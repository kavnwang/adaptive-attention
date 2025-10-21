# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding, GatedMLP
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


class Compress(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        compression_ratio: float = 0.5,
        compression_depth: int = 2,
        seq_len: int = 8192,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.head_dim = head_dim

        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_dim = self.num_kv_heads * head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.compression_ratio = compression_ratio
        self.compression_depth = compression_depth
        self.seq_len = seq_len
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_init = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_init = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        self.attn_norm = RMSNorm(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.rotary = RotaryEmbedding(dim=head_dim, base=rope_theta)
        self.compression = nn.Linear(seq_len, int(seq_len * compression_ratio))
        self.mlp = GatedMLP(hidden_size)

        # Lightweight learnable compressed token embeddings initialized with sinusoidal positions
        init_emb = self._build_sinusoidal_embeddings(self.num_compressed, hidden_size)
        # Register as a Parameter so it can be learned; dtype/device will be managed by precision policies
        self.compressed_token_emb = nn.Parameter(init_emb, requires_grad=True)

    @staticmethod
    def _build_sinusoidal_embeddings(n_positions: int, dim: int) -> torch.Tensor:
        """Create [n_positions, dim] sinusoidal embeddings (sin/cos interleaved).

        Returns float32; model precision policies (FSDP/mp) will cast as needed.
        If dim is odd, the last column is zero-padded.
        """
        if n_positions <= 0:
            n_positions = 1
        half_dim = dim // 2
        positions = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
        # classic transformer angles: 10000^{-2i/d}
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32) * (-math.log(10000.0) / max(1, half_dim))
        )
        angles = positions * div_term  # [n, half]
        emb = torch.empty(n_positions, dim, dtype=torch.float32)
        emb[:, 0:2 * half_dim:2] = torch.sin(angles)
        emb[:, 1:2 * half_dim:2] = torch.cos(angles)
        if dim % 2 == 1:
            emb[:, -1] = 0.0
        return emb
    
    def init_compressed_tokens_mlp(
        self,
        hidden_states: torch.Tensor, #b s d
    ):  
        '''
        compression_length = int(hidden_states.shape[1] * self.compression_ratio)
        compressed_tokens = hidden_states[:,compression_length:,:]
        '''
        compressed_tokens = self.compression(hidden_states.transpose(-1,-2)).transpose(-1,-2)
        return compressed_tokens

    def init_compressed_tokens_suffix(
        self,
        hidden_states: torch.Tensor, #b s d
    ):  
        '''
        compression_length = int(hidden_states.shape[1] * self.compression_ratio)
        compressed_tokens = hidden_states[:,compression_length:,:]
        '''
        num_compressed = int(self.seq_len * self.compression_ratio)
        compressed_tokens = hidden_states[:,-num_compressed:,:]
        return compressed_tokens

    def init_compressed_tokens_attn(
        self,
        hidden_states: torch.Tensor, #b s d
    ):  
        """Return B x num_compressed x hidden_size learnable tokens.

        Efficient: uses a single small parameter matrix (num_compressed x hidden_size)
        initialized from sinusoidal positions; batched by expand without extra copies.
        """
        bsz = hidden_states.shape[0]
        emb = self.compressed_token_emb
        # Ensure dtype/device match the incoming hidden states for compute efficiency
        if emb.dtype != hidden_states.dtype:
            emb = emb.to(hidden_states.dtype)
        if emb.device != hidden_states.device:
            emb = emb.to(hidden_states.device)
        return emb.unsqueeze(0).expand(bsz, -1, -1)


    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, _, _ = hidden_states.shape
        compressed_tokens = self.init_compressed_tokens_suffix(hidden_states)
        
        for _ in range(self.compression_depth):
            residual = torch.cat([hidden_states, compressed_tokens], dim=-2)
            compressed_tokens = self.attn_norm(compressed_tokens)
            q = rearrange(self.q_proj(compressed_tokens), "... (h d) -> ... h d", d=self.head_dim)
            k = rearrange(self.k_proj(residual), "... (h d) -> ... h d", d=self.head_dim)
            v = rearrange(self.v_proj(residual), "... (h d) -> ... h d", d=self.head_dim)
            if self.num_kv_groups != 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=-2)
                v = v.repeat_interleave(self.num_kv_groups, dim=-2)
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            # insert RoPE here
            qk_similarities = torch.einsum("bsnk,btnk->bsnt", q, k)
            qk_similarities = qk_similarities / (self.head_dim ** 0.5)
            attn_scores = torch.softmax(qk_similarities, dim=-1)
            o = torch.einsum("bsnt,btnk->bsnk", attn_scores, v)
            o = o.reshape(batch_size, o.shape[1], -1)
            output = self.o_proj(o)
            compressed_tokens = output + compressed_tokens
            compressed_tokens = self.mlp_norm(compressed_tokens)
            output = self.mlp(compressed_tokens)
            compressed_tokens = output + compressed_tokens
        return compressed_tokens



