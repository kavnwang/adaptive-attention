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
        compression_ratio: float = 0.0625,
        compression_depth: int = 2,
        seq_len: int = 8192,
        init_method: str = "suffix",
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        attention_bias: bool = False,
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
        # Pre-compute expected compressed and key lengths from config
        self.compressed_len = int(self.seq_len * self.compression_ratio)
        self.keys_len = self.seq_len + self.compressed_len
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.init_method = init_method
        self.use_attn_bias = attention_bias

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
        #self.compression = nn.Linear(seq_len, int(seq_len * compression_ratio))
        self.mlp = GatedMLP(hidden_size)

        # Optional key-side (sink) attention bias per head: [H, L+C]
        # This biases certain key positions for every query.
        if self.use_attn_bias:
            self.max_keys_len = self.keys_len
            self.sink_bias_keys = nn.Parameter(torch.zeros(self.num_heads, self.max_keys_len))
        else:
            self.max_keys_len = None
            self.sink_bias_keys = None

    def init_compressed_tokens_mlp(
        self,
        hidden_states: torch.Tensor, #b s d
    ):  
        '''
        compression_length = int(hidden_states.shape[1] * self.compression_ratio)
        compressed_tokens = hidden_states[:,compression_length:,:]
        '''
        #compressed_tokens = self.compression(hidden_states.transpose(-1,-2)).transpose(-1,-2)
        #return compressed_tokens
        pass

    def init_compressed_tokens_suffix(
        self,
        hidden_states: torch.Tensor, 
        compression_ratio: float,#b s d
    ):  
        '''
        compression_length = int(hidden_states.shape[1] * self.compression_ratio)
        compressed_tokens = hidden_states[:,compression_length:,:]
        '''
        num_compressed = int(hidden_states.shape[1] * compression_ratio)
        compressed_tokens = hidden_states[:,-num_compressed:,:]
        return compressed_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        compression_ratio: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, _, _ = hidden_states.shape
        if self.init_method == "suffix":
            compressed_tokens = self.init_compressed_tokens_suffix(hidden_states, compression_ratio)
        elif self.init_method == "mlp":
            compressed_tokens = self.init_compressed_tokens_mlp(hidden_states)
        else:
            compressed_tokens = self.init_compressed_tokens_mlp(hidden_states)
        
        for _ in range(self.compression_depth):
            # Build residual and sizes
            residual = torch.cat([hidden_states, compressed_tokens], dim=-2)
            L = hidden_states.shape[1]
            C = compressed_tokens.shape[1]
            K = residual.shape[1]  # L + C

            # 1) Update original hidden states with FlashAttention (causal within original segment)
            if L > 0:
                orig = self.attn_norm(hidden_states)
                q_o = rearrange(self.q_proj(orig), "... (h d) -> ... h d", d=self.head_dim)
                k_o = rearrange(self.k_proj(orig), "... (h d) -> ... h d", d=self.head_dim)
                v_o = rearrange(self.v_proj(orig), "... (h d) -> ... h d", d=self.head_dim)
                if self.num_kv_groups != 1:
                    k_o = k_o.repeat_interleave(self.num_kv_groups, dim=-2)
                    v_o = v_o.repeat_interleave(self.num_kv_groups, dim=-2)
                if self.qk_norm:
                    q_o = self.q_norm(q_o)
                    k_o = self.k_norm(k_o)
                # RoPE for original tokens: positions [0..L-1]
                q_o, k_o = self.rotary(q_o, k_o, seqlen_offset=0, max_seqlen=L)
                o = flash_attn_func(
                    q_o, k_o, v_o,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                )
                o = o.reshape(orig.shape[0], L, -1)
                out_o = self.o_proj(o)
                hidden_states = hidden_states + out_o
                # MLP on original tokens
                tmp = self.mlp_norm(hidden_states)
                hidden_states = hidden_states + self.mlp(tmp)

            # 2) Update compressed tokens with FlashAttention over residual (keys length K, queries length C)
            compressed_tokens = self.attn_norm(compressed_tokens)
            q = rearrange(self.q_proj(compressed_tokens), "... (h d) -> ... h d", d=self.head_dim)
            # Rebuild residual after original update
            residual = torch.cat([hidden_states, compressed_tokens], dim=-2)
            k = rearrange(self.k_proj(residual), "... (h d) -> ... h d", d=self.head_dim)
            v = rearrange(self.v_proj(residual), "... (h d) -> ... h d", d=self.head_dim)
            if self.num_kv_groups != 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=-2)
                v = v.repeat_interleave(self.num_kv_groups, dim=-2)
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            # RoPE with distinct offsets for queries and keys
            q, _ = self.rotary(q, q, seqlen_offset=L, max_seqlen=K)
            _, k = self.rotary(k, k, seqlen_offset=0, max_seqlen=K)
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
            o = o.reshape(batch_size, C, -1)
            output = self.o_proj(o)
            compressed_tokens = compressed_tokens + output
            compressed_tokens = self.mlp_norm(compressed_tokens)
            output = self.mlp(compressed_tokens)
            compressed_tokens = compressed_tokens + output
        return compressed_tokens
