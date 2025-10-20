# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

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


class Upsample(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        seq_len: int = 128,
        upsample_ratio: float = 2,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        upsample_depth: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        if num_kv_heads is None:
            self.num_kv_heads = num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_dim = self.num_kv_heads * head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.seq_len = seq_len
        self.upsample_ratio = upsample_ratio
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.upsample_depth = upsample_depth
        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        self.attn_norm = RMSNorm(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.upsample = nn.Linear(seq_len, int(seq_len * upsample_ratio))
        self.rotary = RotaryEmbedding(dim=head_dim, base=rope_theta)
        self.mlp = GatedMLP(hidden_size)
    
    def init_upsampled_tokens(
        self,
        hidden_states: torch.Tensor, #b s d
    ):
        return self.upsample(hidden_states.transpose(-1,-2)).transpose(-1,-2)


    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        upsampled_tokens = self.init_upsampled_tokens(hidden_states)
        
        for _ in range(self.upsample_depth):
            residual = torch.cat([hidden_states, upsampled_tokens], dim=-2)
            upsampled_tokens = self.attn_norm(upsampled_tokens)
            q = rearrange(self.q_proj(upsampled_tokens), "... (h d) -> ... h d", d=self.head_dim)
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
            upsampled_tokens = output + upsampled_tokens
            upsampled_tokens = self.mlp_norm(upsampled_tokens)
            output = self.mlp(upsampled_tokens)
            upsampled_tokens = output + upsampled_tokens
        return upsampled_tokens

