# llmonade/modules/joyce_controller/ratio_controller.py
from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_scalar_embedding(x: torch.Tensor, dim: int, min_freq: float = 1e-4, max_freq: float = 1.0):
    """
    Sin-cos embedding for a scalar x (shape [B] or [B,1]) with log-spaced frequencies.
    Returns [B, dim].
    """
    x = x.view(-1, 1)  # [B,1]
    # log-spaced frequencies from min_freq..max_freq (inclusive)
    freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), dim // 2, device=x.device)
    angles = x * freqs  # [B, dim//2]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, dim- (dim%2)]
    if emb.shape[-1] < dim:  # pad if odd dim
        emb = F.pad(emb, (0, 1), value=0.0)
    return emb


class JoyceRatioController(nn.Module):
    """
    Predicts keep-ratio r in [0,1] from:
      - shrunken hidden states S in R^{B x T' x D_s} (from upsampling init)
      - a scalar loss budget Δ* (allowed NLL increase)
    Design: summarize S with LN+MLP then mean+max pool; concatenate with sinusoidal Δ* embedding;
    push through a small MLP and squash with sigmoid.
    """
    def __init__(
        self,
        d_shrunken: int,          # channel dim of the shrunken states
        d_model: int = 256,
        delta_emb_dim: int = 64,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_shrunken)
        self.proj = nn.Sequential(
            nn.Linear(d_shrunken, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.delta_emb_dim = delta_emb_dim
        self.head = nn.Sequential(
            nn.Linear(2 * d_model + delta_emb_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

        # optional learned scale/shift for Δ*
        self.delta_affine = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))  # scale, shift

    def forward(self, shrunken_states: torch.Tensor, delta_budget: torch.Tensor) -> torch.Tensor:
        """
        shrunken_states: [B, T', D_s]
        delta_budget:    [B] (scalar Δ* per sample, in nats)
        Returns:
          keep_ratio r \in [0,1], shape [B]
        """
        B, Tp, Ds = shrunken_states.shape
        x = self.norm(shrunken_states)            # [B, T', Ds]
        x = self.proj(x)                          # [B, T', d_model]
        mean_pool = x.mean(dim=1)                 # [B, d_model]
        max_pool, _ = x.max(dim=1)                # [B, d_model]
        seq_summary = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2*d_model]

        # affine + embedding for Δ*
        scale, shift = self.delta_affine[0], self.delta_affine[1]
        delta = scale * delta_budget + shift      # [B]
        delta_emb = sinusoidal_scalar_embedding(delta, self.delta_emb_dim)  # [B, delta_emb_dim]

        h = torch.cat([seq_summary, delta_emb], dim=-1)  # [B, 2*d_model + delta_emb_dim]
        r = torch.sigmoid(self.head(h)).squeeze(-1)      # [B], keep-ratio in (0,1)
        return r
