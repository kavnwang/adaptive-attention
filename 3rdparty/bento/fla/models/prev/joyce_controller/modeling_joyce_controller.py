# bento/fla/models/joyce_controller/modeling_joyce_controller.py
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_joyce_controller import JoyceControllerConfig


def _sinusoidal_embed_scalar(x: torch.Tensor, num_freqs: int, scale: float) -> torch.Tensor:
    """
    x: [B] (scalar Δ per example)
    returns: [B, 2 * num_freqs]
    """
    # Use log-spaced frequencies so we cover small & large Δ smoothly.
    # freq_k = 10^(k / (num_freqs-1)) * scale  (k=0..num_freqs-1)
    B = x.shape[0]
    device = x.device
    k = torch.arange(num_freqs, device=device, dtype=x.dtype)
    freqs = (10.0 ** (k / max(1, num_freqs - 1))) * scale  # [num_freqs]
    x = x[:, None] * freqs[None, :]                         # [B, num_freqs]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [B, 2 * num_freqs]


class _TinySelfAttnPool(nn.Module):
    """Optional attention pooler over the shrunken sequence."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln = nn.LayerNorm(hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_shrunk, H]
        q = self.q(x.mean(1, keepdim=True))  # [B,1,H]
        k = self.k(x)                         # [B,T,H]
        v = self.v(x)                         # [B,T,H]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1]), dim=-1)  # [B,1,T]
        pooled = attn @ v  # [B,1,H]
        return self.ln(pooled.squeeze(1))     # [B,H]


@dataclass
class JoyceControllerOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    ratio: torch.Tensor = None          # [B] in (0,1]
    logits: torch.Tensor = None         # [B] unbounded (pre-sigmoid)
    pooled: Optional[torch.Tensor] = None  # [B,H] diagnostic


class JoyceControllerForCompressionRatio(PreTrainedModel):
    """
    Input:
      - shrunken_states: FloatTensor [B, T_shrunk, H]  (from Joyce up-sampling init)
      - delta:           FloatTensor [B]               (Δ = NLL(comp) - NLL(full))
      - target_ratio:    Optional FloatTensor [B]      (supervision r* in (0,1])
    Output:
      - ratio prediction r_hat in (0,1], with optional MSE loss if target provided.
    """
    config_class = JoyceControllerConfig

    def __init__(self, config: JoyceControllerConfig):
        super().__init__(config)
        H = config.hidden_size
        self.ln = nn.LayerNorm(H, eps=config.layer_norm_epsilon)
        self.pooler_type = config.pooler
        if self.pooler_type == "attn":
            self.pooler = _TinySelfAttnPool(H)
        else:
            self.pooler = None

        self.delta_embed_dim = 2 * config.num_delta_freqs
        self.delta_ln = nn.LayerNorm(self.delta_embed_dim, eps=config.layer_norm_epsilon)

        self.proj_in = nn.Sequential(
            nn.Linear(H, config.controller_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.controller_hidden_size + self.delta_embed_dim, config.controller_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.controller_hidden_size, 1),
        )
        # We produce a logit, then squash to (ratio_min, ratio_max] with a scaled sigmoid
        self.ratio_min = float(config.ratio_min)
        self.ratio_max = float(config.ratio_max)

        self.post_init()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        x = self.ln(x)
        if self.pooler_type == "attn":
            return self.pooler(x)                           # [B,H]
        elif self.pooler_type == "mean_ln":
            return x.mean(dim=1)                           # [B,H]
        else:
            return x.mean(dim=1)                           # [B,H]

    def forward(
        self,
        shrunken_states: torch.Tensor,   # [B, T_shrunk, H]
        delta: torch.Tensor,             # [B]
        target_ratio: Optional[torch.Tensor] = None,  # [B], optional supervision
        return_dict: bool = True,
    ) -> Union[Tuple, JoyceControllerOutput]:
        assert shrunken_states.dim() == 3, "shrunken_states must be [B, T_shrunk, H]"
        B = shrunken_states.shape[0]

        # Pool the shrunken sequence to a single vector per example
        pooled = self._pool(shrunken_states)                       # [B,H]
        pooled = self.proj_in(pooled)                              # [B,Hc]

        # Sinusoidal embedding of Δ
        delta = delta.view(B)
        delta = delta.clamp(min=-self.config.max_abs_delta, max=self.config.max_abs_delta)
        delta_embed = _sinusoidal_embed_scalar(
            delta * self.config.delta_embed_scale,
            self.config.num_delta_freqs,
            scale=1.0
        )                                                          # [B, 2F]
        delta_embed = self.delta_ln(delta_embed)

        # Regress ratio (pre-sigmoid logits -> (r_min, r_max])
        h = torch.cat([pooled, delta_embed], dim=-1)               # [B,Hc + 2F]
        logits = self.mlp(h).squeeze(-1)                           # [B]
        ratio = torch.sigmoid(logits)                              # (0,1)
        # scale to (ratio_min, ratio_max]
        ratio = self.ratio_min + (self.ratio_max - self.ratio_min) * ratio

        loss = None
        if target_ratio is not None:
            # Smooth L1 is stable here; you can use MSE, too.
            loss = nn.functional.smooth_l1_loss(ratio, target_ratio)

        if not return_dict:
            return (loss, ratio, logits, pooled)
        return JoyceControllerOutput(loss=loss, ratio=ratio, logits=logits, pooled=pooled)
