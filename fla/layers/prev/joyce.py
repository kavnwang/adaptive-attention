# 3rdparty/bento/fla/layers/joyce.py
# Minimal, dependency-light building blocks used by Joyce.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weight is on the same device as input tensor
        weight = self.weight.to(x.device, dtype=x.dtype)
        return _rms_norm(x, self.eps) * weight


class SwiGLULinear(nn.Module):
    """SwiGLU MLP as in PaLM / LLaMA (gate, up, down)."""
    def __init__(self, dim: int, hidden_ratio: float = 4.0, bias: bool = False):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.gate = nn.Linear(dim, hidden, bias=bias)
        self.up = nn.Linear(dim, hidden, bias=bias)
        self.down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CrossAttention(nn.Module):
    """Vanilla SDPA-based cross attention (Q from x, K/V from ctx)."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,          # (B, Tx, D) queries
        ctx: torch.Tensor,        # (B, Tc, D) keys/values
        attn_mask: Optional[torch.Tensor] = None,  # broadcastable additive mask
    ) -> torch.Tensor:
        B, Tx, D = x.shape
        Tc = ctx.shape[1]

        # Ensure input tensors have the same dtype
        x = x.to(ctx.dtype)
        
        q = self.q(x).view(B, Tx, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tx, Hd)
        k = self.k(ctx).view(B, Tc, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tc, Hd)
        v = self.v(ctx).view(B, Tc, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tc, Hd)

        # PyTorch SDPA expects (B*H, T, Hd)
        q = q.reshape(B * self.num_heads, Tx, self.head_dim)
        k = k.reshape(B * self.num_heads, Tc, self.head_dim)
        v = v.reshape(B * self.num_heads, Tc, self.head_dim)

        # attn_mask: broadcast to (B*H, Tx, Tc) additive mask in logit space.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B*H, Tx, Hd)

        y = y.reshape(B, self.num_heads, Tx, self.head_dim).transpose(1, 2).reshape(B, Tx, D)
        y = self.o(y)
        return y


class TokenMixDown(nn.Module):
    """
    Learnable linear mixing across the *sequence* axis: T -> C.
    Implemented as a right-multiply with a (C x T) matrix on latent channels.
    """
    def __init__(self, t_in: int, t_out: int):
        super().__init__()
        # Initialize to something close to strided average pooling.
        W = torch.zeros(t_out, t_in)
        idx = torch.linspace(0, t_in - 1, steps=t_out)
        for c, pos in enumerate(idx):
            # Soft binning around center 'pos'
            sigma = max(1.0, (t_in / t_out) / 2)
            grid = torch.arange(t_in)
            W[c] = torch.exp(-0.5 * ((grid - pos) / sigma) ** 2)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-6)
        self.weight = nn.Parameter(W)  # (C, T)

    def forward(self, x_latent: torch.Tensor) -> torch.Tensor:
        # x_latent: (B, T, H) → (B, C, H) via einsum over sequence
        # Ensure weight is on the same device as input tensor
        weight = self.weight.to(x_latent.device, dtype=x_latent.dtype)
        return torch.einsum('b t h, c t -> b c h', x_latent, weight)


class TokenMixUp(nn.Module):
    """Learnable linear mixing across the sequence axis: C -> T."""
    def __init__(self, t_out: int, t_in: int, tied_down: Optional[TokenMixDown] = None):
        super().__init__()
        if tied_down is not None:
            # Optional weight tying: Up is the transpose of Down.
            self.weight = tied_down.weight.T  # (T, C), shared storage
        else:
            W = torch.zeros(t_out, t_in)
            idx = torch.linspace(0, t_out - 1, steps=t_in)
            for c, pos in enumerate(idx):
                sigma = max(1.0, (t_out / t_in) / 2)
                grid = torch.arange(t_out)
                W[:, c] = torch.exp(-0.5 * ((grid - pos) / sigma) ** 2)
            W = W / (W.sum(dim=0, keepdim=True) + 1e-6)
            self.weight = nn.Parameter(W)  # (T, C)

    def forward(self, x_latent: torch.Tensor) -> torch.Tensor:
        # x_latent: (B, C, H) → (B, T, H)
        # Ensure weight is on the same device as input tensor
        weight = self.weight.to(x_latent.device, dtype=x_latent.dtype)
        return torch.einsum('b c h, t c -> b t h', x_latent, weight)


@dataclass
class JoyceBlockCfg:
    dim: int
    latent_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    norm_eps: float = 1e-6


class JoyceCompressionBlock(nn.Module):
    """
    Compress T tokens → C tokens, then refine compressed tokens with
    cross-attention over the original T tokens and a SwiGLU MLP.
    """
    def __init__(
        self,
        cfg: JoyceBlockCfg,
        t_in: int,
        t_out: int,
        depth: int = 1,
        tie_up_down: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.t_in, self.t_out = t_in, t_out
        self.depth = depth

        self.down_proj = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.up_proj = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.mix_down = TokenMixDown(t_in=t_in, t_out=t_out)  # (C, T)
        self.mix_up = TokenMixUp(t_out=t_in, t_in=t_out, tied_down=self.mix_down if tie_up_down else None)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict(dict(
                q_norm=RMSNorm(cfg.dim, cfg.norm_eps),
                kv_norm=RMSNorm(cfg.dim, cfg.norm_eps),
                attn=CrossAttention(cfg.dim, cfg.num_heads, qkv_bias=False, dropout=cfg.dropout),
                mlp_norm=RMSNorm(cfg.dim, cfg.norm_eps),
                mlp=SwiGLULinear(cfg.dim, cfg.mlp_ratio, bias=False),
            )))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        assert T == self.t_in and D == self.cfg.dim, "Unexpected shape"

        # 1) Sequence-axis downmix in latent channel space
        x_lat = self.down_proj(x)                           # (B, T, H)
        z0_lat = self.mix_down(x_lat)                       # (B, C, H)
        z = self.up_proj(z0_lat)                            # (B, C, D)

        # 2) Depth × [CrossAttn(z ← x) + MLP]
        for blk in self.blocks:
            z = z + blk["attn"](blk["q_norm"](z), blk["kv_norm"](x), attn_mask)
            z = z + blk["mlp"](blk["mlp_norm"](z))
        return z  # (B, C, D)


class JoyceUpsamplingBlock(nn.Module):
    """
    Expand C tokens → T tokens via sequence-axis upmix and optionally refine the
    reconstructed tokens by attending to the compressed tokens.
    """
    def __init__(
        self,
        cfg: JoyceBlockCfg,
        t_out: int,
        t_in: int,
        depth: int = 1,
        tie_up_down: Optional[TokenMixDown] = None,
        use_refine_attn: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.t_out, self.t_in = t_out, t_in
        self.use_refine_attn = use_refine_attn
        # Exposed feature tap: shrunken latent sequence before any expansion
        self._last_shrunken: Optional[torch.Tensor] = None  # (B, C, H)

        self.down_proj = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.up_proj = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.mix_up = TokenMixUp(t_out=t_out, t_in=t_in, tied_down=tie_up_down)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict(dict(
                # refinement: y attends to z (Q from y, K/V from z)
                q_norm=RMSNorm(cfg.dim, cfg.norm_eps),
                kv_norm=RMSNorm(cfg.dim, cfg.norm_eps),
                attn=CrossAttention(cfg.dim, cfg.num_heads, qkv_bias=False, dropout=cfg.dropout),
                mlp_norm=RMSNorm(cfg.dim, cfg.norm_eps),
                mlp=SwiGLULinear(cfg.dim, cfg.mlp_ratio, bias=False),
            )))

    def forward(
        self,
        z: torch.Tensor,             # (B, C, D) compressed tokens
        x_ctx: Optional[torch.Tensor] = None,  # (B, T, D) optional to pass along
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, D = z.shape
        assert C == self.t_in and D == self.cfg.dim

        # 1) Sequence-axis upmix in latent space to reconstruct T tokens
        z_lat = self.down_proj(z)             # (B, C, H)
        # Keep a copy of the shrunken sequence for external consumers (e.g., controllers)
        self._last_shrunken = z_lat
        y_lat = self.mix_up(z_lat)            # (B, T, H)
        y = self.up_proj(y_lat)               # (B, T, D)

        if not self.use_refine_attn or len(self.blocks) == 0:
            return y

        # 2) refine via cross-attn y ← z plus MLP
        for blk in self.blocks:
            y = y + blk["attn"](blk["q_norm"](y), blk["kv_norm"](z), attn_mask)
            y = y + blk["mlp"](blk["mlp_norm"](y))
        return y

    def get_last_shrunken(self) -> Optional[torch.Tensor]:
        """
        Returns the most recent shrunken latent sequence before upmixing.
        Shape: (B, C, H) where C is the compressed length (T') and H is latent_dim (D_s).
        """
        return self._last_shrunken
