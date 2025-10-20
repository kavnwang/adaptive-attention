# 3rdparty/bento/fla/models/joyce/modeling_joyce_pretrain.py
from __future__ import annotations
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput
from torch.nn.modules.module import _IncompatibleKeys

from .config_joyce import JoyceAEConfig
from ...layers.joyce import (
    JoyceBlockCfg,
    JoyceCompressionBlock,
    JoyceUpsamplingBlock,
)


class JoyceAEReturn(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon: Optional[torch.Tensor] = None     # (B, T, D) reconstructed hiddens
    target: Optional[torch.Tensor] = None    # (B, T, D) original hiddens
    compressed: Optional[torch.Tensor] = None# (B, C, D) compressed tokens


class CompressionAutoEncoder(nn.Module):
    """Pure AE: given layer-L hidden states (B, T, D) → compress (B, C, D) → reconstruct (B, T, D)."""
    def __init__(self, cfg: JoyceAEConfig):
        super().__init__()
        blk = JoyceBlockCfg(
            dim=cfg.hidden_size,
            latent_dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            norm_eps=cfg.norm_eps,
        )
        self.compress = JoyceCompressionBlock(
            cfg=blk,
            t_in=cfg.seq_len,
            t_out=cfg.num_compressed_tokens,
            depth=cfg.compression_depth,
            tie_up_down=cfg.tie_mixers,
        )
        self.upsample = JoyceUpsamplingBlock(
            cfg=blk,
            t_out=cfg.seq_len,
            t_in=cfg.num_compressed_tokens,
            depth=cfg.upsample_depth,
            tie_up_down=self.compress.mix_down if cfg.tie_mixers else None,
            use_refine_attn=cfg.use_refine_attn,
        )

    def forward(
        self,
        h_L: torch.Tensor,                       # (B, T, D) layer-L hidden states
        attn_mask: Optional[torch.Tensor] = None # optional additive mask for SDPA
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.compress(h_L, attn_mask=attn_mask)  # (B, C, D)
        y = self.upsample(z, x_ctx=h_L, attn_mask=attn_mask)  # (B, T, D)
        return z, y


class JoyceAutoencoderForPreTraining(nn.Module):
    """
    Wrapper that:
      * runs a frozen base model to obtain hidden states at layer L
      * applies CompressionAutoEncoder
      * computes L2 reconstruction loss (sum over tokens and channels)
    The base model must support `output_hidden_states=True`.
    """
    def __init__(self, base_model: nn.Module, cfg: JoyceAEConfig):
        super().__init__()
        self.base_model = base_model.eval()  # frozen
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        self.cfg = cfg
        ae = CompressionAutoEncoder(cfg)
        # Register compressor and upsampler directly for cleaner checkpointing.
        self.compressor = ae.compress
        self.upsampler = ae.upsample
        # Preserve backwards compatibility for code expecting `model.ae`.
        self.ae = SimpleNamespace(compress=self.compressor, upsample=self.upsampler)
        self.register_buffer(
            "attn_mask_cache",
            torch.zeros(1, cfg.seq_len, cfg.seq_len), persistent=False
        )  # kept for interface completeness (unused by default)

    @torch.no_grad()
    def _take_layer_L(self, bm_out: BaseModelOutputWithPast) -> torch.Tensor:
        # HF models usually return hidden_states as tuple(len = num_layers + 1)
        hs = bm_out.hidden_states
        assert hs is not None and len(hs) > self.cfg.compress_after_layer, \
            "Base model did not return enough hidden states; set output_hidden_states=True"
        return hs[self.cfg.compress_after_layer]  # (B, T, D)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **bm_kwargs,
    ) -> JoyceAEReturn:
        bm_kwargs = dict(bm_kwargs)
        bm_kwargs.update(dict(
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        ))
        with torch.no_grad():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **bm_kwargs,
            )
        hL = self._take_layer_L(base_out)  # (B, T, D)

        z = self.compressor(hL, attn_mask=None)
        recon = self.upsampler(z, x_ctx=hL, attn_mask=None)

        # L2 reconstruction loss (sum over T and D, mean over batch)
        diff = recon - hL
        loss = (diff.pow(2).sum(dim=(1, 2))).mean()

        return JoyceAEReturn(
            loss=loss,
            recon=recon,
            target=hL.detach(),
            compressed=z.detach(),
        )

    # ------------------------------------------------------------------
    # Custom checkpointing helpers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Expose only compressor / upsampler weights (base model is frozen).
        """
        if destination is None:
            destination = OrderedDict()
        # Preserve destination type if caller provided one.
        dest = destination

        comp_sd = self.compressor.state_dict(
            destination=None, prefix="", keep_vars=keep_vars
        )
        for k, v in comp_sd.items():
            dest[prefix + f"compressor.{k}"] = v

        ups_sd = self.upsampler.state_dict(
            destination=None, prefix="", keep_vars=keep_vars
        )
        for k, v in ups_sd.items():
            dest[prefix + f"upsampler.{k}"] = v

        return dest

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Accept checkpoints saved either in the new split format
        (compressor.* / upsampler.*) or the legacy ae.compress / ae.upsample
        naming. Frozen base model weights are ignored if present.
        """
        comp_state: Dict[str, torch.Tensor] = OrderedDict()
        up_state: Dict[str, torch.Tensor] = OrderedDict()
        unexpected: list[str] = []

        for key, value in state_dict.items():
            if key.startswith("compressor."):
                comp_state[key[len("compressor."):]] = value
            elif key.startswith("ae.compress."):
                comp_state[key[len("ae.compress."):]] = value
            elif key.startswith("upsampler."):
                up_state[key[len("upsampler."):]] = value
            elif key.startswith("ae.upsample."):
                up_state[key[len("ae.upsample."):]] = value
            elif key.startswith("base_model."):
                # Base model stays frozen; silently drop.
                continue
            else:
                unexpected.append(key)

        comp_result = self.compressor.load_state_dict(comp_state, strict=strict)
        up_result = self.upsampler.load_state_dict(up_state, strict=strict)

        missing = [f"compressor.{k}" for k in comp_result.missing_keys]
        missing += [f"upsampler.{k}" for k in up_result.missing_keys]

        unexpected_keys = (
            [f"compressor.{k}" for k in comp_result.unexpected_keys]
            + [f"upsampler.{k}" for k in up_result.unexpected_keys]
            + unexpected
        )

        if strict and (missing or unexpected_keys):
            raise RuntimeError(
                f"Error loading Joyce AE state_dict. "
                f"Missing keys: {missing}; Unexpected keys: {unexpected_keys}"
            )

        return _IncompatibleKeys(missing, unexpected_keys)
