#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint / continued training for Joyce:
- Input per sample: 2 * seq_len tokens (first window + second window).
- Run base layers [0..L] across the whole 2*seq_len.
- Compress FIRST window (length S = seq_len) with Joyce.
- Branch A (reconstruction): upsample compressed -> run [L+1..end] on length S -> CE loss A over first window.
- Branch B (extended context): prepend compressed to SECOND window -> run [L+1..end] on length (C+S) -> CE loss B over last S only.
- Optimize base + Joyce modules jointly; initialize Joyce from AE checkpoints if provided.

This script intentionally avoids framework-specific trainers and uses only PyTorch + HF Datasets/Transformers.
"""

from __future__ import annotations
import os
import math
import time
import argparse
from typing import Optional, Iterable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# --- Import Joyce blocks from your bento submodule ---
# Path: 3rdparty/bento/fla/layers/joyce.py
from fla.layers.joyce import (
    JoyceBlockCfg, JoyceCompressionBlock, JoyceUpsamplingBlock
)


# ----------------------- Utilities -----------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    save_dir: str,
    step: int,
    base_model: nn.Module,
    compress: nn.Module,
    upsample: nn.Module,
):
    os.makedirs(save_dir, exist_ok=True)
    step_dir = os.path.join(save_dir, f"checkpoint-{step}")
    os.makedirs(step_dir, exist_ok=True)

    # save torch weights
    base_model.save_pretrained(step_dir)  # works for HF models
    torch.save(compress.state_dict(), os.path.join(step_dir, "compressor.pt"))
    torch.save(upsample.state_dict(), os.path.join(step_dir, "upsampler.pt"))
    print(f"[ckpt] saved to {step_dir}")


def maybe_load_state_dict(module: nn.Module, path: Optional[str]):
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    # Accept plain state_dict OR a packaged dict
    sd = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    try:
        module.load_state_dict(sd, strict=True)
    except RuntimeError:
        # try to fish submodule keys (e.g., saved as "compress.xxx")
        prefix = "compress." if any(k.startswith("compress.") for k in sd) else "upsample."
        sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        module.load_state_dict(sub_sd, strict=False)


# ----------------------- Dataset (streaming) -----------------------

class StreamingSequenceDataset(IterableDataset):
    """
    Streams a text dataset and yields fixed-length sequences of token ids.
    Each sample has length = block_size (we expect block_size = 2 * seq_len).
    """
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: AutoTokenizer,
        text_key: str,
        block_size: int,
        streaming: bool = True,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.block_size = block_size
        self.streaming = streaming
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        ds = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)

        buf = []
        for ex in ds:
            text = ex[self.text_key]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.bos_token_id is not None:
                buf.append(self.bos_token_id)
            buf.extend(ids)
            if self.eos_token_id is not None:
                buf.append(self.eos_token_id)

            while len(buf) >= self.block_size:
                chunk = buf[:self.block_size]
                buf = buf[self.block_size:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long)
                }


# ----------------------- Base model adapter -----------------------

class _BaseStackAdapter:
    """
    Small shim to run layers [start:end] and access embeddings/lm_head
    without editing the underlying model code.
    """
    def __init__(self, base_model: nn.Module):
        self.base = base_model

        # embeddings
        if hasattr(base_model, "get_input_embeddings"):
            self.tok_embed = base_model.get_input_embeddings()
        elif hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
            self.tok_embed = base_model.model.embed_tokens
        else:
            raise AttributeError("Could not find token embedding on base model")

        # block list
        self.layers = (
            getattr(getattr(base_model, "model", base_model), "layers", None)
            or getattr(getattr(base_model, "transformer", base_model), "layers", None)
            or getattr(getattr(base_model, "model", base_model), "h", None)
            or getattr(getattr(base_model, "transformer", base_model), "h", None)
            or getattr(base_model, "layers", None)
        )
        if not isinstance(self.layers, (list, nn.ModuleList)):
            raise AttributeError("Could not find decoder layers list on base model")

        # final norm
        self.final_norm = (
            getattr(getattr(base_model, "model", base_model), "norm", None)
            or getattr(base_model, "norm", None)
            or getattr(getattr(base_model, "transformer", base_model), "ln_f", None)
            or getattr(base_model, "ln_f", None)
        )
        if self.final_norm is None:
            raise AttributeError("Could not find final norm on base model")

        # lm head
        if hasattr(base_model, "lm_head"):
            self.lm_head = base_model.lm_head
        elif hasattr(base_model, "get_output_embeddings"):
            self.lm_head = base_model.get_output_embeddings()
        else:
            raise AttributeError("Could not find lm_head on base model")

    def _call_block(self, block: nn.Module, x: torch.Tensor, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        try:
            return block(x, position_ids=position_ids, use_cache=False)
        except TypeError:
            try:
                return block(x, position_ids=position_ids)
            except TypeError:
                return block(x)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embed(input_ids)

    def run_layers(self, x: torch.Tensor, start: int, end: int, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        for i in range(start, end):
            x = self._call_block(self.layers[i], x, position_ids)
        return x

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        h = self.final_norm(h)
        return self.lm_head(h)


# ----------------------- Joint model wrapper -----------------------

class TransformerWithJoyce(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_heads: int,
        latent_dim: int,
        norm_eps: float,
        mlp_ratio: float,
        dropout: float,
        seq_len: int,
        num_compressed: int,
        layer_L: int,
        use_refine_attn: bool = True,
        tie_mixers: bool = True,
    ):
        super().__init__()
        self.adapter = _BaseStackAdapter(base_model)
        self.base = base_model

        cfg = JoyceBlockCfg(
            dim=hidden_size,
            latent_dim=latent_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.compress = JoyceCompressionBlock(
            cfg=cfg, t_in=seq_len, t_out=num_compressed, depth=1, tie_up_down=tie_mixers
        )
        self.upsample = JoyceUpsamplingBlock(
            cfg=cfg, t_out=seq_len, t_in=num_compressed, depth=1,
            tie_up_down=self.compress.mix_down if tie_mixers else None,
            use_refine_attn=use_refine_attn
        )

        self.S = seq_len
        self.C = num_compressed
        self.L = layer_L
        self.num_layers = len(self.adapter.layers)

    def _pos(self, B: int, T: int, device: torch.device, offset: int = 0) -> torch.Tensor:
        return (torch.arange(offset, offset + T, device=device)[None, :]).expand(B, T)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        input_ids: (B, 2*S)
        returns: {"loss": scalar, "lossA": ..., "lossB": ...}
        """
        B, T = input_ids.shape
        assert T == 2 * self.S, f"Expected input length 2*S={2*self.S}, got {T}"
        device = input_ids.device

        # Embedding
        h = self.adapter.embed(input_ids)  # (B, 2S, D)

        # Layers 0..L
        pos_full = self._pos(B, T, device)
        hL = self.adapter.run_layers(h, 0, self.L + 1, pos_full)

        # Split
        h_first = hL[:, :self.S, :]
        h_second = hL[:, self.S:, :]

        # Compress
        z = self.compress(h_first)  # (B, C, D)

        # Branch A: upsample → L+1..end on length S
        yA = self.upsample(z, x_ctx=h_first)
        posA = self._pos(B, self.S, device)
        yA = self.adapter.run_layers(yA, self.L + 1, self.num_layers, posA)
        logitsA = self.adapter.logits(yA)  # (B, S, V)
        lossA = self._ntp_ce(logitsA, input_ids[:, :self.S])

        # Branch B: prepend z to second window → L+1..end
        yB_in = torch.cat([z, h_second], dim=1)  # (B, C+S, D)
        posB = self._pos(B, self.C + self.S, device)
        yB = self.adapter.run_layers(yB_in, self.L + 1, self.num_layers, posB)
        logitsB = self.adapter.logits(yB)        # (B, C+S, V)
        logitsB_tail = logitsB[:, self.C:, :]    # only the last S
        lossB = self._ntp_ce(logitsB_tail, input_ids[:, self.S:])

        loss = 0.5 * (lossA + lossB)
        return {
            "loss": loss,
            "lossA": lossA.detach(),
            "lossB": lossB.detach(),
            "loss_first_half": lossA.detach(),
            "loss_second_half": lossB.detach(),
        }

    @staticmethod
    def _ntp_ce(logits: torch.Tensor, input_ids_window: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        tgt = input_ids_window[:, 1:].contiguous().view(B * (T - 1))
        pred = logits[:, :-1, :].contiguous().view(B * (T - 1), V)
        return F.cross_entropy(pred, tgt, reduction="mean")


# ----------------------- Training loop -----------------------

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    wandb_module = None
    if args.enable_wandb:
        try:
            import wandb  # type: ignore

            wandb_module = wandb
            project = args.wandb_project or os.getenv("WANDB_PROJECT") or "joyce-joint"
            entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
            mode = args.wandb_mode or os.getenv("WANDB_MODE")
            run_name = args.wandb_run_name or os.getenv("WANDB_NAME")
            tags = args.wandb_tags or []

            init_kwargs: Dict[str, Any] = {
                "project": project,
                "config": {
                    "model_name_or_path": args.model_name_or_path,
                    "tokenizer_name_or_path": args.tokenizer_name_or_path,
                    "dataset": args.dataset,
                    "dataset_split": args.dataset_split,
                    "text_key": args.text_key,
                    "seq_len": args.seq_len,
                    "layer_L": args.layer_L,
                    "num_compressed_states": args.num_compressed_states,
                    "latent_dim": args.latent_dim,
                    "mlp_ratio": args.mlp_ratio,
                    "dropout": args.dropout,
                    "num_compressed": args.num_compressed_states,
                    "warmup_steps": args.warmup_steps,
                    "train_steps": args.train_steps,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "fp16": args.fp16,
                    "save_every": args.save_every,
                },
            }
            if entity:
                init_kwargs["entity"] = entity
            if mode:
                init_kwargs["mode"] = mode
            if run_name:
                init_kwargs["name"] = run_name
            if tags:
                init_kwargs["tags"] = tags

            wandb_run = wandb_module.init(**init_kwargs)
            wandb_module.define_metric("train/step")
            wandb_module.define_metric("train/*", step_metric="train/step")
        except ModuleNotFoundError:
            print("⚠️  wandb not installed. Install with: pip install wandb")
        except Exception as exc:
            print(f"⚠️  Failed to init wandb ({exc}). Continuing without wandb logging.")
            wandb_module = None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else args.bos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id

    # Base model
    base_cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=base_cfg, trust_remote_code=True)
    base_model.to(device)

    # Joyce wrapper
    model = TransformerWithJoyce(
        base_model=base_model,
        hidden_size=getattr(base_cfg, "hidden_size"),
        num_heads=getattr(base_cfg, "num_attention_heads", None) or getattr(base_cfg, "num_heads"),
        latent_dim=args.latent_dim,
        norm_eps=getattr(base_cfg, "rms_norm_eps", None) or getattr(base_cfg, "norm_eps", 1e-6),
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        seq_len=args.seq_len,
        num_compressed=args.num_compressed_states,
        layer_L=args.layer_L,
        use_refine_attn=not args.no_refine_attn,
        tie_mixers=not args.no_tie_mixers,
    ).to(device)

    # Optionally load compressor/upsampler init (from AE pretraining)
    maybe_load_state_dict(model.compress, args.compressor_ckpt)
    maybe_load_state_dict(model.upsample, args.upsampler_ckpt)

    # Optimizer
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )

    # Scheduler: linear warmup, cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.train_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # Data
    block_size = 2 * args.seq_len
    ds = StreamingSequenceDataset(
        dataset_name=args.dataset,
        split=args.dataset_split,
        tokenizer=tokenizer,
        text_key=args.text_key,
        block_size=block_size,
        streaming=True,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=2)

    # AMP
    use_fp16 = args.fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # Training
    model.train()
    step, tok_loss_avg = 0, 0.0
    optim.zero_grad(set_to_none=True)
    step_start_time = time.time()

    for batch in dl:
        input_ids = batch["input_ids"].to(device, non_blocking=True)  # (B, 2*S)
        with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
            out = model(input_ids)
            loss = out["loss"] / args.grad_accum

        scaler.scale(loss).backward()

        grad_norm_val = None
        if (step + 1) % args.grad_accum == 0:
            # clip
            grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            scheduler.step()

        step += 1
        tok_loss_avg = 0.9 * tok_loss_avg + 0.1 * out["loss"].item()
        now = time.time()
        step_time = now - step_start_time
        tokens_this_step = input_ids.numel()

        if step % 10 == 0:
            print(
                f"[step {step}] loss={out['loss'].item():.4f}  ema={tok_loss_avg:.4f}  "
                f"first_half_loss={out['loss_first_half'].item():.4f} "
                f"second_half_loss={out['loss_second_half'].item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if wandb_module:
            log_data = {
                "train/step": step,
                "train/loss": out["loss"].item(),
                "train/lossA": out["lossA"].item(),
                "train/lossB": out["lossB"].item(),
                "train/loss_first_half": out["loss_first_half"].item(),
                "train/loss_second_half": out["loss_second_half"].item(),
                "train/ema_loss": tok_loss_avg,
                "train/lr": optim.param_groups[0]["lr"],
                "train/tokens": tokens_this_step,
                "train/tokens_per_sec": tokens_this_step / step_time if step_time > 0 else float("inf"),
                "train/step_time_s": step_time,
            }
            if grad_norm_val is not None:
                log_data["train/grad_norm"] = float(grad_norm_val)
            wandb_module.log(log_data, step=step)
        step_start_time = now

        if (args.save_every > 0) and (step % args.save_every == 0):
            save_checkpoint(args.save_dir, step, base_model, model.compress, model.upsample)

        if step >= args.train_steps:
            break

    # final save
    save_checkpoint(args.save_dir, step, base_model, model.compress, model.upsample)
    if wandb_module:
        final_log = {"train/step": step}
        if "out" in locals():
            final_log["train/final_loss"] = out["loss"].item()
        wandb_module.log(final_log, step=step)
        wandb_module.finish()
    print("[Joyce Joint] done.")


# ----------------------- CLI -----------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--tokenizer_name_or_path", type=str, required=True)

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_key", type=str, default="text")

    # Geometry
    p.add_argument("--seq_len", type=int, required=True, help="Half-window S (total length per sample = 2*S).")
    p.add_argument("--layer_L", type=int, required=True)
    p.add_argument("--num_compressed_states", type=int, required=True)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no_refine_attn", action="store_true")
    p.add_argument("--no_tie_mixers", action="store_true")

    # Optim
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--train_steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)

    # Checkpointing
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--save_dir", type=str, default="checkpoints_joint")
    p.add_argument("--compressor_ckpt", type=str, default=None)
    p.add_argument("--upsampler_ckpt", type=str, default=None)

    # Tokens
    p.add_argument("--bos_token_id", type=int, default=None)
    p.add_argument("--eos_token_id", type=int, default=None)

    # Precision
    p.add_argument("--fp16", action="store_true")

    # WandB
    p.add_argument("--enable_wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name.")
    p.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team or username).")
    p.add_argument("--wandb_run_name", type=str, default=None, help="Optional run name for WandB.")
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Optional tags for the WandB run.")
    p.add_argument("--wandb_mode", type=str, default=None, help="WandB mode (online, offline, disabled).")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
