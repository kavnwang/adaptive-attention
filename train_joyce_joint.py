
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint (continued) training for Joyce: compress-at-layer-L + upsample training,
and "compressed prefix" conditioning for the second half of the sequence.

This script implements the "Combined Training" phase described in the Joyce
technical write-up. It assumes you have already pre-trained:
  1) a base causal language model (next-token prediction), and
  2) a Joyce compression block + upsampling block (reconstruction L2 loss).

During each training step with sequence length 2*S (S = args.seq_len), we:
  • Run layers 0..L on the first S tokens ("A"), obtain H_A (B, S, d).
  • Compress H_A -> C (B, T, d), where T = args.num_compressed_states.
  • Upsample C -> U_A (B, S, d) and run layers L+1..end on U_A to get logits_A.
    Compute standard LM loss on "A" (next-token prediction). This loss updates
    base model, compressor, and upsampler.
  • Run layers 0..L on the last S tokens ("B") **with position offset S** to
    obtain H_B (B, S, d).  Build [C ; H_B] (B, T+S, d) and run layers L+1..end
    to compute logits over T+S positions. Compute LM loss **only over B** tokens
    (ignore the C prefix). This loss updates the base model and the compressor
    (upsampler is not used in this branch).

Notes / assumptions:
  • The base model is LLaMA-like (Hugging Face style), with attributes:
      base.model.embed_tokens, base.model.layers (list), base.model.norm, base.lm_head
    and with blocks that accept (hidden_states, attention_mask, position_ids, ...).
    If your blocks have a different forward signature, adapt _block_forward().
  • Rotary / positional encoding: we provide explicit position_ids so "B" sees
    position indices offset by S in the lower (0..L) stack. Above L, when we
    concatenate [C ; H_B], the causal mask is constructed so "B" can attend to
    "C" and its own previous tokens but not vice versa.
  • The compressor and upsampler modules are imported from your bento/fla
    implementation. We perform a flexible import so you can name the modules
    as you like. See _load_joyce_modules().

Author: (drop-in for your repo)
"""
from __future__ import annotations

import os
import sys
import math
import time
import json
import types
import random
import inspect
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, set_seed
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------

def add_bento_to_sys_path(repo_root: str):
    """
    Add 3rdparty/bento to sys.path so we can import modules from the submodule.
    """
    candidate = os.path.join(repo_root, "3rdparty", "bento")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)


def maybe_all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        t = t / torch.distributed.get_world_size()
    return t


def setup_ddp(args):
    if args.distributed and torch.cuda.is_available():
        # infer if launched with torchrun
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        else:
            # single-process fallback
            args.distributed = False
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0


def is_main(args) -> bool:
    return (not args.distributed) or (args.rank == 0)


# ------------------------------------------------------------------------------------
# Joyce module loader (import from bento/fla or a local path)
# ------------------------------------------------------------------------------------

def _load_joyce_modules(args):
    """
    Attempt to import compressor & upsampler from your submodule.
    You can adapt names here to match your concrete implementation.

    Expected classes:
      - JoyceCompressor: nn.Module with forward(hidden_states: (B,S,d)) -> (B,T,d)
      - JoyceUpSampler:  nn.Module with forward(compressed: (B,T,d), seq_len: int) -> (B,S,d)
    """
    import_errors = []
    # Try common paths
    candidate_imports = [
        ("fla.models.joyce", "JoyceCompressor", "JoyceUpSampler"),
        ("fla.models.joyce", "JoyceCompression", "JoyceUpSampling"),
        ("bento.fla.models.joyce", "JoyceCompressor", "JoyceUpSampler"),
        ("bento.fla.models.joyce", "JoyceCompression", "JoyceUpSampling"),
        ("fla.models", "JoyceCompressor", "JoyceUpSampler"),
        ("fla.models", "JoyceCompression", "JoyceUpSampling"),
    ]
    for mod_name, comp_name, up_name in candidate_imports:
        try:
            mod = __import__(mod_name, fromlist=[comp_name, up_name])
            Compressor = getattr(mod, comp_name)
            UpSampler = getattr(mod, up_name)
            return Compressor, UpSampler
        except Exception as e:
            import_errors.append(f"{mod_name}: {e!r}")
    raise ImportError(
        "Could not import Joyce compressor/upsampler modules. Tried:\n  - "
        + "\n  - ".join(import_errors)
    )


# ------------------------------------------------------------------------------------
# Model "split" helpers for LLaMA-style models
# ------------------------------------------------------------------------------------

@dataclass
class SplitHandles:
    embed_tokens: nn.Module
    layers: List[nn.Module]
    norm: nn.Module
    lm_head: nn.Linear
    hidden_size: int


def _get_split_handles(base: nn.Module) -> SplitHandles:
    """
    Extracts references to embed -> layers -> norm -> lm_head from a LLaMA-like model.
    Works for Hugging Face LLaMA-based classes where `base.model` holds the stack.
    """
    if not hasattr(base, "model"):
        raise AttributeError("Base model missing `.model` attribute (expected HF LLaMA-like).")
    core = base.model
    for attr in ["embed_tokens", "layers", "norm"]:
        if not hasattr(core, attr):
            raise AttributeError(f"Base model `.model` missing `{attr}` attribute.")
    if not hasattr(base, "lm_head"):
        raise AttributeError("Base model missing `lm_head` attribute.")
    layers = core.layers
    hidden_size = getattr(core, "hidden_size", None) or getattr(core, "hidden_dim", None)
    if hidden_size is None:
        # try to infer from first layer norm or an attn dim
        try:
            sample = next(core.layers[0].parameters())
            hidden_size = sample.shape[-1]
        except Exception:
            raise AttributeError("Could not infer hidden size; please set `core.hidden_size`.")
    return SplitHandles(
        embed_tokens=core.embed_tokens,
        layers=list(layers),
        norm=core.norm,
        lm_head=base.lm_head,
        hidden_size=hidden_size,
    )


def _maybe_get_block_argspec(block: nn.Module):
    sig = inspect.signature(block.forward)
    return sig


def _block_forward(block: nn.Module, hidden_states: torch.Tensor,
                   attention_mask: Optional[torch.Tensor] = None,
                   position_ids: Optional[torch.Tensor] = None,
                   **kwargs) -> torch.Tensor:
    """
    Call a transformer block in a way that tolerates different forward signatures.
    """
    sig = _maybe_get_block_argspec(block)
    fwd_kwargs = {}
    if "attention_mask" in sig.parameters:
        fwd_kwargs["attention_mask"] = attention_mask
    if "position_ids" in sig.parameters:
        fwd_kwargs["position_ids"] = position_ids
    # common flags
    if "use_cache" in sig.parameters:
        fwd_kwargs["use_cache"] = False
    if "past_key_value" in sig.parameters:
        fwd_kwargs["past_key_value"] = None
    # allow any extra kwargs to flow if supported
    for k, v in kwargs.items():
        if k in sig.parameters:
            fwd_kwargs[k] = v
    out = block(hidden_states, **fwd_kwargs)
    # Some blocks return tuple (hidden_states, ...) and others just hidden_states
    if isinstance(out, tuple):
        return out[0]
    return out


def run_layers_range(handles: SplitHandles,
                     x: torch.Tensor,
                     start: int,
                     end: int,
                     attention_mask: Optional[torch.Tensor] = None,
                     position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Run a contiguous range of transformer blocks [start, end) on `x`.
    """
    for i in range(start, end):
        x = _block_forward(handles.layers[i], x, attention_mask=attention_mask, position_ids=position_ids)
    return x


def build_causal_mask(B: int, S: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Standard causal mask for sequence length S. Shape (B, 1, S, S) in HF convention.
    """
    mask = torch.full((S, S), fill_value=-float("inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)  # upper triangular (i<j allowed)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)
    return mask


def build_causal_mask_with_prefix(B: int, T: int, S: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Causal mask for concatenated sequence [C (T); B (S)].
    - C tokens can attend among themselves (lower triangle in T-by-T).
    - B tokens can attend to all C and previous B tokens.
    Shape (B,1,T+S,T+S).
    """
    total = T + S
    m = torch.full((total, total), fill_value=-float("inf"), device=device, dtype=dtype)
    # Allow attention to previous positions (including same group)
    m = torch.triu(m, diagonal=1)  # (i<j) masked
    m = m.clone()
    # No special changes needed; the standard causal mask already allows positions i to attend to [0..i].
    # With [C;B], B positions (>=T) can attend to C (indices < T) and previous B positions.
    m = m.unsqueeze(0).unsqueeze(0).expand(B, 1, total, total)
    return m


# ------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------

class StreamingTextDataset(torch.utils.data.IterableDataset):
    """
    Minimal iterable dataset that yields tokenized, concatenated chunks of length 2*seq_len.
    Uses Hugging Face datasets streaming for large corpora if available.
    """
    def __init__(self, tokenizer, dataset_name_or_path: str, split: str,
                 seq_len: int, seed: int = 42, text_key: str = "text"):
        super().__init__()
        assert HAS_DATASETS, "Install `datasets` to use StreamingTextDataset."
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.total_len = 2 * seq_len
        self.text_key = text_key
        self.ds = load_dataset(dataset_name_or_path, split=split, streaming=True)
        self.rng = random.Random(seed)

    def __iter__(self):
        buffer = []
        for ex in self.ds:
            txt = ex.get(self.text_key, None)
            if not txt:
                continue
            ids = self.tokenizer(txt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            buffer.extend(ids)
            while len(buffer) >= self.total_len + 1:
                chunk = buffer[: self.total_len + 1]  # +1 for next-token shift
                buffer = buffer[self.total_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def collate_batch(examples: List[dict]) -> dict:
    # stack to (B, 2S) for both input_ids and labels
    input_ids = torch.stack([e["input_ids"] for e in examples], dim=0)
    labels = torch.stack([e["labels"] for e in examples], dim=0)
    return {"input_ids": input_ids, "labels": labels}


# ------------------------------------------------------------------------------------
# Training step implementing Joyce combined losses
# ------------------------------------------------------------------------------------

def joint_forward_loss(
    base,
    handles: SplitHandles,
    compressor: nn.Module,
    upsampler: nn.Module,
    input_ids: torch.Tensor,   # (B, 2S)
    labels: torch.Tensor,      # (B, 2S)
    L: int,
    num_compressed_states: int,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the two-branch joint loss described in the Joyce write-up.
    Returns (total_loss, metrics_dict).
    """
    device = input_ids.device
    B, twoS = input_ids.shape
    assert twoS % 2 == 0, "Expect even sequence length."
    S = twoS // 2
    T = num_compressed_states

    # Split into A (first S) and B (last S)
    ids_A = input_ids[:, :S]                  # (B,S)
    ids_B = input_ids[:, S:]                  # (B,S)
    labels_A = labels[:, :S]                  # (B,S)
    labels_B = labels[:, S:]                  # (B,S)

    # Embeddings
    # NOTE: many LLaMA-like models embed in core.model.embed_tokens
    x_A = handles.embed_tokens(ids_A)         # (B,S,d)
    x_B = handles.embed_tokens(ids_B)         # (B,S,d)

    # Build causal masks for lower stack (A-only and B-only)
    attn_A = build_causal_mask(B, S, device=device, dtype=x_A.dtype)
    # For B lower-stack, we also want a causal mask of size S, but with position offset S.
    # Position IDs: (B,S) with offset
    position_ids_A = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
    position_ids_B = torch.arange(S, 2*S, device=device).unsqueeze(0).expand(B, S)

    # Run 0..L on A and B separately (avoid leakage below L)
    h_A_L = run_layers_range(handles, x_A, start=0, end=L, attention_mask=attn_A, position_ids=position_ids_A)
    h_B_L = run_layers_range(handles, x_B, start=0, end=L, attention_mask=attn_A, position_ids=position_ids_B)

    # Compress A
    C = compressor(h_A_L)                     # (B,T,d)

    # ---------------------- Branch 1: A -> compress -> upsample -> top ----------------------
    U_A = upsampler(C, seq_len=S)             # (B,S,d)
    # Run L..end on U_A (include block L if your "insert" is AFTER L; we proceed with L+1)
    h_A_top = run_layers_range(handles, U_A, start=L, end=len(handles.layers),
                               attention_mask=attn_A, position_ids=position_ids_A)
    # head
    h_A_top = handles.norm(h_A_top)           # (B,S,d)
    logits_A = handles.lm_head(h_A_top)       # (B,S,V)

    # Compute NTP loss for A
    # shift: predict labels_A[:,1:] from logits_A[:,:-1]
    logits_A_shift = logits_A[:, :-1, :].contiguous()
    labels_A_shift = labels_A[:, 1:].contiguous()
    loss_A = F.cross_entropy(
        logits_A_shift.view(-1, logits_A_shift.size(-1)),
        labels_A_shift.view(-1),
        ignore_index=ignore_index
    )

    # ---------------------- Branch 2: [C ; B] at layer L -> top -----------------------------
    # Build causal mask over T+S
    attn_CB = build_causal_mask_with_prefix(B, T=T, S=S, device=device, dtype=h_B_L.dtype)
    # position ids for the concatenated stream above L: we keep monotonic positions
    position_ids_CB = torch.arange(0, T + S, device=device).unsqueeze(0).expand(B, T + S)

    # Concat along sequence dimension
    H_CB_L = torch.cat([C, h_B_L], dim=1)     # (B, T+S, d)

    # Run top stack
    h_CB_top = run_layers_range(handles, H_CB_L, start=L, end=len(handles.layers),
                                attention_mask=attn_CB, position_ids=position_ids_CB)
    h_CB_top = handles.norm(h_CB_top)         # (B, T+S, d)
    logits_CB = handles.lm_head(h_CB_top)     # (B, T+S, V)

    # Only compute loss on the B region (skip the prefix C)
    logits_B = logits_CB[:, T:, :]            # (B, S, V)
    logits_B_shift = logits_B[:, :-1, :].contiguous()
    labels_B_shift = labels_B[:, 1:].contiguous()
    loss_B = F.cross_entropy(
        logits_B_shift.view(-1, logits_B_shift.size(-1)),
        labels_B_shift.view(-1),
        ignore_index=ignore_index
    )

    total_loss = loss_A + loss_B
    with torch.no_grad():
        ppl_A = torch.exp(torch.clamp(loss_A, max=80.0))
        ppl_B = torch.exp(torch.clamp(loss_B, max=80.0))
        ppl = torch.exp(torch.clamp(total_loss, max=80.0))
    metrics = {
        "loss_A": loss_A.detach(),
        "loss_B": loss_B.detach(),
        "loss": total_loss.detach(),
        "ppl_A": ppl_A.detach(),
        "ppl_B": ppl_B.detach(),
        "ppl": ppl.detach(),
    }
    return total_loss, metrics


# ------------------------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Joyce Joint Training (Combined Stage)")
    parser.add_argument("--repo_root", type=str, default=".", help="Path to adaptive-attention repo root.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF base model checkpoint (pretrained).")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="HF tokenizer (defaults to model).")
    parser.add_argument("--dataset", type=str, default=None, help="HF dataset name or path. If None, script expects you to provide your own dataloader.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--seq_len", type=int, default=1024, help="S (half-length). Actual tokens per sample = 2*S.")
    parser.add_argument("--layer_L", type=int, required=True, help="Index of layer after which we insert Joyce (0-based).")
    parser.add_argument("--num_compressed_states", type=int, default=64, help="T (compressed token count).")
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="checkpoints_joint")
    parser.add_argument("--compressor_ckpt", type=str, default=None, help="Optional path to load compressor weights.")
    parser.add_argument("--upsampler_ckpt", type=str, default=None, help="Optional path to load upsampler weights.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed (NCCL).")
    parser.add_argument("--compile", action="store_true", help="torch.compile the joint step.")
    args = parser.parse_args()

    setup_ddp(args)
    if is_main(args):
        os.makedirs(args.save_dir, exist_ok=True)

    # Repro
    if HAS_TRANSFORMERS:
        set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Add bento to import path
    add_bento_to_sys_path(args.repo_root)

    # Load Joyce modules
    CompressorCls, UpSamplerCls = _load_joyce_modules(args)

    # Load base model + tokenizer
    assert HAS_TRANSFORMERS, "Install transformers to run this script."
    tok_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    base.train()

    # Split handles
    handles = _get_split_handles(base)

    # Instantiate Joyce blocks
    compressor = CompressorCls(hidden_size=handles.hidden_size, num_compressed_states=args.num_compressed_states)
    upsampler  = UpSamplerCls(hidden_size=handles.hidden_size, num_compressed_states=args.num_compressed_states)

    # Optionally load pre-trained weights
    if args.compressor_ckpt and os.path.isfile(args.compressor_ckpt):
        sd = torch.load(args.compressor_ckpt, map_location="cpu")
        compressor.load_state_dict(sd, strict=False)
    if args.upsampler_ckpt and os.path.isfile(args.upsampler_ckpt):
        sd = torch.load(args.upsampler_ckpt, map_location="cpu")
        upsampler.load_state_dict(sd, strict=False)

    # Move to device(s)
    device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    base.to(device)
    compressor.to(device)
    upsampler.to(device)

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # Optimizer (base + compressor + upsampler)
    # Weight decay only for non-norm, non-bias params
    def param_groups(module: nn.Module):
        decay, no_decay = [], []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in ["bias", "norm", "ln_", "layernorm", "LayerNorm"]):
                no_decay.append(p)
            else:
                decay.append(p)
        return [{"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    optim = torch.optim.AdamW(
        param_groups(base) + param_groups(compressor) + param_groups(upsampler),
        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps
    )

    # Cosine LR with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        progress = float(step - args.warmup_steps) / max(1, args.train_steps - args.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # Data
    if args.dataset is None:
        raise ValueError("Please provide --dataset (HF datasets name or local dataset script).")
    ds = StreamingTextDataset(tokenizer, args.dataset, split=args.dataset_split,
                              seq_len=args.seq_len, seed=args.seed, text_key=args.text_key)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_batch)

    # Optionally compile the joint forward for speed
    joint_fn = joint_forward_loss
    if args.compile and hasattr(torch, "compile"):
        joint_fn = torch.compile(joint_fn, mode="max-autotune")

    # Training loop
    step = 0
    running = {"loss": 0.0, "loss_A": 0.0, "loss_B": 0.0}
    start_time = time.time()

    base.zero_grad(set_to_none=True)
    compressor.zero_grad(set_to_none=True)
    upsampler.zero_grad(set_to_none=True)

    for batch in dl:
        if step >= args.train_steps:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        # FP16 mixed precision if requested
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, metrics = joint_fn(
                    base, handles, compressor, upsampler,
                    input_ids, labels, L=args.layer_L, num_compressed_states=args.num_compressed_states
                )
            scaler.scale(loss / args.grad_accum).backward()
        else:
            loss, metrics = joint_fn(
                base, handles, compressor, upsampler,
                input_ids, labels, L=args.layer_L, num_compressed_states=args.num_compressed_states
            )
            (loss / args.grad_accum).backward()

        # Log running averages (main process only)
        running["loss"] += metrics["loss"].item()
        running["loss_A"] += metrics["loss_A"].item()
        running["loss_B"] += metrics["loss_B"].item()

        if (step + 1) % args.grad_accum == 0:
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
            scheduler.step()

        if is_main(args) and (step % 20 == 0):
            elapsed = time.time() - start_time
            avg_loss = running["loss"] / max(1, step + 1)
            avg_A = running["loss_A"] / max(1, step + 1)
            avg_B = running["loss_B"] / max(1, step + 1)
            lr_now = scheduler.get_last_lr()[0]
            print(f"[step {step:06d}] loss={avg_loss:.4f} (A={avg_A:.4f}, B={avg_B:.4f})  lr={lr_now:.2e}  elapsed={elapsed:.1f}s")

        if is_main(args) and (step > 0) and (step % args.save_every == 0):
            ckpt_dir = os.path.join(args.save_dir, f"step_{step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(base.state_dict(), os.path.join(ckpt_dir, "base.pt"))
            torch.save(compressor.state_dict(), os.path.join(ckpt_dir, "compressor.pt"))
            torch.save(upsampler.state_dict(), os.path.join(ckpt_dir, "upsampler.pt"))
            with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            print(f"Saved checkpoint to: {ckpt_dir}")

        step += 1

    # Final save
    if is_main(args):
        ckpt_dir = os.path.join(args.save_dir, "final")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(base.state_dict(), os.path.join(ckpt_dir, "base.pt"))
        torch.save(compressor.state_dict(), os.path.join(ckpt_dir, "compressor.pt"))
        torch.save(upsampler.state_dict(), os.path.join(ckpt_dir, "upsampler.pt"))
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Training completed. Final checkpoint -> {ckpt_dir}")


if __name__ == "__main__":
    main()
