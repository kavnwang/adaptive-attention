import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

from fla.models.joyce.autoencoder import JoyceAutoEncoder, JoyceConfig


def parse_args():
    p = argparse.ArgumentParser("Joyce pre-training (auto-encoder on hidden states at layer L)")
    p.add_argument("--model.ckpt", type=str, required=True, help="Path or HF repo of the base transformer checkpoint.")
    p.add_argument("--model.config", type=str, required=True, help="JSON config containing JoyceConfig + runtime args.")
    p.add_argument("--model.tokenizer_path", type=str, required=True)

    # data
    p.add_argument("--training.dataset", type=str, default="/cache/ufw/Ultra-FineWeb")
    p.add_argument("--training.dataset_name", type=str, default="sample-100BT")
    p.add_argument("--training.dataset_split", type=str, default="en")

    # optim
    p.add_argument("--optimizer.name", type=str, default="AdamW")
    p.add_argument("--optimizer.lr", type=float, default=1e-3)
    p.add_argument("--optimizer.eps", type=float, default=1e-8)
    p.add_argument("--optimizer.wd", type=float, default=0.0)
    p.add_argument("--lr_scheduler.warmup_steps", type=int, default=1024)
    p.add_argument("--lr_scheduler.decay_type", type=str, default="cosine")
    p.add_argument("--lr_scheduler.lr_min", type=float, default=0.1)

    # train
    p.add_argument("--training.batch_size", type=int, default=1)
    p.add_argument("--training.seq_len", type=int, default=4096)
    p.add_argument("--training.steps", type=int, default=20480)
    p.add_argument("--training.gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--training.compile", action="store_true")
    p.add_argument("--training.max_norm", type=float, default=1.0)
    p.add_argument("--training.seed", type=int, default=42)
    p.add_argument("--training.num_workers", type=int, default=8)
    p.add_argument("--training.prefetch_factor", type=int, default=2)
    p.add_argument("--training.skip_nan_inf", action="store_true")

    # logging/checkpoint
    p.add_argument("--job.dump_folder", type=str, default="exp/joyce_pretrain")
    p.add_argument("--checkpoint.interval", type=int, default=2048)
    p.add_argument("--checkpoint.keep_latest_k", type=int, default=2)

    return p.parse_args()


def build_tokenizer(path: str):
    return AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)


def build_base_model(ckpt: str):
    """
    Load your base transformer in eval/frozen mode.
    If you maintain HF conversion utilities, this can be from HF hub or local.
    """
    # This example assumes a HF-compatible model class with `from_pretrained`.
    # Replace with your repo's loader if different (e.g., bento loader).
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.requires_grad_(False).eval()
    return model


def make_dataloader(tokenizer, args):
    seq_len = args.training_seq_len if hasattr(args, "training_seq_len") else args.__dict__["training.seq_len"]
    name = args.__dict__["training.dataset_name"]
    split = args.__dict__["training.dataset_split"]
    root = args.__dict__["training.dataset"]

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=name, split=split, cache_dir=root)

    def _tok(batch):
        return tokenizer(
            batch["text"],
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    ds = ds.map(_tok, batched=True, num_proc=8, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=args.__dict__["training.batch_size"], shuffle=True,
                      num_workers=args.__dict__["training.num_workers"],
                      prefetch_factor=args.__dict__["training.prefetch_factor"], pin_memory=True, drop_last=True)


def cosine_lr(step, warmup, total_steps, base_lr, lr_min):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total_steps - warmup)
    return lr_min + 0.5 * (base_lr - lr_min) * (1 + torch.cos(torch.tensor(torch.pi * p)))


def main():
    args = parse_args()

    # Read JoyceConfig (JSON)
    cfg_path = Path(args.__dict__["model.config"])
    jcfg_dict = json.loads(cfg_path.read_text())
    jcfg = JoyceConfig(**jcfg_dict)

    torch.manual_seed(args.__dict__["training.seed"])

    tok = build_tokenizer(args.__dict__["model.tokenizer_path"])
    base = build_base_model(args.__dict__["model.ckpt"])

    model = JoyceAutoEncoder(base, jcfg)

    if jcfg.compile or args.__dict__["training.compile"]:
        model.compressor = torch.compile(model.compressor)
        model.upsampler = torch.compile(model.upsampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dl = make_dataloader(tok, args)

    # Optim on Joyce parameters only
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.__dict__["optimizer.lr"],
        eps=args.__dict__["optimizer.eps"],
        weight_decay=args.__dict__["optimizer.wd"],
    )

    total_steps = args.__dict__["training.steps"]
    warmup = args.__dict__["lr_scheduler.warmup_steps"]
    base_lr = args.__dict__["optimizer.lr"]
    lr_min = base_lr * args.__dict__["lr_scheduler.lr_min"]

    dump = Path(args.__dict__["job.dump_folder"])
    dump.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(jcfg.amp == "fp16"))
    use_autocast = jcfg.amp in ("bf16", "fp16")
    autocast_dtype = torch.bfloat16 if jcfg.amp == "bf16" else torch.float16

    step = 0
    model.train()
    for batch in dl:
        if step >= total_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        for pgroup in optim.param_groups:
            pgroup["lr"] = float(cosine_lr(step, warmup, total_steps, base_lr, lr_min))

        try:
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        except (FloatingPointError, RuntimeError) as e:
            if args.__dict__["training.skip_nan_inf"]:
                print(f"[warn] Skip step {step} due to FP error: {e}")
                step += 1
                continue
            raise

        scaler.scale(loss).backward()

        if (step + 1) % args.__dict__["training.gradient_accumulation_steps"] == 0:
            if args.__dict__["training.max_norm"] > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.__dict__["training.max_norm"])
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        if step % args.__dict__["checkpoint.interval"] == 0:
            ckpt_path = dump / f"joyce_pretrain_step{step}.pt"
            torch.save({"step": step, "state_dict": model.state_dict(), "cfg": jcfg.__dict__}, ckpt_path)
            # logs
            with open(dump / "train.log", "a") as f:
                f.write(f"{step}\t{loss.item():.6f}\n")

        step += 1

    # final save
    ckpt_path = dump / f"joyce_pretrain_final.pt"
    torch.save({"step": step, "state_dict": model.state_dict(), "cfg": jcfg.__dict__}, ckpt_path)
    print(f"[done] saved to {ckpt_path}")


if __name__ == "__main__":
    main()
