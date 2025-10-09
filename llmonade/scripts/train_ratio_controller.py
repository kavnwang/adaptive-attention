# llmonade/scripts/train_ratio_controller.py
from __future__ import annotations
import argparse, glob, os, math
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm

from llmonade.modules.joyce_controller.ratio_controller import JoyceRatioController
from llmonade.modules.joyce_controller.dataset import DeltaCurveShardDataset

def collate(batch):
    # Pad S to the max T' in the batch if variable-length; else stack directly.
    Ts = [b["S"].shape[0] for b in batch]
    D = batch[0]["S"].shape[1]
    Tm = max(Ts)
    S_pad = []
    for b in batch:
        S = b["S"]
        if S.shape[0] < Tm:
            pad = torch.zeros(Tm - S.shape[0], D, dtype=S.dtype)
            S = torch.cat([S, pad], dim=0)
        S_pad.append(S)
    S = torch.stack(S_pad, dim=0)  # [B, Tm, D]
    delta = torch.stack([b["delta_star"] for b in batch], dim=0)  # [B]
    r = torch.stack([b["r_star"] for b in batch], dim=0)          # [B]
    return {"S": S, "delta": delta, "r": r}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shard_paths = sorted(glob.glob(os.path.join(args.curves_dir, "*.pt")))
    ds = DeltaCurveShardDataset(shard_paths, budget_sampling=args.budget_sampling,
                                n_budgets_per_curve=args.n_budgets_per_curve,
                                fixed_budgets=None, normalize_delta=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)

    # infer D_s from one sample
    probe = ds[0]["S"]
    d_shrunken = probe.shape[-1]
    model = JoyceRatioController(
        d_shrunken=d_shrunken,
        d_model=args.d_model,
        delta_emb_dim=args.delta_emb_dim,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses, maes = [], []
        for batch in tqdm(dl, desc=f"Train epoch {epoch}"):
            S = batch["S"].to(device)      # [B, T', D_s]
            delta = batch["delta"].to(device)  # [B]
            r_tgt = batch["r"].to(device)  # [B]

            # Controller expects *unnormalized* delta; dataset normalized by mean/std.
            # Undo normalization for embedding:
            if hasattr(ds, "delta_mean"):
                delta_unnorm = delta * ds.delta_std + ds.delta_mean
            else:
                delta_unnorm = delta

            r_pred = model(S, delta_unnorm)          # [B]
            loss = F.mse_loss(r_pred, r_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()

            losses.append(loss.item())
            maes.append(torch.mean(torch.abs(r_pred.detach() - r_tgt)).item())

        sched.step()
        msg = f"epoch={epoch} loss={sum(losses)/len(losses):.4f} mae={sum(maes)/len(maes):.4f} lr={sched.get_last_lr()[0]:.2e}"
        print(msg)

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "delta_mean": ds.delta_mean,
            "delta_std": ds.delta_std,
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"controller_epoch{epoch:03d}.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--curves_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--delta_emb_dim", type=int, default=64)
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--budget_sampling", type=str, choices=["quantiles", "uniform", "median"], default="quantiles")
    p.add_argument("--n_budgets_per_curve", type=int, default=1)
    args = p.parse_args()
    main(args)
