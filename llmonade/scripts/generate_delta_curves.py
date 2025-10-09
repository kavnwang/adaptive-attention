# llmonade/scripts/generate_delta_curves.py
from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# import your model + tokenizer + data pipeline
# from llmonade.models.joyce import JoyceModel
# from llmonade.data.datasets import PromptDataset
# from llmonade.models.joyce.upsampling_init import UpsamplingInit

# ---------------------- helpers ----------------------

def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], targets: [B, T]
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
    return loss  # nats if your CE uses natural log; else multiply by ln(2) if logits are base-2

def set_keep_ratio(model, r: float):
    """
    Wire this to your compression module (e.g., number of compressed tokens t = ceil(r * seq_len)
    or whatever control knob you use at layer L).
    """
    if hasattr(model, "set_keep_ratio"):
        model.set_keep_ratio(r)
    else:
        # implement a setter in your compression block and reach it here
        model.compression.set_keep_ratio(r)

@dataclass
class CurveExample:
    shrunken_states: torch.Tensor   # [T', D_s] (single window)
    keep_ratios: torch.Tensor       # [R]
    deltas: torch.Tensor            # [R]  (Δ(r) vs full)
    seq_len: int
    meta: Dict[str, Any]

# ---------------------- main builder ----------------------

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model/tokenizer/dataset per your repo conventions
    # model = JoyceModel.from_config(args.config).to(device).eval()
    # tok = load_tokenizer(args.tokenizer)
    # ds  = PromptDataset(args.dataset, split=args.split)
    # dl  = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    # For demonstration, assume you have a dataloader yielding dicts with:
    #   'input_ids': [1, T], 'labels': [1, T]
    dl = ...  # TODO: plug your dataset

    keep_ratios = torch.tensor([1.00, 0.75, 0.50, 0.25], dtype=torch.float32, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    shard = []
    shard_sz = args.shard_size
    shard_idx = 0

    for batch in tqdm(dl, desc="Δ-curve collection"):
        input_ids: torch.Tensor = batch["input_ids"].to(device)  # [1, T]
        labels:     torch.Tensor = batch["labels"].to(device)     # [1, T]

        # ----- Full pass (no compression; r=1.0 assumed) -----
        set_keep_ratio(model=None, r=1.0)  # if your model treats 1.0 as "no compression", ensure it's respected
        # logits_full, _ = model(input_ids)      # adjust to your API
        # L_full = nll_from_logits(logits_full, labels)
        L_full = ...  # compute per your API

        # Capture the shrunken states for this window from upsampling init:
        # S = model.upsampling_init.get_last_shrunken()[0].detach().cpu()  # [T', D_s]
        S = ...  # obtain from your hook; CPU copy

        # ----- Compressed passes over the grid -----
        deltas = []
        for r in keep_ratios:
            # Skip r=1.0 if you've already computed above; but re-run to capture consistent states if needed
            set_keep_ratio(model=None, r=float(r.item()))
            # logits_r, _ = model(input_ids)
            # L_r = nll_from_logits(logits_r, labels)
            L_r = ...  # per your API
            deltas.append((L_r - L_full).detach().float())

        deltas = torch.stack(deltas, dim=0).cpu()  # [R]

        ex = CurveExample(
            shrunken_states=S.cpu(), keep_ratios=keep_ratios.cpu(),
            deltas=deltas.cpu(), seq_len=int(labels.shape[-1]),
            meta={"doc_id": batch.get("doc_id", None)}
        )
        shard.append(ex)

        if len(shard) >= shard_sz:
            torch.save(shard, os.path.join(args.out_dir, f"curves_{shard_idx:05d}.pt"))
            shard.clear()
            shard_idx += 1

    if shard:
        torch.save(shard, os.path.join(args.out_dir, f"curves_{shard_idx:05d}.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=False)
    p.add_argument("--tokenizer", type=str, required=False)
    p.add_argument("--dataset", type=str, required=False)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--shard_size", type=int, default=256)
    args = p.parse_args()
    main(args)
