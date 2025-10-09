# llmonade/modules/joyce_controller/dataset.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import math
import random
import torch
from torch.utils.data import Dataset

class DeltaCurveShardDataset(Dataset):
    """
    Reads a list of torch shards produced by generate_delta_curves.py, each a list of CurveExample dataclasses.

    For each __getitem__:
      - sample a curve,
      - sample a budget Δ* from a strategy (uniform between min/max Δ, or from a predefined set),
      - compute the label r* = min r s.t. Δ(r) <= Δ*; if none, r* = 1.0 (no compression),
      - return (S, Δ*, r*).
    """
    def __init__(
        self,
        shard_paths: List[str],
        budget_sampling: str = "quantiles",
        n_budgets_per_curve: int = 1,
        fixed_budgets: Optional[List[float]] = None,
        normalize_delta: bool = True,
    ):
        super().__init__()
        self.examples = []
        for sp in shard_paths:
            shard = torch.load(sp, map_location="cpu")
            self.examples.extend(shard)
        self.normalize_delta = normalize_delta
        self.n_budgets = n_budgets_per_curve
        self.fixed_budgets = fixed_budgets
        self.budget_sampling = budget_sampling

        # Precompute dataset-scale mean/std of Δ(r) for normalization of Δ*
        all_deltas = torch.cat([ex.deltas.flatten() for ex in self.examples], dim=0)
        self.delta_mean = all_deltas.mean().item()
        self.delta_std = all_deltas.std().item() + 1e-8

    def __len__(self):
        return len(self.examples) * self.n_budgets

    def _sample_budget(self, deltas: torch.Tensor) -> float:
        # deltas: [R]
        if self.fixed_budgets is not None:
            return random.choice(self.fixed_budgets)
        if self.budget_sampling == "quantiles":
            q = random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
            return torch.quantile(deltas, q).item()
        elif self.budget_sampling == "uniform":
            return float(torch.empty(1).uniform_(float(deltas.min().item()), float(deltas.max().item())).item())
        else:
            return float(torch.median(deltas).item())

    def __getitem__(self, idx: int):
        ex = self.examples[idx // self.n_budgets]
        S: torch.Tensor = ex.shrunken_states.float()     # [T', D_s]
        keep_ratios: torch.Tensor = ex.keep_ratios.float()
        deltas: torch.Tensor = ex.deltas.float()         # [R]
        # sample budget Δ*
        budget = self._sample_budget(deltas)             # float
        # label r* = min r with Δ(r) <= Δ* ; else 1.0 (no compression)
        mask = deltas <= budget + 1e-8
        if mask.any():
            j = int(torch.nonzero(mask, as_tuple=False)[0].item())  # first such r
            r_star = float(keep_ratios[j].item())
        else:
            r_star = 1.0

        delta_star = torch.tensor([budget], dtype=torch.float32)
        if self.normalize_delta:
            delta_star = (delta_star - self.delta_mean) / self.delta_std

        return {
            "S": S,                           # [T', D_s]
            "delta_star": delta_star.squeeze(0),  # scalar
            "r_star": torch.tensor([r_star], dtype=torch.float32).squeeze(0),
        }
