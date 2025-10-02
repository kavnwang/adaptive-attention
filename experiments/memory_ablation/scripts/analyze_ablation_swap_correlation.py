#!/usr/bin/env python3
"""
Analyze correlation between ablation ease and swap ease in memory keys.
Compares ablation_results.json and swap_results.json to determine if keys
that are easy to ablate are also easy to swap.
"""

import json
from collections import defaultdict
from typing import Dict
import numpy as np


def load_ablation_results(path: str) -> Dict:
    """Load and parse ablation results."""
    with open(path, "r") as f:
        data = json.load(f)

    # Extract QA -> ablation success rate mapping
    qa_ablation_stats = defaultdict(lambda: {"successful": 0, "total": 0})

    for entry in data:
        qa = entry["question"].replace("Q: Where is ", "").replace("?", "").strip()
        answer = entry["answer"]

        # Check each ablation attempt
        for ablation in entry.get("ablations", []):
            qa_ablation_stats[qa]["total"] += 1
            if ablation["correct_generation"]:
                qa_ablation_stats[qa]["successful"] += 1

    # Calculate success rates
    qa_ablation_rates = {}
    for qa, stats in qa_ablation_stats.items():
        if stats["total"] > 0:
            qa_ablation_rates[qa] = stats["successful"] / stats["total"]

    return qa_ablation_rates


def load_swap_results(path: str) -> Dict:
    """Load and parse swap results."""
    with open(path, "r") as f:
        data = json.load(f)

    # Extract QA -> swap success mapping
    qa_swap_stats = defaultdict(lambda: {"swapped": 0, "total": 0})

    for result in data["results"]:
        qa1 = (
            result["qa1"]["question"]
            .replace("Q: Where is ", "")
            .replace("?", "")
            .strip()
        )
        qa2 = (
            result["qa2"]["question"]
            .replace("Q: Where is ", "")
            .replace("?", "")
            .strip()
        )

        # Process QA1 swaps
        answer_tokens_len = len(result["qa1"]["answer_tokens"])
        num_swapped = result["qa1"]["num_swapped"]
        qa_swap_stats[qa1]["total"] += answer_tokens_len
        qa_swap_stats[qa1]["swapped"] += num_swapped

        # Process QA2 swaps
        answer_tokens_len = len(result["qa2"]["answer_tokens"])
        num_swapped = result["qa2"]["num_swapped"]
        qa_swap_stats[qa2]["total"] += answer_tokens_len
        qa_swap_stats[qa2]["swapped"] += num_swapped

    # Calculate swap rates
    qa_swap_rates = {}
    for qa, stats in qa_swap_stats.items():
        if stats["total"] > 0:
            qa_swap_rates[qa] = stats["swapped"] / stats["total"]

    return qa_swap_rates


def analyze_correlation(ablation_rates: Dict, swap_rates: Dict) -> None:
    """Analyze correlation between ablation and swap rates."""
    # Find common QAs
    common_qas = set(ablation_rates.keys()) & set(swap_rates.keys())
    print(f"Found {len(common_qas)} QAs in both datasets")

    if len(common_qas) == 0:
        print("No common QAs found between datasets!")
        return

    # Extract paired rates
    ablation_values = []
    swap_values = []

    for qa in common_qas:
        ablation_values.append(ablation_rates[qa])
        swap_values.append(swap_rates[qa])

    ablation_values = np.array(ablation_values)
    swap_values = np.array(swap_values)

    # Calculate correlation
    correlation = np.corrcoef(ablation_values, swap_values)[0, 1]
    print(f"\nPearson correlation between ablation and swap rates: {correlation:.4f}")

    # Analyze by quartiles
    print("\nAnalysis by ablation difficulty quartiles:")
    print("-" * 50)

    # Sort by ablation rate
    sorted_indices = np.argsort(ablation_values)
    quartile_size = len(sorted_indices) // 4

    for i in range(4):
        start_idx = i * quartile_size
        end_idx = (i + 1) * quartile_size if i < 3 else len(sorted_indices)

        quartile_indices = sorted_indices[start_idx:end_idx]
        quartile_ablation = ablation_values[quartile_indices]
        quartile_swap = swap_values[quartile_indices]

        print(
            f"\nQuartile {i + 1} (ablation rate {quartile_ablation.min():.3f} - {quartile_ablation.max():.3f}):"
        )
        print(f"  Mean ablation rate: {quartile_ablation.mean():.3f}")
        print(f"  Mean swap rate: {quartile_swap.mean():.3f}")
        print(
            f"  Correlation within quartile: {np.corrcoef(quartile_ablation, quartile_swap)[0, 1]:.3f}"
        )

    # Identify outliers
    print("\n\nOutlier Analysis:")
    print("-" * 50)

    # High ablation but low swap (easy to ablate but hard to swap)
    high_ablation_low_swap = []
    # Low ablation but high swap (hard to ablate but easy to swap)
    low_ablation_high_swap = []

    for i, qa in enumerate(common_qas):
        abl_rate = ablation_rates[qa]
        swap_rate = swap_rates[qa]

        if abl_rate > 0.7 and swap_rate < 0.3:
            high_ablation_low_swap.append((qa, abl_rate, swap_rate))
        elif abl_rate < 0.3 and swap_rate > 0.7:
            low_ablation_high_swap.append((qa, abl_rate, swap_rate))

    print(f"\nEasy to ablate but hard to swap ({len(high_ablation_low_swap)} QAs):")
    for qa, abl, swap in sorted(
        high_ablation_low_swap, key=lambda x: x[1] - x[2], reverse=True
    )[:10]:
        print(f"  {qa}: ablation={abl:.3f}, swap={swap:.3f}")

    print(f"\nHard to ablate but easy to swap ({len(low_ablation_high_swap)} QAs):")
    for qa, abl, swap in sorted(
        low_ablation_high_swap, key=lambda x: x[2] - x[1], reverse=True
    )[:10]:
        print(f"  {qa}: ablation={abl:.3f}, swap={swap:.3f}")

    # Overall statistics
    print("\n\nOverall Statistics:")
    print("-" * 50)
    print(
        f"Mean ablation rate: {ablation_values.mean():.3f} (±{ablation_values.std():.3f})"
    )
    print(f"Mean swap rate: {swap_values.mean():.3f} (±{swap_values.std():.3f})")

    # Check if generally easy-to-ablate keys are easy to swap
    threshold = 0.5
    easy_ablate = ablation_values > threshold
    easy_swap = swap_values > threshold

    both_easy = np.sum(easy_ablate & easy_swap)
    both_hard = np.sum(~easy_ablate & ~easy_swap)
    consistent = both_easy + both_hard

    print(f"\nConsistency analysis (threshold={threshold}):")
    print(f"  Both easy: {both_easy} ({both_easy / len(common_qas) * 100:.1f}%)")
    print(f"  Both hard: {both_hard} ({both_hard / len(common_qas) * 100:.1f}%)")
    print(f"  Consistent: {consistent} ({consistent / len(common_qas) * 100:.1f}%)")


def main():
    # Paths to data files
    ablation_path = "experiments/synthetic_qa/results/ablation_results/memory_1layer_2K/ablation_results.json"
    swap_path = "memory_1layer_2K_no_mlp_swap_results.json"

    print("Loading ablation results...")
    ablation_rates = load_ablation_results(ablation_path)
    print(f"Loaded ablation data for {len(ablation_rates)} QAs")

    print("\nLoading swap results...")
    swap_rates = load_swap_results(swap_path)
    print(f"Loaded swap data for {len(swap_rates)} QAs")

    print("\nAnalyzing correlation...")
    analyze_correlation(ablation_rates, swap_rates)


if __name__ == "__main__":
    main()
