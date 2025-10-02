#!/usr/bin/env python3
"""
Analyze ablation success rates by token position in the answer sequence.
"""

import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def analyze_ablations_by_position(ablation_file: str):
    """Analyze ablation success rates by token position."""

    print("Loading ablation results...")
    with open(ablation_file, "r") as f:
        data = json.load(f)

    # Track ablations by position
    position_stats = defaultdict(lambda: {"successful": 0, "total": 0})

    # Also track which positions were ablated for each attempt
    ablation_details = []

    for entry in data:
        for ablation in entry.get("ablations", []):
            # Find which position was ablated by comparing clean vs ablated activations
            ablated_positions = []

            # Check each layer's activations
            for layer_idx, layer_activations in ablation["ablated_activations"].items():
                for activation in layer_activations:
                    if activation.get("ablated", False):
                        position = activation["position"]
                        ablated_positions.append(position)

                        # Record statistics
                        position_stats[position]["total"] += 1
                        if ablation["correct_generation"]:
                            position_stats[position]["successful"] += 1

            ablation_details.append(
                {
                    "positions": ablated_positions,
                    "success": ablation["correct_generation"],
                }
            )

    # Calculate success rates by position
    positions = sorted(position_stats.keys())
    success_rates = []
    totals = []

    print("\nAblation success rates by token position:")
    print("-" * 50)
    print("Position | Success Rate | Total Ablations")
    print("-" * 50)

    for pos in positions:
        stats = position_stats[pos]
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        success_rates.append(rate)
        totals.append(stats["total"])

        print(f"{pos:8} | {rate:11.1%} | {stats['total']:15,}")

    # Plot results
    plt.figure(figsize=(12, 6))

    # Subplot 1: Success rates by position
    plt.subplot(1, 2, 1)
    bars = plt.bar(
        positions, [r * 100 for r in success_rates], color="skyblue", edgecolor="navy"
    )
    plt.xlabel("Token Position")
    plt.ylabel("Ablation Success Rate (%)")
    plt.title("Ablation Success Rate by Token Position")
    plt.ylim(0, 100)

    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Subplot 2: Number of ablations by position
    plt.subplot(1, 2, 2)
    plt.bar(positions, totals, color="lightcoral", edgecolor="darkred")
    plt.xlabel("Token Position")
    plt.ylabel("Number of Ablations")
    plt.title("Total Ablations by Token Position")

    plt.tight_layout()
    plt.savefig("ablation_by_position.png", dpi=150)
    print("\nPlot saved to ablation_by_position.png")

    # Additional analysis: multi-position ablations
    print("\n\nMulti-position ablation analysis:")
    print("-" * 50)

    multi_position_count = defaultdict(int)
    multi_position_success = defaultdict(int)

    for detail in ablation_details:
        num_positions = len(
            set(detail["positions"])
        )  # Use set to count unique positions
        multi_position_count[num_positions] += 1
        if detail["success"]:
            multi_position_success[num_positions] += 1

    print("Positions Ablated | Count | Success Rate")
    print("-" * 50)
    for num_pos in sorted(multi_position_count.keys()):
        count = multi_position_count[num_pos]
        success = multi_position_success[num_pos]
        rate = success / count if count > 0 else 0
        print(f"{num_pos:17} | {count:5} | {rate:11.1%}")

    # Analyze answer length distribution
    answer_lengths = []
    for entry in data:
        # Count tokens in answer (approximate by counting underscores + 1)
        answer_token_count = entry["answer"].count("_") + 1
        answer_lengths.append(answer_token_count)

    print("\n\nAnswer token statistics:")
    print(f"  Mean length: {np.mean(answer_lengths):.1f} tokens")
    print(f"  Std dev: {np.std(answer_lengths):.1f} tokens")
    print(f"  Min: {np.min(answer_lengths)} tokens")
    print(f"  Max: {np.max(answer_lengths)} tokens")

    return position_stats


def main():
    ablation_file = "experiments/synthetic_qa/results/ablation_results/memory_1layer_2K/ablation_results.json"
    analyze_ablations_by_position(ablation_file)


if __name__ == "__main__":
    main()
