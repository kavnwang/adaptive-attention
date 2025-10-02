#!/usr/bin/env python3
"""
Fix the JSON file and analyze correlation between repeating digits and ablation flips.
"""

import json
import re
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def fix_json_file(input_file, output_file):
    """Fix the malformed JSON file by adding missing closing brackets."""
    with open(input_file, "r") as f:
        content = f.read()

    # Fix clean_logits lines
    content = re.sub(
        r'"clean_logits": \[ ([^\]]*?)\n', r'"clean_logits": [ \1 ],\n', content
    )

    # Fix ablated_logits lines
    content = re.sub(
        r'"ablated_logits": \[ ([^\]]*?)\n', r'"ablated_logits": [ \1 ],\n', content
    )

    # Write fixed content
    with open(output_file, "w") as f:
        f.write(content)

    print(f"Fixed JSON saved to {output_file}")


def parse_ablation_data(json_file):
    """Parse the ablation data and extract relevant information."""
    with open(json_file, "r") as f:
        data = json.load(f)

    results = []

    for item in data:
        question = item["question"]
        answer = item["answer"]
        clean_logits = item["clean_logits"]

        # Extract key from question (e.g., "Q: Where is e174774_?" -> "e174774_")
        key_match = re.search(r"Where is (e\d+_?)\?", question)
        if not key_match:
            continue
        key = key_match.group(1)

        # Clean prediction is based on the argmax of clean_logits
        clean_pred_idx = np.argmax(clean_logits)

        # Process each ablation
        for ablation in item.get("ablations", []):
            ablated_logits = ablation["ablated_logits"]
            ablated_pred_idx = np.argmax(ablated_logits)

            # Check if prediction changed
            flipped = 1 if clean_pred_idx != ablated_pred_idx else 0

            results.append(
                {
                    "key": key,
                    "value": answer,
                    "clean_pred_idx": clean_pred_idx,
                    "ablated_pred_idx": ablated_pred_idx,
                    "flipped": flipped,
                    "layer": ablation.get("layer", 1),
                    "idx1": ablation.get("idx1"),
                    "idx2": ablation.get("idx2"),
                }
            )

    return results


def count_repeating_digits_with_diff_values(key, value):
    """
    Count the number of repeating digits in the key where the corresponding
    digits in the value are different.

    Example:
    - key="e00", value="e89" -> 2 (both 0s in key correspond to different digits 8,9 in value)
    - key="e12", value="e12" -> 0 (no repeating digits in key)
    - key="e11", value="e11" -> 0 (repeating digits but values are same)
    - key="e11", value="e89" -> 2 (both 1s in key correspond to different digits 8,9 in value)
    """
    # Extract just the digit parts (assuming format like "eXXXXXX")
    key_digits = "".join(c for c in key if c.isdigit())
    value_digits = "".join(c for c in value if c.isdigit())

    if len(key_digits) != len(value_digits):
        return 0

    # Count occurrences of each digit in key
    digit_positions = defaultdict(list)
    for i, digit in enumerate(key_digits):
        digit_positions[digit].append(i)

    # Count repeating digits where corresponding value digits differ
    count = 0
    for digit, positions in digit_positions.items():
        if len(positions) > 1:  # This digit repeats in the key
            # Check if all corresponding value digits are different from key digit
            for pos in positions:
                if key_digits[pos] != value_digits[pos]:
                    count += 1

    return count


def analyze_correlation(results):
    """Analyze correlation between repeating digits and ablation flips."""

    # Group by key-value pair and calculate flip rate
    pair_stats = defaultdict(lambda: {"flips": 0, "total": 0})

    for result in results:
        key = result["key"]
        value = result["value"]
        pair = (key, value)

        pair_stats[pair]["total"] += 1
        pair_stats[pair]["flips"] += result["flipped"]

    # Calculate statistics for each pair
    repeating_counts = []
    flip_rates = []
    flip_details = defaultdict(list)

    for (key, value), stats in pair_stats.items():
        repeat_count = count_repeating_digits_with_diff_values(key, value)
        flip_rate = stats["flips"] / stats["total"] if stats["total"] > 0 else 0

        repeating_counts.append(repeat_count)
        flip_rates.append(flip_rate)

        flip_details[repeat_count].append(
            {
                "key": key,
                "value": value,
                "flip_rate": flip_rate,
                "flips": stats["flips"],
                "total": stats["total"],
            }
        )

    # Convert to numpy arrays
    repeating_counts = np.array(repeating_counts)
    flip_rates = np.array(flip_rates)

    # Calculate correlation
    if len(set(repeating_counts)) > 1:
        pearson_r, pearson_p = pearsonr(repeating_counts, flip_rates)
        spearman_r, spearman_p = spearmanr(repeating_counts, flip_rates)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan

    # Calculate average flip rates by repeat count
    avg_flip_rates = {}
    for count in sorted(set(repeating_counts)):
        mask = repeating_counts == count
        avg_rate = flip_rates[mask].mean()
        total_pairs = mask.sum()
        avg_flip_rates[count] = {
            "rate": avg_rate,
            "total_pairs": total_pairs,
            "pairs": flip_details[count],
        }

    # Print results
    print("=== Correlation Analysis ===")
    print(f"\nTotal unique key-value pairs analyzed: {len(pair_stats)}")
    print(f"Total ablations performed: {len(results)}")
    print(f"Overall flip rate: {sum(r['flipped'] for r in results) / len(results):.3f}")

    print(f"\nPearson correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")
    print(f"Spearman correlation: r={spearman_r:.3f}, p={spearman_p:.3f}")

    print("\n=== Average Flip Rates by Repeating Digit Count ===")
    print("Repeat Count | Avg Flip Rate | Num Pairs | Total Appearances")
    print("-" * 60)
    for count in sorted(avg_flip_rates.keys()):
        info = avg_flip_rates[count]
        # Calculate total appearances (ablations) for this group
        total_appearances = sum(pair["total"] for pair in info["pairs"])
        print(
            f"{count:12d} | {info['rate']:13.3f} | {info['total_pairs']:10d} | {total_appearances:17d}"
        )

    # Create visualizations
    create_visualizations(repeating_counts, flip_rates, avg_flip_rates, flip_details)

    return avg_flip_rates, flip_details


def create_visualizations(repeating_counts, flip_rates, avg_flip_rates, flip_details):
    """Create visualizations of the correlation analysis."""

    # Set up the plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar plot of average flip rates by repeat count
    ax1 = axes[0, 0]
    counts = sorted(avg_flip_rates.keys())
    rates = [avg_flip_rates[c]["rate"] for c in counts]
    totals = [avg_flip_rates[c]["total_pairs"] for c in counts]

    bars = ax1.bar(counts, rates)
    ax1.set_xlabel("Number of Repeating Digits with Different Values")
    ax1.set_ylabel("Average Flip Rate")
    ax1.set_title("Ablation Flip Rate by Repeating Digit Count")
    ax1.set_ylim(0, max(rates) * 1.1 if rates else 1)

    # Add sample counts on bars
    for i, (bar, total) in enumerate(zip(bars, totals)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"n={total}",
            ha="center",
            va="bottom",
        )

    # 2. Scatter plot of flip rates
    ax2 = axes[0, 1]
    ax2.scatter(repeating_counts, flip_rates, alpha=0.5)
    ax2.set_xlabel("Number of Repeating Digits with Different Values")
    ax2.set_ylabel("Flip Rate (per key-value pair)")
    ax2.set_title("Scatter Plot of Repeating Digits vs Flip Rate")

    # Add trend line if there's correlation
    if len(set(repeating_counts)) > 1:
        z = np.polyfit(repeating_counts, flip_rates, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(repeating_counts.min(), repeating_counts.max(), 100)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label="Trend line")
        ax2.legend()

    # 3. Distribution of repeating digit counts
    ax3 = axes[1, 0]
    unique_counts, count_freq = np.unique(repeating_counts, return_counts=True)
    ax3.bar(unique_counts, count_freq)
    ax3.set_xlabel("Number of Repeating Digits with Different Values")
    ax3.set_ylabel("Number of Key-Value Pairs")
    ax3.set_title("Distribution of Repeating Digit Counts in Dataset")

    # 4. Examples table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Show examples with highest flip rates
    examples_text = "Examples with High Flip Rates:\n\n"
    all_examples = []
    for count, pairs_list in flip_details.items():
        for pair in pairs_list:
            if pair["flip_rate"] > 0:
                all_examples.append((count, pair))

    # Sort by flip rate and show top examples
    all_examples.sort(key=lambda x: x[1]["flip_rate"], reverse=True)

    for i, (count, ex) in enumerate(all_examples[:5]):
        examples_text += f"Repeat count {count}:\n"
        examples_text += f"  {ex['key']} → {ex['value']}\n"
        examples_text += f"  Flip rate: {ex['flip_rate']:.3f} ({ex['flips']}/{ex['total']} ablations)\n\n"

    ax4.text(
        0.05,
        0.95,
        examples_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax4.set_title("Examples with Highest Flip Rates")

    plt.tight_layout()
    plt.savefig(
        "repeating_digits_correlation_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("\nVisualization saved as 'repeating_digits_correlation_analysis.png'")

    # Create a detailed DataFrame for further analysis
    df_data = []
    for count in flip_details:
        for item in flip_details[count]:
            df_data.append(
                {
                    "repeat_count": count,
                    "key": item["key"],
                    "value": item["value"],
                    "flip_rate": item["flip_rate"],
                    "flips": item["flips"],
                    "total_ablations": item["total"],
                }
            )

    df = pd.DataFrame(df_data)
    df = df.sort_values(["repeat_count", "flip_rate"], ascending=[True, False])
    df.to_csv("repeating_digits_analysis_detailed.csv", index=False)
    print("Detailed results saved as 'repeating_digits_analysis_detailed.csv'")


if __name__ == "__main__":
    input_file = "/home/kevin/LLMonade_interp/experiments/synthetic_qa/results/ablation_results/50K/ablation_results_50K_new.json"
    fixed_file = "/home/kevin/LLMonade_interp/experiments/synthetic_qa/results/ablation_results/50K/ablation_results_50K_fixed.json"

    # First, fix the JSON file
    fix_json_file(input_file, fixed_file)

    # Parse the data
    results = parse_ablation_data(fixed_file)

    # Analyze correlation
    avg_flip_rates, flip_details = analyze_correlation(results)
