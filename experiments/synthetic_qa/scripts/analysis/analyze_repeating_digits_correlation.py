#!/usr/bin/env python3
"""
Analyze correlation between repeating digits in keys where corresponding values differ
and the probability that ablation flips the prediction.

Example: e00 -> e89 has 2 repeating digits (00) where the corresponding value digits differ (89).
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
    # Extract just the digit parts (assuming format like "eXX" or "abcdXX")
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
            value_digits_at_positions = [value_digits[pos] for pos in positions]
            # Check if any of these positions have different digits in the value
            for pos in positions:
                if key_digits[pos] != value_digits[pos]:
                    count += 1

    return count


def analyze_correlation(json_file):
    """Analyze correlation between repeating digits and ablation flips."""

    # Load data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Collect data for analysis
    repeating_counts = []
    flip_occurred = []
    flip_details = defaultdict(list)

    for item in data:
        prompt = item["prompt"]
        original_pred = item["original_prediction"]
        ablated_pred = item["ablated_prediction"]

        # Extract key-value pair from prompt
        # Assuming format like "What is abcd00?" and answer is "abcd89"
        if "What is " in prompt and "?" in prompt:
            key = prompt.split("What is ")[1].split("?")[0].strip()
            value = original_pred

            # Count repeating digits with different values
            repeat_count = count_repeating_digits_with_diff_values(key, value)

            # Check if ablation caused a flip
            flipped = 1 if original_pred != ablated_pred else 0

            repeating_counts.append(repeat_count)
            flip_occurred.append(flipped)

            # Store details for further analysis
            flip_details[repeat_count].append(
                {
                    "key": key,
                    "value": value,
                    "original": original_pred,
                    "ablated": ablated_pred,
                    "flipped": flipped,
                }
            )

    # Convert to numpy arrays
    repeating_counts = np.array(repeating_counts)
    flip_occurred = np.array(flip_occurred)

    # Calculate correlation
    if len(set(repeating_counts)) > 1:  # Need variation to calculate correlation
        pearson_r, pearson_p = stats.pearsonr(repeating_counts, flip_occurred)
        spearman_r, spearman_p = stats.spearmanr(repeating_counts, flip_occurred)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan

    # Calculate flip rates by repeat count
    flip_rates = {}
    for count in sorted(set(repeating_counts)):
        mask = repeating_counts == count
        flip_rate = flip_occurred[mask].mean()
        total = mask.sum()
        flipped = flip_occurred[mask].sum()
        flip_rates[count] = {"rate": flip_rate, "flipped": flipped, "total": total}

    # Print results
    print("=== Correlation Analysis ===")
    print(f"\nTotal samples analyzed: {len(repeating_counts)}")
    print(f"Overall flip rate: {flip_occurred.mean():.3f}")

    print(f"\nPearson correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")
    print(f"Spearman correlation: r={spearman_r:.3f}, p={spearman_p:.3f}")

    print("\n=== Flip Rates by Repeating Digit Count ===")
    print("Repeat Count | Flip Rate | Flipped/Total")
    print("-" * 40)
    for count in sorted(flip_rates.keys()):
        info = flip_rates[count]
        print(
            f"{count:12d} | {info['rate']:9.3f} | {info['flipped']:6d}/{info['total']:<6d}"
        )

    # Create visualizations
    create_visualizations(repeating_counts, flip_occurred, flip_rates, flip_details)

    return flip_rates, flip_details


def create_visualizations(repeating_counts, flip_occurred, flip_rates, flip_details):
    """Create visualizations of the correlation analysis."""

    # Set up the plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar plot of flip rates by repeat count
    ax1 = axes[0, 0]
    counts = sorted(flip_rates.keys())
    rates = [flip_rates[c]["rate"] for c in counts]
    totals = [flip_rates[c]["total"] for c in counts]

    bars = ax1.bar(counts, rates)
    ax1.set_xlabel("Number of Repeating Digits with Different Values")
    ax1.set_ylabel("Flip Rate")
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

    # 2. Scatter plot with jitter
    ax2 = axes[0, 1]
    # Add jitter for better visualization
    jittered_counts = repeating_counts + np.random.normal(
        0, 0.05, size=len(repeating_counts)
    )
    jittered_flips = flip_occurred + np.random.normal(0, 0.02, size=len(flip_occurred))

    ax2.scatter(jittered_counts, jittered_flips, alpha=0.5, s=20)
    ax2.set_xlabel("Number of Repeating Digits with Different Values")
    ax2.set_ylabel("Flipped (1) or Not (0)")
    ax2.set_title("Scatter Plot of Repeating Digits vs Flip Occurrence")

    # Add trend line if there's correlation
    if len(set(repeating_counts)) > 1:
        z = np.polyfit(repeating_counts, flip_occurred, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(repeating_counts.min(), repeating_counts.max(), 100)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label="Trend line")
        ax2.legend()

    # 3. Distribution of repeating digit counts
    ax3 = axes[1, 0]
    unique_counts, count_freq = np.unique(repeating_counts, return_counts=True)
    ax3.bar(unique_counts, count_freq)
    ax3.set_xlabel("Number of Repeating Digits with Different Values")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Repeating Digit Counts in Dataset")

    # 4. Examples of flipped cases
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Show some examples of flipped cases
    examples_text = "Examples of Flipped Cases:\n\n"
    max_examples = 5
    shown_counts = set()

    for count in sorted(flip_details.keys(), reverse=True):
        flipped_examples = [ex for ex in flip_details[count] if ex["flipped"]]
        if flipped_examples and len(shown_counts) < max_examples:
            shown_counts.add(count)
            ex = flipped_examples[0]
            examples_text += f"Repeat count {count}:\n"
            examples_text += f"  {ex['key']} → {ex['value']}\n"
            examples_text += (
                f"  Original: {ex['original']}, Ablated: {ex['ablated']}\n\n"
            )

    ax4.text(
        0.05,
        0.95,
        examples_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax4.set_title("Example Flipped Cases")

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
                    "flipped": item["flipped"],
                    "original_pred": item["original"],
                    "ablated_pred": item["ablated"],
                }
            )

    df = pd.DataFrame(df_data)
    df.to_csv("repeating_digits_analysis_detailed.csv", index=False)
    print("Detailed results saved as 'repeating_digits_analysis_detailed.csv'")


if __name__ == "__main__":
    json_file = "/home/kevin/LLMonade_interp/experiments/synthetic_qa/results/ablation_results/50K/ablation_results_50K_new.json"
    flip_rates, flip_details = analyze_correlation(json_file)
