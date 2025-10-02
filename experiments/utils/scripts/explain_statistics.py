#!/usr/bin/env python3
"""
Detailed explanation of ablation and swap statistics calculation.
"""

import json
from collections import defaultdict


def explain_ablation_calculation():
    """Show exactly how ablation rates are calculated."""
    print("=== ABLATION RATE CALCULATION ===\n")

    # Load data
    with open(
        "experiments/synthetic_qa/results/ablation_results/memory_1layer_2K/ablation_results.json",
        "r",
    ) as f:
        data = json.load(f)

    # Take first entry as example
    entry = data[0]
    qa_key = entry["question"].replace("Q: Where is ", "").replace("?", "").strip()

    print("Example QA pair:")
    print(f"  Question: {entry['question']}")
    print(f"  Answer: {entry['answer']}")
    print(f"  QA key extracted: '{qa_key}'")
    print(f"  Number of ablation attempts: {len(entry['ablations'])}")

    # Show ablation attempts
    print("\nAblation attempts:")
    successful = 0
    for i, abl in enumerate(entry["ablations"][:5]):  # Show first 5
        print(f"\n  Attempt {i + 1}:")
        print(f"    Ablated memory key: idx1={abl['idx1']}, idx2={abl['idx2']}")
        print(f"    Correct generation: {abl['correct_generation']}")
        if abl["correct_generation"]:
            successful += 1

    print(f"\n  ... ({len(entry['ablations'])} total attempts)")

    # Calculate rate for this QA
    total_successful = sum(1 for abl in entry["ablations"] if abl["correct_generation"])
    rate = total_successful / len(entry["ablations"])

    print(f"\nAblation rate calculation for '{qa_key}':")
    print(f"  Successful ablations: {total_successful}")
    print(f"  Total ablations: {len(entry['ablations'])}")
    print(f"  Rate: {total_successful}/{len(entry['ablations'])} = {rate:.3f}")

    # Show aggregation across all QAs
    print("\n\n=== AGGREGATION ACROSS ALL QAs ===")
    qa_stats = defaultdict(lambda: {"successful": 0, "total": 0})

    for entry in data:
        qa = entry["question"].replace("Q: Where is ", "").replace("?", "").strip()
        for ablation in entry.get("ablations", []):
            qa_stats[qa]["total"] += 1
            if ablation["correct_generation"]:
                qa_stats[qa]["successful"] += 1

    print(f"Total unique QAs: {len(qa_stats)}")
    print("\nSample of QA ablation rates:")
    for i, (qa, stats) in enumerate(list(qa_stats.items())[:5]):
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {qa}: {stats['successful']}/{stats['total']} = {rate:.3f}")


def explain_swap_calculation():
    """Show exactly how swap rates are calculated."""
    print("\n\n=== SWAP RATE CALCULATION ===\n")

    # Load data
    with open("memory_1layer_2K_no_mlp_swap_results.json", "r") as f:
        data = json.load(f)

    # Take first result as example
    result = data["results"][0]

    print("Example swap experiment:")
    print(f"  Swap type: {result['swap_type']}")
    print(f"  QA1: {result['qa1']['question']}")
    print(f"  QA2: {result['qa2']['question']}")

    # Explain QA1 swapping
    qa1 = result["qa1"]
    print("\nQA1 swap details:")
    print(f"  Answer: {qa1['answer']}")
    print(f"  Answer tokens: {qa1['answer_tokens']}")
    print(f"  Generated tokens: {qa1['generated_tokens']}")
    print(f"  Swapped positions: {qa1['swapped_positions']}")
    print(f"  Number of tokens swapped: {qa1['num_swapped']}")

    # Show token-by-token comparison
    print("\n  Token-by-token comparison:")
    for i, (ans_tok, gen_tok) in enumerate(
        zip(qa1["answer_tokens"], qa1["generated_tokens"])
    ):
        swapped = i in qa1["swapped_positions"]
        print(
            f"    Position {i}: {ans_tok} -> {gen_tok} {'[SWAPPED]' if swapped else '[SAME]'}"
        )

    # Calculate rate
    swap_rate = qa1["num_swapped"] / len(qa1["answer_tokens"])
    print(
        f"\n  Swap rate: {qa1['num_swapped']}/{len(qa1['answer_tokens'])} = {swap_rate:.3f}"
    )

    # Show aggregation
    print("\n\n=== AGGREGATION ACROSS ALL EXPERIMENTS ===")
    qa_swap_stats = defaultdict(lambda: {"swapped": 0, "total": 0})

    for result in data["results"][:100]:  # First 100 for speed
        qa1_key = (
            result["qa1"]["question"]
            .replace("Q: Where is ", "")
            .replace("?", "")
            .strip()
        )
        qa2_key = (
            result["qa2"]["question"]
            .replace("Q: Where is ", "")
            .replace("?", "")
            .strip()
        )

        qa_swap_stats[qa1_key]["total"] += len(result["qa1"]["answer_tokens"])
        qa_swap_stats[qa1_key]["swapped"] += result["qa1"]["num_swapped"]

        qa_swap_stats[qa2_key]["total"] += len(result["qa2"]["answer_tokens"])
        qa_swap_stats[qa2_key]["swapped"] += result["qa2"]["num_swapped"]

    print(f"Unique QAs seen (in first 100 experiments): {len(qa_swap_stats)}")
    print("\nSample of QA swap rates:")
    for i, (qa, stats) in enumerate(list(qa_swap_stats.items())[:5]):
        rate = stats["swapped"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {qa}: {stats['swapped']}/{stats['total']} tokens = {rate:.3f}")


def explain_correlation():
    """Explain correlation calculation."""
    print("\n\n=== CORRELATION CALCULATION ===\n")

    print("1. For each QA that appears in BOTH datasets:")
    print("   - Get its ablation rate (successful ablations / total ablations)")
    print("   - Get its swap rate (swapped tokens / total tokens)")
    print("   - Create paired data points (ablation_rate, swap_rate)")
    print("\n2. Calculate Pearson correlation coefficient:")
    print("   - Measures linear relationship between ablation and swap rates")
    print("   - Range: -1 (perfect negative) to +1 (perfect positive)")
    print("   - 0 means no linear relationship")
    print("\n3. Our result: r = -0.0019")
    print("   - This is essentially 0, meaning NO correlation")
    print("   - Keys easy to ablate are NOT more/less likely to be easy to swap")

    print("\n\n=== CONSISTENCY ANALYSIS ===\n")
    print("Threshold = 0.5 (50% rate)")
    print("- 'Easy' = rate > 0.5")
    print("- 'Hard' = rate ≤ 0.5")
    print("\nCategories:")
    print("- Both easy: QA has >50% ablation rate AND >50% swap rate")
    print("- Both hard: QA has ≤50% ablation rate AND ≤50% swap rate")
    print("- Inconsistent: One easy, one hard")
    print("\nOur results:")
    print("- 45.9% are both easy")
    print("- 8.8% are both hard")
    print("- 54.8% total consistent (both easy OR both hard)")
    print("- This is barely better than random (50%)")


if __name__ == "__main__":
    explain_ablation_calculation()
    explain_swap_calculation()
    explain_correlation()
