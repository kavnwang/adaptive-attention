#!/usr/bin/env python3
"""
Analyze QA pairs and their key statistics from memory activation and ablation results.

For each QA pair, this script generates:
- Total score (sum of top key scores)
- Number of significant ablations in layer 0
- Number of significant ablations in layer 1
- Ratio of largest to fifth largest key score
"""

import argparse
import json
from pathlib import Path


def load_json_file(filepath):
    """Load JSON data from file."""
    with open(filepath, "r") as f:
        return json.load(f)


def calculate_qa_statistics(dataset_size):
    """
    Calculate statistics for each QA pair based on memory activations and ablation results.

    Args:
        dataset_size: One of "200", "2K", "10K", "50K"

    Returns:
        List of dictionaries with QA statistics
    """
    # Construct file paths
    activation_file = (
        f"../results/memory_key_activations/memory_key_activations_{dataset_size}.json"
    )
    ablation_file = f"../results/ablation_results/ablation_results_{dataset_size}.json"

    # Load data
    print(f"Loading activation data from {activation_file}...")
    activations = load_json_file(activation_file)

    print(f"Loading ablation data from {ablation_file}...")
    ablations = load_json_file(ablation_file)

    # Create mapping from question to ablation results
    ablation_map = {item["question"]: item for item in ablations}

    # Process each QA pair
    results = []

    for qa in activations:
        question = qa["question"]
        answer = qa["answer"]

        # Get clean logit sum from ablation results if available
        clean_logit_sum = None
        if question in ablation_map:
            ablation_data = ablation_map[question]
            # Get the first clean_logits_sum as they should all be the same for a given QA pair
            for layer_results in ablation_data.get(
                "ablation_results_per_layer", {}
            ).values():
                if layer_results and len(layer_results) > 0:
                    clean_logit_sum = layer_results[0].get("clean_logits_sum", None)
                    break

        # Calculate total score from activation data
        total_activation_score = 0
        layer_0_top_scores = []
        layer_1_top_scores = []

        for layer_name, keys in qa["top_keys_per_layer"].items():
            for key in keys:
                total_activation_score += key["total_score"]
                if layer_name == "layer_0":
                    layer_0_top_scores.append(key["total_score"])
                elif layer_name == "layer_1":
                    layer_1_top_scores.append(key["total_score"])

        # Calculate ratio of largest to fifth largest key score per layer
        layer_0_ratio = None
        layer_1_ratio = None

        # Layer 0 ratio
        if layer_0_top_scores and len(layer_0_top_scores) >= 5:
            layer_0_top_scores_sorted = sorted(layer_0_top_scores, reverse=True)
            layer_0_ratio = (
                layer_0_top_scores_sorted[0] / layer_0_top_scores_sorted[4]
                if layer_0_top_scores_sorted[4] > 0
                else float("inf")
            )

        # Layer 1 ratio
        if layer_1_top_scores and len(layer_1_top_scores) >= 5:
            layer_1_top_scores_sorted = sorted(layer_1_top_scores, reverse=True)
            layer_1_ratio = (
                layer_1_top_scores_sorted[0] / layer_1_top_scores_sorted[4]
                if layer_1_top_scores_sorted[4] > 0
                else float("inf")
            )

        # Count ablations that change the output
        layer_0_ablations = 0
        layer_1_ablations = 0

        if question in ablation_map:
            ablation_data = ablation_map[question]

            for layer_name, layer_results in ablation_data.get(
                "ablation_results_per_layer", {}
            ).items():
                for key_result in layer_results:
                    # Check if the ablation changed the generated output
                    output_changed = key_result.get("output_changed", False)

                    if output_changed:
                        if layer_name == "layer_0":
                            layer_0_ablations += 1
                        elif layer_name == "layer_1":
                            layer_1_ablations += 1

        # Compile results
        results.append(
            {
                "question": question,
                "answer": answer,
                "clean_logit_sum": round(clean_logit_sum, 4)
                if clean_logit_sum is not None
                else None,
                "total_activation_score": round(total_activation_score, 4),
                "layer_0_ablations": layer_0_ablations,
                "layer_1_ablations": layer_1_ablations,
                "layer_0_score_ratio_1st_to_5th": round(layer_0_ratio, 4)
                if layer_0_ratio is not None
                else None,
                "layer_1_score_ratio_1st_to_5th": round(layer_1_ratio, 4)
                if layer_1_ratio is not None
                else None,
                "layer_0_max_score": round(max(layer_0_top_scores), 4)
                if layer_0_top_scores
                else 0,
                "layer_1_max_score": round(max(layer_1_top_scores), 4)
                if layer_1_top_scores
                else 0,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QA key statistics from memory activations and ablations"
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        required=True,
        choices=["200", "2K", "10K", "50K"],
        help="Dataset size to analyze",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file (default: results/qa_key_statistics_{dataset_size}.json)",
    )

    args = parser.parse_args()

    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = (
            f"../results/qa_key_statistics/qa_key_statistics_{args.dataset_size}.json"
        )

    print(f"\n{'=' * 60}")
    print("QA KEY STATISTICS ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Output file: {args.output_file}")
    print(f"{'=' * 60}\n")

    # Calculate statistics
    try:
        results = calculate_qa_statistics(args.dataset_size)

        # Sort by clean logit sum (descending), falling back to activation score if no logit sum
        results.sort(
            key=lambda x: x["clean_logit_sum"]
            if x["clean_logit_sum"] is not None
            else float("-inf"),
            reverse=True,
        )

        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary statistics
        print(f"\nProcessed {len(results)} QA pairs")

        if results:
            clean_logit_sums = [
                r["clean_logit_sum"]
                for r in results
                if r["clean_logit_sum"] is not None
            ]
            total_activation_scores = [r["total_activation_score"] for r in results]
            layer_0_ablations = [r["layer_0_ablations"] for r in results]
            layer_1_ablations = [r["layer_1_ablations"] for r in results]
            layer_0_ratios = [
                r["layer_0_score_ratio_1st_to_5th"]
                for r in results
                if r["layer_0_score_ratio_1st_to_5th"] is not None
            ]
            layer_1_ratios = [
                r["layer_1_score_ratio_1st_to_5th"]
                for r in results
                if r["layer_1_score_ratio_1st_to_5th"] is not None
            ]

            if clean_logit_sums:
                print("\nClean Logit Sum Statistics:")
                print(f"  Min: {min(clean_logit_sums):.2f}")
                print(f"  Max: {max(clean_logit_sums):.2f}")
                print(f"  Mean: {sum(clean_logit_sums) / len(clean_logit_sums):.2f}")

            print("\nTotal Activation Score Statistics:")
            print(f"  Min: {min(total_activation_scores):.2f}")
            print(f"  Max: {max(total_activation_scores):.2f}")
            print(
                f"  Mean: {sum(total_activation_scores) / len(total_activation_scores):.2f}"
            )

            print("\nAblation Counts:")
            print(
                f"  Layer 0: {sum(layer_0_ablations)} total, {sum(1 for x in layer_0_ablations if x > 0)} QA pairs with ablations"
            )
            print(
                f"  Layer 1: {sum(layer_1_ablations)} total, {sum(1 for x in layer_1_ablations if x > 0)} QA pairs with ablations"
            )

            if layer_0_ratios:
                print("\nLayer 0 Score Ratio (1st/5th) Statistics:")
                print(f"  Min: {min(layer_0_ratios):.2f}")
                print(f"  Max: {max(layer_0_ratios):.2f}")
                print(f"  Mean: {sum(layer_0_ratios) / len(layer_0_ratios):.2f}")

            if layer_1_ratios:
                print("\nLayer 1 Score Ratio (1st/5th) Statistics:")
                print(f"  Min: {min(layer_1_ratios):.2f}")
                print(f"  Max: {max(layer_1_ratios):.2f}")
                print(f"  Mean: {sum(layer_1_ratios) / len(layer_1_ratios):.2f}")

            print("\nTop 5 QA pairs by clean logit sum:")
            for i, qa in enumerate(results[:5]):
                print(f"  {i + 1}. {qa['question']} -> {qa['answer']}")
                logit_str = (
                    f"{qa['clean_logit_sum']:.2f}"
                    if qa["clean_logit_sum"] is not None
                    else "N/A"
                )
                print(
                    f"     Clean logit sum: {logit_str}, L0 ablations: {qa['layer_0_ablations']}, L1 ablations: {qa['layer_1_ablations']}"
                )

        print(f"\nResults saved to: {args.output_file}")

    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print(
            f"Make sure you have run both extract_memory_key_activations.py and ablate_memory_keys.py for dataset size {args.dataset_size}"
        )
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
