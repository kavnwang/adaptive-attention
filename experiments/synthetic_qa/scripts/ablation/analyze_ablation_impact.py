#!/usr/bin/env python3
"""
Analyze ablation results to find QA pairs where memory key ablation has significant impact.
"""

import argparse
import json
import numpy as np
from collections import defaultdict
from pathlib import Path


def analyze_ablation_results(results, alpha=0.1, verbose=True):
    """
    Analyze ablation results to find significant impacts.

    Args:
        results: List of ablation results
        alpha: Threshold for "significant" impact (in log probability difference)
        verbose: Whether to print detailed information

    Returns:
        Dictionary with analysis results
    """
    total_qa_pairs = len(results)
    qa_with_impact = set()
    layer_impact_counts = defaultdict(int)
    layer_impact_magnitudes = defaultdict(list)

    # Track the most impactful ablations
    most_impactful = []

    for qa_idx, qa_result in enumerate(results):
        question = qa_result["question"]
        answer = qa_result["answer"]
        predicted = qa_result.get("predicted_answer", answer)

        qa_has_impact = False
        max_impact_for_qa = 0

        for layer_name, layer_results in qa_result[
            "ablation_results_per_layer"
        ].items():
            for key_result in layer_results:
                impact = abs(key_result["logit_difference"])
                layer_impact_magnitudes[layer_name].append(impact)

                if impact > alpha:
                    qa_has_impact = True
                    layer_impact_counts[layer_name] += 1
                    max_impact_for_qa = max(max_impact_for_qa, impact)

                    # Track most impactful ablations
                    most_impactful.append(
                        {
                            "qa_idx": qa_idx,
                            "question": question,
                            "answer": answer,
                            "predicted": predicted,
                            "layer": layer_name,
                            "idx1": key_result["idx1"],
                            "idx2": key_result["idx2"],
                            "impact": impact,
                            "clean_sum": key_result["clean_logits_sum"],
                            "ablated_sum": key_result["ablated_logits_sum"],
                        }
                    )

        if qa_has_impact:
            qa_with_impact.add(qa_idx)

    # Sort most impactful by impact magnitude
    most_impactful.sort(key=lambda x: x["impact"], reverse=True)

    # Calculate statistics
    analysis = {
        "total_qa_pairs": total_qa_pairs,
        "qa_pairs_with_significant_impact": len(qa_with_impact),
        "percentage_with_impact": len(qa_with_impact) / total_qa_pairs * 100,
        "threshold_alpha": alpha,
        "layer_statistics": {},
        "all_impactful_ablations": most_impactful,  # All ablations above threshold, sorted
    }

    # Calculate per-layer statistics
    for layer_name in sorted(layer_impact_magnitudes.keys()):
        impacts = layer_impact_magnitudes[layer_name]
        analysis["layer_statistics"][layer_name] = {
            "total_ablations": len(impacts),
            "significant_ablations": layer_impact_counts[layer_name],
            "percentage_significant": layer_impact_counts[layer_name]
            / len(impacts)
            * 100,
            "mean_impact": np.mean(impacts),
            "median_impact": np.median(impacts),
            "max_impact": np.max(impacts),
            "std_impact": np.std(impacts),
        }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"ABLATION IMPACT ANALYSIS (α = {alpha})")
        print(f"{'=' * 60}")
        print(f"\nTotal QA pairs analyzed: {total_qa_pairs}")
        print(
            f"QA pairs with significant impact: {len(qa_with_impact)} ({analysis['percentage_with_impact']:.2f}%)"
        )

        print(f"\n{'=' * 60}")
        print("PER-LAYER STATISTICS")
        print(f"{'=' * 60}")

        for layer_name, stats in analysis["layer_statistics"].items():
            print(f"\n{layer_name}:")
            print(f"  Total ablations: {stats['total_ablations']}")
            print(
                f"  Significant ablations: {stats['significant_ablations']} ({stats['percentage_significant']:.2f}%)"
            )
            print(f"  Mean impact: {stats['mean_impact']:.6f}")
            print(f"  Median impact: {stats['median_impact']:.6f}")
            print(f"  Max impact: {stats['max_impact']:.6f}")
            print(f"  Std deviation: {stats['std_impact']:.6f}")

        if len(most_impactful) > 0:
            print(f"\n{'=' * 60}")
            print(f"IMPACTFUL ABLATIONS (Total: {len(most_impactful)})")
            print(f"{'=' * 60}")

            # Show first 10 as examples
            show_count = min(10, len(most_impactful))
            print(f"\nShowing top {show_count} examples:")

            for i, item in enumerate(most_impactful[:show_count]):
                print(f"\n{i + 1}. Impact: {item['impact']:.6f}")
                print(f"   Question: {item['question']}")
                print(f"   Answer: {item['answer']}")
                print(
                    f"   Layer: {item['layer']}, Key: ({item['idx1']}, {item['idx2']})"
                )
                print(
                    f"   Logit change: {item['clean_sum']:.6f} → {item['ablated_sum']:.6f}"
                )

    # Find QA pairs where ALL ablations have minimal impact
    qa_no_impact = []
    for qa_idx, qa_result in enumerate(results):
        all_minimal = True
        max_impact = 0

        for layer_results in qa_result["ablation_results_per_layer"].values():
            for key_result in layer_results:
                impact = abs(key_result["logit_difference"])
                max_impact = max(max_impact, impact)
                if impact > alpha / 10:  # Even 10x smaller than threshold
                    all_minimal = False
                    break
            if not all_minimal:
                break

        if all_minimal:
            qa_no_impact.append(
                {
                    "qa_idx": qa_idx,
                    "question": qa_result["question"],
                    "answer": qa_result["answer"],
                    "max_impact": max_impact,
                }
            )

    analysis["qa_pairs_with_minimal_impact"] = len(qa_no_impact)

    if verbose and qa_no_impact:
        print(f"\n{'=' * 60}")
        print(f"QA PAIRS WITH MINIMAL IMPACT (< {alpha / 10:.6f})")
        print(f"{'=' * 60}")
        print(
            f"Found {len(qa_no_impact)} QA pairs with minimal impact across all ablations"
        )

        for item in qa_no_impact[:5]:  # Show first 5
            print(f"\nQuestion: {item['question']}")
            print(f"Answer: {item['answer']}")
            print(f"Max impact: {item['max_impact']:.8f}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation impact")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Ablation results JSON file"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Threshold for significant impact"
    )
    parser.add_argument(
        "--output_file", type=str, help="Optional output file for detailed analysis"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Load results
    print(f"Loading ablation results from {args.input_file}...")
    with open(args.input_file, "r") as f:
        results = json.load(f)

    # Analyze results
    analysis = analyze_ablation_results(
        results, alpha=args.alpha, verbose=not args.quiet
    )

    # Save detailed analysis if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"\nDetailed analysis saved to {args.output_file}")

    # Always print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Threshold (α): {args.alpha}")
    print(
        f"QA pairs with significant impact: {analysis['qa_pairs_with_significant_impact']}/{analysis['total_qa_pairs']} ({analysis['percentage_with_impact']:.2f}%)"
    )
    print(f"QA pairs with minimal impact: {analysis['qa_pairs_with_minimal_impact']}")


if __name__ == "__main__":
    main()
