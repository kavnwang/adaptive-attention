#!/usr/bin/env python3
"""
Analyze which memory keys appear across multiple QA pairs in the activation data.
"""

import json
from collections import defaultdict


def analyze_key_occurrences(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Track which QA pairs each key appears in
    key_to_qa_pairs = defaultdict(set)

    for entry in data:
        question = entry["question"]
        answer = entry["answer"]
        qa_identifier = f"{question} -> {answer}"

        # Process each layer
        for layer_name, keys in entry["top_keys_per_layer"].items():
            for key_info in keys:
                idx1 = key_info["idx1"]
                idx2 = key_info["idx2"]
                key_tuple = (layer_name, idx1, idx2)
                key_to_qa_pairs[key_tuple].add(qa_identifier)

    # Convert to list and sort by number of QA pairs
    key_occurrences = []
    for key_tuple, qa_pairs in key_to_qa_pairs.items():
        layer_name, idx1, idx2 = key_tuple
        key_occurrences.append(
            {
                "layer": layer_name,
                "idx1": idx1,
                "idx2": idx2,
                "num_qa_pairs": len(qa_pairs),
                "qa_pairs": list(qa_pairs)[:5],  # Show first 5 examples
            }
        )

    # Sort by number of QA pairs (descending)
    key_occurrences.sort(key=lambda x: x["num_qa_pairs"], reverse=True)

    return key_occurrences


def main():
    json_path = "memory_1layer_2K_no_mlp_key_activations.json"

    print(f"Analyzing key occurrences in {json_path}...")
    key_occurrences = analyze_key_occurrences(json_path)

    # Print top 20 most common keys
    print("\nTop 20 most common keys across QA pairs:")
    print(
        f"{'Layer':<10} {'Key (idx1, idx2)':<20} {'# QA pairs':<12} {'Example QA pairs'}"
    )
    print("-" * 100)

    for i, key_info in enumerate(key_occurrences[:20]):
        layer = key_info["layer"]
        idx1 = key_info["idx1"]
        idx2 = key_info["idx2"]
        num_qa = key_info["num_qa_pairs"]

        print(f"{layer:<10} ({idx1:>4}, {idx2:>4})        {num_qa:<12}", end="")

        # Show first 2 example QA pairs
        examples = key_info["qa_pairs"][:2]
        if examples:
            print(f"{examples[0][:50]}...")
            if len(examples) > 1:
                print(f"{'':<44}{examples[1][:50]}...")
        print()

    # Statistics
    print("\nStatistics:")
    print(f"Total unique keys: {len(key_occurrences)}")
    print(
        f"Keys appearing in >10 QA pairs: {sum(1 for k in key_occurrences if k['num_qa_pairs'] > 10)}"
    )
    print(
        f"Keys appearing in >50 QA pairs: {sum(1 for k in key_occurrences if k['num_qa_pairs'] > 50)}"
    )
    print(
        f"Keys appearing in >100 QA pairs: {sum(1 for k in key_occurrences if k['num_qa_pairs'] > 100)}"
    )

    # Distribution analysis
    print("\nDistribution of key occurrences:")
    buckets = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    for i in range(len(buckets)):
        if i == 0:
            count = sum(1 for k in key_occurrences if k["num_qa_pairs"] == buckets[i])
            print(f"  Exactly {buckets[i]} QA pair: {count} keys")
        else:
            lower = buckets[i - 1] + 1
            upper = buckets[i]
            count = sum(
                1 for k in key_occurrences if lower <= k["num_qa_pairs"] <= upper
            )
            print(f"  {lower}-{upper} QA pairs: {count} keys")

    # Keys appearing in many QA pairs
    print("\nKeys appearing in >100 QA pairs:")
    for key_info in key_occurrences:
        if key_info["num_qa_pairs"] > 100:
            layer = key_info["layer"]
            idx1 = key_info["idx1"]
            idx2 = key_info["idx2"]
            num_qa = key_info["num_qa_pairs"]
            print(f"  {layer} ({idx1}, {idx2}): {num_qa} QA pairs")


if __name__ == "__main__":
    main()
