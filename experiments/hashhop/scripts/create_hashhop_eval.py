#!/usr/bin/env python3
"""
Create hash-hop evaluation dataset for 340M memory MLP router model
"""

import json
import pickle
from pathlib import Path
from hashhop import MultiHopEval


def create_eval_set(output_dir="hashhop_eval_340m"):
    """Create evaluation sets with varying difficulty levels"""

    Path(output_dir).mkdir(exist_ok=True)

    # Configuration for 340M model - start conservative with context length
    configs = [
        # Easy: Short context, 1 hop
        {
            "name": "easy_1hop",
            "n_samples": 50,
            "n_chars_problem": 10_000,  # ~3K tokens
            "num_queries": 5,
            "hops": 1,
            "hash_pair_str_length": 8,
            "chain_of_thought": False,
        },
        # Medium: Medium context, 2 hops
        {
            "name": "medium_2hop",
            "n_samples": 50,
            "n_chars_problem": 50_000,  # ~15K tokens
            "num_queries": 5,
            "hops": 2,
            "hash_pair_str_length": 12,
            "chain_of_thought": False,
        },
        # Hard: Longer context, 3 hops
        {
            "name": "hard_3hop",
            "n_samples": 25,
            "n_chars_problem": 100_000,  # ~30K tokens
            "num_queries": 5,
            "hops": 3,
            "hash_pair_str_length": 16,
            "chain_of_thought": False,
        },
        # Challenge: Max context for 340M model, 4 hops
        {
            "name": "challenge_4hop",
            "n_samples": 10,
            "n_chars_problem": 200_000,  # ~60K tokens
            "num_queries": 5,
            "hops": 4,
            "hash_pair_str_length": 16,
            "chain_of_thought": False,
        },
        # Chain-of-thought versions for comparison
        {
            "name": "medium_2hop_cot",
            "n_samples": 25,
            "n_chars_problem": 50_000,
            "num_queries": 5,
            "hops": 2,
            "hash_pair_str_length": 12,
            "chain_of_thought": True,
        },
        {
            "name": "hard_3hop_cot",
            "n_samples": 25,
            "n_chars_problem": 100_000,
            "num_queries": 5,
            "hops": 3,
            "hash_pair_str_length": 16,
            "chain_of_thought": True,
        },
    ]

    all_datasets = {}

    for config in configs:
        print(f"Creating {config['name']} dataset...")
        name = config.pop("name")
        n_samples = config.pop("n_samples")

        samples = []
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Generated {i}/{n_samples} samples")

            sample = MultiHopEval.make_one(**config)
            samples.append(
                {
                    "prompt": sample.prompt,
                    "completion": sample.completion,
                    "targets": sample.targets,
                    "metadata": {
                        "hops": config["hops"],
                        "n_chars": config["n_chars_problem"],
                        "chain_of_thought": config["chain_of_thought"],
                        "hash_length": config["hash_pair_str_length"],
                    },
                }
            )

        all_datasets[name] = samples

        # Save individual dataset
        with open(f"{output_dir}/{name}.json", "w") as f:
            json.dump(samples, f, indent=2)

        print(f"  Saved {n_samples} samples to {output_dir}/{name}.json")

    # Save all datasets in one file
    with open(f"{output_dir}/all_eval_sets.json", "w") as f:
        json.dump(all_datasets, f, indent=2)

    # Also save as pickle for faster loading
    with open(f"{output_dir}/all_eval_sets.pkl", "wb") as f:
        pickle.dump(all_datasets, f)

    # Create metadata file
    metadata = {
        "model": "340M memory MLP router",
        "total_samples": sum(len(samples) for samples in all_datasets.values()),
        "datasets": {
            name: {
                "n_samples": len(samples),
                "config": samples[0]["metadata"] if samples else {},
            }
            for name, samples in all_datasets.items()
        },
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset creation complete!")
    print(f"Total samples created: {metadata['total_samples']}")
    print(f"Output directory: {output_dir}/")
    print("\nDatasets created:")
    for name, info in metadata["datasets"].items():
        print(f"  - {name}: {info['n_samples']} samples")


if __name__ == "__main__":
    create_eval_set()
