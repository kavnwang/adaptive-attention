#!/usr/bin/env python3
"""
Create a hash-hop training dataset with high diversity to prevent memorization.
Keeps sequences under 8192 tokens (~24K chars assuming 3 chars/token).
"""

import json
import random
from pathlib import Path
from hashhop import MultiHopEval
from tqdm import tqdm
import hashlib


def estimate_tokens(n_chars):
    """Estimate token count from character count (roughly 3 chars per token)"""
    return n_chars // 3


def replicate_and_shuffle_dataset(
    samples: List[Dict], num_copies: int, base_seed: int
) -> List[Dict]:
    """
    Create multiple shuffled copies of the dataset and concatenate them.

    Args:
        samples: Original dataset samples
        num_copies: Number of copies to create (100)
        base_seed: Base random seed

    Returns:
        Concatenated list of all shuffled copies
    """
    all_samples = []

    for i in range(num_copies):
        # Create a copy of the samples
        copy_samples = samples.copy()

        # Shuffle with a different seed for each copy
        random.seed(base_seed + i)
        random.shuffle(copy_samples)

        # Add to the full list
        all_samples.extend(copy_samples)

    return all_samples


def create_diverse_hashhop_dataset(
    n_samples=10000,
    max_chars=24000,  # ~8K tokens
    output_dir="hashhop_training_data",
    num_copies=100,
    seed=42,
):
    """
    Create a training dataset where hash patterns rarely repeat.
    Uses varying parameters to ensure diversity.
    """

    Path(output_dir).mkdir(exist_ok=True)

    # Parameter ranges for diversity
    hop_distribution = [1, 1, 2, 2, 2, 3, 3, 4]  # Weighted towards 2-3 hops
    hash_lengths = [8, 10, 12, 14, 16, 18, 20]  # Varying hash string lengths

    # Track hash string usage to ensure uniqueness
    used_hashes = set()
    collision_count = 0

    samples = []
    unique_prompts = set()

    print(f"Creating {n_samples} diverse hash-hop training samples...")

    with tqdm(total=n_samples) as pbar:
        attempts = 0
        while len(samples) < n_samples and attempts < n_samples * 3:
            attempts += 1

            # Randomly select parameters
            hops = random.choice(hop_distribution)
            hash_length = random.choice(hash_lengths)

            # Vary prompt size (staying well under token limit)
            # Smaller prompts = more variety in the dataset
            min_chars = 5000
            prompt_chars = random.randint(
                min_chars, max_chars - 1000
            )  # Leave room for completion

            # Vary number of queries
            num_queries = random.randint(3, 8)

            # Use chain of thought randomly (30% of the time)
            chain_of_thought = random.random() < 0.3

            try:
                # Generate sample
                sample = MultiHopEval.make_one(
                    n_chars_problem=prompt_chars,
                    num_queries=num_queries,
                    hops=hops,
                    hash_pair_str_length=hash_length,
                    chain_of_thought=chain_of_thought,
                )

                # Check prompt uniqueness (using hash for efficiency)
                prompt_hash = hashlib.md5(sample.prompt.encode()).hexdigest()
                if prompt_hash in unique_prompts:
                    collision_count += 1
                    continue

                unique_prompts.add(prompt_hash)

                # Extract all hash strings from the sample to track usage
                hash_strings = set()
                for line in sample.prompt.split("\n"):
                    if " = " in line:
                        parts = line.split(" = ")
                        for part in parts:
                            cleaned = part.strip().strip("'")
                            if (
                                len(cleaned) == hash_length
                                and cleaned.replace("_", "").isalnum()
                            ):
                                hash_strings.add(cleaned)

                # Track hash usage
                used_hashes.update(hash_strings)

                # Create training format
                # Format: prompt + completion as one sequence
                full_text = sample.prompt + "\n\n" + sample.completion

                # Estimate tokens
                estimated_tokens = estimate_tokens(len(full_text))

                if estimated_tokens > 8192:
                    # Skip if too long
                    continue

                # Store sample with metadata
                training_sample = {
                    "text": full_text,
                    "metadata": {
                        "hops": hops,
                        "hash_length": hash_length,
                        "num_queries": num_queries,
                        "chain_of_thought": chain_of_thought,
                        "prompt_chars": len(sample.prompt),
                        "total_chars": len(full_text),
                        "estimated_tokens": estimated_tokens,
                        "unique_hashes": len(hash_strings),
                    },
                }

                samples.append(training_sample)
                pbar.update(1)

            except Exception as e:
                print(f"Error generating sample: {e}")
                continue

    print(f"\nGenerated {len(samples)} unique samples")
    print(f"Unique hash strings used: {len(used_hashes)}")
    print(f"Prompt collisions avoided: {collision_count}")

    # Split into train/validation
    random.seed(seed)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.95)
    base_train_samples = samples[:split_idx]
    base_val_samples = samples[split_idx:]

    # Create multiple shuffled copies
    print(f"\nCreating {num_copies} shuffled copies for each split...")
    train_samples = replicate_and_shuffle_dataset(
        base_train_samples, num_copies, seed + 1000
    )
    val_samples = replicate_and_shuffle_dataset(
        base_val_samples, num_copies, seed + 2000
    )

    # Calculate statistics
    stats = {
        "unique_samples": len(samples),
        "unique_train_samples": len(base_train_samples),
        "unique_val_samples": len(base_val_samples),
        "total_train_samples": len(train_samples),
        "total_val_samples": len(val_samples),
        "num_copies": num_copies,
        "unique_hashes": len(used_hashes),
        "avg_chars": sum(s["metadata"]["total_chars"] for s in samples) / len(samples),
        "avg_tokens": sum(s["metadata"]["estimated_tokens"] for s in samples)
        / len(samples),
        "hop_distribution": {},
        "hash_length_distribution": {},
        "token_distribution": {},
    }

    # Calculate distributions
    for s in samples:
        meta = s["metadata"]
        stats["hop_distribution"][meta["hops"]] = (
            stats["hop_distribution"].get(meta["hops"], 0) + 1
        )
        stats["hash_length_distribution"][meta["hash_length"]] = (
            stats["hash_length_distribution"].get(meta["hash_length"], 0) + 1
        )

        token_bucket = (
            meta["estimated_tokens"] // 1000
        ) * 1000  # Round to nearest 1000
        stats["token_distribution"][f"{token_bucket}-{token_bucket + 1000}"] = (
            stats["token_distribution"].get(f"{token_bucket}-{token_bucket + 1000}", 0)
            + 1
        )

    # Save datasets
    print("\nSaving datasets...")

    # Save as JSONL for training (all copies)
    with open(f"{output_dir}/train.jsonl", "w") as f:
        for sample in train_samples:
            f.write(json.dumps({"text": sample["text"]}) + "\n")

    with open(f"{output_dir}/validation.jsonl", "w") as f:
        for sample in val_samples:
            f.write(json.dumps({"text": sample["text"]}) + "\n")

    # Save base samples with metadata for analysis (without copies)
    with open(f"{output_dir}/train_with_metadata.json", "w") as f:
        json.dump(base_train_samples, f, indent=2)

    with open(f"{output_dir}/validation_with_metadata.json", "w") as f:
        json.dump(base_val_samples, f, indent=2)

    # Save statistics
    with open(f"{output_dir}/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Unique samples generated: {stats['unique_samples']}")
    print(f"Number of copies per split: {stats['num_copies']}")
    print(f"Unique train samples: {stats['unique_train_samples']}")
    print(f"Unique validation samples: {stats['unique_val_samples']}")
    print(f"Total train samples (with copies): {stats['total_train_samples']}")
    print(f"Total validation samples (with copies): {stats['total_val_samples']}")
    print(f"Unique hash strings: {stats['unique_hashes']:,}")
    print(f"Average characters: {stats['avg_chars']:.0f}")
    print(f"Average tokens: {stats['avg_tokens']:.0f}")

    print("\nHop distribution:")
    for hops in sorted(stats["hop_distribution"].keys()):
        count = stats["hop_distribution"][hops]
        print(f"  {hops} hops: {count} ({count / len(samples) * 100:.1f}%)")

    print("\nToken length distribution:")
    for bucket in sorted(stats["token_distribution"].keys()):
        count = stats["token_distribution"][bucket]
        print(f"  {bucket} tokens: {count} ({count / len(samples) * 100:.1f}%)")

    print(f"\nDataset saved to: {output_dir}/")
    print("Files created:")
    print("  - train.jsonl (for training)")
    print("  - validation.jsonl (for evaluation)")
    print("  - train_with_metadata.json (for analysis)")
    print("  - validation_with_metadata.json (for analysis)")
    print("  - dataset_stats.json (statistics)")

    # Estimate hash reuse
    total_hash_opportunities = sum(s["metadata"]["unique_hashes"] for s in samples)
    reuse_rate = 1 - (len(used_hashes) / total_hash_opportunities)
    print(f"\nHash reuse rate: {reuse_rate:.2%}")
    print(
        f"Average hash appearances: {total_hash_opportunities / len(used_hashes):.2f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of unique samples to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hashhop_training_data",
        help="Output directory",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=24000,
        help="Max characters per sample (~8K tokens)",
    )
    parser.add_argument(
        "--num_copies",
        type=int,
        default=100,
        help="Number of shuffled copies to concatenate (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    create_diverse_hashhop_dataset(
        n_samples=args.n_samples,
        max_chars=args.max_chars,
        output_dir=args.output_dir,
        num_copies=args.num_copies,
        seed=args.seed,
    )
