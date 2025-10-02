#!/usr/bin/env python3
"""
Create a large 1-hop hash-hop training dataset to prevent memorization.
Generates enough samples so each appears at most once during 10k training steps.
"""

import json
import random
from pathlib import Path
from hashhop import MultiHopEval
from tqdm import tqdm
import hashlib


def create_1hop_training_dataset(
    n_samples=100000,  # 10x training steps = minimal repetition
    max_chars=20000,  # ~6.6K tokens, leaving room for batch processing
    output_dir="hashhop_1hop_training",
):
    """
    Create a 1-hop only training dataset with extreme diversity.
    """

    Path(output_dir).mkdir(exist_ok=True)

    # Parameter ranges for diversity
    hash_lengths = [4]  # Use 4-character hashes to reduce tokenization overhead

    # Track statistics
    used_hashes = set()
    unique_prompts = set()
    collision_count = 0

    samples = []

    print(f"Creating {n_samples:,} 1-hop hash-hop training samples...")
    print("This will take a while due to the large number of samples...\n")

    with tqdm(total=n_samples) as pbar:
        attempts = 0
        while len(samples) < n_samples and attempts < n_samples * 2:
            attempts += 1

            # Randomly select parameters
            hash_length = random.choice(hash_lengths)

            # Vary prompt size significantly to prevent pattern memorization
            # Smaller prompts = more samples fit in memory during training
            prompt_chars = random.randint(3000, max_chars)

            # Vary number of queries (1-10 for 1-hop is reasonable)
            num_queries = random.randint(1, 10)

            # Never use chain of thought for 1-hop (it's just A = 'B')
            chain_of_thought = False

            try:
                # Generate sample
                sample = MultiHopEval.make_one(
                    n_chars_problem=prompt_chars,
                    num_queries=num_queries,
                    hops=1,  # ONLY 1-hop
                    hash_pair_str_length=hash_length,
                    chain_of_thought=chain_of_thought,
                )

                # Check prompt uniqueness
                prompt_hash = hashlib.md5(sample.prompt.encode()).hexdigest()
                if prompt_hash in unique_prompts:
                    collision_count += 1
                    continue

                unique_prompts.add(prompt_hash)

                # Extract hash strings to track diversity
                hash_strings = set()
                for line in sample.prompt.split("\n"):
                    if " = " in line:
                        parts = line.split(" = ")
                        for part in parts:
                            cleaned = part.strip().strip("'")
                            if len(cleaned) >= 6 and cleaned.replace("_", "").isalnum():
                                hash_strings.add(cleaned)

                used_hashes.update(hash_strings)

                # Create training format
                full_text = sample.prompt + "\n\n" + sample.completion

                # Estimate tokens (conservative: 2.5 chars per token for hash-heavy content)
                estimated_tokens = len(full_text) // 2.5

                if estimated_tokens > 8192:
                    continue

                # Store sample
                training_sample = {
                    "text": full_text,
                    "metadata": {
                        "hash_length": hash_length,
                        "num_queries": num_queries,
                        "prompt_chars": len(sample.prompt),
                        "total_chars": len(full_text),
                        "estimated_tokens": int(estimated_tokens),
                        "unique_hashes": len(hash_strings),
                    },
                }

                samples.append(training_sample)
                pbar.update(1)

                # Periodic memory cleanup
                if len(samples) % 10000 == 0:
                    print(
                        f"\nCheckpoint: {len(samples):,} samples, {len(used_hashes):,} unique hashes"
                    )

            except Exception:
                if attempts % 1000 == 0:
                    print(
                        f"\nWarning: {attempts} attempts made, {len(samples)} successful"
                    )
                continue

    print(f"\n\nGenerated {len(samples):,} samples")
    print(f"Unique hash strings used: {len(used_hashes):,}")
    print(f"Prompt collisions avoided: {collision_count:,}")

    # Calculate hash uniqueness
    total_hash_slots = sum(s["metadata"]["unique_hashes"] for s in samples)
    hash_reuse_rate = 1 - (len(used_hashes) / total_hash_slots)
    print(f"Hash reuse rate: {hash_reuse_rate:.2%}")
    print(f"Average times each hash appears: {total_hash_slots / len(used_hashes):.2f}")

    # Split into train/validation (98/2 split due to large dataset)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.98)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save datasets
    print("\nSaving datasets...")

    # Save as JSONL for training (text only)
    print("Saving train.jsonl...")
    with open(f"{output_dir}/train.jsonl", "w") as f:
        for sample in tqdm(train_samples, desc="Writing training data"):
            f.write(json.dumps({"text": sample["text"]}) + "\n")

    print("Saving validation.jsonl...")
    with open(f"{output_dir}/validation.jsonl", "w") as f:
        for sample in tqdm(val_samples, desc="Writing validation data"):
            f.write(json.dumps({"text": sample["text"]}) + "\n")

    # Save smaller sample with metadata for analysis
    print("Saving sample with metadata...")
    sample_size = min(1000, len(train_samples))
    with open(f"{output_dir}/train_sample_with_metadata.json", "w") as f:
        json.dump(train_samples[:sample_size], f, indent=2)

    # Calculate and save statistics
    stats = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "unique_hashes": len(used_hashes),
        "hash_reuse_rate": hash_reuse_rate,
        "avg_chars": sum(s["metadata"]["total_chars"] for s in samples) / len(samples),
        "avg_tokens": sum(s["metadata"]["estimated_tokens"] for s in samples)
        / len(samples),
        "hash_length_distribution": {},
        "query_distribution": {},
        "token_distribution": {},
    }

    # Calculate distributions
    for s in samples:
        meta = s["metadata"]

        # Hash length distribution
        hl = meta["hash_length"]
        stats["hash_length_distribution"][hl] = (
            stats["hash_length_distribution"].get(hl, 0) + 1
        )

        # Query distribution
        nq = meta["num_queries"]
        stats["query_distribution"][nq] = stats["query_distribution"].get(nq, 0) + 1

        # Token distribution
        token_bucket = (meta["estimated_tokens"] // 1000) * 1000
        bucket_label = f"{token_bucket}-{token_bucket + 1000}"
        stats["token_distribution"][bucket_label] = (
            stats["token_distribution"].get(bucket_label, 0) + 1
        )

    with open(f"{output_dir}/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("1-HOP TRAINING DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Train samples: {stats['train_samples']:,}")
    print(f"Validation samples: {stats['val_samples']:,}")
    print(f"Unique hash strings: {stats['unique_hashes']:,}")
    print(f"Hash reuse rate: {stats['hash_reuse_rate']:.2%}")
    print(f"Average characters: {stats['avg_chars']:.0f}")
    print(f"Average tokens: {stats['avg_tokens']:.0f}")

    print("\nHash length distribution:")
    for hl in sorted(stats["hash_length_distribution"].keys()):
        count = stats["hash_length_distribution"][hl]
        print(f"  {hl:2d} chars: {count:6,} ({count / len(samples) * 100:4.1f}%)")

    print("\nQueries per sample:")
    for nq in sorted(stats["query_distribution"].keys()):
        count = stats["query_distribution"][nq]
        print(f"  {nq:2d} queries: {count:6,} ({count / len(samples) * 100:4.1f}%)")

    print(f"\nDataset saved to: {output_dir}/")

    # Training recommendations
    print("\n" + "=" * 60)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 60)
    print(f"With {len(train_samples):,} training samples:")
    print(f"- At batch size 8: {len(train_samples) // 8:,} gradient steps per epoch")
    print(f"- At batch size 16: {len(train_samples) // 16:,} gradient steps per epoch")
    print(f"- At batch size 32: {len(train_samples) // 32:,} gradient steps per epoch")
    print("\nFor 10k training steps:")
    print(
        f"- Batch size 8: {10000 / (len(train_samples) // 8):.2f} epochs (low repetition)"
    )
    print(
        f"- Batch size 16: {10000 / (len(train_samples) // 16):.2f} epochs (very low repetition)"
    )
    print(
        f"- Batch size 32: {10000 / (len(train_samples) // 32):.2f} epochs (minimal repetition)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100000,
        help="Number of samples (default: 100k for 10k training steps)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hashhop_1hop_training",
        help="Output directory",
    )

    args = parser.parse_args()

    create_1hop_training_dataset(n_samples=args.n_samples, output_dir=args.output_dir)
