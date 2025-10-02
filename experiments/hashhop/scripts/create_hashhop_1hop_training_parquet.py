#!/usr/bin/env python3
"""
Create a large 1-hop hash-hop training dataset in parquet format.
Compatible with LLMonade's training pipeline.
"""

import json
import random
from pathlib import Path
from hashhop import MultiHopEval
from tqdm import tqdm
import hashlib
from datasets import Dataset


def create_1hop_training_dataset(
    n_samples=100000,  # 10x training steps = minimal repetition
    max_chars=20000,  # ~6.6K tokens, leaving room for batch processing
    output_dir="hashhop_1hop_training_parquet",
):
    """
    Create a 1-hop only training dataset with extreme diversity.
    Save in parquet format compatible with LLMonade training.
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

                # Store in format expected by datasets
                training_sample = {"text": full_text}

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

    # Split into train/validation (98/2 split due to large dataset)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.98)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save datasets in parquet format
    print("\nSaving datasets in parquet format...")

    # Create subdirectories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Save train set
    print("Creating train dataset...")
    train_dataset = Dataset.from_list(train_samples)
    train_dataset.to_parquet(str(train_dir / "data.parquet"))
    print(f"Saved {len(train_samples)} train samples to {train_dir}/data.parquet")

    # Save validation set
    print("Creating validation dataset...")
    val_dataset = Dataset.from_list(val_samples)
    val_dataset.to_parquet(str(val_dir / "data.parquet"))
    print(f"Saved {len(val_samples)} validation samples to {val_dir}/data.parquet")

    # Save metadata
    metadata = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "unique_hashes": len(used_hashes),
        "format": "parquet",
        "hops": 1,
        "description": "1-hop hash-hop training dataset for memory layer evaluation",
    }

    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("1-HOP TRAINING DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(samples):,}")
    print(f"Train samples: {len(train_samples):,}")
    print(f"Validation samples: {len(val_samples):,}")
    print(f"Unique hash strings: {len(used_hashes):,}")
    print(f"Dataset saved to: {output_dir}/")
    print("\nFiles created:")
    print("  - train/data.parquet")
    print("  - validation/data.parquet")
    print("  - metadata.json")

    # Training recommendations
    print("\n" + "=" * 60)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 60)
    print(f"With {len(train_samples):,} training samples:")
    print(f"- At batch size 8: {len(train_samples) // 8:,} gradient steps per epoch")
    print(f"- At batch size 16: {len(train_samples) // 16:,} gradient steps per epoch")
    print(f"- At batch size 32: {len(train_samples) // 32:,} gradient steps per epoch")
    print("\nFor 20k training steps:")
    print(f"- Batch size 8: {20000 / (len(train_samples) // 8):.2f} epochs")
    print(f"- Batch size 16: {20000 / (len(train_samples) // 16):.2f} epochs")
    print(f"- Batch size 32: {20000 / (len(train_samples) // 32):.2f} epochs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100000,
        help="Number of samples (default: 100k for 20k training steps)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hashhop_1hop_training_parquet",
        help="Output directory",
    )

    args = parser.parse_args()

    create_1hop_training_dataset(n_samples=args.n_samples, output_dir=args.output_dir)
