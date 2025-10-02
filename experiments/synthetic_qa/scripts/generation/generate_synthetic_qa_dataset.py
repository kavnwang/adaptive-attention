#!/usr/bin/env python3
"""
Generate synthetic QA-recall dataset for memory layer training.

This script creates a dataset with unique (entity_x, entity_y) pairs,
each repeated multiple times with different exposure frequencies to study
memorization vs. generalization in memory-augmented language models.

Key features:
- Entity format: 6-digit numbers (000000 to 999999)
- Digit lookup format: "key:value " with space separation
- Controllable exposure frequency distribution
- Train/val/test splits with held-out items in each frequency bucket
- JSONL/Parquet output format compatible with LLMonade training infrastructure
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np


def generate_entity_id(entity_num: int) -> str:
    """Generate entity ID as 6-digit number."""
    return f"{entity_num:06d}"


def create_entity_pairs(
    num_pairs: int, vocab_size: int = 2_000_000
) -> List[Tuple[str, str]]:
    """
    Create unique (entity_x, entity_y) pairs.

    Args:
        num_pairs: Number of unique pairs to generate
        vocab_size: Total vocabulary size (should be >= 2 * num_pairs)

    Returns:
        List of (entity_x, entity_y) tuples
    """
    if vocab_size < 2 * num_pairs:
        raise ValueError(f"Vocab size {vocab_size} too small for {num_pairs} pairs")

    print(f"Generating {num_pairs} unique entity pairs...")

    # Sample entities without replacement
    entity_indices = random.sample(range(vocab_size), 2 * num_pairs)

    pairs = []
    for i in range(num_pairs):
        entity_x = generate_entity_id(entity_indices[2 * i])
        entity_y = generate_entity_id(entity_indices[2 * i + 1])
        pairs.append((entity_x, entity_y))

    return pairs


def create_exposure_buckets(
    pairs: List[Tuple[str, str]],
    freq_less: int = 1,
    freq_more: int = 10,
    ratio_less: float = 0.5,
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Distribute pairs into two frequency buckets.

    Args:
        pairs: List of (entity_x, entity_y) pairs
        freq_less: Number of times to repeat pairs in the low frequency bucket
        freq_more: Number of times to repeat pairs in the high frequency bucket
        ratio_less: Fraction of pairs to put in the low frequency bucket

    Returns:
        Dict mapping frequency -> list of pairs for that frequency
    """
    # Shuffle pairs
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)

    # Split into two buckets
    n_less = int(len(pairs) * ratio_less)

    buckets = {freq_less: shuffled_pairs[:n_less], freq_more: shuffled_pairs[n_less:]}

    print("Frequency distribution:")
    print(f"  Bucket {freq_less}x: {len(buckets[freq_less])} pairs")
    print(f"  Bucket {freq_more}x: {len(buckets[freq_more])} pairs")
    print(
        f"Total examples to be generated: {n_less * freq_less + (len(pairs) - n_less) * freq_more}"
    )

    return buckets


def create_train_val_test_split(
    buckets: Dict[int, List[Tuple[str, str]]],
) -> Tuple[Dict, Dict, Dict]:
    """
    Create train/val/test splits where all sets are identical.
    This allows testing pure memorization without generalization.

    Args:
        buckets: Frequency buckets

    Returns:
        Tuple of (train_buckets, val_buckets, test_buckets) - all identical
    """
    # All splits get the same data
    train_buckets = buckets.copy()
    val_buckets = buckets.copy()
    test_buckets = buckets.copy()

    print("\nSplit information:")
    print(
        "Train, validation, and test sets are IDENTICAL for pure memorization testing"
    )

    return train_buckets, val_buckets, test_buckets


def generate_qa_examples(
    buckets: Dict[int, List[Tuple[str, str]]], include_inverse: bool = False
) -> List[Dict]:
    """
    Generate QA examples from frequency buckets using digit lookup format.

    Args:
        buckets: Frequency buckets
        include_inverse: Whether to include inverse recall questions

    Returns:
        List of QA examples in digit lookup format ("key:value ")
    """
    examples = []

    for freq, pairs in buckets.items():
        for entity_x, entity_y in pairs:
            # Generate 'freq' copies of this pair
            for _ in range(freq):
                # Forward recall: "<X>:" -> "<Y>"
                examples.append(
                    {
                        "text": f"{entity_x}:{entity_y} ",
                        "frequency": freq,
                        "entity_x": entity_x,
                        "entity_y": entity_y,
                        "type": "forward",
                    }
                )

                # Inverse recall: "<Y>:" -> "<X>"
                if include_inverse:
                    examples.append(
                        {
                            "text": f"{entity_y}:{entity_x} ",
                            "frequency": freq,
                            "entity_x": entity_x,
                            "entity_y": entity_y,
                            "type": "inverse",
                        }
                    )

    return examples


def save_jsonl(examples: List[Dict], filepath: Path):
    """Save examples to JSONL file."""
    with open(filepath, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(examples)} examples to {filepath}")


def save_arrow(examples: List[Dict], filepath: Path):
    """Save examples to Arrow format using HuggingFace datasets."""
    from datasets import Dataset

    # Convert list of dicts to dataset
    dataset = Dataset.from_list(examples)

    # Create the directory if it doesn't exist
    filepath.mkdir(parents=True, exist_ok=True)

    # Save as parquet file (widely compatible format)
    parquet_file = filepath / "data.parquet"
    dataset.to_parquet(str(parquet_file))
    print(f"Saved {len(examples)} examples to {parquet_file} in Parquet format")


def replicate_and_shuffle_dataset(
    examples: List[Dict], num_copies: int, base_seed: int
) -> List[Dict]:
    """
    Create multiple shuffled copies of the dataset and concatenate them.

    Args:
        examples: Original dataset examples
        num_copies: Number of copies to create (100)
        base_seed: Base random seed

    Returns:
        Concatenated list of all shuffled copies
    """
    all_examples = []

    for i in range(num_copies):
        # Create a copy of the examples
        copy_examples = examples.copy()

        # Shuffle with a different seed for each copy
        random.seed(base_seed + i)
        random.shuffle(copy_examples)

        # Add to the full list
        all_examples.extend(copy_examples)

    return all_examples


def generate_entity_vocab(vocab_size: int, output_path: Path):
    """Generate entity vocabulary file."""
    entities = [generate_entity_id(i) for i in range(vocab_size)]

    with open(output_path, "w") as f:
        for entity in entities:
            f.write(entity + "\n")

    print(f"Saved entity vocabulary ({vocab_size} entities) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic QA-recall dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_qa_data",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--num-pairs", type=int, default=500, help="Number of unique entity pairs"
    )
    parser.add_argument(
        "--freq-less", type=int, default=1, help="Frequency for low-frequency bucket"
    )
    parser.add_argument(
        "--freq-more", type=int, default=10, help="Frequency for high-frequency bucket"
    )
    parser.add_argument(
        "--ratio-less",
        type=float,
        default=0.5,
        help="Fraction of pairs in low-frequency bucket",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=10000, help="Total entity vocabulary size"
    )
    parser.add_argument(
        "--include-inverse",
        action="store_true",
        help="Include inverse recall questions",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["parquet", "jsonl"],
        help="Output format for dataset files (parquet or jsonl)",
    )
    parser.add_argument(
        "--num-copies",
        type=int,
        default=100,
        help="Number of shuffled copies to concatenate (default: 100)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic QA dataset with {args.num_pairs} pairs...")
    print(f"Low frequency: {args.freq_less}, High frequency: {args.freq_more}")
    print(f"Ratio in low frequency bucket: {args.ratio_less}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("NOTE: Test and validation sets will be IDENTICAL to train set")

    # Generate entity pairs
    pairs = create_entity_pairs(args.num_pairs, args.vocab_size)

    # Create two frequency buckets
    buckets = create_exposure_buckets(
        pairs, args.freq_less, args.freq_more, args.ratio_less
    )

    # Create identical train/val/test splits
    train_buckets, val_buckets, test_buckets = create_train_val_test_split(buckets)

    # Generate QA examples
    print("Generating training examples...")
    base_train_examples = generate_qa_examples(train_buckets, args.include_inverse)

    print("Generating validation examples (identical to train)...")
    base_val_examples = generate_qa_examples(val_buckets, args.include_inverse)

    print("Generating test examples (identical to train)...")
    base_test_examples = generate_qa_examples(test_buckets, args.include_inverse)

    # Create multiple shuffled copies
    print(f"\nCreating {args.num_copies} shuffled copies for each split...")
    train_examples = replicate_and_shuffle_dataset(
        base_train_examples, args.num_copies, args.seed + 0
    )
    val_examples = replicate_and_shuffle_dataset(
        base_val_examples, args.num_copies, args.seed + 1000
    )
    test_examples = replicate_and_shuffle_dataset(
        base_test_examples, args.num_copies, args.seed + 2000
    )

    # Save datasets
    if args.output_format == "parquet":
        # Create subdirectories for parquet format
        save_arrow(train_examples, output_dir / "train")
        save_arrow(val_examples, output_dir / "validation")
        save_arrow(test_examples, output_dir / "test")
    else:
        save_jsonl(train_examples, output_dir / "train.jsonl")
        save_jsonl(val_examples, output_dir / "validation.jsonl")
        save_jsonl(test_examples, output_dir / "test.jsonl")

    # Generate entity vocabulary
    generate_entity_vocab(args.vocab_size, output_dir / "entity_vocab.txt")

    # Save metadata
    metadata = {
        "num_pairs": args.num_pairs,
        "freq_less": args.freq_less,
        "freq_more": args.freq_more,
        "ratio_less": args.ratio_less,
        "vocab_size": args.vocab_size,
        "include_inverse": args.include_inverse,
        "test_equals_train": True,
        "val_equals_train": True,
        "seed": args.seed,
        "num_copies": args.num_copies,
        "frequency_buckets": list(buckets.keys()),
        "pairs_per_bucket": {str(k): len(v) for k, v in buckets.items()},
        "unique_examples_per_split": len(base_train_examples),
        "total_train_examples": len(train_examples),
        "total_val_examples": len(val_examples),
        "total_test_examples": len(test_examples),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset generation complete!")
    print(f"Total unique examples per split: {len(base_train_examples)}")
    print(
        f"Total examples per split (with {args.num_copies} copies): {len(train_examples)}"
    )
    print(
        f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}"
    )
    print("All splits are IDENTICAL for pure memorization testing")


if __name__ == "__main__":
    main()
