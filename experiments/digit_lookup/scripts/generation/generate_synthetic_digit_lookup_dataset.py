#!/usr/bin/env python3
"""
Generate synthetic digit lookup dataset.

This script creates a dataset with key:value pairs where:
- Keys are N-digit numbers where each digit is in range [1, N]
- Values are computed by: Value[i] = Key[Key[i]-1] (using 1-based indexing)
- num_keys distinct keys are sampled from the vocabulary
- Each split contains 100 shuffled copies concatenated together

Format: "key:value " with space separation between pairs.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np


def determine_n_and_vocab_size(num_keys: int) -> Tuple[int, int]:
    """
    Calculate the minimum N such that N^N >= num_keys.

    Args:
        num_keys: Number of distinct keys needed

    Returns:
        Tuple of (N, vocab_size) where vocab_size = N^N
    """
    n = 1
    while n**n < num_keys:
        n += 1

    vocab_size = n**n
    return n, vocab_size


def generate_all_valid_numbers(n: int) -> List[str]:
    """
    Generate all valid N-digit numbers where each digit is in range [1, N].

    Args:
        n: Number of digits and maximum digit value

    Returns:
        List of all valid N-digit numbers as strings
    """
    import itertools

    # Generate all combinations of digits [1, 2, ..., n]
    digits = [str(i) for i in range(1, n + 1)]

    # Generate all n-length combinations with replacement
    all_numbers = []
    for combo in itertools.product(digits, repeat=n):
        number = "".join(combo)
        all_numbers.append(number)

    return all_numbers


def compute_value_from_key(key: str, n: int, recursive_steps: int = 1) -> str:
    """
    Apply the transformation rule to compute value from key.
    Value[i] = Key[Key[i]-1] where indexing is 1-based for digit values.
    For recursive_steps > 1, apply the transformation multiple times.

    Args:
        key: N-digit key string
        n: Number of digits
        recursive_steps: Number of times to apply the transformation (default: 1)

    Returns:
        N-digit value string after applying transformation recursive_steps times
    """
    current_value = key

    for step in range(recursive_steps):
        next_value = []

        for i in range(n):
            # Get the digit at position i in the current value
            digit_at_i = int(current_value[i])

            # Use that digit to index into the current value (subtract 1 for 0-based indexing)
            # digit_at_i is in range [1, n], so digit_at_i - 1 is in range [0, n-1]
            index = digit_at_i - 1

            # Get the digit at that index in the current value
            value_digit = current_value[index]

            # Add to our next value
            next_value.append(value_digit)

        current_value = "".join(next_value)

    return current_value


def create_dataset(
    num_keys: int, n: int, seed: int, recursive_steps: int = 1
) -> List[Tuple[str, str]]:
    """
    Create the dataset by generating all valid numbers, shuffling, and taking num_keys.

    Args:
        num_keys: Number of distinct keys to use
        n: Number of digits
        seed: Random seed for shuffling
        recursive_steps: Number of times to apply the transformation (default: 1)

    Returns:
        List of (key, value) tuples
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Generate all valid numbers
    all_numbers = generate_all_valid_numbers(n)

    # Shuffle the list
    random.shuffle(all_numbers)

    # Take first num_keys as keys
    keys = all_numbers[:num_keys]

    # Compute value for each key
    pairs = []
    for key in keys:
        value = compute_value_from_key(key, n, recursive_steps)
        pairs.append((key, value))

    return pairs


def format_dataset_examples(pairs: List[Tuple[str, str]]) -> List[Dict]:
    """
    Format key-value pairs as dataset examples.

    Args:
        pairs: List of (key, value) tuples

    Returns:
        List of dataset examples with "text" field
    """
    examples = []
    for key, value in pairs:
        # Format as "key:value " with trailing space
        text = f"{key}:{value} "
        examples.append({"text": text})

    return examples


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


def generate_test_sets(
    training_pairs: List[Tuple[str, str]],
    n: int,
    num_test_pairs: int,
    seed: int,
    recursive_steps: int = 1,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generate two test sets:
    1. Known pairs test: All training pairs
    2. Novel pairs test: New pairs with keys not seen in training

    Args:
        training_pairs: Pairs used in training
        n: Number of digits
        num_test_pairs: Number of novel pairs to generate
        seed: Random seed
        recursive_steps: Number of times to apply the transformation (default: 1)

    Returns:
        Tuple of (known_pairs_test, novel_pairs_test)
    """
    # Test Set 1: All training pairs
    known_pairs_test = training_pairs.copy()

    # Test Set 2: Novel pairs
    # Get all training keys
    training_keys = set(key for key, _ in training_pairs)

    # Generate all valid numbers
    all_numbers = generate_all_valid_numbers(n)

    # Filter out training keys
    novel_keys = [num for num in all_numbers if num not in training_keys]

    # Shuffle novel keys
    random.seed(seed + 10000)  # Different seed for novel pairs
    random.shuffle(novel_keys)

    # Take up to num_test_pairs novel keys
    selected_novel_keys = novel_keys[:num_test_pairs]

    # Generate pairs for novel keys
    novel_pairs_test = []
    for key in selected_novel_keys:
        value = compute_value_from_key(key, n, recursive_steps)
        novel_pairs_test.append((key, value))

    return known_pairs_test, novel_pairs_test


def save_dataset(
    examples: List[Dict], output_dir: Path, output_format: str, split_name: str
):
    """
    Save dataset to file in specified format.

    Args:
        examples: List of dataset examples
        output_dir: Output directory
        output_format: Format (jsonl or parquet)
        split_name: Name of split (train/val/test)
    """
    if output_format == "jsonl":
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(examples)} examples to {filepath}")
    else:  # parquet format
        from datasets import Dataset

        # Create subdirectory for parquet format
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Convert to dataset and save
        dataset = Dataset.from_list(examples)
        parquet_file = split_dir / "data.parquet"
        dataset.to_parquet(str(parquet_file))
        print(f"Saved {len(examples)} examples to {parquet_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic digit lookup dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_digit_lookup_data",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--num-keys", type=int, default=1000, help="Number of distinct keys to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["parquet", "jsonl"],
        help="Output format for dataset files",
    )
    parser.add_argument(
        "--num-copies",
        type=int,
        default=100,
        help="Number of shuffled copies to concatenate (default: 100)",
    )
    parser.add_argument(
        "--num-test-pairs",
        type=int,
        default=100,
        help="Number of novel pairs for generalization test set",
    )
    parser.add_argument(
        "--recursive-steps",
        type=int,
        default=1,
        help="Number of recursive lookup steps to apply (default: 1)",
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory with num_keys in the name
    base_output_dir = Path(args.output_dir)
    # If the output dir doesn't already include the number of keys, add it
    if f"_{args.num_keys}" not in str(base_output_dir):
        output_dir = base_output_dir.parent / f"{base_output_dir.name}_{args.num_keys}"
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic digit lookup dataset...")
    print(f"Number of keys: {args.num_keys}")
    print(f"Number of copies per split: {args.num_copies}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")

    # Determine N and vocabulary size
    n, vocab_size = determine_n_and_vocab_size(args.num_keys)
    print(f"\nUsing N={n} (digits 1-{n}), vocabulary size: {vocab_size:,}")

    # Create base dataset
    print("\nCreating base dataset...")
    print(f"Recursive steps: {args.recursive_steps}")
    pairs = create_dataset(args.num_keys, n, args.seed, args.recursive_steps)
    print(f"Created {len(pairs)} key-value pairs")

    # Format as dataset examples
    examples = format_dataset_examples(pairs)

    # Create train/val/test splits (each with num_copies shuffled copies)
    print(f"\nCreating {args.num_copies} shuffled copies for each split...")

    for split_name, split_seed_offset in [
        ("train", 0),
        ("validation", 1000),
        ("test", 2000),
    ]:
        print(f"\nGenerating {split_name} split...")
        split_examples = replicate_and_shuffle_dataset(
            examples, args.num_copies, args.seed + split_seed_offset
        )
        save_dataset(split_examples, output_dir, args.output_format, split_name)

    # Generate and save test sets
    print("\nGenerating test sets...")
    known_pairs_test, novel_pairs_test = generate_test_sets(
        pairs, n, args.num_test_pairs, args.seed, args.recursive_steps
    )

    # Create separate directory for special test sets
    special_tests_dir = output_dir.parent / f"{output_dir.name}_special_tests"
    special_tests_dir.mkdir(parents=True, exist_ok=True)

    # Save known pairs test set
    print(f"\nTest Set 1: {len(known_pairs_test)} known pairs (same as training)")
    known_test_examples = format_dataset_examples(known_pairs_test)
    save_dataset(
        known_test_examples, special_tests_dir, args.output_format, "test_known_pairs"
    )

    # Save novel pairs test set
    print(f"\nTest Set 2: {len(novel_pairs_test)} novel pairs (not seen in training)")
    novel_test_examples = format_dataset_examples(novel_pairs_test)
    save_dataset(
        novel_test_examples, special_tests_dir, args.output_format, "test_novel_pairs"
    )

    # Save metadata
    metadata = {
        "num_keys": args.num_keys,
        "n": n,
        "vocab_size": vocab_size,
        "seed": args.seed,
        "num_copies": args.num_copies,
        "recursive_steps": args.recursive_steps,
        "total_examples_per_split": len(examples) * args.num_copies,
        "format": args.output_format,
        "transformation_rule": f"Value[i] = Key[Key[i]-1] (1-based indexing) applied {args.recursive_steps} time(s)",
        "test_sets": {
            "known_pairs_size": len(known_pairs_test),
            "novel_pairs_size": len(novel_pairs_test),
            "novel_pairs_requested": args.num_test_pairs,
            "possible_novel_pairs": vocab_size - args.num_keys,
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset generation complete!")
    print(f"Total examples per split: {len(examples) * args.num_copies:,}")
    print(f"Known pairs test: {len(known_pairs_test)} pairs")
    print(f"Novel pairs test: {len(novel_pairs_test)} pairs")
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
