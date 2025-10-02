#!/usr/bin/env python3
"""
Generate synthetic two-hop reasoning dataset.

This script creates a dataset with key:value pairs where:
- num_pairs: Number of base single-hop pairs
- num_two_hop: Number of two-hop pairs generated from chains (a,b) + (b,c) -> (a,c)

Format: "key:value " with space separation between pairs.
"""

import json
import random
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np


def determine_vocab_size(num_pairs: int) -> Tuple[int, int]:
    """
    Calculate the number of digits needed and max value for vocabulary.

    Args:
        num_pairs: Number of key-value pairs to generate

    Returns:
        Tuple of (num_digits, max_value)
    """
    # Need num_pairs unique values (each appears once as key, once as value)
    total_values_needed = num_pairs

    # Calculate minimum number of digits needed
    if total_values_needed <= 1:
        num_digits = 1
    else:
        num_digits = math.ceil(math.log10(total_values_needed))

    # Maximum value with num_digits
    max_value = 10**num_digits - 1

    # Ensure we have enough values
    if max_value + 1 < total_values_needed:
        num_digits += 1
        max_value = 10**num_digits - 1

    return num_digits, max_value


def create_base_pairs(
    num_pairs: int, num_digits: int, max_value: int
) -> List[Tuple[str, str]]:
    """
    Create base single-hop key-value pairs.
    Each value appears exactly once as a key and once as a value.

    Args:
        num_pairs: Number of pairs to create
        num_digits: Number of digits for formatting
        max_value: Maximum value in vocabulary

    Returns:
        List of (key, value) tuples with zero-padded strings
    """
    # Use values from 0 to num_pairs-1
    values = list(range(num_pairs))

    # Create a random permutation for the values
    # This ensures each value appears once as key and once as value
    keys = values.copy()
    targets = values.copy()
    random.shuffle(targets)

    # Ensure no self-loops (key != value)
    # If we find a self-loop, swap with another position
    for i in range(num_pairs):
        if keys[i] == targets[i]:
            # Find another position j where we can swap
            for j in range(num_pairs):
                if j != i and keys[j] != targets[i] and keys[i] != targets[j]:
                    targets[i], targets[j] = targets[j], targets[i]
                    break

    # Create pairs with zero-padding
    pairs = []
    for i in range(num_pairs):
        key_str = str(keys[i]).zfill(num_digits)
        value_str = str(targets[i]).zfill(num_digits)
        pairs.append((key_str, value_str))

    return pairs


def find_chainable_pairs(
    pairs: List[Tuple[str, str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Set[str]]:
    """
    Build data structures for efficient chain finding.

    Args:
        pairs: List of (key, value) tuples

    Returns:
        Tuple of:
        - key_to_value: Dict mapping keys to values
        - value_to_keys: Dict mapping values to list of keys
        - bridge_values: Set of values that can serve as bridges
    """
    # Build key_to_value mapping
    key_to_value = {}
    for key, value in pairs:
        key_to_value[key] = value

    # Build value_to_keys mapping
    value_to_keys = {}
    for key, value in pairs:
        if value not in value_to_keys:
            value_to_keys[value] = []
        value_to_keys[value].append(key)

    # Find bridge_values = values that are also keys
    all_keys = set(key_to_value.keys())
    all_values = set(value_to_keys.keys())
    bridge_values = all_keys & all_values

    return key_to_value, value_to_keys, bridge_values


def generate_two_hop_pairs(
    num_two_hop: int,
    key_to_value: Dict[str, str],
    value_to_keys: Dict[str, List[str]],
    bridge_values: Set[str],
    existing_pairs: Set[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Generate two-hop pairs from existing chains.

    Args:
        num_two_hop: Number of two-hop pairs to generate
        key_to_value: Mapping of keys to values
        value_to_keys: Mapping of values to keys that map to them
        bridge_values: Set of values that can serve as bridges
        existing_pairs: Set of existing pairs to avoid conflicts

    Returns:
        List of new two-hop (key, value) pairs
    """
    two_hop_pairs = []
    attempts = 0
    max_attempts = num_two_hop * 100  # Prevent infinite loops

    while len(two_hop_pairs) < num_two_hop and attempts < max_attempts:
        attempts += 1

        # If no bridge values exist, we can't create two-hop pairs
        if not bridge_values:
            print(
                f"Warning: No bridge values available. Generated {len(two_hop_pairs)} two-hop pairs."
            )
            break

        # 1. Randomly select bridge value b
        bridge = random.choice(list(bridge_values))

        # 2. Get all possible source keys a where a->b
        source_keys = value_to_keys.get(bridge, [])
        if not source_keys:
            continue

        # 3. Get target value c where b->c
        if bridge not in key_to_value:
            continue
        target_value = key_to_value[bridge]

        # 4. Randomly select source key
        source_key = random.choice(source_keys)

        # 5. Create new pair (source_key, target_value)
        new_pair = (source_key, target_value)

        # Check for conflicts
        if new_pair not in existing_pairs and source_key != target_value:
            two_hop_pairs.append(new_pair)
            existing_pairs.add(new_pair)

    if attempts >= max_attempts:
        print(
            f"Warning: Reached max attempts. Generated {len(two_hop_pairs)} two-hop pairs."
        )

    return two_hop_pairs


def validate_two_hop_consistency(
    base_pairs: List[Tuple[str, str]], two_hop_pairs: List[Tuple[str, str]]
) -> bool:
    """
    Validate that there are no duplicate pairs.
    Note: Keys mapping to multiple values is allowed (represents different reasoning paths).

    Args:
        base_pairs: Original single-hop pairs
        two_hop_pairs: Generated two-hop pairs

    Returns:
        True if no duplicates, False otherwise
    """
    # Check for duplicates within base pairs
    base_set = set(base_pairs)
    if len(base_set) != len(base_pairs):
        print("Warning: Duplicate pairs found in base pairs")
        return False

    # Check for duplicates within two-hop pairs
    two_hop_set = set(two_hop_pairs)
    if len(two_hop_set) != len(two_hop_pairs):
        print("Warning: Duplicate pairs found in two-hop pairs")
        return False

    # Check for overlap between base and two-hop
    overlap = base_set & two_hop_set
    if overlap:
        print(f"Warning: {len(overlap)} pairs appear in both base and two-hop sets")
        return False

    return True


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


def generate_dataset_examples(
    base_pairs: List[Tuple[str, str]], two_hop_pairs: List[Tuple[str, str]]
) -> List[Dict]:
    """
    Generate dataset examples in "key:value " format.

    Args:
        base_pairs: Single-hop pairs
        two_hop_pairs: Two-hop pairs

    Returns:
        List of examples with metadata
    """
    examples = []

    # Combine all pairs
    all_pairs = []

    # Add base pairs with metadata
    for key, value in base_pairs:
        all_pairs.append(
            {"pair": f"{key}:{value} ", "key": key, "value": value, "type": "base"}
        )

    # Add two-hop pairs with metadata
    for key, value in two_hop_pairs:
        all_pairs.append(
            {"pair": f"{key}:{value} ", "key": key, "value": value, "type": "two-hop"}
        )

    # Shuffle all pairs
    random.shuffle(all_pairs)

    # Create examples by concatenating pairs into sequences
    # Each example contains multiple key:value pairs
    pairs_per_example = 1  # Number of pairs per training example

    for i in range(0, len(all_pairs), pairs_per_example):
        batch = all_pairs[i : i + pairs_per_example]
        if batch:  # Only create example if we have pairs
            text = "".join([p["pair"] for p in batch])
            examples.append(
                {
                    "text": text.strip(),  # Remove trailing space
                    "num_pairs": len(batch),
                    "contains_two_hop": any(p["type"] == "two-hop" for p in batch),
                }
            )

    return examples


def generate_test_sets(
    base_pairs: List[Tuple[str, str]],
    two_hop_pairs: List[Tuple[str, str]],
    key_to_value: Dict[str, str],
    value_to_keys: Dict[str, List[str]],
    bridge_values: Set[str],
    num_test_pairs: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generate two test sets:
    1. Known pairs test: All base + two-hop pairs from training
    2. Novel two-hop test: New two-hop pairs not seen in training

    Args:
        base_pairs: Base single-hop pairs
        two_hop_pairs: Two-hop pairs used in training
        key_to_value: Mapping of keys to values
        value_to_keys: Mapping of values to keys
        bridge_values: Set of bridge values
        num_test_pairs: Number of novel pairs to generate

    Returns:
        Tuple of (known_pairs_test, novel_two_hop_test)
    """
    # Test Set 1: All training pairs
    known_pairs_test = base_pairs + two_hop_pairs

    # Test Set 2: Novel two-hop pairs
    training_pairs = set(base_pairs + two_hop_pairs)
    novel_two_hop_test = []
    novel_two_hop_set = set()

    while len(novel_two_hop_test) < num_test_pairs:
        if not bridge_values:
            raise ValueError(
                "No bridge values available - cannot generate two-hop pairs"
            )

        # Generate a two-hop pair
        bridge = random.choice(list(bridge_values))
        source_keys = value_to_keys.get(bridge, [])
        if not source_keys or bridge not in key_to_value:
            continue

        source_key = random.choice(source_keys)
        target_value = key_to_value[bridge]

        new_pair = (source_key, target_value)

        # Only add if it's novel (not in training) and not a duplicate
        if (
            new_pair not in training_pairs
            and new_pair not in novel_two_hop_set
            and source_key != target_value
        ):
            novel_two_hop_test.append(new_pair)
            novel_two_hop_set.add(new_pair)

    return known_pairs_test, novel_two_hop_test


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
        description="Generate synthetic two-hop reasoning dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_two_hop_data",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--num-pairs", type=int, default=1000, help="Number of base single-hop pairs"
    )
    parser.add_argument(
        "--num-two-hop",
        type=int,
        default=500,
        help="Number of two-hop pairs to generate",
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
        "--num-test-pairs",
        type=int,
        default=100,
        help="Number of novel two-hop pairs for generalization test set",
    )
    parser.add_argument(
        "--num-copies",
        type=int,
        default=100,
        help="Number of shuffled copies to concatenate (default: 100)",
    )

    args = parser.parse_args()

    # 1. Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic two-hop dataset...")
    print(f"Base pairs: {args.num_pairs}")
    print(f"Two-hop pairs: {args.num_two_hop}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")

    # 2. Determine vocabulary size
    num_digits, max_value = determine_vocab_size(args.num_pairs)
    print(f"Using {num_digits} digits, vocabulary size: {max_value + 1}")

    # 3. Create base pairs
    print("\nCreating base pairs...")
    base_pairs = create_base_pairs(args.num_pairs, num_digits, max_value)
    print(f"Created {len(base_pairs)} base pairs")

    # 4. Find chainable pairs
    print("\nFinding chainable pairs...")
    key_to_value, value_to_keys, bridge_values = find_chainable_pairs(base_pairs)
    print(f"Found {len(bridge_values)} potential bridge values")

    # 5. Generate two-hop pairs
    print("\nGenerating two-hop pairs...")
    existing_pairs = set(base_pairs)
    two_hop_pairs = generate_two_hop_pairs(
        args.num_two_hop, key_to_value, value_to_keys, bridge_values, existing_pairs
    )
    print(f"Generated {len(two_hop_pairs)} two-hop pairs")

    # 6. Validate consistency
    print("\nValidating consistency...")
    if validate_two_hop_consistency(base_pairs, two_hop_pairs):
        print("All pairs are consistent!")
    else:
        print("Warning: Found conflicts in pairs")

    # 7. Generate examples
    print("\nGenerating dataset examples...")
    base_examples = generate_dataset_examples(base_pairs, two_hop_pairs)

    # 8. Create multiple shuffled copies for each split
    print(f"\nCreating {args.num_copies} shuffled copies for each split...")
    train_examples = replicate_and_shuffle_dataset(
        base_examples, args.num_copies, args.seed + 0
    )
    val_examples = replicate_and_shuffle_dataset(
        base_examples, args.num_copies, args.seed + 1000
    )
    test_examples = replicate_and_shuffle_dataset(
        base_examples, args.num_copies, args.seed + 2000
    )

    # 9. Save train/val/test splits
    print("\nSaving datasets...")
    save_dataset(train_examples, output_dir, args.output_format, "train")
    save_dataset(val_examples, output_dir, args.output_format, "validation")
    save_dataset(test_examples, output_dir, args.output_format, "test")

    # 9. Generate and save test sets
    print("\nGenerating test sets...")
    known_pairs_test, novel_two_hop_test = generate_test_sets(
        base_pairs,
        two_hop_pairs,
        key_to_value,
        value_to_keys,
        bridge_values,
        args.num_test_pairs,
    )

    # Create separate directory for special test sets
    special_tests_dir = output_dir.parent / f"{output_dir.name}_special_tests"
    special_tests_dir.mkdir(parents=True, exist_ok=True)

    # Create examples for test set 1 (known pairs)
    print(f"\nTest Set 1: {len(known_pairs_test)} known pairs")
    known_test_examples = []
    for key, value in known_pairs_test:
        known_test_examples.append(
            {"text": f"{key}:{value}", "key": key, "value": value, "type": "known"}
        )
    save_dataset(
        known_test_examples, special_tests_dir, args.output_format, "test_known_pairs"
    )

    # Create examples for test set 2 (novel two-hop)
    print(f"\nTest Set 2: {len(novel_two_hop_test)} novel two-hop pairs")
    novel_test_examples = []
    for key, value in novel_two_hop_test:
        novel_test_examples.append(
            {
                "text": f"{key}:{value}",
                "key": key,
                "value": value,
                "type": "novel_two_hop",
            }
        )
    save_dataset(
        novel_test_examples, special_tests_dir, args.output_format, "test_novel_two_hop"
    )

    # 10. Save metadata
    metadata = {
        "num_pairs": args.num_pairs,
        "num_two_hop": args.num_two_hop,
        "num_test_pairs": args.num_test_pairs,
        "num_digits": num_digits,
        "vocab_size": max_value + 1,
        "seed": args.seed,
        "num_copies": args.num_copies,
        "num_bridge_values": len(bridge_values),
        "total_base_pairs": len(base_pairs),
        "total_two_hop_pairs": len(two_hop_pairs),
        "unique_examples": len(base_examples),
        "total_examples_per_split": len(train_examples),
        "test_equals_train": True,
        "val_equals_train": True,
        "known_pairs_test_size": len(known_pairs_test),
        "novel_two_hop_test_size": len(novel_two_hop_test),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset generation complete!")
    print(f"Unique training examples: {len(base_examples)}")
    print(
        f"Total examples per split (with {args.num_copies} copies): {len(train_examples)}"
    )
    print(f"Total training pairs: {len(base_pairs) + len(two_hop_pairs)}")
    print(f"Known pairs test: {len(known_pairs_test)} pairs")
    print(f"Novel two-hop test: {len(novel_two_hop_test)} pairs")
    print("Train/val/test splits are IDENTICAL for pure memorization testing")


if __name__ == "__main__":
    main()
