#!/usr/bin/env python3
"""
Generate synthetic MQAR (Multi-Query Associative Recall) dataset.

This script creates a dataset where each example consists of:
1. num_kv_pairs key-value pairs (for storage)
2. num_kv_pairs queries (keys followed by their corresponding values)

The query order is determined by a power law distribution, where keys
appearing earlier in the context are more likely to be queried first.

Format: Each sequence contains multiple concatenated MQAR examples.
No padding or random tokens are used.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from transformers import AutoTokenizer


def generate_single_mqar_example(
    num_kv_pairs: int,
    token_pool: List[int],
    pool_index: int,
    power_a: float,
    rng: np.random.RandomState
) -> Tuple[List[int], int]:
    """
    Generate a single MQAR example with power law query ordering.

    Args:
        num_kv_pairs: Number of key-value pairs
        token_pool: Pre-shuffled pool of tokens to use
        pool_index: Current index in the token pool
        power_a: Power law parameter for query ordering
        rng: Random number generator

    Returns:
        Tuple of (token list for the example, updated pool_index)
    """
    example_tokens = []
    keys = []
    values = []
    
    # Build KV pairs section using sequential tokens from pool
    for _ in range(num_kv_pairs):
        key = token_pool[pool_index % len(token_pool)]
        pool_index += 1
        value = token_pool[pool_index % len(token_pool)]
        pool_index += 1
        
        keys.append(key)
        values.append(value)
        example_tokens.extend([key, value])
    
    # Determine query order using power law
    query_order = get_query_order_power_law(num_kv_pairs, power_a, rng)
    
    # Build queries section
    for idx in query_order:
        example_tokens.extend([keys[idx], values[idx]])
    
    return example_tokens, pool_index


def get_query_order_power_law(
    num_kv_pairs: int,
    power_a: float,
    rng: np.random.RandomState
) -> List[int]:
    """
    Determine query order using power law distribution.
    
    The power law is applied to the "distance" between where a key appeared
    in the KV context and where it's queried. Smaller distances are more likely.
    
    Args:
        num_kv_pairs: Number of KV pairs
        power_a: Power law parameter (smaller = more concentrated near beginning)
        rng: Random number generator
        
    Returns:
        List of indices indicating query order
    """
    # For each position, calculate probability based on distance from KV section
    # Distance 1 means querying the last KV pair first, distance num_kv_pairs means querying the first
    distances = np.arange(1, num_kv_pairs + 1)
    
    # Power law: P(distance) ∝ distance^(power_a - 1)
    # With power_a = 0.01, smaller distances (recent KVs) are much more likely
    probabilities = power_a * distances ** (power_a - 1)
    probabilities = probabilities / probabilities.sum()
    
    # For each query position, sample which key to query based on distance
    query_order = []
    available_indices = list(range(num_kv_pairs))
    
    for _ in range(num_kv_pairs):
        # Calculate current distances for available keys
        current_distances = []
        current_probs = []
        
        for idx in available_indices:
            # Distance from end of KV section (reverse index)
            distance = num_kv_pairs - idx
            prob = power_a * distance ** (power_a - 1)
            current_distances.append(distance)
            current_probs.append(prob)
        
        # Normalize probabilities
        current_probs = np.array(current_probs)
        current_probs = current_probs / current_probs.sum()
        
        # Sample next query
        chosen_idx = rng.choice(len(available_indices), p=current_probs)
        query_idx = available_indices[chosen_idx]
        
        query_order.append(query_idx)
        available_indices.remove(query_idx)
    
    return query_order


def generate_mqar_sequence(
    seq_len: int,
    num_kv_pairs: int,
    vocab_size: int,
    power_a: float,
    seed: int,
    tokenizer
) -> str:
    """
    Generate a full MQAR sequence by concatenating multiple examples.
    
    Args:
        seq_len: Target sequence length
        num_kv_pairs: Number of KV pairs per example
        vocab_size: Total vocabulary size
        power_a: Power law parameter
        seed: Random seed
        tokenizer: Tokenizer for decoding
        
    Returns:
        Text string for the full sequence
    """
    rng = np.random.RandomState(seed)
    
    # Shuffle entire vocabulary for this sequence
    all_tokens = list(range(vocab_size))
    rng.shuffle(all_tokens)
    token_pool = all_tokens[:8192]  # Take first 8192 tokens as pool
    pool_index = 0
    
    # Generate tokens for the full sequence
    sequence_tokens = []
    
    # Keep generating MQAR examples until we fill the sequence
    while len(sequence_tokens) < seq_len:
        # Generate one example
        example_tokens, pool_index = generate_single_mqar_example(
            num_kv_pairs, token_pool, pool_index, power_a, rng
        )
        sequence_tokens.extend(example_tokens)
    
    # Truncate to exact sequence length
    sequence_tokens = sequence_tokens[:seq_len]
    
    # Decode entire sequence at once
    full_text = tokenizer.decode(sequence_tokens)
    
    return full_text


def create_dataset(
    num_samples: int,
    seq_len: int,
    num_kv_pairs: int,
    vocab_size: int,
    power_a: float,
    seed: int,
    tokenizer
) -> List[Dict]:
    """
    Create the full dataset.
    
    Args:
        num_samples: Number of sequences to generate
        seq_len: Sequence length
        num_kv_pairs: Number of KV pairs per example
        vocab_size: Vocabulary size
        power_a: Power law parameter
        seed: Base random seed
        
    Returns:
        List of dataset examples
    """
    examples = []
    
    for i in range(num_samples):
        # Use different seed for each example
        example_seed = seed + i
        text = generate_mqar_sequence(
            seq_len, num_kv_pairs, vocab_size, power_a, example_seed, tokenizer
        )
        
        # Create example dict with only text field
        example = {
            "text": text
        }
        
        examples.append(example)
    
    return examples


def replicate_and_shuffle_dataset(
    examples: List[Dict],
    num_copies: int,
    base_seed: int
) -> List[Dict]:
    """
    Create multiple shuffled copies of the dataset and concatenate them.
    This follows the pattern used in other synthetic dataset generators.
    
    Args:
        examples: Original dataset examples
        num_copies: Number of copies to create
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


def save_dataset(
    examples: List[Dict],
    output_dir: Path,
    output_format: str,
    split_name: str
):
    """
    Save dataset in the specified format.
    
    Args:
        examples: List of dataset examples
        output_dir: Output directory
        output_format: Format (jsonl or parquet)
        split_name: Name of split (train/validation/test)
    """
    if output_format == "jsonl":
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(examples)} examples to {filepath}")
    else:  # parquet
        from datasets import Dataset
        
        # Create directory for this split
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(examples)
        
        # Save as parquet
        parquet_file = split_dir / "data.parquet"
        dataset.to_parquet(str(parquet_file))
        print(f"Saved {len(examples)} examples to {parquet_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic MQAR (Multi-Query Associative Recall) dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_mqar_data",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30000,
        help="Number of training steps (will calculate sequences as steps × batch_size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Global batch size (batch_size × num_gpus × gradient_accumulation)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length (should be divisible by 4*num_kv_pairs)",
    )
    parser.add_argument(
        "--num-kv-pairs",
        type=int,
        default=8,
        help="Number of key-value pairs per MQAR example",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Total vocabulary size (will be split evenly between keys and values)",
    )
    parser.add_argument(
        "--power-a",
        type=float,
        default=0.01,
        help="Power law parameter for query ordering (smaller = queries favor recent keys)",
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
        default=1,
        help="Number of shuffled copies to concatenate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate number of sequences needed based on training steps
    num_sequences = args.num_samples * args.batch_size
    val_sequences = num_sequences // 10  # 10% for validation
    test_sequences = num_sequences // 10  # 10% for test
    
    # Calculate example length
    example_len = 4 * args.num_kv_pairs
    examples_per_seq = args.seq_len // example_len
    
    print("Generating MQAR dataset with the following parameters:")
    print(f"  Output directory: {output_dir}")
    print(f"  Training steps: {args.num_samples:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequences to generate: {num_sequences:,} (train), {val_sequences:,} (val), {test_sequences:,} (test)")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  KV pairs per example: {args.num_kv_pairs}")
    print(f"  Example length: {example_len} tokens")
    print(f"  Examples per sequence: {examples_per_seq}")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Power law parameter: {args.power_a}")
    print(f"  Number of copies: {args.num_copies}")
    print(f"  Random seed: {args.seed}")
    print(f"\nNote: With batch_size={args.batch_size}, {args.num_samples:,} training steps requires {num_sequences:,} sequences")
    
    # Generate base datasets
    print("\nGenerating training examples...")
    train_examples = create_dataset(
        num_sequences,
        args.seq_len,
        args.num_kv_pairs,
        args.vocab_size,
        args.power_a,
        args.seed,
        tokenizer
    )
    
    print("Generating validation examples...")
    val_examples = create_dataset(
        val_sequences,
        args.seq_len,
        args.num_kv_pairs,
        args.vocab_size,
        args.power_a,
        args.seed + 1000000,  # Different seed
        tokenizer
    )
    
    print("Generating test examples...")
    test_examples = create_dataset(
        test_sequences,
        args.seq_len,
        args.num_kv_pairs,
        args.vocab_size,
        args.power_a,
        args.seed + 2000000,  # Different seed
        tokenizer
    )
    
    # Create multiple shuffled copies
    print(f"\nCreating {args.num_copies} shuffled copies for each split...")
    train_examples = replicate_and_shuffle_dataset(
        train_examples, args.num_copies, args.seed
    )
    val_examples = replicate_and_shuffle_dataset(
        val_examples, args.num_copies, args.seed + 1000
    )
    test_examples = replicate_and_shuffle_dataset(
        test_examples, args.num_copies, args.seed + 2000
    )
    
    # Save datasets
    print("\nSaving datasets...")
    save_dataset(train_examples, output_dir, args.output_format, "train")
    save_dataset(val_examples, output_dir, args.output_format, "validation")
    save_dataset(test_examples, output_dir, args.output_format, "test")
    
    # Save metadata
    metadata = {
        "dataset_type": "mqar",
        "training_steps": args.num_samples,
        "batch_size": args.batch_size,
        "sequences_per_split": {
            "train": num_sequences,
            "validation": val_sequences,
            "test": test_sequences,
        },
        "total_examples_per_split": {
            "train": len(train_examples),
            "validation": len(val_examples),
            "test": len(test_examples),
        },
        "seq_len": args.seq_len,
        "num_kv_pairs": args.num_kv_pairs,
        "example_length": example_len,
        "examples_per_sequence": examples_per_seq,
        "vocab_size": args.vocab_size,
        "key_vocab_size": args.vocab_size // 2,
        "value_vocab_range": [args.vocab_size // 2, args.vocab_size],
        "power_a": args.power_a,
        "num_copies": args.num_copies,
        "seed": args.seed,
        "format": args.output_format,
        "tokens_per_step": args.batch_size * args.seq_len,
        "total_training_tokens": args.num_samples * args.batch_size * args.seq_len,
        "description": "MQAR dataset where each example contains KV pairs followed by queries ordered by power law",
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDataset generation complete!")
    print(f"Total examples: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")


if __name__ == "__main__":
    main()