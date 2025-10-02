#!/usr/bin/env python3
"""
Generate synthetic MQAR (Multi-Query Associative Recall) dataset with curriculum learning.

This script creates a dataset with a curriculum progression:
- Stages 1-4: 1, 2, 4, 8 KV pairs (no noise)
- Stages 5-10: 4, 8, 16, 32, 32, 32 KV pairs (with 0-2 noise tokens between KV pairs)

Each stage contains num_samples sequences, for a total of 10 * num_samples sequences.
Each sequence is exactly seq_len tokens.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoTokenizer


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


def generate_single_mqar_example_tokens(
    num_kv_pairs: int,
    token_pool: List[int],
    pool_index: int,
    power_a: float,
    use_noise: bool,
    rng: np.random.RandomState
) -> Tuple[List[int], int]:
    """
    Generate a single MQAR example as token IDs with optional noise between KV pairs.

    Args:
        num_kv_pairs: Number of key-value pairs
        token_pool: Pre-shuffled pool of tokens to use
        pool_index: Current index in the token pool
        power_a: Power law parameter for query ordering
        use_noise: Whether to add 0-2 noise tokens between KV pairs
        rng: Random number generator

    Returns:
        Tuple of (token list for the example, updated pool_index)
    """
    tokens = []
    keys = []
    values = []
    
    # Build KV pairs section
    for i in range(num_kv_pairs):
        # Get key-value pair from pool
        key = token_pool[pool_index % len(token_pool)]
        pool_index += 1
        value = token_pool[pool_index % len(token_pool)]
        pool_index += 1
        
        keys.append(key)
        values.append(value)
        tokens.extend([key, value])
        
        # Add noise only if enabled and not after last KV pair
        if use_noise and i < num_kv_pairs - 1:
            num_noise = rng.randint(0, 3)  # 0, 1, or 2 tokens
            for _ in range(num_noise):
                # Get noise token from pool
                noise_token = token_pool[pool_index % len(token_pool)]
                pool_index += 1
                tokens.append(noise_token)
    
    # Determine query order using power law
    query_order = get_query_order_power_law(num_kv_pairs, power_a, rng)
    
    # Build queries section (no noise)
    for idx in query_order:
        tokens.extend([keys[idx], values[idx]])
    
    return tokens, pool_index


def generate_single_mqar_example(
    num_kv_pairs: int,
    vocab_size: int,
    power_a: float,
    use_noise: bool,
    rng: np.random.RandomState,
    tokenizer
) -> str:
    """
    Generate a single MQAR example with optional noise between KV pairs.

    Args:
        num_kv_pairs: Number of key-value pairs
        key_vocab_size: Size of key vocabulary (first half of vocab)
        vocab_size: Total vocabulary size
        power_a: Power law parameter for query ordering
        use_noise: Whether to add 0-2 noise tokens between KV pairs
        rng: Random number generator
        tokenizer: Tokenizer for decoding

    Returns:
        Text string for the example
    """
    # Sample unique keys and values from full vocabulary
    # Ensure keys and values don't overlap by sampling 2*num_kv_pairs unique tokens
    all_tokens = rng.choice(range(1, vocab_size), size=2*num_kv_pairs, replace=False)
    keys = all_tokens[:num_kv_pairs]
    values = all_tokens[num_kv_pairs:]
    
    # Build KV pairs section
    text_tokens = []
    for i in range(num_kv_pairs):
        # Add key-value pair
        k_str = tokenizer.decode([int(keys[i])])
        v_str = tokenizer.decode([int(values[i])])
        text_tokens.extend([k_str, v_str])
        
        # Add noise only if enabled and not after last KV pair
        if use_noise and i < num_kv_pairs - 1:
            num_noise = rng.randint(0, 3)  # 0, 1, or 2 tokens
            for _ in range(num_noise):
                # Sample noise from full vocabulary
                noise_token = rng.randint(1, vocab_size)
                noise_str = tokenizer.decode([noise_token])
                text_tokens.append(noise_str)
    
    # Determine query order using power law
    query_order = get_query_order_power_law(num_kv_pairs, power_a, rng)
    
    # Build queries section (no noise)
    for idx in query_order:
        k_str = tokenizer.decode([int(keys[idx])])
        v_str = tokenizer.decode([int(values[idx])])
        text_tokens.extend([k_str, v_str])
    
    return " ".join(text_tokens)


def generate_mqar_sequence(
    seq_len: int,
    num_kv_pairs: int,
    vocab_size: int,
    power_a: float,
    use_noise: bool,
    seed: int,
    tokenizer
) -> str:
    """
    Generate a full MQAR sequence with EXACT seq_len tokens.
    
    Args:
        seq_len: Target sequence length (exact)
        num_kv_pairs: Number of KV pairs per example
        vocab_size: Total vocabulary size
        power_a: Power law parameter
        use_noise: Whether to add noise between KV pairs
        seed: Random seed
        tokenizer: Tokenizer
        
    Returns:
        Text string with exactly seq_len tokens
    """
    rng = np.random.RandomState(seed)
    
    # Shuffle entire vocabulary for this sequence
    all_vocab_tokens = list(range(vocab_size))
    rng.shuffle(all_vocab_tokens)
    token_pool = all_vocab_tokens[:8192]  # Take first 8192 tokens as pool
    pool_index = 0
    
    # Keep generating and concatenating examples until we have enough tokens
    all_tokens = []
    
    while len(all_tokens) < seq_len:
        # Generate one example as token IDs
        example_tokens, pool_index = generate_single_mqar_example_tokens(
            num_kv_pairs=num_kv_pairs,
            token_pool=token_pool,
            pool_index=pool_index,
            power_a=power_a,
            use_noise=use_noise,
            rng=rng
        )
        all_tokens.extend(example_tokens)
    
    # Truncate to EXACT seq_len
    all_tokens = all_tokens[:seq_len]
    
    # Decode to text
    final_text = tokenizer.decode(all_tokens)
    
    return final_text


def create_curriculum_dataset(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    power_a: float,
    seed: int,
    tokenizer
) -> List[Dict]:
    """
    Create MQAR curriculum dataset with 10 stages.
    
    Args:
        num_samples: Number of sequences per stage
        seq_len: Sequence length (exact)
        vocab_size: Vocabulary size
        power_a: Power law parameter
        seed: Random seed
        tokenizer: Tokenizer
        
    Returns:
        List of dataset examples
    """
    examples = []
    
    # Define curriculum: (num_kv_pairs, use_noise)
    curriculum = [
        (1, False),   # Stage 1: 1 KV pair, no noise
        (2, False),   # Stage 2: 2 KV pairs, no noise
        (4, False),   # Stage 3: 4 KV pairs, no noise
        (8, False),   # Stage 4: 8 KV pairs, no noise
        (4, True),    # Stage 5: 4 KV pairs with noise
        (8, True),    # Stage 6: 8 KV pairs with noise
        (16, True),   # Stage 7: 16 KV pairs with noise
        (32, True),   # Stage 8: 32 KV pairs with noise
        (32, True),   # Stage 9: 32 KV pairs with noise (repeat)
        (32, True),   # Stage 10: 32 KV pairs with noise (repeat)
    ]
    
    example_idx = 0
    for stage_num, (num_kv_pairs, use_noise) in enumerate(curriculum):
        noise_str = "with noise" if use_noise else "no noise"
        print(f"Stage {stage_num+1}/10: {num_kv_pairs} KV pairs ({noise_str})...")
        
        for i in range(num_samples):
            text = generate_mqar_sequence(
                seq_len=seq_len,
                num_kv_pairs=num_kv_pairs,
                vocab_size=vocab_size,
                power_a=power_a,
                use_noise=use_noise,
                seed=seed + example_idx,
                tokenizer=tokenizer
            )
            examples.append({"text": text})
            example_idx += 1
    
    return examples


def replicate_and_shuffle_dataset(
    examples: List[Dict],
    num_copies: int,
    base_seed: int
) -> List[Dict]:
    """
    Create multiple shuffled copies of the dataset and concatenate them.
    
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
        description="Generate synthetic MQAR curriculum dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_mqar_curriculum_data",
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
        help="Sequence length (will be exact)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Total vocabulary size (keys and values sampled from full range)",
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
    
    print("Generating MQAR curriculum dataset with the following parameters:")
    print(f"  Output directory: {output_dir}")
    print(f"  Training steps: {args.num_samples:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequences per stage: {num_sequences:,}")
    print(f"  Total sequences: {10 * num_sequences:,} (10 stages)")
    print(f"  Sequence length: {args.seq_len} (exact)")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Power law parameter: {args.power_a}")
    print(f"  Number of copies: {args.num_copies}")
    print(f"  Random seed: {args.seed}")
    
    # Generate base datasets
    print("\nGenerating training examples...")
    train_examples = create_curriculum_dataset(
        num_sequences,
        args.seq_len,
        args.vocab_size,
        args.power_a,
        args.seed,
        tokenizer
    )
    
    print("Generating validation examples...")
    val_examples = create_curriculum_dataset(
        val_sequences,
        args.seq_len,
        args.vocab_size,
        args.power_a,
        args.seed + 1000000,  # Different seed
        tokenizer
    )
    
    print("Generating test examples...")
    test_examples = create_curriculum_dataset(
        test_sequences,
        args.seq_len,
        args.vocab_size,
        args.power_a,
        args.seed + 2000000,  # Different seed
        tokenizer
    )
    
    # Create multiple shuffled copies if requested
    if args.num_copies > 1:
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
        "dataset_type": "mqar_curriculum",
        "training_steps": args.num_samples,
        "batch_size": args.batch_size,
        "sequences_per_stage": num_sequences,
        "total_sequences": {
            "train": len(train_examples),
            "validation": len(val_examples),
            "test": len(test_examples),
        },
        "seq_len": args.seq_len,
        "curriculum": [
            {"stage": 1, "kv_pairs": 1, "noise": False},
            {"stage": 2, "kv_pairs": 2, "noise": False},
            {"stage": 3, "kv_pairs": 4, "noise": False},
            {"stage": 4, "kv_pairs": 8, "noise": False},
            {"stage": 5, "kv_pairs": 4, "noise": True},
            {"stage": 6, "kv_pairs": 8, "noise": True},
            {"stage": 7, "kv_pairs": 16, "noise": True},
            {"stage": 8, "kv_pairs": 32, "noise": True},
            {"stage": 9, "kv_pairs": 32, "noise": True},
            {"stage": 10, "kv_pairs": 32, "noise": True},
        ],
        "vocab_size": args.vocab_size,
        "key_value_sampling": "Keys and values sampled from full vocabulary without overlap",
        "power_a": args.power_a,
        "num_copies": args.num_copies,
        "seed": args.seed,
        "format": args.output_format,
        "tokens_per_step": args.batch_size * args.seq_len,
        "total_training_tokens": 10 * args.num_samples * args.batch_size * args.seq_len,
        "description": "MQAR curriculum dataset with progressive KV pair counts and noise injection",
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDataset generation complete!")
    print(f"Total examples: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")


if __name__ == "__main__":
    main()