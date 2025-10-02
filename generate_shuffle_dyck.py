#!/usr/bin/env python3
"""
Generate k-shuffle Dyck dataset with optional tokenization.
Mirrors the pre-pretraining module commands but in a single clean script.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import trange
from transformers import AutoTokenizer


def generate_shuff_dyck(k: int, max_length: int = 2048, p_open: float = 0.5, max_depth: int = 16) -> List[int]:
    """
    Generate a k-shuffle Dyck sequence, truncated at max_length.
    When max depth is reached, close one bracket and continue.

    Args:
        k: Number of different types of brackets
        max_length: Target maximum length of the sequence
        p_open: Probability of opening a new bracket
        max_depth: Maximum nesting depth allowed

    Returns:
        Generated sequence where i represents opening bracket i
        and i+k represents closing bracket i
    """
    sequence = []
    counts = [0] * k  # Track open brackets of each type

    while len(sequence) < max_length:
        depth = sum(counts)

        # Must open if all brackets are closed
        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        # If at max depth, force a close
        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        # Randomly choose to open or close
        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            # Close an existing bracket
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


def generate_and_save_text_file(
    output_dir: Path,
    num_symbols: int,
    n: int,
    seq_len: int,
    p_open: float,
    max_depth: int,
    seed: int
) -> Path:
    """Generate text file with k-shuffle Dyck sequences."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"dyck_sequences_cross_serial_{num_symbols}_{p_open}.txt"
    filepath = output_dir / filename
    
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Generating {n} k-shuffle Dyck sequences...")
    print(f"Parameters: k={num_symbols}, seq_len={seq_len}, p_open={p_open}, max_depth={max_depth}")
    
    with open(filepath, "w") as f:
        for i in trange(n, desc="Generating sequences"):
            result = generate_shuff_dyck(num_symbols, seq_len, p_open, max_depth)
            dyck_str = " ".join([str(x) for x in result[:seq_len]])
            f.write(f"{dyck_str}\n")
    
    return filepath


def tokenize_and_save(
    text_file: Path,
    output_dir: Path,
    tokenizer_name: str,
    seq_len: int
):
    """Tokenize text file and save as HuggingFace dataset."""
    print(f"\nTokenizing with {tokenizer_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    
    # Load as text dataset
    dataset = load_dataset("text", data_files=str(text_file), split="train")
    
    # Tokenize with proper max_length
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=seq_len,
        ),
        batched=True,
        desc="Tokenizing"
    ).remove_columns(["text"])
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"Saved tokenized dataset to {output_dir}")
    print(f"Dataset size: {len(dataset)} examples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate k-shuffle Dyck dataset with optional tokenization"
    )
    
    # Generation parameters
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./synthetic_shuffle_dyck",
        help="Base output directory (default: ./synthetic_shuffle_dyck)"
    )
    parser.add_argument(
        "-k", "--num-symbols", 
        type=int, 
        default=64,
        help="Number of distinct bracket types (default: 64)"
    )
    parser.add_argument(
        "--seq-len", 
        type=int, 
        default=2048,
        help="Sequence length (default: 2048)"
    )
    parser.add_argument(
        "-n", 
        type=int, 
        default=100000,
        help="Number of sequences to generate (default: 100000)"
    )
    parser.add_argument(
        "-p", "--p-open", 
        type=float, 
        default=0.5,
        help="Probability of opening a new bracket (default: 0.5)"
    )
    parser.add_argument(
        "--max-depth", 
        type=int, 
        default=16,
        help="Maximum nesting depth (default: 16)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Tokenization parameters
    parser.add_argument(
        "--tokenize", 
        action="store_true",
        help="Also create tokenized dataset"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="EleutherAI/pythia-160m",
        help="Tokenizer to use (default: EleutherAI/pythia-160m)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Generate text file
    text_file = generate_and_save_text_file(
        output_dir=Path(args.output_dir),
        num_symbols=args.num_symbols,
        n=args.n,
        seq_len=args.seq_len,
        p_open=args.p_open,
        max_depth=args.max_depth,
        seed=args.seed
    )
    print(f"\nSaved text file to: {text_file}")
    
    # Step 2: Tokenize (optional)
    if args.tokenize:
        tokenized_dir = Path(args.output_dir) / "tokenized"
        tokenize_and_save(
            text_file=text_file,
            output_dir=tokenized_dir,
            tokenizer_name=args.tokenizer,
            seq_len=args.seq_len
        )
    
    print("\nGeneration complete!")
    if args.tokenize:
        print(f"Text file: {text_file}")
        print(f"Tokenized data: {Path(args.output_dir) / 'tokenized'}")
    else:
        print("To tokenize, run again with --tokenize flag")


if __name__ == "__main__":
    main()