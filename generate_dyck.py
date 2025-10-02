#!/usr/bin/env python3
"""
Generate standard Dyck dataset with optional tokenization.
Mirrors the pre-pretraining module commands but in a single clean script.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import trange
from transformers import AutoTokenizer


def generate_dyck(
    num_symbols: int, 
    min_depth: int = 1, 
    max_depth: int = 4, 
    max_length: int = 510, 
    offset: Optional[int] = None
) -> Optional[List[int]]:
    """
    Generates a Dyck sequence with specified number of symbols and depth constraints.

    Args:
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        min_depth: Minimum required depth of nested brackets.
        max_depth: Maximum allowed depth of nested brackets.
        max_length: The maximum length of the generated sequence.
        offset: Offset for closing brackets (defaults to num_symbols).

    Returns:
        A list representing the Dyck sequence, or None if generation fails.
    """
    result = []
    stack = []

    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")

    if offset is None:
        offset = num_symbols

    # Initialize with minimum depth
    for _ in range(min_depth):
        opening_symbol = np.random.randint(0, num_symbols)
        result.append(opening_symbol)
        stack.append(opening_symbol)

    while len(result) < max_length:
        if (
            len(stack) < max_depth and random.random() < 0.5
        ):  # Try to open if under max depth
            if len(result) >= max_length - 1:
                closing_symbol = stack.pop() + offset
                result.append(closing_symbol)
                continue
            opening_symbol = np.random.randint(0, num_symbols)
            result.append(opening_symbol)
            stack.append(opening_symbol)
        else:  # Close existing bracket
            closing_symbol = stack.pop() + offset
            result.append(closing_symbol)
            if not stack:
                break

    # pop remaining stuff on the stack if any
    while stack:
        closing_symbol = stack.pop() + offset
        result.append(closing_symbol)

    return result if not stack else None


def generate_and_save_text_file(
    output_dir: Path,
    num_symbols: int,
    n: int,
    seq_len: int,
    min_depth: int,
    max_depth: int,
    seed: int
) -> Path:
    """Generate text file with standard Dyck sequences."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"dyck_sequences_{num_symbols}_{min_depth}_{max_depth}.txt"
    filepath = output_dir / filename
    
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Generating {n} standard Dyck sequences...")
    print(f"Parameters: k={num_symbols}, seq_len={seq_len}, min_depth={min_depth}, max_depth={max_depth}")
    
    with open(filepath, "w") as f:
        for i in trange(n, desc="Generating sequences"):
            result = []
            while len(result) < seq_len:
                new_seq = generate_dyck(
                    num_symbols, min_depth=min_depth, max_depth=max_depth
                )
                if new_seq is None:
                    continue
                result.extend(new_seq)
            
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
        description="Generate standard Dyck dataset with optional tokenization"
    )
    
    # Generation parameters
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./synthetic_dyck",
        help="Base output directory (default: ./synthetic_dyck)"
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
        "--min-depth", 
        type=int, 
        default=1,
        help="Minimum nesting depth (default: 1)"
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
        min_depth=args.min_depth,
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