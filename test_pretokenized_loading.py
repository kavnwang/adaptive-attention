#!/usr/bin/env python3
"""Test loading pre-tokenized datasets with modified LLMonade data loader."""

import torch
from transformers import AutoTokenizer
from llmonade.data import build_dataset
import logging

logging.basicConfig(level=logging.INFO)

def test_pretokenized_loading():
    """Test loading pre-tokenized Dyck dataset."""
    
    print("Testing pre-tokenized dataset loading...")
    
    # Load tokenizer (needed for dataset class even if not used)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    
    # Test loading pre-tokenized Dyck dataset
    dataset = build_dataset(
        dataset="pre-pretraining/data/tokenized/shuff_dyck",
        dataset_split="train",
        streaming=True,
        dp_degree=1,
        num_workers=1,
        seed=42
    )
    
    # Create data loader
    from llmonade.data import OnlineTokenizedIterableDataset
    tokenized_dataset = OnlineTokenizedIterableDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        seq_len=2048,
        rank=0,
        world_size=1
    )
    
    # Check if pre-tokenized detection worked
    print(f"Is pre-tokenized: {tokenized_dataset.is_pretokenized}")
    
    # Get first few examples
    data_iter = iter(tokenized_dataset)
    for i in range(3):
        batch = next(data_iter)
        print(f"\nExample {i+1}:")
        print(f"  Shape: {batch['input_ids'].shape}")
        print(f"  First 20 tokens: {batch['input_ids'][:20].tolist()}")
        print(f"  Min/Max values: {batch['input_ids'].min().item()}, {batch['input_ids'].max().item()}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_pretokenized_loading()