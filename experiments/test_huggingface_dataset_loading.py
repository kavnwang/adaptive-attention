#!/usr/bin/env python3
"""
Test script to verify HuggingFace dataset loading integration.

This script tests loading various HuggingFace datasets (Piqa, GSM8K) 
and verifies they are properly preprocessed for training.
"""

import sys
sys.path.insert(0, '.')

from llmonade.data import build_dataset, preprocess_huggingface_dataset
from datasets import load_dataset
import torch
from transformers import AutoTokenizer


def test_piqa_dataset():
    """Test loading and preprocessing Piqa dataset."""
    print("\n=== Testing Piqa Dataset ===")
    
    # Load a small sample for testing
    dataset = load_dataset("piqa", split="train[:10]")
    print(f"Original Piqa columns: {dataset.column_names}")
    print(f"Sample original data: {dataset[0]}")
    
    # Preprocess
    processed_dataset = preprocess_huggingface_dataset(dataset, "piqa")
    print(f"Processed columns: {processed_dataset.column_names}")
    print(f"Sample processed text: {processed_dataset[0]['text']}")
    
    # Verify the format
    for i in range(min(3, len(processed_dataset))):
        example = dataset[i]
        processed = processed_dataset[i]
        expected_solution = example["sol1"] if example["label"] == 0 else example["sol2"]
        expected_text = f"{example['goal']} {expected_solution}"
        assert processed["text"] == expected_text, f"Mismatch at index {i}"
        print(f"✓ Example {i} processed correctly")
    
    print("✅ Piqa dataset test passed!")
    return True


def test_gsm8k_dataset():
    """Test loading and preprocessing GSM8K dataset."""
    print("\n=== Testing GSM8K Dataset ===")
    
    # Load a small sample for testing  
    dataset = load_dataset("openai/gsm8k", "main", split="train[:10]")
    print(f"Original GSM8K columns: {dataset.column_names}")
    print(f"Sample original data: {dataset[0]}")
    
    # Preprocess
    processed_dataset = preprocess_huggingface_dataset(dataset, "openai/gsm8k")
    print(f"Processed columns: {processed_dataset.column_names}")
    print(f"Sample processed text (truncated): {processed_dataset[0]['text'][:200]}...")
    
    # Verify the format
    for i in range(min(3, len(processed_dataset))):
        example = dataset[i]
        processed = processed_dataset[i]
        expected_text = f"{example['question']} {example['answer']}"
        assert processed["text"] == expected_text, f"Mismatch at index {i}"
        print(f"✓ Example {i} processed correctly")
    
    print("✅ GSM8K dataset test passed!")
    return True


def test_build_dataset_integration():
    """Test the full build_dataset integration with HuggingFace datasets."""
    print("\n=== Testing build_dataset Integration ===")
    
    # Test single dataset
    print("Testing single Piqa dataset...")
    dataset = build_dataset(
        dataset="piqa",
        dataset_split="train",
        streaming=True,
        num_workers=1,
        seed=42
    )
    
    # Get a few examples
    examples = []
    for i, example in enumerate(dataset):
        examples.append(example)
        if i >= 4:
            break
    
    print(f"✓ Successfully loaded {len(examples)} examples")
    print(f"Sample text: {examples[0]['text'][:100]}...")
    
    # Test multiple datasets with interleaving
    print("\nTesting multiple datasets with interleaving...")
    multi_dataset = build_dataset(
        dataset="piqa,openai/gsm8k",
        dataset_name=",main",  # No name for piqa, "main" for gsm8k
        dataset_split="train,train",
        data_probs="0.5,0.5",
        streaming=True,
        num_workers=1,
        seed=42
    )
    
    # Get a few examples
    multi_examples = []
    for i, example in enumerate(multi_dataset):
        multi_examples.append(example)
        if i >= 9:
            break
    
    print(f"✓ Successfully loaded {len(multi_examples)} interleaved examples")
    
    print("✅ build_dataset integration test passed!")
    return True


def test_with_tokenizer():
    """Test the full pipeline with tokenization."""
    print("\n=== Testing with Tokenizer ===")
    
    # Load a small tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build dataset
    dataset = build_dataset(
        dataset="piqa", 
        dataset_split="validation",
        streaming=True,
        num_workers=1,
        seed=42
    )
    
    # Test tokenization on a few examples
    for i, example in enumerate(dataset):
        text = example["text"]
        tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        print(f"Example {i}: {len(tokens['input_ids'][0])} tokens")
        print(f"  Text preview: {text[:100]}...")
        print(f"  Decoded tokens preview: {tokenizer.decode(tokens['input_ids'][0][:20])}...")
        
        if i >= 2:
            break
    
    print("✅ Tokenizer integration test passed!")
    return True


def main():
    """Run all tests."""
    print("Testing HuggingFace dataset loading integration...")
    
    tests = [
        test_piqa_dataset,
        test_gsm8k_dataset,
        test_build_dataset_integration,
        test_with_tokenizer,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n🎉 All tests passed!")
    return 0


if __name__ == "__main__":
    exit(main())