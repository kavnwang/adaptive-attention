#!/usr/bin/env python3
"""
Extract and compute average norm of value vectors per token position across key-value examples.
Based on extract_memory_key_activations.py but focuses on value vector norms.
"""

import json
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
import argparse
import numpy as np

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM, MemoryLayer


def patch_memory_layers_for_value_extraction(model, value_norms_dict):
    """Patch all memory layers to extract value vector norms during forward pass."""
    original_methods = []
    found_memory_layers = []

    for i, layer in enumerate(model.model.layers):
        # Check if this layer's mlp is a MemoryLayer
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            found_memory_layers.append(i)
            original_method = patch_memory_layer_for_value_tracking(
                layer.mlp, i, value_norms_dict
            )
            original_methods.append((layer.mlp, original_method))

    return original_methods, found_memory_layers


def patch_memory_layer_for_value_tracking(memory_layer, layer_idx, value_norms_dict):
    """Patch a MemoryLayer's forward method to track value vector norms."""
    original_forward = memory_layer.forward

    def tracked_forward(x):
        # Call original forward
        result = original_forward(x)

        # During forward, the memory layer retrieves values using indices
        # We need to hook into the value retrieval process
        # The values are retrieved in memory_layer.values(indices, scores)

        # For now, let's patch the embedded bag call
        if hasattr(memory_layer, "_last_retrieved_values"):
            # Store value norms for each position
            values = memory_layer._last_retrieved_values  # (bs, v_dim)
            value_norms = torch.norm(values, dim=-1)  # (bs,)

            if layer_idx not in value_norms_dict:
                value_norms_dict[layer_idx] = []

            # Store norms for all positions in the batch
            value_norms_dict[layer_idx].extend(value_norms.cpu().numpy().tolist())

        return result

    # Also patch the values embedding bag to capture retrieved values
    original_values_forward = memory_layer.values.forward

    def tracked_values_forward(indices, offsets=None, per_sample_weights=None):
        # Call original
        result = original_values_forward(indices, offsets, per_sample_weights)

        # Store the result for access in the main forward
        memory_layer._last_retrieved_values = result

        return result

    memory_layer.values.forward = tracked_values_forward
    memory_layer.forward = tracked_forward

    return (original_forward, original_values_forward)


def restore_original_methods(original_methods):
    """Restore original forward methods for memory layers."""
    for memory_layer, (orig_forward, orig_values_forward) in original_methods:
        memory_layer.forward = orig_forward
        memory_layer.values.forward = orig_values_forward
        if hasattr(memory_layer, "_last_retrieved_values"):
            delattr(memory_layer, "_last_retrieved_values")


def get_value_token_positions(tokenizer, key, value):
    """
    Find the token positions corresponding to the value in the full key:value text.
    Returns a list of position indices where value tokens appear.
    """
    # Tokenize the full key:value pair
    full_text = f"{key}{value} "
    full_encoding = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    full_ids = full_encoding["input_ids"][0]

    # Tokenize just the value (without special tokens)
    value_encoding = tokenizer(value, return_tensors="pt", add_special_tokens=False)
    value_ids = value_encoding["input_ids"][0]

    # Find where value tokens start in the full sequence
    value_len = len(value_ids)
    value_positions = []

    for i in range(len(full_ids) - value_len + 1):
        if torch.equal(full_ids[i : i + value_len], value_ids):
            value_positions = list(range(i, i + value_len))
            break

    return value_positions


def extract_value_vector_norms(model, tokenizer, dataset, batch_size=32):
    """
    Run model over dataset and extract value vector norms for each position.
    Returns average norms per position across all examples.
    """
    model.eval()
    device = next(model.parameters()).device

    # Dictionary to accumulate norms per position per layer
    # layer_idx -> position -> list of norms
    position_value_norms = defaultdict(lambda: defaultdict(list))

    # Track max sequence length seen
    max_seq_len = 0

    # Process dataset in batches
    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(dataset), batch_size), desc="Processing batches"
        ):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_texts = []
            batch_qa_info = []

            # Prepare batch
            for i in range(batch_start, batch_end):
                sample = dataset[i]
                text = sample["text"]

                # Extract key:value
                parts = text.split(":")
                if len(parts) == 2:
                    key = parts[0].strip() + ":"
                    value = parts[1].strip()

                    # Find value token positions
                    value_positions = get_value_token_positions(tokenizer, key, value)

                    if value_positions:
                        batch_texts.append(text)
                        batch_qa_info.append(
                            {
                                "key": key,
                                "value": value,
                                "value_positions": value_positions,
                            }
                        )

            if not batch_texts:
                continue

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Dictionary to store value norms
            value_norms_dict = {}

            # Patch memory layers
            original_methods, found_memory_layers = (
                patch_memory_layers_for_value_extraction(model, value_norms_dict)
            )

            # Forward pass
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = input_ids.shape[1]
            max_seq_len = max(max_seq_len, seq_len)

            # Process token by token to get position-specific norms
            for pos in range(seq_len):
                # Clear previous value norms
                value_norms_dict.clear()

                # Forward pass for single position
                position_input = input_ids[:, : pos + 1]
                position_mask = attention_mask[:, : pos + 1]

                if position_input.shape[1] > 0:
                    outputs = model(position_input, attention_mask=position_mask)

                    # Collect norms for this position across all samples in batch
                    for layer_idx, norms in value_norms_dict.items():
                        # norms contains values for all samples in the batch
                        for sample_idx, norm in enumerate(norms):
                            if sample_idx < len(batch_qa_info):
                                position_value_norms[layer_idx][pos].append(norm)

            # Restore original methods
            restore_original_methods(original_methods)

    # Compute average norms per position
    avg_norms_per_position = {}
    for layer_idx, position_norms in position_value_norms.items():
        avg_norms_per_position[f"layer_{layer_idx}"] = {}
        for pos in range(max_seq_len):
            if pos in position_norms and len(position_norms[pos]) > 0:
                avg_norms_per_position[f"layer_{layer_idx}"][str(pos)] = {
                    "mean": float(np.mean(position_norms[pos])),
                    "std": float(np.std(position_norms[pos])),
                    "count": len(position_norms[pos]),
                }

    return avg_norms_per_position


def main():
    parser = argparse.ArgumentParser(
        description="Extract average value vector norms per token position"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="exp/memory_2layer_qa_1M",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        choices=["200", "2K", "10K", "50K"],
        required=True,
        help="Dataset size to use (200, 2K, 10K, or 50K)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer (if different from model)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="value_vector_norms_per_position.json",
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all)",
    )

    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model = MemoryForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset (size: {args.dataset_size})...")
    dataset_path = (
        f"experiments/synthetic_qa/datasets/synthetic_qa_data_{args.dataset_size}"
    )
    dataset = load_dataset(dataset_path, split="test")
    test_dataset = dataset

    # Limit number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(args.num_samples))
        print(f"Limited to {args.num_samples} samples")

    print(f"Extracting value vector norms from {len(test_dataset)} samples...")
    # Extract norms
    avg_norms = extract_value_vector_norms(
        model, tokenizer, test_dataset, batch_size=args.batch_size
    )

    # Save results
    output_path = args.output_path
    with open(output_path, "w") as f:
        json.dump(avg_norms, f, indent=2)

    print(f"\nSaved average value vector norms to {output_path}")

    # Print summary statistics
    for layer_name, position_data in avg_norms.items():
        print(f"\n{layer_name}:")
        positions = sorted([int(p) for p in position_data.keys()])
        if positions:
            print(f"  Positions analyzed: {min(positions)} to {max(positions)}")
            all_means = [position_data[str(p)]["mean"] for p in positions]
            print(f"  Overall mean norm: {np.mean(all_means):.4f}")
            print(f"  Overall std of means: {np.std(all_means):.4f}")


if __name__ == "__main__":
    main()
