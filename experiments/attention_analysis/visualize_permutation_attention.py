#!/usr/bin/env python3
"""
Visualize attention patterns for transformer models on digit lookup tasks.
Creates side-by-side comparisons of attention on known vs novel test samples.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Add bento to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "3rdparty", "bento")
)


def load_model_and_tokenizer(model_path, device="cuda"):
    """
    Load HF transformer model and tokenizer.

    TODO: Implementation details
    - Use AutoModelForCausalLM.from_pretrained() with torch_dtype=torch.bfloat16
    - Load tokenizer from "fla-hub/transformer-1.3B-100B"
    - Set pad_token if needed
    - Set model to eval mode
    - Return model, tokenizer tuple
    """
    print(f"Loading model from {model_path}")

    # Load model - since we imported bento, AutoModel should recognize TransformerForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Load tokenizer (use the standard tokenizer)
    tokenizer_path = "fla-hub/transformer-1.3B-100B"
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_test_data(data_path, test_type, num_samples=200, seed=42):
    """
    Load existing test data from parquet files.

    Args:
        data_path: Base path to dataset directory
        test_type: Either "test_known_pairs" or "test_novel_pairs"
        num_samples: Number of samples to load
        seed: Random seed for sampling

    Returns:
        List of text samples

    TODO: Implementation details
    - Load dataset using load_dataset("parquet", data_files=...)
    - Sample num_samples if dataset is larger
    - Extract "text" field from each sample
    - Return list of strings
    """
    # Load dataset from parquet file
    dataset_path = f"{data_path}/{test_type}"
    parquet_file = f"{dataset_path}/data.parquet"
    print(f"Loading {test_type} from {parquet_file}")

    dataset = load_dataset("parquet", data_files=parquet_file, split="train")

    # Sample subset if needed
    total_size = len(dataset)
    if num_samples > 0 and num_samples < total_size:
        np.random.seed(seed)
        indices = np.random.choice(total_size, size=num_samples, replace=False)
        dataset = dataset.select(indices)
        print(f"Sampled {num_samples} examples from {total_size} total")
    else:
        print(f"Using all {total_size} examples")

    # Extract text field
    samples = dataset["text"]
    return samples


def extract_attention_weights(model, tokenizer, text, layer_idx):
    """
    Extract attention weights for a specific layer by computing them manually.

    Args:
        model: The transformer model
        tokenizer: Tokenizer for processing text
        text: Input text to process
        layer_idx: Which layer to extract attention from

    Returns:
        Attention weights tensor of shape [num_heads, seq_len, seq_len]
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get hidden states by running through earlier layers
    hidden_states = model.model.embeddings(inputs["input_ids"])

    # Process through layers up to the target layer
    for i in range(layer_idx):
        layer_outputs = model.model.layers[i](hidden_states)
        hidden_states = layer_outputs[0]

    # Now extract Q, K, V from the target layer's attention module
    attn_module = model.model.layers[layer_idx].attn

    with torch.no_grad():
        # Project to Q, K, V
        q = attn_module.q_proj(hidden_states)
        k = attn_module.k_proj(hidden_states)
        v = attn_module.v_proj(hidden_states)

        # Reshape to separate heads
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = attn_module.head_dim
        num_heads = attn_module.num_heads
        num_kv_heads = (
            attn_module.num_kv_heads
            if hasattr(attn_module, "num_kv_heads")
            else num_heads
        )

        # Reshape Q with num_heads
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Reshape K and V with num_kv_heads (for Multi-Query Attention)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply QK normalization if used
        if attn_module.qk_norm:
            q = attn_module.q_norm(q)
            k = attn_module.k_norm(k)

        # Repeat K and V heads to match Q heads if using MQA
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.bool(), float("-inf"))

        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Move to CPU and remove batch dimension
        attention_weights = attention_weights.squeeze(0).cpu()

    return attention_weights


def compute_average_attention(model, tokenizer, samples, layer_idx):
    """
    Process multiple samples and compute average attention patterns.

    Args:
        model: The transformer model
        tokenizer: Tokenizer for processing text
        samples: List of text samples
        layer_idx: Which layer to extract attention from

    Returns:
        Averaged attention tensor [num_heads, seq_len, seq_len]

    TODO: Implementation details
    - Initialize accumulator for attention weights
    - Process each sample and extract attention
    - Handle variable sequence lengths (pad/truncate to consistent size)
    - Average across all samples
    - Return averaged attention
    """
    # First, determine the maximum sequence length we'll use
    max_len = 128  # Use a fixed max length for consistency

    # Initialize accumulator
    attention_sum = None
    valid_count = 0

    print(f"Processing {len(samples)} samples for layer {layer_idx}...")

    for i, text in enumerate(tqdm(samples, desc=f"Layer {layer_idx}")):
        try:
            # Extract attention for this sample
            attn = extract_attention_weights(model, tokenizer, text, layer_idx)

            # Get current shape
            num_heads, seq_len, _ = attn.shape

            # Pad or truncate to max_len
            if seq_len < max_len:
                # Pad with zeros
                pad_size = max_len - seq_len
                attn = torch.nn.functional.pad(attn, (0, pad_size, 0, pad_size))
            elif seq_len > max_len:
                # Truncate
                attn = attn[:, :max_len, :max_len]

            # Add to accumulator
            if attention_sum is None:
                attention_sum = attn
            else:
                attention_sum += attn

            valid_count += 1

        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")
            continue

    if valid_count == 0:
        raise ValueError("No valid samples processed")

    # Average
    avg_attention = attention_sum / valid_count
    print(f"Averaged {valid_count} valid samples")

    return avg_attention


def visualize_attention_comparison(
    known_attn,
    novel_attn,
    model_name,
    layer_idx,
    head_idx,
    output_path,
    tokenizer=None,
    sample_text=None,
):
    """
    Create side-by-side visualization of attention patterns using seaborn heatmaps.

    Args:
        known_attn: Attention weights for known pairs [seq_len, seq_len]
        novel_attn: Attention weights for novel pairs [seq_len, seq_len]
        model_name: Name of the model for title
        layer_idx: Layer index for title
        head_idx: Head index for title
        output_path: Where to save the figure
        tokenizer: Tokenizer for decoding tokens
        sample_text: Sample text for token labels
    """
    # Set seaborn style
    sns.set_style("white")

    # Convert to float32 for matplotlib (from bfloat16)
    known_attn = known_attn.float().numpy()
    novel_attn = novel_attn.float().numpy()

    # Get token labels if tokenizer and sample text provided
    token_labels = None
    if tokenizer and sample_text:
        # Tokenize to get the actual tokens
        token_ids = tokenizer(sample_text, return_tensors="pt")["input_ids"][0]
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Clean up tokens to show actual characters
        cleaned_tokens = []
        for tok in tokens:
            # Remove special tokens and whitespace indicators
            tok = tok.replace("▁", "").strip()
            if tok == "<s>":
                tok = "BOS"
            elif tok == "</s>":
                tok = "EOS"
            elif not tok:  # Empty after cleaning
                tok = "_"  # Show space as underscore
            cleaned_tokens.append(tok)
        token_labels = cleaned_tokens

    # Find actual sequence length (up to the last answer token)
    seq_len = known_attn.shape[0]
    # For digit lookup task, we typically have format: BOS + 6 key digits + : + 6 value digits + space
    # So we expect around 15-16 tokens total
    if seq_len > 16:
        seq_len = 16

    # Truncate to actual sequence length
    known_attn = known_attn[:seq_len, :seq_len]
    novel_attn = novel_attn[:seq_len, :seq_len]

    # Create figure with side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Determine color scale
    vmax = max(known_attn.max(), novel_attn.max())

    # Create mask for upper triangle (since attention is causal)
    mask = np.triu(np.ones_like(known_attn), k=1).astype(bool)

    # Plot known pairs attention with seaborn
    sns.heatmap(
        known_attn,
        ax=ax1,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar=False,
        xticklabels=token_labels[:seq_len] if token_labels else range(seq_len),
        yticklabels=token_labels[:seq_len] if token_labels else range(seq_len),
    )
    ax1.set_title("Known Pairs (Training Keys)", fontsize=12, pad=5)
    ax1.set_xlabel("Key Position", fontsize=10)
    ax1.set_ylabel("Query Position", fontsize=10)

    # Rotate x labels for better readability
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.tick_params(axis="y", rotation=0, labelsize=8)

    # Plot novel pairs attention with seaborn
    sns.heatmap(
        novel_attn,
        ax=ax2,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar=True,
        cbar_kws={"label": "Attention Weight", "shrink": 0.7},
        xticklabels=token_labels[:seq_len] if token_labels else range(seq_len),
        yticklabels=token_labels[:seq_len] if token_labels else range(seq_len),
    )
    ax2.set_title("Novel Pairs (Unseen Keys)", fontsize=12, pad=5)
    ax2.set_xlabel("Key Position", fontsize=10)
    ax2.set_ylabel("Query Position", fontsize=10)

    # Rotate x labels for better readability
    ax2.tick_params(axis="x", rotation=45, labelsize=8)
    ax2.tick_params(axis="y", rotation=0, labelsize=8)

    # Add main title
    fig.suptitle(
        f"{model_name} - Layer {layer_idx} Head {head_idx}", fontsize=14, y=0.98
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved visualization to {output_path}")


def main():
    """
    Main processing logic.

    TODO: Implementation details
    - Parse command line arguments
    - Load model and get architecture info (num_layers, num_heads)
    - Extract model name from path for output directory
    - Create output directory structure
    - Load both test datasets (known and novel)
    - For each layer:
        - Compute average attention for known samples
        - Compute average attention for novel samples
        - For each head:
            - Create visualization
            - Save as layer_{i}_head_{j}.png
    """
    parser = argparse.ArgumentParser(
        description="Visualize attention patterns for transformer models"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to HF model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="synthetic_digit_lookup_data_20000_special_tests",
        help="Path to test data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/attention_analysis/permutations",
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples", type=int, default=200, help="Number of samples per test set"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("ATTENTION VISUALIZATION FOR TRANSFORMER MODELS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Samples per test set: {args.num_samples}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Get model architecture info
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = (
        config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads
    )

    print("\nModel architecture:")
    print(f"- Layers: {num_layers}")
    print(f"- Heads per layer: {num_heads}")

    # Extract model name from path
    model_name = Path(args.model_path).name

    # Create output directory
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load test data
    print("\nLoading test data...")
    known_samples = load_test_data(
        args.data_path, "test_known_pairs", args.num_samples, args.seed
    )
    novel_samples = load_test_data(
        args.data_path, "test_novel_pairs", args.num_samples, args.seed
    )

    # Process each layer
    for layer_idx in range(num_layers):
        print(f"\n{'=' * 60}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'=' * 60}")

        # Compute average attention for known and novel samples
        print("\nComputing attention for known pairs...")
        known_attention = compute_average_attention(
            model, tokenizer, known_samples, layer_idx
        )

        print("\nComputing attention for novel pairs...")
        novel_attention = compute_average_attention(
            model, tokenizer, novel_samples, layer_idx
        )

        # Visualize each head
        print(f"\nCreating visualizations for {num_heads} heads...")
        for head_idx in range(num_heads):
            # Extract attention for this head
            known_head_attn = known_attention[head_idx]
            novel_head_attn = novel_attention[head_idx]

            # Create output filename
            if num_layers == 1:
                output_path = output_dir / f"head_{head_idx}.png"
            else:
                output_path = output_dir / f"layer_{layer_idx}_head_{head_idx}.png"

            # Create visualization
            # Use first sample as reference for token labels
            sample_text = known_samples[0] if known_samples else None
            visualize_attention_comparison(
                known_head_attn,
                novel_head_attn,
                model_name,
                layer_idx,
                head_idx,
                output_path,
                tokenizer,
                sample_text,
            )

    print(f"\n{'=' * 60}")
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
