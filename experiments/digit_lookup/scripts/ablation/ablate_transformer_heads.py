#!/usr/bin/env python3
"""
Ablate attention heads one by one in transformer models and evaluate accuracy
by digit and position for known and novel test sets.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import sys
import os

# Add bento to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../3rdparty/bento")
)


def load_transformer_model(model_path, device="cuda"):
    """
    Load HF transformer model and tokenizer.
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


def load_test_data(dataset_type, num_samples=400, seed=42):
    """
    Load test data from parquet files.

    Args:
        dataset_type: Either "1hop" or "2steps"
        num_samples: Number of samples to load per test set
        seed: Random seed for sampling

    Returns:
        Tuple of (known_samples, novel_samples)
    """
    # Determine dataset path based on type
    if dataset_type == "1hop":
        data_path = "experiments/digit_lookup/datasets/synthetic_digit_lookup_data_20000_special_tests"
    elif dataset_type == "2steps":
        data_path = "experiments/digit_lookup/datasets/synthetic_digit_lookup_data_2steps_20000_special_tests"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Load known pairs
    known_path = f"{data_path}/test_known_pairs/data.parquet"
    print(f"Loading known pairs from {known_path}")
    known_dataset = load_dataset("parquet", data_files=known_path, split="train")

    # Load novel pairs
    novel_path = f"{data_path}/test_novel_pairs/data.parquet"
    print(f"Loading novel pairs from {novel_path}")
    novel_dataset = load_dataset("parquet", data_files=novel_path, split="train")

    # Sample subset if needed
    np.random.seed(seed)

    # Sample known pairs
    known_size = len(known_dataset)
    if num_samples > 0 and num_samples < known_size:
        known_indices = np.random.choice(known_size, size=num_samples, replace=False)
        known_samples = [known_dataset[int(i)]["text"] for i in known_indices]
    else:
        known_samples = known_dataset["text"][:num_samples]

    # Sample novel pairs
    novel_size = len(novel_dataset)
    if num_samples > 0 and num_samples < novel_size:
        novel_indices = np.random.choice(novel_size, size=num_samples, replace=False)
        novel_samples = [novel_dataset[int(i)]["text"] for i in novel_indices]
    else:
        novel_samples = novel_dataset["text"][:num_samples]

    print(
        f"Loaded {len(known_samples)} known samples and {len(novel_samples)} novel samples"
    )
    return known_samples, novel_samples


def hook_attention_output(module, args, output):
    """
    Hook function to zero out specific attention head outputs.
    """
    # For transformer models, we need to modify the attention computation
    # We'll use a different approach - modify the attention weights directly
    return output


def register_attention_hooks(model, layer_idx, head_idx):
    """
    Register hooks to ablate a specific attention head.
    """
    hooks = []

    # Find the attention module in the specified layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if layer_idx < len(model.model.layers):
            layer = model.model.layers[layer_idx]
            if hasattr(layer, "attn"):
                attn_module = layer.attn

                # Store ablation info on the module
                attn_module._ablate_layer_idx = layer_idx
                attn_module._ablate_head_idx = head_idx

                # Hook to modify attention weights before they're used
                def attention_hook(module, args, kwargs, output):
                    # Output is (hidden_states, attn_weights, past_key_value)
                    hidden_states = output[0]

                    # Get stored ablation info
                    head_idx_to_ablate = getattr(module, "_ablate_head_idx", None)

                    if head_idx_to_ablate is not None:
                        batch_size, seq_len, hidden_size = hidden_states.shape
                        num_heads = module.num_heads
                        head_dim = hidden_size // num_heads

                        # Reshape to separate heads
                        hidden_states = hidden_states.view(
                            batch_size, seq_len, num_heads, head_dim
                        )

                        # Zero out the specific head
                        hidden_states[:, :, head_idx_to_ablate, :] = 0

                        # Reshape back
                        hidden_states = hidden_states.view(
                            batch_size, seq_len, hidden_size
                        )

                        # Return modified output
                        return (hidden_states,) + output[1:]

                    return output

                # Register the hook
                hook = attn_module.register_forward_hook(
                    attention_hook, with_kwargs=True
                )
                hooks.append((attn_module, hook))

    return hooks


def remove_hooks(hooks):
    """
    Remove all registered hooks.
    """
    for module, hook in hooks:
        hook.remove()
        # Clean up temporary attributes
        if hasattr(module, "_ablate_layer_idx"):
            delattr(module, "_ablate_layer_idx")
        if hasattr(module, "_ablate_head_idx"):
            delattr(module, "_ablate_head_idx")


def evaluate_sample(
    model, tokenizer, text, layer_idx=None, head_idx=None, device="cuda"
):
    """
    Evaluate a single digit lookup sample with optional head ablation.
    """
    # Extract key and value from text
    if ":" in text:
        parts = text.split(":", 1)
        key = parts[0].strip()
        expected_value = parts[1].strip()
    else:
        return None

    # Register ablation hooks if specified
    hooks = []
    if layer_idx is not None and head_idx is not None:
        hooks = register_attention_hooks(model, layer_idx, head_idx)

    # Prepare input
    prompt = key + ":"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate token by token
    generated_tokens = []
    position_correct = []
    digit_correct = []

    with torch.no_grad():
        current_ids = inputs["input_ids"]

        for pos in range(len(expected_value)):
            # Generate next token
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]
            predicted_token = next_logits.argmax().item()

            # Decode predicted token
            predicted_text = tokenizer.decode(
                [predicted_token], skip_special_tokens=True
            )

            # Check if correct
            expected_char = expected_value[pos] if pos < len(expected_value) else ""
            is_correct = predicted_text == expected_char

            position_correct.append(is_correct)
            if expected_char.isdigit():
                digit_correct.append(
                    {
                        "digit": int(expected_char),
                        "correct": is_correct,
                        "position": pos,
                    }
                )

            generated_tokens.append(predicted_token)

            # Add predicted token to continue generation
            next_token_tensor = torch.tensor([[predicted_token]], device=device)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

    # Remove hooks
    remove_hooks(hooks)

    # Decode full generation
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "key": key,
        "expected_value": expected_value,
        "generated_value": generated_text,
        "position_correct": position_correct,
        "digit_correct": digit_correct,
        "fully_correct": generated_text == expected_value,
    }


def compute_accuracy_by_position(results):
    """
    Compute accuracy statistics by token position.
    """
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        if result is None:
            continue

        for pos, is_correct in enumerate(result["position_correct"]):
            position_stats[pos]["total"] += 1
            if is_correct:
                position_stats[pos]["correct"] += 1

    # Compute accuracies
    position_accuracies = {}
    for pos in sorted(position_stats.keys()):
        stats = position_stats[pos]
        if stats["total"] > 0:
            position_accuracies[pos] = stats["correct"] / stats["total"]
        else:
            position_accuracies[pos] = 0.0

    return position_accuracies


def compute_accuracy_by_digit(results):
    """
    Compute accuracy statistics by digit value.
    """
    digit_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        if result is None:
            continue

        for digit_info in result["digit_correct"]:
            digit = digit_info["digit"]
            digit_stats[digit]["total"] += 1
            if digit_info["correct"]:
                digit_stats[digit]["correct"] += 1

    # Compute accuracies
    digit_accuracies = {}
    for digit in range(10):
        stats = digit_stats[digit]
        if stats["total"] > 0:
            digit_accuracies[digit] = stats["correct"] / stats["total"]
        else:
            digit_accuracies[digit] = 0.0

    return digit_accuracies


def visualize_ablation_effects(
    position_effects,
    digit_effects,
    model_name,
    test_type,
    output_dir,
    baseline_pos_acc=None,
    baseline_digit_acc=None,
):
    """
    Create heatmap visualizations of ablation effects.

    Args:
        position_effects: Dict mapping (layer, head) to position accuracies
        digit_effects: Dict mapping (layer, head) to digit accuracies
        model_name: Name of the model for titles
        test_type: "known" or "novel"
        output_dir: Directory to save visualizations
        baseline_pos_acc: Baseline position accuracies for computing drops
        baseline_digit_acc: Baseline digit accuracies for computing drops
    """
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract dimensions
    layer_head_pairs = sorted(position_effects.keys())
    if not layer_head_pairs:
        print("No ablation data to visualize")
        return

    # Separate layers and heads
    layers = sorted(set(lh[0] for lh in layer_head_pairs))
    heads_per_layer = {}
    for layer in layers:
        heads_per_layer[layer] = sorted(
            set(lh[1] for lh in layer_head_pairs if lh[0] == layer)
        )

    # Figure 1: Position-based accuracy drop heatmap
    # Dynamic sizing based on content
    positions = sorted(next(iter(position_effects.values())).keys())
    total_heads = sum(len(heads_per_layer[layer]) for layer in layers)

    # Calculate figure dimensions - 0.8 inch per position, 0.6 inch per head
    fig_width = max(15, len(positions) * 0.8 + 3)  # +3 for margins and colorbar
    fig_height = max(8, total_heads * 0.6 + 2 * len(layers))  # +2 per layer for labels

    fig1, axes1 = plt.subplots(len(layers), 1, figsize=(fig_width, fig_height))
    if len(layers) == 1:
        axes1 = [axes1]

    for layer_idx, layer in enumerate(layers):
        heads = heads_per_layer[layer]

        # Create matrix for accuracy drops
        matrix = np.zeros((len(heads), len(positions)))

        for head_idx, head in enumerate(heads):
            if (layer, head) in position_effects:
                for pos_idx, pos in enumerate(positions):
                    ablated_acc = position_effects[(layer, head)].get(pos, 0)
                    baseline_acc = (
                        baseline_pos_acc.get(pos, 0) if baseline_pos_acc else 1
                    )
                    # Calculate accuracy drop as percentage
                    matrix[head_idx, pos_idx] = (baseline_acc - ablated_acc) * 100

        # Plot heatmap
        ax = axes1[layer_idx]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="Reds",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 10},
            cbar_kws={"label": "Accuracy Drop (%)"},
            xticklabels=[f"{p}" for p in positions],
            yticklabels=[f"Head {h} -" for h in heads],
            square=True,
        )
        ax.set_title(
            f"Layer {layer}: Head Ablation Effects on Answer Positions",
            fontsize=14,
            pad=10,
        )
        ax.set_xlabel("Position in Answer", fontsize=12)
        ax.set_ylabel("Attention Head", fontsize=12)

    plt.tight_layout()
    position_path = Path(output_dir) / f"{model_name}_{test_type}_position_heatmap.png"
    plt.savefig(position_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Digit-based accuracy drop heatmap
    # Dynamic sizing for digit heatmap - 10 digits fixed
    fig_width = max(12, 10 * 0.8 + 3)  # 10 digits, 0.8 inch each + margins
    fig_height = max(8, total_heads * 0.6 + 2 * len(layers))  # Same height calculation

    fig2, axes2 = plt.subplots(len(layers), 1, figsize=(fig_width, fig_height))
    if len(layers) == 1:
        axes2 = [axes2]

    for layer_idx, layer in enumerate(layers):
        heads = heads_per_layer[layer]

        # Create matrix for accuracy drops
        matrix = np.zeros((len(heads), 10))  # 10 digits

        for head_idx, head in enumerate(heads):
            if (layer, head) in digit_effects:
                for digit in range(10):
                    ablated_acc = digit_effects[(layer, head)].get(digit, 0)
                    baseline_acc = (
                        baseline_digit_acc.get(digit, 0) if baseline_digit_acc else 1
                    )
                    # Calculate accuracy drop as percentage
                    matrix[head_idx, digit] = (baseline_acc - ablated_acc) * 100

        # Plot heatmap
        ax = axes2[layer_idx]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 10},
            cbar_kws={"label": "Accuracy Drop (%)"},
            xticklabels=[str(d) for d in range(10)],
            yticklabels=[f"Head {h} -" for h in heads],
            square=True,
        )
        ax.set_title(
            f"Layer {layer}: Head Ablation Effects on Digit Values", fontsize=14, pad=10
        )
        ax.set_xlabel("Digit Value", fontsize=12)
        ax.set_ylabel("Attention Head", fontsize=12)

    plt.tight_layout()
    digit_path = Path(output_dir) / f"{model_name}_{test_type}_digit_heatmap.png"
    plt.savefig(digit_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved visualizations:")
    print(f"  - {position_path}")
    print(f"  - {digit_path}")


def main():
    """
    Main function to orchestrate head ablation analysis.
    """
    parser = argparse.ArgumentParser(
        description="Ablate attention heads and analyze digit lookup accuracy"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to HF model"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name for output files"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["1hop", "2steps"],
        required=True,
        help="Type of dataset to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/digit_lookup/results/ablation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=400, help="Number of samples per test set"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("ATTENTION HEAD ABLATION ANALYSIS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Samples per set: {args.num_samples}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_transformer_model(args.model_path, args.device)

    # Get model architecture info
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = (
        config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads
    )

    print("\nModel architecture:")
    print(f"- Layers: {num_layers}")
    print(f"- Heads per layer: {num_heads}")

    # Load test datasets
    print("\nLoading test data...")
    known_samples, novel_samples = load_test_data(
        args.dataset_type, args.num_samples, args.seed
    )

    # First evaluate without ablation (baseline)
    print("\nEvaluating baseline performance...")

    # Baseline on known pairs
    known_baseline_results = []
    for text in tqdm(known_samples, desc="Baseline (known)"):
        result = evaluate_sample(model, tokenizer, text, device=args.device)
        known_baseline_results.append(result)

    known_baseline_pos_acc = compute_accuracy_by_position(known_baseline_results)
    known_baseline_digit_acc = compute_accuracy_by_digit(known_baseline_results)

    print(f"Known pairs baseline accuracy by position: {known_baseline_pos_acc}")

    # Baseline on novel pairs
    novel_baseline_results = []
    for text in tqdm(novel_samples, desc="Baseline (novel)"):
        result = evaluate_sample(model, tokenizer, text, device=args.device)
        novel_baseline_results.append(result)

    novel_baseline_pos_acc = compute_accuracy_by_position(novel_baseline_results)
    novel_baseline_digit_acc = compute_accuracy_by_digit(novel_baseline_results)

    print(f"Novel pairs baseline accuracy by position: {novel_baseline_pos_acc}")

    # Initialize results storage
    all_results = {
        "config": {
            "model_path": args.model_path,
            "model_name": args.model_name,
            "dataset_type": args.dataset_type,
            "num_samples": args.num_samples,
            "num_layers": num_layers,
            "num_heads": num_heads,
        },
        "baseline": {
            "known": {
                "position_accuracy": known_baseline_pos_acc,
                "digit_accuracy": known_baseline_digit_acc,
            },
            "novel": {
                "position_accuracy": novel_baseline_pos_acc,
                "digit_accuracy": novel_baseline_digit_acc,
            },
        },
        "ablations": {},
    }

    # Ablation results storage
    known_position_effects = {}
    known_digit_effects = {}
    novel_position_effects = {}
    novel_digit_effects = {}

    # Perform ablation for each layer and head
    print("\nPerforming head ablations...")
    total_ablations = num_layers * num_heads
    ablation_count = 0

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            ablation_count += 1
            print(
                f"\nAblation {ablation_count}/{total_ablations}: Layer {layer_idx}, Head {head_idx}"
            )

            # Evaluate on known pairs with ablation
            known_ablation_results = []
            for text in tqdm(
                known_samples, desc=f"L{layer_idx}H{head_idx} (known)", leave=False
            ):
                result = evaluate_sample(
                    model, tokenizer, text, layer_idx, head_idx, args.device
                )
                known_ablation_results.append(result)

            known_pos_acc = compute_accuracy_by_position(known_ablation_results)
            known_digit_acc = compute_accuracy_by_digit(known_ablation_results)

            # Evaluate on novel pairs with ablation
            novel_ablation_results = []
            for text in tqdm(
                novel_samples, desc=f"L{layer_idx}H{head_idx} (novel)", leave=False
            ):
                result = evaluate_sample(
                    model, tokenizer, text, layer_idx, head_idx, args.device
                )
                novel_ablation_results.append(result)

            novel_pos_acc = compute_accuracy_by_position(novel_ablation_results)
            novel_digit_acc = compute_accuracy_by_digit(novel_ablation_results)

            # Store results
            known_position_effects[(layer_idx, head_idx)] = known_pos_acc
            known_digit_effects[(layer_idx, head_idx)] = known_digit_acc
            novel_position_effects[(layer_idx, head_idx)] = novel_pos_acc
            novel_digit_effects[(layer_idx, head_idx)] = novel_digit_acc

            # Store in full results
            ablation_key = f"L{layer_idx}H{head_idx}"
            all_results["ablations"][ablation_key] = {
                "layer": layer_idx,
                "head": head_idx,
                "known": {
                    "position_accuracy": known_pos_acc,
                    "digit_accuracy": known_digit_acc,
                },
                "novel": {
                    "position_accuracy": novel_pos_acc,
                    "digit_accuracy": novel_digit_acc,
                },
            }

    # Create visualizations
    print("\nCreating visualizations...")
    output_dir = Path(args.output_dir)

    # Visualize known pairs results
    visualize_ablation_effects(
        known_position_effects,
        known_digit_effects,
        args.model_name,
        "known",
        output_dir,
        baseline_pos_acc=known_baseline_pos_acc,
        baseline_digit_acc=known_baseline_digit_acc,
    )

    # Visualize novel pairs results
    visualize_ablation_effects(
        novel_position_effects,
        novel_digit_effects,
        args.model_name,
        "novel",
        output_dir,
        baseline_pos_acc=novel_baseline_pos_acc,
        baseline_digit_acc=novel_baseline_digit_acc,
    )

    # Save full results to JSON
    results_path = output_dir / f"{args.model_name}_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved full results to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find heads with largest impact
    print("\nHeads with largest impact on known pairs (position accuracy):")
    for (layer, head), pos_acc in sorted(
        known_position_effects.items(), key=lambda x: np.mean(list(x[1].values()))
    ):
        avg_acc = np.mean(list(pos_acc.values()))
        baseline_avg = np.mean(list(known_baseline_pos_acc.values()))
        drop = baseline_avg - avg_acc
        if drop > 0.1:  # Show only significant drops
            print(f"  L{layer}H{head}: {avg_acc:.2f} (drop: {drop:.2f})")

    print("\nHeads with largest impact on novel pairs (position accuracy):")
    for (layer, head), pos_acc in sorted(
        novel_position_effects.items(), key=lambda x: np.mean(list(x[1].values()))
    ):
        avg_acc = np.mean(list(pos_acc.values()))
        baseline_avg = np.mean(list(novel_baseline_pos_acc.values()))
        drop = baseline_avg - avg_acc
        if drop > 0.1:  # Show only significant drops
            print(f"  L{layer}H{head}: {avg_acc:.2f} (drop: {drop:.2f})")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
