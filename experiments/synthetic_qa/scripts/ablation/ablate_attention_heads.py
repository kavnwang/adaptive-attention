#!/usr/bin/env python3
"""
Head-wise ablation for synthetic QA evaluation.

For each attention head:
1. Insert a hook that zeros its output before it's added to the residual stream
2. Measure token-level accuracy for each position of the value string

This tests the hypothesis that each head is responsible for a specific digit position.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM
from bento.layers.sequence_mixers.attn import Attention


def hook_attention_output(module, args, output):
    """Hook function to zero out specific attention head outputs."""
    # output is a tuple: (hidden_states, attentions, past_key_values)
    hidden_states = output[0]

    # Get the head mask from the module (set during ablation)
    if hasattr(module, "_ablate_head_mask"):
        head_mask = module._ablate_head_mask  # (num_heads,)
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_heads = len(head_mask)
        head_dim = hidden_size // num_heads

        # Reshape to separate heads
        hidden_states = hidden_states.view(batch_size, seq_len, num_heads, head_dim)

        # Apply mask (zero out ablated heads)
        for head_idx in range(num_heads):
            if head_mask[head_idx] == 0:
                hidden_states[:, :, head_idx, :] = 0

        # Reshape back
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        # Return modified output
        return (hidden_states,) + output[1:]

    return output


def hook_attention_patterns(module, args, output):
    """Hook to capture attention patterns."""
    # output is a tuple: (hidden_states, attentions, past_key_values)
    if len(output) >= 2 and output[1] is not None:
        # Store attention weights in the module
        module._last_attention_weights = output[1]
    return output


def register_attention_hooks(model, layer_idx, head_idx):
    """Register hooks to ablate a specific attention head."""
    hooks = []

    # Find the attention module in the specified layer
    if layer_idx < len(model.model.layers):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "attn") and isinstance(layer.attn, Attention):
            attn_module = layer.attn

            # Create head mask (1 for active heads, 0 for ablated)
            num_heads = attn_module.num_heads
            head_mask = torch.ones(num_heads)
            head_mask[head_idx] = 0

            # Set the mask on the module
            attn_module._ablate_head_mask = head_mask

            # Register the hook
            hook = attn_module.register_forward_hook(hook_attention_output)
            hooks.append((attn_module, hook))

    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks and clean up."""
    for module, hook in hooks:
        hook.remove()
        if hasattr(module, "_ablate_head_mask"):
            delattr(module, "_ablate_head_mask")


def capture_attention_patterns(model, tokenizer, question, answer, device="cuda"):
    """Capture attention patterns while generating answer."""
    # Tokenize inputs
    inputs = tokenizer(question, return_tensors="pt").to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)

    # Register hooks to capture attention
    hooks = []
    attention_patterns = {}

    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "attn") and isinstance(layer.attn, Attention):
            # Force output_attentions=True by modifying forward
            original_forward = layer.attn.forward

            def make_forward_with_attention(orig_forward, layer_idx):
                def forward_with_attention(*args, **kwargs):
                    kwargs["output_attentions"] = True
                    output = orig_forward(*args, **kwargs)
                    # Store attention weights
                    if len(output) >= 2 and output[1] is not None:
                        attention_patterns[layer_idx] = output[1].detach()
                    return output

                return forward_with_attention

            layer.attn.forward = make_forward_with_attention(
                original_forward, layer_idx
            )
            hooks.append((layer.attn, original_forward))

    # Generate and capture attention
    current_ids = inputs["input_ids"]
    with torch.no_grad():
        # Run full generation to capture attention at answer positions
        full_ids = torch.cat([current_ids, answer_token_ids.unsqueeze(0)], dim=1)
        outputs = model(input_ids=full_ids, output_attentions=True)

    # Restore original forwards
    for module, original_forward in hooks:
        module.forward = original_forward

    return attention_patterns


def evaluate_token_accuracy(
    model,
    tokenizer,
    question,
    answer,
    layer_idx=None,
    head_idx=None,
    device="cuda",
    capture_attention=False,
):
    """
    Evaluate token-level accuracy for each position in the answer.
    If layer_idx and head_idx are specified, ablate that head during evaluation.

    Returns:
        - token_accuracies: List of booleans indicating if each token was predicted correctly
        - predicted_tokens: List of predicted token ids
        - predicted_text: The generated text
        - attention_patterns: (optional) Captured attention patterns if capture_attention=True
    """
    # Tokenize inputs
    inputs = tokenizer(question, return_tensors="pt").to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)

    # Register ablation hooks if specified
    hooks = []
    if layer_idx is not None and head_idx is not None:
        hooks = register_attention_hooks(model, layer_idx, head_idx)

    # Capture attention patterns if requested (only for clean runs)
    attention_patterns = None
    if capture_attention and layer_idx is None:
        attention_patterns = capture_attention_patterns(
            model, tokenizer, question, answer, device
        )

    # Generate answer token by token
    predicted_tokens = []
    token_accuracies = []
    current_ids = inputs["input_ids"]

    with torch.no_grad():
        for i in range(len(answer_token_ids)):
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]

            # Get predicted token
            predicted_token = next_logits.argmax().item()
            predicted_tokens.append(predicted_token)

            # Check if correct
            is_correct = predicted_token == answer_token_ids[i].item()
            token_accuracies.append(is_correct)

            # Add predicted token to continue generation
            next_token_tensor = torch.tensor([[predicted_token]], device=device)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

    # Remove hooks
    remove_hooks(hooks)

    # Decode predicted text
    predicted_token_tensor = torch.tensor(predicted_tokens, device=device)
    predicted_text = tokenizer.decode(predicted_token_tensor, skip_special_tokens=True)

    result = {
        "token_accuracies": token_accuracies,
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_text,
        "answer_tokens": answer_token_ids.tolist(),
    }

    if attention_patterns is not None:
        result["attention_patterns"] = attention_patterns

    return result


def analyze_head_specialization(results):
    """
    Analyze if each head is specialized for a specific digit position.

    Returns a summary of which heads affect which positions.
    """
    # Collect statistics: head -> position -> accuracy drop
    head_position_effects = defaultdict(lambda: defaultdict(list))

    for qa_result in results:
        clean_acc = qa_result["clean_eval"]["token_accuracies"]

        for ablation in qa_result["ablations"]:
            layer_idx = ablation["layer_idx"]
            head_idx = ablation["head_idx"]
            ablated_acc = ablation["token_accuracies"]

            # Calculate accuracy drop for each position
            for pos in range(len(clean_acc)):
                if clean_acc[
                    pos
                ]:  # Only consider positions that were correct originally
                    accuracy_drop = 1.0 if not ablated_acc[pos] else 0.0
                    head_position_effects[(layer_idx, head_idx)][pos].append(
                        accuracy_drop
                    )

    # Compute average effects
    head_specialization = {}
    for (layer_idx, head_idx), position_effects in head_position_effects.items():
        head_key = f"L{layer_idx}H{head_idx}"
        head_specialization[head_key] = {}

        for pos, drops in position_effects.items():
            avg_drop = sum(drops) / len(drops) if drops else 0
            head_specialization[head_key][f"pos_{pos}"] = {
                "avg_accuracy_drop": avg_drop,
                "num_samples": len(drops),
            }

    return head_specialization


def analyze_digit_specialization(results, tokenizer):
    """
    Analyze if each head is specialized for specific digit values (0-9).

    Returns a summary of which heads affect which digit values.
    """
    # Collect statistics: head -> digit -> accuracy drop
    head_digit_effects = defaultdict(lambda: defaultdict(list))

    for qa_result in results:
        clean_acc = qa_result["clean_eval"]["token_accuracies"]
        answer_tokens = qa_result["clean_eval"]["answer_tokens"]
        answer_text = qa_result["answer"]

        # Decode each token to get the digit value
        digit_values = []
        for token_id in answer_tokens:
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            # Extract digit if present
            if token_text and token_text[0].isdigit():
                digit_values.append(int(token_text[0]))
            else:
                digit_values.append(-1)  # Non-digit token

        for ablation in qa_result["ablations"]:
            layer_idx = ablation["layer_idx"]
            head_idx = ablation["head_idx"]
            ablated_acc = ablation["token_accuracies"]

            # Calculate accuracy drop for each digit
            for pos in range(len(clean_acc)):
                if (
                    clean_acc[pos] and digit_values[pos] >= 0
                ):  # Only consider correct digits
                    digit = digit_values[pos]
                    accuracy_drop = 1.0 if not ablated_acc[pos] else 0.0
                    head_digit_effects[(layer_idx, head_idx)][digit].append(
                        accuracy_drop
                    )

    # Compute average effects
    digit_specialization = {}
    for (layer_idx, head_idx), digit_effects in head_digit_effects.items():
        head_key = f"L{layer_idx}H{head_idx}"
        digit_specialization[head_key] = {}

        for digit in range(10):  # 0-9
            if digit in digit_effects:
                drops = digit_effects[digit]
                avg_drop = sum(drops) / len(drops) if drops else 0
                digit_specialization[head_key][f"digit_{digit}"] = {
                    "avg_accuracy_drop": avg_drop,
                    "num_samples": len(drops),
                }
            else:
                digit_specialization[head_key][f"digit_{digit}"] = {
                    "avg_accuracy_drop": 0,
                    "num_samples": 0,
                }

    return digit_specialization


def visualize_head_effects(
    head_specialization, layers_tested, heads_tested, max_answer_length, output_path
):
    """Create a heatmap visualization of head ablation effects on each position."""
    # Create a matrix for visualization
    n_layers = len(layers_tested)
    n_heads = len(heads_tested)

    # Adjust figure size to ensure readability
    # Width: at least 0.8 inch per digit position, height: 0.6 inch per head
    fig_width = max(15, max_answer_length * 0.8 + 2)  # +2 for margins
    fig_height = max(6, n_heads * 0.6 + 2) * n_layers  # +2 for margins per layer

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(n_layers, 1, figsize=(fig_width, fig_height))
    if n_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        layer = layers_tested[layer_idx]

        # Create matrix for this layer
        matrix = np.zeros((n_heads, max_answer_length))

        for head_idx_in_list, actual_head_idx in enumerate(heads_tested):
            head_key = f"L{layer}H{actual_head_idx}"
            if head_key in head_specialization:
                for pos in range(max_answer_length):
                    pos_key = f"pos_{pos}"
                    if pos_key in head_specialization[head_key]:
                        matrix[head_idx_in_list, pos] = head_specialization[head_key][
                            pos_key
                        ]["avg_accuracy_drop"]

        # Create heatmap with better formatting
        sns.heatmap(
            matrix * 100,  # Convert to percentage
            ax=ax,
            cmap="Reds",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 8},  # Smaller annotation font
            cbar_kws={"label": "Accuracy Drop (%)", "shrink": 0.8},
            xticklabels=[f"Pos {i}" for i in range(max_answer_length)],
            yticklabels=[f"Head {h}" for h in heads_tested],
            square=True,  # Make cells square for better readability
        )

        ax.set_title(
            f"Layer {layer}: Head Ablation Effects on Answer Positions",
            fontsize=14,
            pad=10,
        )
        ax.set_xlabel("Position in Answer", fontsize=12)
        ax.set_ylabel("Attention Head", fontsize=12)

        # Rotate x labels if there are many positions
        if max_answer_length > 10:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_path}")


def visualize_digit_effects(
    digit_specialization, layers_tested, heads_tested, output_path
):
    """Create a heatmap visualization of head ablation effects on each digit value (0-9)."""
    # Create a matrix for visualization
    n_layers = len(layers_tested)
    n_heads = len(heads_tested)

    # Fixed width for 10 digits
    fig_width = max(12, 10 * 0.8 + 2)  # 10 digits
    fig_height = max(6, n_heads * 0.6 + 2) * n_layers

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(n_layers, 1, figsize=(fig_width, fig_height))
    if n_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        layer = layers_tested[layer_idx]

        # Create matrix for this layer (heads x 10 digits)
        matrix = np.zeros((n_heads, 10))

        for head_idx_in_list, actual_head_idx in enumerate(heads_tested):
            head_key = f"L{layer}H{actual_head_idx}"
            if head_key in digit_specialization:
                for digit in range(10):
                    digit_key = f"digit_{digit}"
                    if digit_key in digit_specialization[head_key]:
                        matrix[head_idx_in_list, digit] = digit_specialization[
                            head_key
                        ][digit_key]["avg_accuracy_drop"]

        # Create heatmap
        sns.heatmap(
            matrix * 100,  # Convert to percentage
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 9},
            cbar_kws={"label": "Accuracy Drop (%)", "shrink": 0.8},
            xticklabels=[str(d) for d in range(10)],
            yticklabels=[f"Head {h}" for h in heads_tested],
            square=True,
        )

        ax.set_title(
            f"Layer {layer}: Head Ablation Effects on Digit Values", fontsize=14, pad=10
        )
        ax.set_xlabel("Digit Value", fontsize=12)
        ax.set_ylabel("Attention Head", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved digit visualization to {output_path}")


def visualize_attention_patterns(attention_patterns, output_path):
    """Visualize attention patterns for a single example."""
    if not attention_patterns:
        print("No attention patterns captured")
        return

    # Get dimensions
    n_layers = len(attention_patterns)
    example_attn = list(attention_patterns.values())[0]
    n_heads = (
        example_attn.shape[0] if len(example_attn.shape) == 3 else example_attn.shape[1]
    )
    seq_len = example_attn.shape[-1]

    # Create figure
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_heads == 1:
        axes = axes.reshape(-1, 1)

    for layer_idx, (layer, attn) in enumerate(sorted(attention_patterns.items())):
        # attn shape: (batch=1, n_heads, seq_len, seq_len) or (n_heads, seq_len, seq_len)
        if len(attn.shape) == 4:
            attn = attn[0]  # Remove batch dimension

        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]

            # Get attention weights for this head
            head_attn = attn[head_idx].cpu().numpy()

            # Plot heatmap
            im = ax.imshow(head_attn, cmap="Blues", aspect="auto")
            ax.set_title(f"L{layer}H{head_idx}", fontsize=8)

            # Only show labels on leftmost and bottom plots
            if head_idx == 0:
                ax.set_ylabel("Query Pos", fontsize=8)
            if layer_idx == n_layers - 1:
                ax.set_xlabel("Key Pos", fontsize=8)

            # Reduce tick labels
            ax.set_xticks(range(0, seq_len, max(1, seq_len // 5)))
            ax.set_yticks(range(0, seq_len, max(1, seq_len // 5)))
            ax.tick_params(labelsize=6)

    plt.suptitle("Attention Patterns by Layer and Head", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved attention patterns to {output_path}")


def visualize_one_to_one_mapping(head_specialization, output_path):
    """Visualize which heads are responsible for which positions."""
    # Find strong connections (>80% accuracy drop)
    connections = []
    head_to_positions = defaultdict(list)
    position_to_heads = defaultdict(list)

    for head_key, position_effects in head_specialization.items():
        for pos_key, effect_data in position_effects.items():
            if effect_data["avg_accuracy_drop"] > 0.8:
                pos = int(pos_key.split("_")[1])
                connections.append((head_key, pos, effect_data["avg_accuracy_drop"]))
                head_to_positions[head_key].append(pos)
                position_to_heads[pos].append(head_key)

    if not connections:
        print("No strong head-position connections found (>80% accuracy drop)")
        return

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Bipartite graph
    heads = sorted(set(conn[0] for conn in connections))
    positions = sorted(set(conn[1] for conn in connections))

    head_y = np.linspace(0, 1, len(heads))
    pos_y = np.linspace(0, 1, len(positions))

    for head, pos, strength in connections:
        head_idx = heads.index(head)
        pos_idx = positions.index(pos)

        # Color based on exclusivity
        if len(position_to_heads[pos]) == 1 and len(head_to_positions[head]) == 1:
            color = "green"  # Exclusive one-to-one
            alpha = 0.8
        else:
            color = "red"  # Shared
            alpha = 0.4

        ax1.plot(
            [0, 1],
            [head_y[head_idx], pos_y[pos_idx]],
            color=color,
            alpha=alpha,
            linewidth=strength * 3,
        )

    # Draw nodes
    ax1.scatter([0] * len(heads), head_y, s=100, c="blue", zorder=5)
    ax1.scatter([1] * len(positions), pos_y, s=100, c="orange", zorder=5)

    # Add labels
    for i, head in enumerate(heads):
        ax1.text(-0.1, head_y[i], head, ha="right", va="center", fontsize=8)

    for i, pos in enumerate(positions):
        ax1.text(1.1, pos_y[i], f"Pos {pos}", ha="left", va="center", fontsize=8)

    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Head-to-Position Mapping\n(Green=Exclusive, Red=Shared)")

    # Right plot: Summary statistics
    ax2.axis("off")
    summary_text = "=== Mapping Analysis ===\n\n"

    # Count exclusive mappings
    exclusive_positions = sum(
        1 for pos, heads in position_to_heads.items() if len(heads) == 1
    )
    exclusive_heads = sum(
        1 for head, positions in head_to_positions.items() if len(positions) == 1
    )

    summary_text += f"Total positions with effects: {len(positions)}\n"
    summary_text += f"Positions with exclusive head: {exclusive_positions}\n"
    summary_text += f"Heads affecting single position: {exclusive_heads}\n\n"

    # List one-to-one mappings
    summary_text += "One-to-One Mappings:\n"
    for pos, heads in sorted(position_to_heads.items()):
        if len(heads) == 1 and len(head_to_positions[heads[0]]) == 1:
            summary_text += f"  Position {pos} ← {heads[0]}\n"

    # List shared positions
    shared_positions = [
        (pos, heads) for pos, heads in position_to_heads.items() if len(heads) > 1
    ]
    if shared_positions:
        summary_text += "\nShared Positions:\n"
        for pos, heads in sorted(shared_positions):
            summary_text += f"  Position {pos} ← {', '.join(heads)}\n"

    ax2.text(
        0.1,
        0.9,
        summary_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved mapping visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablate attention heads and measure impact on digit positions"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        choices=["200", "2K", "10K", "50K"],
        required=True,
        help="Dataset size to use",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file with ablation results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of QA pairs to process",
    )
    parser.add_argument(
        "--layers_to_test",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer indices to test (default: all layers with attention)",
    )
    parser.add_argument(
        "--heads_to_test",
        type=int,
        nargs="+",
        default=None,
        help="Specific head indices to test (default: all heads)",
    )

    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model = MemoryForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset (size: {args.dataset_size})...")
    dataset_path = (
        f"experiments/synthetic_qa/datasets/synthetic_qa_data_{args.dataset_size}"
    )
    dataset = load_dataset(dataset_path, split="test")

    # Limit dataset size if requested
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)} samples")

    # Determine which layers and heads to test
    if args.layers_to_test is None:
        # Find all layers with attention
        layers_to_test = []
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "attn"):
                layers_to_test.append(i)
    else:
        layers_to_test = args.layers_to_test

    # Get number of heads from first attention layer
    num_heads = None
    for layer in model.model.layers:
        if hasattr(layer, "attn"):
            num_heads = layer.attn.num_heads
            break

    if num_heads is None:
        print("Error: Could not determine number of attention heads")
        return

    heads_to_test = args.heads_to_test if args.heads_to_test else list(range(num_heads))

    print(f"Testing layers: {layers_to_test}")
    print(f"Testing heads: {heads_to_test}")
    print(f"Total ablations per sample: {len(layers_to_test) * len(heads_to_test)}")

    # Setup output directory
    output_path = Path(args.output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process QA pairs
    results = []

    for i in tqdm(range(len(dataset)), desc="Processing QA pairs"):
        sample = dataset[i]
        text = sample["text"]

        # Extract Q&A
        parts = text.split(":")
        if len(parts) != 2:
            continue

        key = parts[0].strip() + ":"
        value = parts[1].strip()

        # For this dataset, values are digit sequences like "007717"
        # We can analyze any value format, so let's not skip any

        # Evaluate without ablation (clean run)
        # Skip attention pattern capture for models that don't support it
        clean_eval = evaluate_token_accuracy(
            model,
            tokenizer,
            key,
            value,
            device=args.device,
            capture_attention=False,  # Disabled for memory models
        )

        # Skip if model doesn't get it right in clean run
        if not all(clean_eval["token_accuracies"]):
            continue

        qa_result = {
            "key": key,
            "value": value,
            "value_length": len(value),
            "clean_eval": clean_eval,
            "ablations": [],
        }

        # Skip attention patterns visualization for memory models

        # Test each head
        for layer_idx in layers_to_test:
            for head_idx in heads_to_test:
                ablation_eval = evaluate_token_accuracy(
                    model,
                    tokenizer,
                    question,
                    answer,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    device=args.device,
                )

                qa_result["ablations"].append(
                    {
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "token_accuracies": ablation_eval["token_accuracies"],
                        "predicted_text": ablation_eval["predicted_text"],
                        "accuracy": sum(ablation_eval["token_accuracies"])
                        / len(ablation_eval["token_accuracies"]),
                    }
                )

        results.append(qa_result)

    # Analyze head specialization
    print("\nAnalyzing head specialization...")
    head_specialization = analyze_head_specialization(results)

    # Analyze digit specialization
    print("\nAnalyzing digit specialization...")
    digit_specialization = analyze_digit_specialization(results, tokenizer)

    # Find max answer length for visualization
    max_answer_length = max(qa["answer_length"] for qa in results) if results else 0

    # Create visualizations
    if results and max_answer_length > 0:
        # Position-based heatmap visualization
        heatmap_path = output_dir / f"{output_path.stem}_position_heatmap.png"
        visualize_head_effects(
            head_specialization,
            layers_to_test,
            heads_to_test,
            max_answer_length,
            heatmap_path,
        )

        # Digit-based heatmap visualization
        digit_heatmap_path = output_dir / f"{output_path.stem}_digit_heatmap.png"
        visualize_digit_effects(
            digit_specialization, layers_to_test, heads_to_test, digit_heatmap_path
        )

        # Mapping visualization
        mapping_path = output_dir / f"{output_path.stem}_mapping.png"
        visualize_one_to_one_mapping(head_specialization, mapping_path)

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    output_data = {
        "config": {
            "model_path": args.model_path,
            "dataset_size": args.dataset_size,
            "num_samples": len(results),
            "layers_tested": layers_to_test,
            "heads_tested": heads_to_test,
            "num_heads": num_heads,
        },
        "results": results,
        "head_specialization_summary": head_specialization,
        "digit_specialization_summary": digit_specialization,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")

    # Print summary of findings
    print("\n=== Head Specialization Summary ===")
    print(
        "Looking for heads that cause specific digit positions to fail when ablated..."
    )

    for head_key, position_effects in head_specialization.items():
        significant_effects = []
        for pos_key, effect_data in position_effects.items():
            if effect_data["avg_accuracy_drop"] > 0.8:  # 80% drop threshold
                pos = int(pos_key.split("_")[1])
                significant_effects.append((pos, effect_data["avg_accuracy_drop"]))

        if significant_effects:
            print(f"\n{head_key}:")
            for pos, drop in sorted(significant_effects):
                print(f"  - Position {pos}: {drop * 100:.1f}% accuracy drop")

    # Check the hypothesis
    print("\n=== Hypothesis Check ===")
    print("Hypothesis: Each head is responsible for exactly one digit position")
    print(
        "Expected: Ablating head h causes digit h to drop to ~10% accuracy, others stay at 100%"
    )

    # For each head, check if it affects exactly one position strongly
    heads_with_single_position = 0
    for head_key, position_effects in head_specialization.items():
        strong_effects = []
        for pos_key, effect_data in position_effects.items():
            if effect_data["avg_accuracy_drop"] > 0.8:
                strong_effects.append(pos_key)

        if len(strong_effects) == 1:
            heads_with_single_position += 1
            print(f"{head_key} affects only {strong_effects[0]} (✓)")
        elif len(strong_effects) > 1:
            print(f"{head_key} affects multiple positions: {strong_effects} (✗)")
        else:
            print(f"{head_key} has no strong effect on any position (✗)")

    total_heads = len(head_specialization)
    print(
        f"\nHeads with single-position specialization: {heads_with_single_position}/{total_heads}"
    )

    if heads_with_single_position == total_heads:
        print(
            "✓ Hypothesis SUPPORTED: Each head specializes in exactly one digit position"
        )
    else:
        print(
            "✗ Hypothesis NOT SUPPORTED: Heads do not show clear one-to-one digit specialization"
        )


if __name__ == "__main__":
    main()
