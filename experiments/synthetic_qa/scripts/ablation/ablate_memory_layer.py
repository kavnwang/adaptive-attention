#!/usr/bin/env python3
"""
Ablate the entire memory layer and measure token-level accuracy for each position.

This script:
1. Hooks into the memory layer to zero out its entire output
2. Measures token-level accuracy for each position in the generated answers
3. Compares performance with and without the memory layer
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
import numpy as np

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM, MemoryLayer


def hook_memory_output(module, args, output):
    """Hook function to zero out memory layer output."""
    # For memory layers, output is typically the hidden states
    # Zero out the entire output
    if isinstance(output, torch.Tensor):
        return torch.zeros_like(output)
    elif isinstance(output, tuple):
        # If output is a tuple, zero out the first element (hidden states)
        return (torch.zeros_like(output[0]),) + output[1:]
    return output


def register_memory_hooks(model):
    """Register hooks to ablate all memory layers."""
    hooks = []

    # Find all memory modules
    for name, module in model.named_modules():
        if isinstance(module, MemoryLayer):
            # Register the hook
            hook = module.register_forward_hook(hook_memory_output)
            hooks.append((module, hook))
            print(f"Registered ablation hook on memory layer: {name}")

    if not hooks:
        print("Warning: No memory layers found in the model!")

    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks."""
    for module, hook in hooks:
        hook.remove()


def evaluate_token_accuracy(
    model, tokenizer, question, answer, ablate_memory=False, device="cuda"
):
    """
    Evaluate token-level accuracy for each position in the answer.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        question: The question string
        answer: The expected answer string
        ablate_memory: Whether to ablate memory layers
        device: Device to use

    Returns:
        dict with:
        - token_accuracies: List of booleans for each position
        - predicted_tokens: List of predicted token ids
        - predicted_text: The generated text
        - answer_tokens: List of expected token ids
    """
    # Tokenize inputs
    inputs = tokenizer(question, return_tensors="pt").to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)

    # Register ablation hooks if requested
    hooks = []
    if ablate_memory:
        hooks = register_memory_hooks(model)

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

    return {
        "token_accuracies": token_accuracies,
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_text,
        "answer_tokens": answer_token_ids.tolist(),
    }


def analyze_position_effects(results):
    """
    Analyze accuracy by position with and without memory layer.

    Returns position-wise accuracy statistics.
    """
    # Collect accuracies by position
    clean_position_acc = defaultdict(list)
    ablated_position_acc = defaultdict(list)

    for result in results:
        clean_acc = result["clean_eval"]["token_accuracies"]
        ablated_acc = result["ablated_eval"]["token_accuracies"]

        # Record accuracy for each position
        for pos, (clean, ablated) in enumerate(zip(clean_acc, ablated_acc)):
            clean_position_acc[pos].append(1.0 if clean else 0.0)
            ablated_position_acc[pos].append(1.0 if ablated else 0.0)

    # Compute average accuracies
    position_stats = {}
    max_pos = (
        max(max(clean_position_acc.keys()), max(ablated_position_acc.keys()))
        if clean_position_acc
        else -1
    )

    for pos in range(max_pos + 1):
        clean_acc_list = clean_position_acc.get(pos, [])
        ablated_acc_list = ablated_position_acc.get(pos, [])

        position_stats[pos] = {
            "clean_accuracy": sum(clean_acc_list) / len(clean_acc_list)
            if clean_acc_list
            else 0,
            "ablated_accuracy": sum(ablated_acc_list) / len(ablated_acc_list)
            if ablated_acc_list
            else 0,
            "accuracy_drop": 0,  # Will compute below
            "num_samples": len(clean_acc_list),
        }

        # Compute accuracy drop
        position_stats[pos]["accuracy_drop"] = (
            position_stats[pos]["clean_accuracy"]
            - position_stats[pos]["ablated_accuracy"]
        )

    return position_stats


def visualize_position_accuracy(position_stats, output_path):
    """Create visualization of position-wise accuracy with and without memory layer."""
    if not position_stats:
        print("No position statistics to visualize")
        return

    positions = sorted(position_stats.keys())
    clean_acc = [position_stats[p]["clean_accuracy"] * 100 for p in positions]
    ablated_acc = [position_stats[p]["ablated_accuracy"] * 100 for p in positions]
    accuracy_drops = [position_stats[p]["accuracy_drop"] * 100 for p in positions]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(positions) * 0.5), 10))

    # Top plot: Clean vs Ablated accuracy
    x = np.arange(len(positions))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        clean_acc,
        width,
        label="With Memory Layer",
        color="blue",
        alpha=0.7,
    )
    bars2 = ax1.bar(
        x + width / 2,
        ablated_acc,
        width,
        label="Without Memory Layer",
        color="red",
        alpha=0.7,
    )

    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title(
        "Token Accuracy by Position: With vs Without Memory Layer", fontsize=14
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Pos {p}" for p in positions])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 105)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only label if > 5%
                ax1.annotate(
                    f"{height:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Bottom plot: Accuracy drop
    bars3 = ax2.bar(x, accuracy_drops, width=0.6, color="darkred", alpha=0.7)

    ax2.set_ylabel("Accuracy Drop (%)", fontsize=12)
    ax2.set_xlabel("Position in Answer", fontsize=12)
    ax2.set_title("Accuracy Drop When Memory Layer is Ablated", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Pos {p}" for p in positions])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        if height > 5:  # Only label if > 5%
            ax2.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Rotate x labels if many positions
    if len(positions) > 10:
        ax1.set_xticklabels([f"Pos {p}" for p in positions], rotation=45, ha="right")
        ax2.set_xticklabels([f"Pos {p}" for p in positions], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablate memory layer and measure impact on position-wise accuracy"
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
        default=100,
        help="Maximum number of QA pairs to process",
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

    # Setup output directory
    output_path = Path(args.output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process QA pairs
    results = []
    total_processed = 0
    total_clean_correct = 0
    total_ablated_correct = 0

    for i in tqdm(range(len(dataset)), desc="Processing QA pairs"):
        sample = dataset[i]
        text = sample["text"]

        # Extract Q&A
        parts = text.split("? ")
        if len(parts) != 2:
            continue

        question = parts[0] + "?"
        answer = parts[1].strip()

        # Evaluate without ablation (clean run)
        clean_eval = evaluate_token_accuracy(
            model, tokenizer, question, answer, ablate_memory=False, device=args.device
        )

        # Skip if model doesn't get it right in clean run
        # (we want to see the effect on correctly answered questions)
        if not all(clean_eval["token_accuracies"]):
            continue

        # Evaluate with memory ablation
        ablated_eval = evaluate_token_accuracy(
            model, tokenizer, question, answer, ablate_memory=True, device=args.device
        )

        result = {
            "question": question,
            "answer": answer,
            "answer_length": len(answer),
            "clean_eval": clean_eval,
            "ablated_eval": ablated_eval,
            "clean_accuracy": sum(clean_eval["token_accuracies"])
            / len(clean_eval["token_accuracies"]),
            "ablated_accuracy": sum(ablated_eval["token_accuracies"])
            / len(ablated_eval["token_accuracies"]),
        }

        results.append(result)
        total_processed += 1

        # Count fully correct predictions
        if all(clean_eval["token_accuracies"]):
            total_clean_correct += 1
        if all(ablated_eval["token_accuracies"]):
            total_ablated_correct += 1

    print(f"\nProcessed {total_processed} samples where clean model was correct")

    # Analyze position effects
    print("\nAnalyzing position-wise effects...")
    position_stats = analyze_position_effects(results)

    # Calculate overall statistics
    overall_clean_acc = (
        sum(r["clean_accuracy"] for r in results) / len(results) if results else 0
    )
    overall_ablated_acc = (
        sum(r["ablated_accuracy"] for r in results) / len(results) if results else 0
    )

    print("\n=== Overall Statistics ===")
    print(f"Average token accuracy (clean): {overall_clean_acc * 100:.1f}%")
    print(f"Average token accuracy (ablated): {overall_ablated_acc * 100:.1f}%")
    print(
        f"Average accuracy drop: {(overall_clean_acc - overall_ablated_acc) * 100:.1f}%"
    )
    print(
        f"Fully correct predictions (clean): {total_clean_correct}/{total_processed} ({total_clean_correct / total_processed * 100:.1f}%)"
    )
    print(
        f"Fully correct predictions (ablated): {total_ablated_correct}/{total_processed} ({total_ablated_correct / total_processed * 100:.1f}%)"
    )

    # Print position-wise statistics
    print("\n=== Position-wise Accuracy ===")
    for pos in sorted(position_stats.keys()):
        stats = position_stats[pos]
        print(
            f"Position {pos}: {stats['clean_accuracy'] * 100:.1f}% → {stats['ablated_accuracy'] * 100:.1f}% "
            f"(drop: {stats['accuracy_drop'] * 100:.1f}%, n={stats['num_samples']})"
        )

    # Create visualization
    if position_stats:
        vis_path = output_dir / f"{output_path.stem}_visualization.png"
        visualize_position_accuracy(position_stats, vis_path)

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    output_data = {
        "config": {
            "model_path": args.model_path,
            "dataset_size": args.dataset_size,
            "num_samples": len(results),
            "max_samples": args.max_samples,
        },
        "overall_statistics": {
            "clean_token_accuracy": overall_clean_acc,
            "ablated_token_accuracy": overall_ablated_acc,
            "accuracy_drop": overall_clean_acc - overall_ablated_acc,
            "clean_fully_correct": total_clean_correct,
            "ablated_fully_correct": total_ablated_correct,
            "total_processed": total_processed,
        },
        "position_statistics": position_stats,
        "results": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
