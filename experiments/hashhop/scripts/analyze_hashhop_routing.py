#!/usr/bin/env python3
"""
Analyze token routing patterns between MLP and memory layers on HashHop dataset.

This script evaluates a trained model with MLP/memory routing to understand:
1. Overall routing statistics per layer (% MLP vs % memory)
2. Token-level routing patterns for hash tokens vs regular tokens
3. Position-based routing (beginning, middle, end of sequences)
4. Accuracy correlation with routing decisions
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import re
import wandb
from typing import Dict, List, Tuple
import sys
import os

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))


def load_model_and_tokenizer(
    model_path: str, tokenizer_path: str, device: str = "cuda"
):
    """Load the model with routing capabilities and tokenizer."""
    print(f"Loading model from {model_path}")

    # Import the custom model class
    from bento.models.memory_mlp_router.modeling_memory import MemoryForCausalLM

    # Load model using the custom class
    model = MemoryForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    model.eval()

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def categorize_hashhop_token(token: str, tokenizer) -> str:
    """Categorize a token in hash-hop context."""
    # Decode if token is an ID
    if isinstance(token, int):
        token = tokenizer.decode([token])

    # Remove leading space if present
    clean_token = token.lstrip("Ġ").lstrip("▁").lstrip()

    # Hash-hop specific categories
    if not clean_token or clean_token.isspace():
        return "whitespace"
    elif clean_token == "=":
        return "equals_sign"
    elif clean_token == "'":
        return "quote"
    elif clean_token == "\n":
        return "newline"
    elif clean_token in ["COMPLETION:", "Hops=", "CoT="]:
        return "instruction"
    elif re.match(r"^[a-zA-Z]{4,20}$", clean_token):  # Hash strings
        return "hash_string"
    elif clean_token.isdigit():
        return "digit"
    elif clean_token in [".", ",", ":", ";"]:
        return "punctuation"
    else:
        return "other"


def parse_hashhop_completion(completion: str) -> Dict[str, str]:
    """Parse expected answers from hash-hop completion."""
    answers = {}

    # Split by lines and look for answer patterns
    lines = completion.split("\n")
    for line in lines:
        # Pattern: "key = 'value'" or "key = 'chain = of = values'"
        if " = '" in line and line.endswith("'"):
            parts = line.split(" = '", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].rstrip("'")
                answers[key] = value

    return answers


def evaluate_hashhop_sample(
    model, tokenizer, sample: Dict, device: str = "cuda"
) -> Tuple[Dict, Dict]:
    """Evaluate a single hash-hop sample and collect routing statistics."""

    # Prepare input
    prompt = sample["prompt"]
    expected_completion = sample["completion"]

    # Add the COMPLETION: prefix to match training format
    input_text = prompt + "\n\nCOMPLETION:\n"

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", padding=False).to(device)
    input_ids = inputs["input_ids"]

    # Get tokens for categorization
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    token_categories = [categorize_hashhop_token(tok, tokenizer) for tok in tokens]

    # Hook to capture routing decisions
    routing_decisions = {}

    def capture_routing_hook(module, input, output, layer_idx):
        """Hook to capture routing decisions from each layer."""
        with torch.no_grad():
            router_input = input[0]  # hidden states after norm
            router_logits = module.router(router_input)
            routing_weights = torch.softmax(router_logits, dim=-1)

            # Store routing weights [batch, seq_len, 2]
            # Index 0 is MLP, index 1 is memory
            routing_decisions[layer_idx] = routing_weights.cpu().float().numpy()

    # Register hooks for each layer
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(
            lambda m, i, o, idx=layer_idx: capture_routing_hook(m, i, o, idx)
        )
        hooks.append(hook)

    # Generate completion
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Decode generated text
    generated_ids = outputs[0][input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Parse expected and generated answers
    expected_answers = parse_hashhop_completion(expected_completion)

    # Check accuracy - look for each expected answer in the generated text
    correct_answers = 0
    total_answers = len(expected_answers)

    for key, expected_value in expected_answers.items():
        # Check if the exact answer appears in generated text
        if f"{key} = '{expected_value}'" in generated_text:
            correct_answers += 1

    # Analyze routing statistics
    routing_stats = {
        "layer_stats": defaultdict(
            lambda: {"mlp_count": 0, "memory_count": 0, "total_count": 0}
        ),
        "category_stats": defaultdict(
            lambda: defaultdict(lambda: {"mlp_count": 0, "memory_count": 0})
        ),
        "position_stats": defaultdict(
            lambda: defaultdict(lambda: {"mlp_count": 0, "memory_count": 0})
        ),
        "routing_matrices": routing_decisions,
    }

    # Process routing decisions for the input tokens only (not generated)
    input_length = input_ids.shape[1]
    seq_len = len(tokens)

    for layer_idx, routing_weights in routing_decisions.items():
        routing_weights = routing_weights[0]  # Remove batch dimension

        # Only analyze input tokens
        for token_idx in range(min(input_length, len(routing_weights))):
            if token_idx >= len(tokens):
                break

            mlp_weight = routing_weights[token_idx, 0]
            memory_weight = routing_weights[token_idx, 1]
            uses_mlp = mlp_weight > memory_weight

            category = token_categories[token_idx]
            position_decile = min(int((token_idx / seq_len) * 10), 9)

            # Update statistics
            if uses_mlp:
                routing_stats["layer_stats"][layer_idx]["mlp_count"] += 1
                routing_stats["category_stats"][layer_idx][category]["mlp_count"] += 1
                routing_stats["position_stats"][layer_idx][position_decile][
                    "mlp_count"
                ] += 1
            else:
                routing_stats["layer_stats"][layer_idx]["memory_count"] += 1
                routing_stats["category_stats"][layer_idx][category][
                    "memory_count"
                ] += 1
                routing_stats["position_stats"][layer_idx][position_decile][
                    "memory_count"
                ] += 1

            routing_stats["layer_stats"][layer_idx]["total_count"] += 1

    result = {
        "correct": correct_answers,
        "total": total_answers,
        "accuracy": correct_answers / total_answers if total_answers > 0 else 0,
        "generated_text": generated_text,
        "expected_answers": expected_answers,
    }

    return result, routing_stats


def aggregate_routing_stats(all_routing_stats: List[Dict]) -> Dict:
    """Aggregate routing statistics across multiple samples."""
    aggregated = {
        "layer_stats": defaultdict(
            lambda: {"mlp_count": 0, "memory_count": 0, "total_count": 0}
        ),
        "category_stats": defaultdict(
            lambda: defaultdict(lambda: {"mlp_count": 0, "memory_count": 0})
        ),
        "position_stats": defaultdict(
            lambda: defaultdict(lambda: {"mlp_count": 0, "memory_count": 0})
        ),
    }

    for stats in all_routing_stats:
        # Aggregate layer stats
        for layer_idx, layer_stats in stats["layer_stats"].items():
            aggregated["layer_stats"][layer_idx]["mlp_count"] += layer_stats[
                "mlp_count"
            ]
            aggregated["layer_stats"][layer_idx]["memory_count"] += layer_stats[
                "memory_count"
            ]
            aggregated["layer_stats"][layer_idx]["total_count"] += layer_stats[
                "total_count"
            ]

        # Aggregate category stats
        for layer_idx, cat_stats in stats["category_stats"].items():
            for category, counts in cat_stats.items():
                aggregated["category_stats"][layer_idx][category]["mlp_count"] += (
                    counts["mlp_count"]
                )
                aggregated["category_stats"][layer_idx][category]["memory_count"] += (
                    counts["memory_count"]
                )

        # Aggregate position stats
        for layer_idx, pos_stats in stats["position_stats"].items():
            for position, counts in pos_stats.items():
                aggregated["position_stats"][layer_idx][position]["mlp_count"] += (
                    counts["mlp_count"]
                )
                aggregated["position_stats"][layer_idx][position]["memory_count"] += (
                    counts["memory_count"]
                )

    return aggregated


def plot_hashhop_routing_analysis(
    routing_stats: Dict, results: List[Dict], output_dir: Path
):
    """Create visualizations specific to hash-hop evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall layer routing percentages
    layers = sorted(routing_stats["layer_stats"].keys())
    mlp_percentages = []
    memory_percentages = []

    for layer in layers:
        stats = routing_stats["layer_stats"][layer]
        total = stats["total_count"]
        if total > 0:
            mlp_pct = (stats["mlp_count"] / total) * 100
            memory_pct = (stats["memory_count"] / total) * 100
        else:
            mlp_pct = memory_pct = 0
        mlp_percentages.append(mlp_pct)
        memory_percentages.append(memory_pct)

    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width / 2, mlp_percentages, width, label="MLP", color="skyblue")
    ax.bar(x + width / 2, memory_percentages, width, label="Memory", color="lightcoral")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Percentage of Tokens")
    ax.set_title("Hash-Hop Token Routing: MLP vs Memory by Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hashhop_layer_routing.png", dpi=300)
    plt.close()

    # 2. Token category analysis - focus on hash strings vs other tokens
    categories = set()
    for layer_data in routing_stats["category_stats"].values():
        categories.update(layer_data.keys())
    categories = sorted(list(categories))

    # Create heatmap data
    heatmap_data = []
    for layer in layers:
        row = []
        for category in categories:
            cat_stats = routing_stats["category_stats"][layer].get(
                category, {"mlp_count": 0, "memory_count": 0}
            )
            total = cat_stats["mlp_count"] + cat_stats["memory_count"]
            if total > 0:
                memory_pct = (cat_stats["memory_count"] / total) * 100
            else:
                memory_pct = 0
            row.append(memory_pct)
        heatmap_data.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=categories,
        yticklabels=[f"Layer {l}" for l in layers],
        cmap="RdBu_r",
        center=50,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "% Routed to Memory"},
    )

    plt.title("Hash-Hop Token Category Routing: % Routed to Memory")
    plt.xlabel("Token Category")
    plt.ylabel("Layer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "hashhop_category_routing.png", dpi=300)
    plt.close()

    # 3. Accuracy vs Routing Correlation
    # Calculate average memory usage per sample and correlate with accuracy
    sample_memory_usage = []
    sample_accuracies = []

    for i, result in enumerate(results):
        sample_accuracies.append(result["accuracy"])
        # This would need to be tracked per-sample in the main loop
        # For now, we'll skip this visualization

    # 4. Hash token specific analysis
    # Compare routing for hash strings vs other tokens
    hash_routing_by_layer = []
    other_routing_by_layer = []

    for layer in layers:
        hash_stats = routing_stats["category_stats"][layer].get(
            "hash_string", {"mlp_count": 0, "memory_count": 0}
        )
        hash_total = hash_stats["mlp_count"] + hash_stats["memory_count"]

        # Aggregate all non-hash categories
        other_mlp = 0
        other_memory = 0
        for category, stats in routing_stats["category_stats"][layer].items():
            if category != "hash_string":
                other_mlp += stats["mlp_count"]
                other_memory += stats["memory_count"]
        other_total = other_mlp + other_memory

        if hash_total > 0:
            hash_memory_pct = (hash_stats["memory_count"] / hash_total) * 100
        else:
            hash_memory_pct = 0

        if other_total > 0:
            other_memory_pct = (other_memory / other_total) * 100
        else:
            other_memory_pct = 0

        hash_routing_by_layer.append(hash_memory_pct)
        other_routing_by_layer.append(other_memory_pct)

    # Plot hash vs non-hash routing
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(layers))
    width = 0.35

    ax.bar(
        x - width / 2,
        hash_routing_by_layer,
        width,
        label="Hash Strings",
        color="darkred",
    )
    ax.bar(
        x + width / 2,
        other_routing_by_layer,
        width,
        label="Other Tokens",
        color="darkblue",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("% Routed to Memory")
    ax.set_title("Memory Routing: Hash Strings vs Other Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hashhop_hash_vs_other_routing.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze routing on Hash-Hop evaluation dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to HuggingFace model"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hashhop_eval_340m/medium_2hop.json",
        help="Path to hash-hop dataset JSON",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hashhop_routing_results",
        help="Directory for outputs",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--use_wandb", action="store_true", help="Log results to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="hashhop-routing", help="W&B project name"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    # Initialize W&B if requested
    if args.use_wandb:
        run_name = args.wandb_run_name or f"hashhop_{Path(args.dataset_path).stem}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.device
    )

    # Load hash-hop dataset
    print(f"\nLoading hash-hop dataset from {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    # Limit samples if requested
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = dataset[: args.num_samples]

    print(f"Evaluating {len(dataset)} hash-hop samples...")

    # Evaluate samples
    all_results = []
    all_routing_stats = []

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        result, routing_stats = evaluate_hashhop_sample(
            model, tokenizer, sample, args.device
        )
        all_results.append(result)
        all_routing_stats.append(routing_stats)

        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            accuracies = [r["accuracy"] for r in all_results]
            avg_accuracy = np.mean(accuracies)
            print(
                f"  Progress: {i + 1}/{len(dataset)}, Avg accuracy: {avg_accuracy:.2%}"
            )

    # Aggregate statistics
    aggregated_routing = aggregate_routing_stats(all_routing_stats)

    # Calculate overall metrics
    total_correct = sum(r["correct"] for r in all_results)
    total_queries = sum(r["total"] for r in all_results)
    overall_accuracy = total_correct / total_queries if total_queries > 0 else 0

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_summary = {
        "config": vars(args),
        "overall_metrics": {
            "accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_queries": total_queries,
            "num_samples": len(dataset),
        },
        "per_sample_results": [
            {
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
            }
            for r in all_results
        ],
        "routing_statistics": {
            "layer_stats": dict(aggregated_routing["layer_stats"]),
            "category_stats": {
                str(k): dict(v) for k, v in aggregated_routing["category_stats"].items()
            },
            "position_stats": {
                str(k): {str(p): dict(s) for p, s in v.items()}
                for k, v in aggregated_routing["position_stats"].items()
            },
        },
    }

    # Save JSON results
    with open(output_dir / "hashhop_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_hashhop_routing_analysis(aggregated_routing, all_results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("HASH-HOP EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset: {Path(args.dataset_path).name}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Overall accuracy: {overall_accuracy:.2%} ({total_correct}/{total_queries})")

    # Accuracy distribution
    accuracies = [r["accuracy"] for r in all_results]
    print("\nAccuracy distribution:")
    print(f"  Min: {min(accuracies):.2%}")
    print(f"  Max: {max(accuracies):.2%}")
    print(f"  Mean: {np.mean(accuracies):.2%}")
    print(f"  Std: {np.std(accuracies):.2%}")

    # Perfect accuracy count
    perfect_samples = sum(1 for r in all_results if r["accuracy"] == 1.0)
    print(f"  Perfect accuracy: {perfect_samples}/{len(dataset)} samples")

    # Routing summary
    print("\nRouting Summary:")
    total_tokens = sum(
        s["total_count"] for s in aggregated_routing["layer_stats"].values()
    )
    total_mlp = sum(s["mlp_count"] for s in aggregated_routing["layer_stats"].values())
    total_memory = sum(
        s["memory_count"] for s in aggregated_routing["layer_stats"].values()
    )

    if total_tokens > 0:
        print(f"  Overall MLP usage: {(total_mlp / total_tokens) * 100:.2f}%")
        print(f"  Overall Memory usage: {(total_memory / total_tokens) * 100:.2f}%")

    # Hash token routing
    print("\nHash String Routing (averaged across layers):")
    hash_mlp_total = 0
    hash_memory_total = 0

    for layer_cats in aggregated_routing["category_stats"].values():
        if "hash_string" in layer_cats:
            hash_mlp_total += layer_cats["hash_string"]["mlp_count"]
            hash_memory_total += layer_cats["hash_string"]["memory_count"]

    hash_total = hash_mlp_total + hash_memory_total
    if hash_total > 0:
        print(
            f"  Hash strings to Memory: {(hash_memory_total / hash_total) * 100:.2f}%"
        )
        print(f"  Total hash tokens: {hash_total}")

    print(f"\nResults saved to: {output_dir}")

    # Log to W&B if enabled
    if args.use_wandb:
        wandb.log(
            {
                "overall_accuracy": overall_accuracy,
                "total_correct": total_correct,
                "total_queries": total_queries,
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "perfect_samples": perfect_samples,
                "mlp_percentage": (total_mlp / total_tokens) * 100
                if total_tokens > 0
                else 0,
                "memory_percentage": (total_memory / total_tokens) * 100
                if total_tokens > 0
                else 0,
                "hash_memory_percentage": (hash_memory_total / hash_total) * 100
                if hash_total > 0
                else 0,
            }
        )

        # Log visualizations
        for img_path in output_dir.glob("*.png"):
            wandb.log({img_path.stem: wandb.Image(str(img_path))})

        # Save results file
        wandb.save(str(output_dir / "hashhop_results.json"))

        wandb.finish()


if __name__ == "__main__":
    main()
