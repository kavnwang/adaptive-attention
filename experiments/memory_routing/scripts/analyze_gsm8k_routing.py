#!/usr/bin/env python3
"""
Analyze token routing patterns between MLP and memory layers on GSM8K dataset.

This script evaluates a trained model with MLP/memory routing to understand:
1. Overall routing statistics per layer (% MLP vs % memory)
2. Token-level routing patterns based on different types of content
3. Visualization of routing decisions
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import re
import wandb
from typing import Dict, List
import sys
import os

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "3rdparty/bento"))


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


def categorize_token(token: str, tokenizer) -> str:
    """Categorize a token into different types for analysis."""
    # Decode if token is an ID
    if isinstance(token, int):
        token = tokenizer.decode([token])

    # Remove leading space if present
    clean_token = token.lstrip("Ġ").lstrip("▁").lstrip()

    # Check for different token categories
    if not clean_token or clean_token.isspace():
        return "whitespace"
    elif clean_token.isdigit():
        return "digit"
    elif clean_token in ["+", "-", "*", "/", "=", "<", ">", "%"]:
        return "math_operator"
    elif clean_token in ["(", ")", "[", "]", "{", "}"]:
        return "bracket"
    elif clean_token in [".", ",", "!", "?", ":", ";", '"', "'", "$"]:
        return "punctuation"
    elif clean_token[0].isupper() and len(clean_token) > 1:
        return "capitalized"
    elif clean_token.isupper():
        return "all_caps"
    elif re.match(r"^[a-zA-Z]+$", clean_token):
        # Common function words
        if clean_token.lower() in [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "must",
            "can",
            "if",
            "then",
            "else",
            "when",
            "where",
            "what",
            "who",
            "which",
            "that",
        ]:
            return "function_word"
        else:
            return "content_word"
    else:
        return "mixed/other"


def analyze_routing_with_hooks(
    model, tokenizer, texts: List[str], device: str = "cuda"
) -> Dict:
    """
    Analyze routing decisions by running forward passes and collecting routing statistics.
    """
    all_results = {
        "layer_stats": defaultdict(
            lambda: {"mlp_count": 0, "memory_count": 0, "total_count": 0}
        ),
        "token_category_stats": defaultdict(
            lambda: defaultdict(lambda: {"mlp_count": 0, "memory_count": 0})
        ),
        "per_token_routing": [],
        "layer_routing_matrices": defaultdict(
            list
        ),  # Store routing decisions per layer
        "position_stats": defaultdict(
            lambda: defaultdict(
                lambda: {"mlp_count": 0, "memory_count": 0, "total_count": 0}
            )
        ),  # Stats by position decile
    }

    # Reset model routing statistics
    model.reset_all_routing_stats()

    for text in tqdm(texts, desc="Analyzing routing"):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(device)
        input_ids = inputs["input_ids"]

        # Get tokens for categorization
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        token_categories = [categorize_token(tok, tokenizer) for tok in tokens]

        # Hook to capture routing decisions
        routing_decisions = {}

        def capture_routing_hook(module, input, output, layer_idx):
            """Hook to capture routing decisions from each layer."""
            # The router output is at module.router
            with torch.no_grad():
                router_input = input[0]  # hidden states after norm
                router_logits = module.router(router_input)
                routing_weights = torch.softmax(router_logits, dim=-1)

                # Store routing weights [batch, seq_len, 2]
                # Index 0 is MLP, index 1 is memory
                # Convert from bfloat16 to float32 before numpy conversion
                routing_decisions[layer_idx] = routing_weights.cpu().float().numpy()

        # Register hooks for each layer
        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            hook = layer.register_forward_hook(
                lambda m, i, o, idx=layer_idx: capture_routing_hook(m, i, o, idx)
            )
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            pass  # outputs = model(input_ids=input_ids, use_cache=False)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process routing decisions
        for layer_idx, routing_weights in routing_decisions.items():
            # routing_weights shape: [1, seq_len, 2]
            routing_weights = routing_weights[0]  # Remove batch dimension

            # Store routing matrix for visualization
            all_results["layer_routing_matrices"][layer_idx].append(routing_weights)

            # Calculate sequence length and position deciles
            seq_len = len(tokens)

            for token_idx, (token, category) in enumerate(
                zip(tokens, token_categories)
            ):
                mlp_weight = routing_weights[token_idx, 0]
                memory_weight = routing_weights[token_idx, 1]

                # Binary decision based on which weight is higher
                uses_mlp = mlp_weight > memory_weight

                # Calculate position decile (0-9 for tenths of the sequence)
                position_decile = min(int((token_idx / seq_len) * 10), 9)

                # Update layer statistics
                if uses_mlp:
                    all_results["layer_stats"][layer_idx]["mlp_count"] += 1
                else:
                    all_results["layer_stats"][layer_idx]["memory_count"] += 1
                all_results["layer_stats"][layer_idx]["total_count"] += 1

                # Update token category statistics
                if uses_mlp:
                    all_results["token_category_stats"][layer_idx][category][
                        "mlp_count"
                    ] += 1
                else:
                    all_results["token_category_stats"][layer_idx][category][
                        "memory_count"
                    ] += 1

                # Update position statistics
                if uses_mlp:
                    all_results["position_stats"][layer_idx][position_decile][
                        "mlp_count"
                    ] += 1
                else:
                    all_results["position_stats"][layer_idx][position_decile][
                        "memory_count"
                    ] += 1
                all_results["position_stats"][layer_idx][position_decile][
                    "total_count"
                ] += 1

                # Store per-token routing info
                all_results["per_token_routing"].append(
                    {
                        "layer": layer_idx,
                        "token": token,
                        "category": category,
                        "position_decile": position_decile,
                        "mlp_weight": float(mlp_weight),
                        "memory_weight": float(memory_weight),
                        "uses_mlp": uses_mlp,
                        "text_idx": len(
                            all_results["layer_routing_matrices"][layer_idx]
                        )
                        - 1,
                        "token_idx": token_idx,
                    }
                )

    # Also get runtime statistics from the model
    runtime_stats = model.get_mlp_memory_routing_stats()
    all_results["runtime_stats"] = runtime_stats

    return all_results


def plot_routing_statistics(results: Dict, output_dir: Path):
    """Create visualizations of routing statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall layer routing percentages
    layers = sorted(results["layer_stats"].keys())
    mlp_percentages = []
    memory_percentages = []

    for layer in layers:
        stats = results["layer_stats"][layer]
        total = stats["total_count"]
        if total > 0:
            mlp_pct = (stats["mlp_count"] / total) * 100
            memory_pct = (stats["memory_count"] / total) * 100
        else:
            mlp_pct = memory_pct = 0
        mlp_percentages.append(mlp_pct)
        memory_percentages.append(memory_pct)

    # Bar plot of routing percentages by layer
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width / 2, mlp_percentages, width, label="MLP", color="skyblue")
    ax.bar(x + width / 2, memory_percentages, width, label="Memory", color="lightcoral")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Percentage of Tokens")
    ax.set_title("Token Routing Distribution: MLP vs Memory by Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "layer_routing_percentages.png", dpi=300)
    plt.close()

    # 1.5. Position-based routing analysis
    # Create a heatmap showing routing preferences by position in sequence
    position_data = []
    position_labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    for layer in layers:
        row = []
        for decile in range(10):
            if decile in results["position_stats"][layer]:
                stats = results["position_stats"][layer][decile]
                total = stats["total_count"]
                if total > 0:
                    memory_pct = (stats["memory_count"] / total) * 100
                else:
                    memory_pct = 0
            else:
                memory_pct = 0
            row.append(memory_pct)
        position_data.append(row)

    # Create position heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        position_data,
        xticklabels=position_labels,
        yticklabels=[f"Layer {layer}" for layer in layers],
        cmap="RdBu_r",
        center=50,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "% Routed to Memory"},
    )

    plt.title("Token Routing by Position in Sequence: % Routed to Memory")
    plt.xlabel("Position in Sequence (Deciles)")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(output_dir / "position_routing_heatmap.png", dpi=300)
    plt.close()

    # Also create a line plot showing average routing by position across all layers
    avg_memory_by_position = []
    for decile in range(10):
        total_memory = 0
        total_count = 0
        for layer in layers:
            if decile in results["position_stats"][layer]:
                stats = results["position_stats"][layer][decile]
                total_memory += stats["memory_count"]
                total_count += stats["total_count"]

        if total_count > 0:
            avg_memory_pct = (total_memory / total_count) * 100
        else:
            avg_memory_pct = 0
        avg_memory_by_position.append(avg_memory_pct)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(10)
    ax.plot(x, avg_memory_by_position, "o-", linewidth=2, markersize=8, color="darkred")
    ax.fill_between(x, avg_memory_by_position, alpha=0.3, color="lightcoral")

    ax.set_xlabel("Position in Sequence (Deciles)")
    ax.set_ylabel("Average % Routed to Memory")
    ax.set_title("Average Memory Routing by Token Position Across All Layers")
    ax.set_xticks(x)
    ax.set_xticklabels(position_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Add value labels on points
    for i, val in enumerate(avg_memory_by_position):
        ax.annotate(
            f"{val:.1f}%",
            (i, val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "position_routing_average.png", dpi=300)
    plt.close()

    # 2. Token category routing heatmap
    # Prepare data for heatmap
    categories = set()
    for layer_data in results["token_category_stats"].values():
        categories.update(layer_data.keys())
    categories = sorted(list(categories))

    # Create matrix: rows are layers, columns are categories
    # Values are % of category tokens routed to memory
    heatmap_data = []
    for layer in layers:
        row = []
        for category in categories:
            cat_stats = results["token_category_stats"][layer].get(
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
        yticklabels=[f"Layer {layer}" for layer in layers],
        cmap="RdBu_r",
        center=50,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "% Routed to Memory"},
    )

    plt.title("Token Category Routing Patterns: % Routed to Memory")
    plt.xlabel("Token Category")
    plt.ylabel("Layer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "category_routing_heatmap.png", dpi=300)
    plt.close()

    # 3. Example routing visualization for a few sequences
    # Show routing patterns for first 3 texts
    num_examples = min(3, len(results["layer_routing_matrices"][0]))

    for example_idx in range(num_examples):
        fig, axes = plt.subplots(
            len(layers), 1, figsize=(15, 2 * len(layers)), sharex=True
        )
        if len(layers) == 1:
            axes = [axes]

        for layer_idx, ax in enumerate(axes):
            if example_idx < len(results["layer_routing_matrices"][layer_idx]):
                routing_matrix = results["layer_routing_matrices"][layer_idx][
                    example_idx
                ]
                # Show memory routing weights
                memory_weights = routing_matrix[:, 1]

                ax.imshow(
                    memory_weights.reshape(1, -1),
                    cmap="RdBu_r",
                    aspect="auto",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                ax.set_ylabel(f"Layer {layer_idx}")
                ax.set_yticks([])

                if layer_idx == 0:
                    ax.set_title(
                        f"Example {example_idx + 1}: Memory Routing Weights by Token Position"
                    )

                if layer_idx == len(layers) - 1:
                    ax.set_xlabel("Token Position")

        plt.colorbar(axes[-1].images[0], ax=axes, label="Memory Weight", shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_dir / f"example_{example_idx}_routing_pattern.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token routing in MLP vs Memory model on GSM8K"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to HuggingFace model"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of GSM8K samples to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="routing_analysis_results",
        help="Directory for outputs",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--use_wandb", action="store_true", help="Log results to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="routing-analysis", help="W&B project name"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    # Initialize W&B if requested
    if args.use_wandb:
        run_name = (
            args.wandb_run_name or f"routing_analysis_{Path(args.model_path).name}"
        )
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.device
    )

    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    gsm8k = load_dataset("openai/gsm8k", "main")
    test_ds = gsm8k["test"]

    # Sample subset for analysis
    total_size = len(test_ds)
    if args.num_samples < total_size:
        indices = np.random.choice(total_size, size=args.num_samples, replace=False)
        selected_samples = [test_ds[int(i)] for i in indices]
    else:
        selected_samples = test_ds

    print(f"Analyzing routing for {len(selected_samples)} GSM8K samples...")

    # Prepare texts for analysis - using the questions
    texts = [sample["question"] for sample in selected_samples]

    # Analyze routing
    results = analyze_routing_with_hooks(model, tokenizer, texts, args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_for_json = {
        "config": vars(args),
        "layer_stats": dict(results["layer_stats"]),
        "token_category_stats": {
            str(k): dict(v) for k, v in results["token_category_stats"].items()
        },
        "position_stats": {
            str(layer): {str(pos): dict(stats) for pos, stats in pos_data.items()}
            for layer, pos_data in results["position_stats"].items()
        },
        "runtime_stats": results["runtime_stats"],
        "summary_stats": {},
    }

    # Calculate summary statistics
    total_tokens = sum(
        stats["total_count"] for stats in results["layer_stats"].values()
    )
    total_mlp = sum(stats["mlp_count"] for stats in results["layer_stats"].values())
    total_memory = sum(
        stats["memory_count"] for stats in results["layer_stats"].values()
    )

    if total_tokens > 0:
        overall_mlp_pct = (total_mlp / total_tokens) * 100
        overall_memory_pct = (total_memory / total_tokens) * 100
    else:
        overall_mlp_pct = overall_memory_pct = 0

    results_for_json["summary_stats"] = {
        "total_tokens_analyzed": total_tokens,
        "overall_mlp_percentage": overall_mlp_pct,
        "overall_memory_percentage": overall_memory_pct,
        "num_layers": len(results["layer_stats"]),
        "num_samples": len(texts),
    }

    # Per-layer summary
    layer_summaries = []
    for layer_idx in sorted(results["layer_stats"].keys()):
        stats = results["layer_stats"][layer_idx]
        if stats["total_count"] > 0:
            mlp_pct = (stats["mlp_count"] / stats["total_count"]) * 100
            memory_pct = (stats["memory_count"] / stats["total_count"]) * 100
        else:
            mlp_pct = memory_pct = 0

        layer_summaries.append(
            {
                "layer": layer_idx,
                "mlp_percentage": mlp_pct,
                "memory_percentage": memory_pct,
                "total_tokens": stats["total_count"],
            }
        )

    results_for_json["layer_summaries"] = layer_summaries

    # Save JSON results
    with open(output_dir / "routing_analysis_results.json", "w") as f:
        json.dump(results_for_json, f, indent=2)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_routing_statistics(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("ROUTING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total tokens analyzed: {total_tokens}")
    print(f"Overall MLP usage: {overall_mlp_pct:.2f}%")
    print(f"Overall Memory usage: {overall_memory_pct:.2f}%")
    print("\nPer-layer breakdown:")
    for summary in layer_summaries:
        print(
            f"  Layer {summary['layer']}: {summary['mlp_percentage']:.2f}% MLP, "
            f"{summary['memory_percentage']:.2f}% Memory ({summary['total_tokens']} tokens)"
        )

    # Token category analysis
    print("\nToken category preferences (averaged across layers):")
    category_preferences = defaultdict(lambda: {"mlp_total": 0, "memory_total": 0})

    for layer_data in results["token_category_stats"].values():
        for category, stats in layer_data.items():
            category_preferences[category]["mlp_total"] += stats["mlp_count"]
            category_preferences[category]["memory_total"] += stats["memory_count"]

    for category in sorted(category_preferences.keys()):
        prefs = category_preferences[category]
        total = prefs["mlp_total"] + prefs["memory_total"]
        if total > 0:
            memory_pct = (prefs["memory_total"] / total) * 100
            print(f"  {category}: {memory_pct:.2f}% prefer Memory ({total} tokens)")

    # Position analysis summary
    print("\nPosition-based routing (averaged across layers):")
    position_labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]
    for decile in range(10):
        total_memory = 0
        total_count = 0
        for layer_data in results["position_stats"].values():
            if decile in layer_data:
                stats = layer_data[decile]
                total_memory += stats["memory_count"]
                total_count += stats["total_count"]

        if total_count > 0:
            memory_pct = (total_memory / total_count) * 100
            print(
                f"  {position_labels[decile]}: {memory_pct:.2f}% to Memory ({total_count} tokens)"
            )

    print(f"\nResults saved to: {output_dir}")

    # Log to W&B if enabled
    if args.use_wandb:
        # Log summary metrics
        wandb.log(
            {
                "overall_mlp_percentage": overall_mlp_pct,
                "overall_memory_percentage": overall_memory_pct,
                "total_tokens_analyzed": total_tokens,
                "num_samples": len(texts),
            }
        )

        # Log per-layer metrics
        for summary in layer_summaries:
            wandb.log(
                {
                    f"layer_{summary['layer']}_mlp_pct": summary["mlp_percentage"],
                    f"layer_{summary['layer']}_memory_pct": summary[
                        "memory_percentage"
                    ],
                }
            )

        # Log visualizations
        for img_path in output_dir.glob("*.png"):
            wandb.log({img_path.stem: wandb.Image(str(img_path))})

        # Log raw results file
        wandb.save(str(output_dir / "routing_analysis_results.json"))

        wandb.finish()


if __name__ == "__main__":
    main()
