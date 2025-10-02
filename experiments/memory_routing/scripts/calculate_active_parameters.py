#!/usr/bin/env python3
"""
Calculate active parameters for memory-based transformer models.
Accounts for sparse memory retrieval and grouped query attention.
"""

import json
import argparse
from typing import Dict, Any


def calculate_active_parameters(config: Dict[str, Any]) -> Dict[str, int]:
    """Calculate active parameters for the memory model."""

    # Extract config values
    hidden_size = config["hidden_size"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    num_layers = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]
    tie_word_embeddings = config["tie_word_embeddings"]

    # Memory-specific parameters
    memory_layers = config.get("memory_layers", [])
    mem_n_keys = config.get("mem_n_keys", 0)
    mem_heads = config.get("mem_heads", 1)
    mem_knn = config.get("mem_knn", 32)
    mem_k_dim = config.get("mem_k_dim", 512)
    mem_v_dim = config.get("mem_v_dim", -1)
    mem_share_values = config.get("mem_share_values", True)
    swilu_projection = config.get("swilu_projection", True)

    # For memory_mlp_router models, all layers have memory if mem_n_keys > 0
    model_type = config.get("model_type", "")
    if model_type == "memory_mlp_router" and mem_n_keys > 0 and not memory_layers:
        memory_layers = list(range(num_layers))

    # Handle mem_v_dim = -1 (uses hidden_size)
    if mem_v_dim == -1:
        mem_v_dim = hidden_size

    # Calculate head dimension
    head_dim = hidden_size // num_heads

    results = {}

    # =============================================================================
    # NON-MEMORY PARAMETERS (Always Active)
    # =============================================================================

    # Token embeddings
    input_embeddings = vocab_size * hidden_size
    output_embeddings = 0 if tie_word_embeddings else vocab_size * hidden_size
    total_embeddings = input_embeddings + output_embeddings

    # Attention parameters per layer (GQA)
    q_proj_per_layer = hidden_size * (num_heads * head_dim)  # Q projection
    k_proj_per_layer = hidden_size * (num_kv_heads * head_dim)  # K projection
    v_proj_per_layer = hidden_size * (num_kv_heads * head_dim)  # V projection
    o_proj_per_layer = (num_heads * head_dim) * hidden_size  # Output projection

    # QK norm parameters if enabled
    qk_norm_per_layer = 0
    if config.get("qk_norm", False):
        qk_norm_per_layer = 2 * hidden_size  # q_norm + k_norm (full hidden_size each)

    attention_per_layer = (
        q_proj_per_layer
        + k_proj_per_layer
        + v_proj_per_layer
        + o_proj_per_layer
        + qk_norm_per_layer
    )
    total_attention = attention_per_layer * num_layers

    # MLP parameters (for transformer models)
    mlp_params = 0
    intermediate_size = config.get("intermediate_size", 0)
    if intermediate_size > 0:
        # Up projection: hidden_size -> intermediate_size
        # Down projection: intermediate_size -> hidden_size
        mlp_per_layer = 2 * (hidden_size * intermediate_size)
        mlp_params = mlp_per_layer * num_layers

    # Layer normalization parameters
    # Pre-attention norm + pre-MLP norm per layer + final norm
    layer_norm_per_layer = 2 * hidden_size
    final_norm = hidden_size
    total_layer_norm = (layer_norm_per_layer * num_layers) + final_norm

    # Non-memory parameters
    non_memory_params = (
        total_embeddings + total_attention + total_layer_norm + mlp_params
    )

    # =============================================================================
    # MEMORY PARAMETERS (Sparse/Active)
    # =============================================================================

    memory_params = 0

    if memory_layers:
        num_memory_layers = len(memory_layers)

        # Active memory keys per layer
        # Each head retrieves mem_knn keys independently
        active_keys_per_layer = mem_heads * mem_knn * mem_k_dim
        total_active_keys = active_keys_per_layer * num_memory_layers

        # Active memory values
        # Each head retrieves mem_knn values independently
        active_values_per_head = mem_knn * mem_v_dim

        if mem_share_values:
            # Values are shared across layers, but each head still retrieves independently
            total_active_values = mem_heads * active_values_per_head
        else:
            # Values are not shared, so multiply by number of memory layers
            total_active_values = mem_heads * active_values_per_head * num_memory_layers

        # Query MLP parameters (always active)
        # Input to output: hidden_size -> mem_heads * mem_k_dim
        query_mlp_per_layer = hidden_size * (mem_heads * mem_k_dim)
        total_query_mlp = query_mlp_per_layer * num_memory_layers

        # Value projection parameters (always active)
        value_proj_per_layer = 0
        if swilu_projection:
            # Value projection: mem_v_dim -> hidden_size
            # SwiLU projection: hidden_size -> mem_v_dim
            value_proj_per_layer = (mem_v_dim * hidden_size) + (hidden_size * mem_v_dim)
        else:
            # Just value projection: mem_v_dim -> hidden_size
            value_proj_per_layer = mem_v_dim * hidden_size

        total_value_proj = value_proj_per_layer * num_memory_layers

        # Load balancing biases (always active, but negligible)
        bias_params_per_layer = 2 * mem_n_keys  # memory_biases_1 + memory_biases_2
        total_bias_params = bias_params_per_layer * num_memory_layers

        # Total active memory parameters
        memory_params = (
            total_active_keys
            + total_active_values
            + total_query_mlp
            + total_value_proj
            + total_bias_params
        )

    # =============================================================================
    # RESULTS
    # =============================================================================

    results = {
        "non_memory_parameters": non_memory_params,
        "active_memory_parameters": memory_params,
        "total_active_parameters": non_memory_params + memory_params,
        # Detailed breakdown
        "embeddings": total_embeddings,
        "attention": total_attention,
        "layer_norm": total_layer_norm,
        "mlp": mlp_params,
        "active_memory_keys": total_active_keys if memory_layers else 0,
        "active_memory_values": total_active_values if memory_layers else 0,
        "query_mlp": total_query_mlp if memory_layers else 0,
        "value_projections": total_value_proj if memory_layers else 0,
        "bias_parameters": total_bias_params if memory_layers else 0,
        # Configuration summary
        "config_summary": {
            "model_type": config.get("model_type", "unknown"),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "memory_layers": memory_layers,
            "mem_heads": mem_heads,
            "mem_knn": mem_knn,
            "mem_share_values": mem_share_values,
        },
    }

    return results


def format_number(n: int) -> str:
    """Format number with commas and abbreviated form."""
    if n >= 1e9:
        return f"{n:,} ({n / 1e9:.2f}B)"
    elif n >= 1e6:
        return f"{n:,} ({n / 1e6:.2f}M)"
    elif n >= 1e3:
        return f"{n:,} ({n / 1e3:.2f}K)"
    else:
        return f"{n:,}"


def print_results(results: Dict[str, int]):
    """Print formatted results."""
    print("=" * 80)
    print("ACTIVE PARAMETER CALCULATION RESULTS")
    print("=" * 80)

    config = results["config_summary"]
    print(f"Model Type: {config['model_type']}")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Layers: {config['num_layers']}")
    print(f"Query Heads: {config['num_heads']}")
    print(f"KV Heads: {config['num_kv_heads']}")
    print(f"Memory Layers: {config['memory_layers']}")
    print(f"Memory Heads: {config['mem_heads']}")
    print(f"Memory KNN: {config['mem_knn']}")
    print(f"Memory Share Values: {config['mem_share_values']}")
    print()

    print("ACTIVE PARAMETERS:")
    print("-" * 40)
    print(
        f"Non-Memory Parameters:      {format_number(results['non_memory_parameters'])}"
    )
    print(
        f"Active Memory Parameters:   {format_number(results['active_memory_parameters'])}"
    )
    print(
        f"Total Active Parameters:    {format_number(results['total_active_parameters'])}"
    )
    print()

    if results["active_memory_parameters"] > 0:
        memory_percentage = (
            results["active_memory_parameters"]
            / results["total_active_parameters"]
            * 100
        )
        print(f"Memory Parameter Ratio:     {memory_percentage:.1f}%")
        print()

    print("DETAILED BREAKDOWN:")
    print("-" * 40)
    print(f"Token Embeddings:           {format_number(results['embeddings'])}")
    print(f"Attention Weights:          {format_number(results['attention'])}")
    print(f"Layer Normalization:        {format_number(results['layer_norm'])}")
    if results["mlp"] > 0:
        print(f"MLP:                        {format_number(results['mlp'])}")

    if results["active_memory_parameters"] > 0:
        print(
            f"Active Memory Keys:         {format_number(results['active_memory_keys'])}"
        )
        print(
            f"Active Memory Values:       {format_number(results['active_memory_values'])}"
        )
        print(f"Query MLP:                  {format_number(results['query_mlp'])}")
        print(
            f"Value Projections:          {format_number(results['value_projections'])}"
        )
        print(
            f"Bias Parameters:            {format_number(results['bias_parameters'])}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate active parameters for memory models"
    )
    parser.add_argument("config_path", help="Path to model config JSON file")
    parser.add_argument("--output", "-o", help="Output file for results (optional)")

    args = parser.parse_args()

    # Load config
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Calculate parameters
    results = calculate_active_parameters(config)

    # Print results
    print_results(results)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
