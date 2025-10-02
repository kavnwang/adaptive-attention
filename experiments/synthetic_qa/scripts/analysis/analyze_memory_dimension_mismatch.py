#!/usr/bin/env python3
"""
Analyze memory layer dimension mismatches between source and target models.
This helps understand why parameter transfer is failing.
"""

import json
import torch
from pathlib import Path


def analyze_checkpoint_dimensions(checkpoint_path, step=10000):
    """Load and analyze dimensions from a checkpoint."""
    # Try to load the checkpoint metadata
    ckpt_dir = Path(checkpoint_path) / "checkpoint" / f"step-{step}"

    if not ckpt_dir.exists():
        print(f"Checkpoint not found at {ckpt_dir}")
        return None

    # Load metadata to understand tensor shapes
    metadata_path = ckpt_dir / ".metadata"
    if metadata_path.exists():
        print(f"\nAnalyzing checkpoint: {checkpoint_path}")
        print(f"Step: {step}")

        # Load a sample checkpoint file to inspect shapes
        ckpt_files = list(ckpt_dir.glob("__*__.dist_cp"))
        if ckpt_files:
            # Load first checkpoint shard
            state_dict = torch.load(ckpt_files[0], map_location="cpu")

            print("\nMemory layer parameters found:")
            memory_params = {}
            for key, tensor in state_dict.items():
                if "memory" in key and "mlp" in key:
                    shape = tensor.shape if hasattr(tensor, "shape") else "unknown"
                    memory_params[key] = shape
                    print(f"  {key}: {shape}")

            return memory_params

    return None


def compare_configs(config1_path, config2_path):
    """Compare two model configurations."""
    with open(config1_path, "r") as f:
        config1 = json.load(f)

    with open(config2_path, "r") as f:
        config2 = json.load(f)

    print("\nComparing configurations:")
    print(f"Config 1: {config1_path}")
    print(f"Config 2: {config2_path}")

    # Key parameters that affect memory layer dimensions
    key_params = [
        "hidden_size",
        "num_heads",
        "num_kv_heads",
        "mem_n_keys",
        "mem_heads",
        "mem_knn",
        "mem_k_dim",
        "mem_v_dim",
        "hidden_ratio",
        "intermediate_size",
    ]

    print("\nKey parameter differences:")
    for param in key_params:
        val1 = config1.get(param, "not set")
        val2 = config2.get(param, "not set")
        if val1 != val2:
            print(f"  {param}: {val1} -> {val2}")

    # Calculate expected dimensions
    print("\nExpected memory layer dimensions:")

    # For config 1
    mem_heads1 = config1.get("mem_heads", 1)
    mem_k_dim1 = config1.get("mem_k_dim", 512)
    mem_n_keys1 = config1.get("mem_n_keys", 256)
    mem_v_dim1 = config1.get("mem_v_dim", -1)
    hidden_size1 = config1.get("hidden_size", 1024)
    num_heads1 = config1.get("num_heads", 8)

    if mem_v_dim1 == -1:
        mem_v_dim1 = hidden_size1

    print("\nConfig 1 dimensions:")
    print(f"  Query projection: {hidden_size1} -> {mem_heads1 * mem_k_dim1}")
    print(f"  Memory keys: ({mem_n_keys1}, {mem_k_dim1})")
    print(
        f"  Memory values: ({mem_n_keys1}^2, {mem_v_dim1}) = ({mem_n_keys1**2}, {mem_v_dim1})"
    )

    # For config 2
    mem_heads2 = config2.get("mem_heads", 1)
    mem_k_dim2 = config2.get("mem_k_dim", 512)
    mem_n_keys2 = config2.get("mem_n_keys", 256)
    mem_v_dim2 = config2.get("mem_v_dim", -1)
    hidden_size2 = config2.get("hidden_size", 1024)
    num_heads2 = config2.get("num_heads", 16)

    if mem_v_dim2 == -1:
        mem_v_dim2 = hidden_size2

    print("\nConfig 2 dimensions:")
    print(f"  Query projection: {hidden_size2} -> {mem_heads2 * mem_k_dim2}")
    print(f"  Memory keys: ({mem_n_keys2}, {mem_k_dim2})")
    print(
        f"  Memory values: ({mem_n_keys2}^2, {mem_v_dim2}) = ({mem_n_keys2**2}, {mem_v_dim2})"
    )


def suggest_fixes():
    """Suggest potential fixes for the dimension mismatch."""
    print("\n" + "=" * 50)
    print("SUGGESTED FIXES:")
    print("=" * 50)

    print("\n1. Modify the target config to match source dimensions:")
    print("   - Change num_heads from 16 to 8 in memory_1layer_synthetic.json")

    print("\n2. Use a more flexible parameter mapping:")
    print("   - Only transfer the memory biases (which are already working)")
    print("   - Skip the mismatched parameters")

    print("\n3. Create an adapter script to resize parameters:")
    print("   - Truncate or pad the source parameters to match target dimensions")
    print(
        "   - This requires careful consideration of how to handle the size differences"
    )

    print("\n4. Train from scratch with the new architecture:")
    print(
        "   - If the dimension differences are intentional, starting fresh may be best"
    )


def main():
    # Paths
    source_checkpoint = "exp/memory_1layer_synthetic_qa_2K"
    source_config = "exp/memory_1layer_synthetic_qa_2K/model_config.json"
    target_config = "llmonade/configs/memory/memory_1layer_synthetic.json"

    # Analyze checkpoint dimensions
    checkpoint_dims = analyze_checkpoint_dimensions(source_checkpoint)

    # Compare configurations
    if Path(source_config).exists() and Path(target_config).exists():
        compare_configs(source_config, target_config)

    # Provide suggestions
    suggest_fixes()


if __name__ == "__main__":
    main()
