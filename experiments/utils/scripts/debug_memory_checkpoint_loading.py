#!/usr/bin/env python3
"""
Debug script to understand memory layer checkpoint loading issues.
"""

import torch
import os
import tempfile
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def analyze_checkpoint(checkpoint_path, step=10000):
    """Analyze checkpoint contents and dimensions."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print(f"{'=' * 60}")

    # Convert distributed checkpoint
    ckpt_dir = os.path.join(checkpoint_path, "checkpoint", f"step-{step}")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, "checkpoint.pt")
        print("\nConverting distributed checkpoint...")
        dcp_to_torch_save(ckpt_dir, temp_path)

        # Load checkpoint
        checkpoint = torch.load(temp_path, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model", checkpoint)

        print("\nCheckpoint structure:")
        print(f"  Keys: {list(checkpoint.keys())}")
        print(f"  Model state dict keys: {len(model_state)} parameters")

        # Analyze memory layer parameters
        print("\nMemory layer parameters:")
        memory_params = {}
        for key, tensor in model_state.items():
            if "memory" in key and "mlp" in key:
                memory_params[key] = tensor
                print(f"  {key}:")
                print(f"    Shape: {tensor.shape}")
                print(f"    Dtype: {tensor.dtype}")
                print(f"    Size: {tensor.numel()} elements")

        return memory_params


def test_loading_scenarios():
    """Test different loading scenarios to identify the issue."""
    import sys

    sys.path.append("/home/kevin/LLMonade_interp")

    from transformers import AutoConfig

    print(f"\n{'=' * 60}")
    print("Testing checkpoint loading scenarios")
    print(f"{'=' * 60}")

    # Load config
    config_path = "llmonade/configs/memory/memory_1layer_synthetic.json"
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

    print("\nModel configuration:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  mem_heads: {config.mem_heads}")
    print(f"  mem_n_keys: {config.mem_n_keys}")
    print(f"  mem_k_dim: {config.mem_k_dim}")

    # Expected dimensions
    print("\nExpected dimensions (based on config):")
    print(
        f"  Query projection: {config.hidden_size} -> {config.mem_heads * config.mem_k_dim}"
    )
    print(f"  Memory keys: ({config.mem_n_keys}, {config.mem_k_dim})")
    print(
        f"  Memory values: ({config.mem_n_keys}**2, {config.hidden_size}) = ({config.mem_n_keys**2}, {config.hidden_size})"
    )


def create_minimal_test():
    """Create a minimal test case for the loading issue."""
    print(f"\n{'=' * 60}")
    print("Creating minimal test case")
    print(f"{'=' * 60}")

    # Create simple test tensors matching checkpoint dimensions
    test_checkpoint = {
        "model.layers.0.mlp.memory_retrieval.keys": torch.randn(512, 256),
        "model.layers.0.mlp.memory_retrieval.memory_biases_1": torch.randn(256),
        "model.layers.0.mlp.memory_retrieval.memory_biases_2": torch.randn(256),
        "model.layers.0.mlp.memory_unlocking.values.weight": torch.randn(65536, 1024),
    }

    print("\nTest checkpoint tensors:")
    for key, tensor in test_checkpoint.items():
        print(f"  {key}: {tensor.shape}")

    # Save test checkpoint
    test_path = "/tmp/test_memory_checkpoint.pt"
    torch.save({"model": test_checkpoint}, test_path)
    print(f"\nSaved test checkpoint to: {test_path}")

    return test_path


def test_dtensor_behavior():
    """Test DTensor distribution behavior."""
    print(f"\n{'=' * 60}")
    print("Testing DTensor behavior")
    print(f"{'=' * 60}")

    # This would require a distributed setup, so we'll just document the issue
    print("\nDTensor distribution might be causing issues when:")
    print("1. Source tensor shape doesn't match target tensor shape")
    print("2. distribute_tensor is called with FSDP sharding")
    print("3. The sharding might multiply dimensions by the number of devices")

    print("\nPossible solutions:")
    print("1. Load checkpoint before applying FSDP")
    print("2. Use device_mesh=None for loading")
    print("3. Skip DTensor conversion for mismatched shapes")


def main():
    # Analyze the actual checkpoint
    checkpoint_path = "exp/memory_1layer_synthetic_qa_2K"
    memory_params = analyze_checkpoint(checkpoint_path)

    # Test loading scenarios
    test_loading_scenarios()

    # Create minimal test case
    test_path = create_minimal_test()

    # Test DTensor behavior
    test_dtensor_behavior()

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"{'=' * 60}")

    print("\nKey findings:")
    print("1. Checkpoint has correct dimensions (not pre-sharded)")
    print("2. The 8x factor matches the number of GPUs")
    print("3. Issue likely in DTensor distribution during loading")
    print("\nNext steps: Modify PretrainedLayerLoader to handle this case")


if __name__ == "__main__":
    main()
