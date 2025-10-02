#!/usr/bin/env python3
import argparse
from safetensors import safe_open
import re
import numpy as np


def inspect_safetensors(file_path):
    print(f"Inspecting safetensors file: {file_path}")

    # Open the safetensors file
    with safe_open(file_path, framework="pt") as f:
        # Get metadata if available
        metadata = f.metadata()
        if metadata:
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

        # Get all keys
        keys = f.keys()
        print(f"\nFound {len(keys)} tensors in file")

        # Look specifically for memory counts and biases
        mom_keys = [k for k in keys if "memory_counts" in k or "memory_biases" in k]

        if mom_keys:
            print("\n=== MomGatedDeltaNet Memory Buffers ===")
            for key in mom_keys:
                tensor = f.get_tensor(key)
                print(f"\n{key}")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Values: {tensor}")
                print(f"  Sum: {tensor.sum().item()}")
                print(f"  Mean: {tensor.mean().item()}")
                print(f"  Any non-zero: {(tensor != 0).any().item()}")

                # Extract layer index
                match = re.search(r"layers\.(\d+)", key)
                if match:
                    layer_idx = match.group(1)
                    print(f"  Layer: {layer_idx}")
        else:
            print("\nNo memory_counts or memory_biases buffers found!")

        # Print some statistics about other tensors
        total_params = 0
        print("\n=== Tensor Statistics ===")
        for key in keys:
            if key in mom_keys:
                continue  # Skip memory buffers, already printed

            tensor = f.get_tensor(key)
            total_params += np.prod(tensor.shape)

            # Print info for small tensors or on request
            if np.prod(tensor.shape) < 100 or args.verbose:
                print(f"\n{key}")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                if np.prod(tensor.shape) < 10:
                    print(f"  Values: {tensor}")

        print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SafeTensors file contents")
    parser.add_argument("file_path", help="Path to the safetensors file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information for all tensors",
    )
    args = parser.parse_args()

    inspect_safetensors(args.file_path)
