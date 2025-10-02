#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to inspect layers in a checkpoint for planning layer transfers.

Usage:
    python inspect_checkpoint_layers.py \
        --checkpoint_path exp/model/checkpoint/step-30000 \
        --model_config llmonade/configs/memory/memory_mlp_router_340M.json \
        --output layer_info.txt
"""

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import timedelta
from io import BytesIO

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

# Import to register custom models
try:
    import bento
    import bento.models.memory
    import bento.models.memory_mlp_router
except ImportError:
    pass

from transformers import AutoConfig


def get_layer_info(state_dict):
    """Extract layer information from state dict."""
    layer_info = defaultdict(dict)
    component_info = defaultdict(list)

    for key, tensor in state_dict.items():
        # Extract layer number
        layer_match = re.search(r"model\.layers\.(\d+)\.", key)
        if layer_match:
            layer_num = int(layer_match.group(1))

            # Extract component name
            component = key.split(f"model.layers.{layer_num}.")[-1]
            param_name = component.split(".")[-1]  # weight, bias, etc.
            component_base = ".".join(component.split(".")[:-1])

            # Store shape info
            if component_base not in layer_info[layer_num]:
                layer_info[layer_num][component_base] = {}
            layer_info[layer_num][component_base][param_name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }

            # Track unique components
            if component not in component_info[layer_num]:
                component_info[layer_num].append(component)

    return layer_info, component_info


def load_checkpoint(checkpoint_path):
    """Load checkpoint from distributed or regular format."""
    if os.path.isdir(checkpoint_path) and os.path.exists(
        os.path.join(checkpoint_path, ".metadata")
    ):
        # Distributed checkpoint
        print(f"Loading distributed checkpoint from {checkpoint_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "checkpoint.pt")
            dcp_to_torch_save(checkpoint_path, temp_path)

            # Add safe globals for loading
            torch.serialization.add_safe_globals([timedelta, BytesIO])
            checkpoint = torch.load(temp_path, map_location="cpu")
            return checkpoint.get("model", checkpoint)
    else:
        # Regular checkpoint
        print(f"Loading regular checkpoint from {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu")


def format_param_count(count):
    """Format parameter count with units."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    else:
        return str(count)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect layers in a checkpoint for planning layer transfers"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint (distributed or regular format)",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to model config JSON (optional, for additional context)",
    )
    parser.add_argument(
        "--output", type=str, help="Output file for layer information (optional)"
    )
    parser.add_argument(
        "--layer", type=int, help="Show detailed info for specific layer number"
    )

    args = parser.parse_args()

    # Load checkpoint
    state_dict = load_checkpoint(args.checkpoint_path)

    # Load model config if provided
    model_config = None
    if args.model_config:
        try:
            # Try loading with AutoConfig (works if model type is registered)
            model_config = AutoConfig.from_pretrained(
                args.model_config, trust_remote_code=True
            )
        except Exception as e:
            # Fall back to loading as raw JSON for unregistered model types
            print(f"Note: Loading config as JSON due to: {e}")
            with open(args.model_config, "r") as f:
                config_dict = json.load(f)

                # Create a simple namespace object to access config attributes
                class SimpleConfig:
                    def __init__(self, config_dict):
                        self.__dict__.update(config_dict)

                model_config = SimpleConfig(config_dict)

    # Extract layer information
    layer_info, component_info = get_layer_info(state_dict)

    # Prepare output
    output_lines = []
    output_lines.append(f"Checkpoint: {args.checkpoint_path}")
    if model_config:
        output_lines.append(f"Model Config: {args.model_config}")
        output_lines.append(f"Model Type: {model_config.model_type}")
        output_lines.append(f"Hidden Size: {model_config.hidden_size}")
        output_lines.append(f"Num Layers: {model_config.num_hidden_layers}")
    output_lines.append("")

    # Summary
    output_lines.append(f"Total layers found: {len(layer_info)}")
    output_lines.append(f"Layer indices: {sorted(layer_info.keys())}")
    output_lines.append("")

    # Layer details
    if args.layer is not None:
        # Detailed view for specific layer
        if args.layer in layer_info:
            output_lines.append(f"=== Layer {args.layer} Details ===")
            for component, params in sorted(layer_info[args.layer].items()):
                output_lines.append(f"\n{component}:")
                for param_name, info in sorted(params.items()):
                    output_lines.append(
                        f"  {param_name}: shape={info['shape']}, "
                        f"dtype={info['dtype']}, "
                        f"params={format_param_count(info['numel'])}"
                    )
        else:
            output_lines.append(f"Layer {args.layer} not found in checkpoint")
    else:
        # Overview of all layers
        output_lines.append("=== Layer Overview ===")
        for layer_num in sorted(layer_info.keys()):
            total_params = sum(
                param_info["numel"]
                for component_params in layer_info[layer_num].values()
                for param_info in component_params.values()
            )

            # Check for special components
            special_components = []
            for component in layer_info[layer_num].keys():
                if "memory" in component:
                    special_components.append("memory")
                elif "attn" in component:
                    special_components.append("attention")
                elif "mlp" in component:
                    special_components.append("mlp")

            special_str = (
                f" ({', '.join(set(special_components))})" if special_components else ""
            )

            output_lines.append(
                f"Layer {layer_num:3d}: {format_param_count(total_params):>8} params{special_str}"
            )

    # Special components section
    output_lines.append("\n=== Special Components ===")
    memory_components = defaultdict(list)
    embeddings = {}
    head_components = {}

    for key, tensor in state_dict.items():
        if "memory" in key:
            layer_match = re.search(r"model\.layers\.(\d+)\.", key)
            if layer_match:
                layer_num = int(layer_match.group(1))
                component = key.split(f"model.layers.{layer_num}.")[-1]
                memory_components[layer_num].append(
                    f"{component}: shape={list(tensor.shape)}"
                )
        elif "embed" in key:
            embeddings[key] = f"shape={list(tensor.shape)}"
        elif "lm_head" in key or "head" in key:
            head_components[key] = f"shape={list(tensor.shape)}"

    if memory_components:
        output_lines.append("\nMemory components:")
        for layer_num in sorted(memory_components.keys()):
            output_lines.append(f"  Layer {layer_num}:")
            for comp in memory_components[layer_num]:
                output_lines.append(f"    {comp}")

    if embeddings:
        output_lines.append("\nEmbedding layers:")
        for key, info in embeddings.items():
            output_lines.append(f"  {key}: {info}")

    if head_components:
        output_lines.append("\nHead components:")
        for key, info in head_components.items():
            output_lines.append(f"  {key}: {info}")

    # Example mapping suggestions
    output_lines.append("\n=== Example Layer Mapping Commands ===")
    if layer_info:
        max_layer = max(layer_info.keys())
        output_lines.append("# Copy all layers:")
        output_lines.append(
            f'--checkpoint.pretrained_layer_map \'{{"0-{max_layer}": "0-{max_layer}"}}\''
        )

        if max_layer >= 11:
            output_lines.append("\n# Copy first 12 layers:")
            output_lines.append(
                '--checkpoint.pretrained_layer_map \'{"0-11": "0-11"}\''
            )

        if memory_components:
            output_lines.append("\n# Copy memory components with pattern:")
            output_lines.append(
                '--checkpoint.pretrained_layer_map \'{"*.memory.*": "*.memory.*"}\''
            )

    # Output results
    output_text = "\n".join(output_lines)
    print(output_text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"\nLayer information saved to: {args.output}")


if __name__ == "__main__":
    main()
