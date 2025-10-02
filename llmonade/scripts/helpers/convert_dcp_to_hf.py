# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import bento  # noqa

# Explicitly import memory_mlp_router to ensure it's registered
try:
    import bento.models.memory_mlp_router  # noqa
except ImportError:
    pass
from torchtitan.tools.logging import init_logger, logger


def fix_memory_sharing_state_dict(state_dict, model_config):
    """
    Fix state dict keys for memory sharing when converting checkpoints.

    When mem_share_values_blocks=1, only the first memory layer should have values,
    but distributed checkpointing saves values for all memory layers.

    Args:
        state_dict: The loaded state dictionary
        model_config: The model configuration

    Returns:
        Modified state dictionary with conflicting memory values removed
    """
    # Only apply fix if this is a memory model with shared values
    if not hasattr(model_config, "mem_share_values_blocks"):
        # Check for old parameter name for backward compatibility
        if hasattr(model_config, "mem_share_values"):
            # Convert old parameter to new one
            model_config.mem_share_values_blocks = (
                1 if model_config.mem_share_values else 0
            )
        else:
            return state_dict

    if model_config.mem_share_values_blocks != 1:
        return state_dict

    # For memory router models, all layers have memory but share values
    if (
        hasattr(model_config, "model_type")
        and model_config.model_type == "memory_mlp_router"
    ):
        # Remove all memory.values except layer 0 (or 1 if that's what's in the checkpoint)
        keys_to_remove = []
        has_layer_0 = any(
            "model.layers.0.memory.values." in key for key in state_dict.keys()
        )
        first_layer = 0 if has_layer_0 else 1

        for key in state_dict.keys():
            if ".memory.values." in key and key.startswith("model.layers."):
                try:
                    layer_num = int(key.split(".")[2])
                    if layer_num != first_layer:
                        keys_to_remove.append(key)
                        logger.info(f"Removing conflicting memory values key: {key}")
                except (ValueError, IndexError):
                    continue

        for key in keys_to_remove:
            del state_dict[key]

        return state_dict

    # Original logic for non-router models
    if not hasattr(model_config, "memory_layers") or not model_config.memory_layers:
        return state_dict

    # Find the first memory layer (the "original" that will have values)
    first_memory_layer = min(model_config.memory_layers)

    # Remove values from non-original memory layers
    keys_to_remove = []
    for key in state_dict.keys():
        # Handle both memory.values and mlp.values patterns
        if key.startswith("model.layers.") and (
            ".memory.values." in key or ".mlp.values." in key
        ):
            # Extract layer number from key like "model.layers.12.memory.values.weight" or "model.layers.12.mlp.values.weight"
            try:
                layer_num = int(key.split(".")[2])
                if (
                    layer_num in model_config.memory_layers
                    and layer_num != first_memory_layer
                ):
                    keys_to_remove.append(key)
                    logger.info(f"Removing conflicting memory values key: {key}")
            except (ValueError, IndexError):
                # Skip if we can't parse the layer number
                continue

    for key in keys_to_remove:
        del state_dict[key]

    return state_dict


@torch.inference_mode()
def save_pretrained(path: str, step: int, config: str, tokenizer: str, output: str = None):
    if not config:
        config = os.path.join(path, "model_config.json")

    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    if step == -1:
        # List files in the path/checkpoint folder and get the latest step
        checkpoint_dir = os.path.join(path, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
        steps = [
            int(f.split("-")[-1])
            for f in os.listdir(checkpoint_dir)
            if f.startswith("step-")
        ]
        step = max(steps)
    
    # Determine output path
    if output is None:
        output = os.path.join(path, f"hf_model_step{step}")
        logger.info(f"No output path specified, using default: {output}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)
    
    logger.info(f"Saving the config to {output}")
    config.save_pretrained(output)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {output}")
    tokenizer.save_pretrained(output)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f"checkpoint/step-{step}")
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

        # Check memory biases
        for key, value in state_dict.items():
            if "memory" in key:
                print(key)
                print(value)

        # Fix memory sharing state dict before any other processing
        state_dict = fix_memory_sharing_state_dict(state_dict, config)

        # Get model configuration to determine if we need to handle LoRA references
        uses_lora = False
        if hasattr(config, "kv_proj_strategy") and isinstance(
            config.kv_proj_strategy, str
        ):
            uses_lora = config.kv_proj_strategy.startswith("lora:")
            logger.info(f"Model uses LoRA strategy: {uses_lora}")

        if uses_lora:
            # MONKEY PATCH!
            for key in list(state_dict.keys()):
                # Handle shared_k and shared_v weights which are identical to shared_k_proj.0 and shared_v_proj.0
                if key.endswith(".weight") and (
                    "shared_k." in key or "shared_v." in key
                ):
                    print(f"Found shared reference: {key}")
                    # Convert shared_k.weight -> shared_k_proj.0.weight
                    prefix = "shared_k" if "shared_k." in key else "shared_v"
                    new_key = key.replace(f"{prefix}.weight", f"{prefix}_proj.0.weight")

                    # If both keys exist, use the _proj version and remove the reference
                    if new_key in state_dict:
                        print(f"Both {key} and {new_key} exist, removing {key}")
                        state_dict.pop(key)
                    else:
                        # Rename the key
                        print(f"Remapping {key} to {new_key}")
                        state_dict[new_key] = state_dict.pop(key)

            # Now add the reverse mapping for each layer - copy from _proj.0 -> shared_
            for key in list(state_dict.keys()):
                if key.endswith(".weight") and (
                    "shared_k_proj.0" in key or "shared_v_proj.0" in key
                ):
                    # Get the base name (model.layers.X.attn.shared_k_proj.0) -> (model.layers.X.attn.shared_k)
                    prefix = "shared_k" if "shared_k_proj" in key else "shared_v"
                    new_key = key.replace(f"{prefix}_proj.0.weight", f"{prefix}.weight")

                    # Create duplicate weights for the shared_k and shared_v parameters
                    if new_key not in state_dict:
                        print(f"Creating reference from {key} to {new_key}")
                        state_dict[new_key] = state_dict[key].clone()

        model.load_state_dict(state_dict)

        logger.info(f"Saving the model to {output}")
        model.save_pretrained(output, safe_serialization=False)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        "Convert DCP format model weights to huggingface-style."
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the experiment directory containing checkpoints")
    parser.add_argument("--step", type=int, default=-1, help="Checkpoint step to convert (default: latest)")
    parser.add_argument("--config", type=str, default=None, help="Path to model config (default: {path}/model_config.json)")
    parser.add_argument(
        "--tokenizer", type=str, default="fla-hub/transformer-1.3B-100B",
        help="Tokenizer to use for the converted model"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory for HuggingFace model (default: {path}/hf_model_step{step})")
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer, args.output)
