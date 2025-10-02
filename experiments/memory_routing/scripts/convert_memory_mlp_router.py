#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified from llmonade/scripts/helpers/convert_dcp_to_hf.py
# Specifically handles memory_mlp_router architecture

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
from torchtitan.tools.logging import init_logger, logger


def fix_memory_router_state_dict(state_dict, model_config):
    """
    Fix state dict keys for memory_mlp_router architecture.

    When mem_share_values=True, only the first layer with memory should have values,
    but distributed checkpointing saves values for all memory layers.

    Args:
        state_dict: The loaded state dictionary
        model_config: The model configuration

    Returns:
        Modified state dictionary with conflicting memory values removed
    """
    # Check if this is a memory_mlp_router model
    if (
        hasattr(model_config, "model_type")
        and model_config.model_type == "memory_mlp_router"
    ):
        logger.info("Detected memory_mlp_router model type")

        # Check if mem_share_values is True
        if hasattr(model_config, "mem_share_values") and model_config.mem_share_values:
            logger.info("Model has mem_share_values=True, fixing memory values...")

            # Find all layers that have memory.values.weight
            memory_value_layers = []
            for key in state_dict.keys():
                if ".memory.values.weight" in key:
                    # Extract layer number from key like "model.layers.12.memory.values.weight"
                    try:
                        parts = key.split(".")
                        if (
                            len(parts) >= 3
                            and parts[0] == "model"
                            and parts[1] == "layers"
                        ):
                            layer_num = int(parts[2])
                            memory_value_layers.append(layer_num)
                    except (ValueError, IndexError):
                        continue

            if memory_value_layers:
                memory_value_layers.sort()
                first_layer = memory_value_layers[0]
                logger.info(f"Found memory value layers: {memory_value_layers}")
                logger.info(f"Keeping values only for first layer: {first_layer}")

                # Remove values.weight from all non-first layers
                keys_to_remove = []
                for layer_num in memory_value_layers[1:]:  # Skip first layer
                    key = f"model.layers.{layer_num}.memory.values.weight"
                    if key in state_dict:
                        keys_to_remove.append(key)
                        logger.info(f"Removing duplicate memory values: {key}")

                for key in keys_to_remove:
                    del state_dict[key]

    # Also apply the original fix for backward compatibility
    if hasattr(model_config, "memory_layers") and hasattr(
        model_config, "mem_share_values"
    ):
        if model_config.mem_share_values and model_config.memory_layers:
            # Find the first memory layer (the "original" that will have values)
            first_memory_layer = min(model_config.memory_layers)

            # Remove values from non-original memory layers (for .mlp.values. pattern)
            keys_to_remove = []
            for key in state_dict.keys():
                if key.startswith("model.layers.") and ".mlp.values." in key:
                    try:
                        layer_num = int(key.split(".")[2])
                        if (
                            layer_num in model_config.memory_layers
                            and layer_num != first_memory_layer
                        ):
                            keys_to_remove.append(key)
                            logger.info(
                                f"Removing conflicting memory values key: {key}"
                            )
                    except (ValueError, IndexError):
                        continue

            for key in keys_to_remove:
                del state_dict[key]

    return state_dict


@torch.inference_mode()
def save_pretrained(path: str, step: int, config: str, tokenizer: str):
    if not config:
        config = os.path.join(path, "model_config.json")

    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Saving the config to {path}")
    config.save_pretrained(path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

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
        logger.info(f"Using latest checkpoint step: {step}")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f"checkpoint/step-{step}")
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        logger.info(f"Converting distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

        # Debug: Check memory-related keys
        logger.info("Memory-related keys in state dict:")
        memory_keys = [k for k in state_dict.keys() if "memory" in k]
        for key in sorted(memory_keys):
            if "values.weight" in key or "memory_biases" in key:
                logger.info(f"  {key}: shape={state_dict[key].shape}")

        # Fix memory router state dict before any other processing
        state_dict = fix_memory_router_state_dict(state_dict, config)

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
                    logger.info(f"Found shared reference: {key}")
                    # Convert shared_k.weight -> shared_k_proj.0.weight
                    prefix = "shared_k" if "shared_k." in key else "shared_v"
                    new_key = key.replace(f"{prefix}.weight", f"{prefix}_proj.0.weight")

                    # If both keys exist, use the _proj version and remove the reference
                    if new_key in state_dict:
                        logger.info(f"Both {key} and {new_key} exist, removing {key}")
                        state_dict.pop(key)
                    else:
                        # Rename the key
                        logger.info(f"Remapping {key} to {new_key}")
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
                        logger.info(f"Creating reference from {key} to {new_key}")
                        state_dict[new_key] = state_dict[key].clone()

        # Try to load the state dict
        logger.info("Loading state dict into model...")
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("State dict loaded successfully!")
        except RuntimeError as e:
            logger.error(f"Error loading state dict with strict=True: {e}")
            logger.info("Attempting to load with strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
                # Only fail if there are unexpected keys after our fixes
                raise RuntimeError(
                    f"Still have unexpected keys after fixes: {unexpected_keys}"
                )

        logger.info(f"Saving the model to {path}")
        model.save_pretrained(path, safe_serialization=False)
        logger.info("Conversion complete!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        "Convert memory_mlp_router DCP format model weights to huggingface-style."
    )
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--tokenizer", type=str, default="fla-hub/transformer-1.3B-100B"
    )
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer)
