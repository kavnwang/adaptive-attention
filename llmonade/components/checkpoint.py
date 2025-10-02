# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.tools.logging import logger


@dataclass
class TrainState(Stateful):
    step: int = 0
    skipped_step: int = 0
    token: int = 0
    elapsed: timedelta = timedelta(0)
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)
    # Two-phase training state
    training_phase: str = "main"  # "synthetic" or "main"
    phase_step: int = 0  # Step counter within current phase
    synthetic_steps_completed: int = 0  # Total synthetic steps completed
    use_custom_synthetic_optimizer: bool = False  # Track optimizer toggle setting

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "skipped_step": torch.tensor(self.skipped_step, dtype=torch.int32),
            "token": torch.tensor(self.token, dtype=torch.int64),
            "elapsed": self.elapsed,
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
            "training_phase": self.training_phase,
            "phase_step": torch.tensor(self.phase_step, dtype=torch.int32),
            "synthetic_steps_completed": torch.tensor(
                self.synthetic_steps_completed, dtype=torch.int32
            ),
            "use_custom_synthetic_optimizer": torch.tensor(
                self.use_custom_synthetic_optimizer, dtype=torch.bool
            ),
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        self.skipped_step = state_dict.get("skipped_step", 0).item()
        self.token = state_dict["token"].item()
        self.elapsed = state_dict["elapsed"]
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)
        # Load two-phase training state with backward compatibility
        self.training_phase = state_dict.get("training_phase", "main")
        self.phase_step = state_dict.get("phase_step", torch.tensor(0)).item()
        self.synthetic_steps_completed = state_dict.get(
            "synthetic_steps_completed", torch.tensor(0)
        ).item()
        self.use_custom_synthetic_optimizer = state_dict.get(
            "use_custom_synthetic_optimizer", torch.tensor(False)
        ).item()


class PretrainedLayerLoader:
    """
    Loads specific layers from a pretrained checkpoint and applies them to a target model.

    Supports:
    - Layer index mapping (e.g., "0-5": "0-5")
    - Component path mapping (e.g., "model.layers.0.memory.values": "model.layers.0.memory.values")
    - Pattern matching with wildcards (e.g., "*.memory.keys": "*.memory.keys")
    """

    def __init__(
        self,
        source_path: str,
        layer_mapping: Optional[str],
        step: int,
        device_mesh: Optional[Any] = None,
        parallel_dims: Optional[Any] = None,
    ):
        self.source_path = source_path
        self.step = step
        self.device_mesh = device_mesh
        self.parallel_dims = parallel_dims
        self.layer_mapping = self._parse_mapping(layer_mapping)

    def _parse_mapping(self, mapping_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse JSON string or load from file."""
        if mapping_str is None:
            return None

        try:
            if mapping_str.startswith("{"):
                # Inline JSON
                return json.loads(mapping_str)
            else:
                # File path
                with open(mapping_str, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to parse layer mapping: {e}")

    def _expand_range(self, range_str: str) -> List[int]:
        """Expand range string like '0-5' to [0, 1, 2, 3, 4, 5]."""
        if "-" in range_str:
            start, end = map(int, range_str.split("-"))
            return list(range(start, end + 1))
        else:
            return [int(range_str)]

    def _get_local_shard(
        self,
        global_tensor: torch.Tensor,
        local_shape: torch.Size,
        device_mesh: Any,
        placements: Any,
    ) -> torch.Tensor:
        """
        Extract the local shard from a global tensor based on device mesh and placements.

        Args:
            global_tensor: The full global tensor
            local_shape: Expected shape of the local shard
            device_mesh: DTensor device mesh
            placements: DTensor placements

        Returns:
            The local shard for the current rank
        """
        # Get current rank in the device mesh
        rank = device_mesh.get_local_rank()
        mesh_size = device_mesh.size()

        # For now, we only handle Shard placement on the first dimension
        if len(placements) == 1 and hasattr(placements[0], "dim"):
            shard_dim = placements[0].dim

            # Calculate the shard size and offset
            global_size = global_tensor.shape[shard_dim]
            local_size = local_shape[shard_dim]

            # Verify the sharding makes sense
            expected_local_size = global_size // mesh_size
            if local_size != expected_local_size:
                logger.warning(
                    f"Unexpected local size: {local_size} vs {expected_local_size}"
                )

            # Calculate start and end indices for this rank's shard
            start_idx = rank * local_size
            end_idx = start_idx + local_size

            # Create the slice
            indices = [slice(None)] * len(global_tensor.shape)
            indices[shard_dim] = slice(start_idx, end_idx)

            local_shard = global_tensor[tuple(indices)]
            logger.info(f"Extracted local shard for rank {rank}: {local_shard.shape}")

            return local_shard
        else:
            # For other placement types, we can't easily shard
            raise NotImplementedError(
                f"Sharding not implemented for placements: {placements}"
            )

    def _expand_layer_mappings(self, mappings: Dict[str, str]) -> Dict[str, str]:
        """Expand range-based mappings to individual layer mappings."""
        expanded = {}

        for src, tgt in mappings.items():
            src_layers = self._expand_range(src)
            tgt_layers = self._expand_range(tgt)

            if len(src_layers) != len(tgt_layers):
                raise ValueError(
                    f"Source range {src} has {len(src_layers)} layers but "
                    f"target range {tgt} has {len(tgt_layers)} layers"
                )

            for s, t in zip(src_layers, tgt_layers):
                expanded[str(s)] = str(t)

        return expanded

    def _get_checkpoint_path(self, base_path: str, step: int) -> str:
        """Find the checkpoint directory for the given step."""
        if step == -1:
            # Find latest checkpoint
            checkpoint_dir = (
                os.path.join(base_path, "checkpoint")
                if "checkpoint" not in base_path
                else base_path
            )
            if not os.path.exists(checkpoint_dir):
                raise ValueError(
                    f"Checkpoint directory {checkpoint_dir} does not exist"
                )

            pattern = r"step-(\d+)"
            steps = []
            for dirname in os.listdir(checkpoint_dir):
                match = re.search(pattern, dirname)
                if match and os.path.isdir(os.path.join(checkpoint_dir, dirname)):
                    steps.append(int(match.group(1)))

            if not steps:
                raise ValueError(f"No checkpoints found in {checkpoint_dir}")

            step = max(steps)

        checkpoint_path = os.path.join(base_path, "checkpoint", f"step-{step}")
        if "checkpoint" in base_path:
            # Handle case where base_path already includes "checkpoint"
            checkpoint_path = os.path.join(base_path, f"step-{step}")

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")

        logger.info(f"Loading from checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _load_checkpoint_state_dict(
        self, checkpoint_path: str
    ) -> Dict[str, torch.Tensor]:
        """Load the state dict from a distributed checkpoint."""
        # Check if this is a distributed checkpoint
        if os.path.isdir(checkpoint_path) and os.path.exists(
            os.path.join(checkpoint_path, ".metadata")
        ):
            # Convert distributed checkpoint to single file for easier handling
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = os.path.join(tmpdir, "checkpoint.pt")
                logger.info("Converting distributed checkpoint to single file...")
                dcp_to_torch_save(checkpoint_path, temp_path)

                # Load with BytesIO support for compatibility
                torch.serialization.add_safe_globals([timedelta, BytesIO])
                checkpoint = torch.load(temp_path, map_location="cpu")
                return checkpoint.get("model", checkpoint)
        else:
            # Regular checkpoint file
            return torch.load(checkpoint_path, map_location="cpu")

    def _apply_pattern_mapping(
        self, source_key: str, pattern: str, target_pattern: str
    ) -> Optional[str]:
        """Apply wildcard pattern mapping to generate target key."""
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace("*", "(.*)")
        regex_pattern = f"^{regex_pattern}$"

        match = re.match(regex_pattern, source_key)
        if match:
            # Replace wildcards in target pattern with captured groups
            target_key = target_pattern
            for i, group in enumerate(match.groups(), 1):
                target_key = target_key.replace("*", group, 1)
            return target_key
        return None

    def _map_state_dict_keys(
        self,
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Map source state dict keys to target model keys based on mapping configuration."""
        if not self.layer_mapping:
            return {}

        mapped_state_dict = {}

        # Handle simple layer index mappings (e.g., {"0": "1", "2-5": "2-5"})
        if all(k.replace("-", "").isdigit() for k in self.layer_mapping.keys()):
            # This is a simple layer index mapping
            expanded_mappings = self._expand_layer_mappings(self.layer_mapping)

            for src_key, src_tensor in source_state_dict.items():
                # Extract layer number from key like "model.layers.0.attn.q_proj.weight"
                layer_match = re.search(r"model\.layers\.(\d+)\.", src_key)
                if layer_match:
                    src_layer = layer_match.group(1)
                    if src_layer in expanded_mappings:
                        tgt_layer = expanded_mappings[src_layer]
                        tgt_key = src_key.replace(
                            f"layers.{src_layer}.", f"layers.{tgt_layer}."
                        )

                        # Check if target key exists and shapes match
                        if tgt_key in target_state_dict:
                            if src_tensor.shape == target_state_dict[tgt_key].shape:
                                mapped_state_dict[tgt_key] = src_tensor
                            else:
                                logger.warning(
                                    f"Shape mismatch for {src_key} -> {tgt_key}: "
                                    f"{src_tensor.shape} vs {target_state_dict[tgt_key].shape}"
                                )
        else:
            # Complex mapping with component paths and patterns
            layer_mappings = self.layer_mapping.get("layer_mappings", {})
            component_mappings = self.layer_mapping.get("component_mappings", {})

            # First handle layer mappings
            if layer_mappings:
                expanded_layer_mappings = self._expand_layer_mappings(layer_mappings)
                for src_key, src_tensor in source_state_dict.items():
                    layer_match = re.search(r"model\.layers\.(\d+)\.", src_key)
                    if layer_match:
                        src_layer = layer_match.group(1)
                        if src_layer in expanded_layer_mappings:
                            tgt_layer = expanded_layer_mappings[src_layer]
                            tgt_key = src_key.replace(
                                f"layers.{src_layer}.", f"layers.{tgt_layer}."
                            )

                            if tgt_key in target_state_dict:
                                if src_tensor.shape == target_state_dict[tgt_key].shape:
                                    mapped_state_dict[tgt_key] = src_tensor

            # Then handle component mappings (can override layer mappings)
            for src_pattern, tgt_pattern in component_mappings.items():
                if "*" in src_pattern:
                    # Pattern matching
                    for src_key, src_tensor in source_state_dict.items():
                        tgt_key = self._apply_pattern_mapping(
                            src_key, src_pattern, tgt_pattern
                        )
                        if tgt_key and tgt_key in target_state_dict:
                            if src_tensor.shape == target_state_dict[tgt_key].shape:
                                mapped_state_dict[tgt_key] = src_tensor
                else:
                    # Direct mapping
                    if src_pattern in source_state_dict:
                        src_tensor = source_state_dict[src_pattern]
                        if tgt_pattern in target_state_dict:
                            if src_tensor.shape == target_state_dict[tgt_pattern].shape:
                                mapped_state_dict[tgt_pattern] = src_tensor

        return mapped_state_dict

    def load_and_apply(self, model: torch.nn.Module) -> List[str]:
        """
        Load specified layers from checkpoint and apply to model.

        Returns:
            List of parameter keys that were successfully transferred.
        """
        # Get checkpoint path
        checkpoint_path = self._get_checkpoint_path(self.source_path, self.step)

        # Load source checkpoint
        logger.info("Loading source checkpoint...")
        source_state_dict = self._load_checkpoint_state_dict(checkpoint_path)

        # Get target model state dict
        target_state_dict = model.state_dict()

        # Map keys and filter based on configuration
        logger.info("Mapping layer keys...")
        mapped_state_dict = self._map_state_dict_keys(
            source_state_dict, target_state_dict
        )

        if not mapped_state_dict:
            raise ValueError("No layers were mapped from source to target model")

        # Apply mapped weights to model with DTensor handling
        logger.info(f"Applying {len(mapped_state_dict)} mapped parameters to model...")
        successfully_loaded = []
        failed_params = []

        with torch.no_grad():
            for param_name, src_tensor in mapped_state_dict.items():
                try:
                    # Navigate to the actual parameter using the parameter name path
                    param = model
                    attrs = param_name.split(".")
                    for attr in attrs[:-1]:
                        param = getattr(param, attr)

                    # Get the final parameter
                    final_attr = attrs[-1]
                    if hasattr(param, final_attr):
                        target_param = getattr(param, final_attr)

                        # Check if target is a DTensor
                        if hasattr(target_param, "_spec"):
                            # Target is a DTensor, need to convert source tensor
                            from torch.distributed._tensor import (
                                DTensor,
                                distribute_tensor,
                            )

                            # Get DTensor specifications from target
                            device_mesh = target_param._spec.mesh
                            placements = target_param._spec.placements

                            # Log debugging information
                            logger.info(f"Loading {param_name}:")
                            logger.info(f"  Source shape: {src_tensor.shape}")
                            logger.info(f"  Target shape: {target_param.shape}")
                            logger.info(
                                f"  Target local shape: {target_param._local_tensor.shape if hasattr(target_param, '_local_tensor') else 'N/A'}"
                            )
                            logger.info(f"  Placements: {placements}")
                            logger.info(f"  Device mesh: {device_mesh}")

                            # Move source tensor to target device
                            src_on_device = src_tensor.to(target_param.device)

                            try:
                                # Check if shapes match globally
                                shapes_match = target_param.shape == src_on_device.shape

                                # Also check if source matches target's local shape (for FSDP sharding)
                                local_shapes_match = False
                                if hasattr(target_param, "_local_tensor"):
                                    local_shapes_match = (
                                        target_param._local_tensor.shape
                                        == src_on_device.shape
                                    )

                                if (
                                    shapes_match
                                    and src_on_device.shape
                                    == target_param._local_tensor.shape
                                ):
                                    # Source is already the correct local size (pre-sharded checkpoint)
                                    logger.info(
                                        "Shapes match globally and locally, using from_local"
                                    )
                                    src_dtensor = DTensor.from_local(
                                        src_on_device,
                                        device_mesh=device_mesh,
                                        placements=placements,
                                        run_check=False,
                                    )
                                    target_param.copy_(src_dtensor)
                                elif (
                                    shapes_match
                                    and src_on_device.shape
                                    != target_param._local_tensor.shape
                                ):
                                    # Global shapes match but source is global size, needs to be sharded
                                    logger.info(
                                        "Global shapes match but source needs sharding"
                                    )
                                    logger.info(
                                        f"  Source is global size: {src_on_device.shape}"
                                    )
                                    logger.info(
                                        f"  Target expects local shards: {target_param._local_tensor.shape}"
                                    )

                                    # Get the local shard for this rank
                                    local_tensor = self._get_local_shard(
                                        src_on_device,
                                        target_param._local_tensor.shape,
                                        device_mesh,
                                        placements,
                                    )

                                    # Directly copy to the local tensor, bypassing DTensor
                                    target_param._local_tensor.copy_(local_tensor)
                                    logger.info(
                                        "Successfully copied local shard directly"
                                    )
                                elif local_shapes_match:
                                    # Source matches target's local shape - this is an unsharded checkpoint
                                    # being loaded into a sharded model
                                    logger.info(
                                        "Source matches target local shape - loading unsharded checkpoint"
                                    )
                                    # Directly copy to local tensor
                                    target_param._local_tensor.copy_(src_on_device)
                                else:
                                    # Shapes don't match - try distribute_tensor
                                    logger.info(
                                        "Shape mismatch, attempting distribute_tensor"
                                    )
                                    logger.info(
                                        f"  Source: {src_on_device.shape}, Target global: {target_param.shape}, Target local: {target_param._local_tensor.shape if hasattr(target_param, '_local_tensor') else 'N/A'}"
                                    )
                                    src_dtensor = distribute_tensor(
                                        src_on_device,
                                        device_mesh=device_mesh,
                                        placements=placements,
                                    )
                                    target_param.copy_(src_dtensor)

                                successfully_loaded.append(param_name)
                                continue  # Skip to next parameter

                            except Exception as dist_error:
                                logger.warning(
                                    f"DTensor conversion failed for {param_name}: {type(dist_error).__name__}: {dist_error}"
                                )
                                # Check if this is a size mismatch error
                                if (
                                    "must match the size" in str(dist_error)
                                    or "size mismatch" in str(dist_error).lower()
                                ):
                                    logger.warning(
                                        f"Size mismatch details: source={src_tensor.shape}, target={target_param.shape}"
                                    )
                                    if hasattr(target_param, "_local_tensor"):
                                        logger.warning(
                                            f"Target local shape: {target_param._local_tensor.shape}"
                                        )
                                # Fall through to record as failed
                                raise  # Re-raise to be caught by outer exception handler
                        else:
                            # Regular tensor - direct copy
                            target_param.copy_(src_tensor)
                            successfully_loaded.append(param_name)
                    else:
                        logger.warning(f"Parameter {param_name} not found in model")
                        failed_params.append(param_name)

                except Exception as e:
                    logger.warning(f"Failed to load {param_name}: {e}")
                    failed_params.append(param_name)

        # Log summary
        if successfully_loaded:
            logger.info(
                f"Successfully transferred {len(successfully_loaded)} parameters:"
            )
            for key in sorted(successfully_loaded)[:10]:  # Show first 10
                logger.info(f"  - {key}")
            if len(successfully_loaded) > 10:
                logger.info(f"  ... and {len(successfully_loaded) - 10} more")

        if failed_params:
            logger.warning(f"Failed to transfer {len(failed_params)} parameters:")
            for key in failed_params[:5]:
                logger.warning(f"  - {key}")
            if len(failed_params) > 5:
                logger.warning(f"  ... and {len(failed_params) - 5} more")

        return successfully_loaded
