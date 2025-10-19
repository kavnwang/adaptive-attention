# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import os
import random
import time
from datetime import timedelta
from typing import Any

import torch
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from llmonade.components.checkpoint import TrainState, PretrainedLayerLoader, freeze_model_layers
from llmonade.config_manager import JobConfig
from llmonade.data import build_dataloader, build_dataset
from llmonade.models.parallelize_bento import parallelize_bento
from llmonade.models.pipeline_bento import pipeline_bento
from llmonade.tools.utils import get_nparams_and_flops
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import (
    build_device_memory_monitor,
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import (
    TrainSpec,
    get_train_spec,
    register_train_spec,
)
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from typing import List, Dict


def _ensure_auto_model_registration(model_config: AutoConfig) -> None:
    """
    Import the package owning the config class so that any AutoModel
    registrations executed at module import time are applied before we
    instantiate from the config.
    """
    module_name = type(model_config).__module__
    package = module_name.rsplit(".", 1)[0] if "." in module_name else module_name

    try:
        importlib.import_module(package)
    except Exception as exc:  # pragma: no cover - best effort (should rarely fail)
        logger.debug(f"Unable to import '{package}' for AutoModel registration: {exc}")




def build_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)


def create_optimizer_for_phase(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    ft_manager,
    parallel_dims,
    phase: str,
) -> Any:
    """Create optimizer with phase-specific hyperparameters."""
    # Only use custom optimizer settings if toggle is enabled
    if phase == "synthetic" and job_config.training.use_custom_synthetic_optimizer:
        # Save original optimizer config
        original_lr = job_config.optimizer.lr
        original_wd = job_config.optimizer.weight_decay
        
        # Override with phase-specific values
        if job_config.training.synthetic_lr is not None:
            job_config.optimizer.lr = job_config.training.synthetic_lr
        if job_config.training.synthetic_weight_decay is not None:
            job_config.optimizer.weight_decay = job_config.training.synthetic_weight_decay
        
        # Build optimizer with phase-specific config
        optimizers = build_optimizers(model_parts, job_config, ft_manager)
        
        # Restore original config
        job_config.optimizer.lr = original_lr
        job_config.optimizer.weight_decay = original_wd
    else:
        # Use unified optimizer settings
        optimizers = build_optimizers(model_parts, job_config, ft_manager)
    
    return optimizers


register_train_spec(
    TrainSpec(
        name="bento",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_bento,
        pipelining_fn=pipeline_bento,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])

    ### RACE CONDITION FIX ###
    global_rank = int(os.environ["RANK"])

    # Setup per-rank Triton cache directory
    base_cache_dir = os.environ.get("TRITON_CACHE_DIR", "~/tmp/triton_cache_user_owned")
    base_cache_dir = os.path.expanduser(base_cache_dir)
    rank_cache_dir = os.path.join(base_cache_dir, f"rank_{global_rank}")
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = rank_cache_dir
    logger.info(f"Using Triton cache directory: {rank_cache_dir}")

    ### END ###

    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )
    dist_utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not supported in this version"
        )
        """
        ! TODO[flame]: We need to fix the pipeline parallelism for flame
        [x] Match the key of models' components with the actual naming
        [ ] Fix the post-init and tie-embedding for pipeline parallelism, HF's transformer automatically
            forces to tie if head is None, we need to handle this case
        [ ]
        """
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    logger.info(f"{tokenizer}")
    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}"
        if job_config.training.dataset_name is not None
        else ""
    )
    dataset = build_dataset(
        dataset=job_config.training.dataset,
        dataset_name=job_config.training.dataset_name,
        dataset_split=job_config.training.dataset_split,
        data_dir=job_config.training.data_dir,
        data_files=job_config.training.data_files,
        data_probs=job_config.training.data_probs,
        streaming=job_config.training.streaming,
        dp_degree=dp_degree,
        num_workers=job_config.training.num_workers,
        seed=job_config.training.seed,
    )

    logger.info("Building dataloader...")
    
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
        apply_qa_mask=job_config.training.apply_qa_mask,
        apply_hashhop_mask=job_config.training.apply_hashhop_mask,
        apply_digit_lookup_mask=job_config.training.apply_digit_lookup_mask,
        ar_assist=job_config.training.ar_assist,
        ar_chunk_len=job_config.training.ar_chunk_len,
        current_step=0,  # initial step
        ar_max_steps=job_config.training.ar_max_steps,
    )

    # Build synthetic dataloader if synthetic training is configured
    synthetic_dataloader = None
    synthetic_steps = 0
    if job_config.training.synthetic_dataset and job_config.training.synthetic_end_step > 0:
        synthetic_steps = job_config.training.synthetic_end_step - job_config.training.synthetic_start_step
        if synthetic_steps > 0:
            logger.info(
                f"Loading synthetic dataset {job_config.training.synthetic_dataset}"
                f":{job_config.training.synthetic_dataset_name}"
                if job_config.training.synthetic_dataset_name is not None
                else ""
            )
            synthetic_dataset = build_dataset(
                dataset=job_config.training.synthetic_dataset,
                dataset_name=job_config.training.synthetic_dataset_name,
                dataset_split=job_config.training.synthetic_dataset_split,
                data_dir=job_config.training.synthetic_data_dir,
                data_files=job_config.training.synthetic_data_files,
                data_probs=None,  # No dataset interleaving for synthetic data
                streaming=job_config.training.streaming,
                dp_degree=dp_degree,
                num_workers=job_config.training.num_workers,
                seed=job_config.training.seed,
            )

            logger.info("Building synthetic dataloader...")
            synthetic_dataloader = build_dataloader(
                dataset=synthetic_dataset,
                tokenizer=tokenizer,
                rank=dp_rank,
                world_size=dp_degree,
                batch_size=job_config.training.batch_size,
                seq_len=job_config.training.seq_len,
                context_len=job_config.training.context_len,
                varlen=job_config.training.varlen,
                num_workers=job_config.training.num_workers,
                pin_memory=job_config.training.pin_memory,
                persistent_workers=job_config.training.persistent_workers,
                snapshot_every_n_steps=job_config.checkpoint.interval,
                apply_qa_mask=job_config.training.apply_qa_mask,
                apply_hashhop_mask=job_config.training.apply_hashhop_mask,
                apply_digit_lookup_mask=job_config.training.apply_digit_lookup_mask,
                ar_assist=False,  # Typically don't apply AR to synthetic data
                ar_chunk_len=job_config.training.ar_chunk_len,
                current_step=0,
                ar_max_steps=job_config.training.ar_max_steps,
            )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)

    # Save model config to experiment directory
    config_save_path = os.path.join(job_config.job.dump_folder, "model_config.json")
    with open(config_save_path, "w") as f:
        json.dump(model_config.to_dict(), f, indent=2)
    logger.info(f"Saved model config to {config_save_path}")

    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len base on inputs
    if parallel_dims.tp_enabled:
        if model_config.fuse_norm:
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled:
        if model_config.fuse_cross_entropy:
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_cross_entropy = False
    model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )

    _ensure_auto_model_registration(model_config)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=8 // parallel_dims.tp
            )
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # calculate model size and flops per token
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                if hasattr(m, 'post_init'):
                    m.post_init()
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(parallel_dims, job_config, color)
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            if hasattr(model, 'post_init'):
                model.post_init()
        model.train()

        model_parts = [model]

    # Initialize layers from one or more pretrained checkpoints if configured
    if job_config.checkpoint.pretrained_layer_path:
        if not job_config.checkpoint.pretrained_layer_map:
            raise ValueError(
                "checkpoint.pretrained_layer_map must be specified when using "
                "checkpoint.pretrained_layer_path"
            )

        # Support comma-separated lists to allow multiple sources and mappings
        def _split_csv(val: str) -> list[str]:
            return [s.strip() for s in val.split(",") if s.strip()]

        src_paths = (
            _split_csv(job_config.checkpoint.pretrained_layer_path)
            if isinstance(job_config.checkpoint.pretrained_layer_path, str)
            else [job_config.checkpoint.pretrained_layer_path]
        )
        maps_raw = job_config.checkpoint.pretrained_layer_map
        map_specs = _split_csv(maps_raw) if isinstance(maps_raw, str) else [maps_raw]

        # Align mapping list length with paths (repeat single map if needed)
        if len(map_specs) == 1 and len(src_paths) > 1:
            map_specs = map_specs * len(src_paths)
        if len(map_specs) != len(src_paths):
            raise ValueError(
                "Number of mapping specs must equal number of pretrained_layer_path entries"
            )

        # Step may be a single int; when multiple paths are provided, use the same step for all
        step_spec = job_config.checkpoint.pretrained_layer_step
        step_vals = []
        if isinstance(step_spec, int):
            step_vals = [step_spec] * len(src_paths)
        else:
            # Fallback: try to parse string; if parsing fails, use -1 (latest)
            try:
                step_vals = [int(s) for s in _split_csv(str(step_spec))]
            except Exception:
                step_vals = [-1] * len(src_paths)
            if len(step_vals) == 1 and len(src_paths) > 1:
                step_vals = step_vals * len(src_paths)
            if len(step_vals) != len(src_paths):
                step_vals = [-1] * len(src_paths)

        total_transferred = 0
        for src_path, map_spec, step in zip(src_paths, map_specs, step_vals):
            try:
                logger.info(
                    f"{color.green}Initializing layers from {src_path} (step={step}){color.reset}"
                )
                loader = PretrainedLayerLoader(
                    source_path=src_path,
                    layer_mapping=map_spec,
                    step=step,
                    device_mesh=world_mesh,
                    parallel_dims=parallel_dims,
                )
                # Apply to all model parts (for pipeline parallelism compatibility)
                for i, model_part in enumerate(model_parts):
                    if parallel_dims.pp_enabled:
                        logger.info(
                            f"Applying pretrained layers to pipeline stage {i}"
                        )
                    transferred_keys = loader.load_and_apply(model_part)
                    total_transferred += len(transferred_keys)
            except Exception as e:
                if job_config.checkpoint.pretrained_layer_strict:
                    raise RuntimeError(
                        f"Failed to initialize pretrained layers from {src_path}: {e}"
                    )
                else:
                    logger.warning(
                        f"{color.yellow}Failed to initialize pretrained layers from {src_path}: {e}. "
                        f"Continuing without this source.{color.reset}"
                    )

        if total_transferred > 0:
            logger.info(
                f"{color.green}Successfully initialized {total_transferred} parameters from pretrained checkpoints{color.reset}"
            )

    # Freeze layers if configured
    if job_config.checkpoint.freeze_layer_map:
        try:
            logger.info(
                f"{color.green}Freezing layers based on freeze_layer_map{color.reset}"
            )

            frozen_params = freeze_model_layers(
                model_parts,
                job_config.checkpoint.freeze_layer_map,
                job_config.checkpoint.freeze_layer_strict,
            )

            if frozen_params > 0:
                logger.info(
                    f"{color.green}Successfully froze {frozen_params:,} parameters{color.reset}"
                )
            else:
                logger.warning(
                    f"{color.yellow}No parameters were frozen. Check your freeze_layer_map patterns.{color.reset}"
                )

        except Exception as e:
            if job_config.checkpoint.freeze_layer_strict:
                raise RuntimeError(f"Failed to freeze layers: {e}")
            else:
                logger.warning(
                    f"{color.yellow}Failed to freeze layers: {e}. "
                    f"Continuing without freezing.{color.reset}"
                )

    # Apply token subset optimization if enabled
    if job_config.training.enable_token_subset_optimization:
        from llmonade.utils.token_subset import apply_token_subset_optimization, identify_used_tokens
        
        logger.info(
            f"{color.green}Applying token subset optimization "
            f"(similar to Unsloth's fix_untrained_tokens){color.reset}"
        )
        
        # First, identify all tokens used across both main and synthetic datasets
        all_used_tokens = set()
        
        # Scan main dataset
        logger.info("Scanning main dataset for used tokens...")
        main_tokens = identify_used_tokens(
            dataset=dataset,
            tokenizer=tokenizer,
            max_samples=job_config.training.token_subset_scan_samples,
            sample_field="input_ids"
        )
        all_used_tokens.update(main_tokens)
        
        # Scan synthetic dataset if it exists
        if 'synthetic_dataset' in locals() and synthetic_dataset is not None:
            logger.info("Scanning synthetic dataset for used tokens...")
            synthetic_tokens = identify_used_tokens(
                dataset=synthetic_dataset,
                tokenizer=tokenizer,
                max_samples=job_config.training.token_subset_scan_samples,
                sample_field="input_ids"
            )
            all_used_tokens.update(synthetic_tokens)
            
            if dp_rank == 0:
                logger.info(
                    f"Found {len(main_tokens)} tokens in main dataset, "
                    f"{len(synthetic_tokens)} in synthetic dataset, "
                    f"{len(all_used_tokens)} total unique tokens"
                )
        
        # Apply optimization to all model parts using combined token set
        from llmonade.utils.token_subset import optimize_embeddings_for_token_subset
        
        for i, model_part in enumerate(model_parts):
            if parallel_dims.pp_enabled:
                logger.info(f"Applying token subset optimization to pipeline stage {i}")
            
            optimize_embeddings_for_token_subset(
                model=model_part,
                tokenizer=tokenizer,
                used_tokens=all_used_tokens,
                eps=job_config.training.token_subset_eps,
                scale_factor=job_config.training.token_subset_scale_factor
            )
        
        if dp_rank == 0:
            logger.info(
                f"{color.green}Token subset optimization complete: "
                f"{len(all_used_tokens)} tokens used out of {tokenizer.vocab_size} total "
                f"({len(all_used_tokens)/tokenizer.vocab_size*100:.1f}%){color.reset}"
            )

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # Initialize train state before optimizer creation
    train_state = TrainState()
    train_state.use_custom_synthetic_optimizer = job_config.training.use_custom_synthetic_optimizer

    # build optimizer after applying parallelisms to the model
    # Use phase-specific hyperparameters for initial optimizer
    initial_phase = "main"  # Default to main
    if synthetic_dataloader is not None and synthetic_steps > 0:
        # Determine initial phase based on current step
        if train_state.step < job_config.training.synthetic_start_step:
            initial_phase = "main"
        elif train_state.step < job_config.training.synthetic_end_step:
            initial_phase = "synthetic"
        else:
            initial_phase = "main"
    
    optimizers = create_optimizer_for_phase(
        model_parts, job_config, ft_manager, parallel_dims, initial_phase
    )
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    # Initialize training phase based on config
    if synthetic_dataloader is not None and synthetic_steps > 0:
        if train_state.step < job_config.training.synthetic_start_step:
            train_state.training_phase = "main"
        elif train_state.step < job_config.training.synthetic_end_step:
            train_state.training_phase = "synthetic"
        else:
            train_state.training_phase = "main"
    else:
        train_state.training_phase = "main"

    # load initial checkpoint
    # Use synthetic dataloader for checkpoint if we're in synthetic phase
    checkpoint_dataloader = (
        synthetic_dataloader
        if train_state.training_phase == "synthetic" and synthetic_dataloader is not None
        else dataloader
    )
    checkpoint = CheckpointManager(
        dataloader=checkpoint_dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, (
            "Must create seed checkpoint using a single device, to disable sharding"
        )
        assert job_config.checkpoint.enable_checkpoint, (
            "Must enable checkpointing when creating a seed checkpoint"
        )
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)

    # After loading checkpoint, determine which phase we're in based on loaded state
    if synthetic_dataloader is not None and synthetic_steps > 0:
        # Determine current phase based on step
        if train_state.step < job_config.training.synthetic_start_step:
            train_state.training_phase = "main"
            current_dataloader = dataloader
        elif train_state.step < job_config.training.synthetic_end_step:
            train_state.training_phase = "synthetic"
            current_dataloader = synthetic_dataloader
        else:
            train_state.training_phase = "main"
            current_dataloader = dataloader
        
        checkpoint.dataloader = current_dataloader
        
        # Recreate optimizer if needed and custom optimizer is enabled
        if job_config.training.use_custom_synthetic_optimizer:
            expected_phase = train_state.training_phase
            if expected_phase != initial_phase and train_state.step > 0:
                logger.info(f"Recreating optimizer for {expected_phase} phase after checkpoint load")
                optimizers = create_optimizer_for_phase(
                    model_parts, job_config, ft_manager, parallel_dims, expected_phase
                )
                lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
                optimizers.register_step_post_hook(
                    lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
                )
                # Update checkpoint manager
                checkpoint.optimizers = optimizers
                checkpoint.lr_schedulers = lr_schedulers
    else:
        train_state.training_phase = "main"
        current_dataloader = dataloader
        checkpoint.dataloader = dataloader
    
    # No initial freezing - weights only get frozen when exiting synthetic phase
    # This ensures we never accidentally freeze before synthetic training

    metric_logger = build_metrics_processor(job_config, parallel_dims)
    # Set dependent attributes for metric_logger
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers  # Pass optimizers if needed by logger logic
    metric_logger.lr_schedulers = (
        lr_schedulers  # Pass schedulers if needed by logger logic
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

    # Always create main_data_iterator from the main UFW dataset
    main_data_iterator = iter(dataloader)  # Always points to main UFW dataset
    
    # Set up iterators based on current phase
    if train_state.training_phase == "synthetic" and synthetic_dataloader is not None:
        synthetic_data_iterator = iter(synthetic_dataloader)
        data_iterator = synthetic_data_iterator  # For compatibility
    else:
        synthetic_data_iterator = None
        data_iterator = main_data_iterator  # For compatibility

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")

    # Log phase-specific information
    if synthetic_dataloader is not None and synthetic_steps > 0:
        logger.info(f"{color.green}  Synthetic training: steps {job_config.training.synthetic_start_step} to {job_config.training.synthetic_end_step}")
        logger.info(f"{color.green}  Use custom synthetic optimizer: {job_config.training.use_custom_synthetic_optimizer}")
        if job_config.training.use_custom_synthetic_optimizer:
            logger.info(f"{color.green}  Synthetic LR: {job_config.training.synthetic_lr or job_config.optimizer.lr}")
            logger.info(f"{color.green}  Synthetic weight decay: {job_config.training.synthetic_weight_decay or job_config.optimizer.weight_decay}")
        if train_state.training_phase == "synthetic":
            logger.info(f"{color.green}  Starting with SYNTHETIC dataset training phase")
        else:
            logger.info(f"{color.green}  Starting with MAIN dataset training")
    else:
        logger.info(f"{color.green}  Starting with MAIN dataset training")
    
    # Log AR status
    if job_config.training.ar_assist:
        logger.info(
            f"{color.green}  Associative Recall augmentation: ENABLED for first "
            f"{job_config.training.ar_max_steps} steps (chunk_len={job_config.training.ar_chunk_len})"
        )
    else:
        logger.info(f"{color.green}  Associative Recall augmentation: DISABLED")

    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )

    with (
        maybe_enable_profiling(
            job_config, global_step=train_state.step
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
    ):
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)
            
            # Check for disabling AR augmentation
            if (
                job_config.training.ar_assist
                and train_state.step == job_config.training.ar_max_steps
                and hasattr(dataloader.dataset, 'ar_assist')
                and dataloader.dataset.ar_assist
            ):
                logger.info(f"{color.red}***** Disabling AR augmentation at step {train_state.step} *****{color.reset}")
                
                # Save checkpoint before transition
                checkpoint.save(train_state.step, force=True)
                
                # Recreate main dataloader with AR disabled
                dataloader = build_dataloader(
                    dataset=dataset,
                    tokenizer=tokenizer,
                    rank=dp_rank,
                    world_size=dp_degree,
                    batch_size=job_config.training.batch_size,
                    seq_len=job_config.training.seq_len,
                    context_len=job_config.training.context_len,
                    varlen=job_config.training.varlen,
                    num_workers=job_config.training.num_workers,
                    pin_memory=True,
                    prefetch_factor=job_config.training.prefetch_factor,
                    persistent_workers=job_config.training.persistent_workers,
                    apply_qa_mask=job_config.training.apply_qa_mask,
                    apply_hashhop_mask=job_config.training.apply_hashhop_mask,
                    apply_digit_lookup_mask=job_config.training.apply_digit_lookup_mask,
                    ar_assist=False,  # Disable AR
                    ar_chunk_len=job_config.training.ar_chunk_len,
                    current_step=train_state.step,
                    ar_max_steps=job_config.training.ar_max_steps,
                )
                
                # Create new iterator
                main_data_iterator = iter(dataloader)
                
                # Update checkpoint's dataloader reference
                checkpoint.dataloader = dataloader
                
                logger.info(f"{color.green}AR augmentation disabled successfully{color.reset}")

            # Check for entering synthetic phase
            if (
                synthetic_dataloader is not None
                and train_state.step == job_config.training.synthetic_start_step
                and train_state.training_phase != "synthetic"
            ):
                logger.info(f"{color.red}***** Starting synthetic dataset training *****{color.reset}")
                logger.info(f"Synthetic weight: {job_config.training.synthetic_weight} (synthetic) / {1.0 - job_config.training.synthetic_weight} (main corpus)")
                train_state.training_phase = "synthetic"
                # Keep both iterators during synthetic phase
                synthetic_data_iterator = iter(synthetic_dataloader)
                # For checkpointing, use synthetic dataloader
                checkpoint.dataloader = synthetic_dataloader
                
                # No need to unfreeze when entering synthetic - weights are already unfrozen
                
                if job_config.training.use_custom_synthetic_optimizer:
                    # Only reset optimizer if using custom settings
                    saved_optimizer_state = None
                    if not job_config.training.reset_optimizer_on_transition:
                        # Save current optimizer state
                        saved_optimizer_state = optimizers.state_dict()
                    
                    logger.info("Creating optimizer for synthetic dataset training")
                    optimizers = create_optimizer_for_phase(
                        model_parts, job_config, ft_manager, parallel_dims, "synthetic"
                    )
                    
                    if saved_optimizer_state and not job_config.training.reset_optimizer_on_transition:
                        # Restore optimizer state
                        optimizers.load_state_dict(saved_optimizer_state)
                    
                    # Re-register hooks and update references
                    optimizers.register_step_post_hook(
                        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
                    )
                    
                    if job_config.training.reset_lr_scheduler_on_transition:
                        lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
                    
                    checkpoint.optimizers = optimizers
                    checkpoint.lr_schedulers = lr_schedulers
                    metric_logger.optimizers = optimizers
                    metric_logger.lr_schedulers = lr_schedulers

            # Check for exiting synthetic phase
            elif (
                synthetic_dataloader is not None
                and train_state.step == job_config.training.synthetic_end_step
                and train_state.training_phase == "synthetic"
            ):
                # Synchronize all workers before transition
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                
                logger.info(f"{color.red}***** Transitioning from synthetic to main dataset training *****{color.reset}")
                logger.info(f"Completed {synthetic_steps} synthetic training steps")
                
                # Save checkpoint before transition
                checkpoint.save(train_state.step, force=True)
                
                train_state.training_phase = "main"
                train_state.synthetic_steps_completed = synthetic_steps
                
                # Back to using only main dataloader
                synthetic_data_iterator = None
                checkpoint.dataloader = dataloader
                
                if job_config.training.use_custom_synthetic_optimizer:
                    # Only reset optimizer if using custom settings
                    saved_optimizer_state = None
                    if not job_config.training.reset_optimizer_on_transition:
                        # Save current optimizer state
                        saved_optimizer_state = optimizers.state_dict()
                    
                    logger.info("Creating optimizer for main dataset training")
                    optimizers = create_optimizer_for_phase(
                        model_parts, job_config, ft_manager, parallel_dims, "main"
                    )
                    
                    if saved_optimizer_state and not job_config.training.reset_optimizer_on_transition:
                        # Restore optimizer state
                        optimizers.load_state_dict(saved_optimizer_state)
                    
                    # Re-register hooks and update references
                    optimizers.register_step_post_hook(
                        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
                    )
                    
                    if job_config.training.reset_lr_scheduler_on_transition:
                        lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
                    
                    checkpoint.optimizers = optimizers
                    checkpoint.lr_schedulers = lr_schedulers
                    metric_logger.optimizers = optimizers
                    metric_logger.lr_schedulers = lr_schedulers
                
                logger.info(
                    f"{color.green}Starting main dataset training with "
                    f"lr={job_config.optimizer.lr}, "
                    f"weight_decay={job_config.optimizer.weight_decay}{color.reset}"
                )
            
            # Update phase step counter
            train_state.phase_step = train_state.step - (
                job_config.training.synthetic_start_step 
                if train_state.training_phase == "synthetic" 
                else job_config.training.synthetic_end_step
            )

            optimizers.zero_grad()
            

            losses = []
            # do gradient accumulation if enabled
            for _ in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()
                
                # During synthetic phase, probabilistically choose dataset
                if train_state.training_phase == "synthetic" and synthetic_data_iterator is not None:
                    if random.random() < job_config.training.synthetic_weight:
                        # Use synthetic dataset
                        batch = next(synthetic_data_iterator)
                    else:
                        # Use main dataset
                        batch = next(main_data_iterator)
                else:
                    # Normal phase - use main dataset
                    batch = next(main_data_iterator)
                
                input_ids, labels = batch["input_ids"], batch["labels"]

                # Update metrics processor state before forward/backward
                metric_logger.ntokens_since_last_log += labels.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                input_ids = input_ids.to(device_type)

                """
                TODO[flame]: We need to carefully handle the position_ids for TP/CP
                Depending on the Models'PE, the position_ids might be different.

                e.g. for TP
                    For RoPE, all ranks have the same position_ids. [FOR HF model]
                    For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

                e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                    Each rank has the coresponding chunked position_ids. [FOR All model]

                """
                labels = labels.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
                if cu_seqlens is not None:
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    position_ids = (
                        torch.arange(0, input_ids.shape[1], device=device_type)
                        .repeat(input_ids.shape[0], 1)
                        .to(torch.int32)
                    )
                # apply context parallelism if cp is enabled
                # ensure CP handles the separate freqs_cis buffer for each pp stage
                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={input_ids, labels, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                # #! TODO[flame], we should distribute the position_ids as well with CP
                if parallel_dims.pp_enabled:
                    raise NotImplementedError(
                        "Pipeline parallelism is not supported in this version"
                    )
                    # Pipeline Parallel forward / backward inside step() call
                    with train_context(optional_context_parallel_ctx):
                        targets, losses = (
                            (labels, []) if has_last_stage else (None, None)
                        )

                        if has_first_stage:
                            pp_schedule.step(input_ids, target=targets, losses=losses)
                        else:
                            pp_schedule.step(target=targets, losses=losses)

                    # accumulate losses across pipeline microbatches
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        output = model(
                            input_ids=input_ids,
                            labels=labels,
                            position_ids=position_ids,
                            cu_seqlens=cu_seqlens,
                        )
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                        loss.backward()

                losses.append(loss)
            loss = sum(losses)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()

            # log metrics - Use MetricsProcessor
            if metric_logger.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    # Use dist_mean/max on the accumulated loss for the step
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                        dist_utils.dist_max(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                    )
                else:
                    # Scale back the loss before logging
                    global_avg_loss = global_max_loss = loss.item()

                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = (
                    time_now - metric_logger.time_last_log
                )  # Use metric_logger's time
                train_state.token += (
                    metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                # Collect routing statistics from all SortingAttention layers for wandb logging
                routing_stats = {}
                for i, m in enumerate(model_parts):
                    for name, module in m.named_modules():
                        if hasattr(module, "get_routing_stats"):
                            layer_stats = module.get_routing_stats()
                            layer_name = f"layer_{i}_{name.replace('.', '_')}"
                            for stat_name, stat_value in layer_stats.items():
                                routing_stats[f"routing/{layer_name}/{stat_name}"] = (
                                    stat_value
                                )

                # Check if model has get_mlp_memory_routing_stats method (for memory_mlp_router models)
                if hasattr(model, "get_mlp_memory_routing_stats"):
                    mlp_memory_stats = model.get_mlp_memory_routing_stats()
                    # Add these stats to routing_stats with appropriate prefix
                    for stat_name, stat_value in mlp_memory_stats.items():
                        routing_stats[f"routing/{stat_name}"] = stat_value

                # Log using the metric processor
                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )

                # Add routing statistics to metrics
                extra_metrics = {
                    "optimizer/lr": last_lr,
                    "optimizer/grad_norm": grad_norm.item(),
                    "optimizer/skipped_step": train_state.skipped_step,
                }
                
                # Add PHi losses if model supports it
                if hasattr(model, 'get_loss_components'):
                    lm_loss, phi_losses = model.get_loss_components()
                    if lm_loss is not None:
                        extra_metrics["loss_metrics/lm_loss"] = lm_loss
                    if phi_losses:
                        # Add total PHi loss
                        if 'total' in phi_losses:
                            extra_metrics["loss_metrics/phi_loss_total"] = phi_losses['total']
                        # Add individual PHi loss components
                        for loss_name, loss_value in phi_losses.items():
                            if loss_name != 'total':
                                extra_metrics[f"loss_metrics/phi_{loss_name}"] = loss_value

                # Add all routing statistics to metrics
                if routing_stats:
                    extra_metrics.update(routing_stats)

                metric_logger.log(
                    train_state.step,
                    global_avg_loss,
                    global_max_loss,
                    extra_metrics=extra_metrics,
                )

                # Optional: Reset routing statistics after logging
                # (Uncomment if you want fresh stats for each logging period)
                # if hasattr(model, "reset_all_routing_stats"):
                #     print("DEBUG: Resetting routing stats for next period")
                #     model.reset_all_routing_stats()

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
