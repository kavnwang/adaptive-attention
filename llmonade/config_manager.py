# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import defaultdict
from typing import Tuple

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtitan.tools.logging import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def string_list(raw_arg):
    """Comma-separated string list argument."""
    return [s.strip() for s in raw_arg.split(",") if s.strip()]


def check_string_list_argument(args_dict: dict[str, any], fullargname: str):
    section, name = fullargname.split(".")
    # Split string list which are still raw strings.
    if (
        section in args_dict
        and name in args_dict[section]
        and isinstance(args_dict[section][name], str)
    ):
        sec = args_dict[section]
        sec[name] = string_list(sec[name])


def parse_optimizer_kwargs(kwargs_list):
    """Parse optimizer kwargs from string list format like ['key1=value1', 'key2=value2']."""
    if not kwargs_list:
        return {}

    kwargs_dict = {}
    for item in kwargs_list:
        if "=" not in item:
            logger.warning(
                f"Skipping invalid optimizer kwarg format: {item}. Expected format: key=value"
            )
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert value to appropriate type
        try:
            # Try boolean first
            if value.lower() in ("true", "false"):
                kwargs_dict[key] = value.lower() == "true"
            # Try float
            elif "." in value:
                kwargs_dict[key] = float(value)
            # Try int
            else:
                kwargs_dict[key] = int(value)
        except ValueError:
            # Keep as string if conversion fails
            kwargs_dict[key] = value

    return kwargs_dict


class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the section name in the toml file
    followed by name of the option in the toml file. For ex,
    model.name translates to:
        [model]
        name
    in the toml file
    """

    def __init__(self):
        self.args_dict = None
        # main parser
        self.parser = argparse.ArgumentParser(description="torchtitan arg parser.")

        self.parser.add_argument(
            "--job.config_file",
            type=str,
            default=None,
            help="Job config file",
        )

        # job level configs
        self.parser.add_argument(
            "--job.dump_folder",
            type=str,
            default="./torchtitan/outputs",
            help="Folder to dump job outputs",
        )
        self.parser.add_argument(
            "--job.description",
            type=str,
            default="default job",
            help="Description of the job",
        )
        self.parser.add_argument(
            "--job.use_for_integration_test",
            action="store_true",
            help="Add this config to the integration test suite",
        )
        self.parser.add_argument(
            "--job.print_args",
            action="store_true",
            help="Print the args to terminal",
        )

        # model configs
        self.parser.add_argument(
            "--model.name",
            type=str,
            default="bento",
            help="Which model to train",
        )
        self.parser.add_argument(
            "--model.config",
            type=str,
            default="fla-hub/transformer-1.3B-100B",
            help="Path to the model config",
        )
        self.parser.add_argument(
            "--model.tokenizer_path",
            type=str,
            default="fla-hub/transformer-1.3B-100B",
            help="Tokenizer path",
        )
        self.parser.add_argument(
            "--model.converters",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Comma separated list of converters to apply to the model.
                For instance, the `float8` converter swaps `torch.nn.Linear`
                with `Float8Linear`. This feature requires you to install 'torchao'
                which can be found here: https://github.com/pytorch/ao
            """,
        )
        self.parser.add_argument(
            "--model.print_after_conversion",
            action="store_true",
            help="""
            If true, model definition will be printed to stdout after all model
            converters have been applied.
            """,
        )

        # profiling configs
        self.parser.add_argument(
            "--profiling.enable_profiling",
            action="store_true",
            help="Whether to enable pytorch profiler",
        )
        self.parser.add_argument(
            "--profiling.save_traces_folder",
            type=str,
            default="profile_traces",
            help="Trace files location",
        )
        self.parser.add_argument(
            "--profiling.profile_freq",
            type=int,
            default=10,
            help="How often to collect profiler traces, in iterations",
        )
        self.parser.add_argument(
            "--profiling.enable_memory_snapshot",
            action="store_true",
            help="Whether to dump memory snapshot",
        )
        self.parser.add_argument(
            "--profiling.save_memory_snapshot_folder",
            type=str,
            default="memory_snapshot",
            help="Memeory snapshot files location",
        )

        # optimizer configs
        self.parser.add_argument(
            "--optimizer.name", type=str, default="AdamW", help="Optimizer to use"
        )
        self.parser.add_argument(
            "--optimizer.eps",
            type=float,
            default=1e-8,
            help="Epsilon value for the optimizer.",
        )
        self.parser.add_argument(
            "--optimizer.lr", type=float, default=8e-4, help="Learning rate to use"
        )
        self.parser.add_argument(
            "--optimizer.beta1",
            type=float,
            default=0.9,
            help="Beta1 parameter for Adam optimizers (first moment decay rate).",
        )
        self.parser.add_argument(
            "--optimizer.beta2",
            type=float,
            default=0.95,
            help="Beta2 parameter for Adam optimizers (second moment decay rate).",
        )
        self.parser.add_argument(
            "--optimizer.weight_decay",
            type=float,
            default=0.1,
            help="Weight decay (L2 penalty) for the optimizer.",
        )
        self.parser.add_argument(
            "--optimizer.kwargs",
            type=string_list,
            nargs="*",
            default=[],
            help="""
                Additional optimizer keyword arguments in format key=value,key2=value2.
                Values are automatically converted to appropriate types (int, float, bool, or string).
                Example: --optimizer.kwargs amsgrad=true,maximize=false
            """,
        )
        self.parser.add_argument(
            "--optimizer.implementation",
            type=str,
            default="fused",
            choices=["for-loop", "foreach", "fused"],
            help="""
            Specify which optimizer implementation to use:
            - 'fused': Use fused implementation (CUDA only) for best performance.
            - 'foreach': Use some horizontal fusion of tensors for better performance.
            - 'for-loop': Use the default implementation for the optimizer (slowest).
            - more info: https://pytorch.org/docs/stable/optim.html
            """,
        )
        self.parser.add_argument(
            "--optimizer.early_step_in_backward",
            action="store_true",
            help="""
            Whether to apply optimizer in the backward. Caution, optimizer_in_backward
            is not compatible with gradients clipping, users should not call
            register_post_accumulate_grad_hook after the optimizer is built.""",
        )

        # Muon optimizer specific configs
        self.parser.add_argument(
            "--optimizer.matched_adamw_rms",
            type=float,
            default=0.3,
            help="The AdamW Update RMS that Muon is designed to match (0.2-0.4 recommended)",
        )
        self.parser.add_argument(
            "--optimizer.momentum",
            type=float,
            default=0.95,
            help="Momentum for Muon SGD component",
        )
        self.parser.add_argument(
            "--optimizer.nesterov",
            action="store_true",
            default=True,
            help="Whether to use Nesterov momentum in Muon",
        )
        self.parser.add_argument(
            "--optimizer.ns_steps",
            type=int,
            default=5,
            help="Number of Newton-Schulz iteration steps for Muon",
        )
        self.parser.add_argument(
            "--optimizer.adamw_betas",
            type=str,
            default="0.95,0.95",
            help="Beta values for AdamW component in Muon (comma-separated)",
        )
        self.parser.add_argument(
            "--optimizer.adamw_eps",
            type=float,
            default=1e-8,
            help="Epsilon for AdamW component in Muon",
        )

        # lr scheduler configs
        self.parser.add_argument(
            "--lr_scheduler.warmup_steps",
            type=int,
            default=200,
            help="Steps for lr scheduler warmup, normally 1/5 of --training.steps",
        )
        self.parser.add_argument(
            "--lr_scheduler.decay_ratio",
            type=float,
            default=None,
            help="""
            Controls the proportion of the training steps allocated to the learning rate decay phase.

            If `None`, the learning rate will begin decaying immediately after the warmup period.
            Otherwise, the learning rate will remain stable after the warmup period and
            only start decaying during the last `decay_ratio` portion of the total training steps.

            This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
            """,
        )
        self.parser.add_argument(
            "--lr_scheduler.decay_type",
            type=str,
            default="linear",
            choices=["linear", "sqrt", "cosine"],
            help="""
            Learning rate decay type to use during training:
            - 'linear': linearly decays learning rate from initial to final value
            - 'sqrt': decays learning rate following a 1 minus square root curve
            - 'cosine': smoothly decays learning rate following a cosine curve
            """,
        )
        self.parser.add_argument(
            "--lr_scheduler.lr_min",
            type=float,
            default=0.0,
            help="""
            Min lr ratio for lr scheduler.

            If provided, the range of decay factor is scaled from 1 to `lr_min`
            to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.lr_min`.
            """,
        )

        # training configs
        self.parser.add_argument(
            "--training.batch_size", type=int, default=8, help="Batch size"
        )
        self.parser.add_argument(
            "--training.seq_len", type=int, default=2048, help="Sequence length"
        )
        self.parser.add_argument(
            "--training.context_len",
            type=int,
            default=2048,
            help="Max length allowed for each sequence",
        )
        self.parser.add_argument(
            "--training.varlen",
            action="store_true",
            help="Whether to take sequences of variable length as input",
        )
        self.parser.add_argument(
            "--training.apply_qa_mask",
            action="store_true",
            help="Apply answer-only masking for QA datasets (masks everything before '?')",
        )
        self.parser.add_argument(
            "--training.apply_hashhop_mask",
            action="store_true",
            help="Apply answer-only masking for HashHop datasets (masks everything except values between quotes)",
        )
        self.parser.add_argument(
            "--training.apply_digit_lookup_mask",
            action="store_true",
            help="Apply answer-only masking for digit lookup datasets (masks everything except values after ':')",
        )
        self.parser.add_argument(
            "--training.gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of steps to accumulate gradients before updating parameters",
        )
        self.parser.add_argument(
            "--training.steps",
            type=int,
            default=10000,
            help="How many train steps to run",
        )
        self.parser.add_argument(
            "--training.max_norm",
            type=float,
            default=1.0,
            help="Max norm for gradient clipping",
        )
        self.parser.add_argument(
            "--training.skip_nan_inf",
            action="store_true",
            help="Skip batch updates when NaN or INF gradients are encountered during training",
        )
        self.parser.add_argument(
            "--training.dataset",
            default="HuggingFaceFW/fineweb-edu",
            help="Dataset to use, with comma separated values",
        )
        self.parser.add_argument(
            "--training.dataset_name",
            default=None,
            help="The name of the dataset config, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.dataset_split",
            default=None,
            help="Dataset split to use, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.data_dir",
            default=None,
            help="Data dirs to use, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.data_files",
            default=None,
            help="Data files to use, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.data_probs",
            default=None,
            help="Data sampling probabilities, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.streaming",
            action="store_true",
            help="Whether to load dataset in streaming mode, used for huge dataset",
        )

        # Synthetic dataset training parameters
        self.parser.add_argument(
            "--training.synthetic_dataset",
            type=str,
            default=None,
            help="Path to synthetic dataset for initial training phase",
        )
        self.parser.add_argument(
            "--training.synthetic_start_step",
            type=int,
            default=0,
            help="Starting step for synthetic dataset training (inclusive)",
        )
        self.parser.add_argument(
            "--training.synthetic_end_step",
            type=int,
            default=0,
            help="Ending step for synthetic dataset training (exclusive). If 0, synthetic training is disabled.",
        )
        self.parser.add_argument(
            "--training.use_custom_synthetic_optimizer",
            action="store_true",
            help="When True, use synthetic_lr and synthetic_weight_decay for synthetic phase. "
                 "When False, use unified optimizer schedule across both phases.",
        )
        self.parser.add_argument(
            "--training.synthetic_lr",
            type=float,
            default=None,
            help="Learning rate for synthetic dataset training phase",
        )
        self.parser.add_argument(
            "--training.synthetic_weight_decay",
            type=float,
            default=None,
            help="Weight decay for synthetic dataset training phase",
        )
        self.parser.add_argument(
            "--training.synthetic_weight",
            type=float,
            default=1.0,
            help="Probability (0-1) of using synthetic dataset during synthetic phase. "
                 "With probability w, use synthetic dataset; with probability 1-w, use general corpus.",
        )
        self.parser.add_argument(
            "--training.reset_optimizer_on_transition",
            action="store_true",
            default=True,
            help="Whether to reset optimizer state when transitioning from synthetic to main dataset",
        )
        self.parser.add_argument(
            "--training.reset_lr_scheduler_on_transition",
            action="store_true",
            default=True,
            help="Whether to reset LR scheduler when transitioning from synthetic to main dataset",
        )
        self.parser.add_argument(
            "--training.synthetic_dataset_name",
            type=str,
            default=None,
            help="Synthetic dataset name if using HuggingFace dataset",
        )
        self.parser.add_argument(
            "--training.synthetic_dataset_split",
            type=str,
            default="train",
            help="Synthetic dataset split to use",
        )
        self.parser.add_argument(
            "--training.synthetic_data_dir",
            type=str,
            default=None,
            help="Synthetic data directory if dataset is stored locally",
        )
        self.parser.add_argument(
            "--training.synthetic_data_files",
            type=str,
            default=None,
            help="Specific synthetic data files to use",
        )
        
        # Associative Recall Augmentation parameters
        self.parser.add_argument(
            "--training.ar_assist",
            action="store_true",
            help="Enable associative recall augmentation for first N steps"
        )
        self.parser.add_argument(
            "--training.ar_chunk_len",
            type=int,
            default=512,
            help="Chunk length for associative recall augmentation"
        )
        self.parser.add_argument(
            "--training.ar_max_steps",
            type=int,
            default=1500,
            help="Maximum steps to apply associative recall augmentation"
        )
        
        self.parser.add_argument(
            "--training.num_workers",
            type=int,
            default=32,
            help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
        )
        self.parser.add_argument(
            "--training.prefetch_factor",
            type=int,
            default=2,
            help="Number of batches loaded in advance by each worker."
            "2 means there will be a total of 2 * num_workers batches prefetched across all workers.",
        )
        self.parser.add_argument(
            "--training.data_parallel_replicate_degree",
            type=int,
            default=1,
            help="""
            The `data_parallel_replicate_degree` argument specifies the degree of
            data parallelism for weight replication. When this value is greater
            than 1, weights will be replicated across `data_parallel_replicate_degree`
            ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
            method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
            parallelism method used is DDP (Distributed Data Parallelism).
            1 means disabled.""",
        )
        self.parser.add_argument(
            "--training.data_parallel_shard_degree",
            type=int,
            default=-1,
            help="""
            The `data_parallel_shard_degree` argument specifies the degree of data
            parallelism for weight sharding. When this value is greater than 1, weights
            will be sharded across `data_parallel_shard_degree` ranks. If
            `data_parallel_replicate_degree` is also greater than 1, the parallelism
            method used is HSDP (Hybrid Sharded Data Parallelism).  Otherwise, the
            parallelism method used is FSDP (Fully Sharded Data Parallelism).

            -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
            only `data_parallel_shard_degree` can be negative. 1 means disabled.""",
        )
        self.parser.add_argument(
            "--training.enable_cpu_offload",
            action="store_true",
            help="""
            Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP""",
        )
        self.parser.add_argument(
            "--training.tensor_parallel_degree",
            type=int,
            default=1,
            help="Tensor Parallelism degree. 1 means disabled.",
        )
        self.parser.add_argument(
            "--training.disable_loss_parallel",
            action="store_true",
            help="Whether to apply loss parallel when sequence parallel is enabled",
        )
        self.parser.add_argument(
            "--training.fsdp_reshard_after_forward",
            type=str,
            default="default",
            choices=["default", "always", "never"],
            help="""
            `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
            within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
            trading off memory and communication. See torch's `fully_shard` API for more documentation
            on `reshard_after_forward`.
            The supported policies include "default", "always" and "never":
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
              scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.
            """,
        )
        self.parser.add_argument(
            "--training.mixed_precision_param",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float32"],
            help="""
                torch dtype to use for parameters when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_shard_degree > 1
            """,
        )
        self.parser.add_argument(
            "--training.mixed_precision_reduce",
            type=str,
            default="float32",
            choices=["float32"],
            help="""
                torch dtype to use for reductions when applying mixed precision via FSDP.
                This feature only takes effect when data_parallel_shard_degree > 1
            """,
        )
        self.parser.add_argument(
            "--training.compile",
            action="store_true",
            help="Whether to compile the model",
        )
        self.parser.add_argument(
            "--training.gc_freq",
            type=int,
            default=50,
            help="Python garbage control scheduling interval, in steps",
        )
        self.parser.add_argument(
            "--training.seed",
            type=int,
            default=42,
            help="Choose the base RNG seed used for training",
        )
        self.parser.add_argument(
            "--training.deterministic",
            action="store_true",
            help="Use deterministic algorithms wherever possible, may be slower",
        )
        
        # Token subset optimization configs (Unsloth-style fix_untrained_tokens)
        self.parser.add_argument(
            "--training.enable_token_subset_optimization",
            action="store_true",
            help="Enable token subset optimization (similar to Unsloth's fix_untrained_tokens). "
                 "This identifies tokens used in the dataset and scales down unused token embeddings.",
        )
        self.parser.add_argument(
            "--training.token_subset_scan_samples",
            type=int,
            default=10000,
            help="Maximum number of dataset samples to scan for identifying used tokens",
        )
        self.parser.add_argument(
            "--training.token_subset_scale_factor",
            type=float,
            default=1e-5,
            help="Scaling factor for unused token embeddings (default: 1e-5)",
        )
        self.parser.add_argument(
            "--training.token_subset_eps",
            type=float,
            default=1e-16,
            help="Small epsilon value for numerical stability in token subset optimization",
        )
        
        # metrics configs
        self.parser.add_argument(
            "--metrics.log_freq",
            type=int,
            default=10,
            help="How often to log metrics to TensorBoard, in iterations",
        )
        self.parser.add_argument(
            "--metrics.enable_tensorboard",
            action="store_true",
            help="Whether to log metrics to TensorBoard",
        )
        self.parser.add_argument(
            "--metrics.disable_color_printing",
            action="store_true",
            help="Whether to disable color printing in logs",
        )
        self.parser.add_argument(
            "--metrics.save_tb_folder",
            type=str,
            default="tb",
            help="Folder to dump TensorBoard states",
        )
        self.parser.add_argument(
            "--metrics.save_for_all_ranks",
            action="store_true",
            default=False,
            help="""
                Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
                When this option is False and pipeline_parallel_degree is > 1, the metrics
                component uses the 0th rank of the last stage pipeline group, which is the
                only stage that computes loss metrics.
            """,
        )
        self.parser.add_argument(
            "--metrics.enable_wandb",
            action="store_true",
            help="Whether to log metrics to Weights & Biases",
        )

        self.parser.add_argument(
            "--experimental.enable_async_tensor_parallel",
            action="store_true",
            help="Whether to apply async tensor parallel (currently only effective when compile is enabled)",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_degree",
            type=int,
            default=1,
            help="""
                Pipeline Parallelism degree, or number of ranks. 1 means disabled.
                If using looped schedules, this still specifies the number of physical ranks, not the number
                of stages.  Stages per rank are inferred from split points degree, and schedule.""",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_split_points",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Specify comma-separated names of modules to use as the beginning of a split point.

                e.g. "layers.0,layers.2" will cause the model to be split into 3 stages,
                the first containing all the layers up to layers.0,
                the second containing layers.0 and up to layers.2,
                the third containing layers.2 and all the remaining layers.

                Note: fully-automated splitting may be enabled in the future,
                but currently the split points must be specified manually.""",
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_schedule",
            type=str,
            default="1F1B",
            help="""
                Specify the Pipeline Parallel schedule to use. The supported schedules are:
                https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
                The schedule must be compatible with the split points and stages_per_rank.

                Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
                and split_points = number of stages - 1
                """,
        )
        self.parser.add_argument(
            "--experimental.pipeline_parallel_schedule_csv",
            type=str,
            default="",
            help="""
                Specify the path to the pipeline parallel schedule csv file to use.
                The pipeline_parallel_schedule argument must be either
                PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
            """,
        )

        self.parser.add_argument(
            "--experimental.pipeline_parallel_microbatches",
            type=int,
            default=None,
            help="""
                How many microbatches to split the global training batch into when using pipeline parallelism.

                The global training batch size must be evenly divisible by the number of microbatches.

                The default value will be the number of pipeline stages, if unspecified.
            """,
        )
        self.parser.add_argument(
            "--experimental.enable_compiled_autograd",
            action="store_true",
            help="Enable CompiledAutograd to compile the backward.",
        )
        self.parser.add_argument(
            "--experimental.context_parallel_degree",
            type=int,
            default=1,
            help="Context parallelism degree. 1 means disabled.",
        )
        self.parser.add_argument(
            "--experimental.context_parallel_rotate_method",
            type=str,
            default="allgather",
            help="""
                The collective to use in context parallel SDPA for kv shards exchange.

                'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,

                'alltoall' means to all-to-all shuffle the kv shards.

                The default value is 'allgather'.
            """,
        )
        # I'm not particularly fond of this. Users can choose to write their own wrapper
        # module and import TorchTitan training loop and execute it, which look cleaner.
        # One reason to provide this option is to allow users to use the existing run script.
        # While the script is pretty trivial now, we may add more logic when integrating
        # with TorchFT.
        # This option is subject to change and may be deleted in the future.
        self.parser.add_argument(
            "--experimental.custom_model_path",
            type=str,
            default="",
            help="""
                The --custom_model_path option allows to specify a custom path to a model module
                that is not natively implemented within TorchTitan.
                Acceptable values are the file system path to the module (e.g., my_models/model_x)
                dotted import module  (e.g., some_package.model_x).
            """,
        )
        # checkpointing configs
        self.parser.add_argument(
            "--checkpoint.enable_checkpoint",
            action="store_true",
            help="Whether to enable checkpoint",
        )
        self.parser.add_argument(
            "--checkpoint.folder",
            type=str,
            default="checkpoint",
            help="""
                The folder to store the checkpoints.
                When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.interval",
            type=int,
            default=500,
            help="Checkpointing interval in steps.",
        )
        self.parser.add_argument(
            "--checkpoint.model_weights_only",
            action="store_true",
            help="""
                When model_weights_only=True, only model weights will be saved at the end of training.
                With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
                When model_weights_only=False, the full checkpoint will be saved.
                A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
                The default value is false.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.export_dtype",
            type=str,
            default="float32",
            choices=["float16", "bfloat16", "float32"],
            help="""
                Converts to the specified precision when training completes and model_weights_only=true.
                Currently supports float32, float16, and bfloat16.
                The default value is float32.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.create_seed_checkpoint",
            action="store_true",
            help="""
                Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
                Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
                Could be implemented as a separate script, but this way shares more code.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.async_mode",
            type=str,
            default="disabled",
            help="""
                Which async checkpoint mode to use. Currently there are 3 different modes.
                1. "disabled": synchronized checkpointing will be used.
                2. "async": torch.distributed.checkpoint.async_save will be used.
                3. "async_with_pinned_mem": this option utilizes a dedicated pinned memory
                   space and creates a separate process for faster GPU->CPU transfer
                   performance and eliminating GIL contention. The cost is increased CPU
                   memory usage. If insufficient CPU memory is available, performance may
                   degrade due to memory paging. For most users, "async" should suffice as
                   the performance overhead is typically small (on the order of tens of
                   seconds) compared to checkpointing frequency. This mode can be employed
                   to pursue near-zero checkpointing times (e.g., < 1 second) given
                   appropriate hardware support such as ample CPU memory and fast PCIe.

                "disabled" is the default mode.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.keep_latest_k",
            type=int,
            default=0,
            help="""
                Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
                0 is the default value. k cannot be 1 as the last one may be in the process of being
                saved. As a result, the metadata of the last one may not be ready yet.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.load_step",
            type=int,
            default=-1,
            help="Load the checkpoint at the specified step. If -1, load the latest checkpoint.",
        )
        self.parser.add_argument(
            "--checkpoint.exclude_from_loading",
            type=string_list,
            nargs="*",
            default=[],
            help="""
                Exclude specific keys from being loaded from the checkpoint.
                Provide a comma-separated list of keys to exclude, e.g. 'optimizer,lr_scheduler,dataloader'.
                This will load the model only, excluding the specified keys.
            """,
        )
        # pretrained layer initialization configs
        self.parser.add_argument(
            "--checkpoint.pretrained_layer_path",
            type=str,
            default=None,
            help="Optional: Path to pretrained checkpoint for layer initialization",
        )
        self.parser.add_argument(
            "--checkpoint.pretrained_layer_map",
            type=str,
            default=None,
            help="Optional: JSON string or file path for explicit layer mapping (required if pretrained_layer_path is set)",
        )
        self.parser.add_argument(
            "--checkpoint.pretrained_layer_step",
            type=int,
            default=-1,
            help="Optional: Step to load from pretrained checkpoint (-1 for latest)",
        )
        self.parser.add_argument(
            "--checkpoint.pretrained_layer_strict",
            action="store_true",
            default=False,
            help="Optional: Whether to fail if layer initialization encounters errors",
        )
        # layer freezing configs
        self.parser.add_argument(
            "--checkpoint.freeze_layer_map",
            type=str,
            default=None,
            help="Optional: JSON string or file path for layer freezing mapping (same format as pretrained_layer_map)",
        )
        self.parser.add_argument(
            "--checkpoint.freeze_layer_strict",
            action="store_true",
            default=False,
            help="Optional: Whether to fail if layer freezing encounters errors",
        )
        # activation checkpointing configs
        self.parser.add_argument(
            "--activation_checkpoint.mode",
            type=str,
            default="selective",
            help="Type of activation checkpointing to use ['none', 'full', 'selective']",
        )
        self.parser.add_argument(
            "--activation_checkpoint.selective_ac_option",
            type=str,
            default="2",  # 2 = checkpoint every other layer
            help="""
                Selective activation checkpointing options ['int', 'op'].
                'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
            """,
        )

        self.parser.add_argument(
            "--activation_offload.mode",
            type=str,
            default="none",
            help="""
                if we are using activation offload or not. Options are ['none', 'full'].
            """,
        )

        # float8 configs
        self.parser.add_argument(
            "--float8.enable_fsdp_float8_all_gather",
            action="store_true",
            help="Whether enable float8 all-gather in FSDP, recommended for tensorwise scaling",
        )
        self.parser.add_argument(
            "--float8.precompute_float8_dynamic_scale_for_fsdp",
            action="store_true",
            help="Whether precompute float8 scales dynamically for FSDP, recommended for tensorwise scaling",
        )
        self.parser.add_argument(
            "--float8.force_recompute_fp8_weight_in_bwd",
            action="store_true",
            help="""
            Whether to force the recomputation of FP8 weights during backward pass.
            When using FSDP with tensorwise scaling, it is recommended to enable
            `force_recompute_fp8_weight_in_bwd` to prevent saving unsharded FP8 weights
            for backward computation.
            """,
        )
        self.parser.add_argument(
            "--float8.recipe_name",
            type=str,
            default=None,
            choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"],
            help="""
            If specified, creates float8 config from recipe name, valid choices are
            `tensorwise`, `rowwise` and `rowwise_with_gw_hp`.
            """,
        )
        self.parser.add_argument(
            "--float8.filter_fqns",
            type=string_list,
            nargs="+",
            default=["swiglu_linear", "down_proj", "lm_head"],
            help="""
            Comma-separated list of fully qualified names of modules to skip applying float8 training to.
            nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
            Example: --float8.filter_fqns "attention.wq,attention.wk,attention.wv,output"
            """,
        )
        self.parser.add_argument(
            "--float8.emulate",
            action="store_true",
            help="""
            If True, emulation is used instead of hardware accelerated gemm. This is for test purpose only,
            as the current CI does not have sm_89 capability, required by Float8.
            Not compatible with torch.compile.
            """,
        )

        # communications library settings
        self.parser.add_argument(
            "--comm.init_timeout_seconds",
            type=int,
            default=300,
            help="Timeout for communication operations, during initialization and first train step.",
        )
        self.parser.add_argument(
            "--comm.train_timeout_seconds",
            type=int,
            default=100,
            help=(
                "Timeout for communication operations after the first train step -- "
                "usually a tighter bound than during initialization."
            ),
        )
        self.parser.add_argument(
            "--comm.trace_buf_size",
            type=int,
            default=20000,
            help="Flight recorder ring buffer size, >0 means recording by default, 0 means disabled",
        )

        # memory estimation settings
        self.parser.add_argument(
            "--memory_estimation.enabled",
            help="Whether to estimate memory usage for FSDP",
            action="store_true",
        )

        self.parser.add_argument(
            "--memory_estimation.disable_fake_mode",
            help="Whether to estimate memory under FakeTensorMode",
            action="store_true",
        )

        self.parser.add_argument(
            "--fault_tolerance.enable",
            action="store_true",
            help="""
                Enable TorchFT integration. When TorchFT is enabled, HSDP will be used.
                And --fault_tolerance.data_parallel_replicate_degree should be 1 and
                --fault_tolerance.group_size will be used to control the maximum
                replicate group size as the replicate group size is dynamic.

                Note that this is still an experimental feature.
            """,
        )

        self.parser.add_argument(
            "--fault_tolerance.replica_id",
            type=int,
            default=0,
            help="The TorchFT replica ID of this run.",
        )

        self.parser.add_argument(
            "--fault_tolerance.group_size",
            type=int,
            default=0,
            help="""
                The number of TorchFT replicate groups. This number will be used for
                dataloader to split the dataset across the replicate groups and FSDP
                dimension
            """,
        )

        self.parser.add_argument(
            "--fault_tolerance.min_replica_size",
            type=int,
            default=1,
            help="The minimum number of FT replica for each step.",
        )

    def to_dict(self):
        return self.args_dict

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                logger.exception(
                    f"Error while loading the configuration file: {config_file}"
                )
                logger.exception(f"Error details: {str(e)}")
                raise e

        # Checking string-list arguments are properly split into a list
        # if split-points came from 'args' (from cmd line) it would have already been parsed into a list by that parser
        string_list_argnames = self._get_string_list_argument_names()
        for n in string_list_argnames:
            check_string_list_argument(args_dict, n)

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        self.args_dict = args_dict

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self) -> None:
        # TODO: Add more mandatory validations
        assert self.model.config
        assert self.model.tokenizer_path

        # Validate optimizer parameters
        if hasattr(self.optimizer, "beta1") and not (0.0 <= self.optimizer.beta1 < 1.0):
            raise ValueError(
                f"optimizer.beta1 must be in [0.0, 1.0), got {self.optimizer.beta1}"
            )
        if hasattr(self.optimizer, "beta2") and not (0.0 <= self.optimizer.beta2 < 1.0):
            raise ValueError(
                f"optimizer.beta2 must be in [0.0, 1.0), got {self.optimizer.beta2}"
            )
        if (
            hasattr(self.optimizer, "weight_decay")
            and self.optimizer.weight_decay < 0.0
        ):
            raise ValueError(
                f"optimizer.weight_decay must be >= 0.0, got {self.optimizer.weight_decay}"
            )

    def _get_string_list_argument_names(self) -> list[str]:
        """Get the parser argument names of type `string_list`."""
        string_list_args = [
            v.dest for v in self.parser._actions if v.type is string_list
        ]
        return string_list_args

    def parse_args_from_command_line(
        self, args_list
    ) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)
        string_list_argnames = set(self._get_string_list_argument_names())

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument(
                    "--" + arg, action="store_true" if val else "store_false"
                )
            elif arg in string_list_argnames:
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
