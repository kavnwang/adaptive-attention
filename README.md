<div align="center">

# 🍋 LLMonade: When life gives you LLMons...

</div>

## Attribution
The initial version, inspiration, and foundational code for this repo come from the excellent [🔥 Flame](https://github.com/fla-org/flame) framework developed by the [fla-org](https://github.com/fla-org) team. Flame is a minimal and efficient training framework built on `torchtitan` for language models with blazing efficiency, providing seamless integration with FLA (Flash Linear Attention) models and transformers. We extend our gratitude to Yu Zhang and Songlin Yang, the primary authors of Flame, for creating such a well-designed and accessible framework for training state-of-the-art language models.

In turn, this repository relies on the publicly maintained tools [TorchTitan](https://github.com/pytorch/torchtitan) by PyTorch and the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) by EleutherAI. TorchTitan serves as the foundational PyTorch native platform for large-scale training of generative AI models, providing essential distributed training capabilities and multi-dimensional parallelism support. The Language Model Evaluation Harness offers a unified framework for evaluating language models across diverse tasks, enabling comprehensive assessment of model performance and capabilities.

## A Practical Guide
You can skip the detailed documentation below this section and follow these quick-start instructions:

### Environment Setup
```bash
git submodule update --init --recursive
uv sync
uv add --editable 3rdparty/bento
uv add --editable 3rdparty/lm-evaluation-harness
uv add --editable 3rdparty/torchtitan
```
When developing a specific module (e.g., `nsa`), make sure to switch to the appropriate branch in both this repo and the `bento`` submodule.

### Configuring Models
Model configurations are located in the `configs/` folder, typically one file per model per size. These configurations are passed directly during model instantiation. To understand how each is used, refer to the corresponding `modeling_***.py` file in bento/bento/models.

### Running Jobs
Pre-configured Slurm scripts are available in the `llmonade/scripts/slurm/` folder with sensible defaults from the literature (e.g., 15B tokens of fw-edu for a 340M parameter model).

To configure a custom job:
1. Change the experiment name
2. Update the config to point to your desired configuration
3. Adjust the scale of the job by modifying `--nnodes` and `--nntasks` in the Slurm job config

Note: When changing scale, compensate your batch size/gradient accumulation accordingly to maintain consistent behavior. For example, if you reduce from 4 nodes to 2 nodes, double your gradient accumulation steps to keep batch sizes consistent.

### Monitoring Jobs
Job logs are stored in the `logs/` folder. You can also use standard Slurm commands such as `squeue` and `sinfo` to monitor jobs.

### Converting Checkpoints
Convert your distributed checkpoints to finalized HuggingFace format using this utility:

```sh
python -m llmonade.scripts.helpers.convert_dcp_to_hf --path <path_to_model> --step <step> --config <path_to_config> --tokenizer <path_to_tokenizer>
```

Important: This utility uses the config to instantiate and save with the model. If your config has diverged from when the model training started, you'll need to revert it for correctness.



## Full Guide

Welcome to 🍋  `LLMonade`, a minimal and efficient framework built on `torchtitan` for language models with blazing efficiency.

**Feature Highlights:**

- 🚀 Minimal, easy-to-use, extensible training framework
- 🤗 Seamless integration with `fla` and `transformers`
- 🔄 Zero-cost data preprocessing: online tokenization, dataset shuffling, and multiple datasets support
- 🔮 4D parallelism (coming soon)



## Dataset Preparation
To download the dataset to your local disk, create a new Python file with the following content and execute it:

```py
from datasets import load_dataset

# load fineweb-edu with parallel processing
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=64, cache_dir="/your/cache/path")

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=64, cache_dir="/your/cache/path")
```

## Training Recipes

Here's an example of training a 340M Transformer model with a LLaMA-like architecture from scratch on a 100BT subset of the Fineweb-edu corpus ~~in streaming mode~~. (Do not use streaming mode if you are concerned about resuming training.)

> [!WARNING]
> If the dataset is not downloaded beforehand, the streaming mode will attempt to fetch it from a remote server and download it on-the-fly, which can be highly unstable during training due to network issues.  
> For stable training, ensure the dataset is downloaded locally (see [**Dataset Preparation**](#dataset-preparation)). Otherwise, we assume you are only testing the new corpus.

```sh
bash train.sh \
  --job.config_file llmonade/models/llmon.toml \
  --job.dump_folder exp/transformer-340M-4K-10B/batch1.seqlen65536.context4096.warmup1024.update1.steps20480.lr1e-3.cosine \
  --model.config llmonade/configs/transformer_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /cache/ufw/Ultra-FineWeb \
  --training.dataset_name sample-100BT \
  --training.dataset_split en \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1
```

You can specify the number of GPUs by setting the environment variable `NGPU`, which defaults to 8.  
**For single-GPU debugging, set `NGPU=1`.**

We provide several [config files](https://github.com/tilde-research/bento/tree/main/llmonade/configs/) for different models.
By default, the learning rate is set to 1e-3 with a cosine scheduler and the weight decay is set to 0.1. Other schedulers, such as WSD (wsd), are also supported.

**Key parameters:**
- `--lr_scheduler.decay_ratio`: The proportion of the steps allocated to the decay phase. The learning rate will remain stable after the warmup period and only start decaying during the last `decay_ratio` portion of the total training steps, which is known as the Warmup-Stable-Decay (WSD) schedule.
- `--lr_scheduler.warmup_steps`: The number of steps for the learning rate warmup phase.
- `--training.steps`: Total number of training steps.
- `--training.batch_size`: Batch size per device, must be 1 if `--training.varlen` is set.
- `--training.seq_len`: The length of each sequence in the batch, which is concatenated from multiple samples.
- `--training.context_len`: The max allowed length of a sample. For non-varlen mode, this is equivalent to `seq_len`.
- `--training.varlen`: Whether to conduct variable-length sequence training.
- `--training.gradient_accumulation_steps`: Number of gradient accumulation steps.

> [!WARNING]
> The total number of tokens processed per batch, referred to as `global_batch_size`, is calculated as batch_size × gradient_accumulation_steps × num_gpus.
> Each step processes `global_batch_size * seq_len` tokens.  
> Monitor the value of `global_batch_size`, `warmup_steps`, and `steps` carefully when modifying any of the hyperparameters!

For a detailed explanation of all parameters, run:

```sh
bash llmonade/scripts/local/train.sh -h
```

<details>
<summary>Usage</summary>

```py
options:
  -h, --help            show this help message and exit
  --job.config_file JOB.CONFIG_FILE
                        Job config file
  --job.dump_folder JOB.DUMP_FOLDER
                        Folder to dump job outputs
  --job.description JOB.DESCRIPTION
                        Description of the job
  --job.use_for_integration_test
                        Add this config to the integration test suite
  --job.print_args      Print the args to terminal
  --model.config MODEL.CONFIG
                        Path to the model config
  --model.norm_type MODEL.NORM_TYPE
                        Type of layer normalization to use [layernorm,
                        np_layernorm, rmsnorm, fused_rmsnorm]
  --model.tokenizer_path MODEL.TOKENIZER_PATH
                        Tokenizer path
  --profiling.enable_profiling
                        Whether to enable pytorch profiler
  --profiling.save_traces_folder PROFILING.SAVE_TRACES_FOLDER
                        Trace files location
  --profiling.profile_freq PROFILING.PROFILE_FREQ
                        How often to collect profiler traces, in iterations
  --profiling.enable_memory_snapshot
                        Whether to dump memory snapshot
  --profiling.save_memory_snapshot_folder PROFILING.SAVE_MEMORY_SNAPSHOT_FOLDER
                        Memeory snapshot files location
  --optimizer.name OPTIMIZER.NAME
                        Optimizer to use
  --optimizer.eps OPTIMIZER.EPS
                        Epsilon value for the optimizer.
  --optimizer.fused     Whether the fused implementation(CUDA only) is used.
  --optimizer.scheduler {wsd,cosine,linear}
                        Scheduler to use. Currently supported: wsd, cosine,
                        and linear.
  --optimizer.lr OPTIMIZER.LR
                        Learning rate to use
  --optimizer.min_lr_ratio OPTIMIZER.MIN_LR_RATIO
                        Min lr ratio for lr scheduler
  --optimizer.early_step_in_backward
                        Whether to apply optimizer in the backward. Caution,
                        optimizer_in_backward is not compatible with gradients
                        clipping, users should not call
                        register_post_accumulate_grad_hook after the optimizer
                        is built.
  --training.batch_size TRAINING.BATCH_SIZE
                        Batch size
  --training.seq_len TRAINING.SEQ_LEN
                        Sequence length
  --training.context_len TRAINING.CONTEXT_LEN
                        Max length allowed for each sequence
  --training.varlen     Whether to take sequences of variable length as input
  --training.warmup_steps TRAINING.WARMUP_STEPS
                        Steps for lr scheduler warmup, normally 1/5 of
                        --training.steps
  --training.gradient_accumulation_steps TRAINING.GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients before
                        updating parameters
  --training.steps TRAINING.STEPS
                        How many train steps to run
  --training.max_norm TRAINING.MAX_NORM
                        Max norm for gradient clipping
  --training.skip_nan_inf
                        Skip batch updates when NaN or INF gradients are
                        encountered during training
  --training.dataset TRAINING.DATASET
                        Dataset to use, with comma separated values
  --training.dataset_name TRAINING.DATASET_NAME
                        The name of the dataset config, with comma separated
                        values if provided
  --training.dataset_split enING.DATASET_SPLIT
                        Dataset split to use, with comma separated values if
                        provided
  --training.data_dir TRAINING.DATA_DIR
                        Data dirs to use, with comma separated values if
                        provided
  --training.data_files TRAINING.DATA_FILES
                        Data files to use, with comma separated values if
                        provided
  --training.data_probs TRAINING.DATA_PROBS
                        Data sampling probabilities, with comma separated
                        values if provided
  --training.streaming  Whether to load dataset in streaming mode, used for
                        huge dataset
  --training.num_workers TRAINING.NUM_WORKERS
                        Number of subprocesses to use for data loading. 0
                        means that the data will be loaded in the main
                        process.
  --training.prefetch_factor TRAINING.PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker.2
                        means there will be a total of 2 * num_workers batches
                        prefetched across all workers.
  --training.data_parallel_replicate_degree TRAINING.DATA_PARALLEL_REPLICATE_DEGREE
                        The `data_parallel_replicate_degree` argument
                        specifies the degree of data parallelism for weight
                        replication. When this value is greater than 1,
                        weights will be replicated across
                        `data_parallel_replicate_degree` ranks. If
                        `data_parallel_shard_degree` is also greater than 1,
                        the parallelism method used is HSDP (Hybrid Sharded
                        Data Parallelism). Otherwise, the parallelism method
                        used is DDP (Distributed Data Parallelism). 1 means
                        disabled.
  --training.data_parallel_shard_degree TRAINING.DATA_PARALLEL_SHARD_DEGREE
                        The `data_parallel_shard_degree` argument specifies
                        the degree of data parallelism for weight sharding.
                        When this value is greater than 1, weights will be
                        sharded across `data_parallel_shard_degree` ranks. If
                        `data_parallel_replicate_degree` is also greater than
                        1, the parallelism method used is HSDP (Hybrid Sharded
                        Data Parallelism). Otherwise, the parallelism method
                        used is FSDP (Fully Sharded Data Parallelism). -1
                        means leftover ranks will be used (After
                        DP_REPLICATE/SP/PP). Note that only
                        `data_parallel_shard_degree` can be negative. 1 means
                        disabled.
  --training.enable_cpu_offload
                        Whether to apply CPU offloading of parameters,
                        gradients, and optimizer states in FSDP
  --training.tensor_parallel_degree TRAINING.TENSOR_PARALLEL_DEGREE
                        Tensor Parallelism degree. 1 means disabled.
  --training.disable_loss_parallel
                        Whether to apply loss parallel when sequence parallel
                        is enabled
  --training.mixed_precision_param {bfloat16,float32}
                        torch dtype to use for parameters when applying mixed
                        precision via FSDP. This feature only takes effect
                        when data_parallel_shard_degree > 1
  --training.mixed_precision_reduce {float32}
                        torch dtype to use for reductions when applying mixed
                        precision via FSDP. This feature only takes effect
                        when data_parallel_shard_degree > 1
  --training.compile    Whether to compile the model
  --training.gc_freq TRAINING.GC_FREQ
                        Python garbage control scheduling interval, in steps
  --training.seed TRAINING.SEED
                        Choose the base RNG seed used for training
  --training.deterministic
                        Use deterministic algorithms wherever possible, may be
                        slower
  --metrics.log_freq METRICS.LOG_FREQ
                        How often to log metrics to TensorBoard, in iterations
  --metrics.enable_tensorboard
                        Whether to log metrics to TensorBoard
  --metrics.disable_color_printing
                        Whether to disable color printing in logs
  --metrics.save_tb_folder METRICS.SAVE_TB_FOLDER
                        Folder to dump TensorBoard states
  --metrics.rank_0_only
                        Whether to save TensorBoard metrics only for rank 0 or
                        for all ranks. When pipeline_parallel_degree is > 1,
                        this option uses the 0th rank of the last stage
                        pipeline group, which is the only stage that computes
                        loss metrics.
  --metrics.enable_wandb
                        Whether to log metrics to Weights & Biases
  --experimental.enable_async_tensor_parallel
                        Whether to apply async tensor parallel (currently only
                        effective when compile is enabled)
  --experimental.pipeline_parallel_degree EXPERIMENTAL.PIPELINE_PARALLEL_DEGREE
                        Pipeline Parallelism degree, or number of ranks. 1
                        means disabled. If using looped schedules, this still
                        specifies the number of physical ranks, not the number
                        of stages. Stages per rank are inferred from split
                        points degree, and schedule.
  --experimental.pipeline_parallel_split_points EXPERIMENTAL.PIPELINE_PARALLEL_SPLIT_POINTS [EXPERIMENTAL.PIPELINE_PARALLEL_SPLIT_POINTS ...]
                        Specify comma-separated names of modules to use as the
                        beginning of a split point. e.g. "layers.0,layers.2"
                        will cause the model to be split into 3 stages, the
                        first containing all the layers up to layers.0, the
                        second containing layers.0 and up to layers.2, the
                        third containing layers.2 and all the remaining
                        layers. Note: fully-automated splitting may be enabled
                        in the future, but currently the split points must be
                        specified manually.
  --experimental.pipeline_parallel_schedule EXPERIMENTAL.PIPELINE_PARALLEL_SCHEDULE
                        Specify the Pipeline Parallel schedule to use. The
                        supported schedules are: https://github.com/pytorch/py
                        torch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/to
                        rch/distributed/pipelining/schedules.py#L2161. The
                        schedule must be compatible with the split points and
                        stages_per_rank. Looped schedules (e.g.
                        Interleaved1F1B) require specifying
                        pipeline_parallel_degree = number of ranks, and
                        split_points = number of stages - 1
  --experimental.pipeline_parallel_schedule_csv EXPERIMENTAL.PIPELINE_PARALLEL_SCHEDULE_CSV
                        Specify the path to the pipeline parallel schedule csv
                        file to use. The pipeline_parallel_schedule argument
                        must be either PipelineScheduleSingle,
                        PipelineScheduleMulti, or _PipelineScheduleRuntime.
  --experimental.pipeline_parallel_microbatches EXPERIMENTAL.PIPELINE_PARALLEL_MICROBATCHES
                        How many microbatches to split the global training
                        batch into when using pipeline parallelism. The global
                        training batch size must be evenly divisible by the
                        number of microbatches. The default value will be the
                        number of pipeline stages, if unspecified.
  --experimental.enable_compiled_autograd
                        Enable CompiledAutograd to compile the backward.
  --experimental.context_parallel_degree EXPERIMENTAL.CONTEXT_PARALLEL_DEGREE
                        Context parallelism degree. 1 means disabled.
  --experimental.context_parallel_rotate_method EXPERIMENTAL.CONTEXT_PARALLEL_ROTATE_METHOD
                        The collective to use in context parallel SDPA for kv
                        shards exchange. 'allgather' means to all-gather all
                        kv shards on ranks after the first sub-SDPA
                        computation, 'alltoall' means to all-to-all shuffle
                        the kv shards. The default value is 'allgather'.
  --checkpoint.enable_checkpoint
                        Whether to enable checkpoint
  --checkpoint.folder CHECKPOINT.FOLDER
                        The folder to store the checkpoints. When
                        enable_checkpoint is set to true, checkpoints will be
                        in {--job.dump_folder}/{--checkpoint.folder}.
  --checkpoint.interval_type CHECKPOINT.INTERVAL_TYPE
                        Checkpointing interval unit of measurement ['step',
                        'seconds']
  --checkpoint.interval CHECKPOINT.INTERVAL
                        Checkpointing interval, in steps or seconds depending
                        on --checkpoint.interval_type
  --checkpoint.model_weights_only
                        When model_weights_only=True, only model weights will
                        be saved at the end of training. With this,
                        checkpoints can be loaded using `torch.load(...,
                        weights_only=True)` after conversion. When
                        model_weights_only=False, the full checkpoint will be
                        saved. A full checkpoint includes model, optimizer and
                        train_state, which can be used to resume training. The
                        default value is false.
  --checkpoint.export_dtype {float16,bfloat16,float32}
                        Converts to the specified precision when training
                        completes and model_weights_only=true. Currently
                        supports float32, float16, and bfloat16. The default
                        value is float32.
  --checkpoint.create_seed_checkpoint
                        Initializes the full model without applying
                        parallelisms, and then saves it as a seed checkpoint.
                        Note: requires user to call train.py without
                        specifying any parallelisms, e.g. NGPU=1. Could be
                        implemented as a separate script, but this way shares
                        more code.
  --checkpoint.async_mode CHECKPOINT.ASYNC_MODE
                        Which async checkpoint mode to use. Currently there
                        are 3 different modes. 1. "disabled": synchronized
                        checkpointing will be used. 2. "async":
                        torch.distributed.checkpoint.async_save will be used.
                        1. "async_with_pinned_mem": this option utilizes a
                        dedicated pinned memory space and creates a separate
                        process for faster GPU->CPU transfer performance and
                        eliminating GIL contention. The cost is increased CPU
                        memory usage. If insufficient CPU memory is available,
                        performance may degrade due to memory paging. For most
                        users, "async" should suffice as the performance
                        overhead is typically small (on the order of tens of
                        seconds) compared to checkpointing frequency. This
                        mode can be employed to pursue near-zero checkpointing
                        times (e.g., < 1 second) given appropriate hardware
                        support such as ample CPU memory and fast PCIe.
                        "disabled" is the default mode.
  --checkpoint.keep_latest_k CHECKPOINT.KEEP_LATEST_K
                        Keeps only the latest k checkpoints, and purging older
                        ones. If 0, keep all checkpoints. 0 is the default
                        value.
  --checkpoint.load_step CHECKPOINT.LOAD_STEP
                        Load the checkpoint at the specified step. If -1, load
                        the latest checkpoint.
  --float8.enable_float8_linear
                        If true, swaps `torch.nn.Linear` with `Float8Linear`.
                        This feature requires you to install 'torchao' which
                        can be found here: https://github.com/pytorch/ao
  --float8.enable_fsdp_float8_all_gather
                        Whether enable float8 all-gather in FSDP
  --float8.precompute_float8_dynamic_scale_for_fsdp
                        Whether precompute float8 scales dynamically for FSDP
  --float8.scaling_type_input {dynamic,delayed}
                        float8 scaling for input, dynamic (default) or delayed
  --float8.scaling_type_weight FLOAT8.SCALING_TYPE_WEIGHT
                        float8 scaling for input, dynamic (default) or delayed
  --float8.scaling_type_grad_output FLOAT8.SCALING_TYPE_GRAD_OUTPUT
                        float8 scaling for input, dynamic (default) or delayed
  --comm.init_timeout_seconds COMM.INIT_TIMEOUT_SECONDS
                        Timeout for communication operations, during
                        initialization and first train step.
  --comm.train_timeout_seconds COMM.TRAIN_TIMEOUT_SECONDS
                        Timeout for communication operations after the first
                        train step -- usually a tighter bound than during
                        initialization.
  --comm.trace_buf_size COMM.TRACE_BUF_SIZE
                        Flight recorder ring buffer size, >0 means recording
                        by default, 0 means disabled
  --memory_estimation.enabled
                        Whether to estimate memory usage for FSDP
  --memory_estimation.disable_fake_mode
                        Whether to estimate memory under FakeTensorMode
```
</details>

### Training with `torch.compile`

Starting from `torch 2.0`, `torch.compile` has been introduced as a new feature to seamlessly accelerate training processes.
In `llmonade`, one can simply enable `torch.compile` by adding `--training.compile` flag to your training script.

However, `bento` has integrated numerous fused kernels for acceleration, which may potentially conflict with `torch.compile`.
We are actively working on resolving these issues to make compilation transparent to users.
In the meantime, please ensure you are using the latest dependencies.

Specifically, **we recommend using `torch>=2.6` and `triton>=3.0`**.

### Training with multiple datasets

If you wish to train a model with all-round capabilities (e.g., code, math, and multilingual ability), it's necessary to train on multiple datasets.
`llmonade` allows training with multiple datasets easily.
For example, you can specify the following arguments to train on 6 datasets with different proportions:

```sh
  --training.dataset /cache/ufw/Ultra-FineWeb,opencsg/Fineweb-Edu-Chinese-V2.1,OpenCoder-LLM/opc-fineweb-code-corpus,math-ai/AutoMathText,EleutherAI/proof-pile-2,OpenCoder-LLM/opc-fineweb-math-corpus   \
  --training.data_probs 0.6,0.15,0.15,0.014,0.058,0.028     \
```

### Resuming Training

#### From a Local Checkpoint

To resume training from a local checkpoint:

```sh
# Resume from the latest checkpoint
bash llmonade/scripts/local/train.sh --checkpoint.load_step -1

# Resume from a specific checkpoint step
bash llmonade/scripts/local/train.sh --checkpoint.load_step 50
```

### ~Finalizing training~

> [!NOTE]  
> We have done this conversion automatically in the training script since our latest updates.

Once training is complete, you may want to convert the distributed checkpoints (DCPs) into the 🤗 format for broader use.
To facilitate this, we provide a straightforward conversion script:

```sh
python -m llmonade.scripts.helpers.convert_dcp_to_hf --path <path_to_model> --step <step> --config <path_to_config> --tokenizer <path_to_tokenizer>
```
After this, your model will be in the 🤗 format, ready to be shared or deployed.
You can then easily publish your model using the `huggingface_hub` for wider accessibility.

### Continual training

If you wish to build upon a strong pre-trained model (in 🤗 format) and continue training, we also offer a script to convert the 🤗 format model back into DCP format.
This allows you to seamlessly resume training with `llmonade`.
```sh
python -m llmonade.scripts.helpers.convert_hf_to_dcp --model <path_to_hf> --checkpoint <path_to_dcp/checkpoint/step-0>
```
Here, `<path_to_dcp>` is the directory where your distributed checkpoints will be stored.
The checkpoint is intentionally saved at `<step-0>` within the checkpoint folder to ensure it is loadable by `llmonade` during the initial training step, similar to how a seed checkpoint is handled.

Once the conversion is complete, you can proceed with training using `llmonade` as usual, continuing from where the pretrained model left off.

## Multi-node training

If you have access to multi-node GPUs, consider leveraging them for optimal performance.
This process is straightforward and well-documented in the PyTorch [docs](https://pytorch.org/docs/stable/elastic/run.html).

To set up multi-node training:
* Set the environment variables `MASTER_ADDR=<ip>` and `MASTER_PORT=<port>` before running the training script across all nodes.
* If you're using a job scheduler like Slurm, it will handle these variables for you.

`torchtitan` provides a [Slurm script](https://github.com/pytorch/torchtitan/blob/main/multinode_trainer.slurm) for multi-node training, which you can use as a reference or starting point.

## Multi-GPU Evaluation with Hugging Face accelerate 🚀

To perform data-parallel evaluation (where each GPU loads a separate full copy of the model), we leverage the accelerate launcher as follows:
```sh
$ PATH='fla-hub/gla-1.3B-100B'
$ accelerate launch -m llmonade.evals.harness --model hf \
    --model_args pretrained=$PATH,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config  
```

If a GPU can't load a full copy of the model, please refer to [this link](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate) for FSDP settings.

> [!Tip]
> If you are using `lm-evaluation-harness` as an external library and can't find (almost) any tasks available, before calling `lm_eval.evaluate()` or `lm_eval.simple_evaluate()`, simply run the following to load the library's stock tasks!
```py
>>> from lm_eval.tasks import TaskManager; TaskManager().initialize_tasks()
```

## Citation

Here are the original citations for `flame`, `torchtitan`, and `lm-evaluation-harness`.

```bib
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```

```bib
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```

```bib
@software{yang2025flame,
  title  = {Flame: Flash Language Modeling Made Easy},
  author = {Zhang, Yu and Yang, Songlin},
  url    = {https://github.com/fla-org/flame},
  month  = jan,
  year   = {2025}
}
```
