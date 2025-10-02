#!/bin/bash
#SBATCH --job-name=1layer_2step_200K
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --time=36:00:00
#SBATCH --output=logs/1layer_2step_200K_%j.log
#SBATCH --exclude=worker-4,worker-5

# Get head node info
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"
echo "Slurm job ID: $SLURM_JOB_ID"

# Important environment variables
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# Network configuration - match the working script
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=23
export NCCL_TIMEOUT=900000

# If on AWS, uncomment these lines
# export FI_PROVIDER="efa"
# export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# Change to working directory
cd $SLURM_SUBMIT_DIR

export TRITON_CACHE_DIR=~/tmp/triton_cache_user_owned
mkdir -p $TRITON_CACHE_DIR
chmod 777 $TRITON_CACHE_DIR

# Launch training for 1-layer memory model on 2-step digit lookup data
# Using the memory_1layer_permute.json config for permutation-style task
# Applying digit lookup masking to train only on value prediction
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun torchrun --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_id=112 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node_ip:29500" \
  -m llmonade.train \
  --job.config_file llmonade/configs/llmon.toml \
  --job.dump_folder exp/permutations/1layer_2step_200K \
  --model.config llmonade/configs/transformer/permutations/transformer_1_layer.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 2e-4 \
  --optimizer.weight_decay 0.06 \
  --lr_scheduler.warmup_steps 1000 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 128 \
  --training.seq_len 16 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 30000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset parquet \
  --training.data_dir experiments/digit_lookup/datasets/synthetic_digit_lookup_data_2steps_200000 \
  --training.dataset_split train \
  --training.streaming \
  --training.apply_digit_lookup_mask \
  --training.mixed_precision_param bfloat16 \
  --training.num_workers 16 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.enable_checkpoint \
  --checkpoint.interval 5000 \
  --checkpoint.export_dtype bfloat16 \
  --checkpoint.load_step 0 \
  --metrics.log_freq 10 \
  --comm.train_timeout_seconds 240
