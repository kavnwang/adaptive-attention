#!/bin/bash
#SBATCH --job-name=memory_340M_hashhop
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=logs/memory_340M_hashhop_%j.log

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

# First generate the hash-hop dataset if it doesn't exist
if [ ! -f "hashhop_experiments/datasets/hashhop_1hop_training_parquet/train/data.parquet" ]; then
    echo "Generating hash-hop dataset in parquet format..."
    python hashhop_experiments/eval_scripts/create_hashhop_1hop_training_parquet.py --n_samples 100000
fi

# Launch training on hash-hop dataset
# Using memory MLP router model to learn hash lookups
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun torchrun --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_id=106 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node_ip:29500" \
  -m llmonade.train \
  --job.config_file llmonade/configs/llmon.toml \
  --job.dump_folder exp/memory_340M_hashhop \
  --model.config llmonade/configs/memory/memory_layer_340M_standard.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-4 \
  --optimizer.weight_decay 0.01 \
  --lr_scheduler.warmup_steps 500 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 8 \
  --training.seq_len 8192 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset parquet \
  --training.data_dir hashhop_experiments/datasets/hashhop_1hop_training_parquet \
  --training.dataset_split train \
  --training.streaming \
  --training.mixed_precision_param bfloat16 \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.enable_checkpoint \
  --checkpoint.interval 2000 \
  --checkpoint.export_dtype bfloat16 \
  --checkpoint.load_step 0 \
  --metrics.log_freq 10 \
  --comm.train_timeout_seconds 240
