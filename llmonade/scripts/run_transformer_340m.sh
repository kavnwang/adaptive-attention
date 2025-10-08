#!/bin/bash
set -e
cd /workspace/adaptive-attention
mkdir -p exp/transformer_340M logs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR=~/tmp/triton_cache_user_owned; mkdir -p "$TRITON_CACHE_DIR"
LOGFILE="logs/train_340M_$(date +%Y%m%d_%H%M%S).log"
echo "Starting 340M base training - logging to $LOGFILE"

torchrun --nproc_per_node=1 --nnodes=1 -m llmonade.train \
  --job.config_file llmonade/configs/llmon.toml \
  --job.dump_folder exp/transformer_340M \
  --model.config llmonade/configs/transformer/transformer_340m.json \
  --model.tokenizer_path EleutherAI/pythia-410m \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --optimizer.weight_decay 0.1 \
  --lr_scheduler.warmup_steps 1000 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 2 \
  --training.seq_len 8192 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 30000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.streaming \
  --training.dataset_name sample-10BT \
  --training.dataset_split train \
  --training.mixed_precision_param bfloat16 \
  --training.num_workers 14 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.enable_checkpoint \
  --checkpoint.interval 6000 \
  --checkpoint.export_dtype bfloat16 \
  --checkpoint.load_step 0 \
  --metrics.log_freq 1 \
  --comm.train_timeout_seconds 240 2>&1 | tee "$LOGFILE"

echo "Base 340M training complete!"
