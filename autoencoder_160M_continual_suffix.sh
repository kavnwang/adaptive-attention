#!/bin/bash
# autoencoder_160M_continual.sh

set -e  # Exit on error

# Activate local virtualenv if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_PATH="$SCRIPT_DIR/.venv/bin/activate"
if [ -f "$ACTIVATE_PATH" ]; then
  # shellcheck source=/dev/null
  source "$ACTIVATE_PATH"
else
  echo "Virtualenv not found at $ACTIVATE_PATH" >&2
  echo "Create it with: uv sync  (or: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt)" >&2
  # continue without exiting, in case system python is intended
fi

# Create necessary directories
mkdir -p exp/autoencoder_continual_160M_suffix
mkdir -p logs

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR=~/tmp/triton_cache_user_owned
mkdir -p $TRITON_CACHE_DIR
# Make vendored bento importable without network installs
export PYTHONPATH="$(pwd)/3rdparty/bento:${PYTHONPATH}"

# Log file with timestamp
LOGFILE="logs/train_autoencoder_160M_continual_$(date +%Y%m%d_%H%M%S).log"

echo "Starting AE continual training - logging to $LOGFILE"

# Notes:
# - We initialize 12 transformer layers from the 160M transformer checkpoint (latest -> step-30000),
#   and the compressor/upsampler from the AE checkpoint (latest -> step-30000).
# - Multiple pretrained sources are provided as comma-separated lists; train.py applies them in order.

torchrun --nproc_per_node=1 --nnodes=1 -m llmonade.train \
  --job.config_file llmonade/configs/llmon.toml \
  --job.dump_folder exp/autoencoder_continual_160M_suffix \
  --model.config llmonade/configs/autoencoder/autoencoder_160m_continual_suffix.json \
  --model.tokenizer_path EleutherAI/pythia-160m \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --optimizer.weight_decay 0.1 \
  --lr_scheduler.warmup_steps 1000 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 8 \
  --training.seq_len 16384 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 10000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset manu/project_gutenberg \
  --training.streaming \
  --training.dataset_split en \
  --training.mixed_precision_param bfloat16 \
  --training.num_workers 8 \
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

echo "AE continual training complete!"
