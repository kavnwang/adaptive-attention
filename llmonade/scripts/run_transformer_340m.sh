#!/usr/bin/env bash
set -euo pipefail

# Derive repository root relative to this script and ensure we run from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Distributed defaults (overridable):
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
RDZV_ID="${RDZV_ID:-101}"
RDZV_PORT="${RDZV_PORT:-29500}"
RDZV_HOST="${RDZV_HOST:-}"

if [[ -z "$RDZV_HOST" ]]; then
  if command -v hostname >/dev/null 2>&1; then
    RDZV_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -z "$RDZV_HOST" ]]; then
      RDZV_HOST="$(hostname -f 2>/dev/null || hostname)"
    fi
  else
    RDZV_HOST="127.0.0.1"
  fi
fi
RDZV_ENDPOINT="${RDZV_HOST}:${RDZV_PORT}"

export LOGLEVEL="${LOGLEVEL:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,en,eth,em,bond}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-900000}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/tmp/triton_cache_user_owned}"

mkdir -p "$TRITON_CACHE_DIR" exp/transformer_340M logs
chmod 777 "$TRITON_CACHE_DIR" || true

LOGFILE="logs/train_340M_$(date +%Y%m%d_%H%M%S).log"
echo "Starting 340M base training - logging to $LOGFILE"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes="$NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$RDZV_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_ENDPOINT" \
  -m llmonade.train \
  --job.config_file llmonade/configs/llmon.toml \
  --job.dump_folder exp/transformer_340M \
  --model.config llmonade/configs/transformer/t340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
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
  --comm.train_timeout_seconds 240 \
  "$@" 2>&1 | tee "$LOGFILE"

echo "Base 340M training complete!"
