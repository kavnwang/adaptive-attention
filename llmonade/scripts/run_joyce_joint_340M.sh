#!/usr/bin/env bash
# ==============================================================================
# run_joyce_joint_340M.sh
# Continued (joint) training for the 340M base + Joyce.
# ==============================================================================

set -euo pipefail

# Derive repository root relative to this script and ensure we run from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "[Joyce Joint 340M] Starting setup..."
git submodule update --init --recursive

# Environment consistency with other launchers
export LOGLEVEL="${LOGLEVEL:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,en,eth,em,bond}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-900000}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/tmp/triton_cache_user_owned}"

mkdir -p "$TRITON_CACHE_DIR" logs
chmod 777 "$TRITON_CACHE_DIR" || true

LOGFILE="logs/joyce_joint_340M_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

# Choose seq_len S such that total per-sample length = 2*S.
S=4096  # with base max_position_embeddings=8192, this keeps 2*S within range

ENABLE_WANDB=${ENABLE_WANDB:-0}
WANDB_ARGS=()
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  echo "[Joyce Joint 340M] WandB logging enabled."
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    WANDB_ARGS+=(--wandb_mode "${WANDB_MODE}")
  fi
  if [[ -n "${WANDB_TAGS:-}" ]]; then
    OLD_IFS=$IFS
    IFS=', '
    read -r -a _wandb_tags <<< "${WANDB_TAGS}"
    IFS=$OLD_IFS
    if [[ ${#_wandb_tags[@]} -gt 0 ]]; then
      WANDB_ARGS+=(--wandb_tags)
      for tag in "${_wandb_tags[@]}"; do
        WANDB_ARGS+=("${tag}")
      done
    fi
    unset _wandb_tags
  fi
  WANDB_ARGS=(--enable_wandb "${WANDB_ARGS[@]}")
else
  echo "[Joyce Joint 340M] WandB logging disabled. Set ENABLE_WANDB=1 to enable."
fi

python train_joyce_joint.py \
  --repo_root . \
  --model_name_or_path exp/transformer_340M/checkpoint-30000 \
  --tokenizer_name_or_path fla-hub/transformer-1.3B-100B \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset_split train \
  --text_key text \
  --seq_len $S \
  --layer_L 12 \
  --num_compressed_states 512 \
  --warmup_steps 1000 \
  --train_steps 30000 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-4 \
  --save_every 1000 \
  --save_dir checkpoints_joint_340M \
  --compressor_ckpt exp/joyce_ae_340M/checkpoint-30000/compressor.pt \
  --upsampler_ckpt exp/joyce_ae_340M/checkpoint-30000/upsampler.pt \
  --fp16 \
  "${WANDB_ARGS[@]}" "$@" 2>&1 | tee "$LOGFILE"

echo "[Joyce Joint 340M] Done."
