#!/bin/bash
# ==============================================================================
# run_joyce_joint.sh
# Launches the continued (joint) training phase for Joyce.
# Must be run from the adaptive-attention/ repo root.
# ==============================================================================

set -e  # exit on first error
set -o pipefail

echo "[Joyce Joint Training] Starting setup..."

# ------------------------------------------------------------------------------
# 1. Ensure submodules are initialized
# ------------------------------------------------------------------------------
git submodule update --init --recursive

# ------------------------------------------------------------------------------
# 2. (Optional) Dependency installation
# Uncomment if you use uv-based environment management as in the repo README.
# ------------------------------------------------------------------------------
# echo "[Joyce Joint Training] Installing dependencies..."
# uv sync
# uv add --editable 3rdparty/bento

# ------------------------------------------------------------------------------
# 3. Launch training
# Replace <> placeholders below with your actual paths/checkpoints.
# ------------------------------------------------------------------------------

ENABLE_WANDB=${ENABLE_WANDB:-0}
WANDB_ARGS=()
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  echo "[Joyce Joint Training] WandB logging enabled."
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
  echo "[Joyce Joint Training] WandB logging disabled. Set ENABLE_WANDB=1 to enable."
fi

python train_joyce_joint.py \
  --repo_root . \
  --model_name_or_path exp/joyce_ae_160M/checkpoint-30000 \
  --tokenizer_name_or_path EleutherAI/pythia-160m \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset_split train \
  --text_key text \
  --seq_len 8192 \
  --layer_L 12 \
  --num_compressed_states 256 \
  --warmup_steps 1000 \
  --train_steps 30000 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-4 \
  --save_every 500 \
  --save_dir checkpoints_joint \
  --compressor_ckpt exp/joyce_ae_160M/checkpoint-30000/compressor.pt \
  --upsampler_ckpt exp/joyce_ae_160M/checkpoint-30000/upsampler.pt \
  --fp16 \
  "${WANDB_ARGS[@]}"

echo "[Joyce Joint Training] Done."
