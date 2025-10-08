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

python train_joyce_joint.py \
  --repo_root . \
  --model_name_or_path exp/joyce_ae_160M/checkpoint-30000 \
  --tokenizer_name_or_path EleutherAI/pythia-160m \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset_split train \
  --text_key text \
  --seq_len 1024 \
  --layer_L 16 \
  --num_compressed_states 64 \
  --train_steps 2000 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-4 \
  --save_every 500 \
  --save_dir checkpoints_joint \
  --compressor_ckpt exp/joyce_ae_160M/checkpoint-30000/compressor.pt \
  --upsampler_ckpt exp/joyce_ae_160M/checkpoint-30000/upsampler.pt \
  --fp16

echo "[Joyce Joint Training] Done."
