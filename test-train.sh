#!/bin/bash
# Quick 1-GPU training test

# Activate virtual environment
source .venv/bin/activate

NNODE=1 NGPU=1 LOG_RANK=0 bash llmonade/scripts/local/train.sh \
  --job.config_file llmonade/configs/llmon.toml \
  --job.dump_folder exp/test-run \
  --model.config llmonade/configs/transformer/transformer_test.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.lr 3e-4 \
  --training.batch_size 2 \
  --training.seq_len 512 \
  --training.steps 10 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.streaming \
  --training.num_workers 2 \
  --metrics.log_freq 1 \
  --metrics.enable_wandb false

