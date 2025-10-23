#!/usr/bin/env bash
set -euo pipefail

# Run three training scripts sequentially.
# Usage: bash run_all_160M_continual.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] Running autoencoder_160M_continual_suffix.sh..."
bash ./autoencoder_160M_continual_suffix.sh

echo "[$(ts)] Running transformer_160M.sh..."
bash ./transformer_160M.sh

echo "[$(ts)] Running transformer_160M_continual.sh..."
bash ./transformer_160M_continual.sh

echo "[$(ts)] All three scripts completed successfully."
