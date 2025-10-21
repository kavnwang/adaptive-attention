#!/usr/bin/env bash

# Dev bootstrap for LLMonade/adaptive-attention
# - Updates submodules
# - Installs uv if missing
# - Runs uv sync
# - Links local lm-evaluation-harness (and torchtitan if needed)
# - Sets per-repo git user/email

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "==> Updating submodules"
git submodule update --init --recursive

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Error: need curl or wget to install uv" >&2
    exit 1
  fi
else
  echo "==> uv already installed"
fi

# ensure uv on PATH for this shell
export PATH="$HOME/.local/bin:$PATH"

echo "==> Syncing dependencies (uv sync)"
uv sync

echo "==> Linking local lm-evaluation-harness"
if ! uv add --editable 3rdparty/lm-evaluation-harness --frozen; then
  echo "   (lock was frozen; retrying without --frozen)"
  uv add --editable 3rdparty/lm-evaluation-harness
fi

# torchtitan is already part of the uv workspace; only add if not configured
if grep -Eq '^\s*torchtitan\s*=\s*\{[^}]*workspace[[:space:]]*=[[:space:]]*true' pyproject.toml \
   || grep -Eq '^\s*"3rdparty/torchtitan"' pyproject.toml; then
  echo "==> torchtitan managed via uv workspace; skipping uv add"
else
  echo "==> Linking local torchtitan"
  if ! uv add --editable 3rdparty/torchtitan --frozen; then
    echo "   (lock was frozen; retrying without --frozen)"
    uv add --editable 3rdparty/torchtitan
  fi
fi

echo "==> Re-syncing after changes"
uv sync

echo "==> Setting repo git identity"
git config user.name "Kevin Wang"
git config user.email "kavnweng@gmail.com"

echo ""
echo "✅ Dev setup complete."
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
