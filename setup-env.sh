#!/bin/bash
# Setup script for RunPod - run this after container restart

set -e

echo "🚀 Setting up environment..."

# Install uv to workspace if not already installed
if [ ! -f "/workspace/bin/uv" ]; then
    echo "📦 Installing uv to /workspace/bin..."
    mkdir -p /workspace/bin
    curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path
    mv ~/.local/bin/uv /workspace/bin/uv
    echo "✅ uv installed to /workspace/bin"
else
    echo "✅ uv already installed"
fi

# Add to PATH
export PATH="/workspace/bin:$PATH"

# Install jq if not present
if ! command -v jq &> /dev/null; then
    echo "📦 Installing jq..."
    apt-get update -qq && apt-get install -y jq > /dev/null 2>&1
    echo "✅ jq installed"
fi

# Sync packages if .venv doesn't have everything
cd /workspace/adaptive-attention
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/python" ]; then
    echo "📦 Installing packages with uv sync..."
    uv sync
    echo "✅ Packages installed"
else
    echo "✅ Virtual environment ready"
    echo "💡 Run 'uv sync' if you need to update packages"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run training test:"
echo "  ./test-train.sh"

