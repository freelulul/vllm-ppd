#!/bin/bash
# ============================================================================
# PPD Setup Script
# Installs vLLM (from vendored source with PPD patch) and PPD dependencies.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "PPD Setup"
echo "=============================================="

# Step 1: Install vLLM from vendored source
echo "[1/2] Installing vLLM from vllm-source/..."
# SETUPTOOLS_SCM_PRETEND_VERSION needed because vllm-source/ is not a standalone git repo
SETUPTOOLS_SCM_PRETEND_VERSION="0.13.0" python -m pip install --no-build-isolation -e "$SCRIPT_DIR/vllm-source"

# Step 2: Install PPD dependencies
echo "[2/2] Installing PPD dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  export MODEL_PATH=/path/to/meta-llama/Llama-3.1-8B"
echo "  ./scripts/server/start_2P_2D.sh"
