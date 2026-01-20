#!/bin/bash
# ============================================================================
# Unified Configuration for vLLM PPD Servers
# ============================================================================
# All model and path configurations should be defined here.
# Other scripts should source this file.
#
# Environment variable overrides (for model scaling experiments):
# - MODEL_PATH: Override model path before sourcing this file
# - MAX_MODEL_LEN: Override max model length
# - GPU_MEMORY_UTILIZATION: Override GPU memory utilization
# ============================================================================

# Model path - change this when switching models, or set MODEL_PATH env var first
export MODEL_PATH="${MODEL_PATH:-/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B}"

# Model name for pkill (derived from MODEL_PATH if not set)
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B}"

# Model settings (support environment variable override)
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# Project directory
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Log directory
export LOG_DIR="${PROJECT_DIR}/logs"

# Standard ports
export PROXY_HTTP_PORT=10001
export PROXY_ZMQ_PORT=30001

# GPU server base ports
export PREFILL_BASE_PORT=8100
export DECODE_BASE_PORT=8200
export REPLICA_BASE_PORT=8300
