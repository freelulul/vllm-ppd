#!/bin/bash
# ============================================================================
# Unified Configuration for vLLM PPD Servers
# ============================================================================
# All model and path configurations should be defined here.
# Other scripts should source this file.
# ============================================================================

# Model path - change this when switching models
export MODEL_PATH="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"

# Model name for pkill (derived from MODEL_PATH)
export MODEL_NAME="Llama-3.1-8B"

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
