"""
Unified configuration for vLLM PPD project.

All model and path configurations should be defined here.
Other modules should import from this file.

Environment variable overrides (for model scaling experiments):
- MODEL_PATH: Override model path
- MAX_MODEL_LEN: Override max model length
- GPU_MEMORY_UTILIZATION: Override GPU memory utilization
"""

import os
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Model configuration
# Change this when switching models, or use environment variables for dynamic override
_DEFAULT_MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
MODEL_NAME = os.environ.get("MODEL_NAME", "Llama-3.1-8B")

# Server ports
PROXY_HTTP_PORT = 10001
PROXY_ZMQ_PORT = 30001
REPLICA_PROXY_PORT = 10002

# GPU server base ports
PREFILL_BASE_PORT = 8100
DECODE_BASE_PORT = 8200
REPLICA_BASE_PORT = 8300

# Timeout settings
REQUEST_TIMEOUT_SEC = 120
WARMUP_TIMEOUT_SEC = 60

# Server settings (support environment variable override for model scaling experiments)
_DEFAULT_MAX_MODEL_LEN = 8192
_DEFAULT_GPU_MEMORY_UTIL = 0.85
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", _DEFAULT_MAX_MODEL_LEN))
GPU_MEMORY_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", _DEFAULT_GPU_MEMORY_UTIL))
