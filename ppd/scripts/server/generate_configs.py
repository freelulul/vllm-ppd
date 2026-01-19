#!/usr/bin/env python3
"""
Generate server startup scripts for all 17 GPU configurations.

GPU Configurations:
- Pure Replica: 4PD
- Pure Disaggregated (9): 1P+3D, 1P+2D+1pD, 1P+1D+2pD, 1P+3pD, 2P+2D, 2P+1D+1pD, 2P+2pD, 3P+1D, 3P+1pD
- Hybrid (7): 1PD+1P+2D, 1PD+1P+1D+1pD, 1PD+1P+2pD, 1PD+2P+1D, 1PD+2P+1pD, 2PD+1P+1D, 2PD+1P+1pD

Machine Types:
- P: Prefill-only (kv_producer)
- D: Decode-only (kv_consumer)
- pD: Prefill-capable Decode (kv_consumer with append_prefill capability)
- PD/R: Replica (full model, no KV transfer)
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class ServerConfig:
    """Configuration for a single server instance."""
    gpu_id: int
    server_type: str  # "P", "D", "pD", "R"
    port: int
    kv_port: int = 0  # 0 for replica


# All 17 configurations
CONFIGURATIONS = {
    # Pure Replica
    "4R": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "R", 8400),
        ServerConfig(2, "R", 8500),
        ServerConfig(3, "R", 8600),
    ],

    # Pure Disaggregated
    "1P_3D": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "D", 8200, 14580),
        ServerConfig(2, "D", 8201, 14581),
        ServerConfig(3, "D", 8202, 14582),
    ],
    "1P_2D_1pD": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "D", 8200, 14580),
        ServerConfig(2, "D", 8201, 14581),
        ServerConfig(3, "pD", 8202, 14582),
    ],
    "1P_1D_2pD": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "D", 8200, 14580),
        ServerConfig(2, "pD", 8201, 14581),
        ServerConfig(3, "pD", 8202, 14582),
    ],
    "1P_3pD": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "pD", 8200, 14580),
        ServerConfig(2, "pD", 8201, 14581),
        ServerConfig(3, "pD", 8202, 14582),
    ],
    "2P_2D": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "P", 8101, 14581),
        ServerConfig(2, "D", 8200, 14580),
        ServerConfig(3, "D", 8201, 14582),
    ],
    "2P_1D_1pD": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "P", 8101, 14581),
        ServerConfig(2, "D", 8200, 14580),
        ServerConfig(3, "pD", 8201, 14582),
    ],
    "2P_2pD": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "P", 8101, 14581),
        ServerConfig(2, "pD", 8200, 14580),
        ServerConfig(3, "pD", 8201, 14582),
    ],
    "3P_1D": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "P", 8101, 14580),
        ServerConfig(2, "P", 8102, 14581),
        ServerConfig(3, "D", 8200, 14582),
    ],
    "3P_1pD": [
        ServerConfig(0, "P", 8100, 14579),
        ServerConfig(1, "P", 8101, 14580),
        ServerConfig(2, "P", 8102, 14581),
        ServerConfig(3, "pD", 8200, 14582),
    ],

    # Hybrid (Replica + Disaggregated)
    "1R_1P_2D": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "P", 8100, 14579),
        ServerConfig(2, "D", 8200, 14580),
        ServerConfig(3, "D", 8201, 14581),
    ],
    "1R_1P_1D_1pD": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "P", 8100, 14579),
        ServerConfig(2, "D", 8200, 14580),
        ServerConfig(3, "pD", 8201, 14581),
    ],
    "1R_1P_2pD": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "P", 8100, 14579),
        ServerConfig(2, "pD", 8200, 14580),
        ServerConfig(3, "pD", 8201, 14581),
    ],
    "1R_2P_1D": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "P", 8100, 14579),
        ServerConfig(2, "P", 8101, 14580),
        ServerConfig(3, "D", 8200, 14581),
    ],
    "1R_2P_1pD": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "P", 8100, 14579),
        ServerConfig(2, "P", 8101, 14580),
        ServerConfig(3, "pD", 8200, 14581),
    ],
    "2R_1P_1D": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "R", 8400),
        ServerConfig(2, "P", 8100, 14579),
        ServerConfig(3, "D", 8200, 14580),
    ],
    "2R_1P_1pD": [
        ServerConfig(0, "R", 8300),
        ServerConfig(1, "R", 8400),
        ServerConfig(2, "P", 8100, 14579),
        ServerConfig(3, "pD", 8200, 14580),
    ],
}


def generate_script(config_name: str, servers: List[ServerConfig]) -> str:
    """Generate a bash script for the given configuration."""

    # Count server types
    p_count = sum(1 for s in servers if s.server_type == "P")
    d_count = sum(1 for s in servers if s.server_type == "D")
    pd_count = sum(1 for s in servers if s.server_type == "pD")
    r_count = sum(1 for s in servers if s.server_type == "R")

    # Determine if this is a pure replica config
    is_pure_replica = r_count == 4
    is_pure_disagg = r_count == 0
    is_hybrid = r_count > 0 and r_count < 4

    # Architecture description
    arch_parts = []
    if p_count > 0:
        arch_parts.append(f"{p_count}P")
    if d_count > 0:
        arch_parts.append(f"{d_count}D")
    if pd_count > 0:
        arch_parts.append(f"{pd_count}pD")
    if r_count > 0:
        arch_parts.append(f"{r_count}R")
    arch_str = " + ".join(arch_parts)

    # GPU allocation description
    gpu_alloc = []
    for s in servers:
        gpu_alloc.append(f"GPU{s.gpu_id}: {s.server_type} (port {s.port})")

    script = f'''#!/bin/bash
# ============================================================================
# Start Script for vLLM Configuration: {config_name}
#
# Architecture: {arch_str}
# '''

    for desc in gpu_alloc:
        script += f"\n#   {desc}"

    script += f'''
#
# Usage: ./start_{config_name}.sh
# ============================================================================

set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source common functions
source "$SCRIPT_DIR/common.sh" 2>/dev/null || true

# Check environment
check_environment || exit 1

# Check GPU availability (GPUs 0-3)
echo ""
check_gpu_availability "0,1,2,3"
gpu_status=$?
if [ $gpu_status -eq 1 ]; then
    exit 1
elif [ $gpu_status -eq 2 ]; then
    force_cleanup
fi

# Source unified configuration
source "$SCRIPT_DIR/config.sh"

# Server parameters
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs/{config_name}"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

echo "=============================================="
echo "Starting vLLM Configuration: {config_name}"
echo "Architecture: {arch_str}"
echo "=============================================="
'''

    # Add NCCL settings if there are disaggregated servers
    if not is_pure_replica:
        script += '''
# NCCL settings for multi-GPU P2P
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=32
'''

    script += '''
# Cleanup existing processes using the comprehensive cleanup script
if [ -f "$SCRIPT_DIR/cleanup_all.sh" ]; then
    bash "$SCRIPT_DIR/cleanup_all.sh"
else
    echo "Cleaning up existing processes..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "comprehensive_proxy" 2>/dev/null || true
    pkill -9 -f "disagg_proxy" 2>/dev/null || true
    pkill -9 -f "simple_replica_proxy" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "$MODEL_NAME" 2>/dev/null || true
    sleep 10
fi
'''

    # Start proxy if not pure replica
    if not is_pure_replica:
        script += f'''
# Start Proxy
echo "[1/{len(servers) + 1}] Starting Proxy..."
PROXY_PORT=10001
PROXY_CONTROL_PORT=30001
python "$SRC_DIR/comprehensive_proxy.py" \\
    --config {config_name} \\
    --http-port $PROXY_PORT \\
    --zmq-port $PROXY_CONTROL_PORT \\
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2
'''
    else:
        script += f'''
# Start Replica Proxy (simple load balancer)
echo "[1/{len(servers) + 1}] Starting Replica Proxy..."
PROXY_PORT=10002
python "$SRC_DIR/simple_replica_proxy.py" \\
    --http-port $PROXY_PORT \\
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2
'''

    # Start each server
    step = 2
    for s in servers:
        server_name = f"{s.server_type}{s.gpu_id}"

        if s.server_type == "R":
            # Replica server (no KV transfer)
            script += f'''
# Start Replica {s.gpu_id} (GPU {s.gpu_id})
echo "[{step}/{len(servers) + 1}] Starting Replica (GPU {s.gpu_id}, port {s.port})..."
CUDA_VISIBLE_DEVICES={s.gpu_id} vllm serve "$MODEL_PATH" \\
    --host 0.0.0.0 --port {s.port} \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEMORY_UTIL \\
    --trust-remote-code --disable-log-requests \\
    --enable-prefix-caching \\
    > "$LOG_DIR/replica{s.gpu_id}.log" 2>&1 &
'''
        elif s.server_type == "P":
            # Prefill server (kv_producer)
            script += f'''
# Start Prefill (GPU {s.gpu_id})
echo "[{step}/{len(servers) + 1}] Starting Prefill (GPU {s.gpu_id}, port {s.port})..."
KV_CONFIG='{{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":{s.kv_port},"kv_connector_extra_config":{{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"{s.port}","send_type":"PUT_ASYNC"}}}}'
CUDA_VISIBLE_DEVICES={s.gpu_id} vllm serve "$MODEL_PATH" \\
    --host 0.0.0.0 --port {s.port} \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEMORY_UTIL \\
    --trust-remote-code --disable-log-requests \\
    --enable-prefix-caching \\
    --kv-transfer-config "$KV_CONFIG" \\
    > "$LOG_DIR/prefill{s.gpu_id}.log" 2>&1 &
'''
        else:  # D or pD
            # Decode server (kv_consumer)
            type_name = "Decode" if s.server_type == "D" else "PPD-Decode"
            log_name = "decode" if s.server_type == "D" else "ppd_decode"
            script += f'''
# Start {type_name} (GPU {s.gpu_id})
echo "[{step}/{len(servers) + 1}] Starting {type_name} (GPU {s.gpu_id}, port {s.port})..."
KV_CONFIG='{{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":10000000000,"kv_port":{s.kv_port},"kv_connector_extra_config":{{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"{s.port}","send_type":"PUT_ASYNC"}}}}'
CUDA_VISIBLE_DEVICES={s.gpu_id} vllm serve "$MODEL_PATH" \\
    --host 0.0.0.0 --port {s.port} \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEMORY_UTIL \\
    --trust-remote-code --disable-log-requests \\
    --enable-prefix-caching \\
    --kv-transfer-config "$KV_CONFIG" \\
    > "$LOG_DIR/{log_name}{s.gpu_id}.log" 2>&1 &
'''
        step += 1

    # Wait for servers
    all_ports = [s.port for s in servers]
    script += f'''
# Wait for servers
echo ""
echo "Waiting for servers to be ready..."
MAX_WAIT=300
WAITED=0

for PORT in {" ".join(str(p) for p in all_ports)}; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 2; WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for port $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting for port $PORT... ($WAITED s)"
    done
    echo "  Port $PORT: READY"
done

sleep 3
'''

    if not is_pure_replica:
        script += '''echo "  Proxy: $(curl -s http://localhost:$PROXY_PORT/status)"
'''

    script += f'''
echo ""
echo "=============================================="
echo "Configuration {config_name} ready!"
echo "Architecture: {arch_str}"
echo "=============================================="
'''

    for s in servers:
        script += f'''echo "{s.server_type}: http://localhost:{s.port} (GPU {s.gpu_id})"
'''

    if not is_pure_replica:
        script += '''echo ""
echo "Proxy: http://localhost:$PROXY_PORT"
'''

    script += f'''echo ""
echo "To stop: ./scripts/server/stop_{config_name}.sh"
echo "=============================================="
'''

    return script


def generate_stop_script(config_name: str) -> str:
    """Generate a stop script for the given configuration.

    All stop scripts simply call cleanup_all.sh for consistency.
    """
    return f'''#!/bin/bash
# Stop script for configuration: {config_name}
# Note: All stop scripts use the unified cleanup_all.sh

SCRIPT_DIR="$(dirname "$0")"
bash "$SCRIPT_DIR/cleanup_all.sh"
'''


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating server scripts for 17 configurations...")

    for config_name, servers in CONFIGURATIONS.items():
        # Generate start script
        start_script = generate_script(config_name, servers)
        start_path = os.path.join(script_dir, f"start_{config_name}.sh")
        with open(start_path, "w") as f:
            f.write(start_script)
        os.chmod(start_path, 0o755)
        print(f"  Generated: start_{config_name}.sh")

        # Generate stop script
        stop_script = generate_stop_script(config_name)
        stop_path = os.path.join(script_dir, f"stop_{config_name}.sh")
        with open(stop_path, "w") as f:
            f.write(stop_script)
        os.chmod(stop_path, 0o755)

    # Create logs directories
    logs_dir = os.path.join(os.path.dirname(script_dir), "..", "logs")
    for config_name in CONFIGURATIONS:
        config_log_dir = os.path.join(logs_dir, config_name)
        os.makedirs(config_log_dir, exist_ok=True)

    print(f"\nGenerated {len(CONFIGURATIONS)} configurations.")
    print("\nConfigurations:")
    for name, servers in CONFIGURATIONS.items():
        p = sum(1 for s in servers if s.server_type == "P")
        d = sum(1 for s in servers if s.server_type == "D")
        pd = sum(1 for s in servers if s.server_type == "pD")
        r = sum(1 for s in servers if s.server_type == "R")
        parts = []
        if r: parts.append(f"{r}R")
        if p: parts.append(f"{p}P")
        if d: parts.append(f"{d}D")
        if pd: parts.append(f"{pd}pD")
        print(f"  {name}: {' + '.join(parts)}")


if __name__ == "__main__":
    main()
