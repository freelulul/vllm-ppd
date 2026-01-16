#!/bin/bash
# ============================================================================
# Common functions for vLLM server scripts
# ============================================================================

# Check if GPUs are available (no processes using significant memory)
check_gpu_availability() {
    local required_gpus="$1"  # e.g., "0,1,2,3" or "0 1 2 3"
    local gpus=$(echo "$required_gpus" | tr ',' ' ')

    echo "Checking GPU availability..."

    # Get GPU memory usage
    local gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    if [ -z "$gpu_info" ]; then
        echo "ERROR: nvidia-smi failed. Are GPUs available?"
        return 1
    fi

    local blocked_gpus=""
    local found_vllm=false

    for gpu in $gpus; do
        # Get memory used for this GPU (in MiB)
        local mem_used=$(echo "$gpu_info" | awk -F', ' -v g="$gpu" '$1 == g {print $2}')
        local mem_total=$(echo "$gpu_info" | awk -F', ' -v g="$gpu" '$1 == g {print $3}')

        if [ -z "$mem_used" ]; then
            echo "  GPU $gpu: Not found"
            blocked_gpus="$blocked_gpus $gpu"
            continue
        fi

        # Check if significant memory is used (>1000 MiB suggests a process is running)
        if [ "$mem_used" -gt 1000 ]; then
            echo "  GPU $gpu: IN USE (${mem_used}/${mem_total} MiB)"
            blocked_gpus="$blocked_gpus $gpu"

            # Check if it's a vLLM process
            local vllm_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -5)
            for pid in $vllm_pids; do
                if [ -n "$pid" ] && ps -p "$pid" -o args= 2>/dev/null | grep -q "vllm"; then
                    found_vllm=true
                    echo "    └─ vLLM process detected (PID: $pid)"
                fi
            done
        else
            echo "  GPU $gpu: Available (${mem_used}/${mem_total} MiB)"
        fi
    done

    if [ -n "$blocked_gpus" ]; then
        echo ""
        echo "WARNING: Some GPUs are in use:$blocked_gpus"

        if $found_vllm; then
            echo ""
            echo "vLLM processes detected, auto-cleaning..."
        fi

        # Non-interactive: auto force cleanup
        echo "Auto force cleanup..."
        return 2  # Signal to force cleanup
    fi

    echo "All required GPUs are available."
    return 0
}

# Kill processes on specific GPUs
kill_gpu_processes() {
    local gpus="$1"  # e.g., "0,1,2,3"

    echo "Cleaning up GPU processes on GPUs: $gpus"

    for gpu in $(echo "$gpus" | tr ',' ' '); do
        local pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        for pid in $pids; do
            if [ -n "$pid" ] && [ "$pid" != "0" ]; then
                echo "  Killing PID $pid on GPU $gpu"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    done

    sleep 2
}

# Force cleanup vLLM and proxy processes
force_cleanup() {
    echo "Force cleanup of vLLM and proxy processes..."

    # Kill vLLM processes
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true

    # Kill proxy processes
    pkill -f "disagg_proxy" 2>/dev/null || true
    pkill -f "optimizer_proxy" 2>/dev/null || true
    pkill -f "simple_replica_proxy" 2>/dev/null || true
    pkill -f "replication_proxy" 2>/dev/null || true

    # Kill EngineCore processes (zombie vLLM workers)
    pkill -9 -f "EngineCore" 2>/dev/null || true

    sleep 3
    echo "Cleanup complete."
}

# Wait for a port to be ready
wait_for_port() {
    local port="$1"
    local max_wait="${2:-300}"
    local waited=0

    while ! curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "Timeout waiting for port $port"
            return 1
        fi
        [ $((waited % 30)) -eq 0 ] && echo "  Still waiting for port $port... ($waited s)"
    done
    echo "  Port $port: READY"
    return 0
}

# Check environment
check_environment() {
    local PYTHON_PATH=$(which python)
    if [[ "$PYTHON_PATH" != *"vllm-ppd"* ]]; then
        echo "ERROR: Please activate vllm-ppd first:"
        echo "  conda activate vllm-ppd"
        return 1
    fi

    if ! python -c "import quart" 2>/dev/null; then
        echo "ERROR: quart not installed. Fix: pip install quart"
        return 1
    fi

    return 0
}
