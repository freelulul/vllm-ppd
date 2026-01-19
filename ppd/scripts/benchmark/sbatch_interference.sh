#!/bin/bash
#SBATCH --job-name=interference
#SBATCH --output=logs/benchmark/interference_%j.out
#SBATCH --error=logs/benchmark/interference_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8

# =============================================================================
# Interference Benchmark - Prefill-Decode干扰测试
#
# 用于论文引入部分的micro-benchmark
# 证明append-prefill与decode共存时干扰很小
#
# 实验内容：
# 1. 核心实验：三线对比 (decode-only vs full-prefill vs append-prefill)
# 2. 敏感性实验：不同append-prefill大小的干扰程度
#
# 预计时间：~45分钟
# =============================================================================

set -e

echo "=============================================="
echo "Interference Benchmark"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate vllm-ppd
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo ""

cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Create directories
mkdir -p logs/benchmark
mkdir -p logs/single_replica
mkdir -p results/interference

# =============================================================================
# Step 1: Start single replica server
# =============================================================================
echo "=============================================="
echo "Step 1: Starting single replica server"
echo "Time: $(date)"
echo "=============================================="

# Use GPU 0
GPU_ID=0
PORT=8300

# Cleanup any existing processes
echo "Cleaning up existing processes..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 5

# Start server
echo "Starting vLLM server on GPU $GPU_ID..."
source scripts/server/config.sh

CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > logs/single_replica/server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
echo "Waiting for server to be ready..."
MAX_WAIT=300
WAITED=0
while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Timeout waiting for server"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting... ($WAITED s)"
done
echo "Server ready! (waited $WAITED s)"

# =============================================================================
# Step 2: Run interference benchmark
# =============================================================================
echo ""
echo "=============================================="
echo "Step 2: Running interference benchmark"
echo "Time: $(date)"
echo "=============================================="

python scripts/tests/interference_benchmark.py \
    --all \
    --server-url "http://localhost:$PORT" \
    2>&1 | tee logs/benchmark/interference_${SLURM_JOB_ID}.log

# =============================================================================
# Step 3: Cleanup
# =============================================================================
echo ""
echo "=============================================="
echo "Step 3: Cleanup"
echo "Time: $(date)"
echo "=============================================="

echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true

echo ""
echo "=============================================="
echo "Interference Benchmark Complete!"
echo "Time: $(date)"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  results/interference/core_experiment.json"
echo "  results/interference/sensitivity_experiment.json"
echo ""
echo "To analyze results:"
echo "  python scripts/analysis/analyze_interference.py"
echo "=============================================="
