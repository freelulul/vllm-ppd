#!/bin/bash
#SBATCH --job-name=mini_test
#SBATCH --output=logs/mini_test_%j.log
#SBATCH --error=logs/mini_test_%j.err
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

# Mini Test: Verify benchmark script works correctly
# Estimated time: ~10 minutes
#
# Usage:
#   sbatch scripts/benchmark/sbatch_mini_test.sh

set -e

echo "=========================================="
echo "MINI TEST: Verify Benchmark Correctness"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

cd /net/projects2/ds3lab/zongzel/vllm/ppd
source ~/.bashrc
conda activate vllm-ppd
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

mkdir -p logs

# Step 1: Verify configuration
echo ""
echo "=========================================="
echo "Step 1: Verify Configuration"
echo "=========================================="
python scripts/tests/verify_trend_config.py

# Step 2: Start servers
echo ""
echo "=========================================="
echo "Step 2: Start Servers"
echo "=========================================="
./scripts/server/start_optimizer_servers.sh

echo ""
echo "Waiting for servers to stabilize (30s)..."
sleep 30

# Step 3: Run mini test
echo ""
echo "=========================================="
echo "Step 3: Run Mini Test (3 requests per mode)"
echo "=========================================="
python scripts/tests/mini_test_benchmark.py

# Step 4: Cleanup
echo ""
echo "=========================================="
echo "Step 4: Cleanup"
echo "=========================================="
pkill -f "vllm serve" || true
pkill -f "proxy" || true

echo ""
echo "=========================================="
echo "MINI TEST COMPLETE"
echo "=========================================="
echo "End: $(date)"
echo ""
echo "If all tests passed, run full benchmark:"
echo "  sbatch scripts/benchmark/sbatch_real_trend.sh"
