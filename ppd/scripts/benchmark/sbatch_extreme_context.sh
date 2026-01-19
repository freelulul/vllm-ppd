#!/bin/bash
#SBATCH --job-name=extreme_context
#SBATCH --output=logs/benchmark/extreme_context_%j.out
#SBATCH --error=logs/benchmark/extreme_context_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# Extreme Context Test - Finding Memory Capacity Limits
#
# Tests fixed config (2P_2pD) at fixed QPS (20) with progressively larger
# context sizes to find true memory capacity upper limit.
#
# Config: 2P_2pD (1 config)
# QPS: 20 (1 point)
# Context sizes: 4096, 6144, 8192, 12288, 16384 tokens (5 points)
# Total: 1 × 1 × 5 = 5 test points
# Estimated time: ~30-90 minutes (may timeout at large contexts)
# =============================================================================

set -e

echo "=============================================="
echo "Extreme Context Test"
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
mkdir -p logs/benchmark results/extreme_context

# Run extreme context test
echo "=============================================="
echo "Starting Extreme Context Test"
echo "Time: $(date)"
echo "=============================================="

python scripts/tests/extreme_context_test.py \
    2>&1 | tee logs/benchmark/extreme_context_${SLURM_JOB_ID}.log

echo ""
echo "=============================================="
echo "Extreme Context Test Complete"
echo "Time: $(date)"
echo "=============================================="

# Final cleanup
bash scripts/server/cleanup_all.sh || true

echo ""
echo "Results saved to: results/extreme_context/"
echo "Log saved to: logs/benchmark/extreme_context_${SLURM_JOB_ID}.log"
