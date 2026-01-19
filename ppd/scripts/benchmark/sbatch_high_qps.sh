#!/bin/bash
#SBATCH --job-name=high_qps_boundary
#SBATCH --output=logs/benchmark/high_qps_%j.out
#SBATCH --error=logs/benchmark/high_qps_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# High QPS Boundary Test - Finding True OOM Limits
#
# Tests multi-prefill configurations at extreme QPS (30-100) to bypass prefill
# bottleneck and find true memory OOM boundary.
#
# Configs: 3P_1D, 3P_1pD, 2P_2D, 2P_2pD (4 configs)
# Workloads: large_very_long_gen, large_huge_paste (2 workloads)
# QPS: 30, 40, 50, 60, 80, 100 (6 points)
# Total: 4 × 2 × 6 = 48 test points
# Estimated time: ~2-3 hours (with early stopping on OOM)
# =============================================================================

set -e

echo "=============================================="
echo "High QPS Boundary Test"
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
mkdir -p logs/benchmark results/oom_boundary

# Run high QPS boundary test
echo "=============================================="
echo "Starting High QPS Boundary Test"
echo "Time: $(date)"
echo "=============================================="

python scripts/tests/high_qps_boundary_test.py \
    2>&1 | tee logs/benchmark/high_qps_${SLURM_JOB_ID}.log

echo ""
echo "=============================================="
echo "High QPS Boundary Test Complete"
echo "Time: $(date)"
echo "=============================================="

# Final cleanup
bash scripts/server/cleanup_all.sh || true

echo ""
echo "Results saved to: results/oom_boundary/"
echo "Log saved to: logs/benchmark/high_qps_${SLURM_JOB_ID}.log"
