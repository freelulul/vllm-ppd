#!/bin/bash
#SBATCH --job-name=bench_batch1
#SBATCH --output=logs/benchmark/batch1_%j.out
#SBATCH --error=logs/benchmark/batch1_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# Batch 1: Pure Replica + First 1P config
# Configs: 4R, 1P_3D
# Estimated time: ~3 hours (2 configs × 1.5h)
# =============================================================================

set -e

CONFIGS="4R 1P_3D"

echo "=============================================="
echo "Comprehensive Benchmark - Batch 1"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Configs: ${CONFIGS}"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate vllm-ppd
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Create directories
mkdir -p logs/benchmark results/comprehensive

# Run benchmark for each config
for CONFIG in ${CONFIGS}; do
    echo ""
    echo "=============================================="
    echo "Starting config: ${CONFIG}"
    echo "Time: $(date)"
    echo "=============================================="

    python scripts/benchmark/comprehensive_benchmark.py \
        --config ${CONFIG} \
        --workload all \
        --output-dir results/comprehensive \
        2>&1 | tee -a logs/benchmark/batch1_${CONFIG}_${SLURM_JOB_ID}.log

    echo "Completed ${CONFIG} at $(date)"

    # Cleanup between configs
    bash scripts/server/cleanup_all.sh || true
    sleep 10
done

echo ""
echo "=============================================="
echo "Batch 1 Complete"
echo "Time: $(date)"
echo "=============================================="
