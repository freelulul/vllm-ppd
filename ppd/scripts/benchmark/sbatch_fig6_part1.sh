#!/bin/bash
#SBATCH --job-name=fig6_AB
#SBATCH --output=logs/fig6_AB_%j.out
#SBATCH --error=logs/fig6_AB_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# Fig6 Benchmark Part 1: Panel A + B
# Estimated time: ~2.5 hours
# =============================================================================

set -e

echo "=============================================="
echo "Fig6 Benchmark - Part 1 (Panel A + B)"
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

cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Create directories
mkdir -p logs results/fig6

# Run Panel A
echo ""
echo "========== Running Panel A =========="
python scripts/tests/fig6_benchmark.py \
    --panels A \
    --output-dir results/fig6 \
    --duration 30

# Run Panel B
echo ""
echo "========== Running Panel B =========="
python scripts/tests/fig6_benchmark.py \
    --panels B \
    --output-dir results/fig6 \
    --duration 30

echo ""
echo "=============================================="
echo "Part 1 (Panel A + B) Complete!"
echo "=============================================="
echo "Done at $(date)"
