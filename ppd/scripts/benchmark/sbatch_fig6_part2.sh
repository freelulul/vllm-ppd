#!/bin/bash
#SBATCH --job-name=fig6_CDE
#SBATCH --output=logs/fig6_CDE_%j.out
#SBATCH --error=logs/fig6_CDE_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# Fig6 Benchmark Part 2: Panel C + D + E
# Estimated time: ~3 hours
# =============================================================================

set -e

echo "=============================================="
echo "Fig6 Benchmark - Part 2 (Panel C + D + E)"
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

# Run Panel C
echo ""
echo "========== Running Panel C =========="
python scripts/tests/fig6_benchmark.py \
    --panels C \
    --output-dir results/fig6 \
    --duration 30

# Run Panel D
echo ""
echo "========== Running Panel D =========="
python scripts/tests/fig6_benchmark.py \
    --panels D \
    --output-dir results/fig6 \
    --duration 30

# Run Panel E
echo ""
echo "========== Running Panel E =========="
python scripts/tests/fig6_benchmark.py \
    --panels E \
    --output-dir results/fig6 \
    --duration 30

echo ""
echo "=============================================="
echo "Part 2 (Panel C + D + E) Complete!"
echo "=============================================="
echo "Done at $(date)"
