#!/bin/bash
#SBATCH --job-name=patch_missing
#SBATCH --output=logs/benchmark/patch_missing_%j.out
#SBATCH --error=logs/benchmark/patch_missing_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# Patch Missing & Failed Tests - 补测缺失和失败的测试点
#
# 运行缺失的测试点 + 重跑0%成功率的测试点
#
# 缺失统计:
# - 1P_3D: 30个缺失 + 9个失败 = 39个测试点
# - 3P_1D: 57个缺失 + 9个失败 = 66个测试点
# 总计: ~105个测试点
# 预计时间: ~2小时
# =============================================================================

set -e

echo "=============================================="
echo "Patch Missing Tests - 补测缺失的测试点"
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

# Run patch script
echo "=============================================="
echo "Starting patch for missing tests"
echo "Time: $(date)"
echo "=============================================="

python scripts/benchmark/patch_missing_tests.py \
    2>&1 | tee logs/benchmark/patch_missing_${SLURM_JOB_ID}.log

echo ""
echo "=============================================="
echo "Patch Complete"
echo "Time: $(date)"
echo "=============================================="

# Final cleanup
bash scripts/server/cleanup_all.sh || true

# Verify results
echo ""
echo "验证补测后的结果:"
for config in 1P_3D 3P_1D; do
    count=$(ls results/comprehensive/${config}/*.json 2>/dev/null | wc -l)
    echo "  ${config}: ${count}/180 测试点"
done
