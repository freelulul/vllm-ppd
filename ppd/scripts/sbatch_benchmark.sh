#!/bin/bash
#SBATCH --job-name=vllm-bench
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# vLLM Benchmark V3 - 3D Orthogonal Workload Design (16 Workloads)
# =============================================================================
# Usage: sbatch --array=0-15 scripts/sbatch_benchmark.sh qps 2
#        sbatch --array=0-15 scripts/sbatch_benchmark.sh replication 2
#        (第二个参数是run编号，输出到 results/run{N}/)
# =============================================================================
#
# T1 Configs (Context Building - C1):
#   S:  256→256  (C1=512)
#   M:  512→512  (C1=1024)
#   L:  1024→1024 (C1=2048)
#   XL: 2048→2048 (C1=4096)
#
# T2 Types (Incremental Request):
#   a: 32→64    (tiny followup, light/light)
#   b: 32→512   (short Q long output, light/heavy)
#   c: 256→256  (medium balanced)
#   d: 1024→64  (big paste short answer, heavy/light)
#
# QPS Lists by T1 Size (auto mode):
#   Base (_a, _b, _c workloads):
#     S:  8 points  [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
#     M:  8 points  [0.05, 0.1, 1.0, 2.0, 4.0, 8.0, 10.0, 12.0]
#     L:  7 points  [0.05, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0]
#     XL: 5 points  [0.05, 0.1, 1.0, 2.0, 4.0]
#   Extended (_d workloads only - prefill-heavy, shows PD TPOT advantage):
#     S:  12 points [... 17.0, 18.0, 19.0, 20.0]
#     M:  12 points [... 13.0, 14.0, 15.0, 16.0]
#     L:  11 points [... 9.0, 10.0, 11.0, 12.0]
#     XL: 9 points  [... 5.0, 6.0, 7.0, 8.0]
#
# =============================================================================

set -e

# Activate conda environment
source ~/.bashrc
conda activate vllm-ppd
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

MODE=${1:-qps}      # "qps" or "replication"
RUN_NUM=${2:-1}     # Run number (1, 2, 3, ...) for multi-run averaging
RUNS=1
BASE_DURATION=30    # Dynamic duration per QPS is handled in Python code
OUTPUT_DIR="results/run${RUN_NUM}"

# 16 workloads: 4 T1 x 4 T2 (array index 0-15)
WORKLOADS=(
    # T1_S (C1=512)
    "S_a"    # 0: Light context + tiny followup
    "S_b"    # 1: Light context + long generation
    "S_c"    # 2: Light context + medium balanced
    "S_d"    # 3: Light context + big paste
    # T1_M (C1=1024)
    "M_a"    # 4: Medium context + tiny followup
    "M_b"    # 5: Medium context + long generation
    "M_c"    # 6: Medium context + medium balanced
    "M_d"    # 7: Medium context + big paste
    # T1_L (C1=2048)
    "L_a"    # 8: Large context + tiny followup
    "L_b"    # 9: Large context + long generation
    "L_c"    # 10: Large context + medium balanced
    "L_d"    # 11: Large context + big paste
    # T1_XL (C1=4096)
    "XL_a"   # 12: XL context + tiny followup
    "XL_b"   # 13: XL context + long generation
    "XL_c"   # 14: XL context + medium balanced
    "XL_d"   # 15: XL context + big paste
)

WORKLOAD=${WORKLOADS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Job: $SLURM_JOB_ID, Array: $SLURM_ARRAY_TASK_ID"
echo "Mode: $MODE, Workload: $WORKLOAD, Run: $RUN_NUM"
echo "Output: $OUTPUT_DIR"
echo "Base Duration: ${BASE_DURATION}s (dynamic per QPS in code)"
echo "QPS: auto (by T1 size)"
echo "========================================"

cd /net/projects2/ds3lab/zongzel/vllm/ppd
mkdir -p logs "$OUTPUT_DIR"

if [ "$MODE" == "qps" ]; then
    # PD/PPD benchmark
    echo "Starting PD/PPD servers..."
    ./scripts/start_servers_4gpu.sh ppd
    sleep 180  # Wait for servers

    echo "Running benchmark: $WORKLOAD"
    python src/qps_benchmark.py \
        --workload $WORKLOAD \
        --qps auto \
        --duration $BASE_DURATION \
        --runs $RUNS \
        --output "${OUTPUT_DIR}/qps_${WORKLOAD}.json"

    ./scripts/stop_servers_4gpu.sh

elif [ "$MODE" == "replication" ]; then
    # Replication benchmark
    echo "Starting Replication servers..."
    ./scripts/start_replication_servers_4gpu.sh
    sleep 180

    echo "Running benchmark: $WORKLOAD"
    python src/replication_benchmark.py \
        --workload $WORKLOAD \
        --qps auto \
        --duration $BASE_DURATION \
        --runs $RUNS \
        --output "${OUTPUT_DIR}/replication_${WORKLOAD}.json"

    ./scripts/stop_replication_servers_4gpu.sh
fi

echo "========================================"
echo "Workload $WORKLOAD completed!"
echo "========================================"
