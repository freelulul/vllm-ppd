#!/bin/bash
#SBATCH --job-name=highqps-d
#SBATCH --output=logs/highqps_d_%A_%a.out
#SBATCH --error=logs/highqps_d_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# High QPS Test for _d Workloads - Extending PD TPOT Advantage Analysis
# =============================================================================
# Usage: sbatch --array=0-7 scripts/sbatch_highqps_d.sh
#
# Array mapping (8 jobs = 4 workloads x 2 modes):
#   0: S_d  QPS  - qps mode (PD/PPD)
#   1: S_d  QPS  - replication mode
#   2: M_d  QPS  - qps mode (PD/PPD)
#   3: M_d  QPS  - replication mode
#   4: L_d  QPS  - qps mode (PD/PPD)
#   5: L_d  QPS  - replication mode
#   6: XL_d QPS  - qps mode (PD/PPD)
#   7: XL_d QPS  - replication mode
#
# QPS ranges (extending from current max where PD shows advantage):
#   S_d:  17, 18, 19, 20  (current max=16, PD +43.9% vs PPD)
#   M_d:  13, 14, 15, 16  (current max=12, PD +50.9% vs PPD)
#   L_d:  9, 10, 11, 12   (current max=8,  PD +5.2% vs PPD)
#   XL_d: 5, 6, 7, 8      (current max=4,  PD +9.1% vs PPD)
# =============================================================================

set -e

source ~/.bashrc
conda activate vllm-ppd
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

cd /net/projects2/ds3lab/zongzel/vllm/ppd
mkdir -p logs results/highqps_d

# Array index determines workload and mode
IDX=$SLURM_ARRAY_TASK_ID

# Workload index (0-3) and mode (0=qps, 1=replication)
WORKLOAD_IDX=$((IDX / 2))
MODE_IDX=$((IDX % 2))

# Workload configs
WORKLOADS=("S_d" "M_d" "L_d" "XL_d")
QPS_LISTS=("17,18,19,20" "13,14,15,16" "9,10,11,12" "5,6,7,8")

WORKLOAD=${WORKLOADS[$WORKLOAD_IDX]}
QPS_LIST=${QPS_LISTS[$WORKLOAD_IDX]}

if [ $MODE_IDX -eq 0 ]; then
    MODE="qps"
else
    MODE="replication"
fi

DURATION=30
RUNS=1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="results/highqps_d/${MODE}_${WORKLOAD}_highqps_${TIMESTAMP}.json"

echo "========================================"
echo "High QPS _d Workload Test"
echo "========================================"
echo "Job: $SLURM_JOB_ID, Array: $SLURM_ARRAY_TASK_ID"
echo "Workload: $WORKLOAD"
echo "Mode: $MODE"
echo "QPS List: $QPS_LIST"
echo "Duration: ${DURATION}s per QPS"
echo "Output: $OUTPUT_FILE"
echo "========================================"

if [ "$MODE" == "qps" ]; then
    echo "Starting PD/PPD servers..."
    ./scripts/start_servers_4gpu.sh ppd
    sleep 180

    echo "Running QPS benchmark: $WORKLOAD @ QPS=$QPS_LIST"
    python src/qps_benchmark.py \
        --workload $WORKLOAD \
        --qps "$QPS_LIST" \
        --duration $DURATION \
        --runs $RUNS \
        --output "$OUTPUT_FILE"

    ./scripts/stop_servers_4gpu.sh

elif [ "$MODE" == "replication" ]; then
    echo "Starting Replication servers..."
    ./scripts/start_replication_servers_4gpu.sh
    sleep 180

    echo "Running Replication benchmark: $WORKLOAD @ QPS=$QPS_LIST"
    python src/replication_benchmark.py \
        --workload $WORKLOAD \
        --qps "$QPS_LIST" \
        --duration $DURATION \
        --runs $RUNS \
        --output "$OUTPUT_FILE"

    ./scripts/stop_replication_servers_4gpu.sh
fi

echo "========================================"
echo "Completed: $WORKLOAD ($MODE) @ QPS=$QPS_LIST"
echo "Output: $OUTPUT_FILE"
echo "========================================"
