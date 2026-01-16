#!/bin/bash
# =============================================================================
# Submit All Fig6 Benchmark Jobs (Panels A-H)
# =============================================================================
#
# Usage:
#   ./scripts/benchmark/submit_fig6_all.sh           # Submit all 3 parts
#   ./scripts/benchmark/submit_fig6_all.sh 1         # Submit only part 1
#   ./scripts/benchmark/submit_fig6_all.sh 2 3       # Submit parts 2 and 3
#
# Jobs are chained with dependencies to avoid GPU conflicts on n001
#
# Benchmark Configuration:
#   - Part 1: Panel A (5 pts) + B (8 pts) = ~2.5 hours
#   - Part 2: Panel C (5 pts) + D (5 pts) + E (5 pts) = ~3 hours
#   - Part 3: Panel F (6 pts) + G (6 pts) + H (5 pts) = ~3.5 hours
#   - Total: ~9 hours (45 data points × 4 modes × 30s duration)
#
# =============================================================================

set -e

cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Get parts from arguments (default: 1 2 3)
PARTS="${@:-1 2 3}"

echo "=============================================="
echo "Fig6 Complete Benchmark Suite (Panels A-H)"
echo "=============================================="
echo "Partition:  ds3lab-own"
echo "Node:       n001"
echo "GPUs:       4 x GPU"
echo "Duration:   30s per data point"
echo "Parts:      ${PARTS}"
echo ""
echo "Panel Summary:"
echo "  A: Objective Type (QPS=16, 5 points)"
echo "  B: QPS Scaling (1-24, 8 points)"
echo "  C: Input Length (QPS=16, 5 points)"
echo "  D: Output Length (QPS=16, 5 points)"
echo "  E: Multi-Turn (QPS=8, 5 points)"
echo "  F: I/O Ratio (QPS=8, 6 points)"
echo "  G: Big Paste/PD Advantage (6 points)"
echo "  H: Context Size (QPS=12, 5 points)"
echo ""
echo "Estimated total time: ~9 hours"
echo "=============================================="
echo ""

mkdir -p logs results/fig6

LAST_JOB=""
TOTAL_JOBS=0

for PART in ${PARTS}; do
    echo "--- Submitting Part ${PART} ---"

    SCRIPT="scripts/benchmark/sbatch_fig6_part${PART}.sh"

    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT"
        continue
    fi

    # Submit with dependency on previous job if exists
    if [ -z "$LAST_JOB" ]; then
        JOB_ID=$(sbatch --parsable "$SCRIPT")
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:${LAST_JOB} "$SCRIPT")
    fi

    case $PART in
        1) PANELS="A, B" ;;
        2) PANELS="C, D, E" ;;
        3) PANELS="F, G, H" ;;
    esac

    echo "  Part ${PART} (Panel ${PANELS}): Job ${JOB_ID}"
    LAST_JOB=${JOB_ID}
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    echo ""
done

echo "=============================================="
echo "Submitted ${TOTAL_JOBS} jobs successfully!"
echo "=============================================="
echo ""
echo "Commands:"
echo "  squeue -u \$USER                    # Check job status"
echo "  tail -f logs/fig6_*.out            # View logs"
echo ""
echo "After all jobs complete:"
echo "  python scripts/benchmark/merge_fig6_results.py    # Merge results"
echo ""
echo "Output files will be in: results/fig6/"
echo "=============================================="
