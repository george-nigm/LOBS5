#!/bin/bash
#
# Parallel LOB Benchmark Orchestrator
# ====================================
# Runs 15 metrics in parallel as background jobs
# Each metric gets ~17 CPU cores (256 total / 15 metrics)
#

# Configuration
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output"
SAVE_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output/brisk-violet-111/results"
STOCK="GOOG"
TIME_PERIOD="2023"
MODEL="s5"

# NOTE: GH200 node has 288 CPU cores, using 256 cores (leaving 32 for system)
# Each metric task gets: 256 / 15 ≈ 17 cores
CORES_PER_METRIC=17
export OMP_NUM_THREADS=$CORES_PER_METRIC
export MKL_NUM_THREADS=$CORES_PER_METRIC
export OPENBLAS_NUM_THREADS=$CORES_PER_METRIC

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench

# List of all metrics from DEFAULT_SCORING_CONFIG
METRICS=(
    "spread"
    "orderbook_imbalance"
    "log_inter_arrival_time"
    "log_time_to_cancel"
    "ask_volume_touch"
    "bid_volume_touch"
    "ask_volume"
    "bid_volume"
    "limit_ask_order_depth"
    "limit_bid_order_depth"
    "ask_cancellation_depth"
    "bid_cancellation_depth"
    "limit_ask_order_levels"
    "limit_bid_order_levels"
    "ask_cancellation_levels"
    "bid_cancellation_levels"
    "vol_per_min"
    "ofi"
    "ofi_up"
    "ofi_stay"
    "ofi_down"
)

echo "============================================"
echo "Starting Parallel LOB Benchmark"
echo "Total metrics: ${#METRICS[@]}"
echo "Cores per metric: $CORES_PER_METRIC"
echo "Total cores used: $((${#METRICS[@]} * CORES_PER_METRIC))"
echo "============================================"

# Array to store background job PIDs
declare -a PIDS

# Launch each metric computation in background
for i in "${!METRICS[@]}"; do
    metric="${METRICS[$i]}"
    echo "[Task $((i+1))/${#METRICS[@]}] Launching: $metric"

    # Run in background and capture PID
    $CONDA_PREFIX/bin/python -u run_bench.py \
        --data_dir="$DATA_DIR" \
        --save_dir="$SAVE_DIR/${metric}" \
        --model_name="$MODEL" \
        --stock="$STOCK" \
        --time_period="$TIME_PERIOD" \
        --metrics "$metric" \
        --uncond_only \
        > "$SAVE_DIR/log_${metric}.out" \
        2> "$SAVE_DIR/log_${metric}.err" &

    PIDS[$i]=$!
    echo "  → PID: ${PIDS[$i]}"
done

echo "============================================"
echo "All ${#METRICS[@]} metrics launched in parallel"
echo "Waiting for completion..."
echo "============================================"

# Wait for all background jobs to complete
failed=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    metric="${METRICS[$i]}"

    wait $pid
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ [$((i+1))/${#METRICS[@]}] $metric completed successfully"
    else
        echo "✗ [$((i+1))/${#METRICS[@]}] $metric FAILED (exit code: $exit_code)"
        ((failed++))
    fi
done

echo "============================================"
echo "Parallel benchmark completed"
echo "Success: $((${#METRICS[@]} - failed))/${#METRICS[@]}"
echo "Failed: $failed/${#METRICS[@]}"
echo "============================================"

# Merge results if all succeeded
if [ $failed -eq 0 ]; then
    echo "[*] All metrics completed successfully"
    echo "[*] Results saved to: $SAVE_DIR"
else
    echo "[WARNING] Some metrics failed. Check individual log files:"
    echo "  $SAVE_DIR/log_*.err"
fi

exit $failed
