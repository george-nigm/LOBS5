#!/bin/bash
###############################################################################
# Parallel LOB Benchmark Job Submitter
# ======================================
# Submits 15+ independent SLURM jobs, each computing one metric
# Each job uses ~20 CPU cores from the 288 available on GH200 nodes
###############################################################################

# Configuration
DATA_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output"
SAVE_DIR="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench/output/brisk-violet-111/results"
STOCK="GOOG"
TIME_PERIOD="2023"
MODEL="s5"

# List of metrics to compute (from DEFAULT_SCORING_CONFIG)
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
echo "Submitting ${#METRICS[@]} parallel SLURM jobs"
echo "============================================"

# Array to store job IDs
declare -a JOBIDS

# Submit one SLURM job for each metric
for metric in "${METRICS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --job-name="bench_${metric}" \
        --output="logs_lobs5/bench_${metric}_%j.out" \
        --error="logs_lobs5/bench_${metric}_%j.err" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=256 \
        --mem=512G \
        --time=00:30:00 \
        --wrap="
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lob
export OMP_NUM_THREADS=256
export MKL_NUM_THREADS=256
export OPENBLAS_NUM_THREADS=256
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/lob_bench
python run_bench.py \
    --data_dir='$DATA_DIR' \
    --save_dir='$SAVE_DIR/${metric}' \
    --model_name='$MODEL' \
    --stock='$STOCK' \
    --time_period='$TIME_PERIOD' \
    --metrics '$metric' \
    --uncond_only
")

    JOBIDS+=($JOB_ID)
    echo "✓ Submitted: $metric (Job ID: $JOB_ID)"
done

echo "============================================"
echo "All ${#METRICS[@]} jobs submitted!"
echo "============================================"
echo ""
echo "Job IDs: ${JOBIDS[@]}"
echo ""
echo "Monitor with:"
echo "  squeue --me"
echo "  squeue -j $(IFS=, ; echo "${JOBIDS[*]}")"
echo ""
echo "Results will be saved to:"
echo "  $SAVE_DIR/<metric_name>/"
echo "============================================"
