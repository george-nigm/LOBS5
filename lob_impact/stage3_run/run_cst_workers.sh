#!/bin/bash
# Launch N parallel CPU workers for CST aggressive scenario
# Usage: bash lob_impact/run_cst_workers.sh [CONFIG] [NUM_WORKERS]
#
# Example:
#   bash lob_impact/run_cst_workers.sh lob_impact/scenarios/4.aggressive_scenario_cst_config.yaml 16
set -e

CONFIG="${1:-lob_impact/scenarios/4.aggressive_scenario_cst_config.yaml}"
NUM_WORKERS="${2:-8}"

# Docker image and volume mounts
IMAGE="georgenigm_25jan"
VOLUMES="-v /scratch/local/homes/80/georgenigm/LOBS5:/app \
  -v /scratch/local/homes/80/georgenigm/LOBS5/Alphatrade:/AlphaTrade \
  -v /homes/groups/finance/data:/home/myuser/data \
  -v /scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences:/home/myuser/data/evalsequences \
  -v /homes/80/georgenigm/scratch_LOB:/home/myuser/scratch"
USER_FLAGS="--user $(id -u):$(id -g) --group-add 652"

# Read save_dir from config
SAVE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['save_dir'])")

# Create shared experiment folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Find next exp index
if [ -d "/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences/$(basename $SAVE_DIR)" ]; then
    MOUNT_DIR="/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences/$(basename $SAVE_DIR)"
    MAX_IDX=$(ls -d "$MOUNT_DIR"/exp_* 2>/dev/null | sed 's/.*exp_\([0-9]*\).*/\1/' | sort -n | tail -1)
    NEXT_IDX=$(( ${MAX_IDX:-0} + 1 ))
else
    NEXT_IDX=1
fi
SAVE_FOLDER="${SAVE_DIR}/exp_${NEXT_IDX}_${TIMESTAMP}"

echo "============================================"
echo "CST Aggressive Scenario — Parallel CPU Run"
echo "============================================"
echo "Config:      $CONFIG"
echo "Workers:     $NUM_WORKERS"
echo "Save folder: $SAVE_FOLDER"
echo "Backend:     CPU (JAX_PLATFORMS=cpu)"
echo "============================================"
echo ""

# Launch workers in parallel
PIDS=()
for W in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Starting worker $W/$NUM_WORKERS ..."
    docker run --rm \
        --name "georgenigm_cst_w${W}" \
        $USER_FLAGS \
        $VOLUMES \
        -e JAX_PLATFORMS=cpu \
        --shm-size=1g \
        -w /app \
        "$IMAGE" \
        python3 -u lob_impact/scenarios/4.aggressive_scenario_cst.py \
            --config "$CONFIG" \
            --worker_id "$W" \
            --num_workers "$NUM_WORKERS" \
            --save_folder "$SAVE_FOLDER" \
        > "/tmp/cst_worker_${W}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $NUM_WORKERS workers launched. PIDs: ${PIDS[*]}"
echo "Logs: /tmp/cst_worker_*.log"
echo ""

# Wait for all workers
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "Worker $i finished OK (PID $PID)"
    else
        echo "Worker $i FAILED (PID $PID) — check /tmp/cst_worker_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "All $NUM_WORKERS workers completed successfully!"
    # Count output files
    N_COND=$(ls "$MOUNT_DIR/exp_${NEXT_IDX}_${TIMESTAMP}/data_cond/" 2>/dev/null | wc -l)
    N_GEN=$(ls "$MOUNT_DIR/exp_${NEXT_IDX}_${TIMESTAMP}/data_gen/" 2>/dev/null | wc -l)
    echo "Output: $N_COND cond files, $N_GEN gen files"
else
    echo "$FAILED/$NUM_WORKERS workers failed!"
    exit 1
fi
