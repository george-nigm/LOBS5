#!/bin/bash
# =============================================================================
# Context 500 experiments with cooling = 10 × insertions
# v3 4K context: j2504167 checkpoint at 4 training steps
#
# Constraint: total_gen = 11*i*mb ≤ 500
#
# Grid (10 pairs × 3 volumes × 2 directions = 60 runs per step):
#   mb=5:  i=3,5,9   → total: 165, 275, 495
#   mb=10: i=2,3,4   → total: 220, 330, 440
#   mb=15: i=2,3     → total: 330, 495
#   mb=20: i=1,2     → total: 220, 440
#
# Volumes: 75, 300, 485 shares per insertion
#
# Checkpoint: j2504167 — 4K context (55M params), 24-token encoding
# Steps: 22820, 38335, 79688, 100378
#
# Usage:
#   chmod +x lob_impact/run_context_500_c10x_v3_4kctx.sh
#   ./lob_impact/run_context_500_c10x_v3_4kctx.sh [step|all]
#
#   step: 22820 | 38335 | 79688 | 100378 | all (default: all)
# =============================================================================
set -eo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_context_500_c10x_v3_4kctx"
LOGS_DIR="${PROJECT_DIR}/output/evalsequences/aggressive_scenario_v3/logs_4kctx"
DOCKER_IMAGE="georgenigm_25jan"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"
SCRIPT_PY="lob_impact/1.aggressive_scenario_s5_v3.py"

SAVE_BASE="/home/myuser/data/evalsequences/aggressive_scenario_v3"
N_COND=500

mkdir -p "$CONFIGS_DIR" "$LOGS_DIR"

# ---- Checkpoint definition (single checkpoint, multiple steps) ----
CKPT_PATH="/home/myuser/data/checkpoints/lobs5_v3/j2504167_y0c4j6l3_2504167"
ALL_STEPS=(22820 38335 79688 100378)

# ---- Parse which step(s) to run ----
STEP_ARG="${1:-all}"
if [ "$STEP_ARG" = "all" ]; then
    RUN_STEPS=("${ALL_STEPS[@]}")
else
    valid=false
    for s in "${ALL_STEPS[@]}"; do
        if [ "$STEP_ARG" = "$s" ]; then
            valid=true
            break
        fi
    done
    if ! $valid; then
        echo "ERROR: Unknown step '$STEP_ARG'. Use: 22820 | 38335 | 79688 | 100378 | all"
        exit 1
    fi
    RUN_STEPS=("$STEP_ARG")
fi

# ---- Helper: write YAML config ----
write_config() {
    local file=$1 n_ins=$2 n_cool=$3 save_dir=$4 vol=$5 ckpt=$6 step=$7
    cat > "$file" << EOF
n_gen_msgs: 50
num_insertions: ${n_ins}
num_coolings: ${n_cool}
n_cond_msgs: ${N_COND}
n_eval_msgs_dataset: 500
event_type: 4
direction: 0
order_volume: ${vol}
use_relative_volume: false
order_volume_ratio: 1.0
checkpoint_step: ${step}
chunk_size: 5
n_samples: 2048
batch_size: 64
rng_seed: 42
stock: "GOOG"
data_dir: "/home/myuser/data/processed_data/GOOG/2023_Jan"
ckpt_path: "${ckpt}"
save_dir: "${save_dir}"
tick_size: 100
sample_top_n: -1
n_vol_series: 500
book_dim: 503
test_split: 0
EOF
}

# ---- Helper: run one experiment ----
run_one() {
    local gpu=$1 cfg_container=$2 mb=$3 dir=$4 job_name=$5 log_dir=$6

    echo "[GPU ${gpu}] START: ${job_name}  $(date '+%H:%M:%S')"

    docker run -d --rm \
        --gpus "\"device=${gpu}\"" \
        --name "${job_name}" \
        --user "$(id -u):$(id -g)" \
        --group-add 652 \
        -v "${PROJECT_DIR}:/app" \
        -v "${PROJECT_DIR}/Alphatrade:/AlphaTrade" \
        -v /homes/groups/finance/data:/home/myuser/data \
        -v "${PROJECT_DIR}/output/evalsequences:/home/myuser/data/evalsequences" \
        -e WANDB_API_KEY="${WANDB_KEY}" \
        --shm-size=1g \
        -w /app \
        "${DOCKER_IMAGE}" \
        python -u "${SCRIPT_PY}" \
            --config "${cfg_container}" \
            --n_gen_msgs "${mb}" \
            --direction "${dir}" \
        > /dev/null

    docker logs -f "${job_name}" > "${log_dir}/${job_name}.log" 2>&1 &

    docker wait "${job_name}" > /dev/null 2>&1

    echo "[GPU ${gpu}] DONE:  ${job_name}  $(date '+%H:%M:%S')"
}

# =============================================================================
# Define grid: (i, mb) pairs where 11*i*mb <= 500
# =============================================================================
GRID=(
    "3 5"
    "5 5"
    "9 5"
    "2 10"
    "3 10"
    "4 10"
    "2 15"
    "3 15"
    "1 20"
    "2 20"
)

# Order volumes to test
VOLUMES=(75 300 485)

# =============================================================================
# Run each step
# =============================================================================
for step in "${RUN_STEPS[@]}"; do
    step_logs="${LOGS_DIR}/step_${step}"
    mkdir -p "$step_logs"

    echo ""
    echo "============================================================"
    echo "  Checkpoint: j2504167 (step ${step})"
    echo "  Path: ${CKPT_PATH}"
    echo "============================================================"

    # ---- Generate configs and build job list ----
    declare -a JOBS=()

    for pair in "${GRID[@]}"; do
        read -r i mb <<< "$pair"
        c=$((i * 10))
        total=$((11 * i * mb))
        cntxt=$((total * 100 / N_COND))

        for vol in "${VOLUMES[@]}"; do
            for dir in 0 1; do
                dir_name=$( [ "$dir" = "0" ] && echo "buy" || echo "sell" )

                folder_name="i${i}_c${c}_mb${mb}_v${vol}_cntxt${cntxt}%"
                save_dir="${SAVE_BASE}/j2504167_step${step}/context_${N_COND}_${dir_name}/${folder_name}"
                workload=$total

                cfg_host="${CONFIGS_DIR}/s${step}_cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
                cfg_container="lob_impact/configs_context_500_c10x_v3_4kctx/s${step}_cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
                write_config "$cfg_host" "$i" "$c" "$save_dir" "$vol" "$CKPT_PATH" "$step"

                job_name="v3_4k_s${step}_${folder_name}_${dir_name}"
                # Replace '%' with 'pct' for Docker container names
                job_name="${job_name//%/pct}"
                JOBS+=("${cfg_container}|${mb}|${dir}|${job_name}|${workload}")
            done
        done
    done

    echo "  Total jobs for step ${step}: ${#JOBS[@]}"

    # ---- Sort by workload (descending) ----
    mapfile -t SORTED_JOBS < <(for j in "${JOBS[@]}"; do echo "$j"; done | sort -t'|' -k5 -rn)

    # ---- Assign to 8 GPUs using snake pattern via temp files ----
    AVAILABLE_GPUS=(4 5 6 7)
    N_GPUS=${#AVAILABLE_GPUS[@]}

    TMPDIR_JOBS=$(mktemp -d)
    for g in "${AVAILABLE_GPUS[@]}"; do
        : > "${TMPDIR_JOBS}/gpu_${g}"
    done

    # Snake pattern: 0,1,2,3,4,5,6,7, 7,6,5,4,3,2,1,0, 0,1,...
    for idx in "${!SORTED_JOBS[@]}"; do
        cycle_pos=$((idx % (N_GPUS * 2)))
        if [ "$cycle_pos" -lt "$N_GPUS" ]; then
            gpu_idx=$cycle_pos
        else
            gpu_idx=$((N_GPUS * 2 - 1 - cycle_pos))
        fi
        gpu=${AVAILABLE_GPUS[$gpu_idx]}
        echo "${SORTED_JOBS[$idx]}" >> "${TMPDIR_JOBS}/gpu_${gpu}"
    done

    # ---- Print assignment ----
    echo ""
    echo "  === GPU Assignment (GPUs 4-7) ==="
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [ ! -s "${TMPDIR_JOBS}/gpu_${gpu}" ]; then
            echo "  GPU ${gpu} [0 jobs, work=0]:"
            continue
        fi
        count=0; total=0; names=""
        while IFS='|' read -r _ _ _ name wl; do
            total=$((total + wl))
            count=$((count + 1))
            names+="  ${name##v3_4k_s${step}_}(${wl})"
        done < "${TMPDIR_JOBS}/gpu_${gpu}"
        echo "  GPU ${gpu} [${count} jobs, work=${total}]:${names}"
    done
    echo ""

    # ---- Launch: each GPU runs its jobs sequentially, all GPUs in parallel ----
    run_gpu_from_file() {
        local gpu=$1 log_dir=$2 job_file=$3
        while IFS='|' read -r cfg mb dir name wl; do
            run_one "$gpu" "$cfg" "$mb" "$dir" "$name" "$log_dir"
        done < "$job_file"
    }

    echo "  === Launching step ${step} on 4 GPUs... $(date) ==="

    for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [ -s "${TMPDIR_JOBS}/gpu_${gpu}" ]; then
            run_gpu_from_file "$gpu" "$step_logs" "${TMPDIR_JOBS}/gpu_${gpu}" &
        fi
    done

    wait
    rm -rf "$TMPDIR_JOBS"
    echo ""
    echo "  === Step ${step} finished! $(date) ==="

    unset JOBS SORTED_JOBS
done

# =============================================================================
# Post-processing: flatten exp_* folders
# =============================================================================
echo ""
echo "=== Post-processing: flattening exp_* folders ==="

OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/aggressive_scenario_v3"

for step in "${RUN_STEPS[@]}"; do
    for context_dir in "${OUTPUT_BASE}/j2504167_step${step}"/context_${N_COND}_*; do
        [ -d "$context_dir" ] || continue
        for folder_dir in "$context_dir"/i*; do
            [ -d "$folder_dir" ] || continue
            exp_dir=$(find "$folder_dir" -maxdepth 1 -type d -name "exp_*" | head -1)
            if [ -n "$exp_dir" ]; then
                echo "  Flattening: j2504167_step${step}/${folder_dir##*/}"
                mv "$exp_dir"/* "$folder_dir"/ 2>/dev/null || true
                rmdir "$exp_dir" 2>/dev/null || true
            fi
        done
    done
done

echo ""
echo "=== All experiments finished! $(date) ==="
