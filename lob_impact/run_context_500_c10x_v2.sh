#!/bin/bash
# =============================================================================
# Context 500 experiments with cooling = 10 × insertions
# v2: added order_volume dimension (75, 300, 485)
#
# Constraint: total_gen = 11*i*mb ≤ 500
#
# Grid (10 pairs × 3 volumes × 2 directions = 60 runs):
#   mb=5:  i=3,5,9   → total: 165, 275, 495
#   mb=10: i=2,3,4   → total: 220, 330, 440
#   mb=15: i=2,3     → total: 330, 495
#   mb=20: i=1,2     → total: 220, 440
#
# Volumes: 75, 300, 485 shares per insertion
#
# Naming: i{i}_c{c}_mb{mb}_v{vol}_cntxt{pct}%
#
# Estimated time: ~4-5 hours on 7 GPUs
#
# Usage:
#   chmod +x lob_impact/run_context_500_c10x_v2.sh
#   ./lob_impact/run_context_500_c10x_v2.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_context_500_c10x_v2"
LOGS_DIR="${PROJECT_DIR}/output/evalsequences/aggressive_scenario/logs_c10x_v2"
DOCKER_IMAGE="georgenigm_25jan"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"
SCRIPT_PY="lob_impact/1.aggressive_scenario_s5.py"

SAVE_BASE="/home/myuser/data/evalsequences/aggressive_scenario"
N_COND=500

mkdir -p "$CONFIGS_DIR" "$LOGS_DIR"

# ---- Helper: write YAML config ----
write_config() {
    local file=$1 n_ins=$2 n_cool=$3 save_dir=$4 vol=$5
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
checkpoint_step: null
chunk_size: 5
n_samples: 2048
batch_size: 64
rng_seed: 42
stock: "GOOG"
data_dir: "/home/myuser/data/processed_data/GOOG/2023_Jan"
ckpt_path: "/home/myuser/data/checkpoints/lobs5_v2/twilight-sound-77_s42sujip"
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
    local gpu=$1 cfg_container=$2 mb=$3 dir=$4 job_name=$5
    # Docker container names don't allow '%' — replace with 'pct'
    local container_name="c10x_v2_${job_name//%/pct}"

    echo "[GPU ${gpu}] START: ${job_name}  $(date '+%H:%M:%S')"

    docker run -d --rm \
        --gpus "\"device=${gpu}\"" \
        --name "${container_name}" \
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

    docker logs -f "${container_name}" > "${LOGS_DIR}/${job_name}.log" 2>&1 &

    docker wait "${container_name}" > /dev/null 2>&1

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
# Generate configs and build job list
# =============================================================================
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
            save_dir="${SAVE_BASE}/context_${N_COND}_${dir_name}/${folder_name}"
            workload=$total

            cfg_host="${CONFIGS_DIR}/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
            cfg_container="lob_impact/configs_context_500_c10x_v2/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
            write_config "$cfg_host" "$i" "$c" "$save_dir" "$vol"

            job_name="${folder_name}_${dir_name}"
            JOBS+=("${cfg_container}|${mb}|${dir}|${job_name}|${workload}")
        done
    done
done

echo "=== Total jobs: ${#JOBS[@]} ==="

# =============================================================================
# Sort by workload (descending) for load balancing
# =============================================================================
IFS=$'\n' SORTED_JOBS=($(for j in "${JOBS[@]}"; do echo "$j"; done | sort -t'|' -k5 -rn))
unset IFS

# =============================================================================
# Assign to 7 GPUs using snake pattern (GPU 5 is busy)
# =============================================================================
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
N_GPUS=${#AVAILABLE_GPUS[@]}

for g in "${AVAILABLE_GPUS[@]}"; do
    eval "declare -a GPU_JOBS_${g}=()"
done

# Snake pattern: 0,1,2,3,4,6,7,7,6,4,3,2,1,0,...
SNAKE=()
forward=("${AVAILABLE_GPUS[@]}")
reverse=()
for (( idx=${#AVAILABLE_GPUS[@]}-1; idx>=0; idx-- )); do
    reverse+=("${AVAILABLE_GPUS[$idx]}")
done
while [ ${#SNAKE[@]} -lt ${#SORTED_JOBS[@]} ]; do
    SNAKE+=("${forward[@]}")
    SNAKE+=("${reverse[@]}")
done

for idx in "${!SORTED_JOBS[@]}"; do
    gpu=${SNAKE[$idx]}
    eval "GPU_JOBS_${gpu}+=(\"${SORTED_JOBS[$idx]}\")"
done

# =============================================================================
# Print assignment
# =============================================================================
echo ""
echo "=== GPU Assignment (8 GPUs) ==="
for gpu in "${AVAILABLE_GPUS[@]}"; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    total=0
    count=0
    names=""
    for j in "${jobs[@]}"; do
        IFS='|' read -r _ _ _ name wl <<< "$j"
        total=$((total + wl))
        count=$((count + 1))
        names+="  ${name}(${wl})"
    done
    echo "GPU ${gpu} [${count} jobs, work=${total}]:${names}"
done
echo ""

# =============================================================================
# Launch: each GPU runs its jobs sequentially, all 7 GPUs in parallel
# =============================================================================
run_gpu() {
    local gpu=$1
    shift
    local jobs=("$@")

    for job_str in "${jobs[@]}"; do
        IFS='|' read -r cfg mb dir name wl <<< "$job_str"
        run_one "$gpu" "$cfg" "$mb" "$dir" "$name"
    done
}

echo "=== Launching on 8 GPUs... $(date) ==="

for gpu in "${AVAILABLE_GPUS[@]}"; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    if [ ${#jobs[@]} -gt 0 ]; then
        run_gpu "$gpu" "${jobs[@]}" &
    fi
done

wait
echo ""
echo "=== All experiments finished! $(date) ==="

# =============================================================================
# Post-processing: flatten exp_* folders
# =============================================================================
echo ""
echo "=== Post-processing: flattening exp_* folders ==="

OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/aggressive_scenario"

for context_dir in "${OUTPUT_BASE}"/context_${N_COND}_*; do
    [ -d "$context_dir" ] || continue
    for folder_dir in "$context_dir"/i*; do
        [ -d "$folder_dir" ] || continue
        exp_dir=$(find "$folder_dir" -maxdepth 1 -type d -name "exp_*" | head -1)
        if [ -n "$exp_dir" ]; then
            echo "  Flattening: ${folder_dir##*/}"
            mv "$exp_dir"/* "$folder_dir"/ 2>/dev/null || true
            rmdir "$exp_dir" 2>/dev/null || true
        fi
    done
done

echo "=== Done! ==="
