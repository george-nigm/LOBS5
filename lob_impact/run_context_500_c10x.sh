#!/bin/bash
# =============================================================================
# Context 500 experiments with cooling = 10 × insertions
# Grid: 10 configs × 2 (buy/sell) = 20 experiments on 7 GPUs (GPU 5 busy)
#
# Constraint: total_gen = (i + 10*i) * mb = 11*i*mb ≤ 500
#
# Grid:
#   mb=5:  i=3,5,9   → total: 165, 275, 495
#   mb=10: i=2,3,4   → total: 220, 330, 440
#   mb=15: i=2,3     → total: 330, 495
#   mb=20: i=1,2     → total: 220, 440
#
# Usage:
#   chmod +x lob_impact/run_context_500_c10x.sh
#   ./lob_impact/run_context_500_c10x.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_context_500_c10x"
LOGS_DIR="${PROJECT_DIR}/output/evalsequences/aggressive_scenario/logs_c10x"
DOCKER_IMAGE="georgenigm_25jan"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"
SCRIPT_PY="lob_impact/1.aggressive_scenario_s5.py"

SAVE_BASE="/home/myuser/data/evalsequences/aggressive_scenario/c10x_v2"
N_COND=500

mkdir -p "$CONFIGS_DIR" "$LOGS_DIR"

# ---- Helper: write YAML config ----
write_config() {
    local file=$1 n_ins=$2 n_cool=$3 save_dir=$4
    cat > "$file" << EOF
n_gen_msgs: 50
num_insertions: ${n_ins}
num_coolings: ${n_cool}
n_cond_msgs: ${N_COND}
n_eval_msgs_dataset: 500
event_type: 4
direction: 0
order_volume: 75
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
    local container_name="c10x_${job_name}"

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
# Format: "i mb"
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

# =============================================================================
# Generate configs and build job list
# =============================================================================
declare -a JOBS=()

for pair in "${GRID[@]}"; do
    read -r i mb <<< "$pair"
    c=$((i * 10))
    total=$((11 * i * mb))
    cntxt=$((total * 100 / N_COND))

    for dir in 0 1; do
        dir_name=$( [ "$dir" = "0" ] && echo "buy" || echo "sell" )
        folder_name="i${i}_c${c}_mb${mb}_cntxt${cntxt}%"
        save_dir="${SAVE_BASE}/context_${N_COND}_${dir_name}/${folder_name}"
        workload=$total

        cfg_host="${CONFIGS_DIR}/cfg_i${i}_c${c}_mb${mb}_${dir_name}.yaml"
        cfg_container="lob_impact/configs_context_500_c10x/cfg_i${i}_c${c}_mb${mb}_${dir_name}.yaml"
        write_config "$cfg_host" "$i" "$c" "$save_dir"

        job_name="i${i}_c${c}_mb${mb}_${dir_name}"
        JOBS+=("${cfg_container}|${mb}|${dir}|${job_name}|${workload}")
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
AVAILABLE_GPUS=(0 1 2 3 4 6 7)
N_GPUS=${#AVAILABLE_GPUS[@]}

for g in "${AVAILABLE_GPUS[@]}"; do
    eval "declare -a GPU_JOBS_${g}=()"
done

# Snake pattern over available GPUs: 0,1,2,3,4,6,7,7,6,4,3,2,1,0,...
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
echo "=== GPU Assignment (7 GPUs, excluding GPU 5) ==="
for gpu in "${AVAILABLE_GPUS[@]}"; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    total=0
    names=""
    for j in "${jobs[@]}"; do
        IFS='|' read -r _ _ _ name wl <<< "$j"
        total=$((total + wl))
        names+="  ${name}(${wl})"
    done
    echo "GPU ${gpu} [total_work=${total}]:${names}"
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

echo "=== Launching on 7 GPUs... $(date) ==="

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

OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/aggressive_scenario/c10x_v2"

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
