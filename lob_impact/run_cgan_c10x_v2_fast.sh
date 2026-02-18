#!/bin/bash
# =============================================================================
# CGAN (Coletta) Aggressive Scenario v2 — c10x_v2 grid (OPTIMIZED)
#
# Uses 5v2.aggressive_scenario_cgan.py with batched inference (batch_size=64).
# ~6-10x faster than v1 due to:
#   - Numpy tracker (no pandas per-message)
#   - Batched CGAN forward pass (64 samples at once)
#   - Vectorized decode
#
# Grid: 10 (i,mb) pairs × 3 volumes × 2 directions = 60 jobs on 7 GPUs.
#
# Usage:
#   chmod +x lob_impact/run_cgan_c10x_v2_fast.sh
#   nohup ./lob_impact/run_cgan_c10x_v2_fast.sh > /dev/null 2>&1 &
# =============================================================================
set -euo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
LOGS_BASE="${PROJECT_DIR}/output/evalsequences/logs_cgan_c10x_v2_fast"
DOCKER_IMAGE="georgenigm_25jan:latest"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"

SAVE_BASE="/home/myuser/data/evalsequences"
N_COND=500

mkdir -p "$LOGS_BASE"

# =============================================================================
# Grid: (i, mb) pairs where 11*i*mb <= 500
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

VOLUMES=(75 300 485)
AVAILABLE_GPUS=(0 1 2 3)

# CGAN model paths (inside container)
CGAN_CKPT="/home/myuser/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_['20221228', '20221229', '20221230']_v2_41/checkpoints/model_39.ckpt"
CGAN_SCALERS="/home/myuser/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_['20221228', '20221229', '20221230']_v2_41/data__scalers.pickle"
CGAN_IAT="/home/myuser/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_['20221228', '20221229', '20221230']_v2_41/interarrival_times"

# =============================================================================
# Config writer (v2: batch_size=64)
# =============================================================================
CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_cgan_c10x_v2"
mkdir -p "$CONFIGS_DIR"

write_config_cgan() {
    local file=$1 n_ins=$2 n_cool=$3 save_dir=$4 vol=$5
    cat > "$file" << EOF
data_dir: "/home/myuser/data/processed_data/GOOG/2023_Jan"
stock: "GOOG"
tick_size: 100

n_cond_msgs: ${N_COND}
n_gen_msgs: 50
n_eval_msgs_dataset: 500
n_levels: 10

num_insertions: ${n_ins}
num_coolings: ${n_cool}
event_type: 4
direction: 0
order_volume: ${vol}

n_samples: 2048
batch_size: 64
rng_seed: 42
test_split: 0

save_dir: "${save_dir}"

cgan_checkpoint: "${CGAN_CKPT}"
cgan_scalers: "${CGAN_SCALERS}"
cgan_interarrival_times: "${CGAN_IAT}"
lookback_window: 100
EOF
}

# =============================================================================
# Docker launcher
# =============================================================================
run_one() {
    local gpu=$1 cfg_container=$2 mb=$3 dir=$4 job_name=$5
    local container_name="cganv2_${job_name//%/pct}"

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
        -e "XLA_FLAGS=--xla_gpu_enable_command_buffer=" \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=true \
        -e XLA_PYTHON_CLIENT_MEM_FRACTION=.40 \
        --shm-size=1g \
        -w /app \
        "${DOCKER_IMAGE}" \
        bash -c "pip install --quiet 'pytorch-lightning>=1.9,<2.0' 2>/dev/null && \
            python -u lob_impact/5v2.aggressive_scenario_cgan.py \
                --config ${cfg_container} \
                --n_gen_msgs ${mb} \
                --direction ${dir}" \
        > /dev/null

    docker logs -f "${container_name}" > "${LOGS_BASE}/${job_name}.log" 2>&1 &

    docker wait "${container_name}" > /dev/null 2>&1

    echo "[GPU ${gpu}] DONE:  ${job_name}  $(date '+%H:%M:%S')"
}

# =============================================================================
# Build job list, sort, assign to GPUs, launch
# =============================================================================
SCENARIO_NAME="cgan"
SAVE_SUBDIR="cgan_aggressive_scenario"

echo "================================================================="
echo "  CGAN Aggressive Scenario v2 (OPTIMIZED, batch_size=64)"
echo "  Model: Coletta LOBGAN v2_41, epoch 39"
echo "  Started: $(date)"
echo "  GPUs: ${AVAILABLE_GPUS[*]}"
echo "  Grid: 10 pairs x 3 volumes x 2 directions = 60 jobs"
echo "================================================================="

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
            save_dir="${SAVE_BASE}/${SAVE_SUBDIR}/c10x_v2/context_${N_COND}_${dir_name}/${folder_name}"
            workload=$total

            cfg_host="${CONFIGS_DIR}/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
            cfg_container="lob_impact/configs_cgan_c10x_v2/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
            write_config_cgan "$cfg_host" "$i" "$c" "$save_dir" "$vol"

            job_name="${folder_name}_${dir_name}"
            JOBS+=("${cfg_container}|${mb}|${dir}|${job_name}|${workload}")
        done
    done
done

echo "=== ${#JOBS[@]} jobs ==="

# Sort by workload descending
IFS=$'\n' SORTED_JOBS=($(for j in "${JOBS[@]}"; do echo "$j"; done | sort -t'|' -k5 -rn))
unset IFS

# Snake assignment to GPUs
N_GPUS=${#AVAILABLE_GPUS[@]}

for g in "${AVAILABLE_GPUS[@]}"; do
    eval "declare -a GPU_JOBS_${g}=()"
done

declare -a SNAKE=()
declare -a forward=("${AVAILABLE_GPUS[@]}")
declare -a reverse=()
for (( idx=${#AVAILABLE_GPUS[@]}-1; idx>=0; idx-- )); do
    reverse+=("${AVAILABLE_GPUS[$idx]}")
done
while [ ${#SNAKE[@]} -lt ${#SORTED_JOBS[@]} ]; do
    SNAKE+=("${forward[@]}")
    SNAKE+=("${reverse[@]}")
done

for idx in "${!SORTED_JOBS[@]}"; do
    local_gpu=${SNAKE[$idx]}
    eval "GPU_JOBS_${local_gpu}+=(\"${SORTED_JOBS[$idx]}\")"
done

# Print assignment
echo ""
echo "=== GPU Assignment ==="
for gpu in "${AVAILABLE_GPUS[@]}"; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    total_wl=0
    count=0
    names=""
    for j in "${jobs[@]}"; do
        IFS='|' read -r _ _ _ name wl <<< "$j"
        total_wl=$((total_wl + wl))
        count=$((count + 1))
        names+="  ${name}(${wl})"
    done
    echo "GPU ${gpu} [${count} jobs, work=${total_wl}]:${names}"
done
echo ""

# Launch: each GPU runs its jobs sequentially, all GPUs in parallel
run_gpu_jobs() {
    local gpu=$1
    shift 1
    local jobs=("$@")

    for job_str in "${jobs[@]}"; do
        IFS='|' read -r cfg mb dir name wl <<< "$job_str"
        run_one "$gpu" "$cfg" "$mb" "$dir" "$name"
    done
}

echo "=== Launching on ${N_GPUS} GPUs... $(date) ==="

for gpu in "${AVAILABLE_GPUS[@]}"; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    if [ ${#jobs[@]} -gt 0 ]; then
        run_gpu_jobs "$gpu" "${jobs[@]}" &
    fi
done

wait
echo ""
echo "=== All CGAN v2 experiments finished! $(date) ==="

# Post-processing: flatten exp_* folders
echo ""
echo "=== Flattening exp_* folders ==="

OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/${SAVE_SUBDIR}/c10x_v2"

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

echo ""
echo "================================================================="
echo "  CGAN v2 scenario completed!"
echo "  Finished: $(date)"
echo "================================================================="
