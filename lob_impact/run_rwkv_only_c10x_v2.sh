#!/bin/bash
# =============================================================================
# RWKV-only rerun for c10x_v2 grid
#
# Rebuilds the Docker image (adds tokenizers+transformers), then runs
# only the RWKV scenario (60 experiments) on 8 GPUs.
#
# Uses checkpoint: bptt_rwkv_7g0.1B/final (1.14M steps, most trained)
#
# Usage:
#   chmod +x lob_impact/run_rwkv_only_c10x_v2.sh
#   ./lob_impact/run_rwkv_only_c10x_v2.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
LOGS_BASE="${PROJECT_DIR}/output/evalsequences/logs_baselines_c10x_v2"
DOCKER_IMAGE="georgenigm_25jan:latest"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"

SAVE_BASE="/home/myuser/data/evalsequences"
N_COND=500

mkdir -p "$LOGS_BASE"

# =============================================================================
# Step 1: Rebuild Docker image with tokenizers+transformers
# =============================================================================
echo "================================================================="
echo "  Rebuilding Docker image: ${DOCKER_IMAGE}"
echo "  Started: $(date)"
echo "================================================================="

docker build -t "${DOCKER_IMAGE}" \
    --build-arg UID=$(id -u) \
    --build-arg MYUSER=$(whoami) \
    -f "${PROJECT_DIR}/Dockerfile_LOBS5" \
    "${PROJECT_DIR}"

echo ""
echo "  Image rebuilt successfully!"
echo ""

# Verify the fix
echo "  Verifying tokenizers import..."
docker run --rm "${DOCKER_IMAGE}" python -c "from tokenizers import Tokenizer; print('tokenizers OK')" 2>&1 | grep "OK" || {
    echo "ERROR: tokenizers still not importable!"
    exit 1
}
docker run --rm "${DOCKER_IMAGE}" python -c "from transformers import PreTrainedTokenizerFast; print('transformers OK')" 2>&1 | grep "OK" || {
    echo "ERROR: transformers still not importable!"
    exit 1
}
echo "  Dependencies verified!"
echo ""

# =============================================================================
# Step 2: Clean up empty RWKV output folders from failed run
# =============================================================================
echo "Cleaning up empty RWKV folders from previous failed run..."
RWKV_OUTPUT="${PROJECT_DIR}/output/evalsequences/rwkv_aggressive_scenario"
for context_dir in "${RWKV_OUTPUT}"/context_${N_COND}_*; do
    [ -d "$context_dir" ] || continue
    for folder_dir in "$context_dir"/i*; do
        [ -d "$folder_dir" ] || continue
        nfiles=$(find "$folder_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
        if [ "$nfiles" -eq 0 ]; then
            rmdir "$folder_dir" 2>/dev/null && echo "  Removed empty: ${folder_dir##*/}" || true
        fi
    done
done
echo ""

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
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

# =============================================================================
# Config writer
# =============================================================================

write_config_rwkv() {
    local file=$1 n_ins=$2 n_cool=$3 save_dir=$4 vol=$5
    cat > "$file" << EOF
lobgen_dir: "/app/lobgen"
model_choice: "7g0.1B"
rwkv_type: "ScanRWKV"
ckpt_path: "/home/myuser/data/checkpoints/rwkv_models/bptt_rwkv_7g0.1B/final"
tokenizer_file: "tokenizers/lob_tok.json"
temperature: 1.0
raw_data_dir: "/home/myuser/data/rawLOBSTER/GOOG/JAN2023"
stock: "GOOG"
tick_size: 100
n_cond_msgs: ${N_COND}
n_gen_msgs: 50
max_tokens_per_block: 2000
num_insertions: ${n_ins}
num_coolings: ${n_cool}
event_type: 4
direction: 0
order_volume: ${vol}
n_samples: 2048
batch_size: 8
rng_seed: 42
n_levels: 10
process_long_seq_padding: 128
max_cond_tokens: 15000
save_dir: "${save_dir}"
EOF
}

# =============================================================================
# Docker launcher
# =============================================================================

run_one() {
    local gpu=$1 script_py=$2 cfg_container=$3 mb=$4 dir=$5 job_name=$6
    local container_name="bl_${job_name//%/pct}"

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
        python -u "${script_py}" \
            --config "${cfg_container}" \
            --n_gen_msgs "${mb}" \
            --direction "${dir}" \
        > /dev/null

    docker logs -f "${container_name}" > "${LOGS_BASE}/${job_name}.log" 2>&1 &

    docker wait "${container_name}" > /dev/null 2>&1

    echo "[GPU ${gpu}] DONE:  ${job_name}  $(date '+%H:%M:%S')"
}

# =============================================================================
# Build job list, sort, assign to GPUs, launch
# =============================================================================

SCENARIO_NAME="rwkv"
SCRIPT_PY="lob_impact/5.aggressive_scenario_rwkv.py"
SAVE_SUBDIR="rwkv_aggressive_scenario"
CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_rwkv_c10x_v2"
mkdir -p "$CONFIGS_DIR"

echo "================================================================="
echo "  RWKV Aggressive Scenario (c10x_v2 grid)"
echo "  Checkpoint: bptt_rwkv_7g0.1B/final (1.14M steps)"
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
            save_dir="${SAVE_BASE}/${SAVE_SUBDIR}/context_${N_COND}_${dir_name}/${folder_name}"
            workload=$total

            cfg_host="${CONFIGS_DIR}/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
            cfg_container="lob_impact/configs_rwkv_c10x_v2/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
            write_config_rwkv "$cfg_host" "$i" "$c" "$save_dir" "$vol"

            job_name="${SCENARIO_NAME}_${folder_name}_${dir_name}"
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
    local gpu=$1 script=$2
    shift 2
    local jobs=("$@")

    for job_str in "${jobs[@]}"; do
        IFS='|' read -r cfg mb dir name wl <<< "$job_str"
        run_one "$gpu" "$script" "$cfg" "$mb" "$dir" "$name"
    done
}

echo "=== Launching on ${N_GPUS} GPUs... $(date) ==="

for gpu in "${AVAILABLE_GPUS[@]}"; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    if [ ${#jobs[@]} -gt 0 ]; then
        run_gpu_jobs "$gpu" "$SCRIPT_PY" "${jobs[@]}" &
    fi
done

wait
echo ""
echo "=== All RWKV experiments finished! $(date) ==="

# Post-processing: flatten exp_* folders
echo ""
echo "=== Flattening exp_* folders ==="

OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/${SAVE_SUBDIR}"

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
echo "  RWKV scenario completed!"
echo "  Finished: $(date)"
echo "================================================================="
