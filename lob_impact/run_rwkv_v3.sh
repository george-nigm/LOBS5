#!/bin/bash
# =============================================================================
# RWKV v3 market impact experiments
# Grid: fixed cooling=5, i*mb+1) ≤ 500  (7 configs × 3 vols × 2 dirs = 42 exps)
#
# Grid (same as S5 v3):
#   mb=5:   i=83  → insertion_phase=498
#   mb=10:  i=45  → insertion_phase=495
#   mb=20:  i=23  → insertion_phase=483
#   mb=25:  i=19  → insertion_phase=494
#   mb=50:  i=9   → insertion_phase=459
#   mb=75:  i=6   → insertion_phase=456
#   mb=100: i=4   → insertion_phase=404
#
# Usage:
#   chmod +x lob_impact/run_rwkv_v3.sh
#   ./lob_impact/run_rwkv_v3.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_rwkv_v3"
LOGS_DIR="${PROJECT_DIR}/output/evalsequences/rwkv_goog2022/logs_v3"
DOCKER_IMAGE="georgenigm_25jan:latest"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"
SCRIPT_PY="lob_impact/5v2.aggressive_scenario_rwkv.py"

SAVE_BASE="/home/myuser/data/evalsequences/rwkv_goog2022/v3"
N_COND=500
N_COOL=5

mkdir -p "$CONFIGS_DIR" "$LOGS_DIR"

# ---- Helper: write YAML config ----
write_config() {
    local file=$1 n_ins=$2 n_cool=$3 vol=$4 save_dir=$5
    cat > "$file" << EOF
lobgen_dir: "/app/lobgen"
model_choice: "6g0.1B"
rwkv_type: "ScanRWKV"
ckpt_path: "/home/myuser/data/checkpoints/rwkv_models/goog2022_rwkv_6g0.1B"
tokenizer_file: "tokenizers/lob_tok.json"
temperature: 1.0
process_long_seq_padding: 128
max_cond_tokens: 15000
max_tokens_per_block: 2000
n_gen_msgs: 50
num_insertions: ${n_ins}
num_coolings: ${n_cool}
n_cond_msgs: ${N_COND}
event_type: 4
direction: 0
order_volume: ${vol}
n_samples: 2048
batch_size: 64
rng_seed: 42
stock: "GOOG"
data_dir: "/home/myuser/data/processed_data/GOOG/2022"
save_dir: "${save_dir}"
tick_size: 100
n_levels: 10
n_eval_msgs_dataset: 500
test_split: 0
EOF
}

# ---- Helper: run one experiment ----
run_one() {
    local gpu=$1 cfg_container=$2 mb=$3 dir=$4 job_name=$5
    local container_name="rwkv_v3_${job_name//%/pct}"

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
# Grid: (i, mb) pairs where i * (mb + 1) ≤ 500
# =============================================================================
GRID=(
    "83 5"
    "45 10"
    "23 20"
    "19 25"
    "9  50"
    "6  75"
    "4  100"
)

VOLUMES=(75 300 485)

# =============================================================================
# Generate configs and build job list
# =============================================================================
declare -a JOBS=()

for pair in "${GRID[@]}"; do
    read -r i mb <<< "$pair"
    insertion_phase=$(( i * (mb + 1) ))
    total=$(( (i + N_COOL) * mb + i ))
    cntxt=$(( insertion_phase * 100 / N_COND ))

    for vol in "${VOLUMES[@]}"; do
        for dir in 0 1; do
            dir_name=$( [ "$dir" = "0" ] && echo "buy" || echo "sell" )
            folder_name="i${i}_c5_mb${mb}_v${vol}_cntxt${cntxt}%"
            save_dir="${SAVE_BASE}/context_${N_COND}_${dir_name}/${folder_name}"

            cfg_host="${CONFIGS_DIR}/cfg_i${i}_c5_mb${mb}_v${vol}_${dir_name}.yaml"
            cfg_container="lob_impact/configs_rwkv_v3/cfg_i${i}_c5_mb${mb}_v${vol}_${dir_name}.yaml"
            write_config "$cfg_host" "$i" "$N_COOL" "$vol" "$save_dir"

            job_name="i${i}_c5_mb${mb}_v${vol}_${dir_name}"
            JOBS+=("${cfg_container}|${mb}|${dir}|${job_name}|${total}")
        done
    done
done

echo "=== Total jobs: ${#JOBS[@]} ==="

# =============================================================================
# Sort by workload (descending)
# =============================================================================
IFS=$'\n' SORTED_JOBS=($(for j in "${JOBS[@]}"; do echo "$j"; done | sort -t'|' -k5 -rn))
unset IFS

# =============================================================================
# Assign to 8 GPUs (snake pattern)
# =============================================================================
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
N_GPUS=${#AVAILABLE_GPUS[@]}

for g in "${AVAILABLE_GPUS[@]}"; do
    eval "declare -a GPU_JOBS_${g}=()"
done

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
    total_wl=0
    names=""
    for j in "${jobs[@]}"; do
        IFS='|' read -r _ _ _ name wl <<< "$j"
        total_wl=$((total_wl + wl))
        names+="  ${name}(${wl})"
    done
    echo "GPU ${gpu} [total_work=${total_wl}]:${names}"
done
echo ""

# =============================================================================
# Launch: each GPU runs sequentially, all GPUs in parallel
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

echo "=== Launching RWKV v3 on ${N_GPUS} GPUs (0-7)... $(date) ==="

for gpu in "${AVAILABLE_GPUS[@]}"; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    if [ ${#jobs[@]} -gt 0 ]; then
        run_gpu "$gpu" "${jobs[@]}" &
    fi
done

wait
echo ""
echo "=== RWKV v3 experiments finished! $(date) ==="

# =============================================================================
# Post-processing: flatten exp_* folders
# =============================================================================
echo ""
echo "=== Post-processing: flattening exp_* folders ==="

OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/rwkv_goog2022/v3"

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
