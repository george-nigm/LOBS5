#!/bin/bash
# =============================================================================
# Baseline scenarios with c10x_v2 grid (same grid as S5 neural model)
#
# Runs 4 scenarios SEQUENTIALLY:
#   1. Historic  — historical replay + aggressive insertions (no model)
#   2. Heuristic — same + price shifting on consumed levels (no model)
#   3. CST       — Stoikov-Talreja parametric model + aggressive insertions
#   4. RWKV      — RWKV neural model + aggressive insertions
#
# Grid per scenario (same as run_context_500_c10x_v2.sh):
#   Constraint: total_gen = 11*i*mb ≤ 500
#   10 (i,mb) pairs × 3 volumes (75,300,485) × 2 directions = 60 runs
#
# Total: 4 × 60 = 240 experiments
# Estimated time: ~3-4h historic/heuristic/CST, ~6-8h RWKV on 8 GPUs
#
# Usage:
#   chmod +x lob_impact/run_baselines_c10x_v2.sh
#   ./lob_impact/run_baselines_c10x_v2.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
LOGS_BASE="${PROJECT_DIR}/output/evalsequences/logs_baselines_c10x_v2"
DOCKER_IMAGE="georgenigm_25jan"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"

SAVE_BASE="/home/myuser/data/evalsequences"
N_COND=500

# Scratch dir for CST params (mounted as /home/myuser/scratch inside container)
SCRATCH_DIR="${HOME}/scratch_LOB"

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
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

# =============================================================================
# Config writers
# =============================================================================

write_config_historic_heuristic() {
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
n_samples: 2048
batch_size: 64
rng_seed: 42
stock: "GOOG"
data_dir: "/home/myuser/data/processed_data/GOOG/2023_Jan"
save_dir: "${save_dir}"
tick_size: 100
n_vol_series: 500
book_dim: 503
test_split: 0
EOF
}

write_config_cst() {
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
n_samples: 2048
batch_size: 64
rng_seed: 42
stock: "GOOG"
data_dir: "/home/myuser/data/processed_data/GOOG/2023_Jan"
save_dir: "${save_dir}"
tick_size: 100
params_file: "/home/myuser/scratch/cst_params_goog.pkl"
num_ticks: 500
n_levels: 10
test_split: 0
EOF
}

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
    local extra_mounts="${7:-}"
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
        ${extra_mounts} \
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
# Run one scenario (generate configs → sort → snake assign → launch → flatten)
# =============================================================================

run_scenario() {
    local scenario_name=$1   # "historic", "heuristic", "cst"
    local script_py=$2       # python script path (container-relative)
    local write_fn=$3        # config writer function name
    local save_subdir=$4     # e.g. "historic_scenario"
    local extra_mounts="${5:-}"

    local configs_dir="${PROJECT_DIR}/lob_impact/configs_${scenario_name}_c10x_v2"
    mkdir -p "$configs_dir"

    echo ""
    echo "############################################################"
    echo "# Scenario: ${scenario_name}"
    echo "# Script:   ${script_py}"
    echo "# Started:  $(date)"
    echo "############################################################"
    echo ""

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
                save_dir="${SAVE_BASE}/${save_subdir}/context_${N_COND}_${dir_name}/${folder_name}"
                workload=$total

                cfg_host="${configs_dir}/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
                cfg_container="lob_impact/configs_${scenario_name}_c10x_v2/cfg_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
                $write_fn "$cfg_host" "$i" "$c" "$save_dir" "$vol"

                job_name="${scenario_name}_${folder_name}_${dir_name}"
                JOBS+=("${cfg_container}|${mb}|${dir}|${job_name}|${workload}")
            done
        done
    done

    echo "=== ${scenario_name}: ${#JOBS[@]} jobs ==="

    # ---- Sort by workload descending ----
    IFS=$'\n' SORTED_JOBS=($(for j in "${JOBS[@]}"; do echo "$j"; done | sort -t'|' -k5 -rn))
    unset IFS

    # ---- Snake assignment to GPUs ----
    local N_GPUS=${#AVAILABLE_GPUS[@]}

    for g in "${AVAILABLE_GPUS[@]}"; do
        eval "declare -a GPU_JOBS_${g}=()"
    done

    local -a SNAKE=()
    local -a forward=("${AVAILABLE_GPUS[@]}")
    local -a reverse=()
    for (( idx=${#AVAILABLE_GPUS[@]}-1; idx>=0; idx-- )); do
        reverse+=("${AVAILABLE_GPUS[$idx]}")
    done
    while [ ${#SNAKE[@]} -lt ${#SORTED_JOBS[@]} ]; do
        SNAKE+=("${forward[@]}")
        SNAKE+=("${reverse[@]}")
    done

    for idx in "${!SORTED_JOBS[@]}"; do
        local gpu=${SNAKE[$idx]}
        eval "GPU_JOBS_${gpu}+=(\"${SORTED_JOBS[$idx]}\")"
    done

    # ---- Print assignment ----
    echo ""
    echo "=== ${scenario_name}: GPU Assignment ==="
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
        local total_wl=0
        local count=0
        local names=""
        for j in "${jobs[@]}"; do
            IFS='|' read -r _ _ _ name wl <<< "$j"
            total_wl=$((total_wl + wl))
            count=$((count + 1))
            names+="  ${name}(${wl})"
        done
        echo "GPU ${gpu} [${count} jobs, work=${total_wl}]:${names}"
    done
    echo ""

    # ---- Launch: each GPU runs its jobs sequentially, all GPUs in parallel ----
    run_gpu_jobs() {
        local gpu=$1 script=$2 mounts="$3"
        shift 3
        local jobs=("$@")

        for job_str in "${jobs[@]}"; do
            IFS='|' read -r cfg mb dir name wl <<< "$job_str"
            run_one "$gpu" "$script" "$cfg" "$mb" "$dir" "$name" "$mounts"
        done
    }

    echo "=== ${scenario_name}: Launching on ${N_GPUS} GPUs... $(date) ==="

    for gpu in "${AVAILABLE_GPUS[@]}"; do
        eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
        if [ ${#jobs[@]} -gt 0 ]; then
            run_gpu_jobs "$gpu" "$script_py" "$extra_mounts" "${jobs[@]}" &
        fi
    done

    wait
    echo ""
    echo "=== ${scenario_name}: All experiments finished! $(date) ==="

    # ---- Post-processing: flatten exp_* folders ----
    echo ""
    echo "=== ${scenario_name}: Flattening exp_* folders ==="

    local OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/${save_subdir}"

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

    echo "=== ${scenario_name}: Done! ==="
}

# =============================================================================
# Run all 4 scenarios sequentially
# =============================================================================

echo "================================================================="
echo "  Scenarios (c10x_v2 grid): historic → heuristic → CST → RWKV"
echo "  Started: $(date)"
echo "  GPUs: ${AVAILABLE_GPUS[*]}"
echo "  Grid: 10 pairs × 3 volumes × 2 directions = 60 jobs per scenario"
echo "  Total: 240 experiments"
echo "================================================================="

# 1. Historic scenario (no extra mounts needed)
run_scenario \
    "historic" \
    "lob_impact/2.historic_scenario.py" \
    "write_config_historic_heuristic" \
    "historic_scenario"

# 2. Heuristic scenario (no extra mounts needed)
run_scenario \
    "heuristic" \
    "lob_impact/3.heuristic_scenario.py" \
    "write_config_historic_heuristic" \
    "heuristic_scenario"

# 3. CST scenario (needs scratch mount for params_file)
run_scenario \
    "cst" \
    "lob_impact/4.aggressive_scenario_cst.py" \
    "write_config_cst" \
    "cst_scenario" \
    "-v ${SCRATCH_DIR}:/home/myuser/scratch"

# 4. RWKV scenario (lobgen dir accessible via /app mount)
run_scenario \
    "rwkv" \
    "lob_impact/5.aggressive_scenario_rwkv.py" \
    "write_config_rwkv" \
    "rwkv_aggressive_scenario"

echo ""
echo "================================================================="
echo "  All 4 scenarios completed!"
echo "  Finished: $(date)"
echo "================================================================="
