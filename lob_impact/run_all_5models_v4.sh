#!/bin/bash
# =============================================================================
# All 5 scenarios — v4 grid (cooling = insertions * 10, same (i,mb) as v3)
#
# Grid design:
#   - Same (i, mb) pairs as v3: i = floor(500 / (mb + 1))
#   - Variable cooling: num_coolings = i * 10
#   - Total messages: (i + i*10) * mb + i = 11*i*mb + i
#
# Grid (7 pairs × 3 volumes × 2 directions = 42 configs per scenario):
#   mb=5:   i=83, c=830  → total=4648
#   mb=10:  i=45, c=450  → total=4995
#   mb=20:  i=23, c=230  → total=5103
#   mb=25:  i=19, c=190  → total=4009
#   mb=50:  i=9,  c=90   → total=4959
#   mb=75:  i=6,  c=60   → total=4956
#   mb=100: i=4,  c=40   → total=4804
#
# Scenarios (sequential): S5 → CGAN → CST → Historic → Heuristic
# Total: 5 × 42 = 210 experiments
#
# Usage:
#   chmod +x lob_impact/run_all_5models_v4.sh
#   ./lob_impact/run_all_5models_v4.sh              # full run
#   ./lob_impact/run_all_5models_v4.sh --dry-run    # generate configs only
# =============================================================================
set -euo pipefail

# --- Dry-run support ---
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN MODE] — configs will be generated, no containers launched"
fi

# =============================================================================
# Global configuration
# =============================================================================
PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
DOCKER_IMAGE="georgenigm_25jan:latest"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"
LOGS_BASE="${PROJECT_DIR}/output/evalsequences/logs_v4"

SAVE_BASE="/home/myuser/data/evalsequences"
N_COND=500

# Scratch dir for CST params
SCRATCH_DIR="${HOME}/scratch_LOB"

# CGAN model paths (inside container)
CGAN_CKPT="/home/myuser/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_['20221228', '20221229', '20221230']_v2_41/checkpoints/model_39.ckpt"
CGAN_SCALERS="/home/myuser/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_['20221228', '20221229', '20221230']_v2_41/data__scalers.pickle"
CGAN_IAT="/home/myuser/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_['20221228', '20221229', '20221230']_v2_41/interarrival_times"

mkdir -p "$LOGS_BASE"

# =============================================================================
# Grid: same (i, mb) pairs as v3
# i = floor(500 / (mb + 1)) — maximum insertions that fit
# cooling = i * 10
# =============================================================================
GRID=(
    "83 5"
    "45 10"
    "23 20"
    "19 25"
    "9 50"
    "6 75"
    "4 100"
)

VOLUMES=(75 300 485)
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

# =============================================================================
# Config writers  (num_coolings is now a parameter, not a constant)
# =============================================================================

write_config_s5() {
    local file=$1 n_ins=$2 save_dir=$3 vol=$4 n_cool=$5
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

write_config_historic_heuristic() {
    local file=$1 n_ins=$2 save_dir=$3 vol=$4 n_cool=$5
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
    local file=$1 n_ins=$2 save_dir=$3 vol=$4 n_cool=$5
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

write_config_cgan() {
    local file=$1 n_ins=$2 save_dir=$3 vol=$4 n_cool=$5
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
    local gpu=$1 script_py=$2 cfg_container=$3 mb=$4 dir=$5 job_name=$6
    local scenario_type="${7:-standard}"
    local extra_mounts="${8:-}"
    local container_name="v4_${scenario_type}_${job_name//%/pct}"

    echo "[GPU ${gpu}] START: ${job_name}  $(date '+%H:%M:%S')"

    if $DRY_RUN; then
        echo "[DRY RUN] docker run --gpus device=${gpu} ... ${script_py} --config ${cfg_container} --n_gen_msgs ${mb} --direction ${dir}"
        return
    fi

    if [[ "$scenario_type" == "cgan" ]]; then
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
            ${extra_mounts} \
            --shm-size=1g \
            -w /app \
            "${DOCKER_IMAGE}" \
            bash -c "pip install --quiet 'pytorch-lightning>=1.9,<2.0' 2>/dev/null && \
                python -u ${script_py} \
                    --config ${cfg_container} \
                    --n_gen_msgs ${mb} \
                    --direction ${dir}" \
            > /dev/null
    else
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
    fi

    docker logs -f "${container_name}" > "${LOGS_BASE}/${job_name}.log" 2>&1 &

    docker wait "${container_name}" > /dev/null 2>&1

    echo "[GPU ${gpu}] DONE:  ${job_name}  $(date '+%H:%M:%S')"
}

# =============================================================================
# Run one scenario (generate configs → sort → snake assign → launch → flatten)
# =============================================================================

run_scenario() {
    local scenario_name=$1   # "s5", "cgan", "cst", "historic", "heuristic"
    local script_py=$2       # python script path (container-relative)
    local write_fn=$3        # config writer function name
    local save_subdir=$4     # e.g. "aggressive_scenario"
    local scenario_type=$5   # "s5", "cgan", "cst", "hist", "heur"
    local extra_mounts="${6:-}"

    local configs_dir="${PROJECT_DIR}/lob_impact/configs_${scenario_name}_v4"
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
        cool=$(( i * 10 ))
        insertion_phase=$(( i * (mb + 1) ))
        total=$(( (i + cool) * mb + i ))
        cntxt=$(( insertion_phase * 100 / N_COND ))

        for vol in "${VOLUMES[@]}"; do
            for dir in 0 1; do
                dir_name=$( [ "$dir" = "0" ] && echo "buy" || echo "sell" )

                folder_name="i${i}_c${cool}_mb${mb}_v${vol}_cntxt${cntxt}%"
                save_dir="${SAVE_BASE}/${save_subdir}/v4/context_${N_COND}_${dir_name}/${folder_name}"
                workload=$total

                cfg_host="${configs_dir}/cfg_i${i}_c${cool}_mb${mb}_v${vol}_${dir_name}.yaml"
                cfg_container="lob_impact/configs_${scenario_name}_v4/cfg_i${i}_c${cool}_mb${mb}_v${vol}_${dir_name}.yaml"
                $write_fn "$cfg_host" "$i" "$save_dir" "$vol" "$cool"

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
        local gpu=$1 script=$2 stype=$3 mounts="$4"
        shift 4
        local jobs=("$@")

        for job_str in "${jobs[@]}"; do
            IFS='|' read -r cfg mb dir name wl <<< "$job_str"
            run_one "$gpu" "$script" "$cfg" "$mb" "$dir" "$name" "$stype" "$mounts"
        done
    }

    echo "=== ${scenario_name}: Launching on ${N_GPUS} GPUs... $(date) ==="

    for gpu in "${AVAILABLE_GPUS[@]}"; do
        eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
        if [ ${#jobs[@]} -gt 0 ]; then
            run_gpu_jobs "$gpu" "$script_py" "$scenario_type" "$extra_mounts" "${jobs[@]}" &
        fi
    done

    wait
    echo ""
    echo "=== ${scenario_name}: All experiments finished! $(date) ==="

    # ---- Post-processing: flatten exp_* folders ----
    if ! $DRY_RUN; then
        echo ""
        echo "=== ${scenario_name}: Flattening exp_* folders ==="

        local OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/${save_subdir}/v4"

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
    fi

    echo "=== ${scenario_name}: Done! ==="
}

# =============================================================================
# Run all 5 scenarios sequentially
# =============================================================================

echo "================================================================="
echo "  All 5 scenarios — v4 grid (cooling = i*10, same (i,mb) as v3)"
echo "  Started: $(date)"
echo "  GPUs: ${AVAILABLE_GPUS[*]}"
echo "  Grid: 7 pairs × 3 volumes × 2 directions = 42 jobs per scenario"
echo "  Scenarios: S5 → CGAN → CST → Historic → Heuristic"
echo "  Total: 210 experiments"
echo "================================================================="

# 1. S5 (slowest — neural model with checkpoint)
run_scenario "s5" \
    "lob_impact/1.aggressive_scenario_s5.py" \
    "write_config_s5" \
    "aggressive_scenario" \
    "s5"

# 2. CGAN (neural model, needs pip install)
run_scenario "cgan" \
    "lob_impact/5v2.aggressive_scenario_cgan.py" \
    "write_config_cgan" \
    "cgan_aggressive_scenario" \
    "cgan"

# 3. CST (parametric model, needs scratch mount)
run_scenario "cst" \
    "lob_impact/4.aggressive_scenario_cst.py" \
    "write_config_cst" \
    "cst_scenario" \
    "cst" \
    "-v ${SCRATCH_DIR}:/home/myuser/scratch"

# 4. Historic (replay, no model)
run_scenario "historic" \
    "lob_impact/2.historic_scenario.py" \
    "write_config_historic_heuristic" \
    "historic_scenario" \
    "hist"

# 5. Heuristic (replay + price shift, no model)
run_scenario "heuristic" \
    "lob_impact/3.heuristic_scenario.py" \
    "write_config_historic_heuristic" \
    "heuristic_scenario" \
    "heur"

echo ""
echo "================================================================="
echo "  All 5 scenarios completed!"
echo "  Finished: $(date)"
echo "================================================================="
