#!/bin/bash
# =============================================================================
# Run all 32 context research experiments across 8 GPUs
# Grid: i=[3,5] × n_cond=[500,250] × mb=[5,15,25,50] × dir=[buy,sell]
#
# Cooling periods:
#   n_cond=500: i=3→c=12, i=5→c=15
#   n_cond=250: i=3→c=30, i=5→c=50
#
# Usage:
#   chmod +x lob_impact/run_context_experiments.sh
#   ./lob_impact/run_context_experiments.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="/scratch/local/homes/80/georgenigm/LOBS5"
CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_context_run"
LOGS_DIR="${PROJECT_DIR}/output/evalsequences/aggressive_scenario/logs"
DOCKER_IMAGE="georgenigm_25jan"
WANDB_KEY="74075d19681454163130e79756ce47db4dcb571f"
SCRIPT_PY="lob_impact/1.aggressive_scenario_s5.py"

SAVE_BASE="/home/myuser/data/evalsequences/aggressive_scenario"

mkdir -p "$CONFIGS_DIR" "$LOGS_DIR"

# ---- Helper: cooling periods lookup ----
get_coolings() {
    local i=$1 n_cond=$2
    if   [ "$n_cond" = "500" ] && [ "$i" = "3" ]; then echo 12
    elif [ "$n_cond" = "500" ] && [ "$i" = "5" ]; then echo 15
    elif [ "$n_cond" = "250" ] && [ "$i" = "3" ]; then echo 30
    elif [ "$n_cond" = "250" ] && [ "$i" = "5" ]; then echo 50
    fi
}

# ---- Helper: write YAML config ----
write_config() {
    local file=$1 n_ins=$2 n_cool=$3 n_cond=$4 save_dir=$5
    cat > "$file" << EOF
n_gen_msgs: 50
num_insertions: ${n_ins}
num_coolings: ${n_cool}
n_cond_msgs: ${n_cond}
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

# ---- Helper: run one experiment (detached docker + docker wait) ----
run_one() {
    local gpu=$1 cfg_container=$2 mb=$3 dir=$4 job_name=$5
    local container_name="ctx_${job_name}"

    echo "[GPU ${gpu}] START: ${job_name}  $(date '+%H:%M:%S')"

    # Launch detached so container survives if parent shell dies
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

    # Stream logs to file in background
    docker logs -f "${container_name}" > "${LOGS_DIR}/${job_name}.log" 2>&1 &

    # Wait for container to finish
    docker wait "${container_name}" > /dev/null 2>&1

    echo "[GPU ${gpu}] DONE:  ${job_name}  $(date '+%H:%M:%S')"
}

# =============================================================================
# Generate all 32 configs and build job list
# =============================================================================
# Job format: "cfg_container_path mb direction job_name workload"
declare -a JOBS=()

for n_cond in 500 250; do
    for i in 3 5; do
        c=$(get_coolings $i $n_cond)
        for mb in 5 15 25 50; do
            for dir in 0 1; do
                dir_name=$( [ "$dir" = "0" ] && echo "buy" || echo "sell" )
                cntxt=$(( (i + c) * mb * 100 / n_cond ))
                folder_name="i${i}_c${c}_mb${mb}_cntxt${cntxt}%"
                save_dir="${SAVE_BASE}/context_${n_cond}_${dir_name}/${folder_name}"
                workload=$(( (i + c) * mb ))

                # Write config file
                cfg_host="${CONFIGS_DIR}/cfg_i${i}_c${c}_cond${n_cond}_mb${mb}_${dir_name}.yaml"
                cfg_container="lob_impact/configs_context_run/cfg_i${i}_c${c}_cond${n_cond}_mb${mb}_${dir_name}.yaml"
                write_config "$cfg_host" "$i" "$c" "$n_cond" "$save_dir"

                job_name="i${i}_c${c}_mb${mb}_${dir_name}_cond${n_cond}"
                JOBS+=("${cfg_container}|${mb}|${dir}|${job_name}|${workload}")
            done
        done
    done
done

echo "=== Total jobs: ${#JOBS[@]} ==="

# =============================================================================
# Sort jobs by workload (descending) for better load balancing
# =============================================================================
IFS=$'\n' SORTED_JOBS=($(for j in "${JOBS[@]}"; do echo "$j"; done | sort -t'|' -k5 -rn))
unset IFS

# =============================================================================
# Assign to 8 GPUs using snake pattern for balanced workload
# GPU assignment: 0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0,0,1,...
# =============================================================================
declare -a GPU_JOBS_0=() GPU_JOBS_1=() GPU_JOBS_2=() GPU_JOBS_3=()
declare -a GPU_JOBS_4=() GPU_JOBS_5=() GPU_JOBS_6=() GPU_JOBS_7=()

SNAKE=(0 1 2 3 4 5 6 7 7 6 5 4 3 2 1 0 0 1 2 3 4 5 6 7 7 6 5 4 3 2 1 0)

for idx in "${!SORTED_JOBS[@]}"; do
    gpu=${SNAKE[$idx]}
    eval "GPU_JOBS_${gpu}+=(\"${SORTED_JOBS[$idx]}\")"
done

# =============================================================================
# Print assignment
# =============================================================================
echo ""
echo "=== GPU Assignment ==="
for gpu in 0 1 2 3 4 5 6 7; do
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
# Launch: each GPU runs its jobs sequentially, all 8 GPUs in parallel
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

for gpu in 0 1 2 3 4 5 6 7; do
    eval "jobs=(\"\${GPU_JOBS_${gpu}[@]}\")"
    run_gpu "$gpu" "${jobs[@]}" &
done

wait
echo ""
echo "=== All experiments finished! $(date) ==="

# =============================================================================
# Post-processing: flatten exp_* folders into notebook-expected structure
# The script creates exp_N_timestamp/ inside save_dir. Notebook expects
# data_cond/ and data_gen/ directly in the folder.
# =============================================================================
echo ""
echo "=== Post-processing: flattening exp_* folders ==="

OUTPUT_BASE="${PROJECT_DIR}/output/evalsequences/aggressive_scenario"

for context_dir in "${OUTPUT_BASE}"/context_*; do
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

echo "=== Done! Data ready for 99.context_research.ipynb ==="
