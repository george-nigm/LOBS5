#!/bin/bash
# =============================================================================
# Isambard SLURM array launch script for market impact experiments
#
# Each model runs as a separate SLURM array job (--array=0-119).
# Each array task = 1 (stock × grid_point × volume × direction) combination.
# Each task uses 1 GPU.
#
# Usage:
#   # Submit all 8 models (8 array jobs × 120 tasks each):
#   bash lob_impact/stage3_run/run_isambard_c10x_v2.sh submit
#
#   # Submit specific model:
#   bash lob_impact/stage3_run/run_isambard_c10x_v2.sh submit s5_150m
#
#   # Direct single-task run (called by SLURM):
#   MODEL=s5_150m sbatch --array=0-119 lob_impact/stage3_run/run_isambard_c10x_v2.sh
# =============================================================================
#SBATCH --job-name=impact
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/impact_%x_%A_%a.out
#SBATCH --error=logs/impact_%x_%A_%a.err

set -euo pipefail

# ── Isambard paths ──
PROJECT_DIR="${PROJECT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5}"
LUS="/lus/lfs1aip2/projects/public/s5e/quant_team"   # quant/ space moved here (2026); checkpoints under $LUS/quant/AlphaTrade/experiments
SAVE_BASE="${PROJECT_DIR}/data/evalsequences/aggressive_scenario_v3"
CST_PARAMS_DIR="${PROJECT_DIR}/data/checkpoints/cst_params"

# ── Model definitions ──
# Format: "label|script|ckpt_path|checkpoint_step|book_dim|data_variant"
declare -A MODELS=(
    # STALE ckpt path: exp_J1-sparse-book-anchoring is gone at the new LUS base — set the current job ID before using.
    [lobs5]="LobS5|lob_impact/scenarios/1.aggressive_scenario_s5_v3.py|${LUS}/quant/AlphaTrade/experiments/exp_J1-sparse-book-anchoring/checkpoints/j2633975_gao5ok51_2633975|61037|503|v3"
    [s5_150m]="S5-150M|lob_impact/scenarios/1.aggressive_scenario_s5_v3.py|${LUS}/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440|135458|503|v3"
    [s5_4k]="S5-4K|lob_impact/scenarios/1.aggressive_scenario_s5_v3.py|${LUS}/quant/AlphaTrade/experiments/exp_H2-context-scale/checkpoints/j2504167_y0c4j6l3_2504167|100378|503|v3"
    # STALE ckpt path: exp_J2_muon_optimizer/checkpoints is empty at the new base; leaderboard now has S5-360m-adamw (different run). Set the current job ID before using.
    [s5_360m]="S5-360M|lob_impact/scenarios/1.aggressive_scenario_s5_v3.py|${LUS}/quant/AlphaTrade/experiments/exp_J2_muon_optimizer/checkpoints/j2731367_u5xps1po_2731367|34158|503|v3"
    [zero]="ZeroInsertions|lob_impact/scenarios/2.historic_scenario.py|||503|base"
    [historic]="Historic|lob_impact/scenarios/2.historic_scenario.py|||503|base"
    [heuristic]="Heuristic|lob_impact/scenarios/3.heuristic_scenario.py|||503|base"
    [cst]="CST|lob_impact/scenarios/4.aggressive_scenario_cst.py|||503|base"
    [cgan]="CGAN|lob_impact/scenarios/5v2.aggressive_scenario_cgan.py|${PROJECT_DIR}/data/checkpoints/cgan|null|503|base"
)

# Default run set excludes lobs5 / s5_360m (stale ckpt paths — re-add once their job IDs are set).
ALL_MODEL_KEYS=(s5_150m s5_4k zero historic heuristic cst cgan)

# Models that need GPU vs CPU-only
GPU_MODELS="lobs5 s5_150m s5_4k s5_360m cgan"
needs_gpu() { [[ " ${GPU_MODELS} " == *" $1 "* ]]; }

# ── Stock data (S&P500) ──
# Preprocessed S&P500 lives as monthly squashfs shards + a JSON index:
#   ${SQUASHFS_DIR}/shard_YYYY-MM.squashfs   (inside: <TICKER>/<TICKER>_<date>_..._{message,orderbook}_10_proc.npy)
#   ${SQUASHFS_DIR}/index_YYYY-MM.json
SQUASHFS_DIR="${LUS}/lob_preproc_sp500_squashfs"
#
# TODO (not yet wired — see README "Known gaps"): the scenario reads a plain dir of *_proc.npy,
# so the target month's shard must first be MOUNTED (squashfuse / apptainer) to expose <TICKER>/ dirs,
# then data_dir = "${DATA_MOUNT}/${stock}". Also: shards are L10 (orderbook 43 cols) whereas the S5
# checkpoints expect the L500 wide book (book_dim=503) — resolve the book-width transform before running.
get_data_dir() {
    local stock=$1 variant=$2
    echo "${DATA_MOUNT:?set DATA_MOUNT to a mounted shard dir}/${stock}"
}

declare -A STOCK_TICK=(
    [GOOG]=100
)

# ── Grid (c10x_v2) ──
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
STOCKS=(GOOG)
VOLUMES=(75 300 485)
N_COND=500

# Total tasks: 1 stock × 10 grid × 3 vol × 2 dir = 60
N_TASKS=60

# ── Config writer ──
write_config() {
    local file=$1 n_ins=$2 n_cool=$3 save_dir=$4 vol=$5
    local stock=$6 data_dir=$7 tick=$8 ckpt=$9 ckpt_step=${10} book_dim=${11}
    mkdir -p "$(dirname "$file")"
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
checkpoint_step: ${ckpt_step}
chunk_size: 5
n_samples: 2048
batch_size: 64
rng_seed: 42
stock: "${stock}"
data_dir: "${data_dir}"
ckpt_path: "${ckpt}"
save_dir: "${save_dir}"
tick_size: ${tick}
sample_top_n: -1
n_vol_series: 500
book_dim: ${book_dim}
test_split: 0
EOF
}

# ── Decode SLURM_ARRAY_TASK_ID → (stock, grid, vol, dir) ──
decode_task_id() {
    local tid=$1
    local stock_idx=0
    local grid_idx=$((tid / 6))
    local rem2=$((tid % 6))
    local vol_idx=$((rem2 / 2))
    local dir_idx=$((rem2 % 2))

    TASK_STOCK="${STOCKS[$stock_idx]}"
    TASK_GRID="${GRID[$grid_idx]}"
    TASK_VOL="${VOLUMES[$vol_idx]}"
    TASK_DIR="$dir_idx"
}

# =============================================================================
# MODE 1: Submit array jobs (called from login node)
# =============================================================================
if [ "${1:-}" = "submit" ]; then
    mkdir -p "${PROJECT_DIR}/logs"
    shift
    if [ $# -gt 0 ]; then
        SUBMIT_KEYS=("$@")
    else
        SUBMIT_KEYS=("${ALL_MODEL_KEYS[@]}")
    fi

    echo "Submitting ${#SUBMIT_KEYS[@]} array jobs (${N_TASKS} tasks each)..."
    for model_key in "${SUBMIT_KEYS[@]}"; do
        IFS='|' read -r label _ _ _ _ _ <<< "${MODELS[$model_key]}"
        if needs_gpu "$model_key"; then
            gres_flag="--gres=gpu:1"
            time_flag="--time=12:00:00"
            tag="GPU"
        else
            gres_flag="--gres=gpu:0"
            time_flag="--time=04:00:00"
            tag="CPU"
        fi
        job_id=$(MODEL="$model_key" sbatch \
            --array=0-$((N_TASKS - 1)) \
            --job-name="impact_${model_key}" \
            ${gres_flag} ${time_flag} \
            --parsable \
            "${PROJECT_DIR}/lob_impact/stage3_run/run_isambard_c10x_v2.sh")
        echo "  ${label} (${model_key}): job ${job_id}, ${N_TASKS} tasks [${tag}]"
    done
    echo "Done. Monitor with: squeue -u \$USER"
    exit 0
fi

# =============================================================================
# MODE 2: Run single task (called by SLURM)
# =============================================================================

# ── Conda (set +u around activate to avoid unbound variable errors in conda scripts) ──
set +u
source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

# ── Env ──
model_key="${MODEL:?MODEL env var required}"
if needs_gpu "$model_key"; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
    export TF_FORCE_GPU_ALLOW_GROWTH=true
else
    export JAX_PLATFORMS=cpu
fi
task_id="${SLURM_ARRAY_TASK_ID:?Must run as SLURM array task}"

IFS='|' read -r label script ckpt ckpt_step book_dim data_variant <<< "${MODELS[$model_key]}"
decode_task_id "$task_id"

read -r i mb <<< "$TASK_GRID"
c=$((i * 10))
total=$((11 * i * mb))
cntxt=$((total * 100 / N_COND))

stock="$TASK_STOCK"
vol="$TASK_VOL"
dir="$TASK_DIR"
dir_name=$( [ "$dir" = "0" ] && echo "buy" || echo "sell" )

data_dir="$(get_data_dir "$stock" "$data_variant")"
tick="${STOCK_TICK[$stock]}"

folder_name="i${i}_c${c}_mb${mb}_v${vol}_cntxt${cntxt}%"
save_dir="${SAVE_BASE}/${label}/context_${N_COND}_${dir_name}/${folder_name}"

# ZeroInsertions: same total messages, zero injections
if [ "$model_key" = "zero" ]; then
    n_ins=0
    n_cool=$((i + c))
else
    n_ins=$i
    n_cool=$c
fi

CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_isambard/${model_key}"
cfg_file="${CONFIGS_DIR}/cfg_${stock,,}_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
write_config "$cfg_file" "$n_ins" "$n_cool" "$save_dir" "$vol" \
    "$stock" "$data_dir" "$tick" "$ckpt" "$ckpt_step" "$book_dim"

# CST model needs extra fields: params_file, n_levels, num_ticks
if [ "$model_key" = "cst" ]; then
    cat >> "$cfg_file" << EOF
params_file: "${CST_PARAMS_DIR}/cst_params_${stock}_dec2025.pkl"
num_ticks: 500
n_levels: 10
EOF
fi

# CGAN model needs checkpoint, scalers, interarrival paths
if [ "$model_key" = "cgan" ]; then
    cgan_dir="/scratch/s5e/aramis.s5e/cgan_runs/GOOG/models/GOOG/NEW_SET_lb100_['20251224', '20251226', '20251230']_v2_41"
    cat >> "$cfg_file" << EOF
cgan_checkpoint: "${cgan_dir}/checkpoints/model_39.ckpt"
cgan_scalers: "${cgan_dir}/data__scalers.pickle"
cgan_interarrival_times: "${cgan_dir}/interarrival_times"
n_levels: 10
EOF
fi

echo "=== [${SLURM_ARRAY_JOB_ID}_${task_id}] ${label} / ${stock} / i${i}_c${c}_mb${mb}_v${vol}_${dir_name} ==="
echo "Config: ${cfg_file}"
echo "Data:   ${data_dir}"
echo "Save:   ${save_dir}"

python -u "${PROJECT_DIR}/${script}" \
    --config "$cfg_file" \
    --n_gen_msgs "$mb" \
    --direction "$dir"

echo "=== Task ${task_id} finished: $(date) ==="
