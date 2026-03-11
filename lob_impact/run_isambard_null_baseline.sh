#!/bin/bash
# =============================================================================
# Isambard SLURM launch script for null baseline experiments
# Runs S5 models without any aggressive order injection
#
# Usage:
#   sbatch lob_impact/run_isambard_null_baseline.sh
# =============================================================================
#SBATCH --job-name=null_baseline
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/null_%j.out
#SBATCH --error=logs/null_%j.err

set -euo pipefail

# ── Isambard paths ──
PROJECT_DIR="/home/s5e/georgenigm.s5e/LOBS5_11_march"
DATA_BASE="/home/s5e/georgenigm.s5e/LOBS5_11_march/data"
CKPT_BASE="/home/s5e/georgenigm.s5e/LOBS5_11_march/data/checkpoints"
SAVE_BASE="/home/s5e/georgenigm.s5e/LOBS5_11_march/data/evalsequences/null_baseline"

# ── Conda ──
source /home/s5e/georgenigm.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

# ── Env ──
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true

SCRIPT_PY="lob_impact/0.null_baseline_s5.py"
N_GEN=500

# ── S5 Models (neural models only - baselines don't need null test) ──
declare -A MODELS=(
    [lobs5]="LobS5|${CKPT_BASE}/lobs5_v2/twilight-sound-77_s42sujip|null|503"
    [s5_120m]="S5-120M|${CKPT_BASE}/lobs5_v3/j2514440_bkotgtm5_2514440|135458|503"
    [s5_4k]="S5-4K|${CKPT_BASE}/lobs5_v3/j2504167_y0c4j6l3_2504167|100378|503"
)

# ── Stocks ──
declare -A STOCK_DATA=(
    [GOOG]="${DATA_BASE}/processed_data/GOOG/2026_Jan"
    [INTC]="${DATA_BASE}/processed_data/INTC/2026_Jan"
)
declare -A STOCK_TICK=(
    [GOOG]=100
    [INTC]=100
)

GPU_IDX=0
N_GPUS=4

for model_key in "${!MODELS[@]}"; do
    IFS='|' read -r label ckpt ckpt_step book_dim <<< "${MODELS[$model_key]}"

    for stock in "${!STOCK_DATA[@]}"; do
        data_dir="${STOCK_DATA[$stock]}"
        tick="${STOCK_TICK[$stock]}"
        save_dir="${SAVE_BASE}/${label}/${stock}"

        cfg_file="${PROJECT_DIR}/lob_impact/configs_isambard_null/cfg_${model_key}_${stock}.yaml"
        mkdir -p "$(dirname "$cfg_file")"

        cat > "$cfg_file" << EOF
n_gen_msgs: ${N_GEN}
n_cond_msgs: 500
n_eval_msgs_dataset: 500
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

        gpu=$((GPU_IDX % N_GPUS))
        GPU_IDX=$((GPU_IDX + 1))

        echo "[GPU ${gpu}] Null baseline: ${label} / ${stock}"
        CUDA_VISIBLE_DEVICES=$gpu python -u "${PROJECT_DIR}/${SCRIPT_PY}" \
            --config "$cfg_file" &

        if (( GPU_IDX % N_GPUS == 0 )); then
            wait
        fi
    done
done

wait
echo "=== Null baseline experiments finished! $(date) ==="
