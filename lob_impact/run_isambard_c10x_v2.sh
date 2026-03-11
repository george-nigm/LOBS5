#!/bin/bash
# =============================================================================
# Isambard SLURM launch script for market impact experiments
# Replaces Docker-based run_context_500_c10x_v2.sh
#
# Usage:
#   # Run all models for both stocks:
#   sbatch lob_impact/run_isambard_c10x_v2.sh
#
#   # Run specific model and stock:
#   MODEL=s5_120m STOCK=GOOG sbatch lob_impact/run_isambard_c10x_v2.sh
# =============================================================================
#SBATCH --job-name=lob_impact
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/impact_%j_%a.out
#SBATCH --error=logs/impact_%j_%a.err

set -euo pipefail

# ── Isambard paths ──
PROJECT_DIR="/lus/lfs1aip2/home/s5e/LOBS5"
DATA_BASE="/lus/lfs1aip2/home/s5e/data"
CKPT_BASE="/lus/lfs1aip2/home/s5e/data/checkpoints"
SAVE_BASE="/lus/lfs1aip2/home/s5e/data/evalsequences/aggressive_scenario_v3"

# ── Conda ──
source /lus/lfs1aip2/home/s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

# ── Env ──
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true

# ── Model definitions ──
# Format: "label|script|ckpt_path|checkpoint_step|book_dim"
declare -A MODELS=(
    [lobs5]="LobS5|lob_impact/1.aggressive_scenario_s5.py|${CKPT_BASE}/lobs5_v2/twilight-sound-77_s42sujip|null|503"
    [s5_120m]="S5-120M|lob_impact/1.aggressive_scenario_s5_v3.py|${CKPT_BASE}/lobs5_v3/j2514440_bkotgtm5_2514440|135458|503"
    [s5_4k]="S5-4K|lob_impact/1.aggressive_scenario_s5_v3.py|${CKPT_BASE}/lobs5_v3/j2504167_y0c4j6l3_2504167|100378|503"
    [historic]="Historic|lob_impact/2.historic_scenario.py|||503"
    [heuristic]="Heuristic|lob_impact/3.heuristic_scenario.py|||503"
    [cst]="CST|lob_impact/4.aggressive_scenario_cst.py|||503"
    [cgan]="CGAN|lob_impact/1.aggressive_scenario_cgan.py|${CKPT_BASE}/cgan/cgan_checkpoint|null|503"
)

# ── Stock definitions ──
declare -A STOCK_DATA=(
    [GOOG]="${DATA_BASE}/processed_data/GOOG/2026_Jan"
    [INTC]="${DATA_BASE}/processed_data/INTC/2026_Jan"
)
declare -A STOCK_TICK=(
    [GOOG]=100
    [INTC]=100
)

# ── Grid (same as c10x_v2) ──
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
N_COND=500

# ── Config writer ──
write_config() {
    local file=$1 n_ins=$2 n_cool=$3 save_dir=$4 vol=$5
    local stock=$6 data_dir=$7 tick=$8 ckpt=$9 ckpt_step=${10} book_dim=${11}
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

# ── Select model and stock ──
MODEL_KEY="${MODEL:-all}"
STOCK_KEY="${STOCK:-all}"

if [ "$MODEL_KEY" = "all" ]; then
    MODEL_KEYS=("${!MODELS[@]}")
else
    MODEL_KEYS=("$MODEL_KEY")
fi

if [ "$STOCK_KEY" = "all" ]; then
    STOCK_KEYS=("${!STOCK_DATA[@]}")
else
    STOCK_KEYS=("$STOCK_KEY")
fi

# ── Generate configs and run ──
GPU_IDX=0
N_GPUS=4  # GH200 has 4 GPUs per node

for model_key in "${MODEL_KEYS[@]}"; do
    IFS='|' read -r label script ckpt ckpt_step book_dim <<< "${MODELS[$model_key]}"

    for stock in "${STOCK_KEYS[@]}"; do
        data_dir="${STOCK_DATA[$stock]}"
        tick="${STOCK_TICK[$stock]}"

        CONFIGS_DIR="${PROJECT_DIR}/lob_impact/configs_isambard_${stock,,}_2026"
        mkdir -p "$CONFIGS_DIR"

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
                    save_dir="${SAVE_BASE}/${label}/context_${N_COND}_${dir_name}/${folder_name}"

                    cfg_file="${CONFIGS_DIR}/cfg_${model_key}_i${i}_c${c}_mb${mb}_v${vol}_${dir_name}.yaml"
                    write_config "$cfg_file" "$i" "$c" "$save_dir" "$vol" \
                        "$stock" "$data_dir" "$tick" "$ckpt" "$ckpt_step" "$book_dim"

                    JOBS+=("${cfg_file}|${mb}|${dir}|${folder_name}_${dir_name}")
                done
            done
        done

        echo "=== ${label} / ${stock}: ${#JOBS[@]} jobs ==="

        # Run jobs round-robin across GPUs
        for job_str in "${JOBS[@]}"; do
            IFS='|' read -r cfg mb dir name <<< "$job_str"
            gpu=$((GPU_IDX % N_GPUS))
            GPU_IDX=$((GPU_IDX + 1))

            echo "[GPU ${gpu}] ${name}"
            CUDA_VISIBLE_DEVICES=$gpu python -u "${PROJECT_DIR}/${script}" \
                --config "$cfg" \
                --n_gen_msgs "$mb" \
                --direction "$dir" &

            # Wait if all GPUs are busy
            if (( GPU_IDX % N_GPUS == 0 )); then
                wait
            fi
        done
        wait

        unset JOBS
    done
done

echo "=== All experiments finished! $(date) ==="
