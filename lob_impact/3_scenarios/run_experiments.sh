#!/bin/bash
#SBATCH --job-name=lobimp_a3
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios/logs/run_%j.out
#
# Action 3 — scenario generation. Self-mounts a month shard and runs the grid.
#   sbatch lob_impact/3_scenarios/run_experiments.sh smoke            # 1 combo, tiny, num_insertions=3
#   sbatch lob_impact/3_scenarios/run_experiments.sh full
#   sbatch lob_impact/3_scenarios/run_experiments.sh smoke s5_4k      # pick model key(s)
# NOTE: neural (S5) models need a GPU — edit the #SBATCH lines to add e.g.
#       --gres=gpu:1 and a GPU partition before submitting s5_* / cgan.
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
HERE="${IMPACT_DIR}/3_scenarios"
REPO_ROOT="$(dirname "$IMPACT_DIR")"
RUN_TS="$(date +%Y%m%d-%H%M%S)"

PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
CKPT_BASE="${CKPT_BASE:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments}"
SAVE_BASE="${SAVE_BASE:-${HERE}/results/run_${RUN_TS}}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

# --- Python env (jax, lob, model deps) ---  (conda activate scripts aren't set -u safe)
set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

# --- Data: self-mount a month shard unless DATA_MOUNT is already provided ---
SRC="${SRC:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}"
SHARD="${SHARD:-shard_2026-01.squashfs}"
if [ -z "${DATA_MOUNT:-}" ]; then
  DATA_MOUNT="${TMPDIR:-/tmp}/s5e_mnt_${SLURM_JOB_ID:-$$}"
  mkdir -p "$DATA_MOUNT"
  echo "[$(date)] mounting ${SHARD} -> ${DATA_MOUNT}"
  squashfuse_ll "${SRC}/${SHARD}" "$DATA_MOUNT"
  trap 'fusermount -u "$DATA_MOUNT" 2>/dev/null; rmdir "$DATA_MOUNT" 2>/dev/null' EXIT
fi

# --- Models: "label|script|ckpt_path|checkpoint_step|book_dim" (script relative to 3_scenarios/) ---
declare -A MODELS=(
  [historic]="Historic|2.historic_scenario.py|||503"
  [s5_150m]="S5-150M|1.aggressive_scenario_s5_v3.py|${CKPT_BASE}/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440|135458|503"
  [s5_4k]="S5-4K|1.aggressive_scenario_s5_v3.py|${CKPT_BASE}/exp_H2-context-scale/checkpoints/j2504167_y0c4j6l3_2504167|100378|503"
)
MODEL_KEYS=(historic)
STOCKS=(EA NVDA AMD)
# shape: "name|num_insertions|num_coolings|template"
SHAPES=(
  "bet_composition|100|0|config_bet_composition.yaml"   # Shape I  -> beta
  "beta_decay|10|100|config_beta_decay.yaml"           # Shape II -> decay
)
MB_VALUES=(5 10 15 20)
DIRECTIONS=(buy sell)
declare -A STOCK_TICK=()        # e.g. ([EA]=100 [NVDA]=100 [AMD]=100)

MODE="${1:-smoke}"; shift || true
[ "$#" -gt 0 ] && MODEL_KEYS=("$@")

SMOKE_N_INS=""
if [ "$MODE" = "smoke" ]; then
  MODEL_KEYS=("${MODEL_KEYS[0]}"); STOCKS=(EA)
  SHAPES=("${SHAPES[0]}"); DIRECTIONS=(buy); MB_VALUES=(5)
  N_SAMPLES_OVERRIDE=64; SMOKE_N_INS=3      # user: 3 insertions, not 100 — just to check
  echo ">>> SMOKE: ${MODEL_KEYS[0]} x EA x bet_composition x buy x mb=5, n_samples=64, num_insertions=3"
elif [ "$MODE" = "full" ]; then
  N_SAMPLES_OVERRIDE=""
  echo ">>> FULL: ${#MODEL_KEYS[@]} models x ${#STOCKS[@]} stocks x ${#SHAPES[@]} shapes x 2 dir x ${#MB_VALUES[@]} mb"
else
  echo "usage: $0 {smoke|full} [model_key ...]" >&2; exit 2
fi
echo ">>> run ${RUN_TS} | save_base ${SAVE_BASE}"

# JAX on CPU for models without a checkpoint (historic/heuristic); neural models use the GPU.
dir_to_int() { [ "$1" = "buy" ] && echo 0 || echo 1; }

render_config() {
  local tmpl="$1" out="$2"
  TMPL="$tmpl" OUT="$out" STOCK="$STOCK" DATA_DIR="$DATA_DIR" CKPT="$CKPT" \
  CKPT_STEP="$CKPT_STEP" BOOK_DIM="$BOOK_DIM" SAVE_DIR="$SAVE_DIR" \
  N_INS="$N_INS" N_COOL="$N_COOL" TICK="$TICK" N_SAMPLES_OVERRIDE="$N_SAMPLES_OVERRIDE" \
  python3 - <<'PY'
import os, yaml
cfg = yaml.safe_load(open(os.environ["TMPL"]))
cfg["stock"]          = os.environ["STOCK"]
cfg["data_dir"]       = os.environ["DATA_DIR"]
cfg["save_dir"]       = os.environ["SAVE_DIR"]
cfg["ckpt_path"]      = os.environ["CKPT"] or None
cfg["book_dim"]       = int(os.environ["BOOK_DIM"])
cfg["num_insertions"] = int(os.environ["N_INS"])
cfg["num_coolings"]   = int(os.environ["N_COOL"])
if os.environ.get("TICK"):            cfg["tick_size"] = int(os.environ["TICK"])
step = os.environ["CKPT_STEP"]
cfg["checkpoint_step"] = None if step in ("", "null", "None", "PLACEHOLDER") else int(step)
ov = os.environ["N_SAMPLES_OVERRIDE"]
if ov: cfg["n_samples"] = int(ov)
yaml.safe_dump(cfg, open(os.environ["OUT"], "w"), sort_keys=False)
PY
}

for model_key in "${MODEL_KEYS[@]}"; do
  IFS='|' read -r LABEL SCRIPT CKPT CKPT_STEP BOOK_DIM <<< "${MODELS[$model_key]}"
  abs_script="${HERE}/${SCRIPT}"
  # no checkpoint -> replay/parametric baseline -> force JAX onto CPU
  if [ -z "$CKPT" ]; then export JAX_PLATFORMS=cpu; else unset JAX_PLATFORMS; fi
  for STOCK in "${STOCKS[@]}"; do
    DATA_DIR="${DATA_MOUNT}/${STOCK}"
    TICK="${STOCK_TICK[$STOCK]:-}"
    for shape_row in "${SHAPES[@]}"; do
      IFS='|' read -r SHAPE_NAME N_INS N_COOL TEMPLATE <<< "$shape_row"
      [ -n "$SMOKE_N_INS" ] && N_INS="$SMOKE_N_INS"
      tmpl_path="${HERE}/${TEMPLATE}"
      for dir in "${DIRECTIONS[@]}"; do
        dir_int="$(dir_to_int "$dir")"
        for mb in "${MB_VALUES[@]}"; do
          run_id="${LABEL}_${STOCK}_${SHAPE_NAME}_${dir}_mb${mb}"
          SAVE_DIR="${SAVE_BASE}/${SHAPE_NAME}/${LABEL}/${STOCK}/${dir}/mb${mb}"
          cfg_dir="${SAVE_BASE}/_configs/${SHAPE_NAME}/${LABEL}/${STOCK}"
          cfg_file="${cfg_dir}/cfg_${run_id}.yaml"
          mkdir -p "$cfg_dir" "$SAVE_DIR"
          render_config "$tmpl_path" "$cfg_file"
          echo "=== ${run_id} ===  cfg=${cfg_file}  save=${SAVE_DIR}"
          python -u "$abs_script" --config "$cfg_file" --n_gen_msgs "$mb" --direction "$dir_int"
        done
      done
    done
  done
done
echo ">>> ${MODE} run complete -> ${SAVE_BASE}"
