#!/bin/bash
# =============================================================================
# lob_impact/3_scenarios/run_experiments.sh
# Reproducible market-impact experiment launcher (Action 3).
#
# Loops 3 models x 3 stocks x 2 scenario shapes x 2 directions x mb-values,
# renders a per-run YAML from a shape template, and runs the scenario script.
# Direct sequential loop (not SLURM) so the pipeline is verifiable step-by-step.
#
#   export DATA_MOUNT=<mounted squashfs root>     # one subdir per ticker
#   export PROJECT_DIR=/home/u6gb/georgenigm.u6gb/LOBS5   # optional (defaults to repo root)
#   bash 3_scenarios/run_experiments.sh smoke            # 1 combo, tiny n_samples
#   bash 3_scenarios/run_experiments.sh full             # whole grid
#   bash 3_scenarios/run_experiments.sh full model_a     # restrict to model key(s)
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../lob_impact/3_scenarios
IMPACT_DIR="$(dirname "$HERE")"                         # .../lob_impact
REPO_ROOT="$(dirname "$IMPACT_DIR")"                    # .../LOBS5

# --- Env-overridable anchors -------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
DATA_MOUNT="${DATA_MOUNT:?set DATA_MOUNT to the mounted squashfs root (one subdir per ticker)}"
CKPT_BASE="${CKPT_BASE:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments}"
SAVE_BASE="${SAVE_BASE:-${IMPACT_DIR}/data/evalsequences/impact_v4}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

# --- Models (PLACEHOLDERS — fill ckpt_path / step / book_dim later) ----------
# Format: "label|script|ckpt_path|checkpoint_step|book_dim"  (script relative to 3_scenarios/)
# Verified S5 checkpoints available now (uncomment + use as model entries):
#   S5-150M -> ${CKPT_BASE}/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440 | 135458 | 503
#   S5-4K   -> ${CKPT_BASE}/exp_H2-context-scale/checkpoints/j2504167_y0c4j6l3_2504167 | 100378 | 503
declare -A MODELS=(
  [model_a]="ModelA|1.aggressive_scenario_s5_v3.py|${CKPT_BASE}/PLACEHOLDER_A|PLACEHOLDER|503"
  [model_b]="ModelB|1.aggressive_scenario_s5_v3.py|${CKPT_BASE}/PLACEHOLDER_B|PLACEHOLDER|503"
  [model_c]="ModelC|1.aggressive_scenario_s5_v3.py|${CKPT_BASE}/PLACEHOLDER_C|PLACEHOLDER|503"
)
MODEL_KEYS=(model_a model_b model_c)

# --- Grid axes ---------------------------------------------------------------
STOCKS=(EA NVDA AMD)
# shape: "name|num_insertions|num_coolings|template"
SHAPES=(
  "bet_composition|100|0|config_bet_composition.yaml"   # Shape I  -> beta
  "beta_decay|10|100|config_beta_decay.yaml"            # Shape II -> decay
)
MB_VALUES=(5 10 15 20)        # messages-between sweep (the only axis the user varies in-config)
DIRECTIONS=(buy sell)
# Optional per-stock tick override (fill if EA/NVDA/AMD tick != template default 100):
declare -A STOCK_TICK=()        # e.g. ([EA]=100 [NVDA]=100 [AMD]=100)

# --- Mode --------------------------------------------------------------------
MODE="${1:-smoke}"; shift || true
if [ "$#" -gt 0 ]; then MODEL_KEYS=("$@"); fi   # optional model-key restriction

if [ "$MODE" = "smoke" ]; then
  MODEL_KEYS=("${MODEL_KEYS[0]}"); STOCKS=("${STOCKS[0]}")
  SHAPES=("${SHAPES[0]}");          DIRECTIONS=(buy); MB_VALUES=(5)
  N_SAMPLES_OVERRIDE=64
  echo ">>> SMOKE: ${MODEL_KEYS[0]} x ${STOCKS[0]} x bet_composition x buy x mb=5, n_samples=64"
elif [ "$MODE" = "full" ]; then
  N_SAMPLES_OVERRIDE=""
  echo ">>> FULL: ${#MODEL_KEYS[@]} models x ${#STOCKS[@]} stocks x ${#SHAPES[@]} shapes x 2 dir x ${#MB_VALUES[@]} mb"
else
  echo "usage: $0 {smoke|full} [model_key ...]" >&2; exit 2
fi

dir_to_int() { [ "$1" = "buy" ] && echo 0 || echo 1; }

# --- Render a per-run config from a template (robust template-merge) ---------
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

# --- Grid --------------------------------------------------------------------
for model_key in "${MODEL_KEYS[@]}"; do
  IFS='|' read -r LABEL SCRIPT CKPT CKPT_STEP BOOK_DIM <<< "${MODELS[$model_key]}"
  abs_script="${HERE}/${SCRIPT}"
  for STOCK in "${STOCKS[@]}"; do
    DATA_DIR="${DATA_MOUNT}/${STOCK}"
    TICK="${STOCK_TICK[$STOCK]:-}"
    for shape_row in "${SHAPES[@]}"; do
      IFS='|' read -r SHAPE_NAME N_INS N_COOL TEMPLATE <<< "$shape_row"
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
          python -u "$abs_script" \
            --config "$cfg_file" \
            --n_gen_msgs "$mb" \
            --direction "$dir_int"
        done
      done
    done
  done
done
echo ">>> ${MODE} run complete."
