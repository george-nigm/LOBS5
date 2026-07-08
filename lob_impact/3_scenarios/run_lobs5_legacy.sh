#!/bin/bash
#SBATCH --job-name=lobs5_legacy
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios/logs/lobs5_legacy_%j.out
#
# twilight-sound-77 (old-codebase S5) on GOOG 2023_Jan. No shard mount: flair06 old-format
# data sits directly on Lustre. Env-overridable knobs render a per-run config:
#   MODE=smoke|full  SHAPE=beta|relaxation  DIR=buy|sell  N_SAMPLES  PER_DAY_CSV
#   sbatch run_lobs5_legacy.sh                      # smoke: 8 samples, 3 ins, mb=100
#   MODE=full SHAPE=beta DIR=buy sbatch run_lobs5_legacy.sh
set -euo pipefail
# --export=ALL leaks the submitter's TMPDIR (login-node /local/user/...) which doesn't exist
# on compute nodes -> torch dies at import in tempfile.mkdtemp. Force a node-valid tmp.
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
IMPACT_DIR=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
HERE="$IMPACT_DIR/3_scenarios"
REPO_ROOT="$(dirname "$IMPACT_DIR")"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Alphatrade:${PYTHONPATH:-}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u

MODE="${MODE:-smoke}"
SHAPE="${SHAPE:-beta}"          # beta: 100 ins / 0 cool | relaxation: 10 ins / 100 cool
DIRN="${DIR:-buy}"
SAVE_BASE="${SAVE_BASE:-/lus/lfs1aip2/projects/u6gb/lob_impact_legacy}"
PER_DAY_CSV="${PER_DAY_CSV:-}"  # per_day_params CSV for GOOG-2023 (day,mult,child,mb); empty = static

if [ "$SHAPE" = beta ]; then N_INS=100; N_COOL=0; else N_INS=10; N_COOL=100; fi
[ "$DIRN" = buy ] && DIR_INT=0 || DIR_INT=1
if [ "$MODE" = smoke ]; then
  N_INS=3; N_COOL=0; MB=100; N_SAMPLES="${N_SAMPLES:-8}"; SAVE_DIR="$SAVE_BASE/smoke/GOOG-LobS5-$SHAPE/$DIRN"
else
  MB=729; N_SAMPLES="${N_SAMPLES:-64}"; SAVE_DIR="$SAVE_BASE/GOOG-LobS5-$SHAPE/$DIRN"
fi

CFG="$HERE/logs/cfg_lobs5_legacy_${SLURM_JOB_ID:-$$}.yaml"
python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$HERE/config_lobs5_legacy_goog.yaml"))
cfg.update(dict(num_insertions=$N_INS, num_coolings=$N_COOL, n_gen_msgs=$MB,
                direction=$DIR_INT, n_samples=$N_SAMPLES, save_dir="$SAVE_DIR"))
pdc = "$PER_DAY_CSV"
if pdc: cfg['per_day_params'] = pdc
yaml.dump(cfg, open("$CFG", "w"))
print("rendered:", "$CFG"); print(cfg)
PY

cd "$REPO_ROOT"
python -u "$HERE/lobs5_scenario.py" --config "$CFG"
echo ">>> lobs5_legacy $MODE complete -> $SAVE_DIR"
