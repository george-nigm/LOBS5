#!/bin/bash
#SBATCH --job-name=lobimp_a5_decay
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/decay/logs/decay_%j.out
#
# Action 5 (decay / Step-5(II) dynamic relaxation) — runs on a COMPUTE node (never the login node:
# it globs thousands of grid CSVs -> Lustre metadata load must stay off the login node).
# CPU-only, no data mount: reads the grid relaxation dirs directly.
#
#   sbatch lob_impact/5_analysis/decay/run_decay.sh              # FULL, all samples, all 7 models
#   MODE=smoke sbatch lob_impact/5_analysis/decay/run_decay.sh   # capped (--max_samples 60) sanity run
#   STOCK=EA MODELS=Historic,Mamba3 sbatch lob_impact/5_analysis/decay/run_decay.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid}"
STOCK="${STOCK:-EA}"
MODELS="${MODELS:-Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k,S5_4k}"
MODE="${MODE:-full}"
mkdir -p "${IMPACT_DIR}/5_analysis/decay/logs"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

CAP=""
[ "$MODE" = "smoke" ] && CAP="--max_samples ${MAX_SAMPLES:-60}"

echo "[$(date)] host $(hostname) | grid ${GRID} | stock ${STOCK} | mode ${MODE} ${CAP}"
echo "         models: ${MODELS}"
cd "$IMPACT_DIR"
python3.11 5_analysis/decay/decay_full.py \
  --grid "$GRID" --stock "$STOCK" --models "$MODELS" $CAP
echo "[$(date)] done -> 5_analysis/decay/results/decay_{master,propagator,scorecard}_${STOCK}.{png,csv}"
