#!/bin/bash
#SBATCH --job-name=lobimp_a5_beta
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/beta_%j.out
#
# Action 5 (beta) — runs the beta analysis on a COMPUTE node (never the login node).
# No data mount needed: reads results/grid (NFS) + the Action-2 daily CSV (NFS), CPU-only.
#
#   sbatch lob_impact/5_analysis/beta/run_beta.sh                       # auto-pick newest OHLC daily csv
#   DAILY=/path/daily_h_l_all.csv sbatch lob_impact/5_analysis/beta/run_beta.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
HERE="${IMPACT_DIR}/5_analysis/beta"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"   # post-audit grid; GRID=... to point elsewhere
STOCKS="${STOCKS:-EA NVDA AMD}"
MODELS="${MODELS:-Historic,Heuristic,Propagator,Hawkes,CST,NMZI,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p "${HERE}/logs"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

# newest Action-2 daily CSV that actually carries OHLC (open_price col)
if [ -z "${DAILY:-}" ]; then
  for f in $(ls -t "${IMPACT_DIR}"/2_daily_stats/results/*/daily_h_l_all.csv 2>/dev/null); do
    head -1 "$f" | grep -q open_price && { DAILY="$f"; break; }
  done
fi
[ -n "${DAILY:-}" ] || { echo "FATAL: no OHLC daily_h_l_all.csv found (run Action 2 first)" >&2; exit 3; }
echo "[$(date)] host $(hostname) | grid ${GRID} | daily ${DAILY}"

cd "$IMPACT_DIR"
echo ">>> HEADLINE: beta binned (signed conditional bin-means, literature-standard δ; no I>0 selection)"
for s in $STOCKS; do
  python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock "$s" --models "$MODELS"
  python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock "$s" --models "$MODELS" --method none
done
echo ">>> beta master curve (Parkinson, intercept estimator) [per-point: E[log I|I>0] biased — diagnostic only]"
python3.11 5_analysis/beta/beta_master_curve.py --grid "$GRID" --daily "$DAILY"
echo ">>> beta 5-method (5 sigma estimators + intercept, day-clustered bootstrap CI) [per-point: biased — diagnostic only]"
python3.11 5_analysis/beta/beta_5method.py --grid "$GRID" --daily "$DAILY"
echo ">>> beta grid 3x5 (rows: all-points | by-k origin | by-k intercept; cols: 5 sigma) [per-point: biased — diagnostic only]"
python3.11 5_analysis/beta/beta_grid.py --grid "$GRID" --daily "$DAILY"
echo "[$(date)] done -> 5_analysis/beta/results/{beta_binned,beta_master,beta_5method,beta_grid}/"
