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
MODELS="${MODELS:-Historic,Heuristic,Propagator,OW,QR,Hawkes,CST,NMZI,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p "${HERE}/logs"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

# newest Action-2 daily CSV that carries OHLC (open_price col) AND covers every requested stock.
# The plain daily_h_l_all.csv only has EA/NVDA/GOOG/AMD; MSFT and AAPL live in the *_plus.csv.
# Picking the wrong one is a SILENT failure: every model reports "0 signed points" and the job
# still exits 0, so glob the _plus variants too and verify coverage before running anything.
if [ -z "${DAILY:-}" ]; then
  for f in $(ls -t "${IMPACT_DIR}"/2_daily_stats/results/*/daily_h_l_all_plus.csv \
                   "${IMPACT_DIR}"/2_daily_stats/results/*/daily_h_l_all.csv 2>/dev/null); do
    head -1 "$f" | grep -q open_price || continue
    missing=""
    for s in $STOCKS; do grep -q "^${s}," "$f" || missing="$missing $s"; done
    [ -z "$missing" ] && { DAILY="$f"; break; }
    echo "[daily] skipping $(basename "$(dirname "$f")")/$(basename "$f") — missing:$missing" >&2
  done
fi
[ -n "${DAILY:-}" ] || { echo "FATAL: no OHLC daily table covering [$STOCKS] (run Action 2 first)" >&2; exit 3; }
for s in $STOCKS; do
  grep -q "^${s}," "$DAILY" || { echo "FATAL: $s absent from $DAILY — every model would silently score 0 points" >&2; exit 4; }
done
echo "[$(date)] host $(hostname) | grid ${GRID} | daily ${DAILY}"

cd "$IMPACT_DIR"
echo ">>> HEADLINE: beta binned (signed conditional bin-means, literature-standard δ; no I>0 selection)"
for s in $STOCKS; do
  python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock "$s" --models "$MODELS"
  python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock "$s" --models "$MODELS" --method none
done
# Everything below is labelled "diagnostic only" in its own banner, and none of it has ever
# completed: every run so far (5x betaOWQR + beta_AAPL, 2026-07-26/27) saved the four headline PNGs
# and then died at the 12 h wall inside beta_master_curve.py. These three scripts take no --stock,
# so each job recomputes ALL stocks -- five concurrent per-stock jobs did the same all-stock work
# five times over. Headline-only is therefore the default; FULL=1 opts back in.
if [ "${FULL:-0}" != "1" ]; then
  echo "[$(date)] HEADLINE_DONE (diagnostic stages skipped; FULL=1 to run them)"
  exit 0
fi
echo ">>> beta master curve (Parkinson, intercept estimator) [per-point: E[log I|I>0] biased — diagnostic only]"
python3.11 5_analysis/beta/beta_master_curve.py --grid "$GRID" --daily "$DAILY"
echo ">>> beta 5-method (5 sigma estimators + intercept, day-clustered bootstrap CI) [per-point: biased — diagnostic only]"
python3.11 5_analysis/beta/beta_5method.py --grid "$GRID" --daily "$DAILY"
echo ">>> beta grid 3x5 (rows: all-points | by-k origin | by-k intercept; cols: 5 sigma) [per-point: biased — diagnostic only]"
python3.11 5_analysis/beta/beta_grid.py --grid "$GRID" --daily "$DAILY"
echo "[$(date)] done -> 5_analysis/beta/results/{beta_binned,beta_master,beta_5method,beta_grid}/"
