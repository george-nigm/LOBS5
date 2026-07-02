#!/bin/bash
#SBATCH --job-name=lobimp_a5_pub
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/pub_%j.out
#
# Action 5 — regenerate the PAPER result figures in publication style (../pubstyle.py),
# on a COMPUTE node (the grid globs thousands of CSVs on Lustre -> never the login node).
#   Fig 5: impact master curves  (beta build-up + relaxation)   -> master_curve.py
#   Fig 6: beta(k) 3 views + hero impact cloud                  -> beta_vs_k_3views.py
# Each script also writes a .npz cache next to its PNG for instant future re-styling.
#
#   sbatch lob_impact/5_analysis/beta/run_pub_figs.sh
#   GRID=/path/to/grid DAILY=/path/daily_h_l_all.csv sbatch .../run_pub_figs.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
HERE="${IMPACT_DIR}/5_analysis/beta"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid}"
STOCK="${STOCK:-EA}"
MODELS="${MODELS:-Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k}"
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
echo "[$(date)] host $(hostname) | grid ${GRID} | stock ${STOCK} | daily ${DAILY}"

cd "$IMPACT_DIR"
BETA_NPZ="${HERE}/results/master_curve/master_curve_${STOCK}_beta.npz"
if [ "${FORCE:-0}" = "1" ] || [ ! -f "$BETA_NPZ" ]; then
  echo ">>> Fig 5a: master curve (beta build-up)"
  python3.11 5_analysis/beta/master_curve.py --grid "$GRID" --stock "$STOCK" --shape beta       --models "$MODELS"
else
  echo ">>> Fig 5a: cached ($BETA_NPZ exists) — skipping grid read (FORCE=1 to redo)"
fi
echo ">>> Fig 5b: master curve (relaxation)"
python3.11 5_analysis/beta/master_curve.py --grid "$GRID" --stock "$STOCK" --shape relaxation --models "$MODELS"
echo ">>> Fig 6: beta(k) 3 views + hero impact cloud"
python3.11 5_analysis/beta/beta_vs_k_3views.py --grid "$GRID" --daily "$DAILY" --stock "$STOCK" --models "$MODELS"
echo "[$(date)] done -> 5_analysis/beta/results/{master_curve,beta_vs_k_3views}/  (+ .npz caches)"
