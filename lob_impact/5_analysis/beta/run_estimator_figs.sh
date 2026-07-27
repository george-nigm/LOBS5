#!/bin/bash
#SBATCH --job-name=estfig
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/estfig_%j.out
#
# Fig. 8 (the 3x3 estimator panel) and the bias exhibit. Both read the grid, both cache every
# plotted array to an npz next to the png -- and NO launcher ran either of them, so their caches
# were months stale and verify_figures_coverage.py reported 10 gaps that nothing in the queue was
# going to close. bias_exhibit.py's own --models default carries only 8 of the 14 canonical models,
# which is why EA and GOOG showed 6-7; the full list is passed explicitly here.
#   STOCK=NVDA sbatch run_estimator_figs.sh
set -uo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
B="${IMPACT_DIR}/5_analysis/beta"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"
STOCK="${STOCK:-EA}"
MODELS="${MODELS:-Historic,Heuristic,Propagator,OW,CST,NMZI,Hawkes,QR,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN}"
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
export JAX_PLATFORMS=cpu
mkdir -p "$B/logs" "$B/results/beta_3x3" "$B/results/bias_exhibit"
cd "$B"

# Same coverage-aware pick as run_beta.sh: the plain daily_h_l_all.csv covers only EA/NVDA/GOOG/AMD,
# MSFT and AAPL live in the *_plus.csv, and picking the wrong one is a SILENT failure -- every model
# scores "0 signed points" and the job still exits 0.
if [ -z "${DAILY:-}" ]; then
  for f in $(ls -t "${IMPACT_DIR}"/2_daily_stats/results/*/daily_h_l_all_plus.csv \
                   "${IMPACT_DIR}"/2_daily_stats/results/*/daily_h_l_all.csv 2>/dev/null); do
    head -1 "$f" | grep -q open_price || continue
    grep -q "^${STOCK}," "$f" && { DAILY="$f"; break; }
    echo "[daily] skipping $f — no $STOCK" >&2
  done
fi
[ -n "${DAILY:-}" ] || { echo "FATAL: no OHLC daily table covering $STOCK" >&2; exit 3; }
echo "[$(date)] host $(hostname) | grid $GRID | stock $STOCK | daily $DAILY"
echo "models: $MODELS"

rc=0
echo ">>> Fig 8: 3x3 estimator panel"
$PY beta_3x3.py --grid "$GRID" --daily "$DAILY" --stock "$STOCK" --models "$MODELS" || rc=$?
echo ">>> bias exhibit"
$PY bias_exhibit.py --grid "$GRID" --daily "$DAILY" --stock "$STOCK" --models "$MODELS" || rc=$?
echo "[$(date)] ESTFIG_DONE $STOCK rc=$rc"
exit $rc
