#!/bin/bash
#SBATCH --job-name=lobimp_stockfull
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/stockfull_%j.out
#
# Full β + decay + impact analysis suite for ONE stock, all models. CPU, reads Lustre grid + daily CSV.
#   sbatch --dependency=afterok:<gen jobids> run_stock_full_analysis.sh NVDA
set -uo pipefail
STOCK="${1:-${STOCK:?need stock}}"
MODELS="${MODELS:-Historic,Heuristic,OW,QR,CST,Mamba3}"
IMPACT=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
PY="${PY:-/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python}"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid}"
B="$IMPACT/5_analysis/beta"; DEC="$IMPACT/5_analysis/decay"
export JAX_PLATFORMS=cpu
mkdir -p "$IMPACT/5_analysis/logs" "$B/results/beta_explorer" "$B/results/beta_cumul_vs_exact" \
         "$B/results/beta_sigma1" "$B/results/beta_grid" "$B/results/mid_impact" "$DEC/results"
# Pick a daily OHLC table that COVERS the requested stock: the plain daily_h_l_all.csv only has
# EA/NVDA/GOOG/AMD; MSFT and AAPL live in *_plus.csv. The wrong one is a SILENT failure (every
# model scores 0 points and the job still exits 0).
if [ -z "${DAILY:-}" ]; then
  for f in $(ls -t "$IMPACT"/2_daily_stats/results/*/daily_h_l_all_plus.csv \
                   "$IMPACT"/2_daily_stats/results/*/daily_h_l_all.csv 2>/dev/null); do
    head -1 "$f" | grep -q open_price || continue
    grep -q "^${STOCK}," "$f" && { DAILY="$f"; break; }
    echo "[daily] skip $(basename "$f") — no ${STOCK}" >&2
  done
fi
echo "[$(date)] FULL analysis $STOCK | grid=$GRID | daily=$DAILY | models=$MODELS"
cd "$B"
echo ">>> [1] multi-model explorer (≤k/==k toggle)"
$PY beta_explorer_multimodel.py --grid "$GRID" --daily "$DAILY" --stock "$STOCK" --models "$MODELS" \
    --out "$B/results/beta_explorer/beta_explorer_multimodel_${STOCK}.html" || echo "  explorer FAILED"
echo ">>> [2] cumul-vs-exact static panel"
$PY beta_cumul_vs_exact.py --grid "$GRID" --daily "$DAILY" --stock "$STOCK" --models "$MODELS" \
    --out "$B/results/beta_cumul_vs_exact/beta_cumul_vs_exact_${STOCK}.png" || echo "  cumul_vs_exact FAILED"
echo ">>> [3] cross-model cumulative panel (paper style)"
$PY beta_sigma1_by_model.py --grid "$GRID" --daily "$DAILY" --stocks "$STOCK" \
    --out_dir "$B/results/beta_sigma1" || echo "  sigma1 FAILED"
echo ">>> [4] mid-price combined impact curve"
$PY mid_impact_curve.py --grid "$GRID" --stock "$STOCK" --models "$MODELS" \
    --out "$B/results/mid_impact/mid_impact_${STOCK}.png" || echo "  mid_impact FAILED"
echo ">>> [5] beta-grid static 3x5 per model"
for m in ${MODELS//,/ }; do
  $PY beta_grid.py --grid "$GRID" --exp "${STOCK}-${m}-beta" --daily "$DAILY" \
      --out_dir "$B/results/beta_grid" || echo "  beta_grid $m FAILED"
done
echo ">>> [6] decay master curve (Shape II)"
$PY "$DEC/decay_master_curve.py" --grid "$GRID" --stock "$STOCK" --models "$MODELS" \
    --out "$DEC/results/decay_${STOCK}.png" || echo "  decay FAILED"
echo "[$(date)] DONE $STOCK"
