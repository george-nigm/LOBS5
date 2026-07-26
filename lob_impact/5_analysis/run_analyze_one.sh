#!/bin/bash
#SBATCH --job-name=lobimp_analyze
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/analyze_%j.out
#
# All analyses for ONE <stock>-<model> once its grid is complete. CPU, reads the Lustre grid
# + the Action-2 daily CSV. Runs the full beta suite (static 3x5 + 5-method + master-curve +
# the interactive exact-k 3x5 HTML) and the book-player HTMLs for BOTH shapes.
# (Decay/relaxation metric for the new grid is still a TODO — only book-players cover relaxation.)
#
#   sbatch --dependency=afterok:<jobids> lob_impact/5_analysis/run_analyze_one.sh EA Mamba3
#   STOCK=EA MODEL=Mamba3 sbatch lob_impact/5_analysis/run_analyze_one.sh        # (or positional)
set -uo pipefail
STOCK="${1:-${STOCK:?need stock}}"; MODEL="${2:-${MODEL:?need model}}"
IMPACT=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
PY="${PY:-/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python}"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid}"
B="$IMPACT/5_analysis/beta"
mkdir -p "$IMPACT/5_analysis/logs" "$B/results/beta_grid" "$B/results/beta_5method" \
         "$B/results/beta_master" "$B/results/beta_explorer"

# newest Action-2 daily CSV carrying OHLC (absolute path)
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
[ -n "${DAILY:-}" ] || { echo "FATAL: no OHLC daily_h_l_all.csv (run Action 2)"; exit 3; }
EXP="${STOCK}-${MODEL}-beta"
echo "[$(date)] analyze $STOCK-$MODEL | grid=$GRID | daily=$DAILY | exp(beta)=$EXP"

cd "$B"
echo ">>> [1/5] beta_grid (static 3x5)"
$PY beta_grid.py        --grid "$GRID" --exp "$EXP" --daily "$DAILY" --out_dir "$B/results/beta_grid"        || echo "  beta_grid FAILED"
echo ">>> [2/5] beta_5method"
$PY beta_5method.py     --grid "$GRID" --exp "$EXP" --daily "$DAILY" --out_dir "$B/results/beta_5method"     || echo "  beta_5method FAILED"
echo ">>> [3/5] beta_master_curve"
$PY beta_master_curve.py --grid "$GRID" --exp "$EXP" --daily "$DAILY" --out_dir "$B/results/beta_master"     || echo "  beta_master FAILED"
echo ">>> [4/5] beta_explorer (interactive exact-k 3x5 HTML)"
$PY beta_explorer_html.py --grid "$GRID" --exp "$EXP" --daily "$DAILY" \
    --out "$B/results/beta_explorer/beta_explorer_grid_${EXP}.html"                                          || echo "  beta_explorer FAILED"
echo ">>> [5/5] book-player HTMLs (beta + relaxation, buy + sell)"
STOCKS="$STOCK" MODELS="$MODEL" SHAPES="beta relaxation" SIDES="buy sell" N_SAMPLES="${N_SAMPLES:-8}" \
    GRID="$GRID" bash "$IMPACT/4_diagnostics/run_book_player.sh"                                             || echo "  book_player FAILED"
echo "[$(date)] DONE $STOCK-$MODEL -> 5_analysis/beta/results/* + 4_diagnostics/results/book_player/"
