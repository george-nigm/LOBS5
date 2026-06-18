#!/bin/bash
# Build self-contained interactive book-player HTMLs for CHOSEN stocks × methods × shapes × sides.
# One HTML per combination -> results/book_player/player_<stock>-<model>-<shape>_<side>.html
# (Light analysis: reads the grid CSVs and writes HTML. No JAX/model. Fine on login for small
#  --n_samples; for big grids/many samples submit it via sbatch on a compute node.)
#
# Pick what to bundle via env (space-separated). Examples:
#   ./run_book_player.sh                                   # everything that exists in the grid
#   STOCKS="EA NVDA" MODELS=Historic SHAPES=beta SIDES=buy N_SAMPLES=12 ./run_book_player.sh
#   GRID=/path/to/results/smoke_2026... N_SAMPLES=8 ./run_book_player.sh   # a smoke dir
#
# Knobs: STOCKS, MODELS (Historic Mamba3 …), SHAPES (beta relaxation), SIDES (buy sell),
#        N_SAMPLES (per HTML, spread across days), MAX_STEPS (0=full; cap to shrink file), GRID, OUT_DIR.
set -uo pipefail
HERE="/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics"
PY="${PY:-/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python}"   # has plotly/numpy
GRID="${GRID:-${HERE}/../3_scenarios/results/grid}"
STOCKS="${STOCKS:-EA NVDA AMD}"
MODELS="${MODELS:-Historic Mamba3}"
SHAPES="${SHAPES:-beta relaxation}"
SIDES="${SIDES:-buy sell}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
OUT_DIR="${OUT_DIR:-${HERE}/results/book_player}"
mkdir -p "$OUT_DIR"
echo "grid=$GRID  n_samples=$N_SAMPLES  out=$OUT_DIR"
made=0; skipped=0
for s in $STOCKS; do for m in $MODELS; do for sh in $SHAPES; do for d in $SIDES; do
  exp="${s}-${m}-${sh}"
  if [ ! -d "${GRID}/${exp}/${d}" ]; then echo "  skip ${exp}/${d} (no data)"; skipped=$((skipped+1)); continue; fi
  out="${OUT_DIR}/player_${exp}_${d}.html"
  if "$PY" "${HERE}/book_player.py" --html --grid "$GRID" --exp "$exp" --side "$d" \
        --n_samples "$N_SAMPLES" --max_steps "$MAX_STEPS" --out "$out" >/dev/null 2>&1; then
    echo "  ✓ ${exp}/${d}  -> $(basename "$out") ($(stat -c%s "$out" 2>/dev/null | awk '{printf "%.1fMB",$1/1e6}'))"; made=$((made+1))
  else
    echo "  ✗ ${exp}/${d}  (book_player failed — no usable samples?)"; skipped=$((skipped+1))
  fi
done; done; done; done
echo "done: ${made} HTML(s) written, ${skipped} skipped -> ${OUT_DIR}"
