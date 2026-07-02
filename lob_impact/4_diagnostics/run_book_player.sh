#!/bin/bash
# Build a FOLDER of self-contained interactive book-player HTMLs — ONE per (stock·method·shape·side).
# Each is offline-ready (Plotly inlined). -> results/book_player/player_<stock>-<model>-<shape>_<side>.html
# (Light analysis: reads grid CSVs, writes HTML, NO JAX. Fine on login for modest N_SAMPLES.)
#
# Pick via env (space-separated). Examples:
#   MODELS=Historic ./run_book_player.sh                          # all stocks, both shapes/sides, Historic
#   STOCKS="EA NVDA AMD" MODELS=Historic SIDES="buy sell" N_SAMPLES=8 ./run_book_player.sh
#   STOCKS=EA MODELS="Historic Mamba3" SHAPES=beta MAX_STEPS=15000 ./run_book_player.sh
#
# Knobs: STOCKS, MODELS, SHAPES (beta relaxation — the experiment TYPE, not sample count!), SIDES,
#        N_SAMPLES (per HTML, spread across days), MAX_STEPS (0=full; cap to shrink big beta files), GRID, OUT_DIR.
set -uo pipefail
HERE="/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics"
PY="${PY:-/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python}"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid}"   # consolidated grid lives on LUSTRE (no quota)
STOCKS="${STOCKS:-EA NVDA AMD}"; MODELS="${MODELS:-Historic Mamba3}"
SHAPES="${SHAPES:-beta relaxation}"; SIDES="${SIDES:-buy sell}"
N_SAMPLES="${N_SAMPLES:-8}"; MAX_STEPS="${MAX_STEPS:-0}"
OUT_DIR="${OUT_DIR:-${HERE}/results/book_player}"
mkdir -p "$OUT_DIR"
echo "grid=$GRID  n_samples=$N_SAMPLES  max_steps=$MAX_STEPS  ->  $OUT_DIR"
made=0; skipped=0
for s in $STOCKS; do for m in $MODELS; do for sh in $SHAPES; do for d in $SIDES; do
  exp="${s}-${m}-${sh}"
  if [ ! -d "${GRID}/${exp}/${d}" ]; then echo "  skip ${exp}/${d} (no data)"; skipped=$((skipped+1)); continue; fi
  out="${OUT_DIR}/player_${exp}_${d}.html"
  if "$PY" "${HERE}/book_player.py" --html --grid "$GRID" --exp "$exp" --side "$d" \
        --n_samples "$N_SAMPLES" --max_steps "$MAX_STEPS" --out "$out" >/dev/null 2>&1; then
    echo "  ✓ ${exp}/${d}  -> $(basename "$out") ($(stat -c%s "$out" 2>/dev/null | awk '{printf "%.1fMB",$1/1e6}'))"; made=$((made+1))
  else
    echo "  ✗ ${exp}/${d}  (no usable samples?)"; skipped=$((skipped+1))
  fi
done; done; done; done
echo "done: ${made} HTML(s), ${skipped} skipped -> ${OUT_DIR}"
