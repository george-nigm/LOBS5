#!/bin/bash
# Build ONE self-contained interactive book-player HTML over the CHOSEN stocks × methods × shapes ×
# sides — a top "набор" selector switches dataset, then the sample dropdown / day table / book /
# insertions all update for that dataset. -> results/book_player/player_combined.html
# (Light analysis: reads the grid CSVs, writes HTML, NO JAX. Fine on login for modest N_SAMPLES;
#  for big grids submit via sbatch. beta samples are long -> use MAX_STEPS to keep the file small.)
#
# Pick via env (space-separated). Examples:
#   MODELS=Historic ./run_book_player.sh                          # all stocks, both shapes/sides, Historic
#   STOCKS="EA NVDA AMD" MODELS=Historic SIDES="buy sell" N_SAMPLES=8 ./run_book_player.sh
#   STOCKS=EA MODELS="Historic Mamba3" SHAPES=beta MAX_STEPS=15000 ./run_book_player.sh
#   SEPARATE=1 ... ./run_book_player.sh                           # one HTML PER combo instead of combined
#
# Knobs: STOCKS, MODELS, SHAPES (beta relaxation — the experiment TYPE, not sample count!), SIDES,
#        N_SAMPLES (per dataset, spread across days), MAX_STEPS (0=full; cap to shrink), GRID, OUT, SEPARATE.
set -uo pipefail
HERE="/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics"
PY="${PY:-/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python}"
GRID="${GRID:-${HERE}/../3_scenarios/results/grid}"
STOCKS="${STOCKS:-EA NVDA AMD}"; MODELS="${MODELS:-Historic Mamba3}"
SHAPES="${SHAPES:-beta relaxation}"; SIDES="${SIDES:-buy sell}"
N_SAMPLES="${N_SAMPLES:-8}"; MAX_STEPS="${MAX_STEPS:-6000}"
csv(){ echo "$*" | tr ' ' ','; }

if [ -n "${SEPARATE:-}" ]; then            # one HTML per (stock·method·shape·side)
  OUT_DIR="${OUT:-${HERE}/results/book_player}"; mkdir -p "$OUT_DIR"
  for s in $STOCKS; do for m in $MODELS; do for sh in $SHAPES; do for d in $SIDES; do
    exp="${s}-${m}-${sh}"; [ -d "${GRID}/${exp}/${d}" ] || { echo "  skip ${exp}/${d}"; continue; }
    o="${OUT_DIR}/player_${exp}_${d}.html"
    "$PY" "${HERE}/book_player.py" --html --grid "$GRID" --exp "$exp" --side "$d" \
       --n_samples "$N_SAMPLES" --max_steps "$MAX_STEPS" --out "$o" >/dev/null 2>&1 \
       && echo "  ✓ ${exp}/${d}" || echo "  ✗ ${exp}/${d}"
  done; done; done; done
else                                       # ONE combined HTML with a top dataset selector (default)
  OUT="${OUT:-${HERE}/results/book_player/player_combined.html}"
  "$PY" "${HERE}/book_player.py" --combined --grid "$GRID" \
     --stocks "$(csv $STOCKS)" --models "$(csv $MODELS)" --shapes "$(csv $SHAPES)" --sides "$(csv $SIDES)" \
     --n_samples "$N_SAMPLES" --max_steps "$MAX_STEPS" --out "$OUT"
fi
