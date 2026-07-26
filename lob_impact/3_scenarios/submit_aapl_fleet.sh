#!/bin/bash
# AAPL panel fleet (mb=344, between MSFT 261 and AMD 401) -> lob_impact_grid_v2.
# Mirror of submit_msft_fleet.sh; the only stock left without an impact panel — the
# grid currently holds AAPL-OW-* only, so the AAPL paper clone sits at 3/26 figures.
#   - replays + cst/nmzi/hawkes: cst_params_AAPL.pkl / hawkes_params_AAPL.pkl /
#     qr_params_AAPL.pkl all exist -> every phase can submit at once
#   - per-day counts scaled off MSFT by the mb ratio (261/344): 52->40, 104->80
#   - 500-ctx neural: N_SLICES=5;  4k neural: BSZ=8, N_SLICES=10
# Usage: bash submit_aapl_fleet.sh [phase]   phase: all|replays|params|neural500|neural4k
set -uo pipefail
HERE=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
RUN="$HERE/run_experiments.sh"
ACC=brics.u6gb; PART=workq
S=AAPL
NPD="${NPD:-40}"; NPD4K="${NPD4K:-80}"
GAP="${GAP:-8}"
PHASE="${1:-all}"
SHP=(beta relaxation); DIR=(buy sell)
mkdir -p "$HERE/logs"
JIDS="$HERE/logs/aapl_fleet_jids_$(date +%Y%m%d-%H%M%S).txt"
echo "submitting AAPL fleet (phase=$PHASE) -> $JIDS"; : > "$JIDS"

if [ "$PHASE" = all ] || [ "$PHASE" = replays ]; then
  for m in historic heuristic propagator; do
    for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
      j=$(sbatch --parsable --account=$ACC --partition=$PART \
        --cpus-per-task=32 --mem=256G --time=12:00:00 \
        --job-name="${m:0:2}_${S}_${sh:0:1}${d:0:1}" --output="$HERE/logs/${m}_${S}_${sh}_${d}_%j.out" \
        --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$S,ONLY_SHAPE=$sh,ONLY_DIR=$d \
        "$RUN" full "$m")
      echo "$m $S/$sh/$d = $j" | tee -a "$JIDS"; sleep "$GAP"
    done; done
  done
fi

if [ "$PHASE" = all ] || [ "$PHASE" = params ]; then
  for m in cst nmzi hawkes; do
    for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
      j=$(sbatch --parsable --account=$ACC --partition=$PART \
        --cpus-per-task=32 --mem=256G --time=12:00:00 \
        --job-name="${m:0:3}_${S:0:1}_${sh:0:1}${d:0:1}" --output="$HERE/logs/${m}_${S}_${sh}_${d}_%j.out" \
        --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$S,ONLY_SHAPE=$sh,ONLY_DIR=$d \
        "$RUN" full "$m")
      echo "$m $S/$sh/$d = $j" | tee -a "$JIDS"; sleep "$GAP"
    done; done
  done
fi

if [ "$PHASE" = all ] || [ "$PHASE" = neural500 ]; then
  for m in mamba3 gdn s5_120m; do
    for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do for k in 0 1 2 3 4; do
      j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
        --cpus-per-task=16 --mem=96G --time=08:00:00 \
        --job-name="a${m:0:3}_${sh:0:1}${d:0:1}${k}" \
        --output="$HERE/logs/${m}_${S}_${sh}_${d}_s${k}_%j.out" \
        --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$S,ONLY_SHAPE=$sh,ONLY_DIR=$d,SAMPLE_SLICE=$k/5 \
        "$RUN" full "$m")
      echo "$m $S/$sh/$d/s$k = $j" | tee -a "$JIDS"; sleep "$GAP"
    done; done; done
  done
fi

if [ "$PHASE" = all ] || [ "$PHASE" = neural4k ]; then
  for m in mamba3_4k s5_4k; do
    for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do for k in 0 1 2 3 4 5 6 7 8 9; do
      j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
        --cpus-per-task=8 --mem=96G --time=12:00:00 \
        --job-name="a${m:0:3}4_${sh:0:1}${d:0:1}${k}" \
        --output="$HERE/logs/${m}_${S}_${sh}_${d}_s${k}_%j.out" \
        --export=ALL,PER_DAY=1,N_PER_DAY=$NPD4K,BSZ=8,STOCKS=$S,ONLY_SHAPE=$sh,ONLY_DIR=$d,SAMPLE_SLICE=$k/10 \
        "$RUN" full "$m")
      echo "$m $S/$sh/$d/s$k = $j" | tee -a "$JIDS"; sleep "$GAP"
    done; done; done
  done
fi
echo "ALL SUBMITTED -> $JIDS"
