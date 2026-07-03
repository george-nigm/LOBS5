#!/bin/bash
# Phase-2 (post-audit) EA fleet -> lob_impact_grid_v2. Staggered sbatch, per-day mode.
# CPU baselines (historic/heuristic/cst/hawkes) + mamba3 GPU + timing PROBES for the 4k models
# (one slice-0 job each; fan the remaining slices out with submit_slices.sh after reading the
# probe walltime). Usage:  bash submit_ea_v2.sh [stock]     (default EA)
set -uo pipefail
HERE=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
RUN="$HERE/run_experiments.sh"
ACC=brics.u6gb; PART=workq
S="${1:-EA}"
NPD="${NPD:-52}"                 # samples/day (2k models): 52*20d ≈ 1040/side
NPD4K="${NPD4K:-104}"            # samples/day for 4k models (bsz 8 -> 13 batches/day)
N_SLICES="${N_SLICES:-8}"
SHP=(beta relaxation); DIR=(buy sell)
GAP="${GAP:-35}"
declare -A HI_MEM=([EA]=96G [NVDA]=192G [AMD]=192G)
mkdir -p "$HERE/logs"
JIDS="$HERE/logs/v2_${S}_jids_$(date +%Y%m%d-%H%M%S).txt"
echo "submitting v2 fleet for $S -> $JIDS"; : > "$JIDS"

# ---- CPU baselines ----
for m in historic heuristic cst hawkes; do
  case "$m" in
    historic|heuristic) CPUS=32; MEM="${HI_MEM[$S]:-96G}"; T=12:00:00 ;;
    cst)                CPUS=16; MEM=64G;  T=12:00:00 ;;
    hawkes)             CPUS=16; MEM=64G;  T=24:00:00 ;;
  esac
  for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
    j=$(sbatch --parsable --account=$ACC --partition=$PART \
      --cpus-per-task=$CPUS --mem=$MEM --time=$T \
      --job-name="${m:0:2}_${S}_${sh}_${d}" --output="$HERE/logs/${m}_${S}_${sh}_${d}_%j.out" \
      --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$S,ONLY_SHAPE=$sh,ONLY_DIR=$d \
      "$RUN" full "$m")
    echo "$m $S/$sh/$d = $j" | tee -a "$JIDS"; sleep "$GAP"
  done; done
done

# ---- mamba3 (500-ctx) GPU fleet ----
for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
    --cpus-per-task=16 --mem=96G --time=08:00:00 \
    --job-name="m3_${S}_${sh}_${d}" --output="$HERE/logs/m3_${S}_${sh}_${d}_%j.out" \
    --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$S,ONLY_SHAPE=$sh,ONLY_DIR=$d \
    "$RUN" full mamba3)
  echo "mamba3 $S/$sh/$d = $j" | tee -a "$JIDS"; sleep "$GAP"
done; done

# ---- 4k model PROBES: slice 0/N of beta/buy only; read walltime, then fan out ----
for m in mamba3_4k s5_4k; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
    --cpus-per-task=16 --mem=96G --time=08:00:00 \
    --job-name="probe_${m}_${S}" --output="$HERE/logs/probe_${m}_${S}_%j.out" \
    --export=ALL,PER_DAY=1,N_PER_DAY=$NPD4K,BSZ=8,STOCKS=$S,ONLY_SHAPE=beta,ONLY_DIR=buy,SAMPLE_SLICE=0/$N_SLICES \
    "$RUN" full "$m")
  echo "PROBE $m $S/beta/buy slice0/$N_SLICES = $j" | tee -a "$JIDS"; sleep "$GAP"
done

echo "=== ALL SUBMITTED ($(grep -c '=' "$JIDS") jobs) -> $JIDS ==="
echo "after probes finish: fan out remaining slices, e.g."
echo "  for sh in beta relaxation; do for d in buy sell; do for k in \$(seq 0 $((N_SLICES-1))); do"
echo "    [ \"\$sh/\$d/\$k\" = beta/buy/0 ] || PER_DAY=1 N_PER_DAY=$NPD4K BSZ=8 ONLY_SHAPE=\$sh ONLY_DIR=\$d SAMPLE_SLICE=\$k/$N_SLICES STOCKS=$S sbatch --gres=gpu:1 --time=08:00:00 --mem=96G $RUN full mamba3_4k; done; done; done"
