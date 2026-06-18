#!/bin/bash
# Staggered overnight submission of the FULL per-day grid (corrected order_volume=p50/day, mb/day).
# One sbatch per (stock × shape × dir) for max parallelism + per-day partial-result banking.
# Mamba3 (GPU) gated on a Mamba3 smoke; Historic (CPU) gated on a Historic smoke — so a broken
# per-day path never releases the 12-job fleet. Safety: all via sbatch, node-local squashfuse mounts,
# staggered sleep 35 between submissions, no metadata storms.
set -uo pipefail
HERE=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
RUN="$HERE/run_experiments.sh"
ACC=brics.u6gb; PART=workq
NPD="${NPD:-52}"                 # samples/day -> 52*20days ≈ 1040 ≈ 1024 target
STK=(EA NVDA AMD); SHP=(beta relaxation); DIR=(buy sell)
mkdir -p "$HERE/logs"
JIDS="$HERE/logs/overnight_jids_$(date +%Y%m%d-%H%M%S).txt"
echo "submitting overnight grid -> $JIDS"; : > "$JIDS"
GAP="${GAP:-35}"

# ---- 1) smokes (gates) ----
M3SMOKE=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
  --cpus-per-task=2 --mem=12G --time=00:15:00 --job-name=m3smoke \
  --output="$HERE/logs/m3smoke_%j.out" --export=ALL,PER_DAY=1,N_PER_DAY=4 \
  "$RUN" smoke mamba3)
echo "GATE mamba3-smoke  = $M3SMOKE" | tee -a "$JIDS"; sleep "$GAP"

HISMOKE=$(sbatch --parsable --account=$ACC --partition=$PART \
  --cpus-per-task=2 --mem=8G --time=00:15:00 --job-name=hismoke \
  --output="$HERE/logs/hismoke_%j.out" --export=ALL,PER_DAY=1,N_PER_DAY=4 \
  "$RUN" smoke historic)
echo "GATE historic-smoke = $HISMOKE" | tee -a "$JIDS"; sleep "$GAP"

# ---- 2) Mamba3 GPU fleet (gated on mamba3 smoke) ----
for s in "${STK[@]}"; do for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
    --cpus-per-task=16 --mem=80G --time=23:00:00 \
    --job-name="m3_${s}_${sh}_${d}" --output="$HERE/logs/m3_${s}_${sh}_${d}_%j.out" \
    --dependency=afterok:$M3SMOKE \
    --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$s,ONLY_SHAPE=$sh,ONLY_DIR=$d \
    "$RUN" full mamba3)
  echo "mamba3   $s/$sh/$d = $j" | tee -a "$JIDS"; sleep "$GAP"
done; done; done

# ---- 3) Historic CPU fleet (gated on historic smoke) ----
for s in "${STK[@]}"; do for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART \
    --cpus-per-task=32 --mem=64G --time=12:00:00 \
    --job-name="hi_${s}_${sh}_${d}" --output="$HERE/logs/hi_${s}_${sh}_${d}_%j.out" \
    --dependency=afterok:$HISMOKE \
    --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$s,ONLY_SHAPE=$sh,ONLY_DIR=$d \
    "$RUN" full historic)
  echo "historic $s/$sh/$d = $j" | tee -a "$JIDS"; sleep "$GAP"
done; done; done

echo "=== ALL SUBMITTED ($(grep -c '=' "$JIDS") lines) -> $JIDS ==="
