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

# NO smoke gates: both per-day paths are already validated (Mamba3 generates a sane evolving L10
# book; Historic EA/AMD/NVDA-beta completed). afterok gates risked DependencyNeverSatisfied (the
# Mamba3 smoke timed out at 20min from model-load+compile) wiping the whole fleet. Submit direct.
# Output root = SAVE_BASE (Lustre, no quota) via run_experiments.sh.

# ---- 1) Mamba3 GPU fleet ----
for s in "${STK[@]}"; do for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
    --cpus-per-task=16 --mem=96G --time=08:00:00 \
    --job-name="m3_${s}_${sh}_${d}" --output="$HERE/logs/m3_${s}_${sh}_${d}_%j.out" \
    --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$s,ONLY_SHAPE=$sh,ONLY_DIR=$d \
    "$RUN" full mamba3)
  echo "mamba3   $s/$sh/$d = $j" | tee -a "$JIDS"; sleep "$GAP"
done; done; done

# ---- 2) Historic CPU fleet (per-stock mem: NVDA/AMD high-volume -> 192G; NVDA OOM'd at 64G) ----
declare -A HI_MEM=([EA]=96G [NVDA]=192G [AMD]=192G)
for s in "${STK[@]}"; do for sh in "${SHP[@]}"; do for d in "${DIR[@]}"; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART \
    --cpus-per-task=32 --mem="${HI_MEM[$s]:-96G}" --time=12:00:00 \
    --job-name="hi_${s}_${sh}_${d}" --output="$HERE/logs/hi_${s}_${sh}_${d}_%j.out" \
    --export=ALL,PER_DAY=1,N_PER_DAY=$NPD,STOCKS=$s,ONLY_SHAPE=$sh,ONLY_DIR=$d \
    "$RUN" full historic)
  echo "historic $s/$sh/$d = $j  (mem=${HI_MEM[$s]:-96G})" | tee -a "$JIDS"; sleep "$GAP"
done; done; done

echo "=== ALL SUBMITTED ($(grep -c '=' "$JIDS") lines) -> $JIDS ==="
