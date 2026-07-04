#!/bin/bash
# mamba3_4k EA slice fleet: 8 slices x {beta,relaxation} x {buy,sell} minus beta/buy/0 (probe 5487210).
# Probe timing: slice0 (2 batches/day) = 3h00m -> 8h limit is safe.
set -uo pipefail
HERE=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
RUN="$HERE/run_experiments.sh"
ACC=brics.u6gb; PART=workq
GAP="${GAP:-20}"
JIDS="$HERE/logs/v2_EA_m34k_fleet_jids_$(date +%Y%m%d-%H%M%S).txt"
: > "$JIDS"
echo "submitting mamba3_4k EA fleet -> $JIDS"
for sh in beta relaxation; do for d in buy sell; do for k in 0 1 2 3 4 5 6 7; do
  [ "$sh/$d/$k" = "beta/buy/0" ] && continue
  j=$(PER_DAY=1 N_PER_DAY=104 BSZ=8 ONLY_SHAPE=$sh ONLY_DIR=$d SAMPLE_SLICE=$k/8 STOCKS=EA \
    sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 \
    --cpus-per-task=8 --mem=96G --time=08:00:00 \
    --job-name="m34k_${sh:0:1}${d:0:1}${k}" \
    --output="$HERE/logs/m34k_EA_${sh}_${d}_s${k}_%j.out" \
    --export=ALL "$RUN" full mamba3_4k)
  echo "m34k EA/$sh/$d/slice$k = $j" | tee -a "$JIDS"
  sleep "$GAP"
done; done; done
echo "ALL SUBMITTED -> $JIDS"
