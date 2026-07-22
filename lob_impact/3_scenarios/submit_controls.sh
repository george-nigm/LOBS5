#!/bin/bash
# Control fleets for ONE stock -> lob_impact_controls_v2: noins (drift/Stage-0
# flow-balance input; 64 free rollouts per model) + invisible (triangle;
# 8/day x 20 days per side, neural only). Recipe recovered verbatim from the
# GOOG/NVDA control fleets (sacct SubmitLine, 2026-07-09).
#   bash submit_controls.sh <STOCK> <MB_OVERRIDE ~ 100 x m_b>
set -uo pipefail
HERE=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
RUN="$HERE/run_experiments.sh"
ACC=brics.u6gb; PART=workq
ST="${1:?stock}"; MBOV="${2:?mb_override}"
CTRL=/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2
GAP="${GAP:-2}"
JIDS="$HERE/logs/controls_${ST}_jids_$(date +%Y%m%d-%H%M%S).txt"; : > "$JIDS"
CPU_MODELS=(historic heuristic propagator cst nmzi hawkes)
GPU_MODELS=(mamba3 gdn s5_120m mamba3_4k s5_4k)
for m in "${CPU_MODELS[@]}"; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART --cpus-per-task=32 --mem=256G --time=12:00:00 \
    --job-name="${ST}_noi_${m:0:4}" --output="$HERE/logs/noins_${ST}_${m}_%j.out" \
    --export=ALL,SAVE_BASE=$CTRL/noins,STOCKS=$ST,ONLY_SHAPE=beta,DIRS=buy,N_SAMPLES=64,N_INS_OVERRIDE=1,MB_OVERRIDE=$MBOV \
    "$RUN" full "$m"); echo "noins $m = $j" | tee -a "$JIDS"; sleep $GAP
done
for m in "${GPU_MODELS[@]}"; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=12:00:00 \
    --job-name="${ST}_noi_${m:0:4}" --output="$HERE/logs/noins_${ST}_${m}_%j.out" \
    --export=ALL,SAVE_BASE=$CTRL/noins,STOCKS=$ST,ONLY_SHAPE=beta,DIRS=buy,N_SAMPLES=64,N_INS_OVERRIDE=1,MB_OVERRIDE=$MBOV \
    "$RUN" full "$m"); echo "noins $m = $j" | tee -a "$JIDS"; sleep $GAP
done
for m in "${GPU_MODELS[@]}"; do for d in buy sell; do
  j=$(sbatch --parsable --account=$ACC --partition=$PART --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=12:00:00 \
    --job-name="${ST}_inv_${m:0:4}_${d:0:1}" --output="$HERE/logs/inv_${ST}_${m}_${d}_%j.out" \
    --export=ALL,SAVE_BASE=$CTRL/invisible,STOCKS=$ST,ONLY_SHAPE=beta,ONLY_DIR=$d,PER_DAY=1,N_PER_DAY=8,BSZ=8,METAORDER_VISIBLE=0 \
    "$RUN" full "$m"); echo "inv $m $d = $j" | tee -a "$JIDS"; sleep $GAP
done; done
echo "controls fleet for $ST submitted -> $JIDS"
