#!/bin/bash
# Fan out a grid across GPUs by SAMPLE SLICE: one job per (stock, slice). Each slice runs a
# contiguous SHARD of the batch partition (array_split in the scenario); n_samples comes from the
# config / N_SAMPLES env — slices no longer imply n_samples = n_slices*64. All slices of an
# experiment MERGE into the consolidated path <SAVE_BASE>/<stock>-<model>-<beta|relaxation>/<dir>/
# (global gen_ids -> disjoint), so the analysis still points at ONE root.
#
#   bash 3_scenarios/submit_slices.sh mamba3_4k 8               # 8 shards/job fleet, default stocks
#   bash 3_scenarios/submit_slices.sh mamba3_4k 8 "EA"          # EA only
#   GRES='--gres=gpu:1' TIME=08:00:00 PER_DAY=1 BSZ=8 ONLY_SHAPE=beta ONLY_DIR=buy \
#     bash 3_scenarios/submit_slices.sh s5_4k 8 "EA"
# Env passed through to run_experiments.sh (sbatch inherits): PER_DAY, N_PER_DAY, BSZ, N_SAMPLES,
# ONLY_SHAPE, ONLY_DIR, SAVE_BASE, METAORDER_VISIBLE, ...
# CPU models: pass GRES=' '.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?usage: submit_slices.sh <model> <n_slices> [stocks]}"
N_SLICES="${2:?n_slices}"
STOCKS="${3:-EA NVDA AMD}"
GRES="${GRES:---gres=gpu:1}"        # CPU models (historic): pass GRES=' '
TIME="${TIME:-04:00:00}"

n=0
for stock in $STOCKS; do
  for k in $(seq 0 $((N_SLICES - 1))); do
    jid=$(SAMPLE_SLICE="${k}/${N_SLICES}" STOCKS="$stock" \
          sbatch $GRES --time="$TIME" --parsable "$HERE/run_experiments.sh" full "$MODEL")
    n=$((n + 1))
    echo "  ${MODEL} ${stock} slice ${k}/${N_SLICES} -> ${jid}"
  done
done
echo ">>> submitted ${n} jobs (${N_SLICES} shards x stocks) for ${MODEL}; n_samples per experiment comes from the config/N_SAMPLES."
echo ">>> all merge into <SAVE_BASE>/<stock>-${MODEL}-<tag>/<dir>/"
