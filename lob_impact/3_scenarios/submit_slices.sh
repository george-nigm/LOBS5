#!/bin/bash
# Fan out a grid across GPUs by SAMPLE SLICE: one job per (stock, slice), each computing one batch
# (batch_size=64 samples). Total n_samples per experiment = n_slices * 64. All slices of an experiment
# MERGE into the consolidated path results/grid/<stock>-<model>-<beta|relaxation>/<dir>/ (slices are
# disjoint -> distinct real_ids), so the analysis still points at ONE root.
#
#   bash 3_scenarios/submit_slices.sh mamba3 32                 # 32 slices = 2048 samples, default stocks
#   bash 3_scenarios/submit_slices.sh mamba3 4 "EA"            # 4 slices = 256 samples, EA only
#   GRES='--gres=gpu:1' TIME=06:00:00 bash 3_scenarios/submit_slices.sh mamba3 8
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
echo ">>> submitted ${n} jobs (${N_SLICES} slices x stocks) for ${MODEL}; n_samples/experiment = $((N_SLICES * 64))."
echo ">>> all merge into results/grid/<stock>-${MODEL}-<tag>/<dir>/  — analysis: diagnostics.py --run_dir results/grid"
