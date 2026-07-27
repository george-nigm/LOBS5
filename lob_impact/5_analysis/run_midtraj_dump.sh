#!/bin/bash
#SBATCH --job-name=mtd_GOOG
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/mtd_%x_%j.out
# mid_trajectory rebuild WITH per-sample k-clock dump (spaghetti exhibit) — grid_v2
set -uo pipefail
STOCK="${1:-GOOG}"; SHAPE="${2:-beta}"
# The grid folders are <stock>-<model>-{beta,relaxation}, but the OUTPUT of the relaxation shape is
# named "decay" everywhere else, so "decay" is what one naturally types. The old guard was
#   TAG=beta; [ "$SHAPE" = relaxation ] && TAG=decay
# which left TAG=beta for SHAPE=decay: mid_trajectory then found no <stock>-<model>-decay folders,
# printed "no data" for every model, and wrote the resulting EMPTY npz over the good BETA cache --
# exit 0, no error. That destroyed the beta caches of NVDA/GOOG/AMD/MSFT on 2026-07-27. Accept
# "decay" as an alias for "relaxation" and refuse anything else.
case "$SHAPE" in
  beta)              TAG=beta ;;
  relaxation|decay)  SHAPE=relaxation; TAG=decay ;;
  *) echo "FATAL: shape '$SHAPE' is not one of beta|relaxation|decay" >&2; exit 2 ;;
esac
export JAX_PLATFORMS=cpu
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
B=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2
DAILY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/results/daily_20260708-131449/daily_h_l_all_plus.csv
PDP=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/1_data_prep/results/per_day_params/per_day_params_${STOCK}.csv
MODELS="Historic,Heuristic,Propagator,OW,Hawkes,QR,CST,NMZI,Mamba3,GDN,S5_120M,Mamba3_4k,S5_4k"
cd "$B"
$PY mid_trajectory.py --grid "$GRID" --stock "$STOCK" --shape "$SHAPE" --models "$MODELS" \
   --daily "$DAILY" --per_day_params "$PDP" --dump_samples \
   --out "$B/results/mid_impact/mid_trajectory_${STOCK}_${TAG}.png"
NPZ="$B/results/mid_impact/mid_trajectory_${STOCK}_${TAG}.npz"
# Second belt: if the run produced a cache with no per-model curves, say so loudly instead of
# letting the next stage fail on an assertion nobody reads.
$PY - "$NPZ" <<'EOP' || { echo "FATAL: $NPZ carries no per-model data — grid folders for shape '$SHAPE' missing?" >&2; exit 3; }
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
sys.exit(0 if any(k.endswith('_k_mean') for k in z.files) else 1)
EOP
$PY fig6_spaghetti.py --npz "$NPZ" --stock "$STOCK"
echo "MTD_DONE $STOCK $TAG"
