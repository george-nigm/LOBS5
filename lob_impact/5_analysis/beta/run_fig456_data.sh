#!/bin/bash
#SBATCH --job-name=lobimp_fig456
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/fig456_%j.out
#
# Regenerate the grid-dependent paper figures on a COMPUTE node (never login — Lustre glob):
#   Fig 4: mid-price trajectory with ALL 6 models (beta + decay)  -> mid_trajectory.py (+ npz)
#   Fig 6 (body): beta(k) 3 views + cloud with sigma=1 (--method none) -> beta_vs_k_3views.py
# The sigma=parkinson Fig 6 (appendix) already exists; not recomputed here.
set -uo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
B="${IMPACT_DIR}/5_analysis/beta"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid}"
STOCK="${STOCK:-EA}"
MODELS="${MODELS:-Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k}"
DAILY="${DAILY:-${IMPACT_DIR}/2_daily_stats/results/daily_20260618-215656/daily_h_l_all.csv}"
PER_DAY="${PER_DAY:-${IMPACT_DIR}/1_data_prep/results/per_day_params/per_day_params_${STOCK}.csv}"
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
export JAX_PLATFORMS=cpu
mkdir -p "${B}/logs"
cd "$B"
echo "[$(date)] host $(hostname) | grid ${GRID} | stock ${STOCK} | models ${MODELS}"

echo ">>> Fig 4a: mid-price trajectory (beta build-up, 6 models)"
$PY mid_trajectory.py --grid "$GRID" --stock "$STOCK" --shape beta --models "$MODELS" \
    --daily "$DAILY" --per_day_params "$PER_DAY" \
    --out "$B/results/mid_impact/mid_trajectory_${STOCK}_beta.png"
echo ">>> Fig 4b: mid-price trajectory (relaxation/decay, 6 models)"
$PY mid_trajectory.py --grid "$GRID" --stock "$STOCK" --shape relaxation --models "$MODELS" \
    --daily "$DAILY" --per_day_params "$PER_DAY" \
    --out "$B/results/mid_impact/mid_trajectory_${STOCK}_decay.png"
echo ">>> Fig 6 (body): beta(k) 3 views + cloud, sigma=1"
$PY beta_vs_k_3views.py --grid "$GRID" --daily "$DAILY" --stock "$STOCK" --models "$MODELS" \
    --method none --out "$B/results/beta_vs_k_3views/beta_vs_k_3views_${STOCK}_sigma1.png"
echo "[$(date)] DONE -> mid_trajectory_${STOCK}_{beta,decay}.npz + beta_vs_k_3views_${STOCK}_sigma1.{png,npz}"
