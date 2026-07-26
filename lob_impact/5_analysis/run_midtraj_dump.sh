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
TAG=beta; [ "$SHAPE" = relaxation ] && TAG=decay
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
$PY fig6_spaghetti.py --npz "$B/results/mid_impact/mid_trajectory_${STOCK}_${TAG}.npz" --stock "$STOCK"
echo "MTD_DONE $STOCK $TAG"
