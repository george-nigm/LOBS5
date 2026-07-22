#!/bin/bash
#SBATCH --job-name=midtraj1
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/midtraj1_%x_%j.out
set -uo pipefail
STOCK="${1:?stock}"; SHAPE="${2:?shape}"   # SHAPE: beta | relaxation
TAG=beta; [ "$SHAPE" = relaxation ] && TAG=decay
export JAX_PLATFORMS=cpu
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
B=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid
DAILY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/results/daily_20260618-215656/daily_h_l_all.csv
PDP=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/1_data_prep/results/per_day_params/per_day_params_${STOCK}.csv
cd "$B"
echo ">>> [$STOCK] $SHAPE -> $TAG"
$PY mid_trajectory.py --grid "$GRID" --stock "$STOCK" --shape "$SHAPE" \
   --daily "$DAILY" --per_day_params "$PDP" \
   --out "$B/results/mid_impact/mid_trajectory_${STOCK}_${TAG}.png"
echo "DONE $STOCK $TAG"
