#!/bin/bash
#SBATCH --job-name=midtraj
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/midtraj_%x_%j.out
set -uo pipefail
STOCK="${1:?need stock}"
export JAX_PLATFORMS=cpu
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
B=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid
cd "$B"
echo ">>> [$STOCK] BETA shape"
DAILY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/results/daily_20260618-215656/daily_h_l_all.csv
$PY mid_trajectory.py --grid "$GRID" --stock "$STOCK" --shape beta --daily "$DAILY" \
   --out "$B/results/mid_impact/mid_trajectory_${STOCK}_beta.png"
echo ">>> [$STOCK] DECAY (relaxation) shape"
$PY mid_trajectory.py --grid "$GRID" --stock "$STOCK" --shape relaxation --daily "$DAILY" \
   --out "$B/results/mid_impact/mid_trajectory_${STOCK}_decay.png"
echo "DONE $STOCK"
