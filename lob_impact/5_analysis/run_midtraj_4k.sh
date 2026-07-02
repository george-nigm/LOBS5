#!/bin/bash
#SBATCH --job-name=midtraj4k
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/midtraj4k_%x_%j.out
set -uo pipefail
SHAPE="${1:?shape}"; TAG=beta; [ "$SHAPE" = relaxation ] && TAG=decay
export JAX_PLATFORMS=cpu
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
B=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid
DAILY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/results/daily_20260618-215656/daily_h_l_all.csv
PDP=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/1_data_prep/results/per_day_params/per_day_params_EA.csv
cd "$B"
echo ">>> EA $SHAPE: Mamba3(500) vs Mamba3_4k(4000) vs S5_4k(4000)"
$PY mid_trajectory.py --grid "$GRID" --stock EA --shape "$SHAPE" \
   --models "${MODELS:-Historic,Mamba3,Mamba3_4k,S5_4k}" --daily "$DAILY" --per_day_params "$PDP" \
   --out "$B/results/mid_impact/mid_traj_4kVS500_EA_${TAG}.png"
echo "DONE $SHAPE"
