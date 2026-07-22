#!/bin/bash
#SBATCH --job-name=midtraj_EA
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/midtraj_EA_%j.out
set -uo pipefail
export JAX_PLATFORMS=cpu
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
B=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid
cd "$B"
$PY mid_trajectory.py --grid "$GRID" --stock EA --shape beta \
   --out "$B/results/mid_impact/mid_trajectory_EA.png"
