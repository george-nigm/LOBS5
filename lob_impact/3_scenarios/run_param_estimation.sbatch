#!/bin/bash
#SBATCH --job-name=est
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios/logs/est_%j.out
# CST + Hawkes parameter estimation for one stock from historic data_cond
# (submit with -d afterok:<historic beta buy job> for a new stock).
set -eo pipefail
STOCK="${1:?stock}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu
cd /home/u6gb/georgenigm.u6gb/LOBS5
export PYTHONPATH="$PWD:$PWD/Alphatrade:${PYTHONPATH:-}"
GRID="${EST_GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"
python -u lob_impact/5_analysis/_cst_estimate_stock.py "$STOCK" "$GRID" 300 100
python -u lob_impact/5_analysis/_hawkes_estimate_stock.py "$STOCK" "$GRID" 300 100
