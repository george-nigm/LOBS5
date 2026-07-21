#!/bin/bash
#SBATCH --job-name=causal
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/causal_%j.out
# Placebo + pre-trend + flow-balance causal tests for one stock.
set -eo pipefail
STOCK="${1:?stock}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2
CTRL=/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2
DAILY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/results/daily_20260708-131449/daily_h_l_all.csv
python -u placebo_pretrend.py --grid "$GRID" --controls "$CTRL" --daily "$DAILY" --stock "$STOCK"
python -u flow_balance.py --controls "$CTRL" --grid "$GRID" --stock "$STOCK"
