#!/bin/bash
#SBATCH --job-name=bsg
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:59:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/b33_%j.out
set -eo pipefail
STOCK="${1:?stock}"
VIEW="${2:-le}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
DAILY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/results/daily_20260708-131449/daily_h_l_all.csv
python -u beta_sigma_grid.py --grid /lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2 \
  --daily "$DAILY" --stock "$STOCK" --view "$VIEW" \
  --points_cache /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/results/beta_sigma_grid/points_${STOCK}.npz
