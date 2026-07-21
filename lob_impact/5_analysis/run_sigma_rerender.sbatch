#!/bin/bash
#SBATCH --job-name=bsgrr
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/bsgrr_%j.out
# Cache-only re-renders that depend on the beta_sigma_grid npz files; run with
# sbatch -d afterok:<bsg jobids> after a sigma-grid fleet.
set -eo pipefail
STOCK="${1:?stock}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
python -u beta_sigma_byestimator.py --stock "$STOCK" --estimator l2
python -u beta_sigma_l2_summary.py --stock "$STOCK"
python -u amplitude_stability.py --stock "$STOCK"
