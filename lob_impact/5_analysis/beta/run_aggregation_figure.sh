#!/bin/bash
#SBATCH --job-name=lobimp_aggfig
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/aggfig_%j.out
#
# Paper aggregation figure (raw spaghetti -> k-aligned mean) + duration-independence check.
#   sbatch lob_impact/5_analysis/beta/run_aggregation_figure.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"
MODELS="${MODELS:-Mamba3,Mamba3_4k,S5_4k}"
mkdir -p "${IMPACT_DIR}/5_analysis/beta/logs"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

cd "$IMPACT_DIR"
echo "[$(date)] host $(hostname) | grid=$GRID | models=$MODELS"
python -u 5_analysis/beta/aggregation_figure.py --grid "$GRID" --models "$MODELS"
echo "[$(date)] JOB_DONE"
