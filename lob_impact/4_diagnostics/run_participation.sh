#!/bin/bash
#SBATCH --job-name=lobimp_a4_partrate
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/partrate_%j.out
#
# Action 4 diagnostic — participation-rate CONFIRMATION (Step-1 eta=10% calibration check).
# Runs on a COMPUTE node (globs thousands of grid CSVs -> Lustre metadata load off the login node).
# Emits, per model, a 3-stock x 2-col figure: windowed sawtooth (floor -> eta target) | cumulative.
#
#   sbatch lob_impact/4_diagnostics/run_participation.sh              # beta shape, count metric
#   SHAPE=relaxation sbatch lob_impact/4_diagnostics/run_participation.sh
#   METRIC=volume    sbatch lob_impact/4_diagnostics/run_participation.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid}"
SHAPE="${SHAPE:-beta}"
METRIC="${METRIC:-count}"
mkdir -p "${IMPACT_DIR}/4_diagnostics/logs"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

echo "[$(date)] host $(hostname) | grid ${GRID} | shape ${SHAPE} | metric ${METRIC}"
cd "$IMPACT_DIR"
MODELS="${MODELS:-Mamba3,Historic}"
python3.11 4_diagnostics/participation_rate.py \
  --grid "$GRID" --shape "$SHAPE" --metric "$METRIC" --models "$MODELS"
echo "[$(date)] done -> 4_diagnostics/results/participation_rate/participation_{${MODELS}}_${SHAPE}_${METRIC}.png"
