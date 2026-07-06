#!/bin/bash
#SBATCH --job-name=lobimp_durind
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/durind_%j.out
#
# Duration-(in)dependence probes from the EXISTING grid (no generation).
#   sbatch lob_impact/5_analysis/beta/run_duration_indata.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
mkdir -p "${IMPACT_DIR}/5_analysis/beta/logs"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

cd "$IMPACT_DIR"
echo "[$(date)] host $(hostname)"
python -u 5_analysis/beta/duration_indata.py
echo "[$(date)] JOB_DONE"
