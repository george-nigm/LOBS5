#!/bin/bash
#SBATCH --job-name=lobimp_tricmp
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/tricmp_%j.out
#
# Cross-model control-triangle comparison (figures + Word doc). Reads the cached
# per-model numbers.json — no grid access.
#   sbatch lob_impact/4_diagnostics/run_triangle_compare.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
MODELS="${MODELS:-Mamba3,Mamba3_4k,S5_4k}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
OUT="${IMPACT_DIR}/4_diagnostics/results/triangle_compare_${RUN_TS}"
mkdir -p "${IMPACT_DIR}/4_diagnostics/logs" "$OUT"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

cd "$IMPACT_DIR"
echo "[$(date)] host $(hostname) | models=$MODELS | out=$OUT"
python -u 4_diagnostics/triangle_compare_report.py --models "$MODELS" --out_dir "$OUT"
echo "[$(date)] TRICMP_DONE -> $OUT"
