#!/bin/bash
#SBATCH --job-name=lobimp_fig4doc
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/fig4doc_%j.out
#
# Explainer doc for the Fig 4 variable-length averaging (2 figures + docx).
#   sbatch lob_impact/4_diagnostics/run_fig4_averaging_docx.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
OUT="${IMPACT_DIR}/4_diagnostics/results/fig4_averaging_${RUN_TS}"
mkdir -p "${IMPACT_DIR}/4_diagnostics/logs" "$OUT"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

cd "$IMPACT_DIR"
echo "[$(date)] host $(hostname) | out=$OUT"
python -u 4_diagnostics/make_fig4_averaging_docx.py --out_dir "$OUT"
echo "[$(date)] FIG4DOC_DONE -> $OUT"
