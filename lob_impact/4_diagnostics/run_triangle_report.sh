#!/bin/bash
#SBATCH --job-name=lobimp_triangle
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/triangle_%j.out
#
# Control-triangle report: figures + numbers + Word doc, on a COMPUTE node.
#   sbatch lob_impact/4_diagnostics/run_triangle_report.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
NS="${NS:-64}"
MODEL="${MODEL:-Mamba3}"          # grid model label: Mamba3 | Mamba3_4k | S5_4k
RUN_TS="$(date +%Y%m%d-%H%M%S)"
OUT="${IMPACT_DIR}/4_diagnostics/results/triangle_${MODEL}_${RUN_TS}"
mkdir -p "${IMPACT_DIR}/4_diagnostics/logs" "$OUT"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

cd "$IMPACT_DIR"
echo "[$(date)] host $(hostname) | model=$MODEL | n_samples=$NS | out=$OUT"
python -u 4_diagnostics/control_triangle_report.py --model "$MODEL" --n_samples "$NS" --out_dir "$OUT"
python -u 4_diagnostics/make_triangle_docx.py --fig_dir "$OUT" --model "$MODEL" \
    --out "$OUT/Control_Triangle_Report_${MODEL}.docx"
echo "[$(date)] TRIANGLE_DONE -> $OUT"
