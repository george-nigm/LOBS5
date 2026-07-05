#!/bin/bash
#SBATCH --job-name=lobimp_verify
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/verify_%j.out
#
# Action 4 — integrity re-audit of grid_v2 samples on a COMPUTE node (Lustre CSV reads).
#   GRID=/path STOCK=EA SHAPES=beta,relaxation sbatch lob_impact/4_diagnostics/run_verify_integrity.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"
STOCK="${STOCK:-EA}"
MODELS="${MODELS:-Mamba3,Mamba3_4k,S5_4k,Historic,Heuristic,Hawkes,CST}"
SHAPES="${SHAPES:-beta,relaxation}"
NS="${NS:-30}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
OUT="${IMPACT_DIR}/4_diagnostics/results/verify_${RUN_TS}"
mkdir -p "${IMPACT_DIR}/4_diagnostics/logs" "$OUT"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

cd "$IMPACT_DIR"
echo "[$(date)] host $(hostname) | grid $GRID | stock $STOCK | models $MODELS | shapes $SHAPES | n=$NS"
python -u 4_diagnostics/verify_integrity.py --grid "$GRID" --stock "$STOCK" \
    --models "$MODELS" --shapes "$SHAPES" --n_samples "$NS" --out_dir "$OUT"
echo "[$(date)] VERIFY_DONE -> $OUT"
