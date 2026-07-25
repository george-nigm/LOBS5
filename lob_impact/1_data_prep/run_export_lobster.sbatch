#!/bin/bash
#SBATCH --job-name=lob_export
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/1_data_prep/logs/lob_export_%j.out
# Export a stock's proc npy month to LOBSTER csv (for external-model retrains).
# Usage: sbatch run_export_lobster.sbatch AAPL
set -euo pipefail
STOCK="${1:?stock}"
OUT="${OUT:-/lus/lfs1aip2/projects/u6gb/lob_impact_v4_external/lobster_export}"
SRC="${SRC:-/lus/lfs1aip2/projects/public/u6gb/projects_public_s5e_quant_team}"
SHARD="${SHARD:-shard_2026-01.squashfs}"

MNT="${TMPDIR:-/tmp}/lob_export_${SLURM_JOB_ID}"
mkdir -p "$MNT"
squashfuse_ll "$SRC/$SHARD" "$MNT"
trap 'fusermount3 -u "$MNT" 2>/dev/null || fusermount -u "$MNT" 2>/dev/null || true' EXIT

set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
python -u /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/1_data_prep/export_lobster_csv.py \
  --mnt "$MNT/$STOCK" --stock "$STOCK" --out "$OUT"
