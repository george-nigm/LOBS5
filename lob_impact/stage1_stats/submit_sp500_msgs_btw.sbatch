#!/bin/bash
#SBATCH --job-name=sp500_mb
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/sp500_universe/slurm_%j.out

set -euo pipefail

SRC=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
SHARD=shard_2026-01.squashfs
OUT=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/sp500_universe
SCRIPT=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/compute_sp500_msgs_btw.py
MNT=${TMPDIR:-/tmp}/s5e_mnt_${SLURM_JOB_ID}

mkdir -p "$OUT" "$MNT"
echo "[$(date)] mounting $SHARD on $(hostname) -> $MNT"
squashfuse_ll "$SRC/$SHARD" "$MNT"
trap 'fusermount -u "$MNT" 2>/dev/null; rmdir "$MNT" 2>/dev/null' EXIT

echo "[$(date)] tickers in shard: $(ls "$MNT" | wc -l)"
python3.11 "$SCRIPT" --mnt "$MNT" --out_dir "$OUT" --workers 48
echo "[$(date)] done"
