#!/bin/bash
#SBATCH --job-name=lobimp_a1_msgsbtw
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/1_data_prep/logs/msgs_btw_%j.out
#
# Action 1 — msgs_between over the whole S&P500 universe -> stock-selection table + histograms.
# Mounts ONE month shard (squashfuse_ll, node-local) and computes the table.
#
#   sbatch lob_impact/1_data_prep/run_msgs_btw.sh                          # default Jan-2026
#   SHARD=shard_2025-12.squashfs sbatch lob_impact/1_data_prep/run_msgs_btw.sh
#
# Output:  1_data_prep/results/msgs_btw_<timestamp>/   Log: the #SBATCH --output above.
set -euo pipefail

IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
HERE="${IMPACT_DIR}/1_data_prep"
SRC="${SRC:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}"
SHARD="${SHARD:-shard_2026-01.squashfs}"          # January 2026
WORKERS="${WORKERS:-48}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
RESULTS="${HERE}/results/msgs_btw_${RUN_TS}"
MNT="${TMPDIR:-/tmp}/s5e_mnt_${SLURM_JOB_ID:-$$}"
mkdir -p "$RESULTS" "$MNT" "${HERE}/logs"

echo "[$(date)] Action 1 | run ${RUN_TS} | host $(hostname)"
echo "[$(date)] mounting ${SHARD} -> ${MNT}"
squashfuse_ll "${SRC}/${SHARD}" "$MNT"
trap 'fusermount -u "$MNT" 2>/dev/null; rmdir "$MNT" 2>/dev/null' EXIT
echo "[$(date)] tickers in shard: $(ls "$MNT" | wc -l)"

python3.11 "${HERE}/compute_sp500_msgs_btw.py" --mnt "$MNT" --out_dir "$RESULTS" --workers "$WORKERS"
python3.11 "${HERE}/postprocess_sp500.py" "$RESULTS"

echo "[$(date)] done -> ${RESULTS}"
echo "  msgs_btw_sp500_{perday,aggregated,daymean}.csv + hist PNGs (pick 3 stocks from the daymean table)"
