#!/bin/bash
#SBATCH --job-name=lobimp_a2_daily
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/logs/daily_%j.out
#
# Action 2 — daily stats (H/L, execution volume) for impact normalization.
# Mounts ONE month shard (squashfuse_ll, node-local) and computes:
#   default              -> daily_h_l_all.csv  (ALL S&P500 tickers x all days, one file)
#   STOCKS="EA NVDA AMD" -> per-stock daily_h_l_<STOCK>.csv (for the analysis loader)
#
#   sbatch lob_impact/2_daily_stats/run_daily_stats.sh                          # ALL, Jan-2026
#   STOCKS="EA NVDA AMD" sbatch lob_impact/2_daily_stats/run_daily_stats.sh     # just these 3
#   SHARD=shard_2025-12.squashfs sbatch lob_impact/2_daily_stats/run_daily_stats.sh
set -euo pipefail

IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
HERE="${IMPACT_DIR}/2_daily_stats"
SRC="${SRC:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}"
SHARD="${SHARD:-shard_2026-01.squashfs}"          # January 2026
WORKERS="${WORKERS:-48}"
STOCKS="${STOCKS:-}"                                # empty => ALL tickers, one combined file

RUN_TS="$(date +%Y%m%d-%H%M%S)"
RESULTS="${HERE}/results/daily_${RUN_TS}"
MNT="${TMPDIR:-/tmp}/s5e_mnt_${SLURM_JOB_ID:-$$}"
mkdir -p "$RESULTS" "$MNT" "${HERE}/logs"

echo "[$(date)] Action 2 | run ${RUN_TS} | host $(hostname) | stocks: ${STOCKS:-ALL}"
echo "[$(date)] mounting ${SHARD} -> ${MNT}"
squashfuse_ll "${SRC}/${SHARD}" "$MNT"
trap 'fusermount -u "$MNT" 2>/dev/null; rmdir "$MNT" 2>/dev/null' EXIT

if [ -z "$STOCKS" ]; then
  python3.11 "${HERE}/compute_daily_stats.py" --mnt "$MNT" --out_dir "$RESULTS" --workers "$WORKERS"
  echo "[$(date)] done -> ${RESULTS}/daily_h_l_all.csv"
else
  for s in $STOCKS; do
    echo "--- ${s} ---"
    python3.11 "${HERE}/compute_daily_stats.py" --mnt "$MNT" --stock "$s" --out_dir "$RESULTS"
  done
  echo "[$(date)] done -> ${RESULTS}/daily_h_l_<STOCK>.csv"
fi
