#!/bin/bash
# Action 2 — daily stats (H/L, execution volume) per chosen stock, for impact normalization.
# Output:  2_daily_stats/results/daily_<timestamp>/   Log: 2_daily_stats/logs/daily_<timestamp>.log
#   export DATA_MOUNT=<mounted squashfs root>
#   bash 2_daily_stats/run_daily_stats.sh            # default stocks below
#   STOCKS="EA NVDA AMD" bash 2_daily_stats/run_daily_stats.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_MOUNT="${DATA_MOUNT:?set DATA_MOUNT to the mounted squashfs root (one subdir per ticker)}"
STOCKS="${STOCKS:-EA NVDA AMD}"
EXTRA="${EXTRA:-}"      # e.g. EXTRA="--aggregate" for one pooled row per stock

RUN_TS="$(date +%Y%m%d-%H%M%S)"
RESULTS="${HERE}/results/daily_${RUN_TS}"
LOG="${HERE}/logs/daily_${RUN_TS}.log"
mkdir -p "$RESULTS" "${HERE}/logs"

{
  echo ">>> Action 2: daily stats  | run ${RUN_TS}  | stocks: ${STOCKS}"
  for s in $STOCKS; do
    echo "--- ${s} ---"
    python3 "${HERE}/compute_daily_stats.py" --mnt "$DATA_MOUNT" --stock "$s" --out_dir "$RESULTS" $EXTRA
  done
  echo ">>> done. Inspect ${RESULTS}/daily_h_l_<STOCK>.csv  (verify H/L look like real prices)."
} 2>&1 | tee "$LOG"
