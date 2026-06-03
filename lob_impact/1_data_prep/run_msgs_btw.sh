#!/bin/bash
# Action 1 — msgs_between over the S&P500 universe -> stock-selection table + histograms.
# Output:  1_data_prep/results/msgs_btw_<timestamp>/   Log: 1_data_prep/logs/msgs_btw_<timestamp>.log
#   export DATA_MOUNT=<mounted squashfs root>
#   bash 1_data_prep/run_msgs_btw.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_MOUNT="${DATA_MOUNT:?set DATA_MOUNT to the mounted squashfs root (one subdir per ticker)}"
WORKERS="${WORKERS:-96}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
RESULTS="${HERE}/results/msgs_btw_${RUN_TS}"
LOG="${HERE}/logs/msgs_btw_${RUN_TS}.log"
mkdir -p "$RESULTS" "${HERE}/logs"

{
  echo ">>> Action 1: msgs_between  | run ${RUN_TS}"
  echo "    mnt=${DATA_MOUNT}  results=${RESULTS}"
  python3 "${HERE}/compute_sp500_msgs_btw.py" --mnt "$DATA_MOUNT" --out_dir "$RESULTS" --workers "$WORKERS"
  python3 "${HERE}/postprocess_sp500.py" "$RESULTS"
  echo ">>> done. Inspect ${RESULTS}/ for: msgs_btw_sp500_{perday,aggregated,daymean}.csv + hist PNGs."
} 2>&1 | tee "$LOG"
