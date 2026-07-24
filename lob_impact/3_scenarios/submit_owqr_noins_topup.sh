#!/bin/bash
# Top-up submitter for the deferred OW/QR stage-0 noins slices (QOS 512-job cap).
# State: logs/owqr_noins_pending.txt ("STOCK model slice" per line). Each run pops
# as many lines as quota allows (keeps HEADROOM slots free), submits, journals to
# logs/owqr_noins_topup.txt. Idempotent: run from the monitor loop until the
# pending file is empty. sbatch-only, safe on the login node.
set -uo pipefail
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
PENDING=logs/owqr_noins_pending.txt
JIDS=logs/owqr_noins_topup.txt
NOINS=/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2/noins
HEADROOM="${HEADROOM:-2}"
QOS_CAP=512
declare -A MBOV=([EA]=13000 [NVDA]=25000 [GOOG]=73000 [AMD]=55000 [MSFT]=26000 [AAPL]=38000)

[ -s "$PENDING" ] || { echo "TOPUP: pending list empty — nothing to do"; exit 0; }
inq=$(squeue -u "$(whoami)" -h 2>/dev/null | wc -l) || { echo "TOPUP: squeue failed, retry next cycle"; exit 0; }
free=$(( QOS_CAP - HEADROOM - inq ))
[ "$free" -gt 0 ] || { echo "TOPUP: no quota (in queue: $inq)"; exit 0; }

n=0
while [ "$n" -lt "$free" ] && [ -s "$PENDING" ]; do
  read -r ST m k < "$PENDING"
  # unique full-stock job name (AMD vs AAPL both start with A) + skip-if-queued guard:
  # sbatch client timeouts can report failure for a job the server accepted.
  JN=s0_${ST}_${m}$k
  if squeue -u "$(whoami)" -h -n "$JN" 2>/dev/null | grep -q .; then
    echo "TOPUP: '$ST $m $k' already queued, dropping from list" >> "$JIDS"
    sed -i '1d' "$PENDING"; continue
  fi
  EXTRA=""; [ "$m" = ow ] && EXTRA=",PROP_KERNEL=exp,PROP_TAU=1000"
  j=$(sbatch --parsable --account=brics.u6gb --partition=workq --cpus-per-task=32 --mem=256G --time=12:00:00 \
    --job-name="$JN" --output=logs/s0_${ST}_${m}_s${k}_%j.out \
    --export=ALL,SAVE_BASE=$NOINS,STOCKS=$ST,ONLY_SHAPE=beta,DIRS=buy,N_SAMPLES=2048,SAMPLE_SLICE=$k/8,N_INS_OVERRIDE=1,MB_OVERRIDE=${MBOV[$ST]}$EXTRA \
    run_experiments.sh full "$m" 2>>"$JIDS.err") || { echo "TOPUP: sbatch failed on '$ST $m $k', stopping this cycle" | tee -a "$JIDS"; exit 0; }
  echo "B $ST $m s$k=$j" >> "$JIDS"
  sed -i '1d' "$PENDING"
  n=$((n+1)); sleep 1
done
echo "TOPUP: submitted $n, remaining $(wc -l < "$PENDING")"
