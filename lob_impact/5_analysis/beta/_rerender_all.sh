#!/bin/bash
# Pure cache re-render of every legend-bearing paper figure (seconds each). Tiny
# allocation on purpose: at priority-1 fairshare only small jobs backfill, and an
# 8-CPU/64G ask sat pending for hours behind the AAPL fleet.
set -uo pipefail
source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
FIG=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/paper/Figures
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
for ST in ${STOCKS_R:-EA NVDA GOOG AMD MSFT}; do
  python -u fig5_combined.py      --stock $ST --copy_to $FIG || echo "skip fig5 $ST"
  python -u beta_3est_body.py     --stock $ST --copy_to $FIG || echo "skip 3est $ST"
  python -u beta_3x3_Y.py         --stock $ST --copy_to $FIG || echo "skip Y $ST"
  python -u fig_decay_protocol.py --stock $ST --copy_to $FIG || echo "skip dp $ST"
  python -u beta_3x3_body.py --stock $ST --mask --suffix suspended --copy_to $FIG || echo "skip b33b $ST"
done
python -u sample_count_table.py || true
echo RERENDER_ALL_DONE
