#!/bin/bash
#SBATCH --job-name=causal
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/causal_%j.out
# Placebo + pre-trend + flow-balance causal tests for one stock.
set -eo pipefail
STOCK="${1:?stock}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2
CTRL=/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2
IMPACT_DIR=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
# The hardcoded daily_h_l_all.csv covers only EA/NVDA/GOOG/AMD -- on MSFT and AAPL every model
# silently scores zero points and the job still exits 0. Pick the newest OHLC table that actually
# carries this stock, and abort loudly if none does.
if [ -z "${DAILY:-}" ]; then
  for f in $(ls -t "$IMPACT_DIR"/2_daily_stats/results/*/daily_h_l_all_plus.csv \
                   "$IMPACT_DIR"/2_daily_stats/results/*/daily_h_l_all.csv 2>/dev/null); do
    head -1 "$f" | grep -q open_price || continue
    grep -q "^${STOCK}," "$f" && { DAILY="$f"; break; }
  done
fi
[ -n "${DAILY:-}" ] || { echo "FATAL: no OHLC daily table covering $STOCK" >&2; exit 3; }
echo "[causal] daily=$DAILY"
# placebo_pretrend.py's own --models default lists 6 models, which is why the cache carried 6 while
# the figure legend claims the full panel; pass the canonical list explicitly.
MODELS="${MODELS:-Historic,Heuristic,Propagator,OW,CST,NMZI,Hawkes,QR,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN}"
python -u placebo_pretrend.py --grid "$GRID" --controls "$CTRL" --daily "$DAILY" --stock "$STOCK" --models "$MODELS"
python -u flow_balance.py --controls "$CTRL" --grid "$GRID" --stock "$STOCK"
