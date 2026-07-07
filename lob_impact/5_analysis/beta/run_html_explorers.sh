#!/bin/bash
#SBATCH --job-name=lobimp_a5_html
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/html_%j.out
#
# Action 5 — regenerate ALL interactive HTML dashboards on a COMPUTE node
# (each script re-globs the whole grid on Lustre -> never the login node):
#   beta_day_explorer / beta_k_explorer / beta_explorer_multimodel
#   impact_plotly_html (impact_trajectory beta + decay, with sqrt-law overlay)
#
#   GRID=/path STOCK=EA MODELS=... sbatch lob_impact/5_analysis/beta/run_html_explorers.sh
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
HERE="${IMPACT_DIR}/5_analysis/beta"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"
STOCK="${STOCK:-EA}"
MODELS="${MODELS:-Historic,Heuristic,Hawkes,CST,Propagator,Mamba3,Mamba3_4k,S5_4k,GDN}"
PER_DAY="${PER_DAY:-${IMPACT_DIR}/1_data_prep/results/per_day_params/per_day_params_${STOCK}.csv}"
mkdir -p "${HERE}/logs"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

if [ -z "${DAILY:-}" ]; then
  for f in $(ls -t "${IMPACT_DIR}"/2_daily_stats/results/*/daily_h_l_all.csv 2>/dev/null); do
    head -1 "$f" | grep -q open_price && { DAILY="$f"; break; }
  done
fi
[ -n "${DAILY:-}" ] || { echo "FATAL: no OHLC daily_h_l_all.csv found" >&2; exit 3; }
echo "[$(date)] host $(hostname) | grid ${GRID} | stock ${STOCK} | models ${MODELS}"

cd "$IMPACT_DIR"
echo ">>> impact_trajectory (beta)"
python 5_analysis/beta/impact_plotly_html.py --grid "$GRID" --stock "$STOCK" --shape beta \
    --models "$MODELS" --daily "$DAILY" --per_day_params "$PER_DAY"
echo ">>> impact_trajectory (relaxation/decay)"
python 5_analysis/beta/impact_plotly_html.py --grid "$GRID" --stock "$STOCK" --shape relaxation \
    --models "$MODELS" --daily "$DAILY" --per_day_params "$PER_DAY"
echo ">>> beta_day_explorer"
python 5_analysis/beta/beta_day_explorer.py --grid "$GRID" --stock "$STOCK" --daily "$DAILY" --models "$MODELS"
echo ">>> beta_k_explorer"
python 5_analysis/beta/beta_k_explorer.py --grid "$GRID" --stock "$STOCK" --daily "$DAILY" --models "$MODELS"
echo ">>> beta_explorer_multimodel"
python 5_analysis/beta/beta_explorer_multimodel.py --grid "$GRID" --stock "$STOCK" --daily "$DAILY" --models "$MODELS"
echo "[$(date)] done -> results/{mid_impact,beta_day_explorer,beta_k_explorer,beta_explorer}/"
