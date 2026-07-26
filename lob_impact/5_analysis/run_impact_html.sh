#!/bin/bash
#SBATCH --job-name=impact_html_EA
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/impact_html_EA_%j.out
# ALL-MODELS interactive Plotly HTML of the mid-price impact trajectory.
# Runs BOTH shapes for EA -> two self-contained HTMLs (open offline, native legend toggling).
set -uo pipefail
export JAX_PLATFORMS=cpu
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
B=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid
DAILY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/2_daily_stats/results/daily_20260618-215656/daily_h_l_all.csv
PERDAY=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/1_data_prep/results/per_day_params/per_day_params_EA.csv
STOCK=EA
MODELS="${MODELS:-Historic,Heuristic,OW,Hawkes,QR,CST,Mamba3,Mamba3_4k,S5_4k}"
mkdir -p "$B/results/mid_impact"
cd "$B"

# Shape I (beta): √-law peak overlay, no propagator
$PY impact_plotly_html.py --grid "$GRID" --stock "$STOCK" --shape beta \
   --models "$MODELS" --daily "$DAILY" --per_day_params "$PERDAY" \
   --out "$B/results/mid_impact/impact_trajectory_${STOCK}_beta.html"

# Shape II (relaxation -> titled "decay"): √-law permanent ref + propagator overlay
$PY impact_plotly_html.py --grid "$GRID" --stock "$STOCK" --shape relaxation \
   --models "$MODELS" --daily "$DAILY" --per_day_params "$PERDAY" \
   --out "$B/results/mid_impact/impact_trajectory_${STOCK}_decay.html"
