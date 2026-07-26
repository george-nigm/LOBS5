#!/bin/bash
#SBATCH --job-name=figsweep
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/figsweep_%j.out
# Idempotent figure sweep: for one stock, re-render every paper figure whose npz cache
# already exists and whose png is missing from paper/Figures. Every step is best-effort
# (|| true) so a not-yet-computed cache just skips instead of killing the job.
# Re-runnable every monitoring cycle as caches land.  Usage: sbatch ... <STOCK>
set -uo pipefail
ST="${1:?stock}"
ROOT=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
FIG="$ROOT/paper/Figures"
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"
DAILY="${DAILY:-$ROOT/2_daily_stats/results/daily_20260708-131449/daily_h_l_all_plus.csv}"
MODELS="${MODELS:-Historic,Heuristic,Propagator,OW,QR,Hawkes,CST,NMZI,Mamba3,GDN,S5_120M,Mamba3_4k,S5_4k}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu
cd "$ROOT/5_analysis/beta"

have() { [ -e "$FIG/$1" ]; }        # png already in the paper
cache() { [ -e "$1" ]; }            # npz cache present

# --- pure cache re-renders (seconds each) -------------------------------------
have "binning_steps_$ST.png"   || cache "results/bias_exhibit/bias_exhibit_$ST.npz" && \
  python -u fig_binning_steps.py --stock "$ST" --copy_to "$FIG" || true
have "loss_surface_$ST.png"    || cache "results/bias_exhibit/bias_exhibit_$ST.npz" && \
  python -u fig_loss_surface.py --stock "$ST" --copy_to "$FIG" || true
have "decay_protocol_$ST.png"  || cache "results/master_curve/master_curve_${ST}_relaxation_v2gated.npz" && \
  python -u fig_decay_protocol.py --stock "$ST" --copy_to "$FIG" || true
have "beta_3x3_body_${ST}_suspended.png" || cache "results/beta_3x3/beta_3x3_$ST.npz" && \
  python -u beta_3x3_body.py --stock "$ST" --copy_to "$FIG" || true
have "beta_3est_$ST.png"       || cache "results/beta_3views_l2/beta_3views_l2_$ST.npz" && \
  python -u beta_3est_body.py --stock "$ST" --copy_to "$FIG" || true
have "beta_3x3_Y_$ST.png"      || cache "results/bias_exhibit/bias_exhibit_$ST.npz" && \
  python -u beta_3x3_Y.py --stock "$ST" --copy_to "$FIG" || true
have "beta_sigma_l2_$ST.png"   || cache "results/beta_sigma_grid/beta_sigma_grid_${ST}_le.npz" && \
  python -u beta_sigma_l2_summary.py --stock "$ST" || true
have "amplitude_stability_$ST.png" || cache "results/beta_sigma_grid/points_$ST.npz" && \
  python -u amplitude_stability.py --stock "$ST" || true
have "beta_pipeline_$ST.png"   || cache "results/bias_exhibit/bias_exhibit_$ST.npz" && \
  python -u beta_pipeline_steps.py --stock "$ST" --models "$MODELS" || true

# --- collect anything the renders wrote under results/ -------------------------
for f in binning_steps loss_surface decay_protocol beta_3x3_body beta_3est beta_3x3_Y \
         beta_sigma_l2 amplitude_stability beta_pipeline; do
  for p in results/*/"${f}_${ST}"*.png; do [ -e "$p" ] && cp -n "$p" "$FIG/" 2>/dev/null; done
done
echo "FIGSWEEP_DONE $ST"
ls "$FIG" | grep -c "_${ST}\." || true
