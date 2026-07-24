#!/bin/bash
# 2026-07-24 fixed analysis waves (C-G). Fixes vs submit_all_waves_0724.sh:
#  - sub() prints ONLY the job id to stdout (journal line goes straight to file),
#    so captured $MJ is clean for --dependency chains;
#  - correct path for run_empresp_curve.sbatch (4_diagnostics, not 5_analysis).
# Deferred to the monitor loop (QOS 512-job cap): OW/QR noins slices for
# EA/GOOG/MSFT/AAPL -- resubmit via submit_all_waves_0724.sh wave B when quota frees.
set -uo pipefail
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
JIDS=logs/analysis_waves_0724_$(date +%H%M%S).txt
AN=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis
DI=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics
CONDA="source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5"
M13="Historic,Heuristic,Propagator,Hawkes,CST,NMZI,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN,OW,QR"
sub() { local tag="$1"; shift; local j; j=$(sbatch --parsable "$@" 2>>"$JIDS.err") || j=FAIL; echo "$tag=$j" >> "$JIDS"; sleep 1; echo "$j"; }

# ---- C: masters with 13 models -> fig5_combined + decay-kernel fit per stock
for ST in EA NVDA GOOG AMD MSFT; do
  MJ=$(sub "C master13 $ST" --time=06:00:00 --export=ALL,STOCK=$ST,MODELS=$M13 "$AN/beta/run_master_v2.sh")
  if [ "$MJ" != FAIL ]; then
    sub "C fig5c $ST" --dependency=afterok:$MJ --export=ALL,STOCKS_F5=$ST "$AN/run_fig5_combined.sbatch" >/dev/null
    sub "C dkf $ST" --dependency=afterok:$MJ --account=brics.u6gb --partition=workq --cpus-per-task=4 --mem=32G --time=00:30:00 \
      --job-name=dkf_$ST --output=$AN/logs/dkf_${ST}_%j.out \
      --wrap="$CONDA; cd $AN/beta; python -u decay_kernel_fit.py --stock $ST" >/dev/null
  else
    echo "C chain $ST skipped (master FAIL)" >> "$JIDS"
  fi
done

# ---- D: b3l2 with OW/QR
for ST in EA NVDA GOOG AMD MSFT; do
  sub "D b3l2 $ST" --export=ALL,MODELS=$M13 "$AN/run_beta_3views_l2.sbatch" $ST >/dev/null
done

# ---- E: binned delta with OW/QR
for ST in EA NVDA GOOG AMD MSFT; do
  sub "E binned $ST" --export=ALL,STOCKS=$ST,MODELS=$M13 "$AN/beta/run_beta.sh" >/dev/null
done

# ---- F: LobS5 flow-balance after legacy noins (5764652)
sub "F lobs5 flowbal" --dependency=afterok:5764652 --account=brics.u6gb --partition=workq --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --job-name=fb_lobs5 --output=$DI/logs/fb_lobs5_%j.out \
  --wrap="$CONDA; export JAX_PLATFORMS=cpu; cd $DI; python -u flow_balance.py --controls /lus/lfs1aip2/projects/u6gb/lob_impact_legacy --grid /lus/lfs1aip2/projects/u6gb/lob_impact_legacy --stock GOOG2023 --models S5" >/dev/null

# ---- G: appendix fill: sigma grids + empresp MSFT/AMD
sub "G bsg AMD ge"  --account=brics.u6gb --partition=workq "$AN/run_beta_sigma_grid.sbatch" AMD ge  >/dev/null
for V in le eq ge; do
  sub "G bsg MSFT $V" --account=brics.u6gb --partition=workq "$AN/run_beta_sigma_grid.sbatch" MSFT $V >/dev/null
done
sub "G empresp MSFT" --time=03:00:00 "$DI/run_empresp_curve.sbatch" MSFT >/dev/null
sub "G empresp AMD"  --time=03:00:00 "$DI/run_empresp_curve.sbatch" AMD  >/dev/null

echo "ANALYSIS_WAVES_DONE ok=$(grep -c '=57' "$JIDS" || true) fail=$(grep -c '=FAIL' "$JIDS" || true)" >> "$JIDS"
tail -1 "$JIDS"
