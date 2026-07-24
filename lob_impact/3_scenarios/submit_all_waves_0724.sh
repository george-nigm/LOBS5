#!/bin/bash
# 2026-07-24 "ставь всё сразу": every remaining wave into the queue at once.
# A: 2 lost stage-0 slices    B: OW/QR stage-0 noins 2048 (6 stocks x 8 slices x 2)
# C: masters with 13 models (5 stocks) -> chained fig5_combined + decay-kernel fit
# D: b3l2 with OW/QR (5)      E: binned delta with OW/QR (5)
# F: LobS5 flow-balance (after legacy noins)   G: appendix: sigma grids + empresp MSFT
set -uo pipefail
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios
JIDS=logs/all_waves_0724_$(date +%H%M%S).txt
AN=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis
DI=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics
NOINS=/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2/noins
CONDA="source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5"
M13="Historic,Heuristic,Propagator,Hawkes,CST,NMZI,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN,OW,QR"
sub() { local tag="$1"; shift; local j; j=$(sbatch --parsable "$@") || j=FAIL; echo "$tag=$j" | tee -a "$JIDS"; sleep 1; echo "$j"; }

# ---- A: lost stage-0 slices
sub "A NVDA s5_4k s11" --account=brics.u6gb --partition=workq --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=10:00:00 \
  --job-name=s0_s5_411 --output=logs/s0_NVDA_s5_4k_s11_%j.out \
  --export=ALL,SAVE_BASE=$NOINS,STOCKS=NVDA,ONLY_SHAPE=beta,DIRS=buy,N_SAMPLES=2048,SAMPLE_SLICE=11/16,N_INS_OVERRIDE=1,MB_OVERRIDE=25000,BSZ=8 \
  run_experiments.sh full s5_4k >/dev/null
sub "A EA s5_120m s0" --account=brics.u6gb --partition=workq --cpus-per-task=16 --mem=96G --time=06:00:00 \
  --job-name=s0E_s5_1 --output=logs/s0_EA_s5_120m_s0_%j.out \
  --export=ALL,SAVE_BASE=$NOINS,STOCKS=EA,ONLY_SHAPE=beta,DIRS=buy,N_SAMPLES=2048,SAMPLE_SLICE=0/8,N_INS_OVERRIDE=1,MB_OVERRIDE=13000 \
  run_experiments.sh full s5_120m >/dev/null

# ---- B: OW/QR stage-0 noins 2048, 8 slices each
declare -A MBOV=([EA]=13000 [NVDA]=25000 [GOOG]=73000 [AMD]=55000 [MSFT]=26000 [AAPL]=38000)
for ST in EA NVDA GOOG AMD MSFT AAPL; do
  for m in ow qr; do
    EXTRA=""; [ "$m" = ow ] && EXTRA=",PROP_KERNEL=exp,PROP_TAU=1000"
    for k in 0 1 2 3 4 5 6 7; do
      sub "B $ST $m s$k" --account=brics.u6gb --partition=workq --cpus-per-task=32 --mem=256G --time=12:00:00 \
        --job-name=s0${ST:0:1}_${m}$k --output=logs/s0_${ST}_${m}_s${k}_%j.out \
        --export=ALL,SAVE_BASE=$NOINS,STOCKS=$ST,ONLY_SHAPE=beta,DIRS=buy,N_SAMPLES=2048,SAMPLE_SLICE=$k/8,N_INS_OVERRIDE=1,MB_OVERRIDE=${MBOV[$ST]}$EXTRA \
        run_experiments.sh full $m >/dev/null
    done
  done
done

# ---- C: masters with 13 models -> fig5_combined + decay-kernel fit per stock
for ST in EA NVDA GOOG AMD MSFT; do
  MJ=$(sub "C master13 $ST" --time=06:00:00 --export=ALL,STOCK=$ST,MODELS=$M13 "$AN/beta/run_master_v2.sh")
  sub "C fig5c $ST" --dependency=afterok:$MJ --export=ALL,STOCKS_F5=$ST "$AN/run_fig5_combined.sbatch" >/dev/null
  sub "C dkf $ST" --dependency=afterok:$MJ --account=brics.u6gb --partition=workq --cpus-per-task=4 --mem=32G --time=00:30:00 \
    --job-name=dkf_$ST --output=$AN/logs/dkf_${ST}_%j.out \
    --wrap="$CONDA; cd $AN/beta; python -u decay_kernel_fit.py --stock $ST" >/dev/null
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

# ---- G: appendix fill: sigma grids + empresp MSFT
sub "G bsg AMD ge"  --account=brics.u6gb --partition=workq "$AN/run_beta_sigma_grid.sbatch" AMD ge  >/dev/null
for V in le eq ge; do
  sub "G bsg MSFT $V" --account=brics.u6gb --partition=workq "$AN/run_beta_sigma_grid.sbatch" MSFT $V >/dev/null
done
sub "G empresp MSFT" --time=03:00:00 "$AN/run_empresp_curve.sbatch" MSFT >/dev/null
sub "G empresp AMD"  --time=03:00:00 "$AN/run_empresp_curve.sbatch" AMD  >/dev/null

echo "ALL_WAVES_SUBMITTED $(grep -c '=' "$JIDS") entries, failures: $(grep -c '=FAIL' "$JIDS" || true)" | tee -a "$JIDS"
