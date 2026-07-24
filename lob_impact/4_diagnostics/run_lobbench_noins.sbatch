#!/bin/bash
#SBATCH --job-name=lbb_noins
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/lbb_noins_%j.out
# LOB-Bench distributional scoring of the noins (free-rollout) controls for one
# stock: KS/L1/WS columns of Tables 7/8, on the SAME stock/month as the impact
# runs. Stages the noins outputs into the layout run_bench.py expects
# ({data_dir}/{model}/{stock}/{period}/data_{real,gen,cond}) via symlinks, then
# scores every model that has generated data. Usage: sbatch ... run_lobbench_noins.sbatch NVDA
set -eo pipefail
STOCK="${1:?stock}"
PERIOD="${PERIOD:-2026_Jan}"
NOINS="${NOINS:-/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2/noins}"
BENCH="${BENCH:-/lus/lfs1aip2/projects/u6gb/public/projects_public_s5e_quant_team_lob_pipeline/lob_bench}"
OUT=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/results/lobbench_noins
STAGE="$OUT/stage_$STOCK"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu

# ---- stage: merge every exp_* of every model into the bench layout
rm -rf "$STAGE"; mkdir -p "$OUT/scores_$STOCK"
MODELS=()
for d in "$NOINS/$STOCK-"*-beta; do
  [ -d "$d" ] || continue
  M=$(basename "$d"); M=${M#"$STOCK"-}; M=${M%-beta}
  ngen=$(find "$d"/buy/exp_*/data_gen -maxdepth 1 -name '*message*gen*.csv' 2>/dev/null | wc -l)
  if [ "$ngen" -eq 0 ]; then echo "SKIP $M (no generated files yet)"; continue; fi
  T="$STAGE/$M/$STOCK/$PERIOD"
  mkdir -p "$T/data_real" "$T/data_gen" "$T/data_cond"
  for sub in data_real data_gen data_cond; do
    for f in "$d"/buy/exp_*/"$sub"/*.csv; do
      [ -e "$f" ] || continue
      ln -sf "$f" "$T/$sub/$(basename "$f")"
    done
  done
  MODELS+=("$M")
  echo "STAGED $M: gen=$ngen real=$(ls "$T/data_real" | wc -l) cond=$(ls "$T/data_cond" | wc -l)"
done
[ "${#MODELS[@]}" -gt 0 ] || { echo "NO MODELS STAGED — abort"; exit 1; }

# ---- score
cd "$BENCH"
python -u run_bench.py --stock "$STOCK" --time_period "$PERIOD" \
  --data_dir "$STAGE" --save_dir "$OUT/scores_$STOCK" \
  --model_name "${MODELS[@]}" --n_workers "${SLURM_CPUS_PER_TASK:-16}"
echo "LOBBENCH_NOINS_DONE $STOCK models=${MODELS[*]}"
