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
# ONLY_MODELS="CST,Hawkes,NMZI" scores a subset. Two half-size jobs schedule far sooner than one
# 440G job under a priority-1 fairshare (backfill fits small jobs into gaps), and the parametric
# baselines carry ~16k generated files each while the neural runs carry ~64 — splitting along that
# line also balances the memory. TAG keeps concurrent subsets from clobbering each other's stage dir.
ONLY_MODELS="${ONLY_MODELS:-}"
TAG="${TAG:-all}"
STAGE="$OUT/stage_${STOCK}_${TAG}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu

# ---- stage: merge every exp_* of every model into the bench layout
rm -rf "$STAGE"; mkdir -p "$OUT/scores_${STOCK}_${TAG}"
MODELS=()
for d in "$NOINS/$STOCK-"*-beta; do
  [ -d "$d" ] || continue
  M=$(basename "$d"); M=${M#"$STOCK"-}; M=${M%-beta}
  ngen=$(find "$d"/buy/exp_*/data_gen -maxdepth 1 -name '*message*gen*.csv' 2>/dev/null | wc -l)
  if [ "$ngen" -eq 0 ]; then echo "SKIP $M (no generated files yet)"; continue; fi
  if [ -n "$ONLY_MODELS" ] && ! echo ",$ONLY_MODELS," | grep -q ",$M,"; then echo "SKIP $M (not in ONLY_MODELS)"; continue; fi
  T="$STAGE/$M/$STOCK/$PERIOD"
  mkdir -p "$T/data_real" "$T/data_gen" "$T/data_cond"
  for sub in data_real data_gen data_cond; do
    for f in "$d"/buy/exp_*/"$sub"/*.csv; do
      [ -e "$f" ] || continue
      ln -sf "$f" "$T/$sub/$(basename "$f")"
    done
  done
  # noins runs don't write data_real; the cond windows ARE real LOBSTER data on the
  # same stock/month (and share the message_real_id_{i} naming) — use them as the
  # real side of the gen-vs-real distributional comparison.
  if [ -z "$(ls -A "$T/data_real" 2>/dev/null)" ]; then
    # The conditioning ORDERBOOK carries n_cond+1 rows (row 0 = the book BEFORE the first
    # message) while the message file carries n_cond; LOB-Bench requires equal lengths
    # ("Length mismatch: expected axis has 501 elements, new values have 500"). Symlink the
    # messages, but write a trimmed copy of each book dropping row 0, so row i is the book
    # AFTER message i — the LOBSTER convention the scorer assumes.
    for f in "$T/data_cond"/*.csv; do
      [ -e "$f" ] || continue
      b=$(basename "$f")
      case "$b" in
        *orderbook*) tail -n +2 "$(readlink -f "$f")" > "$T/data_real/$b" ;;
        *)           ln -sf "$(readlink -f "$f")" "$T/data_real/$b" ;;
      esac
    done
    echo "REAL=cond-proxy for $M (orderbook rows trimmed by 1 to match message length)"
  fi
  MODELS+=("$M")
  echo "STAGED $M: gen=$ngen real=$(ls "$T/data_real" | wc -l) cond=$(ls "$T/data_cond" | wc -l)"
done
[ "${#MODELS[@]}" -gt 0 ] || { echo "NO MODELS STAGED — abort"; exit 1; }

# ---- score
cd "$BENCH"
echo y | python -u run_bench.py --stock "$STOCK" --time_period "$PERIOD" \
  --data_dir "$STAGE" --save_dir "$OUT/scores_${STOCK}_${TAG}" \
  --model_name "${MODELS[@]}" --n_workers "${SLURM_CPUS_PER_TASK:-16}"
echo "LOBBENCH_NOINS_DONE $STOCK models=${MODELS[*]}"
