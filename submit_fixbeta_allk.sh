#!/bin/bash
# Submit fixed-β=0.5 analysis jobs for all stocks (V5r2 + V4)
set -e

cd /home/s5e/georgenigm.s5e/LOBS5_11_march
mkdir -p logs

V5_BASE="/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v5"
V4_BASE="/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v3"

CONDA_CMD="source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh && conda activate lobs5"
WORK_DIR="cd /home/s5e/georgenigm.s5e/LOBS5_11_march"

echo "=== Submitting V5r2 jobs ==="

# V5r2 AAPL
sbatch --job-name="fixb_allk_AAPL" --partition=workq --nodes=1 --ntasks=1 --gres=gpu:0 --time=02:00:00 \
  --output="logs/fixb_allk_AAPL_%j.out" --error="logs/fixb_allk_AAPL_%j.err" \
  --wrap="${CONDA_CMD} && ${WORK_DIR} && python -u lob_impact/fixbeta_allk.py --pickle_base ${V5_BASE}/pickles --stock AAPL --daily_hl lob_impact/daily_h_l_AAPL.csv"

# V5r2 AMZN
sbatch --job-name="fixb_allk_AMZN" --partition=workq --nodes=1 --ntasks=1 --gres=gpu:0 --time=02:00:00 \
  --output="logs/fixb_allk_AMZN_%j.out" --error="logs/fixb_allk_AMZN_%j.err" \
  --wrap="${CONDA_CMD} && ${WORK_DIR} && python -u lob_impact/fixbeta_allk.py --pickle_base ${V5_BASE}/pickles --stock AMZN --daily_hl lob_impact/daily_h_l_AMZN.csv"

echo "=== Submitting V4 jobs ==="

# V4 GOOG
sbatch --job-name="fixb_allk_GOOG" --partition=workq --nodes=1 --ntasks=1 --gres=gpu:0 --time=02:00:00 \
  --output="logs/fixb_allk_GOOG_%j.out" --error="logs/fixb_allk_GOOG_%j.err" \
  --wrap="${CONDA_CMD} && ${WORK_DIR} && python -u lob_impact/fixbeta_allk.py --pickle_base ${V4_BASE}/pickles --stock GOOG --daily_hl lob_impact/daily_h_l_GOOG.csv"

# V4 INTC
sbatch --job-name="fixb_allk_INTC" --partition=workq --nodes=1 --ntasks=1 --gres=gpu:0 --time=02:00:00 \
  --output="logs/fixb_allk_INTC_%j.out" --error="logs/fixb_allk_INTC_%j.err" \
  --wrap="${CONDA_CMD} && ${WORK_DIR} && python -u lob_impact/fixbeta_allk.py --pickle_base ${V4_BASE}/pickles --stock INTC --daily_hl lob_impact/daily_h_l_INTC.csv"

echo "=== All jobs submitted. Check with: squeue -u \$(whoami) ==="
