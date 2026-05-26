#!/bin/bash
#SBATCH --job-name=stock_stats
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=logs/stock_stats_%j.out
#SBATCH --error=logs/stock_stats_%j.err

PROJECT_DIR="/home/s5e/georgenigm.s5e/LOBS5_11_march"
mkdir -p "${PROJECT_DIR}/logs"
source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
cd "${PROJECT_DIR}"

echo "=== GOOG ===" && python -u lob_impact/compute_stock_stats.py --stock GOOG
echo ""
echo "=== INTC ===" && python -u lob_impact/compute_stock_stats.py --stock INTC
