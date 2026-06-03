#!/bin/bash
# =============================================================================
# Compute depth-at-best statistics from existing pickles (CPU-only, ~5 min).
#
# Reads ONE model's raw pickle (Historic) per stock, outputs depth percentiles
# to help calibrate order_volume for future experiments.
#
# Usage:
#   sbatch lob_impact/compute_depth_stats_slurm.sh
#   sbatch lob_impact/compute_depth_stats_slurm.sh INTC
#   sbatch lob_impact/compute_depth_stats_slurm.sh GOOG Historic
# =============================================================================
#SBATCH --job-name=depth_stats
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=logs/depth_stats_%j.out
#SBATCH --error=logs/depth_stats_%j.err

STOCK="${1:-GOOG}"
MODEL="${2:-Historic}"

PROJECT_DIR="/home/s5e/georgenigm.s5e/LOBS5_11_march"
mkdir -p "${PROJECT_DIR}/logs"

echo "=== Depth Stats: $(date) ==="
echo "Stock: ${STOCK}"
echo "Model: ${MODEL}"
echo "RAM:   $(free -g | awk '/Mem:/{print $2}') GB"
echo ""

source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
cd "${PROJECT_DIR}"

python -u lob_impact/compute_depth_stats.py \
    --stock "$STOCK" \
    --model "$MODEL" \
    --output "lob_impact/depth_stats_${STOCK}.csv"

echo ""
echo "=== Done: $(date) ==="
