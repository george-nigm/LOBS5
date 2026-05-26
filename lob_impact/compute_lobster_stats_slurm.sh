#!/bin/bash
# =============================================================================
# Compute LOBSTER data statistics for framework calibration (CPU-only, ~10 min).
#
# Reads preprocessed .npy files for GOOG and INTC (20 days each),
# outputs per-day CSV + printed aggregate summary.
#
# Usage:
#   sbatch lob_impact/compute_lobster_stats_slurm.sh
#   sbatch lob_impact/compute_lobster_stats_slurm.sh GOOG
#   sbatch lob_impact/compute_lobster_stats_slurm.sh INTC 10
# =============================================================================
#SBATCH --job-name=lobster_stats
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=logs/lobster_stats_%j.out
#SBATCH --error=logs/lobster_stats_%j.err

STOCK="${1:-}"
N_DAYS="${2:-20}"

PROJECT_DIR="/home/s5e/georgenigm.s5e/LOBS5_11_march"
mkdir -p "${PROJECT_DIR}/logs"

echo "=== LOBSTER Stats: $(date) ==="
echo "Stock: ${STOCK:-ALL}"
echo "N_days: ${N_DAYS}"
echo "RAM:   $(free -g | awk '/Mem:/{print $2}') GB"
echo ""

source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
cd "${PROJECT_DIR}"

if [ -n "${STOCK}" ]; then
    python -u lob_impact/compute_lobster_stats.py --stock "$STOCK" --n_days "$N_DAYS"
else
    python -u lob_impact/compute_lobster_stats.py --n_days "$N_DAYS"
fi

echo ""
echo "=== Done: $(date) ==="
