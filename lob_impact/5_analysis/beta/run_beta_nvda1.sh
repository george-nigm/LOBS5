#!/bin/bash
#SBATCH --job-name=lobimp_beta_NVDA1
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/beta_NVDA1_%j.out
#
# NVDA beta wave 1: the 7 models complete by 18:30 (CPU panel + Mamba3). Wave 2
# (GDN/S5_120M/Mamba3_4k/S5_4k) runs when their slices land — same script, MODELS override.
set -uo pipefail
IMPACT_DIR=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2
DAILY="$IMPACT_DIR/2_daily_stats/results/daily_20260708-131449/daily_h_l_all.csv"
M="${MODELS:-Historic,Heuristic,Propagator,Hawkes,CST,NMZI,Mamba3}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
cd "$IMPACT_DIR"
echo ">>> NVDA binned ($M)"
python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock NVDA --models "$M" || true
python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock NVDA --models "$M" --method none || true
for name in ${M//,/ }; do
  e="NVDA-${name}-beta"
  echo ">>> 5method $e";      python3.11 5_analysis/beta/beta_5method.py      --grid "$GRID" --daily "$DAILY" --exp "$e" || true
  echo ">>> grid $e";         python3.11 5_analysis/beta/beta_grid.py         --grid "$GRID" --daily "$DAILY" --exp "$e" || true
  echo ">>> master-beta $e";  python3.11 5_analysis/beta/beta_master_curve.py --grid "$GRID" --daily "$DAILY" --exp "$e" || true
done
echo "[$(date)] beta NVDA wave done ($M)"
