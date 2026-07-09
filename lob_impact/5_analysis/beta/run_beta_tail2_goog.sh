#!/bin/bash
#SBATCH --job-name=lobimp_betatail2_G
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/betatail2_G_%j.out
#
# GOOG beta part 2: the 4 models that completed after betatail started (CST, NMZI,
# Mamba3_4k, S5_4k) + the FULL 11-model binned pass for the unified headline figure.
set -uo pipefail
IMPACT_DIR=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2
DAILY="$IMPACT_DIR/2_daily_stats/results/daily_20260708-131449/daily_h_l_all.csv"
MALL="Historic,Heuristic,Propagator,Hawkes,CST,NMZI,Mamba3,GDN,S5_120M,Mamba3_4k,S5_4k"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
cd "$IMPACT_DIR"
echo ">>> FULL 11-model binned"
python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock GOOG --models "$MALL" || true
python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock GOOG --models "$MALL" --method none || true
for name in CST NMZI Mamba3_4k S5_4k; do
  e="GOOG-${name}-beta"
  echo ">>> 5method $e";      python3.11 5_analysis/beta/beta_5method.py      --grid "$GRID" --daily "$DAILY" --exp "$e" || true
  echo ">>> grid $e";         python3.11 5_analysis/beta/beta_grid.py         --grid "$GRID" --daily "$DAILY" --exp "$e" || true
  echo ">>> master-beta $e";  python3.11 5_analysis/beta/beta_master_curve.py --grid "$GRID" --daily "$DAILY" --exp "$e" || true
done
echo "[$(date)] betatail2 GOOG done"
