#!/bin/bash
#SBATCH --job-name=lobimp_betatail_G
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/betatail_G_%j.out
#
# GOOG beta "tail": the methods run_beta.sh could not reach in 4h on GOOG-size data
# (each pass re-reads ~200k long CSVs). Runs binned(+Hawkes now that it's complete),
# binned --method none, then per-experiment 5method/grid/master via --exp so incomplete
# experiments (CST sell, NMZI) don't poison the pass. Rerun full-panel when all models land.
set -uo pipefail
IMPACT_DIR=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact
HERE="$IMPACT_DIR/5_analysis/beta"
GRID=/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2
DAILY="$IMPACT_DIR/2_daily_stats/results/daily_20260708-131449/daily_h_l_all.csv"
M="Historic,Heuristic,Propagator,Hawkes,Mamba3,GDN,S5_120M"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
cd "$IMPACT_DIR"
echo ">>> binned incl Hawkes"
python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock GOOG --models "$M" || true
python3.11 5_analysis/beta/beta_binned.py --grid "$GRID" --daily "$DAILY" --stock GOOG --models "$M" --method none || true
for name in Historic Heuristic Propagator Hawkes Mamba3 GDN S5_120M; do
  e="GOOG-${name}-beta"
  echo ">>> 5method $e";      python3.11 5_analysis/beta/beta_5method.py      --grid "$GRID" --daily "$DAILY" --exp "$e" || true
  echo ">>> grid $e";         python3.11 5_analysis/beta/beta_grid.py         --grid "$GRID" --daily "$DAILY" --exp "$e" || true
  echo ">>> master-beta $e";  python3.11 5_analysis/beta/beta_master_curve.py --grid "$GRID" --daily "$DAILY" --exp "$e" || true
done
echo "[$(date)] betatail GOOG done"
