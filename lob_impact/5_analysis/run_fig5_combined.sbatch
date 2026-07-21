#!/bin/bash
#SBATCH --job-name=fig5c
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/logs/fig5c_%j.out
# Pure cache re-render: combined 2x2 Figure 5 with shared bottom legend, all stocks.
set -eo pipefail
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
export JAX_PLATFORMS=cpu
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
FIGDIR=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/paper/Figures
for ST in EA NVDA GOOG; do
  python -u fig5_combined.py --stock "$ST" --copy_to "$FIGDIR"
done
