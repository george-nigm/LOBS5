#!/bin/bash
#SBATCH --job-name=emprc
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/emprc_%j.out
set -eo pipefail
STOCK="${1:-NVDA}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics
python -u empirical_response_curve.py \
  --data_cond "$(ls -d /lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2/${STOCK}-Mamba3-beta/buy/*data_cond* /lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2/${STOCK}-Mamba3-beta/buy/exp_*/data_cond 2>/dev/null | head -1)" \
  --stock "$STOCK" --n_files 400 --m_max 250 \
  --out_dir results/empresp_curve_${STOCK}
