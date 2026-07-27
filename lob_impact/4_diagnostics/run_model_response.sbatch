#!/bin/bash
#SBATCH --job-name=mresp
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/mresp_%j.out
set -eo pipefail
STOCK="${1:?stock}"
set +u; source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh; conda activate lobs5; set -u
cd /home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics
# model_response_curve.py's own --models default omits OW and QR, and this launcher never passed
# --models — so the two kernel baselines could not appear in Figure 7 no matter what the plotting
# script did. Pass the canonical list explicitly.
MODELS="${MODELS:-Historic,Heuristic,Propagator,OW,CST,NMZI,Hawkes,QR,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN}"
python -u model_response_curve.py --grid /lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2 \
  --models "$MODELS" --stock "$STOCK" --m_max 250 --k_max 12 --n_files 300 --out_dir results/mresp_${STOCK}
python -u event_response_v2.py --stock "$STOCK" \
  --model_npz results/mresp_${STOCK}/model_response_${STOCK}.npz \
  --anchor results/empresp_curve_${STOCK}/empresp_curve_${STOCK}.npz \
  --out results/empresp_curve_${STOCK}/event_response_v2_${STOCK}.png
