#!/bin/bash
#SBATCH --job-name=lobimp_master
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta/logs/master_v2_%j.out
#
# Paper Fig 5 refresh on grid_v2: master curves (both shapes) with the significance gate.
set -uo pipefail
B=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/5_analysis/beta
GRID="${GRID:-/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2}"
MODELS="${MODELS:-Historic,Heuristic,Propagator,Hawkes,CST,NMZI,S5,Mamba3,Mamba3_4k,S5_4k,S5_120M,GDN}"
STOCK="${STOCK:-EA}"
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
export JAX_PLATFORMS=cpu
cd "$B"
for SH in beta relaxation; do
  echo ">>> master curve $SH (grid_v2, gated, $STOCK)"
  $PY master_curve.py --grid "$GRID" --stock "$STOCK" --shape "$SH" --models "$MODELS" \
      --out "$B/results/master_curve/master_curve_${STOCK}_${SH}_v2gated.png"
done
echo "MASTER_V2_DONE"
