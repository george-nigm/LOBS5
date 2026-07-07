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
MODELS="${MODELS:-Historic,Heuristic,Hawkes,CST,NMZI,Propagator,Mamba3,Mamba3_4k,S5_4k,GDN}"
PY=/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5/bin/python
export JAX_PLATFORMS=cpu
cd "$B"
for SH in beta relaxation; do
  echo ">>> master curve $SH (grid_v2, gated)"
  $PY master_curve.py --grid "$GRID" --stock EA --shape "$SH" --models "$MODELS" \
      --out "$B/results/master_curve/master_curve_EA_${SH}_v2gated.png"
done
echo "MASTER_V2_DONE"
