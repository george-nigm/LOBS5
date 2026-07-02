#!/usr/bin/env bash
#SBATCH --job-name=lobimp_explorer
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/explorer_%j.out
# JupyterLab for interactive_explorer.ipynb on a COMPUTE node (no GPU — pure CSV+plotly).
# Launch:   sbatch lob_impact/4_diagnostics/run_explorer.sbatch
# Then:     tail -f lob_impact/4_diagnostics/logs/explorer_<jobid>.out
#           copy the printed `ssh -N -L ...` line to your LAPTOP, open the http://127.0.0.1 URL.
set -euo pipefail

PORT="${EXPLORER_PORT:-8899}"
HERE="/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics"
PROJECT_DIR="/home/u6gb/georgenigm.u6gb/LOBS5"

# conda activate scripts trip `set -u` (unbound ADDR2LINE in binutils activate.d) → relax around it
set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

NODE="$(hostname)"
LOGIN="login40"
cat <<EOF

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  JupyterLab is starting on compute node:  ${NODE}  (port ${PORT})
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║  1) On your LAPTOP, open a terminal and run this tunnel (keep it open):
  ║
  ║       ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@${LOGIN}
  ║
  ║  2) Open the http://127.0.0.1:${PORT}/lab?token=... URL printed just below.
  ║  3) When done: scancel ${SLURM_JOB_ID}
  ╚══════════════════════════════════════════════════════════════════════════╝

EOF

cd "$HERE"
exec jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}" \
     --ServerApp.root_dir="$HERE" --ServerApp.open_browser=False
