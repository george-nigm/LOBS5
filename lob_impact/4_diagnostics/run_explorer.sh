#!/usr/bin/env bash
# Launch JupyterLab for interactive_explorer.ipynb with the right env + PYTHONPATH.
# Usage:  bash 4_diagnostics/run_explorer.sh [PORT]
# Then copy the printed SSH-tunnel line to your laptop and open the URL it prints.
set -euo pipefail

PORT="${1:-8899}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../lob_impact/4_diagnostics
PROJECT_DIR="$(cd "$HERE/../.." && pwd)"                       # .../LOBS5  (repo root)

# --- Python env (jupyterlab + plotly + pandas live in conda `lobs5`) ---
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"

# lob/ s5/ Alphatrade resolvable from inside the notebook
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

NODE="$(hostname)"
cat <<EOF

  ── JupyterLab launching on ${NODE}:${PORT} ───────────────────────────────
  On your LAPTOP, open a second terminal and run the tunnel:

      ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@${NODE}

  Then open the http://127.0.0.1:${PORT}/lab?token=... URL printed below.
  ──────────────────────────────────────────────────────────────────────────

EOF

cd "$HERE"
exec jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}" \
     --ServerApp.root_dir="$HERE" --ServerApp.open_browser=False
