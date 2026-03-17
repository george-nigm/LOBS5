#!/bin/bash
# =============================================================================
# Smoke test: run all 9 market impact models with minimal parameters
#
# Verifies that each scenario script launches, loads data/checkpoints,
# and writes output files in the expected format.
#
# Step 0: CST param estimation (sequential, ~1-2 min on 1 day of data)
# Step 1: 9 parallel srun tasks (5 GPU + 4 CPU-only)
#
# Architecture: single sbatch, 9 srun --exclusive tasks in parallel.
# Each GH200 node = 1 GPU + 72 cores; GPU tasks use the local GPU,
# CPU-only tasks (zero, historic, heuristic, cst) ignore it.
#
# Expected runtime: 1-5 min per task, 30 min wall limit.
# Resource cost: 9 nodes × 0.5 h = 4.5 node-hours.
#
# Usage:
#   mkdir -p logs
#   sbatch lob_impact/run_isambard_smoke_test.sh
# =============================================================================
#SBATCH --job-name=smoke_test
#SBATCH --partition=workq
#SBATCH --nodes=9
#SBATCH --ntasks=9
#SBATCH --gres=gpu:5
#SBATCH --time=00:30:00
#SBATCH --time-min=00:10:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

# ── Paths ──
PROJECT_DIR="/home/s5e/georgenigm.s5e/LOBS5_11_march"
CONFIGS="${PROJECT_DIR}/lob_impact/configs_smoke_test"
LOGDIR="${PROJECT_DIR}/logs/smoke_${SLURM_JOB_ID}"
L10_SOURCE="/lus/lfs1aip2/projects/s5e/lob_reconstruction/GOOG/reconstructed_l10"
CST_PARAMS_DIR="${PROJECT_DIR}/data/checkpoints/cst_params"
mkdir -p "$LOGDIR"

echo "=== Smoke test started: $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes:  ${SLURM_JOB_NODELIST}"
echo "Tasks:  9 (5 GPU + 4 CPU-only)"
echo "Logs:   ${LOGDIR}"
echo ""

# ── Common environment setup (embedded in each srun via bash -c) ──
# Use direct Python path (like aramis) to avoid conda activate issues in srun
CONDA_ENV="/home/s5e/satyamaga.s5e/miniforge3/envs/lobs5"
SETUP_ENV="
export PATH='${CONDA_ENV}/bin':\${PATH}
export LD_LIBRARY_PATH='${CONDA_ENV}/lib':\${LD_LIBRARY_PATH:-}
export PYTHONPATH='${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PROJECT_DIR}/lob_bench:${PROJECT_DIR}/abides_worldmodel_offline/abides-markets:\${PYTHONPATH:-}'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
find '${PROJECT_DIR}/lob' '${PROJECT_DIR}/s5' -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
cd '${PROJECT_DIR}'
"

# =============================================================================
# Step 0: CST Parameter Estimation (sequential, on first node)
# Uses 1 day of L10 data for smoke test (~1-2 min)
# =============================================================================
echo "--- Step 0: CST param estimation ---"

CST_DATA_DIR="${CST_PARAMS_DIR}/data_smoke"
CST_PARAMS_FILE="${CST_PARAMS_DIR}/cst_params_GOOG_smoke.pkl"
mkdir -p "$CST_DATA_DIR"

# Create symlinks to Dec 1, 2025 L10 data (before Jan 2026 test period)
find "$CST_DATA_DIR" -type l -delete 2>/dev/null || true
for suffix in message orderbook; do
    src="${L10_SOURCE}/GOOG_2025-12-01_34200000_57600000_${suffix}_10.csv"
    ln -sf "$src" "${CST_DATA_DIR}/"
done

# Run param estimation inline (no srun — runs on first node)
set +u
source /home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
set -u
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/lob_bench:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
cd "${PROJECT_DIR}"

python -u -c "
import sys
sys.path.insert(0, 'lob_bench/cst_model')
sys.path.insert(0, 'lob_bench')
import param_estimation

print('[*] Estimating CST parameters from 1 day (smoke)...')
aggr_params = param_estimation.estimate_from_data_files(
    '${CST_DATA_DIR}',
    save_path='${CST_PARAMS_FILE}',
    tick_size=100,
    num_ticks=500,
    recompute_existing=True,
)
print(f'[*] Done. Keys: {list(aggr_params.keys())}')
print(f'[*] Saved to: ${CST_PARAMS_FILE}')
" 2>&1 | tee "${LOGDIR}/cst_param_est.log"

if [ ! -f "$CST_PARAMS_FILE" ]; then
    echo "FATAL: CST param estimation failed — no params file produced"
    exit 1
fi
echo "--- Step 0 complete ---"
echo ""

# Unset CPU-only JAX override from Step 0 before launching GPU tasks
unset JAX_PLATFORMS

# =============================================================================
# Step 1: Launch all 9 models in parallel
# =============================================================================
# Export LD_LIBRARY_PATH at parent level so srun children inherit it
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "--- Step 1: Launching 9 models ---"

# ── GPU tasks (5): S5 models + CGAN ──

# 0. LobS5 (original S5, 24-tok encoding — same as v3 since checkpoint j2633975 uses 24-tok)
srun --nodes=1 --ntasks=1 --gpus=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [lobs5] START: \$(date) ==='
python -u lob_impact/1.aggressive_scenario_s5_v3.py \
    --config '${CONFIGS}/cfg_lobs5.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/lobs5.log'
echo '=== [lobs5] DONE: \$(date) ==='" &

# 1. S5-150M (scaled model, 24-tok encoding)
srun --nodes=1 --ntasks=1 --gpus=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [s5_150m] START: \$(date) ==='
python -u lob_impact/1.aggressive_scenario_s5_v3.py \
    --config '${CONFIGS}/cfg_s5_150m.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/s5_150m.log'
echo '=== [s5_150m] DONE: \$(date) ==='" &

# 2. S5-4K (4K context model, 24-tok encoding)
srun --nodes=1 --ntasks=1 --gpus=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [s5_4k] START: \$(date) ==='
python -u lob_impact/1.aggressive_scenario_s5_v3.py \
    --config '${CONFIGS}/cfg_s5_4k.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/s5_4k.log'
echo '=== [s5_4k] DONE: \$(date) ==='" &

# 3. S5-360M (360M params, 24-tok encoding)
srun --nodes=1 --ntasks=1 --gpus=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [s5_360m] START: \$(date) ==='
python -u lob_impact/1.aggressive_scenario_s5_v3.py \
    --config '${CONFIGS}/cfg_s5_360m.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/s5_360m.log'
echo '=== [s5_360m] DONE: \$(date) ==='" &

# 4. CGAN (Coletta generative model)
srun --nodes=1 --ntasks=1 --gpus=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [cgan] START: \$(date) ==='
python -u lob_impact/5v2.aggressive_scenario_cgan.py \
    --config '${CONFIGS}/cfg_cgan.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/cgan.log'
echo '=== [cgan] DONE: \$(date) ==='" &

# ── CPU-only tasks (4): baselines ──

# 5. ZeroInsertions (historic replay, 0 aggressive orders)
srun --nodes=1 --ntasks=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [zero] START: \$(date) ==='
python -u lob_impact/2.historic_scenario.py \
    --config '${CONFIGS}/cfg_zero.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/zero.log'
echo '=== [zero] DONE: \$(date) ==='" &

# 6. Historic (historic replay with aggressive insertions)
srun --nodes=1 --ntasks=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [historic] START: \$(date) ==='
python -u lob_impact/2.historic_scenario.py \
    --config '${CONFIGS}/cfg_historic.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/historic.log'
echo '=== [historic] DONE: \$(date) ==='" &

# 7. Heuristic (historic replay + price shift heuristic)
srun --nodes=1 --ntasks=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [heuristic] START: \$(date) ==='
python -u lob_impact/3.heuristic_scenario.py \
    --config '${CONFIGS}/cfg_heuristic.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/heuristic.log'
echo '=== [heuristic] DONE: \$(date) ==='" &

# 8. CST (Cont-Stoikov-Talreja parametric model)
srun --nodes=1 --ntasks=1 --exclusive \
    bash -c "${SETUP_ENV}
echo '=== [cst] START: \$(date) ==='
python -u lob_impact/4.aggressive_scenario_cst.py \
    --config '${CONFIGS}/cfg_cst.yaml' \
    --direction 0 \
    2>&1 | tee '${LOGDIR}/cst.log'
echo '=== [cst] DONE: \$(date) ==='" &

# ── Wait for all tasks ──
echo "All 9 tasks launched, waiting..."
wait
RC=$?

echo ""
echo "=== Smoke test finished: $(date) ==="
echo ""

# ── Summary (disable errexit — summary is best-effort) ──
set +eo pipefail
SMOKE_DIR="${PROJECT_DIR}/data/evalsequences/smoke_test"
MODELS=(LobS5 S5-150M S5-4K S5-360M ZeroInsertions Historic Heuristic CGAN CST)
KEYS=(lobs5 s5_150m s5_4k s5_360m zero historic heuristic cgan cst)

n_pass=0
n_fail=0
for idx in "${!MODELS[@]}"; do
    model="${MODELS[$idx]}"
    key="${KEYS[$idx]}"
    log="${LOGDIR}/${key}.log"
    out_dir="${SMOKE_DIR}/${model}/GOOG"

    # Find the latest experiment folder
    latest_exp=$(ls -td "${out_dir}"/exp_* 2>/dev/null | head -1)

    if [ -n "$latest_exp" ] && [ -f "${latest_exp}/experiment.log" ]; then
        last_line=$(tail -1 "${latest_exp}/experiment.log" 2>/dev/null)
        n_cond=$(ls "${latest_exp}/data_cond/" 2>/dev/null | wc -l)
        n_gen=$(ls "${latest_exp}/data_gen/" 2>/dev/null | wc -l)
        echo "  PASS  ${model} — cond=${n_cond} gen=${n_gen} (${latest_exp##*/})"
        n_pass=$((n_pass + 1))
    elif [ -f "$log" ] && grep -q "Traceback\|Error\|Exception" "$log" 2>/dev/null; then
        err=$(grep -m1 "Error\|Exception" "$log" | head -c 120)
        echo "  FAIL  ${model} — ${err}"
        n_fail=$((n_fail + 1))
    else
        echo "  ????  ${model} — no output found (check ${log})"
        n_fail=$((n_fail + 1))
    fi
done

echo ""
echo "Results: ${n_pass} passed, ${n_fail} failed out of ${#MODELS[@]} models"
echo "Logs: ${LOGDIR}/"
echo "Output: ${SMOKE_DIR}/"

exit $RC
