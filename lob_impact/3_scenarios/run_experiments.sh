#!/bin/bash
#SBATCH --job-name=lobimp_a3
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/3_scenarios/logs/run_%j.out
#
# Action 3 — scenario generation. Self-mounts a month shard and runs the grid.
#   sbatch lob_impact/3_scenarios/run_experiments.sh smoke            # 1 combo, tiny, num_insertions=3
#   sbatch lob_impact/3_scenarios/run_experiments.sh full
#   sbatch lob_impact/3_scenarios/run_experiments.sh smoke s5_4k      # pick model key(s)
# NOTE: neural (S5) models need a GPU — edit the #SBATCH lines to add e.g.
#       --gres=gpu:1 and a GPU partition before submitting s5_* / cgan.
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
HERE="${IMPACT_DIR}/3_scenarios"
REPO_ROOT="$(dirname "$IMPACT_DIR")"
RUN_TS="$(date +%Y%m%d-%H%M%S)"

PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
# U6GB project mirror (our own project, full group access — replaces the s5e world-read dependency)
CKPT_BASE="${CKPT_BASE:-/lus/lfs1aip2/projects/public/u6gb/projects_public_s5e_quant_team_quant/AlphaTrade/experiments}"
SAVE_BASE="${SAVE_BASE:-${HERE}/results/grid}"   # STABLE consolidated root: all jobs/slices merge here
                                                  # -> analysis points at ONE path (results/grid), not per-run dirs
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/Alphatrade:${PYTHONPATH:-}"

# --- per-day calibration (order_volume = daily-median MO = p50, mb per day) ---
# PER_DAY=1 makes the scenario read per_day_params_<STOCK>.csv (Action-1 derived) and use a
# per-day child(order_volume)=p50 + mb=msgs_btw, instead of the hardcoded template order_volume=75.
PER_DAY="${PER_DAY:-}"
PDP_DIR="${PDP_DIR:-${IMPACT_DIR}/1_data_prep/results/per_day_params}"
N_PER_DAY="${N_PER_DAY:-8}"          # samples generated per day (also the batch size) in per-day mode

# --- Python env (jax, lob, model deps) ---  (conda activate scripts aren't set -u safe)
set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

# --- Data: self-mount a month shard unless DATA_MOUNT is already provided ---
# U6GB mirror (flat dir of monthly shards, our project) — was s5e/.../lob_preproc_sp500_squashfs
SRC="${SRC:-/lus/lfs1aip2/projects/public/u6gb/projects_public_s5e_quant_team}"
SHARD="${SHARD:-shard_2026-01.squashfs}"
if [ -z "${DATA_MOUNT:-}" ]; then
  # $TMPDIR=/local/user/<uid> is a per-user-session tmpfs that gets torn down ~19s into the job
  # (taking the squashfuse mount + staged data with it -> empty-glob IndexError). Use a job-private
  # dir under node-global /dev/shm instead (override with LOB_SCRATCH).
  TMPROOT="${LOB_SCRATCH:-/dev/shm/lob_${SLURM_JOB_ID:-$$}}"
  mkdir -p "$TMPROOT"
  # Hold an fd on the node-local scratch root: it is an autofs auto-mount that idle-expires after
  # ~30-40s, and during the long model load nothing touches it -> autofs unmounts it, wiping the
  # squashfuse mount AND staged data under it (-> get_dataset sees an empty dir -> IndexError).
  # NOTE: do NOT append 2>/dev/null to a command-less `exec` — that makes the redirect PERMANENT and
  # silences the whole script's stderr. Keep the fd-hold on its own line.
  exec 9<"$TMPROOT" || echo "[scratch] WARN: could not hold fd on $TMPROOT" >&2
  echo "[scratch] $TMPROOT fstype=$(stat -f -c %T "$TMPROOT" 2>/dev/null) | $(grep -m1 " $TMPROOT " /proc/self/mounts 2>/dev/null)" >&2
  # DIAG: which node-local paths exist + are stable (TMPROOT=tmpfs vanishes mid-job — find a stable one)
  for _p in "$TMPDIR" /tmp /dev/shm /local/scratch "${SLURM_TMPDIR:-}" "/local/user/${SLURM_JOB_ID:-}"; do
    [ -n "$_p" ] && echo "[paths] $_p exists=$([ -d "$_p" ] && echo 1 || echo 0) fstype=$(stat -f -c %T "$_p" 2>/dev/null) writable=$([ -w "$_p" ] && echo 1 || echo 0)" >&2
  done
  DATA_MOUNT="${TMPROOT}/s5e_mnt_${SLURM_JOB_ID:-$$}"
  mkdir -p "$DATA_MOUNT"
  echo "[$(date)] mounting ${SHARD} -> ${DATA_MOUNT}"
  squashfuse_ll "${SRC}/${SHARD}" "$DATA_MOUNT"
  trap 'fusermount -u "$DATA_MOUNT" 2>/dev/null; rm -rf "$TMPROOT" 2>/dev/null' EXIT
  # Fail LOUDLY (not silently with empty data) if the mount didn't populate — retry once.
  if [ -z "$(ls -A "$DATA_MOUNT" 2>/dev/null)" ]; then
    echo "[$(date)] WARN mount empty, retrying squashfuse_ll once" >&2
    fusermount -u "$DATA_MOUNT" 2>/dev/null || true; sleep 3
    squashfuse_ll "${SRC}/${SHARD}" "$DATA_MOUNT"
  fi
  [ -n "$(ls -A "$DATA_MOUNT" 2>/dev/null)" ] || { echo "[$(date)] FATAL: ${SHARD} mount empty at ${DATA_MOUNT}" >&2; exit 3; }
  echo "[$(date)] mounted: $(ls "$DATA_MOUNT" | wc -l) tickers"
fi

# --- Models: "label|script|ckpt_path|checkpoint_step|book_dim" (script relative to 3_scenarios/) ---
# Active models only. The old S5 scenarios (L500/24-tok, incompatible with the new L10/26-tok data)
# live in 3_scenarios/to_implement/ and are not wired here.
declare -A MODELS=(
  [historic]="Historic|historic_scenario.py|||503"
  [mamba3]="Mamba3|mamba3_scenario.py|${CKPT_BASE}/exp_R1_Mamba3/checkpoints/j3417629_pw8u0edj_3417629|46050|503"
)
MODEL_KEYS=(historic mamba3)
read -ra STOCKS <<< "${STOCKS:-EA NVDA AMD}"        # env-overridable: STOCKS="EA" for per-stock jobs
# mb (messages-between) is FIXED per stock = its msgs_btw (eta=10% participation), from 1_data_prep.
# NOT a sweep — one mb per stock so participation rate stays at the target.
declare -A STOCK_MB=([EA]=122 [NVDA]=250 [AMD]=401)
# shape: "name|num_insertions|num_coolings|template|tag"   (tag = folder suffix: beta | relaxation)
SHAPES=(
  "bet_composition|100|0|config_bet_composition.yaml|beta"        # Shape I  -> beta
  "beta_decay|10|100|config_beta_decay.yaml|relaxation"          # Shape II -> decay/relaxation
)
DIRECTIONS=(buy sell)
declare -A STOCK_TICK=()        # e.g. ([EA]=100 [NVDA]=100 [AMD]=100)

MODE="${1:-smoke}"; shift || true
[ "$#" -gt 0 ] && MODEL_KEYS=("$@")

SMOKE_N_INS=""; SMOKE_MB=""
if [ "$MODE" = "smoke" ]; then
  MODEL_KEYS=("${MODEL_KEYS[0]}"); STOCKS=(EA)
  SHAPES=("${SHAPES[0]}"); DIRECTIONS=(buy)
  N_SAMPLES_OVERRIDE=64; SMOKE_N_INS=3; SMOKE_MB=5   # tiny: 3 insertions, mb=5 — just to check
  SAVE_BASE="${HERE}/results/smoke_${RUN_TS}"        # throwaway, not the consolidated grid root
  echo ">>> SMOKE: ${MODEL_KEYS[0]} x EA x bet_composition x buy x mb=5, n_samples=64, num_insertions=3"
elif [ "$MODE" = "full" ]; then
  N_SAMPLES_OVERRIDE="${N_SAMPLES:-}"        # env-overridable; empty => use the config's n_samples
  echo ">>> FULL: ${#MODEL_KEYS[@]} models x ${#STOCKS[@]} stocks x ${#SHAPES[@]} shapes x 2 dir x per-stock mb${N_SAMPLES_OVERRIDE:+ | n_samples=$N_SAMPLES_OVERRIDE}"
else
  echo "usage: $0 {smoke|full} [model_key ...]" >&2; exit 2
fi

# SAMPLE_SLICE="k/N": this job runs only slice k of N (one batch). Total n_samples = N * batch_size;
# the scenario computes the deterministic partition and runs only batch k. N jobs fan out across GPUs
# and merge into the consolidated path (slices are disjoint). Set submit_slices.sh for the fan-out.
SLICE_K="${SLICE_K:-}"
if [ -n "${SAMPLE_SLICE:-}" ]; then
  SLICE_K="${SAMPLE_SLICE%%/*}"; N_SLICES="${SAMPLE_SLICE##*/}"
  N_SAMPLES_OVERRIDE=$(( N_SLICES * ${BATCH_SIZE:-64} ))     # total; only batch SLICE_K runs here
  echo ">>> SLICE ${SLICE_K}/${N_SLICES} | batch_size ${BATCH_SIZE:-64} | total n_samples ${N_SAMPLES_OVERRIDE}"
fi
echo ">>> run ${RUN_TS} | save_base ${SAVE_BASE}"

# JAX on CPU for models without a checkpoint (historic/heuristic); neural models use the GPU.
dir_to_int() { [ "$1" = "buy" ] && echo 0 || echo 1; }

render_config() {
  local tmpl="$1" out="$2"
  TMPL="$tmpl" OUT="$out" STOCK="$STOCK" DATA_DIR="$DATA_DIR" CKPT="$CKPT" \
  CKPT_STEP="$CKPT_STEP" BOOK_DIM="$BOOK_DIM" SAVE_DIR="$SAVE_DIR" \
  N_INS="$N_INS" N_COOL="$N_COOL" TICK="$TICK" N_SAMPLES_OVERRIDE="$N_SAMPLES_OVERRIDE" SLICE_K="$SLICE_K" \
  PER_DAY="$PER_DAY" PDP_DIR="$PDP_DIR" NPD="$N_PER_DAY" \
  python3 - <<'PY'
import os, yaml
cfg = yaml.safe_load(open(os.environ["TMPL"]))
if os.environ.get("SLICE_K", "") != "":
    cfg["sample_slice"] = int(os.environ["SLICE_K"])
cfg["stock"]          = os.environ["STOCK"]
cfg["data_dir"]       = os.environ["DATA_DIR"]
cfg["save_dir"]       = os.environ["SAVE_DIR"]
cfg["ckpt_path"]      = os.environ["CKPT"] or None
cfg["book_dim"]       = int(os.environ["BOOK_DIM"])
cfg["num_insertions"] = int(os.environ["N_INS"])
cfg["num_coolings"]   = int(os.environ["N_COOL"])
if os.environ.get("TICK"):            cfg["tick_size"] = int(os.environ["TICK"])
step = os.environ["CKPT_STEP"]
cfg["checkpoint_step"] = None if step in ("", "null", "None", "PLACEHOLDER") else int(step)
ov = os.environ["N_SAMPLES_OVERRIDE"]
if ov: cfg["n_samples"] = int(ov)
if os.environ.get("PER_DAY"):
    # per-day calibration: child(order_volume)=p50 and mb come from the CSV per day (NOT the
    # hardcoded 75). batch_size == n_samples_per_day so each day = one batch.
    cfg["per_day_params"]    = os.path.join(os.environ["PDP_DIR"], f"per_day_params_{os.environ['STOCK']}.csv")
    cfg["order_volume_mult"] = 1.0
    cfg["n_samples_per_day"] = int(os.environ["NPD"])
    cfg["batch_size"]        = int(os.environ["NPD"])
yaml.safe_dump(cfg, open(os.environ["OUT"], "w"), sort_keys=False)
PY
}

for model_key in "${MODEL_KEYS[@]}"; do
  IFS='|' read -r LABEL SCRIPT CKPT CKPT_STEP BOOK_DIM <<< "${MODELS[$model_key]}"
  abs_script="${HERE}/${SCRIPT}"
  # no checkpoint -> replay/parametric baseline -> force JAX onto CPU
  if [ -z "$CKPT" ]; then export JAX_PLATFORMS=cpu; else unset JAX_PLATFORMS; fi
  for STOCK in "${STOCKS[@]}"; do
    RAW_DIR="${DATA_MOUNT}/${STOCK}"
    # STAGE the stock's .npy from the (fresh) squashfuse mount to node-local NVMe BEFORE python/JAX
    # starts: the squashfuse daemon dies ~40s in under model-load memory pressure, so reading data
    # lazily during the run hits an empty dir (get_dataset IndexError). Copy now while the mount is
    # healthy; the scenario then reads stable local files. (Lustre-safe: node-local glob/cp only.)
    if [ "${STAGE_LOCAL:-1}" = "1" ]; then
      DATA_DIR="${DATA_MOUNT%/*}/staged_${STOCK}"
      mkdir -p "$DATA_DIR"
      cp -n "$RAW_DIR"/*.npy "$DATA_DIR"/ 2>/dev/null || true
      nst=$(ls "$DATA_DIR"/*message*.npy 2>/dev/null | wc -l)
      echo "[stage] $STOCK: $nst message .npy -> $DATA_DIR" >&2
      [ "$nst" -gt 0 ] || { echo "FATAL: staging $STOCK got 0 files (mount died during copy?)" >&2; exit 4; }
    else
      DATA_DIR="$RAW_DIR"
    fi
    TICK="${STOCK_TICK[$STOCK]:-}"
    for shape_row in "${SHAPES[@]}"; do
      IFS='|' read -r SHAPE_NAME N_INS N_COOL TEMPLATE TAG <<< "$shape_row"
      # ONLY_SHAPE=beta|relaxation -> run a single shape (one experiment per job for max parallelism)
      [ -n "${ONLY_SHAPE:-}" ] && [ "$TAG" != "$ONLY_SHAPE" ] && continue
      [ -n "$SMOKE_N_INS" ] && N_INS="$SMOKE_N_INS"
      tmpl_path="${HERE}/${TEMPLATE}"
      for dir in "${DIRECTIONS[@]}"; do
        # ONLY_DIR=buy|sell -> run a single side
        [ -n "${ONLY_DIR:-}" ] && [ "$dir" != "$ONLY_DIR" ] && continue
        dir_int="$(dir_to_int "$dir")"
        mb="${SMOKE_MB:-${STOCK_MB[$STOCK]:-50}}"   # FIXED per-stock msgs_btw (eta=10%), not swept
        scen="${STOCK}-${LABEL}-${TAG}"             # one experiment = stock-model-(beta|relaxation)/dir
        SAVE_DIR="${SAVE_BASE}/${scen}/${dir}"
        cfg_dir="${SAVE_BASE}/_configs/${scen}"
        # cfg must be PER-SLICE: concurrent slice jobs each bake their OWN node-local data_dir into the
        # config; a shared cfg would race and the loser would read the winner's (invalid) mount path.
        cfg_file="${cfg_dir}/cfg_${scen}_${dir}${SLICE_K:+_slice${SLICE_K}}.yaml"
        mkdir -p "$cfg_dir"
        # Consolidation: one experiment per path. On a fresh (non-slice) run, clear it so reruns
        # don't pile up exp_* and so the analysis sees a single clean experiment per (stock,model,shape,dir).
        [ -z "${SAMPLE_SLICE:-}" ] && rm -rf "$SAVE_DIR"
        mkdir -p "$SAVE_DIR"
        render_config "$tmpl_path" "$cfg_file"
        echo "=== ${scen}/${dir} (mb=${mb}) ===  save=${SAVE_DIR}"
        # Verify the squashfuse mount is visible to PYTHON's glob (the warm bash listing right after
        # mount can succeed while python's get_dataset — called tens of seconds later, after model
        # load — sees an empty dir -> IndexError on day_indeces). Probe with python, log bash/python
        # counts + squashfuse liveness, and wait until python sees the .npy files.
        for _try in $(seq 1 10); do
          pyn=$(python -c "import glob;print(len(glob.glob('${DATA_DIR}/*message*.npy')))" 2>/dev/null)
          bn=$(ls "${DATA_DIR}"/*message*.npy 2>/dev/null | wc -l)
          echo "  [mount-probe ${_try}] bash=${bn} python=${pyn:-?} squashfuse=$(pgrep -c squashfuse_ll 2>/dev/null)" >&2
          [ "${pyn:-0}" -gt 0 ] && break
          sleep 3
        done
        # Keepalive: touch the autofs scratch every 5s for the duration so it never idle-expires
        # while python is busy loading the model (belt-and-suspenders alongside the held fd 9).
        ( while :; do echo "[ka $(date +%H:%M:%S)] TMPROOT=$([ -d "${TMPROOT:-/tmp}" ] && echo 1||echo 0) staged=$([ -d "$DATA_DIR" ] && echo 1||echo 0) mount=$([ -d "$DATA_MOUNT" ] && echo 1||echo 0) sqfuse=$(pgrep -c squashfuse_ll 2>/dev/null)" >&2; sleep 3; done ) &
        _KEEPALIVE=$!
        python -u "$abs_script" --config "$cfg_file" --n_gen_msgs "$mb" --direction "$dir_int"
        kill "$_KEEPALIVE" 2>/dev/null || true
      done
    done
  done
done
echo ">>> ${MODE} run complete -> ${SAVE_BASE}"
