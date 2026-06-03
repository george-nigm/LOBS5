# CLAUDE.md — working notes for `lob_impact` (agent-facing)

This file is for Claude: decisions, conventions, file map, gaps, and the working agreement.
The human-facing doc is `README.md` — keep README about *what it is + how to run*; keep
agent reasoning, TODOs, and conventions HERE.

## Working agreement
- **Step by step.** Build/verify one action at a time. Do NOT move to the next action until the
  current one actually generates its output and the user has eyeballed it. (User: "пока по каждому
  не сгенерируем — дальше не идём".)
- Data + checkpoints live on the old `s5e` project and will be copied into this `u6gb` checkout
  later. Everything is parameterized via env (`DATA_MOUNT`, `PROJECT_DIR`, `CKPT_BASE`, `SAVE_BASE`).
- No push, no merge. Commit in focused steps, scoped to `lob_impact`.

## Output / logging convention (apply to every new script & launcher)
- Each action folder has **`results/`** (outputs) and **`logs/`** (run logs); both git-ignored
  except `.gitkeep` (see `.gitignore`).
- Every run is **timestamped** `YYYYMMDD-HHMMSS`. Outputs → `<action>/results/<name>_<ts>/…`;
  the full run log → `<action>/logs/<name>_<ts>.log`.
- Each action's **launcher `run_*.sh` lives in its own action folder**. Pattern: compute `RUN_TS`,
  make `results/<name>_$RUN_TS` + `logs/`, run python, `tee` everything to the log.
- **Shard mounting**: actions 1/2 launchers are SLURM scripts that self-mount one month shard via
  `squashfuse_ll "$SRC/$SHARD" "$MNT"` (node-local `$TMPDIR`, trap-unmount on exit), default
  `SHARD=shard_2026-01.squashfs` from `/lus/.../public/s5e/quant_team/lob_preproc_sp500_squashfs`
  (recovered from the user's original `submit_sp500_msgs_btw.sbatch`). Action 3 still takes a
  pre-set `DATA_MOUNT` — give it the same self-mount when wiring its cluster job.

## Structure rule (load-bearing)
- **Only `core/` is importable** (`lob_impact.core`) — digit-prefixed folders (`1_…`–`5_…`) can't be
  Python packages. They hold scripts run as files (`python -u path/x.py`) or notebooks. Scenario
  scripts run as files and `import lob_impact.core.*` (legal — `core/` is non-digit).
- `scenarios → 3_scenarios` was depth-preserving: each scenario's `parent_folder_path =
  dirname(dirname(script_dir))` still resolves to the LOBS5 repo root, so sys.path/imports are
  unchanged. **Do not rename** `1.aggressive_scenario_s5.py` / `…_s5_v3.py` (v3 loads its sibling by filename).

## The 5 actions & file map
1. **`1_data_prep/`** — `compute_sp500_msgs_btw.py` (msgs_between per ticker, `--mnt`/`--out_dir`),
   `postprocess_sp500.py` (day-mean table + 3-panel histogram; dir via `argv[1]`).
   Launcher: `run_msgs_btw.sh`. Goal: pick 3 stocks (EA/NVDA/AMD).
2. **`2_daily_stats/`** — `compute_daily_stats.py` (NEW: daily H/L + execution_sum per stock →
   `daily_h_l_<STOCK>.csv` for Parkinson σ; `--per-day` default / `--aggregate`).
   `compute_depth_stats_outdated.py` — NOT used: order_volume is now chosen from historical market
   orders, not book-depth percentiles (kept for reference; PICKLE_BASE points at the old s5e path).
   Launcher: `run_daily_stats.sh`.
3. **`3_scenarios/`** — 8 scenario scripts + two shape configs + `run_experiments.sh`. THE core.
4. **`4_diagnostics/`** — master curve / book update / participation + interactive notebook (to build).
5. **`5_analysis/{beta,decay}/`** — beta (Shape I) and decay (Shape II) analysis (to build).
- **`core/`** — `inference_w_insertions.py` (impact generation), `_cgan_mocks.py` (CGAN-only ABIDES
  shim: mocks side-effecty modules in `sys.modules` before importing ganmodels/gan_utils).

## Scenario shapes (Action 3)
- `config_bet_composition.yaml` = Shape I `(num_insertions=100, num_coolings=0)` → Beta.
- `config_beta_decay.yaml` = Shape II `(num_insertions=10, num_coolings=100)` → Decay.
- User varies only **stock, mb (`--n_gen_msgs`), model**. Scripts CLI-override ONLY
  `--n_gen_msgs` and `--direction`; everything else comes from the rendered YAML.
- `run_experiments.sh`: env-anchored, self-mounts the shard (squashfuse_ll), `STOCKS=(EA NVDA AMD)`,
  both shapes × buy/sell × `MB_VALUES`, `smoke`(n_samples=64, 1 combo, num_insertions=3)/`full`. Per-run
  config rendered by an inline `python3` yaml-merge. Folder naming: `<stock>-<model>-<beta|relaxation>`.
  Activates conda `lobs5` (jax). CPU models (no ckpt) → `JAX_PLATFORMS=cpu`; neural → GPU.

## Mamba3 integration (DONE, validated on GPU)
The new data is L10/26-tok; OLD S5 (L500/24-tok) is incompatible, so we run the NEW leaderboard models.
- **New codebase** (same project, newer): `/lus/.../quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3`
  — has `s5/mamba3*.py`, `s5/gdn*.py`, `lob/encoding_26tok.py`, new `init_train`/`inference_no_errcorr`.
  We DON'T copy it; the mamba3 scenario `sys.path.insert(0, EXP)` so `lob.*`/`s5.*` resolve there
  (our old scenarios keep using our repo-root lob/s5).
- `3_scenarios/1.aggressive_scenario_mamba3.py`: loads Mamba3 via new `init_train` (ssm_type from
  metadata), **`MAMBA3_LEGACY_NORM=1` (REQUIRED for j3417629 — else garbage)**, `TOKEN_MODE=26tok`,
  `sys.modules['lob.encoding']=encoding_26tok`, mamba3 hidden carry recipe (from `sample_new`).
- `core/inference_w_insertions_mamba3.py`: NEW `inference_no_errcorr.py` + the insertion graft
  (12-tuple `_generate_msg`, `insertion_schedule`+`chunk_size` through scannable/generate/generate_batched).
- Checkpoint `exp_R1_Mamba3/checkpoints/j3417629_pw8u0edj_3417629 @46050` (Mamba3-78M, 26-tok, book_dim 503,
  merging=padded). `load_checkpoint(partial_restore=True)` falls back past the muon optimizer-state mismatch.
- Run: `sbatch --gres=gpu:1 lob_impact/3_scenarios/run_experiments.sh smoke mamba3`. Validated: generates
  a sane evolving L10 book with real EA prices showing buy-side price impact (~72s for the smoke).
- GDN is similarly available (`s5/gdn.py`) — same pattern, different `ssm_type`, when wanted.

## Known gaps / TODO (carry forward)
- **Models = placeholders** in `run_experiments.sh`. Verified-resolving checkpoints (new LUS base
  `…/public/s5e/quant_team`): S5-150M `exp_H1-scaling-law/…/j2514440_bkotgtm5_2514440` @135458,
  S5-4K `exp_H2-context-scale/…/j2504167_y0c4j6l3_2504167` @100378 (book_dim 503). LobS5/S5-360M
  old exps are gone at the new base — need current job IDs.
- **L10 vs L500**: squashfs proc data is L10 (orderbook 43 cols); S5 configs use `book_dim=503`
  (wide L500). Resolve the transform before a real run; smoke surfaces it fast.
- **`compute_daily_stats.py`** RESOLVED: proc `.npy` columns are event=1, **price(abs)=3** (col 4 is
  price-relative-to-mid), size=5. Daily H/L from **executions only** (event_type==4) — the only
  definition giving sane Parkinson σ (NVDA 1.4%, AMD 2.5%); all-orders methods are garbage on proc
  data (far resting limit orders → H/L 18-40×). EA is genuinely low-vol (~0.12%).
- **`compute_depth_stats.py` `PICKLE_BASE`** still points at the old `s5e/lob_pipeline` path — fix
  when wiring it; it reads Action-3 pickles, so it's really post-Action-3 calibration (not Action 2).
- **`tick_size`** defaults to 100 (GOOG); set per-stock `STOCK_TICK` in the launcher if EA/NVDA/AMD differ.
- **CST** needs `cst.py`/`param_estimation.py` vendored into `lob_bench/cst_model/` (currently empty).
- **Analysis (4/5) rewrite**: extract metric functions from `5_analysis/run_300_analyze_one.py`
  (`compute_global_beta`, `bootstrap_beta`, `compute_master_curve`, `compute_relaxation_ratio`,
  `fit_decay`, `compute_hurst_dfa`, `compute_propagator`, `compute_kyle_lambda`, `stability_vote`)
  into `core/impact_metrics.py`; then `run_beta_report.py`'s `import lob_impact.analysis.*` becomes
  `lob_impact.core.impact_metrics`, and split beta/decay.

## Dependency boundary (provided by parent LOBS5, not this submodule)
`lob/` (model/tokenizer/inference/checkpointing), `preproc.py`, `s5/`, `Alphatrade/gymnax_exchange`
(symlinked at `LOBS5/Alphatrade`). Run from the repo root.
