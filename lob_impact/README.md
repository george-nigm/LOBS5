# LOB-Impact

Market-impact evaluation for generative LOB models — a submodule of **LOBS5** that runs alongside
[`lob_bench`](../lob_bench). Where LOB-Bench measures *distributional* realism, LOB-Impact measures
*response* realism: inject an aggressive metaorder into a model's generated order book and check the
reaction against known microstructure laws (square-root impact `β ≈ 0.5`, decay toward a `2/3`
permanent level, Kyle's λ, propagator, Hurst). See `FRAMEWORK.md` for the full methodology, and the
two reference images (`1. FRAMEWORK-*.jpg`, `2. models-leaderboard.jpeg`).

## Pipeline — 5 actions

```
lob_impact/
├── core/            # the ONLY importable package (lob_impact.core): shared impact/model logic
├── 1_data_prep/     # Action 1: msgs_between over the S&P500 universe → choose stocks
├── 2_daily_stats/   # Action 2: per-day / aggregated daily stats (H/L, exec-vol, depth) for normalization
├── 3_scenarios/     # Action 3: scenario generation + experiment launcher          ← runnable now
├── 4_diagnostics/   # Action 4: master curves / book updates / participation + interactive notebook
└── 5_analysis/      # Action 5: beta/  and  decay/
```

| # | Action | Run | Output |
|---|--------|-----|--------|
| 1 | **Data prep** — `msgs_btw` (η=10%), `trade_frac`, MO-volume over S&P500; pick 3 stocks | `1_data_prep/compute_sp500_msgs_btw.py --mnt <root> --out_dir <d>` then `postprocess_sp500.py` | `msgs_btw_sp500_*.csv` + histograms |
| 2 | **Daily stats** — daily H/L (Parkinson σ), execution volume, depth | `2_daily_stats/compute_daily_stats.py --mnt <root> --stock EA [--aggregate]`; `compute_depth_stats.py` | `daily_h_l_<STOCK>.csv`, depth CSV |
| 3 | **Scenarios** — inject metaorders, generate sequences | `3_scenarios/run_experiments.sh {smoke\|full}` | `save_dir/{data_cond,data_gen}/`, `aggressive_indices.csv` |
| 4 | **Diagnostics** — master curve, book update, participation | `4_diagnostics/` (notebook + `diagnostics.py`, next round) | figures |
| 5 | **Analysis** — `beta/` (Shape I) and `decay/` (Shape II) | `5_analysis/{beta,decay}/` (next round) | β, relaxation→2/3, γ, Hurst, propagator, scorecard |

## Two scenario shapes (Action 3)

The launcher runs both shapes per model×stock; in each config you vary only **stock, mb, model**:

| Config | Shape (i, c) | Feeds |
|--------|-------------|-------|
| `config_bet_composition.yaml` | I = (`num_insertions=100`, `num_coolings=0`) | `5_analysis/beta` |
| `config_beta_decay.yaml` | II = (`num_insertions=10`, `num_coolings=100`) | `5_analysis/decay` |

### Run the experiments (Action 3)

```bash
# 1. mount the target month's squashfs shard (team step) -> $MNT  (one subdir per ticker)
export DATA_MOUNT="$MNT"
export PROJECT_DIR=/home/u6gb/georgenigm.u6gb/LOBS5     # optional (defaults to repo root)
# 2. fill the MODELS array in 3_scenarios/run_experiments.sh (verified S5 paths are in comments)
# 3. smoke first (1 combo, n_samples=64), then full
bash lob_impact/3_scenarios/run_experiments.sh smoke
bash lob_impact/3_scenarios/run_experiments.sh full
```

Results are written under `SAVE_BASE` (default `lob_impact/data/evalsequences/impact_v4/`, on u6gb),
laid out as `<shape>/<model>/<stock>/<dir>/mb<mb>/` so the analysis stages can glob by shape.

## Structure rules

- **Only `core/` is importable** (`lob_impact.core`). Digit-prefixed folders (`1_…`–`5_…`) hold
  scripts run as files (`python -u path/to/x.py`) or notebooks — never `import`ed (a package name
  can't start with a digit). Scenario scripts run as files and `import lob_impact.core.*`, which is
  legal because `core/` is non-digit.
- **Dependency boundary** — provided by the parent **LOBS5** (not this submodule): `lob/` (model,
  tokenizer, inference, checkpointing), `preproc.py`, `s5/`, and the `Alphatrade/gymnax_exchange`
  JAX-LOB simulator (symlinked at `LOBS5/Alphatrade`). Run everything from the repo root.

## Known gaps / TODO

- **Models are placeholders** in `run_experiments.sh`; S5-150M and S5-4K checkpoint paths are in the
  comments (verified to resolve). LobS5 / S5-360M need current job IDs.
- **L10 vs L500 book width** — squashfs proc data is L10 (orderbook 43 cols); S5 configs use
  `book_dim=503` (wide L500). Resolve the transform before a real run (smoke surfaces it fast).
- **`compute_daily_stats.py` price column** — `COL_PRICE` is flagged for first-run verification
  (proc `.npy` reorders columns vs raw LOBSTER); confirm H/L look like real prices.
- **`tick_size`** defaults to 100 (GOOG) — set per-stock `STOCK_TICK` in the launcher if EA/NVDA/AMD differ.
- **CST** scenario needs `cst.py`/`param_estimation.py` vendored into `lob_bench/cst_model/`.
- **Analysis (4/5)** internals are the next rewrite: extract the metric functions from
  `5_analysis/run_300_analyze_one.py` into `core/impact_metrics.py`, then split beta/decay.
