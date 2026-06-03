# LOB-Impact

Market-impact evaluation for generative LOB models. A submodule of **LOBS5** that runs
alongside [`lob_bench`](../lob_bench): where LOB-Bench measures *distributional* realism,
LOB-Impact measures *response* realism — how a model's order book reacts when an aggressive
metaorder is injected, and whether that reaction obeys known market-microstructure laws
(square-root impact `β ≈ 0.5`, decay toward a `2/3` permanent level, Kyle's λ, propagator, Hurst).

See [`FRAMEWORK.md`](FRAMEWORK.md) for the full methodology spec and glossary, and
[`FINDINGS_MARKET_IMPACT.md`](FINDINGS_MARKET_IMPACT.md) for the readable results companion.

## The 5-stage framework → repository layout

```
lob_impact/
├── stage1_stats/   # 1. STOCKS & STATISTICS — pick S&P500 stocks, compute msgs_btw / trade_frac / MO volume
├── scenarios/      # 2. SCENARIOS (i, c, mb) — the reusable core: inject metaorders, measure impact
├── stage3_run/     # 3. RUN MODELS — run_isambard_c10x_v2.sh (writes its own configs at runtime)
├── analysis/       # 4. DIAGNOSE + 5. ANALYSIS — master curve, participation, beta, decay
├── core/           # shared logic imported by scenarios (impact-generation + model glue)
├── FRAMEWORK.md    # methodology spec
└── FINDINGS_MARKET_IMPACT.md
```

| Stage | What it does | Where |
|-------|--------------|-------|
| **1. Stocks & statistics** | S&P500 (Jan-2026, out-of-sample) `msgs_btw` (η=10%), `trade_frac`, MO-volume, liquidity cohorts → choose stocks (EA / NVDA / AMD) | `stage1_stats/compute_sp500_msgs_btw.py`, `postprocess_sp500.py`, `compute_depth_stats.py` |
| **2. Scenarios** | Inject aggressive metaorders into generated sequences and measure price impact. Shapes I=(100,0,mb), II=(10,100,mb) | `scenarios/` (see below) |
| **3. Run models** | Generate eval sequences across model architectures + baselines (SLURM array; the launcher writes per-(model×stock×grid) configs into `configs_isambard/` at runtime) | `stage3_run/run_isambard_c10x_v2.sh` |
| **4. Diagnose** | Visual sanity: master curve, book updates, participation rate | `analysis/run_300_*.py`, `analysis/190.*.ipynb`, `analysis/220.*.ipynb` |
| **5. Analysis** | Beta analysis (β vs k, Kyle λ, bootstrap, interception map) + decay (relaxation ratio → 2/3, propagator, Hurst) + scorecard | `analysis/run_beta_report.py`, `analysis/run_210_analysis.py`, `analysis/210.paper_v3_final.ipynb` |

## Scenarios (Stage 2 — the reusable core)

Each scenario script reads a YAML config and writes eval sequences. They share the impact-generation
logic in `core/`.

| Script | Model | Notes |
|--------|-------|-------|
| `scenarios/1.aggressive_scenario_s5.py` | S5 (neural) | Canonical aggressive scenario; pattern for all neural models |
| `scenarios/1.aggressive_scenario_s5_v3.py` | S5 (24-tok) | Encoding bridge for v3 checkpoints |
| `scenarios/0.null_baseline_s5.py` | S5 | Null/drift counterfactual (no injections) |
| `scenarios/2.historic_scenario.py` | — | Historic replay baseline |
| `scenarios/3.heuristic_scenario.py` | — | Heuristic price-shift baseline |
| `scenarios/4.aggressive_scenario_cst.py` | Stoikov–Talreja | Parametric baseline — **see Known gaps** |
| `scenarios/5v2.aggressive_scenario_cgan.py` | CGAN | Needs `abides_markets` on path |
| `scenarios/6.twap_scenario_s5.py` | S5 | TWAP passive-order variant |

Run a single scenario (from the repo root, so `lob_impact.*` and `lob.*` resolve):

```bash
python lob_impact/scenarios/1.aggressive_scenario_s5.py \
    --config lob_impact/scenarios/1.aggressive_scenario_config.yaml   # the one example config
```

Or launch the full grid (8 models × stocks × grid × direction) on Isambard. The launcher
generates its own YAML configs at runtime — no pre-baked config tree is kept in the repo:

```bash
sbatch lob_impact/stage3_run/run_isambard_c10x_v2.sh
```

## Dependency boundary — what the parent LOBS5 must provide

LOB-Impact owns its impact logic but relies on the parent project for the model and simulator.
Run everything from the **repo root** (the scenarios insert it on `sys.path`), like `lob_bench`.

Provided by **LOBS5** (not part of this submodule):
- `lob/` — model & tokenizer: `encoding` / `encoding_24tok`, `inference`, `inference_no_errcorr`,
  `init_train` (`init_train_state`, `load_checkpoint`, `load_metadata`), `validation_helpers`
- `preproc.py` (repo root), `s5/` (SSM framework)
- `Alphatrade/gymnax_exchange/` — JAX-LOB simulator (`OrderBook`, `JaxOrderBookArrays`, …), a git submodule
- model checkpoints + preprocessed `.npy` data (mounted, gitignored)

Owned **here** (`core/`):
- `core/inference_w_insertions.py` — impact-specific generation with insertion schedules
  (moved out of `lob/` so the submodule owns its own impact logic)
- `core/_cgan_mocks.py` — `sys.modules` glue required before CGAN imports

## Known gaps

- **CST scenario needs vendored modules.** `scenarios/4.aggressive_scenario_cst.py` imports
  `cst` and `param_estimation`, expected at `../lob_bench/cst_model/`. Those files were **not
  transferred** with this copy. To run CST, vendor `cst.py` + `param_estimation.py` (and the
  `params_file` the launchers reference) from the original `lob_bench/cst_model/`. Until then the
  S5 / historic / heuristic / CGAN scenarios are fully runnable; CST is not.
- **Launcher is wired for the original `s5e` account.** `run_isambard_c10x_v2.sh` hardcodes
  `PROJECT_DIR=/home/s5e/georgenigm.s5e/LOBS5_11_march`, `LUS=/lus/lfs1aip2/projects/s5e`
  (data + checkpoints), and a conda env under another user's home. Update these for your
  account/checkout before launching, and ensure `Alphatrade` resolves at `$PROJECT_DIR/Alphatrade`.
