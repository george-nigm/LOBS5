# Unified Market Impact Evaluation Framework

**Version:** 1.0
**Date:** April 10, 2026
**Authors:** George Nigm, Claude Code
**Scope:** Methodology specification for evaluating generative LOB models via price impact experiments.
**Status:** Reference document — no conclusions, only methodology and what to compute.

---

## Glossary

| Symbol | Name | Definition |
|--------|------|-----------|
| I | Impact | Price impact of the metaorder. Primary: VWAP implementation shortfall. `I = \|VWAP − mid_arrival\| / mid_arrival` |
| Q | Metaorder volume | Total shares in the metaorder. `Q = Σ size_k` across all `i` child orders |
| V | Daily volume | Total executed volume on the trading day (from `daily_h_l_*.csv`) |
| σ | Daily volatility | Parkinson estimator: `σ = ln(H/L) / (2√ln2) = ln(H/L) / 1.6651092` |
| β | Scaling exponent | Slope in `log(I) = α + β · log(Q/V)`. Target: β ≈ 0.5 (square-root law) |
| α | Intercept | Level parameter in the log-log regression. Captures absolute impact magnitude |
| η | Participation rate | `η = child / (child + mb × trade_frac × trade_p50)`. Fraction of executed volume attributable to our metaorder |
| φ | Participation rate (alt) | Same concept as η; used interchangeably in earlier versions |
| mb | Messages between | Number of model-generated messages between consecutive child order insertions |
| i | Insertions | Number of child orders (aggressive orders) per metaorder |
| c | Cooling periods | Number of cooling phases (no insertions, model generates freely). V1–V6 used c>0; V7+ use c=0 |
| child | Child order size | Shares per individual aggressive order (one insertion) |
| k | Insertion index | k=1 is the first child order, k=K is the last. 1-indexed |
| K | Total insertions | K = i, the total number of insertions. At k=K, the full metaorder is complete |
| n_cond_msgs | Context length | Number of historical (conditioning) messages fed to the model before generation. Default: 500 |
| n_gen_msgs | Generation count | Same as mb — messages generated between insertions |
| R | Message rate | Messages per second in the conditioning data |
| H, L | Daily high/low | Highest and lowest execution prices in a trading day (from `daily_h_l_*.csv`) |
| λ | Kyle lambda | Per-insertion price impact in ticks per share: `λ_k = \|exec_price_k − mid_before_k\| / (tick × size_k)` |
| γ | Decay exponent | Rate of post-peak impact decay: `I(u) ~ (1+u)^{-γ}` for u > u_peak |
| G(l) | Propagator | Impact memory kernel at lag l: `G(l) = E[Δp_{t+l} · ε_t]` |
| H_dfa | Hurst exponent | Long memory parameter from DFA. Target: H ≈ 0.7 |
| u | Normalized time | `u = message_index / (i × (mb + 1))`. u=1 is end of insertion phase |

---

## Pipeline Diagram

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                    DATA PREPARATION                              │
                    │                                                                  │
                    │  LOBSTER CSVs ──→ compute_daily_high_low.py ──→ daily_h_l_*.csv │
                    │       │                                                          │
                    │       ├──────→ compute_stock_stats.py ──→ lobster_stats_*.csv    │
                    │       │                                                          │
                    │       └──────→ compute_depth_stats.py ──→ depth_stats_*.csv      │
                    └──────────────────────────┬──────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────────────┐
                    │                    CONFIG GENERATION                             │
                    │                                                                  │
                    │  run_isambard_v{N}_*.sh generates YAML configs per               │
                    │  (model × stock × direction × grid_point)                        │
                    └──────────────────────────┬──────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────────────┐
                    │                    EXPERIMENT EXECUTION                          │
                    │                                                                  │
                    │  SLURM array jobs → 1.aggressive_scenario_s5_v3.py               │
                    │                     2.historic_scenario.py                        │
                    │                     3.heuristic_scenario.py                       │
                    │                     4.aggressive_scenario_cst.py                  │
                    │                     5v2.aggressive_scenario_cgan.py               │
                    │                     6.twap_scenario_s5.py                         │
                    │                                                                  │
                    │  Output: CSVs (books, messages, aggressive_indices) per sample    │
                    └──────────────────────────┬──────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────────────┐
                    │                    DATA AGGREGATION                              │
                    │                                                                  │
                    │  run_300_compute.py: CSV → raw pickle (~15 GB per model/stock)   │
                    │  run_300_analyze_one.py: raw pickle → metrics pickle (~1 MB)     │
                    └──────────────────────────┬──────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────────────┐
                    │                    ANALYSIS & FIGURES                            │
                    │                                                                  │
                    │  run_300_figures.py: metrics pickles → 20 publication figures     │
                    │  run_v8_report.py: per-K β × 7 σ estimators                      │
                    │  analyze_volatility_sweep.py: σ method comparison                │
                    │                                                                  │
                    │  Output: summary_statistics.csv + PDF report                     │
                    └─────────────────────────────────────────────────────────────────┘
```

---

## Section 1: Framework Overview

### 1.1 The Square-Root Law

The empirical square-root law (Kyle 1985, Tóth et al. 2011, Bouchaud et al. 2018):

```
I = Y · σ · (Q/V)^β    where β ≈ 0.5
```

In log-log space:

```
log(I) = α + β · log(Q/V)
```

where `α = log(Y) + log(σ)` absorbs the level parameters.

A generative LOB model that produces realistic order flow should yield β close to 0.5. A model with unrealistic microstructure will have β far from 0.5. This provides a single number — β — that captures the quality of the generated price-volume dynamics.

### 1.2 What β Measures

β captures the combined effect of four model properties:

1. **Book depth generation** — Does the model maintain realistic depth at the best levels? Deeper book → more volume needed to move the price → affects β.
2. **Book resilience** — After an aggressive order depletes depth, how quickly does the model restore it? Faster restoration → lower subsequent marginal impact → affects β through mb dependence.
3. **Order flow realism** — Do the model's messages look like real market activity? A realistic mix of limit orders, cancellations, and executions → realistic impact dynamics.
4. **Price discovery** — Does the midprice respond correctly to order flow imbalance? Correct response → β closer to 0.5.

### 1.3 What β Does NOT Measure

- **Absolute impact magnitude**: Two models can have the same β but different α. α captures the level; β captures the scaling.
- **Low participation rate regime**: Our experiments run at η > 80%. We cannot test the linear regime (η < 1%) where β → 1 (Bucci, Lillo, Bouchaud 2019).
- **Multi-day effects**: Each experiment is within one conditioning context (~500 messages ≈ minutes of real time). Cross-day persistence is not captured.
- **Permanent vs temporary decomposition in isolation**: β mixes both components; the no-arb scorecard (§5.5) is needed to separate them.

### 1.4 Scenario Types

| # | Scenario | Script | Model Type | Hidden State | GPU | Key Property |
|---|----------|--------|-----------|-------------|-----|-------------|
| 1 | S5 Neural | `1.aggressive_scenario_s5.py` (v2 enc) | S5 autoregressive | Continuous SSM | Yes | Full neural generation |
| 1v3 | S5 Neural (v3) | `1.aggressive_scenario_s5_v3.py` (v3 enc) | S5 autoregressive | Continuous SSM | Yes | 24-token encoding |
| 2 | Historic | `2.historic_scenario.py` | None (replay) | N/A | No | Ground truth replay |
| 3 | Heuristic | `3.heuristic_scenario.py` | None (replay + shift) | N/A | No | Proportional price shift after injection |
| 4 | CST | `4.aggressive_scenario_cst.py` | Cont-Stoikov-Talreja parametric | N/A | No | Closed-form parametric model |
| 5 | CGAN | `5v2.aggressive_scenario_cgan.py` | Conditional GAN (~2M) | N/A | Yes | GAN-based generation |
| 5r | RWKV | `5v2.aggressive_scenario_rwkv.py` | RWKV (experimental) | Linear attention | Yes | Alternative architecture |
| 6 | TWAP | `6.twap_scenario_s5.py` | S5 (passive LO) | Continuous SSM | Yes | Passive limit order injection |
| 0 | Null Baseline | `0.null_baseline_s5.py` | S5 (no insertions) | Continuous SSM | Yes | Drift measurement |

**Source:** `FINDINGS_MARKET_IMPACT.md` Part I + Part III

---

## Section 2: Data Selection

### 2.1 Stock Selection

**Criteria:** LOBSTER L2 data available, preprocessed `.npy` files, daily high/low/volume computable, depth statistics computable.

**Data location:** `${LUS}/lob_pipeline/data/{STOCK}_jan2026/` where `LUS=/lus/lfs1aip2/projects/s5e`

**Current stock coverage:**

| Stock | daily_h_l | lobster_stats | depth_stats | V3/V4 exps | V5–V7 exps | V8 exps | V9 exps |
|-------|-----------|---------------|-------------|-----------|-----------|---------|---------|
| GOOG | Yes | Yes | Yes | Yes (10 models) | — | — | — |
| INTC | Yes | Yes | Yes | Yes (8 models) | — | — | — |
| AAPL | Yes | — | — | — | Yes (6 models) | Yes | Planned |
| AMZN | Yes | — | — | — | Yes (6 models) | — | — |
| META | Yes | — | — | — | — | Yes | Planned |
| MSFT | Yes | — | — | — | — | Yes | Planned |
| NVDA | Yes | — | — | — | — | Yes | Planned |
| TSLA | Yes | — | — | — | — | Yes | Planned |
| GOOG_2023 | Yes | Yes | — | Legacy only | — | — | — |

**Scripts:**
- `compute_daily_high_low.py` — Daily H/L/V from raw LOBSTER message CSVs
- `compute_stock_stats.py` — Message rate R, execution fraction, trade sizes, depth percentiles
- `compute_depth_stats.py` — Depth-at-best distribution (p25/p50/p75/p95/p99)
- `compute_lobster_stats.py` — Full per-day LOBSTER statistics

### 2.2 Test Day Selection

Each stock's preprocessing yields ~20 trading days per month (Jan 2026). The experiment system maps samples to days via `create_sample_day_map.py`, distributing 2048 samples across available test days. Cross-day β variation captures regime differences (high-vol vs low-vol days).

**Key parameters from daily_h_l CSV:**
- `highest_price` — Daily highest execution price (IQR-filtered)
- `lowest_price` — Daily lowest execution price (IQR-filtered)
- `execution_sum` — Total daily executed volume
- `filename` — LOBSTER source file (contains date)

### 2.3 Generation Capacity

The total number of generated messages per sample is:

```
total_gen = i × mb    (when c=0, as in V7+)
total_gen = (i + c) × mb    (when c > 0, as in V1–V6)
```

This must not exceed the model's generation capacity:
- **ctx=500 models** (LobS5, S5-120M, S5-360M, LobS5-v2): capacity ≈ 500–600 messages
- **ctx=4000 model** (S5-4K): capacity ≈ 4000 messages
- **Replay models** (Historic, Heuristic): capacity = `n_eval_msgs_dataset` (up to 7000)
- **Parametric models** (CST): unlimited (stateless)

### 2.4 Book Depth Analysis

Order volumes must be calibrated to each stock's book depth. The `compute_depth_stats.py` script computes the distribution of depth at the best ask (for buys) or best bid (for sells) from the conditioning data.

**From 122,880 observations per stock (V3/V4 experiments):**

```
         GOOG    INTC
p25       45     275
p50      105     590
p75      166    1110
p95      325    3120
p99      645   10710
```

Volume calibration uses p50/p75/p95 of depth at best per stock, ensuring a range from "sometimes penetrates first level" to "always penetrates."

**Source:** `compute_depth_stats.py`, depth_stats CSVs, `FINDINGS_MARKET_IMPACT.md` Finding 3

---

## Section 3: Volume and Activity Management

### 3.1 Participation Rate (η)

The effective participation rate is:

```
η = child / (child + mb × trade_frac × trade_p50)
```

where:
- `child` = shares per insertion (our child order)
- `mb` = messages between insertions
- `trade_frac` = fraction of messages that are executions (event type 4 or 5)
- `trade_p50` = median trade size in the stock

**Key insight (Finding 4):** In our simulation, η > 80% regardless of mb, because the model generates mostly limit orders and cancellations (only 0.8–2.7% of messages are executions). Our aggressive orders dominate. This does NOT invalidate results — it means we run a stress test of model microstructure.

**Evolution of η across versions:**

| Version | η (target) | η (actual) | child source | mb source |
|---------|-----------|-----------|-------------|----------|
| V1–V2 | Not considered | ~99% | Fixed 75/300/485 | Fixed 50 |
| V3 | Not considered | ~99% | Fixed 75/300/485 | Grid: 5/10/15/20 |
| V4 | Not considered | ~98% | Depth percentiles | Grid: 5/10/15/20 |
| V5–V6 | 10%/50%/90% | ~10%/50%/90% | η-derived (child=2–4) | Fixed 36 |
| V7 | 10%/50%/90% | ~10%/50%/90% | Same as V6 | Fixed 36 |
| V8 | 10% | ~10% | Median MO per stock | Calibrated per stock |
| V9 | 5%/10%/20% | ~5%/10%/20% | Median MO per stock | Calibrated per stock |

### 3.2 Child Order Sizing

**V1–V4 approach (DEPRECATED):** Fixed volumes (75/300/485) or depth percentiles. Problems: INTC 75 shares = p6 of depth (order never eats first level). Volume must scale with liquidity.

**V5–V7 approach (FLAWED):** Child derived from target η with fixed mb=36. Result: child=2–4 shares for AAPL/AMZN. These are unrealistically small — no institutional trader submits 2-share child orders.

**V8+ approach (RECOMMENDED):** Child = median market order size per stock (from historical LOBSTER data). mb is then calibrated to achieve target η.

**Per-stock child order sizes (V8+, Jan 2026 LOBSTER data):**

| Stock | child (shares) | median MO | msg_rate | trade_frac | trade_p50 | price ($) |
|-------|---------------|----------|---------|-----------|----------|---------|
| AAPL | 35 | 35 | 165 | 2.73% | 40 | 243 |
| META | 10 | 10 | 72 | 3.43% | 12 | 617 |
| MSFT | 17 | 17 | 97 | 3.56% | 20 | 430 |
| NVDA | 25 | 25 | 400 | 3.70% | 28 | 140 |
| TSLA | 15 | 15 | 232 | 3.02% | 17 | 390 |
| GOOG | ~36 | 36 | 137 | 0.8% | 24 | 290 |
| INTC | ~99 | 99 | 91 | 2.7% | 112 | 47 |

**Source:** `run_isambard_v8_realistic.sh` STOCK_CHILD, `run_v8_report.py` STOCK_PARAMS, `FINDINGS_MARKET_IMPACT.md` Part VI

### 3.3 mb Calibration

mb is solved from the target participation rate:

```
mb = child × (1/η − 1) / (trade_frac × trade_p50)
```

mb controls **book resilience**, not participation rate (Finding 5):
- Low mb (3–5): Depth barely restored between insertions → large impact → higher β
- High mb (200+): Depth partially restored → smaller marginal impact → lower β

**Per-stock mb values (V8/V9):**

| Stock | mb (η=5%) | mb (η=10%) | mb (η=20%) | total_gen (η=10%) |
|-------|----------|-----------|-----------|-----------------|
| AAPL | 650 | 308 | 137 | 3,080 |
| META | 581 | 275 | 122 | 2,750 |
| MSFT | 532 | 252 | 112 | 2,520 |
| NVDA | 557 | 264 | 117 | 2,640 |
| TSLA | 562 | 266 | 118 | 2,660 |

**Source:** `run_isambard_v9_cross_eta.sh` STOCK_MB_* arrays

### 3.4 Insertion Mechanics

Each metaorder consists of `i` child orders separated by `mb` model-generated messages.

**Insertion phase:** For k=1..i: inject child order → model generates mb messages → next injection.

**Cooling phase (if c > 0):** After all insertions, model generates c × mb additional messages without injections. Used in V1–V6 to observe relaxation. Removed in V7+ because β(K) dynamics at each k provide richer information.

**Evolution of cooling:**

| Version | Cooling | Rationale |
|---------|---------|----------|
| V1 | c = 4×i to 10×i | Original design: long post-injection observation |
| V2 (c10x) | c = 10×i | Standardized cooling, budget constraint `11×i×mb ≤ 500` |
| V3–V4 | c = 10×i | Same as V2, multiple stocks/models |
| V5–V6 | c = 100 | Very long cooling (100 × mb = 3600 messages) |
| V7 | c = 0 | No cooling — per-K dynamics replace post-injection observation |
| V8–V9 | c = 0 | Same as V7 |

### 3.5 Complete Version Grid

| Version | Script | i | c | mb | child | η | Stocks | Models | Key Question |
|---------|--------|---|---|-----|-------|---|--------|--------|-------------|
| V1 | `run_context_experiments.sh` | 3,5 | 12–50 | 5,15,25,50 (fixed) | 75 | ~99% | GOOG | LobS5-v2 | First grid: does mb affect β? |
| V2 (c10x) | `run_context_500_c10x.sh` | 1–9 | 10×i | 5,10,15,20 | 75 | ~99% | GOOG | LobS5-v2 | Standardize cooling |
| V2b (c10x_v2) | `run_context_500_c10x_v2.sh` | 1–9 | 10×i | 5,10,15,20 | 75,300,485 | ~99% | GOOG | LobS5-v2 | Add volume dimension |
| V3 (c10x_v3) | `run_context_500_c10x_v3.sh` | 1–9 | 10×i | 5,10,15,20 | 75,300,485 | ~99% | GOOG | S5-120M, S5-4K, (360M) | v3 encoding (24-tok) checkpoints |
| V4 (c10x_v4) | `run_isambard_c10x_v4.sh` | 1–9 | 10×i | 5,10,15,20 | p50/p75/p95 depth | ~98% | GOOG, INTC | 10 models | Calibrated volumes; full model comparison |
| V4b (beta_grid) | `run_isambard_beta_grid.sh` | 5,10 | 0 | 5,10 | 25–800 | varies | GOOG, INTC | 10 models | Pure volume scaling (fixed i,mb) |
| V5 | `run_isambard_v5_low_phi.sh` | 10 | 100 | 36 | η-derived (2–357) | 10%/50%/90% | AAPL, AMZN | 6 models | First η-calibrated experiments |
| V6 | (same script, SAVE → v6) | 10 | 100 | 36 | η-derived | 10%/50%/90% | AAPL, AMZN | 6 models | Iteration on V5 |
| V7 | `run_isambard_v7_extended.sh` | 30 | 0 | 36 | Same as V6 | 10%/50%/90% | AAPL, AMZN | 6 models | More insertions, no cooling |
| V8 | `run_isambard_v8_realistic.sh` | 10 | 0 | calibrated | median MO | 10% | AAPL,META,MSFT,NVDA,TSLA | 6 models | Realistic child + mb, 5 stocks |
| V9 | `run_isambard_v9_cross_eta.sh` | 10 | 0 | calibrated | median MO | 5%/10%/20% | AAPL,META,MSFT,NVDA,TSLA | 6 models | Cross-η comparison |

---

## Section 4: Regression Methodology

### 4.1 Origin vs Intercept Estimator

**DECISION: Always use the intercept estimator.**

**Origin estimator** (DEPRECATED): `log(I/σ) = β · log(Q/V)` — forced through (0,0).

```
β_origin = Σ(x_i · y_i) / Σ(x_i²)
```

This is mathematically biased when the true model has α ≠ 0:

```
bias = (α − E[log σ]) · E[x] / E[x²]
```

With typical data (α ≈ −9, E[log σ] ≈ −3.5, E[x] ≈ −6.8, E[x²] ≈ 50.4), the bias is **+0.4 to +0.5**. The origin estimator gives β ≈ 0.5 for ALL models — no differentiation. This was proven on 80,000 synthetic data points in `diag_1_synthetic_calibration.py`.

**Intercept estimator** (REQUIRED): `log(I) = α + β · log(Q/V)` — free intercept.

```
β_intercept = (n·Σx_iy_i − Σx_i·Σy_i) / (n·Σx_i² − (Σx_i)²)
```

Unbiased regardless of α. This is standard OLS.

**Ratio estimator** (informational only): `β_ratio = mean(log(I/σ) / log(Q/V))`.

All three are computed in `run_300_analyze_one.py:compute_global_beta()` and stored, but β = β_intercept is the PRIMARY metric.

**Source:** `FINDINGS_MARKET_IMPACT.md` Finding 1, `diag_1_synthetic_calibration.py`, `plot_origin_bias_proof.py`

### 4.2 Volatility Estimators (7 Methods)

Seven methods for computing daily σ, implemented in `analyze_volatility_sweep.py:compute_sigma_methods()`:

| # | Name | Formula | Notes |
|---|------|---------|-------|
| 1 | `parkinson` | `ln(H/L) / (2√ln2)` | Correct Parkinson (1980). Primary method |
| 2 | `parkinson_2x` | `ln(H/L) / √ln2` | Old code (2× overestimate, for backward compat) |
| 3 | `log_range` | `ln(H/L)` | Raw log range |
| 4 | `no_sigma` | `σ = 1` | No volatility normalization |
| 5 | `constant` | `σ = median(parkinson)` | Same σ for all days |
| 6 | `sqrt_range` | `√ln(H/L)` | Variance proxy |
| 7 | `realized_vol` | Per-sample midprice RV | `std(Δlog(mid))` within each sample |

**Key finding:** β_slope (intercept estimator) is **invariant to σ choice** — changing σ only shifts the regression vertically (changes α), not the slope. R² differs: Parkinson is highest because it captures cross-day variance. `no_sigma` gives identical β but lower R².

**Resolution:** σ choice does not affect β ranking. Report all 7 for transparency. Use Parkinson as primary for R² comparison.

**Source:** `analyze_volatility_sweep.py`, `run_v8_report.py` SIGMA_METHODS, `FINDINGS_MARKET_IMPACT.md` Finding 2 (implicit)

### 4.3 Impact Definitions

Three impact measures computed per sample in `run_300_analyze_one.py:extract_point_cloud()`:

| Name | Column | Formula | When to Use |
|------|--------|---------|------------|
| VWAP (primary) | `I` | `\|VWAP_all_fills − mid_arrival\| / mid_arrival` | Primary metric. child ≥ depth_p25 |
| Midprice | `I_mid` | `\|mid_after − mid_before\| / mid_before` | Always works. Use when VWAP breaks |
| Instantaneous | `I_inst` | `\|exec_price_k − mid_before_k\| / mid_before_k` | Per-insertion Kyle λ analysis |

**VWAP breaks** when child orders are very small (child=2–4 shares in V5–V7): the aggressive order may execute at mid or better, giving VWAP ≈ mid → I ≈ 0 → log(I) undefined.

**Decision:** Use VWAP for child ≥ depth_p25 (the order meaningfully perturbs the book). Use midprice otherwise. Both are always computed and stored.

**Source:** `diag_5_impact_definitions.py`, `run_300_analyze_one.py` lines 177–215

### 4.4 Per-K vs Pooled β

**Per-K regression:** At each k=1..K, fit a separate regression on ~2048 samples. Each sample has:
- `Q_k = k × child` (cumulative volume, identical across samples at same k)
- `V = daily_vol` (varies across ~20 trading days → gives x-axis variation)
- `I_k = cumulative_impact_after_k_insertions`

This gives β(k): how the scaling exponent evolves with metaorder execution progress. Implemented in `run_v8_report.py:analyze_stock()`.

**Pooled regression:** Mix all k=1..K across all samples. More Q variation (from different k values), but mixes different phenomena (early impact ≠ late impact).

**Decision:**
- **Per-K** for dynamics analysis: β(k) trajectory reveals model differentiation (at k=1 all models are identical; divergence grows with k).
- **k=K_max** (last insertion) for the **primary β**: this represents the full metaorder's impact.
- **Pooled** only for backward compatibility or when Q variation from V alone is insufficient.

**Source:** `run_v8_report.py` per-K loop, `FINDINGS_MARKET_IMPACT.md` Finding 2

### 4.5 One Metaorder = One Data Point

**RULE:** Never use per-insertion cross-sectional regression. Each (config, sample, direction) at k=K produces exactly ONE observation: (Q_cumulative, I_cumulative).

Within one simulation run, the trajectory (Q_1, I_1), ..., (Q_K, I_K) is a TIME SERIES, not a cross-section. Points share the same conditioning context, the same model-generated messages, and cumulative Q that grows deterministically. The per-insertion slope measures within-run impact growth, NOT the cross-sectional square-root law.

The empirical square-root law was established on cross-sectional data: thousands of different metaorders from different traders on different days (Tóth et al. 2011: ~500K metaorders). Our approach matches this: each metaorder is an independent sample with its own conditioning context.

**Exception:** Per-K regression (§4.4) uses samples at a single k value. At each k, samples are independent (different conditioning contexts, different days). This is valid cross-sectional analysis.

**Source:** `FINDINGS_MARKET_IMPACT.md` Finding 2, `run_cross_sectional_stratified.py`

---

## Section 5: Metrics Catalog

### 5.1 Primary Metrics

| Metric | Function | Script | Notes |
|--------|----------|--------|-------|
| β (intercept OLS) | `compute_global_beta()` | `run_300_analyze_one.py:246` | Primary. `log(I) = α + β·log(Q/V)` |
| α (intercept) | `compute_global_beta()` | `run_300_analyze_one.py:246` | Level parameter |
| R² (intercept) | `compute_global_beta()` | `run_300_analyze_one.py:246` | Goodness of fit |
| 95% CI (bootstrap) | `bootstrap_beta()` | `run_300_analyze_one.py:309` | 1000 resamples, group-level bootstrap (resample by sample_id) |
| β_origin (legacy) | `compute_global_beta()` | `run_300_analyze_one.py:246` | Through-origin estimator, for comparison only |
| β_ratio | `compute_global_beta()` | `run_300_analyze_one.py:246` | `mean(log(I/σ)/log(Q/V))` |
| n (sample count) | `compute_global_beta()` | `run_300_analyze_one.py:246` | Number of valid data points |

### 5.2 Dynamic Metrics

| Metric | Function | Script | Target | Notes |
|--------|----------|--------|--------|-------|
| Master curve I(u) | `compute_master_curve()` | `run_300_analyze_one.py:389` | Concave, peak at u=1 | (buy−sell)/2, interpolated to 200 u-points |
| Relaxation ratio | `compute_relaxation_ratio()` | `run_300_analyze_one.py:407` | ≈ 2/3 (Bouchaud) | I(u=3)/I(u=1) |
| Stability vote | `stability_vote()` | `run_300_analyze_one.py:425` | > 50% stable | 3-criteria vote: slope, level shift, exp fit |
| Decay exponent γ | `fit_decay()` | `run_300_analyze_one.py:414` | 0.3–1.0 | Power-law fit to post-peak decay |

### 5.3 Per-Insertion Metrics

| Metric | Function | Script | Notes |
|--------|----------|--------|-------|
| Kyle λ(k) | `compute_kyle_lambda()` | `run_300_analyze_one.py:498` | `\|exec_price − mid_before\| / (tick × size)`. **Strongest differentiator** between model families |
| Midprice response I_mid(k) | Extracted in `extract_point_cloud()` | `run_300_analyze_one.py:206` | `\|mid_after − mid_before\| / mid_before` per insertion |
| Spread dynamics | `compute_spread()` | `run_300_analyze_one.py:472` | Spread trajectory normalized by injection count |
| β(K) per σ | `analyze_stock()` | `run_v8_report.py:121` | β at each K=1..10 for each σ estimator |
| Depth at best (k=1) | `compute_depth_at_best_stats()` | `run_300_analyze_one.py:521` | Conditioning book depth distribution |

### 5.4 Memory Metrics

| Metric | Function | Script | Target | Notes |
|--------|----------|--------|--------|-------|
| Hurst exponent H | `compute_hurst_dfa()` | `run_300_analyze_one.py:441` | H ≈ 0.7 | DFA on trade sign series |
| Propagator G(l) | `compute_propagator()` | `run_300_analyze_one.py:456` | G ~ l^{−0.5} | E[Δp_{t+l} · ε_t] |

### 5.5 Decomposition and No-Arb

| Metric | Function | Script | Notes |
|--------|----------|--------|-------|
| β_perm | `b_perm = β × relaxation_ratio` | `run_300_analyze_one.py:740` | Permanent component |
| β_temp | `β − β_perm` | implied | Temporary component |
| No-arb score (5 tests) | Scorecard in `run()` | `run_300_analyze_one.py:741–746` | A: β<1, B: 0.7≤β_perm≤1.3, C: 0.3≤γ≤1.0, D: 0.5≤r≤1.0, E: β≤1/(1+2γ) |

### 5.6 Stratified β

Computed in `run_300_figures.py` figures 10–12 and in `run_cross_sectional_stratified.py`:

| Stratification | Axis | Source | Notes |
|---------------|------|--------|-------|
| β vs mb | Messages between insertions | fig 10 | Resilience dependence |
| β vs vol | Child order volume | fig 11 | Volume scaling |
| β vs day | Trading day | fig 12 | Cross-day stability |
| β vs direction | Buy/sell | `run_cross_sectional_stratified.py` | Asymmetry check |
| β vs i | Number of insertions | `analyze_per_experiment_beta.py` | Grid dependence |

### 5.7 Alternative β Estimates

| Metric | Column | Script | Notes |
|--------|--------|--------|-------|
| β(I_mid) | `beta_I_mid` | `run_300_analyze_one.py:627` | Midprice-based impact |
| β(I_inst) | `beta_I_inst` | `run_300_analyze_one.py:627` | Instantaneous impact |
| β(V_local) | `beta_Vlocal` | `run_300_analyze_one.py:664` | Local volume normalization |
| β(k≥3) | `beta_k3plus` | `run_300_analyze_one.py:681` | Only insertions where models have diverged |
| β_incremental | `beta_incremental` | `run_300_analyze_one.py:697` | Per-insertion (size_k, I_inst) instead of cumulative |

---

## Section 6: Visualization Specification

### 6.1 Standard 20-Figure Catalog

Defined in `run_300_figures.py:FIGURE_CATALOG`:

| Fig | Name | What It Shows | Key Metric | Code |
|-----|------|--------------|-----------|------|
| 0 | Null Baseline Drift | Midprice drift without insertions | drift (ticks) | `run_300_figures.py` |
| 1 | Master Curves | I(u) per config per model | Master curve shape | `run_300_figures.py` |
| 2 | Average Master Curve | Mean I(u) per model | Model ranking by shape | `run_300_figures.py` |
| 3 | Beta Regression Lines | log(I/σ) vs log(Q/V) scatter + OLS fit | β, R² | `run_300_figures.py` |
| 4 | Bootstrap Beta Distributions | Histogram of 1000 bootstrap β | β, 95% CI | `run_300_figures.py` |
| 5 | Relaxation Ratio | I(u=3)/I(u=1) per model | Target: ≈ 2/3 | `run_300_figures.py` |
| 6 | Fraction Stable | Stability vote per model | Target: > 50% | `run_300_figures.py` |
| 7 | Hurst Exponent | DFA-based H per model | Target: H ≈ 0.7 | `run_300_figures.py` |
| 8 | Propagator G(l) | Impact memory kernel | Decay rate | `run_300_figures.py` |
| 9 | Spread Dynamics | Spread trajectory after shock | Recovery time | `run_300_figures.py` |
| 10 | Beta vs mb | β at different message gaps | Resilience dependence | `run_300_figures.py` |
| 11 | Beta vs Volume | β at different child order sizes | Volume scaling | `run_300_figures.py` |
| 12 | Per-Day Beta | β per trading day | Cross-day stability | `run_300_figures.py` |
| 13 | Perm Temp Decomposition | β_perm vs β_temp | Permanent fraction | `run_300_figures.py` |
| 14 | No-Arb Scatter | β × relaxation scorecard | 5-test pass/fail | `run_300_figures.py` |
| 15 | Beta Estimator Comparison | Origin vs Intercept vs Ratio | Estimator bias | `run_300_figures.py` |
| 16 | Per-Insertion Midprice Response | I_mid(k) per model | k-dynamics | `run_300_figures.py` |
| 17 | Kyle Lambda per Insertion | λ(k) per model | **Strongest differentiator** | `run_300_figures.py` |
| 18 | Beta Vlocal Comparison | β with V_local normalization | Local volume effect | `run_300_figures.py` |
| 19 | Conditional Beta Comparison | β(k≥3) vs β(all k) | Model divergence timing | `run_300_figures.py` |

### 6.2 Extended Figures

Generated by specialized analysis scripts:

| Figure Set | Script | Content |
|-----------|--------|---------|
| σ sweep (7 panels) | `analyze_volatility_sweep.py` | β vs η for each σ method, cross-η comparison |
| β(K) per σ | `run_v8_report.py` | Per-K dynamics across 7 σ estimators per stock |
| Origin vs intercept | `run_v8_report.py` | Side-by-side grouped bar chart at K=K_max |
| Scatter per-η | `plot_scatter_per_eta.py` | Point clouds colored by η |
| Scatter β | `plot_scatter_beta.py` | Cross-model scatter comparison |
| Volume comparison | `plot_scatter_vol_comparison.py` | Old vs calibrated volume scatter |
| Origin bias proof | `plot_origin_bias_proof.py` | Synthetic data demonstrating estimator bias |
| β–φ figures | `generate_beta_phi_figures.py` | β vs participation rate curves |
| Cross-stock comparison | `run_v8_report.py` | Grouped bar charts (β per model × stock) |

### 6.3 Scatter Plot Specification

For publication-quality scatter plots (fig 3):

1. **Per-K clouds**: Plot points at each k=1..K with color gradient (lighter for k=1, darker for k=K)
2. **Individual regression**: Thin regression line per k
3. **Primary regression**: Thick regression line at k=K_max only
4. **Annotations**: β, R², n, 95% CI
5. **Reference**: β=0.5 reference line (red dashed)
6. **Max scatter points**: Limited to MAX_SCATTER=800 per panel for readability

### 6.4 Cross-Stock Comparison Plots

Generated by `run_v8_report.py` for each key σ estimator (`parkinson`, `sqrt_range`, `no_sigma`):

- **Grouped bar chart**: β per model × stock, origin vs intercept side-by-side
- **Layout**: 1 row × 2 columns (origin | intercept), x-axis = models, grouped bars = stocks

---

## Section 7: Existing Data Inventory

### 7.1 Experiment Data on Lustre

Base path: `/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/`

| Directory | Version | Stocks | Models | Grid Size | Pickle Status | Data Files |
|-----------|---------|--------|--------|-----------|--------------|-----------|
| `aggressive_scenario_v3/` | V3 (c10x_v3) | GOOG | S5-120M, S5-4K, S5-360M | 10×3×2=60 per model | Yes (GOOG) | 20 pickles |
| `aggressive_scenario_v4/` | V4 (c10x_v4) | GOOG, INTC | 10 models | 10×3×2=60 per model | Yes (both) | 20 pickles |
| `aggressive_scenario_v5/` | V5 | AAPL, AMZN | 6 models | 3η×2dir=6 per model | Yes | 20 pickles |
| `aggressive_scenario_v6/` | V6 | AAPL, AMZN | 6 models | 3η×2dir=6 per model | Yes | 12 pickles |
| `aggressive_scenario_v7/` | V7 | AAPL, AMZN | 6 models | 3η×2dir=6 per model | Yes | 12 pickles |
| `aggressive_scenario_v8/` | V8 | AAPL,META,MSFT,NVDA,TSLA | 6 models | 1η×2dir per model/stock | Yes | 26 pickles |
| `aggressive_scenario_v9/` | V9 | (planned) | 6 models | 2η×2dir per model/stock | Not yet | 0 |
| `beta_grid_v1/` | beta_grid | GOOG, INTC | 10 models | 2×6×2=24 per model | No | 0 (27 GB raw) |
| `beta_test_i7_mb50/` | beta_test | GOOG | subset | fixed i=7,mb=50 | No | 0 (19 GB raw) |
| `mb_sweep_tsla/` | mb_sweep | TSLA | subset | mb sweep | No | 0 |
| `null_baseline/` | null | GOOG | ZeroInsertions | — | No | 0 |
| `smoke_test/` | test | — | — | — | No | 0 |
| `twap_smoke/` | TWAP | — | — | — | No | 0 |

### 7.2 Precomputed Statistics

| File Pattern | Source Script | Stocks Available | Content |
|-------------|-------------|-----------------|---------|
| `daily_h_l_{STOCK}.csv` | `compute_daily_high_low.py` | GOOG, INTC, AAPL, AMZN, META, MSFT, NVDA, TSLA + others (11 total) | Per-day H, L, V |
| `lobster_stats_{STOCK}.csv` | `compute_lobster_stats.py` | GOOG, INTC + up to 9 total | Per-day message rate, exec fraction, trade sizes |
| `depth_stats_{STOCK}.csv` | `compute_depth_stats.py` | GOOG, INTC | Depth-at-best percentiles |

### 7.3 What Can Be Analyzed Now

These datasets have pickles ready — just run the analysis pipeline:

| Dataset | Stocks | Models | Analysis Script | Output Dir |
|---------|--------|--------|----------------|-----------|
| V3/V4 | GOOG, INTC | All 10 (INTC: 8, no CGAN) | `run_300_figures.py` | `pics_for_300_GOOG/`, `pics_for_v4_300_GOOG/`, `pics_for_v4_300_INTC/` |
| V5/V6 | AAPL, AMZN | Historic, Heuristic, CST, S5-120M, S5-360M, S5-4K | `run_v5_report.py` | `pics_for_v5_report/` |
| V7 | AAPL, AMZN | Same 6 models | `analyze_volatility_sweep.py` | `pics_for_vol_sweep/` |
| V8 | AAPL, META, MSFT, NVDA, TSLA | Same 6 models | `run_v8_report.py` | `pics_for_v8_report/` |

### 7.4 What Needs New Experiments or Computation

| Gap | What's Missing | Action | Priority |
|-----|---------------|--------|---------|
| V9 experiments | η=5% and η=20% data for 5 stocks | Run `run_isambard_v9_cross_eta.sh submit` | High |
| depth_stats expansion | Only GOOG/INTC have depth_stats | Run `compute_depth_stats.py` for AAPL,AMZN,META,MSFT,NVDA,TSLA | Medium |
| lobster_stats expansion | Only GOOG/INTC fully computed | Run `compute_lobster_stats.py` for remaining stocks | Medium |
| CST params (missing) | No CST params for META, MSFT, NVDA, TSLA | Run `run_cst_param_estimation.sh` | Low (CST is a baseline) |
| beta_grid pickles | 27 GB raw data, no pickles | Run `run_300_compute.py --stock GOOG` on beta_grid data | Low |
| TWAP experiments | Only smoke test exists | Run `run_isambard_twap_smoke.sh` (full) | Future |
| RWKV integration | Script exists, no production runs | `5v2.aggressive_scenario_rwkv.py` ready | Future |

---

## Section 8: Unified Run Specification

### 8.1 One-Shot Launch (Experiment Generation)

**Entry point:** `run_isambard_v{N}_*.sh submit`

**What it does:**
1. Generates YAML configs per (model × stock × direction × grid_point)
2. Submits SLURM array jobs (one per model × stock)
3. Each array task runs one experiment (one direction)

**Interface:**

```bash
# Full grid
bash lob_impact/run_isambard_v8_realistic.sh submit

# Single stock
bash lob_impact/run_isambard_v8_realistic.sh submit --stock AAPL

# Single model + stock
bash lob_impact/run_isambard_v8_realistic.sh submit s5_360m --stock NVDA

# Smoke test (n_samples=50)
bash lob_impact/run_isambard_v8_realistic.sh submit --smoke
```

**YAML config contract (input):**

```yaml
# Scenario parameters
n_gen_msgs: <mb>                    # messages between insertions
num_insertions: <i>                 # child orders per metaorder
num_coolings: <c>                   # cooling periods (0 for V7+)
n_cond_msgs: 500                    # conditioning context length
n_eval_msgs_dataset: 4000           # historical messages for replay models
event_type: 4                       # market order (aggressive)
direction: 0                        # 0=buy, 1=sell (overridden by CLI --direction)
order_volume: <child>               # shares per child order

# Sampling
n_samples: 2048                     # number of independent samples
batch_size: 64                      # batch size for generation
rng_seed: 42                        # reproducibility
chunk_size: 5                       # save every N batches

# Model
stock: "AAPL"                       # stock ticker
data_dir: "/path/to/stock_data"     # preprocessed .npy files
ckpt_path: "/path/to/checkpoint"    # model checkpoint (S5 only)
checkpoint_step: 135458             # checkpoint step (null = latest)
save_dir: "/path/to/output"         # experiment output directory
tick_size: 100                      # price tick size
sample_top_n: -1                    # top-N sampling (-1 = no restriction)
n_vol_series: 500                   # historical series for volatility
book_dim: 503                       # book representation dimension
test_split: 0                       # test split index
```

### 8.2 One-Shot Analysis (Metrics + Figures)

**Pipeline (3 stages):**

```bash
# Stage 1: CSV → raw pickle (run once, ~2h per model/stock)
python lob_impact/run_300_compute.py --model all --stock GOOG

# Stage 2: raw pickle → metrics pickle (run once, ~10min per model)
python lob_impact/run_300_analyze_one.py --model LobS5 --stock GOOG \
    --daily_hl lob_impact/daily_h_l_GOOG.csv

# Stage 3: metrics pickles → figures + summary CSV
python lob_impact/run_300_figures.py --stock GOOG \
    --daily_hl lob_impact/daily_h_l_GOOG.csv
```

**SLURM wrapper:** `run_300_v4_slurm.sh` runs all 3 stages for a stock.

**V8-specific analysis:**

```bash
# Full V8 report with per-K β × 7 σ estimators
python lob_impact/run_v8_report.py --out_dir pics_for_v8_report --stocks AAPL META MSFT NVDA TSLA
```

**Volatility sweep:**

```bash
python lob_impact/analyze_volatility_sweep.py \
    --stock AAPL --pickle_base /path/to/pickles \
    --daily_hl lob_impact/daily_h_l_AAPL.csv --version v8 --out_dir pics_for_vol_sweep
```

### 8.3 Input Contract

| Parameter | Type | Required | Source |
|-----------|------|----------|--------|
| stock | string | Yes | User specification |
| version | string | Yes | Experiment version (v3, v4, v8, ...) |
| pickle_base | path | Yes | Lustre path to raw pickles |
| daily_hl | path | Yes | `daily_h_l_{STOCK}.csv` |
| models | list[str] | Optional | Default: all available in pickle dir |
| sigma_methods | list[str] | Optional | Default: all 7 |
| out_dir | path | Optional | Default: `pics_for_{version}_{STOCK}/` |

### 8.4 Output Contract

| File | Format | Content |
|------|--------|---------|
| `summary_statistics.csv` | CSV | Per-model: β, α, R², CI, n, β_origin, β_ratio, relaxation, stability, Hurst, Kyle λ |
| `{MODEL}.metrics.pkl` | Pickle | All computed metrics (§5) for one model |
| `v{N}_report.pdf` | PDF | All figures (§6.1), multi-page |
| `vol_sweep_{STOCK}_{version}.csv` | CSV | β × σ × η × k_filter grid |
| `v8_report_all.csv` | CSV | Per-K β × σ × model × stock |
| `v8_report.pdf` | PDF | Cross-stock comparison, per-K dynamics |

---

## Section 9: Experiment Version Changelog

### V1: Context Run (December 2024)

**Script:** `run_context_experiments.sh`
**What changed:** First grid exploration of market impact.
**Parameters:** i∈{3,5}, n_cond∈{500,250}, mb∈{5,15,25,50}, dir∈{buy,sell}. Fixed child=75, c=12–50.
**Grid size:** 32 experiments (on flair-node-12, Docker, 8 GPUs).
**What it resolved:** Confirmed that mb affects β. Discovered that β ≈ 0.82 for all models (later attributed to origin estimator bias).
**Data:** Legacy (flair-node-12 local storage, not on Lustre).

### V2: c10x (January 2025)

**Script:** `run_context_500_c10x.sh`
**What changed:** Standardized cooling to c=10×i with budget constraint `11×i×mb ≤ 500`.
**Parameters:** i∈{1–9}, mb∈{5,10,15,20}, child=75. 10 grid points × 2 dir = 20 experiments.
**What it resolved:** More systematic exploration of i × mb space.
**Data:** Legacy (flair-node-12).

### V2b: c10x_v2 — Volume Dimension (February 2025)

**Script:** `run_context_500_c10x_v2.sh`
**What changed:** Added volume dimension: child ∈ {75, 300, 485}.
**Parameters:** Same grid × 3 volumes = 60 runs.
**What it resolved:** Showed that β depends on volume (larger orders → lower β due to persistent depletion).
**Data:** Legacy (flair-node-12).

### V3: c10x_v3 — v3 Encoding Checkpoints (March 2025)

**Script:** `run_context_500_c10x_v3.sh`
**What changed:** Migrated to v3 encoding (24-token). Tested 3 new checkpoints: S5-120M (j2514440), S5-4K (j2504167), S5-360M (j2504227→j2731367).
**Parameters:** Same 60-run grid per checkpoint.
**What it resolved:** Validated that v3 checkpoints produce similar β as v2, confirming encoding change is benign.
**Data:** `aggressive_scenario_v3/` on Lustre (20 pickles for GOOG).

### V4: Calibrated Volumes + Full Model Comparison (April 2025)

**Script:** `run_isambard_c10x_v4.sh`
**What changed:**
1. Volumes calibrated from depth-at-best: GOOG {105,165,325}, INTC {590,1110,3120}
2. All 10 models tested (Historic, Heuristic, CST, CGAN, LobS5, S5-120M, S5-4K, S5-360M, LobS5-v2, ZeroInsertions)
3. Two stocks: GOOG and INTC
4. Migrated to Isambard (SLURM)
**Parameters:** 10 grid points × 3 volumes × 2 directions = 60 tasks per model/stock.
**What it resolved:**
- Found origin estimator bias (Finding 1)
- Discovered per-metaorder approach (Finding 2)
- Calibrated volumes fix INTC (Finding 3)
- Achieved model differentiation: S5 β≈0.46, CST β≈0.35, CGAN β≈0.28
**Data:** `aggressive_scenario_v4/` on Lustre (20 pickles for GOOG+INTC).

### V4b: Beta Grid — Pure Volume Scaling (April 2025)

**Script:** `run_isambard_beta_grid.sh`
**What changed:** Fixed i and mb (i=10,mb=5 or i=5,mb=10), vary only vol ∈ {25,50,100,200,400,800}.
**Purpose:** Test if β is robust to i,mb choice (it should depend only on volume scaling).
**Data:** `beta_grid_v1/` on Lustre (27 GB raw, no pickles yet).

### V5: Low Participation Rate (May–June 2025)

**Script:** `run_isambard_v5_low_phi.sh`
**What changed:**
1. First η-calibrated experiments: child derived from target η∈{10%,50%,90%}
2. Fixed i=10, c=100, mb=36
3. Two new stocks: AAPL, AMZN
**Problems discovered:** Child=2–4 shares is unrealistically small. VWAP breaks for tiny orders.
**Data:** `aggressive_scenario_v5/` on Lustre (20 pickles).

### V6: Iteration on V5 (June 2025)

**Script:** Same as V5 (saves to `aggressive_scenario_v6/`).
**What changed:** Minor parameter tweaks on V5 grid. Added more η values.
**Data:** `aggressive_scenario_v6/` on Lustre (12 pickles).

### V7: Extended Insertions, No Cooling (July 2025)

**Script:** `run_isambard_v7_extended.sh`
**What changed:**
1. i=30 (3× more insertions than V5/V6)
2. c=0 (no cooling — insertion phase only)
3. Total = 30×36 = 1080 messages
**Purpose:** Determine if β(k) plateaus before k=30.
**What it resolved:** Per-K dynamics are more informative than post-injection relaxation. Cooling is unnecessary.
**Data:** `aggressive_scenario_v7/` on Lustre (12 pickles).

### V8: Realistic Participation Rate (August–September 2025)

**Script:** `run_isambard_v8_realistic.sh`
**What changed:**
1. Child = median MO size per stock (from historical data): AAPL=35, META=10, MSFT=17, NVDA=25, TSLA=15
2. mb calibrated for η≈10% per stock: AAPL=308, META=275, MSFT=252, NVDA=264, TSLA=266
3. i=10, c=0
4. Five stocks: AAPL, META, MSFT, NVDA, TSLA
5. Six models: Historic, Heuristic, CST, S5-120M, S5-360M, S5-4K
**What it resolved:** Realistic child order sizes. Cross-stock comparison. Per-K β × 7 σ estimators.
**Data:** `aggressive_scenario_v8/` on Lustre (26 pickles).

### V9: Cross-Participation-Rate (In Progress)

**Script:** `run_isambard_v9_cross_eta.sh`
**What changed:**
1. Same child sizes as V8 (median MO)
2. Three η targets: 5%, 10%, 20% (via mb calibration)
3. η=10% reused from V8
4. η=5% requires n_eval_msgs_dataset=7000
**Purpose:** Does β change with η? Is the square-root law regime-dependent in our simulation?
**Data:** `aggressive_scenario_v9/` on Lustre (0 pickles — not yet run).

---

## Section 10: Model Registry

### 10.1 Production Models (V4+)

| Key | Label | Script | Checkpoint Path | Step | Enc | book_dim | GPU | Description |
|-----|-------|--------|----------------|------|-----|---------|-----|-------------|
| `lobs5` | LobS5 | `1.aggressive_scenario_s5_v3.py` | `exp_J1-sparse-book-anchoring/.../j2633975_gao5ok51_2633975` | 61037 | v3 | 503 | Yes | S5 7M, ctx=500, sparse book anchoring |
| `s5_120m` | S5-120M | `1.aggressive_scenario_s5_v3.py` | `exp_H1-scaling-law/.../j2514440_bkotgtm5_2514440` | 135458 | v3 | 503 | Yes | S5 120M, ctx=500, best KS=0.0759 |
| `s5_4k` | S5-4K | `1.aggressive_scenario_s5_v3.py` | `exp_H2-context-scale/.../j2504167_y0c4j6l3_2504167` | 100378 | v3 | 503 | Yes | S5 ~50M, ctx=4000 |
| `s5_360m` | S5-360M | `1.aggressive_scenario_s5_v3.py` | `exp_J2_muon_optimizer/.../j2731367_u5xps1po_2731367` | 34158 | v3 | 503 | Yes | S5 360M, ctx=500, Muon optimizer, best KS=0.0678 |

### 10.2 Baseline Models

| Key | Label | Script | Checkpoint | GPU | Description |
|-----|-------|--------|-----------|-----|-------------|
| `historic` | Historic | `2.historic_scenario.py` | None | No | Replay historical messages verbatim |
| `heuristic` | Heuristic | `3.heuristic_scenario.py` | None | No | Replay + proportional price shift after injection |
| `cst` | CST | `4.aggressive_scenario_cst.py` | `cst_params_{STOCK}_dec2025.pkl` | No | Cont-Stoikov-Talreja parametric (per-stock params) |
| `zero` | ZeroInsertions | `2.historic_scenario.py` | None | No | Replay with no insertions (drift measurement) |

### 10.3 Legacy / Experimental Models

| Key | Label | Script | Checkpoint | GPU | Notes |
|-----|-------|--------|-----------|-----|-------|
| `lobs5_v2` | LobS5-v2 | `1.aggressive_scenario_s5.py` | `twilight-sound-77_s42sujip` | Yes | S5 7M, v2 encoding (older), only in V3/V4 |
| `cgan` | CGAN | `5v2.aggressive_scenario_cgan.py` | `data/checkpoints/cgan` | Yes | ~2M params. INTC skipped (only 10 epochs). Only in V3/V4 |
| `rwkv` | RWKV | `5v2.aggressive_scenario_rwkv.py` | TBD | Yes | Experimental. No production runs yet |

### 10.4 Model Visualization Properties

Defined in `run_300_figures.py:MODEL_META`:

| Model | Color | Line Style | Marker | Marker Size |
|-------|-------|-----------|--------|------------|
| ZeroInsertions | `#7CAE7A` | `:` | `D` | 5 |
| Historic | `#90939C` | `:` | `x` | 6 |
| Heuristic | `#546884` | `-.` | `d` | 5 |
| CST | `#213552` | `--` | `^` | 5 |
| CGAN | `#7B4F9E` | `(0,(8,4))` | `s` | 5 |
| LobS5 | `#C88A3A` | `-` | `o` | 5 |
| S5-120M | `#D95F02` | `-` | `v` | 5 |
| S5-4K | `#5B7BBF` | `-` | `*` | 7 |
| S5-360M | `#B5446E` | `-` | `H` | 6 |
| LobS5-v2 | `#2CA02C` | `-` | `P` | 6 |

---

## Appendix A: Complete Script Reference

### A.1 Scenario Execution Scripts

| Script | Scenario | Input | Output |
|--------|----------|-------|--------|
| `0.null_baseline_s5.py` | Null baseline (no insertions) | YAML config | Books/messages CSVs |
| `1.aggressive_scenario_s5.py` | S5 aggressive (v2 encoding) | YAML config | Books/messages/aggressive_indices CSVs |
| `1.aggressive_scenario_s5_v3.py` | S5 aggressive (v3 encoding) | YAML config | Books/messages/aggressive_indices CSVs |
| `2.historic_scenario.py` | Historic replay | YAML config | Books/messages/aggressive_indices CSVs |
| `3.heuristic_scenario.py` | Heuristic (replay + shift) | YAML config | Books/messages/aggressive_indices CSVs |
| `4.aggressive_scenario_cst.py` | CST parametric | YAML config + params_file | Books/messages/aggressive_indices CSVs |
| `5v2.aggressive_scenario_cgan.py` | CGAN | YAML config | Books/messages/aggressive_indices CSVs |
| `5v2.aggressive_scenario_rwkv.py` | RWKV (experimental) | YAML config | Books/messages/aggressive_indices CSVs |
| `6.twap_scenario_s5.py` | TWAP passive LO | YAML config | Books/messages CSVs |

### A.2 Experiment Launch Scripts (SLURM)

| Script | Version | Stocks | Notes |
|--------|---------|--------|-------|
| `run_context_experiments.sh` | V1 | GOOG | Docker (flair-node-12) |
| `run_context_500_c10x.sh` | V2 | GOOG | Docker (flair-node-12) |
| `run_context_500_c10x_v2.sh` | V2b | GOOG | +volume dimension |
| `run_context_500_c10x_v3.sh` | V3 | GOOG | v3 encoding checkpoints |
| `run_context_500_c10x_v3_4kctx.sh` | V3 (4K) | GOOG | S5-4K specifically |
| `run_isambard_c10x_v4.sh` | V4 | GOOG, INTC | Calibrated volumes, 10 models |
| `run_isambard_c10x_v4_hii.sh` | V4 ext | GOOG, INTC | High-i extension (i=10,12,15) |
| `run_isambard_beta_grid.sh` | V4b | GOOG, INTC | Pure volume scaling |
| `run_isambard_v5_low_phi.sh` | V5/V6 | AAPL, AMZN | η-calibrated |
| `run_isambard_v7_extended.sh` | V7 | AAPL, AMZN | i=30, c=0 |
| `run_isambard_v8_realistic.sh` | V8 | 5 stocks | Realistic child+mb |
| `run_isambard_v9_cross_eta.sh` | V9 | 5 stocks | Cross-η |
| `run_isambard_null_baseline.sh` | Null | — | ZeroInsertions |
| `run_isambard_smoke_test.sh` | Smoke | — | Quick validation |
| `run_isambard_twap_smoke.sh` | TWAP | — | TWAP agent test |
| `run_isambard_mb_sweep.sh` | Sweep | TSLA | mb sweep |

### A.3 Data Preparation Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `compute_daily_high_low.py` | Daily H/L/V from LOBSTER CSVs | `daily_h_l_{STOCK}.csv` |
| `compute_stock_stats.py` | Message rate, trade sizes, depth, mb recommendation | Console + CSV |
| `compute_depth_stats.py` | Depth-at-best distribution | `depth_stats_{STOCK}.csv` |
| `compute_lobster_stats.py` | Full per-day LOBSTER statistics | `lobster_stats_{STOCK}.csv` |
| `create_sample_day_map.py` | Map samples to trading days | Sample-day mapping |

### A.4 Analysis Pipeline Scripts

| Script | Stage | Input | Output |
|--------|-------|-------|--------|
| `run_300_compute.py` | 1. Aggregation | Experiment CSVs | Raw pickle (~15 GB) |
| `run_300_analyze_one.py` | 2. Metrics | Raw pickle + daily_h_l | Metrics pickle (~1 MB) |
| `run_300_figures.py` | 3. Figures | Metrics pickles | 20 figures + summary CSV |
| `run_300_v4_slurm.sh` | All 3 stages | Stock name | Full analysis |
| `run_300_slurm.sh` | Legacy pipeline | — | — |
| `run_300_figures_slurm.sh` | Stage 3 only | — | Figures |
| `run_300_v4_figures_only.sh` | Stage 3 only (v4) | — | Figures |

### A.5 Specialized Analysis Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `run_v8_report.py` | V8 per-K β × 7 σ estimators | V8 pickles | PDF + CSV |
| `analyze_volatility_sweep.py` | σ method comparison | Pickles + daily_h_l | CSV + PDF |
| `analyze_vol_sweep_fixed_eta.py` | Volume sweep at fixed η | Pickles | CSV |
| `analyze_beta_by_k.py` | β at each k | Pickles | CSV |
| `analyze_beta_fixed_eta.py` | β at fixed η | Pickles | CSV |
| `analyze_beta_per_phi.py` | β per participation rate | Pickles | CSV |
| `analyze_fixed_beta.py` | Fixed-β analysis | Pickles | CSV |
| `analyze_metaorder_beta.py` | Per-metaorder β | Pickles | CSV |
| `analyze_per_experiment_beta.py` | Per-experiment β | Pickles | CSV |
| `analyze_per_k_beta.py` | Per-k β | Pickles | CSV |
| `analyze_realized_eta.py` | Realized η from experiments | Pickles | CSV |
| `analyze_single_config.py` | Single config deep-dive | Pickles | Console |
| `run_v5_filter_and_analyze.py` | V5-specific filtering | V5 pickles | Filtered data |
| `run_v5_report.py` | V5 report | V5 pickles | PDF |
| `run_210_analysis.py` | 9-model grid analysis | Notebook 210 data | Figures |

### A.6 Report Generation Scripts

| Script | Report Type | Output |
|--------|-----------|--------|
| `run_framework_report.py` | Framework overview PDF (21 pages) | `pics_for_framework_report/` |
| `run_beta_report.py` | Comprehensive β PDF (27 pages) | `pics_for_beta_report/` |
| `run_cross_sectional_stratified.py` | Stratified analysis PDF (17 pages) | `pics_for_cross_sectional/` |
| `run_counterfactual_beta.py` | Counterfactual (MarS-style) correction | `pics_for_counterfactual/` |
| `run_beta_test_analysis.py` | Beta test analysis | `pics_for_beta_test_GOOG/` |
| `run_experiment_catalog.py` | Experiment catalog | `pics_for_experiment_catalog/` |
| `run_lifecycle_validation.py` | Lifecycle validation | `pics_for_lifecycle/` |
| `run_master_report.py` | Master report | `pics_for_master_report/` |
| `run_mega_report.py` | Mega report (all analyses) | `pics_for_mega_report/` |
| `run_article_v4.py` | Article figures (V4) | `pics_for_article_v4/` |

### A.7 Diagnostic Scripts

| Script | What It Tests | Evidence |
|--------|-------------|---------|
| `diag_1_synthetic_calibration.py` | Estimator bias on synthetic data | Origin biased +0.5, intercept unbiased |
| `diag_2_data_integrity.py` | Data completeness and validity | Missing configs, corrupt CSVs |
| `diag_3_distribution_analysis.py` | Point cloud distributions | Impact distribution shape |
| `diag_4_normalizations.py` | V normalization methods | Daily V vs V_local |
| `diag_5_impact_definitions.py` | VWAP vs midprice vs instantaneous | When VWAP breaks |
| `diag_6_reproduce_140.py` | Reproduce old notebook 140 | Backward compatibility |
| `diag_7_stratified.py` | Full stratification analysis | β × mb × vol × day |

### A.8 Visualization Scripts

| Script | Plot Type |
|--------|----------|
| `plot_origin_bias_proof.py` | Synthetic proof of estimator bias |
| `plot_scatter_beta.py` | Cross-model β scatter |
| `plot_scatter_per_eta.py` | Scatter colored by η |
| `plot_scatter_vol_comparison.py` | Old vs calibrated volumes |
| `generate_beta_phi_figures.py` | β vs φ curves |
| `generate_article_figures_180.py` | Article-ready figures |
| `220.figures_from_cache.py` | Regenerate figures from cached metrics |

### A.9 Legacy / Notebook-Related Scripts

| Script | Notes |
|--------|-------|
| `170_run.py`, `171_run.py`, `172_run.py` | Legacy notebook runners |
| `_gen_nb120.py`, `_gen_nb121.py`, `_gen_nb130.py`, `_gen_nb200.py` | Notebook generators |
| `_cgan_mocks.py` | CGAN test mocks |
| `heuristic_historical_scenario_run_quantile_fixing.py` | Legacy heuristic variant |
| `migrate_grids.py` | Grid migration between versions |
| `step_4_sample_day_mapping.py` | Legacy sample-day mapping |
| `fixbeta_allk.py` | Fix β computation for all k |
| `run_300_analysis.py` | Legacy predecessor of run_300_analyze_one.py |

### A.10 SLURM Wrapper Scripts

These `.sh` files wrap the corresponding Python analysis scripts with `sbatch` headers for Isambard:

| Wrapper Script | Wraps |
|---------------|-------|
| `compute_daily_hl_from_npy.sh` | `compute_daily_high_low.py` (from .npy) |
| `compute_lobster_stats_slurm.sh` | `compute_lobster_stats.py` |
| `compute_stock_stats_slurm.sh` | `compute_stock_stats.py` |
| `compute_depth_stats_slurm.sh` | `compute_depth_stats.py` |
| `run_beta_by_k.sh` | `analyze_beta_by_k.py` |
| `run_beta_fixed_eta.sh` | `analyze_beta_fixed_eta.py` |
| `run_beta_grid_analysis.sh` | Beta grid analysis pipeline |
| `run_beta_report_slurm.sh` | `run_beta_report.py` |
| `run_beta_test.sh` | `run_beta_test_analysis.py` |
| `run_beta_test_analysis_slurm.sh` | `run_beta_test_analysis.py` |
| `run_counterfactual_slurm.sh` | `run_counterfactual_beta.py` |
| `run_cross_sectional_slurm.sh` | `run_cross_sectional_stratified.py` |
| `run_diagnostics_slurm.sh` | `diag_1` through `diag_7` |
| `run_framework_slurm.sh` | `run_framework_report.py` |
| `run_lifecycle_slurm.sh` | `run_lifecycle_validation.py` |
| `run_metaorder_beta.sh` | `analyze_metaorder_beta.py` |
| `run_per_exp_beta.sh` | `analyze_per_experiment_beta.py` |
| `run_realized_eta.sh` | `analyze_realized_eta.py` |
| `run_scatter_beta.sh` | `plot_scatter_beta.py` |
| `run_scatter_per_eta.sh` | `plot_scatter_per_eta.py` |
| `run_v7_analysis.sh` | V7 analysis pipeline |
| `run_v7_full_pipeline.sh` | V7 compute + analyze + figures |
| `run_v6_full_pipeline.sh` | V6 compute + analyze + figures |
| `run_v8_analysis.sh` | V8 analysis pipeline |
| `run_v8_report_slurm.sh` | `run_v8_report.py` |
| `run_vol_sweep.sh` | `analyze_volatility_sweep.py` |
| `run_vol_sweep_eta9.sh` | Volatility sweep at η=9% |
| `run_vol_sweep_v8.sh` | Volatility sweep for V8 data |
| `run_vol_comparison.sh` | `plot_scatter_vol_comparison.py` |
| `run_cst_param_estimation.sh` | CST parameter estimation |

### A.11 Multi-Model and Variant Launch Scripts

| Script | Purpose |
|--------|---------|
| `run_all_5models_v3.sh` | Launch V3 experiments for all 5 S5 models |
| `run_all_5models_v4.sh` | Launch V4 experiments for all 5 S5 models |
| `run_baselines_c10x_v2.sh` | Launch V2 experiments for baseline models |
| `run_cgan_c10x_v2_fast.sh` | Fast CGAN V2 experiments |
| `run_cst_test.sh` | CST model testing |
| `run_cst_workers.sh` | CST worker process management |
| `run_rwkv_c10x_v2.sh` | RWKV V2 experiments |
| `run_rwkv_v3.sh` | RWKV V3 experiments |

---

## Appendix B: Stock Statistics Reference

### B.1 GOOG and INTC (from FINDINGS, Jan 2026)

```
                          GOOG        INTC
─────────────────────────────────────────────
Message rate (msg/sec)     137          91
Execution fraction         0.8%        2.7%
Avg trade size (shares)     36         112
Median trade size            24          99
Daily volume (shares)    264,430    689,290
Depth at best p50           105         590
Depth at best p95           325       3,120
Parkinson σ (median)      0.028       0.022
Price (approx)            $290         $47
Spread (ticks, median)       5           1
─────────────────────────────────────────────
```

### B.2 V8 Stocks (from run_v8_report.py, Jan 2026)

```
Stock  child   mb   sec/ins  Q_meta   Q ($)      Q/ADV     η     mkt_vol
─────────────────────────────────────────────────────────────────────────
AAPL     35   308    1.9s     350     $85,050   0.0054%  10.1%   ~34sh
META     10   275    3.8s     100     $61,700   0.0092%  10.5%   ~11sh
MSFT     17   252    2.6s     170     $73,100   0.0062%  10.2%   ~18sh
NVDA     25   264    0.7s     250     $35,000   0.0013%  10.3%   ~27sh
TSLA     15   266    1.1s     150     $58,500   0.0022%  10.2%   ~14sh
─────────────────────────────────────────────────────────────────────────
```

Derived: `sec/ins = mb / msg_rate`, `Q_meta = 10 × child`, `mkt_vol = mb × trade_frac × trade_p50`

### B.3 V9 mb Calibration Table

```
Stock  child   mb(5%)  mb(10%)  mb(20%)  total(5%)  total(10%)  total(20%)
──────────────────────────────────────────────────────────────────────────
AAPL     35     650     308      137       6500       3080        1370
META     10     581     275      122       5810       2750        1220
MSFT     17     532     252      112       5320       2520        1120
NVDA     25     557     264      117       5570       2640        1170
TSLA     15     562     266      118       5620       2660        1180
──────────────────────────────────────────────────────────────────────────

MO rate and median MO (source for mb derivation):
  AAPL: MO_rate=2.92%, q_med=35 → denominator=1.022
  META: MO_rate=2.55%, q_med=14 → denominator=0.357
  MSFT: MO_rate=2.72%, q_med=25 → denominator=0.680
  NVDA: MO_rate=3.24%, q_med=28 → denominator=0.907
  TSLA: MO_rate=2.82%, q_med=20 → denominator=0.564
```

---

## Appendix C: Methodological Decisions Summary

All methodological contradictions encountered during framework development, and their resolutions:

| # | Question | Options Considered | Decision | Rationale | Source |
|---|---------|-------------------|----------|-----------|--------|
| 1 | Origin vs intercept? | Origin (through 0), intercept (free α) | **Intercept** | Origin biased by +0.4–0.5; proven on synthetic data | Finding 1, diag_1 |
| 2 | Per-insertion or per-metaorder? | Per-insertion (autocorrelated), per-metaorder (independent) | **Per-metaorder** at k=K | Per-insertion violates OLS independence; per-K at fixed k is valid cross-section | Finding 2 |
| 3 | Which σ estimator? | 7 methods | **Parkinson (primary)**, all 7 reported | β_slope invariant to σ; Parkinson gives best R² | §4.2, vol_sweep |
| 4 | VWAP or midprice? | VWAP, midprice, instantaneous | **VWAP** (child ≥ p25), **midprice** (fallback) | VWAP is standard in literature; breaks for tiny orders | §4.3, diag_5 |
| 5 | Per-K or pooled? | Per-K (separate regression), pooled (mix all k) | **Per-K** for dynamics, **k=K_max** for primary β | Per-K reveals model differentiation timing | §4.4 |
| 6 | How to calibrate child? | Fixed, depth percentiles, η-derived, median MO | **Median MO** (V8+) | Realistic institutional order sizes | Finding 3, §3.2 |
| 7 | How to set mb? | Fixed, grid, η-calibrated | **η-calibrated** per stock | Controls participation rate consistently | §3.3, Finding 5 |
| 8 | Cooling or not? | c>0 (observe relaxation), c=0 (per-K dynamics) | **c=0** (V7+) | Per-K β(k) more informative than post-injection observation | §3.4 |
| 9 | Daily V or V_local? | V = daily execution volume, V_local = sample execution volume | **Daily V** (primary), V_local (informational) | Daily V matches literature convention; V_local is correlated with model quality | §5.7 |
| 10 | Bootstrap unit? | Per-point, per-sample_id (group) | **Per-sample_id** (group bootstrap) | Points within a sample are correlated | §5.1 |

---

*End of FRAMEWORK.md — Version 1.0*
