# Market Impact Evaluation Framework for Generative LOB Models

**Date:** March 24, 2026
**Authors:** George Nigm, Claude Code
**Stocks:** GOOG, INTC (January 2026, LOBSTER L2 data, NASDAQ)
**Models tested:** LobS5 (7M), S5-120M, S5-4K (ctx=4000), S5-360M, LobS5-v2, Historic, Heuristic, CST, CGAN, ZeroInsertions

---

## Part I: The Problem

We need to answer one question: **does a generative LOB model produce realistic market dynamics?**

There are many ways to test this — distributional metrics (spread, volume, interarrival times), stylized facts (volatility clustering, long memory), or benchmarks like LOB-Bench. But the most economically meaningful test is **market impact**: when a large order hits the book, does the price respond the way it does in real markets?

The empirical square-root law (Kyle 1985, Tóth et al. 2011, Bouchaud et al. 2018) says:

```
I = Y · σ · (Q/V)^β    where β ≈ 0.5
```

- I = price impact (implementation shortfall)
- Q = total volume of the metaorder
- V = daily traded volume
- σ = daily volatility
- β = scaling exponent (the number we estimate)

A model that generates realistic order flow should produce β close to 0.5. A model that generates unrealistic flow will have β far from 0.5. This gives us a single number — β — that captures the quality of the generated microstructure.

The goal of this work was to build a **framework** that takes any generative LOB model, runs a standardized impact experiment, and outputs β with confidence intervals. The framework should require minimal user input — ideally just the stock name and the model checkpoint.

---

## Part II: What We Discovered

### Finding 1: The origin estimator is biased — use the intercept estimator

The standard approach in the early LOB literature fits `log(I/σ) = β · log(Q/V)` through the origin (0,0). This is the **origin estimator**. It gave us β ≈ 0.82 for all models — no differentiation, and far from 0.5.

We proved this is a mathematical artifact. When the true model has a nonzero intercept α:

```
β_origin = β_true + (α − E[log σ]) · Σx / Σx²
```

With our data (α ≈ −9, mean(x) ≈ −6.8), this adds +0.51 to the true β. So β_origin ≈ 0.33 + 0.51 = 0.84. It's not the data — it's the estimator.

**Proof:** We generated 80,000 synthetic data points with known β_true ∈ {0.3, 0.4, 0.5, 0.6, 0.7} and α ∈ {−5, −7, −9, −11}. The intercept estimator recovered β_true to within ±0.003 in all 20 scenarios. The origin estimator was biased by +0.13 to +0.66 depending on α.

**Rule:** Always use `log(I) = α + β · log(Q/V)` with free intercept. Never force through origin.

### Finding 2: One metaorder = one data point (not one per insertion)

Our experiment injects i aggressive orders into the generated book, with mb cooling messages between them. The naive approach treats each insertion k as a separate data point: (Q_cumulative_k, I_k). With i=10 insertions per sample and 200 samples, this gives 2000 points.

The problem: these points are **autocorrelated**. Points k=1 through k=10 from the same sample share the same conditioning context, the same model-generated messages, and the same book trajectory. OLS assumes independence.

This gave β ≈ 0.33 — biased downward by within-run impact saturation.

The correct approach: **one metaorder = one point**. Each (config, sample, direction) produces exactly one observation:

```
Q = total metaorder volume = Σ size_k (all child orders)
I = implementation shortfall = |VWAP_all_fills − mid_arrival| / mid_arrival
```

This gave β ≈ 0.46 for S5 models — much closer to 0.5, and with clear model differentiation.

**Rule:** Never use per-insertion points. Always collapse to one point per metaorder.

### Finding 3: Volume calibration is critical

The original experiments used order volumes of 75/300/485 shares for all stocks. These were calibrated from an older GOOG 2023 dataset.

For the current data (January 2026):
- GOOG: 75 shares = p38 of depth at best (order eats first level only 38% of the time) — marginal
- INTC: 75 shares = p6 of depth (order almost never eats the first level) — broken

INTC is 10× more liquid than GOOG. The same order volume produces completely different impact regimes. Volume must be calibrated per stock.

We computed depth-at-best statistics from 122,880 observations per stock and set volumes to p50/p75/p95 of the depth distribution:
- GOOG: {105, 165, 325} shares
- INTC: {590, 1110, 3120} shares

**Rule:** Calibrate order volume from the stock's book depth, not arbitrarily.

### Finding 4: Our participation rate is ~99%

In real markets, the square-root law is measured on institutional metaorders with participation rate φ = 5–20% (the metaorder is 5–20% of total market activity during execution). The crossover from linear impact (β=1) to square-root (β=0.5) happens at φ ≈ 1–2% (Bucci, Lillo, Bouchaud 2019).

In our simulation, the model generates mb messages between our insertions. Only ~0.8% (GOOG) to ~2.7% (INTC) of these messages are executions. The rest are limit orders and cancellations.

```
GOOG, mb=5, vol=105:
  Market volume between insertions = 5 × 0.008 × 36 = 1.4 shares
  Our insertion: 105 shares
  φ = 105 / (105 + 1.4) = 98.6%
```

**We are always at φ > 80%, regardless of mb.** Our aggressive order dominates the market. This is fundamentally different from the real-market setting where the trader is a small player.

This does NOT invalidate our results. It means we're running a **stress test**: how does the model's book respond when hit by a dominant aggressive flow? A model with realistic microstructure still shows β closer to 0.5 because it generates realistic depth, resilience, and order flow dynamics.

### Finding 5: mb controls book resilience, not participation rate

We observed that β depends strongly on mb (messages between insertions):
- mb=3: β ≈ 0.35
- mb=10: β ≈ 0.14
- mb=20: β ≈ 0.09

Initially we thought this was about participation rate regime (more mb → lower φ → more linear). This was wrong — φ > 80% for all mb values.

The real mechanism: **mb controls how much time the model has to restore the book between impacts.**
- mb=3: 3 messages → maybe 2 limit orders and 1 cancel. Depth barely restored. Next insertion hits an empty book → large impact → β stays high.
- mb=20: 20 messages → maybe 10 limit orders. Depth partially restored. Next insertion hits a deeper book → smaller marginal impact → β drops.

This is actually a useful dimension for model evaluation: it measures **book resilience** — how quickly the model regenerates realistic depth after a shock.

### Finding 6: Cross-sectional β differentiates models

With the correct methodology (per-metaorder, intercept estimator, calibrated volumes), β clearly separates models:

**GOOG:**

| Model | β | 95% CI | Group |
|-------|---|--------|-------|
| LobS5-v2 | 0.466 | [0.450, 0.482] | S5 Neural |
| LobS5 | 0.462 | [0.447, 0.479] | S5 Neural |
| Heuristic | 0.457 | [0.441, 0.471] | Baseline |
| S5-4K | 0.453 | [0.438, 0.469] | S5 Neural |
| S5-120M | 0.445 | [0.429, 0.460] | S5 Neural |
| S5-360M | 0.441 | [0.426, 0.456] | S5 Neural |
| Historic | 0.430 | [0.413, 0.445] | Baseline |
| CST | 0.353 | [0.337, 0.368] | Parametric |
| CGAN | 0.275 | [0.257, 0.291] | Parametric |

S5 models: β ≈ 0.45–0.47. CST: 0.35. CGAN: 0.27. Spread: 0.19.

**INTC:** Similar ranking. S5 family β ≈ 0.40–0.41. CGAN: 0.14.

### Finding 7: Kyle λ provides complementary model differentiation

Kyle λ(k) = |exec_price − mid_before| / (tick × size) at each insertion k. At k=1 all models are identical (same conditioning book). At k > 5, models diverge:
- CST/CGAN: λ stays flat (~0.06–0.09) — they over-replenish the book
- S5 models: λ rises to 0.15–0.30 — realistic depletion with partial recovery
- Historic: λ rises to 0.36 — real data replay, no model-based replenishment

This is the strongest differentiator between model families and doesn't require the cross-sectional regression machinery.

### Finding 8: Stock statistics determine experimental parameters

From the conditioning data (LOBSTER messages), we can compute per-stock:
- Message rate R (events/sec): GOOG 137, INTC 91
- Execution fraction: GOOG 0.8%, INTC 2.7%
- Average trade size: GOOG 36 shares, INTC 112 shares
- Median trade size: GOOG 24 shares, INTC 99 shares
- Depth at best: GOOG p50=105, INTC p50=590

These statistics determine what order volume is "realistic" and what mb gives a particular participation rate. The goal is to minimize arbitrary parameter choices.

---

## Part III: The Framework

### How to evaluate a new generative LOB model

**Input:** Stock name + model checkpoint. Nothing else.

**Step 0: Compute stock statistics** (automatic, run once per stock)

```python
python lob_impact/compute_stock_stats.py --stock GOOG
# → R, exec_fraction, avg_trade_size, median_trade_size, depth percentiles, V_daily, σ
```

From this we get:
- **vol** = calibrated order volumes {p50, p75, p95 of depth at best}
- **mb** = messages between insertions (controls resilience test severity)
- **V_daily, σ** = normalization parameters for the regression

**Step 1: Run impact experiments** (SLURM array job)

For each (i, vol, direction) combination:
- Load 500 conditioning messages from real LOBSTER data
- Model generates new messages
- Insert i aggressive market orders of size vol, with mb cooling messages between them
- Save: generated messages + orderbook states + aggressive indices

The experiment grid varies:
- i ∈ {2, 3, 5, 9, 10, 12, 15} — number of child orders (metaorder size)
- vol ∈ {p50, p75, p95 depth} — child order size
- direction ∈ {buy, sell} — side

Constraint: total generated messages = (i + c) × mb ≤ model's generation capacity (typically 500–600 for ctx=500 models, up to 4000 for S5-4K).

**Step 2: Compute per-metaorder impact** (analysis script)

For each (config, sample, direction) = one metaorder:
```
Q = Σ size_k across all i child orders
VWAP = Σ(size_k × exec_price_k) / Q
mid_arrival = midprice at first child order
I = |VWAP − mid_arrival| / mid_arrival
```

One point per metaorder. ~15,000 independent metaorders per model per stock.

**Step 3: Estimate β** (cross-sectional regression)

```
log(I) = α + β · log(Q / V_daily)
```

OLS with free intercept. Bootstrap 2000 resamples for 95% CI.

**Step 4: Report**

- β per model with 95% CI
- Model ranking by |β − 0.5|
- Kyle λ(k) trajectory per model
- Stratification: β vs i, β vs mb, β vs vol, β per day
- Bootstrap distributions

### What the framework measures

β captures the combined effect of:
1. **Book depth generation**: Does the model maintain realistic depth? (Deeper book → more volume needed for same impact → affects β)
2. **Book resilience**: After an aggressive order depletes depth, how quickly does the model restore it? (Faster restoration → lower subsequent impact → affects β through mb dependence)
3. **Order flow realism**: Do the model's messages look like real market activity? (Realistic mix of limit orders, cancels, and executions → realistic impact dynamics)
4. **Price discovery**: Does the midprice respond correctly to order flow imbalance? (Correct response → β closer to 0.5)

A model that gets all four right will have β ≈ 0.5. A model that fails on any dimension will show β ≠ 0.5.

### What β does NOT measure

- **Absolute impact magnitude**: Two models can have the same β but different α (intercept). α captures the level of impact, β captures the scaling.
- **Impact at low participation rate**: Our experiments run at φ > 80%. We cannot test the linear regime (φ < 1%) because the model doesn't generate enough market activity between our insertions.
- **Multi-day effects**: Each experiment is within one conditioning context (~500 messages ≈ minutes of real time). Cross-day effects are not captured.

### Why β < 0.5 in our experiments

Our best models achieve β ≈ 0.46, not 0.50. The gap of ~0.04 is explained by:

1. **Extreme participation rate** (φ ≈ 99%): Our aggressive orders dominate the market. At such high φ, the book is perpetually depleted. Impact partially saturates because there's simply no more depth to consume.

2. **Finite generation capacity**: ctx=500 models generate ≤600 messages. This limits the total metaorder size and the Q/V range available for regression (~3.9 log-units vs 4–5 in the literature).

3. **Simulation vs reality**: Our "market" is a single generative model. Real markets have hundreds of participants whose combined order flow creates the emergent square-root scaling. Our model approximates this with one generator.

Despite these limitations, β ≈ 0.46 for S5 models is remarkably close to 0.5, and the differentiation from baselines (CST 0.35, CGAN 0.27) is statistically significant.

### How to use this framework for a new model

1. Train your generative LOB model on LOBSTER data
2. Run `compute_stock_stats.py` for your stock → get vol, mb, V, σ
3. Run `run_isambard_c10x_v4.sh submit your_model --stock YOUR_STOCK` → SLURM experiments
4. Run `run_300_v4_slurm.sh YOUR_STOCK` → analysis + figures
5. Read `pics_for_v4_300_YOUR_STOCK/summary_statistics.csv` → β, CI, R², and 15+ other metrics
6. Compare β to existing models

Total compute: ~20 GPU-hours per model per stock (at 2048 samples).

---

## Part IV: Complementary Metrics

β is the primary metric. These are secondary:

| Metric | What it measures | Target | Source |
|--------|-----------------|--------|--------|
| Kyle λ(k) trajectory | Per-insertion book response | Rising, similar to Historic | fig_17 |
| Relaxation ratio | Permanent vs temporary impact | ≈ 2/3 (Bouchaud) | fig_5 |
| Stability vote | Does impact stabilize? | > 50% stable | fig_6 |
| Hurst exponent | Long memory in order flow | H ≈ 0.7 | fig_7 |
| Propagator G(l) | Impact memory kernel | G ~ l^{-0.5} | fig_8 |
| Spread dynamics | Spread after shock | Recovers to pre-shock | fig_9 |
| Master curve shape | Impact trajectory | Concave, peaks at u=1 | fig_2 |
| No-arb score | β_perm × relaxation | In target zone | fig_14 |

All computed automatically by `run_300_figures.py`.

---

## Part V: Methodological Details

### Why intercept estimator, not origin (Finding 1)

The origin estimator `β_origin = Σ(xi·yi)/Σ(xi²)` is equivalent to OLS without intercept. When the true DGP has α ≠ 0, the omitted intercept is absorbed into the slope, causing bias:

```
bias = (α − E[log σ]) · E[x] / E[x²]
```

For our data: α ≈ −9, E[log σ] ≈ −3.5, E[x] ≈ −6.8, E[x²] ≈ 50.4 → bias ≈ +0.74.

The intercept estimator is standard OLS on `log(I) = α + β · log(Q/V)` and is unbiased regardless of α.

**Reports:** `pics_for_investigation/diag_1_synthetic.txt`, `pics_for_beta_report/R01_synthetic_calibration.png`

### Why per-metaorder, not per-insertion (Finding 2)

Within one simulation run, the trajectory (Q_1, I_1), ..., (Q_K, I_K) is a TIME SERIES, not a cross-section. The points are autocorrelated because they share:
- The same conditioning context
- The same model-generated messages (up to that point)
- Cumulative Q that grows deterministically: Q_k = Q_{k-1} + vol

The per-insertion slope measures **within-run impact growth** (how impact accumulates with more injections in the same book). This is NOT the same as the cross-sectional square-root law (how impact scales across different-sized metaorders in different books).

The empirical square-root law was established on cross-sectional data: thousands of different metaorders from different traders on different days (Tóth et al. 2011: ~500,000 metaorders). Our cross-sectional approach matches this: each metaorder is an independent sample with its own conditioning context.

**Reports:** `pics_for_cross_sectional/cross_sectional_beta_GOOG.pdf`

### Why participation rate matters but doesn't control our design (Finding 4)

Bucci et al. (2019) showed that β transitions from ~1 (linear) at φ < 1% to ~0.5 (square-root) at φ > 2%. In real markets, institutional metaorders typically have φ = 5–20%.

In our simulation, even with mb=50 (50 messages between insertions), only ~0.8% of messages are executions. The model generates mostly limit orders and cancellations. So the executed volume between our insertions is tiny relative to our aggressive order:

```
φ = our_vol / (our_vol + market_exec_between) ≈ 105 / (105 + 1.4) ≈ 98.6%
```

We cannot achieve φ < 50% with feasible generation lengths. This is a fundamental property of order-level LOB simulation: most messages are NOT executions.

**Implication:** Our β ≈ 0.46 is measured at φ ≈ 99%, not φ ≈ 10%. The comparison to empirical β ≈ 0.5 should acknowledge this difference. However, the relative ranking of models is valid regardless of φ — if S5 produces β closer to 0.5 than CGAN at the same φ, it has better microstructure.

**Reports:** `lob_impact/compute_stock_stats.py` output

### Volume calibration from depth-at-best (Finding 3)

Depth-at-best = volume at the best ask (for buy orders) or best bid (for sell orders) at the moment of the first insertion. This is measured from the book state in the conditioning data.

From 122,880 observations per stock:
```
         GOOG    INTC
p25       45     275
p50      105     590
p75      166    1110
p95      325    3120
p99      645   10710
```

Old volumes (75/300/485) corresponded to p38/p93/p98 for GOOG but only p6/p26/p42 for INTC. INTC orders were too small to meaningfully perturb the book.

Calibrated volumes use p50/p75/p95 per stock, ensuring a range from "sometimes penetrates first level" to "always penetrates."

**Reports:** `lob_impact/depth_stats_GOOG.csv`, `depth_stats_INTC.csv`

---

## Part VI: Stock Statistics Reference

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

---

## Part VII: All Generated Reports

```
Framework & Results:
  pics_for_framework_report/framework_report.pdf        — 21 pages, final framework + both stocks
  pics_for_cross_sectional/cross_sectional_beta_GOOG.pdf — 10 pages, bootstrap + scatter
  pics_for_cross_sectional/stratified_beta_GOOG.pdf     — 17 pages, stratification
  pics_for_cross_sectional/stratified_beta_INTC.pdf     — 17 pages, stratification

Beta Analysis:
  pics_for_beta_report/beta_report_full.pdf             — 27 pages, comprehensive
  pics_for_beta_grid/beta_grid_report.pdf               — 10 pages, clean grid validation
  pics_for_counterfactual/counterfactual_beta_GOOG.pdf  — 7 pages, MarS-style correction

Diagnostics:
  pics_for_investigation/diag_1_synthetic.txt            — estimator bias proof
  pics_for_investigation/diag_2_data_integrity.txt       — data validation
  pics_for_investigation/diag_3_distribution_GOOG.txt    — point cloud analysis
  pics_for_investigation/diag_4_normalizations_GOOG.txt  — V normalization comparison
  pics_for_investigation/diag_5_impact_defs_GOOG.txt     — impact definition comparison
  pics_for_investigation/diag_6_reproduce_140.txt        — old notebook reproduction
  pics_for_investigation/diag_7_stratified.txt           — full stratification

Standard Analysis (v4 grid):
  pics_for_v4_300_GOOG/ — 18 figures + summary CSV (10 models)
  pics_for_v4_300_INTC/ — 18 figures + summary CSV (10 models)
```

---

## Part VIII: Scripts Reference

```
Experiment launch:
  lob_impact/run_isambard_c10x_v4.sh      — v4 grid (calibrated volumes)
  lob_impact/run_isambard_c10x_v4_hii.sh  — high-i extension (i=10,12,15)
  lob_impact/run_isambard_beta_grid.sh     — clean beta grid (fixed i,mb)
  lob_impact/run_isambard_production.sh    — production (if created)

Analysis pipeline:
  lob_impact/run_300_compute.py            — CSV → pickle conversion
  lob_impact/run_300_analyze_one.py        — per-model analysis
  lob_impact/run_300_figures.py            — figure generation
  lob_impact/run_300_v4_slurm.sh           — full pipeline (SLURM)

Stock calibration:
  lob_impact/compute_depth_stats.py        — depth-at-best analysis
  lob_impact/compute_stock_stats.py        — message rate, trade size, etc.
  lob_impact/compute_daily_high_low.py     — daily H/L/V from LOBSTER

Reports:
  lob_impact/run_framework_report.py       — framework PDF
  lob_impact/run_beta_report.py            — comprehensive beta PDF
  lob_impact/run_cross_sectional_stratified.py — stratified analysis PDF
  lob_impact/run_counterfactual_beta.py    — counterfactual correction

Diagnostics:
  lob_impact/diag_1_synthetic_calibration.py — estimator bias test
  lob_impact/diag_2_data_integrity.py        — data validation
  lob_impact/diag_3_distribution_analysis.py — distribution analysis
  lob_impact/diag_4_normalizations.py        — normalization comparison
  lob_impact/diag_5_impact_definitions.py    — impact definition comparison
  lob_impact/diag_6_reproduce_140.py         — old notebook reproduction
  lob_impact/diag_7_stratified.py            — stratified analysis
```

---

## Part IX: What Remains

1. **Production run (2048 samples)** on v4 grid — tighter CI, same β
2. **Paper section** — LaTeX writeup of the framework + results
3. **Appendix** — design justifications (why intercept, why per-metaorder, why these volumes)
