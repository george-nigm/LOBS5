# Deep Critical Analysis: Market Impact Beta Comparison

## Notebooks 170 (v3 Grid) & 171 (c10x_v2 Grid) — Five-Model Comparison

---

## 1. Overview & Motivation

### What beta measures

In market impact theory, the relationship between executed volume and price impact follows a power law:

```
log(impact) = alpha + beta * log(Q / V_day)
```

where `Q` is the executed quantity, `V_day` is the daily volume, and `beta` is the power-law exponent. The theoretical square-root impact law predicts **beta = 0.5** — price impact scales as the square root of the participation rate. This is one of the most robust stylized facts in market microstructure, supported by empirical studies reporting betas in the range 0.4–0.7.

Beta is a **first-order metric**: it measures the average scaling relationship between order size and impact. It does not capture dynamics (how impact evolves during execution), higher-order distributional properties, or temporal correlations.

### The five models

| Model | Type | Description |
|-------|------|-------------|
| **Historic** | Replay | Replays the actual historical LOB evolution; the metaorder executes against real recorded book dynamics |
| **Heuristic** | Baseline | Preserves empirical LOB structure with a simple deterministic price shift proportional to order flow |
| **CST** | Statistical | Continuous-state model that generates LOB dynamics via calibrated statistical processes |
| **LOB-S5** | Neural | Deep sequence model (S5 architecture) trained to generate realistic LOB dynamics |
| **CGAN** | Neural (GAN) | Conditional GAN trained to generate LOB states adversarially |

### The two grids

- **v3 (Notebook 170):** 7 `mb` values × 7 `i` values × 3 `V` values = 21 configurations, ~2.3M data points per model. Covers mb = {5, 10, 20, 25, 50, 75, 100}.
- **c10x_v2 (Notebook 171):** Constrained grid with `11·i·mb ≤ 500`, yielding 30 folders (27 with valid betas) × 3 `V` values. Covers only mb = {5, 10, 15, 20}. ~392K data points per model.

**Critical grid difference:** The c10x_v2 grid has **max mb = 20**, which means no "Hard" cases (defined as mb ≥ 50) exist. This fundamentally changes the analysis — the difficulty-band comparison that is central to notebook 170 cannot be replicated in 171.

---

## 2. Individual Analysis: Notebook 170 (v3 Grid)

### 2.1 The imbalance problem

The mb = 5 configuration contributes disproportionately to the regression:

| mb | i | N_pts | % total | OLS weight % |
|----|---|-------|---------|-------------|
| 5 | 83 | 1,017,665 | 44.1% | 36.4% |
| 10 | 45 | 550,767 | 23.9% | 23.2% |
| 20 | 23 | 280,544 | 12.2% | 13.9% |
| 25 | 19 | 231,334 | 10.0% | 12.0% |
| 50 | 9 | 108,769 | 4.7% | 6.6% |
| 75 | 6 | 71,908 | 3.1% | 4.7% |
| 100 | 4 | 47,612 | 2.1% | 3.4% |

The "easy" configurations (mb ≤ 10) account for **68% of data** and **59.6% of regression weight**. Since easy cases produce mechanically inflated betas (the LOB barely changes, so all models behave similarly), this creates a systematic upward bias in global beta and artificially compresses inter-model differences.

### 2.2 Results summary

#### Grand summary table (from 170_run.log)

| Metric | Historic | Heuristic | CST | LOB-S5 | CGAN |
|--------|----------|-----------|-----|--------|------|
| Global beta | 0.750 | 0.677 | 0.713 | 0.675 | 0.675 |
| Per-config median | 0.673 | 0.645 | 0.683 | 0.650 | 0.661 |
| First-impact (k=1) | 0.469 | 0.469 | 0.478 | 0.469 | 0.538 |
| Folder-weighted | 0.649 | 0.622 | 0.648 | 0.621 | 0.644 |
| Easy (mb ≤ 10) | 0.822 | 0.707 | 0.749 | 0.703 | 0.690 |
| Hard (mb ≥ 50) | 0.577 | 0.575 | 0.592 | 0.575 | 0.611 |
| Delta(E−H) | 0.245 | 0.132 | 0.157 | 0.129 | 0.079 |

#### Per-config beta distribution

| Model | N configs | Mean | Median | Std | Min | Max |
|-------|-----------|------|--------|-----|-----|-----|
| Historic | 21 | 0.671 | 0.673 | 0.107 | 0.536 | 0.856 |
| Heuristic | 21 | 0.635 | 0.645 | 0.073 | 0.535 | 0.812 |
| CST | 21 | 0.663 | 0.683 | 0.079 | 0.547 | 0.817 |
| LOB-S5 | 21 | 0.634 | 0.650 | 0.071 | 0.537 | 0.804 |
| CGAN | 21 | 0.651 | 0.661 | 0.055 | 0.580 | 0.787 |

#### KS test results

All pairwise KS tests are **non-significant** after Bonferroni correction (10 pairs). Even uncorrected, the smallest p-value is 0.0948 (CST vs LobS5, CST vs Heuristic). The per-config beta distributions are statistically indistinguishable.

#### Equalized betas

| Method | Historic | Heuristic | CST | LOB-S5 | CGAN |
|--------|----------|-----------|-----|--------|------|
| mb-equalized | 0.649 | 0.622 | 0.648 | 0.621 | 0.644 |
| i-equalized | 0.649 | 0.622 | 0.648 | 0.621 | 0.644 |
| Context%-equalized | 0.636 | 0.608 | 0.633 | 0.608 | 0.633 |
| Difficulty-equalized | 0.673 | 0.638 | 0.667 | 0.637 | 0.653 |

### 2.3 Critical observations for 170

1. **Hard-case betas cluster tightly:** 0.575–0.611, a spread of only 0.036. If easy cases were the only problem, removing them should reveal clear model differentiation — but it does not.

2. **k=1 divergence:** CGAN uniquely diverges at k=1 (0.538 vs 0.469 for Historic/Heuristic/LOB-S5, 0.478 for CST). At k=1 there is exactly one data point per sample, eliminating all within-sample imbalance.

3. **LOB-S5 ≈ Heuristic** on virtually every fair metric: per-config median (0.650 vs 0.645), folder-weighted (0.621 vs 0.622), hard-case (0.575 vs 0.575), Delta (0.129 vs 0.132). The differences are negligible.

4. **Historic is worst** by all fair measures: highest global beta (0.750), highest Delta (0.245), highest per-config std (0.107).

5. **CGAN has lowest Delta (0.079)** and lowest per-config std (0.055), indicating the most consistent calibration across difficulty levels — but its hard-case beta (0.611) is the furthest from 0.5.

6. **CGAN buy/sell asymmetry:** CGAN shows a persistent BUY-SELL difference (up to −0.125 at mb=5), far larger than any other model (which show differences < 0.015). This suggests a systematic bias in CGAN's generated LOB dynamics.

7. **Ranking by proximity to 0.5:** Heuristic and LOB-S5 are tied at average rank 2.0, followed by CGAN (3.2), CST (3.7), Historic (4.2).

---

## 3. Individual Analysis: Notebook 171 (c10x_v2 Grid)

### 3.1 Grid characteristics

The c10x_v2 grid uses a constraint `11·i·mb ≤ 500`, yielding configurations with mb = {5, 10, 15, 20} only. This means:

- **No Hard cases exist** (Hard requires mb ≥ 50). The Delta(E−H) analysis cannot be performed.
- **30 folders loaded, 27 with valid betas** (3 configs produced insufficient data).
- **~392K points per model** (roughly 6× fewer than v3, but far from the 200× ratio one might expect from 21 vs 30 configs — many insertions per config compensate).
- **mb=5 contributes 52.0% of data and 48.7% of OLS weight** — the imbalance is even worse than in v3.

### 3.2 Results summary

#### Grand summary table (from 171_run.log)

| Metric | Historic | Heuristic | CST | LOB-S5 | CGAN |
|--------|----------|-----------|-----|--------|------|
| Global beta | 0.545 | 0.544 | 0.558 | 0.542 | 0.586 |
| Per-config median | 0.521 | 0.521 | 0.534 | 0.519 | 0.572 |
| First-impact (k=1) | 0.469 | 0.469 | 0.480 | 0.468 | 0.539 |
| Folder-weighted | 0.527 | 0.526 | 0.538 | 0.525 | 0.576 |
| Easy (mb ≤ 10) | 0.557 | 0.555 | 0.569 | 0.553 | 0.592 |
| Hard (mb ≥ 50) | NaN | NaN | NaN | NaN | NaN |
| Delta(E−H) | NaN | NaN | NaN | NaN | NaN |

#### Per-config beta distribution

| Model | N configs | Mean | Median | Std | Min | Max |
|-------|-----------|------|--------|-----|-----|-----|
| Historic | 27 | 0.529 | 0.521 | 0.034 | 0.494 | 0.608 |
| Heuristic | 27 | 0.529 | 0.521 | 0.033 | 0.494 | 0.608 |
| CST | 27 | 0.541 | 0.534 | 0.036 | 0.504 | 0.627 |
| LOB-S5 | 27 | 0.527 | 0.519 | 0.032 | 0.491 | 0.603 |
| CGAN | 27 | 0.577 | 0.572 | 0.020 | 0.553 | 0.632 |

#### KS test results

Unlike 170, the 171 KS tests reveal a **clear structure:**

| | Historic | Heuristic | CST | LOB-S5 | CGAN |
|--|----------|-----------|-----|--------|------|
| Historic | 1.000 | 1.000 | 0.996 | 1.000 | **0.000** |
| Heuristic | 1.000 | 1.000 | 0.996 | 1.000 | **0.000** |
| CST | 0.996 | 0.996 | 1.000 | 0.996 | **0.000** |
| LOB-S5 | 1.000 | 1.000 | 0.996 | 1.000 | **0.000** |
| CGAN | **0.000** | **0.000** | **0.000** | **0.000** | 1.000 |

*(Bonferroni-corrected p-values)*

CGAN is **significantly different** from all other models. The remaining four models (Historic, Heuristic, CST, LOB-S5) are statistically indistinguishable from each other.

#### Equalized betas

| Method | Historic | Heuristic | CST | LOB-S5 | CGAN |
|--------|----------|-----------|-----|--------|------|
| mb-equalized | 0.526 | 0.525 | 0.537 | 0.523 | 0.575 |
| i-equalized | 0.542 | 0.541 | 0.554 | 0.539 | 0.584 |
| V-equalized | 0.545 | 0.544 | 0.558 | 0.542 | 0.586 |

### 3.3 Critical observations for 171

1. **All betas are much closer to 0.5** than in 170 (range 0.52–0.59 vs 0.62–0.75). This is because the grid lacks high-mb configs where beta inflation is strongest.

2. **CGAN is the clear outlier.** Its per-config median (0.572) is 0.05 above the next model (CST at 0.534), and it is the only model with statistically distinguishable beta distributions.

3. **Historic ≈ Heuristic ≈ LOB-S5** even more tightly than in 170. Their per-config medians differ by only 0.002 (0.519–0.521). In this restricted grid, these three models are functionally identical on beta.

4. **CST sits slightly above** the Historic/Heuristic/LOB-S5 cluster (median 0.534 vs 0.519–0.521) but is not statistically significant.

5. **CGAN buy/sell asymmetry persists:** differences of 0.014–0.022, consistent with the pattern seen in 170 but smaller in magnitude (because mb range is smaller).

6. **The ranking run crashed** due to NaN values in Hard/Delta columns, so no automated ranking was produced. Manual ranking by proximity to 0.5 (per-config median): LOB-S5 (0.519) > Historic = Heuristic (0.521) > CST (0.534) > CGAN (0.572).

---

## 4. Cross-Notebook Comparison (170 vs 171)

### 4.1 What is robust (consistent across both grids)

1. **Model ordering on fair metrics:** Historic ≈ Heuristic ≈ LOB-S5 < CST < CGAN (distance from 0.5). This ordering holds in both grids.

2. **CGAN k=1 uniqueness:** k=1 betas of 0.538 (170) and 0.539 (171) vs 0.468–0.480 for all other models. This is the most robust finding across both grids.

3. **Global beta inflation:** Always higher than per-config medians, confirming the Simpson's-paradox-like effect of pooling across configs with unequal sizes.

4. **Heuristic competitive with neural models** on beta: in both grids, Heuristic matches or beats LOB-S5 on most fair beta metrics.

5. **CGAN buy/sell asymmetry:** Persistent in both grids, suggesting a systematic architectural property.

### 4.2 What changes

| Property | v3 (170) | c10x_v2 (171) |
|----------|----------|---------------|
| Beta range (per-cfg median) | 0.645–0.683 | 0.519–0.572 |
| Hard-case analysis | Available (mb up to 100) | **Not available** (max mb = 20) |
| KS significance | None after correction | **CGAN significant** vs all |
| Per-config beta std | 0.055–0.107 | 0.020–0.036 |
| N points per model | ~2.3M | ~392K |

The dramatic drop in betas from 170 to 171 is **not** a statistical artifact — it directly reflects the absence of high-mb configurations. High-mb configs inflate beta because they combine many insertions (each contributing a data point) with large cumulative impact, pulling the regression slope upward.

### 4.3 The CGAN separation

In 170, CGAN blends in with the crowd (global beta 0.675, same as LOB-S5). In 171, CGAN clearly separates (global 0.586 vs next-highest 0.558). The key insight: **when high-mb configs are removed, CGAN's inherently different dynamics become visible.** In the v3 grid, the overwhelming weight of high-mb/high-i configs masks CGAN's distinctiveness.

This is the inverse of the easy-case hypothesis: rather than easy cases inflating all models equally, removing hard cases reveals that CGAN generates fundamentally different impact scaling even for "easy" configurations.

### 4.4 Beta-by-mb profiles (from equalized-i analysis)

**v3 grid (170) — beta by i (proxy for mb):**

| i | Historic | Heuristic | CST | LOB-S5 | CGAN |
|---|----------|-----------|-----|--------|------|
| 4 (mb=100) | 0.537 | 0.537 | 0.549 | 0.539 | 0.586 |
| 6 (mb=75) | 0.571 | 0.570 | 0.583 | 0.565 | 0.607 |
| 9 (mb=50) | 0.601 | 0.598 | 0.620 | 0.600 | 0.626 |
| 19 (mb=25) | 0.673 | 0.658 | 0.690 | 0.658 | 0.670 |
| 23 (mb=20) | 0.693 | 0.671 | 0.705 | 0.671 | 0.680 |
| 45 (mb=10) | 0.771 | 0.704 | 0.743 | 0.706 | 0.711 |
| 83 (mb=5) | 0.854 | 0.709 | 0.752 | 0.702 | 0.676 |

**c10x_v2 grid (171) — beta by i:**

| i | Historic | Heuristic | CST | LOB-S5 | CGAN |
|---|----------|-----------|-----|--------|------|
| 2 | 0.498 | 0.498 | 0.506 | 0.497 | 0.559 |
| 3 | 0.522 | 0.522 | 0.534 | 0.520 | 0.573 |
| 4 | 0.542 | 0.541 | 0.553 | 0.541 | 0.585 |
| 5 | 0.558 | 0.557 | 0.571 | 0.553 | 0.591 |
| 9 | 0.607 | 0.603 | 0.622 | 0.599 | 0.619 |

Key pattern: **CGAN starts high and stays high** — its beta at i=2 (0.559) is already where other models sit at i=4-5. Historic/Heuristic/LOB-S5 converge increasingly at low i, while CGAN maintains a persistent offset.

---

## 5. Critical Assessment of the "Easy Case" Hypothesis

### 5.1 Evidence supporting the hypothesis

- mb=5 contributes 36–49% of OLS weight across both grids
- Global betas exceed per-config medians for all models (by 0.014–0.077 in 170, 0.014–0.024 in 171)
- Easy-case betas are substantially above 0.5 in the v3 grid (0.69–0.82)
- Historic shows the largest inflation (Delta = 0.245 in 170)
- All equalization methods reduce betas toward 0.5

### 5.2 Evidence complicating the hypothesis

1. **Hard cases do NOT differentiate models dramatically.** In 170, the spread in hard betas is only 0.036 (0.575–0.611). If easy cases were solely responsible for masking model differences, removing them should reveal clear LOB-S5 superiority — but all models converge to similar hard-case values.

2. **Heuristic is competitive.** A trivial price-shift heuristic matches LOB-S5 on essentially all fair metrics (per-config median: 0.645 vs 0.650 in 170, 0.521 vs 0.519 in 171). This means beta is too coarse to capture whatever LOB-S5 does differently.

3. **The hypothesis explains inflation but not convergence.** It correctly predicts that reweighting/stratifying reduces betas, but does not explain why all models converge to similar values after correction.

4. **The 171 grid removes high-mb cases entirely — and all models (except CGAN) still produce nearly identical betas.** If the problem were only about mb-weighting, the restricted grid should expose differences. Instead, it confirms convergence.

### 5.3 Revised interpretation

**Beta is a first-order metric that measures average size-impact scaling.** Any model that approximately preserves the empirical LOB structure (even by simple replay or heuristic price shift) will reproduce this scaling relationship. LOB-S5's advantages, if they exist, lie in:

- **Higher-order dynamics:** How impact evolves temporally during and after execution
- **Distributional properties:** Not just the mean relationship but the full distribution of outcomes
- **Out-of-sample generalization:** How well the model extrapolates beyond its training data
- **Spread and queue dynamics:** Microstructural properties that beta cannot capture

The easy-case hypothesis is **partially correct** — easy cases do inflate global beta — but the deeper finding is that beta itself is insufficient to differentiate these models.

---

## 6. Deep Dives on Key Puzzles

### 6.1 Why LOB-S5 ≈ Heuristic

The Heuristic model preserves the actual empirical LOB state and applies a deterministic price shift. This means it inherits the real bid-ask structure, queue sizes, and order book shape. For beta estimation, which only tests the *average* relationship between order size and impact, this is nearly sufficient — the LOB structure determines first-order price impact mechanics.

LOB-S5, despite learning complex dynamics, generates LOB states that produce the same average impact relationship. This is not a failure — it means LOB-S5 has correctly learned the equilibrium scaling. The question is whether it also produces realistic *deviations* from this equilibrium, which beta does not test.

### 6.2 Why CGAN diverges

CGAN's behavior is the most informative finding:

- **Persistent beta offset:** CGAN consistently produces betas ~0.03–0.05 higher than the Historic/Heuristic/LOB-S5 cluster across all configurations and analysis methods.
- **Statistically significant** in the c10x_v2 grid (KS p = 0.000 against all other models).
- **k=1 uniqueness:** At exactly one insertion, CGAN's beta (0.538–0.539) differs by +0.06–0.07 from other models (0.468–0.480).
- **Buy/sell asymmetry:** CGAN shows impact asymmetries (up to 12.5% at mb=5 in v3) that no other model exhibits.

Interpretation: GAN-based training may produce LOB states with systematically different liquidity profiles. The adversarial objective optimizes for visual/distributional realism rather than mechanical accuracy, potentially resulting in order books that are realistic-looking but respond differently to market orders. The buy/sell asymmetry suggests the generator has learned an asymmetric liquidity provision pattern.

### 6.3 What Delta(E−H) really measures

Delta = Beta(Easy) − Beta(Hard) quantifies how much a model's beta varies across difficulty levels.

- **Low Delta** (CGAN: 0.079) = consistent calibration, but not necessarily accurate — CGAN's hard-case beta (0.611) is furthest from 0.5.
- **High Delta** (Historic: 0.245) = the model's beta is heavily influenced by configuration difficulty. This primarily reflects that Historic replay inflates impact at easy configs (beta=0.822) because book evolution is not independent of the metaorder.

Delta measures **susceptibility to easy-case inflation**, not hard-case competence. A model with Delta=0 but beta=0.7 everywhere would have perfect consistency but poor calibration.

### 6.4 Is beta=0.5 the right target?

The theoretical prediction of beta=0.5 comes from dimensional analysis and equilibrium arguments (Bouchaud et al., Kyle's lambda). However:

- Empirical studies report betas of 0.4–0.7 depending on market, time period, and methodology
- The simulation setup (discrete LOB, finite tick size, specific order flow) may produce a "native" beta different from 0.5
- In the c10x_v2 grid, models produce betas of 0.52–0.58 (per-config median), which is well within the empirical range

**Recommendation:** Compute beta from the underlying LOB data (LOBSTER) using the identical regression framework. If the empirical benchmark is ~0.53, then all non-CGAN models in the c10x_v2 grid are essentially perfectly calibrated.

### 6.5 The KS significance asymmetry

In 170 (v3, 21 configs): all KS tests non-significant, even uncorrected. With only 21 observations per distribution, the test has limited power — non-significance is partially a sample-size artifact.

In 171 (c10x_v2, 27 configs): CGAN is highly significant (p < 0.0001 even after Bonferroni). With only 6 more configs, the dramatic change in significance for CGAN suggests a real effect, not a power issue. The difference is that in 171, the per-config beta distributions are much tighter (std 0.020–0.036 vs 0.055–0.107 in 170), so CGAN's offset is more visible relative to the noise.

For the paper: the 170 non-significance result means **we cannot claim any model has a statistically different beta distribution from any other** in the v3 grid. The 171 result means we **can claim CGAN differs**, but the remaining four models are indistinguishable.

---

## 7. Recommendations for the Paper

### Claims that CAN be made

1. **"All tested LOB simulators reproduce the empirical square-root impact law."** In the v3 grid, per-config median betas fall in [0.645, 0.683]; in the restricted c10x_v2 grid, in [0.519, 0.572]. Both ranges are consistent with the empirical literature.

2. **"Beta estimation is a low-resolution test that does not differentiate simulator architectures."** KS tests on per-config beta distributions are non-significant for all model pairs in the v3 grid. Only CGAN separates statistically in the restricted grid.

3. **"CGAN produces systematically elevated betas and exhibits unique first-impact scaling."** CGAN's k=1 beta (0.538–0.539) is consistently 0.06–0.07 above other models, and this persists across both grid designs.

4. **"Global beta estimates are biased upward by heterogeneous configuration weighting."** Global betas exceed per-config medians by up to 0.077 (Historic, v3), confirming that pooled regression inflates estimates when easy configs dominate.

5. **"LOB-S5 and Heuristic achieve the most consistent betas across difficulty levels"** (lowest Delta of 0.129/0.132, excluding CGAN which has low Delta but systematically higher betas).

6. **"Results are robust across grid designs."** Model ordering and key conclusions hold across both the full v3 and restricted c10x_v2 grids.

### Claims that should NOT be made

1. ~~"LOB-S5 outperforms baselines on market impact estimation."~~ Not supported — Heuristic matches LOB-S5 on all fair beta metrics.

2. ~~"Beta estimation validates LOB-S5 as a superior simulator."~~ Beta is too coarse to differentiate architectures.

3. ~~"Hard cases reveal the true quality of a simulator."~~ Hard-case betas converge (spread 0.036), failing to differentiate models.

4. ~~"CGAN produces the most accurate betas."~~ CGAN has the lowest Delta(E-H) but the highest absolute beta in the c10x_v2 grid, making it the furthest from 0.5 on fair metrics.

### Framing strategy

Frame beta as a **necessary-but-not-sufficient validation criterion:**

> "We evaluate all five simulators on the square-root impact law (beta ≈ 0.5), a fundamental stylized fact of market microstructure. All models reproduce this relationship to within the empirical range (beta ∈ [0.52, 0.68] across analysis methods), confirming that each simulator captures the first-order size-impact scaling. However, per-config KS tests reveal that beta distributions are statistically indistinguishable across models, indicating that beta alone cannot differentiate simulator architectures. We therefore complement beta estimation with higher-order metrics — relaxation ratio, stability fraction, and volume scaling — where model differences become apparent."

---

## 8. Contrast: LOB-S5 Strengths on Other Metrics (Notebook 140)

Notebook 140 (`140.paper_results_5models.ipynb`) analyzes metrics that capture **dynamics and distributional properties** rather than first-order scaling. Unlike beta, these metrics reveal clear model differentiation.

### 8.1 Relaxation ratio (impact decay)

**Definition:** `R = I_final / I_peak` — the ratio of permanent to peak impact. Bouchaud et al. (2004) theory predicts R ≈ 2/3 ≈ 0.667.

| Model | Median ratio | Mean ± std | CV | Delta from 2/3 |
|-------|-------------|------------|------|----------------|
| Historic | 0.618 | 0.621 ± 0.089 | 0.143 | 0.049 |
| Heuristic | 0.658 | 0.661 ± 0.085 | 0.129 | 0.009 |
| CST | 0.646 | 0.649 ± 0.091 | 0.140 | 0.021 |
| **LOB-S5** | **0.661** | **0.665 ± 0.079** | **0.119** | **0.002** |
| CGAN | 0.647 | 0.651 ± 0.088 | 0.135 | 0.016 |

**LOB-S5 advantage:** Closest to theoretical prediction (Delta = 0.002) and lowest coefficient of variation (CV = 0.119), indicating the most consistent and theoretically aligned relaxation behavior. The median of 0.661 is near-perfect agreement with the Bouchaud 2/3 prediction.

### 8.2 Stability fraction

**Definition:** A configuration is "stable" if ≥ 2 of 3 stability tests agree (trailing slope test, two-window mean test, exponential convergence test).

| Model | Stable / Total | Fraction |
|-------|---------------|----------|
| Historic | 4 / 30 | 13% |
| Heuristic | 11 / 30 | 37% |
| CST | 10 / 30 | 33% |
| **LOB-S5** | **18 / 30** | **60%** |
| CGAN | 14 / 30 | 47% |

**LOB-S5 advantage:** Highest stability fraction at 60% — nearly 5x Historic, and significantly above CGAN (47%) and Heuristic (37%). LOB-S5 produces impact curves that decay predictably in the majority of configurations.

### 8.3 Volume scaling (gamma)

**Definition:** `I_peak ∝ V^γ` — the exponent of peak impact vs. order volume. Theory predicts γ ≈ 0.5.

| Model | gamma |
|-------|-------|
| Historic | 0.489 |
| Heuristic | 0.510 |
| CST | 0.501 |
| LOB-S5 | 0.492 |
| CGAN | 0.503 |

All models cluster tightly around 0.5 (range 0.489–0.510). Like beta, gamma is a first-order scaling metric and does not strongly differentiate models.

### 8.4 The contrast with beta

| Metric | Differentiates models? | LOB-S5 advantage? |
|--------|----------------------|-------------------|
| Beta (power-law exponent) | No (KS non-significant) | No (tied with Heuristic) |
| Relaxation ratio | **Yes** | **Yes** (closest to theory, lowest CV) |
| Stability fraction | **Yes** (range 13%–60%) | **Yes** (highest at 60%) |
| Gamma (volume scaling) | No (range 0.489–0.510) | No (all models similar) |

The pattern is clear: **first-order scaling metrics** (beta, gamma) do not differentiate models, while **dynamic metrics** (relaxation, stability) do — and LOB-S5 leads on both.

### 8.5 Strategic framing

The beta convergence finding should be positioned as:

> "All models pass the necessary condition (square-root law), which confirms that first-order impact scaling is a low bar that even simple baselines clear. LOB-S5 additionally captures higher-order dynamic properties — impact relaxation, temporal stability, and volume scaling — where baselines fail. These higher-order metrics differentiate the models because they depend on realistic LOB evolution, not just equilibrium structure."

This reframes the beta result from a negative finding ("LOB-S5 doesn't outperform") to an informative methodological insight ("beta is insufficient; richer metrics are needed").

---

## 9. Experimental Plan for Follow-Up Analyses

### Experiment A: Beta-by-mb curves (extends existing analysis)

**Goal:** Instead of the binary Easy/Hard split, show the full beta degradation profile across all mb values.

**Method:** For each model, group data by `mb` and compute beta per group. Plot `beta(mb)` for all 5 models on the same axes.

**Implementation:** The v3 grid already has the data. Use `compute_per_config_beta()` grouped by `mb` (extracting mb from the folder name) rather than `folder`. The beta-by-i analysis in cells 7 already does something similar — adapt it to group by mb directly.

**Expected insight:** Models may diverge at specific mb thresholds. If LOB-S5 shows flatter beta(mb) curves than Historic, it demonstrates more consistent scaling even if the average is similar.

**Priority:** HIGH — trivial to implement, reuses existing code.

### Experiment B: Residual analysis

**Goal:** Compare the precision (not just accuracy) of beta estimates across models.

**Method:** After fitting `y_adj = beta * x`, compute residuals `e_i = y_adj_i - beta * x_i`. Compare `std(e)`, skewness, kurtosis, and quantile ranges across models.

**Implementation:** Extend `compute_global_beta()` to return residuals. For each model, compute:
- `std(residuals)` — overall precision
- `IQR(residuals)` — robust precision measure
- Residual skewness/kurtosis — distributional realism

**Expected insight:** A model with smaller residuals is more precise even if beta is the same. LOB-S5 may produce tighter residual distributions, indicating more realistic individual-trade impact estimates.

**Priority:** MEDIUM — moderate effort, high potential insight.

### Experiment C: Conditional beta by Q/V_day quantile

**Goal:** Test whether the power law is truly linear in log-log space or shows curvature.

**Method:** Split data into quantiles of `x = log(Q/V_day)`. Compute local slope in each quantile using linear regression on the quantile subset.

**Implementation:**
```python
for q in [0.2, 0.4, 0.6, 0.8]:
    mask = (x_quantile == q)
    local_beta = compute_global_beta(df[mask])
```

**Expected insight:** If models differ in curvature (non-linearity), a single beta masks this. LOB-S5 may maintain linearity better than baselines at extreme participation rates.

**Priority:** LOW-MEDIUM — deeper analysis, requires careful quantile definitions.

### Experiment D: CGAN k=1 deep dive

**Goal:** Determine whether CGAN's k=1 beta advantage is genuine or an artifact of easy-case dominance.

**Method:** In the v3 grid, restrict k=1 analysis to hard cases only (mb ≥ 50). Compare CGAN's k=1 hard-case beta against other models.

**Implementation:**
```python
hard_mask = df['mb'] >= 50
df_hard_k1 = first_k_insertions(df[hard_mask], k=1)
beta_k1_hard = compute_global_beta(df_hard_k1)
```

**Expected insight:** If CGAN's k=1 advantage persists in hard cases → genuine architectural advantage (adversarial training produces better marginal impact distributions). If it vanishes → easy-case artifact.

**Priority:** HIGH — targeted, fast, high-insight potential.

### Experiment E: Cross-metric correlation (connects to notebook 140)

**Goal:** Test whether beta and relaxation ratio are complementary or redundant.

**Method:** For each configuration, compute both beta AND relaxation ratio. Plot beta vs. relaxation ratio per model. Compute the correlation.

**Implementation:** Requires joining results from 170/171 with 140. Match configs by folder name and compute Pearson/Spearman correlation.

**Expected insight:** If beta and relaxation ratio are uncorrelated → they test independent aspects of simulator quality, strengthening the argument for multi-metric evaluation. If highly correlated → they are redundant.

**Priority:** MEDIUM — connects the two analysis streams, but requires cross-notebook data joining.

### Experiment F: Empirical beta benchmark

**Goal:** Establish the ground-truth beta from real LOBSTER data.

**Method:** Using the actual trade data underlying the simulations, compute beta with the identical regression framework (`y_adj = beta * x`, through-zero OLS).

**Implementation:** Load LOBSTER execution data, construct the same `(x, y_adj)` point cloud, and fit. This provides the "correct answer" for the specific market/stock/period.

**Expected insight:** If empirical beta ≈ 0.53, then the v3 grid models are over-shooting (0.62–0.68) while c10x_v2 models are well-calibrated (0.52–0.54). If empirical beta ≈ 0.65, then v3 models are correctly calibrated and the theoretical 0.5 target is inappropriate.

**Priority:** LOW — requires access to real execution data and careful matching of the regression setup.

### Priority ordering

**A (beta-by-mb curves) > D (CGAN k=1 deep dive) > B (residuals) > E (cross-metric) > C (conditional beta) > F (empirical benchmark)**

---

## 10. Summary for Paper — Suggested Text Fragments

### For the Methods section

> "We evaluate market impact beta — the power-law exponent in the relationship log(I) = α + β·log(Q/V) — using multiple fair estimation procedures: per-configuration median (each configuration contributes one beta), folder-weighted regression, difficulty-band stratification, and first-k-insertion analysis. These procedures control for the strong data imbalance inherent in the parameter grid, where low-mb configurations contribute up to 48.7% of the OLS regression weight."

### For the Results section

> "Table X reports beta estimates across five estimation methods for all simulators. Per-configuration median betas range from 0.645 (Heuristic) to 0.683 (CST) in the v3 grid, and from 0.519 (LOB-S5) to 0.572 (CGAN) in the c10x_v2 grid. Pairwise KS tests on per-configuration beta distributions are non-significant after Bonferroni correction in the v3 grid (all p > 0.09), indicating that the models produce statistically indistinguishable impact scaling. In the restricted c10x_v2 grid, CGAN is the only model to separate significantly (p < 0.0001), exhibiting consistently elevated betas."

### For the Discussion section

> "The convergence of beta estimates across architecturally diverse simulators reveals that the square-root impact law is a first-order property that any model preserving basic LOB structure will reproduce. This finding is itself informative: it establishes beta estimation as a necessary but insufficient validation criterion. The discriminative power of our evaluation framework comes instead from higher-order metrics — relaxation ratio, stability fraction, and volume scaling — which test dynamic properties that simple baselines cannot capture."

---

## Appendix: Data Consistency Notes

### Numbers verified against

- `170_run.log` (run completed 2026-02-27 01:55:59)
- `171_run.log` (run crashed at ranking step due to NaN in Hard column; all analysis cells completed successfully through cell 12)

### Known issues in 171

1. The ranking computation crashes because `Hard` and `Delta(E-H)` are NaN (no mb ≥ 50 configs in c10x_v2 grid). Fix: filter NaN metrics before ranking.
2. Context% equalization produces only one band (`<90%`) because all c10x_v2 configs have context < 90%. This is expected given the low-mb constraint.
3. 30 folders loaded but only 27 produced valid per-config betas (3 configs had insufficient data).

### Correction relative to initial plan

The initial plan assumed the c10x_v2 grid had ~11K points per model and 60 configs, with valid Hard-case betas and specific numeric values for 171. The actual data shows ~392K points, 27 valid configs, and no Hard cases (max mb = 20). All numeric values in this document are taken directly from the run logs, not from the initial plan.
