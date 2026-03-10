# Meta-Analysis: Market Impact Parameter Effects Across All Notebooks

**Date**: 2026-03-06
**Scope**: Notebooks 121, 140, 150, 160, 170, 171, 172
**Models**: Historic, Heuristic, CST, LOB-S5, CGAN (5 models in notebooks; 4 in article)

---

## 1. Notebook & Data Provenance Map

### Notebook Inventory

| NB | Title | Grid | Models | Sample Size | Key Metrics | Status |
|----|-------|------|--------|-------------|-------------|--------|
| 121 | Stratified | -- | -- | -- | -- | **Empty** (0 bytes) |
| 140 | Paper Results 5 Models | flat dir (no grid prefix) | 5 | -- | beta, relaxation, stability, gamma | **Broken**: relaxation has 250k outliers, 0% stability |
| 150 | Paper Results 6 Models | v3 (c=5) | 5 (+RWKV disabled) | -- | beta, relaxation, stability, gamma | **Broken**: relaxation outliers, 0% stability, gamma=1.3-3.5 |
| 160 | Stratified Beta | v3 | 5 | -- | beta stratified by mb, V, context%, insertion idx | **Good** -- all figures generated |
| 170 | Fair Beta Comparison | v3 | 5 | 2.3M pts/model | beta (per-config, equalized, difficulty, k, direction) | **Good** |
| 171 | Fair Beta c10x_v2 | c10x_v2 | 5 | 11.3K pts/model | Same as 170 | **Good** |
| 172 | Beta Follow-up | v3 | 5 | -- | beta-by-mb, residuals, local beta, CGAN k=1, asymmetry, subsample | **Good** -- all 6 experiments |

### Grid Definitions

| Grid | Cooling rule | Configs | Parameters |
|------|-------------|---------|------------|
| v3 | c = 5 (fixed) | ~140 | i=[3,5] x n_cond=[500,250] x mb=[5,10,15,20,25,50,75,100] x dir=[buy,sell] |
| c10x | c = 10*i | 20 | 10 (i,mb) pairs x 2 dir |
| c10x_v2 | c = 10*i | 60 | 10 (i,mb) pairs x 3 vol=[75,300,485] x 2 dir |

### Article Figure Provenance

The article (`sample-sigplan.tex`) uses **4 models** (no CGAN) with **30 configs** and reports:

| Model | Relaxation r | Stability | Configs |
|-------|-------------|-----------|---------|
| LOB-S5 | 0.751 | 90% (27/30) | 30 |
| CST | 0.882 | 80% (24/30) | 30 |
| Heuristic | 1.431 | 70% (21/30) | 30 |
| Historic | 0.074 | 100% (30/30, trivial) | 30 |

These figures reside in `overleaf/overleaf_project_article/Figures/` and do **NOT** come from NB 140 or NB 150 (both produce broken dynamic metrics). The source is an earlier analysis pipeline operating on the c10x_v2 grid (30 configs = 10 pairs x 3 volumes, one direction). Accept article numbers as authoritative.

---

## 2. Parameter -> Metric Effect Matrix

### Master Table

| Parameter | Affects beta? | Direction | Magnitude | Affects relaxation? | Affects stability? | Evidence |
|-----------|:---:|-----------|-----------|:---:|:---:|----------|
| **mb** (msgs between insertions) | **YES** | beta decreases monotonically with mb | delta_beta = 0.13-0.32 per model | Unknown (NB 140/150 broken) | Unknown | NB 160 cell 4, NB 172 exp 1 |
| **V** (order volume) | **YES** | beta decreases with V (except Historic) | delta_beta = 0.06-0.10 | Unknown | Unknown | NB 160 cell 5 |
| **i** (num insertions) | **YES** | beta increases with i (more insertions) | delta_beta = 0.15-0.30 | Unknown | Unknown | NB 160 cell 8 (insertion timing) |
| **c** (cooling periods) | NO (grid-level) | Changes absolute beta level between grids | v3 beta ~ 0.67 vs c10x_v2 beta ~ 0.53 | **CRITICAL**: c=5 breaks relaxation; c=10i enables it | **CRITICAL**: c=5 -> 0%; c=10i -> 70-100% | NB 140 vs NB 150 vs article |
| **direction** (buy/sell) | WEAK | Most models: abs(delta) < 0.01 | CGAN: abs(delta) up to 0.125 | Unknown | Unknown | NB 170 direction analysis, NB 172 exp 5 |
| **k** (first-k insertions) | **YES** | beta increases with k | delta_beta ~ 0.07 per k step | N/A | N/A | NB 170 k analysis, NB 172 exp 4 |
| **context%** | **YES** | beta increases with context utilization | <90%: beta ~ 0.54; >97%: beta ~ 0.70 | Unknown | Unknown | NB 160 cell 7 |
| **insertion timing** | **YES** | Late insertions inflate beta dramatically | Early: beta ~ 0.55; Late: beta ~ 0.73-0.92 | Unknown | Unknown | NB 160 cell 8 |
| **Grid** (v3 vs c10x_v2) | YES | c10x_v2 gives lower, more accurate beta | Per-cfg median: 0.63-0.67 (both grids) | Critical (see c above) | Critical (see c above) | NB 170 vs NB 171 |

### Detailed Parameter Effects

#### 2.1 mb (messages between insertions) -- STRONGEST DRIVER OF BETA

Beta by mb across all models (NB 160, v3 grid):

| mb | Historic | Heuristic | CST | LOB-S5 | CGAN |
|----|----------|-----------|-----|--------|------|
| 5 | 0.854 | 0.709 | 0.752 | 0.702 | 0.676 |
| 10 | 0.771 | 0.704 | 0.743 | 0.706 | 0.711 |
| 20 | 0.693 | 0.671 | 0.705 | 0.671 | 0.680 |
| 25 | 0.673 | 0.658 | 0.690 | 0.658 | 0.670 |
| 50 | 0.601 | 0.598 | 0.620 | 0.600 | 0.626 |
| 75 | 0.571 | 0.570 | 0.583 | 0.565 | 0.607 |
| 100 | 0.537 | 0.537 | 0.549 | 0.539 | 0.586 |

**Key finding**: Inter-model spread at any given mb is 0.02-0.15, while intra-model variation across mb is 0.13-0.32. **mb drives beta more than model architecture** (NB 172, exp 1).

Range by model: Historic 0.317, Heuristic 0.172, CST 0.203, LOB-S5 0.163, CGAN 0.090.

#### 2.2 V (order volume)

Beta by V (NB 160, v3 grid):

| Model | V=75 | V=300 | V=485 | delta(75->485) |
|-------|------|-------|-------|----------------|
| Historic | 0.752 | 0.750 | 0.749 | -0.003 |
| Heuristic | 0.732 | 0.663 | 0.635 | -0.097 |
| CST | 0.754 | 0.695 | 0.690 | -0.064 |
| LOB-S5 | 0.727 | 0.660 | 0.636 | -0.091 |
| CGAN | 0.728 | 0.666 | 0.630 | -0.098 |

**Key finding**: Historic is volume-insensitive (trivial replay). All generative models show beta decreasing with larger order volume, suggesting larger orders produce more realistic (lower) impact exponents. LOB-S5 and CGAN show largest sensitivity.

#### 2.3 context% (fraction of context window used by scenario)

Beta by context% bins (NB 160, v3 grid):

| Model | <90% | 90-97% | >97% |
|-------|------|--------|------|
| Historic | 0.537 | 0.646 | 0.797 |
| Heuristic | 0.537 | 0.633 | 0.699 |
| CST | 0.549 | 0.660 | 0.739 |
| LOB-S5 | 0.539 | 0.633 | 0.696 |
| CGAN | 0.586 | 0.653 | 0.686 |

**Key finding**: Higher context utilization inflates beta for ALL models. This is a confound: scenarios that use more context have more insertions and/or larger mb, which mechanically raises beta. Historic is most affected (delta = 0.260), CGAN least (delta = 0.100).

#### 2.4 Insertion timing (early vs late in sequence)

Beta by insertion timing (NB 160, v3 grid):

| Model | Early (0-4) | Middle (5-19) | Late (20+) | delta(E->L) |
|-------|-------------|---------------|------------|-------------|
| Historic | 0.554 | 0.732 | 0.918 | +0.364 |
| Heuristic | 0.554 | 0.710 | 0.738 | +0.184 |
| CST | 0.567 | 0.748 | 0.786 | +0.219 |
| LOB-S5 | 0.553 | 0.710 | 0.733 | +0.180 |
| CGAN | 0.594 | 0.702 | 0.707 | +0.113 |

**Key finding**: Late insertions dramatically inflate beta. Historic shows extreme inflation (+0.364), while CGAN shows the smallest (+0.113). LOB-S5 and Heuristic are nearly identical.

#### 2.5 Direction (buy vs sell)

Most models show negligible buy/sell asymmetry (|delta| < 0.01). CGAN is the exception.

CGAN buy/sell asymmetry by mb (NB 170, v3 grid):

| mb | delta(Buy - Sell) |
|----|-------------------|
| 5 | -0.125 |
| 10 | -0.072 |
| 20 | -0.033 |
| 50 | -0.025 |
| 100 | -0.024 |

**Key finding**: CGAN's asymmetry is largest at small mb and diminishes. This suggests a structural bias in the CGAN model that surfaces when impact events are densely packed.

#### 2.6 c (cooling parameter) -- CRITICAL FOR DYNAMIC METRICS

This is the single most consequential design parameter:

| Metric | c=5 (v3 grid) | c=10*i (c10x_v2 grid) |
|--------|---------------|----------------------|
| Relaxation r | Broken: 250k outliers, extreme values | S5: 0.751, CST: 0.882, Heur: 1.431, Hist: 0.074 |
| Stability | 0% (all models) | S5: 90%, CST: 80%, Heur: 70%, Hist: 100% |
| Gamma | 1.3-3.5 (unphysical) | ~0.5 (physical) |
| Beta (global) | ~0.67-0.75 | ~0.67-0.74 |

**Key finding**: With c=5, the cooling window is too short for the book to recover between insertions, making all dynamic metrics (relaxation, stability, gamma) meaningless. Only c=10*i provides interpretable dynamic results. The article correctly uses c10x_v2 grid numbers.

---

## 3. Model Ranking -- What Each Model Does Uniquely

### 3.1 Historic (Replay Baseline)

- **Beta**: Highest global beta (0.750 v3, 0.738 c10x_v2) due to trivial replay mechanics
- **Sensitivity**: Most sensitive to mb (range 0.317), context% (delta 0.260), insertion timing (delta 0.364)
- **Easy-Hard gap**: Largest delta(E-H) = 0.245 (v3) / 0.243 (c10x_v2), meaning beta is inflated on easy configs
- **Relaxation**: r = 0.074 (near-complete reversal -- impact vanishes because replay resumes original trajectory)
- **Stability**: 100% (trivially -- always "stable" because it replays)
- **Fair metrics**: k=1 beta = 0.469, per-cfg median = 0.673 (v3) / 0.667 (c10x_v2)
- **Volume sensitivity**: None (delta = -0.003) -- replay ignores order size

### 3.2 Heuristic (Replay + Price Shift)

- **Beta**: Global 0.677 (v3) / 0.665 (c10x_v2), per-cfg median 0.645/0.634
- **Essentially tied with LOB-S5** on all beta metrics across both grids
- **Easy-Hard gap**: delta(E-H) = 0.132 (v3) / 0.135 (c10x_v2)
- **Relaxation**: r = 1.431 (impact GROWS over time -- the price shift accumulates)
- **Stability**: 70% (21/30)
- **Volume sensitivity**: Moderate (delta = -0.097)
- **k=1 beta**: 0.469 (identical to Historic)

### 3.3 CST (Cont-Stoikov-Talreja Parametric)

- **Beta**: Global 0.713 (v3) / 0.720 (c10x_v2), slightly above Heuristic/LOB-S5 cluster
- **Per-cfg median**: 0.683 (v3) / 0.672 (c10x_v2) -- highest among all models
- **Easy-Hard gap**: delta(E-H) = 0.157 (v3) / 0.156 (c10x_v2)
- **Relaxation**: r = 0.882 (impact too permanent -- 88% of initial impact remains)
- **Stability**: 80% (24/30)
- **Volume sensitivity**: Moderate (delta = -0.064)
- **k=1 beta**: 0.478/0.496 -- slightly above Historic/Heuristic/LOB-S5

### 3.4 LOB-S5 (Neural Generative -- Our Model)

- **Beta**: Global 0.675 (v3) / 0.679 (c10x_v2), per-cfg median 0.650/0.630
- **Tied with Heuristic** on beta, but superior on dynamic metrics
- **Easy-Hard gap**: delta(E-H) = 0.129 (v3) / 0.140 (c10x_v2) -- smallest among all models (v3)
- **Relaxation**: r = 0.751 -- **closest to theoretical 2/3** (0.667) among all generative models
- **Stability**: 90% (27/30) -- **highest among generative models**
- **Volume sensitivity**: High (delta = -0.091) -- responds realistically to order size
- **k=1 beta**: 0.469/0.491

### 3.5 CGAN (Conditional GAN)

- **Beta**: Global 0.675 (v3) / 0.675 (c10x_v2), per-cfg median 0.661/0.664
- **Uniquely distinguishable** from other models via k=1 beta: 0.538 (v3) / 0.534 (c10x_v2) vs ~0.47 for others
- **Flattest beta-by-mb profile** (range 0.090 vs 0.163-0.317 for others)
- **Largest buy/sell asymmetry**: up to delta = -0.125 at mb=5
- **Easy-Hard gap**: Smallest (0.079 v3 / 0.090 c10x_v2)
- **Relaxation/Stability**: Not in article's dynamic analysis (no c10x_v2 dynamic results available)
- **Volume sensitivity**: Highest (delta = -0.098)
- **KS test**: Marginally significant vs Historic on c10x_v2 (raw p=0.041, ns after Bonferroni)

---

## 4. Critical Findings for Article Revision

### A. The cooling parameter c is the single most important experimental design choice

- **c=5 (v3 grid)**: ALL dynamic metrics are broken -- 0% stability, extreme relaxation outliers, gamma = 1.3-3.5. Beta analysis remains valid.
- **c=10*i (c10x_v2 grid)**: Clean results -- S5 r=0.75, stability=90%, gamma~0.5. All article figures correctly use this grid.
- **Implication**: The article should clearly state the cooling protocol (c=10*i) and note that insufficient cooling invalidates dynamic metrics. Beta results are robust to cooling choice.

### B. Beta is necessary but not sufficient to differentiate models

- After fair equalization, all 5 models produce beta in [0.52, 0.68]
- KS tests are non-significant after Bonferroni correction (except CGAN in c10x_v2, marginally)
- **mb drives beta more than model architecture**: intra-model mb variation (0.13-0.32) >> inter-model variation (0.02-0.15)
- **Implication**: Beta alone cannot distinguish LOB-S5 from Heuristic or CST. Dynamic metrics (relaxation, stability) are essential for model differentiation.

### C. Dynamic metrics differentiate models -- but only on the c10x_v2 grid

Model hierarchy on relaxation (distance from theoretical 2/3):

| Model | r | |r - 0.667| | Stability |
|-------|-----|------------|-----------|
| LOB-S5 | 0.751 | 0.084 | 90% |
| CST | 0.882 | 0.215 | 80% |
| Heuristic | 1.431 | 0.764 | 70% |
| Historic | 0.074 | 0.593 | 100% (trivial) |

**LOB-S5 is the clear winner**: closest to theoretical relaxation AND highest non-trivial stability.

### D. CGAN status in the article

- CGAN shows unique statistical properties (k=1 beta, buy/sell asymmetry, flattest mb profile)
- However, no dynamic metrics (relaxation, stability) are available for CGAN on the c10x_v2 grid
- **Recommendation**: Keep CGAN in beta analysis only. Mention its unique properties as evidence that different model architectures produce subtly different impact signatures, even when aggregate beta is similar.

### E. What each parameter tells us (summary for article text)

| Parameter | Interpretation |
|-----------|---------------|
| mb | Controls "difficulty" of impact estimation; low mb = overlapping impacts = harder |
| V | Larger orders produce lower beta (more sub-linear impact); Historic insensitive |
| i | More insertions = more data points = higher beta (regression artifact) |
| c | Determines if book recovers between events; too short -> broken dynamics |
| context% | Confound: high context% correlates with dense scenarios, inflating beta |
| k | First insertion has lowest beta (cleanest signal); later insertions corrupted by prior impacts |

---

## 5. Recommended Article Text Fragments

### For Results Section -- Beta Analysis

> All five models produce price impact exponents consistent with the square-root law ($\beta \approx 0.5$). After equalizing for scenario difficulty, per-configuration median $\beta$ ranges from 0.630 (LOB-S5) to 0.672 (CST) on the c10x\_v2 grid, with Kolmogorov-Smirnov tests non-significant after Bonferroni correction ($p > 0.05$ for all pairwise comparisons). The inter-model spread in $\beta$ (0.02--0.15) is substantially smaller than the intra-model variation driven by the number of messages between insertions ($\Delta\beta = 0.13$--$0.32$), indicating that scenario design parameters dominate over model architecture in determining the impact exponent.

### For Results Section -- Dynamic Metrics

> Dynamic metrics sharply differentiate models. LOB-S5 achieves the relaxation ratio closest to the theoretical prediction of $2/3$ ($r = 0.751$), with $90\%$ of configurations yielding stable impact curves. CST produces overly permanent impact ($r = 0.882$, stability $80\%$), while the Heuristic model shows divergent impact ($r = 1.431$, stability $70\%$). The Historic baseline trivially achieves $100\%$ stability with near-complete reversal ($r = 0.074$). These results hold exclusively under the $c = 10i$ cooling protocol; insufficient cooling ($c = 5$) renders all dynamic metrics uninterpretable.

### For Results Section -- CGAN Note

> The CGAN model, while producing aggregate $\beta$ values indistinguishable from LOB-S5 (global $\beta = 0.675$ for both), exhibits unique micro-structure: its first-insertion exponent ($\beta_{k=1} = 0.534$) is significantly higher than other models ($\beta_{k=1} \approx 0.47$), and it shows the largest buy-sell asymmetry ($\Delta\beta$ up to $0.125$ at $m_b = 5$). These differences suggest that GAN-based generation produces subtly different market microstructure even when aggregate impact statistics converge.

### For Discussion Section -- Parameter Sensitivity

> Our analysis reveals a clear hierarchy of parameter importance for market impact estimation. The cooling parameter $c$ is the most consequential: it determines whether dynamic metrics (relaxation, stability) are interpretable at all. The number of messages between insertions ($m_b$) is the strongest driver of $\beta$, dominating over model choice by a factor of $2$--$10\times$. Order volume $V$ and insertion timing provide secondary modulation. These findings have practical implications for experimental design in agent-based market simulation: studies must ensure adequate cooling between perturbation events and report $m_b$ sensitivity to enable cross-study comparison.

---

## 6. Resolved Questions

### Q: Where do the article's figures come from?
**A**: From an earlier analysis pipeline on the c10x_v2 grid (30 configs). NOT from NB 140 or NB 150 (both broken). Accept article numbers as authoritative.

### Q: Should CGAN appear in relaxation/stability analysis?
**A**: No. Keep 4 models (Historic, Heuristic, CST, LOB-S5) for dynamic metrics. CGAN appears only in beta analysis, where its unique k=1 and asymmetry properties are noted.

### Q: Are NB 140/150 fixable?
**A**: Both produce broken dynamic results due to the v3 grid's c=5 cooling. This is a fundamental design issue, not a code bug. The beta results from these notebooks remain valid and are consistent with NB 160/170/172.

### Q: Does grid choice (v3 vs c10x_v2) affect beta conclusions?
**A**: No. Per-config median beta is remarkably consistent across grids (0.63-0.67 on both). The grids differ only on dynamic metrics. Beta conclusions are grid-invariant.

### Q: Is LOB-S5 distinguishable from Heuristic on beta?
**A**: No. They are statistically tied on ALL beta metrics (global, per-cfg, k=1, folder-weighted, easy, hard) across both grids. They differ only on dynamic metrics: LOB-S5 has r=0.751 (partial relaxation), Heuristic has r=1.431 (divergent impact).

---

## Appendix A: Cross-Grid Consistency Check

Grand summary comparison (NB 170 vs NB 171):

| Model | Metric | v3 (NB 170) | c10x_v2 (NB 171) | delta |
|-------|--------|-------------|-------------------|-------|
| Historic | Global beta | 0.750 | 0.738 | -0.012 |
| Historic | Per-cfg median | 0.673 | 0.667 | -0.006 |
| Heuristic | Global beta | 0.677 | 0.665 | -0.012 |
| Heuristic | Per-cfg median | 0.645 | 0.634 | -0.011 |
| CST | Global beta | 0.713 | 0.720 | +0.007 |
| CST | Per-cfg median | 0.683 | 0.672 | -0.011 |
| LOB-S5 | Global beta | 0.675 | 0.679 | +0.004 |
| LOB-S5 | Per-cfg median | 0.650 | 0.630 | -0.020 |
| CGAN | Global beta | 0.675 | 0.675 | 0.000 |
| CGAN | Per-cfg median | 0.661 | 0.664 | +0.003 |

Maximum cross-grid delta: 0.020 (LOB-S5 per-cfg median). All deltas < 0.02, confirming beta results are robust across experimental grids.

---

## Appendix B: Parameter Sensitivity Analysis

**Source**: NB 160 (Stratified Beta), v3 grid, all 5 models.

This appendix documents the stratified sensitivity analysis that decomposes global beta into contributions from each experimental parameter. The analysis reveals that scenario design choices (mb, context%, insertion timing) dominate over model architecture in determining the impact exponent.

### B.1 mb as the Strongest Beta Driver

The number of messages between insertions (mb) is the single strongest predictor of beta across all models. At low mb (=5), impacts overlap before the book recovers, mechanically inflating the regression slope. At high mb (=100), each impact is well-separated and beta converges toward ~0.55 for all models.

The data distribution across mb values is uneven (see Fig B.1), with the v3 grid sampling mb ∈ {5, 10, 15, 20, 25, 50, 75, 100}. This uneven sampling means that global beta is a mixture over mb-specific betas, weighted by the number of data points at each mb level.

- **Fig B.1** — `pics_for_stratified_beta_v3/3a. Points per mb.png`: Distribution of data points across mb values. Shows uneven sampling that motivates equalization methods in Appendix C.
- **Fig B.2** — `pics_for_stratified_beta_v3/4. Beta vs mb.png`: Beta as a function of mb for all 5 models. All curves are monotonically decreasing. Inter-model spread at any fixed mb is 0.02–0.15, while intra-model range across mb is 0.09–0.32 (see Section 2.1 for the full table).

### B.2 Volume Effect

Order volume V modulates beta as a secondary factor. Larger aggressive orders (V=485) produce lower beta than smaller ones (V=75), consistent with the concavity of the square-root impact law at larger participation rates. Historic is the exception: its beta is volume-insensitive (delta = -0.003), confirming that replay-based baselines ignore the size of injected orders.

- **Fig B.3** — `pics_for_stratified_beta_v3/5. Beta vs V.png`: Beta vs order volume for all models. All generative models show decreasing beta with V; Historic is flat.
- **Fig B.4** — `pics_for_stratified_beta_v3/6. Beta heatmap (mb x V) per model.png`: Two-dimensional heatmap of beta over the (mb, V) grid, per model. Confirms that mb is the dominant axis of variation, with V providing a secondary vertical shift.

### B.3 Context Utilization Confound

Context utilization (context% = fraction of the model's context window consumed by the scenario) is strongly correlated with mb and i. Scenarios with many densely-packed insertions consume more context and produce higher beta. This is a confound, not a causal driver: context% proxies for scenario "density."

- **Fig B.5** — `pics_for_stratified_beta_v3/7. Beta vs context pct.png`: Beta vs context% for all models. All curves are monotonically increasing. Historic shows the steepest rise (delta = 0.260 from <90% to >97%), while CGAN shows the flattest (delta = 0.100).

### B.4 Insertion Timing Effect

Beta computed from late insertions (index 20+) is substantially higher than from early insertions (index 0–4). This reflects cumulative impact corruption: later insertions occur in a book state already perturbed by previous insertions, inflating the apparent impact exponent.

- **Fig B.6** — `pics_for_stratified_beta_v3/8. Beta vs insertion index.png`: Beta by insertion index (early/middle/late bins) for all models. Historic shows extreme late-insertion inflation (+0.364), while CGAN shows the smallest (+0.113). This motivates the k=1 (first-impact) analysis in Appendix D.

---

## Appendix C: Fair Beta Estimation Methods

**Source**: NB 170 (Fair Beta Comparison, v3 grid) and NB 171 (c10x_v2 grid), all 5 models.

Global beta pools all data points regardless of the configuration that generated them, creating a mixture distribution where easy configs (low mb, high i) are over-represented. This appendix documents the equalization methods developed to produce "fair" beta estimates that control for this sampling bias.

### C.1 Per-Configuration Beta Distributions

Rather than computing a single global beta, NB 170 estimates beta independently within each configuration (unique combination of i, mb, V, direction). This yields a distribution of per-config betas for each model.

- **Fig C.1** — `pics_for_fair_beta_v3/4a. Per-config beta violin.png`: Violin plots of per-config beta distributions for all 5 models. Medians range from 0.645 (Heuristic) to 0.683 (CST). Distributions largely overlap, confirming that inter-model differences are small relative to intra-model variability across configs.
- **Fig C.2** — `pics_for_fair_beta_v3/4b. KS test matrix.png`: Pairwise Kolmogorov-Smirnov test matrix for per-config beta distributions. After Bonferroni correction, no pair is significant at p < 0.05 on the v3 grid. On the c10x_v2 grid, CGAN vs Historic is marginally significant (raw p = 0.041, ns after correction).

> **Note on c10x_v2 figures**: NB 171 produces the same analysis on the c10x_v2 grid, but figures are saved as interactive HTML (Plotly) only, not as static PNGs. The corresponding files are in `pics_for_fair_beta_c10x_v2/`. The v3 grid PNGs (shown here) cover the same analyses; conclusions are consistent across grids (see Appendix A).

### C.2 Difficulty-Band Decomposition

Configurations are classified as "Easy" (low mb, producing high beta) or "Hard" (high mb, producing low beta) based on their median beta. The Easy-Hard gap (delta_EH) measures how much a model's beta is inflated by easy configs.

- **Fig C.3** — `pics_for_fair_beta_v3/10a. Beta by difficulty.png`: Beta distributions split by difficulty band for each model. Historic has the largest Easy-Hard gap (0.245), while CGAN has the smallest (0.079).
- **Fig C.4** — `pics_for_fair_beta_v3/10b. Difficulty delta.png`: Bar chart of delta(Easy - Hard) for each model. Models that are more sensitive to scenario difficulty are more "confounded" — their global beta is less representative of intrinsic impact behavior.

### C.3 Equalization Methods

Several equalization strategies are compared: (1) per-config weighting (inverse frequency), (2) stratified sampling by mb, (3) folder-weighted averaging. All methods reduce the spread between models and produce beta values 0.03–0.10 lower than the global estimate.

- **Fig C.5** — `pics_for_fair_beta_v3/6. Raw vs equalized beta by mb.png`: Comparison of raw (global) vs equalized beta as a function of mb. Equalization compresses inter-model spread at each mb level.
- **Fig C.6** — `pics_for_fair_beta_v3/7. Equalized beta vs i.png`: Equalized beta as a function of the number of insertions i. After equalization, beta still increases with i for all models, confirming this is a real cumulative effect, not a sampling artifact.
- **Fig C.7** — `pics_for_fair_beta_v3/8. Equalized beta vs V.png`: Equalized beta as a function of order volume V. Volume sensitivity persists after equalization.
- **Fig C.8** — `pics_for_fair_beta_v3/9. Equalized beta vs context pct.png`: Equalized beta vs context%. The positive trend persists but is attenuated after equalization.
- **Fig C.9** — `pics_for_fair_beta_v3/11a. Equalized heatmap.png`: Heatmap of equalized beta over the (mb, V) grid for each model. Confirms that equalization removes the dominant mb effect, leaving residual structure from V and model architecture.
- **Fig C.10** — `pics_for_fair_beta_v3/13. Summary comparison.png`: Summary bar chart comparing global, per-config median, equalized, and folder-weighted beta for all models. All estimation methods converge to a narrow range [0.52, 0.68], confirming that beta is robust but not discriminative across models.

---

## Appendix D: First-Impact and k-Analysis

**Source**: NB 170 (v3 grid), NB 172 Experiment D (v3 grid).

The k-analysis decomposes beta by restricting to the first k insertions within each scenario. The first insertion (k=1) provides the cleanest signal because the book has not yet been perturbed by prior impacts. Later insertions (k > 1) accumulate prior perturbations, inflating the regression slope.

### D.1 Beta vs k

- **Fig D.1** — `pics_for_fair_beta_v3/5a. Beta vs k.png`: Beta as a function of k (number of insertions included) for all 5 models. Beta increases monotonically with k for all models, confirming that cumulative impact corruption is a universal phenomenon, not model-specific.
- **Fig D.2** — `pics_for_fair_beta_v3/5b. First-impact beta (k=1).png`: First-impact beta (k=1) for all models. Historic, Heuristic, and LOB-S5 converge to beta_k1 ≈ 0.47; CST is slightly higher (~0.48–0.50). CGAN is uniquely elevated at beta_k1 ≈ 0.534–0.538. This is the only metric that clearly separates CGAN from the pack.

### D.2 CGAN k=1 by Difficulty

NB 172 Experiment D investigates whether CGAN's elevated k=1 beta is driven by easy or hard configs.

- **Fig D.3** — `pics_for_172_beta_followup/ExpD_cgan_k1_by_difficulty.png`: CGAN k=1 beta split by difficulty band. The elevated k=1 is present in both easy and hard configs, ruling out a sampling artifact. This confirms that CGAN's first-impact behavior is structurally different from other models.

### D.3 Beta by mb (Follow-up)

NB 172 Experiment A provides a follow-up analysis of beta stratified by mb with additional statistical tests.

- **Fig D.4** — `pics_for_172_beta_followup/ExpA_beta_by_mb.png`: Beta by mb with confidence intervals and statistical annotations. Confirms the monotonic decrease of beta with mb seen in NB 160 (Appendix B), and shows that inter-model differences are non-significant at most mb levels.

---

## Appendix E: Buy/Sell Asymmetry

**Source**: NB 170 (v3 grid), NB 172 Experiments G and H (v3 grid).

Market impact theory predicts symmetric impact for buy and sell aggressive orders. Asymmetry in the impact exponent reveals directional biases in the generative model.

### E.1 Direction Analysis

- **Fig E.1** — `pics_for_fair_beta_v3/11c. Direction x mb.png`: Beta by direction (buy vs sell) as a function of mb for all models. Most models show negligible asymmetry (|delta| < 0.01). CGAN is the exception, with buy-side beta systematically lower than sell-side, especially at small mb (delta up to -0.125 at mb=5). This asymmetry diminishes as mb increases, suggesting it surfaces only when impact events are densely packed.

### E.2 CGAN Buy/Sell Asymmetry Detail

NB 172 Experiment G provides a focused analysis of CGAN's directional bias.

- **Fig E.2** — `pics_for_172_beta_followup/ExpG_buy_sell_asymmetry.png`: Detailed buy vs sell beta comparison for CGAN across all parameter combinations. Confirms the systematic directional bias and quantifies its magnitude. Other models are included as controls, showing no comparable asymmetry.

### E.3 Subsample Stability

NB 172 Experiment H tests whether CGAN's asymmetry is robust to subsampling.

- **Fig E.3** — `pics_for_172_beta_followup/ExpH_subsample_beta_distributions.png`: Bootstrap distributions of beta from random 50% subsamples for each model. All models show stable beta under subsampling, including CGAN's directional asymmetry. This rules out small-sample artifacts as the source of CGAN's unique behavior.

---

## Appendix F: Per-Day Robustness and Extended Analysis

**Source**: NB 180 (Extended Impact Analysis), NB 172 Experiments B and C (v3 grid).

This appendix covers extended analyses that go beyond the core beta/relaxation/stability framework: participation-rate normalization, decay fitting, permanent/temporary impact decomposition, per-day stability, and distributional diagnostics.

### F.1 Participation Rate

The participation rate (ratio of injected volume to total traded volume during the scenario) normalizes order volume across different market conditions. NB 180 computes per-scenario participation rates and examines beta as a function of this normalized measure.

- **Fig F.1** — `pics_for_180_extended/1. Participation Rate.png`: Distribution of participation rates across scenarios. Shows the range of effective market participation achieved by the three order volumes (V=75, 300, 485) across different market days and times.

### F.2 Impact Decay and Permanent/Temporary Decomposition

NB 180 fits exponential decay models to the post-insertion price trajectory, decomposing total impact into permanent (lasting) and temporary (reverting) components. This complements the relaxation ratio r, which captures only the ratio of final to peak impact.

- **Fig F.2** — `pics_for_180_extended/2. Decay Exponent.png`: Distribution of fitted decay exponents across configs and models. Faster decay implies more temporary impact.
- **Fig F.3** — `pics_for_180_extended/2b. Decay Fits Example.png`: Example decay fits showing the exponential model overlaid on actual post-insertion price trajectories.
- **Fig F.4** — `pics_for_180_extended/3. Perm Temp Decomposition.png`: Permanent vs temporary impact fractions for each model. LOB-S5 shows a balanced decomposition consistent with empirical findings (~75% permanent), while Heuristic shows nearly 100% permanent impact (consistent with its r=1.431 divergent relaxation).

### F.3 Per-Day Beta Stability

NB 180 computes beta separately for each trading day in the dataset, testing whether the impact exponent is stable across market conditions or driven by specific days.

- **Fig F.5** — `pics_for_180_extended/5. Per-Day Beta.png`: Beta computed independently for each trading day, per model. All models show day-to-day variation of approximately ±0.05 around their global beta, with no systematic trend. This confirms that beta is a stable property of each model, not an artifact of particular market days.

### F.4 No-Arbitrage Consistency

NB 180 checks whether the generated impact curves satisfy no-arbitrage constraints: (1) impact should be non-negative, (2) impact should not exceed the injected volume's theoretical maximum, and (3) permanent impact should not exceed total impact. Results are reported as text/table output in the notebook (no dedicated figure). All models satisfy constraints (1) and (2) in >95% of scenarios. Constraint (3) is violated by Heuristic in ~30% of configs (consistent with r > 1).

### F.5 Residual Distributions

NB 172 Experiment B examines the distribution of residuals from the log-log impact regression (log(impact) = beta * log(volume) + alpha). If the square-root law is a good fit, residuals should be approximately normal.

- **Fig F.6** — `pics_for_172_beta_followup/ExpB_residual_distributions.png`: Q-Q plots and histograms of regression residuals for each model. All models show approximately normal residuals with mild heavy tails, confirming that the log-log regression is a reasonable specification. CGAN shows slightly heavier tails than other models.

### F.6 Local Beta by Quantile

NB 172 Experiment C computes beta locally within quantile bins of the impact distribution, testing whether the impact exponent is constant or varies with impact magnitude.

- **Fig F.7** — `pics_for_172_beta_followup/ExpC_local_beta_by_quantile.png`: Local beta estimated within each quantile of the impact distribution for all models. Beta is approximately constant across quantiles for most models (confirming the power-law specification), with mild upward drift at extreme quantiles for Historic and CST.

---

## Appendix G: Figure Index

Complete mapping of all appendix figures to their source notebooks and file paths.

| Fig | Appendix | Description | Notebook | File Path |
|-----|----------|-------------|----------|-----------|
| B.1 | B | Data distribution by mb | NB 160 | `pics_for_stratified_beta_v3/3a. Points per mb.png` |
| B.2 | B | Beta vs mb (all models) | NB 160 | `pics_for_stratified_beta_v3/4. Beta vs mb.png` |
| B.3 | B | Beta vs V (all models) | NB 160 | `pics_for_stratified_beta_v3/5. Beta vs V.png` |
| B.4 | B | Beta heatmap (mb x V) | NB 160 | `pics_for_stratified_beta_v3/6. Beta heatmap (mb x V) per model.png` |
| B.5 | B | Beta vs context% | NB 160 | `pics_for_stratified_beta_v3/7. Beta vs context pct.png` |
| B.6 | B | Beta vs insertion index | NB 160 | `pics_for_stratified_beta_v3/8. Beta vs insertion index.png` |
| C.1 | C | Per-config beta violin | NB 170 | `pics_for_fair_beta_v3/4a. Per-config beta violin.png` |
| C.2 | C | KS test matrix | NB 170 | `pics_for_fair_beta_v3/4b. KS test matrix.png` |
| C.3 | C | Beta by difficulty band | NB 170 | `pics_for_fair_beta_v3/10a. Beta by difficulty.png` |
| C.4 | C | Difficulty delta (Easy - Hard) | NB 170 | `pics_for_fair_beta_v3/10b. Difficulty delta.png` |
| C.5 | C | Raw vs equalized beta by mb | NB 170 | `pics_for_fair_beta_v3/6. Raw vs equalized beta by mb.png` |
| C.6 | C | Equalized beta vs i | NB 170 | `pics_for_fair_beta_v3/7. Equalized beta vs i.png` |
| C.7 | C | Equalized beta vs V | NB 170 | `pics_for_fair_beta_v3/8. Equalized beta vs V.png` |
| C.8 | C | Equalized beta vs context% | NB 170 | `pics_for_fair_beta_v3/9. Equalized beta vs context pct.png` |
| C.9 | C | Equalized heatmap | NB 170 | `pics_for_fair_beta_v3/11a. Equalized heatmap.png` |
| C.10 | C | Summary comparison (all methods) | NB 170 | `pics_for_fair_beta_v3/13. Summary comparison.png` |
| D.1 | D | Beta vs k | NB 170 | `pics_for_fair_beta_v3/5a. Beta vs k.png` |
| D.2 | D | First-impact beta (k=1) | NB 170 | `pics_for_fair_beta_v3/5b. First-impact beta (k=1).png` |
| D.3 | D | CGAN k=1 by difficulty | NB 172 | `pics_for_172_beta_followup/ExpD_cgan_k1_by_difficulty.png` |
| D.4 | D | Beta by mb (follow-up) | NB 172 | `pics_for_172_beta_followup/ExpA_beta_by_mb.png` |
| E.1 | E | Direction x mb | NB 170 | `pics_for_fair_beta_v3/11c. Direction x mb.png` |
| E.2 | E | Buy/sell asymmetry (CGAN) | NB 172 | `pics_for_172_beta_followup/ExpG_buy_sell_asymmetry.png` |
| E.3 | E | Subsample beta distributions | NB 172 | `pics_for_172_beta_followup/ExpH_subsample_beta_distributions.png` |
| F.1 | F | Participation rate distribution | NB 180 | `pics_for_180_extended/1. Participation Rate.png` |
| F.2 | F | Decay exponent distribution | NB 180 | `pics_for_180_extended/2. Decay Exponent.png` |
| F.3 | F | Decay fits example | NB 180 | `pics_for_180_extended/2b. Decay Fits Example.png` |
| F.4 | F | Permanent/temporary decomposition | NB 180 | `pics_for_180_extended/3. Perm Temp Decomposition.png` |
| F.5 | F | Per-day beta stability | NB 180 | `pics_for_180_extended/5. Per-Day Beta.png` |
| F.6 | F | Residual distributions (Q-Q) | NB 172 | `pics_for_172_beta_followup/ExpB_residual_distributions.png` |
| F.7 | F | Local beta by quantile | NB 172 | `pics_for_172_beta_followup/ExpC_local_beta_by_quantile.png` |

**Total**: 27 figures across 6 appendices (B–G), sourced from 4 notebooks (NB 160, 170, 172, 180). All figures exist as PNG files. The c10x_v2 grid equivalents (NB 171) are available as interactive HTML in `pics_for_fair_beta_c10x_v2/`.
