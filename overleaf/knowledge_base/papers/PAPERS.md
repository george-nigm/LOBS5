# σ₀ — Publications & Papers

*Reverse chronological order — newest first*

*Для proposal: этот список — полная публикационная история проекта σ₀. Все статьи связаны единой линией: foundation model для limit order book → симулятор → бенчмарк → scaling → downstream tasks (execution, market making, forecasting).*

---

## Published / Accepted

### 7. Painting the Market: Generative Diffusion Models for Financial Limit Order Book Simulation and Forecasting
**Authors:** A Backhouse, K Li, J Foerster, A Calinescu, S Zohren
**Venue:** arXiv 2025
**Citations:** —
**Link:** https://arxiv.org/abs/2509.05107
**PDF:** `7_Painting-the-Market_arXiv2025.pdf`
**Summary:** Diffusion-based generative model (DiT) for LOB simulation and mid-price forecasting. Two-stage classification head: flat vs directional movement. Complementary approach to autoregressive S5 — addresses mode collapse in flat-dominated data (~90%). Explores bidirectional refinement via denoising.
**σ₀ relevance:** DiT/DFM workstream (Aramis). Backup approach if compound error confirmed in autoregressive generation.

---

### 6. JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading
**Authors:** V Mohl, S Frey, R Leyland, K Li, **G Nigmatulin**, M Cucuringu, S Zohren, A Calinescu, J Foerster
**Venue:** 🏆 **[Best Paper] ICAIF 2025**
**Citations:** 1
**Link:** https://arxiv.org/abs/2511.02136
**PDF:** `6_JaxMARL-HFT_ICAIF2025_bestpaper.pdf`
**Summary:** GPU-accelerated multi-agent RL environment for HFT built on JAX-LOB. Enables large-scale parallel training of trading agents in realistic order book environments. Foundation for EGGROLL execution experiments.
**σ₀ relevance:** Direct infrastructure for EGGROLL (Valentin) and market making RL (Aramis). George is co-author.

---

### 5. Discrete Flow Matching is a Surprisingly Effective Post-training Method to Address Compound Error in Autoregressive Models
**Authors:** K Li, B Sarkar, Z Xiong, S Frey, Z Wang, F Zejnullahu, A Backhouse, ...
**Venue:** 🎤 **[Oral] ICAIF 2025**
**Citations:** 1
**Link:** https://dl.acm.org/doi/10.1145/3768292.3770442
**PDF:** `5_DFM-Compound-Error_ICAIF2025_oral.pdf`
**Summary:** Discrete Flow Matching (DFM) as post-training correction for compound error in autoregressive sequence generation. Shows bidirectional refinement can fix distribution drift without retraining from scratch. Key question for σ₀: is compound error real for LOBS5 at 360M+ scale?
**σ₀ relevance:** Core DFM workstream. If compound error confirmed on 360M model → apply DFM to LOBS5. Currently P3 priority (backup).

---

### 4. Mixtures of Experts for Scaling Up Neural Networks in Order Execution
**Authors:** K Li, M Cucuringu, L Sánchez-Betancourt, T Willi
**Venue:** ICAIF 2024
**Citations:** 5
**Link:** https://dl.acm.org/doi/10.1145/3677052.3698691
**PDF:** `4_MoE-Order-Execution_ICAIF2024.pdf`
**Summary:** MoE architecture for scaling order execution models. 16 experts per layer, activate 2 → enables 3B active / 30B total parameters. Demonstrates efficient scaling for financial sequence models.
**σ₀ relevance:** MoE/lobmax workstream (Kang). MaxText-based architecture now core infrastructure ("no S5 inside"). MFU 20–50%.

---

### 3. LOB-Bench: Benchmarking Generative AI for Finance, an Application to Limit Order Book Data
**Authors:** P Nagy, S Frey, K Li, B Sarkar, S Vyetrenko, S Zohren, A Calinescu, ...
**Venue:** **ICML 2025**
**Citations:** 5
**Link:** https://arxiv.org/abs/2502.09172
**PDF:** `3_LOB-Bench_ICML2025.pdf`
**Summary:** Universal evaluation suite for generative LOB models. Metrics: L1 distance, Wasserstein, BTS distance, impact curves, time-lagged conditional evals. Establishes standardised comparison across all model variants.
**σ₀ relevance:** LOB-Bench workstream (Satyam, Sascha). P0 priority — every model change evaluated through LOB-Bench. PR with extended metrics ready to submit.

---

### 2. JAX-LOB: A GPU-Accelerated Limit Order Book Simulator to Unlock Large Scale Reinforcement Learning for Trading
**Authors:** SY Frey, K Li, P Nagy, S Sapora, C Lu, S Zohren, J Foerster, A Calinescu
**Venue:** 🏆 **[Best Paper] ICAIF 2023**
**Citations:** 34
**Link:** https://arxiv.org/abs/2308.13289
**PDF:** `2_JAX-LOB_ICAIF2023_bestpaper.pdf`
**Summary:** Vectorised limit order book matching engine in JAX. Enables massively parallel LOB simulation on GPU for RL training. Core simulation infrastructure for all downstream experiments.
**σ₀ relevance:** Foundation infrastructure. All execution (EGGROLL), market making, and evaluation runs through JAX-LOB. Known bottleneck: D2H transfers, SIMD mismatch (XProf profiling Jan 16).

---

### 1. Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network
**Authors:** P Nagy, S Frey, S Sapora, K Li, A Calinescu, S Zohren, J Foerster
**Venue:** 🎤 **[Oral] ICAIF 2023**
**Citations:** 34
**Link:** https://arxiv.org/abs/2309.00638
**PDF:** `1_LOBS5_ICAIF2023_oral.pdf`
**Summary:** The founding paper of LOBS5. S5-based autoregressive model treating order book messages as token sequences. First demonstration that a deep state space network can generate realistic order flow. Established the tokenisation scheme (24 tokens/message) and training pipeline.
**σ₀ relevance:** This is the origin — everything builds on this. Current work extends to 360M+ parameters, TBPTT for long context, full-depth data, and downstream tasks.

---

## Paper Narrative for Proposal

Проект σ₀ развивается по чёткой линии:

1. **Foundation model** (ICAIF 2023 Oral) — доказали что S5 может генерировать реалистичный order flow
2. **Simulator** (ICAIF 2023 Best Paper) — JAX-LOB как GPU-инфраструктура для масштабного RL
3. **Scaling** (ICAIF 2024) — MoE для масштабирования до 30B параметров
4. **Evaluation** (ICML 2025) — LOB-Bench как универсальный бенчмарк
5. **Error correction** (ICAIF 2025 Oral) — DFM для исправления compound error
6. **Multi-agent** (ICAIF 2025 Best Paper) — JaxMARL-HFT для multi-agent RL
7. **Diffusion** (arXiv 2025) — альтернативный generative подход через DiT

**Итого:** 2× Best Paper, 2× Oral, 1× ICML, 78+ citations. Полный pipeline от данных до trading.

---

## In Progress / Planned Papers

| Working Title | Lead | Status | Target |
|---|---|---|---|
| LOBS5 v2: Conditional Generation + Scaling | Sascha, Kang | Data fix needed (full-depth LOBSTER) | TBD |
| EGGROLL: Foundation Model + ES for Execution | Valentin | First results working, needs multi-window | TBD |
| Market Impact via Foundation Models | George | Breakthrough results (Feb 6), needs robustness | TBD |
| TBPTT for Long-Horizon LOB Modelling | Kang | Training in progress, LR tuning | TBD |
| Tokenisation Ablation Study | Aramis, Sascha | Not started | TBD |
