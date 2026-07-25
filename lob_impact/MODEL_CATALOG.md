# Model Catalog — message-level LOB world models for the impact framework

The fullest known list of order/message-level LOB generative models, split into (A) models already
integrated in this framework, (B) external models with public checkpoints (v4 integration targets),
(C) external models runnable only after retraining, (D) closed/unusable, (E) out-of-scope
(not message-level). Survey date: **2026-07-25** (web-verified repos/weights). Companion memory:
`external-lob-world-models`.

Framework contract a model must satisfy: condition on a real LOBSTER message history → generate
messages (type/side/price/size/time) applied to the JAX-LOB simulator → accept injected aggressive
orders (metaorder protocol) → write `data_cond`/`data_gen` CSVs (see `FRAMEWORK.md`).

## A. Integrated (run in v2/v3 rounds)

| Label | Class | Ckpt / params | Notes |
|---|---|---|---|
| S5 (LobS5) | neural SSM, token AR | trained in-house (8 tickers 2022-25) | original Nagy et al. line |
| S5_120M | neural SSM, scaled | in-house | |
| S5_4k | neural SSM, 4k context | in-house | long-context line |
| Mamba3 | neural SSM (Mamba3) | in-house (exp_R1_Mamba3) | flagship |
| Mamba3_4k | neural SSM, 4k context | in-house | |
| GDN | neural (gated delta net) | in-house | classic neural profile (β≈0.97 EA) |
| Historic | replay of real data | — | ground-truth control |
| Heuristic | replay + permanent tick shift | — | mechanical control |
| CST | Cont-Stoikov-Talreja parametric | per-stock pkl (estimated) | stationary null |
| NMZI | zero-intelligence + sign feedback | per-stock pkl | informative null at literature κ |
| Hawkes | Hawkes depth model | per-stock pkl | |
| QR | queue-reactive (Huang-Lehalle-Rosenbaum) | per-stock pkl | |
| OW / Propagator | Obizhaeva-Wang exponential-resilience propagator | params npz | PROP_KERNEL=exp |

## B. External, public checkpoints — v4 targets («заявленные стоки»)

| Label (v4) | Model | Source | Ckpt | Claimed stock(s) | Integration |
|---|---|---|---|---|---|
| MarketGPT | Wheeler & Varner 2024 (arXiv:2411.16585) — 94.6M Llama-style AR over ITCH message tokens (24 tok/msg), KV-cache | github.com/aaron-wheeler/MarketGPT (MIT) | HF `aaronwheeler/MarketGPT-100m`, `ckpt_finetune_AAPL_v3.zip` 1.05 GB | **AAPL** (ITCH 2019, 8 days) | `marketgpt_scenario.py`: pure next-msg sampler + our JAX-LOB as single book; prices decoded relative to live mid → 2026 anchoring free |
| TRADES | Berti et al. 2025 (arXiv:2502.07071) — transformer DDPM, 1 order/call, cond = 255 orders + 256 L10 states | github.com/LeonardoBerti00/DeepMarket (MIT) | GDrive folder `1fg5G9KzmzC6E4FUYSCjObJ7sCEdjo43W` | **TSLA, INTC** (Jan-2015) | `trades_scenario.py`; 2015 z-score constants verbatim (constants.py) + mandatory affine price re-anchor of the 2026 window; DDIM-10 default |
| CGAN | Coletta-style cGAN (DeepMarket reimpl.) — LSTM cond on 9 price-free market features | same repo | git-LFS in repo, 2 MB each | **TSLA, INTC** (Jan-2015) | `cgan_dm_scenario.py`; needs 512-msg context; gamma-fit interarrivals |
| KNN | Giegrich, Oomen, Reisinger 2024 (arXiv:2409.06514) — K-NN state resampling, K=20, Euclidean on vol profile | no code — reimplemented from paper | none needed (pool = our own data) | any (we run panel + TSLA/INTC) | `knn_scenario.py`; message-level adaptation of their state-transition resampler, price_rel re-anchor |

## C. External, code public, **no weights** (retrain required)

| Model | Source | Blocker |
|---|---|---|
| Hultin RNN-LOB (Quant. Finance 2023) | github.com/lobrnn/lob-rnn (Apache-2.0) | TensorFlow 2.4; no ckpt; Nasdaq Nordic data gated. Factorized conditionals — insertion-friendly after retrain |
| RWKV4 / RWKV6 LOB (LOB-Bench baselines, ICML 2025) | training stack = LOBS5-fork (peernagy) | weights not published; same token stream as our S5 — could retrain in-house |

## D. Closed / weight-locked / vaporware

| Model | Status (2026-07-25) |
|---|---|
| MarS / LMM (Microsoft, ICLR'25) | engine MIT (github.com/microsoft/MarS) but LMM weights private since 2024 («awaiting review», HF issue #9 unanswered); trained on Chinese A-shares. Cite as unavailable |
| Coletta world agent (JPM, ICAIF'21/'22) | no code, no weights ever released (proprietary). De-facto public implementation = DeepMarket CGAN (row B) |
| Stock-GAN (Li et al., AAAI 2020) | no official repo; appendix pseudocode only |
| Shi-Cartlidge NS-ABM (AAMAS'23) | neural-Hawkes ABIDES hybrid; only the KDD'22 event-prediction part released |
| DiGA (MSRA, AAAI'25) | diffusion meta-agent emitting orders; no code |
| DFM post-training (Oxford/UCLA, ICAIF'25) | discrete flow matching over AR LOB models; no repo |
| LOBERT (arXiv:2511.12563) | encoder-only (BERT), not a sampler; no code |
| Cont-Cucuringu-Kochems-Prenzel GAN (SSRN 4512356) | next book state, proprietary broker data, no code |

## E. Not message-level (excluded from the baseline slot)

| Model | Granularity |
|---|---|
| DiffLOB (arXiv:2602.03776, code public) | L2 trajectories (counterfactual-native, but no messages) |
| Painting the Market (Oxford, arXiv:2509.05107) | L2-as-image diffusion inpainting; no code |
| DiffVolume (arXiv:2508.08698) | volumes only |
| Financial Wind Tunnel (arXiv:2503.17909) | candlestick/cross-sectional |
| ABIDES / agent-based configs | rule agents, not learned world models (host platform only) |

## Fairness caveat for the paper

Every public checkpoint (B) is trained on another period (2015/2019) and, for TRADES/CGAN, another
market regime; off-the-shelf runs are **transfer baselines**. In-distribution comparison would
require retraining on our 8-ticker 2022-25 corpus (possible for MarketGPT ~100M and TRADES ~12M;
out of scope for v4). MarketGPT/AAPL is the cleanest case: same stock, same venue, price-level-free
decoding.
