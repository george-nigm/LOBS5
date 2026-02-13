# Прогресс

## 2026-02-11
- Прочитал все исходные ноутбуки (103, 104, 105) — извлёк reusable код
- Проверил структуру V2 папок: 30 в context_500_buy/, 30 в context_500_sell/
- Создал 110.market_impact_comprehensive.ipynb — 44 ячеек, 7 секций:
  - S0: Setup & Data Loading (discover_v2_folders + auto-regex)
  - S1: Decay Curves (faceted 3x10 grid + overlay by V)
  - S2: Square-Root Law / Beta (point cloud, scatter, by-V, by-mb, bootstrap, per-day)
  - S3: Volume Dimension (V-scaling gamma, impact ratios, cross-V beta)
  - S4: Quality Heatmaps (decay, beta, combined)
  - S5: Volume-Time Dynamics (Q-groups, relaxation ratio, master curve)
  - S6: Stability + Recommendations (3-method vote, master table, dashboard, export)
- Notebook готов к запуску. Не запускался (нужна среда с данными).
- Исправлен баг в cell 43 (export): относительный путь `Path('lob_impact/...')` заменён на абсолютный с Docker/host auto-detect (`_LOB_IMPACT`)
- Decay_ratio: цвет теперь по расстоянию от 2/3 (Bouchaud) — зелёный (<0.05), жёлтый, оранж, красный
- Добавлена ячейка "Beta by 3 modes" (после per-day beta): @a, [:a], [a:] vs iteration
- Cell 7: добавлен `beta_from` в metrics_df
- Cell 21 (markdown): объяснение V-scaling gamma (γ)
- Cell 26 (heatmap): colorscale центрирован на 2/3
- Cell 41 (dashboard): новый layout — 1.1 scatter, 1.2 beta by V, 1.3 gamma; 2.1 master curve, 2.2 decay heatmap, 2.3 stability

## 2026-02-11 (continued) — Historic & Heuristic Scenario Scripts
- Создал `lob_impact/2.historic_scenario.py` — historical replay + aggressive order injection (no model, no price correction)
- Создал `lob_impact/3.heuristic_scenario.py` — same as historic + price shifting when level consumed (shift_ticks counter)
- Создал `lob_impact/2.historic_scenario_config.yaml` и `lob_impact/3.heuristic_scenario_config.yaml`
- Оба скрипта:
  - Переиспользуют инфраструктуру из `1.aggressive_scenario_s5.py` (TeeLogger, setup_logging, create_experiment_folder, parse_args)
  - Без загрузки модели (no Vocab, Message_Tokenizer, init_train_state, load_checkpoint)
  - Используют `get_sims_vmap`, `get_dataset`, `msg_to_jnp`, `msg_to_lobster_format`, `book_to_lobster_format` из `lob/inference_no_errcorr.py`
  - Выходной формат: LOBSTER CSV (data_cond/ + data_gen/ + aggressive_indices.csv + config.yaml)
- Ключевое отличие `3.heuristic_scenario.py`: `shift_prices()` — сдвигает цены limit/execution сообщений на `tick_size * shift_ticks` когда aggressive order полностью потребляет level
- Формула сдвига: `(1 - 2*direction) * tick_size * shift_ticks` (buy→вверх, sell→вниз), адаптировано из `heuristic_historical_scenario_run_quantile_fixing.py`

## 2026-02-11 — Fix Order IDs in Historic & Heuristic Scripts
- Убран фиксированный `AGGRESSIVE_ORDER_ID = 77777777` из обоих скриптов
- Добавлен параметр `order_id: jax.Array` в `create_aggressive_order()` вместо константы
- Обновлён vmap `in_axes` — `order_id` теперь батчится (8-й аргумент, `in_axes=0`)
- Добавлен descending counter `n_msg_todo` в main loop (инициализация `jnp.full(batch_size, total_steps)`)
- Aggressive orders получают `order_id = n_msg_todo - 1`, далее `n_msg_todo -= 1`
- Historical messages сохраняют свои оригинальные order_id, но counter тоже декрементируется
- Паттерн соответствует `inference_no_errcorr_w_insertions.py:do_insert()` (line 760/783)
- Добавлена `data_real/` папка (match `1.aggressive_scenario_s5.py` line 397)

## 2026-02-11 — CST Model Aggressive Scenario Script
- Создал `lob_impact/4.aggressive_scenario_cst.py` — CST parametric model baseline для market impact
- Создал `lob_impact/4.aggressive_scenario_cst_config.yaml`
- Ключевые решения:
  - CST модуль импортируется как `stoikov` (alias) чтобы не конфликтовать с `gymnax_exchange.jaxob.jaxob_constants as cst`
  - `msg_to_jaxlob()`, `update_oid()` — inlined из `lob_bench/cst_model/lobster_conversion.py` (разные import paths для JAX-LOB)
  - `load_params` — импортируется напрямую из `param_estimation`
  - Processing per-sample (no vmap) — CST достаточно быстрый (~15ms/sample после JIT warmup)
- Новые функции:
  - `apply_aggressive_to_cst_book()` — обновляет CST Book после aggressive order (использует `stoikov._change_vol`)
  - `make_cst_scan_fn()` — JAX-scannable step: CST step → msg_to_jaxlob → update_oid → process_order_array → get_L2_state
  - `create_aggressive_order()` — строит 8-field sim_msg + 14-field msg_decoded из JAX-LOB state
- Выходной формат полностью совпадает с S5 scenario: data_cond/, data_gen/, aggressive_indices.csv, config.yaml
- Aggressive orders используют фиксированный `AGGRESSIVE_ORDER_ID = 77777777` (CST не нужен descending counter)
- CST direction convention: bid_side=direction (0=ask/buy, 1=bid/sell) — verified against `_apply_event` MO logic

## 2026-02-11 — Bug Fixes & Full CPU Runs
- Fixed CPU compatibility: `jax.devices()[0]` instead of `jax.devices('gpu')[0]` in both scripts
- **Critical bug fix**: swapped ask/bid volume indices in `get_best_bid_and_ask_inclQuants` usage
  - Was: `best_bid_ask[1][1]` for "ask volume" (actually bid volume)
  - Fixed: `best_ask_pv, best_bid_pv = sim.get_best_bid_and_ask_inclQuants(sim_state)` then `best_ask_pv[1]`/`best_bid_pv[1]`
  - Same bug exists in `1.aggressive_scenario_s5.py` but is dead code there
- Full 2048-sample CPU runs completed:
  - Historic: `output/evalsequences/historic_scenario/exp_3_20260211_221812/` — 2048 files, ~86% orders at full 75
  - Heuristic: `output/evalsequences/heuristic_scenario/exp_10_20260211_221813/` — 2048 files, verified
- Remaining <75 sizes are legitimate caps where best level had insufficient volume

## 2026-02-11 — CGAN (Coletta) Aggressive Scenario
- Создал `lob_impact/5.aggressive_scenario_cgan.py` (1096 строк) — CGAN aggressive scenario
- Создал `lob_impact/5.aggressive_scenario_cgan_config.yaml` — шаблон конфига
- Создал `lob_impact/cgan_convert_lobster.py` (249 строк) — конвертация LOBSTER CSV → ABIDES pickle
- Создал `lob_impact/cgan_train.py` (151 строк) — обёртка обучения CGAN
- Все файлы прошли синтаксическую проверку
- Ключевые решения:
  - `CGANStateTracker` class: трекинг market state features (imbalance, spread, returns, exec imbalance) из JAX-LOB L2
  - Tick-based exec imbalance: подсчёт по последним 128/256 market orders (не по временным окнам)
  - `build_lookback_tensor()`: 5-sec discrete time bucketing + ffill + MinMaxScaler normalization → torch.Tensor
  - `decode_cgan_output()`: порт worldagent_vgan.unnormalized_order() + sanitize_action()
  - `cgan_order_to_sim_msg()`: конвертация action_dict → JAX-LOB 8-field msg
  - Interarrival time: Gamma(k, theta) из обученного файла или дефолт
  - `ganmodels.abides_test = True` — CGAN генерирует только один ордер без orderbook simulation внутри
  - Нормализация lookback: standalone версия gan_utils.normalize_input_data() без ABIDES-зависимостей
- Структура скрипта повторяет CST scenario: batch loop → sample loop → block loop → msg loop
- Выходной формат: LOBSTER CSV (data_cond/, data_gen/, aggressive_indices.csv, config.yaml) — идентичен S5/CST
- cgan_convert_lobster.py: standalone port lobster_converted.py без ABIDES-зависимостей (Side enum, datetime_str_to_ns)
- cgan_train.py: thin wrapper вызывающий ganmodels.main() с auto-discovery дат

## 2026-02-11 — Перезапуск с n_cond_msgs=500
- Конфиги обновлены: `n_cond_msgs: 2048` → `500` в обоих:
  - `lob_impact/2.historic_scenario_config.yaml`
  - `lob_impact/3.heuristic_scenario_config.yaml`
- Shell сломался (exit code 1 на всех командах) — не удалось удалить старые эксперименты и перезапустить
- **НУЖНО СДЕЛАТЬ в новом терминале**:
  1. Удалить ВСЕ старые эксперименты:
     - `output/evalsequences/historic_scenario/exp_*` (2 папки: exp_1, exp_2)
     - `output/evalsequences/heuristic_scenario/exp_*` (9 папок: exp_1 — exp_9)
  2. Запустить оба скрипта с n_cond_msgs=500, n_samples=2048 на CPU
  3. Проверить результаты

## Текущее состояние файлов
- `lob_impact/2.historic_scenario.py` — готов (order_id fix, volume fix, CPU fix)
- `lob_impact/3.heuristic_scenario.py` — готов (order_id fix, volume fix, CPU fix)
- `lob_impact/2.historic_scenario_config.yaml` — n_cond_msgs=500, n_samples=2048, order_volume=75
- `lob_impact/3.heuristic_scenario_config.yaml` — n_cond_msgs=500, n_samples=2048, order_volume=75

## Ключевые фиксы в скриптах (для справки)
1. **Order ID**: descending counter `n_msg_todo` вместо фиксированного `77777777`
2. **CPU**: `jax.devices()[0]` вместо `jax.devices('gpu')[0]`, `device` вместо `gpu_device`
3. **Volume bug**: `best_ask_pv, best_bid_pv = sim.get_best_bid_and_ask_inclQuants(sim_state)` — правильная деструктуризация (было swapped)

## 2026-02-11 — Перезапуск с n_cond_msgs=500 (ВЫПОЛНЕНО)
- Удалены все старые exp_* из historic_scenario/ и heuristic_scenario/
- Запущены оба сценария в Docker контейнере `georgenigm_lobs5_viz` на CPU (`CUDA_VISIBLE_DEVICES=''`)
- **Historic scenario**: `exp_1_20260211_225650/` — 2048 samples (4096 файлов gen+cond), config n_cond_msgs=500, direction=0, 3 мин 3 сек
- **Heuristic scenario**: `exp_1_20260211_225804/` — 2048 samples (4096 файлов gen+cond), config n_cond_msgs=500, direction=0, 3 мин 25 сек
- Оба сценария: 32 batches по 64 samples, 280 steps (275 historical + 5 aggressive)

## Что осталось
- ~~СРОЧНО: удалить старые exp_* и перезапустить с n_cond_msgs=500~~ ✓ DONE
- Task #2: Run full v2 grid (60 configs × 2 scenarios = 120 runs) on CPU workers
- Запустить cell-by-cell в Jupyter (Docker или host) для 110.market_impact_comprehensive.ipynb
- Проверить что discover обнаруживает 30 папок
- Проверить beta global ~ 0.54-0.56
- Проверить mb=20 как "poor"/"avoid"
- Запустить CST scenario с маленьким конфигом для верификации
- **CGAN**: обучить модель на GOOG данных (нужны raw LOBSTER CSV)
- **CGAN**: smoke test `5.aggressive_scenario_cgan.py` с n_samples=2, batch_size=1
- **CGAN**: создать YAML-конфиги для grid запуска (аналог configs_context_500_c10x_v2/)

## 2026-02-11 — RWKV Aggressive Scenario Script
- Создал `lob_impact/5.aggressive_scenario_rwkv.py` (~680 строк) — RWKV aggressive market impact scenario
- Создал `lob_impact/5.aggressive_scenario_rwkv_config.yaml` — конфиг для GOOG 2023 Jan / bptt_rwkv_7g0.1B/final
- Ключевые решения:
  - **jax.lax.scan + early stopping mask**: решает проблему variable-length BPE (20-35 tokens/message)
    - Фиксированное `max_tokens_per_block` шагов scan, каждый sample считает newlines (token 36)
    - После `n_gen_msgs` newlines → sample переключается на PAD (token 0), state замораживается
    - ~30% overhead от PAD-шагов — приемлемо для vmapped batching
  - **make_generate_block_fn()**: factory для scan-based generation, vmapped по batch
  - **process_long_seq**: заимствована из lobgen/evaluate.py — chunked processing через scan
  - **v_process_long_seq / v_generate_block**: JIT + vmap для batched GPU processing
  - **Data loading**: raw LOBSTER CSV → convert_to_nanoseconds → differentiate → tokenize (matches training format)
  - **Simulator**: replay conditioning messages + replay generated tokens (dual-pass architecture):
    - Pass 1 (online): update sim_states during generation for aggressive order placement
    - Pass 2 (post-process): clean replay for accurate L2 states in output CSVs
  - **Aggressive order injection**: format as RWKV text → tokenize → feed through model (v_process_long_seq)
  - **Token ↔ message parsing**: parse_block_tokens() — handles <time> delta accumulation, <tag>/value pairs, orderbook skip
  - **Direction convention**: LOBSTER raw {1=buy, -1=sell} → RWKV text uses same → construct_sim_msg expects {0=buy, 1=sell}
  - **update_oid()**: reused from CST/CGAN scripts — fixes order IDs for cancel/execution messages
- Структура: load_day_data → enumerate_samples → batch loop → v_process_long_seq (conditioning) →
  v_generate_block (blocks) → parse_block_tokens → sim replay → msg_to_lobster_format
- Выходной формат: LOBSTER CSV (data_cond/, data_gen/, aggressive_indices.csv, config.yaml) — идентичен S5/CST/CGAN
- NEEDS: smoke test in Docker с маленьким конфигом (n_samples=1, batch_size=1, n_gen_msgs=5, num_insertions=1, num_coolings=2)

## 2026-02-12 — Paper Draft (sample-sigplan.tex)
- Полностью переписан `overleaf_project/sample-sigplan.tex` — bullet-point placeholders → full prose
- Секции: Abstract (204w), Intro (443w), Background (523w), Methodology (665w), Setup (229w), Results (627w), Discussion (394w), Conclusion (153w) ≈ 3238w total
- 6 figure placeholders (protocol, architectures, midprice 4-panel, loglog scatter, perday beta, decay overlay)
- 2 tables (parameter grid, model comparison)
- 5 key equations (sqrt law, combined impact, Parkinson, loglog regression, through-origin OLS)
- 19 citations — all keys verified against bib
- CCS codes updated (Neural networks, Probabilistic algorithms, Economics)
- Metadata: ICAIF '25, anonymous mode, 2025 year
- `sample-base.bib` — добавлено 18 domain-specific BibTeX entries (Kyle, Almgren, Cont, Bouchaud, Toth, etc.)
- LaTeX syntax verified: balanced braces, matched environments, no orphaned refs
- No LaTeX compiler available on host — push to Overleaf for PDF verification

## 2026-02-12 — CGAN Training (correct data split)
- Обнаружено: старая конвертация (9 дней Jan 2023) была на ТЕСТОВЫХ данных → нельзя обучать на тесте
- Правильный split: train = 3 дня Dec 2022 (20221228, 20221229, 20221230), test = 9 дней Jan 2023
- Обновлён `run_cgan_train.sh`: LOBSTER_DIR → 2022/, CONVERTED_DIR → converted_train/, DATES → 3 дня Dec
- Raw LOBSTER data: 2.17M messages/day для GOOG Dec 2022
- Конвертация запущена в Docker контейнере `georgenigm_cgan_train` на GPU 7
- Статус: конвертация LOBSTER → ABIDES pickle идёт (Step 1), затем обучение 40 эпох (Step 2)

## 2026-02-12 — Notebook 120: Cross-Scenario Market Impact Analysis
- Created `lob_impact/120.market_impact_all_scenarios.ipynb` — 53 cells (20 markdown + 33 code)
- Generated from `lob_impact/_gen_nb120.py` (generator script)
- All code cells pass Python syntax validation
- Structure:
  - **Section 0**: Setup, imports, scenario registry, auto-discovery, full function library (parameterized from 110)
  - **Section 1**: Per-scenario processing loop with `process_scenario()` master function + gc.collect()
  - **Part A** (per-scenario): Decay curves, Beta analysis, Gamma, Quality heatmaps, Volume-time, Stability
  - **Part B** (cross-scenario): Side-by-side heatmaps, overlay plots, grouped bars, radar chart
  - **Section 14**: Model ranking — scorecard, radar chart, composite score, per-config winner map, paired bootstrap
  - **Section 15**: Dashboard & CSV export
- Key changes from 110:
  - `scenario` → `direction` rename (scenario now = model type)
  - All functions parameterized (no globals BUY_PATH/SELL_PATH)
  - Memory management: one scenario at a time, del+gc.collect()
  - 6-model registry: S5, Historic, Heuristic, CST, RWKV (placeholder), Coletta (placeholder)
  - Auto-skip missing scenarios
- Output: `/homes/80/georgenigm/LOBS5/lob_impact/120.market_impact_all_scenarios.ipynb`
- NOT YET RUN — needs Jupyter environment with data access

## 2026-02-12 — CGAN Training Pipeline Fixes & Launch
- Конвертация 3 дней Dec 2022 завершена (43 мин):
  - 20221228: 2.17M msgs (1.6 GB), 20221229: 1.81M msgs (1.4 GB), 20221230: 1.89M msgs (1.4 GB)
- Fix 1: `cgan_train.py` — добавлен `__path__` к мокам `scripts` и `scripts.ganworldagent` → Python находит реальный `interrarival_time.py`
- Fix 2: удалён мок `scripts.ganworldagent.utils` → Python импортирует реальный модуль (нужен `enrich_cancellations_orig`)
- Fix 3: `run_cgan_train.sh` — `pytorch-lightning>=2.0` → `>=1.9,<2.0` (v2 убрал `validation_epoch_end`)
- Fix 4: monkey-patch `LOBGAN.__log_results_tb` → WandB histograms вместо TensorBoard `add_histogram`
- Обучение запущено на GPU 7, контейнер `georgenigm_cgan_train`:
  - Train: 3,054,692 samples, Val: 763,598 samples
  - Model: Generator 167K + Discriminator 598K = 765K params
  - 59660 steps/epoch, ~3.6 it/s, 40 epochs, loss falling (0.73→0.06 за 22 steps)
  - WandB: https://wandb.ai/george-nigm/cgan-lob/runs/vjt9ezgx
  - ETA: ~4.6h/epoch → ~7-8 дней на 40 эпох (можно остановить раньше если converged)

## 2026-02-12 — Paper: Align with σ₀ Research Narrative
- Added 5 BibTeX entries to `sample-base.bib`: frey2023jaxlob, nagy2025lobbench, mohl2025jaxmarlhft, li2025dfm, backhouse2025painting
- **Abstract**: LOB-Bench positioning sentence, JAX-LOB citation for simulator, "complementing distributional benchmarks" closing
- **Section 1 (Intro)**: LOB-Bench replaces generic metrics (para 1), world model motivation with JaxMARL-HFT (para 2), LOB-Bench replaces cont2001empirical (para 3), 4th contribution (evaluation hierarchy)
- **Section 2.2**: Expanded with JAX-LOB pipeline, O(n) complexity, 22-24 token encoding, diffusion alternative (backhouse2025painting), compound error (li2025dfm)
- **Section 2.3**: Rewritten to position explicitly vs LOB-Bench (L1/Wasserstein, individual-event vs meta-order)
- **Section 3**: JAX-LOB matching engine replaces lobster2023 cite for simulator
- **Section 4**: JAX-LOB simulator description added (NASDAQ ITCH, vectorized JAX)
- **Section 6**: New "From microscopic to macroscopic evaluation" paragraph, JaxMARL-HFT practical implications, compound error decay limitation
- **Section 7**: Pipeline conclusion sentence (LOB-Bench + JaxMARL-HFT + market impact = evaluation stack)
- Fixed: "three contributions" → "four contributions"
- Fixed: redundant "diagonal structured state-space layers" repetition in §2.2
- Verified: all 23 cite keys exist in bib, zero anonymity violations
- Total: ~200 words added, 0 deleted — fits within 6-8 page budget

## 2026-02-12 — RWKV Debugging & Price Mismatch Discovery

### Проблемы найдены и исправлены
1. **ImportError tokenizers**: Docker image `georgenigm_25jan:latest` не содержит tokenizers/transformers (cached layers). Fix: `pip install` at runtime.
2. **NaN checkpoint**: `bptt_rwkv_7g0.1B/final` → NaN. Сканировал все 8 моделей × все шаги → `rwkv_7g0.1B/final` работает.
3. **Tiny test пройден**: 8 samples, GOOG 2018 data, `rwkv_7g0.1B/final` — генерация работает.

### БЛОКЕР: Price mismatch
- **Все GOOG RWKV чекпоинты обучены на GOOG 2017** (~$1040, pre-split)
- **Тестовые данные GOOG Jan 2023** (~$91, post-split после 20:1 сплита Jul 2022)
- RWKV использует абсолютные цены как текст → генерирует в диапазоне ~$1040 независимо от кондиционирования
- S5 не имеет этой проблемы (относительные цены, тики от mid-price)
- В `lobgen/run_all.py` было задумано обучение на `goog2022.npy` → чекпоинты `goog2022_rwkv_*`, но ни данных, ни чекпоинтов нет

### Что есть
| Модель | Данные | Цены | Статус |
|--------|--------|------|--------|
| rwkv_7g0.1B (final OK) | GOOG 2017 | ~$1040 | ✓ работает, но не совместима с Jan 2023 |
| rwkv_6g0.1B (все OK) | GOOG 2017 | ~$1040 | то же |
| bptt_rwkv_7g0.1B (step 10 OK, rest NaN) | GOOG 2017 | ~$1040 | то же |
| bptt_rwkv_6g0.1B (все OK) | GOOG 2017 | ~$1040 | то же |
| intc2022_rwkv_* | INTC 2022 | другой тикер | неприменимо |

### Что нужно
- Чекпоинт RWKV обученный на **GOOG post-split 2022** (авг-дек, ~$88-93)
- Написано сообщение для Sascha с вопросом есть ли такой чекпоинт / сколько заняло обучение
- Raw GOOG 2022 данные есть: `/home/myuser/data/rawLOBSTER/GOOG/2022/` (251 день)

### Архитектура RWKV v6 vs v7
- `6g0.1B` = RWKV-x060 (173M, Pile pretrained May 2024)
- `7g0.1B` = RWKV-x070 (168M, Pile pretrained Nov 2024)
- `g` = GptTokenizer, `w` = WorldTokenizer
- Обучение: `train_shuffle.py`, JAX, 8 GPU, seq_len=16384, ~1B tok/epoch, optimizer=dadapt_adamw

### Evaluate.py контекст
- Стандартная eval: 500 msgs conditioning → генерация 500 msgs (max 11000 tokens)
- `parallel_processing=500` samples одновременно

### Файлы
- Config обновлён на GOOG 2018: `5.aggressive_scenario_rwkv_config_tiny.yaml` (raw_data_dir → 2018)
- Тест с 2018 данными запущен (в процессе)
- `run_rwkv_only_c10x_v2.sh` и 60 YAML конфигов всё ещё указывают на `bptt_rwkv_7g0.1B/final` — НЕ обновлены (ждём решения по чекпоинту)

### TODO при возвращении
1. Дождаться ответа от Sascha про чекпоинт GOOG 2022
2. Если нет — обучить самим (нужно: preprocess_data.py → goog2022_postsplit.npy → train_shuffle.py)
3. Обновить все 60 конфигов + launch script с правильным чекпоинтом
4. Запустить full grid (60 экспериментов)

## 2026-02-13 — Notebook 130 + Paper Rewrite (4-model results)

### Ноутбук 130
- Создан `lob_impact/_gen_nb130.py` — генератор ноутбука
- Сгенерирован `lob_impact/130.paper_results.ipynb` (21 ячейка):
  - Cell 0: Title
  - Cell 1-2: Imports, config, paths, scenario registry (4 модели)
  - Cell 3: Data I/O functions (discover, load)
  - Cell 4: Beta functions (extract_point_cloud, compute_global_beta, bootstrap)
  - Cell 5: Master curves + relaxation functions
  - Cell 6: Stability functions (3-method vote)
  - Cell 7: Main processing loop (load all 4 scenarios, compute all metrics, free memory)
  - Cells 8-11: Section 1 — Beta: Table 1, Fig 3 (regression), Fig 4 (bootstrap)
  - Cells 12-15: Section 2 — Master Curves: Fig 1 (2x2), Fig 2 (overlay), Table 2 + Fig 5 (relaxation)
  - Cells 16-17: Section 3 — Stability: Table 3 + Fig 6 (fraction stable)
  - Cells 18-19: Section 4 — Gamma: Fig 99
  - Cell 20: Summary table
- Все code cells прошли синтаксическую проверку
- НЕ ЗАПУЩЕН — нужна Jupyter среда с доступом к данным

### Статья (sample-sigplan.tex)
- **Новый нарратив**: ВСЕ 4 модели дают β≈0.5 (a не только S5). Различие — в динамике.
- **Abstract**: полностью переписан. Ключевое: β∈[0.545, 0.561] для всех 4 моделей, различие в relaxation
- **Intro**: contributions #3 → "Universal β, divergent dynamics", #4 → "Hierarchical evaluation"
- **Section 5 Results**: полностью переписан:
  - 5.1 Volume-Time Master Curves (Fig 3: 2x2, Fig 4: overlay)
  - 5.2 Square-Root Law: Universal Across All Models (Table 2: beta, Fig 5: regression, Fig 6: bootstrap)
  - 5.3 Impact Relaxation: The Key Differentiator (Table 3: relaxation, Fig 7: box plot)
  - 5.4 Stability Analysis (Table 4: fraction stable, Fig 8: bar chart)
- **Section 6 Discussion**: полностью переписан:
  - "Static vs dynamic emergence" (ключевой инсайт)
  - "Role of hidden state" (обновлён с реальными числами)
  - "Evaluation hierarchy" (3 уровня: distributional, static β, dynamic relaxation)
  - "Practical implications" (обновлён)
  - "Limitations" (обновлён, убран "TBD baseline results")
- **Section 7 Conclusion**: полностью переписан (separation of concerns, 122,880 sims per model)
- **Table 2 (beta)**: реальные данные: S5=0.545, Historic=0.549, Heuristic=0.548, CST=0.561
- **Table 3 (relaxation)**: S5=0.76, Historic=0.07, Heuristic=1.43, CST=0.87
- **Table 4 (stability)**: S5=90%, Historic=100%, Heuristic=70%, CST=80%
- Валидация: 24 cite-ключа ✓, все ref/label ✓, все таблицы ✓
- 7 фигур referenced (нужно скопировать картинки в Figures/ и переименовать)
- 1 placeholder остаётся (Fig 2: architecture comparison)

### Маппинг картинок для Overleaf
| Файл в pics_for_transfer_4_methods/ | LaTeX ссылка |
|------|------|
| `1. Master Curves.png` | `Figures/master_curves_4panel.png` |
| `2. Average Master Curve.png` | `Figures/avg_master_curve.png` |
| `3. Beta Regression Lines.png` | `Figures/beta_regression.png` |
| `4. Bootstrap Beta Distributions.png` | `Figures/bootstrap_beta.png` |
| `5. Relaxation Ratio.png` | `Figures/relaxation_ratio.png` |
| `6. Fraction Stable.png` | `Figures/fraction_stable.png` |

### Ключевые числа из nb120
| Модель | β | R² | N | CI | Relaxation | Stable |
|--------|------|------|---------|------|------|------|
| S5 | 0.545 | 0.948 | 368,481 | [0.543, 0.548] | 0.76 | 90% |
| Historic | 0.549 | 0.948 | 368,283 | [0.546, 0.551] | 0.07 | 100% |
| Heuristic | 0.548 | 0.948 | 368,619 | [0.545, 0.550] | 1.43 | 70% |
| CST | 0.561 | 0.956 | 364,112 | [0.560, 0.563] | 0.87 | 80% |
| Theory | 0.5 | — | — | — | 0.667 | — |
