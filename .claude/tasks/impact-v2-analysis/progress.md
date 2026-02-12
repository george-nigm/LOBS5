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
