# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LOBS5 is a generative AI model for end-to-end limit order book (LOB) modelling. It uses token-level autoregressive generation of NASDAQ order book messages via Simplified Structured State-Space Networks (S5). Generated messages are validated by a JAX-LOB simulator (Alphatrade submodule). The project is a fork of the [original S5 repository](https://github.com/lindermanlab/S5).

## Common Commands

### Docker-based workflow (primary)
```bash
make build              # Build Docker image (Dockerfile_LOBS5, NVIDIA JAX CUDA 13 base)
make run                # Interactive shell in container
make test               # pytest ./tests/
make train_small        # bin/run_experiments/run_lobster_padded_small.sh
make train_large        # bin/run_experiments/run_lobster_padded_large.sh
make inference          # Run inference for AMZN stock
make eval               # Run evaluation via bin/eval_bash/eval_local.sh
make benchmark          # Run LOB-Bench benchmarks
```

### Direct entry points (inside container or with deps installed)
```bash
# Preprocessing LOBSTER data
python preproc.py --data_dir /path/to/data/GOOG/ --save_dir /path/to/save/ --n_tick_range 500 --use_raw_book_repr

# Training
python3 run_train.py --USE_WANDB=True --d_model=512 --n_layers=12 --ssm_size_base=512 --blocks=16 --bsz=24 --msg_seq_len=500 --use_book_data=True --masking=none --epochs=100

# Inference (generation)
python3 run_inference.py --stock AMZN --checkpoint_step 37 --test_split 1 --batch_size 32 --n_sequences 1024

# Evaluation
python3 run_eval.py --restore /path/to/checkpoint --restore_step 0 --dir_name /path/to/data

# Market impact simulation
python lob_impact/1.aggressive_scenario_s5.py --config lob_impact/configs_context_run/some_config.yaml

# LOB-Bench benchmarking
python3 lob_bench/run_bench.py --stock AMZN --model_version ruby-aardvark --data_dir ./eval_local --save_dir ./benchmark_local/
```

### Install dependencies
```bash
pip install -r requirements.txt
# JAX GPU packages are commented out in requirements.txt; Docker handles CUDA JAX install
```

## Architecture

### Core Pipeline

```
LOBSTER CSV → preproc.py → tokenized .npy files → run_train.py → checkpoints → run_inference.py → generated sequences → lob_bench/ evaluation
```

### Key Modules

**`lob/`** — Core model and training code:
- `encoding.py` — `Message_Tokenizer` and `Vocab` classes. Custom tokenization converts LOB message integer fields (order ID, event type, direction, price, size, timestamps) into digit-group tokens. JAX-jitted encode/decode. Special values: MASK_VAL, HIDDEN_VAL, NA_VAL, START_VAL.
- `lob_seq_model.py` — `LobPredModel` wrapping a `StackedEncoderModel` (S5 layers) with embedding + decoder. Supports both `__call_ar__` (autoregressive) and `__call_rnn__` (scan-based RNN) forward passes with separate hidden state management.
- `lobster_dataloader.py` — `LOBSTER_Dataset` with masking strategies (causal, random, last_pos, none), L2 state transformation, and PyTorch DataLoader integration.
- `inference.py` / `inference_no_errcorr.py` — Autoregressive token generation with `generate_messages()` and `apply_tokens_to_book()`. Supports conditional (from historical context) and unconditional generation.
- `train.py` / `train_helpers.py` — Training loop with multi-GPU pmap, gradient clipping, LR schedules (warmup, cosine annealing, reduce-on-plateau), orbax checkpointing, WandB logging.
- `validation_helpers.py` — Per-token CE loss and accuracy evaluation.

**`s5/`** — Original S5 framework (state-space model):
- `ssm.py` — Discretized continuous-time linear SSM with binary parallel scan. Diagonal Lambda matrices with HiPPO initialization.
- `seq_model.py` — `StackedEncoderModel` stacking sequence layers with normalization and activation.
- `ssm_init.py` — HiPPO/DPLR initialization for structured state matrices.

**`Alphatrade/`** — Git submodule containing JAX-LOB order book simulator:
- `gymnax_exchange/jaxob/JaxOrderBookArrays.py` — Array-based JAX order book (fully jitted, GPU-accelerated).
- `gymnax_exchange/jaxob/jorderbook.py` — `OrderBook` class used during inference to validate generated messages.
- Has its own test suite in `tests_claude/` with ~93% coverage.

**`lob_bench/`** — LOB-Bench evaluation framework:
- Distributional metrics (spread, interarrival time, imbalance, volume, order depths) with L1/Wasserstein distances.
- Impact-response analysis for 6 event types (MO, LO, CA with/without mid-price change).
- `run_bench.py` — CLI entry point; `eval.py` — pipeline orchestration.

**`lob_impact/`** — Market impact analysis experiments:
- `1.aggressive_scenario_s5.py` — Simulates price impact by injecting synthetic aggressive orders into generated sequences.
- YAML configs in `configs_context_run/` and `configs_context_500_c10x/`.
- Jupyter notebooks (`80-82.*.ipynb`, `100-103.*.ipynb`) for visualization and analysis.

### Data Flow During Inference

1. Load trained checkpoint via `init_train_state()` + `load_checkpoint()` (orbax)
2. Load LOBSTER dataset, select test split
3. Feed historical context messages (tokenized) through model
4. Generate tokens autoregressively, decode to messages
5. Apply decoded messages to JAX-LOB simulator (`OrderBook`)
6. Extract L2 book states for evaluation

### Important Patterns

- **JAX-first**: Heavy use of `@jax.jit`, `@jax.vmap`, `jax.pmap` throughout. Data operations use JAX arrays.
- **Worker process GPU isolation**: `run_train.py`, `run_eval.py`, and `run_inference.py` set `CUDA_VISIBLE_DEVICES="-1"` at module level for worker processes (before `__main__` guard), then re-enable GPUs inside `__main__`.
- **Multi-device training**: Uses `jax.pmap` with `num_devices` parameter; batch size must be divisible by device count.
- **Experiment tracking**: WandB integration for training metrics, hyperparameter sweeps (`lob/sweep.py`), and evaluation tables.
- **Checkpoints**: Orbax-based checkpoint saving/loading in `lob/init_train.py`.

### Environment Variables (set in Docker/entry points)
```
XLA_PYTHON_CLIENT_PREALLOCATE=true/false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85-0.99
TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Data

- LOBSTER format: NASDAQ LOB data from lobsterdata.com
- Preprocessed to `.npy` files via `preproc.py`
- Data directories mounted via Docker volumes (see Makefile `BASE_FLAGS`)
- Gitignored: `.npz`, `.csv`, `.npy`, `.7z`, `checkpoints*/`, `cache_dir/`

## Docker Experiment Launch Rules

**ВСЕ эксперименты запускаются в Docker-контейнерах.** Никогда не запускай python-скрипты напрямую на хосте.

### Базовый образ и сборка
- Dockerfile: `Dockerfile_LOBS5`, базовый образ: `nvcr.io/nvidia/jax:25.01-py3` (NVIDIA JAX + CUDA 13)
- Сборка: `make build` или `docker build -f Dockerfile_LOBS5 --build-arg USE_CUDA=true --build-arg MYUSER=myuser --build-arg UID=$(id -u)`
- Образ по умолчанию в Makefile: `lobs5_sascha:latest`
- Образ для market impact экспериментов: `georgenigm_25jan:latest`

### GPU выбор
- Makefile по умолчанию: `--gpus '"device=7"'` (одна GPU)
- Market impact эксперименты: каждый контейнер на отдельной GPU `--gpus '"device=${gpu}"'`
- Доступные GPU: `0,1,2,3,4,5,6,7` (8 штук), иногда GPU 5 исключена
- Распределение нагрузки: snake pattern (0→7, 7→0, 0→7...) сортировка по убыванию workload
- Для multi-GPU тренировки: `--num_devices=6-8` внутри контейнера (JAX pmap)

### Монтирование томов (volume mounts)

**Makefile (тренировка, инференс, eval):**
```
-v ${PWD}:/home/myuser                              # код проекта
-v $(DATADIR):/home/myuser/data                     # данные LOBSTER
-v $(SCRATCH_DIR):/home/myuser/scratch              # scratch хранилище
-v $(GYMNAX_DIR):/home/myuser/gymnax_exchange       # JAX-LOB симулятор
-v $(LOBBENCH_DIR):/home/myuser/lob_bench           # LOB-Bench
--shm-size 20G
```
- `DATADIR`: `/homes/80/sascha/data` (на flair-node-12) или `~/data`
- `SCRATCH_DIR`: `~/scratch_LOB`
- `GYMNAX_DIR`: `/homes/80/sascha/AlphaTrade/gymnax_exchange`

**Market impact эксперименты (run_context_*.sh):**
```
-v "${PROJECT_DIR}:/app"                             # код проекта → /app
-v "${PROJECT_DIR}/Alphatrade:/AlphaTrade"          # симулятор → /AlphaTrade
-v /homes/groups/finance/data:/home/myuser/data     # shared finance data
-v "${PROJECT_DIR}/output/evalsequences:/home/myuser/data/evalsequences"  # выходные данные
-e WANDB_API_KEY="${WANDB_KEY}"                     # WandB ключ
--shm-size=1g
-w /app                                             # рабочая директория
--user "$(id -u):$(id -g)" --group-add 652          # права доступа
```

### Порты
```
-p 8060:80      # HTTP
-p 8064:6006    # TensorBoard
```

### Переменные окружения в контейнере
```
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85-0.90
TF_FORCE_GPU_ALLOW_GROWTH=true
NCCL_TIMEOUT=600              # multi-GPU тренировка
NCCL_IB_DISABLE=0
NCCL_P2P_DISABLE=0
PYTHONPATH="/home/${MYUSER}"
```

### Типы запуска экспериментов

**1. Тренировка** (`make train_small` / `make train_large`):
```bash
docker run -it --rm ${GPUS} ${BASE_FLAGS} ${PORT_FLAGS} ${IMAGE} \
  /bin/bash -c "sh bin/run_experiments/run_lobster_padded_small.sh"
```
Ключевые параметры: `--d_model`, `--n_layers`, `--ssm_size_base`, `--blocks`, `--bsz`, `--epochs`, `--num_devices`, `--restore`, `--restore_step`, `--dir_name`

**2. Инференс** (`make inference`):
```bash
docker run ... python3 run_inference.py --stock AMZN --checkpoint_step 37 --test_split 1 --batch_size 32 --n_sequences 1024
```

**3. Market impact — массовый запуск** (`lob_impact/run_context_*.sh`):
- Скрипт генерирует YAML-конфиги динамически
- Запускает N detached контейнеров (`docker run -d`) по одному на GPU
- Каждый контейнер: `python -u lob_impact/1.aggressive_scenario_s5.py --config <yaml> --n_gen_msgs <mb> --direction <dir>`
- Логи: `docker logs -f <container> > logfile`
- Ожидание: `docker wait <container>`
- После завершения: flatten `exp_*/` в родительскую папку

### YAML-конфиг для market impact экспериментов
```yaml
# Сценарий
n_gen_msgs: 50                 # сообщений между insertions/coolings
num_insertions: 3              # количество aggressive order insertions
num_coolings: 12               # cooling periods (c = 10*i в v2)
n_cond_msgs: 500               # длина контекста (historical)
n_eval_msgs_dataset: 500

# Aggressive order
event_type: 4                  # market order
direction: 0                   # 0=buy, 1=sell
order_volume: 75               # акций за ордер (75, 300, 485 в v2)

# Sampling
n_samples: 2048
batch_size: 64
rng_seed: 42
chunk_size: 5

# Пути (внутри контейнера)
stock: "GOOG"
data_dir: "/home/myuser/data/processed_data/GOOG/2023_Jan"
ckpt_path: "/home/myuser/data/checkpoints/lobs5_v2/twilight-sound-77_s42sujip"
save_dir: "/home/myuser/data/evalsequences/aggressive_scenario/..."

# Модель
tick_size: 100
sample_top_n: -1
n_vol_series: 500
book_dim: 503
checkpoint_step: null          # null = latest
test_split: 0
```

### Грид параметров для market impact
- **v1** (`configs_context_run/`): `i=[3,5] × n_cond=[500,250] × mb=[5,15,25,50] × dir=[buy,sell]` = 32 эксперимента
- **c10x** (`configs_context_500_c10x/`): `c=10*i`, constraint `11*i*mb ≤ 500`, 10 пар × 2 dir = 20 экспериментов
- **c10x_v2** (`configs_context_500_c10x_v2/`): добавлен `order_volume=[75,300,485]`, 10 пар × 3 vol × 2 dir = 60 экспериментов

### Именование
- Конфиги: `cfg_i{i}_c{c}_cond{n_cond}_mb{mb}_{direction}.yaml` или `cfg_i{i}_c{c}_mb{mb}_v{vol}_{direction}.yaml`
- Выходные папки: `context_{n_cond}_{direction}/i{i}_c{c}_mb{mb}_v{vol}_cntxt{pct}%/`
- `cntxt%` = `(11*i*mb)*100/n_cond_msgs` — доля контекста, использованная при генерации

### 4 типа сценариев
| Сценарий | Скрипт | Модель | Hidden state | Checkpoint |
|----------|--------|--------|-------------|------------|
| Aggressive S5 | `1.aggressive_scenario_s5.py` | S5 neural | Continuous | Да |
| Historic | `2.historic_scenario.py` | Нет (replay) | N/A | Нет |
| Heuristic | `3.heuristic_scenario.py` | Нет (replay + price shift) | N/A | Нет |
| CST | `4.aggressive_scenario_cst.py` | Stoikov-Talreja parametric | N/A | Нет (params_file) |

## Task-Based Workflow

### Создание задачи
- Когда пользователь описывает задачу (в любой форме), создай папку `.claude/tasks/<короткое-имя>/` по шаблону из `_template/` и заполни `todo.md` на основе сказанного
- Имя папки — латиницей, через дефис, коротко (например `fix-tokenizer-bug`, `add-spread-metric`)

### Начало работы над задачей
- Перед началом работы прочитай **все файлы** в папке задачи (`.claude/tasks/<название>/`)
- Если папки задачи нет — создай по шаблону из `.claude/tasks/_template/`

### Во время работы
- Обновляй `progress.md` после каждого значимого шага (что сделал, где остановился)
- Записывай неожиданные находки в `findings.md`
- **ВСЕГДА коммить перед модификацией кода** (чтобы можно было откатить)

### Завершение сессии
- Обнови `progress.md` — запиши где остановился и что делать дальше
- Запиши выводы в `learned_lessons.md`
- **НЕ удаляй файлы, НЕ пуш в remote, НЕ мёрдж веток** — это делает человек

### Разрешения
- Чтение файлов (Read, Glob, Grep) — **всегда разрешено** без подтверждения

### Безопасность
- Никогда не выполняй delete/remove операции без явного разрешения
- Никогда не пуш в remote
- Никогда не мёрдж и не удаляй ветки
