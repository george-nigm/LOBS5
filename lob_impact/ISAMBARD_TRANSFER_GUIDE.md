# Isambard Transfer Guide — Market Impact v3

## Зачем всё это

### Проблема
Модель LOBS5 обучена на 8 тикеров × 4 года данных. В v2 статьи мы тестировали на **GOOG Jan 2023** — но эти данные **входили в обучающую выборку**. Это **data leakage** — ревьюеры заметят.

### Решение
Перейти на **January 2026** данные (Year 5, out-of-sample). Плюс добавить **Intel (INTC)** для cross-stock валидации.

### Что уже сделано (Phase A)
Все скрипты подготовлены:
- `0.null_baseline_s5.py` — нулевой контроль (генерация без инъекций)
- `run_isambard_c10x_v2.sh` — SLURM скрипт для всех моделей
- `run_isambard_null_baseline.sh` — SLURM для нулевой базовой линии
- `210.paper_v3_final.ipynb` — ноутбук с анализом (включая Hurst, propagator, spread)
- Статья обновлена (tex/bib) с новыми цитатами и секциями

### Что надо сделать (Phase B-D)
1. Перенести файлы на Isambard
2. Подготовить GOOG/INTC Jan 2026 данные
3. Запустить 960 экспериментов
4. Прогнать анализ, обновить статью

---

## Что скачать и перенести на Isambard

### 1. Код проекта (ОБЯЗАТЕЛЬНО)

Весь репо. На Isambard уже есть клон — нужно `git pull` на ветке `lob_impact`.

```bash
# На Isambard:
cd /lus/lfs1aip2/home/s5e/LOBS5
git pull origin lob_impact
```

Новые файлы (после коммита):
- `lob_impact/0.null_baseline_s5.py`
- `lob_impact/run_isambard_c10x_v2.sh`
- `lob_impact/run_isambard_null_baseline.sh`
- `lob_impact/210.paper_v3_final.ipynb`

---

### 2. CGAN — нужно скачать (~33 MB)

CGAN — это единственная модель, которую **нельзя** пересчитать на Isambard (обучена на специфических данных, требует ABIDES framework).

**Скачать эти файлы с flair:**

| Файл | Путь на flair | Размер |
|------|--------------|--------|
| Чекпоинт | `/homes/groups/finance/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_['20221228', '20221229', '20221230']_v2_41/checkpoints/model_39.ckpt` | 8.9 MB |
| Скейлеры | `.../data__scalers.pickle` | 2 KB |
| Interarrival | `.../interarrival_times` | 152 B |
| Hyperparams | `.../hyperpars.txt` | 384 B |
| Сценарий | `lob_impact/5v2.aggressive_scenario_cgan.py` | 40 KB |
| Моки | `lob_impact/_cgan_mocks.py` | 8 KB |
| **ABIDES код** | `abides_worldmodel_offline/` | **24 MB** |

**Быстрая сборка архива на flair:**
```bash
cd /scratch/local/homes/80/georgenigm/LOBS5
mkdir -p /tmp/cgan_transfer/cgan_checkpoint
cp /homes/groups/finance/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_\[\'20221228\',\ \'20221229\',\ \'20221230\'\]_v2_41/checkpoints/model_39.ckpt /tmp/cgan_transfer/cgan_checkpoint/
cp /homes/groups/finance/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_\[\'20221228\',\ \'20221229\',\ \'20221230\'\]_v2_41/data__scalers.pickle /tmp/cgan_transfer/cgan_checkpoint/
cp /homes/groups/finance/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_\[\'20221228\',\ \'20221229\',\ \'20221230\'\]_v2_41/interarrival_times /tmp/cgan_transfer/cgan_checkpoint/
cp /homes/groups/finance/data/cgan/GOOG/models/GOOG/NEW_SET_lb100_\[\'20221228\',\ \'20221229\',\ \'20221230\'\]_v2_41/hyperpars.txt /tmp/cgan_transfer/cgan_checkpoint/
cp lob_impact/5v2.aggressive_scenario_cgan.py /tmp/cgan_transfer/
cp lob_impact/_cgan_mocks.py /tmp/cgan_transfer/
cp -r abides_worldmodel_offline/ /tmp/cgan_transfer/
tar czf ~/cgan_transfer.tar.gz -C /tmp cgan_transfer/
echo "Архив: ~/cgan_transfer.tar.gz ($(du -sh ~/cgan_transfer.tar.gz | cut -f1))"
```

**На Isambard — распаковать:**
```bash
cd /lus/lfs1aip2/home/s5e/LOBS5
tar xzf cgan_transfer.tar.gz
# Переместить чекпоинт:
mkdir -p /lus/lfs1aip2/home/s5e/data/checkpoints/cgan
mv cgan_transfer/cgan_checkpoint/* /lus/lfs1aip2/home/s5e/data/checkpoints/cgan/
# Код:
mv cgan_transfer/5v2.aggressive_scenario_cgan.py lob_impact/
mv cgan_transfer/_cgan_mocks.py lob_impact/
mv cgan_transfer/abides_worldmodel_offline/ .
```

---

### 3. CST — пересчитать на Isambard (НЕ скачивать)

CST — это параметрическая модель (Cont-Stoikov-Talreja). Параметры **калибруются из сырых LOBSTER данных**. Так как мы переходим на **Jan 2026** данные, старые параметры (калиброванные на Jan 2023) всё равно нерелевантны.

**На Isambard:**
```bash
cd /lus/lfs1aip2/home/s5e/LOBS5

# 1. Калибровка параметров из новых данных GOOG Jan 2026
python3 -c "
import sys
sys.path.insert(0, 'lob_bench/cst_model')
from param_estimation import estimate_from_data_files
params = estimate_from_data_files(
    '/lus/lfs1aip2/home/s5e/data/rawLOBSTER/GOOG/JAN2026',
    save_path='/lus/lfs1aip2/home/s5e/data/cst_params_goog_2026.pkl',
    tick_size=100,
    num_ticks=500,
)
print('CST params:', list(params.keys()))
"

# 2. То же для INTC
python3 -c "
import sys
sys.path.insert(0, 'lob_bench/cst_model')
from param_estimation import estimate_from_data_files
params = estimate_from_data_files(
    '/lus/lfs1aip2/home/s5e/data/rawLOBSTER/INTC/JAN2026',
    save_path='/lus/lfs1aip2/home/s5e/data/cst_params_intc_2026.pkl',
    tick_size=100,
    num_ticks=500,
)
"
```

CST код уже в репо: `lob_bench/cst_model/` и `lob_impact/4.aggressive_scenario_cst.py`.

---

### 4. Данные LOBSTER Jan 2026

**Нужны для обоих тикеров:**

| Тикер | Что нужно | Назначение |
|-------|-----------|------------|
| GOOG Jan 2026 | rawLOBSTER + preprocessed .npy | Основные результаты |
| INTC Jan 2026 | rawLOBSTER + preprocessed .npy | Кросс-валидация (Appendix) |

**Предобработка (если ещё нет .npy):**
```bash
python preproc.py \
    --data_dir /lus/.../rawLOBSTER/GOOG/JAN2026 \
    --save_dir /lus/.../processed_data/GOOG/2026_Jan \
    --n_tick_range 500 --use_raw_book_repr

python preproc.py \
    --data_dir /lus/.../rawLOBSTER/INTC/JAN2026 \
    --save_dir /lus/.../processed_data/INTC/2026_Jan \
    --n_tick_range 500 --use_raw_book_repr
```

---

## Эксперименты на Isambard

### Порядок запуска

```
1. Подготовить данные (preproc.py)
2. Калибровать CST (см. выше)
3. Тест — один эксперимент:
   MODEL=lobs5 STOCK=GOOG sbatch lob_impact/run_isambard_c10x_v2.sh
4. Если ОК — все S5 модели:
   MODEL=lobs5 sbatch lob_impact/run_isambard_c10x_v2.sh
   MODEL=s5_120m sbatch lob_impact/run_isambard_c10x_v2.sh
   MODEL=s5_4k sbatch lob_impact/run_isambard_c10x_v2.sh
5. Базовые линии (Historic, Heuristic, CST):
   MODEL=historic sbatch ...
   MODEL=heuristic sbatch ...
   MODEL=cst sbatch ...
6. CGAN (после переноса):
   MODEL=cgan sbatch ...
7. Null baseline:
   sbatch lob_impact/run_isambard_null_baseline.sh
```

### Общий объём
- 60 конфигов × 8 моделей × 2 стока = **960 экспериментов**
- ~2048 samples × 64 batch = 32 батча на каждый
- На 4 GPU (GH200) параллельно: ~4-8 часов на модель × сток

---

## Структура файлов проекта

```
LOBS5/
├── lob_impact/
│   ├── 0.null_baseline_s5.py          ← NEW: нулевой контроль
│   ├── 1.aggressive_scenario_s5.py     ← S5 v2 (22-tok)
│   ├── 1.aggressive_scenario_s5_v3.py  ← S5 v3 (24-tok)
│   ├── 2.historic_scenario.py          ← Historic replay
│   ├── 3.heuristic_scenario.py         ← Heuristic price-shift
│   ├── 4.aggressive_scenario_cst.py    ← CST parametric
│   ├── 5v2.aggressive_scenario_cgan.py ← CGAN (нужно перенести)
│   ├── _cgan_mocks.py                  ← CGAN mock dependencies
│   ├── run_isambard_c10x_v2.sh         ← NEW: SLURM скрипт
│   ├── run_isambard_null_baseline.sh   ← NEW: SLURM null baseline
│   ├── 210.paper_v3_final.ipynb        ← NEW: анализ
│   └── configs_isambard_*/             ← генерируются скриптом
├── lob_bench/cst_model/                ← CST код (уже в репо)
├── abides_worldmodel_offline/          ← CGAN ABIDES (нужно перенести)
└── overleaf/.../sample-sigplan.tex     ← обновлённая статья
```

---

## Чеклист для скачивания

- [ ] `git pull` на Isambard (ветка `lob_impact`)
- [ ] Архив `~/cgan_transfer.tar.gz` (~33 MB) — скачать, перенести на Isambard
- [ ] Данные GOOG Jan 2026 rawLOBSTER — проверить что на Isambard есть
- [ ] Данные INTC Jan 2026 rawLOBSTER — купить/скачать с lobsterdata.com если нет
- [ ] Overleaf tex/bib — скопировать вручную в Overleaf проект
