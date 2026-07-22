# 5_analysis/methodology — методический PDF про маркет-импакт + аудит расчёта β

Один самодостаточный PDF: (1) обзор классики и свежих статей о том, как ПРАВИЛЬНО оценивать
маркет-импакт и воспроизводят ли его генеративные модели LOB; (2) точное описание того, как β
считается в `5_analysis/beta/`; (3) аудит «что правильно ✓ / что чинить ✗» с привязкой к литературе.

Проза — по-русски, формулы/термины/цитаты — по-английски. Документ портативный (только методология
и формулы, без вставки result-PNG проекта).

## Запуск
```bash
bash run_methodology_pdf.sh            # auto: fpdf2 -> matplotlib fallback
ENGINE=matplotlib bash run_methodology_pdf.sh   # форсить fallback (нулевые доп. deps)
```
Запускать в conda env `lobs5` (matplotlib/numpy), НЕ на login-ноде с тяжёлым окружением.
Лаунчер источает conda.sh (`CONDA_SH`, по умолчанию s5e miniforge), ставит `fpdf2 --user` при отсутствии,
пишет timestamped-лог в `logs/` и кладёт результат в `results/`.

## Выход
- `results/market_impact_methodology.pdf` — стабильная копия (последняя сборка).
- `results/market_impact_methodology_<ts>.pdf` — timestamped.
- `logs/methodology_pdf_<ts>.log` — лог сборки.

## Файлы
- `content.py` — весь текст/формулы/таблицы/ссылки (список «блоков»). Правки контента — здесь.
- `make_methodology_pdf.py` — рендерер: формулы → PNG через matplotlib mathtext, вёрстка через fpdf2
  (fallback — matplotlib `PdfPages`). Кириллица — шрифт DejaVuSans из пакета matplotlib.
- `run_methodology_pdf.sh` — лаунчер (конвенция `results/` + `logs/`, timestamp).

Источник фактуры для аудита (read-only): `../beta/*.py`, `../beta/vol_estimators.py`,
`../run_300_analyze_one.py`, `../../3_scenarios/config_bet_composition.yaml`, `run_experiments.sh`,
`../../1_data_prep/compute_sp500_msgs_btw.py`, `../../core/inference_w_insertions.py`.
