# Задача: Comprehensive Market Impact Analysis (v2)

## Цель
Создать один системный ноутбук `110.market_impact_comprehensive.ipynb`, заменяющий 6 устаревших (100-105).
Два вопроса: (1) модель воспроизводит market impact? (beta~0.5, реалистичный decay)
(2) при каких условиях использовать? (какие (i,mb,V) работают)

## Шаги
- [ ] Создать notebook: Section 0 — Setup & Data Loading (cells 0-7)
- [ ] Section 1 — Decay Curves (cells 8-11)
- [ ] Section 2 — Square-Root Law / Beta (cells 12-18)
- [ ] Section 3 — Volume Dimension (cells 19-24)
- [ ] Section 4 — Quality Heatmaps (cells 25-28)
- [ ] Section 5 — Volume-Time Dynamics (cells 29-35)
- [ ] Section 6 — Stability + Recommendations (cells 36-43)
- [ ] Проверить что discover обнаружил 30 папок (60 buy+sell)

## Критерии готовности
- Notebook запускается cell-by-cell
- discover обнаруживает 30 конфигов
- metrics_df имеет 30 строк с ненулевыми значениями
- Beta global ~0.54-0.56
- Heatmap показывает mb=20 как "poor"/"avoid"
