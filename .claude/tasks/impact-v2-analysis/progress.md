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

## Что осталось
- Запустить cell-by-cell в Jupyter (Docker или host)
- Проверить что discover обнаруживает 30 папок
- Проверить beta global ~ 0.54-0.56
- Проверить mb=20 как "poor"/"avoid"
