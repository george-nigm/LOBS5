# Находки

- V2 папки в context_500_buy/ и context_500_sell/ — формат i{i}_c{c}_mb{mb}_v{V}_cntxt{pct}%
- Старые V1 папки (без _v) удалены — только V2 остались
- 30 папок x 2 стороны = 60 buy+sell экспериментов
- aggressive_indices.csv одинаковы для всех V при фиксированных (i,mb) — как и ожидалось
- sample_day_map.csv: 2048 строк, 9 тестовых дней
- Код из 103/104/105 почти полностью reusable, главное изменение — parse_folder_params_v2 с V-группой и discover_v2_folders вместо хардкода
- В 103 compute_combined_impact использовал parse_folder_params (3 return values), в V2 нужен parse_folder_params_v2 (4 return values с V)
- compute_beta_3modes теперь принимает buy_path/sell_path явно (вместо глобальных BUY_PATH/SELL_PATH + folder)
