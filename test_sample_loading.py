#!/usr/bin/env python3
import json
import os

# Тестируем загрузку файла с индексами
sample_file_path = "random_sample_indices_b64_bs16_ins100_cool20.json"
batch_size = 16
start_batch = 0
end_batch = 64

print(f"Текущая директория: {os.getcwd()}")
print(f"Файл существует: {os.path.exists(sample_file_path)}")

if os.path.exists(sample_file_path):
    try:
        with open(sample_file_path, "r") as f:
            sample_i_full = json.load(f)
        
        print(f"Загружено батчей: {len(sample_i_full)}")
        print(f"Первый батч: {sample_i_full[0]}")
        print(f"Размер первого батча: {len(sample_i_full[0])}")
        
        # Режем по start/end
        sample_i = sample_i_full[start_batch: end_batch if end_batch != -1 else None]
        print(f"Выбранных батчей: {len(sample_i)}")
        
        # Проверяем, что каждый батч нужного размера
        for i, batch in enumerate(sample_i):
            if len(batch) != batch_size:
                print(f"ОШИБКА: Batch {i} имеет неправильный размер {len(batch)}, ожидается {batch_size}")
                break
            if i < 3:  # Показываем только первые 3 батча
                print(f"Batch {i}: {batch}")
        
        print("Все проверки пройдены успешно!")
        
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
else:
    print("Файл не найден!")









