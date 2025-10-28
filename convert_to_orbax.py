#!/usr/bin/env python3
"""
Скрипт для конвертации legacy чекпоинтов (msgpack, pickle, старый формат Flax) в формат Orbax.

Поддерживает:
1. Flax msgpack чекпоинты
2. Legacy Flax чекпоинты  
3. Pickle чекпоинты
4. PureJAXRL-style чекпоинты

Использование:
    python convert_to_orbax.py --input /путь/к/чекпоинту --output /путь/к/orbax_чекпоинту
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import serialization


def load_msgpack_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Загрузить msgpack чекпоинт (legacy формат Flax)."""
    print(f"Загрузка msgpack чекпоинта из: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        data = f.read()
    restored = serialization.msgpack_restore(data)
    print(f"Msgpack чекпоинт успешно загружен")
    return restored


def _robust_load_file(file_path: Path) -> Any:
    """Попытаться загрузить файл различными способами.

    Порядок:
    1) Flax msgpack
    2) pickle
    3) numpy (np.load для npy/npz)
    4) fallback: сырые байты
    """
    # 1) msgpack
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return serialization.msgpack_restore(data)
    except Exception:
        pass

    # 2) pickle
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # 3) numpy (на случай npy/npz под нестандартным расширением)
    try:
        obj = np.load(str(file_path), allow_pickle=True)
        # Для npz возвращается NpzFile, приводим к dict
        if hasattr(obj, "files"):
            return {k: obj[k] for k in obj.files}
        return obj
    except Exception:
        pass

    # 4) fallback сырые байты
    with open(file_path, "rb") as f:
        return f.read()


def detect_checkpoint_type(checkpoint_path: str) -> str:
    """Определить тип чекпоинта."""
    path = Path(checkpoint_path)
    
    if path.is_dir():
        if (path / "metadata.txt").exists():
            return "flax_legacy_dir"
        elif (path / "_CHECKPOINT_METADATA").exists():
            return "orbax"
        else:
            return "unknown_dir"
    
    elif path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".msgpack":
            return "msgpack"
        elif suffix in [".pkl", ".pickle"]:
            return "pickle"
        elif suffix == ".model":
            return "flax_model"
    
    return "unknown"


def load_checkpoint_auto(checkpoint_path: str) -> Dict[str, Any]:
    """Автоматически определить и загрузить чекпоинт."""
    ckpt_type = detect_checkpoint_type(checkpoint_path)
    print(f"Обнаружен тип чекпоинта: {ckpt_type}")

    # Вспомогательная функция для поиска вложенной директории шага
    def _find_nested_flax_dir(root: Path) -> Optional[Path]:
        # 1) Если есть final_old — берём его
        final_old = root / "final_old"
        if final_old.is_dir():
            return final_old
        # 2) Ищем наибольшую по номеру директорию со структурой *.model
        numeric_dirs = []
        for child in root.iterdir():
            if child.is_dir() and child.name.isdigit():
                if any((child / name).exists() for name in ["params.model", "state.model", "optimizer.model", "metadata.txt"]):
                    numeric_dirs.append(int(child.name))
        if numeric_dirs:
            chosen = root / str(max(numeric_dirs))
            return chosen
        # 3) На крайний случай — поддеректория metadata с *.model
        meta_dir = root / "metadata"
        if meta_dir.is_dir() and any((meta_dir / name).exists() for name in ["params.model", "state.model", "optimizer.model", "metadata.txt"]):
            return meta_dir
        return None

    if ckpt_type == "msgpack" or ckpt_type == "flax_model":
        return load_msgpack_checkpoint(checkpoint_path)
    elif ckpt_type == "pickle":
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    elif ckpt_type == "flax_legacy_dir":
        path = Path(checkpoint_path)
        # Ищем .msgpack или .model файлы
        msgpack_files = list(path.glob("*.msgpack")) + list(path.glob("*.model"))
        if msgpack_files:
            # Если это директория со множеством файлов, загрузим все ключевые и соберём dict
            result: Dict[str, Any] = {}
            for fp in msgpack_files:
                name = fp.stem
                try:
                    result[name] = _robust_load_file(fp)
                except Exception as e:
                    raise ValueError(f"Не удалось загрузить {fp}: {e}")
            if result:
                return result
            # как запасной вариант — попробовать первый как msgpack
            return load_msgpack_checkpoint(str(msgpack_files[0]))
        else:
            raise ValueError(f"Не найдены файлы чекпоинта в {checkpoint_path}")
    else:
        # Обработка случая, когда задана корневая директория с поддиректориями шагов (1, 2, ..., final_old)
        root = Path(checkpoint_path)
        if root.is_dir():
            nested = _find_nested_flax_dir(root)
            if nested is not None:
                print(f"Найдена вложенная директория шага: {nested}")
                msgpack_files = list(nested.glob("*.msgpack")) + list(nested.glob("*.model"))
                if msgpack_files:
                    result: Dict[str, Any] = {}
                    for fp in msgpack_files:
                        name = fp.stem
                        try:
                            result[name] = _robust_load_file(fp)
                        except Exception as e:
                            raise ValueError(f"Не удалось загрузить {fp}: {e}")
                    if result:
                        return result
                raise ValueError(f"Не найдены файлы чекпоинта в {nested}")
        raise ValueError(f"Неизвестный тип чекпоинта: {ckpt_type}")


def save_orbax_checkpoint(
    checkpoint_data: Dict[str, Any],
    output_dir: str,
    step: int = 0,
) -> None:
    """Сохранить чекпоинт в формате Orbax с использованием CheckpointManager."""
    print(f"Сохранение чекпоинта в формат Orbax: {output_dir}")
    
    # Orbax требует абсолютный путь для директории чекпоинта
    output_path = Path(output_dir).resolve()
    
    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        create=True,
    )
    
    # Подготовим состояние модели.
    # Если в legacy есть полноценный снимок 'state', используем его как есть.
    if isinstance(checkpoint_data, dict) and 'state' in checkpoint_data:
        state_to_save = checkpoint_data['state']
    elif isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        state_to_save = {'params': checkpoint_data['params']}
    else:
        state_to_save = {'params': checkpoint_data}
    # Добавим пустые batch_stats, если отсутствуют
    if isinstance(state_to_save, dict) and 'batch_stats' not in state_to_save:
        state_to_save['batch_stats'] = {}

    # Минимальная конфигурация, необходимая для init_train_state
    default_config = {
        "USE_WANDB": True,
        "wandb_project": "LOBS5",
        "wandb_entity": "george-nigm",
        "dir_name": "./data/GOOG",
        "dataset": "lobster-prediction",
        "masking": "causal",
        "use_book_data": True,
        "use_simple_book": False,
        "book_transform": True,
        "book_depth": 500,
        "restore": None,
        "restore_step": None,
        "msg_seq_len": 500,
        "n_data_workers": 4,
        "n_message_layers": 2,
        "n_book_pre_layers": 1,
        "n_book_post_layers": 1,
        "n_layers": 12,
        "d_model": 512,
        "ssm_size_base": 512,
        "blocks": 16,
        "C_init": "trunc_standard_normal",
        "discretization": "zoh",
        "mode": "pool",
        "activation_fn": "half_glu1",
        "conj_sym": True,
        "clip_eigs": True,
        "bidirectional": True,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "prenorm": True,
        "batchnorm": True,
        "bn_momentum": 0.95,
        "bsz": 128,
        "num_devices": 8,
        "epochs": 100,
        "early_stop_patience": 1000,
        "ssm_lr_base": 5e-4,
        "lr_factor": 1.0,
        "dt_global": False,
        "lr_min": 0,
        "cosine_anneal": True,
        "warmup_end": 1,
        "lr_patience": 1_000_000,
        "reduce_factor": 1.0,
        "p_dropout": 0.0,
        "weight_decay": 0.05,
        "opt_config": "standard",
        "jax_seed": 42,
    }

    # Сохраняем state + metadata как составной чекпоинт.
    with ocp.CheckpointManager(
        output_path,
        item_names=("state", "metadata"),
        options=options,
    ) as mngr:
        composite_args = ocp.args.Composite(
            state=ocp.args.StandardSave(state_to_save),
            metadata=ocp.args.JsonSave(default_config),
        )
        mngr.save(step, args=composite_args)

    # На всякий случай создадим файл metadata/metadata по пути шага, который читает пользовательский код
    step_dir = output_path / str(step) / "default" / "metadata"
    step_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(step_dir / "metadata", "w") as f:
        _json.dump(default_config, f)

    # Совместимость с путём вида <ckpt>/0/default: создаём вложенную папку шага
    import shutil as _shutil
    outer_default = output_path / str(step) / "default"
    inner_step = outer_default / str(step)
    if not inner_step.exists():
        inner_step.mkdir(parents=True, exist_ok=True)
        # скопируем только 'metadata' если есть
        for sub in ["metadata", "_CHECKPOINT_METADATA"]:
            src = outer_default / sub
            dst = inner_step / sub
            if src.exists():
                if src.is_dir():
                    _shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    _shutil.copy2(src, dst)

    # Убедимся, что структура шагов присутствует
    
    print(f"Чекпоинт успешно сохранен в формате Orbax")


def convert_checkpoint(
    input_path: str,
    output_dir: str,
    step: int = 0,
    dry_run: bool = False,
) -> None:
    """Основная функция конвертации."""
    print("=" * 60)
    print("Конвертация чекпоинта в формат Orbax")
    print("=" * 60)
    
    checkpoint_data = load_checkpoint_auto(input_path)
    
    if dry_run:
        print("\nDRY RUN: Структура чекпоинта:")
        import jax.tree_util as jtu
        print(jtu.tree_map(lambda x: f"{type(x).__name__} {getattr(x, 'shape', '')}", checkpoint_data))
        return
    
    save_orbax_checkpoint(checkpoint_data, output_dir, step)
    
    print("\n" + "=" * 60)
    print("Конвертация завершена!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация legacy чекпоинтов в формат Orbax"
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Путь к входному чекпоинту (файл или директория)")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Путь к выходной директории для Orbax чекпоинта")
    parser.add_argument("--step", type=int, default=0,
                       help="Номер шага для чекпоинта (по умолчанию: 0)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Показать структуру без сохранения")
    
    args = parser.parse_args()
    
    convert_checkpoint(
        input_path=args.input,
        output_dir=args.output,
        step=args.step,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()



# Конвертировать директорию с чекпоинтом
# python convert_to_orbax.py --input ./goog2022_rwkv_6g0.1B --output ./orbax_checkpoint

# docker run --rm --gpus '"device=7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name convert_to_orbax_7 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u convert_to_orbax.py --input checkpoints/goog2022_rwkv_6g0.1B --output checkpoints/orbax_goog2022_rwkv_6g0.1B"