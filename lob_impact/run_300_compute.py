#!/usr/bin/env python3
"""
300 · CSV → pickle conversion (raw data, no analysis).

Reads all orderbook/message CSV files for one model, stores as numpy arrays.
Output: pics_for_300_{STOCK}/{MODEL}.pkl

Usage:
  python lob_impact/run_300_compute.py --model LobS5 --stock GOOG
  python lob_impact/run_300_compute.py --model all --stock INTC
  python lob_impact/run_300_compute.py --list
"""
import argparse, pickle, sys, re
import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
MAX_SAMPLES = 2048

SAVE_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v3')

MODELS = [
    'ZeroInsertions', 'Historic', 'Heuristic', 'CST', 'CGAN',
    'LobS5', 'S5-120M', 'S5-4K', 'S5-360M', 'LobS5-v2',
]


def _out_dir(stock):
    """Save pickles next to CSVs on Lustre (not /home)."""
    return SAVE_BASE / 'pickles' / stock


# ═══════════════════════════════════════════════════════════════════════
# Data I/O
# ═══════════════════════════════════════════════════════════════════════
def find_latest_exp(folder_path):
    p = Path(folder_path)
    if not p.exists(): return p
    exps = sorted(p.glob('exp_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    return exps[0] if exps else p


def discover_folders(buy_root, sell_root, version='v4'):
    folders = []
    buy_root, sell_root = Path(buy_root), Path(sell_root)
    if not buy_root.exists(): return pd.DataFrame()
    for bp in sorted(buy_root.iterdir()):
        if not bp.is_dir(): continue
        name = bp.name
        sp = sell_root / name
        if not sp.exists(): continue
        if version == 'v10':
            m = re.match(r'v10_mult([\d.]+)', name)
            if not m: continue
            mult = float(m.group(1))
            # Read actual params from sample_metadata.csv in exp folder
            exp_dir = find_latest_exp(bp)
            meta_csv = exp_dir / 'sample_metadata.csv'
            if meta_csv.exists():
                import pandas as _pd
                _meta = _pd.read_csv(meta_csv)
                child_med = int(_meta['child'].median())
                mb_med = int(_meta['mb_day'].median())
            else:
                child_med, mb_med = 0, 0
            folders.append(dict(
                folder=name, i=10, c=0,
                mb=mb_med, vol=10 * child_med,  # Q = i * child
                cntxt_pct=0, child=child_med, mult=mult,
                buy_path=str(find_latest_exp(bp)), sell_path=str(find_latest_exp(sp))))
        elif version in ('v5', 'v7', 'v8', 'v9', 'final'):
            m = re.match(r'i(\d+)_c(\d+)_mb(\d+)_child(\d+)_Q(\d+)', name)
            if not m: continue
            folders.append(dict(
                folder=name, i=int(m.group(1)), c=int(m.group(2)),
                mb=int(m.group(3)), vol=int(m.group(5)),  # Q → vol for compat
                cntxt_pct=int(100 * 11 * int(m.group(1)) * int(m.group(3)) / 500),
                child=int(m.group(4)),
                buy_path=str(find_latest_exp(bp)), sell_path=str(find_latest_exp(sp))))
        else:
            m = re.match(r'i(\d+)_c(\d+)_mb(\d+)_v(\d+)_cntxt(\d+)%', name)
            if not m: continue
            folders.append(dict(
                folder=name, i=int(m.group(1)), c=int(m.group(2)),
                mb=int(m.group(3)), vol=int(m.group(4)), cntxt_pct=int(m.group(5)),
                buy_path=str(find_latest_exp(bp)), sell_path=str(find_latest_exp(sp))))
    return pd.DataFrame(folders)


def _parse_day(filename):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return m.group(1) if m else None


def load_folder_raw(folder_row, max_samples):
    """Load ALL CSV files for one folder. Returns raw numpy arrays."""
    buy_gen = Path(folder_row['buy_path']) / 'data_gen'
    sell_gen = Path(folder_row['sell_path']) / 'data_gen'
    result = {}
    for direction, gen_dir in [('buy', buy_gen), ('sell', sell_gen)]:
        books, msgs, days = [], [], []
        if not gen_dir.exists():
            result[direction] = dict(books=[], msgs=[], days=[])
            continue
        ob_files = sorted(gen_dir.glob('*_orderbook_*_gen_id_0.csv'))[:max_samples]
        for ob_f in ob_files:
            try:
                book = pd.read_csv(ob_f, header=None).values
                books.append(book)
                days.append(_parse_day(ob_f.name))
                msg_f = ob_f.parent / ob_f.name.replace('_orderbook_', '_message_')
                if msg_f.exists():
                    msgs.append(pd.read_csv(msg_f, header=None).values)
            except:
                pass
        result[direction] = dict(books=books, msgs=msgs, days=days)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def run_convert(model, stock, max_samples, base=None, version='v4'):
    base = base or SAVE_BASE
    out = base / 'pickles' / stock
    out.mkdir(parents=True, exist_ok=True)

    if version in ('v5', 'v7', 'v8', 'v9', 'v10', 'final'):
        buy_root = base / model / f'{version}_buy' / stock
        sell_root = base / model / f'{version}_sell' / stock
    else:
        buy_root = base / model / 'context_500_buy' / stock
        sell_root = base / model / 'context_500_sell' / stock

    print(f'Loading {stock}/{model} (version={version})...')
    grid_df = discover_folders(str(buy_root), str(sell_root), version=version)
    if grid_df.empty:
        print(f'  No data for {model}')
        sys.exit(1)

    data = {}
    n_books = 0
    for _, row in grid_df.iterrows():
        folder = row['folder']
        fd = load_folder_raw(row, max_samples)
        data[folder] = fd
        nb = len(fd['buy']['books']) + len(fd['sell']['books'])
        n_books += nb
        print(f'  {folder}: {len(fd["buy"]["books"])} buy + {len(fd["sell"]["books"])} sell')

    cache = dict(
        model=model,
        stock=stock,
        grid=grid_df,
        data=data,
    )

    pkl_path = out / f'{model}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = pkl_path.stat().st_size / (1024*1024)
    print(f'\nSaved: {pkl_path} ({size_mb:.0f} MB, {n_books} samples)')
    return cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='300 · CSV → pickle (raw data)')
    parser.add_argument('--model', type=str, help='Model name (or "all")')
    parser.add_argument('--stock', type=str, default='GOOG')
    parser.add_argument('--max', type=int, default=MAX_SAMPLES)
    parser.add_argument('--base', type=str, default=None,
                        help='Override SAVE_BASE path (e.g. for v4 experiments)')
    parser.add_argument('--version', type=str, default='v4', choices=['v3', 'v4', 'v5', 'v7', 'v8', 'v9', 'v10', 'final'],
                        help='Experiment version (affects folder naming)')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    base = Path(args.base) if args.base else None

    if args.list:
        print('Available models:')
        for i, m in enumerate(MODELS):
            print(f'  [{i}] {m}')
        sys.exit(0)

    if not args.model:
        parser.print_help()
        sys.exit(1)

    if args.model == 'all':
        for m in MODELS:
            print(f'\n{"="*60}')
            run_convert(m, args.stock, args.max, base, version=args.version)
    else:
        run_convert(args.model, args.stock, args.max, base, version=args.version)
