#!/usr/bin/env python3
"""
Compute depth-at-best statistics from existing experiment pickles.

Reads ONE model's raw pickle (Historic recommended — conditioning is identical
across models), extracts book depth at first aggressive order insertion, and
outputs percentile statistics per stock.

This helps calibrate order_volume for future experiments:
  - vol_small  = p25 depth  (order does NOT eat first level)
  - vol_medium = p75 depth  (eats first level ~75% of the time)
  - vol_large  = p95 depth  (eats first level almost always)
  - vol_xlarge = p99 depth  (eats multiple levels)

Usage:
  python lob_impact/compute_depth_stats.py --stock GOOG
  python lob_impact/compute_depth_stats.py --stock INTC --model Historic
  python lob_impact/compute_depth_stats.py --stock GOOG --model LobS5
"""
import argparse, pickle, sys
import numpy as np
import pandas as pd
from pathlib import Path

PICKLE_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v3/pickles')
TICK_SIZE = 100


def load_aggressive_indices(exp_path):
    aggr_file = Path(exp_path) / 'aggressive_indices.csv'
    if not aggr_file.exists():
        return None
    vals = np.loadtxt(aggr_file, dtype=int)
    return np.atleast_1d(vals)


def compute_depth_stats(stock, model='Historic'):
    raw_pkl = PICKLE_BASE / stock / f'{model}.pkl'
    if not raw_pkl.exists():
        print(f'ERROR: {raw_pkl} not found')
        sys.exit(1)

    print(f'Loading {stock}/{model} ({raw_pkl.stat().st_size/1e9:.1f} GB)...')
    with open(raw_pkl, 'rb') as f:
        md = pickle.load(f)
    grid_df = md['grid']

    _aggr_cache = {}
    records = []

    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in md['data']:
            continue
        fd = md['data'][folder]

        bp, sp = row['buy_path'], row['sell_path']
        if bp not in _aggr_cache:
            _aggr_cache[bp] = load_aggressive_indices(bp)
        if sp not in _aggr_cache:
            _aggr_cache[sp] = load_aggressive_indices(sp)
        aggr_buy = _aggr_cache[bp]
        aggr_sell = _aggr_cache[sp]
        if aggr_buy is None or aggr_sell is None:
            continue

        for direction, aggr_idx, books_list, days_list in [
            ('buy',  aggr_buy,  fd['buy']['books'],  fd['buy']['days']),
            ('sell', aggr_sell, fd['sell']['books'], fd['sell']['days']),
        ]:
            n_aggr = len(aggr_idx)
            if n_aggr < 1:
                continue
            for j, book in enumerate(books_list):
                if aggr_idx[0] >= len(book):
                    continue
                # At first insertion point (k=0): this is the conditioning book
                ask_price = float(book[aggr_idx[0], 0])
                ask_vol   = float(book[aggr_idx[0], 1])
                bid_price = float(book[aggr_idx[0], 2])
                bid_vol   = float(book[aggr_idx[0], 3])
                mid = (ask_price + bid_price) / 2.0
                spread = (ask_price - bid_price) / TICK_SIZE if ask_price > bid_price else 0

                # For buy orders: relevant depth is ask side
                # For sell orders: relevant depth is bid side
                depth_relevant = ask_vol if direction == 'buy' else bid_vol

                # Also collect deeper levels (L2-L5)
                total_ask_5 = sum(float(book[aggr_idx[0], 1 + 4*lv]) for lv in range(min(5, book.shape[1]//4)))
                total_bid_5 = sum(float(book[aggr_idx[0], 3 + 4*lv]) for lv in range(min(5, book.shape[1]//4)))

                day = days_list[j] if j < len(days_list) else None

                records.append(dict(
                    folder=folder, direction=direction, sample_j=j, day=day,
                    ask_vol_best=ask_vol, bid_vol_best=bid_vol,
                    depth_relevant=depth_relevant,
                    total_ask_5lvl=total_ask_5, total_bid_5lvl=total_bid_5,
                    mid=mid, spread_ticks=spread,
                    i=row['i'], mb=row['mb'], vol=row['vol'],
                ))

    df = pd.DataFrame(records)
    print(f'\nCollected {len(df)} depth observations from {len(grid_df)} configs')

    if df.empty:
        print('No data found.')
        return df

    # Percentile summary
    print(f'\n{"="*60}')
    print(f'  Depth Statistics: {stock} (model={model})')
    print(f'{"="*60}')

    for col, label in [
        ('depth_relevant', 'Depth at best (relevant side)'),
        ('ask_vol_best',   'Ask volume at best'),
        ('bid_vol_best',   'Bid volume at best'),
        ('total_ask_5lvl', 'Total ask volume (5 levels)'),
        ('total_bid_5lvl', 'Total bid volume (5 levels)'),
        ('spread_ticks',   'Spread (ticks)'),
    ]:
        vals = df[col].dropna().values
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        print(f'\n  {label} (n={len(vals)}):')
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f'    p{p:>2} = {np.percentile(vals, p):>10.0f}')
        print(f'    mean = {np.mean(vals):>10.0f}')

    # Suggested volumes
    depth = df['depth_relevant'].dropna().values
    depth = depth[depth > 0]
    if len(depth) > 0:
        print(f'\n{"="*60}')
        print(f'  SUGGESTED ORDER VOLUMES for {stock}:')
        print(f'{"="*60}')
        p25 = np.percentile(depth, 25)
        p75 = np.percentile(depth, 75)
        p95 = np.percentile(depth, 95)
        p99 = np.percentile(depth, 99)
        print(f'  vol_small  = {p25:>6.0f}  (p25 — does NOT eat first level)')
        print(f'  vol_medium = {p75:>6.0f}  (p75 — eats at ~75% of samples)')
        print(f'  vol_large  = {p95:>6.0f}  (p95 — eats almost always)')
        print(f'  vol_xlarge = {p99:>6.0f}  (p99 — eats multiple levels)')
        print(f'\n  Current volumes: 75, 300, 485')
        depth_sorted = np.sort(depth)
        n = len(depth_sorted)
        for v in [75, 300, 485]:
            pct = int(np.searchsorted(depth_sorted, v) * 100 // n)
            print(f'  {v} shares = p{pct} of depth')

    return df


def main():
    parser = argparse.ArgumentParser(description='Compute depth-at-best statistics')
    parser.add_argument('--stock', type=str, default='GOOG')
    parser.add_argument('--model', type=str, default='Historic',
                        help='Model pickle to read (Historic recommended)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: lob_impact/depth_stats_{STOCK}.csv)')
    args = parser.parse_args()

    stock = args.stock.upper()
    df = compute_depth_stats(stock, args.model)

    if not df.empty:
        out_path = args.output or f'lob_impact/depth_stats_{stock}.csv'
        df.to_csv(out_path, index=False)
        print(f'\nSaved raw data: {out_path} ({len(df)} rows)')


if __name__ == '__main__':
    main()
