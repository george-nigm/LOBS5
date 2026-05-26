#!/usr/bin/env python3
"""Diagnostic 4: Alternative Volume Normalizations.

Tests whether V_daily is the wrong normalizer by trying V_local, V=1, depth, etc.
Square-root law: I/σ = C·(Q/V)^β — if V is wrong scale, β is biased.

Usage:
    python lob_impact/diag_4_normalizations.py --stock GOOG
"""
import argparse, pickle, sys, re
import numpy as np
import pandas as pd
from pathlib import Path

TICK_SIZE = 100
PICKLE_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v4/pickles')
OUT_DIR = Path('pics_for_investigation')
MODELS_TEST = ['Historic', 'LobS5', 'CST']


def load_daily_params(stock):
    hl_path = Path(f'lob_impact/daily_h_l_{stock}.csv')
    sdm_path = Path(f'lob_impact/sample_day_map_{stock}.csv')
    params = {}
    if sdm_path.exists():
        sdm = pd.read_csv(sdm_path)
        for _, row in sdm.iterrows():
            day = row.get('day', '')
            if not day:
                continue
            H, L = row['highest_price'], row['lowest_price']
            V = row['execution_sum']
            sigma = np.log(H / L) / 0.8325546 if H > 0 and L > 0 else 1.0
            params[day] = dict(V=V, sigma=sigma)
    elif hl_path.exists():
        df = pd.read_csv(hl_path)
        if 'day' not in df.columns:
            df['day'] = df['filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
        for _, row in df.iterrows():
            day = row.get('day', '')
            if pd.isna(day):
                continue
            H, L = row['highest_price'], row['lowest_price']
            V = row['execution_sum']
            sigma = np.log(H / L) / 0.8325546 if H > 0 and L > 0 else 1.0
            params[day] = dict(V=V, sigma=sigma)
    return params


def load_aggressive_indices(exp_path):
    f = Path(exp_path) / 'aggressive_indices.csv'
    if not f.exists():
        return None
    vals = np.loadtxt(f, dtype=int)
    return np.atleast_1d(vals)


def extract_multi_norm_cloud(md, daily_params):
    """Extract point cloud with multiple normalizations per point."""
    grid_df = md['grid']
    points = []
    _aggr_cache = {}

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
        aggr_buy, aggr_sell = _aggr_cache[bp], _aggr_cache[sp]
        if aggr_buy is None or aggr_sell is None:
            continue

        for direction, aggr_idx, msgs_list, books_list, days_list in [
            ('buy', aggr_buy, fd['buy']['msgs'], fd['buy']['books'], fd['buy']['days']),
            ('sell', aggr_sell, fd['sell']['msgs'], fd['sell']['books'], fd['sell']['days']),
        ]:
            n_aggr = len(aggr_idx)
            if n_aggr < 2:
                continue
            for j in range(min(len(msgs_list), len(books_list))):
                msg = msgs_list[j]
                book = books_list[j]
                if len(msg) == 0 or aggr_idx.max() >= len(msg) or aggr_idx.max() >= len(book):
                    continue

                ref_mid = (float(book[aggr_idx[0], 0]) + float(book[aggr_idx[0], 2])) / 2.0
                if ref_mid <= 0:
                    continue
                sizes = msg[aggr_idx, 3].astype(float)
                prices = msg[aggr_idx, 4].astype(float)
                if np.any(sizes <= 0) or np.any(prices <= 0):
                    continue

                Q_cum = np.cumsum(sizes)
                vwap = np.cumsum(sizes * prices) / Q_cum
                impact = np.abs((vwap - ref_mid) / ref_mid) if direction == 'buy' else np.abs((ref_mid - vwap) / ref_mid)

                # V_local: total exec volume in sample
                exec_mask = (msg[:, 1] == 4)
                V_local = float(msg[exec_mask, 3].astype(float).sum()) if exec_mask.any() else 1.0

                # Depth at best at first insertion
                depth_best = float(book[aggr_idx[0], 1]) if direction == 'buy' else float(book[aggr_idx[0], 3])
                depth_best = max(depth_best, 1.0)

                # Daily params
                day = days_list[j] if j < len(days_list) else None
                V_daily = 1e6  # fallback
                sigma = 1.0
                if day and daily_params:
                    # Try exact match or find closest
                    if day in daily_params:
                        V_daily = daily_params[day]['V']
                        sigma = daily_params[day]['sigma']
                    else:
                        # Ordinal mapping
                        all_days = sorted(daily_params.keys())
                        if all_days:
                            V_daily = daily_params[all_days[0]]['V']
                            sigma = daily_params[all_days[0]]['sigma']

                for k in range(n_aggr):
                    if impact[k] <= 1e-10 or Q_cum[k] <= 0:
                        continue
                    I = float(impact[k])
                    Q = float(Q_cum[k])
                    points.append(dict(
                        I=I, Q=Q, sigma=sigma, V_daily=V_daily, V_local=V_local,
                        depth=depth_best, ref_mid=ref_mid, size_k=float(sizes[k]),
                        k=k+1, vol=row['vol'], i=row['i'], mb=row['mb'],
                    ))

    return pd.DataFrame(points)


def compute_beta_intercept(x, y):
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    xv, yv = x[ok], y[ok]
    if len(xv) < 2:
        return np.nan, np.nan, 0
    coeffs = np.polyfit(xv, yv, 1)
    beta = float(coeffs[0])
    yhat = beta * xv + coeffs[1]
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2, int(ok.sum())


def compute_beta_origin(x, y_adj):
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    xv, yv = x[ok], y_adj[ok]
    if len(xv) < 2:
        return np.nan
    return float(np.dot(xv, yv) / np.dot(xv, xv))


def run(stock):
    OUT_DIR.mkdir(exist_ok=True)
    daily_params = load_daily_params(stock)
    lines = [f'=== Diagnostic 4: Alternative Normalizations ({stock}) ===\n']

    results = []
    for model in MODELS_TEST:
        pkl_path = PICKLE_BASE / stock / f'{model}.pkl'
        if not pkl_path.exists():
            lines.append(f'SKIP {model}: {pkl_path} not found')
            continue
        print(f'Loading {model}...')
        with open(pkl_path, 'rb') as f:
            md = pickle.load(f)

        pc = extract_multi_norm_cloud(md, daily_params)
        if pc.empty:
            lines.append(f'SKIP {model}: empty point cloud')
            continue
        print(f'  {model}: {len(pc)} points')

        y = np.log(pc['I'].values)

        normalizations = {
            'V_daily':  np.log(pc['Q'].values / pc['V_daily'].values),
            'V_local':  np.log(pc['Q'].values / pc['V_local'].values),
            'V=1':      np.log(pc['Q'].values),
            'depth':    np.log(pc['Q'].values / pc['depth'].values),
            'Q_dollars': np.log(pc['Q'].values * pc['ref_mid'].values / pc['V_daily'].values),
        }

        for norm_name, x in normalizations.items():
            y_adj = y - np.log(np.maximum(pc['sigma'].values, 1e-10))
            beta_int, r2, n = compute_beta_intercept(x, y)
            beta_orig = compute_beta_origin(x, y_adj)
            x_ok = x[np.isfinite(x)]
            results.append(dict(
                model=model, normalization=norm_name,
                beta_intercept=beta_int, beta_origin=beta_orig, r2=r2, n=n,
                x_min=float(x_ok.min()) if len(x_ok) > 0 else np.nan,
                x_max=float(x_ok.max()) if len(x_ok) > 0 else np.nan,
                x_range=float(x_ok.max() - x_ok.min()) if len(x_ok) > 0 else 0,
            ))

        del md

    # Table
    lines.append(f'\n{"Normalization":<14} {"Model":<12} {"β_int":>7} {"β_orig":>7} {"R²":>7} {"N":>7} '
                 f'{"x_min":>7} {"x_max":>7} {"x_range":>8} {"Δ(0.5)":>7}')
    lines.append('-' * 100)
    for r in results:
        delta = abs(r['beta_intercept'] - 0.5)
        lines.append(f'{r["normalization"]:<14} {r["model"]:<12} {r["beta_intercept"]:7.4f} '
                     f'{r["beta_origin"]:7.4f} {r["r2"]:7.4f} {r["n"]:7d} '
                     f'{r["x_min"]:7.1f} {r["x_max"]:7.1f} {r["x_range"]:8.1f} {delta:7.4f}')

    # Best normalization
    if results:
        best = min(results, key=lambda r: abs(r['beta_intercept'] - 0.5))
        lines.append(f'\nClosest to β=0.5: {best["normalization"]} on {best["model"]} → '
                     f'β_int={best["beta_intercept"]:.4f} (Δ={abs(best["beta_intercept"]-0.5):.4f})')

    report = '\n'.join(lines)
    print(report)
    out_file = OUT_DIR / f'diag_4_normalizations_{stock}.txt'
    with open(out_file, 'w') as f:
        f.write(report)
    print(f'\nSaved: {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', default='GOOG')
    args = parser.parse_args()
    run(args.stock.upper())
