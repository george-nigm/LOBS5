#!/usr/bin/env python3
"""Diagnostic 5: β across different impact definitions.

Compares VWAP cumulative, midprice response, instantaneous, terminal, permanent.
Checks if any definition gives β ≈ 0.5.

Usage:
    python lob_impact/diag_5_impact_definitions.py --stock GOOG
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
    if not hl_path.exists():
        return {}
    df = pd.read_csv(hl_path)
    if 'day' not in df.columns:
        df['day'] = df['filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    params = {}
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
    return np.atleast_1d(np.loadtxt(f, dtype=int))


def get_midprice(book, idx):
    return (float(book[idx, 0]) + float(book[idx, 2])) / 2.0


def extract_impacts(md, daily_params):
    """Extract multiple impact definitions per aggressive order."""
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

                ref_mid = get_midprice(book, aggr_idx[0])
                if ref_mid <= 0:
                    continue
                sizes = msg[aggr_idx, 3].astype(float)
                prices = msg[aggr_idx, 4].astype(float)
                if np.any(sizes <= 0) or np.any(prices <= 0):
                    continue

                # Cumulative VWAP
                Q_cum = np.cumsum(sizes)
                vwap = np.cumsum(sizes * prices) / Q_cum
                if direction == 'buy':
                    I_vwap = np.abs((vwap - ref_mid) / ref_mid)
                else:
                    I_vwap = np.abs((ref_mid - vwap) / ref_mid)

                # Terminal midprice impact (whole sample)
                mid_final = get_midprice(book, -1)
                I_terminal = abs(mid_final - ref_mid) / ref_mid if mid_final > 0 else np.nan

                # Daily params
                day = days_list[j] if j < len(days_list) else None
                V_daily = 1e6
                sigma = 1.0
                if day and day in daily_params:
                    V_daily = daily_params[day]['V']
                    sigma = daily_params[day]['sigma']

                for k in range(n_aggr):
                    if Q_cum[k] <= 0:
                        continue
                    mid_before = get_midprice(book, aggr_idx[k])
                    if mid_before <= 0:
                        continue

                    # Per-insertion metrics
                    I_inst_k = abs(prices[k] - mid_before) / mid_before

                    # Mid response (mid after cooling → before next insertion)
                    if k + 1 < n_aggr:
                        mid_after = get_midprice(book, aggr_idx[k + 1])
                    else:
                        mid_after = get_midprice(book, len(book) - 1)
                    I_mid_k = abs(mid_after - mid_before) / mid_before if mid_after > 0 else np.nan

                    # Permanent: midprice change from this insertion to end of sample
                    I_perm_k = abs(get_midprice(book, -1) - mid_before) / mid_before

                    points.append(dict(
                        Q=float(Q_cum[k]), size_k=float(sizes[k]),
                        I_vwap=float(I_vwap[k]),
                        I_inst=float(I_inst_k),
                        I_mid=float(I_mid_k) if np.isfinite(I_mid_k) else np.nan,
                        I_perm=float(I_perm_k),
                        I_terminal=float(I_terminal) if np.isfinite(I_terminal) else np.nan,
                        V_daily=V_daily, sigma=sigma,
                        k=k + 1, vol=row['vol'], i=row['i'],
                    ))

    return pd.DataFrame(points)


def compute_beta(x, y):
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y > -30)
    xv, yv = x[ok], y[ok]
    if len(xv) < 10:
        return np.nan, np.nan, 0
    coeffs = np.polyfit(xv, yv, 1)
    beta = float(coeffs[0])
    yhat = beta * xv + coeffs[1]
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2, int(ok.sum())


def run(stock):
    OUT_DIR.mkdir(exist_ok=True)
    daily_params = load_daily_params(stock)
    lines = [f'=== Diagnostic 5: Impact Definitions ({stock}) ===\n']
    results = []

    for model in MODELS_TEST:
        pkl_path = PICKLE_BASE / stock / f'{model}.pkl'
        if not pkl_path.exists():
            lines.append(f'SKIP {model}')
            continue
        print(f'Loading {model}...')
        with open(pkl_path, 'rb') as f:
            md = pickle.load(f)

        pc = extract_impacts(md, daily_params)
        if pc.empty:
            continue
        print(f'  {model}: {len(pc)} points')

        x_cum = np.log(pc['Q'].values / pc['V_daily'].values)
        x_incr = np.log(pc['size_k'].values)

        # Filter for k >= 3 for incremental
        pc_k3 = pc[pc['k'] >= 3]
        x_incr_k3 = np.log(pc_k3['size_k'].values) if len(pc_k3) > 0 else np.array([])

        for impact_name in ['I_vwap', 'I_inst', 'I_mid', 'I_perm', 'I_terminal']:
            vals = pc[impact_name].values
            valid = vals > 1e-10
            if valid.sum() < 10:
                continue

            # Cumulative (all k)
            y = np.log(np.where(valid, vals, np.nan))
            beta_c, r2_c, n_c = compute_beta(x_cum, y)
            results.append(dict(model=model, impact=impact_name, scope='cumulative(all)',
                                beta=beta_c, r2=r2_c, n=n_c))

            # Incremental (k>=3)
            if len(pc_k3) > 0:
                vals_k3 = pc_k3[impact_name].values
                valid_k3 = vals_k3 > 1e-10
                if valid_k3.sum() >= 10:
                    y_k3 = np.log(np.where(valid_k3, vals_k3, np.nan))
                    beta_i, r2_i, n_i = compute_beta(x_incr_k3, y_k3)
                    results.append(dict(model=model, impact=impact_name, scope='incremental(k≥3)',
                                        beta=beta_i, r2=r2_i, n=n_i))

        del md

    # Table
    lines.append(f'{"Impact":<14} {"Scope":<20} {"Model":<12} {"β":>7} {"R²":>7} {"N":>7} {"Δ(0.5)":>7}')
    lines.append('-' * 85)
    for r in results:
        delta = abs(r['beta'] - 0.5)
        lines.append(f'{r["impact"]:<14} {r["scope"]:<20} {r["model"]:<12} '
                     f'{r["beta"]:7.4f} {r["r2"]:7.4f} {r["n"]:7d} {delta:7.4f}')

    # Best combination
    if results:
        best = min(results, key=lambda r: abs(r['beta'] - 0.5))
        lines.append(f'\nClosest to β=0.5: {best["impact"]} / {best["scope"]} / {best["model"]} '
                     f'→ β={best["beta"]:.4f}')

        # Group by impact type
        lines.append(f'\n--- Average β by impact type (cumulative, all models) ---')
        for imp in ['I_vwap', 'I_inst', 'I_mid', 'I_perm', 'I_terminal']:
            subset = [r for r in results if r['impact'] == imp and 'cumulative' in r['scope']]
            if subset:
                avg_b = np.mean([r['beta'] for r in subset])
                lines.append(f'  {imp:<14}: β_avg = {avg_b:.4f}')

    report = '\n'.join(lines)
    print(report)
    out_file = OUT_DIR / f'diag_5_impact_defs_{stock}.txt'
    with open(out_file, 'w') as f:
        f.write(report)
    print(f'\nSaved: {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', default='GOOG')
    args = parser.parse_args()
    run(args.stock.upper())
