#!/usr/bin/env python3
"""
β(φ) analysis: compute β at each participation rate level separately.

Uses existing V5r2 pickles (AAPL, AMZN) and V4 pickles (GOOG, INTC).
Filters point cloud by child size (= volume level) to get β per φ.

Also extracts: relaxation, Kyle λ, master curve shape per φ.

Usage:
    python lob_impact/analyze_beta_per_phi.py
"""
import pickle, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, compute_global_beta,
    bootstrap_beta, load_daily_params, collect_days,
    compute_kyle_lambda, compute_master_curve, compute_relaxation_ratio,
    TICK_SIZE, N_BOOTSTRAP,
)


def analyze_one_model(pkl_path, daily_hl_path, model_name, stock):
    """Load pickle, extract point cloud, compute β per volume level."""
    if not Path(pkl_path).exists():
        return None

    print(f'  Loading {model_name} ({Path(pkl_path).stat().st_size/1e9:.1f} GB)...')
    with open(pkl_path, 'rb') as f:
        md = pickle.load(f)

    exp_days = collect_days(md)
    daily_params = load_daily_params(daily_hl_path, exp_days)
    filtered, n_total, n_skip = filter_model(md)
    grid_df = md['grid']

    pc = extract_point_cloud(filtered, grid_df, daily_params)
    if pc.empty:
        return None

    # Get unique volume levels (= child sizes)
    vol_levels = sorted(pc['vol'].unique()) if 'vol' in pc.columns else []
    if not vol_levels:
        return None

    results = []
    for vol in vol_levels:
        pc_vol = pc[pc['vol'] == vol].copy()
        if len(pc_vol) < 50:
            continue

        # β (intercept, VWAP)
        res = compute_global_beta(pc_vol)
        boot = bootstrap_beta(pc_vol, n_boot=1000)
        boots = boot.get('boots', np.array([]))
        ci = (np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)) \
            if len(boots) > 10 else (np.nan, np.nan)

        # β (midprice)
        pc_mid = pc_vol[pc_vol['I_mid'].notna() & (pc_vol['I_mid'] > 1e-12)].copy()
        res_mid = compute_global_beta(pc_mid, impact_col='I_mid') if len(pc_mid) > 20 else {}

        # Kyle λ
        kyle = compute_kyle_lambda(pc_vol)
        k_max = int(pc_vol['k'].max())
        lam_k1 = kyle.get(1, {}).get('mean', np.nan)
        lam_kmax = kyle.get(k_max, {}).get('mean', np.nan)

        # Compute actual φ
        # φ = child / (child + mb * r * q_med)
        # We can estimate from the data: mb is in grid, child = vol
        mb_vals = pc_vol['mb'].unique()
        mb = float(mb_vals[0]) if len(mb_vals) == 1 else float(np.median(mb_vals))

        # N per direction
        n_buy = len(pc_vol[pc_vol['direction'] == 'buy'])
        n_sell = len(pc_vol[pc_vol['direction'] == 'sell'])

        results.append(dict(
            model=model_name,
            stock=stock,
            vol=int(vol),
            Q_at_kmax=int(vol) * k_max if 'vol' in pc_vol.columns else np.nan,
            mb=mb,
            beta_int=res.get('beta', np.nan),
            beta_origin=res.get('beta_origin', np.nan),
            ci_lo=ci[0],
            ci_hi=ci[1],
            r2=res.get('r2', np.nan),
            n=res.get('n', 0),
            beta_mid=res_mid.get('beta', np.nan) if res_mid else np.nan,
            lam_k1=lam_k1,
            lam_kmax=lam_kmax,
            lam_trend='↓' if lam_kmax < lam_k1 else '↑',
            n_buy=n_buy,
            n_sell=n_sell,
        ))

    return results


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path('pics_for_beta_phi')
    out_dir.mkdir(exist_ok=True)

    V4_PKL = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v3/pickles')
    V5_PKL = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v5/pickles')

    COLORS = {
        'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
        'CGAN': '#7B4F9E', 'LobS5': '#C88A3A', 'S5-120M': '#D95F02',
        'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E', 'LobS5-v2': '#2CA02C',
    }

    # Known φ values for each (stock, vol) combination
    # V4 GOOG: vol=105/165/325, depth=166, mb=5-20 → φ≈95-99%
    # V4 INTC: vol=590/1110/3120, depth=1248, mb=5-20 → φ≈95-99%
    # V5r2 AAPL: vol=10/40/100, depth=300, mb=36, mkt=39.3 → φ≈20%/50%/72%
    # V5r2 AMZN: vol=7/25/100, depth=224, mb=36, mkt=19.5 → φ≈26%/56%/84%
    PHI_MAP = {
        ('GOOG', 105): 97, ('GOOG', 165): 98, ('GOOG', 325): 99,
        ('INTC', 590): 96, ('INTC', 1110): 98, ('INTC', 3120): 99,
        ('AAPL', 10): 20, ('AAPL', 40): 50, ('AAPL', 100): 72,
        ('AMZN', 7): 26, ('AMZN', 25): 56, ('AMZN', 100): 84,
        # V5r1 small child
        ('AAPL', 2): 5, ('AAPL', 4): 9, ('AAPL', 9): 19,
        ('AMZN', 1): 5, ('AMZN', 2): 9, ('AMZN', 4): 17,
    }

    all_results = []

    # V4 GOOG + INTC
    for stock in ['GOOG', 'INTC']:
        pkl_base = V4_PKL / stock
        daily_hl = f'lob_impact/daily_h_l_{stock}.csv'
        models = [f.stem for f in sorted(pkl_base.glob('*.pkl')) if f.stem != 'ZeroInsertions']
        print(f'\n{"="*60}\n  {stock} (V4)\n{"="*60}')
        for model in models:
            res = analyze_one_model(str(pkl_base / f'{model}.pkl'), daily_hl, model, stock)
            if res:
                all_results.extend(res)

    # V5r2 AAPL + AMZN
    for stock in ['AAPL', 'AMZN']:
        pkl_base = V5_PKL / stock
        daily_hl = f'lob_impact/daily_h_l_{stock}.csv'
        models = [f.stem for f in sorted(pkl_base.glob('*.pkl')) if f.stem != 'ZeroInsertions']
        print(f'\n{"="*60}\n  {stock} (V5r2)\n{"="*60}')
        for model in models:
            res = analyze_one_model(str(pkl_base / f'{model}.pkl'), daily_hl, model, stock)
            if res:
                all_results.extend(res)

    df = pd.DataFrame(all_results)
    df['phi'] = df.apply(lambda r: PHI_MAP.get((r['stock'], r['vol']), None), axis=1)

    # Save CSV
    csv_path = out_dir / 'beta_per_phi.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path} ({len(df)} rows)')

    # Print summary tables
    print(f'\n{"="*80}')
    print(f'  β(φ) SUMMARY — ALL STOCKS, ALL MODELS')
    print(f'{"="*80}')
    print(f'{"Model":>12s}  {"Stock":>5s}  {"vol":>6s}  {"φ%":>4s}  {"β_int":>7s}  {"CI":>16s}  {"R²":>7s}  {"β_mid":>7s}  {"λ1":>6s}  {"λk":>6s}  {"λ":>2s}  {"N":>7s}')
    for _, r in df.sort_values(['model', 'stock', 'phi']).iterrows():
        phi_s = f'{r["phi"]:.0f}' if pd.notna(r['phi']) else '?'
        ci_s = f'[{r["ci_lo"]:.3f},{r["ci_hi"]:.3f}]' if pd.notna(r['ci_lo']) else ''
        print(f'{r["model"]:>12s}  {r["stock"]:>5s}  {r["vol"]:>6.0f}  {phi_s:>4s}  {r["beta_int"]:>7.3f}  {ci_s:>16s}  {r["r2"]:>7.4f}  {r.get("beta_mid",np.nan):>7.3f}  {r["lam_k1"]:>6.3f}  {r["lam_kmax"]:>6.3f}  {r["lam_trend"]:>2s}  {r["n"]:>7.0f}')

    # β(φ) per model (S5-4K focus)
    print(f'\n{"="*60}')
    print(f'  β(φ) CURVE — S5-4K across all stocks')
    print(f'{"="*60}')
    s5_4k = df[df['model'] == 'S5-4K'].sort_values('phi')
    if not s5_4k.empty:
        print(f'{"Stock":>5s}  {"φ%":>4s}  {"β_int":>7s}  {"R²":>7s}  {"λ trend":>8s}')
        for _, r in s5_4k.iterrows():
            phi_s = f'{r["phi"]:.0f}' if pd.notna(r['phi']) else '?'
            print(f'{r["stock"]:>5s}  {phi_s:>4s}  {r["beta_int"]:>7.3f}  {r["r2"]:>7.4f}  {r["lam_k1"]:.3f}→{r["lam_kmax"]:.3f} {r["lam_trend"]}')


if __name__ == '__main__':
    main()
