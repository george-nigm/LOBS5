#!/usr/bin/env python3
"""
Fixed β=0.5 analysis with ALL k pooled (not just k=last).

For each model, loads the raw pickle, extracts the full point cloud
(all insertion indices k), and computes:
  - α_vwap  = mean(log(I_vwap) - 0.5·log(Q/V))   [VWAP impact, all k]
  - α_mid   = mean(log(I_mid)  - 0.5·log(Q/V))    [midprice impact, all k]
  - α_last  = mean(log(I_vwap) - 0.5·log(Q/V))    [VWAP impact, k=last only]

Also reports free-β OLS for comparison.

Usage:
    python -u lob_impact/fixbeta_allk.py \
        --pickle_base /path/to/pickles --stock GOOG \
        --daily_hl lob_impact/daily_h_l_GOOG.csv
"""
import argparse, pickle, sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, load_daily_params, collect_days,
    TICK_SIZE,
)


def fixed_beta_stats(pc, impact_col, beta=0.5, label=''):
    """Compute α = mean(log(I) - β·log(Q/V)) and OLS free-β."""
    if pc.empty or 'daily_vol' not in pc.columns:
        return dict(label=label, alpha=np.nan, alpha_se=np.nan,
                    exp_alpha=np.nan, r2_fixed=np.nan,
                    beta_free=np.nan, alpha_free=np.nan, r2_free=np.nan, n=0)

    I = pc[impact_col].values
    Q = pc['Q'].values
    V = pc['daily_vol'].values

    valid = (I > 1e-12) & (Q > 0) & (V > 0) & np.isfinite(I) & np.isfinite(Q) & np.isfinite(V)
    I, Q, V = I[valid], Q[valid], V[valid]
    n = len(I)
    if n < 10:
        return dict(label=label, alpha=np.nan, alpha_se=np.nan,
                    exp_alpha=np.nan, r2_fixed=np.nan,
                    beta_free=np.nan, alpha_free=np.nan, r2_free=np.nan, n=n)

    log_I = np.log(I)
    log_QV = np.log(Q / V)

    # Fixed β
    residuals = log_I - beta * log_QV
    alpha = float(np.mean(residuals))
    alpha_se = float(np.std(residuals) / np.sqrt(n))

    y_pred = alpha + beta * log_QV
    ss_res = np.sum((log_I - y_pred) ** 2)
    ss_tot = np.sum((log_I - np.mean(log_I)) ** 2)
    r2_fixed = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Free β (OLS with intercept)
    A = np.column_stack([log_QV, np.ones(n)])
    coeffs, residuals_ols, _, _ = np.linalg.lstsq(A, log_I, rcond=None)
    beta_free = float(coeffs[0])
    alpha_free = float(coeffs[1])
    y_pred_free = beta_free * log_QV + alpha_free
    ss_res_free = np.sum((log_I - y_pred_free) ** 2)
    r2_free = 1 - ss_res_free / ss_tot if ss_tot > 0 else 0.0

    # Bootstrap CI for α (fixed β)
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_alphas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_alphas[b] = np.mean(log_I[idx] - beta * log_QV[idx])
    ci_lo, ci_hi = np.percentile(boot_alphas, [2.5, 97.5])

    return dict(
        label=label, alpha=alpha, alpha_se=alpha_se,
        ci_lo=ci_lo, ci_hi=ci_hi,
        exp_alpha=np.exp(alpha), r2_fixed=r2_fixed,
        beta_free=beta_free, alpha_free=alpha_free, r2_free=r2_free, n=n,
    )


def main():
    parser = argparse.ArgumentParser(description='Fixed β=0.5 analysis, all k pooled')
    parser.add_argument('--pickle_base', required=True, help='Directory containing {stock}/*.pkl')
    parser.add_argument('--stock', required=True)
    parser.add_argument('--daily_hl', required=True, help='Path to daily_h_l_{stock}.csv')
    parser.add_argument('--beta', type=float, default=0.5, help='Fixed β value')
    parser.add_argument('--models', nargs='+', default=None, help='Subset of models (default: all)')
    args = parser.parse_args()

    pkl_dir = Path(args.pickle_base) / args.stock
    if not pkl_dir.exists():
        print(f'ERROR: pickle dir not found: {pkl_dir}')
        sys.exit(1)

    if args.models:
        models = args.models
    else:
        models = sorted([f.stem for f in pkl_dir.glob('*.pkl')
                         if f.stem != 'ZeroInsertions'])
    print(f'Stock: {args.stock}')
    print(f'Pickle dir: {pkl_dir}')
    print(f'Models: {models}')
    print(f'Fixed β: {args.beta}')
    print()

    all_rows = []

    for model in models:
        pkl_path = pkl_dir / f'{model}.pkl'
        if not pkl_path.exists():
            print(f'  SKIP {model}: not found')
            continue

        t0 = time.time()
        size_gb = pkl_path.stat().st_size / 1e9
        print(f'{"=" * 65}')
        print(f'  {model} / {args.stock}  ({size_gb:.1f} GB)')
        print(f'{"=" * 65}')

        with open(pkl_path, 'rb') as f:
            md = pickle.load(f)
        t_load = time.time() - t0
        print(f'  Loaded in {t_load:.0f}s')

        # Daily params
        exp_days = collect_days(md)
        daily_params = load_daily_params(args.daily_hl, exp_days)
        print(f'  Daily params: {len(daily_params)} days')

        # Filter
        filtered, n_total, n_skip = filter_model(md)
        print(f'  Filtered: {n_skip}/{n_total} outliers removed')

        # Extract full point cloud (ALL k)
        pc = extract_point_cloud(filtered, md['grid'], daily_params)
        print(f'  Total points (all k): {len(pc)}')

        if pc.empty:
            print(f'  WARNING: empty point cloud, skipping')
            continue

        k_max = int(pc['k'].max())
        pc_last = pc[pc['k'] == k_max].copy()
        print(f'  Points with k=last (k={k_max}): {len(pc_last)}')

        # --- Analysis 1: VWAP impact, ALL k ---
        r_vwap_allk = fixed_beta_stats(pc, 'I', args.beta, label=f'{model} VWAP all-k')
        print(f'  [VWAP all-k]  α={r_vwap_allk["alpha"]:.4f}  CI=[{r_vwap_allk.get("ci_lo",0):.4f},{r_vwap_allk.get("ci_hi",0):.4f}]  '
              f'β_free={r_vwap_allk["beta_free"]:.4f}  R²={r_vwap_allk["r2_fixed"]:.4f}  n={r_vwap_allk["n"]}')

        # --- Analysis 2: Midprice impact, ALL k ---
        has_mid = 'I_mid' in pc.columns and pc['I_mid'].notna().sum() > 10
        if has_mid:
            pc_mid = pc[pc['I_mid'].notna() & (pc['I_mid'] > 1e-12)].copy()
            r_mid_allk = fixed_beta_stats(pc_mid, 'I_mid', args.beta, label=f'{model} Mid all-k')
            print(f'  [Mid  all-k]  α={r_mid_allk["alpha"]:.4f}  CI=[{r_mid_allk.get("ci_lo",0):.4f},{r_mid_allk.get("ci_hi",0):.4f}]  '
                  f'β_free={r_mid_allk["beta_free"]:.4f}  R²={r_mid_allk["r2_fixed"]:.4f}  n={r_mid_allk["n"]}')
        else:
            r_mid_allk = dict(label=f'{model} Mid all-k', alpha=np.nan, alpha_se=np.nan,
                              exp_alpha=np.nan, r2_fixed=np.nan, beta_free=np.nan,
                              alpha_free=np.nan, r2_free=np.nan, n=0, ci_lo=np.nan, ci_hi=np.nan)
            print(f'  [Mid  all-k]  no valid I_mid data')

        # --- Analysis 3: VWAP impact, k=last only ---
        r_vwap_last = fixed_beta_stats(pc_last, 'I', args.beta, label=f'{model} VWAP k=last')
        print(f'  [VWAP k=last] α={r_vwap_last["alpha"]:.4f}  CI=[{r_vwap_last.get("ci_lo",0):.4f},{r_vwap_last.get("ci_hi",0):.4f}]  '
              f'β_free={r_vwap_last["beta_free"]:.4f}  R²={r_vwap_last["r2_fixed"]:.4f}  n={r_vwap_last["n"]}')

        # --- Analysis 4: Midprice impact, k=last only ---
        if has_mid:
            pc_mid_last = pc_last[pc_last['I_mid'].notna() & (pc_last['I_mid'] > 1e-12)].copy()
            r_mid_last = fixed_beta_stats(pc_mid_last, 'I_mid', args.beta, label=f'{model} Mid k=last')
            print(f'  [Mid  k=last] α={r_mid_last["alpha"]:.4f}  CI=[{r_mid_last.get("ci_lo",0):.4f},{r_mid_last.get("ci_hi",0):.4f}]  '
                  f'β_free={r_mid_last["beta_free"]:.4f}  R²={r_mid_last["r2_fixed"]:.4f}  n={r_mid_last["n"]}')
        else:
            r_mid_last = dict(label=f'{model} Mid k=last', alpha=np.nan, alpha_se=np.nan,
                              exp_alpha=np.nan, r2_fixed=np.nan, beta_free=np.nan,
                              alpha_free=np.nan, r2_free=np.nan, n=0, ci_lo=np.nan, ci_hi=np.nan)

        all_rows.append(dict(
            model=model,
            # VWAP all-k
            alpha_vwap_allk=r_vwap_allk['alpha'],
            ci_lo_vwap_allk=r_vwap_allk.get('ci_lo', np.nan),
            ci_hi_vwap_allk=r_vwap_allk.get('ci_hi', np.nan),
            beta_free_vwap_allk=r_vwap_allk['beta_free'],
            r2_vwap_allk=r_vwap_allk['r2_fixed'],
            n_vwap_allk=r_vwap_allk['n'],
            # Mid all-k
            alpha_mid_allk=r_mid_allk['alpha'],
            ci_lo_mid_allk=r_mid_allk.get('ci_lo', np.nan),
            ci_hi_mid_allk=r_mid_allk.get('ci_hi', np.nan),
            beta_free_mid_allk=r_mid_allk['beta_free'],
            r2_mid_allk=r_mid_allk['r2_fixed'],
            n_mid_allk=r_mid_allk['n'],
            # VWAP k=last
            alpha_vwap_last=r_vwap_last['alpha'],
            ci_lo_vwap_last=r_vwap_last.get('ci_lo', np.nan),
            ci_hi_vwap_last=r_vwap_last.get('ci_hi', np.nan),
            beta_free_vwap_last=r_vwap_last['beta_free'],
            r2_vwap_last=r_vwap_last['r2_fixed'],
            n_vwap_last=r_vwap_last['n'],
            # Mid k=last
            alpha_mid_last=r_mid_last['alpha'],
            ci_lo_mid_last=r_mid_last.get('ci_lo', np.nan),
            ci_hi_mid_last=r_mid_last.get('ci_hi', np.nan),
            beta_free_mid_last=r_mid_last['beta_free'],
            r2_mid_last=r_mid_last['r2_fixed'],
            n_mid_last=r_mid_last['n'],
        ))

        del md, filtered, pc, pc_last
        print(f'  Done in {time.time() - t0:.0f}s total\n')

    # ═══════════════════════════════════════════════════════════════════
    # Summary tables
    # ═══════════════════════════════════════════════════════════════════
    if not all_rows:
        print('No results.')
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    def print_table(title, alpha_col, ci_lo_col, ci_hi_col, beta_col, r2_col, n_col):
        print(f'\n{"=" * 85}')
        print(f'  {title}  |  β fixed = {args.beta}  |  {args.stock}')
        print(f'{"=" * 85}')
        print(f'{"Model":>12s}  {"α":>8s}  {"95% CI":>20s}  {"exp(α)":>10s}  {"β_free":>8s}  {"R²":>8s}  {"N":>8s}')
        print(f'{"-" * 85}')
        for _, row in df.iterrows():
            a = row[alpha_col]
            lo = row[ci_lo_col]
            hi = row[ci_hi_col]
            bf = row[beta_col]
            r2 = row[r2_col]
            nn = int(row[n_col])
            if np.isnan(a):
                print(f'{row["model"]:>12s}  {"N/A":>8s}  {"N/A":>20s}  {"N/A":>10s}  {"N/A":>8s}  {"N/A":>8s}  {nn:>8d}')
            else:
                ci_str = f'[{lo:.4f}, {hi:.4f}]'
                print(f'{row["model"]:>12s}  {a:>8.4f}  {ci_str:>20s}  {np.exp(a):>10.6f}  {bf:>8.4f}  {r2:>8.4f}  {nn:>8d}')

    print_table('VWAP Impact — ALL k pooled',
                'alpha_vwap_allk', 'ci_lo_vwap_allk', 'ci_hi_vwap_allk',
                'beta_free_vwap_allk', 'r2_vwap_allk', 'n_vwap_allk')

    print_table('Midprice Impact — ALL k pooled',
                'alpha_mid_allk', 'ci_lo_mid_allk', 'ci_hi_mid_allk',
                'beta_free_mid_allk', 'r2_mid_allk', 'n_mid_allk')

    print_table('VWAP Impact — k=last only',
                'alpha_vwap_last', 'ci_lo_vwap_last', 'ci_hi_vwap_last',
                'beta_free_vwap_last', 'r2_vwap_last', 'n_vwap_last')

    print_table('Midprice Impact — k=last only',
                'alpha_mid_last', 'ci_lo_mid_last', 'ci_hi_mid_last',
                'beta_free_mid_last', 'r2_mid_last', 'n_mid_last')

    print(f'\nDone. Processed {len(df)} models for {args.stock}.')


if __name__ == '__main__':
    main()
