#!/usr/bin/env python3
"""
Volatility sweep: try multiple σ estimators and see how β changes.

For each (model, stock, η) × (σ method) × (k filter):
  β_origin = dot(x, y_adj) / dot(x, x)
  where x = log(Q/V), y_adj = log(I/σ)

σ methods:
  1. parkinson_2x  — current code: ln(H/L) / √(ln2)  [WRONG: 2× overestimate]
  2. parkinson     — correct: ln(H/L) / (2√(ln2))
  3. log_range     — raw: ln(H/L)
  4. no_sigma      — σ=1 (no normalization)
  5. constant      — σ = median(parkinson) across all days
  6. sqrt_range    — √(ln(H/L)) [variance proxy]
  7. range_over_V  — ln(H/L) / √(V/V_med) [volume-adjusted]

k filters:
  A. k=10 only (full metaorder)
  B. all k=1..10

Usage:
    python lob_impact/analyze_volatility_sweep.py \
        --stock AAPL --pickle_base /path/to/pickles \
        --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v5 --out_dir pics_for_vol_sweep
"""
import argparse, pickle, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, load_daily_params, collect_days,
)

MODELS_ORDER = ['Historic', 'Heuristic', 'CST', 'S5-120M', 'S5-360M', 'S5-4K']

ETA_MAP = {
    ('AAPL', 20): 5, ('AAPL', 40): 9, ('AAPL', 90): 19,
    ('AAPL', 100): 22, ('AAPL', 400): 50, ('AAPL', 1000): 72,
    ('AMZN', 10): 5, ('AMZN', 20): 9, ('AMZN', 40): 17,
    ('AMZN', 70): 26, ('AMZN', 250): 56, ('AMZN', 1000): 84,
    ('AAPL', 3570): 90, ('AAPL', 390): 50,
    ('AMZN', 190): 50, ('AMZN', 1770): 90,
}


def compute_beta_origin(x, y_adj):
    """β = dot(x, y_adj) / dot(x, x), with R²."""
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    xv, yv = x[ok], y_adj[ok]
    if len(xv) < 2:
        return np.nan, np.nan, 0
    beta = float(np.dot(xv, yv) / np.dot(xv, xv))
    ss_res = np.sum((yv - beta * xv) ** 2)
    ss_tot = np.sum(yv ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2, int(ok.sum())


def compute_sigma_methods(daily_sigma, sigma_rv=None):
    """Given daily σ (now corrected Parkinson), compute all σ variants."""
    # After Parkinson fix: daily_sigma = ln(H/L) / 1.6651 (correct)
    # Recover ln(H/L) = daily_sigma × 1.6651
    ln_HL = daily_sigma * 1.6651092

    methods = {}
    methods['parkinson'] = daily_sigma                       # correct Parkinson
    methods['parkinson_2x'] = daily_sigma * 2.0              # old code (wrong, for comparison)
    methods['log_range'] = ln_HL                             # raw ln(H/L)
    methods['no_sigma'] = np.ones_like(daily_sigma)          # σ=1
    median_sigma = np.median(daily_sigma[daily_sigma > 0])
    methods['constant'] = np.full_like(daily_sigma, median_sigma)
    methods['sqrt_range'] = np.sqrt(np.maximum(ln_HL, 1e-12))
    if sigma_rv is not None:
        methods['realized_vol'] = sigma_rv                   # per-sample midprice RV

    return methods


def analyze_one_model(pkl_path, daily_hl_path, model_name, stock):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f'  SKIP {model_name}')
        return []

    print(f'\n  Loading {model_name} ({pkl_path.stat().st_size / 1e9:.1f} GB)...')
    with open(pkl_path, 'rb') as f:
        md = pickle.load(f)

    exp_days = collect_days(md)
    daily_params = load_daily_params(daily_hl_path, exp_days)
    filtered, n_total, n_skip = filter_model(md)
    grid_df = md['grid']
    pc = extract_point_cloud(filtered, grid_df, daily_params)
    del md, filtered
    gc.collect()

    if pc.empty or 'daily_sigma' not in pc.columns:
        return []

    k_max = int(pc['k'].max())
    vol_levels = sorted(pc['vol'].unique())
    results = []

    # Precompute x (same for all methods)
    x_all = np.log(pc['Q'].values / pc['daily_vol'].values)
    I_all = pc['I'].values
    sigma_all = pc['daily_sigma'].values

    # Compute all σ methods
    sigma_rv_all = pc['sigma_rv'].values if 'sigma_rv' in pc.columns else None
    sigma_methods = compute_sigma_methods(sigma_all, sigma_rv_all)

    for vol in vol_levels:
        mask_vol = (pc['vol'] == vol).values
        eta = ETA_MAP.get((stock, int(vol)), None)
        if 'child' in grid_df.columns:
            child_vals = grid_df[grid_df['vol'] == vol]['child'].unique()
            child = int(child_vals[0]) if len(child_vals) > 0 else int(vol)
        else:
            child = int(vol)

        for k_filter_name, k_mask in [('k10', (pc['k'] == k_max).values),
                                       ('allk', np.ones(len(pc), dtype=bool))]:
            mask = mask_vol & k_mask
            if mask.sum() < 30:
                continue

            x_sub = x_all[mask]
            I_sub = I_all[mask]

            for method_name, sigma_arr in sigma_methods.items():
                sigma_sub = sigma_arr[mask]
                y_adj = np.log(np.maximum(I_sub, 1e-20) / np.maximum(sigma_sub, 1e-20))
                beta_orig, r2_orig, n = compute_beta_origin(x_sub, y_adj)
                # Intercept β
                ok = np.isfinite(x_sub) & np.isfinite(y_adj) & (x_sub != 0)
                if ok.sum() > 2:
                    coeffs = np.polyfit(x_sub[ok], y_adj[ok], 1)
                    beta_slope = float(coeffs[0])
                else:
                    beta_slope = np.nan

                results.append(dict(
                    model=model_name, stock=stock, child=child,
                    vol=int(vol), eta=eta, k_filter=k_filter_name,
                    sigma_method=method_name,
                    beta_origin=beta_orig, beta_slope=beta_slope,
                    r2=r2_orig, n=n,
                ))

    # Cross-η (pool all child sizes)
    for k_filter_name, k_mask in [('k10', (pc['k'] == k_max).values),
                                   ('allk', np.ones(len(pc), dtype=bool))]:
        mask = k_mask
        if mask.sum() < 50:
            continue
        x_sub = x_all[mask]
        I_sub = I_all[mask]

        for method_name, sigma_arr in sigma_methods.items():
            sigma_sub = sigma_arr[mask]
            y_adj = np.log(np.maximum(I_sub, 1e-20) / np.maximum(sigma_sub, 1e-20))
            beta_orig, r2_orig, n = compute_beta_origin(x_sub, y_adj)
            ok = np.isfinite(x_sub) & np.isfinite(y_adj) & (x_sub != 0)
            if ok.sum() > 2:
                coeffs = np.polyfit(x_sub[ok], y_adj[ok], 1)
                beta_slope = float(coeffs[0])
            else:
                beta_slope = np.nan

            results.append(dict(
                model=model_name, stock=stock, child=-1,
                vol=-1, eta=-1, k_filter=k_filter_name,
                sigma_method=method_name,
                beta_origin=beta_orig, beta_slope=beta_slope,
                r2=r2_orig, n=n,
            ))

    return results


def main():
    parser = argparse.ArgumentParser(description='Volatility sweep for β')
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v5')
    parser.add_argument('--out_dir', default='pics_for_vol_sweep')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'{"=" * 70}')
    print(f'  Volatility Sweep — {args.stock} ({args.version})')
    print(f'  Models: {models}')
    print(f'{"=" * 70}')

    all_results = []
    for model in models:
        results = analyze_one_model(
            str(pkl_dir / f'{model}.pkl'), args.daily_hl, model, args.stock)
        all_results.extend(results)
        gc.collect()

    if not all_results:
        print('ERROR: no results'); sys.exit(1)

    df = pd.DataFrame(all_results)
    csv_path = out_dir / f'vol_sweep_{args.stock}_{args.version}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path} ({len(df)} rows)')

    # ── Print pivot tables ──
    sigma_methods = ['parkinson_2x', 'parkinson', 'log_range', 'no_sigma', 'constant', 'sqrt_range']

    for k_filter in ['k10', 'allk']:
        print(f'\n{"=" * 120}')
        print(f'  k_filter={k_filter} — {args.stock} ({args.version})')
        print(f'  Formula: log(I/σ) = β × log(Q/V), origin estimator')
        print(f'{"=" * 120}')

        header = f'{"Model":>12s} {"child":>5s} {"η%":>4s}'
        for m in sigma_methods:
            header += f'  {m:>13s}'
        print(header)
        print('-' * 120)

        df_k = df[df['k_filter'] == k_filter]

        for model in models:
            df_m = df_k[df_k['model'] == model]
            # Per-η rows
            etas = sorted([e for e in df_m['eta'].unique() if e > 0])
            for eta_val in etas:
                df_e = df_m[df_m['eta'] == eta_val]
                child = int(df_e['child'].iloc[0]) if not df_e.empty else 0
                row = f'{model:>12s} {child:>5d} {eta_val:>4.0f}'
                for sm in sigma_methods:
                    val = df_e[df_e['sigma_method'] == sm]['beta_origin']
                    if not val.empty:
                        row += f'  {float(val.iloc[0]):>+13.4f}'
                    else:
                        row += f'  {"—":>13s}'
                print(row)

            # ALL row
            df_all = df_m[df_m['eta'] == -1]
            if not df_all.empty:
                row = f'{model:>12s} {"ALL":>5s} {"ALL":>4s}'
                for sm in sigma_methods:
                    val = df_all[df_all['sigma_method'] == sm]['beta_origin']
                    if not val.empty:
                        row += f'  {float(val.iloc[0]):>+13.4f}'
                    else:
                        row += f'  {"—":>13s}'
                print(row)
            print()

    # ── PDF ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    COLORS = {
        'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
        'S5-120M': '#D95F02', 'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E',
    }

    pdf_path = out_dir / f'vol_sweep_{args.stock}_{args.version}.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        for k_filter in ['k10', 'allk']:
            df_k = df[df['k_filter'] == k_filter]
            df_eta = df_k[df_k['eta'] > 0]

            # β vs η for each σ method
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'{args.stock} ({args.version}) — k={k_filter}: β(η) by σ method',
                         fontsize=14, fontweight='bold')

            for ax, sm in zip(axes.flat, sigma_methods):
                for model in models:
                    mdf = df_eta[(df_eta['model'] == model) & (df_eta['sigma_method'] == sm)]
                    mdf = mdf.sort_values('eta')
                    if mdf.empty:
                        continue
                    c = COLORS.get(model, 'gray')
                    ax.plot(mdf['eta'], mdf['beta_origin'], 'o-', color=c, lw=2, markersize=5, label=model)
                ax.axhline(0.5, ls='--', color='red', lw=1, alpha=0.7)
                ax.axhline(0, ls='-', color='gray', lw=0.5)
                ax.set_title(f'σ = {sm}')
                ax.set_xlabel('η (%)')
                ax.set_ylabel('β')
                ax.legend(fontsize=6)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig); plt.close(fig)

            # Cross-η bar chart comparing methods
            df_cross = df_k[df_k['eta'] == -1]
            if not df_cross.empty:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f'{args.stock} — k={k_filter}: cross-η β by σ method', fontsize=14)

                # Left: grouped bar chart
                ax = axes[0]
                x_pos = np.arange(len(models))
                width = 0.12
                for i, sm in enumerate(sigma_methods):
                    vals = []
                    for model in models:
                        v = df_cross[(df_cross['model'] == model) & (df_cross['sigma_method'] == sm)]['beta_origin']
                        vals.append(float(v.iloc[0]) if not v.empty else 0)
                    ax.bar(x_pos + i * width, vals, width, label=sm, alpha=0.7)
                ax.set_xticks(x_pos + width * len(sigma_methods) / 2)
                ax.set_xticklabels(models, rotation=30, fontsize=8)
                ax.axhline(0.5, ls='--', color='red', lw=1.5)
                ax.axhline(0, ls='-', color='gray', lw=0.5)
                ax.set_ylabel('β')
                ax.legend(fontsize=7)
                ax.set_title('β by model × σ method')

                # Right: Historic β for each method
                ax = axes[1]
                hist_vals = []
                for sm in sigma_methods:
                    v = df_cross[(df_cross['model'] == 'Historic') & (df_cross['sigma_method'] == sm)]['beta_origin']
                    hist_vals.append(float(v.iloc[0]) if not v.empty else 0)
                ax.barh(sigma_methods, hist_vals, color='#90939C', alpha=0.7)
                ax.axvline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
                ax.axvline(0, ls='-', color='gray', lw=0.5)
                ax.set_xlabel('β (Historic only)')
                ax.set_title('Historic β — which σ gives β ≈ 0?')
                ax.legend()

                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

    print(f'Saved: {pdf_path}')


if __name__ == '__main__':
    main()
