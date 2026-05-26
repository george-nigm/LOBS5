#!/usr/bin/env python3
"""
Per-experiment β with full breakdown by η and k-filter.

For each (model, stock, η): 4 β values:
  β_k10_orig   — k=10 only, origin estimator (ratio log(I/σ)/log(Q/V))
  β_k10_slope  — k=10 only, intercept estimator (slope, noisy when Q=const)
  β_allk_orig  — all k=1..10, origin estimator
  β_allk_slope — all k=1..10, intercept estimator (slope, Q varies child..10×child)

Plus cross-η at k=10 (pool all child sizes → Q varies → good slope estimate).

Usage:
    python lob_impact/analyze_per_experiment_beta.py \
        --stock AAPL --pickle_base /path/to/pickles \
        --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v5 --out_dir pics_for_per_exp_beta
"""
import argparse, pickle, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, compute_global_beta,
    bootstrap_beta, load_daily_params, collect_days,
)

ETA_MAP = {
    ('AAPL', 20): 5, ('AAPL', 40): 9, ('AAPL', 90): 19,
    ('AAPL', 100): 22, ('AAPL', 400): 50, ('AAPL', 1000): 72,
    ('AMZN', 10): 5, ('AMZN', 20): 9, ('AMZN', 40): 17,
    ('AMZN', 70): 26, ('AMZN', 250): 56, ('AMZN', 1000): 84,
    ('AAPL', 3570): 90, ('AAPL', 390): 50,
    ('AMZN', 190): 50, ('AMZN', 1770): 90,
}

MODELS_ORDER = ['Historic', 'Heuristic', 'CST', 'S5-120M', 'S5-360M', 'S5-4K']
COLORS = {
    'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
    'S5-120M': '#D95F02', 'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E',
}


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

    if pc.empty:
        return []

    k_max = int(pc['k'].max())
    vol_levels = sorted(pc['vol'].unique())
    results = []

    # ── Per-η rows ──
    for vol in vol_levels:
        pc_vol = pc[pc['vol'] == vol].copy()
        eta = ETA_MAP.get((stock, int(vol)), None)

        if 'child' in grid_df.columns:
            child_vals = grid_df[grid_df['vol'] == vol]['child'].unique()
            child = int(child_vals[0]) if len(child_vals) > 0 else int(vol)
        else:
            child = int(vol)

        if len(pc_vol) < 30:
            continue

        pc_k10 = pc_vol[pc_vol['k'] == k_max]

        # All k
        res_allk = compute_global_beta(pc_vol)
        # k=10 only
        res_k10 = compute_global_beta(pc_k10) if len(pc_k10) > 20 else {}
        # Bootstrap (all k, origin)
        boot = bootstrap_beta(pc_vol, n_boot=1000)
        boots_orig = boot.get('boots_origin', np.array([]))
        ci = (np.nanpercentile(boots_orig, 2.5), np.nanpercentile(boots_orig, 97.5)) \
            if len(boots_orig) > 10 else (np.nan, np.nan)

        n_days = pc_vol['day'].nunique() if 'day' in pc_vol.columns else 0

        results.append(dict(
            model=model_name, stock=stock, child=child, vol=int(vol),
            eta=eta, k_max=k_max, n_days=n_days, row_type='per_eta',
            Q_min=int(pc_vol['Q'].min()), Q_max=int(pc_vol['Q'].max()),
            n_allk=res_allk.get('n', 0), n_k10=res_k10.get('n', 0),
            # 4 β values
            beta_k10_orig=res_k10.get('beta_origin', np.nan),
            beta_k10_slope=res_k10.get('beta_intercept', np.nan),
            beta_allk_orig=res_allk.get('beta_origin', np.nan),
            beta_allk_slope=res_allk.get('beta_intercept', np.nan),
            # CI (all k, origin)
            ci_lo=ci[0], ci_hi=ci[1],
            # R²
            r2_allk_orig=res_allk.get('r2_origin', np.nan),
            r2_allk_slope=res_allk.get('r2_intercept', np.nan),
            # Impact stats
            I_median=float(np.median(pc_vol['I'])),
        ))

        r = results[-1]
        print(f'    child={child:>4d} η={eta}%: '
              f'k10_orig={r["beta_k10_orig"]:+.3f}  k10_slope={r["beta_k10_slope"]:+.4f}  '
              f'allk_orig={r["beta_allk_orig"]:+.3f}  allk_slope={r["beta_allk_slope"]:+.4f}')

    # ── Cross-η row: pool ALL child sizes at k=10 ──
    pc_k10_all = pc[pc['k'] == k_max].copy()
    if len(pc_k10_all) > 50:
        res_cross = compute_global_beta(pc_k10_all)
        boot_cross = bootstrap_beta(pc_k10_all, n_boot=1000)
        boots_cross = boot_cross.get('boots_intercept', np.array([]))
        ci_cross = (np.nanpercentile(boots_cross, 2.5), np.nanpercentile(boots_cross, 97.5)) \
            if len(boots_cross) > 10 else (np.nan, np.nan)

        results.append(dict(
            model=model_name, stock=stock, child=-1, vol=-1,
            eta=-1, k_max=k_max, n_days=pc_k10_all['day'].nunique() if 'day' in pc_k10_all.columns else 0,
            row_type='cross_eta_k10',
            Q_min=int(pc_k10_all['Q'].min()), Q_max=int(pc_k10_all['Q'].max()),
            n_allk=0, n_k10=res_cross.get('n', 0),
            beta_k10_orig=res_cross.get('beta_origin', np.nan),
            beta_k10_slope=res_cross.get('beta_intercept', np.nan),
            beta_allk_orig=np.nan, beta_allk_slope=np.nan,
            ci_lo=ci_cross[0], ci_hi=ci_cross[1],
            r2_allk_orig=np.nan, r2_allk_slope=np.nan,
            I_median=float(np.median(pc_k10_all['I'])),
        ))
        print(f'    >>> ALL η k={k_max}: '
              f'k10_orig={res_cross.get("beta_origin", np.nan):+.3f}  '
              f'k10_slope={res_cross.get("beta_intercept", np.nan):+.4f}  '
              f'Q={int(pc_k10_all["Q"].min())}..{int(pc_k10_all["Q"].max())}  '
              f'N={res_cross.get("n", 0)}')

    # ── Cross-η row: pool ALL child sizes, ALL k ──
    if len(pc) > 50:
        res_cross_all = compute_global_beta(pc)
        results.append(dict(
            model=model_name, stock=stock, child=-1, vol=-1,
            eta=-1, k_max=k_max, n_days=pc['day'].nunique() if 'day' in pc.columns else 0,
            row_type='cross_eta_allk',
            Q_min=int(pc['Q'].min()), Q_max=int(pc['Q'].max()),
            n_allk=res_cross_all.get('n', 0), n_k10=0,
            beta_k10_orig=np.nan, beta_k10_slope=np.nan,
            beta_allk_orig=res_cross_all.get('beta_origin', np.nan),
            beta_allk_slope=res_cross_all.get('beta_intercept', np.nan),
            ci_lo=np.nan, ci_hi=np.nan,
            r2_allk_orig=res_cross_all.get('r2_origin', np.nan),
            r2_allk_slope=res_cross_all.get('r2_intercept', np.nan),
            I_median=float(np.median(pc['I'])),
        ))
        print(f'    >>> ALL η ALL k: '
              f'allk_orig={res_cross_all.get("beta_origin", np.nan):+.3f}  '
              f'allk_slope={res_cross_all.get("beta_intercept", np.nan):+.4f}')

    return results


def make_pdf(all_results, stock, version_label, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.3})

    df = pd.DataFrame(all_results)
    if df.empty:
        return

    csv_path = out_dir / f'per_experiment_beta_{stock}_{version_label}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  Saved CSV: {csv_path} ({len(df)} rows)')

    pdf_path = out_dir / f'per_experiment_beta_{stock}_{version_label}.pdf'
    models = [m for m in MODELS_ORDER if m in df['model'].unique()]
    df_eta = df[df['row_type'] == 'per_eta']
    df_cross_k10 = df[df['row_type'] == 'cross_eta_k10']
    df_cross_allk = df[df['row_type'] == 'cross_eta_allk']

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Full table ──
        fig = plt.figure(figsize=(18, 12))
        fig.text(0.5, 0.97,
                 f'Per-Experiment β — {stock} ({version_label})',
                 ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.95,
                 'k10 = k=10 only (full metaorder) | allk = all k=1..10 | orig = origin | slope = intercept',
                 ha='center', fontsize=9, color='gray')

        header = (f'{"Model":>12s} {"child":>5s} {"η%":>4s} '
                  f'{"β_k10_orig":>10s} {"β_k10_slope":>11s} '
                  f'{"β_allk_orig":>11s} {"β_allk_slope":>12s} '
                  f'{"R²_allk_sl":>10s} {"N_allk":>7s} {"N_k10":>6s}')
        y = 0.91
        fig.text(0.02, y, header, fontsize=7.5, fontfamily='monospace', fontweight='bold')
        y -= 0.014

        for model in models:
            mdf = df_eta[df_eta['model'] == model].sort_values('eta')
            for _, r in mdf.iterrows():
                eta_s = f'{r["eta"]:.0f}' if pd.notna(r['eta']) else '?'
                line = (f'{r["model"]:>12s} {r["child"]:>5.0f} {eta_s:>4s} '
                        f'{r["beta_k10_orig"]:>+10.4f} {r["beta_k10_slope"]:>+11.4f} '
                        f'{r["beta_allk_orig"]:>+11.4f} {r["beta_allk_slope"]:>+12.4f} '
                        f'{r["r2_allk_slope"]:>10.4f} {r["n_allk"]:>7.0f} {r["n_k10"]:>6.0f}')
                fig.text(0.02, y, line, fontsize=7, fontfamily='monospace')
                y -= 0.012
            # Cross-η k=10
            cr_k10 = df_cross_k10[df_cross_k10['model'] == model]
            if not cr_k10.empty:
                r = cr_k10.iloc[0]
                ci_s = f'[{r["ci_lo"]:.3f},{r["ci_hi"]:.3f}]'
                line = (f'{model:>12s} {"ALL":>5s} {"ALL":>4s} '
                        f'{r["beta_k10_orig"]:>+10.4f} {r["beta_k10_slope"]:>+11.4f} '
                        f'{"—":>11s} {"—":>12s} '
                        f'{"—":>10s} {"—":>7s} {r["n_k10"]:>6.0f}')
                fig.text(0.02, y, line, fontsize=7, fontfamily='monospace', fontweight='bold')
                y -= 0.012
            # Cross-η allk
            cr_allk = df_cross_allk[df_cross_allk['model'] == model]
            if not cr_allk.empty:
                r = cr_allk.iloc[0]
                line = (f'{model:>12s} {"ALL":>5s} {"ALL":>4s} '
                        f'{"—":>10s} {"—":>11s} '
                        f'{r["beta_allk_orig"]:>+11.4f} {r["beta_allk_slope"]:>+12.4f} '
                        f'{r["r2_allk_slope"]:>10.4f} {r["n_allk"]:>7.0f} {"—":>6s}')
                fig.text(0.02, y, line, fontsize=7, fontfamily='monospace', fontweight='bold')
                y -= 0.012
            y -= 0.006
            if y < 0.03:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(18, 12)); y = 0.95
        pdf.savefig(fig); plt.close(fig)

        # ── Page 2: β(η) curves — 4 panels ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        titles = ['β_k10_orig', 'β_k10_slope', 'β_allk_orig', 'β_allk_slope']
        cols = ['beta_k10_orig', 'beta_k10_slope', 'beta_allk_orig', 'beta_allk_slope']

        for ax, title, col in zip(axes.flat, titles, cols):
            for model in models:
                mdf = df_eta[df_eta['model'] == model].sort_values('eta').dropna(subset=['eta'])
                if mdf.empty or mdf[col].isna().all():
                    continue
                c = COLORS.get(model, 'gray')
                ax.plot(mdf['eta'], mdf[col], 'o-', color=c, lw=2, markersize=5, label=model)
            ax.axhline(0.5, ls='--', color='red', lw=1, alpha=0.7)
            ax.axhline(0, ls='-', color='gray', lw=0.5)
            ax.set_xlabel('η (%)')
            ax.set_ylabel('β')
            ax.set_title(title)
            ax.legend(fontsize=6)

        fig.suptitle(f'{stock} ({version_label}) — β(η) four estimators', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # ── Page 3: Cross-η bar chart (k=10, slope) ──
        if not df_cross_k10.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            cr = df_cross_k10.sort_values('beta_k10_slope')
            colors_s = [COLORS.get(m, 'gray') for m in cr['model']]
            ax.barh(cr['model'], cr['beta_k10_slope'], color=colors_s, alpha=0.7)
            if cr['ci_lo'].notna().any():
                yerr_lo = (cr['beta_k10_slope'] - cr['ci_lo']).clip(lower=0)
                yerr_hi = (cr['ci_hi'] - cr['beta_k10_slope']).clip(lower=0)
                ax.errorbar(cr['beta_k10_slope'], cr['model'],
                            xerr=[yerr_lo, yerr_hi], fmt='none', color='black', capsize=4)
            ax.axvline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
            ax.axvline(0, ls='-', color='gray', lw=0.5)
            ax.set_xlabel('β (intercept/slope estimator)')
            ax.set_title(f'{stock} — Cross-η β at k=10 (all child sizes pooled)')
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f'  Saved PDF: {pdf_path}')


def main():
    parser = argparse.ArgumentParser(description='Per-experiment β analysis')
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v5')
    parser.add_argument('--out_dir', default='pics_for_per_exp_beta')
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = args.models or [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'{"=" * 70}')
    print(f'  Per-Experiment β — {args.stock} ({args.version})')
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

    make_pdf(all_results, args.stock, args.version, out_dir)

    # ── Clean comparison table ──
    df = pd.DataFrame(all_results)
    df_eta = df[df['row_type'] == 'per_eta']
    df_ck10 = df[df['row_type'] == 'cross_eta_k10']
    df_call = df[df['row_type'] == 'cross_eta_allk']
    models = [m for m in MODELS_ORDER if m in df_eta['model'].unique()]

    print(f'\n{"=" * 100}')
    print(f'  β TABLE — {args.stock} ({args.version})')
    print(f'{"=" * 100}')
    print(f'{"Model":>12s} {"child":>5s} {"η%":>4s}  '
          f'{"β_k10_orig":>10s} {"β_k10_slope":>11s}  '
          f'{"β_allk_orig":>11s} {"β_allk_slope":>12s}')
    print(f'{"-" * 100}')

    for model in models:
        mdf = df_eta[df_eta['model'] == model].sort_values('eta')
        for _, r in mdf.iterrows():
            eta_s = f'{r["eta"]:.0f}' if pd.notna(r['eta']) else '?'
            print(f'{model:>12s} {r["child"]:>5.0f} {eta_s:>4s}  '
                  f'{r["beta_k10_orig"]:>+10.4f} {r["beta_k10_slope"]:>+11.4f}  '
                  f'{r["beta_allk_orig"]:>+11.4f} {r["beta_allk_slope"]:>+12.4f}')
        # Cross-η: one ALL row with all 4 numbers
        ck = df_ck10[df_ck10['model'] == model]
        ca = df_call[df_call['model'] == model]
        k10_orig = ck.iloc[0]['beta_k10_orig'] if not ck.empty else np.nan
        k10_slope = ck.iloc[0]['beta_k10_slope'] if not ck.empty else np.nan
        allk_orig = ca.iloc[0]['beta_allk_orig'] if not ca.empty else np.nan
        allk_slope = ca.iloc[0]['beta_allk_slope'] if not ca.empty else np.nan
        k10o = f'{k10_orig:>+10.4f}' if np.isfinite(k10_orig) else f'{"—":>10s}'
        k10s = f'{k10_slope:>+11.4f}' if np.isfinite(k10_slope) else f'{"—":>11s}'
        ako = f'{allk_orig:>+11.4f}' if np.isfinite(allk_orig) else f'{"—":>11s}'
        aks = f'{allk_slope:>+12.4f}' if np.isfinite(allk_slope) else f'{"—":>12s}'
        print(f'{model:>12s} {"ALL":>5s} {"ALL":>4s}  {k10o} {k10s}  {ako} {aks}')
        print()


if __name__ == '__main__':
    main()
