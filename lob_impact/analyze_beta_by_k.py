#!/usr/bin/env python3
"""
β(k): how does the scaling exponent change with the number of insertions?

For each k = 1, 2, ..., k_max:
  - Pool across all η (cross-η) at that specific k
  - x = log(Q_cumulative / V), where Q_cumulative = k × Q_child
  - y = log(I_at_k / σ)
  - Fit intercept: y = α + β × x  →  β_slope(k)
  - Also fit origin: β_origin(k)

Output:
  - CSV with β(k) per model
  - PDF with β(k) curves + Kyle λ(k) for comparison

Usage:
    python lob_impact/analyze_beta_by_k.py --stock AAPL \
        --pickle_base /path/to/pickles --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v6 --out_dir pics_for_beta_by_k
"""
import argparse, pickle, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, load_daily_params, collect_days,
)

MODELS_ORDER = ['Historic', 'Heuristic', 'CST', 'S5-4K', 'S5-4K-4000', 'S5-120M', 'S5-360M']

COLORS = {
    'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
    'S5-4K': '#5B7BBF', 'S5-4K-4000': '#1B4F72', 'S5-120M': '#D95F02', 'S5-360M': '#B5446E',
}


def fit_both(x, y):
    """Return (beta_origin, beta_slope, alpha, r2_slope, n)."""
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    xv, yv = x[ok], y[ok]
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, n

    # Origin
    beta_orig = float(np.dot(xv, yv) / np.dot(xv, xv))

    # Intercept
    coeffs = np.polyfit(xv, yv, 1)
    beta_slope = float(coeffs[0])
    alpha = float(coeffs[1])
    yhat = beta_slope * xv + alpha
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta_orig, beta_slope, alpha, r2, n


def analyze_model(pkl_path, daily_hl_path, model_name, stock):
    """Compute β(k) for each k=1..k_max."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f'  SKIP {model_name}')
        return []

    print(f'\n  Loading {model_name} ({pkl_path.stat().st_size / 1e9:.1f} GB)...')
    with open(pkl_path, 'rb') as f:
        md = pickle.load(f)

    exp_days = collect_days(md)
    daily_params = load_daily_params(daily_hl_path, exp_days)
    filtered, _, _ = filter_model(md)
    pc = extract_point_cloud(filtered, md['grid'], daily_params)
    del md, filtered
    gc.collect()

    if pc.empty or 'daily_sigma' not in pc.columns:
        return []

    k_max = int(pc['k'].max())
    results = []

    for k in range(1, k_max + 1):
        pc_k = pc[pc['k'] == k]
        if len(pc_k) < 50:
            continue

        x = np.log(pc_k['Q'].values / pc_k['daily_vol'].values)
        y = np.log(pc_k['I'].values / pc_k['daily_sigma'].values)

        beta_orig, beta_slope, alpha, r2, n = fit_both(x, y)

        # Also with realized vol
        beta_rv_slope = np.nan
        if 'sigma_rv' in pc_k.columns:
            y_rv = np.log(pc_k['I'].values / pc_k['sigma_rv'].values)
            _, beta_rv_slope, _, _, _ = fit_both(x, y_rv)

        # Q range info
        Q_vals = pc_k['Q'].values
        log_Q_range = np.log(Q_vals.max()) - np.log(Q_vals[Q_vals > 0].min()) if Q_vals.max() > 0 else 0

        # Median impact
        I_med = float(np.median(pc_k['I'].values))

        results.append(dict(
            model=model_name, stock=stock, k=k, k_max=k_max,
            beta_origin=beta_orig, beta_slope=beta_slope, beta_rv_slope=beta_rv_slope,
            alpha=alpha, r2=r2, n=n,
            log_Q_range=log_Q_range, I_median=I_med,
        ))

    return results


def make_plots(df, stock, version, out_dir):
    """β(k) curves for all models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    models = [m for m in MODELS_ORDER if m in df['model'].unique()]
    k_max = int(df['k'].max())

    pdf_path = out_dir / f'beta_by_k_{stock}_{version}.pdf'
    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: β_slope(k) ──
        fig, ax = plt.subplots(figsize=(12, 7))
        for model in models:
            mdf = df[df['model'] == model].sort_values('k')
            c = COLORS.get(model, 'gray')
            ax.plot(mdf['k'], mdf['beta_slope'], 'o-', color=c, lw=2.5,
                    markersize=7, label=model)
        ax.axhline(0.5, ls='--', color='red', lw=1.5, alpha=0.7, label='Theory β=0.5')
        ax.axhline(0, ls='-', color='gray', lw=0.5)
        ax.set_xlabel('Number of insertions (k)', fontsize=13)
        ax.set_ylabel('β (intercept estimator, cross-η)', fontsize=13)
        ax.set_title(f'{stock} ({version}) — β(k): Impact scaling vs number of insertions\n'
                     f'Cross-η pooled, Parkinson σ', fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, k_max + 1))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Page 2: β_rv_slope(k) ──
        if 'beta_rv_slope' in df.columns and df['beta_rv_slope'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 7))
            for model in models:
                mdf = df[df['model'] == model].sort_values('k')
                c = COLORS.get(model, 'gray')
                ax.plot(mdf['k'], mdf['beta_rv_slope'], 'o-', color=c, lw=2.5,
                        markersize=7, label=model)
            ax.axhline(0.5, ls='--', color='red', lw=1.5, alpha=0.7, label='Theory β=0.5')
            ax.axhline(0, ls='-', color='gray', lw=0.5)
            ax.set_xlabel('Number of insertions (k)', fontsize=13)
            ax.set_ylabel('β (intercept, realised σ_RV)', fontsize=13)
            ax.set_title(f'{stock} ({version}) — β(k) with realised midprice vol\n'
                         f'Historic should be ≈ 0 at all k', fontsize=13, fontweight='bold')
            ax.set_xticks(range(1, k_max + 1))
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ── Page 3: β_origin(k) for comparison ──
        fig, ax = plt.subplots(figsize=(12, 7))
        for model in models:
            mdf = df[df['model'] == model].sort_values('k')
            c = COLORS.get(model, 'gray')
            ax.plot(mdf['k'], mdf['beta_origin'], 'o-', color=c, lw=2.5,
                    markersize=7, label=model)
        ax.axhline(0.5, ls='--', color='red', lw=1.5, alpha=0.7, label='Theory β=0.5')
        ax.set_xlabel('Number of insertions (k)', fontsize=13)
        ax.set_ylabel('β (origin estimator, cross-η)', fontsize=13)
        ax.set_title(f'{stock} ({version}) — β_origin(k): shows ratio bias at all k\n'
                     f'All models ≈ 0.5 regardless of k', fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, k_max + 1))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Page 4: Slope vs Origin side by side ──
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, col, title in [
            (axes[0], 'beta_slope', 'Intercept β(k)'),
            (axes[1], 'beta_origin', 'Origin β(k) — ratio bias'),
        ]:
            for model in models:
                mdf = df[df['model'] == model].sort_values('k')
                c = COLORS.get(model, 'gray')
                ax.plot(mdf['k'], mdf[col], 'o-', color=c, lw=2, markersize=6, label=model)
            ax.axhline(0.5, ls='--', color='red', lw=1.5, alpha=0.7)
            ax.axhline(0, ls='-', color='gray', lw=0.5)
            ax.set_xlabel('k')
            ax.set_ylabel('β')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(range(1, k_max + 1))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'{stock} ({version}) — Intercept vs Origin β(k)', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # ── Page 5: R² and n(k) diagnostics ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for model in models:
            mdf = df[df['model'] == model].sort_values('k')
            c = COLORS.get(model, 'gray')
            axes[0].plot(mdf['k'], mdf['r2'], 'o-', color=c, lw=2, markersize=5, label=model)
            axes[1].plot(mdf['k'], mdf['n'], 'o-', color=c, lw=2, markersize=5, label=model)
        axes[0].set_ylabel('R²')
        axes[0].set_title('R² of intercept fit at each k')
        axes[1].set_ylabel('N points')
        axes[1].set_title('Sample size at each k')
        for ax in axes:
            ax.set_xlabel('k')
            ax.set_xticks(range(1, k_max + 1))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Page 6: Median I(k) — raw impact buildup ──
        fig, ax = plt.subplots(figsize=(12, 7))
        for model in models:
            mdf = df[df['model'] == model].sort_values('k')
            c = COLORS.get(model, 'gray')
            ax.plot(mdf['k'], mdf['I_median'], 'o-', color=c, lw=2.5,
                    markersize=7, label=model)
        ax.set_xlabel('Number of insertions (k)', fontsize=13)
        ax.set_ylabel('Median |I| (implementation shortfall)', fontsize=13)
        ax.set_title(f'{stock} ({version}) — Raw impact buildup\n'
                     f'How much impact accumulates with each insertion',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, k_max + 1))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f'Saved: {pdf_path}')


def main():
    parser = argparse.ArgumentParser(description='β(k) analysis')
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v6')
    parser.add_argument('--out_dir', default='pics_for_beta_by_k')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'β(k) analysis — {args.stock} ({args.version})')
    print(f'Models: {models}')

    all_results = []
    for model in models:
        results = analyze_model(
            str(pkl_dir / f'{model}.pkl'), args.daily_hl, model, args.stock)
        all_results.extend(results)
        gc.collect()

    if not all_results:
        print('ERROR: no results')
        sys.exit(1)

    df = pd.DataFrame(all_results)
    csv_path = out_dir / f'beta_by_k_{args.stock}_{args.version}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path} ({len(df)} rows)')

    # Print table
    print(f'\n{"=" * 90}')
    print(f'  β_slope(k) — {args.stock} ({args.version}), cross-η, Parkinson σ')
    print(f'{"=" * 90}')
    header = f'{"k":>3s}'
    for m in models:
        header += f'  {m:>12s}'
    print(header)
    print('-' * 90)
    for k in sorted(df['k'].unique()):
        row = f'{int(k):>3d}'
        for m in models:
            val = df[(df['model'] == m) & (df['k'] == k)]['beta_slope']
            if not val.empty:
                row += f'  {float(val.iloc[0]):>+12.4f}'
            else:
                row += f'  {"—":>12s}'
        print(row)

    make_plots(df, args.stock, args.version, out_dir)


if __name__ == '__main__':
    main()
