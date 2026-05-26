#!/usr/bin/env python3
"""
V8 Comprehensive Report: β × σ estimator × k dynamics.

For each stock (AAPL, META, MSFT, NVDA, TSLA):
  1. Parameter selection summary (child, mb, η, time between insertions)
  2. For each σ estimator: β_origin, β_slope (intercept), α, R²
  3. β(K) sliding dynamics per σ estimator — how β evolves with insertion number
  4. Scatter plots: log(I/σ) vs log(Q/V) for each σ method

Output: multi-page PDF + CSV summary.

Usage:
    python lob_impact/run_v8_report.py --out_dir pics_for_v8_report
"""
import argparse, pickle, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, load_daily_params, collect_days,
)
from lob_impact.analyze_volatility_sweep import compute_sigma_methods

# ── Constants ──
LUS = "/lus/lfs1aip2/projects/s5e"
PICKLE_BASE = f"{LUS}/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v8/pickles"
PROJECT_DIR = "/home/s5e/georgenigm.s5e/LOBS5_11_march"

MODELS_ORDER = ['Historic', 'Heuristic', 'CST', 'S5-120M', 'S5-360M', 'S5-4K']
COLORS = {
    'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
    'S5-120M': '#D95F02', 'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E',
}

SIGMA_METHODS = ['parkinson', 'parkinson_2x', 'log_range', 'no_sigma', 'constant',
                 'sqrt_range', 'realized_vol']
SIGMA_LABELS = {
    'parkinson': 'Parkinson (correct)',
    'parkinson_2x': 'Parkinson (2x, old)',
    'log_range': 'ln(H/L)',
    'no_sigma': 'σ = 1',
    'constant': 'σ = median',
    'sqrt_range': '√ln(H/L)',
    'realized_vol': 'Realized Vol',
}

STOCK_PARAMS = {
    'AAPL': dict(child=35, mb=308, msg_rate=165, trade_frac=0.0273, trade_p50=40,
                 adv=6427111, price=243),
    'META': dict(child=10, mb=275, msg_rate=72,  trade_frac=0.0343, trade_p50=12,
                 adv=1089218, price=617),
    'MSFT': dict(child=17, mb=252, msg_rate=97,  trade_frac=0.0356, trade_p50=20,
                 adv=2729404, price=430),
    'NVDA': dict(child=25, mb=264, msg_rate=400, trade_frac=0.0370, trade_p50=28,
                 adv=19235892, price=140),
    'TSLA': dict(child=15, mb=266, msg_rate=232, trade_frac=0.0302, trade_p50=17,
                 adv=6764052, price=390),
}

ALL_STOCKS = ['AAPL', 'META', 'MSFT', 'NVDA', 'TSLA']


def fit_ols(x, y):
    """OLS with intercept: y = α + β·x. Returns (β, α, R², n)."""
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    xv, yv = x[ok], y[ok]
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, np.nan, n
    coeffs = np.polyfit(xv, yv, 1)
    beta, alpha = float(coeffs[0]), float(coeffs[1])
    yhat = beta * xv + alpha
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, alpha, r2, n


def fit_origin(x, y_adj):
    """Origin estimator: y = β·x (no intercept)."""
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    xv, yv = x[ok], y_adj[ok]
    if len(xv) < 2:
        return np.nan, np.nan, 0
    beta = float(np.dot(xv, yv) / np.dot(xv, xv))
    ss_res = np.sum((yv - beta * xv) ** 2)
    ss_tot = np.sum(yv ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2, int(ok.sum())


def load_model_data(stock, model_name):
    """Load pickle and extract point cloud."""
    pkl_dir = Path(PICKLE_BASE) / stock
    pkl_path = pkl_dir / f'{model_name}.pkl'
    if not pkl_path.exists():
        return None

    daily_hl_path = f'{PROJECT_DIR}/lob_impact/daily_h_l_{stock}.csv'

    print(f'  Loading {model_name}/{stock} ({pkl_path.stat().st_size / 1e9:.1f} GB)...')
    with open(pkl_path, 'rb') as f:
        md = pickle.load(f)

    exp_days = collect_days(md)
    daily_params = load_daily_params(daily_hl_path, exp_days)
    filtered, _, _ = filter_model(md)
    pc = extract_point_cloud(filtered, md['grid'], daily_params)
    del md, filtered
    gc.collect()

    if pc.empty or 'daily_sigma' not in pc.columns:
        return None
    return pc


def analyze_stock(stock):
    """Full analysis for one stock: β × σ × k."""
    print(f'\n{"=" * 70}')
    print(f'  V8 Report — {stock}')
    print(f'{"=" * 70}')

    pkl_dir = Path(PICKLE_BASE) / stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]
    print(f'  Models: {models}')

    all_results = []
    model_pcs = {}

    for model in models:
        pc = load_model_data(stock, model)
        if pc is None:
            continue
        model_pcs[model] = pc

        k_max = int(pc['k'].max())
        x_all = np.log(pc['Q'].values / pc['daily_vol'].values)
        I_all = pc['I'].values
        sigma_all = pc['daily_sigma'].values
        sigma_rv = pc['sigma_rv'].values if 'sigma_rv' in pc.columns else None
        sigma_methods = compute_sigma_methods(sigma_all, sigma_rv)

        # ── Per-K β: each K is a separate regression on ~2048 samples ──
        # At K=k, each sample gives one point:
        #   Q = k × child (cumulative volume)
        #   I = cumulative impact after k insertions
        # The 2048 samples all have the same Q but different V (daily vol),
        # so x = log(Q/V) varies across ~20 trading days.
        for K in range(1, k_max + 1):
            mask_K = pc['k'].values == K  # ONLY k=K, not k≤K
            x_K = x_all[mask_K]
            I_K = I_all[mask_K]
            sigma_K = sigma_all[mask_K]
            sigma_rv_K = sigma_rv[mask_K] if sigma_rv is not None else None
            sigma_methods_K = compute_sigma_methods(sigma_K, sigma_rv_K)

            for method_name, sigma_arr in sigma_methods_K.items():
                y_adj = np.log(np.maximum(I_K, 1e-20) / np.maximum(sigma_arr, 1e-20))
                beta_orig, r2_orig, n_orig = fit_origin(x_K, y_adj)
                beta_slope, alpha, r2_slope, n_slope = fit_ols(x_K, y_adj)

                all_results.append(dict(
                    stock=stock, model=model, sigma_method=method_name,
                    K=K, beta_origin=beta_orig, beta_slope=beta_slope,
                    alpha=alpha, r2_origin=r2_orig, r2_slope=r2_slope, n=n_slope,
                ))

        del pc
        gc.collect()

    return pd.DataFrame(all_results), models, model_pcs


def make_report_pdf(all_df, out_dir):
    """Generate multi-page PDF report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = out_dir / 'v8_report.pdf'

    with PdfPages(str(pdf_path)) as pdf:

        # ════════════════════════════════════════════════════════════════
        # PAGE 1: Title + Parameter Summary
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')

        title_text = "V8: Realistic Participation Rate — Comprehensive Report"
        ax.text(0.5, 0.95, title_text, transform=ax.transAxes, fontsize=18,
                fontweight='bold', ha='center', va='top')

        # Parameter table
        header = f'{"Stock":>6s} {"child":>6s} {"mb":>5s} {"sec/ins":>8s} {"Q_meta":>7s} {"Q ($)":>10s} {"Q/ADV":>9s} {"η":>6s} {"mkt_vol":>9s}'
        lines = [header, '-' * 80]
        for s in ALL_STOCKS:
            p = STOCK_PARAMS[s]
            sec = p['mb'] / p['msg_rate']
            Q = 10 * p['child']
            Q_d = Q * p['price']
            q_adv = Q / p['adv'] * 100
            mkt_vol = p['mb'] * p['trade_frac'] * p['trade_p50']
            eta = p['child'] / (p['child'] + mkt_vol) * 100
            lines.append(f'{s:>6s} {p["child"]:>6d} {p["mb"]:>5d} {sec:>7.1f}s {Q:>7d} ${Q_d:>9,d} {q_adv:>8.4f}% {eta:>5.1f}% {mkt_vol:>8.0f}sh')

        table_text = '\n'.join(lines)
        ax.text(0.05, 0.78, table_text, transform=ax.transAxes, fontsize=10,
                fontfamily='monospace', va='top')

        ax.text(0.05, 0.40, "Protocol: i=10 insertions, c=0 cooling, n_cond=500, n_samples=2048\n"
                "child = median MO size per stock (Jan 2026 LOBSTER data)\n"
                "mb calibrated for η ≈ 10% (realistic institutional participation rate)\n\n"
                "Models: Historic, Heuristic, CST, S5-120M, S5-360M, S5-4K\n"
                "σ estimators: Parkinson, Parkinson(2x), ln(H/L), σ=1, σ=median, √ln(H/L), Realized Vol\n\n"
                "Analysis: per-insertion VWAP impact, intercept estimator (OLS with free α)",
                transform=ax.transAxes, fontsize=11, va='top')

        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # For each stock...
        for stock in ALL_STOCKS:
            df_stock = all_df[all_df['stock'] == stock]
            if df_stock.empty:
                continue

            models = [m for m in MODELS_ORDER if m in df_stock['model'].unique()]
            sigma_available = [s for s in SIGMA_METHODS if s in df_stock['sigma_method'].unique()]
            k_max = int(df_stock['K'].max())

            df_all_k = df_stock[df_stock['K'] == k_max]
            x_pos = np.arange(len(models))
            n_methods = len(sigma_available)
            width = 0.8 / max(n_methods, 1)

            # ════════════════════════════════════════════════════════════
            # PAGE 1: Origin vs Intercept (at K=k_max)
            # ════════════════════════════════════════════════════════════
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle(f'{stock} — Origin vs Intercept estimator (K={k_max}, metaorder = {STOCK_PARAMS[stock]["child"]}×{k_max} shares)',
                         fontsize=14, fontweight='bold')

            for ax_idx, (est_name, col) in enumerate([('Origin (β through 0)', 'beta_origin'),
                                                       ('Intercept (free α)', 'beta_slope')]):
                ax = axes[ax_idx]
                for i, sm in enumerate(sigma_available):
                    vals = []
                    for m in models:
                        v = df_all_k[(df_all_k['model'] == m) & (df_all_k['sigma_method'] == sm)][col]
                        vals.append(float(v.iloc[0]) if not v.empty else 0)
                    ax.bar(x_pos + i * width - 0.4, vals, width,
                           label=SIGMA_LABELS.get(sm, sm), alpha=0.75)
                ax.axhline(0.5, ls='--', color='red', lw=2, alpha=0.7)
                ax.axhline(0, ls='-', color='gray', lw=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(models, rotation=30, fontsize=9)
                ax.set_ylabel('β', fontsize=11)
                ax.set_title(est_name)
                if ax_idx == 0:
                    ax.legend(fontsize=6, loc='upper left')
                ax.grid(True, alpha=0.2)

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig); plt.close(fig)

            # ════════════════════════════════════════════════════════════
            # PAGE 2: β(K) for ALL σ estimators (intercept fit)
            # Each K = separate regression on ~2048 samples
            # K = number of child orders in the metaorder
            # ════════════════════════════════════════════════════════════
            df_per_k = df_stock.copy()
            df_per_k['K'] = df_per_k['K'].astype(int)

            n_sigma = len(sigma_available)
            ncols_full = 3
            nrows_full = (n_sigma + ncols_full - 1) // ncols_full

            fig, axes = plt.subplots(nrows_full, ncols_full, figsize=(20, 6 * nrows_full))
            axes = np.atleast_2d(axes).flat
            fig.suptitle(f'{stock} — β(K) for all σ estimators (intercept fit)\n'
                         f'Each K: β from ~2048 samples, metaorder = K child orders, impact measured after K-th insertion',
                         fontsize=13, fontweight='bold')

            for ax_i, sm in enumerate(sigma_available):
                ax = axes[ax_i]
                for model in models:
                    mdf = df_per_k[(df_per_k['model'] == model) &
                                   (df_per_k['sigma_method'] == sm)].sort_values('K')
                    if mdf.empty:
                        continue
                    c = COLORS.get(model, 'gray')
                    ax.plot(mdf['K'], mdf['beta_slope'], 'o-', color=c, lw=2,
                            markersize=4, label=model)
                ax.axhline(0.5, ls='--', color='red', lw=1.5, alpha=0.7)
                ax.axhline(0, ls='-', color='gray', lw=0.5)
                ax.set_xlabel('K (child orders in metaorder)')
                ax.set_ylabel('β (intercept)')
                ax.set_title(f'{SIGMA_LABELS.get(sm, sm)}')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(1, k_max + 1))

            for j in range(n_sigma, nrows_full * ncols_full):
                axes[j].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig); plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # CROSS-STOCK: β comparison across all stocks (at K=10)
        # ════════════════════════════════════════════════════════════════
        df_all_k = all_df[all_df['K'] == 10]

        for sm in ['parkinson', 'sqrt_range', 'no_sigma']:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle(f'Cross-stock β comparison — σ = {SIGMA_LABELS.get(sm, sm)}',
                         fontsize=14, fontweight='bold')

            for ax_idx, (est_name, col) in enumerate([('Origin', 'beta_origin'),
                                                       ('Intercept', 'beta_slope')]):
                ax = axes[ax_idx]
                all_models = list(set(df_all_k['model']))
                all_models = [m for m in MODELS_ORDER if m in all_models]
                x_pos = np.arange(len(all_models))
                width = 0.8 / max(len(ALL_STOCKS), 1)

                for i, stock in enumerate(ALL_STOCKS):
                    vals = []
                    for m in all_models:
                        v = df_all_k[(df_all_k['stock'] == stock) &
                                     (df_all_k['model'] == m) &
                                     (df_all_k['sigma_method'] == sm)][col]
                        vals.append(float(v.iloc[0]) if not v.empty else np.nan)
                    ax.bar(x_pos + i * width - 0.4, vals, width, label=stock, alpha=0.75)

                ax.axhline(0.5, ls='--', color='red', lw=2, alpha=0.7)
                ax.axhline(0, ls='-', color='gray', lw=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(all_models, rotation=30, fontsize=9)
                ax.set_ylabel('β', fontsize=11)
                ax.set_title(est_name)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.2)

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig); plt.close(fig)

    print(f'\nSaved: {pdf_path}')
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description='V8 Comprehensive Report')
    parser.add_argument('--out_dir', default='pics_for_v8_report')
    parser.add_argument('--stocks', nargs='+', default=ALL_STOCKS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for stock in args.stocks:
        df_stock, models, model_pcs = analyze_stock(stock)
        all_dfs.append(df_stock)
        # Free memory
        del model_pcs
        gc.collect()

    all_df = pd.concat(all_dfs, ignore_index=True)

    # Save CSV
    csv_path = out_dir / 'v8_report_all.csv'
    all_df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path} ({len(all_df)} rows)')

    # Print summary table (K=10 = full metaorder)
    df_summary = all_df[all_df['K'] == 10].copy()
    for sm in ['parkinson', 'sqrt_range', 'no_sigma', 'realized_vol']:
        if sm not in df_summary['sigma_method'].unique():
            continue
        print(f'\n{"=" * 100}')
        print(f'  σ = {SIGMA_LABELS.get(sm, sm)}')
        print(f'{"=" * 100}')
        print(f'{"Stock":>6s} {"Model":>12s} {"β_origin":>10s} {"β_slope":>10s} {"α":>10s} {"R²_slope":>10s} {"n":>8s}')
        print('-' * 70)
        for stock in args.stocks:
            models_in_stock = [m for m in MODELS_ORDER if m in df_summary[df_summary['stock'] == stock]['model'].unique()]
            for m in models_in_stock:
                row = df_summary[(df_summary['stock'] == stock) &
                                 (df_summary['model'] == m) &
                                 (df_summary['sigma_method'] == sm)]
                if row.empty:
                    continue
                r = row.iloc[0]
                print(f'{stock:>6s} {m:>12s} {r["beta_origin"]:>+10.4f} {r["beta_slope"]:>+10.4f} {r["alpha"]:>+10.4f} {r["r2_slope"]:>10.4f} {int(r["n"]):>8d}')
            if stock != args.stocks[-1]:
                print()

    # Generate PDF
    make_report_pdf(all_df, out_dir)


if __name__ == '__main__':
    main()
