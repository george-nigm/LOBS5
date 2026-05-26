#!/usr/bin/env python3
"""
Fix β=0.5, fit α (impact coefficient) per model.

Instead of estimating β from log(I) = α + β·log(Q/V),
we fix β=0.5 and compute α = mean(log(I) - 0.5·log(Q/V)).

This follows the MarS approach: confirm the square-root FORM,
compare models by impact LEVEL (α), not exponent (β).

Usage:
    python lob_impact/analyze_fixed_beta.py \
        --summary pics_for_v4_300_GOOG/summary_statistics.csv \
        --daily_hl lob_impact/daily_h_l_GOOG.csv \
        --pickle_base /path/to/pickles --stock GOOG \
        --out_dir pics_for_fixed_beta_GOOG
"""
import argparse, pickle, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, load_daily_params, collect_days,
    TICK_SIZE, N_BOOTSTRAP,
)


def compute_fixed_beta_alpha(pc_df, beta_fixed=0.5):
    """Fix β, compute α = mean(log(I) - β·log(Q/V))."""
    if pc_df.empty or 'daily_vol' not in pc_df.columns:
        return dict(alpha=np.nan, alpha_std=np.nan, n=0, residual_std=np.nan)

    I = pc_df['I'].values
    Q = pc_df['Q'].values
    V = pc_df['daily_vol'].values

    valid = (I > 1e-12) & (Q > 0) & (V > 0) & np.isfinite(I)
    I, Q, V = I[valid], Q[valid], V[valid]

    if len(I) < 10:
        return dict(alpha=np.nan, alpha_std=np.nan, n=0, residual_std=np.nan)

    log_I = np.log(I)
    log_QV = np.log(Q / V)

    # α = mean(log(I) - β·log(Q/V))
    residuals = log_I - beta_fixed * log_QV
    alpha = float(np.mean(residuals))
    alpha_std = float(np.std(residuals) / np.sqrt(len(residuals)))
    residual_std = float(np.std(residuals))

    # R² of the fixed-β model
    y_pred = alpha + beta_fixed * log_QV
    ss_res = np.sum((log_I - y_pred)**2)
    ss_tot = np.sum((log_I - np.mean(log_I))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return dict(alpha=alpha, alpha_std=alpha_std, n=len(I),
                residual_std=residual_std, r2=r2, beta_fixed=beta_fixed)


def bootstrap_alpha(pc_df, beta_fixed=0.5, n_boot=2000):
    if pc_df.empty or 'daily_vol' not in pc_df.columns:
        return np.full(n_boot, np.nan)

    I = pc_df['I'].values
    Q = pc_df['Q'].values
    V = pc_df['daily_vol'].values

    valid = (I > 1e-12) & (Q > 0) & (V > 0) & np.isfinite(I)
    log_I = np.log(I[valid])
    log_QV = np.log(Q[valid] / V[valid])
    n = len(log_I)
    if n < 10:
        return np.full(n_boot, np.nan)

    alphas = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        alphas.append(np.mean(log_I[idx] - beta_fixed * log_QV[idx]))
    return np.array(alphas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--stock', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--beta_fixed', type=float, default=0.5)
    parser.add_argument('--version', default='v4', choices=['v3', 'v4', 'v5'])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle_base = Path(args.pickle_base)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    COLORS = {
        'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
        'CGAN': '#7B4F9E', 'LobS5': '#C88A3A', 'S5-120M': '#D95F02',
        'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E', 'LobS5-v2': '#2CA02C',
    }

    # Find available models
    pkl_dir = pickle_base / args.stock
    if args.models:
        models = args.models
    else:
        models = [f.stem for f in sorted(pkl_dir.glob('*.pkl'))
                  if f.stem != 'ZeroInsertions']
    print(f'Models: {models}')

    all_results = {}

    for model in models:
        pkl_path = pkl_dir / f'{model}.pkl'
        if not pkl_path.exists():
            print(f'  SKIP {model}: {pkl_path} not found')
            continue

        print(f'\n{"="*60}')
        print(f'  {model} / {args.stock} (β fixed = {args.beta_fixed})')
        print(f'{"="*60}')
        print(f'Loading {pkl_path} ({pkl_path.stat().st_size/1e9:.1f} GB)...')

        with open(pkl_path, 'rb') as f:
            md = pickle.load(f)

        # Daily params
        exp_days = collect_days(md)
        daily_params = load_daily_params(args.daily_hl, exp_days)
        print(f'  Daily params: {len(daily_params)} days')

        # Filter
        filtered, n_total, n_skip = filter_model(md)
        print(f'  Filtered: {n_skip}/{n_total} outliers')

        # Point cloud
        pc = extract_point_cloud(filtered, md['grid'], daily_params)
        print(f'  Points: {len(pc)}')

        if pc.empty:
            continue

        # Full metaorder (k=max only)
        k_max = int(pc['k'].max())
        pc_last = pc[pc['k'] == k_max].copy()
        print(f'  Full metaorder (k={k_max}): {len(pc_last)} points')

        # Fix β, fit α
        res = compute_fixed_beta_alpha(pc_last, args.beta_fixed)
        boots = bootstrap_alpha(pc_last, args.beta_fixed)
        ci_lo, ci_hi = np.nanpercentile(boots, [2.5, 97.5])

        # Also compute free β for comparison
        if 'daily_vol' in pc_last.columns:
            x = np.log(pc_last['Q'].values / pc_last['daily_vol'].values)
            y = np.log(pc_last['I'].values)
            valid = np.isfinite(x) & np.isfinite(y) & (pc_last['I'].values > 1e-12)
            x, y = x[valid], y[valid]
            if len(x) > 10:
                A = np.column_stack([x, np.ones(len(x))])
                coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
                beta_free = coeffs[0]
            else:
                beta_free = np.nan
        else:
            beta_free = np.nan

        print(f'  α (fixed β={args.beta_fixed}): {res["alpha"]:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]')
        print(f'  α means: exp(α) = {np.exp(res["alpha"]):.6f} (impact level)')
        print(f'  R² (fixed β): {res["r2"]:.4f}')
        print(f'  β (free):     {beta_free:.4f}')
        print(f'  N:            {res["n"]}')

        all_results[model] = dict(
            res=res, boots=boots, ci_lo=ci_lo, ci_hi=ci_hi,
            beta_free=beta_free, pc_last=pc_last,
        )

    if not all_results:
        print('No results')
        sys.exit(1)

    models_done = list(all_results.keys())

    # ═══════════════════════════════════════════════════════
    # PDF Report
    # ═══════════════════════════════════════════════════════
    pdf_path = out_dir / f'fixed_beta_report_{args.stock}.pdf'
    with PdfPages(str(pdf_path)) as pdf_out:

        # Summary table
        fig = plt.figure(figsize=(11, 8))
        fig.text(0.5, 0.95, f'Fixed β={args.beta_fixed} Analysis — {args.stock}',
                 ha='center', fontsize=16, fontweight='bold')

        y = 0.85
        fig.text(0.05, y, f'{"Model":>12s}  {"α":>8s}  {"CI_lo":>8s}  {"CI_hi":>8s}  {"exp(α)":>10s}  {"R²_fixed":>8s}  {"β_free":>8s}  {"N":>8s}',
                 fontsize=9, fontfamily='monospace')
        y -= 0.025
        for m in models_done:
            r = all_results[m]
            fig.text(0.05, y,
                     f'{m:>12s}  {r["res"]["alpha"]:>8.3f}  {r["ci_lo"]:>8.3f}  {r["ci_hi"]:>8.3f}  '
                     f'{np.exp(r["res"]["alpha"]):>10.6f}  {r["res"]["r2"]:>8.4f}  '
                     f'{r["beta_free"]:>8.3f}  {r["res"]["n"]:>8d}',
                     fontsize=9, fontfamily='monospace')
            y -= 0.022

        y -= 0.03
        fig.text(0.05, y, 'Interpretation:', fontsize=11, fontweight='bold')
        y -= 0.025
        fig.text(0.05, y, f'α = log-impact level when Q/V = 1 (normalized).', fontsize=10)
        y -= 0.022
        fig.text(0.05, y, f'exp(α) = multiplicative impact coefficient.', fontsize=10)
        y -= 0.022
        fig.text(0.05, y, f'Higher α = model generates MORE price impact per unit of (Q/V)^0.5.', fontsize=10)
        y -= 0.022
        fig.text(0.05, y, f'R²_fixed = how well the fixed-β model fits vs free intercept.', fontsize=10)
        y -= 0.035
        fig.text(0.05, y, f'β_free = unconstrained OLS slope (for comparison).', fontsize=10)
        pdf_out.savefig(fig); plt.close(fig)

        # Bar chart of α values
        fig, ax = plt.subplots(figsize=(10, 6))
        names = models_done
        alphas = [all_results[m]['res']['alpha'] for m in names]
        ci_los = [all_results[m]['ci_lo'] for m in names]
        ci_his = [all_results[m]['ci_hi'] for m in names]
        colors = [COLORS.get(m, 'gray') for m in names]
        errs = [[a - lo for a, lo in zip(alphas, ci_los)],
                [hi - a for a, hi in zip(alphas, ci_his)]]
        bars = ax.bar(range(len(names)), alphas, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax.errorbar(range(len(names)), alphas, yerr=errs, fmt='none', ecolor='black', capsize=5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel(f'α (impact level, β fixed at {args.beta_fixed})')
        ax.set_title(f'{args.stock} — Impact Coefficient α (fixed β={args.beta_fixed})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf_out.savefig(fig); plt.close(fig)

        # Scatter with fixed-β line
        n_models = len(models_done)
        nc = min(3, n_models)
        nr = (n_models + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr), squeeze=False)
        for idx, model in enumerate(models_done):
            r, c_idx = idx // nc, idx % nc
            ax = axes[r][c_idx]
            pc = all_results[model]['pc_last']
            if 'daily_vol' not in pc.columns:
                continue
            x = np.log(pc['Q'].values / pc['daily_vol'].values)
            y_vals = np.log(pc['I'].values)
            valid = np.isfinite(x) & np.isfinite(y_vals) & (pc['I'].values > 1e-12)
            xv, yv = x[valid], y_vals[valid]
            ax.scatter(xv, yv, alpha=0.03, s=2, color=COLORS.get(model, 'C0'))
            # Fixed β line
            alpha_val = all_results[model]['res']['alpha']
            xl = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xl, alpha_val + args.beta_fixed * xl, 'r-', lw=2, label=f'β={args.beta_fixed}')
            # Free β line
            bf = all_results[model]['beta_free']
            if np.isfinite(bf):
                A = np.column_stack([xv, np.ones(len(xv))])
                coeffs = np.linalg.lstsq(A, yv, rcond=None)[0]
                ax.plot(xl, coeffs[0]*xl + coeffs[1], 'k--', lw=1.5, label=f'β={bf:.2f}')
            ax.set_title(f'{model}\nα={alpha_val:.3f}', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlabel('log(Q/V)')
            ax.set_ylabel('log(I)')
        for idx in range(len(models_done), nr*nc):
            axes[idx//nc][idx%nc].set_visible(False)
        fig.suptitle(f'{args.stock} — Fixed β={args.beta_fixed} vs Free β', fontsize=14, fontweight='bold')
        fig.tight_layout()
        pdf_out.savefig(fig); plt.close(fig)

        # Bootstrap distributions of α
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in models_done:
            boots = all_results[model]['boots']
            valid_boots = boots[np.isfinite(boots)]
            if len(valid_boots) > 10:
                ax.hist(valid_boots, bins=50, alpha=0.5, label=model,
                        color=COLORS.get(model, 'gray'), density=True)
        ax.set_xlabel(f'α (fixed β={args.beta_fixed})')
        ax.set_ylabel('Density')
        ax.set_title(f'{args.stock} — Bootstrap α Distributions')
        ax.legend(fontsize=9)
        fig.tight_layout()
        pdf_out.savefig(fig); plt.close(fig)

    print(f'\nSaved: {pdf_path}')

    # Summary to stdout
    print(f'\n{"="*70}')
    print(f'  SUMMARY: {args.stock} (β fixed at {args.beta_fixed})')
    print(f'{"="*70}')
    print(f'{"Model":>12s}  {"α":>8s}  {"CI":>18s}  {"exp(α)":>10s}  {"R²":>8s}  {"β_free":>8s}  {"N":>8s}')
    for m in models_done:
        r = all_results[m]
        print(f'{m:>12s}  {r["res"]["alpha"]:>8.3f}  [{r["ci_lo"]:.3f},{r["ci_hi"]:.3f}]  '
              f'{np.exp(r["res"]["alpha"]):>10.6f}  {r["res"]["r2"]:>8.4f}  '
              f'{r["beta_free"]:>8.3f}  {r["res"]["n"]:>8d}')


if __name__ == '__main__':
    main()
