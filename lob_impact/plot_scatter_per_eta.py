#!/usr/bin/env python3
"""
Per-η scatter: regression lines for EACH participation rate + combined.

For each (model, stock): one PNG with:
  - All k=1..10 points colored by η
  - One regression line per η (intercept estimator)
  - One thick combined line (all η pooled)
  - Legend: β_η for each + β_pooled

Also: comparison grid (all 6 models on one page).

Usage:
    python lob_impact/plot_scatter_per_eta.py --stock AAPL \
        --pickle_base /path/to/pickles --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v6 --out_dir pics_for_scatter_per_eta
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

# Diverging colormap for η: cool → warm
ETA_CMAP = {
    5: '#2166ac', 9: '#4393c3', 17: '#92c5de', 19: '#92c5de',
    22: '#d1e5f0', 26: '#fddbc7', 50: '#f4a582',
    56: '#f4a582', 72: '#d6604d', 84: '#d6604d', 90: '#b2182b',
}


def fit_intercept(x, y):
    """Fit log(I/σ) = α + β × log(Q/V). Returns β, α, R²."""
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    xv, yv = x[ok], y[ok]
    if len(xv) < 5:
        return np.nan, np.nan, np.nan, 0
    coeffs = np.polyfit(xv, yv, 1)
    beta, alpha = float(coeffs[0]), float(coeffs[1])
    yhat = beta * xv + alpha
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, alpha, r2, int(ok.sum())


def make_per_eta_png(model, stock, pc, out_dir, version):
    """One PNG per model: scatter colored by η, regression line per η + combined."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'daily_sigma' not in pc.columns or 'daily_vol' not in pc.columns:
        return None

    x_all = np.log(pc['Q'].values / pc['daily_vol'].values)
    y_all = np.log(pc['I'].values / pc['daily_sigma'].values)
    vols = pc['vol'].values

    ok_all = np.isfinite(x_all) & np.isfinite(y_all) & (x_all != 0)
    if ok_all.sum() < 50:
        return None

    # Combined fit
    beta_all, alpha_all, r2_all, n_all = fit_intercept(x_all, y_all)

    # Per-η fits
    unique_vols = sorted(pc['vol'].unique())
    eta_fits = []
    for vol in unique_vols:
        mask = vols == vol
        eta = ETA_MAP.get((stock, int(vol)), 0)
        beta_e, alpha_e, r2_e, n_e = fit_intercept(x_all[mask], y_all[mask])
        eta_fits.append(dict(vol=int(vol), eta=eta, beta=beta_e, alpha=alpha_e, r2=r2_e, n=n_e))

    # Plot
    fig, ax = plt.subplots(figsize=(11, 8))

    # Scatter per η
    for vol in unique_vols:
        mask = vols == vol
        eta = ETA_MAP.get((stock, int(vol)), 0)
        c = ETA_CMAP.get(eta, '#888888')
        x_sub = x_all[mask & ok_all]
        y_sub = y_all[mask & ok_all]
        ax.scatter(x_sub, y_sub, s=2, alpha=0.08, color=c, rasterized=True)

    # Per-η regression lines
    x_range = np.linspace(x_all[ok_all].min() - 0.3, x_all[ok_all].max() + 0.3, 100)
    for ef in eta_fits:
        if not np.isfinite(ef['beta']):
            continue
        c = ETA_CMAP.get(ef['eta'], '#888888')
        ax.plot(x_range, ef['beta'] * x_range + ef['alpha'], '-', color=c, lw=1.5, alpha=0.8,
                label=f'η={ef["eta"]}%: β={ef["beta"]:.3f}')

    # Combined line (thick)
    ax.plot(x_range, beta_all * x_range + alpha_all, 'k-', lw=3, alpha=0.9,
            label=f'ALL η: β={beta_all:.3f}')

    # Theory
    ax.plot(x_range, 0.5 * x_range, 'k:', lw=1.5, alpha=0.4, label='Theory β=0.5')

    ax.set_xlabel('log(Q / V_daily)', fontsize=13)
    ax.set_ylabel('log(I / σ)', fontsize=13)
    ax.set_title(f'{model} — {stock} ({version})\nPer-η regression lines (all k=1..10, intercept estimator)',
                 fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / f'{model}_{stock}_{version}_per_eta_lines.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    {png_path.name}')

    return dict(model=model, beta_all=beta_all, eta_fits=eta_fits)


def make_comparison(all_data, stock, version, out_dir):
    """6 models on one grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = [m for m in MODELS_ORDER if m in all_data]
    if not models:
        return

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    for i, model in enumerate(models):
        ax = axes.flat[i]
        info = all_data[model]
        pc = info['pc']

        x_all = np.log(pc['Q'].values / pc['daily_vol'].values)
        y_all = np.log(pc['I'].values / pc['daily_sigma'].values)
        vols = pc['vol'].values
        ok_all = np.isfinite(x_all) & np.isfinite(y_all) & (x_all != 0)

        # Scatter
        for vol in sorted(pc['vol'].unique()):
            mask = (vols == vol) & ok_all
            eta = ETA_MAP.get((stock, int(vol)), 0)
            c = ETA_CMAP.get(eta, '#888888')
            ax.scatter(x_all[mask], y_all[mask], s=1.5, alpha=0.06, color=c, rasterized=True)

        x_range = np.linspace(x_all[ok_all].min() - 0.2, x_all[ok_all].max() + 0.2, 50)

        # Per-η lines
        for ef in info['eta_fits']:
            if not np.isfinite(ef['beta']):
                continue
            c = ETA_CMAP.get(ef['eta'], '#888888')
            ax.plot(x_range, ef['beta'] * x_range + ef['alpha'], '-', color=c, lw=1.2, alpha=0.7,
                    label=f'η={ef["eta"]}%: β={ef["beta"]:.2f}')

        # Combined
        beta_all = info['beta_all']
        alpha_all = info['alpha_all']
        ax.plot(x_range, beta_all * x_range + alpha_all, 'k-', lw=2.5,
                label=f'ALL: β={beta_all:.3f}')

        ax.set_title(f'{model} (β_pooled = {beta_all:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('log(Q/V)')
        ax.set_ylabel('log(I/σ)')
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(True, alpha=0.3)

    for j in range(len(models), 6):
        axes.flat[j].set_visible(False)

    fig.suptitle(f'{stock} ({version}) — Per-η β (intercept, all k=1..10)\n'
                 f'Each color = one participation rate, black = all pooled',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = out_dir / f'comparison_per_eta_{stock}_{version}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {png_path.name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v6')
    parser.add_argument('--out_dir', default='pics_for_scatter_per_eta')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'Per-η scatter — {args.stock} ({args.version})')
    all_data = {}

    for model in models:
        pkl_path = pkl_dir / f'{model}.pkl'
        print(f'\n  {model} ({pkl_path.stat().st_size / 1e9:.1f} GB)...')

        with open(pkl_path, 'rb') as f:
            md = pickle.load(f)
        exp_days = collect_days(md)
        daily_params = load_daily_params(args.daily_hl, exp_days)
        filtered, _, _ = filter_model(md)
        pc = extract_point_cloud(filtered, md['grid'], daily_params)
        del md, filtered
        gc.collect()

        if pc.empty:
            continue

        info = make_per_eta_png(model, args.stock, pc, out_dir, args.version)
        if info:
            info['pc'] = pc
            info['alpha_all'] = fit_intercept(
                np.log(pc['Q'].values / pc['daily_vol'].values),
                np.log(pc['I'].values / pc['daily_sigma'].values)
            )[1]
            all_data[model] = info
        gc.collect()

    make_comparison(all_data, args.stock, args.version, out_dir)


if __name__ == '__main__':
    main()
