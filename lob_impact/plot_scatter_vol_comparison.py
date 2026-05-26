#!/usr/bin/env python3
"""
Side-by-side scatter: how different σ estimators shift points but preserve slope.

For each stock: 6-model grid, 3 panels per model (Parkinson, σ=1, realized vol).
Shows: points move up/down but regression slope (intercept estimator) stays similar.

Usage:
    python lob_impact/plot_scatter_vol_comparison.py --stock AAPL \
        --pickle_base /path/to/pickles --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v6 --out_dir pics_for_vol_comparison
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

ETA_COLORS = {
    5: '#2166ac', 9: '#4393c3', 17: '#92c5de', 19: '#92c5de',
    22: '#d1e5f0', 26: '#d1e5f0', 50: '#f4a582',
    56: '#f4a582', 72: '#d6604d', 84: '#d6604d', 90: '#b2182b',
}

SIGMA_METHODS = [
    ('Parkinson $\\sigma$', 'parkinson'),
    ('No $\\sigma$ ($\\sigma = 1$)', 'no_sigma'),
    ('Realised $\\sigma_{RV}$', 'realized_vol'),
]


def compute_sigma_variants(daily_sigma, sigma_rv):
    """Return dict of sigma arrays."""
    return {
        'parkinson': daily_sigma,
        'no_sigma': np.ones_like(daily_sigma),
        'realized_vol': sigma_rv if sigma_rv is not None else daily_sigma,
    }


def make_model_triple(model, stock, pc_k10, out_dir, version):
    """3-panel figure for one model: Parkinson / no_sigma / realized_vol."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'daily_sigma' not in pc_k10.columns or 'daily_vol' not in pc_k10.columns:
        return None

    x = np.log(pc_k10['Q'].values / pc_k10['daily_vol'].values)
    I_arr = pc_k10['I'].values
    sigma_park = pc_k10['daily_sigma'].values
    sigma_rv = pc_k10['sigma_rv'].values if 'sigma_rv' in pc_k10.columns else None
    vols = pc_k10['vol'].values

    sigmas = compute_sigma_variants(sigma_park, sigma_rv)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    results = {}
    for ax, (label, method) in zip(axes, SIGMA_METHODS):
        sigma = sigmas[method]
        y = np.log(np.maximum(I_arr, 1e-20) / np.maximum(sigma, 1e-20))
        ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
        xv, yv = x[ok], y[ok]

        if len(xv) < 10:
            continue

        # Scatter colored by η
        vols_ok = vols[ok]
        eta_arr = np.array([ETA_MAP.get((stock, int(v)), 0) for v in vols_ok])
        for eta_val in sorted(set(eta_arr)):
            mask = eta_arr == eta_val
            c = ETA_COLORS.get(eta_val, '#888888')
            ax.scatter(xv[mask], yv[mask], s=2, alpha=0.08, color=c, rasterized=True)

        # Origin fit
        beta_orig = float(np.dot(xv, yv) / np.dot(xv, xv))
        # Intercept fit
        coeffs = np.polyfit(xv, yv, 1)
        beta_int, alpha_int = float(coeffs[0]), float(coeffs[1])

        x_range = np.linspace(xv.min() - 0.3, xv.max() + 0.3, 50)
        ax.plot(x_range, beta_orig * x_range, 'r-', lw=2,
                label=f'Origin $\\beta = {beta_orig:.3f}$')
        ax.plot(x_range, beta_int * x_range + alpha_int, 'b--', lw=2,
                label=f'Intercept $\\beta = {beta_int:.3f}$')
        ax.plot(x_range, 0.5 * x_range, 'k:', lw=1, alpha=0.5)

        ax.set_xlabel('$\\ln(Q/V)$')
        ax.set_ylabel(f'$\\ln(I/{label.split("$")[1] if "$" in label else "\\\\sigma"})$')
        ax.set_title(f'{label}\n$\\beta_{{slope}} = {beta_int:.3f}$', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        results[method] = dict(beta_orig=beta_orig, beta_int=beta_int)

    fig.suptitle(f'{model} — {stock} ({version}) k=10 cross-$\\eta$: '
                 f'Effect of volatility normalisation',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = out_dir / f'{model}_{stock}_{version}_vol_comparison.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    {png_path.name}')
    return results


def make_grid(all_data, stock, version, out_dir):
    """6 models × 3 σ methods = 18-panel grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = [m for m in MODELS_ORDER if m in all_data]
    if not models:
        return

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(22, 4 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for i, model in enumerate(models):
        info = all_data[model]
        pc_k10 = info['pc']
        x = np.log(pc_k10['Q'].values / pc_k10['daily_vol'].values)
        I_arr = pc_k10['I'].values
        sigma_park = pc_k10['daily_sigma'].values
        sigma_rv = pc_k10['sigma_rv'].values if 'sigma_rv' in pc_k10.columns else None
        vols = pc_k10['vol'].values
        sigmas = compute_sigma_variants(sigma_park, sigma_rv)

        for j, (label, method) in enumerate(SIGMA_METHODS):
            ax = axes[i, j]
            sigma = sigmas[method]
            y = np.log(np.maximum(I_arr, 1e-20) / np.maximum(sigma, 1e-20))
            ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
            xv, yv = x[ok], y[ok]

            if len(xv) < 10:
                continue

            # Scatter (grey, fast)
            ax.scatter(xv, yv, s=1, alpha=0.04, color='#555555', rasterized=True)

            beta_orig = float(np.dot(xv, yv) / np.dot(xv, xv))
            coeffs = np.polyfit(xv, yv, 1)
            beta_int = float(coeffs[0])
            alpha_int = float(coeffs[1])

            x_range = np.linspace(xv.min() - 0.2, xv.max() + 0.2, 50)
            ax.plot(x_range, beta_orig * x_range, 'r-', lw=1.5,
                    label=f'Orig {beta_orig:.2f}')
            ax.plot(x_range, beta_int * x_range + alpha_int, 'b--', lw=1.5,
                    label=f'Slope {beta_int:.2f}')

            if i == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{model}\n$\\ln(I/\\sigma)$', fontsize=9)
            ax.legend(fontsize=6, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    fig.suptitle(f'{stock} ({version}) — Volatility normalisation comparison\n'
                 f'Points shift vertically with $\\sigma$ choice, slope preserved',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = out_dir / f'grid_vol_comparison_{stock}_{version}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {png_path.name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v6')
    parser.add_argument('--out_dir', default='pics_for_vol_comparison')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'Vol comparison scatter — {args.stock} ({args.version})')
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

        k_max = int(pc['k'].max())
        pc_k10 = pc[pc['k'] == k_max].copy()

        results = make_model_triple(model, args.stock, pc_k10, out_dir, args.version)
        if results:
            all_data[model] = dict(pc=pc_k10, results=results)
        gc.collect()

    make_grid(all_data, args.stock, args.version, out_dir)


if __name__ == '__main__':
    main()
