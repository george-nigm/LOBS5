#!/usr/bin/env python3
"""
Scatter plots: log(Q/V) vs log(I/σ) with origin and intercept regression lines.

For each (model, stock) at k=10 cross-η:
  1. scatter_raw.png    — just points, colored by η
  2. scatter_origin.png — + origin line (β_origin, through 0)
  3. scatter_interc.png — + intercept line (β_slope, free α)

Usage:
    python lob_impact/plot_scatter_beta.py --stock AAPL \
        --pickle_base /path/to/pickles --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v6 --out_dir pics_for_scatter_beta
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


def make_scatter_pngs(model_name, stock, pc_k10, out_dir, version):
    """Generate 3 PNGs for one model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.3})

    if 'daily_sigma' not in pc_k10.columns or 'daily_vol' not in pc_k10.columns:
        return

    x = np.log(pc_k10['Q'].values / pc_k10['daily_vol'].values)
    y_adj = np.log(pc_k10['I'].values / pc_k10['daily_sigma'].values)
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    x, y_adj = x[ok], y_adj[ok]
    vols = pc_k10['vol'].values[ok]

    if len(x) < 10:
        return

    # Compute both betas
    beta_origin = float(np.dot(x, y_adj) / np.dot(x, x))
    coeffs = np.polyfit(x, y_adj, 1)
    beta_interc = float(coeffs[0])
    alpha_interc = float(coeffs[1])

    # η labels per point
    eta_arr = np.array([ETA_MAP.get((stock, int(v)), 0) for v in vols])
    unique_etas = sorted(set(eta_arr))

    x_range = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)

    prefix = f'{model_name}_{stock}_{version}'

    for plot_type in ['raw', 'origin', 'intercept', 'both']:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Scatter colored by η
        for eta_val in unique_etas:
            mask = eta_arr == eta_val
            c = ETA_COLORS.get(eta_val, '#888888')
            ax.scatter(x[mask], y_adj[mask], s=3, alpha=0.12, color=c,
                       label=f'η={eta_val}%', rasterized=True)

        # Regression lines
        if plot_type in ('origin', 'both'):
            y_origin = beta_origin * x_range
            ax.plot(x_range, y_origin, 'r-', lw=2.5,
                    label=f'Origin: β={beta_origin:.3f}')

        if plot_type in ('intercept', 'both'):
            y_interc = beta_interc * x_range + alpha_interc
            ax.plot(x_range, y_interc, 'b--', lw=2.5,
                    label=f'Intercept: β={beta_interc:.3f}, α={alpha_interc:.2f}')

        if plot_type == 'both':
            # Theory line
            y_theory = 0.5 * x_range
            ax.plot(x_range, y_theory, 'k:', lw=1.5, alpha=0.5,
                    label='Theory: β=0.5')

        ax.set_xlabel('log(Q / V_daily)', fontsize=13)
        ax.set_ylabel('log(I / σ)', fontsize=13)

        title_map = {
            'raw': 'Raw scatter (no fit)',
            'origin': f'Origin fit: log(I/σ) = {beta_origin:.3f} × log(Q/V)',
            'intercept': f'Intercept fit: log(I/σ) = {alpha_interc:.2f} + {beta_interc:.3f} × log(Q/V)',
            'both': 'Origin vs Intercept comparison',
        }
        ax.set_title(f'{model_name} — {stock} ({version}) k=10 cross-η\n{title_map[plot_type]}',
                      fontsize=12)
        ax.legend(fontsize=9, markerscale=4, loc='upper left')

        fig.tight_layout()
        png_path = out_dir / f'{prefix}_scatter_{plot_type}.png'
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved: {png_path.name}')


def make_comparison_png(all_data, stock, version, out_dir):
    """One big comparison: all models side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = [m for m in MODELS_ORDER if m in all_data]
    if not models:
        return

    n = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flat

    for i, model in enumerate(models):
        ax = axes[i]
        pc_k10 = all_data[model]

        x = np.log(pc_k10['Q'].values / pc_k10['daily_vol'].values)
        y_adj = np.log(pc_k10['I'].values / pc_k10['daily_sigma'].values)
        ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
        x, y_adj = x[ok], y_adj[ok]
        vols = pc_k10['vol'].values[ok]

        if len(x) < 10:
            continue

        beta_origin = float(np.dot(x, y_adj) / np.dot(x, x))
        coeffs = np.polyfit(x, y_adj, 1)
        beta_interc = float(coeffs[0])
        alpha_interc = float(coeffs[1])

        # Color by η
        eta_arr = np.array([ETA_MAP.get((stock, int(v)), 0) for v in vols])
        for eta_val in sorted(set(eta_arr)):
            mask = eta_arr == eta_val
            c = ETA_COLORS.get(eta_val, '#888888')
            ax.scatter(x[mask], y_adj[mask], s=2, alpha=0.1, color=c, rasterized=True)

        x_range = np.linspace(x.min() - 0.3, x.max() + 0.3, 50)
        ax.plot(x_range, beta_origin * x_range, 'r-', lw=2,
                label=f'Origin β={beta_origin:.3f}')
        ax.plot(x_range, beta_interc * x_range + alpha_interc, 'b--', lw=2,
                label=f'Slope β={beta_interc:.3f}')
        ax.plot(x_range, 0.5 * x_range, 'k:', lw=1, alpha=0.5)

        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.set_xlabel('log(Q/V)')
        ax.set_ylabel('log(I/σ)')
        ax.legend(fontsize=8)

    # Hide unused subplot
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{stock} ({version}) — k=10 cross-η: Origin vs Intercept',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = out_dir / f'comparison_{stock}_{version}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {png_path.name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v6')
    parser.add_argument('--out_dir', default='pics_for_scatter_beta')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'Scatter plots — {args.stock} ({args.version})')
    all_k10_data = {}

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
        print(f'    k={k_max}, {len(pc_k10)} points')

        make_scatter_pngs(model, args.stock, pc_k10, out_dir, args.version)
        all_k10_data[model] = pc_k10
        gc.collect()

    make_comparison_png(all_k10_data, args.stock, args.version, out_dir)


if __name__ == '__main__':
    main()
