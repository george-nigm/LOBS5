#!/usr/bin/env python3
"""
Volatility sweep at fixed η — how does σ choice affect β?

For each σ method × model: compute sliding β(K) at fixed η from V7 data.

Usage:
    python lob_impact/analyze_vol_sweep_fixed_eta.py --stock AAPL \
        --pickle_base /path/to/pickles --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v7 --target_eta 9 --out_dir pics_for_vol_sweep_eta9
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
COLORS = {
    'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
    'S5-120M': '#D95F02', 'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E',
}

ETA_TO_VOL = {
    ('AAPL', 9, 'v7'): 120, ('AMZN', 9, 'v7'): 60,
    ('AAPL', 50, 'v7'): 1170, ('AMZN', 50, 'v7'): 570,
    ('AAPL', 90, 'v7'): 10710, ('AMZN', 90, 'v7'): 5310,
}

SIGMA_METHODS = ['parkinson', 'no_sigma', 'realized_vol']
SIGMA_LABELS = {'parkinson': 'Parkinson σ', 'no_sigma': 'No σ (σ=1)', 'realized_vol': 'Realized σ_RV'}


def fit_intercept(x, y):
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    xv, yv = x[ok], y[ok]
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, n
    coeffs = np.polyfit(xv, yv, 1)
    return float(coeffs[0]), float(coeffs[1]), n


def get_y(pc, method):
    """Compute log(I/σ) for given σ method."""
    I = pc['I'].values
    if method == 'parkinson':
        sigma = pc['daily_sigma'].values
    elif method == 'no_sigma':
        sigma = np.ones(len(pc))
    elif method == 'realized_vol':
        sigma = pc['sigma_rv'].values if 'sigma_rv' in pc.columns else pc['daily_sigma'].values
    else:
        sigma = pc['daily_sigma'].values
    return np.log(np.maximum(I, 1e-20) / np.maximum(sigma, 1e-20))


def analyze_model(pkl_path, daily_hl_path, model_name, stock, target_vol, version):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
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

    if pc.empty:
        return []

    pc_eta = pc[pc['vol'] == target_vol].copy()
    if pc_eta.empty:
        return []

    x = np.log(pc_eta['Q'].values / pc_eta['daily_vol'].values)
    k_max = int(pc_eta['k'].max())
    k_arr = pc_eta['k'].values

    results = []
    for method in SIGMA_METHODS:
        y = get_y(pc_eta, method)

        # Sliding β(K)
        for K in range(3, k_max + 1):
            mask = k_arr <= K
            beta, alpha, n = fit_intercept(x[mask], y[mask])
            results.append(dict(
                model=model_name, stock=stock, sigma_method=method,
                K_max=K, beta_slope=beta, n=n,
            ))

        # Full pooled
        beta_all, alpha_all, n_all = fit_intercept(x, y)
        print(f'    {model_name} / {method}: β(all k) = {beta_all:+.4f}, n={n_all}')

    return results


def make_plots(df, stock, version, target_eta, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = [m for m in MODELS_ORDER if m in df['model'].unique()]

    # ── Plot 1: One panel per model, 3 σ lines ──
    n_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    sigma_styles = {'parkinson': ('-', 2.5), 'no_sigma': ('--', 2), 'realized_vol': (':', 2.5)}
    sigma_colors = {'parkinson': '#2166ac', 'no_sigma': '#b2182b', 'realized_vol': '#4daf4a'}

    for i, model in enumerate(models[:6]):
        ax = axes.flat[i]
        mdf = df[df['model'] == model]
        for method in SIGMA_METHODS:
            sdf = mdf[mdf['sigma_method'] == method].sort_values('K_max')
            ls, lw = sigma_styles[method]
            c = sigma_colors[method]
            beta_30 = sdf[sdf['K_max'] == sdf['K_max'].max()]['beta_slope'].iloc[0] if not sdf.empty else np.nan
            ax.plot(sdf['K_max'], sdf['beta_slope'], ls, color=c, lw=lw,
                    label=f'{SIGMA_LABELS[method]} (β₃₀={beta_30:.3f})')
        ax.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.5)
        ax.axhline(0, ls='-', color='gray', lw=0.5)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xlabel('K')
        ax.set_ylabel('β')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.35)

    for j in range(len(models), 6):
        axes.flat[j].set_visible(False)

    fig.suptitle(f'{stock} ({version}) — Volatility estimator comparison at η={target_eta}%\n'
                 f'β(K) sliding window, k=1..K pooled',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png1 = out_dir / f'vol_sweep_eta{target_eta}_{stock}_{version}.png'
    fig.savefig(png1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png1.name}')

    # ── Plot 2: Summary bar chart at K=30 ──
    fig, ax = plt.subplots(figsize=(14, 7))
    K_max = df['K_max'].max()
    final = df[df['K_max'] == K_max]
    x_pos = np.arange(len(models))
    width = 0.25

    for j, method in enumerate(SIGMA_METHODS):
        vals = [final[(final['model'] == m) & (final['sigma_method'] == method)]['beta_slope'].iloc[0]
                if not final[(final['model'] == m) & (final['sigma_method'] == method)].empty else 0
                for m in models]
        ax.bar(x_pos + j * width, vals, width, label=SIGMA_LABELS[method],
               color=sigma_colors[method], alpha=0.8)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('β at K=30', fontsize=13)
    ax.set_title(f'{stock} ({version}) — β at K=30 by σ method (η={target_eta}%)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='gray', lw=0.5)
    fig.tight_layout()
    png2 = out_dir / f'vol_sweep_bar_eta{target_eta}_{stock}_{version}.png'
    fig.savefig(png2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png2.name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v7')
    parser.add_argument('--target_eta', type=int, default=9)
    parser.add_argument('--out_dir', default='pics_for_vol_sweep_eta9')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_vol = ETA_TO_VOL.get((args.stock, args.target_eta, args.version))
    if target_vol is None:
        sys.exit(f'No vol mapping for ({args.stock}, η={args.target_eta}%, {args.version})')

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'Vol sweep at η={args.target_eta}% — {args.stock} ({args.version})')
    all_results = []
    for model in models:
        results = analyze_model(
            str(pkl_dir / f'{model}.pkl'), args.daily_hl, model, args.stock,
            target_vol, args.version)
        all_results.extend(results)
        gc.collect()

    df = pd.DataFrame(all_results)
    csv_path = out_dir / f'vol_sweep_eta{args.target_eta}_{args.stock}_{args.version}.csv'
    df.to_csv(csv_path, index=False)

    # Summary table at K=30
    K_max = df['K_max'].max()
    final = df[df['K_max'] == K_max]
    print(f'\n{"=" * 70}')
    print(f'  β at K={K_max}, η={args.target_eta}% — {args.stock}')
    print(f'{"=" * 70}')
    pivot = final.pivot(index='model', columns='sigma_method', values='beta_slope')
    pivot = pivot.reindex(models)[SIGMA_METHODS]
    print(pivot.round(3).to_string())

    make_plots(df, args.stock, args.version, args.target_eta, out_dir)


if __name__ == '__main__':
    main()
