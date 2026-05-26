#!/usr/bin/env python3
"""
β at fixed participation rate η, using k-variation from V7 data (i=30).

At fixed η, Q = k × Q_child for k=1..30. Pool all k to get Q variation.
Compute:
  1. β from all k pooled (one number per model)
  2. Sliding β: fit from k=1..K for K=3..30 (how β evolves as metaorder extends)
  3. Scatter plot: log(I/σ) vs log(Q/V), colored by k

Usage:
    python lob_impact/analyze_beta_fixed_eta.py --stock AAPL \
        --pickle_base /path/to/pickles --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --version v7 --target_eta 9 --out_dir pics_for_beta_fixed_eta
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

# η target → vol (Q_meta) mapping per version
# V7: vol = 30 × child.  V6: vol = 10 × child.
ETA_TO_VOL = {
    ('AAPL', 9, 'v7'): 120,   ('AAPL', 50, 'v7'): 1170,  ('AAPL', 90, 'v7'): 10710,
    ('AMZN', 9, 'v7'): 60,    ('AMZN', 50, 'v7'): 570,   ('AMZN', 90, 'v7'): 5310,
    ('AAPL', 9, 'v6'): 40,    ('AAPL', 50, 'v6'): 390,   ('AAPL', 90, 'v6'): 3570,
    ('AMZN', 9, 'v6'): 20,    ('AMZN', 50, 'v6'): 190,   ('AMZN', 90, 'v6'): 1770,
    # V8: realistic child = median MO, η≈10%, vol = 10 × child
    ('AAPL', 10, 'v8'): 350,  ('META', 10, 'v8'): 100,   ('MSFT', 10, 'v8'): 170,
    ('NVDA', 10, 'v8'): 250,  ('TSLA', 10, 'v8'): 150,
    # V9: i=100, same child sizes as V7. vol = 100 × child.
    ('AAPL', 9, 'v9'): 400,   ('AAPL', 50, 'v9'): 3900,  ('AAPL', 90, 'v9'): 35700,
}


def fit_intercept(x, y):
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


def analyze_model(pkl_path, daily_hl_path, model_name, stock, target_vol, version):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f'  SKIP {model_name}')
        return [], None

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
        return [], None

    # Filter to target η
    pc_eta = pc[pc['vol'] == target_vol].copy()
    if pc_eta.empty:
        print(f'    WARNING: no data for vol={target_vol}')
        # Try matching by child size instead
        child_map = {120: 4, 1170: 39, 10710: 357, 60: 2, 570: 19, 5310: 177,
                     40: 4, 390: 39, 3570: 357, 20: 2, 190: 19, 1770: 177}
        child = child_map.get(target_vol)
        if child and 'child' in pc.columns:
            pc_eta = pc[pc['child'] == child].copy()
        if pc_eta.empty:
            print(f'    No data after child filter either')
            return [], None

    k_max = int(pc_eta['k'].max())
    print(f'    η filter: {len(pc_eta)} points, k_max={k_max}')

    x_all = np.log(pc_eta['Q'].values / pc_eta['daily_vol'].values)
    y_all = np.log(pc_eta['I'].values / pc_eta['daily_sigma'].values)

    results = []

    # 1. Sliding window: β from k=1..K
    for K in range(3, k_max + 1):
        mask = pc_eta['k'].values <= K
        beta, alpha, r2, n = fit_intercept(x_all[mask], y_all[mask])
        results.append(dict(
            model=model_name, stock=stock, target_vol=target_vol,
            K_max=K, beta_slope=beta, alpha=alpha, r2=r2, n=n,
            type='sliding',
        ))

    # 2. Per-k β (using all k up to that point — cumulative)
    # Already done above. Also compute per-k-only for diagnostics:
    for k in range(1, k_max + 1):
        mask_k = pc_eta['k'].values == k
        n_k = mask_k.sum()
        I_med = float(np.median(pc_eta['I'].values[mask_k])) if n_k > 0 else np.nan
        Q_med = float(np.median(pc_eta['Q'].values[mask_k])) if n_k > 0 else np.nan
        results.append(dict(
            model=model_name, stock=stock, target_vol=target_vol,
            K_max=k, beta_slope=np.nan, alpha=np.nan, r2=np.nan, n=n_k,
            type='per_k_stats', I_median=I_med, Q_median=Q_med,
        ))

    return results, pc_eta


def make_plots(df, all_pc, stock, version, target_eta, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = [m for m in MODELS_ORDER if m in df['model'].unique()]
    sliding = df[df['type'] == 'sliding']

    # ── Plot 1: Sliding β(K) ──
    fig, ax = plt.subplots(figsize=(14, 8))
    for model in models:
        mdf = sliding[sliding['model'] == model].sort_values('K_max')
        c = COLORS.get(model, 'gray')
        ax.plot(mdf['K_max'], mdf['beta_slope'], 'o-', color=c, lw=2.5,
                markersize=5, label=model)
    ax.axhline(0.5, ls='--', color='red', lw=2, alpha=0.7, label='Theory β=0.5')
    ax.axhline(0, ls='-', color='gray', lw=0.5)
    ax.set_xlabel('Metaorder length K (insertions 1..K pooled)', fontsize=13)
    ax.set_ylabel('β (intercept estimator)', fontsize=13)
    ax.set_title(f'{stock} ({version}) — β at fixed η={target_eta}%\n'
                 f'Sliding window: fit from k=1..K',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, int(sliding['K_max'].max()) + 1, 5))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png1 = out_dir / f'beta_sliding_eta{target_eta}_{stock}_{version}.png'
    fig.savefig(png1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png1.name}')

    # ── Plot 2: Scatter per model (2×3 grid) ──
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    k_cmap = plt.cm.viridis

    for i, model in enumerate(models[:6]):
        ax = axes.flat[i]
        if model not in all_pc:
            ax.set_visible(False)
            continue

        pc = all_pc[model]
        x = np.log(pc['Q'].values / pc['daily_vol'].values)
        y = np.log(pc['I'].values / pc['daily_sigma'].values)
        k_vals = pc['k'].values
        k_max = int(k_vals.max())
        ok = np.isfinite(x) & np.isfinite(y) & (x != 0)

        sc = ax.scatter(x[ok], y[ok], c=k_vals[ok], cmap=k_cmap,
                        s=1.5, alpha=0.08, vmin=1, vmax=k_max, rasterized=True)

        # Fit line
        beta, alpha, r2, n = fit_intercept(x, y)
        x_range = np.linspace(x[ok].min() - 0.2, x[ok].max() + 0.2, 50)
        ax.plot(x_range, beta * x_range + alpha, 'r-', lw=2.5,
                label=f'β = {beta:.3f} (R²={r2:.3f})')
        ax.plot(x_range, 0.5 * x_range + alpha, 'k:', lw=1.5, alpha=0.5,
                label='β = 0.5')

        ax.set_title(f'{model} (β = {beta:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('log(Q/V)')
        ax.set_ylabel('log(I/σ)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    for j in range(len(models), 6):
        axes.flat[j].set_visible(False)

    fig.suptitle(f'{stock} ({version}) — Scatter at fixed η={target_eta}%\n'
                 f'Color = insertion number k (1..{k_max}), Q = k × Q_child',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(cmap=k_cmap, norm=plt.Normalize(1, k_max)),
                 cax=cbar_ax, label='k')
    png2 = out_dir / f'scatter_eta{target_eta}_{stock}_{version}.png'
    fig.savefig(png2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png2.name}')


def main():
    parser = argparse.ArgumentParser(description='β at fixed η')
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--version', default='v7')
    parser.add_argument('--target_eta', type=int, default=9)
    parser.add_argument('--out_dir', default='pics_for_beta_fixed_eta')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_vol = ETA_TO_VOL.get((args.stock, args.target_eta, args.version))
    if target_vol is None:
        print(f'ERROR: no vol mapping for ({args.stock}, η={args.target_eta}%, {args.version})')
        sys.exit(1)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'Fixed-η β — {args.stock} ({args.version}), η={args.target_eta}%, vol={target_vol}')
    all_results = []
    all_pc = {}

    for model in models:
        results, pc_eta = analyze_model(
            str(pkl_dir / f'{model}.pkl'), args.daily_hl, model, args.stock,
            target_vol, args.version)
        all_results.extend(results)
        if pc_eta is not None and not pc_eta.empty:
            all_pc[model] = pc_eta
        gc.collect()

    if not all_results:
        print('ERROR: no results')
        sys.exit(1)

    df = pd.DataFrame(all_results)
    csv_path = out_dir / f'beta_fixed_eta{args.target_eta}_{args.stock}_{args.version}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path} ({len(df)} rows)')

    # Print summary
    sliding = df[df['type'] == 'sliding']
    print(f'\n{"=" * 80}')
    print(f'  Sliding β(K) at η={args.target_eta}% — {args.stock} ({args.version})')
    print(f'{"=" * 80}')
    header = f'{"K":>4s}'
    for m in models:
        header += f'  {m:>12s}'
    print(header)
    print('-' * 80)
    for K in sorted(sliding['K_max'].unique()):
        if K % 5 != 0 and K != 3 and K != int(sliding['K_max'].max()):
            continue
        row = f'{int(K):>4d}'
        for m in models:
            val = sliding[(sliding['model'] == m) & (sliding['K_max'] == K)]['beta_slope']
            if not val.empty:
                row += f'  {float(val.iloc[0]):>+12.4f}'
            else:
                row += f'  {"—":>12s}'
        print(row)

    make_plots(df, all_pc, args.stock, args.version, args.target_eta, out_dir)


if __name__ == '__main__':
    main()
