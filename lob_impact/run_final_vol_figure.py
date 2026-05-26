#!/usr/bin/env python3
"""
Final experiment: β(k) with 5 volatility estimators, 7 models.
Uses the SAME extract_point_cloud as analyze_beta_by_k.py (VWAP impact).

Top row: origin (no intercept).  Bottom row: intercept.

5 σ estimators:
  1. σ = 1 (no vol normalization)
  2. Parkinson σ
  3. Close-to-close σ
  4. High-Low range σ
  5. Realized σ_RV (per-sample, from midprice returns)
"""
import argparse, pickle, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, load_daily_params, collect_days,
)

MODELS_ORDER = ['Historic', 'Heuristic', 'CST', 'S5-4K', 'S5-4K-4000', 'S5-120M', 'S5-360M']
MODEL_COLORS = {
    'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
    'S5-4K': '#5B7BBF', 'S5-4K-4000': '#1B4F72', 'S5-120M': '#D95F02', 'S5-360M': '#B5446E',
}

SIGMA_METHODS = ['no_sigma', 'parkinson', 'close_to_close', 'hl_range', 'realized']
SIGMA_LABELS = {
    'no_sigma': r'$\sigma = 1$',
    'parkinson': r'Parkinson $\sigma$',
    'close_to_close': r'Close-to-Close $\sigma$',
    'hl_range': r'HL-Range $\sigma$',
    'realized': r'Realized $\sigma_{RV}$',
}


def compute_daily_sigmas(daily_hl_path):
    """Compute 5 σ estimators per day."""
    df = pd.read_csv(daily_hl_path)
    if 'day' not in df.columns:
        df['day'] = df['filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    df = df.sort_values('day').reset_index(drop=True)

    H = df['highest_price'].values.astype(float)
    L = df['lowest_price'].values.astype(float)
    mid = (H + L) / 2.0

    sig_park = np.log(H / L) / 1.6651092

    sig_cc = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        sig_cc[i] = abs(np.log(mid[i] / mid[i-1]))
    sig_cc[0] = sig_park[0]
    sig_cc[np.isnan(sig_cc) | (sig_cc < 1e-10)] = sig_park[np.isnan(sig_cc) | (sig_cc < 1e-10)]

    sig_hl = (H - L) / mid

    result = {}
    for i, row in df.iterrows():
        result[row['day']] = {
            'no_sigma': 1.0,
            'parkinson': float(sig_park[i]),
            'close_to_close': float(sig_cc[i]),
            'hl_range': float(sig_hl[i]),
        }
    return result


def fit_both(x, y):
    """Returns (beta_origin, beta_intercept, alpha)."""
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    xv, yv = x[ok], y[ok]
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, np.nan, n
    b_orig = float(np.dot(xv, yv) / np.dot(xv, xv))
    coeffs = np.polyfit(xv, yv, 1)
    b_int, alpha = float(coeffs[0]), float(coeffs[1])
    return b_orig, b_int, alpha, n


def compute_beta_by_k_5sigma(pc, daily_sigmas, k_max=100):
    """Per-k β(K): at each K, use ONLY points with k==K.

    Each K-th insertion is treated as a meta-order of size Q = K × child.
    Variation in x = log(Q/V) comes from 20 different days (different V_daily).
    Shows how β evolves along the execution trajectory:
      small K = mechanical (consuming best levels)
      large K = indirect impact (book reaction, latent liquidity)
    """
    # Pre-compute x and y arrays for all σ methods
    x_all = np.log(pc['Q'].values / pc['daily_vol'].values)
    k_arr = pc['k'].values

    y_all = {}
    for method in SIGMA_METHODS:
        if method == 'no_sigma':
            y_all[method] = np.log(pc['I'].values)
        elif method == 'parkinson':
            y_all[method] = np.log(pc['I'].values / pc['daily_sigma'].values)
        elif method == 'realized':
            sigma_rv = pc['sigma_rv'].values if 'sigma_rv' in pc.columns else pc['daily_sigma'].values
            sigma_rv = np.where(sigma_rv > 1e-15, sigma_rv, pc['daily_sigma'].values)
            y_all[method] = np.log(pc['I'].values / sigma_rv)
        else:
            days = pc['day'].values if 'day' in pc.columns else [None]*len(pc)
            sigma_arr = np.array([
                daily_sigmas.get(d, {}).get(method, 1.0) if d else 1.0
                for d in days
            ])
            sigma_arr = np.where(sigma_arr > 1e-15, sigma_arr, 1.0)
            y_all[method] = np.log(pc['I'].values / sigma_arr)

    results = []
    for K in range(1, k_max + 1):
        mask = k_arr == K
        if mask.sum() < 10:
            continue
        x_k = x_all[mask]

        for method in SIGMA_METHODS:
            y_k = y_all[method][mask]
            b_orig, b_int, alpha, n = fit_both(x_k, y_k)
            results.append(dict(
                k=K, sigma_method=method,
                beta_origin=b_orig, beta_intercept=b_int, alpha=alpha, n=n))

    return pd.DataFrame(results)


def make_figure(all_bk, stock, out_dir, dpi=200, stats=None):
    """4×5 grid: 5 σ columns, 4 rows (origin full, origin zoom, intercept full, intercept zoom)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 5, figsize=(30, 20), dpi=dpi)

    # Stats info line
    if stats:
        info = (f"child={stats['child']}, mb={stats['mb']}, i={stats['i']}, "
                f"n_days={stats['n_days']}, n_samples={stats['n_samples']}, "
                f"η≈{stats['eta']:.1f}%")
    else:
        info = ""

    fig.suptitle(f'{stock} — β(k) at fixed k, K=1..100 — 5 σ methods × 2 estimators × 2 scales\n'
                 f'Rows 1-2: Origin (no intercept). Rows 3-4: Intercept. Even rows: zoomed.\n'
                 f'{info}',
                 fontsize=14, fontweight='bold', y=0.99)

    for col_idx, method in enumerate(SIGMA_METHODS):
        ax_orig_full = axes[0, col_idx]
        ax_orig_zoom = axes[1, col_idx]
        ax_int_full  = axes[2, col_idx]
        ax_int_zoom  = axes[3, col_idx]

        for model_name, bk_df in all_bk.items():
            if bk_df.empty:
                continue
            bk = bk_df[bk_df['sigma_method'] == method].sort_values('k')
            if bk.empty:
                continue

            color = MODEL_COLORS.get(model_name, '#333')
            ls = '-' if 'S5' in model_name else ('--' if model_name == 'CST' else ':')
            lw = 2.5 if 'S5' in model_name else 1.8
            b_final_o = bk['beta_origin'].iloc[-1] if len(bk) > 0 else 0
            b_final_i = bk['beta_intercept'].iloc[-1] if len(bk) > 0 else 0
            n_final = int(bk['n'].iloc[-1]) if len(bk) > 0 else 0

            for ax in [ax_orig_full, ax_orig_zoom]:
                ax.plot(bk['k'], bk['beta_origin'], color=color, ls=ls, lw=lw,
                        label=f'{model_name} ({b_final_o:.2f}, n={n_final})', alpha=0.85)
            for ax in [ax_int_full, ax_int_zoom]:
                ax.plot(bk['k'], bk['beta_intercept'], color=color, ls=ls, lw=lw,
                        label=f'{model_name} ({b_final_i:.2f}, n={n_final})', alpha=0.85)

        for ax in [ax_orig_full, ax_orig_zoom, ax_int_full, ax_int_zoom]:
            ax.axhline(0.5, color='red', ls='--', lw=1.5, alpha=0.7)
            ax.axhspan(0.45, 0.55, alpha=0.06, color='red')
            ax.set_xlim(0, 102)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='best')

            # Stats box in each panel
            if stats:
                stat_text = (f"child={stats['child']} sh, mb={stats['mb']} msgs\n"
                             f"Q_total={stats['i']}×{stats['child']}={stats['i']*stats['child']}\n"
                             f"total_msgs={stats['i']}×{stats['mb']}={stats['i']*stats['mb']}\n"
                             f"{stats['n_days']} days, {stats['n_samples']} samples\n"
                             f"η≈{stats['eta']:.1f}%")
                ax.text(0.98, 0.02, stat_text, transform=ax.transAxes,
                        fontsize=5.5, ha='right', va='bottom', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

        # Full scale
        ax_orig_full.set_ylim(0, 1.0)
        ax_int_full.set_ylim(0, 1.0)
        # Zoomed
        ax_orig_zoom.set_ylim(0.25, 0.65)
        ax_int_zoom.set_ylim(0.25, 0.65)

        ax_orig_full.set_title(SIGMA_LABELS[method], fontsize=11, fontweight='bold')
        ax_int_zoom.set_xlabel('K', fontsize=10)

    axes[0, 0].set_ylabel(r'$\beta_{origin}$', fontsize=11)
    axes[1, 0].set_ylabel(r'$\beta_{origin}$ (zoom)', fontsize=11)
    axes[2, 0].set_ylabel(r'$\beta_{intercept}$', fontsize=11)
    axes[3, 0].set_ylabel(r'$\beta_{intercept}$ (zoom)', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(out_dir) / f'beta_k_4x5_{stock}.png'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='β(k) with 5 σ methods')
    parser.add_argument('--stock', type=str, default='NFLX')
    parser.add_argument('--pickle_base', type=str,
                        default='/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/final_experiment/pickles')
    parser.add_argument('--daily_hl', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    stock = args.stock.upper()
    daily_hl = args.daily_hl or f'lob_impact/daily_h_l_{stock}.csv'
    out_dir = args.out_dir or f'pics_for_final_{stock}'
    Path(out_dir).mkdir(exist_ok=True)

    # Daily sigmas (for CC and HL methods)
    daily_sigmas = compute_daily_sigmas(daily_hl)

    # Daily params for extract_point_cloud (Parkinson)
    pkl_base = Path(args.pickle_base) / stock

    all_bk = OrderedDict()

    for model in MODELS_ORDER:
        pkl = pkl_base / f'{model}.pkl'
        if not pkl.exists():
            print(f'  [skip] {model}')
            continue
        print(f'Loading {model} ({pkl.stat().st_size/1e9:.1f} GB)...')
        with open(pkl, 'rb') as f:
            md = pickle.load(f)

        # Use the SAME pipeline as analyze_beta_by_k.py
        filtered, n_total, n_skip = filter_model(md)
        print(f'  Filtered: {n_skip}/{n_total}')

        exp_days = set()
        for fd in md['data'].values():
            for d in fd['buy']['days'] + fd['sell']['days']:
                if d: exp_days.add(d)
        daily_params = load_daily_params(daily_hl, exp_days)

        pc = extract_point_cloud(filtered, md['grid'], daily_params)

        # Collect stats from first model
        if not all_bk:  # first model
            grid = md['grid']
            child_val = int(grid['child'].iloc[0]) if 'child' in grid else 0
            mb_val = int(grid['mb'].iloc[0]) if 'mb' in grid else 0
            i_val = int(grid['i'].iloc[0]) if 'i' in grid else 0
            n_days = pc['day'].nunique() if 'day' in pc.columns else 0
            n_samples = n_total
            # η = child / (child + mb × trade_frac × trade_p50)
            # approximate: just report
            fig_stats = dict(
                child=child_val, mb=mb_val, i=i_val,
                n_days=n_days, n_samples=n_samples,
                eta=10.0,  # calibrated target
            )

        del md; gc.collect()

        if pc.empty:
            print(f'  {model}: empty point cloud')
            continue

        print(f'  {model}: {len(pc)} points, k_max={int(pc["k"].max())}')

        # Compute β(k) with all 5 σ methods
        bk = compute_beta_by_k_5sigma(pc, daily_sigmas, k_max=int(pc['k'].max()))
        all_bk[model] = bk
        del pc; gc.collect()

    if not all_bk:
        print('ERROR: no data')
        sys.exit(1)

    # Print summary table
    print(f'\n{"="*90}')
    print(f'  {stock} — β at k=100 (5 σ methods × origin / intercept)')
    print(f'{"="*90}')
    header = f'  {"Model":<12s}'
    for method in SIGMA_METHODS:
        short = {'no_sigma': 'σ=1', 'parkinson': 'Park', 'close_to_close': 'CC',
                 'hl_range': 'HL', 'realized': 'RV'}[method]
        header += f'  {short+"_o":>7s} {short+"_i":>7s}'
    print(header)
    print(f'  {"-"*86}')
    for model_name, bk in all_bk.items():
        row_str = f'  {model_name:<12s}'
        for method in SIGMA_METHODS:
            bk_m = bk[(bk['sigma_method'] == method) & (bk['k'] == bk['k'].max())]
            if not bk_m.empty:
                row_str += f'  {bk_m.iloc[0]["beta_origin"]:>7.4f} {bk_m.iloc[0]["beta_intercept"]:>7.4f}'
            else:
                row_str += f'  {"—":>7s} {"—":>7s}'
        print(row_str)

    # Save CSV
    all_rows = []
    for model_name, bk in all_bk.items():
        bk = bk.copy()
        bk['model'] = model_name
        bk['stock'] = stock
        all_rows.append(bk)
    df = pd.concat(all_rows, ignore_index=True)
    csv_path = Path(out_dir) / f'beta_k_5sigma_{stock}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # Generate figure
    make_figure(all_bk, stock, out_dir, dpi=args.dpi, stats=fig_stats)
    print('Done!')


if __name__ == '__main__':
    main()
