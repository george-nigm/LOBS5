#!/usr/bin/env python3
"""
Single-config β analysis: treat each sample as one complete metaorder.

Uses only the LAST insertion (k=max) as the impact measurement point.
Q is fixed; V varies by day → log(I) = α + β·log(Q/V).

Usage:
    python lob_impact/analyze_single_config.py \
        --pickle_base /path/to/pickles --model S5-4K --stock AAPL \
        --daily_hl lob_impact/daily_h_l_AAPL.csv \
        --out_dir pics_for_v5_single_AAPL
"""
import argparse, pickle, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    get_midprice, filter_model, extract_point_cloud,
    compute_global_beta, load_daily_params, collect_days,
    compute_master_curve, compute_combined_impact,
    N_BOOTSTRAP, TICK_SIZE,
)


def bootstrap_beta_simple(pc_df, n_boot=N_BOOTSTRAP):
    """Bootstrap β using sample-level resampling on k=last points."""
    if 'daily_vol' not in pc_df.columns or len(pc_df) < 10:
        return np.full(n_boot, np.nan)
    x = np.log(pc_df['Q'].values / pc_df['daily_vol'].values)
    y = np.log(pc_df['I'].values)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return np.full(n_boot, np.nan)
    betas = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        A = np.column_stack([x[idx], np.ones(n)])
        try:
            coeffs = np.linalg.lstsq(A, y[idx], rcond=None)[0]
            betas.append(coeffs[0])
        except:
            betas.append(np.nan)
    return np.array(betas)


def main():
    parser = argparse.ArgumentParser(description='Single-config β analysis (last insertion)')
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--model', required=True, nargs='+', help='Model name(s)')
    parser.add_argument('--stock', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--q_filter', type=int, default=None, help='Keep only folders with this Q')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle_base = Path(args.pickle_base)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

    COLORS = {
        'Historic': '#1f77b4', 'Heuristic': '#ff7f0e', 'CST': '#2ca02c',
        'S5-4K': '#d62728', 'S5-120M': '#9467bd', 'S5-360M': '#8c564b',
        'LobS5': '#e377c2', 'LobS5-v2': '#7f7f7f', 'CGAN': '#bcbd22',
    }

    all_results = {}

    for model in args.model:
        raw_pkl = pickle_base / args.stock / f'{model}.pkl'
        if not raw_pkl.exists():
            print(f'SKIP {model}: {raw_pkl} not found')
            continue

        print(f'\n{"="*60}')
        print(f'  {model} / {args.stock}')
        print(f'{"="*60}')
        print(f'Loading {raw_pkl} ({raw_pkl.stat().st_size/1e9:.1f} GB)...')

        with open(raw_pkl, 'rb') as f:
            md = pickle.load(f)

        grid_df = md['grid']

        # Optional Q filter
        if args.q_filter:
            q_tag = f'Q{args.q_filter}'
            mask = grid_df['folder'].str.contains(q_tag)
            grid_df = grid_df[mask].reset_index(drop=True)
            md['data'] = {k: v for k, v in md['data'].items() if q_tag in k}
            print(f'  Filtered to {q_tag}: {len(grid_df)} folders')

        # Daily params
        exp_days = collect_days(md)
        daily_params = load_daily_params(args.daily_hl, exp_days)
        print(f'  Daily params: {len(daily_params)} days')

        # Filter outliers
        filtered, n_total, n_skip = filter_model(md)
        print(f'  Filtered: {n_skip}/{n_total} outliers')

        # Extract point cloud (all k)
        print(f'  Extracting point cloud...')
        pc = extract_point_cloud(filtered, grid_df, daily_params)
        print(f'  Total points: {len(pc)} (k=1..{pc["k"].max() if len(pc) else 0})')

        if pc.empty:
            print(f'  No valid points for {model}')
            continue

        k_max = int(pc['k'].max())

        # Filter to LAST insertion only (full metaorder)
        pc_last = pc[pc['k'] == k_max].copy()
        print(f'  Points at k={k_max} (full metaorder): {len(pc_last)}')

        # ── Compute CUMULATIVE MIDPRICE impact ──
        # mid_before_k at k=1 is arrival price, at k=k_max is midprice before last insertion
        # Group by sample_id, get mid at k=1 and k=k_max
        pc_k1 = pc[pc['k'] == 1][['sample_id', 'mid_before_k', 'direction']].copy()
        pc_k1 = pc_k1.rename(columns={'mid_before_k': 'mid_arrival'})
        pc_kmax = pc[pc['k'] == k_max][['sample_id', 'mid_before_k', 'direction']].copy()
        pc_kmax = pc_kmax.rename(columns={'mid_before_k': 'mid_final'})
        pc_mid = pc_kmax.merge(pc_k1[['sample_id', 'mid_arrival']], on='sample_id', how='inner')
        pc_mid['I_mid_cum'] = np.where(
            pc_mid['direction'] == 'buy',
            (pc_mid['mid_final'] - pc_mid['mid_arrival']) / pc_mid['mid_arrival'],
            (pc_mid['mid_arrival'] - pc_mid['mid_final']) / pc_mid['mid_arrival']
        )
        pc_mid['I_mid_cum'] = pc_mid['I_mid_cum'].abs()
        pc_mid = pc_mid[pc_mid['I_mid_cum'] > 1e-12]

        # Add Q and daily params from pc_last
        merge_cols = ['sample_id', 'Q']
        if 'daily_vol' in pc_last.columns:
            merge_cols += ['daily_vol', 'daily_sigma']
        pc_mid = pc_mid.merge(pc_last[merge_cols], on='sample_id', how='inner')

        print(f'  Midprice cumulative impact: {len(pc_mid)} points, mean={pc_mid["I_mid_cum"].mean():.6f}')

        # Compute β (VWAP — for reference)
        res = compute_global_beta(pc_last)
        boots = bootstrap_beta_simple(pc_last)
        ci_lo, ci_hi = np.nanpercentile(boots, [2.5, 97.5])

        print(f'  VWAP:  β={res["beta_intercept"]:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]  R²={res["r2_intercept"]:.4f}  N={res["n"]}')

        # Compute β (MIDPRICE CUMULATIVE — primary metric)
        res_mid = {'beta_intercept': np.nan, 'r2_intercept': np.nan, 'n': 0, 'alpha': np.nan}
        boots_mid = np.full(N_BOOTSTRAP, np.nan)
        ci_lo_mid, ci_hi_mid = np.nan, np.nan
        if len(pc_mid) > 10 and 'daily_vol' in pc_mid.columns:
            x = np.log(pc_mid['Q'].values / pc_mid['daily_vol'].values)
            y = np.log(pc_mid['I_mid_cum'].values)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) > 10:
                A = np.column_stack([x, np.ones(len(x))])
                coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
                y_pred = A @ coeffs
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - y.mean())**2)
                res_mid = {
                    'beta_intercept': coeffs[0], 'alpha': coeffs[1],
                    'r2_intercept': 1 - ss_res/ss_tot if ss_tot > 0 else 0,
                    'n': len(x),
                }
                # Bootstrap
                pc_mid_boot = pc_mid.copy()
                pc_mid_boot['_I'] = pc_mid_boot['I_mid_cum']
                pc_mid_boot['_Q'] = pc_mid_boot['Q']
                pc_mid_boot['_V'] = pc_mid_boot['daily_vol']
                boots_mid = bootstrap_beta_simple(
                    pd.DataFrame({'I': pc_mid_boot['_I'], 'Q': pc_mid_boot['_Q'], 'daily_vol': pc_mid_boot['_V']})
                )
                ci_lo_mid, ci_hi_mid = np.nanpercentile(boots_mid, [2.5, 97.5])

        print(f'  MID:   β={res_mid["beta_intercept"]:.4f}  CI=[{ci_lo_mid:.4f}, {ci_hi_mid:.4f}]  R²={res_mid["r2_intercept"]:.4f}  N={res_mid["n"]}')

        # Kyle λ per insertion
        kyle = {}
        for k in range(1, k_max + 1):
            pk = pc[pc['k'] == k]
            if len(pk) == 0:
                continue
            valid = pk['mid_before_k'].values > 0
            if valid.sum() == 0:
                continue
            lam = np.abs(pk.loc[valid, 'exec_price_k'].values - pk.loc[valid, 'mid_before_k'].values) / (TICK_SIZE * pk.loc[valid, 'size_k'].values)
            kyle[k] = {'mean': float(np.nanmean(lam)), 'std': float(np.nanstd(lam)),
                        'median': float(np.nanmedian(lam)), 'n': int(valid.sum())}
        print(f'  Kyle λ: k=1 mean={kyle.get(1,{}).get("mean",0):.4f}, k={k_max} mean={kyle.get(k_max,{}).get("mean",0):.4f}')

        # Master curve
        mcs = {}
        for folder in grid_df['folder']:
            if folder not in filtered:
                continue
            fd = filtered[folder]
            i_val = int(grid_df[grid_df['folder'] == folder]['i'].iloc[0])
            mb_val = int(grid_df[grid_df['folder'] == folder]['mb'].iloc[0])
            mc = compute_master_curve(fd['buy']['books'], fd['sell']['books'], i_val, mb_val)
            if mc is not None:
                mcs[folder] = mc

        # Mean impact at k=last
        mean_I = float(pc_last['I'].mean()) if len(pc_last) else 0
        mean_I_mid = float(pc_mid['I_mid_cum'].mean()) if len(pc_mid) else 0

        all_results[model] = dict(
            res=res, boots=boots, ci_lo=ci_lo, ci_hi=ci_hi,
            res_mid=res_mid, boots_mid=boots_mid, ci_lo_mid=ci_lo_mid, ci_hi_mid=ci_hi_mid,
            pc_last=pc_last, pc_mid=pc_mid, pc_all=pc, kyle=kyle, mcs=mcs,
            k_max=k_max, mean_I=mean_I, mean_I_mid=mean_I_mid,
        )

    if not all_results:
        print('No results to plot')
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════
    models_done = list(all_results.keys())

    # ── Fig 1: Scatter — VWAP vs Midprice (2 rows) ──
    n_models = len(models_done)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 9), squeeze=False)
    for idx, model in enumerate(models_done):
        r = all_results[model]
        c = COLORS.get(model, 'C0')
        # Row 0: VWAP impact
        ax = axes[0][idx]
        pc = r['pc_last']
        if 'daily_vol' in pc.columns and len(pc) > 2:
            x = np.log(pc['Q'].values / pc['daily_vol'].values)
            y = np.log(pc['I'].values)
            valid = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[valid], y[valid]
            ax.scatter(xv, yv, alpha=0.05, s=3, color=c)
            if len(xv) > 2:
                A = np.column_stack([xv, np.ones(len(xv))])
                coeffs = np.linalg.lstsq(A, yv, rcond=None)[0]
                xl = np.linspace(xv.min(), xv.max(), 100)
                ax.plot(xl, coeffs[0]*xl + coeffs[1], 'k-', lw=2)
        beta = r['res']['beta_intercept']
        ax.set_title(f'{model}\nVWAP β={beta:.3f}', fontsize=10)
        ax.set_ylabel('log(I_vwap)')
        # Row 1: Midprice cumulative impact
        ax = axes[1][idx]
        pc_m = r['pc_mid']
        if 'daily_vol' in pc_m.columns and len(pc_m) > 2:
            x = np.log(pc_m['Q'].values / pc_m['daily_vol'].values)
            y = np.log(pc_m['I_mid_cum'].values)
            valid = np.isfinite(x) & np.isfinite(y)
            xm, ym = x[valid], y[valid]
            ax.scatter(xm, ym, alpha=0.05, s=3, color=c)
            if len(xm) > 2:
                A = np.column_stack([xm, np.ones(len(xm))])
                coeffs = np.linalg.lstsq(A, ym, rcond=None)[0]
                xl = np.linspace(xm.min(), xm.max(), 100)
                ax.plot(xl, coeffs[0]*xl + coeffs[1], 'k-', lw=2)
        beta_m = r['res_mid']['beta_intercept']
        ax.set_title(f'MID β={beta_m:.3f}', fontsize=10)
        ax.set_xlabel('log(Q/V)')
        ax.set_ylabel('log(I_mid_cum)')
    fig.suptitle(f'{args.stock} — VWAP vs Midprice Impact (k={all_results[models_done[0]]["k_max"]})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / '1. Scatter VWAP vs Midprice.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: 1. Scatter VWAP vs Midprice')

    # ── Fig 2: Average Master Curve (all models on one plot) ──
    fig, ax = plt.subplots(figsize=(11, 5))
    for model in models_done:
        mcs = all_results[model]['mcs']
        if not mcs:
            continue
        all_m = [mc['mean'] for mc in mcs.values()]
        u = list(mcs.values())[0]['u']
        avg = np.nanmean(all_m, axis=0)
        std = np.nanstd(all_m, axis=0)
        c = COLORS.get(model, 'gray')
        ax.plot(u, avg, color=c, lw=2.5, label=model)
        ax.fill_between(u, avg - std, avg + std, color=c, alpha=0.1)
    ax.axvline(1.0, ls='--', color='gray', lw=1)
    ax.set(xlabel='u (volume-time)', ylabel='Normalised impact', title=f'{args.stock} — Average Master Curve')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / '2. Average Master Curve.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: 2. Average Master Curve')

    # ── Fig 3: Impact distribution at k=last ──
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    for idx, model in enumerate(models_done):
        ax = axes[0][idx]
        pc = all_results[model]['pc_last']
        I_vals = pc['I'].values
        I_vals = I_vals[np.isfinite(I_vals) & (I_vals > 0)]
        if len(I_vals) > 0:
            ax.hist(np.log10(I_vals), bins=50, color=COLORS.get(model, 'C0'), alpha=0.7, edgecolor='black', lw=0.3)
        ax.set_title(f'{model}\nmean={np.mean(I_vals):.2e}', fontsize=10)
        ax.set_xlabel('log10(I)')
        ax.set_ylabel('count')
    fig.suptitle(f'{args.stock} — Impact Distribution (full metaorder, k={all_results[models_done[0]]["k_max"]})', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / '3. Impact Distribution.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: 3. Impact Distribution')

    # ── Fig 4: Kyle λ(k) per insertion ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models_done:
        kyle = all_results[model]['kyle']
        ks = sorted(kyle.keys())
        means = [kyle[k]['mean'] for k in ks]
        ax.plot(ks, means, 'o-', color=COLORS.get(model, 'gray'), lw=2, markersize=6, label=model)
    ax.set(xlabel='Insertion k', ylabel='Kyle λ (ticks/share)',
           title=f'{args.stock} — Kyle λ per Insertion')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / '4. Kyle Lambda per Insertion.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: 4. Kyle Lambda')

    # ── Fig 5: β comparison bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    names = models_done
    betas = [all_results[m]['res']['beta_intercept'] for m in names]
    ci_los = [all_results[m]['ci_lo'] for m in names]
    ci_his = [all_results[m]['ci_hi'] for m in names]
    colors = [COLORS.get(m, 'gray') for m in names]
    errs = [[b - lo for b, lo in zip(betas, ci_los)], [hi - b for b, hi in zip(betas, ci_his)]]
    ax.bar(range(len(names)), betas, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
    ax.errorbar(range(len(names)), betas, yerr=errs, fmt='none', ecolor='black', capsize=5)
    ax.axhline(0.5, ls='--', color='red', lw=1.5, label='Theory β=0.5')
    ax.axhline(0, ls='-', color='gray', lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set(ylabel='β (intercept estimator)', title=f'{args.stock} — β at Full Metaorder (k=last)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / '5. Beta Comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: 5. Beta Comparison')

    # ── Summary table ──
    print(f'\n{"="*80}')
    print(f'  SUMMARY: {args.stock}')
    print(f'{"="*80}')
    print(f'{"Model":>12s}  {"β_vwap":>7s}  {"β_mid":>7s}  {"CI_lo":>7s}  {"CI_hi":>7s}  {"R²_mid":>7s}  {"N":>6s}  {"mean_I_mid":>10s}  {"λ(1)":>7s}  {"λ(k)":>7s}')
    for m in models_done:
        r = all_results[m]
        k_max = r['k_max']
        print(f'{m:>12s}  {r["res"]["beta_intercept"]:>7.3f}  {r["res_mid"]["beta_intercept"]:>7.3f}  '
              f'{r["ci_lo_mid"]:>7.3f}  {r["ci_hi_mid"]:>7.3f}  '
              f'{r["res_mid"]["r2_intercept"]:>7.4f}  {r["res_mid"]["n"]:>6d}  {r["mean_I_mid"]:>10.2e}  '
              f'{r["kyle"].get(1,{}).get("mean",0):>7.4f}  {r["kyle"].get(k_max,{}).get("mean",0):>7.4f}')

    print(f'\nFigures saved to: {out_dir}/')


if __name__ == '__main__':
    main()
