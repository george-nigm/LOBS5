#!/usr/bin/env python3
"""
Metaorder β analysis: β computed from FULL metaorder only (k=k_max).

For each (model, stock, child_size) separately:
  - Filter point cloud to k=k_max (last insertion, metaorder complete)
  - Fit β: log(I) = α + β × log(Q/V)
  - Bootstrap distribution of β
  - Impact distributions

Output: PDF report + CSV summary per stock.

Usage:
    python lob_impact/analyze_metaorder_beta.py --stock AAPL \
        --pickle_base /path/to/pickles --daily_hl lob_impact/daily_h_l_AAPL.csv
"""
import argparse, pickle, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, compute_global_beta,
    bootstrap_beta, load_daily_params, collect_days,
)

# ═══════════════════════════════════════════════════════════════════════
# η mapping: (stock, child) → participation rate %
# ═══════════════════════════════════════════════════════════════════════
ETA_MAP = {
    ('AAPL', 2): 5, ('AAPL', 4): 9, ('AAPL', 9): 19,
    ('AAPL', 10): 22, ('AAPL', 40): 50, ('AAPL', 100): 72,
    ('AAPL', 400): 91, ('AAPL', 1000): 96,
    ('AMZN', 1): 5, ('AMZN', 2): 9, ('AMZN', 4): 17,
    ('AMZN', 7): 26, ('AMZN', 25): 56, ('AMZN', 100): 84,
    ('AMZN', 250): 93, ('AMZN', 1000): 98,
}

MODELS_ORDER = ['Historic', 'Heuristic', 'CST', 'S5-120M', 'S5-360M', 'S5-4K']

COLORS = {
    'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
    'S5-120M': '#D95F02', 'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E',
    'CGAN': '#7B4F9E', 'LobS5': '#C88A3A',
}


def analyze_one_model(pkl_path, daily_hl_path, model_name, stock):
    """Load pickle, extract point cloud, filter k=k_max, compute β per child."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f'  SKIP {model_name}: {pkl_path} not found')
        return [], {}

    print(f'\n  Loading {model_name} ({pkl_path.stat().st_size / 1e9:.1f} GB)...')
    with open(pkl_path, 'rb') as f:
        md = pickle.load(f)

    exp_days = collect_days(md)
    daily_params = load_daily_params(daily_hl_path, exp_days)
    filtered, n_total, n_skip = filter_model(md)
    print(f'  Filtered: {n_skip}/{n_total} outliers, {len(daily_params)} daily params')

    pc = extract_point_cloud(filtered, md['grid'], daily_params)
    # Free memory early
    del md, filtered
    gc.collect()

    if pc.empty:
        print(f'  WARNING: empty point cloud for {model_name}')
        return [], {}

    k_max = int(pc['k'].max())
    pc_last = pc[pc['k'] == k_max].copy()
    print(f'  k_max={k_max}, total points={len(pc)}, k={k_max} points={len(pc_last)}')

    vol_levels = sorted(pc_last['vol'].unique()) if 'vol' in pc_last.columns else []
    if not vol_levels:
        return [], {}

    results = []
    scatter_data = {}

    for vol in vol_levels:
        pc_vol = pc_last[pc_last['vol'] == vol].copy()
        child = int(vol)
        eta = ETA_MAP.get((stock, child), None)
        Q_total = child * k_max

        if len(pc_vol) < 20:
            print(f'    child={child}: only {len(pc_vol)} points, skip')
            continue

        # β (intercept estimator, primary)
        res = compute_global_beta(pc_vol)
        boot = bootstrap_beta(pc_vol, n_boot=1000)
        boots = boot.get('boots', np.array([]))
        ci = (np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)) \
            if len(boots) > 10 else (np.nan, np.nan)

        # β (midprice)
        pc_mid = pc_vol[pc_vol['I_mid'].notna() & (pc_vol['I_mid'] > 1e-12)].copy()
        res_mid = compute_global_beta(pc_mid, impact_col='I_mid') if len(pc_mid) > 20 else {}
        boot_mid = bootstrap_beta(pc_mid, n_boot=1000, impact_col='I_mid') if len(pc_mid) > 20 else {}
        boots_mid = boot_mid.get('boots', np.array([]))

        n_buy = len(pc_vol[pc_vol['direction'] == 'buy'])
        n_sell = len(pc_vol[pc_vol['direction'] == 'sell'])

        # Unique days
        n_days = pc_vol['day'].nunique() if 'day' in pc_vol.columns else 0

        results.append(dict(
            model=model_name,
            stock=stock,
            child=child,
            Q_total=Q_total,
            k_max=k_max,
            eta=eta,
            beta=res.get('beta', np.nan),
            beta_origin=res.get('beta_origin', np.nan),
            alpha=res.get('alpha', np.nan),
            ci_lo=ci[0],
            ci_hi=ci[1],
            r2=res.get('r2', np.nan),
            n=res.get('n', 0),
            n_buy=n_buy,
            n_sell=n_sell,
            n_days=n_days,
            beta_mid=res_mid.get('beta', np.nan) if res_mid else np.nan,
            I_median=float(np.median(pc_vol['I'])),
            I_mean=float(np.mean(pc_vol['I'])),
            I_std=float(np.std(pc_vol['I'])),
        ))

        # Store scatter data for plotting
        scatter_data[child] = dict(
            log_QV=np.log(pc_vol['Q'].values / pc_vol['daily_vol'].values)
                if 'daily_vol' in pc_vol.columns else np.array([]),
            log_I=np.log(pc_vol['I'].values),
            I_raw=pc_vol['I'].values.copy(),
            direction=pc_vol['direction'].values.copy(),
            boots=boots.copy(),
            boots_mid=boots_mid.copy() if len(boots_mid) > 0 else np.array([]),
            beta=res.get('beta', np.nan),
            alpha=res.get('alpha', np.nan),
        )

        print(f'    child={child} (η≈{eta}%): β={res.get("beta", np.nan):.4f} '
              f'[{ci[0]:.3f}, {ci[1]:.3f}]  R²={res.get("r2", np.nan):.4f}  '
              f'N={res.get("n", 0)}  I_med={np.median(pc_vol["I"]):.6f}')

    return results, scatter_data


def make_pdf(all_results, all_scatter, stock, out_dir):
    """Generate comprehensive PDF report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.grid': True, 'grid.alpha': 0.3,
    })

    df = pd.DataFrame(all_results)
    if df.empty:
        print('  No results to plot')
        return

    pdf_path = out_dir / f'metaorder_beta_{stock}.pdf'
    csv_path = out_dir / f'metaorder_beta_{stock}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  Saved CSV: {csv_path} ({len(df)} rows)')

    child_sizes = sorted(df['child'].unique())
    models = [m for m in MODELS_ORDER if m in df['model'].unique()]

    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1+: Scatter plots per child size ──
        for child in child_sizes:
            eta = ETA_MAP.get((stock, child), '?')
            Q_total = child * 10

            fig, ax = plt.subplots(figsize=(10, 7))
            for model in models:
                sd = all_scatter.get(model, {}).get(child)
                if sd is None or len(sd['log_QV']) == 0:
                    continue
                c = COLORS.get(model, 'gray')
                x, y = sd['log_QV'], sd['log_I']
                ok = np.isfinite(x) & np.isfinite(y)
                ax.scatter(x[ok], y[ok], s=4, alpha=0.15, color=c, label=model, rasterized=True)
                # Regression line
                if np.isfinite(sd['beta']) and np.isfinite(sd['alpha']):
                    xr = np.linspace(np.min(x[ok]), np.max(x[ok]), 50)
                    ax.plot(xr, sd['beta'] * xr + sd['alpha'], '-', color=c, lw=2)

            ax.set_xlabel('log(Q / V_daily)')
            ax.set_ylabel('log(I_vwap)')
            ax.set_title(f'{stock} — child={child}, η≈{eta}%, Q={Q_total} (k=10 only)')
            ax.legend(fontsize=8, markerscale=3)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ── Bootstrap β distributions per child ──
        for child in child_sizes:
            eta = ETA_MAP.get((stock, child), '?')

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # VWAP β
            ax = axes[0]
            for model in models:
                sd = all_scatter.get(model, {}).get(child)
                if sd is None or len(sd['boots']) == 0:
                    continue
                c = COLORS.get(model, 'gray')
                ax.hist(sd['boots'], bins=50, alpha=0.4, color=c, label=model, density=True)
            ax.axvline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
            ax.axvline(0, ls='-', color='gray', lw=0.5)
            ax.set_xlabel('β (VWAP intercept)')
            ax.set_ylabel('Density')
            ax.set_title(f'{stock} child={child} (η≈{eta}%) — β distribution (VWAP)')
            ax.legend(fontsize=7)

            # Midprice β
            ax = axes[1]
            for model in models:
                sd = all_scatter.get(model, {}).get(child)
                if sd is None or len(sd.get('boots_mid', [])) == 0:
                    continue
                c = COLORS.get(model, 'gray')
                ax.hist(sd['boots_mid'], bins=50, alpha=0.4, color=c, label=model, density=True)
            ax.axvline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
            ax.axvline(0, ls='-', color='gray', lw=0.5)
            ax.set_xlabel('β (midprice)')
            ax.set_title(f'{stock} child={child} (η≈{eta}%) — β distribution (midprice)')
            ax.legend(fontsize=7)

            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ── Impact distributions per child ──
        for child in child_sizes:
            eta = ETA_MAP.get((stock, child), '?')

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Raw I boxplot
            ax = axes[0]
            data_box = []
            labels_box = []
            for model in models:
                sd = all_scatter.get(model, {}).get(child)
                if sd is None or len(sd['I_raw']) == 0:
                    continue
                data_box.append(sd['I_raw'])
                labels_box.append(model)
            if data_box:
                bp = ax.boxplot(data_box, labels=labels_box, showfliers=False, patch_artist=True)
                for patch, model in zip(bp['boxes'], labels_box):
                    patch.set_facecolor(COLORS.get(model, 'gray'))
                    patch.set_alpha(0.6)
            ax.set_ylabel('I (VWAP fractional impact)')
            ax.set_title(f'{stock} child={child} (η≈{eta}%) — Impact at k=10')
            ax.tick_params(axis='x', rotation=30)

            # log(I) histogram
            ax = axes[1]
            for model in models:
                sd = all_scatter.get(model, {}).get(child)
                if sd is None or len(sd['log_I']) == 0:
                    continue
                c = COLORS.get(model, 'gray')
                vals = sd['log_I'][np.isfinite(sd['log_I'])]
                ax.hist(vals, bins=60, alpha=0.35, color=c, label=model, density=True)
            ax.set_xlabel('log(I)')
            ax.set_title(f'{stock} child={child} (η≈{eta}%) — log(Impact) distribution')
            ax.legend(fontsize=7)

            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ── β(η) curve ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # VWAP β
        ax = axes[0]
        for model in models:
            mdf = df[df['model'] == model].sort_values('eta')
            if mdf.empty:
                continue
            c = COLORS.get(model, 'gray')
            ax.errorbar(mdf['eta'], mdf['beta'],
                        yerr=[mdf['beta'] - mdf['ci_lo'], mdf['ci_hi'] - mdf['beta']],
                        fmt='o-', color=c, lw=2, markersize=6, capsize=4, label=model)
        ax.axhline(0.5, ls='--', color='red', lw=1.5, alpha=0.7, label='β=0.5')
        ax.axhline(0, ls='-', color='gray', lw=0.5)
        ax.set_xlabel('η (participation rate, %)')
        ax.set_ylabel('β (VWAP, intercept)')
        ax.set_title(f'{stock} — β(η) curve (k=10, full metaorder)')
        ax.legend(fontsize=8)

        # Midprice β
        ax = axes[1]
        for model in models:
            mdf = df[df['model'] == model].sort_values('eta')
            if mdf.empty or mdf['beta_mid'].isna().all():
                continue
            c = COLORS.get(model, 'gray')
            ax.plot(mdf['eta'], mdf['beta_mid'], 'o-', color=c, lw=2, markersize=6, label=model)
        ax.axhline(0.5, ls='--', color='red', lw=1.5, alpha=0.7, label='β=0.5')
        ax.axhline(0, ls='-', color='gray', lw=0.5)
        ax.set_xlabel('η (participation rate, %)')
        ax.set_ylabel('β (midprice)')
        ax.set_title(f'{stock} — β_mid(η) curve')
        ax.legend(fontsize=8)

        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── Summary table ──
        fig = plt.figure(figsize=(14, 10))
        fig.text(0.5, 0.97, f'Metaorder β Summary — {stock} (k=10 only)',
                 ha='center', fontsize=16, fontweight='bold')

        header = (f'{"Model":>12s}  {"child":>6s}  {"η%":>4s}  {"Q":>6s}  '
                  f'{"β_vwap":>8s}  {"CI":>18s}  {"R²":>8s}  {"β_mid":>8s}  '
                  f'{"I_med":>10s}  {"N":>6s}  {"days":>4s}')
        y = 0.93
        fig.text(0.03, y, header, fontsize=8, fontfamily='monospace', fontweight='bold')
        y -= 0.018

        for _, r in df.sort_values(['model', 'eta']).iterrows():
            eta_s = f'{r["eta"]:.0f}' if pd.notna(r['eta']) else '?'
            ci_s = f'[{r["ci_lo"]:.3f},{r["ci_hi"]:.3f}]' if pd.notna(r['ci_lo']) else ''
            line = (f'{r["model"]:>12s}  {r["child"]:>6.0f}  {eta_s:>4s}  {r["Q_total"]:>6.0f}  '
                    f'{r["beta"]:>8.4f}  {ci_s:>18s}  {r["r2"]:>8.4f}  '
                    f'{r.get("beta_mid", np.nan):>8.4f}  '
                    f'{r["I_median"]:>10.6f}  {r["n"]:>6.0f}  {r["n_days"]:>4.0f}')
            fig.text(0.03, y, line, fontsize=7, fontfamily='monospace')
            y -= 0.014
            if y < 0.03:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(14, 10)); y = 0.95

        pdf.savefig(fig); plt.close(fig)

    print(f'  Saved PDF: {pdf_path}')


def main():
    parser = argparse.ArgumentParser(description='Metaorder β analysis (k=last only)')
    parser.add_argument('--stock', required=True, choices=['AAPL', 'AMZN', 'GOOG', 'INTC'])
    parser.add_argument('--pickle_base', required=True,
                        help='Base directory containing {STOCK}/*.pkl')
    parser.add_argument('--daily_hl', required=True,
                        help='Path to daily_h_l_{STOCK}.csv')
    parser.add_argument('--out_dir', default='pics_for_metaorder_beta')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to analyze (default: all found)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    if not pkl_dir.exists():
        print(f'ERROR: {pkl_dir} does not exist')
        sys.exit(1)

    if args.models:
        models = args.models
    else:
        models = [m for m in MODELS_ORDER
                  if (pkl_dir / f'{m}.pkl').exists()]

    print(f'{"=" * 70}')
    print(f'  Metaorder β Analysis — {args.stock}')
    print(f'  Models: {models}')
    print(f'  Pickles: {pkl_dir}')
    print(f'{"=" * 70}')

    all_results = []
    all_scatter = {}  # model -> child -> scatter_data

    for model in models:
        pkl_path = pkl_dir / f'{model}.pkl'
        results, scatter = analyze_one_model(
            str(pkl_path), args.daily_hl, model, args.stock)
        all_results.extend(results)
        all_scatter[model] = scatter
        gc.collect()

    if not all_results:
        print('ERROR: no results')
        sys.exit(1)

    make_pdf(all_results, all_scatter, args.stock, out_dir)

    # Print final summary
    df = pd.DataFrame(all_results)
    print(f'\n{"=" * 70}')
    print(f'  FINAL SUMMARY — {args.stock}')
    print(f'{"=" * 70}')
    print(f'{"Model":>12s}  {"child":>6s}  {"η%":>4s}  {"β":>8s}  {"CI":>18s}  {"R²":>8s}  {"N":>6s}')
    for _, r in df.sort_values(['model', 'eta']).iterrows():
        eta_s = f'{r["eta"]:.0f}' if pd.notna(r['eta']) else '?'
        ci_s = f'[{r["ci_lo"]:.3f},{r["ci_hi"]:.3f}]' if pd.notna(r['ci_lo']) else ''
        print(f'{r["model"]:>12s}  {r["child"]:>6.0f}  {eta_s:>4s}  '
              f'{r["beta"]:>8.4f}  {ci_s:>18s}  {r["r2"]:>8.4f}  {r["n"]:>6.0f}')


if __name__ == '__main__':
    main()
