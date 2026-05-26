#!/usr/bin/env python3
"""
Beta hypothesis test analysis.

Reads CSV output from beta_test experiments (i=7, mb=50, c=70),
computes beta for each model, generates comparison figure.

Hypothesis: β ≈ 1.0 on v3 grid is due to insufficient insertions.
With i=7 and mb=50 (total 3850 msgs), we expect:
  - S5-4K: β < 1.0 (good neural model)
  - Historic/Heuristic: β ≈ 1.0 (replay baselines)
  - CST: β ≈ 1.0 (parametric baseline)

Usage:
  python lob_impact/run_beta_test_analysis.py --base /path/to/beta_test_i7_mb50 --stock GOOG --daily_hl lob_impact/daily_h_l_GOOG.csv
"""
import argparse, pickle, sys, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
TICK_SIZE = 100
N_BOOTSTRAP = 1000

MODELS = OrderedDict([
    ('S5-4K',     dict(color='#5B7BBF', marker='*', ms=10)),
    ('Historic',  dict(color='#90939C', marker='x', ms=8)),
    ('Heuristic', dict(color='#546884', marker='d', ms=7)),
    ('CST',       dict(color='#213552', marker='^', ms=7)),
])


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════
def find_latest_exp(folder_path):
    p = Path(folder_path)
    if not p.exists(): return p
    exps = sorted(p.glob('exp_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    return exps[0] if exps else p


def load_aggressive_indices(exp_path):
    aggr_file = Path(exp_path) / 'aggressive_indices.csv'
    if not aggr_file.exists(): return None
    return np.atleast_1d(np.loadtxt(aggr_file, dtype=int))


def load_csvs(gen_dir, max_samples=2048):
    books, msgs, days = [], [], []
    gen_dir = Path(gen_dir)
    if not gen_dir.exists():
        return books, msgs, days
    ob_files = sorted(gen_dir.glob('*_orderbook_*_gen_id_0.csv'))[:max_samples]
    for ob_f in ob_files:
        try:
            book = pd.read_csv(ob_f, header=None).values
            books.append(book)
            m = re.search(r'(\d{4}-\d{2}-\d{2})', ob_f.name)
            days.append(m.group(1) if m else None)
            msg_f = ob_f.parent / ob_f.name.replace('_orderbook_', '_message_')
            if msg_f.exists():
                msgs.append(pd.read_csv(msg_f, header=None).values)
        except:
            pass
    return books, msgs, days


def load_daily_params(daily_hl_path, exp_days):
    df = pd.read_csv(daily_hl_path)
    if 'day' not in df.columns:
        df['day'] = df['filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    raw_days = sorted(df['day'].dropna().unique())
    exp_days_sorted = sorted(set(exp_days))
    day_map = dict(zip(raw_days, exp_days_sorted))
    params = {}
    for _, row in df.iterrows():
        raw_day = row.get('day', None)
        if raw_day is None or pd.isna(raw_day): continue
        exp_day = day_map.get(raw_day, raw_day)
        H, L = row['highest_price'], row['lowest_price']
        V = row['execution_sum']
        sigma = np.log(H / L) / 0.8325546 if H > 0 and L > 0 else 1.0
        params[exp_day] = dict(H=H, L=L, V=V, sigma=sigma)
    return params


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════
def get_midprice(book_arr):
    return (book_arr[:, 0].astype(float) + book_arr[:, 2].astype(float)) / 2.0


def extract_point_cloud(books_buy, msgs_buy, days_buy, aggr_buy,
                        books_sell, msgs_sell, days_sell, aggr_sell,
                        daily_params=None):
    """VWAP per-order impact. ref_price = book midprice at first aggressive order."""
    points = []
    for direction, aggr_idx, msgs_list, books_list, days_list in [
        ('buy',  aggr_buy,  msgs_buy,  books_buy,  days_buy),
        ('sell', aggr_sell, msgs_sell, books_sell, days_sell),
    ]:
        n_aggr = len(aggr_idx)
        if n_aggr < 2: continue
        for j in range(min(len(msgs_list), len(books_list))):
            msg, book = msgs_list[j], books_list[j]
            if len(msg) == 0: continue
            if aggr_idx.max() >= len(msg) or aggr_idx.max() >= len(book): continue

            ref_mid = (float(book[aggr_idx[0], 0]) + float(book[aggr_idx[0], 2])) / 2.0
            if ref_mid <= 0: continue

            sizes = msg[aggr_idx, 3].astype(float)
            prices = msg[aggr_idx, 4].astype(float)
            if np.any(sizes <= 0) or np.any(prices <= 0): continue

            Q_cum = np.cumsum(sizes)
            vwap = np.cumsum(sizes * prices) / Q_cum
            if direction == 'buy':
                impact = (vwap - ref_mid) / ref_mid
            else:
                impact = (ref_mid - vwap) / ref_mid
            impact = np.abs(impact)

            sample_id = f'{direction}_{j}'

            # Per-insertion midprice response + instantaneous impact
            I_mid_arr = np.full(n_aggr, np.nan)
            I_inst_arr = np.full(n_aggr, np.nan)
            for k in range(n_aggr):
                mid_before = (float(book[aggr_idx[k], 0]) + float(book[aggr_idx[k], 2])) / 2.0
                if mid_before <= 0:
                    continue
                if k + 1 < n_aggr:
                    mid_after = (float(book[aggr_idx[k+1], 0]) + float(book[aggr_idx[k+1], 2])) / 2.0
                else:
                    mid_after = (float(book[-1, 0]) + float(book[-1, 2])) / 2.0
                if mid_after > 0:
                    I_mid_arr[k] = abs(mid_after - mid_before) / mid_before
                I_inst_arr[k] = abs(prices[k] - mid_before) / mid_before

            for k in range(n_aggr):
                if impact[k] <= 1e-10 or Q_cum[k] <= 0: continue
                pt = dict(Q=float(Q_cum[k]), I=float(impact[k]),
                          I_mid=float(I_mid_arr[k]) if np.isfinite(I_mid_arr[k]) else np.nan,
                          I_inst=float(I_inst_arr[k]) if np.isfinite(I_inst_arr[k]) else np.nan,
                          k=k+1, sample_id=sample_id)
                if j < len(days_list) and days_list[j]:
                    pt['day'] = days_list[j]
                if daily_params and pt.get('day') in daily_params:
                    dp = daily_params[pt['day']]
                    pt['daily_vol'] = dp['V']
                    pt['daily_sigma'] = dp['sigma']
                points.append(pt)
    return pd.DataFrame(points)


def compute_beta_ols(pc_df, impact_col='I'):
    """Three beta estimators: OLS-origin, OLS-intercept, ratio.

    Primary metric: beta = beta_intercept.
    """
    empty = dict(beta=np.nan, beta_origin=np.nan, beta_intercept=np.nan,
                 beta_ratio=np.nan, alpha=np.nan,
                 r2=np.nan, r2_origin=np.nan, r2_intercept=np.nan, n=0,
                 x=np.array([]), y=np.array([]), y_raw=np.array([]))
    if pc_df.empty:
        return empty

    I_vals = pc_df[impact_col].values
    has_daily = 'daily_vol' in pc_df.columns

    if has_daily:
        x = np.log(pc_df['Q'].values / pc_df['daily_vol'].values)
        y_adj = np.log(I_vals / pc_df['daily_sigma'].values)
        y_raw = np.log(I_vals)
    else:
        x = np.log(pc_df['Q'].values / 1e6)
        y_adj = np.log(I_vals / 1.0)
        y_raw = np.log(I_vals)

    ok = np.isfinite(x) & np.isfinite(y_adj) & np.isfinite(y_raw) & (x != 0)
    xv, yv_adj, yv_raw = x[ok], y_adj[ok], y_raw[ok]
    if len(xv) < 2:
        return empty

    # 1. OLS through origin
    beta_origin = float(np.dot(xv, yv_adj) / np.dot(xv, xv))
    ss_res_o = np.sum((yv_adj - beta_origin * xv)**2)
    ss_tot_o = np.sum(yv_adj**2)
    r2_origin = 1 - ss_res_o / ss_tot_o if ss_tot_o > 0 else 0.0

    # 2. OLS with free intercept
    coeffs = np.polyfit(xv, yv_raw, 1)
    beta_intercept = float(coeffs[0])
    alpha = float(coeffs[1])
    yhat = beta_intercept * xv + alpha
    ss_res_i = np.sum((yv_raw - yhat)**2)
    ss_tot_i = np.sum((yv_raw - np.mean(yv_raw))**2)
    r2_intercept = 1 - ss_res_i / ss_tot_i if ss_tot_i > 0 else 0.0

    # 3. Ratio estimator
    beta_ratio = float(np.mean(yv_adj / xv))

    return dict(
        beta=beta_intercept,
        beta_origin=beta_origin,
        beta_intercept=beta_intercept,
        beta_ratio=beta_ratio,
        alpha=alpha,
        r2=r2_intercept,
        r2_origin=r2_origin,
        r2_intercept=r2_intercept,
        n=int(ok.sum()),
        x=xv, y=yv_adj, y_raw=yv_raw,
    )


def bootstrap_beta(pc_df, n_boot=N_BOOTSTRAP, impact_col='I'):
    """Bootstrap all three beta estimators. Vectorized resampling."""
    empty = dict(boots=np.array([]), boots_origin=np.array([]),
                 boots_intercept=np.array([]), boots_ratio=np.array([]))
    if pc_df.empty:
        return empty

    I_vals = pc_df[impact_col].values
    has_daily = 'daily_vol' in pc_df.columns

    if has_daily:
        x_all = np.log(pc_df['Q'].values / pc_df['daily_vol'].values)
        y_adj_all = np.log(I_vals / pc_df['daily_sigma'].values)
        y_raw_all = np.log(I_vals)
    else:
        x_all = np.log(pc_df['Q'].values / 1e6)
        y_adj_all = np.log(I_vals / 1.0)
        y_raw_all = np.log(I_vals)

    ok_all = np.isfinite(x_all) & np.isfinite(y_adj_all) & np.isfinite(y_raw_all) & (x_all != 0)
    x_f, y_adj_f, y_raw_f = x_all[ok_all], y_adj_all[ok_all], y_raw_all[ok_all]
    sid_f = pc_df['sample_id'].values[ok_all]
    if len(x_f) < 2:
        return empty

    unique_ids = np.unique(sid_f)
    id_to_int = {sid: i for i, sid in enumerate(unique_ids)}
    group_labels = np.array([id_to_int[s] for s in sid_f])
    n_groups = len(unique_ids)
    group_rows = [np.where(group_labels == g)[0] for g in range(n_groups)]
    max_gs = max(len(g) for g in group_rows)
    gmat = np.full((n_groups, max_gs), -1, dtype=np.int64)
    for g, rows in enumerate(group_rows):
        gmat[g, :len(rows)] = rows

    rng = np.random.default_rng(42)
    b_origin = np.empty(n_boot)
    b_intercept = np.empty(n_boot)
    b_ratio = np.empty(n_boot)

    for b in range(n_boot):
        bg = rng.choice(n_groups, size=n_groups, replace=True)
        flat = gmat[bg].ravel()
        row_idx = flat[flat >= 0]
        xo = x_f[row_idx]; yo_a = y_adj_f[row_idx]; yo_r = y_raw_f[row_idx]
        n = len(xo)
        if n < 2:
            b_origin[b] = b_intercept[b] = b_ratio[b] = np.nan
            continue
        b_origin[b] = np.dot(xo, yo_a) / np.dot(xo, xo)
        sx = xo.sum(); sy = yo_r.sum()
        sxy = np.dot(xo, yo_r); sx2 = np.dot(xo, xo)
        denom = n * sx2 - sx * sx
        b_intercept[b] = (n * sxy - sx * sy) / denom if abs(denom) > 1e-30 else np.nan
        b_ratio[b] = np.mean(yo_a / xo)

    return dict(
        boots=b_intercept, boots_origin=b_origin,
        boots_intercept=b_intercept, boots_ratio=b_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════
def make_figures(results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 13,
        'axes.titlesize': 15, 'axes.titleweight': 'bold',
        'axes.grid': True, 'grid.alpha': 0.25,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Fig 1: Beta regression lines (intercept + origin) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    rng = np.random.default_rng(42)
    for label, res in results.items():
        meta = MODELS[label]
        x, y = res.get('x', []), res.get('y', [])
        y_raw = res.get('y_raw', [])
        if len(x) == 0: continue
        n = len(x)
        if n > 800:
            idx = rng.choice(n, 800, replace=False)
        else:
            idx = np.arange(n)
        xl = np.array([x.min(), x.max()])

        # Left: free intercept
        if len(y_raw) > 0:
            ax1.scatter(x[idx], y_raw[idx], s=3, color=meta['color'], alpha=0.12)
            alpha = res.get('alpha', 0.0)
            ax1.plot(xl, res['beta'] * xl + alpha, color=meta['color'], lw=3,
                     label=f"{label}: β={res['beta']:.3f} [{res['ci_lo']:.2f}, {res['ci_hi']:.2f}]")

        # Right: through origin (legacy)
        ax2.scatter(x[idx], y[idx], s=3, color=meta['color'], alpha=0.12)
        beta_o = res.get('beta_origin', res['beta'])
        ci_o = (res.get('ci_origin_lo', np.nan), res.get('ci_origin_hi', np.nan))
        ax2.plot(xl, beta_o * xl, color=meta['color'], lw=3,
                 label=f"{label}: β={beta_o:.3f} [{ci_o[0]:.2f}, {ci_o[1]:.2f}]")

    for ax in (ax1, ax2):
        ax.plot([-15, -3], [0.5*-15, 0.5*-3], 'k--', lw=2, label='Theory: β=0.5')
        ax.legend(fontsize=8, loc='lower left')
    ax1.set(title=r'Free Intercept: $\log I = \alpha + \beta \log(Q/V)$',
            xlabel='log(Q/V)', ylabel='log(I)')
    ax2.set(title=r'Through Origin: $\log(I/\sigma) = \beta \log(Q/V)$',
            xlabel='log(Q/V)', ylabel=r'log(I/$\sigma$)')
    fig.suptitle('Beta Test: i=7, mb=50, c=70 (3850 msgs)', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'beta_test_regression.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_dir}/beta_test_regression.png')

    # ── Fig 2: Bootstrap distributions (intercept + origin) ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for label, res in results.items():
        meta = MODELS[label]
        boots_i = res.get('boots', [])
        boots_o = res.get('boots_origin', [])
        if len(boots_i) > 0:
            ax1.hist(boots_i, bins=50, alpha=0.45, color=meta['color'],
                     label=f"{label}: β={res['beta']:.3f}", edgecolor='none')
        if len(boots_o) > 0:
            ax2.hist(boots_o, bins=50, alpha=0.45, color=meta['color'],
                     label=f"{label}: β={res.get('beta_origin', np.nan):.3f}", edgecolor='none')
    for ax in (ax1, ax2):
        ax.axvline(0.5, ls='--', color='red', lw=2.5, label='Theory: 0.5')
        ax.axvline(1.0, ls=':', color='gray', lw=1.5, label='Linear: 1.0')
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylabel('Count')
    ax1.set_title(r'$\beta$ (free intercept)', fontsize=13, fontweight='bold')
    ax2.set_title(r'$\beta$ (through origin, legacy)', fontsize=13, fontweight='bold')
    ax2.set_xlabel(r'$\beta$')
    fig.suptitle('Bootstrap Beta Distributions (i=7, mb=50)', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'beta_test_bootstrap.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_dir}/beta_test_bootstrap.png')

    # ── Fig 3: Bar chart — 3 estimators ──
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_list = list(results.keys())
    n_models = len(labels_list)
    x_pos = np.arange(n_models)
    w = 0.25

    for offset, (est_key, ci_lo_key, ci_hi_key, est_label) in enumerate([
        ('beta_origin', 'ci_origin_lo', 'ci_origin_hi', 'Origin'),
        ('beta', 'ci_lo', 'ci_hi', 'Intercept'),
        ('beta_ratio', None, None, 'Ratio'),
    ]):
        vals = [results[l].get(est_key, np.nan) for l in labels_list]
        if ci_lo_key:
            lo = [results[l].get(ci_lo_key, np.nan) for l in labels_list]
            hi = [results[l].get(ci_hi_key, np.nan) for l in labels_list]
            yerr = [[v - l for v, l in zip(vals, lo)],
                    [h - v for v, h in zip(vals, hi)]]
        else:
            yerr = None
        colors = [MODELS[l]['color'] for l in labels_list]
        ax.bar(x_pos + (offset - 1) * w, vals, w * 0.9,
               color=colors if offset == 1 else None,
               edgecolor='gray', linewidth=0.5,
               yerr=yerr, capsize=3, label=est_label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, fontsize=11)
    ax.axhline(0.5, ls='--', color='red', lw=2, label='Theory: 0.5')
    ax.axhline(1.0, ls=':', color='gray', lw=1.5, label='Linear: 1.0')
    ax.set(title=r'$\beta$ Comparison: 3 Estimators (i=7, mb=50, c=70)',
           ylabel=r'$\beta$')
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'beta_test_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_dir}/beta_test_comparison.png')

    # ── Fig 4: Per-insertion impact metrics ──
    has_alt = any(results[l].get('beta_I_mid') is not None for l in labels_list)
    if has_alt:
        fig, ax = plt.subplots(figsize=(10, 5))
        for offset, (key, lbl) in enumerate([
            ('beta', 'VWAP'), ('beta_I_mid', 'Midprice'), ('beta_I_inst', 'Instant.')
        ]):
            vals = [results[l].get(key, np.nan) for l in labels_list]
            ax.bar(x_pos + (offset - 1) * w, vals, w * 0.9,
                   color=[MODELS[l]['color'] for l in labels_list] if offset == 0 else None,
                   edgecolor='gray', linewidth=0.5, label=lbl, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, fontsize=11)
        ax.axhline(0.5, ls='--', color='red', lw=2)
        ax.set(title=r'$\beta$ by Impact Metric (i=7, mb=50)', ylabel=r'$\beta$')
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / 'beta_test_impact_metrics.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_dir}/beta_test_impact_metrics.png')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Beta hypothesis test analysis')
    parser.add_argument('--base', type=str, required=True,
                        help='Base directory with experiment output')
    parser.add_argument('--stock', type=str, default='GOOG')
    parser.add_argument('--daily_hl', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base = Path(args.base)
    stock = args.stock
    out_dir = Path(args.out) if args.out else Path(f'pics_for_beta_test_{stock}')

    print('=' * 60)
    print(f'  Beta Hypothesis Test: {stock}')
    print(f'  Base: {base}')
    print('=' * 60)

    # Collect all days for daily params mapping
    all_days = set()
    results = {}

    for label in MODELS:
        print(f'\n── {label} ──')
        buy_root = base / label / f'context_500_buy' / stock
        sell_root = base / label / f'context_500_sell' / stock

        if not buy_root.exists():
            print(f'  SKIP: {buy_root} not found')
            continue

        # Find experiment folders
        folders = []
        for bp in sorted(buy_root.iterdir()):
            if not bp.is_dir(): continue
            sp = sell_root / bp.name
            if not sp.exists(): continue
            m = re.match(r'i(\d+)_c(\d+)_mb(\d+)_v(\d+)_cntxt(\d+)%', bp.name)
            if not m: continue
            folders.append(dict(
                name=bp.name,
                buy_exp=find_latest_exp(bp),
                sell_exp=find_latest_exp(sp)))

        if not folders:
            print(f'  SKIP: no experiment folders found')
            continue

        for fd in folders:
            print(f'  Loading {fd["name"]}...')
            buy_books, buy_msgs, buy_days = load_csvs(fd['buy_exp'] / 'data_gen')
            sell_books, sell_msgs, sell_days = load_csvs(fd['sell_exp'] / 'data_gen')
            print(f'    {len(buy_books)} buy + {len(sell_books)} sell samples')

            aggr_buy = load_aggressive_indices(fd['buy_exp'])
            aggr_sell = load_aggressive_indices(fd['sell_exp'])
            if aggr_buy is None or aggr_sell is None:
                print(f'    SKIP: no aggressive_indices.csv')
                continue

            # Collect days
            for d in buy_days + sell_days:
                if d: all_days.add(d)

            # Load daily params
            daily_params = None
            if args.daily_hl:
                daily_params = load_daily_params(args.daily_hl, all_days)
                print(f'    Daily params: {len(daily_params)} days')

            # Point cloud
            pc = extract_point_cloud(
                buy_books, buy_msgs, buy_days, aggr_buy,
                sell_books, sell_msgs, sell_days, aggr_sell,
                daily_params)
            print(f'    Points: {len(pc)}')

            if pc.empty:
                print(f'    SKIP: empty point cloud')
                continue

            # Beta (3 estimators)
            res = compute_beta_ols(pc)
            boot_dict = bootstrap_beta(pc)
            boots = boot_dict['boots']
            ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) > 0 else (np.nan, np.nan)
            ci_origin = (np.nan, np.nan)
            if len(boot_dict['boots_origin']) > 0:
                ci_origin = (np.percentile(boot_dict['boots_origin'], 2.5),
                             np.percentile(boot_dict['boots_origin'], 97.5))

            results[label] = dict(
                beta=res['beta'], beta_origin=res['beta_origin'],
                beta_ratio=res['beta_ratio'], alpha=res['alpha'],
                r2=res['r2'], r2_origin=res['r2_origin'], n=res['n'],
                ci_lo=ci[0], ci_hi=ci[1], boots=boots,
                ci_origin_lo=ci_origin[0], ci_origin_hi=ci_origin[1],
                boots_origin=boot_dict['boots_origin'],
                x=res.get('x', np.array([])),
                y=res.get('y', np.array([])),
                y_raw=res.get('y_raw', np.array([])),
                n_buy=len(buy_books), n_sell=len(sell_books))

            # Beta on alternative impact metrics
            for alt_col in ('I_mid', 'I_inst'):
                if alt_col in pc.columns:
                    pc_alt = pc[pc[alt_col].notna() & (pc[alt_col] > 1e-10)].copy()
                    if not pc_alt.empty:
                        res_alt = compute_beta_ols(pc_alt, impact_col=alt_col)
                        results[label][f'beta_{alt_col}'] = res_alt['beta']
                        print(f'    β({alt_col}) = {res_alt["beta"]:.4f}')

            print(f'    β_intercept = {res["beta"]:.4f}  [{ci[0]:.3f}, {ci[1]:.3f}]  R²={res["r2"]:.4f}')
            print(f'    β_origin    = {res["beta_origin"]:.4f}  [{ci_origin[0]:.3f}, {ci_origin[1]:.3f}]  R²={res["r2_origin"]:.4f}')
            print(f'    β_ratio     = {res["beta_ratio"]:.4f}  n={res["n"]}')

    # Summary
    print('\n' + '=' * 60)
    print('  RESULTS (3 estimators)')
    print('=' * 60)
    print(f'  {"Model":12s}  {"β_intercept":>12s}  {"β_origin":>10s}  {"β_ratio":>9s}  {"β(I_mid)":>9s}  {"β(I_inst)":>10s}')
    print(f'  {"-"*12}  {"-"*12}  {"-"*10}  {"-"*9}  {"-"*9}  {"-"*10}')
    for label, res in results.items():
        b_int = f'{res["beta"]:.4f}  [{res["ci_lo"]:.2f},{res["ci_hi"]:.2f}]'
        b_ori = f'{res["beta_origin"]:.4f}'
        b_rat = f'{res["beta_ratio"]:.4f}'
        b_mid = f'{res.get("beta_I_mid", np.nan):.4f}' if not np.isnan(res.get('beta_I_mid', np.nan)) else '   N/A'
        b_inst = f'{res.get("beta_I_inst", np.nan):.4f}' if not np.isnan(res.get('beta_I_inst', np.nan)) else '    N/A'
        print(f'  {label:12s}  {b_int:>12s}  {b_ori:>10s}  {b_rat:>9s}  {b_mid:>9s}  {b_inst:>10s}')

    # Verdict (using intercept beta as primary)
    print('\n' + '=' * 60)
    s5_beta = results.get('S5-4K', {}).get('beta', np.nan)
    baseline_betas = [results[m]['beta'] for m in ('Historic', 'Heuristic', 'CST') if m in results]
    if not np.isnan(s5_beta) and baseline_betas:
        baseline_mean = np.mean(baseline_betas)
        print(f'  S5-4K β_intercept     = {s5_beta:.4f}')
        print(f'  Baselines β_intercept = {baseline_mean:.4f} (mean)')
        if s5_beta < 0.85 and baseline_mean > 0.9:
            print('  VERDICT: HYPOTHESIS CONFIRMED')
        elif s5_beta < baseline_mean - 0.1:
            print('  VERDICT: PARTIAL — S5-4K is better but not conclusively sub-linear')
        else:
            print('  VERDICT: HYPOTHESIS NOT CONFIRMED')
            print('  S5-4K and baselines show similar β. Issue may be elsewhere.')
    else:
        print('  VERDICT: INSUFFICIENT DATA')
    print('=' * 60)

    # Figures
    if results:
        make_figures(results, out_dir)

    # Save CSV summary
    out_dir.mkdir(exist_ok=True)
    rows = [dict(Model=l,
                 beta_intercept=r['beta'], beta_origin=r['beta_origin'],
                 beta_ratio=r['beta_ratio'],
                 CI_lo=r['ci_lo'], CI_hi=r['ci_hi'],
                 R2=r['r2'], R2_origin=r['r2_origin'], N=r['n'],
                 beta_I_mid=r.get('beta_I_mid', np.nan),
                 beta_I_inst=r.get('beta_I_inst', np.nan),
                 N_buy=r['n_buy'], N_sell=r['n_sell'])
            for l, r in results.items()]
    df = pd.DataFrame(rows)
    csv_path = out_dir / 'beta_test_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  Summary CSV: {csv_path}')


if __name__ == '__main__':
    main()
