#!/usr/bin/env python3
"""
Diag 7 · Stratified beta analysis — find subsets where beta ~ 0.5.

Loads metrics pickles from pics_for_v4_300_{STOCK}/ (and optionally v3),
slices beta by day, config (mb, vol), insertion number k, and stock.
Also does per-k analysis from raw Lustre pickle for one reference model.

Outputs a text report to pics_for_investigation/diag_7_stratified.txt.

Usage:
  python lob_impact/diag_7_stratified.py --stock GOOG
  python lob_impact/diag_7_stratified.py --stock INTC
  python lob_impact/diag_7_stratified.py --stock GOOG --skip_raw
"""
import argparse, pickle, io
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
MIDPRICE_OUTLIER_FACTOR = 1.5

V4_PICKLE_BASE = Path(
    '/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/'
    'evalsequences/aggressive_scenario_v4/pickles'
)
V3_PICKLE_BASE = Path(
    '/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/'
    'evalsequences/aggressive_scenario_v3/pickles'
)

MODELS_ORDER = [
    'Historic', 'Heuristic', 'CST', 'CGAN',
    'LobS5', 'S5-120M', 'S5-4K', 'S5-360M', 'LobS5-v2',
]

# Reference model for per-k raw analysis
REF_MODEL = 'Historic'

# k values to test in per-k analysis
K_VALUES = [1, 2, 3, 5, 7, 10, 15]


# ═══════════════════════════════════════════════════════════════════════
# Helpers (reused from run_300_analyze_one.py)
# ═══════════════════════════════════════════════════════════════════════
def get_midprice(book_arr):
    return (book_arr[:, 0].astype(float) + book_arr[:, 2].astype(float)) / 2.0


def is_midprice_outlier(book_arr, factor=MIDPRICE_OUTLIER_FACTOR):
    mp = get_midprice(book_arr)
    mp_valid = mp[mp > 0]
    if len(mp_valid) < 2:
        return True
    ref = mp_valid[0]
    return np.any(mp_valid > ref * factor) or np.any(mp_valid < ref / factor)


def load_aggressive_indices(exp_path):
    aggr_file = Path(exp_path) / 'aggressive_indices.csv'
    if not aggr_file.exists():
        return None
    vals = np.loadtxt(aggr_file, dtype=int)
    return np.atleast_1d(vals)


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
        if raw_day is None or pd.isna(raw_day):
            continue
        exp_day = day_map.get(raw_day, raw_day)
        H, L = row['highest_price'], row['lowest_price']
        V = row['execution_sum']
        sigma = np.log(H / L) / 0.8325546 if H > 0 and L > 0 else 1.0
        params[exp_day] = dict(H=H, L=L, V=V, sigma=sigma)
    return params


def compute_global_beta(pc_df, impact_col='I'):
    """OLS with free intercept: log(I) = alpha + beta * log(Q/V)."""
    empty = dict(beta=np.nan, alpha=np.nan, r2=np.nan, n=0)
    if pc_df.empty:
        return empty

    I_vals = pc_df[impact_col].values
    V_denom = pc_df['daily_vol'].values if 'daily_vol' in pc_df.columns else 1e6
    x = np.log(pc_df['Q'].values / V_denom)
    y_raw = np.log(I_vals)

    ok = np.isfinite(x) & np.isfinite(y_raw) & (x != 0)
    xv, yv = x[ok], y_raw[ok]
    if len(xv) < 2:
        return empty

    coeffs = np.polyfit(xv, yv, 1)
    beta = float(coeffs[0])
    alpha = float(coeffs[1])
    yhat = beta * xv + alpha
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return dict(beta=beta, alpha=alpha, r2=r2, n=int(ok.sum()))


# ═══════════════════════════════════════════════════════════════════════
# Load metrics pickles
# ═══════════════════════════════════════════════════════════════════════
def load_all_metrics(stock, src_dir):
    """Load all .metrics.pkl from a directory."""
    src = Path(src_dir)
    if not src.exists():
        print(f'  WARNING: {src} not found, skipping')
        return {}

    result = {}
    for model in MODELS_ORDER:
        pkl = src / f'{model}.metrics.pkl'
        if not pkl.exists():
            continue
        with open(pkl, 'rb') as f:
            mc = pickle.load(f)
        result[model] = mc
    return result


# ═══════════════════════════════════════════════════════════════════════
# Per-k analysis from raw pickle
# ═══════════════════════════════════════════════════════════════════════
def extract_point_cloud_from_raw(md, daily_params=None):
    """Extract full point cloud from already-loaded raw pickle data.

    Args:
        md: dict with 'grid' (DataFrame) and 'data' (dict of folder data).
        daily_params: optional dict mapping day -> {V, sigma, ...}.
    """
    grid_df = md['grid']

    # Filter
    filtered = {}
    for folder, fd in md['data'].items():
        filt = {}
        for direction in ('buy', 'sell'):
            books, msgs, days = [], [], []
            src = fd[direction]
            for j, book in enumerate(src['books']):
                if is_midprice_outlier(book):
                    continue
                books.append(book)
                days.append(src['days'][j] if j < len(src['days']) else None)
                if j < len(src['msgs']):
                    msgs.append(src['msgs'][j])
            filt[direction] = dict(books=books, msgs=msgs, days=days)
        filtered[folder] = filt

    # Extract points
    points = []
    _aggr_cache = {}
    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in filtered:
            continue
        fd = filtered[folder]

        bp, sp = row['buy_path'], row['sell_path']
        if bp not in _aggr_cache:
            _aggr_cache[bp] = load_aggressive_indices(bp)
        if sp not in _aggr_cache:
            _aggr_cache[sp] = load_aggressive_indices(sp)
        aggr_buy = _aggr_cache[bp]
        aggr_sell = _aggr_cache[sp]
        if aggr_buy is None or aggr_sell is None:
            continue

        for direction, aggr_idx, msgs_list, books_list, days_list in [
            ('buy', aggr_buy, fd['buy']['msgs'], fd['buy']['books'],
             fd['buy']['days']),
            ('sell', aggr_sell, fd['sell']['msgs'], fd['sell']['books'],
             fd['sell']['days']),
        ]:
            n_aggr = len(aggr_idx)
            if n_aggr < 2:
                continue

            for j in range(min(len(msgs_list), len(books_list))):
                msg = msgs_list[j]
                book = books_list[j]
                if len(msg) == 0:
                    continue
                if aggr_idx.max() >= len(msg) or aggr_idx.max() >= len(book):
                    continue

                ref_mid = (float(book[aggr_idx[0], 0])
                           + float(book[aggr_idx[0], 2])) / 2.0
                if ref_mid <= 0:
                    continue

                sizes = msg[aggr_idx, 3].astype(float)
                prices = msg[aggr_idx, 4].astype(float)
                if np.any(sizes <= 0) or np.any(prices <= 0):
                    continue

                Q_cum = np.cumsum(sizes)
                vwap = np.cumsum(sizes * prices) / Q_cum

                if direction == 'buy':
                    impact = (vwap - ref_mid) / ref_mid
                else:
                    impact = (ref_mid - vwap) / ref_mid
                impact = np.abs(impact)

                sample_id = f'{folder}_{direction}_{j}'

                for k in range(n_aggr):
                    if impact[k] <= 1e-10 or Q_cum[k] <= 0:
                        continue
                    pt = dict(
                        Q=float(Q_cum[k]), I=float(impact[k]),
                        vol=row['vol'], i=row['i'], mb=row['mb'],
                        k=k + 1, folder=folder, sample_id=sample_id,
                        direction=direction,
                    )
                    if j < len(days_list) and days_list[j]:
                        pt['day'] = days_list[j]
                    if daily_params and pt.get('day') in daily_params:
                        dp = daily_params[pt['day']]
                        pt['daily_vol'] = dp['V']
                        pt['daily_sigma'] = dp['sigma']
                    points.append(pt)

    print(f'  Extracted {len(points)} points from {len(grid_df)} configs')
    return pd.DataFrame(points)


# ═══════════════════════════════════════════════════════════════════════
# Analysis sections
# ═══════════════════════════════════════════════════════════════════════
def section_per_day(metrics, stock, out):
    """1. Per-day beta analysis across models."""
    out.write('\n' + '=' * 70 + '\n')
    out.write('1. PER-DAY BETA ANALYSIS\n')
    out.write('=' * 70 + '\n\n')

    all_day_betas = []
    day_beta_by_model = {}

    for model, mc in metrics.items():
        day_betas = mc.get('day_betas', {})
        if not day_betas:
            continue
        day_beta_by_model[model] = day_betas
        for day, beta in day_betas.items():
            all_day_betas.append(dict(model=model, day=day, beta=beta))

    if not all_day_betas:
        out.write('  No per-day betas available.\n')
        return

    df = pd.DataFrame(all_day_betas)

    # Histogram stats
    betas = df['beta'].values
    out.write(f'Total per-day beta values: {len(betas)}\n')
    out.write(f'  Mean:   {np.mean(betas):.4f}\n')
    out.write(f'  Median: {np.median(betas):.4f}\n')
    out.write(f'  Std:    {np.std(betas):.4f}\n')
    out.write(f'  Min:    {np.min(betas):.4f}\n')
    out.write(f'  Max:    {np.max(betas):.4f}\n')
    out.write(f'  P5:     {np.percentile(betas, 5):.4f}\n')
    out.write(f'  P25:    {np.percentile(betas, 25):.4f}\n')
    out.write(f'  P75:    {np.percentile(betas, 75):.4f}\n')
    out.write(f'  P95:    {np.percentile(betas, 95):.4f}\n\n')

    # Count in target range [0.40, 0.60]
    in_range = np.sum((betas >= 0.40) & (betas <= 0.60))
    out.write(f'  In [0.40, 0.60]: {in_range}/{len(betas)} '
              f'({100 * in_range / len(betas):.1f}%)\n\n')

    # Per-model breakdown
    out.write('Per-model day-beta statistics:\n')
    header = f'  {"Model":<15} {"N":>4} {"Mean":>7} {"Median":>7} '
    header += f'{"Std":>7} {"Min":>7} {"Max":>7} {"N~0.5":>6}\n'
    out.write(header)
    out.write('  ' + '-' * 68 + '\n')

    for model in MODELS_ORDER:
        if model not in day_beta_by_model:
            continue
        db = day_beta_by_model[model]
        vals = np.array(list(db.values()))
        if len(vals) == 0:
            continue
        n_near = np.sum((vals >= 0.40) & (vals <= 0.60))
        out.write(f'  {model:<15} {len(vals):>4} {np.mean(vals):>7.4f} '
                  f'{np.median(vals):>7.4f} {np.std(vals):>7.4f} '
                  f'{np.min(vals):>7.4f} {np.max(vals):>7.4f} '
                  f'{n_near:>6}\n')
    out.write('\n')

    # Days closest to 0.5 (across all models)
    df['dist_05'] = np.abs(df['beta'] - 0.5)
    closest = df.nsmallest(15, 'dist_05')
    out.write('Days closest to beta=0.5 (top 15):\n')
    out.write(f'  {"Day":<12} {"Model":<15} {"Beta":>7}\n')
    out.write('  ' + '-' * 36 + '\n')
    for _, row in closest.iterrows():
        out.write(f'  {row["day"]:<12} {row["model"]:<15} '
                  f'{row["beta"]:>7.4f}\n')
    out.write('\n')


def section_per_config(metrics, stock, out):
    """2. Per-config analysis: beta vs mb, beta vs vol."""
    out.write('\n' + '=' * 70 + '\n')
    out.write('2. PER-CONFIG BETA ANALYSIS (mb, vol)\n')
    out.write('=' * 70 + '\n\n')

    # Beta vs mb
    out.write('Beta vs mb (messages between insertions):\n')
    header = f'  {"Model":<15}'
    all_mbs = set()
    for model in MODELS_ORDER:
        if model not in metrics:
            continue
        mb_betas = metrics[model].get('mb_betas', {})
        all_mbs.update(mb_betas.keys())
    all_mbs = sorted(all_mbs)

    if all_mbs:
        header += ''.join(f' mb={mb:>3}' for mb in all_mbs) + '\n'
        out.write(header)
        out.write('  ' + '-' * (15 + 7 * len(all_mbs)) + '\n')

        for model in MODELS_ORDER:
            if model not in metrics:
                continue
            mb_betas = metrics[model].get('mb_betas', {})
            if not mb_betas:
                continue
            line = f'  {model:<15}'
            for mb in all_mbs:
                val = mb_betas.get(mb, np.nan)
                if np.isnan(val):
                    line += '     - '
                else:
                    line += f' {val:>6.3f}'
            out.write(line + '\n')
        out.write('\n')

        # Find configs closest to 0.5
        best_mb = []
        for model in MODELS_ORDER:
            if model not in metrics:
                continue
            for mb, beta in metrics[model].get('mb_betas', {}).items():
                best_mb.append(dict(model=model, mb=mb, beta=beta,
                                    dist=abs(beta - 0.5)))
        if best_mb:
            best_mb_df = pd.DataFrame(best_mb).nsmallest(10, 'dist')
            out.write('Configs closest to beta=0.5 by mb (top 10):\n')
            for _, row in best_mb_df.iterrows():
                out.write(f'  {row["model"]:<15} mb={row["mb"]:>3}  '
                          f'beta={row["beta"]:.4f}\n')
            out.write('\n')
    else:
        out.write('  No mb_betas available.\n\n')

    # Beta vs vol
    out.write('Beta vs vol (order volume):\n')
    all_vols = set()
    for model in MODELS_ORDER:
        if model not in metrics:
            continue
        vol_betas = metrics[model].get('vol_betas', {})
        all_vols.update(vol_betas.keys())
    all_vols = sorted(all_vols)

    if all_vols:
        header = f'  {"Model":<15}'
        header += ''.join(f' v={v:>3}' for v in all_vols) + '\n'
        out.write(header)
        out.write('  ' + '-' * (15 + 7 * len(all_vols)) + '\n')

        for model in MODELS_ORDER:
            if model not in metrics:
                continue
            vol_betas = metrics[model].get('vol_betas', {})
            if not vol_betas:
                continue
            line = f'  {model:<15}'
            for v in all_vols:
                val = vol_betas.get(v, np.nan)
                if np.isnan(val):
                    line += '     - '
                else:
                    line += f' {val:>6.3f}'
            out.write(line + '\n')
        out.write('\n')

        # Do larger volumes give higher beta?
        out.write('Trend: does larger volume -> higher beta?\n')
        for model in MODELS_ORDER:
            if model not in metrics:
                continue
            vol_betas = metrics[model].get('vol_betas', {})
            if len(vol_betas) < 2:
                continue
            vols = sorted(vol_betas.keys())
            vals = [vol_betas[v] for v in vols]
            # Simple correlation
            if len(vals) >= 2:
                corr = np.corrcoef(vols, vals)[0, 1]
                out.write(f'  {model:<15} corr(vol, beta)={corr:+.3f}  '
                          f'[{vals[0]:.3f} -> {vals[-1]:.3f}]\n')
        out.write('\n')
    else:
        out.write('  No vol_betas available.\n\n')


def section_per_k(pc_df, stock, out):
    """3. Per-k analysis: does beta increase with k?"""
    out.write('\n' + '=' * 70 + '\n')
    out.write(f'3. PER-K BETA ANALYSIS (from raw pickle, {stock})\n')
    out.write('=' * 70 + '\n\n')

    if pc_df.empty:
        out.write('  No point cloud data available (raw pickle not loaded).\n')
        return

    available_k = sorted(pc_df['k'].unique())
    out.write(f'Available k values: {available_k}\n')
    out.write(f'Total points: {len(pc_df)}\n\n')

    header = f'  {"k":>3} {"N":>7} {"Beta":>7} {"Alpha":>7} {"R2":>6}\n'
    out.write(header)
    out.write('  ' + '-' * 33 + '\n')

    k_betas = {}
    for k_val in K_VALUES:
        if k_val not in available_k:
            continue
        pc_k = pc_df[pc_df['k'] == k_val]
        if len(pc_k) < 5:
            continue
        res = compute_global_beta(pc_k)
        k_betas[k_val] = res['beta']
        out.write(f'  {k_val:>3} {res["n"]:>7} {res["beta"]:>7.4f} '
                  f'{res["alpha"]:>7.4f} {res["r2"]:>6.4f}\n')
    out.write('\n')

    # Also try cumulative: k >= threshold
    out.write('Cumulative (k >= threshold):\n')
    header = f'  {"k>=":<5} {"N":>7} {"Beta":>7} {"Alpha":>7} {"R2":>6}\n'
    out.write(header)
    out.write('  ' + '-' * 35 + '\n')
    for k_thresh in [1, 2, 3, 5, 7, 10]:
        pc_cum = pc_df[pc_df['k'] >= k_thresh]
        if len(pc_cum) < 5:
            continue
        res = compute_global_beta(pc_cum)
        out.write(f'  {k_thresh:>3}   {res["n"]:>7} {res["beta"]:>7.4f} '
                  f'{res["alpha"]:>7.4f} {res["r2"]:>6.4f}\n')
    out.write('\n')

    # Trend
    if len(k_betas) >= 2:
        ks = sorted(k_betas.keys())
        vals = [k_betas[kk] for kk in ks]
        corr = np.corrcoef(ks, vals)[0, 1] if len(ks) >= 2 else np.nan
        out.write(f'Trend: corr(k, beta) = {corr:+.4f}\n')
        out.write(f'  k=1 beta={k_betas.get(1, np.nan):.4f}, '
                  f'k={max(ks)} beta={k_betas[max(ks)]:.4f}\n\n')


def section_cross_stock(out):
    """4. Cross-stock comparison (GOOG vs INTC)."""
    out.write('\n' + '=' * 70 + '\n')
    out.write('4. CROSS-STOCK COMPARISON (GOOG vs INTC)\n')
    out.write('=' * 70 + '\n\n')

    stock_data = {}
    for stock in ('GOOG', 'INTC'):
        src = Path(f'pics_for_v4_300_{stock}')
        metrics = load_all_metrics(stock, src)
        if metrics:
            stock_data[stock] = metrics

    if len(stock_data) < 2:
        out.write('  Need both GOOG and INTC metrics for comparison.\n')
        if stock_data:
            out.write(f'  Available: {list(stock_data.keys())}\n')
        return

    # Build comparison table
    header = (f'  {"Model":<15} {"GOOG_beta":>10} {"INTC_beta":>10} '
              f'{"GOOG_R2":>8} {"INTC_R2":>8} {"GOOG_n":>7} {"INTC_n":>7}\n')
    out.write(header)
    out.write('  ' + '-' * 68 + '\n')

    for model in MODELS_ORDER:
        goog = stock_data.get('GOOG', {}).get(model, {})
        intc = stock_data.get('INTC', {}).get(model, {})

        goog_beta = goog.get('beta', {})
        intc_beta = intc.get('beta', {})

        gb = goog_beta.get('beta', np.nan) if isinstance(goog_beta, dict) else np.nan
        ib = intc_beta.get('beta', np.nan) if isinstance(intc_beta, dict) else np.nan
        gr = goog_beta.get('r2', np.nan) if isinstance(goog_beta, dict) else np.nan
        ir = intc_beta.get('r2', np.nan) if isinstance(intc_beta, dict) else np.nan
        gn = goog_beta.get('n', 0) if isinstance(goog_beta, dict) else 0
        in_ = intc_beta.get('n', 0) if isinstance(intc_beta, dict) else 0

        if np.isnan(gb) and np.isnan(ib):
            continue

        out.write(f'  {model:<15} {gb:>10.4f} {ib:>10.4f} '
                  f'{gr:>8.4f} {ir:>8.4f} {gn:>7} {in_:>7}\n')
    out.write('\n')

    # Scorecard comparison
    out.write('Scorecard comparison:\n')
    header = f'  {"Model":<15} {"GOOG_score":>10} {"INTC_score":>10}\n'
    out.write(header)
    out.write('  ' + '-' * 37 + '\n')
    for model in MODELS_ORDER:
        goog = stock_data.get('GOOG', {}).get(model, {})
        intc = stock_data.get('INTC', {}).get(model, {})
        gs = goog.get('scorecard', {}).get('score', '-')
        is_ = intc.get('scorecard', {}).get('score', '-')
        if gs == '-' and is_ == '-':
            continue
        out.write(f'  {model:<15} {str(gs):>10} {str(is_):>10}\n')
    out.write('\n')


def section_low_high_i(stock, out):
    """5. Low-i (v3 grid) vs high-i (v4 grid) comparison."""
    out.write('\n' + '=' * 70 + '\n')
    out.write('5. LOW-i (v3 grid) vs HIGH-i (v4 grid) COMPARISON\n')
    out.write('=' * 70 + '\n\n')

    v3_src = Path(f'pics_for_300_{stock}')
    v4_src = Path(f'pics_for_v4_300_{stock}')

    v3_metrics = load_all_metrics(stock, v3_src)
    v4_metrics = load_all_metrics(stock, v4_src)

    if not v3_metrics:
        out.write(f'  No v3 metrics found at {v3_src}\n')
    if not v4_metrics:
        out.write(f'  No v4 metrics found at {v4_src}\n')
    if not v3_metrics or not v4_metrics:
        return

    out.write('v3 grid: i in {3,5}, mb in {5,10,15,25,50} (lower i, more configs)\n')
    out.write('v4 grid: i in {10,15}, mb in {5,10,15,25,50}, vol in {75,300,485} '
              '(higher i, calibrated volumes)\n\n')

    header = (f'  {"Model":<15} {"v3_beta":>8} {"v3_R2":>6} {"v3_n":>6} '
              f'{"v4_beta":>8} {"v4_R2":>6} {"v4_n":>6} {"delta":>7}\n')
    out.write(header)
    out.write('  ' + '-' * 65 + '\n')

    for model in MODELS_ORDER:
        v3 = v3_metrics.get(model, {})
        v4 = v4_metrics.get(model, {})

        v3b = v3.get('beta', {})
        v4b = v4.get('beta', {})

        b3 = v3b.get('beta', np.nan) if isinstance(v3b, dict) else np.nan
        b4 = v4b.get('beta', np.nan) if isinstance(v4b, dict) else np.nan
        r3 = v3b.get('r2', np.nan) if isinstance(v3b, dict) else np.nan
        r4 = v4b.get('r2', np.nan) if isinstance(v4b, dict) else np.nan
        n3 = v3b.get('n', 0) if isinstance(v3b, dict) else 0
        n4 = v4b.get('n', 0) if isinstance(v4b, dict) else 0

        if np.isnan(b3) and np.isnan(b4):
            continue

        delta = b4 - b3 if not (np.isnan(b3) or np.isnan(b4)) else np.nan
        delta_str = f'{delta:>+7.4f}' if not np.isnan(delta) else '      -'

        out.write(f'  {model:<15} {b3:>8.4f} {r3:>6.4f} {n3:>6} '
                  f'{b4:>8.4f} {r4:>6.4f} {n4:>6} {delta_str}\n')

    out.write('\n')
    out.write('Note: v4 grid has more insertions (i=10,15) — '
              'expect beta closer to 0.5\n')
    out.write('      if the square-root law emerges with sufficient '
              'market depth turnover.\n\n')


def section_conditional_betas(metrics, stock, out):
    """6. Conditional beta variants from metrics pickle."""
    out.write('\n' + '=' * 70 + '\n')
    out.write('6. CONDITIONAL BETA VARIANTS\n')
    out.write('=' * 70 + '\n\n')

    out.write('beta:           log(I) = a + b*log(Q/V)  (VWAP cumulative)\n')
    out.write('beta_k3plus:    same, but k >= 3 only\n')
    out.write('beta_Vlocal:    x = log(Q / V_local)\n')
    out.write('beta_I_mid:     impact = midprice shift after cooling\n')
    out.write('beta_I_inst:    impact = instantaneous exec price vs mid\n')
    out.write('beta_incremental: per-insertion (size_k, I_inst), k>=3\n\n')

    variants = ['beta', 'beta_k3plus', 'beta_Vlocal',
                'beta_I_mid', 'beta_I_inst', 'beta_incremental']

    header = f'  {"Model":<15}'
    for v in variants:
        label = v.replace('beta_', '').replace('beta', 'main')[:8]
        header += f' {label:>8}'
    header += '\n'
    out.write(header)
    out.write('  ' + '-' * (15 + 9 * len(variants)) + '\n')

    for model in MODELS_ORDER:
        if model not in metrics:
            continue
        mc = metrics[model]
        line = f'  {model:<15}'
        for v in variants:
            data = mc.get(v, {})
            val = data.get('beta', np.nan) if isinstance(data, dict) else np.nan
            if np.isnan(val):
                line += '        -'
            else:
                line += f' {val:>8.4f}'
        out.write(line + '\n')
    out.write('\n')

    # Find any variant with beta near 0.5
    out.write('Variants closest to beta=0.5:\n')
    rows = []
    for model in MODELS_ORDER:
        if model not in metrics:
            continue
        mc = metrics[model]
        for v in variants:
            data = mc.get(v, {})
            val = data.get('beta', np.nan) if isinstance(data, dict) else np.nan
            if not np.isnan(val):
                rows.append(dict(model=model, variant=v, beta=val,
                                 dist=abs(val - 0.5)))
    if rows:
        rows_df = pd.DataFrame(rows).nsmallest(15, 'dist')
        for _, row in rows_df.iterrows():
            out.write(f'  {row["model"]:<15} {row["variant"]:<20} '
                      f'beta={row["beta"]:.4f}\n')
    out.write('\n')


def section_conclusion(metrics, stock, pc_df, out):
    """7. Summary conclusion."""
    out.write('\n' + '=' * 70 + '\n')
    out.write('7. CONCLUSION\n')
    out.write('=' * 70 + '\n\n')

    # Collect all beta values we've seen
    all_betas = []
    for model, mc in metrics.items():
        beta_main = mc.get('beta', {})
        if isinstance(beta_main, dict):
            b = beta_main.get('beta', np.nan)
            if not np.isnan(b):
                all_betas.append(dict(model=model, variant='main', beta=b))
        for v in ['beta_k3plus', 'beta_Vlocal', 'beta_I_mid',
                   'beta_I_inst', 'beta_incremental']:
            data = mc.get(v, {})
            val = data.get('beta', np.nan) if isinstance(data, dict) else np.nan
            if not np.isnan(val):
                all_betas.append(dict(model=model, variant=v, beta=val))
        for day, db in mc.get('day_betas', {}).items():
            all_betas.append(dict(model=model, variant=f'day:{day}',
                                  beta=db))
        for mb, mb_b in mc.get('mb_betas', {}).items():
            all_betas.append(dict(model=model, variant=f'mb:{mb}',
                                  beta=mb_b))
        for vol, vol_b in mc.get('vol_betas', {}).items():
            all_betas.append(dict(model=model, variant=f'vol:{vol}',
                                  beta=vol_b))

    if not all_betas:
        out.write('  No beta values collected.\n')
        return

    df = pd.DataFrame(all_betas)
    n_near_05 = ((df['beta'] >= 0.40) & (df['beta'] <= 0.60)).sum()

    out.write(f'Total beta estimates examined: {len(df)}\n')
    out.write(f'In range [0.40, 0.60]: {n_near_05} '
              f'({100 * n_near_05 / len(df):.1f}%)\n')
    out.write(f'In range [0.45, 0.55]: '
              f'{((df["beta"] >= 0.45) & (df["beta"] <= 0.55)).sum()}\n\n')

    out.write(f'Overall beta range: [{df["beta"].min():.4f}, '
              f'{df["beta"].max():.4f}]\n')
    out.write(f'Overall mean beta:  {df["beta"].mean():.4f}\n\n')

    # Best subsets
    df['dist_05'] = (df['beta'] - 0.5).abs()
    top = df.nsmallest(20, 'dist_05')
    out.write('Top 20 subsets closest to beta=0.5:\n')
    out.write(f'  {"Model":<15} {"Variant":<25} {"Beta":>7}\n')
    out.write('  ' + '-' * 49 + '\n')
    for _, row in top.iterrows():
        out.write(f'  {row["model"]:<15} {row["variant"]:<25} '
                  f'{row["beta"]:>7.4f}\n')
    out.write('\n')

    # Is there a natural subset with beta ~ 0.5?
    best = top.iloc[0]
    out.write('ANSWER: ')
    if best['dist_05'] < 0.05:
        out.write(f'YES — {best["model"]}/{best["variant"]} '
                  f'has beta={best["beta"]:.4f}\n')
    elif best['dist_05'] < 0.10:
        out.write(f'MARGINAL — closest is {best["model"]}/{best["variant"]} '
                  f'with beta={best["beta"]:.4f}\n')
    else:
        out.write(f'NO — no subset has beta near 0.5. '
                  f'Closest: {best["model"]}/{best["variant"]} '
                  f'beta={best["beta"]:.4f}\n')
    out.write('\n')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Diag 7: Stratified beta analysis')
    parser.add_argument('--stock', type=str, default='GOOG',
                        help='Stock to analyze (default: GOOG)')
    parser.add_argument('--skip_raw', action='store_true',
                        help='Skip raw pickle loading (per-k analysis)')
    parser.add_argument('--ref_model', type=str, default=REF_MODEL,
                        help=f'Reference model for per-k analysis '
                             f'(default: {REF_MODEL})')
    parser.add_argument('--out_dir', type=str, default='pics_for_investigation',
                        help='Output directory (default: pics_for_investigation)')
    args = parser.parse_args()

    stock = args.stock
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'diag_7_stratified.txt'

    print(f'Diag 7: Stratified beta analysis for {stock}')
    print(f'Output: {out_path}')

    # Load v4 metrics
    v4_src = Path(f'pics_for_v4_300_{stock}')
    print(f'\nLoading v4 metrics from {v4_src}...')
    metrics = load_all_metrics(stock, v4_src)
    print(f'  Loaded {len(metrics)} models: {list(metrics.keys())}')

    # Per-k: load raw pickle (once)
    pc_df = pd.DataFrame()
    if not args.skip_raw:
        raw_pkl = V4_PICKLE_BASE / stock / f'{args.ref_model}.pkl'
        if raw_pkl.exists():
            print(f'\nLoading raw pickle: {raw_pkl} '
                  f'({raw_pkl.stat().st_size / 1e9:.1f} GB)...')
            with open(raw_pkl, 'rb') as f:
                md_raw = pickle.load(f)

            # Extract experiment days for daily params
            daily_params = None
            daily_hl = Path(f'lob_impact/daily_h_l_{stock}.csv')
            if daily_hl.exists():
                exp_days = set()
                for fd in md_raw['data'].values():
                    for d in fd['buy']['days'] + fd['sell']['days']:
                        if d:
                            exp_days.add(d)
                daily_params = load_daily_params(str(daily_hl), exp_days)
                print(f'  Daily params: {len(daily_params)} days')

            print(f'Extracting point cloud from {args.ref_model}...')
            pc_df = extract_point_cloud_from_raw(md_raw, daily_params)
            del md_raw  # free memory
        else:
            print(f'\n  Raw pickle not found: {raw_pkl}')
            print('  Use --skip_raw to skip per-k analysis')
    else:
        print('\n  Skipping raw pickle (--skip_raw)')

    # Write report
    buf = io.StringIO()
    buf.write(f'DIAG 7: STRATIFIED BETA ANALYSIS — {stock}\n')
    buf.write(f'{"=" * 70}\n')
    buf.write(f'Source (v4): pics_for_v4_300_{stock}/\n')
    buf.write(f'Source (v3): pics_for_300_{stock}/\n')
    buf.write(f'Models loaded: {len(metrics)}\n')
    buf.write(f'Raw pickle per-k: '
              f'{"yes" if not pc_df.empty else "no (skipped or unavailable)"}\n')

    section_per_day(metrics, stock, buf)
    section_per_config(metrics, stock, buf)
    section_per_k(pc_df, stock, buf)
    section_cross_stock(buf)
    section_low_high_i(stock, buf)
    section_conditional_betas(metrics, stock, buf)
    section_conclusion(metrics, stock, pc_df, buf)

    report = buf.getvalue()

    # Write to file
    with open(out_path, 'w') as f:
        f.write(report)

    # Also print to stdout
    print('\n' + report)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
