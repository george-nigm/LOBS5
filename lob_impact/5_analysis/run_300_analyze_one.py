#!/usr/bin/env python3
"""
300 · Analyze ONE model from raw pickle → save metrics pickle (small).

Reads raw pickle (~15 GB) from Lustre, applies filtering + daily normalization,
computes all metrics, saves ~1 MB metrics pickle to pics_for_300_{STOCK}/.

Usage:
  python lob_impact/run_300_analyze_one.py --model LobS5 --stock GOOG --daily_hl lob_impact/daily_h_l_GOOG.csv
"""
import argparse, pickle, sys, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
TICK_SIZE = 100
N_BOOTSTRAP = 1000
MIDPRICE_OUTLIER_FACTOR = 1.5

PICKLE_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v3/pickles')

MODELS = [
    'ZeroInsertions', 'Historic', 'Heuristic', 'CST', 'CGAN',
    'LobS5', 'S5-120M', 'S5-4K', 'S5-360M', 'LobS5-v2',
]

# ═══════════════════════════════════════════════════════════════════════
# Daily params
# ═══════════════════════════════════════════════════════════════════════
def load_daily_params(daily_hl_path, exp_days):
    df = pd.read_csv(daily_hl_path)
    if 'day' not in df.columns:
        df['day'] = df['filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    raw_days = sorted(df['day'].dropna().unique())
    exp_days_sorted = sorted(set(exp_days))
    if len(raw_days) != len(exp_days_sorted):
        print(f'  WARNING: {len(raw_days)} raw days vs {len(exp_days_sorted)} exp days')
    day_map = dict(zip(raw_days, exp_days_sorted))
    params = {}
    for _, row in df.iterrows():
        raw_day = row.get('day', None)
        if raw_day is None or pd.isna(raw_day): continue
        exp_day = day_map.get(raw_day, raw_day)
        H, L = row['highest_price'], row['lowest_price']
        V = row['execution_sum']
        # Parkinson (1980): σ = ln(H/L) / (2√(ln2)) = ln(H/L) / 1.6651092
        sigma = np.log(H / L) / 1.6651092 if H > 0 and L > 0 else 1.0
        params[exp_day] = dict(H=H, L=L, V=V, sigma=sigma)
    return params

def collect_days(model_data):
    days = set()
    for fd in model_data['data'].values():
        for d in fd['buy']['days'] + fd['sell']['days']:
            if d: days.add(d)
    return days

def load_aggressive_indices(exp_path):
    """Load aggressive order indices from aggressive_indices.csv in experiment folder."""
    aggr_file = Path(exp_path) / 'aggressive_indices.csv'
    if not aggr_file.exists():
        return None
    vals = np.loadtxt(aggr_file, dtype=int)
    return np.atleast_1d(vals)

# ═══════════════════════════════════════════════════════════════════════
# Filtering
# ═══════════════════════════════════════════════════════════════════════
def get_midprice(book_arr):
    return (book_arr[:, 0].astype(float) + book_arr[:, 2].astype(float)) / 2.0

def is_midprice_outlier(book_arr, factor=MIDPRICE_OUTLIER_FACTOR):
    mp = get_midprice(book_arr)
    mp_valid = mp[mp > 0]
    if len(mp_valid) < 2: return True
    ref = mp_valid[0]
    return np.any(mp_valid > ref * factor) or np.any(mp_valid < ref / factor)

def filter_model(model_data):
    filtered = {}
    n_total, n_skip = 0, 0
    for folder, fd in model_data['data'].items():
        filt = {}
        for direction in ('buy', 'sell'):
            books, msgs, days = [], [], []
            src = fd[direction]
            for j, book in enumerate(src['books']):
                n_total += 1
                if is_midprice_outlier(book):
                    n_skip += 1; continue
                books.append(book)
                days.append(src['days'][j] if j < len(src['days']) else None)
                if j < len(src['msgs']): msgs.append(src['msgs'][j])
            filt[direction] = dict(books=books, msgs=msgs, days=days)
        filtered[folder] = filt
    return filtered, n_total, n_skip

# ═══════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════
def compute_combined_impact(buy_books, sell_books):
    """Returns fractional midprice returns: (mid - mid[0]) / mid[0]."""
    buy_rets, sell_rets = [], []
    for b in buy_books:
        mid = get_midprice(b); mid = mid[mid > 0]
        if len(mid) >= 2: buy_rets.append((mid - mid[0]) / mid[0])
    for b in sell_books:
        mid = get_midprice(b); mid = mid[mid > 0]
        if len(mid) >= 2: sell_rets.append((mid - mid[0]) / mid[0])
    return buy_rets, sell_rets

def extract_point_cloud(filtered_data, grid_df, daily_params=None):
    """VWAP per-order impact: multiple points per sample (one per aggressive order).

    - BUY and SELL processed independently (not paired)
    - VWAP cumulative impact at each aggressive order
    - ref_price = book midprice at first aggressive order
    - Skip configs with < 2 aggressive orders (i=1)
    """
    points = []
    _aggr_cache = {}
    n_configs = len(grid_df)
    print(f'  extract_point_cloud: {n_configs} configs, {len(filtered_data)} folders')

    for cfg_i, (_, row) in enumerate(grid_df.iterrows()):
        folder = row['folder']
        if folder not in filtered_data: continue
        fd = filtered_data[folder]

        # Load aggressive indices from experiment folders on Lustre (cached)
        bp, sp = row['buy_path'], row['sell_path']
        if bp not in _aggr_cache:
            _aggr_cache[bp] = load_aggressive_indices(bp)
        if sp not in _aggr_cache:
            _aggr_cache[sp] = load_aggressive_indices(sp)
        aggr_buy = _aggr_cache[bp]
        aggr_sell = _aggr_cache[sp]
        if aggr_buy is None or aggr_sell is None:
            continue

        # Process BUY and SELL independently (not paired)
        for direction, aggr_idx, msgs_list, books_list, days_list in [
            ('buy',  aggr_buy,  fd['buy']['msgs'],  fd['buy']['books'],  fd['buy']['days']),
            ('sell', aggr_sell, fd['sell']['msgs'], fd['sell']['books'], fd['sell']['days']),
        ]:
            n_aggr = len(aggr_idx)
            if n_aggr < 2:  # skip i=1 configs (need cumulative variation)
                continue

            for j in range(min(len(msgs_list), len(books_list))):
                msg = msgs_list[j]
                book = books_list[j]
                if len(msg) == 0: continue

                # Check all aggressive indices are within bounds
                if aggr_idx.max() >= len(msg) or aggr_idx.max() >= len(book):
                    continue

                # Reference = book midprice at first aggressive order
                ref_mid = (float(book[aggr_idx[0], 0]) + float(book[aggr_idx[0], 2])) / 2.0
                if ref_mid <= 0: continue

                # Sizes and prices at aggressive order positions
                sizes = msg[aggr_idx, 3].astype(float)
                prices = msg[aggr_idx, 4].astype(float)
                if np.any(sizes <= 0) or np.any(prices <= 0): continue

                # Cumulative VWAP
                Q_cum = np.cumsum(sizes)
                vwap = np.cumsum(sizes * prices) / Q_cum

                # Fractional impact
                if direction == 'buy':
                    impact = (vwap - ref_mid) / ref_mid
                else:
                    impact = (ref_mid - vwap) / ref_mid
                impact = np.abs(impact)

                sample_id = f'{folder}_{direction}_{j}'

                # Realized midprice volatility (per-sample, from book)
                mid_all = (book[:, 0].astype(float) + book[:, 2].astype(float)) / 2.0
                mid_pos = mid_all[mid_all > 0]
                if len(mid_pos) > 10:
                    log_ret = np.diff(np.log(mid_pos))
                    sigma_rv = float(np.std(log_ret)) if len(log_ret) > 1 else 1.0
                else:
                    sigma_rv = 1.0

                # V_local: total executed volume in this sample
                exec_mask = (msg[:, 1] == 4)
                V_local = float(msg[exec_mask, 3].astype(float).sum()) if exec_mask.any() else 1.0

                # Per-insertion midprice response + instantaneous impact
                mid_before_arr = np.full(n_aggr, np.nan)
                I_mid_arr = np.full(n_aggr, np.nan)
                I_inst_arr = np.full(n_aggr, np.nan)
                for k in range(n_aggr):
                    mid_before = (float(book[aggr_idx[k], 0]) + float(book[aggr_idx[k], 2])) / 2.0
                    mid_before_arr[k] = mid_before
                    if mid_before <= 0:
                        continue
                    # Midprice after cooling (before next insertion, or at end)
                    if k + 1 < n_aggr:
                        mid_after = (float(book[aggr_idx[k+1], 0]) + float(book[aggr_idx[k+1], 2])) / 2.0
                    else:
                        mid_after = (float(book[-1, 0]) + float(book[-1, 2])) / 2.0
                    if mid_after > 0:
                        I_mid_arr[k] = abs(mid_after - mid_before) / mid_before
                    # Instantaneous execution impact
                    I_inst_arr[k] = abs(prices[k] - mid_before) / mid_before

                for k in range(n_aggr):
                    if impact[k] <= 1e-10 or Q_cum[k] <= 0: continue
                    # Depth at best on the relevant side
                    if direction == 'buy':
                        depth_best = float(book[aggr_idx[k], 1])  # ask volume
                    else:
                        depth_best = float(book[aggr_idx[k], 3])  # bid volume
                    pt = dict(Q=float(Q_cum[k]), I=float(impact[k]),
                              I_mid=float(I_mid_arr[k]) if np.isfinite(I_mid_arr[k]) else np.nan,
                              I_inst=float(I_inst_arr[k]) if np.isfinite(I_inst_arr[k]) else np.nan,
                              size_k=float(sizes[k]),
                              exec_price_k=float(prices[k]),
                              mid_before_k=float(mid_before_arr[k]),
                              V_local=V_local,
                              sigma_rv=sigma_rv,
                              depth_at_best=depth_best,
                              direction=direction,
                              vol=row['vol'], i=row['i'], mb=row['mb'],
                              k=k+1, folder=folder, sample_id=sample_id)
                    if j < len(days_list) and days_list[j]:
                        pt['day'] = days_list[j]
                    if daily_params and pt.get('day') in daily_params:
                        dp = daily_params[pt['day']]
                        pt['daily_vol'] = dp['V']
                        pt['daily_sigma'] = dp['sigma']
                    points.append(pt)
    print(f'  extract_point_cloud: {len(points)} points from {n_configs} configs')
    return pd.DataFrame(points)

def compute_global_beta(pc_df, impact_col='I'):
    """Three beta estimators: OLS-origin, OLS-intercept, ratio.

    - beta_origin:    log(I/sigma) = beta * log(Q/V)  (through origin, legacy)
    - beta_intercept: log(I) = alpha + beta * log(Q/V) (free intercept, unbiased)
    - beta_ratio:     mean(log(I/sigma) / log(Q/V))    (ratio estimator)

    Primary metric: beta = beta_intercept.
    """
    empty = dict(beta=np.nan, beta_origin=np.nan, beta_intercept=np.nan,
                 beta_ratio=np.nan, alpha=np.nan,
                 r2=np.nan, r2_origin=np.nan, r2_intercept=np.nan, n=0)
    if pc_df.empty:
        return empty

    I_vals = pc_df[impact_col].values
    has_daily = 'daily_vol' in pc_df.columns

    if has_daily:
        x = np.log(pc_df['Q'].values / pc_df['daily_vol'].values)
        y_adj = np.log(I_vals / pc_df['daily_sigma'].values)   # for origin + ratio
        y_raw = np.log(I_vals)                                  # for intercept
    else:
        x = np.log(pc_df['Q'].values / 1e6)
        y_adj = np.log(I_vals / 1.0)
        y_raw = np.log(I_vals)

    ok = np.isfinite(x) & np.isfinite(y_adj) & np.isfinite(y_raw) & (x != 0)
    xv, yv_adj, yv_raw = x[ok], y_adj[ok], y_raw[ok]
    if len(xv) < 2:
        return empty

    # 1. OLS through origin: log(I/sigma) = beta * log(Q/V)
    beta_origin = float(np.dot(xv, yv_adj) / np.dot(xv, xv))
    ss_res_o = np.sum((yv_adj - beta_origin * xv)**2)
    ss_tot_o = np.sum(yv_adj**2)
    r2_origin = 1 - ss_res_o / ss_tot_o if ss_tot_o > 0 else 0.0

    # 2. OLS with free intercept: log(I/sigma) = alpha + beta * log(Q/V)
    #    (uses y_adj = log(I/sigma), not y_raw, to avoid omitted-variable bias from σ-V correlation)
    coeffs = np.polyfit(xv, yv_adj, 1)
    beta_intercept = float(coeffs[0])
    alpha = float(coeffs[1])
    yhat = beta_intercept * xv + alpha
    ss_res_i = np.sum((yv_adj - yhat)**2)
    ss_tot_i = np.sum((yv_adj - np.mean(yv_adj))**2)
    r2_intercept = 1 - ss_res_i / ss_tot_i if ss_tot_i > 0 else 0.0

    # 3. Ratio estimator: beta = mean(log(I/sigma) / log(Q/V))
    beta_ratio = float(np.mean(yv_adj / xv))

    return dict(
        beta=beta_intercept,               # PRIMARY
        beta_origin=beta_origin,
        beta_intercept=beta_intercept,
        beta_ratio=beta_ratio,
        alpha=alpha,
        r2=r2_intercept,                   # PRIMARY R²
        r2_origin=r2_origin,
        r2_intercept=r2_intercept,
        n=int(ok.sum()),
    )

def bootstrap_beta(pc_df, n_boot=N_BOOTSTRAP, impact_col='I'):
    """Bootstrap all three beta estimators (origin, intercept, ratio).

    Uses padded group matrix + closed-form OLS for speed (~1s for 400K points).
    """
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

    # Pre-filter once: only finite, nonzero x
    ok_all = np.isfinite(x_all) & np.isfinite(y_adj_all) & np.isfinite(y_raw_all) & (x_all != 0)
    x_f, y_adj_f, y_raw_f = x_all[ok_all], y_adj_all[ok_all], y_raw_all[ok_all]
    sid_f = pc_df['sample_id'].values[ok_all]
    if len(x_f) < 2:
        return empty

    # Build padded group matrix for vectorized resampling
    unique_ids = np.unique(sid_f)
    id_to_int = {sid: i for i, sid in enumerate(unique_ids)}
    group_labels = np.array([id_to_int[s] for s in sid_f])
    n_groups = len(unique_ids)
    group_rows = [np.where(group_labels == g)[0] for g in range(n_groups)]
    max_gs = max(len(g) for g in group_rows)
    # Padded matrix: (n_groups, max_group_size), pad with -1
    gmat = np.full((n_groups, max_gs), -1, dtype=np.int64)
    gsizes = np.zeros(n_groups, dtype=np.int64)
    for g, rows in enumerate(group_rows):
        gmat[g, :len(rows)] = rows
        gsizes[g] = len(rows)

    rng = np.random.default_rng(42)
    b_origin = np.empty(n_boot)
    b_intercept = np.empty(n_boot)
    b_ratio = np.empty(n_boot)

    for b in range(n_boot):
        bg = rng.choice(n_groups, size=n_groups, replace=True)
        # Vectorized row selection via padded matrix
        block = gmat[bg]           # (n_groups, max_gs) — single fancy-index op
        flat = block.ravel()
        row_idx = flat[flat >= 0]  # drop padding

        xo = x_f[row_idx]
        yo_a = y_adj_f[row_idx]
        yo_r = y_raw_f[row_idx]
        n = len(xo)
        if n < 2:
            b_origin[b] = b_intercept[b] = b_ratio[b] = np.nan
            continue

        # Origin: dot(x,y)/dot(x,x)
        b_origin[b] = np.dot(xo, yo_a) / np.dot(xo, xo)
        # Intercept: closed-form OLS (no polyfit overhead)
        sx = xo.sum(); sy = yo_r.sum()
        sxy = np.dot(xo, yo_r); sx2 = np.dot(xo, xo)
        denom = n * sx2 - sx * sx
        b_intercept[b] = (n * sxy - sx * sy) / denom if abs(denom) > 1e-30 else np.nan
        # Ratio: mean(y_adj / x)
        b_ratio[b] = np.mean(yo_a / xo)

    return dict(
        boots=b_intercept,                    # PRIMARY (backward compat)
        boots_origin=b_origin,
        boots_intercept=b_intercept,
        boots_ratio=b_ratio,
    )

def compute_master_curve(buy_books, sell_books, i_val, mb_val, n_vol_u=200):
    L = i_val * (mb_val + 1)
    buy_rets, sell_rets = compute_combined_impact(buy_books, sell_books)
    n_pairs = min(len(buy_rets), len(sell_rets))
    if n_pairs == 0: return None
    max_len = max(max((len(r) for r in buy_rets), default=0),
                  max((len(r) for r in sell_rets), default=0))
    if max_len == 0: return None
    u_grid = np.linspace(0, max_len / L, n_vol_u)
    curves = []
    for j in range(n_pairs):
        br, sr = buy_rets[j], sell_rets[j]
        ml = min(len(br), len(sr))
        combined = (br[:ml] - sr[:ml]) / 2.0
        curves.append(np.interp(u_grid, np.arange(ml) / L, combined))
    arr = np.array(curves)
    return dict(u=u_grid, mean=np.nanmean(arr, axis=0), std=np.nanstd(arr, axis=0), n=n_pairs)

def compute_relaxation_ratio(mc, u_peak=1.0, u_final=3.0):
    if mc is None: return np.nan
    u, m = mc['u'], mc['mean']
    pi = np.argmin(np.abs(u - u_peak)); fi = np.argmin(np.abs(u - u_final))
    if m[pi] < 1e-10: return np.nan
    return m[fi] / m[pi]

def fit_decay(u_grid, mean_curve, u_peak=1.0):
    pi = np.argmin(np.abs(u_grid - u_peak))
    post_u, post_y = u_grid[pi:] - u_peak, mean_curve[pi:]
    if len(post_y) < 5 or post_y[0] < 1e-10: return np.nan
    y_norm = post_y / post_y[0]
    try:
        mask = post_u > 0
        popt, _ = optimize.curve_fit(lambda u, g, c: c*(1+u)**(-g), post_u[mask], y_norm[mask], p0=[0.5,1.0], maxfev=5000)
        return popt[0]
    except: return np.nan

def stability_vote(mc, u_peak=1.0):
    if mc is None: return False
    u, m = mc['u'], mc['mean']
    post = m[np.argmin(np.abs(u - u_peak)):]
    if len(post) < 10: return False
    n = len(post); tail = post[int(n*0.80):]
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) > 1 else 1.0
    m1 = abs(slope) / (abs(np.mean(tail)) + 1e-10) < 0.05
    mid = n // 2
    m2 = abs(np.mean(post[-max(n//8,3):]) - np.mean(post[max(0,mid-n//8):mid+n//8])) / (abs(np.mean(tail))+1e-10) < 0.03
    try:
        popt, _ = optimize.curve_fit(lambda x,a,b,c: a*np.exp(-b*x)+c, np.arange(n), post, p0=[post[0]-post[-1], 0.1, post[-1]], maxfev=5000)
        m3 = (1 - abs(popt[0]*np.exp(-popt[1]*n))/(abs(popt[2])+1e-10)) > 0.95
    except: m3 = False
    return sum([m1, m2, m3]) >= 2

def compute_hurst_dfa(signs, max_lag=200):
    if len(signs) < max_lag * 2: return np.nan
    cumsum = np.cumsum(signs - np.mean(signs))
    scales = np.unique(np.logspace(1, np.log10(max_lag), 20).astype(int))
    scales = scales[scales >= 4]
    flucts = []
    for sc in scales:
        n_seg = len(cumsum) // sc
        if n_seg < 1: continue
        F2 = sum(np.mean((cumsum[s*sc:(s+1)*sc] - np.polyval(np.polyfit(np.arange(sc), cumsum[s*sc:(s+1)*sc], 1), np.arange(sc)))**2) for s in range(n_seg))
        flucts.append(np.sqrt(F2 / n_seg))
    if len(flucts) < 3: return np.nan
    H, _ = np.polyfit(np.log(scales[:len(flucts)]), np.log(flucts), 1)
    return H

def compute_propagator(filtered_data, max_lag=200):
    G_sum, G_count = np.zeros(max_lag), np.zeros(max_lag)
    for fd in filtered_data.values():
        for direction in ('buy', 'sell'):
            for msgs, books in zip(fd[direction]['msgs'], fd[direction]['books']):
                if len(msgs) < max_lag + 10: continue
                mid = get_midprice(books) / TICK_SIZE; dp = np.diff(mid)
                eps = np.zeros(len(msgs))
                eps[(msgs[:,1]==4)&(msgs[:,2]==0)] = 1
                eps[(msgs[:,1]==4)&(msgs[:,2]==1)] = -1
                n = min(len(dp), len(eps)-1)
                for lag in range(min(max_lag, n)):
                    G_sum[lag] += np.sum(dp[lag:lag+n-lag] * eps[:n-lag])
                    G_count[lag] += n - lag
    return np.where(G_count > 0, G_sum / G_count, 0), np.arange(max_lag)

def compute_spread(filtered_data, grid_df):
    all_sc = []
    for _, frow in grid_df.iterrows():
        folder = frow['folder']
        if folder not in filtered_data: continue
        fd = filtered_data[folder]
        all_books = fd['buy']['books'] + fd['sell']['books']
        all_sp = []
        n_inj = frow['i'] * (frow['mb'] + 1)
        for books in all_books:
            if len(books) < 10: continue
            sp = (books[:,0].astype(float) - books[:,2].astype(float)) / TICK_SIZE
            sp[sp <= 0] = np.nan; sp[sp > 100] = np.nan
            all_sp.append(sp)
        if not all_sp: continue
        L = max(n_inj, 1)
        u_grid = np.linspace(0, max(len(s) for s in all_sp) / L, 200)
        interp = [np.interp(u_grid, np.arange(len(s))[~np.isnan(s)] / L, s[~np.isnan(s)]) for s in all_sp if (~np.isnan(s)).sum() >= 5]
        if interp:
            all_sc.append(dict(u=u_grid, mean=np.nanmean(interp, axis=0)))
    if not all_sc: return None
    return dict(u=all_sc[0]['u'], mean=np.nanmean([c['mean'] for c in all_sc], axis=0))

# ═══════════════════════════════════════════════════════════════════════
# Kyle lambda + depth stats
# ═══════════════════════════════════════════════════════════════════════
def compute_kyle_lambda(pc_df):
    """Per-insertion Kyle lambda: ticks moved per share.

    λ_k = |exec_price_k - mid_before_k| / (TICK_SIZE × size_k)

    Expected: k=1 identical for all models (conditioning book),
              k>3 lower for S5 (replenishes book → lower λ).
    """
    if pc_df.empty or 'exec_price_k' not in pc_df.columns:
        return {}
    lam = np.abs(pc_df['exec_price_k'] - pc_df['mid_before_k']) / (TICK_SIZE * pc_df['size_k'])
    valid = np.isfinite(lam) & (lam > 0) & (pc_df['size_k'] > 0) & (pc_df['mid_before_k'] > 0)
    if valid.sum() == 0:
        return {}
    result = {}
    for k_val in sorted(pc_df.loc[valid, 'k'].unique()):
        vals = lam[valid & (pc_df['k'] == k_val)].values
        result[int(k_val)] = dict(
            mean=float(np.mean(vals)), std=float(np.std(vals)),
            median=float(np.median(vals)), n=len(vals))
    return result


def compute_depth_at_best_stats(pc_df):
    """Depth at best level statistics from point cloud (k=1 only).

    k=1 uses the conditioning book (identical across models).
    Output: percentiles of depth distribution.
    """
    if pc_df.empty or 'depth_at_best' not in pc_df.columns:
        return {}
    k1 = pc_df[pc_df['k'] == 1]
    depths = k1['depth_at_best'].dropna().values
    depths = depths[depths > 0]
    if len(depths) == 0:
        return {}
    return dict(
        p25=float(np.percentile(depths, 25)),
        p50=float(np.percentile(depths, 50)),
        p75=float(np.percentile(depths, 75)),
        p90=float(np.percentile(depths, 90)),
        p95=float(np.percentile(depths, 95)),
        p99=float(np.percentile(depths, 99)),
        mean=float(np.mean(depths)),
        n=len(depths))


# ═══════════════════════════════════════════════════════════════════════
# Main: analyze one model
# ═══════════════════════════════════════════════════════════════════════
def run(model, stock, daily_hl_path=None, pickle_base=None, out_dir=None):
    base = pickle_base or PICKLE_BASE
    raw_pkl = base / stock / f'{model}.pkl'
    if not raw_pkl.exists():
        print(f'ERROR: {raw_pkl} not found'); sys.exit(1)

    print(f'Loading {stock}/{model} ({raw_pkl.stat().st_size/1e9:.1f} GB)...')
    with open(raw_pkl, 'rb') as f:
        md = pickle.load(f)
    grid_df = md['grid']

    # Daily params
    daily_params = None
    if daily_hl_path:
        exp_days = collect_days(md)
        daily_params = load_daily_params(daily_hl_path, exp_days)
        print(f'  Daily params: {len(daily_params)} days')

    # Filter
    filtered, n_total, n_skip = filter_model(md)
    print(f'  Filtered: {n_skip}/{n_total} outliers')

    cache = dict(model=model, stock=stock, n_raw=n_total, n_filtered=n_skip)

    # ZeroInsertions
    if model == 'ZeroInsertions':
        drifts = []
        for fd in filtered.values():
            for b in fd['buy']['books'] + fd['sell']['books']:
                mid = get_midprice(b); mid = mid[mid > 0]
                if len(mid) >= 2: drifts.append((mid[-1] - mid[0]) / TICK_SIZE)
        cache['null_drifts'] = np.array(drifts)
        print(f'  Drift: {np.mean(cache["null_drifts"]):.3f} +/- {np.std(cache["null_drifts"]):.3f}')
    else:
        # Point cloud + beta (3 estimators × 3 impact metrics)
        print(f'  [step 1] extract_point_cloud...')
        pc = extract_point_cloud(filtered, grid_df, daily_params)
        print(f'  [step 2] compute_global_beta ({len(pc)} points)...')
        res = compute_global_beta(pc)
        print(f'  [step 3] bootstrap_beta...')
        boot_dict = bootstrap_beta(pc)
        boots = boot_dict['boots']
        ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) > 0 else (np.nan, np.nan)
        ci_origin = (np.nan, np.nan)
        if len(boot_dict['boots_origin']) > 0:
            ci_origin = (np.percentile(boot_dict['boots_origin'], 2.5),
                         np.percentile(boot_dict['boots_origin'], 97.5))

        # Scatter data for figures
        if not pc.empty and 'daily_vol' in pc.columns:
            pc_x = np.log(pc['Q'].values / pc['daily_vol'].values)
            pc_y_adj = np.log(pc['I'].values / pc['daily_sigma'].values)
            pc_y_raw = np.log(pc['I'].values)
        elif not pc.empty:
            pc_x = np.log(pc['Q'].values / 1e6)
            pc_y_adj = np.log(pc['I'].values / 1.0)
            pc_y_raw = np.log(pc['I'].values)
        else:
            pc_x, pc_y_adj, pc_y_raw = np.array([]), np.array([]), np.array([])

        cache['beta'] = dict(
            # Primary (intercept)
            beta=res['beta'], r2=res['r2'], n=res['n'],
            ci_lo=ci[0], ci_hi=ci[1], boots=boots,
            alpha=res['alpha'],
            # Origin (legacy)
            beta_origin=res['beta_origin'], r2_origin=res['r2_origin'],
            ci_origin_lo=ci_origin[0], ci_origin_hi=ci_origin[1],
            boots_origin=boot_dict['boots_origin'],
            # Ratio
            beta_ratio=res['beta_ratio'],
            boots_ratio=boot_dict['boots_ratio'],
            # Scatter data
            pc_x=pc_x, pc_y=pc_y_adj, pc_y_raw=pc_y_raw,
        )
        print(f'  beta_intercept={res["beta"]:.4f} beta_origin={res["beta_origin"]:.4f} '
              f'beta_ratio={res["beta_ratio"]:.4f} R2={res["r2"]:.4f} n={res["n"]}')

        # Beta on alternative impact metrics (I_mid, I_inst)
        for alt_col in ('I_mid', 'I_inst'):
            if alt_col in pc.columns:
                pc_alt = pc[pc[alt_col].notna() & (pc[alt_col] > 1e-10)].copy()
                if not pc_alt.empty:
                    res_alt = compute_global_beta(pc_alt, impact_col=alt_col)
                    boot_alt = bootstrap_beta(pc_alt, impact_col=alt_col)
                    b_alt = boot_alt['boots']
                    ci_alt = (np.percentile(b_alt, 2.5), np.percentile(b_alt, 97.5)) if len(b_alt) > 0 else (np.nan, np.nan)
                    cache[f'beta_{alt_col}'] = dict(
                        beta=res_alt['beta'], beta_origin=res_alt['beta_origin'],
                        beta_ratio=res_alt['beta_ratio'], alpha=res_alt['alpha'],
                        r2=res_alt['r2'], n=res_alt['n'],
                        ci_lo=ci_alt[0], ci_hi=ci_alt[1],
                        boots=b_alt,
                    )
                    print(f'  beta({alt_col}): intercept={res_alt["beta"]:.4f} '
                          f'origin={res_alt["beta_origin"]:.4f} n={res_alt["n"]}')

        # Kyle lambda: ticks per share per insertion
        print(f'  [step 4] compute_kyle_lambda...')
        kyle = compute_kyle_lambda(pc)
        cache['kyle_lambda'] = kyle
        if kyle:
            k1 = kyle.get(1, {})
            kmax = kyle.get(max(kyle.keys()), {})
            print(f'  kyle_lambda: k=1 mean={k1.get("mean",0):.4f}, '
                  f'k={max(kyle.keys())} mean={kmax.get("mean",0):.4f}')

        # Depth at best level statistics (k=1 = conditioning book)
        depth_stats = compute_depth_at_best_stats(pc)
        cache['depth_stats'] = depth_stats
        if depth_stats:
            print(f'  depth_at_best (k=1): p50={depth_stats["p50"]:.0f} '
                  f'p75={depth_stats["p75"]:.0f} p95={depth_stats["p95"]:.0f} '
                  f'n={depth_stats["n"]}')

        # Beta with V_local normalization: x = log(Q / V_local)
        if not pc.empty and 'V_local' in pc.columns:
            pc_vl = pc[pc['V_local'] > 0].copy()
            if not pc_vl.empty:
                pc_vl['daily_vol'] = pc_vl['V_local']
                pc_vl['daily_sigma'] = 1.0
                res_vl = compute_global_beta(pc_vl)
                boot_vl = bootstrap_beta(pc_vl)
                boots_vl = boot_vl['boots']
                ci_vl = (np.percentile(boots_vl, 2.5), np.percentile(boots_vl, 97.5)) \
                    if len(boots_vl) > 0 else (np.nan, np.nan)
                cache['beta_Vlocal'] = dict(
                    beta=res_vl['beta'], r2=res_vl['r2'], n=res_vl['n'],
                    alpha=res_vl['alpha'],
                    ci_lo=ci_vl[0], ci_hi=ci_vl[1], boots=boots_vl)
                print(f'  beta(V_local): {res_vl["beta"]:.4f} R2={res_vl["r2"]:.4f} n={res_vl["n"]}')

        # Conditional beta: k >= 3 only (where model differences accumulate)
        pc_k3 = pc[pc['k'] >= 3].copy()
        if len(pc_k3) >= 10:
            print(f'  [step 5] beta(k>=3) on {len(pc_k3)} points...')
            res_k3 = compute_global_beta(pc_k3)
            boot_k3 = bootstrap_beta(pc_k3)
            boots_k3 = boot_k3['boots']
            ci_k3 = (np.percentile(boots_k3, 2.5), np.percentile(boots_k3, 97.5)) \
                if len(boots_k3) > 0 else (np.nan, np.nan)
            cache['beta_k3plus'] = dict(
                beta=res_k3['beta'], r2=res_k3['r2'], n=res_k3['n'],
                alpha=res_k3['alpha'],
                ci_lo=ci_k3[0], ci_hi=ci_k3[1], boots=boots_k3)
            print(f'  beta(k>=3): {res_k3["beta"]:.4f} R2={res_k3["r2"]:.4f} n={res_k3["n"]}')

        # Incremental beta: per-insertion (size_k, I_inst) instead of cumulative (Q, I_vwap)
        # Uses only k >= 3 where models have diverged
        pc_incr = pc_k3[pc_k3['I_inst'].notna() & (pc_k3['I_inst'] > 1e-10)].copy() \
            if len(pc_k3) > 0 else pd.DataFrame()
        if len(pc_incr) >= 10:
            print(f'  [step 6] beta_incremental on {len(pc_incr)} points...')
            pc_incr = pc_incr.copy()
            pc_incr['Q'] = pc_incr['size_k']    # individual order size, not cumulative
            pc_incr['I'] = pc_incr['I_inst']     # instantaneous impact, not cumulative VWAP
            res_inc = compute_global_beta(pc_incr)
            boot_inc = bootstrap_beta(pc_incr)
            boots_inc = boot_inc['boots']
            ci_inc = (np.percentile(boots_inc, 2.5), np.percentile(boots_inc, 97.5)) \
                if len(boots_inc) > 0 else (np.nan, np.nan)
            cache['beta_incremental'] = dict(
                beta=res_inc['beta'], r2=res_inc['r2'], n=res_inc['n'],
                alpha=res_inc['alpha'],
                ci_lo=ci_inc[0], ci_hi=ci_inc[1], boots=boots_inc)
            print(f'  beta_incr(k>=3): {res_inc["beta"]:.4f} R2={res_inc["r2"]:.4f} '
                  f'n={res_inc["n"]} alpha={res_inc["alpha"]:.4f}')

        # Master curves
        mcs = {}
        for _, frow in grid_df.iterrows():
            folder = frow['folder']
            if folder not in filtered: continue
            fd = filtered[folder]
            mc = compute_master_curve(fd['buy']['books'], fd['sell']['books'], frow['i'], frow['mb'])
            if mc is not None: mcs[folder] = mc
        cache['master_curves'] = mcs

        # Relaxation
        ratios = [compute_relaxation_ratio(mc) for mc in mcs.values()]
        cache['relax'] = [r for r in ratios if not np.isnan(r)]

        # Stability
        sc = sum(1 for mc in mcs.values() if stability_vote(mc))
        cache['stability'] = dict(stable=sc, total=len(mcs), frac=sc/max(len(mcs),1))

        # Gamma + Scorecard
        gammas = [fit_decay(mc['u'], mc['mean']) for mc in mcs.values()]
        gammas = [g for g in gammas if not np.isnan(g)]
        beta_val = res['beta']
        r_mean = np.mean(cache['relax']) if cache['relax'] else np.nan
        gamma = np.mean(gammas) if gammas else np.nan
        b_perm_sc = beta_val * (r_mean if not np.isnan(r_mean) else 1.0)
        A = beta_val < 1
        B = 0.7 <= b_perm_sc <= 1.3 if not np.isnan(b_perm_sc) else False
        C = 0.3 <= gamma <= 1.0 if not np.isnan(gamma) else False
        D = 0.5 <= r_mean <= 1.0 if not np.isnan(r_mean) else False
        E = beta_val <= 1/(1+2*gamma) if not np.isnan(gamma) else False
        cache['scorecard'] = dict(b_perm=b_perm_sc, r=r_mean, gamma=gamma,
                                   A=A, B=B, C=C, D=D, E=E, score=sum([A,B,C,D,E]))

        # Hurst
        all_signs = []
        for fd in filtered.values():
            for direction in ('buy', 'sell'):
                for msgs in fd[direction]['msgs']:
                    mask = msgs[:, 1] == 4
                    if mask.sum() >= 2: all_signs.append(np.where(msgs[mask, 2] == 0, 1, -1))
        if all_signs:
            concat = np.concatenate(all_signs)
            if len(concat) >= 100: cache['hurst'] = compute_hurst_dfa(concat)

        # Propagator
        G, lags = compute_propagator(filtered)
        cache['propagator'] = dict(G=G, lags=lags)

        # Spread
        sp = compute_spread(filtered, grid_df)
        if sp: cache['spread'] = sp

        # Beta vs mb, vol
        mb_b, vol_b = {}, {}
        for _, frow in grid_df.iterrows():
            folder = frow['folder']
            if folder not in filtered: continue
            pc_f = extract_point_cloud({folder: filtered[folder]}, pd.DataFrame([frow]), daily_params)
            if pc_f.empty: continue
            res_f = compute_global_beta(pc_f)
            if not np.isnan(res_f['beta']):
                mb_b.setdefault(frow['mb'], []).append(res_f['beta'])
                vol_b.setdefault(frow['vol'], []).append(res_f['beta'])
        cache['mb_betas'] = {k: np.mean(v) for k, v in mb_b.items()}
        cache['vol_betas'] = {k: np.mean(v) for k, v in vol_b.items()}

        # Per-day beta
        if not pc.empty and 'day' in pc.columns:
            day_betas = {}
            for day in pc['day'].dropna().unique():
                pc_day = pc[pc['day'] == day]
                if len(pc_day) < 5: continue
                b = compute_global_beta(pc_day)
                if not np.isnan(b['beta']): day_betas[day] = b['beta']
            cache['day_betas'] = day_betas

        # Perm/temp
        perm_pts, temp_pts = [], []
        for _, frow in grid_df.iterrows():
            folder = frow['folder']
            if folder not in mcs: continue
            mc = mcs[folder]; u, m = mc['u'], mc['mean']
            I_peak = float(np.interp(1.0, u, m))
            I_final = float(np.interp(3.0, u, m))
            Q = frow['i'] * frow['vol']
            if I_peak > 1e-10 and I_final > 1e-10 and Q > 0:
                perm_pts.append(dict(Q=Q, I_perm=I_final))
                temp_pts.append(dict(Q=Q, I_temp=max(I_peak - I_final, 1e-10)))
        cache['perm_temp'] = dict(perm=perm_pts, temp=temp_pts)
        if perm_pts:
            x = np.log(np.array([p['Q'] for p in perm_pts]) / 1e6)
            y = np.log(np.array([p['I_perm'] for p in perm_pts]))
            if len(x) > 2 and np.any(x != 0):
                coeffs = np.polyfit(x, y, 1)
                cache['beta_perm'] = float(coeffs[0])

    # Save metrics pickle (small!) to /home
    out_dir = Path(out_dir) if out_dir else Path(f'pics_for_300_{stock}')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'{model}.metrics.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f'Saved: {out_path} ({out_path.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='300 · Analyze one model')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--stock', type=str, default='GOOG')
    parser.add_argument('--daily_hl', type=str, default=None)
    parser.add_argument('--pickle_base', type=str, default=None,
                        help='Override PICKLE_BASE path (e.g. for v4 experiments)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Override output directory (default: pics_for_300_{STOCK})')
    args = parser.parse_args()
    pb = Path(args.pickle_base) if args.pickle_base else None
    run(args.model, args.stock, args.daily_hl, pickle_base=pb, out_dir=args.out_dir)
