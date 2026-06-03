#!/usr/bin/env python3
"""
300 · Full analysis + figures from raw pickle data.

Reads per-model pickles (raw books/msgs), applies filtering + daily normalization,
computes all metrics, generates publication figures.

Usage:
  python lob_impact/run_300_figures.py --stock GOOG --daily_hl lob_impact/daily_h_l_GOOG.csv
  python lob_impact/run_300_figures.py --stock INTC --daily_hl lob_impact/daily_h_l_INTC.csv --figs 3,12,14
  python lob_impact/run_300_figures.py --list
"""
import argparse, pickle, sys, re, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
N_BOOTSTRAP       = 1000
TICK_SIZE         = 100
MIDPRICE_OUTLIER_FACTOR = 1.5
DPI               = 200
MAX_SCATTER       = 800

MODEL_META = OrderedDict([
    ('ZeroInsertions',  dict(color='#7CAE7A', ls=':',       marker='D', ms=5)),
    ('Historic',        dict(color='#90939C', ls=':',       marker='x', ms=6)),
    ('Heuristic',       dict(color='#546884', ls='-.',      marker='d', ms=5)),
    ('CST',             dict(color='#213552', ls='--',      marker='^', ms=5)),
    ('CGAN',            dict(color='#7B4F9E', ls=(0,(8,4)), marker='s', ms=5)),
    ('LobS5',           dict(color='#C88A3A', ls='-',       marker='o', ms=5)),
    ('S5-120M',         dict(color='#D95F02', ls='-',       marker='v', ms=5)),
    ('S5-4K',           dict(color='#5B7BBF', ls='-',       marker='*', ms=7)),
    ('S5-4K-4000',      dict(color='#1B4F72', ls='-',       marker='P', ms=6)),
    ('S5-360M',         dict(color='#B5446E', ls='-',       marker='H', ms=6)),
    ('LobS5-v2',        dict(color='#2CA02C', ls='-',       marker='P', ms=6)),
])

MODEL_INFO = OrderedDict([
    ('ZeroInsertions',  dict(desc='Null baseline (no insertions)')),
    ('Historic',        dict(desc='Replay historical messages')),
    ('Heuristic',       dict(desc='Replay + proportional price shift')),
    ('CST',             dict(desc='Cont-Stoikov-Talreja parametric')),
    ('CGAN',            dict(desc='Conditional GAN, ~2M params')),
    ('LobS5',           dict(desc='S5 7M, ctx=500, GOOG')),
    ('S5-120M',         dict(desc='S5 120M, ctx=500, GOOG')),
    ('S5-4K',           dict(desc='S5 7M, ctx=4000, GOOG')),
    ('S5-4K-4000',      dict(desc='S5 7M, ctx=4000, n_cond=4000')),
    ('S5-360M',         dict(desc='S5 360M, ctx=500, GOOG')),
    ('LobS5-v2',        dict(desc='S5 7M v2 encoding, ctx=500')),
])

def _c(m):  return MODEL_META[m]['color']
def _ls(m): return MODEL_META[m]['ls']
def _mk(m): return MODEL_META[m]['marker']
def _ms(m): return MODEL_META[m]['ms']

FIGURE_CATALOG = OrderedDict([
    (0,  'Null Baseline Drift'),
    (1,  'Master Curves'),
    (2,  'Average Master Curve'),
    (3,  'Beta Regression Lines'),
    (4,  'Bootstrap Beta Distributions'),
    (5,  'Relaxation Ratio'),
    (6,  'Fraction Stable'),
    (7,  'Hurst Exponent'),
    (8,  'Propagator G(l)'),
    (9,  'Spread Dynamics'),
    (10, 'Beta vs mb'),
    (11, 'Beta vs Volume'),
    (12, 'Per-Day Beta'),
    (13, 'Perm Temp Decomposition'),
    (14, 'No-Arb Scatter'),
    (15, 'Beta Estimator Comparison'),
    (16, 'Per-Insertion Midprice Response'),
    (17, 'Kyle Lambda per Insertion'),
    (18, 'Beta Vlocal Comparison'),
    (19, 'Conditional Beta Comparison'),
])


# ═══════════════════════════════════════════════════════════════════════
# Load raw pickles
# ═══════════════════════════════════════════════════════════════════════
def load_metrics_pickles(stock, src_dir=None):
    """Load small metrics pickles from pics_for_300_{STOCK}/ or custom dir."""
    src = Path(src_dir) if src_dir else Path(f'pics_for_300_{stock}')
    if not src.exists():
        print(f'ERROR: {src} not found.'); sys.exit(1)

    R = dict(models=[], impact_models=[], stock=stock,
             beta={}, master_curves={}, relax={}, stability={},
             scorecard={}, hurst={}, propagator={}, spread={},
             mb_betas={}, vol_betas={}, day_betas={},
             perm_temp={}, beta_perm={},
             beta_I_mid={}, beta_I_inst={},
             kyle_lambda={}, depth_stats={}, beta_Vlocal={},
             beta_k3plus={}, beta_incremental={})

    loaded = 0
    for model in MODEL_META:
        pkl = src / f'{model}.metrics.pkl'
        if not pkl.exists():
            print(f'  [skip] {model}'); continue
        with open(pkl, 'rb') as f:
            mc = pickle.load(f)
        R['models'].append(model)
        loaded += 1

        if model == 'ZeroInsertions':
            R['null_drifts'] = mc.get('null_drifts', np.array([]))
        else:
            R['impact_models'].append(model)
            R['beta'][model] = mc.get('beta', {})
            R['master_curves'][model] = mc.get('master_curves', {})
            R['relax'][model] = mc.get('relax', [])
            R['stability'][model] = mc.get('stability', {})
            R['scorecard'][model] = mc.get('scorecard', {})
            R['hurst'][model] = mc.get('hurst', np.nan)
            R['propagator'][model] = mc.get('propagator', {})
            R['spread'][model] = mc.get('spread', {})
            R['mb_betas'][model] = mc.get('mb_betas', {})
            R['vol_betas'][model] = mc.get('vol_betas', {})
            R['day_betas'][model] = mc.get('day_betas', {})
            R['perm_temp'][model] = mc.get('perm_temp', {})
            R['beta_perm'][model] = mc.get('beta_perm', np.nan)
            R['beta_I_mid'][model] = mc.get('beta_I_mid', {})
            R['beta_I_inst'][model] = mc.get('beta_I_inst', {})
            R['kyle_lambda'][model] = mc.get('kyle_lambda', {})
            R['depth_stats'][model] = mc.get('depth_stats', {})
            R['beta_Vlocal'][model] = mc.get('beta_Vlocal', {})
            R['beta_k3plus'][model] = mc.get('beta_k3plus', {})
            R['beta_incremental'][model] = mc.get('beta_incremental', {})

    print(f'Loaded {loaded}/{len(MODEL_META)} metrics from {src}/')
    return R


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
        print(f'  WARNING: {len(raw_days)} raw days vs {len(exp_days_sorted)} experiment days')
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
    print(f'  Daily params: {len(params)} days')
    return params


def collect_experiment_days(raw_models):
    days = set()
    for model_data in raw_models.values():
        for fd in model_data['data'].values():
            for d in fd['buy']['days'] + fd['sell']['days']:
                if d: days.add(d)
    return days


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


def filter_model_data(model_data):
    """Filter one model's raw data. Returns filtered copy + stats."""
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
                if j < len(src['msgs']):
                    msgs.append(src['msgs'][j])
            filt[direction] = dict(books=books, msgs=msgs, days=days)
        filtered[folder] = filt
    return filtered, n_total, n_skip


# ═══════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════
def load_aggressive_indices(exp_path):
    """Load aggressive order indices from aggressive_indices.csv in experiment folder."""
    aggr_file = Path(exp_path) / 'aggressive_indices.csv'
    if not aggr_file.exists():
        return None
    vals = np.loadtxt(aggr_file, dtype=int)
    return np.atleast_1d(vals)

def compute_combined_impact(buy_books, sell_books):
    buy_rets, sell_rets = [], []
    for b in buy_books:
        mid = get_midprice(b); mid = mid[mid > 0]
        if len(mid) >= 2: buy_rets.append((mid - mid[0]) / TICK_SIZE)
    for b in sell_books:
        mid = get_midprice(b); mid = mid[mid > 0]
        if len(mid) >= 2: sell_rets.append((mid - mid[0]) / TICK_SIZE)
    return buy_rets, sell_rets


def extract_point_cloud(filtered_data, grid_df, daily_params=None):
    """VWAP per-order impact: multiple points per sample (one per aggressive order).
    ref_price = book midprice at first aggressive order."""
    points = []
    _aggr_cache = {}

    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in filtered_data: continue
        fd = filtered_data[folder]

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
            ('buy',  aggr_buy,  fd['buy']['msgs'],  fd['buy']['books'],  fd['buy']['days']),
            ('sell', aggr_sell, fd['sell']['msgs'], fd['sell']['books'], fd['sell']['days']),
        ]:
            n_aggr = len(aggr_idx)
            if n_aggr < 2:
                continue
            for j in range(min(len(msgs_list), len(books_list))):
                msg = msgs_list[j]
                book = books_list[j]
                if len(msg) == 0: continue
                if aggr_idx.max() >= len(msg) or aggr_idx.max() >= len(book): continue
                # Reference = book midprice at first aggressive order
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
                sample_id = f'{folder}_{direction}_{j}'

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
                              vol=row['vol'], i=row['i'], mb=row['mb'],
                              k=k+1, folder=folder, sample_id=sample_id)
                    if j < len(days_list) and days_list[j]:
                        pt['day'] = days_list[j]
                    if daily_params and pt.get('day') in daily_params:
                        dp = daily_params[pt['day']]
                        pt['daily_vol'] = dp['V']
                        pt['daily_sigma'] = dp['sigma']
                    points.append(pt)
    return pd.DataFrame(points)


def compute_global_beta(pc_df, impact_col='I'):
    """Three beta estimators: OLS-origin, OLS-intercept, ratio.

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
    )


def bootstrap_beta(pc_df, n_boot=N_BOOTSTRAP, impact_col='I'):
    """Bootstrap all three beta estimators.

    Uses padded group matrix + closed-form OLS for speed.
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


def compute_master_curve(buy_books, sell_books, i_val, mb_val, n_vol_u=200):
    L = i_val * (mb_val + 1)
    buy_rets, sell_rets = compute_combined_impact(buy_books, sell_books)
    n_pairs = min(len(buy_rets), len(sell_rets))
    if n_pairs == 0: return None
    max_len = max(max((len(r) for r in buy_rets), default=0),
                  max((len(r) for r in sell_rets), default=0))
    if max_len == 0: return None
    u_grid = np.linspace(0, max_len / L, n_vol_u)
    combined_curves = []
    for j in range(n_pairs):
        br, sr = buy_rets[j], sell_rets[j]
        ml = min(len(br), len(sr))
        combined = (br[:ml] - sr[:ml]) / 2.0
        u_raw = np.arange(ml) / L
        combined_curves.append(np.interp(u_grid, u_raw, combined))
    curves = np.array(combined_curves)
    return dict(u=u_grid, mean=np.nanmean(curves, axis=0),
                std=np.nanstd(curves, axis=0), n=n_pairs)


def compute_relaxation_ratio(mc, u_peak=1.0, u_final=3.0):
    if mc is None: return np.nan
    u, m = mc['u'], mc['mean']
    pi, fi = np.argmin(np.abs(u - u_peak)), np.argmin(np.abs(u - u_final))
    if m[pi] < 1e-10: return np.nan
    return m[fi] / m[pi]


def fit_decay(u_grid, mean_curve, u_peak=1.0):
    pi = np.argmin(np.abs(u_grid - u_peak))
    post_u, post_y = u_grid[pi:] - u_peak, mean_curve[pi:]
    if len(post_y) < 5 or post_y[0] < 1e-10: return np.nan
    y_norm = post_y / post_y[0]
    try:
        mask = post_u > 0
        popt, _ = optimize.curve_fit(lambda u, g, c: c*(1+u)**(-g),
                                      post_u[mask], y_norm[mask], p0=[0.5,1.0], maxfev=5000)
        return popt[0]
    except: return np.nan


def stability_vote(mc, u_peak=1.0):
    if mc is None: return False
    u, m = mc['u'], mc['mean']
    post = m[np.argmin(np.abs(u - u_peak)):]
    if len(post) < 10: return False
    n = len(post)
    tail = post[int(n*0.80):]
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) > 1 else 1.0
    m1 = abs(slope) / (abs(np.mean(tail)) + 1e-10) < 0.05
    mid = n // 2
    m2 = abs(np.mean(post[-max(n//8,3):]) - np.mean(post[max(0,mid-n//8):mid+n//8])) / (abs(np.mean(tail))+1e-10) < 0.03
    try:
        popt, _ = optimize.curve_fit(lambda x,a,b,c: a*np.exp(-b*x)+c, np.arange(n), post,
                                      p0=[post[0]-post[-1], 0.1, post[-1]], maxfev=5000)
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
        F2 = sum(np.mean((cumsum[s*sc:(s+1)*sc] -
                 np.polyval(np.polyfit(np.arange(sc), cumsum[s*sc:(s+1)*sc], 1), np.arange(sc)))**2)
                 for s in range(n_seg))
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
                mid = get_midprice(books) / TICK_SIZE
                dp = np.diff(mid)
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
        interp = []
        for s in all_sp:
            v = ~np.isnan(s)
            if v.sum() < 5: continue
            interp.append(np.interp(u_grid, np.arange(len(s))[v] / L, s[v]))
        if interp:
            arr = np.array(interp)
            all_sc.append(dict(u=u_grid, mean=np.nanmean(arr, axis=0)))
    if not all_sc: return None
    u = all_sc[0]['u']
    avg = np.nanmean([c['mean'] for c in all_sc], axis=0)
    return dict(u=u, mean=avg)


# ═══════════════════════════════════════════════════════════════════════
# Full analysis pipeline (from raw data → metrics)
# ═══════════════════════════════════════════════════════════════════════
def analyze_all(raw_models, daily_params=None):
    """Run full analysis on all models. Returns combined results dict."""
    R = dict(models=[], impact_models=[],
             beta={}, master_curves={}, relax={}, stability={},
             scorecard={}, hurst={}, propagator={}, spread={},
             mb_betas={}, vol_betas={}, day_betas={},
             perm_temp={}, beta_perm={}, pc_data={})

    for model, md in raw_models.items():
        R['models'].append(model)
        grid_df = md['grid']

        # Filter
        filtered, n_total, n_skip = filter_model_data(md)
        if n_skip:
            print(f'  {model}: filtered {n_skip}/{n_total} outliers')

        if model == 'ZeroInsertions':
            drifts = []
            for fd in filtered.values():
                for b in fd['buy']['books'] + fd['sell']['books']:
                    mid = get_midprice(b); mid = mid[mid > 0]
                    if len(mid) >= 2: drifts.append((mid[-1] - mid[0]) / TICK_SIZE)
            R['null_drifts'] = np.array(drifts)
            continue

        R['impact_models'].append(model)

        # Point cloud + beta (3 estimators × 3 impact metrics)
        pc = extract_point_cloud(filtered, grid_df, daily_params)
        res = compute_global_beta(pc)
        boot_dict = bootstrap_beta(pc)
        boots = boot_dict['boots']
        ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) > 0 else (np.nan, np.nan)
        ci_origin = (np.nan, np.nan)
        if len(boot_dict['boots_origin']) > 0:
            ci_origin = (np.percentile(boot_dict['boots_origin'], 2.5),
                         np.percentile(boot_dict['boots_origin'], 97.5))

        if not pc.empty and 'daily_vol' in pc.columns:
            pc_x = np.log(pc['Q'].values / pc['daily_vol'].values)
            pc_y = np.log(pc['I'].values / pc['daily_sigma'].values)
            pc_y_raw = np.log(pc['I'].values)
        elif not pc.empty:
            pc_x = np.log(pc['Q'].values / 1e6)
            pc_y = np.log(pc['I'].values / 1.0)
            pc_y_raw = np.log(pc['I'].values)
        else:
            pc_x, pc_y, pc_y_raw = np.array([]), np.array([]), np.array([])

        R['beta'][model] = dict(
            beta=res['beta'], r2=res['r2'], n=res['n'],
            ci_lo=ci[0], ci_hi=ci[1], boots=boots, alpha=res['alpha'],
            beta_origin=res['beta_origin'], r2_origin=res['r2_origin'],
            ci_origin_lo=ci_origin[0], ci_origin_hi=ci_origin[1],
            boots_origin=boot_dict['boots_origin'],
            beta_ratio=res['beta_ratio'],
            boots_ratio=boot_dict['boots_ratio'],
            pc_x=pc_x, pc_y=pc_y, pc_y_raw=pc_y_raw,
        )
        R['pc_data'][model] = pc
        print(f'  {model}: beta_intercept={res["beta"]:.4f} beta_origin={res["beta_origin"]:.4f} '
              f'beta_ratio={res["beta_ratio"]:.4f} R2={res["r2"]:.4f} n={res["n"]}')

        # Master curves
        mcs = {}
        for _, frow in grid_df.iterrows():
            folder = frow['folder']
            if folder not in filtered: continue
            fd = filtered[folder]
            mc = compute_master_curve(fd['buy']['books'], fd['sell']['books'],
                                      frow['i'], frow['mb'])
            if mc is not None: mcs[folder] = mc
        R['master_curves'][model] = mcs

        # Relaxation
        ratios = [compute_relaxation_ratio(mc) for mc in mcs.values()]
        R['relax'][model] = [r for r in ratios if not np.isnan(r)]

        # Stability
        sc = sum(1 for mc in mcs.values() if stability_vote(mc))
        tc = len(mcs)
        R['stability'][model] = dict(stable=sc, total=tc, frac=sc/max(tc,1))

        # Decay gamma
        gammas = [fit_decay(mc['u'], mc['mean']) for mc in mcs.values()]
        gammas = [g for g in gammas if not np.isnan(g)]

        # Scorecard
        beta_val = res['beta']
        r_mean = np.mean(R['relax'][model]) if R['relax'][model] else np.nan
        gamma = np.mean(gammas) if gammas else np.nan
        b_perm_sc = beta_val * (r_mean if not np.isnan(r_mean) else 1.0)
        A = beta_val < 1
        B = 0.7 <= b_perm_sc <= 1.3 if not np.isnan(b_perm_sc) else False
        C = 0.3 <= gamma <= 1.0 if not np.isnan(gamma) else False
        D = 0.5 <= r_mean <= 1.0 if not np.isnan(r_mean) else False
        E = beta_val <= 1/(1+2*gamma) if not np.isnan(gamma) else False
        R['scorecard'][model] = dict(b_perm=b_perm_sc, r=r_mean, gamma=gamma,
                                      A=A, B=B, C=C, D=D, E=E, score=sum([A,B,C,D,E]))

        # Hurst
        all_signs = []
        for fd in filtered.values():
            for direction in ('buy', 'sell'):
                for msgs in fd[direction]['msgs']:
                    mask = msgs[:, 1] == 4
                    if mask.sum() >= 2:
                        all_signs.append(np.where(msgs[mask, 2] == 0, 1, -1))
        if all_signs:
            concat = np.concatenate(all_signs)
            if len(concat) >= 100:
                R['hurst'][model] = compute_hurst_dfa(concat)

        # Propagator
        G, lags = compute_propagator(filtered)
        R['propagator'][model] = dict(G=G, lags=lags)

        # Spread
        sp = compute_spread(filtered, grid_df)
        if sp: R['spread'][model] = sp

        # Beta vs mb, vol
        mb_b, vol_b = {}, {}
        for _, frow in grid_df.iterrows():
            folder = frow['folder']
            if folder not in filtered: continue
            fd_single = {folder: filtered[folder]}
            pc_f = extract_point_cloud(fd_single, pd.DataFrame([frow]), daily_params)
            if pc_f.empty: continue
            res_f = compute_global_beta(pc_f)
            if not np.isnan(res_f['beta']):
                mb_b.setdefault(frow['mb'], []).append(res_f['beta'])
                vol_b.setdefault(frow['vol'], []).append(res_f['beta'])
        R['mb_betas'][model] = {k: np.mean(v) for k, v in mb_b.items()}
        R['vol_betas'][model] = {k: np.mean(v) for k, v in vol_b.items()}

        # Per-day beta
        if not pc.empty and 'day' in pc.columns:
            day_betas = {}
            for day in pc['day'].dropna().unique():
                pc_day = pc[pc['day'] == day]
                if len(pc_day) < 5: continue
                b = compute_global_beta(pc_day)
                if not np.isnan(b['beta']):
                    day_betas[day] = b['beta']
            R['day_betas'][model] = day_betas

        # Perm/temp decomposition
        perm_pts, temp_pts = [], []
        for _, frow in grid_df.iterrows():
            folder = frow['folder']
            if folder not in mcs: continue
            mc = mcs[folder]
            u, m = mc['u'], mc['mean']
            I_peak = float(np.interp(1.0, u, m))
            I_final = float(np.interp(3.0, u, m))
            Q = frow['i'] * frow['vol']
            if I_peak > 1e-10 and I_final > 1e-10 and Q > 0:
                perm_pts.append(dict(Q=Q, I_perm=I_final))
                temp_pts.append(dict(Q=Q, I_temp=max(I_peak - I_final, 1e-10)))
        R['perm_temp'][model] = dict(perm=perm_pts, temp=temp_pts)
        if perm_pts:
            x = np.log(np.array([p['Q'] for p in perm_pts]) / 1e6)
            y = np.log(np.array([p['I_perm'] for p in perm_pts]))
            if len(x) > 2 and np.any(x != 0):
                coeffs = np.polyfit(x, y, 1)
                R['beta_perm'][model] = float(coeffs[0])

    return R


# ═══════════════════════════════════════════════════════════════════════
# matplotlib setup + save
# ═══════════════════════════════════════════════════════════════════════
SAVE_DIR = None

def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 13, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
        'axes.labelsize': 13, 'legend.fontsize': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    })
    return plt

def add_metadata_box(fig, stock):
    """Add small metadata annotation at bottom-right of figure."""
    fig.text(0.99, 0.01, f'Test: {stock}  |  VWAP per-order impact',
             fontsize=7, color='gray', style='italic',
             ha='right', va='bottom', transform=fig.transFigure)

def save(plt, fig, num, name, w=10.8, h=5.0):
    fig.set_size_inches(w, h); fig.tight_layout()
    fig.savefig(SAVE_DIR / f'{num}. {name}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [{num:>2}] {name}')


# ═══════════════════════════════════════════════════════════════════════
# Figure functions (same as before, using R dict)
# ═══════════════════════════════════════════════════════════════════════

def fig_0(plt, R):
    if 'null_drifts' not in R: return
    fig, ax = plt.subplots()
    ax.hist(R['null_drifts'], bins=100, color=_c('ZeroInsertions'), edgecolor='white', lw=0.3)
    ax.axvline(0, ls='--', color='black', lw=1)
    ax.set(title='Null Baseline Drift', xlabel='Drift (ticks)', ylabel='Count')
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 0, 'Null Baseline Drift', w=7, h=4.5)

def fig_1(plt, R):
    imp = R['impact_models']
    available = [m for m in imp if R['master_curves'].get(m)]
    if not available: return
    nc = 2; nr = math.ceil(len(available) / nc)
    palette = [f'C{i}' for i in range(30)]
    fig, axes = plt.subplots(nr, nc, figsize=(13, 4*nr), sharex=True, squeeze=False)
    for idx, label in enumerate(available):
        r, c = idx // nc, idx % nc; ax = axes[r][c]
        for fi, mc in enumerate(R['master_curves'][label].values()):
            mask = mc['u'] <= 3.0
            ax.plot(mc['u'][mask], mc['mean'][mask], color=palette[fi%len(palette)], lw=0.8, alpha=0.6)
        ax.axvline(1.0, ls='--', color='gray', lw=0.8)
        ax.set_title(label, fontsize=12, fontweight='bold')
    for idx in range(len(available), nr*nc): axes[idx//nc][idx%nc].set_visible(False)
    fig.suptitle('Cross-Model: Master Curves', fontsize=16, fontweight='bold', y=1.01)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 1, 'Master Curves', w=13, h=4*nr)

def fig_2(plt, R):
    fig, ax = plt.subplots()
    for label in R['impact_models']:
        mcs = R['master_curves'].get(label, {})
        if not mcs: continue
        all_m = [mc['mean'] for mc in mcs.values()]
        u = list(mcs.values())[0]['u']
        avg, std = np.nanmean(all_m, axis=0), np.nanstd(all_m, axis=0)
        ax.plot(u, avg, color=_c(label), ls=_ls(label), lw=3, label=label)
        ax.fill_between(u, avg-std, avg+std, color=_c(label), alpha=0.08)
    ax.axvline(1.0, ls='--', color='gray', lw=1.5)
    ax.set(title='Average Master Curve', xlabel='u', ylabel=r'$I_{\rm norm}(u)$')
    ax.legend(fontsize=9, ncol=2)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 2, 'Average Master Curve', w=11, h=5)

def fig_3(plt, R):
    """Beta regression: intercept (primary, solid) vs origin (legacy, dashed)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    rng = np.random.default_rng(42)

    for label in R['impact_models']:
        br = R['beta'].get(label, {})
        x, y_adj = br.get('pc_x', []), br.get('pc_y', [])
        y_raw = br.get('pc_y_raw', [])
        if len(x) == 0: continue
        n = len(x)
        if n > MAX_SCATTER:
            idx = rng.choice(n, MAX_SCATTER, replace=False)
        else:
            idx = np.arange(n)

        # Left panel: OLS with free intercept — log(I) = alpha + beta*log(Q/V)
        if len(y_raw) > 0:
            ax1.scatter(x[idx], y_raw[idx], s=3, color=_c(label), alpha=0.12)
            xl = np.array([x.min(), x.max()])
            alpha = br.get('alpha', 0.0)
            ax1.plot(xl, br['beta']*xl + alpha, color=_c(label), ls=_ls(label), lw=3,
                     label=f"{label}: {br['beta']:.3f}")

        # Right panel: OLS through origin (legacy) — log(I/sigma) = beta*log(Q/V)
        ax2.scatter(x[idx], y_adj[idx], s=3, color=_c(label), alpha=0.12)
        xl = np.array([x.min(), x.max()])
        beta_o = br.get('beta_origin', br.get('beta', np.nan))
        ax2.plot(xl, beta_o*xl, color=_c(label), ls=_ls(label), lw=3,
                 label=f"{label}: {beta_o:.3f}")

    ax1.plot([-15,-3], [0.5*-15, 0.5*-3], 'k--', lw=2, label=r'$\beta$=0.5')
    ax1.set(title=r'Free Intercept: $\log I = \alpha + \beta \log(Q/V)$',
            xlabel='log(Q/V)', ylabel='log(I)')
    ax1.legend(fontsize=8, loc='lower left', ncol=2)

    ax2.plot([-15,-3], [0.5*-15, 0.5*-3], 'k--', lw=2, label=r'$\beta$=0.5')
    ax2.set(title=r'Through Origin: $\log(I/\sigma) = \beta \log(Q/V)$',
            xlabel='log(Q/V)', ylabel=r'log(I/$\sigma$)')
    ax2.legend(fontsize=8, loc='lower left', ncol=2)

    fig.suptitle('Beta Regression: Intercept vs Origin', fontsize=16, fontweight='bold', y=1.01)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 3, 'Beta Regression Lines', w=18, h=7)

def fig_4(plt, R):
    """Bootstrap distributions: intercept (top), origin (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for label in R['impact_models']:
        br = R['beta'].get(label, {})
        boots_i = br.get('boots', br.get('boots_intercept', []))
        boots_o = br.get('boots_origin', [])
        if len(boots_i) > 0:
            ax1.hist(boots_i, bins=50, alpha=0.45, color=_c(label), label=label, edgecolor='none')
        if len(boots_o) > 0:
            ax2.hist(boots_o, bins=50, alpha=0.45, color=_c(label), label=label, edgecolor='none')
    for ax in (ax1, ax2):
        ax.axvline(0.5, ls='--', color='red', lw=2.5)
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylabel('Count')
    ax1.set_title(r'$\beta$ (free intercept)', fontsize=13, fontweight='bold')
    ax2.set_title(r'$\beta$ (through origin, legacy)', fontsize=13, fontweight='bold')
    ax2.set_xlabel(r'$\beta$')
    fig.suptitle('Bootstrap Beta Distributions', fontsize=15, fontweight='bold', y=1.01)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 4, 'Bootstrap Beta Distributions', w=9, h=8)

def fig_5(plt, R):
    bdata, blabels = [], []
    for label in R['impact_models']:
        ratios = R['relax'].get(label, [])
        if ratios: bdata.append(ratios); blabels.append(label)
    if not bdata: return
    fig, ax = plt.subplots()
    bp = ax.boxplot(bdata, patch_artist=True, widths=0.55, flierprops=dict(markersize=3, alpha=0.4))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(_c(blabels[i])); patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(blabels)+1))
    ax.set_xticklabels(blabels, rotation=45, ha='right', fontsize=9)
    ax.axhline(2/3, ls='--', color='red', lw=2)
    ax.set(title='Relaxation Ratio', ylabel=r'$I_{\rm final}/I_{\rm peak}$')
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 5, 'Relaxation Ratio', w=9, h=5)

def fig_6(plt, R):
    stab = R['stability']
    labels_s = [m for m in stab if m in MODEL_META]
    if not labels_s: return
    fig, ax = plt.subplots()
    ax.bar(range(len(labels_s)), [stab[l]['frac'] for l in labels_s],
           color=[_c(l) for l in labels_s], width=0.55, edgecolor='white')
    ax.set_xticks(range(len(labels_s)))
    ax.set_xticklabels(labels_s, rotation=45, ha='right', fontsize=9)
    ax.axhline(0.5, ls='--', color='gray', lw=1.5); ax.set_ylim(0, 1.05)
    ax.set(title='Fraction Stable', ylabel='Fraction')
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 6, 'Fraction Stable', w=7, h=4.5)

def fig_7(plt, R):
    hurst = R.get('hurst', {})
    models_h = [m for m in hurst if not np.isnan(hurst.get(m, np.nan))]
    if not models_h: return
    fig, ax = plt.subplots()
    ax.bar(range(len(models_h)), [hurst[m] for m in models_h],
           color=[_c(m) for m in models_h], width=0.55, edgecolor='white')
    ax.set_xticks(range(len(models_h)))
    ax.set_xticklabels(models_h, rotation=45, ha='right', fontsize=9)
    ax.axhline(0.7, ls='--', color='black', lw=1.5)
    ax.set(title='Hurst Exponent', ylabel='H (DFA)')
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 7, 'Hurst Exponent', w=7, h=4.5)

def fig_8(plt, R):
    prop = R.get('propagator', {})
    if not prop: return
    fig, ax = plt.subplots()
    for label, pr in prop.items():
        G, lags = pr['G'], pr['lags']
        mask = (lags > 0) & (G > 0)
        if mask.sum() < 3: continue
        ax.plot(np.log10(lags[mask]), np.log10(G[mask]), color=_c(label), ls=_ls(label), lw=2.5, label=label)
    l_th = np.logspace(0, 2.3, 50)
    ax.plot(np.log10(l_th), np.log10(l_th**(-0.5)*0.1), 'k--', lw=1.5, label=r'$l^{-0.5}$')
    ax.set(title='Propagator G(l)', xlabel='log10(l)', ylabel='log10(G)')
    ax.legend(fontsize=8, ncol=2)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 8, 'Propagator G(l)', w=10, h=5)

def fig_9(plt, R):
    spread = R.get('spread', {})
    if not spread: return
    fig, ax = plt.subplots()
    for label, sr in spread.items():
        ax.plot(sr['u'], sr['mean'], color=_c(label), ls=_ls(label), lw=2.5, label=label)
    ax.axvline(1.0, ls='--', color='gray', lw=1)
    ax.set(title='Spread Dynamics', xlabel='Volume time u', ylabel='Spread (ticks)')
    ax.legend(fontsize=8, ncol=2)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 9, 'Spread Dynamics', w=10, h=5)

def fig_10(plt, R):
    fig, ax = plt.subplots(); has = False
    for label in R['impact_models']:
        d = R['mb_betas'].get(label, {})
        if not d: continue; has = True
        mbs = sorted(d.keys())
        ax.plot(mbs, [d[m] for m in mbs], color=_c(label), ls=_ls(label),
                lw=2, marker=_mk(label), ms=_ms(label), label=label)
    if not has: plt.close(fig); return
    ax.axhline(0.5, ls='--', color='black', lw=1)
    ax.set(title=r'$\beta$ vs mb', xlabel='mb', ylabel=r'$\beta$')
    ax.legend(fontsize=8, ncol=2)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 10, 'Beta vs mb', w=7, h=4.5)

def fig_11(plt, R):
    fig, ax = plt.subplots(); has = False
    for label in R['impact_models']:
        d = R['vol_betas'].get(label, {})
        if not d: continue; has = True
        vols = sorted(d.keys())
        ax.plot(vols, [d[v] for v in vols], color=_c(label), ls=_ls(label),
                lw=2, marker=_mk(label), ms=_ms(label), label=label)
    if not has: plt.close(fig); return
    ax.axhline(0.5, ls='--', color='black', lw=1)
    ax.set(title=r'$\beta$ vs volume', xlabel='Volume', ylabel=r'$\beta$')
    ax.legend(fontsize=8, ncol=2)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 11, 'Beta vs Volume', w=7, h=4.5)

def fig_12(plt, R):
    bdata, blabels = [], []
    for label in R['impact_models']:
        db = R.get('day_betas', {}).get(label, {})
        if db: bdata.append(list(db.values())); blabels.append(label)
    if not bdata: return
    fig, ax = plt.subplots()
    bp = ax.boxplot(bdata, patch_artist=True, widths=0.55, flierprops=dict(markersize=3, alpha=0.4))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(_c(blabels[i])); patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(blabels)+1))
    ax.set_xticklabels(blabels, rotation=45, ha='right', fontsize=9)
    ax.axhline(0.5, ls='--', color='red', lw=2)
    ax.set(title=r'Per-Day $\beta$', ylabel=r'$\beta$')
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 12, 'Per-Day Beta', w=9, h=5)

def fig_13(plt, R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label in R['impact_models']:
        d = R.get('perm_temp', {}).get(label, {})
        perm, temp = d.get('perm', []), d.get('temp', [])
        if not perm: continue
        Qp = np.array([p['Q'] for p in perm])
        Ip = np.array([p['I_perm'] for p in perm])
        It = np.array([p['I_temp'] for p in temp]) if temp else np.array([])
        mask_p = (Qp > 0) & (Ip > 0)
        if mask_p.sum() > 0:
            ax1.scatter(np.log(Qp[mask_p]/1e6), np.log(Ip[mask_p]),
                       s=15, color=_c(label), alpha=0.6, label=label, marker=_mk(label))
        if len(It) > 0:
            mask_t = (Qp > 0) & (It > 0)
            if mask_t.sum() > 0:
                ax2.scatter(np.log(Qp[mask_t]/1e6), np.log(It[mask_t]),
                           s=15, color=_c(label), alpha=0.6, label=label, marker=_mk(label))
    xl = np.array([-12, -5])
    ax1.plot(xl, 1.0*xl, 'k--', lw=1.5, label=r'$\beta_{\rm perm}=1$')
    ax1.set(title='Permanent Impact', xlabel='log(Q/V)', ylabel=r'log($I_{\rm perm}$)')
    ax2.set(title='Temporary Impact', xlabel='log(Q/V)', ylabel=r'log($I_{\rm temp}$)')
    ax1.legend(fontsize=7, ncol=2); ax2.legend(fontsize=7, ncol=2)
    fig.suptitle('Permanent--Temporary Decomposition', fontsize=15, fontweight='bold')
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 13, 'Perm Temp Decomposition', w=12, h=5)

def fig_14(plt, R):
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    for label in R['impact_models']:
        bp = R.get('beta_perm', {}).get(label, np.nan)
        relax = R.get('relax', {}).get(label, [])
        if np.isnan(bp) or not relax: continue
        r_med = np.median(relax)
        ax.scatter(bp, r_med, s=100, color=_c(label), marker=_mk(label),
                   edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(label, (bp, r_med), fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax.scatter(1.0, 2/3, s=200, color='gold', marker='*', edgecolors='black',
               linewidths=1.5, zorder=10, label='Target (1.0, 2/3)')
    rect = Rectangle((0.7, 0.5), 0.6, 0.5, linewidth=1.5, edgecolor='green', facecolor='green', alpha=0.08)
    ax.add_patch(rect)
    ax.axhline(2/3, ls=':', color='gray', lw=0.8)
    ax.axvline(1.0, ls=':', color='gray', lw=0.8)
    ax.set(title=r'No-Arb: $\beta_{\rm perm}$ vs Relaxation',
           xlabel=r'$\beta_{\rm perm}$', ylabel='Relaxation ratio $r$')
    ax.legend(fontsize=9, loc='upper left')
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 14, 'No-Arb Scatter', w=7, h=6)


def fig_15(plt, R):
    """Bar chart: 3 estimators (origin, intercept, ratio) × all models."""
    fig, ax = plt.subplots(figsize=(12, 5))
    labels_list = R['impact_models']
    n = len(labels_list)
    if n == 0: return
    x_pos = np.arange(n)
    w = 0.25

    for offset, (est_key, est_label) in enumerate([
        ('beta_origin', 'Origin'),
        ('beta', 'Intercept'),
        ('beta_ratio', 'Ratio'),
    ]):
        vals = []
        for label in labels_list:
            br = R['beta'].get(label, {})
            vals.append(br.get(est_key, np.nan))
        ax.bar(x_pos + (offset - 1) * w, vals, w * 0.9,
               label=est_label, alpha=0.8,
               color=[_c(l) for l in labels_list] if offset == 1 else None,
               edgecolor='gray', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
    ax.axhline(0.5, ls='--', color='red', lw=2, label=r'$\beta$=0.5')
    ax.axhline(1.0, ls=':', color='gray', lw=1.5)
    ax.set(title=r'$\beta$ Estimator Comparison', ylabel=r'$\beta$')
    ax.legend(fontsize=9, ncol=2)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 15, 'Beta Estimator Comparison', w=12, h=5)


def fig_16(plt, R):
    """Per-insertion midprice response beta comparison across models."""
    # Needs beta_I_mid data from metrics pickles
    has_data = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels_list = R['impact_models']
    beta_vwap, beta_mid, beta_inst = [], [], []
    valid_labels = []

    for label in labels_list:
        br = R['beta'].get(label, {})
        bm = R.get('beta_I_mid', {}).get(label, {})
        bi = R.get('beta_I_inst', {}).get(label, {})
        if br.get('beta') is not None and not np.isnan(br.get('beta', np.nan)):
            beta_vwap.append(br['beta'])
            beta_mid.append(bm.get('beta', np.nan) if bm else np.nan)
            beta_inst.append(bi.get('beta', np.nan) if bi else np.nan)
            valid_labels.append(label)
            has_data = True

    if not has_data:
        plt.close(fig); return

    n = len(valid_labels)
    x_pos = np.arange(n)
    w = 0.25

    # Left: grouped bar chart
    ax1.bar(x_pos - w, beta_vwap, w * 0.9, label='VWAP (cum.)',
            color=[_c(l) for l in valid_labels], edgecolor='gray', linewidth=0.5)
    ax1.bar(x_pos, beta_mid, w * 0.9, label='Midprice resp.',
            color=[_c(l) for l in valid_labels], alpha=0.5, edgecolor='gray', linewidth=0.5)
    ax1.bar(x_pos + w, beta_inst, w * 0.9, label='Instant. exec.',
            color=[_c(l) for l in valid_labels], alpha=0.3, edgecolor='gray', linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
    ax1.axhline(0.5, ls='--', color='red', lw=2)
    ax1.set(title=r'$\beta$ by Impact Metric', ylabel=r'$\beta$ (free intercept)')
    ax1.legend(fontsize=8, ncol=1)

    # Right: model differentiation — range of beta across models
    metrics = ['VWAP', 'Midprice', 'Instant.']
    spreads = []
    for vals in [beta_vwap, beta_mid, beta_inst]:
        finite = [v for v in vals if np.isfinite(v)]
        if len(finite) >= 2:
            spreads.append(max(finite) - min(finite))
        else:
            spreads.append(0)
    ax2.bar(range(3), spreads, color=['#5B7BBF', '#D95F02', '#B5446E'],
            width=0.55, edgecolor='white')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set(title='Cross-Model Spread', ylabel=r'max($\beta$) - min($\beta$)')

    fig.suptitle('Per-Insertion Midprice Response', fontsize=15, fontweight='bold', y=1.01)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 16, 'Per-Insertion Midprice Response', w=14, h=5)


def fig_17(plt, R):
    """Kyle lambda per insertion k across models.

    λ_k = |exec_price - mid_before| / (tick × size).
    k=1: all models identical (conditioning book).
    k>3: S5 should replenish book → lower λ.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    has_data = False
    for label in R['impact_models']:
        kl = R.get('kyle_lambda', {}).get(label, {})
        if not kl: continue
        has_data = True
        ks = sorted(kl.keys())
        means = [kl[k]['mean'] for k in ks]
        stds = [kl[k]['std'] for k in ks]
        medians = [kl[k]['median'] for k in ks]
        ax1.errorbar(ks, means, yerr=stds, color=_c(label), ls=_ls(label),
                     lw=2, marker=_mk(label), ms=_ms(label), label=label,
                     capsize=3, alpha=0.8)
        ax2.plot(ks, medians, color=_c(label), ls=_ls(label),
                 lw=2, marker=_mk(label), ms=_ms(label), label=label)
    if not has_data:
        plt.close(fig); return
    for ax, title in [(ax1, r'Mean $\lambda_k$ ($\pm$ std)'),
                       (ax2, r'Median $\lambda_k$')]:
        ax.set(xlabel='Insertion k', ylabel=r'$\lambda_k$ (ticks / share)', title=title)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(r'Kyle $\lambda$: Price Impact per Share per Insertion',
                 fontsize=15, fontweight='bold', y=1.01)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 17, 'Kyle Lambda per Insertion', w=16, h=5.5)


def fig_18(plt, R):
    """Beta comparison: V_daily vs V_local normalization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    labels_list = R['impact_models']
    if not labels_list: return

    # Left panel: grouped bar chart
    n = len(labels_list)
    x_pos = np.arange(n)
    w = 0.35

    beta_daily = [R['beta'].get(l, {}).get('beta', np.nan) for l in labels_list]
    beta_vlocal = [R.get('beta_Vlocal', {}).get(l, {}).get('beta', np.nan) for l in labels_list]

    ax1.bar(x_pos - w/2, beta_daily, w * 0.9, label=r'$V_{\rm daily}$',
            color=[_c(l) for l in labels_list], edgecolor='gray', linewidth=0.5)
    ax1.bar(x_pos + w/2, beta_vlocal, w * 0.9, label=r'$V_{\rm local}$',
            color=[_c(l) for l in labels_list], alpha=0.5, edgecolor='gray',
            linewidth=0.5, hatch='//')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
    ax1.axhline(0.5, ls='--', color='red', lw=2, label=r'$\beta$=0.5')
    ax1.set(title=r'$\beta$: $V_{\rm daily}$ vs $V_{\rm local}$',
            ylabel=r'$\beta$ (free intercept)')
    ax1.legend(fontsize=9, ncol=2)

    # Right panel: depth at best (k=1) across models
    has_depth = False
    depth_labels, depth_p50, depth_p95 = [], [], []
    for label in labels_list:
        ds = R.get('depth_stats', {}).get(label, {})
        if ds and 'p50' in ds:
            depth_labels.append(label)
            depth_p50.append(ds['p50'])
            depth_p95.append(ds['p95'])
            has_depth = True
    if has_depth:
        x2 = np.arange(len(depth_labels))
        ax2.bar(x2 - w/2, depth_p50, w * 0.9, label='p50',
                color=[_c(l) for l in depth_labels], edgecolor='gray', linewidth=0.5)
        ax2.bar(x2 + w/2, depth_p95, w * 0.9, label='p95',
                color=[_c(l) for l in depth_labels], alpha=0.5, edgecolor='gray',
                linewidth=0.5, hatch='//')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(depth_labels, rotation=45, ha='right', fontsize=9)
        # Mark the old order volumes for reference
        for v, ls in [(75, ':'), (300, '--'), (485, '-.')]:
            ax2.axhline(v, ls=ls, color='gray', lw=1, alpha=0.7, label=f'vol={v}')
        ax2.set(title='Depth at Best (k=1, conditioning book)',
                ylabel='Volume (shares)')
        ax2.legend(fontsize=8, ncol=2)
    else:
        ax2.set_visible(False)

    fig.suptitle(r'Volume Calibration: $\beta$ Normalization & Book Depth',
                 fontsize=15, fontweight='bold', y=1.01)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 18, 'Beta Vlocal Comparison', w=16, h=5.5)


def fig_19(plt, R):
    """Conditional beta: all-k vs k>=3, cumulative vs incremental."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    labels_list = R['impact_models']
    if not labels_list: return
    n = len(labels_list)
    x_pos = np.arange(n)
    w = 0.22

    # Left: 3 estimators side by side — β(all), β(k≥3), β_incremental(k≥3)
    beta_all = [R['beta'].get(l, {}).get('beta', np.nan) for l in labels_list]
    beta_k3 = [R.get('beta_k3plus', {}).get(l, {}).get('beta', np.nan) for l in labels_list]
    beta_inc = [R.get('beta_incremental', {}).get(l, {}).get('beta', np.nan) for l in labels_list]

    ax1.bar(x_pos - w, beta_all, w * 0.9, label=r'$\beta$ (all k, cum.)',
            color=[_c(l) for l in labels_list], edgecolor='gray', linewidth=0.5)
    ax1.bar(x_pos, beta_k3, w * 0.9, label=r'$\beta$ (k$\geq$3, cum.)',
            color=[_c(l) for l in labels_list], alpha=0.6, edgecolor='gray',
            linewidth=0.5, hatch='//')
    ax1.bar(x_pos + w, beta_inc, w * 0.9, label=r'$\beta_{\rm incr}$ (k$\geq$3, per-ins.)',
            color=[_c(l) for l in labels_list], alpha=0.35, edgecolor='gray',
            linewidth=0.5, hatch='xx')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
    ax1.axhline(0.5, ls='--', color='red', lw=2, label=r'$\beta$=0.5')
    ax1.set(title=r'$\beta$ Estimators: Cumulative vs Incremental',
            ylabel=r'$\beta$ (free intercept)')
    ax1.legend(fontsize=8, ncol=2, loc='upper left')

    # Right: α (intercept) comparison — this captures impact LEVEL differences
    alpha_all = [R['beta'].get(l, {}).get('alpha', np.nan) for l in labels_list]
    alpha_k3 = [R.get('beta_k3plus', {}).get(l, {}).get('alpha', np.nan) for l in labels_list]
    alpha_inc = [R.get('beta_incremental', {}).get(l, {}).get('alpha', np.nan) for l in labels_list]

    ax2.bar(x_pos - w, alpha_all, w * 0.9, label=r'$\alpha$ (all k)',
            color=[_c(l) for l in labels_list], edgecolor='gray', linewidth=0.5)
    ax2.bar(x_pos, alpha_k3, w * 0.9, label=r'$\alpha$ (k$\geq$3)',
            color=[_c(l) for l in labels_list], alpha=0.6, edgecolor='gray',
            linewidth=0.5, hatch='//')
    ax2.bar(x_pos + w, alpha_inc, w * 0.9, label=r'$\alpha_{\rm incr}$ (k$\geq$3)',
            color=[_c(l) for l in labels_list], alpha=0.35, edgecolor='gray',
            linewidth=0.5, hatch='xx')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
    ax2.set(title=r'Intercept $\alpha$: Impact Level by Model',
            ylabel=r'$\alpha$ (log-scale intercept)')
    ax2.legend(fontsize=8, ncol=2, loc='upper left')

    fig.suptitle(r'Conditional $\beta$: All Insertions vs k$\geq$3, Cumulative vs Incremental',
                 fontsize=14, fontweight='bold', y=1.01)
    add_metadata_box(fig, R.get('stock', ''))
    save(plt, fig, 19, 'Conditional Beta Comparison', w=16, h=6)


FIGURE_FUNCS = {
    0: fig_0, 1: fig_1, 2: fig_2, 3: fig_3, 4: fig_4, 5: fig_5,
    6: fig_6, 7: fig_7, 8: fig_8, 9: fig_9, 10: fig_10, 11: fig_11,
    12: fig_12, 13: fig_13, 14: fig_14, 15: fig_15, 16: fig_16,
    17: fig_17, 18: fig_18, 19: fig_19,
}


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
def export_summary(R):
    rows = []
    for label in R['impact_models']:
        br = R['beta'].get(label, {})
        bm = R.get('beta_I_mid', {}).get(label, {})
        bi = R.get('beta_I_inst', {}).get(label, {})
        bvl = R.get('beta_Vlocal', {}).get(label, {})
        kl = R.get('kyle_lambda', {}).get(label, {})
        ds = R.get('depth_stats', {}).get(label, {})
        rows.append(dict(
            Model=label,
            beta_intercept=br.get('beta', np.nan),
            beta_origin=br.get('beta_origin', np.nan),
            beta_ratio=br.get('beta_ratio', np.nan),
            CI_lo=br.get('ci_lo', np.nan), CI_hi=br.get('ci_hi', np.nan),
            R2=br.get('r2', np.nan), N=br.get('n', 0),
            beta_I_mid=bm.get('beta', np.nan) if bm else np.nan,
            beta_I_inst=bi.get('beta', np.nan) if bi else np.nan,
            beta_Vlocal=bvl.get('beta', np.nan) if bvl else np.nan,
            R2_Vlocal=bvl.get('r2', np.nan) if bvl else np.nan,
            kyle_lambda_k1=kl.get(1, {}).get('mean', np.nan) if kl else np.nan,
            kyle_lambda_kmax=kl.get(max(kl.keys()), {}).get('mean', np.nan) if kl else np.nan,
            depth_p50=ds.get('p50', np.nan) if ds else np.nan,
            depth_p95=ds.get('p95', np.nan) if ds else np.nan,
            beta_k3plus=R.get('beta_k3plus', {}).get(label, {}).get('beta', np.nan),
            beta_incremental=R.get('beta_incremental', {}).get(label, {}).get('beta', np.nan),
            alpha_incremental=R.get('beta_incremental', {}).get(label, {}).get('alpha', np.nan),
            relaxation=np.median(R['relax'].get(label, [np.nan])),
            stable_frac=R['stability'].get(label, {}).get('frac', np.nan),
            Hurst=R['hurst'].get(label, np.nan),
            beta_perm=R.get('beta_perm', {}).get(label, np.nan),
            no_arb_score=R['scorecard'].get(label, {}).get('score', np.nan)))
    df = pd.DataFrame(rows)
    out = SAVE_DIR / 'summary_statistics.csv'
    df.to_csv(out, index=False)
    print(f'\n  Summary CSV -> {out}')
    return df


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def main():
    global SAVE_DIR, DPI

    parser = argparse.ArgumentParser(description='300 · Full analysis + figures from raw pickles')
    parser.add_argument('--stock', type=str, default='GOOG')
    parser.add_argument('--daily_hl', type=str, default=None,
                        help='Path to daily_h_l CSV for per-day V/sigma normalization')
    parser.add_argument('--figs', type=str, default=None, help='e.g. 1,3,5')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    if args.list:
        print('\nAvailable figures:')
        for num, name in FIGURE_CATALOG.items():
            print(f'  [{num:>2}] {name}')
        sys.exit(0)

    DPI = args.dpi
    stock = args.stock.upper()
    SAVE_DIR = Path(args.out) if args.out else Path(f'pics_for_300_{stock}')
    SAVE_DIR.mkdir(exist_ok=True)

    # Load pre-computed metrics pickles (small, fast)
    print('=' * 60)
    print(f'  Loading metrics for {stock}')
    print('=' * 60)
    R = load_metrics_pickles(stock, src_dir=str(SAVE_DIR) if args.out else None)

    # Figures
    want = set(int(x) for x in args.figs.split(',')) if args.figs else set(FIGURE_CATALOG.keys())
    plt = setup_mpl()
    print('\n' + '=' * 60)
    print(f'  Generating {len(want)} figures')
    print('=' * 60)
    for num in sorted(want):
        if num in FIGURE_FUNCS:
            FIGURE_FUNCS[num](plt, R)

    export_summary(R)
    print(f'\n  Done: {len(list(SAVE_DIR.glob("*.png")))} PNGs in {SAVE_DIR}/')


if __name__ == '__main__':
    main()
