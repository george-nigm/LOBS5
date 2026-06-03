#!/usr/bin/env python3
"""
210 · Full Grid Analysis — 9 Models, GOOG Jan 2026

Two-phase workflow:
  Phase 1 (slow, once):   python run_210_analysis.py compute
  Phase 2 (fast, repeat): python run_210_analysis.py figures [--figs 1,3,5]

Phase 1 loads all data, computes everything, saves to pickle.
Phase 2 reads pickle, generates selected figures in seconds.

Examples:
  python lob_impact/run_210_analysis.py compute              # full compute
  python lob_impact/run_210_analysis.py compute --max 2      # quick test
  python lob_impact/run_210_analysis.py figures               # all figures
  python lob_impact/run_210_analysis.py figures --figs 1,2,3  # selected
  python lob_impact/run_210_analysis.py figures --list        # show available
"""
import argparse, pickle, sys
import numpy as np
import pandas as pd
import re, math
from pathlib import Path
from collections import OrderedDict
from scipy import optimize
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# <<<  EDIT THESE  >>>
# ═══════════════════════════════════════════════════════════════════════
MAX_SAMPLES   = 2048       # set to 2 for quick test, 2048 for full run
N_BOOTSTRAP   = 1000
TICK_SIZE     = 100
STOCK         = 'GOOG'
# Outlier filter: skip samples where midprice deviates >50% from its initial value.
# Stock-agnostic: GOOG ($320) rejects above $480, INTC ($20) rejects above $30.
# A few hundred messages should never cause 50% price moves.
MIDPRICE_OUTLIER_FACTOR = 1.5

SAVE_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v3')

def _dirs(stock):
    d = Path(f'pics_for_210_{stock}')
    return d, d / 'results_cache.pkl'

FIG_DIR, CACHE_FILE = _dirs(STOCK)  # defaults, overridden by CLI

# ═══════════════════════════════════════════════════════════════════════
# Model palette
# ═══════════════════════════════════════════════════════════════════════
MODEL_META = OrderedDict([
    ('ZeroInsertions',  dict(color='#7CAE7A', dash=':', marker='D')),
    ('Historic',        dict(color='#90939C', dash=':', marker='x')),
    ('Heuristic',       dict(color='#546884', dash='-.', marker='d')),
    ('CST',             dict(color='#213552', dash='--', marker='^')),
    ('CGAN',            dict(color='#7B4F9E', dash=(0,(8,4)), marker='s')),
    ('LobS5',           dict(color='#C88A3A', dash='-', marker='o')),
    ('S5-120M',         dict(color='#D95F02', dash='-', marker='v')),
    ('S5-4K',           dict(color='#5B7BBF', dash='-', marker='*')),
    ('S5-360M',         dict(color='#B5446E', dash='-', marker='H')),
    ('LobS5-v2',        dict(color='#2CA02C', dash='-', marker='P')),
])
MODEL_LABELS = list(MODEL_META.keys())

def _build_paths(stock):
    paths = OrderedDict()
    for label in MODEL_LABELS:
        paths[label] = dict(
            buy=str(SAVE_BASE / label / 'context_500_buy' / stock),
            sell=str(SAVE_BASE / label / 'context_500_sell' / stock),
        )
    return paths

def _c(m):  return MODEL_META[m]['color']
def _ls(m): return MODEL_META[m]['dash']
def _mk(m): return MODEL_META[m]['marker']


# ═══════════════════════════════════════════════════════════════════════
# Data I/O
# ═══════════════════════════════════════════════════════════════════════
def find_latest_exp(folder_path):
    p = Path(folder_path)
    if not p.exists(): return p
    exps = sorted(p.glob('exp_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    return exps[0] if exps else p

def discover_folders(buy_root, sell_root):
    folders = []
    buy_root, sell_root = Path(buy_root), Path(sell_root)
    if not buy_root.exists(): return pd.DataFrame()
    for bp in sorted(buy_root.iterdir()):
        if not bp.is_dir(): continue
        name = bp.name
        sp = sell_root / name
        if not sp.exists(): continue
        m = re.match(r'i(\d+)_c(\d+)_mb(\d+)_v(\d+)_cntxt(\d+)%', name)
        if not m: continue
        folders.append(dict(
            folder=name, i=int(m.group(1)), c=int(m.group(2)),
            mb=int(m.group(3)), vol=int(m.group(4)), cntxt_pct=int(m.group(5)),
            buy_path=str(find_latest_exp(bp)), sell_path=str(find_latest_exp(sp))))
    return pd.DataFrame(folders)

def is_midprice_outlier(book_array, factor=MIDPRICE_OUTLIER_FACTOR):
    """True if midprice deviates >factor from its initial value, or is non-positive."""
    mp = get_midprice(book_array)
    mp_valid = mp[mp > 0]
    if len(mp_valid) < 2:
        return True
    ref = mp_valid[0]
    return np.any(mp_valid > ref * factor) or np.any(mp_valid < ref / factor)


def load_folder_data(folder_row, max_samples):
    buy_gen = Path(folder_row['buy_path']) / 'data_gen'
    sell_gen = Path(folder_row['sell_path']) / 'data_gen'
    buy_books, sell_books, buy_msgs, sell_msgs = [], [], [], []
    n_skipped = 0
    for gen_dir, books, msgs in [(buy_gen, buy_books, buy_msgs),
                                 (sell_gen, sell_books, sell_msgs)]:
        if not gen_dir.exists(): continue
        ob_files = sorted(gen_dir.glob('*_orderbook_*_gen_id_0.csv'))[:max_samples]
        for ob_f in ob_files:
            try:
                book = pd.read_csv(ob_f, header=None).values
                if is_midprice_outlier(book):
                    n_skipped += 1
                    continue
                books.append(book)
                msg_f = ob_f.parent / ob_f.name.replace('_orderbook_', '_message_')
                if msg_f.exists():
                    msgs.append(pd.read_csv(msg_f, header=None).values)
            except:
                pass
    if n_skipped:
        print(f'    midprice filter: skipped {n_skipped} samples (>{MIDPRICE_OUTLIER_FACTOR}x deviation)')
    return dict(buy_books=buy_books, sell_books=sell_books,
                buy_msgs=buy_msgs, sell_msgs=sell_msgs,
                n_buy=len(buy_books), n_sell=len(sell_books))

def load_all(paths_dict, max_samples):
    results = OrderedDict()
    for model_name, paths in paths_dict.items():
        print(f"Loading {STOCK}/{model_name}...")
        grid_df = discover_folders(paths['buy'], paths['sell'])
        if grid_df.empty:
            print(f"  No data"); continue
        data = {}
        for _, row in grid_df.iterrows():
            data[row['folder']] = load_folder_data(row, max_samples)
        n_buy = sum(d['n_buy'] for d in data.values())
        n_sell = sum(d['n_sell'] for d in data.values())
        results[model_name] = dict(grid=grid_df, data=data)
        print(f"  {len(grid_df)} folders, {n_buy} buy + {n_sell} sell samples")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════
def get_midprice(book_arr):
    return (book_arr[:, 0].astype(float) + book_arr[:, 2].astype(float)) / 2.0

def compute_combined_impact(buy_books, sell_books):
    buy_rets, sell_rets = [], []
    for b in buy_books:
        mid = get_midprice(b); mid = mid[mid > 0]
        if len(mid) >= 2: buy_rets.append((mid - mid[0]) / TICK_SIZE)
    for b in sell_books:
        mid = get_midprice(b); mid = mid[mid > 0]
        if len(mid) >= 2: sell_rets.append((mid - mid[0]) / TICK_SIZE)
    return buy_rets, sell_rets

def extract_point_cloud(data, grid_df):
    points = []
    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in data: continue
        fd = data[folder]
        buy_rets, sell_rets = compute_combined_impact(fd['buy_books'], fd['sell_books'])
        n_pairs = min(len(buy_rets), len(sell_rets))
        for j in range(n_pairs):
            br, sr = buy_rets[j], sell_rets[j]
            L = min(len(br), len(sr))
            if L < 2: continue
            insert_end = min(row['i'] * (row['mb'] + 1), L - 1)
            combined = (br[insert_end] - sr[insert_end]) / 2.0
            if combined <= 1e-10: continue
            points.append(dict(Q=row['i'] * row['vol'], I=combined,
                              vol=row['vol'], i=row['i'], mb=row['mb'], sample_id=j))
    return pd.DataFrame(points)

def compute_global_beta(pc_df, daily_vol=1e6, daily_sigma=1.0):
    if pc_df.empty: return dict(beta=np.nan, r2=np.nan, n=0)
    x = np.log(pc_df['Q'].values / daily_vol)
    y = np.log(pc_df['I'].values / daily_sigma)
    beta = np.dot(x, y) / np.dot(x, x)
    y_pred = beta * x
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum(y**2)
    return dict(beta=beta, r2=r2, n=len(pc_df))

def bootstrap_beta(pc_df, n_boot=N_BOOTSTRAP):
    if pc_df.empty: return np.array([])
    sample_ids = pc_df['sample_id'].unique()
    rng = np.random.default_rng(42)
    betas = []
    for _ in range(n_boot):
        boot_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        boot_df = pd.concat([pc_df[pc_df['sample_id'] == sid] for sid in boot_ids], ignore_index=True)
        betas.append(compute_global_beta(boot_df)['beta'])
    return np.array(betas)

def compute_master_curve(data_dict, folder, n_vol_u=200):
    i_val = int(re.search(r'i(\d+)', folder).group(1))
    mb_val = int(re.search(r'mb(\d+)', folder).group(1))
    L = i_val * (mb_val + 1)
    buy_rets, sell_rets = compute_combined_impact(data_dict['buy_books'], data_dict['sell_books'])
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
                std=np.nanstd(curves, axis=0), n=n_pairs, folder=folder)

def compute_relaxation_ratio(mc, u_peak=1.0, u_final=3.0):
    if mc is None: return np.nan
    u, m = mc['u'], mc['mean']
    pi, fi = np.argmin(np.abs(u - u_peak)), np.argmin(np.abs(u - u_final))
    if m[pi] < 1e-10: return np.nan
    return m[fi] / m[pi]

def fit_decay(u_grid, mean_curve, u_peak=1.0):
    pi = np.argmin(np.abs(u_grid - u_peak))
    post_u, post_y = u_grid[pi:] - u_peak, mean_curve[pi:]
    if len(post_y) < 5 or post_y[0] < 1e-10: return dict(gamma_power=np.nan)
    y_norm = post_y / post_y[0]
    try:
        mask = post_u > 0
        popt, _ = optimize.curve_fit(lambda u, g, c: c*(1+u)**(-g),
                                      post_u[mask], y_norm[mask], p0=[0.5,1.0], maxfev=5000)
        return dict(gamma_power=popt[0])
    except: return dict(gamma_power=np.nan)

def stability_vote(mc, u_peak=1.0):
    if mc is None: return dict(stable=False, votes=0)
    u, m = mc['u'], mc['mean']
    post = m[np.argmin(np.abs(u - u_peak)):]
    if len(post) < 10: return dict(stable=False, votes=0)
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
    votes = sum([m1, m2, m3])
    return dict(stable=votes >= 2, votes=votes)

def extract_order_signs(msgs_list):
    out = []
    for msgs in msgs_list:
        mask = msgs[:, 1] == 4
        if mask.sum() < 2: continue
        out.append(np.where(msgs[mask, 2] == 0, 1, -1))
    return out

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

def compute_propagator(msgs_list, books_list, max_lag=200):
    G_sum, G_count = np.zeros(max_lag), np.zeros(max_lag)
    for msgs, books in zip(msgs_list, books_list):
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

def compute_spread_trajectory(buy_books, sell_books, n_inj):
    all_sp = []
    for books in buy_books + sell_books:
        if len(books) < 10: continue
        sp = (books[:,0].astype(float) - books[:,2].astype(float)) / TICK_SIZE
        sp[sp <= 0] = np.nan; sp[sp > 100] = np.nan
        all_sp.append(sp)
    if not all_sp: return None
    L = max(n_inj, 1)
    u_grid = np.linspace(0, max(len(s) for s in all_sp) / L, 200)
    interp = []
    for s in all_sp:
        v = ~np.isnan(s)
        if v.sum() < 5: continue
        interp.append(np.interp(u_grid, np.arange(len(s))[v] / L, s[v]))
    if not interp: return None
    arr = np.array(interp)
    return dict(u=u_grid, mean=np.nanmean(arr, axis=0), std=np.nanstd(arr, axis=0))


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: COMPUTE
# ═══════════════════════════════════════════════════════════════════════
def run_compute(stock, max_samples):
    global FIG_DIR, CACHE_FILE
    FIG_DIR, CACHE_FILE = _dirs(stock)
    FIG_DIR.mkdir(exist_ok=True)
    PATHS = _build_paths(stock)
    print('=' * 70)
    print(f'  PHASE 1: COMPUTE — {stock}, 9 models, MAX_SAMPLES={max_samples}')
    print('=' * 70)

    R_raw = load_all(PATHS, max_samples)
    print(f'\nLoaded {len(R_raw)} / {len(MODEL_LABELS)} models')

    R_impact = OrderedDict((k, v) for k, v in R_raw.items() if k != 'ZeroInsertions')
    cache = dict(models=list(R_raw.keys()), impact_models=list(R_impact.keys()))

    # Null baseline drifts
    if 'ZeroInsertions' in R_raw:
        zr = R_raw['ZeroInsertions']
        drifts = []
        for fd in zr['data'].values():
            for b in fd['buy_books'] + fd['sell_books']:
                mid = get_midprice(b); mid = mid[mid > 0]
                if len(mid) >= 2: drifts.append((mid[-1] - mid[0]) / TICK_SIZE)
        cache['null_drifts'] = np.array(drifts)
        print(f'  ZeroInsertions drift = {np.mean(cache["null_drifts"]):.3f} +/- {np.std(cache["null_drifts"]):.3f}')

    # Beta
    beta_results = {}
    for label, rd in R_impact.items():
        pc = extract_point_cloud(rd['data'], rd['grid'])
        res = compute_global_beta(pc)
        boots = bootstrap_beta(pc)
        ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) > 0 else (np.nan, np.nan)
        beta_results[label] = dict(beta=res['beta'], r2=res['r2'], n=res['n'],
                                    ci_lo=ci[0], ci_hi=ci[1], boots=boots,
                                    pc_x=np.log(pc['Q'].values/1e6) if not pc.empty else np.array([]),
                                    pc_y=np.log(pc['I'].values/1.0) if not pc.empty else np.array([]))
        print(f'  {label}: beta={res["beta"]:.4f} R2={res["r2"]:.4f} n={res["n"]}')
    cache['beta'] = beta_results

    # Master curves
    master_curves = OrderedDict()
    for label, rd in R_impact.items():
        mcs = {}
        for _, frow in rd['grid'].iterrows():
            folder = frow['folder']
            if folder not in rd['data']: continue
            mc = compute_master_curve(rd['data'][folder], folder)
            if mc is not None: mcs[folder] = mc
        master_curves[label] = mcs
        print(f'  {label}: {len(mcs)} master curves')
    cache['master_curves'] = master_curves

    # Relaxation
    relax_data = OrderedDict()
    for label, mcs in master_curves.items():
        ratios = [compute_relaxation_ratio(mc) for mc in mcs.values()]
        relax_data[label] = [r for r in ratios if not np.isnan(r)]
    cache['relax'] = relax_data

    # Stability
    stab_data = OrderedDict()
    for label, rd in R_impact.items():
        sc, tc = 0, 0
        for _, frow in rd['grid'].iterrows():
            folder = frow['folder']
            if folder not in rd['data']: continue
            mc = compute_master_curve(rd['data'][folder], folder)
            s = stability_vote(mc)
            tc += 1
            if s['stable']: sc += 1
        stab_data[label] = dict(stable=sc, total=tc, frac=sc/max(tc,1))
        print(f'  {label}: {sc}/{tc} stable ({sc/max(tc,1):.0%})')
    cache['stability'] = stab_data

    # Scorecard
    scorecard = OrderedDict()
    for label, rd in R_impact.items():
        beta = beta_results[label]['beta']
        ratios, gammas = relax_data[label], []
        for mc in master_curves[label].values():
            fd = fit_decay(mc['u'], mc['mean'])
            if not np.isnan(fd['gamma_power']): gammas.append(fd['gamma_power'])
        r_mean = np.mean(ratios) if ratios else np.nan
        gamma = np.mean(gammas) if gammas else np.nan
        b_perm = beta * (r_mean if not np.isnan(r_mean) else 1.0)
        A = beta < 1
        B = 0.7 <= b_perm <= 1.3 if not np.isnan(b_perm) else False
        C = 0.3 <= gamma <= 1.0 if not np.isnan(gamma) else False
        D = 0.5 <= r_mean <= 1.0 if not np.isnan(r_mean) else False
        E = beta <= 1/(1+2*gamma) if not np.isnan(gamma) else False
        scorecard[label] = dict(b_perm=b_perm, r=r_mean, gamma=gamma,
                                 A=A, B=B, C=C, D=D, E=E, score=sum([A,B,C,D,E]))
    cache['scorecard'] = scorecard

    # Hurst
    hurst_results = OrderedDict()
    for label, rd in R_impact.items():
        all_signs = []
        for fd in rd['data'].values():
            for msgs in fd.get('buy_msgs',[]) + fd.get('sell_msgs',[]):
                all_signs.extend(extract_order_signs([msgs]))
        if all_signs:
            concat = np.concatenate(all_signs)
            if len(concat) >= 100:
                hurst_results[label] = compute_hurst_dfa(concat)
    cache['hurst'] = hurst_results

    # Propagator
    prop_results = OrderedDict()
    for label, rd in R_impact.items():
        ma, ba = [], []
        for fd in rd['data'].values():
            for m, b in zip(fd.get('buy_msgs',[]), fd.get('buy_books',[])):
                ma.append(m); ba.append(b)
            for m, b in zip(fd.get('sell_msgs',[]), fd.get('sell_books',[])):
                ma.append(m); ba.append(b)
        if ma:
            G, lags = compute_propagator(ma, ba)
            prop_results[label] = dict(G=G, lags=lags)
    cache['propagator'] = prop_results

    # Spread
    spread_results = OrderedDict()
    for label, rd in R_impact.items():
        all_sc = []
        for _, frow in rd['grid'].iterrows():
            folder = frow['folder']
            if folder not in rd['data']: continue
            fd = rd['data'][folder]
            sc = compute_spread_trajectory(fd['buy_books'], fd['sell_books'],
                                           frow['i'] * (frow['mb'] + 1))
            if sc is not None: all_sc.append(sc)
        if all_sc:
            u = all_sc[0]['u']
            avg = np.nanmean([c['mean'] for c in all_sc], axis=0)
            spread_results[label] = dict(u=u, mean=avg)
    cache['spread'] = spread_results

    # Param sensitivity (beta vs mb, beta vs vol)
    mb_betas_all, vol_betas_all = OrderedDict(), OrderedDict()
    for label, rd in R_impact.items():
        mb_b, vol_b = {}, {}
        for _, frow in rd['grid'].iterrows():
            folder = frow['folder']
            if folder not in rd['data']: continue
            pc = extract_point_cloud({folder: rd['data'][folder]}, pd.DataFrame([frow]))
            if pc.empty: continue
            res = compute_global_beta(pc)
            if not np.isnan(res['beta']):
                mb_b.setdefault(frow['mb'], []).append(res['beta'])
                vol_b.setdefault(frow['vol'], []).append(res['beta'])
        mb_betas_all[label] = {k: np.mean(v) for k, v in mb_b.items()}
        vol_betas_all[label] = {k: np.mean(v) for k, v in vol_b.items()}
    cache['mb_betas'] = mb_betas_all
    cache['vol_betas'] = vol_betas_all

    # Save pickle (legacy)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'\nPickle saved: {CACHE_FILE} ({CACHE_FILE.stat().st_size/1024:.0f} KB)')

    # Save per-model parquets
    save_parquets(cache, FIG_DIR)
    return cache


def save_parquets(cache, out_dir):
    """Save one parquet per model + null_drifts + summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    imp = cache.get('impact_models', [])

    # ── Null drifts ──
    if 'null_drifts' in cache:
        pd.DataFrame({'drift': cache['null_drifts']}).to_parquet(
            out_dir / 'ZeroInsertions.parquet', index=False)
        print(f'  ZeroInsertions.parquet ({len(cache["null_drifts"])} drifts)')

    # ── Per-model parquets ──
    for label in imp:
        rows = []

        # Master curves: table='mc', folder=name, x=u, y=mean, y2=std
        for folder, mc in cache.get('master_curves', {}).get(label, {}).items():
            for j in range(len(mc['u'])):
                rows.append({'table': 'mc', 'folder': folder,
                             'x': float(mc['u'][j]), 'y': float(mc['mean'][j]),
                             'y2': float(mc['std'][j])})

        # Point cloud: table='pc', x=pc_x, y=pc_y
        br = cache.get('beta', {}).get(label, {})
        pc_x, pc_y = br.get('pc_x', np.array([])), br.get('pc_y', np.array([]))
        for j in range(len(pc_x)):
            rows.append({'table': 'pc', 'folder': '', 'x': float(pc_x[j]),
                         'y': float(pc_y[j]), 'y2': np.nan})

        # Bootstrap: table='boot', y=beta_sample
        for b in br.get('boots', []):
            rows.append({'table': 'boot', 'folder': '', 'x': np.nan,
                         'y': float(b), 'y2': np.nan})

        # Relaxation: table='relax', y=ratio
        for i, r in enumerate(cache.get('relax', {}).get(label, [])):
            rows.append({'table': 'relax', 'folder': '', 'x': float(i),
                         'y': float(r), 'y2': np.nan})

        # Propagator: table='prop', x=lag, y=G
        pr = cache.get('propagator', {}).get(label, {})
        if pr:
            for j in range(len(pr['lags'])):
                rows.append({'table': 'prop', 'folder': '', 'x': float(pr['lags'][j]),
                             'y': float(pr['G'][j]), 'y2': np.nan})

        # Spread: table='spread', x=u, y=mean
        sp = cache.get('spread', {}).get(label, {})
        if sp:
            for j in range(len(sp['u'])):
                rows.append({'table': 'spread', 'folder': '', 'x': float(sp['u'][j]),
                             'y': float(sp['mean'][j]), 'y2': np.nan})

        # mb_betas: table='mb_beta', x=mb, y=beta
        for mb, beta in cache.get('mb_betas', {}).get(label, {}).items():
            rows.append({'table': 'mb_beta', 'folder': '', 'x': float(mb),
                         'y': float(beta), 'y2': np.nan})

        # vol_betas: table='vol_beta', x=vol, y=beta
        for vol, beta in cache.get('vol_betas', {}).get(label, {}).items():
            rows.append({'table': 'vol_beta', 'folder': '', 'x': float(vol),
                         'y': float(beta), 'y2': np.nan})

        df = pd.DataFrame(rows)

        # Model-level scalars → parquet metadata
        meta = {
            'beta': str(br.get('beta', '')),
            'r2': str(br.get('r2', '')),
            'n': str(br.get('n', '')),
            'ci_lo': str(br.get('ci_lo', '')),
            'ci_hi': str(br.get('ci_hi', '')),
            'hurst': str(cache.get('hurst', {}).get(label, '')),
            'stable': str(cache.get('stability', {}).get(label, {}).get('stable', '')),
            'total': str(cache.get('stability', {}).get(label, {}).get('total', '')),
            'stable_frac': str(cache.get('stability', {}).get(label, {}).get('frac', '')),
        }
        sc = cache.get('scorecard', {}).get(label, {})
        for k in ('b_perm', 'r', 'gamma', 'A', 'B', 'C', 'D', 'E', 'score'):
            meta[k] = str(sc.get(k, ''))

        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df)
        combined_meta = {**(table.schema.metadata or {}),
                         **{k.encode(): v.encode() for k, v in meta.items()}}
        table = table.replace_schema_metadata(combined_meta)

        path = out_dir / f'{label}.parquet'
        pq.write_table(table, path)
        print(f'  {label}.parquet ({len(df)} rows, {path.stat().st_size/1024:.0f} KB)')


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: FIGURES
# ═══════════════════════════════════════════════════════════════════════
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
])

DPI = 300

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

def save(plt, fig, num, name, w=10.8, h=5.0):
    fig.set_size_inches(w, h)
    fig.tight_layout()
    fname = f'{num}. {name}'
    fig.savefig(FIG_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG_DIR / f'{fname}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [{num:>2}] {name}')

def latex_table(headers, rows, caption=""):
    fmt = 'l' + 'c' * (len(headers) - 1)
    lines = [f"\n── LaTeX: {caption} ──",
             f"\\begin{{tabular}}{{{fmt}}}", "\\toprule",
             " & ".join(headers) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(str(v) for v in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    print('\n'.join(lines))


def run_figures(cache, selected=None):
    plt = setup_mpl()
    FIG_DIR.mkdir(exist_ok=True)
    imp = cache['impact_models']
    want = set(selected) if selected else set(FIGURE_CATALOG.keys())

    print('=' * 70)
    print(f'  PHASE 2: FIGURES — generating {len(want)} figures')
    print('=' * 70)

    # ── 0. Null Baseline ──
    if 0 in want and 'null_drifts' in cache:
        drifts = cache['null_drifts']
        fig, ax = plt.subplots()
        ax.hist(drifts, bins=100, color=_c('ZeroInsertions'), edgecolor='white', lw=0.3)
        ax.axvline(0, ls='--', color='black', lw=1)
        ax.set(title='Null Baseline: Mid-Price Drift (ZeroInsertions)',
               xlabel='Drift (ticks)', ylabel='Count')
        save(plt, fig, 0, 'Null Baseline Drift', w=7, h=4.5)

    # ── 1. Master Curves (panels) ──
    if 1 in want:
        n_m = len(imp); nc = 2; nr = math.ceil(n_m / nc)
        palette = plt.cm.tab10(np.linspace(0, 1, 30))
        fig, axes = plt.subplots(nr, nc, figsize=(13, 4*nr), sharex=True, squeeze=False)
        for idx, label in enumerate(imp):
            r, c = idx // nc, idx % nc; ax = axes[r][c]
            fi = 0
            for mc in cache['master_curves'][label].values():
                mask = mc['u'] <= 3.0
                ax.plot(mc['u'][mask], mc['mean'][mask], color=palette[fi%len(palette)], lw=0.8, alpha=0.6)
                fi += 1
            ax.axvline(1.0, ls='--', color='gray', lw=0.8)
            ax.set_title(label, fontsize=12, fontweight='bold')
            if c == 0: ax.set_ylabel(r'$I_{\rm norm}(u)$')
            if r == nr-1: ax.set_xlabel('u')
        for idx in range(n_m, nr*nc): axes[idx//nc][idx%nc].set_visible(False)
        fig.suptitle('Cross-Model: Master Curves', fontsize=16, fontweight='bold', y=1.01)
        save(plt, fig, 1, 'Master Curves', w=13, h=4*nr)

    # ── 2. Average Master Curve ──
    if 2 in want:
        fig, ax = plt.subplots()
        for label in imp:
            mcs = cache['master_curves'][label]
            if not mcs: continue
            all_m = [mc['mean'] for mc in mcs.values()]
            u = list(mcs.values())[0]['u']
            avg, std = np.nanmean(all_m, axis=0), np.nanstd(all_m, axis=0)
            ax.plot(u, avg, color=_c(label), ls=_ls(label), lw=3, label=label)
            ax.fill_between(u, avg-std, avg+std, color=_c(label), alpha=0.08)
        ax.axvline(1.0, ls='--', color='gray', lw=1.5)
        ax.set(title='Cross-Model: Average Master Curve', xlabel='u', ylabel=r'$I_{\rm norm}(u)$')
        ax.legend(fontsize=9, ncol=2)
        save(plt, fig, 2, 'Average Master Curve', w=11, h=5)

    # ── 3. Beta Regression Lines ──
    if 3 in want:
        fig, ax = plt.subplots()
        beta = cache['beta']
        for label in imp:
            br = beta[label]; x, y = br['pc_x'], br['pc_y']
            if len(x) == 0: continue
            ax.scatter(x, y, s=3, color=_c(label), alpha=0.12, rasterized=True)
            xl = np.array([x.min(), x.max()])
            ax.plot(xl, br['beta']*xl, color=_c(label), ls=_ls(label), lw=3,
                    label=f"{label}: {br['beta']:.3f}")
        ax.plot([-15,-5], [0.5*-15, 0.5*-5], 'k--', lw=2, label='Theory: 0.5')
        ax.set(title='Cross-Model: Beta Regression Lines',
               xlabel='log(Q/V)', ylabel=r'log(I/$\sigma$)')
        ax.legend(fontsize=8, loc='lower left', ncol=2)
        save(plt, fig, 3, 'Beta Regression Lines', w=9, h=7)

        # LaTeX Table 1
        t1 = []
        for label in imp:
            br = beta[label]
            ci = f"[{br['ci_lo']:.3f}, {br['ci_hi']:.3f}]" if not np.isnan(br['ci_lo']) else '---'
            t1.append([label, f"{br['beta']:.3f}", f"{br['r2']:.3f}", f"{br['n']:,}", ci])
        latex_table(['Model', r'$\beta$', r'$R^2$', '$N$', '95\\% CI'], t1, 'Table 1: Global Beta')

    # ── 4. Bootstrap Beta ──
    if 4 in want:
        fig, ax = plt.subplots()
        for label in imp:
            boots = cache['beta'][label]['boots']
            if len(boots) == 0: continue
            ax.hist(boots, bins=50, alpha=0.45, color=_c(label), label=label, edgecolor='none')
        ax.axvline(0.5, ls='--', color='red', lw=2.5)
        ax.set(title='Cross-Model: Bootstrap Beta Distributions', xlabel=r'$\beta$', ylabel='Count')
        ax.legend(fontsize=8, ncol=2)
        save(plt, fig, 4, 'Bootstrap Beta Distributions', w=9, h=5)

    # ── 5. Relaxation Ratio ──
    if 5 in want:
        fig, ax = plt.subplots()
        bdata, blabels = [], []
        for label in imp:
            ratios = cache['relax'][label]
            if ratios: bdata.append(ratios); blabels.append(label)
        bp = ax.boxplot(bdata, patch_artist=True, widths=0.55,
                        flierprops=dict(markersize=3, alpha=0.4))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(_c(blabels[i])); patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(blabels)+1))
        ax.set_xticklabels(blabels, rotation=45, ha='right', fontsize=9)
        ax.axhline(2/3, ls='--', color='red', lw=2)
        ax.text(len(blabels)+0.4, 2/3, 'Bouchaud 2/3', va='center', fontsize=9, color='red')
        ax.set(title='Cross-Model: Relaxation Ratio', ylabel=r'$I_{\rm final}/I_{\rm peak}$')
        save(plt, fig, 5, 'Relaxation Ratio', w=9, h=5)

        # LaTeX Table 2
        t2 = []
        for label in imp:
            arr = cache['relax'][label]
            if not arr: continue
            arr = np.array(arr); med = np.median(arr)
            t2.append([label, f"{med:.3f}", f"{np.mean(arr):.3f} $\\pm$ {np.std(arr):.3f}",
                        f"{np.std(arr)/(abs(np.mean(arr))+1e-10):.3f}", f"{abs(med-2/3):.3f}"])
        latex_table(['Model', 'Median', r'Mean $\pm$ std', 'CV', r'$|\Delta|$ from 2/3'], t2, 'Table 2: Relaxation')

    # ── 6. Fraction Stable ──
    if 6 in want:
        fig, ax = plt.subplots()
        labels_s = list(cache['stability'].keys())
        vals = [cache['stability'][l]['frac'] for l in labels_s]
        ax.bar(range(len(labels_s)), vals, color=[_c(l) for l in labels_s], width=0.55, edgecolor='white')
        ax.set_xticks(range(len(labels_s)))
        ax.set_xticklabels(labels_s, rotation=45, ha='right', fontsize=9)
        ax.axhline(0.5, ls='--', color='gray', lw=1.5); ax.set_ylim(0, 1.05)
        ax.set(title='Cross-Model: Fraction Stable', ylabel='Fraction (2+/3 votes)')
        save(plt, fig, 6, 'Fraction Stable', w=7, h=4.5)

        # LaTeX Table 3
        t3 = [[l, f"{cache['stability'][l]['stable']}/{cache['stability'][l]['total']}",
               f"{cache['stability'][l]['frac']:.0%}"] for l in labels_s]
        latex_table(['Model', 'Stable/Total', 'Fraction'], t3, 'Table 3: Stability')

    # ── 7. Hurst ──
    if 7 in want and cache['hurst']:
        fig, ax = plt.subplots()
        models_h = [m for m in cache['hurst'] if not np.isnan(cache['hurst'][m])]
        if models_h:
            vals = [cache['hurst'][m] for m in models_h]
            ax.bar(range(len(models_h)), vals, color=[_c(m) for m in models_h], width=0.55, edgecolor='white')
            ax.set_xticks(range(len(models_h)))
            ax.set_xticklabels(models_h, rotation=45, ha='right', fontsize=9)
            ax.axhline(0.7, ls='--', color='black', lw=1.5)
            ax.axhline(0.5, ls=':', color='gray', lw=1)
            ax.set(title='Cross-Model: Hurst Exponent', ylabel='H (DFA)')
            save(plt, fig, 7, 'Hurst Exponent', w=7, h=4.5)

    # ── 8. Propagator ──
    if 8 in want and cache['propagator']:
        fig, ax = plt.subplots()
        for label, pr in cache['propagator'].items():
            G, lags = pr['G'], pr['lags']
            mask = (lags > 0) & (G > 0)
            if mask.sum() < 3: continue
            ax.plot(np.log10(lags[mask]), np.log10(G[mask]),
                    color=_c(label), ls=_ls(label), lw=2.5, label=label)
        l_th = np.logspace(0, 2.3, 50)
        ax.plot(np.log10(l_th), np.log10(l_th**(-0.5)*0.1), 'k--', lw=1.5, label=r'$l^{-0.5}$')
        ax.set(title='Cross-Model: Propagator G(l)', xlabel='log10(l)', ylabel='log10(G)')
        ax.legend(fontsize=8, ncol=2)
        save(plt, fig, 8, 'Propagator G(l)', w=10, h=5)

    # ── 9. Spread ──
    if 9 in want and cache['spread']:
        fig, ax = plt.subplots()
        for label, sr in cache['spread'].items():
            ax.plot(sr['u'], sr['mean'], color=_c(label), ls=_ls(label), lw=2.5, label=label)
        ax.axvline(1.0, ls='--', color='gray', lw=1)
        ax.set(title='Cross-Model: Spread Dynamics', xlabel='Volume time u', ylabel='Spread (ticks)')
        ax.legend(fontsize=8, ncol=2)
        save(plt, fig, 9, 'Spread Dynamics', w=10, h=5)

    # ── 10. Beta vs mb ──
    if 10 in want:
        fig, ax = plt.subplots()
        for label in imp:
            d = cache['mb_betas'][label]
            if d:
                mbs = sorted(d.keys())
                ax.plot(mbs, [d[m] for m in mbs], color=_c(label), ls=_ls(label),
                        lw=2, marker=_mk(label), ms=7, label=label)
        ax.axhline(0.5, ls='--', color='black', lw=1)
        ax.set(title=r'$\beta$ vs messages-between', xlabel='mb', ylabel=r'$\beta$')
        ax.legend(fontsize=8, ncol=2)
        save(plt, fig, 10, 'Beta vs mb', w=7, h=4.5)

    # ── 11. Beta vs Volume ──
    if 11 in want:
        fig, ax = plt.subplots()
        for label in imp:
            d = cache['vol_betas'][label]
            if d:
                vols = sorted(d.keys())
                ax.plot(vols, [d[v] for v in vols], color=_c(label), ls=_ls(label),
                        lw=2, marker=_mk(label), ms=7, label=label)
        ax.axhline(0.5, ls='--', color='black', lw=1)
        ax.set(title=r'$\beta$ vs order volume', xlabel='Order volume', ylabel=r'$\beta$')
        ax.legend(fontsize=8, ncol=2)
        save(plt, fig, 11, 'Beta vs Volume', w=7, h=4.5)

    # ── Summary CSV ──
    summary = []
    for label in imp:
        br = cache['beta'][label]
        summary.append(dict(
            Model=label, beta=br['beta'], CI_lo=br['ci_lo'], CI_hi=br['ci_hi'],
            R2=br['r2'], N=br['n'],
            relaxation=np.median(cache['relax'].get(label, [np.nan])),
            stable_frac=cache['stability'].get(label, {}).get('frac', np.nan),
            Hurst=cache['hurst'].get(label, np.nan),
            no_arb_score=cache['scorecard'].get(label, {}).get('score', np.nan)))
    pd.DataFrame(summary).to_csv(FIG_DIR / 'summary_statistics.csv', index=False)
    print(f'\n  Summary → {FIG_DIR}/summary_statistics.csv')
    print(f'  {len(list(FIG_DIR.glob("*.png")))} figures in {FIG_DIR}/')


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='210 Full Grid Analysis')
    sub = parser.add_subparsers(dest='cmd')

    p_comp = sub.add_parser('compute', help='Phase 1: load data + compute')
    p_comp.add_argument('--stock', type=str, default='GOOG', help='Stock ticker (GOOG, INTC)')
    p_comp.add_argument('--max', type=int, default=MAX_SAMPLES, help='MAX_SAMPLES override')

    p_fig = sub.add_parser('figures', help='Phase 2: generate figures from cache')
    p_fig.add_argument('--stock', type=str, default='GOOG', help='Stock ticker (GOOG, INTC)')
    p_fig.add_argument('--figs', type=str, default=None, help='Comma-separated figure numbers, e.g. 1,3,5')
    p_fig.add_argument('--list', action='store_true', help='List available figures')

    args = parser.parse_args()

    if args.cmd == 'compute':
        run_compute(args.stock, args.max)
    elif args.cmd == 'figures':
        if args.list:
            print('\nAvailable figures:')
            for num, name in FIGURE_CATALOG.items():
                print(f'  [{num:>2}] {name}')
            sys.exit(0)
        FIG_DIR, CACHE_FILE = _dirs(args.stock)
        if not CACHE_FILE.exists():
            print(f'ERROR: {CACHE_FILE} not found. Run "compute --stock {args.stock}" first.')
            sys.exit(1)
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        selected = [int(x) for x in args.figs.split(',')] if args.figs else None
        run_figures(cache, selected)
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python lob_impact/run_210_analysis.py compute --stock GOOG')
        print('  python lob_impact/run_210_analysis.py compute --stock INTC --max 2')
        print('  python lob_impact/run_210_analysis.py figures --stock GOOG')
        print('  python lob_impact/run_210_analysis.py figures --stock INTC --figs 1,3,5')
