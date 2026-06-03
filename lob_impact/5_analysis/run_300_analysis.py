#!/usr/bin/env python3
"""
300 · LobS5-v2 (twilight-sound-77) standalone analysis

Quick-look at the original 2022 LobS5 model on Jan 2026 data.
Same metrics as 210, but just one model — fast to run and inspect.

Usage:
  python lob_impact/run_300_analysis.py compute --stock GOOG
  python lob_impact/run_300_analysis.py compute --stock INTC --max 2
  python lob_impact/run_300_analysis.py figures --stock GOOG
  python lob_impact/run_300_analysis.py figures --stock GOOG --figs 2,3,5
  python lob_impact/run_300_analysis.py figures --list
"""
import argparse, pickle, sys, re, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from scipy import optimize
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
MAX_SAMPLES   = 2048
N_BOOTSTRAP   = 1000
TICK_SIZE     = 100
STOCK         = 'GOOG'

SAVE_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v3')

MODEL_NAME  = 'LobS5-v2'
MODEL_COLOR = '#2CA02C'
MODEL_DASH  = '--'
MODEL_MK    = 'p'

def _dirs(stock):
    d = Path(f'pics_for_300_{stock}')
    return d, d / 'results_cache.pkl'

FIG_DIR, CACHE_FILE = _dirs(STOCK)


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

def _parse_day(filename):
    """Extract day string (YYYY-MM-DD) from experiment filename."""
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return m.group(1) if m else None

def load_folder_data(folder_row, max_samples):
    buy_gen = Path(folder_row['buy_path']) / 'data_gen'
    sell_gen = Path(folder_row['sell_path']) / 'data_gen'
    buy_books, sell_books, buy_msgs, sell_msgs = [], [], [], []
    buy_days, sell_days = [], []
    for gen_dir, books, msgs, days in [
        (buy_gen, buy_books, buy_msgs, buy_days),
        (sell_gen, sell_books, sell_msgs, sell_days),
    ]:
        if not gen_dir.exists(): continue
        for f in sorted(gen_dir.glob('*_orderbook_*_gen_id_0.csv'))[:max_samples]:
            try:
                books.append(pd.read_csv(f, header=None).values)
                days.append(_parse_day(f.name))
            except: pass
        for f in sorted(gen_dir.glob('*_message_*_gen_id_0.csv'))[:max_samples]:
            try: msgs.append(pd.read_csv(f, header=None).values)
            except: pass
    return dict(buy_books=buy_books, sell_books=sell_books,
                buy_msgs=buy_msgs, sell_msgs=sell_msgs,
                buy_days=buy_days, sell_days=sell_days,
                n_buy=len(buy_books), n_sell=len(sell_books))

def load_model(stock, max_samples):
    buy_root = str(SAVE_BASE / MODEL_NAME / 'context_500_buy' / stock)
    sell_root = str(SAVE_BASE / MODEL_NAME / 'context_500_sell' / stock)
    print(f"Loading {stock}/{MODEL_NAME}...")
    grid_df = discover_folders(buy_root, sell_root)
    if grid_df.empty:
        print("  No data found!"); return None
    data = {}
    for _, row in grid_df.iterrows():
        data[row['folder']] = load_folder_data(row, max_samples)
    n_buy = sum(d['n_buy'] for d in data.values())
    n_sell = sum(d['n_sell'] for d in data.values())
    print(f"  {len(grid_df)} folders, {n_buy} buy + {n_sell} sell samples")
    return dict(grid=grid_df, data=data)


def load_daily_params(daily_hl_path, exp_days):
    """Load daily H/L CSV and build per-day params dict keyed by experiment day."""
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
        if raw_day is None or pd.isna(raw_day):
            continue
        exp_day = day_map.get(raw_day, raw_day)
        H, L = row['highest_price'], row['lowest_price']
        V = row['execution_sum']
        sigma = np.log(H / L) / 0.8325546 if H > 0 and L > 0 else 1.0
        params[exp_day] = dict(H=H, L=L, V=V, sigma=sigma)

    print(f'  Loaded daily params for {len(params)} days')
    for d in sorted(params):
        p = params[d]
        print(f'    {d}: H={p["H"]:.0f}  L={p["L"]:.0f}  V={p["V"]:.0f}  sigma={p["sigma"]:.6f}')
    return params


def _collect_experiment_days(data):
    """Collect all unique experiment days from loaded data."""
    days = set()
    for fd in data.values():
        for d in fd.get('buy_days', []):
            if d: days.add(d)
        for d in fd.get('sell_days', []):
            if d: days.add(d)
    return days


# ═══════════════════════════════════════════════════════════════════════
# Analysis helpers (same as 210)
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

def extract_point_cloud(data, grid_df, daily_params=None):
    points = []
    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in data: continue
        fd = data[folder]
        buy_rets, sell_rets = compute_combined_impact(fd['buy_books'], fd['sell_books'])
        buy_days = fd.get('buy_days', [None]*len(fd['buy_books']))
        sell_days = fd.get('sell_days', [None]*len(fd['sell_books']))
        n_pairs = min(len(buy_rets), len(sell_rets))
        for j in range(n_pairs):
            br, sr = buy_rets[j], sell_rets[j]
            L = min(len(br), len(sr))
            if L < 2: continue
            insert_end = min(row['i'] * (row['mb'] + 1), L - 1)
            combined = (br[insert_end] - sr[insert_end]) / 2.0
            if combined <= 1e-10: continue
            pt = dict(Q=row['i'] * row['vol'], I=combined,
                      vol=row['vol'], i=row['i'], mb=row['mb'], sample_id=j)
            if daily_params and j < len(buy_days) and buy_days[j] in daily_params:
                dp = daily_params[buy_days[j]]
                pt['daily_vol'] = dp['V']
                pt['daily_sigma'] = dp['sigma']
            points.append(pt)
    return pd.DataFrame(points)

def compute_global_beta(pc_df, daily_vol=1e6, daily_sigma=1.0):
    if pc_df.empty: return dict(beta=np.nan, r2=np.nan, n=0)
    if 'daily_vol' in pc_df.columns:
        x = np.log(pc_df['Q'].values / pc_df['daily_vol'].values)
        y = np.log(pc_df['I'].values / pc_df['daily_sigma'].values)
    else:
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
def run_compute(stock, max_samples, daily_hl_path=None):
    global FIG_DIR, CACHE_FILE
    FIG_DIR, CACHE_FILE = _dirs(stock)
    FIG_DIR.mkdir(exist_ok=True)

    print('=' * 70)
    print(f'  300: COMPUTE — {stock}, {MODEL_NAME}, MAX_SAMPLES={max_samples}')
    print('=' * 70)

    rd = load_model(stock, max_samples)
    if rd is None:
        print("No data — exiting."); sys.exit(1)

    # Load per-day normalization parameters
    daily_params = None
    if daily_hl_path:
        exp_days = _collect_experiment_days(rd['data'])
        daily_params = load_daily_params(daily_hl_path, exp_days)

    cache = dict(stock=stock, model=MODEL_NAME)

    # Beta
    pc = extract_point_cloud(rd['data'], rd['grid'], daily_params)
    res = compute_global_beta(pc)
    boots = bootstrap_beta(pc)
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) > 0 else (np.nan, np.nan)
    if not pc.empty and 'daily_vol' in pc.columns:
        pc_x = np.log(pc['Q'].values / pc['daily_vol'].values)
        pc_y = np.log(pc['I'].values / pc['daily_sigma'].values)
    elif not pc.empty:
        pc_x = np.log(pc['Q'].values / 1e6)
        pc_y = np.log(pc['I'].values / 1.0)
    else:
        pc_x, pc_y = np.array([]), np.array([])
    cache['beta'] = dict(beta=res['beta'], r2=res['r2'], n=res['n'],
                          ci_lo=ci[0], ci_hi=ci[1], boots=boots,
                          pc_x=pc_x, pc_y=pc_y)
    print(f'  beta={res["beta"]:.4f}  R2={res["r2"]:.4f}  n={res["n"]}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]')

    # Master curves
    mcs = {}
    for _, frow in rd['grid'].iterrows():
        folder = frow['folder']
        if folder not in rd['data']: continue
        mc = compute_master_curve(rd['data'][folder], folder)
        if mc is not None: mcs[folder] = mc
    cache['master_curves'] = mcs
    print(f'  {len(mcs)} master curves')

    # Relaxation
    ratios = [compute_relaxation_ratio(mc) for mc in mcs.values()]
    cache['relax'] = [r for r in ratios if not np.isnan(r)]
    if cache['relax']:
        print(f'  relaxation median={np.median(cache["relax"]):.3f}')

    # Stability
    sc, tc = 0, 0
    for _, frow in rd['grid'].iterrows():
        folder = frow['folder']
        if folder not in rd['data']: continue
        mc = compute_master_curve(rd['data'][folder], folder)
        s = stability_vote(mc)
        tc += 1
        if s['stable']: sc += 1
    cache['stability'] = dict(stable=sc, total=tc, frac=sc/max(tc,1))
    print(f'  stability: {sc}/{tc} ({sc/max(tc,1):.0%})')

    # Decay gamma
    gammas = []
    for mc in mcs.values():
        fd = fit_decay(mc['u'], mc['mean'])
        if not np.isnan(fd['gamma_power']): gammas.append(fd['gamma_power'])
    cache['gamma'] = np.mean(gammas) if gammas else np.nan
    if gammas:
        print(f'  gamma={cache["gamma"]:.3f}')

    # Scorecard
    beta = res['beta']
    r_mean = np.mean(cache['relax']) if cache['relax'] else np.nan
    gamma = cache['gamma']
    b_perm = beta * (r_mean if not np.isnan(r_mean) else 1.0)
    A = beta < 1
    B = 0.7 <= b_perm <= 1.3 if not np.isnan(b_perm) else False
    C = 0.3 <= gamma <= 1.0 if not np.isnan(gamma) else False
    D = 0.5 <= r_mean <= 1.0 if not np.isnan(r_mean) else False
    E = beta <= 1/(1+2*gamma) if not np.isnan(gamma) else False
    cache['scorecard'] = dict(b_perm=b_perm, r=r_mean, gamma=gamma,
                               A=A, B=B, C=C, D=D, E=E, score=sum([A,B,C,D,E]))
    print(f'  scorecard: {sum([A,B,C,D,E])}/5  (A={A} B={B} C={C} D={D} E={E})')

    # Hurst
    all_signs = []
    for fd in rd['data'].values():
        for msgs in fd.get('buy_msgs',[]) + fd.get('sell_msgs',[]):
            all_signs.extend(extract_order_signs([msgs]))
    if all_signs:
        concat = np.concatenate(all_signs)
        if len(concat) >= 100:
            cache['hurst'] = compute_hurst_dfa(concat)
            print(f'  Hurst={cache["hurst"]:.3f}')

    # Propagator
    ma, ba = [], []
    for fd in rd['data'].values():
        for m, b in zip(fd.get('buy_msgs',[]), fd.get('buy_books',[])):
            ma.append(m); ba.append(b)
        for m, b in zip(fd.get('sell_msgs',[]), fd.get('sell_books',[])):
            ma.append(m); ba.append(b)
    if ma:
        G, lags = compute_propagator(ma, ba)
        cache['propagator'] = dict(G=G, lags=lags)

    # Spread
    all_sc = []
    for _, frow in rd['grid'].iterrows():
        folder = frow['folder']
        if folder not in rd['data']: continue
        fd = rd['data'][folder]
        sc_res = compute_spread_trajectory(fd['buy_books'], fd['sell_books'],
                                           frow['i'] * (frow['mb'] + 1))
        if sc_res is not None: all_sc.append(sc_res)
    if all_sc:
        u = all_sc[0]['u']
        avg = np.nanmean([c['mean'] for c in all_sc], axis=0)
        cache['spread'] = dict(u=u, mean=avg)

    # Beta vs mb, vol
    mb_b, vol_b = {}, {}
    for _, frow in rd['grid'].iterrows():
        folder = frow['folder']
        if folder not in rd['data']: continue
        pc_f = extract_point_cloud({folder: rd['data'][folder]}, pd.DataFrame([frow]), daily_params)
        if pc_f.empty: continue
        res_f = compute_global_beta(pc_f)
        if not np.isnan(res_f['beta']):
            mb_b.setdefault(frow['mb'], []).append(res_f['beta'])
            vol_b.setdefault(frow['vol'], []).append(res_f['beta'])
    cache['mb_betas'] = {k: np.mean(v) for k, v in mb_b.items()}
    cache['vol_betas'] = {k: np.mean(v) for k, v in vol_b.items()}

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'\nCache saved: {CACHE_FILE} ({CACHE_FILE.stat().st_size/1024:.0f} KB)')
    return cache


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: FIGURES
# ═══════════════════════════════════════════════════════════════════════
FIGURE_CATALOG = OrderedDict([
    (1,  'Master Curves'),
    (2,  'Average Master Curve'),
    (3,  'Beta Regression'),
    (4,  'Bootstrap Beta'),
    (5,  'Relaxation Ratio'),
    (6,  'Propagator G(l)'),
    (7,  'Spread Dynamics'),
    (8,  'Beta vs mb'),
    (9,  'Beta vs Volume'),
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


def run_figures(cache, selected=None):
    plt = setup_mpl()
    FIG_DIR.mkdir(exist_ok=True)
    want = set(selected) if selected else set(FIGURE_CATALOG.keys())
    stock = cache.get('stock', STOCK)

    print('=' * 70)
    print(f'  300: FIGURES — {stock}, {MODEL_NAME}, {len(want)} figures')
    print('=' * 70)

    C = MODEL_COLOR

    # ── 1. Master Curves (all on one panel) ──
    if 1 in want and cache.get('master_curves'):
        palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(cache['master_curves'])))
        fig, ax = plt.subplots()
        for fi, (folder, mc) in enumerate(cache['master_curves'].items()):
            mask = mc['u'] <= 3.0
            ax.plot(mc['u'][mask], mc['mean'][mask], color=palette[fi], lw=0.8, alpha=0.7)
        ax.axvline(1.0, ls='--', color='gray', lw=1)
        ax.set(title=f'{MODEL_NAME} — Master Curves ({stock})', xlabel='u', ylabel=r'$I_{\rm norm}(u)$')
        save(plt, fig, 1, 'Master Curves', w=9, h=5)

    # ── 2. Average Master Curve ──
    if 2 in want and cache.get('master_curves'):
        mcs = cache['master_curves']
        all_m = [mc['mean'] for mc in mcs.values()]
        u = list(mcs.values())[0]['u']
        avg, std = np.nanmean(all_m, axis=0), np.nanstd(all_m, axis=0)
        fig, ax = plt.subplots()
        ax.plot(u, avg, color=C, lw=3, label=MODEL_NAME)
        ax.fill_between(u, avg-std, avg+std, color=C, alpha=0.15)
        ax.axvline(1.0, ls='--', color='gray', lw=1.5)
        ax.set(title=f'{MODEL_NAME} — Average Master Curve ({stock})', xlabel='u', ylabel=r'$I_{\rm norm}(u)$')
        ax.legend()
        save(plt, fig, 2, 'Average Master Curve', w=9, h=5)

    # ── 3. Beta Regression ──
    if 3 in want and cache.get('beta'):
        br = cache['beta']
        x, y = br['pc_x'], br['pc_y']
        fig, ax = plt.subplots()
        if len(x) > 0:
            ax.scatter(x, y, s=5, color=C, alpha=0.15, rasterized=True)
            xl = np.array([x.min(), x.max()])
            ax.plot(xl, br['beta']*xl, color=C, lw=3, label=f"{MODEL_NAME}: {br['beta']:.3f}")
            ax.plot([-15,-5], [0.5*-15, 0.5*-5], 'k--', lw=2, label='Theory: 0.5')
        ax.set(title=f'{MODEL_NAME} — Beta Regression ({stock})',
               xlabel='log(Q/V)', ylabel=r'log(I/$\sigma$)')
        ax.legend()
        save(plt, fig, 3, 'Beta Regression', w=8, h=6)

    # ── 4. Bootstrap Beta ──
    if 4 in want and cache.get('beta'):
        boots = cache['beta']['boots']
        if len(boots) > 0:
            fig, ax = plt.subplots()
            ax.hist(boots, bins=50, color=C, edgecolor='white', lw=0.3, alpha=0.7)
            ax.axvline(0.5, ls='--', color='red', lw=2.5, label='Theory: 0.5')
            ax.axvline(np.median(boots), ls='-', color='black', lw=2,
                       label=f'Median: {np.median(boots):.3f}')
            ax.set(title=f'{MODEL_NAME} — Bootstrap Beta ({stock})', xlabel=r'$\beta$', ylabel='Count')
            ax.legend()
            save(plt, fig, 4, 'Bootstrap Beta', w=7, h=4.5)

    # ── 5. Relaxation Ratio ──
    if 5 in want and cache.get('relax'):
        ratios = cache['relax']
        if ratios:
            fig, ax = plt.subplots()
            ax.hist(ratios, bins=30, color=C, edgecolor='white', lw=0.3, alpha=0.7)
            ax.axvline(2/3, ls='--', color='red', lw=2, label='Bouchaud 2/3')
            ax.axvline(np.median(ratios), ls='-', color='black', lw=2,
                       label=f'Median: {np.median(ratios):.3f}')
            ax.set(title=f'{MODEL_NAME} — Relaxation Ratio ({stock})',
                   xlabel=r'$I_{\rm final}/I_{\rm peak}$', ylabel='Count')
            ax.legend()
            save(plt, fig, 5, 'Relaxation Ratio', w=7, h=4.5)

    # ── 6. Propagator ──
    if 6 in want and cache.get('propagator'):
        pr = cache['propagator']
        G, lags = pr['G'], pr['lags']
        mask = (lags > 0) & (G > 0)
        if mask.sum() >= 3:
            fig, ax = plt.subplots()
            ax.plot(np.log10(lags[mask]), np.log10(G[mask]), color=C, lw=2.5, label=MODEL_NAME)
            l_th = np.logspace(0, 2.3, 50)
            ax.plot(np.log10(l_th), np.log10(l_th**(-0.5)*0.1), 'k--', lw=1.5, label=r'$l^{-0.5}$')
            ax.set(title=f'{MODEL_NAME} — Propagator G(l) ({stock})',
                   xlabel='log10(l)', ylabel='log10(G)')
            ax.legend()
            save(plt, fig, 6, 'Propagator G(l)', w=9, h=5)

    # ── 7. Spread ──
    if 7 in want and cache.get('spread'):
        sr = cache['spread']
        fig, ax = plt.subplots()
        ax.plot(sr['u'], sr['mean'], color=C, lw=2.5, label=MODEL_NAME)
        ax.axvline(1.0, ls='--', color='gray', lw=1)
        ax.set(title=f'{MODEL_NAME} — Spread Dynamics ({stock})',
               xlabel='Volume time u', ylabel='Spread (ticks)')
        ax.legend()
        save(plt, fig, 7, 'Spread Dynamics', w=9, h=5)

    # ── 8. Beta vs mb ──
    if 8 in want and cache.get('mb_betas'):
        d = cache['mb_betas']
        if d:
            fig, ax = plt.subplots()
            mbs = sorted(d.keys())
            ax.plot(mbs, [d[m] for m in mbs], color=C, lw=2, marker=MODEL_MK, ms=8)
            ax.axhline(0.5, ls='--', color='black', lw=1)
            ax.set(title=f'{MODEL_NAME} — Beta vs mb ({stock})',
                   xlabel='mb', ylabel=r'$\beta$')
            save(plt, fig, 8, 'Beta vs mb', w=7, h=4.5)

    # ── 9. Beta vs Volume ──
    if 9 in want and cache.get('vol_betas'):
        d = cache['vol_betas']
        if d:
            fig, ax = plt.subplots()
            vols = sorted(d.keys())
            ax.plot(vols, [d[v] for v in vols], color=C, lw=2, marker=MODEL_MK, ms=8)
            ax.axhline(0.5, ls='--', color='black', lw=1)
            ax.set(title=f'{MODEL_NAME} — Beta vs Volume ({stock})',
                   xlabel='Order volume', ylabel=r'$\beta$')
            save(plt, fig, 9, 'Beta vs Volume', w=7, h=4.5)

    # ── Summary text ──
    br = cache['beta']
    sc = cache['scorecard']
    lines = [
        f"{'='*50}",
        f"  {MODEL_NAME} — {stock} Summary",
        f"{'='*50}",
        f"  beta       = {br['beta']:.4f}  CI=[{br['ci_lo']:.4f}, {br['ci_hi']:.4f}]",
        f"  R2         = {br['r2']:.4f}",
        f"  N          = {br['n']}",
        f"  relaxation = {sc['r']:.3f}" if not np.isnan(sc['r']) else "  relaxation = N/A",
        f"  gamma      = {sc['gamma']:.3f}" if not np.isnan(sc['gamma']) else "  gamma      = N/A",
        f"  Hurst      = {cache.get('hurst', np.nan):.3f}" if not np.isnan(cache.get('hurst', np.nan)) else "  Hurst      = N/A",
        f"  stability  = {cache['stability']['stable']}/{cache['stability']['total']}",
        f"  scorecard  = {sc['score']}/5",
    ]
    print('\n'.join(lines))

    summary = pd.DataFrame([dict(
        Model=MODEL_NAME, beta=br['beta'], CI_lo=br['ci_lo'], CI_hi=br['ci_hi'],
        R2=br['r2'], N=br['n'],
        relaxation=np.median(cache['relax']) if cache['relax'] else np.nan,
        stable_frac=cache['stability']['frac'],
        Hurst=cache.get('hurst', np.nan),
        gamma=sc['gamma'], b_perm=sc['b_perm'],
        no_arb_score=sc['score'])])
    summary.to_csv(FIG_DIR / 'summary_statistics.csv', index=False)
    print(f'\n  Summary -> {FIG_DIR}/summary_statistics.csv')
    print(f'  {len(list(FIG_DIR.glob("*.png")))} figures in {FIG_DIR}/')


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'300: {MODEL_NAME} standalone analysis')
    sub = parser.add_subparsers(dest='cmd')

    p_comp = sub.add_parser('compute', help='Phase 1: load data + compute')
    p_comp.add_argument('--stock', type=str, default='GOOG')
    p_comp.add_argument('--max', type=int, default=MAX_SAMPLES)
    p_comp.add_argument('--daily_hl', type=str, default=None,
                        help='Path to daily_h_l CSV for per-day V/sigma normalization')

    p_fig = sub.add_parser('figures', help='Phase 2: generate figures from cache')
    p_fig.add_argument('--stock', type=str, default='GOOG')
    p_fig.add_argument('--figs', type=str, default=None)
    p_fig.add_argument('--list', action='store_true')

    args = parser.parse_args()

    if args.cmd == 'compute':
        run_compute(args.stock, args.max, getattr(args, 'daily_hl', None))
    elif args.cmd == 'figures':
        if args.list:
            print(f'\n{MODEL_NAME} — Available figures:')
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
        print(f'\nExamples:')
        print(f'  python lob_impact/run_300_analysis.py compute --stock GOOG')
        print(f'  python lob_impact/run_300_analysis.py figures --stock GOOG')
        print(f'  python lob_impact/run_300_analysis.py figures --stock GOOG --figs 2,3,5')
