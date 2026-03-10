#!/usr/bin/env python3
# Auto-generated from 170.fair_beta_comparison.ipynb
# MAX_SAMPLES = 2048

import sys, time
print("Start:", time.strftime("%Y-%m-%d %H:%M:%S"))
sys.stdout.flush()

# ============================================================
# Cell 0  (id: cell-0-setup)
# ============================================================
print("\n" + "="*60)
print("Running cell 0 (cell-0-setup)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 0: Setup & imports
# ══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import re, gc, math, json
from pathlib import Path
from collections import OrderedDict
from scipy.stats import linregress, ks_2samp, mannwhitneyu
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──
TICK_SIZE = 100
MAX_SAMPLES = 2048
MIDPRICE_MAX = None
N_BOOTSTRAP = 1000
GRID = 'v3'
N_COND = 500  # n_cond_msgs for context% calculation
N_REPEATS = 100  # for repeated subsampling

ENABLED = [
    'Historic',
    'Heuristic',
    'CST',
    'LobS5',
    'CGAN',
]

_ALL_SCENARIOS = OrderedDict([
    ('Historic',    {'key': 'historic_scenario',            'color': '#8F939A', 'dash': 'dash'}),
    ('Heuristic',   {'key': 'heuristic_scenario',           'color': '#2F5DA3', 'dash': 'dot'}),
    ('CST',         {'key': 'cst_scenario',                 'color': '#5B4B8A', 'dash': 'dashdot'}),
    ('LobS5',      {'key': 'aggressive_scenario',           'color': '#D09A3C', 'dash': 'solid'}),
    ('CGAN',        {'key': 'cgan_aggressive_scenario',     'color': '#7B4F9E', 'dash': 'longdash'}),
    ('RWKV',        {'key': 'rwkv_aggressive_scenario',     'color': '#D1637B', 'dash': 'longdashdot'}),
])
SCENARIOS = OrderedDict((k, v) for k, v in _ALL_SCENARIOS.items() if k in ENABLED)

_GRID_DIRS = [GRID] if GRID != 'all' else ['c10x_v2', 'v3', 'v4']

# ── Paths (auto-detect Docker vs host) ──
_BASE = [Path('/app/output/evalsequences'),
         Path('/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences')]
EVAL_BASE = next((p for p in _BASE if p.exists()), _BASE[-1])

_SDM = [Path('/app/lob_impact/sample_day_map.csv'),
        Path('/scratch/local/homes/80/georgenigm/LOBS5/lob_impact/sample_day_map.csv')]
SDM_PATH = next((p for p in _SDM if p.exists()), _SDM[-1])
SAMPLE_DAY_MAP = pd.read_csv(SDM_PATH)

_FIG = [Path('/app/pics_for_fair_beta_v3'),
        Path('/homes/80/georgenigm/LOBS5/pics_for_fair_beta_v3')]
FIG_DIR = next((p for p in _FIG if p.exists() or p.parent.exists()), _FIG[0])
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f'GRID      : {GRID}')
print(f'ENABLED   : {list(SCENARIOS.keys())}  ({len(SCENARIOS)}/{len(_ALL_SCENARIOS)})')
print(f'EVAL_BASE : {EVAL_BASE}')
print(f'SDM       : {len(SAMPLE_DAY_MAP)} rows')
print(f'FIG_DIR   : {FIG_DIR}')
print(f'MAX_SAMPLES: {MAX_SAMPLES}')

# ── Helpers ──
def hex_to_rgba(hex_color, alpha=0.12):
    """Convert '#RRGGBB' to 'rgba(R,G,B,alpha)' for Plotly fillcolor."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ── Publication figure style ──
SINGLE_W = 520
FULL_W   = 1080
IMG_SCALE = 3

_AX = dict(
    showline=True, linewidth=1.5, linecolor='black', mirror=True,
    showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.08)',
    ticks='outside', tickwidth=1, ticklen=4, tickcolor='black',
    zeroline=False,
)

def pub_layout(fig, width=SINGLE_W, height=None, legend_pos='tr', **kw):
    if height is None:
        height = int(width * 0.75)
    leg = {
        'tr': dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
        'br': dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom'),
        'tl': dict(x=0.02, y=0.98, xanchor='left',  yanchor='top'),
        'bl': dict(x=0.02, y=0.02, xanchor='left',  yanchor='bottom'),
        'tc': dict(x=0.3, y=0.98, xanchor='center', yanchor='top'),
        'none': dict(visible=False),
    }.get(legend_pos, {})
    fig.update_layout(
        width=width, height=height,
        template='plotly_white',
        font=dict(family='Times New Roman, DejaVu Serif, serif', size=13, color='black'),
        title=None,
        margin=dict(l=60, r=15, t=15, b=55),
        legend=dict(**leg, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='black', borderwidth=1, font_size=11),
        **kw,
    )
    fig.update_xaxes(**_AX)
    fig.update_yaxes(**_AX)
    return fig

def save_fig(fig, name):
    try:
        fig.write_image(str(FIG_DIR / name), scale=IMG_SCALE)
        print(f'  Saved: {name}')
    except Exception as e:
        # Fallback to HTML if kaleido not available
        html_name = name.rsplit('.', 1)[0] + '.html'
        fig.write_html(str(FIG_DIR / html_name))
        print(f'  Saved (HTML fallback): {html_name}  (kaleido error: {e})')

print(f"Cell 0 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 1  (id: cell-1-data-loading)
# ============================================================
print("\n" + "="*60)
print("Running cell 1 (cell-1-data-loading)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 1: Data I/O helpers + load all models
# ══════════════════════════════════════════════════════════════════

def discover_v2_folders(buy_path, sell_path):
    pattern = re.compile(r'^i(\d+)_c(\d+)_mb(\d+)_v(\d+)_cntxt(.+)$')
    rows = []
    for p in sorted(buy_path.iterdir()):
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        i, c, mb, V = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        sell_p = sell_path / p.name
        if not sell_p.exists():
            continue
        rows.append({'folder': p.name, 'i': i, 'c': c, 'mb': mb, 'V': V,
                     'Q_total': i * V, 'buy_path': p, 'sell_path': sell_p})
    return pd.DataFrame(rows)


def parse_folder_params_v2(folder_name):
    m = re.match(r'i(\d+)_c(\d+)_mb(\d+)_v(\d+)_cntxt(.+)', folder_name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None, None, None, None


def compute_midprice(book_array):
    return (book_array[:, 0] + book_array[:, 2]) / 2


def is_midprice_outlier(book_array, max_mp):
    mp = compute_midprice(book_array)
    return np.any(mp > max_mp) or np.any(mp <= 0)


def load_aggressive_indices(data_path):
    f = data_path / 'aggressive_indices.csv'
    if not f.exists():
        return np.array([], dtype=int)
    idx = np.loadtxt(f, dtype=int)
    return np.atleast_1d(idx)


def discover_data_params(data_path, max_samples=None):
    cond_dir = data_path / 'data_cond'
    pat = re.compile(r'^(.+?)_(\d{4}-\d{2}-\d{2})_orderbook_real_id_(\d+)\.csv$')
    samples = []
    for f in cond_dir.glob('*_orderbook_real_id_*.csv'):
        m = pat.match(f.name)
        if m:
            samples.append((m.group(1), m.group(2), int(m.group(3))))
    samples.sort()
    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    return samples


def load_folder_data(data_path, max_samples=None, max_midprice=None):
    samples = discover_data_params(data_path, max_samples)
    gen_books, gen_msgs, cond_lens = {}, {}, {}
    for ticker, date, sid in samples:
        cond_bp = data_path / f'data_cond/{ticker}_{date}_orderbook_real_id_{sid}.csv'
        gen_bp  = data_path / f'data_gen/{ticker}_{date}_orderbook_real_id_{sid}_gen_id_0.csv'
        gen_mp  = data_path / f'data_gen/{ticker}_{date}_message_real_id_{sid}_gen_id_0.csv'
        if not gen_bp.exists():
            continue
        cond_book = np.loadtxt(cond_bp, delimiter=',')
        gen_book  = np.loadtxt(gen_bp, delimiter=',')
        full_book = np.vstack([cond_book, gen_book])
        if max_midprice and is_midprice_outlier(full_book, max_midprice):
            continue
        gen_msg  = np.loadtxt(gen_mp, delimiter=',')
        cond_mp  = data_path / f'data_cond/{ticker}_{date}_message_real_id_{sid}.csv'
        cond_msg = np.loadtxt(cond_mp, delimiter=',')
        key = (date, sid)
        cond_lens[key] = cond_book.shape[0]
        gen_books[key] = full_book
        gen_msgs[key]  = np.vstack([cond_msg, gen_msg])
    return gen_books, gen_msgs, cond_lens


def load_all_v2(grid_df):
    all_data = {}
    for _, row in tqdm(grid_df.iterrows(), total=len(grid_df), desc='Loading'):
        try:
            bb, bm, bc = load_folder_data(row['buy_path'],  MAX_SAMPLES, MIDPRICE_MAX)
            sb, sm, sc = load_folder_data(row['sell_path'], MAX_SAMPLES, MIDPRICE_MAX)
            all_data[row['folder']] = {
                'buy':  {'books': bb, 'msgs': bm, 'cond_lens': bc},
                'sell': {'books': sb, 'msgs': sm, 'cond_lens': sc},
            }
        except Exception as e:
            print(f'  ERR {row["folder"]}: {e}')
    return all_data


# ── Load all models ──
MODEL_DATA = {}  # {model_name: {'grid': DataFrame, 'data': dict}}

for model_name, sc in SCENARIOS.items():
    print(f'\n{"="*60}')
    print(f'Loading: {model_name} ({sc["key"]})')
    print(f'{"="*60}')
    grids = []
    for gdir in _GRID_DIRS:
        buy_p  = EVAL_BASE / sc['key'] / gdir / 'context_500_buy'
        sell_p = EVAL_BASE / sc['key'] / gdir / 'context_500_sell'
        if buy_p.exists() and sell_p.exists():
            gdf = discover_v2_folders(buy_p, sell_p)
            if not gdf.empty:
                grids.append(gdf)
    if not grids:
        print(f'  No data found for {model_name}')
        continue
    grid_df = pd.concat(grids, ignore_index=True)
    print(f'  Found {len(grid_df)} configs')
    data = load_all_v2(grid_df)
    MODEL_DATA[model_name] = {'grid': grid_df, 'data': data}
    print(f'  Loaded {len(data)}/{len(grid_df)} folders')

print(f'\nModels loaded: {list(MODEL_DATA.keys())}')

print(f"Cell 1 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 2  (id: cell-2-point-cloud)
# ============================================================
print("\n" + "="*60)
print("Running cell 2 (cell-2-point-cloud)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 2: Point cloud extraction + fairness utilities
# ══════════════════════════════════════════════════════════════════

def extract_point_cloud_extended(data, grid_df):
    """Like extract_point_cloud from NB 150, but stores extra stratification fields."""
    eps = 1e-12
    rows = []
    for _, grow in grid_df.iterrows():
        folder = grow['folder']
        if folder not in data:
            continue
        d = data[folder]
        mb_val = grow['mb']
        i_val  = grow['i']
        V_val  = grow['V']
        aggr_buy  = load_aggressive_indices(grow['buy_path'])
        aggr_sell = load_aggressive_indices(grow['sell_path'])
        for direction, side, aggr_gen in [('BUY', d['buy'], aggr_buy),
                                           ('SELL', d['sell'], aggr_sell)]:
            if len(aggr_gen) == 0:
                continue
            books, msgs, conds = side['books'], side['msgs'], side['cond_lens']
            for sid in books:
                msg_arr, book_arr = msgs[sid], books[sid]
                junction = conds[sid]
                sample_id = sid[1]
                day = SAMPLE_DAY_MAP[SAMPLE_DAY_MAP['sample_id'] == sample_id]
                if day.empty:
                    continue
                H = float(day.iloc[0]['highest_price']) / TICK_SIZE
                L = float(day.iloc[0]['lowest_price'])  / TICK_SIZE
                V_day = float(day.iloc[0]['execution_sum'])
                if H <= L or L <= 0 or V_day <= eps:
                    continue
                sigma = np.log(H / L) / 0.8325546
                alpha = np.log(max(sigma, eps))
                aggr_idx = junction + aggr_gen
                aggr_idx = aggr_idx[aggr_idx < len(msg_arr)]
                if len(aggr_idx) < 2:
                    continue
                sizes  = msg_arr[aggr_idx, 3].astype(float)
                prices = msg_arr[aggr_idx, 4].astype(float)
                ref = (book_arr[aggr_idx[0], 0] + book_arr[aggr_idx[0], 2]) / 2
                if ref <= 0:
                    continue
                Q_cum = np.cumsum(sizes)
                vwap  = np.cumsum(sizes * prices) / np.maximum(Q_cum, eps)
                imp   = np.abs((vwap - ref) / ref) if direction == 'BUY' else np.abs((ref - vwap) / ref)
                # Context% = fraction of generated sequence used by insertions+coolings
                context_pct = i_val * (mb_val + 1) * 100.0 / N_COND
                for a in range(len(aggr_idx)):
                    if imp[a] > eps:
                        rows.append({
                            'x': np.log(Q_cum[a] / V_day),
                            'y': np.log(imp[a]),
                            'alpha': alpha,
                            'sample_id': sample_id,
                            'folder': folder,
                            'direction': direction,
                            'mb': mb_val,
                            'i': i_val,
                            'V': V_val,
                            'insertion_idx': a,
                            'Q_cum_a': float(Q_cum[a]),
                            'context_pct': context_pct,
                        })
    cols = ['x', 'y', 'alpha', 'sample_id', 'folder', 'direction',
            'mb', 'i', 'V', 'insertion_idx', 'Q_cum_a', 'context_pct']
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def compute_global_beta(df):
    if df.empty:
        return {'beta': np.nan, 'r2': np.nan, 'n': 0}
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    xv, yv = x[ok], y_adj[ok]
    if len(xv) < 2:
        return {'beta': np.nan, 'r2': np.nan, 'n': 0}
    beta = float(np.dot(xv, yv) / np.dot(xv, xv))
    ss_res = np.sum((yv - beta * xv) ** 2)
    ss_tot = np.sum(yv ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'beta': beta, 'r2': r2, 'n': int(ok.sum())}


def bootstrap_beta(df, n_boot=1000):
    if df.empty:
        return np.array([])
    pc = df[['x', 'y', 'alpha', 'sample_id']].copy()
    pc['y_adj'] = pc['y'] - pc['alpha']
    ok = np.isfinite(pc['x']) & np.isfinite(pc['y_adj']) & (pc['x'] != 0)
    pc = pc[ok]
    groups = {sid: g[['x', 'y_adj']].values for sid, g in pc.groupby('sample_id')}
    ids = np.array(list(groups.keys()))
    n = len(ids)
    if n == 0:
        return np.array([])
    rng = np.random.RandomState(42)
    betas = np.zeros(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(ids, size=n, replace=True)
        pool = np.vstack([groups[s] for s in chosen])
        x, y = pool[:, 0], pool[:, 1]
        betas[b] = np.dot(x, y) / np.dot(x, x)
    return betas


# ══════════════════════════════════════════════════════════════════
# NEW: Fairness utility functions
# ══════════════════════════════════════════════════════════════════

def subsample_to_n(df, group_col, target_n, seed=42):
    """Subsample each group to exactly target_n rows.
    Groups with fewer than target_n rows are kept as-is."""
    rng = np.random.RandomState(seed)
    parts = []
    for _, grp in df.groupby(group_col):
        if len(grp) <= target_n:
            parts.append(grp)
        else:
            idx = rng.choice(len(grp), size=target_n, replace=False)
            parts.append(grp.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def compute_per_config_beta(df, group_col='folder'):
    """Compute 1 beta per config group → each config contributes equally."""
    results = []
    for key, grp in df.groupby(group_col):
        g = compute_global_beta(grp)
        if np.isfinite(g['beta']) and g['n'] >= 2:
            results.append({'group': key, 'beta': g['beta'], 'r2': g['r2'], 'n': g['n']})
    return pd.DataFrame(results)


def weighted_beta(df, weight_col):
    """OLS beta with inverse-frequency weights: w_i = 1 / N_group_i."""
    if df.empty:
        return {'beta': np.nan, 'r2': np.nan, 'n': 0}
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    w = df[weight_col].values
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0) & np.isfinite(w) & (w > 0)
    xv, yv, wv = x[ok], y_adj[ok], w[ok]
    if len(xv) < 2:
        return {'beta': np.nan, 'r2': np.nan, 'n': 0}
    beta = float(np.dot(wv * xv, yv) / np.dot(wv * xv, xv))
    ss_res = np.sum(wv * (yv - beta * xv) ** 2)
    ss_tot = np.sum(wv * yv ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'beta': beta, 'r2': r2, 'n': int(ok.sum())}


def first_k_insertions(df, k):
    """Filter to only the first k insertions (insertion_idx < k)."""
    return df[df['insertion_idx'] < k].copy()


def difficulty_band(mb):
    """Classify mb into difficulty bands."""
    if mb <= 10:
        return 'Easy'
    elif mb <= 25:
        return 'Medium'
    else:
        return 'Hard'


def repeated_subsample_beta(df, group_col, n_repeats=100, seed=42):
    """Repeated subsampling: subsample each group to min N, compute beta.
    Returns array of n_repeats beta values."""
    group_sizes = df.groupby(group_col).size()
    min_n = int(group_sizes.min())
    if min_n < 2:
        return np.array([])
    rng = np.random.RandomState(seed)
    betas = np.zeros(n_repeats)
    for rep in range(n_repeats):
        sub = subsample_to_n(df, group_col, min_n, seed=rng.randint(0, 2**31))
        g = compute_global_beta(sub)
        betas[rep] = g['beta']
    return betas


def bootstrap_beta_from_subset(df, n_boot=500):
    """Convenience wrapper returning beta + CI dict."""
    g = compute_global_beta(df)
    boots = bootstrap_beta(df, n_boot)
    if len(boots) > 0:
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    else:
        ci_lo, ci_hi = np.nan, np.nan
    return {'beta': g['beta'], 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'r2': g['r2'], 'n': g['n']}


# ── Build point clouds for all models ──
CLOUDS = {}  # {model_name: DataFrame}
for model_name, md in MODEL_DATA.items():
    print(f'Extracting point cloud: {model_name} ...')
    pc = extract_point_cloud_extended(md['data'], md['grid'])
    CLOUDS[model_name] = pc
    g = compute_global_beta(pc)
    print(f'  {model_name}: {len(pc)} points, beta={g["beta"]:.3f}, R2={g["r2"]:.3f}')

print(f'\nPoint clouds built: {list(CLOUDS.keys())}')

print(f"Cell 2 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 3  (id: cell-3-imbalance)
# ============================================================
print("\n" + "="*60)
print("Running cell 3 (cell-3-imbalance)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 3: Diagnostic — The Imbalance Problem
# ══════════════════════════════════════════════════════════════════

_ref_model = list(CLOUDS.keys())[0]
_ref_pc = CLOUDS[_ref_model]

# ── Table: N points per (mb, V) pair ──
print('Data point distribution (reference model: {}):\n'.format(_ref_model))
mb_vals = sorted(_ref_pc['mb'].unique())
V_vals = sorted(_ref_pc['V'].unique())
total = len(_ref_pc)

print(f'{"mb":>5} {"i":>5} {"N_pts":>8} {"% total":>8}')
print('-' * 32)
for mb_val in mb_vals:
    sub = _ref_pc[_ref_pc['mb'] == mb_val]
    i_val = sub['i'].iloc[0]
    n = len(sub)
    print(f'{mb_val:5d} {i_val:5d} {n:8d} {100*n/total:7.1f}%')

# ── Per-config regression weight: w = sum(x^2_config) / sum(x^2_total) ──
print('\n\n=== Per-config regression weight decomposition ===')
print('(How much each mb group contributes to the OLS beta estimate)\n')

pc = _ref_pc.copy()
pc['y_adj'] = pc['y'] - pc['alpha']
ok = np.isfinite(pc['x']) & np.isfinite(pc['y_adj']) & (pc['x'] != 0)
pc_ok = pc[ok]
x2_total = np.sum(pc_ok['x'].values ** 2)

weight_rows = []
for mb_val in mb_vals:
    sub = pc_ok[pc_ok['mb'] == mb_val]
    x2_mb = np.sum(sub['x'].values ** 2)
    w = x2_mb / x2_total * 100
    weight_rows.append({'mb': mb_val, 'i': sub['i'].iloc[0] if len(sub) > 0 else 0,
                        'N': len(sub), 'x2_weight_pct': w})

wdf = pd.DataFrame(weight_rows)
print(f'{"mb":>5} {"i":>5} {"N":>8} {"OLS weight%":>12}')
print('-' * 35)
for _, r in wdf.iterrows():
    print(f'{int(r["mb"]):5d} {int(r["i"]):5d} {int(r["N"]):8d} {r["x2_weight_pct"]:11.1f}%')

top_mb = wdf.loc[wdf['x2_weight_pct'].idxmax()]
print(f'\n=> mb={int(top_mb["mb"])} contributes {top_mb["x2_weight_pct"]:.1f}% of regression weight '
      f'despite being the most mechanical scenario')

# ── Plot 3a: Points per mb ──
fig3a = go.Figure()
for model_name in SCENARIOS:
    if model_name not in CLOUDS:
        continue
    pc = CLOUDS[model_name]
    counts = [len(pc[pc['mb'] == mb]) for mb in mb_vals]
    fig3a.add_trace(go.Bar(
        x=[str(mb) for mb in mb_vals], y=counts, name=model_name,
        marker_color=SCENARIOS[model_name]['color'],
    ))
pub_layout(fig3a, width=FULL_W, height=400)
fig3a.update_layout(
    barmode='group',
    xaxis_title='Messages between insertions (mb)',
    yaxis_title='Number of data points',
)
save_fig(fig3a, '3a. Points per mb.png')
fig3a# .show()  # skip in script mode

# ── Plot 3b: Stacked bar — cumulative regression weight by mb ──
fig3b = go.Figure()
# Compute weight decomposition for each model
for model_name in SCENARIOS:
    if model_name not in CLOUDS:
        continue
    pc = CLOUDS[model_name].copy()
    pc['y_adj'] = pc['y'] - pc['alpha']
    ok = np.isfinite(pc['x']) & np.isfinite(pc['y_adj']) & (pc['x'] != 0)
    pc_ok = pc[ok]
    x2_tot = np.sum(pc_ok['x'].values ** 2)
    weights = []
    for mb in mb_vals:
        sub = pc_ok[pc_ok['mb'] == mb]
        w = np.sum(sub['x'].values ** 2) / x2_tot * 100 if x2_tot > 0 else 0
        weights.append(w)
    fig3b.add_trace(go.Bar(
        x=[str(mb) for mb in mb_vals], y=weights, name=model_name,
        marker_color=SCENARIOS[model_name]['color'],
    ))

pub_layout(fig3b, width=FULL_W, height=400)
fig3b.update_layout(
    barmode='group',
    xaxis_title='Messages between insertions (mb)',
    yaxis_title='OLS regression weight (%)',
)
save_fig(fig3b, '3b. Config weight decomposition.png')
fig3b# .show()  # skip in script mode

print(f"Cell 3 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 4  (id: cell-4-per-config-beta)
# ============================================================
print("\n" + "="*60)
print("Running cell 4 (cell-4-per-config-beta)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 4: Per-Config Beta Distributions (most fair methodology)
# ══════════════════════════════════════════════════════════════════

# Compute 1 beta per config folder → 21 betas per model
model_names = [m for m in SCENARIOS if m in CLOUDS]

per_config_betas = {}  # {model: DataFrame with columns [group, beta, r2, n]}
for model_name in model_names:
    pcb = compute_per_config_beta(CLOUDS[model_name], group_col='folder')
    per_config_betas[model_name] = pcb

# ── Summary table ──
print('Per-config beta statistics (1 beta per folder, each folder contributes equally):\n')
print(f'{"Model":<12} {"N configs":>10} {"Mean":>8} {"Median":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
print('-' * 70)
for model_name in model_names:
    pcb = per_config_betas[model_name]
    if len(pcb) == 0:
        continue
    b = pcb['beta']
    print(f'{model_name:<12} {len(b):>10} {b.mean():>8.3f} {b.median():>8.3f} '
          f'{b.std():>8.3f} {b.min():>8.3f} {b.max():>8.3f}')

# ── Plot 4a: Violin/box plot of per-config beta ──
all_pcb = []
for model_name in model_names:
    pcb = per_config_betas[model_name].copy()
    pcb['model'] = model_name
    all_pcb.append(pcb)
all_pcb_df = pd.concat(all_pcb, ignore_index=True)

fig4a = go.Figure()
for model_name in model_names:
    sub = all_pcb_df[all_pcb_df['model'] == model_name]
    fig4a.add_trace(go.Violin(
        y=sub['beta'], name=model_name,
        box_visible=True, meanline_visible=True,
        fillcolor=hex_to_rgba(SCENARIOS[model_name]['color'], 0.3),
        line_color=SCENARIOS[model_name]['color'],
        points='all', jitter=0.3, pointpos=-0.5,
    ))

fig4a.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
               annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig4a, width=FULL_W, height=500, legend_pos='none')
fig4a.update_layout(
    xaxis_title='Model',
    yaxis_title='Per-config beta',
)
save_fig(fig4a, '4a. Per-config beta violin.png')
fig4a# .show()  # skip in script mode

# ── Pairwise KS tests with Bonferroni correction ──
n_pairs = len(model_names) * (len(model_names) - 1) // 2
print(f'\nPairwise KS tests (Bonferroni-corrected, {n_pairs} pairs):\n')

ks_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)
ks_raw = pd.DataFrame(np.nan, index=model_names, columns=model_names)
for i, m1 in enumerate(model_names):
    ks_matrix.loc[m1, m1] = 1.0
    ks_raw.loc[m1, m1] = 1.0
    for j, m2 in enumerate(model_names):
        if j <= i:
            continue
        b1 = per_config_betas[m1]['beta'].values
        b2 = per_config_betas[m2]['beta'].values
        stat, p_raw = ks_2samp(b1, b2)
        p_corrected = min(p_raw * n_pairs, 1.0)  # Bonferroni
        ks_matrix.loc[m1, m2] = p_corrected
        ks_matrix.loc[m2, m1] = p_corrected
        ks_raw.loc[m1, m2] = p_raw
        ks_raw.loc[m2, m1] = p_raw

print('Bonferroni-corrected p-values:')
print(ks_matrix.to_string(float_format='{:.4f}'.format))

print('\nRaw p-values:')
print(ks_raw.to_string(float_format='{:.4f}'.format))

# ── Plot 4b: KS p-value matrix heatmap ──
fig4b = go.Figure(go.Heatmap(
    z=ks_matrix.values.astype(float),
    x=model_names, y=model_names,
    text=np.vectorize(lambda x: f'{x:.3f}')(ks_matrix.values.astype(float)),
    texttemplate='%{text}',
    colorscale='RdYlGn', zmin=0, zmax=1,
    colorbar=dict(title='p-value'),
))
pub_layout(fig4b, width=SINGLE_W, height=SINGLE_W)
save_fig(fig4b, '4b. KS test matrix.png')
fig4b# .show()  # skip in script mode

print(f"Cell 4 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 5  (id: cell-5-first-k)
# ============================================================
print("\n" + "="*60)
print("Running cell 5 (cell-5-first-k)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 5: First-k Insertions Analysis (natural equalization)
# ══════════════════════════════════════════════════════════════════

# min insertions across all configs = 4 (for i=4, mb=100)
# At k=1: exactly 1 data point per sample for ALL configs — zero imbalance

model_names = [m for m in SCENARIOS if m in CLOUDS]
k_values = [1, 2, 3, 4]

beta_by_k = {}  # {model: {k: {beta, ci_lo, ci_hi}}}
for model_name in model_names:
    beta_by_k[model_name] = {}
    for k in k_values:
        sub = first_k_insertions(CLOUDS[model_name], k)
        res = bootstrap_beta_from_subset(sub)
        beta_by_k[model_name][k] = res

# Print table
print('Beta vs k (first k insertions only):\n')
print(f'{"Model":<12}', end='')
for k in k_values:
    print(f'  k={k:<12}', end='')
print()
print('-' * (12 + 15 * len(k_values)))
for model_name in model_names:
    print(f'{model_name:<12}', end='')
    for k in k_values:
        d = beta_by_k[model_name][k]
        print(f'  {d["beta"]:.3f} [{d["ci_lo"]:.2f},{d["ci_hi"]:.2f}]', end='')
    print()

# ── Plot 5a: Line chart, beta vs k per model with CI bands ──
fig5a = go.Figure()
for model_name in model_names:
    sc = SCENARIOS[model_name]
    betas = [beta_by_k[model_name][k]['beta'] for k in k_values]
    ci_lo = [beta_by_k[model_name][k]['ci_lo'] for k in k_values]
    ci_hi = [beta_by_k[model_name][k]['ci_hi'] for k in k_values]
    # CI band
    fig5a.add_trace(go.Scatter(
        x=k_values + k_values[::-1],
        y=ci_hi + ci_lo[::-1],
        fill='toself', fillcolor=hex_to_rgba(sc['color']),
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig5a.add_trace(go.Scatter(
        x=k_values, y=betas, name=model_name, mode='lines+markers',
        line=dict(color=sc['color'], dash=sc['dash'], width=2),
        marker=dict(size=7),
    ))

fig5a.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
               annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig5a, width=FULL_W, height=450, legend_pos='tr')
fig5a.update_layout(
    xaxis_title='Number of insertions (k)',
    yaxis_title='Beta (power-law exponent)',
    xaxis=dict(dtick=1),
)
save_fig(fig5a, '5a. Beta vs k.png')
fig5a# .show()  # skip in script mode

# ── Plot 5b: Bar chart at k=1 — "first-impact" beta ──
fig5b = go.Figure()
betas_k1 = [beta_by_k[m][1]['beta'] for m in model_names]
ci_lo_k1 = [beta_by_k[m][1]['ci_lo'] for m in model_names]
ci_hi_k1 = [beta_by_k[m][1]['ci_hi'] for m in model_names]
colors_k1 = [SCENARIOS[m]['color'] for m in model_names]

fig5b.add_trace(go.Bar(
    x=model_names, y=betas_k1,
    marker_color=colors_k1,
    error_y=dict(type='data', symmetric=False,
                 array=[hi - b for b, hi in zip(betas_k1, ci_hi_k1)],
                 arrayminus=[b - lo for b, lo in zip(betas_k1, ci_lo_k1)]),
))
fig5b.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
               annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig5b, width=SINGLE_W, height=400, legend_pos='none')
fig5b.update_layout(
    xaxis_title='Model',
    yaxis_title='First-impact beta (k=1)',
)
save_fig(fig5b, '5b. First-impact beta (k=1).png')
fig5b# .show()  # skip in script mode

print(f"Cell 5 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 6  (id: cell-6-equalized-mb)
# ============================================================
print("\n" + "="*60)
print("Running cell 6 (cell-6-equalized-mb)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 6: N-Equalized Beta by mb
# ══════════════════════════════════════════════════════════════════

model_names = [m for m in SCENARIOS if m in CLOUDS]
mb_vals = sorted(CLOUDS[model_names[0]]['mb'].unique())
N_REPEATS = 100

print('N-Equalized beta by mb (repeated subsampling, {} reps):\n'.format(N_REPEATS))

# ── Raw beta by mb (for comparison) ──
raw_beta_mb = {}  # {model: {mb: beta}}
for model_name in model_names:
    raw_beta_mb[model_name] = {}
    for mb in mb_vals:
        sub = CLOUDS[model_name][CLOUDS[model_name]['mb'] == mb]
        g = compute_global_beta(sub)
        raw_beta_mb[model_name][mb] = g['beta']

# ── Equalized: subsample each mb group to min N, repeat ──
eq_beta_mb = {}  # {model: {mb: {mean, lo, hi}}}
for model_name in model_names:
    pc = CLOUDS[model_name]
    eq_beta_mb[model_name] = {}
    for mb in mb_vals:
        sub = pc[pc['mb'] == mb]
        eq_beta_mb[model_name][mb] = bootstrap_beta_from_subset(sub)

# Equalized global beta: subsample all mb groups to min N, compute global
eq_global = {}  # {model: {mean, lo, hi}}
for model_name in model_names:
    pc = CLOUDS[model_name]
    betas = repeated_subsample_beta(pc, 'mb', N_REPEATS)
    if len(betas) > 0:
        eq_global[model_name] = {
            'mean': np.mean(betas), 'lo': np.percentile(betas, 2.5),
            'hi': np.percentile(betas, 97.5), 'std': np.std(betas),
        }
    else:
        eq_global[model_name] = {'mean': np.nan, 'lo': np.nan, 'hi': np.nan, 'std': np.nan}

# Print: raw vs equalized global beta
print(f'{"Model":<12} {"Raw global":>12} {"Eq global":>12} {"Eq 95% CI":>20}')
print('-' * 60)
for model_name in model_names:
    g = compute_global_beta(CLOUDS[model_name])
    eq = eq_global[model_name]
    print(f'{model_name:<12} {g["beta"]:>12.3f} {eq["mean"]:>12.3f} '
          f'[{eq["lo"]:.3f}, {eq["hi"]:.3f}]')

# ── Plot 6: Side-by-side — raw vs equalized ──
fig6 = make_subplots(rows=1, cols=2,
    subplot_titles=['Raw beta by mb', 'N-equalized global beta'],
    horizontal_spacing=0.12,
)

# Left: raw beta by mb (line chart)
for model_name in model_names:
    sc = SCENARIOS[model_name]
    betas = [raw_beta_mb[model_name][mb] for mb in mb_vals]
    fig6.add_trace(go.Scatter(
        x=mb_vals, y=betas, name=model_name, mode='lines+markers',
        line=dict(color=sc['color'], dash=sc['dash'], width=2),
        marker=dict(size=6), showlegend=True,
    ), row=1, col=1)

fig6.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1, row=1, col=1)

# Right: equalized global beta (bar chart)
eq_betas = [eq_global[m]['mean'] for m in model_names]
eq_lo = [eq_global[m]['lo'] for m in model_names]
eq_hi = [eq_global[m]['hi'] for m in model_names]
colors = [SCENARIOS[m]['color'] for m in model_names]

fig6.add_trace(go.Bar(
    x=model_names, y=eq_betas,
    marker_color=colors,
    error_y=dict(type='data', symmetric=False,
                 array=[hi - b for b, hi in zip(eq_betas, eq_hi)],
                 arrayminus=[b - lo for b, lo in zip(eq_betas, eq_lo)]),
    showlegend=False,
), row=1, col=2)

fig6.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1, row=1, col=2)

pub_layout(fig6, width=FULL_W, height=450, legend_pos='tl')
fig6.update_xaxes(title_text='mb', row=1, col=1)
fig6.update_yaxes(title_text='Beta', row=1, col=1)
fig6.update_xaxes(title_text='Model', row=1, col=2)
fig6.update_yaxes(title_text='Equalized beta', row=1, col=2)
fig6.update_layout(margin=dict(t=40))
save_fig(fig6, '6. Raw vs equalized beta by mb.png')
fig6# .show()  # skip in script mode

print(f"Cell 6 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 7  (id: cell-7-equalized-i)
# ============================================================
print("\n" + "="*60)
print("Running cell 7 (cell-7-equalized-i)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 7: N-Equalized Beta by i (number of insertions)
# ══════════════════════════════════════════════════════════════════

model_names = [m for m in SCENARIOS if m in CLOUDS]
i_vals = sorted(CLOUDS[model_names[0]]['i'].unique())

print('N-Equalized beta by i (number of insertions):\n')

# Raw beta by i
raw_beta_i = {}
for model_name in model_names:
    raw_beta_i[model_name] = {}
    for i_val in i_vals:
        sub = CLOUDS[model_name][CLOUDS[model_name]['i'] == i_val]
        g = compute_global_beta(sub)
        raw_beta_i[model_name][i_val] = g['beta']

# Print raw beta by i
print(f'{"Model":<12}', end='')
for i_val in i_vals:
    print(f'  i={i_val:<5}', end='')
print()
print('-' * (12 + 9 * len(i_vals)))
for model_name in model_names:
    print(f'{model_name:<12}', end='')
    for i_val in i_vals:
        b = raw_beta_i[model_name][i_val]
        print(f'  {b:6.3f}  ', end='')
    print()

# Equalized global beta by i-strata
eq_global_i = {}
for model_name in model_names:
    pc = CLOUDS[model_name]
    betas = repeated_subsample_beta(pc, 'i', N_REPEATS)
    if len(betas) > 0:
        eq_global_i[model_name] = {
            'mean': np.mean(betas), 'lo': np.percentile(betas, 2.5),
            'hi': np.percentile(betas, 97.5),
        }
    else:
        eq_global_i[model_name] = {'mean': np.nan, 'lo': np.nan, 'hi': np.nan}

print(f'\nEqualized global beta (by i-strata):')
for model_name in model_names:
    eq = eq_global_i[model_name]
    print(f'  {model_name:<12}: {eq["mean"]:.3f} [{eq["lo"]:.3f}, {eq["hi"]:.3f}]')

# ── Plot 7: Equalized beta vs i ──
fig7 = go.Figure()
for model_name in model_names:
    sc = SCENARIOS[model_name]
    betas = [raw_beta_i[model_name][i_val] for i_val in i_vals]
    fig7.add_trace(go.Scatter(
        x=i_vals, y=betas, name=model_name, mode='lines+markers',
        line=dict(color=sc['color'], dash=sc['dash'], width=2),
        marker=dict(size=6),
    ))

fig7.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
               annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig7, width=FULL_W, height=450, legend_pos='tr')
fig7.update_layout(
    xaxis_title='Number of insertions (i)',
    yaxis_title='Beta (power-law exponent)',
)
save_fig(fig7, '7. Equalized beta vs i.png')
fig7# .show()  # skip in script mode

print(f"Cell 7 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 8  (id: cell-8-equalized-V)
# ============================================================
print("\n" + "="*60)
print("Running cell 8 (cell-8-equalized-V)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 8: N-Equalized Beta by V (order volume)
# ══════════════════════════════════════════════════════════════════

model_names = [m for m in SCENARIOS if m in CLOUDS]
V_vals = sorted(CLOUDS[model_names[0]]['V'].unique())

print('N-Equalized beta by V (order volume):\n')

# Raw beta by V
raw_beta_V = {}
for model_name in model_names:
    raw_beta_V[model_name] = {}
    for V in V_vals:
        sub = CLOUDS[model_name][CLOUDS[model_name]['V'] == V]
        g = compute_global_beta(sub)
        raw_beta_V[model_name][V] = bootstrap_beta_from_subset(sub)

# Print
print(f'{"Model":<12}', end='')
for V in V_vals:
    print(f'  V={V:<12}', end='')
print()
print('-' * (12 + 15 * len(V_vals)))
for model_name in model_names:
    print(f'{model_name:<12}', end='')
    for V in V_vals:
        d = raw_beta_V[model_name][V]
        print(f'  {d["beta"]:.3f} [{d["ci_lo"]:.2f},{d["ci_hi"]:.2f}]', end='')
    print()

# Equalized by V-strata
eq_global_V = {}
for model_name in model_names:
    pc = CLOUDS[model_name]
    betas = repeated_subsample_beta(pc, 'V', N_REPEATS)
    if len(betas) > 0:
        eq_global_V[model_name] = {
            'mean': np.mean(betas), 'lo': np.percentile(betas, 2.5),
            'hi': np.percentile(betas, 97.5),
        }
    else:
        eq_global_V[model_name] = {'mean': np.nan, 'lo': np.nan, 'hi': np.nan}

print(f'\nEqualized global beta (by V-strata):')
for model_name in model_names:
    eq = eq_global_V[model_name]
    print(f'  {model_name:<12}: {eq["mean"]:.3f} [{eq["lo"]:.3f}, {eq["hi"]:.3f}]')

# ── Plot 8: Beta by V ──
fig8 = go.Figure()
for model_name in model_names:
    sc = SCENARIOS[model_name]
    betas = [raw_beta_V[model_name][V]['beta'] for V in V_vals]
    ci_lo = [raw_beta_V[model_name][V]['ci_lo'] for V in V_vals]
    ci_hi = [raw_beta_V[model_name][V]['ci_hi'] for V in V_vals]
    fig8.add_trace(go.Scatter(
        x=V_vals + V_vals[::-1],
        y=ci_hi + ci_lo[::-1],
        fill='toself', fillcolor=hex_to_rgba(sc['color']),
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig8.add_trace(go.Scatter(
        x=V_vals, y=betas, name=model_name, mode='lines+markers',
        line=dict(color=sc['color'], dash=sc['dash'], width=2),
        marker=dict(size=7),
    ))

fig8.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
               annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig8, width=FULL_W, height=450, legend_pos='tr')
fig8.update_layout(
    xaxis_title='Order volume (V)',
    yaxis_title='Beta (power-law exponent)',
)
save_fig(fig8, '8. Equalized beta vs V.png')
fig8# .show()  # skip in script mode

print(f"Cell 8 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 9  (id: cell-9-equalized-context)
# ============================================================
print("\n" + "="*60)
print("Running cell 9 (cell-9-equalized-context)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 9: N-Equalized Beta by context%
# ══════════════════════════════════════════════════════════════════

def context_band(pct):
    if pct < 90:
        return '<90%'
    elif pct < 97:
        return '90-97%'
    else:
        return '>97%'

bands = ['<90%', '90-97%', '>97%']
model_names = [m for m in SCENARIOS if m in CLOUDS]

# Add context band to each cloud
for model_name in model_names:
    CLOUDS[model_name]['ctx_band'] = CLOUDS[model_name]['context_pct'].apply(context_band)

# Raw beta by context band
raw_beta_ctx = {}
for model_name in model_names:
    raw_beta_ctx[model_name] = {}
    for band in bands:
        sub = CLOUDS[model_name][CLOUDS[model_name]['ctx_band'] == band]
        raw_beta_ctx[model_name][band] = bootstrap_beta_from_subset(sub)

# Print
print('Beta by context% band:\n')
print(f'{"Model":<12}', end='')
for band in bands:
    print(f'  {band:>18}', end='')
print()
print('-' * (12 + 20 * len(bands)))
for model_name in model_names:
    print(f'{model_name:<12}', end='')
    for band in bands:
        d = raw_beta_ctx[model_name][band]
        if d['n'] > 0:
            print(f'  {d["beta"]:6.3f} [{d["ci_lo"]:.2f},{d["ci_hi"]:.2f}]', end='')
        else:
            print(f'  {"---":>18}', end='')
    print()

# N-equalized by context band
eq_global_ctx = {}
for model_name in model_names:
    pc = CLOUDS[model_name]
    betas = repeated_subsample_beta(pc, 'ctx_band', N_REPEATS)
    if len(betas) > 0:
        eq_global_ctx[model_name] = {
            'mean': np.mean(betas), 'lo': np.percentile(betas, 2.5),
            'hi': np.percentile(betas, 97.5),
        }
    else:
        eq_global_ctx[model_name] = {'mean': np.nan, 'lo': np.nan, 'hi': np.nan}

print(f'\nEqualized global beta (by context% bands):')
for model_name in model_names:
    eq = eq_global_ctx[model_name]
    print(f'  {model_name:<12}: {eq["mean"]:.3f} [{eq["lo"]:.3f}, {eq["hi"]:.3f}]')

# ── Plot 9: Beta by context% band ──
fig9 = go.Figure()
for model_name in model_names:
    sc = SCENARIOS[model_name]
    betas = [raw_beta_ctx[model_name][b]['beta'] for b in bands]
    ci_lo = [raw_beta_ctx[model_name][b]['ci_lo'] for b in bands]
    ci_hi = [raw_beta_ctx[model_name][b]['ci_hi'] for b in bands]
    errs_minus = [b - lo for b, lo in zip(betas, ci_lo)]
    errs_plus  = [hi - b for b, hi in zip(betas, ci_hi)]
    fig9.add_trace(go.Bar(
        x=bands, y=betas, name=model_name,
        marker_color=sc['color'],
        error_y=dict(type='data', symmetric=False,
                     array=errs_plus, arrayminus=errs_minus),
    ))

fig9.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
               annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig9, width=FULL_W, height=450)
fig9.update_layout(
    barmode='group',
    xaxis_title='Context utilization',
    yaxis_title='Beta (power-law exponent)',
)
save_fig(fig9, '9. Equalized beta vs context pct.png')
fig9# .show()  # skip in script mode

print(f"Cell 9 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 10  (id: cell-10-difficulty)
# ============================================================
print("\n" + "="*60)
print("Running cell 10 (cell-10-difficulty)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 10: Difficulty-Stratified Analysis (key insight cell)
# ══════════════════════════════════════════════════════════════════
#
# Easy   (mb=5,10):    book barely changes → mechanical beta → all models equivalent
# Medium (mb=20,25):   some evolution → models start diverging
# Hard   (mb=50,75,100): significant evolution → only good models produce realistic beta

model_names = [m for m in SCENARIOS if m in CLOUDS]
diff_bands = ['Easy', 'Medium', 'Hard']
diff_colors = {'Easy': '#72B7B2', 'Medium': '#F58518', 'Hard': '#E45756'}

# Add difficulty band to clouds
for model_name in model_names:
    CLOUDS[model_name]['difficulty'] = CLOUDS[model_name]['mb'].apply(difficulty_band)

# Beta per difficulty band per model
beta_diff = {}  # {model: {band: {beta, ci_lo, ci_hi, n}}}
for model_name in model_names:
    beta_diff[model_name] = {}
    for band in diff_bands:
        sub = CLOUDS[model_name][CLOUDS[model_name]['difficulty'] == band]
        beta_diff[model_name][band] = bootstrap_beta_from_subset(sub)

# Print table
print('Beta by difficulty band:\n')
print(f'{"Model":<12}', end='')
for band in diff_bands:
    print(f'  {band:>18}', end='')
print(f'  {"Delta(E-H)":>12}')
print('-' * (12 + 20 * len(diff_bands) + 14))
for model_name in model_names:
    print(f'{model_name:<12}', end='')
    for band in diff_bands:
        d = beta_diff[model_name][band]
        print(f'  {d["beta"]:6.3f} [{d["ci_lo"]:.2f},{d["ci_hi"]:.2f}]', end='')
    delta = beta_diff[model_name]['Easy']['beta'] - beta_diff[model_name]['Hard']['beta']
    print(f'  {delta:>+12.3f}')

# N-equalized across difficulty bands
print(f'\nN-equalized across difficulty bands ({N_REPEATS} reps):')
eq_diff = {}
for model_name in model_names:
    pc = CLOUDS[model_name]
    betas = repeated_subsample_beta(pc, 'difficulty', N_REPEATS)
    if len(betas) > 0:
        eq_diff[model_name] = {
            'mean': np.mean(betas), 'lo': np.percentile(betas, 2.5),
            'hi': np.percentile(betas, 97.5),
        }
    else:
        eq_diff[model_name] = {'mean': np.nan, 'lo': np.nan, 'hi': np.nan}
    print(f'  {model_name:<12}: {eq_diff[model_name]["mean"]:.3f} '
          f'[{eq_diff[model_name]["lo"]:.3f}, {eq_diff[model_name]["hi"]:.3f}]')

# ── Plot 10a: Beta by difficulty ──
fig10a = go.Figure()
for band in diff_bands:
    betas = [beta_diff[m][band]['beta'] for m in model_names]
    ci_lo = [beta_diff[m][band]['ci_lo'] for m in model_names]
    ci_hi = [beta_diff[m][band]['ci_hi'] for m in model_names]
    errs_minus = [b - lo for b, lo in zip(betas, ci_lo)]
    errs_plus  = [hi - b for b, hi in zip(betas, ci_hi)]
    fig10a.add_trace(go.Bar(
        x=model_names, y=betas, name=band,
        marker_color=diff_colors[band],
        error_y=dict(type='data', symmetric=False,
                     array=errs_plus, arrayminus=errs_minus),
    ))

fig10a.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
                annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig10a, width=FULL_W, height=450)
fig10a.update_layout(
    barmode='group',
    xaxis_title='Model',
    yaxis_title='Beta (power-law exponent)',
)
save_fig(fig10a, '10a. Beta by difficulty.png')
fig10a# .show()  # skip in script mode

# ── Plot 10b: Delta chart (Easy - Hard) ──
deltas = [beta_diff[m]['Easy']['beta'] - beta_diff[m]['Hard']['beta'] for m in model_names]
colors_delta = [SCENARIOS[m]['color'] for m in model_names]

fig10b = go.Figure(go.Bar(
    x=model_names, y=deltas,
    marker_color=colors_delta,
    text=[f'{d:+.3f}' for d in deltas],
    textposition='outside',
))
fig10b.add_hline(y=0, line_dash='solid', line_color='black', line_width=1)

pub_layout(fig10b, width=SINGLE_W, height=400, legend_pos='none')
fig10b.update_layout(
    xaxis_title='Model',
    yaxis_title='Beta(Easy) - Beta(Hard)',
)
save_fig(fig10b, '10b. Difficulty delta.png')
fig10b# .show()  # skip in script mode

print('\nInterpretation:')
print('- Large positive delta = model\'s global beta is inflated by easy configs')
print('- Small/zero delta = model is consistent across difficulty levels')
print('- Negative delta = model performs better on hard configs (unlikely but interesting)')

print(f"Cell 10 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 11  (id: cell-11-cross-strat)
# ============================================================
print("\n" + "="*60)
print("Running cell 11 (cell-11-cross-strat)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 11: Cross-Stratification Heatmaps
# ══════════════════════════════════════════════════════════════════

model_names = [m for m in SCENARIOS if m in CLOUDS]
mb_vals = sorted(CLOUDS[model_names[0]]['mb'].unique())
V_vals = sorted(CLOUDS[model_names[0]]['V'].unique())
n_models = len(model_names)

# ── 11a: (mb x V) heatmap per model with equalized N per cell ──

# Find min N across all (mb, V) cells (across all models)
min_cell_n = np.inf
for model_name in model_names:
    pc = CLOUDS[model_name]
    for mb in mb_vals:
        for V in V_vals:
            n = len(pc[(pc['mb'] == mb) & (pc['V'] == V)])
            if n > 0:
                min_cell_n = min(min_cell_n, n)
min_cell_n = int(min_cell_n)
print(f'Min N per (mb, V) cell: {min_cell_n}')

# Compute equalized beta per cell
eq_beta_grid = {}  # {model: 2D array}
for model_name in model_names:
    pc = CLOUDS[model_name]
    grid = np.full((len(mb_vals), len(V_vals)), np.nan)
    for mi, mb in enumerate(mb_vals):
        for vi, V in enumerate(V_vals):
            sub = pc[(pc['mb'] == mb) & (pc['V'] == V)]
            if len(sub) < 2:
                continue
            target = min(min_cell_n, len(sub))
            # Repeated subsampling for stability
            betas = []
            rng = np.random.RandomState(42)
            for _ in range(50):
                idx = rng.choice(len(sub), size=target, replace=False) if len(sub) > target else np.arange(len(sub))
                g = compute_global_beta(sub.iloc[idx])
                if np.isfinite(g['beta']):
                    betas.append(g['beta'])
            grid[mi, vi] = np.mean(betas) if betas else np.nan
    eq_beta_grid[model_name] = grid

# Shared colorscale
all_betas = np.concatenate([g.ravel() for g in eq_beta_grid.values()])
all_betas = all_betas[np.isfinite(all_betas)]
zmin, zmax = np.percentile(all_betas, [2, 98]) if len(all_betas) > 0 else (0, 1)

ncols = min(3, n_models)
nrows = math.ceil(n_models / ncols)
fig11a = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=model_names,
    horizontal_spacing=0.08, vertical_spacing=0.12,
)
for idx, model_name in enumerate(model_names):
    r, c = divmod(idx, ncols)
    grid = eq_beta_grid[model_name]
    text = np.where(np.isfinite(grid),
                    np.vectorize(lambda x: f'{x:.2f}')(grid), '')
    fig11a.add_trace(go.Heatmap(
        z=grid, x=[str(v) for v in V_vals], y=[str(mb) for mb in mb_vals],
        text=text, texttemplate='%{text}',
        colorscale='RdBu_r', zmid=0.5, zmin=zmin, zmax=zmax,
        showscale=(idx == 0),
        colorbar=dict(title='Beta', len=0.4, y=0.5) if idx == 0 else None,
    ), row=r+1, col=c+1)
    fig11a.update_xaxes(title_text='V' if r == nrows-1 else '', row=r+1, col=c+1)
    fig11a.update_yaxes(title_text='mb' if c == 0 else '', row=r+1, col=c+1)

pub_layout(fig11a, width=FULL_W, height=300*nrows, legend_pos='none')
fig11a.update_layout(margin=dict(t=40))
save_fig(fig11a, '11a. Equalized heatmap.png')
fig11a# .show()  # skip in script mode

# ── 11b: Difference heatmap: beta(model) - beta(LobS5) ──
if 'LobS5' in eq_beta_grid:
    ref_grid = eq_beta_grid['LobS5']
    other_models = [m for m in model_names if m != 'LobS5']
    n_other = len(other_models)
    ncols_d = min(3, n_other)
    nrows_d = math.ceil(n_other / ncols_d)

    fig11b = make_subplots(
        rows=nrows_d, cols=ncols_d,
        subplot_titles=[f'{m} - LobS5' for m in other_models],
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    diff_all = []
    for model_name in other_models:
        diff = eq_beta_grid[model_name] - ref_grid
        diff_all.append(diff[np.isfinite(diff)])

    if diff_all:
        diff_concat = np.concatenate(diff_all)
        dmax = np.percentile(np.abs(diff_concat), 98) if len(diff_concat) > 0 else 0.5
    else:
        dmax = 0.5

    for idx, model_name in enumerate(other_models):
        r, c = divmod(idx, ncols_d)
        diff = eq_beta_grid[model_name] - ref_grid
        text = np.where(np.isfinite(diff),
                        np.vectorize(lambda x: f'{x:+.2f}')(diff), '')
        fig11b.add_trace(go.Heatmap(
            z=diff, x=[str(v) for v in V_vals], y=[str(mb) for mb in mb_vals],
            text=text, texttemplate='%{text}',
            colorscale='RdBu_r', zmid=0, zmin=-dmax, zmax=dmax,
            showscale=(idx == 0),
            colorbar=dict(title='Delta', len=0.4, y=0.5) if idx == 0 else None,
        ), row=r+1, col=c+1)
        fig11b.update_xaxes(title_text='V' if r == nrows_d-1 else '', row=r+1, col=c+1)
        fig11b.update_yaxes(title_text='mb' if c == 0 else '', row=r+1, col=c+1)

    pub_layout(fig11b, width=FULL_W, height=300*nrows_d, legend_pos='none')
    fig11b.update_layout(margin=dict(t=40))
    save_fig(fig11b, '11b. Difference from LobS5.png')
    fig11b# .show()  # skip in script mode
else:
    print('LobS5 not available — skipping difference heatmap')

# ── 11c: (mb x direction) analysis ──
directions = ['BUY', 'SELL']
print('\nBeta by (mb x direction):\n')
for model_name in model_names:
    print(f'\n{model_name}:')
    pc = CLOUDS[model_name]
    print(f'{"mb":>5}', end='')
    for d in directions:
        print(f'  {d:>8}', end='')
    print(f'  {"Diff":>8}')
    print('-' * 35)
    for mb in mb_vals:
        print(f'{mb:5d}', end='')
        b_vals = {}
        for d in directions:
            sub = pc[(pc['mb'] == mb) & (pc['direction'] == d)]
            g = compute_global_beta(sub)
            b_vals[d] = g['beta']
            print(f'  {g["beta"]:8.3f}', end='')
        diff = b_vals.get('BUY', np.nan) - b_vals.get('SELL', np.nan)
        print(f'  {diff:+8.3f}')

# Direction heatmap for each model
ncols_c = min(3, n_models)
nrows_c = math.ceil(n_models / ncols_c)
fig11c = make_subplots(
    rows=nrows_c, cols=ncols_c,
    subplot_titles=model_names,
    horizontal_spacing=0.08, vertical_spacing=0.12,
)

dir_grids = {}
for model_name in model_names:
    pc = CLOUDS[model_name]
    grid = np.full((len(mb_vals), 2), np.nan)
    for mi, mb in enumerate(mb_vals):
        for di, d in enumerate(directions):
            sub = pc[(pc['mb'] == mb) & (pc['direction'] == d)]
            g = compute_global_beta(sub)
            grid[mi, di] = g['beta']
    dir_grids[model_name] = grid

all_dir = np.concatenate([g.ravel() for g in dir_grids.values()])
all_dir = all_dir[np.isfinite(all_dir)]
dzmin, dzmax = np.percentile(all_dir, [2, 98]) if len(all_dir) > 0 else (0, 1)

for idx, model_name in enumerate(model_names):
    r, c = divmod(idx, ncols_c)
    grid = dir_grids[model_name]
    text = np.where(np.isfinite(grid),
                    np.vectorize(lambda x: f'{x:.2f}')(grid), '')
    fig11c.add_trace(go.Heatmap(
        z=grid, x=directions, y=[str(mb) for mb in mb_vals],
        text=text, texttemplate='%{text}',
        colorscale='RdBu_r', zmid=0.5, zmin=dzmin, zmax=dzmax,
        showscale=(idx == 0),
        colorbar=dict(title='Beta', len=0.4, y=0.5) if idx == 0 else None,
    ), row=r+1, col=c+1)
    fig11c.update_xaxes(title_text='Direction' if r == nrows_c-1 else '', row=r+1, col=c+1)
    fig11c.update_yaxes(title_text='mb' if c == 0 else '', row=r+1, col=c+1)

pub_layout(fig11c, width=FULL_W, height=300*nrows_c, legend_pos='none')
fig11c.update_layout(margin=dict(t=40))
save_fig(fig11c, '11c. Direction x mb.png')
fig11c# .show()  # skip in script mode

print(f"Cell 11 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 12  (id: cell-12-weighted)
# ============================================================
print("\n" + "="*60)
print("Running cell 12 (cell-12-weighted)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 12: Weighted vs Unweighted Beta
# ══════════════════════════════════════════════════════════════════

model_names = [m for m in SCENARIOS if m in CLOUDS]

results_w = []  # [{model, unweighted, folder_weighted, mb_weighted}]

for model_name in model_names:
    pc = CLOUDS[model_name].copy()

    # Unweighted (standard OLS)
    g_uw = compute_global_beta(pc)

    # Folder-weighted: w_i = 1 / N_folder_i
    folder_counts = pc.groupby('folder').size().to_dict()
    pc['w_folder'] = pc['folder'].map(lambda f: 1.0 / folder_counts[f])
    g_fw = weighted_beta(pc, 'w_folder')

    # mb-weighted: w_i = 1 / N_mb_i
    mb_counts = pc.groupby('mb').size().to_dict()
    pc['w_mb'] = pc['mb'].map(lambda m: 1.0 / mb_counts[m])
    g_mw = weighted_beta(pc, 'w_mb')

    results_w.append({
        'model': model_name,
        'unweighted': g_uw['beta'],
        'folder_weighted': g_fw['beta'],
        'mb_weighted': g_mw['beta'],
    })

rw_df = pd.DataFrame(results_w)

# Print table
print('Weighted vs Unweighted Beta:\n')
print(f'{"Model":<12} {"Unweighted":>12} {"Folder-wt":>12} {"mb-wt":>12} '
      f'{"Bias(UW-FW)":>12} {"Bias(UW-MW)":>12}')
print('-' * 75)
for _, r in rw_df.iterrows():
    bias_fw = abs(r['unweighted'] - r['folder_weighted'])
    bias_mw = abs(r['unweighted'] - r['mb_weighted'])
    print(f'{r["model"]:<12} {r["unweighted"]:>12.3f} {r["folder_weighted"]:>12.3f} '
          f'{r["mb_weighted"]:>12.3f} {bias_fw:>12.3f} {bias_mw:>12.3f}')

# ── Plot 12: Grouped bars — 3 bars per model ──
fig12 = go.Figure()
bar_types = ['unweighted', 'folder_weighted', 'mb_weighted']
bar_labels = ['Unweighted', 'Folder-weighted', 'mb-weighted']
bar_colors = ['#4C78A8', '#F58518', '#E45756']

for bt, bl, bc in zip(bar_types, bar_labels, bar_colors):
    vals = rw_df[bt].values
    fig12.add_trace(go.Bar(
        x=rw_df['model'], y=vals, name=bl,
        marker_color=bc,
    ))

fig12.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1,
               annotation_text='theory (0.5)', annotation_position='bottom right')

pub_layout(fig12, width=FULL_W, height=450)
fig12.update_layout(
    barmode='group',
    xaxis_title='Model',
    yaxis_title='Beta (power-law exponent)',
)
save_fig(fig12, '12. Weighted vs unweighted.png')
fig12# .show()  # skip in script mode

print(f"Cell 12 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

# ============================================================
# Cell 13  (id: cell-13-summary)
# ============================================================
print("\n" + "="*60)
print("Running cell 13 (cell-13-summary)...")
print("="*60)
_t0 = time.time()

# ══════════════════════════════════════════════════════════════════
# Cell 13: Summary + Recommendations
# ══════════════════════════════════════════════════════════════════

model_names = [m for m in SCENARIOS if m in CLOUDS]

# ── Grand summary table ──
summary_rows = []
for model_name in model_names:
    # Global beta
    g = compute_global_beta(CLOUDS[model_name])

    # Per-config median beta
    pcb = per_config_betas[model_name]
    pc_median = pcb['beta'].median() if len(pcb) > 0 else np.nan

    # First-impact beta (k=1)
    k1_beta = beta_by_k[model_name][1]['beta']

    # Weighted beta (folder-weighted)
    fw_row = rw_df[rw_df['model'] == model_name]
    fw_beta = fw_row['folder_weighted'].values[0] if len(fw_row) > 0 else np.nan

    # Easy / Hard beta
    easy_beta = beta_diff[model_name]['Easy']['beta']
    hard_beta = beta_diff[model_name]['Hard']['beta']
    delta = easy_beta - hard_beta

    summary_rows.append({
        'Model': model_name,
        'Global': g['beta'],
        'Per-config median': pc_median,
        'First-impact (k=1)': k1_beta,
        'Folder-weighted': fw_beta,
        'Easy': easy_beta,
        'Hard': hard_beta,
        'Delta(E-H)': delta,
    })

summary_df = pd.DataFrame(summary_rows)

print('='*80)
print('GRAND SUMMARY TABLE')
print('='*80)
print(summary_df.to_string(index=False, float_format='{:.3f}'.format))

# ── Ranking table: rank by proximity to beta=0.5 ──
print('\n\n' + '='*80)
print('RANKING: Proximity to theoretical beta=0.5')
print('='*80)

metrics = ['Global', 'Per-config median', 'First-impact (k=1)',
           'Folder-weighted', 'Easy', 'Hard']
rank_df = pd.DataFrame({'Model': model_names})
for metric in metrics:
    vals = summary_df[metric].values
    dists = np.abs(vals - 0.5)
    ranks = pd.Series(dists).rank().astype(int).values
    rank_df[metric] = ranks

# Average rank
rank_df['Avg rank'] = rank_df[metrics].mean(axis=1)
rank_df = rank_df.sort_values('Avg rank')
print(rank_df.to_string(index=False))

# ── LaTeX summary table ──
print('\n\n=== LaTeX Table ===')
cols = summary_df.columns.tolist()
print('\\begin{tabular}{l' + 'c' * (len(cols) - 1) + '}')
print('\\toprule')
print(' & '.join(cols) + ' \\\\')
print('\\midrule')
for _, row in summary_df.iterrows():
    vals = [row['Model']] + [f'{row[c]:.3f}' for c in cols[1:]]
    print(' & '.join(vals) + ' \\\\')
print('\\bottomrule')
print('\\end{tabular}')

# ── Plot 13: Summary comparison ──
# Heatmap of summary table (metrics x models)
plot_metrics = ['Global', 'Per-config median', 'First-impact (k=1)',
                'Folder-weighted', 'Easy', 'Hard']
z_data = summary_df[plot_metrics].values.T  # (n_metrics x n_models)
text_data = np.vectorize(lambda x: f'{x:.3f}')(z_data)

fig13 = go.Figure(go.Heatmap(
    z=z_data,
    x=summary_df['Model'].values,
    y=plot_metrics,
    text=text_data, texttemplate='%{text}',
    colorscale='RdBu_r', zmid=0.5,
    colorbar=dict(title='Beta'),
))

pub_layout(fig13, width=FULL_W, height=450, legend_pos='none')
fig13.update_layout(
    xaxis_title='Model',
    yaxis_title='Methodology',
)
save_fig(fig13, '13. Summary comparison.png')
fig13# .show()  # skip in script mode

# ── Recommendations ──
print('\n\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)
print('''
1. METHODOLOGY: Per-config beta (Cell 4) is the fairest single metric.
   Each of the 21 configs contributes exactly 1 beta value.
   KS tests reveal whether distributions truly differ between models.

2. FIRST-IMPACT (k=1): The purest comparison — exactly 1 point per sample
   per config, zero imbalance. Report this alongside global beta.

3. DIFFICULTY BANDS: Report Easy/Medium/Hard separately.
   Hard configs (mb>=50) are where model quality matters most.
   Delta(Easy-Hard) quantifies how much global beta is inflated.

4. FOR THE PAPER: Use the summary table (Global + Per-config median +
   Hard beta) to demonstrate that fair evaluation reveals model differences
   that are hidden by naive pooled regression.

5. WEIGHTED REGRESSION: Folder-weighted or mb-weighted beta as a
   robustness check. If weighted beta differs substantially from
   unweighted, the results are sensitive to the grid design.
''')

print(f"Cell 13 done in {time.time()-_t0:.1f}s")
sys.stdout.flush()

print("\nFinished:", time.strftime("%Y-%m-%d %H:%M:%S"))