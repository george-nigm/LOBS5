#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
172. Follow-Up Beta Experiments

Standalone script converted from 172.beta_followup_experiments.ipynb.
Implements experiments A and D (cells 0-3, 6) from the notebook.

Usage:
    python -u lob_impact/172_run.py
"""

# ══════════════════════════════════════════════════════════════════
# Cell 0: Setup & imports (reused from 170)
# ══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import re, gc, math, json
from pathlib import Path
from collections import OrderedDict
from scipy.stats import linregress, ks_2samp, mannwhitneyu, skew, kurtosis
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
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
N_COND = 500
N_REPEATS = 100

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

_FIG = [Path('/app/pics_for_172_beta_followup'),
        Path('/scratch/local/homes/80/georgenigm/LOBS5/pics_for_172_beta_followup')]
FIG_DIR = next((p for p in _FIG if p.parent.exists()), _FIG[-1])
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──
def hex_to_rgba(hex_color, alpha=0.12):
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
        html_name = name.rsplit('.', 1)[0] + '.html'
        fig.write_html(str(FIG_DIR / html_name))
        print(f'  Saved (HTML fallback): {html_name}  (kaleido error: {e})')

# ── Data I/O helpers (from 170) ──
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

# ── Point cloud & beta functions (from 170) ──
def extract_point_cloud_extended(data, grid_df):
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

def compute_per_config_beta(df, group_col='folder'):
    results = []
    for key, grp in df.groupby(group_col):
        g = compute_global_beta(grp)
        if np.isfinite(g['beta']) and g['n'] >= 2:
            results.append({'group': key, 'beta': g['beta'], 'r2': g['r2'], 'n': g['n']})
    return pd.DataFrame(results)

def subsample_to_n(df, group_col, target_n, seed=42):
    rng = np.random.RandomState(seed)
    parts = []
    for _, grp in df.groupby(group_col):
        if len(grp) <= target_n:
            parts.append(grp)
        else:
            idx = rng.choice(len(grp), size=target_n, replace=False)
            parts.append(grp.iloc[idx])
    return pd.concat(parts, ignore_index=True)

def first_k_insertions(df, k):
    return df[df['insertion_idx'] < k].copy()

def difficulty_band(mb):
    if mb <= 10:
        return 'Easy'
    elif mb <= 25:
        return 'Medium'
    else:
        return 'Hard'

def weighted_beta(df, weight_col):
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

def repeated_subsample_beta(df, group_col, n_repeats=100, seed=42):
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
    g = compute_global_beta(df)
    boots = bootstrap_beta(df, n_boot)
    if len(boots) > 0:
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    else:
        ci_lo, ci_hi = np.nan, np.nan
    return {'beta': g['beta'], 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'r2': g['r2'], 'n': g['n']}

# ── NEW functions for this notebook ──

def compute_global_beta_with_residuals(df):
    """Fit global beta and return residuals array."""
    if df.empty:
        return {'beta': np.nan, 'r2': np.nan, 'n': 0, 'residuals': np.array([])}
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    xv, yv = x[ok], y_adj[ok]
    if len(xv) < 2:
        return {'beta': np.nan, 'r2': np.nan, 'n': 0, 'residuals': np.array([])}
    beta = float(np.dot(xv, yv) / np.dot(xv, xv))
    residuals = yv - beta * xv
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(yv ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'beta': beta, 'r2': r2, 'n': int(ok.sum()), 'residuals': residuals,
            'x': xv, 'y_adj': yv}

def compute_local_beta_by_quantile(df, n_quantiles=5):
    """Split data by x-quantile and fit local beta in each."""
    if df.empty:
        return pd.DataFrame()
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    xv, yv = x[ok], y_adj[ok]
    if len(xv) < n_quantiles * 5:
        return pd.DataFrame()
    quantiles = np.percentile(xv, np.linspace(0, 100, n_quantiles + 1))
    rows = []
    for q in range(n_quantiles):
        lo, hi = quantiles[q], quantiles[q + 1]
        if q == n_quantiles - 1:
            mask = (xv >= lo) & (xv <= hi)
        else:
            mask = (xv >= lo) & (xv < hi)
        xi, yi = xv[mask], yv[mask]
        if len(xi) < 2:
            continue
        beta_q = float(np.dot(xi, yi) / np.dot(xi, xi))
        midpoint = (lo + hi) / 2
        rows.append({'quantile': q, 'x_lo': lo, 'x_hi': hi, 'x_mid': midpoint,
                     'beta': beta_q, 'n': len(xi)})
    return pd.DataFrame(rows)

def compute_point_beta_ratio(df):
    """Compute per-point ratio y_adj / x (instantaneous local beta)."""
    if df.empty:
        return np.array([])
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    return y_adj[ok] / x[ok]


# ══════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print(f'GRID      : {GRID}')
    print(f'ENABLED   : {list(SCENARIOS.keys())}  ({len(SCENARIOS)}/{len(_ALL_SCENARIOS)})')
    print(f'EVAL_BASE : {EVAL_BASE}')
    print(f'SDM       : {len(SAMPLE_DAY_MAP)} rows')
    print(f'FIG_DIR   : {FIG_DIR}')
    print(f'MAX_SAMPLES: {MAX_SAMPLES}')
    print('Setup complete.')

    # ══════════════════════════════════════════════════════════════
    # Cell 1: Data Loading (reuses 170 logic for v3 grid)
    # ══════════════════════════════════════════════════════════════

    R = OrderedDict()  # results dict

    for label, cfg in SCENARIOS.items():
        # ── Discover folders across selected grid subdirectories ──
        grid_frames = []
        for gdir in _GRID_DIRS:
            buy_p  = EVAL_BASE / cfg['key'] / gdir / 'context_500_buy'
            sell_p = EVAL_BASE / cfg['key'] / gdir / 'context_500_sell'
            if buy_p.exists() and sell_p.exists():
                gf = discover_v2_folders(buy_p, sell_p)
                if not gf.empty:
                    gf['grid_version'] = gdir
                    grid_frames.append(gf)
        if not grid_frames:
            print(f"SKIP {label}: no folders found in {_GRID_DIRS}")
            continue
        grid = pd.concat(grid_frames, ignore_index=True)
        print(f"\n{'='*60}\n  {label}: {len(grid)} configs (grids: {grid['grid_version'].unique().tolist()})")
        data = load_all_v2(grid)
        print(f"  Loaded {len(data)} folders")

        R[label] = {'grid': grid, 'data': data}
        del data
        gc.collect()

    print(f"\n{'='*60}\nLoaded {len(R)} scenarios: {list(R.keys())}")

    # ══════════════════════════════════════════════════════════════
    # Cell 2: Extract Point Clouds for all models
    # ══════════════════════════════════════════════════════════════

    for label in list(R.keys()):
        grid = R[label]['grid']
        data = R[label]['data']
        pc = extract_point_cloud_extended(data, grid)
        R[label]['pc'] = pc
        bstat = compute_global_beta(pc)
        R[label]['beta'] = bstat
        print(f"{label:12s}  N={len(pc):>10,}  beta={bstat['beta']:.4f}  R2={bstat['r2']:.4f}")

    print("\nPoint clouds extracted.")

    # ══════════════════════════════════════════════════════════════
    # Cell 3: Experiment A -- Beta-by-mb curves
    # Shows full beta degradation profile across all mb values
    # ══════════════════════════════════════════════════════════════

    # Compute beta per (model, mb) group
    beta_by_mb = {}
    for label in R:
        pc = R[label]['pc']
        if pc.empty:
            continue
        rows = []
        for mb_val, grp in pc.groupby('mb'):
            g = compute_global_beta(grp)
            boots = bootstrap_beta(grp, n_boot=500)
            ci_lo, ci_hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 0
                            else (np.nan, np.nan))
            rows.append({'mb': mb_val, 'beta': g['beta'], 'r2': g['r2'], 'n': g['n'],
                         'ci_lo': ci_lo, 'ci_hi': ci_hi})
        beta_by_mb[label] = pd.DataFrame(rows).sort_values('mb')

    # Print table
    print("-- Beta by mb --")
    header = f"{'mb':>5s}"
    for label in beta_by_mb:
        header += f"  {label:>12s}"
    print(header)
    print("-" * len(header))
    all_mbs = sorted(set(mb for df in beta_by_mb.values() for mb in df['mb']))
    for mb in all_mbs:
        line = f"{mb:5d}"
        for label, df in beta_by_mb.items():
            row = df[df['mb'] == mb]
            if not row.empty:
                line += f"  {row.iloc[0]['beta']:12.4f}"
            else:
                line += f"  {'---':>12s}"
        print(line)

    # ── Figure: Beta-by-mb curves ──
    fig = go.Figure()
    for label in beta_by_mb:
        df = beta_by_mb[label]
        color = SCENARIOS[label]['color']
        dash = SCENARIOS[label]['dash']
        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([df['mb'], df['mb'][::-1]]),
            y=pd.concat([df['ci_hi'], df['ci_lo'][::-1]]),
            fill='toself', fillcolor=hex_to_rgba(color, 0.12),
            line=dict(width=0), showlegend=False))
        # Main line
        fig.add_trace(go.Scatter(
            x=df['mb'], y=df['beta'], mode='lines+markers',
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=6, color=color),
            name=label))

    fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1.5,
                  annotation_text='beta = 0.5', annotation_font_size=11,
                  annotation_position='bottom right')

    pub_layout(fig, width=SINGLE_W, height=420, legend_pos='tr')
    fig.update_xaxes(title_text='Messages between insertions (m_b)', type='log')
    fig.update_yaxes(title_text='beta')
    save_fig(fig, 'ExpA_beta_by_mb.png')

    print("\n-- Key observation --")
    for label, df in beta_by_mb.items():
        lo = df['beta'].min()
        hi = df['beta'].max()
        delta = hi - lo
        print(f"  {label:12s}  range [{lo:.3f}, {hi:.3f}]  delta={delta:.3f}")

    # ══════════════════════════════════════════════════════════════
    # Cell 6: Experiment D -- CGAN k=1 Deep Dive
    # Tests whether CGAN's k=1 beta advantage persists in hard cases
    # ══════════════════════════════════════════════════════════════

    # Define difficulty bands
    def get_difficulty(mb):
        if mb <= 10:
            return 'Easy'
        elif mb <= 25:
            return 'Medium'
        else:
            return 'Hard'

    # k=1 analysis: restrict to first insertion only
    k_values = [1, 2, 3]
    difficulty_bands = ['Easy', 'Medium', 'Hard', 'All']

    print("-- Experiment D: CGAN k=1 Deep Dive --\n")

    # Build table: for each model, compute k=1 beta in each difficulty band
    rows = []
    for label in R:
        pc = R[label]['pc']
        if pc.empty:
            continue
        pc = pc.copy()
        pc['difficulty'] = pc['mb'].apply(get_difficulty)
        for k in k_values:
            pc_k = first_k_insertions(pc, k)
            for band in difficulty_bands:
                if band == 'All':
                    sub = pc_k
                else:
                    sub = pc_k[pc_k['difficulty'] == band]
                if sub.empty or len(sub) < 10:
                    rows.append({'Model': label, 'k': k, 'Band': band,
                                 'beta': np.nan, 'n': 0})
                    continue
                g = compute_global_beta(sub)
                rows.append({'Model': label, 'k': k, 'Band': band,
                             'beta': g['beta'], 'n': g['n']})

    results_d = pd.DataFrame(rows)

    # Print k=1 table
    print("k=1 Beta by Difficulty Band:")
    print("-" * 65)
    k1 = results_d[results_d['k'] == 1]
    header = f"{'Model':>12s}"
    for band in difficulty_bands:
        header += f"  {band:>10s}"
    print(header)
    for label in R:
        line = f"{label:>12s}"
        for band in difficulty_bands:
            row = k1[(k1['Model'] == label) & (k1['Band'] == band)]
            if not row.empty and np.isfinite(row.iloc[0]['beta']):
                line += f"  {row.iloc[0]['beta']:10.4f}"
            else:
                line += f"  {'---':>10s}"
        print(line)

    # Compute delta: CGAN minus average of other 4 models
    print("\n-- CGAN offset from non-CGAN average (k=1) --")
    for band in difficulty_bands:
        cgan_row = k1[(k1['Model'] == 'CGAN') & (k1['Band'] == band)]
        others = k1[(k1['Model'] != 'CGAN') & (k1['Band'] == band)]
        if cgan_row.empty or others.empty:
            continue
        cgan_beta = cgan_row.iloc[0]['beta']
        others_mean = others['beta'].mean()
        if np.isfinite(cgan_beta) and np.isfinite(others_mean):
            delta = cgan_beta - others_mean
            print(f"  {band:>10s}: CGAN={cgan_beta:.4f}  others_mean={others_mean:.4f}  delta={delta:+.4f}")

    # ── Figure: k=1 beta by difficulty band ──
    fig = go.Figure()
    x_pos = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    for label in R:
        color = SCENARIOS[label]['color']
        betas = []
        xs = []
        for band in ['Easy', 'Medium', 'Hard']:
            row = k1[(k1['Model'] == label) & (k1['Band'] == band)]
            if not row.empty and np.isfinite(row.iloc[0]['beta']):
                betas.append(row.iloc[0]['beta'])
                xs.append(band)
        if betas:
            fig.add_trace(go.Scatter(
                x=xs, y=betas, mode='lines+markers',
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                name=label))

    fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1.5)

    pub_layout(fig, width=SINGLE_W, height=380, legend_pos='tr')
    fig.update_xaxes(title_text='Difficulty Band')
    fig.update_yaxes(title_text='beta (k=1)')
    save_fig(fig, 'ExpD_cgan_k1_by_difficulty.png')

    # ── Additional: k comparison for CGAN ──
    print("\n-- CGAN beta by k --")
    cgan_rows = results_d[results_d['Model'] == 'CGAN']
    for k in k_values:
        line = f"  k={k}:"
        for band in difficulty_bands:
            row = cgan_rows[(cgan_rows['k'] == k) & (cgan_rows['Band'] == band)]
            if not row.empty and np.isfinite(row.iloc[0]['beta']):
                line += f"  {band}={row.iloc[0]['beta']:.4f}"
        print(line)

    print("\n-- Conclusion --")
    cgan_k1_hard = k1[(k1['Model'] == 'CGAN') & (k1['Band'] == 'Hard')]
    others_k1_hard = k1[(k1['Model'] != 'CGAN') & (k1['Band'] == 'Hard')]
    if not cgan_k1_hard.empty and not others_k1_hard.empty:
        c_b = cgan_k1_hard.iloc[0]['beta']
        o_b = others_k1_hard['beta'].mean()
        if np.isfinite(c_b) and np.isfinite(o_b):
            if abs(c_b - o_b) > 0.02:
                print(f"CGAN k=1 advantage PERSISTS in hard cases (delta={c_b - o_b:+.4f})")
                print("-> Genuine architectural property, not easy-case artifact")
            else:
                print(f"CGAN k=1 advantage VANISHES in hard cases (delta={c_b - o_b:+.4f})")
                print("-> Easy-case artifact")

    print("\nDone.")
