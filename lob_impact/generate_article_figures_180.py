#!/usr/bin/env python3
"""
Generate article figures from NB 180 extended impact analysis.

Produces 3 figures for the overleaf article:
  1. perday_beta.png       — Per-day beta boxplot (SINGLE_W)
  2. perm_temp_decomp.png  — Permanent/Temporary decomposition 2-panel (FULL_W)
  3. noarb_scatter.png     — 2D scatter: beta_perm vs relaxation ratio (SINGLE_W)

Run in Docker:
  docker run --rm --gpus '"device=7"' \
    -v "${PWD}:/app" \
    -v "${PWD}/Alphatrade:/AlphaTrade" \
    -v /homes/groups/finance/data:/home/myuser/data \
    -v "${PWD}/output/evalsequences:/home/myuser/data/evalsequences" \
    --shm-size=1g -w /app \
    --user "$(id -u):$(id -g)" --group-add 652 \
    georgenigm_25jan:latest \
    python -u lob_impact/generate_article_figures_180.py

WARNING: Numbers below are from NB 180 with GRID=c10x_v2.
         User must verify these match the actual V2 grid data.
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections import OrderedDict
from scipy.stats import linregress
from tqdm.auto import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────

TICK_SIZE = 100
MAX_SAMPLES = 2048
MIDPRICE_MAX = None
N_BOOTSTRAP = 1000
N_COND_MSGS = 500
GRID = 'c10x_v2'

# 4 models only (no CGAN)
ENABLED = ['Historic', 'Heuristic', 'CST', 'LobS5']

_ALL_SCENARIOS = OrderedDict([
    ('Historic',  {'key': 'historic_scenario',  'color': '#8F939A', 'dash': 'dash'}),
    ('Heuristic', {'key': 'heuristic_scenario', 'color': '#2F5DA3', 'dash': 'dot'}),
    ('CST',       {'key': 'cst_scenario',       'color': '#5B4B8A', 'dash': 'dashdot'}),
    ('LobS5',    {'key': 'aggressive_scenario', 'color': '#D09A3C', 'dash': 'solid'}),
])
SCENARIOS = OrderedDict((k, v) for k, v in _ALL_SCENARIOS.items() if k in ENABLED)

_GRID_DIRS = [GRID]

_BASE = [Path('/app/output/evalsequences'),
         Path('/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences')]
EVAL_BASE = next((p for p in _BASE if p.exists()), _BASE[-1])

_SDM = [Path('/app/lob_impact/sample_day_map.csv'),
        Path('/scratch/local/homes/80/georgenigm/LOBS5/lob_impact/sample_day_map.csv')]
SDM_PATH = next((p for p in _SDM if p.exists()), _SDM[-1])
SAMPLE_DAY_MAP = pd.read_csv(SDM_PATH)

# Output to overleaf Figures
_FIG_DIRS = [
    Path('/app/overleaf/overleaf_project_article/Figures'),
    Path('/scratch/local/homes/80/georgenigm/LOBS5/overleaf/overleaf_project_article/Figures'),
]
FIG_DIR = next((p for p in _FIG_DIRS if p.exists()), _FIG_DIRS[-1])

SINGLE_W = 520
FULL_W = 1080
IMG_SCALE = 3

print(f'EVAL_BASE : {EVAL_BASE}')
print(f'FIG_DIR   : {FIG_DIR}')
print(f'GRID      : {GRID}')
print(f'MODELS    : {list(SCENARIOS.keys())}')

# ── Publication style ─────────────────────────────────────────────

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
        'tl': dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        'bl': dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom'),
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
    path = FIG_DIR / name
    fig.write_image(str(path), scale=IMG_SCALE)
    print(f'  Saved: {path}')

# ── Data I/O helpers (from NB 180) ───────────────────────────────

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
        if max_midprice:
            mp = compute_midprice(full_book)
            if np.any(mp > max_midprice) or np.any(mp <= 0):
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
            bb, bm, bc = load_folder_data(row['buy_path'], MAX_SAMPLES, MIDPRICE_MAX)
            sb, sm, sc = load_folder_data(row['sell_path'], MAX_SAMPLES, MIDPRICE_MAX)
            all_data[row['folder']] = {
                'buy':  {'books': bb, 'msgs': bm, 'cond_lens': bc},
                'sell': {'books': sb, 'msgs': sm, 'cond_lens': sc},
            }
        except Exception as e:
            print(f'  ERR {row["folder"]}: {e}')
    return all_data

# ── Beta helpers ──────────────────────────────────────────────────

def extract_point_cloud(data, grid_df):
    eps = 1e-12
    rows = []
    for _, grow in grid_df.iterrows():
        folder = grow['folder']
        if folder not in data:
            continue
        d = data[folder]
        mb_val = grow['mb']
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
                for a in range(len(aggr_idx)):
                    if imp[a] > eps:
                        rows.append({'x': np.log(Q_cum[a] / V_day),
                                     'y': np.log(imp[a]),
                                     'alpha': alpha,
                                     'sample_id': sample_id,
                                     'folder': folder, 'direction': direction,
                                     'mb': mb_val,
                                     'insertion_idx': a})
    if not rows:
        return pd.DataFrame(columns=['x', 'y', 'alpha', 'sample_id', 'folder',
                                     'direction', 'mb', 'insertion_idx'])
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

# ── Raw curve + relaxation helpers ────────────────────────────────

def compute_raw_curve(buy_data, sell_data, folder, aggr_gen,
                      u_max=11.0, n_pts=500):
    i, c, mb, V = parse_folder_params_v2(folder)
    if len(aggr_gen) < 2:
        return None
    s_gen = int(aggr_gen[0])
    e_gen = int(aggr_gen[-1])
    L = e_gen - s_gen
    if L == 0:
        return None
    bb, sb = buy_data['books'], sell_data['books']
    if not bb or not sb:
        return None
    min_len = min(min(b.shape[0] for b in bb.values()),
                  min(b.shape[0] for b in sb.values()))
    junction = list(buy_data['cond_lens'].values())[0]
    u_cap = min(u_max, (min_len - 1 - junction - s_gen) / L)
    if u_cap <= 0:
        return None
    u_grid = np.linspace(0, u_cap, n_pts)

    def side_impacts(books, conds):
        imps = []
        for sid, bk in books.items():
            j = conds[sid]
            s_abs = j + s_gen
            if s_abs < 1 or s_abs >= min_len:
                continue
            mid = compute_midprice(bk[:min_len])
            ref = mid[s_abs - 1]
            raw = mid[s_abs:min_len] - ref
            u_raw = np.arange(len(raw)) / L
            imps.append(np.interp(u_grid, u_raw, raw))
        return np.array(imps) if imps else None

    bi = side_impacts(bb, buy_data['cond_lens'])
    si = side_impacts(sb, sell_data['cond_lens'])
    if bi is None or si is None:
        return None
    mean = (np.mean(bi, axis=0) - np.mean(si, axis=0)) / 2
    std  = np.sqrt(np.std(bi, axis=0)**2 + np.std(si, axis=0)**2) / 2
    return {'u_grid': u_grid, 'combined_mean': mean, 'combined_std': std,
            'L': L, 'i': i, 'c': c, 'mb': mb, 'V': V, 'Q': i * V}

# ── Sigma-normalized master curves ────────────────────────────────

def compute_master_curve(buy_data, sell_data, folder, aggr_gen,
                         u_max=11.0, n_pts=500):
    i, c, mb, V = parse_folder_params_v2(folder)
    if len(aggr_gen) < 2:
        return None
    s_gen = int(aggr_gen[0])
    e_gen = int(aggr_gen[-1])
    L = e_gen - s_gen
    if L == 0:
        return None
    bb, sb = buy_data['books'], sell_data['books']
    if not bb or not sb:
        return None
    min_len = min(min(b.shape[0] for b in bb.values()),
                  min(b.shape[0] for b in sb.values()))
    junction = list(buy_data['cond_lens'].values())[0]
    u_cap = min(u_max, (min_len - 1 - junction - s_gen) / L)
    if u_cap <= 0:
        return None
    u_grid = np.linspace(0, u_cap, n_pts)

    def side_impacts(books, conds):
        imps = []
        for sid, bk in books.items():
            sample_id = sid[1]
            day = SAMPLE_DAY_MAP[SAMPLE_DAY_MAP['sample_id'] == sample_id]
            if day.empty:
                continue
            H = float(day.iloc[0]['highest_price']) / TICK_SIZE
            Lp = float(day.iloc[0]['lowest_price'])  / TICK_SIZE
            if H <= Lp or Lp <= 0:
                continue
            sigma = np.log(H / Lp) / 0.8325546
            if sigma <= 0:
                continue
            j = conds[sid]
            s_abs = j + s_gen
            if s_abs < 1 or s_abs >= min_len:
                continue
            mid = compute_midprice(bk[:min_len])
            ref = mid[s_abs - 1]
            if ref <= 0:
                continue
            raw = (mid[s_abs:min_len] - ref) / (ref * sigma)
            u_raw = np.arange(len(raw)) / L
            imps.append(np.interp(u_grid, u_raw, raw))
        return np.array(imps) if imps else None

    bi = side_impacts(bb, buy_data['cond_lens'])
    si = side_impacts(sb, sell_data['cond_lens'])
    if bi is None or si is None:
        return None
    mean = (np.mean(bi, axis=0) - np.mean(si, axis=0)) / 2
    std  = np.sqrt(np.std(bi, axis=0)**2 + np.std(si, axis=0)**2) / 2
    return {'u_grid': u_grid, 'combined_mean': mean, 'combined_std': std,
            'L': L, 'i': i, 'c': c, 'mb': mb, 'V': V, 'Q': i * V}


# ======================================================================
# MAIN
# ======================================================================

def main():
    # ── Load data ─────────────────────────────────────────────────
    R = OrderedDict()

    for label, cfg in SCENARIOS.items():
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
            print(f'SKIP {label}: no folders found')
            continue
        grid = pd.concat(grid_frames, ignore_index=True)
        print(f"\n{'='*60}\n  {label}: {len(grid)} configs")
        data = load_all_v2(grid)
        print(f'  Loaded {len(data)} folders')

        # Beta
        pc_all = extract_point_cloud(data, grid)
        pc = pc_all[pc_all['mb'] != 20] if len(pc_all) > 0 else pc_all
        bstat = compute_global_beta(pc)

        # Raw curves + relaxation
        raw_curves = {}
        for _, row in grid.iterrows():
            f = row['folder']
            if f not in data:
                continue
            aggr = load_aggressive_indices(row['buy_path'])
            cv = compute_raw_curve(data[f]['buy'], data[f]['sell'], f, aggr)
            if cv is not None:
                raw_curves[f] = cv

        # Sigma-normalized curves
        curves = {}
        for _, row in grid.iterrows():
            f = row['folder']
            if f not in data:
                continue
            aggr = load_aggressive_indices(row['buy_path'])
            cv = compute_master_curve(data[f]['buy'], data[f]['sell'], f, aggr)
            if cv is not None:
                curves[f] = cv

        relax_rows = []
        for f, cv in raw_curves.items():
            u, m = cv['u_grid'], cv['combined_mean']
            I_peak = float(np.interp(1.0, u, m))
            if abs(I_peak) < 1e-12:
                continue
            I_final = float(m[-1])
            relax_rows.append({'folder': f, 'I_peak': I_peak, 'I_final': I_final,
                               'ratio': I_final / I_peak,
                               'mb': cv['mb'], 'V': cv['V'], 'i': cv['i'],
                               'Q': cv['Q']})
        relax_df = pd.DataFrame(relax_rows) if relax_rows else pd.DataFrame()

        R[label] = {
            'grid': grid, 'data': data, 'pc': pc, 'beta': bstat,
            'curves': curves, 'raw_curves': raw_curves, 'relax_df': relax_df,
        }
        relax_med = relax_df['ratio'].median() if not relax_df.empty else np.nan
        print(f"  beta={bstat['beta']:.4f}  R2={bstat['r2']:.4f}  n={bstat['n']:,}")
        print(f'  Relax median: {relax_med:.3f}')

    print(f"\n{'='*60}\nLoaded {len(R)} scenarios: {list(R.keys())}")

    # ── Compute decomposition (Section 3 from NB 180) ────────────

    decomp_results = OrderedDict()
    for label, r in R.items():
        rdf = r['relax_df'].copy()
        if rdf.empty or len(rdf) < 3:
            continue
        rdf['I_perm'] = rdf['I_final']
        rdf['I_temp'] = rdf['I_peak'] - rdf['I_final']
        pc = r['pc']

        folder_vday = {}
        for f in rdf['folder'].unique():
            pc_f = pc[pc['folder'] == f]
            if not pc_f.empty:
                i_val, c_val, mb_val, V_val = parse_folder_params_v2(f)
                Q_total = i_val * V_val
                last_ins = pc_f[pc_f['insertion_idx'] == pc_f['insertion_idx'].max()]
                if not last_ins.empty:
                    med_x = last_ins['x'].median()
                    folder_vday[f] = Q_total / np.exp(med_x)

        folder_sigma = {}
        for f in rdf['folder'].unique():
            pc_f = pc[pc['folder'] == f]
            if not pc_f.empty:
                folder_sigma[f] = np.exp(pc_f['alpha'].median())

        valid = []
        for _, row in rdf.iterrows():
            f = row['folder']
            if f not in folder_vday or f not in folder_sigma:
                continue
            V_day = folder_vday[f]
            sigma = folder_sigma[f]
            QV = row['Q'] / V_day
            if QV <= 0 or sigma <= 0:
                continue
            valid.append({
                'folder': f, 'Q': row['Q'], 'V': row['V'],
                'QV': QV, 'sigma': sigma,
                'I_perm': row['I_perm'], 'I_temp': row['I_temp'],
                'I_peak': row['I_peak'],
                'ln_QV': np.log(QV),
                'ln_Iperm_s': np.log(abs(row['I_perm']) / sigma + 1e-30),
                'ln_Itemp_s': np.log(abs(row['I_temp']) / sigma + 1e-30),
            })
        vdf = pd.DataFrame(valid)
        if len(vdf) < 3:
            continue

        ok_p = np.isfinite(vdf['ln_QV']) & np.isfinite(vdf['ln_Iperm_s']) & (vdf['I_perm'] > 0)
        if ok_p.sum() >= 3:
            sl_p = linregress(vdf.loc[ok_p, 'ln_QV'], vdf.loc[ok_p, 'ln_Iperm_s'])
            beta_perm = sl_p.slope
            r2_perm = sl_p.rvalue**2
        else:
            beta_perm, r2_perm = np.nan, np.nan

        ok_t = np.isfinite(vdf['ln_QV']) & np.isfinite(vdf['ln_Itemp_s']) & (vdf['I_temp'] > 0)
        if ok_t.sum() >= 3:
            sl_t = linregress(vdf.loc[ok_t, 'ln_QV'], vdf.loc[ok_t, 'ln_Itemp_s'])
            beta_temp = sl_t.slope
            r2_temp = sl_t.rvalue**2
        else:
            beta_temp, r2_temp = np.nan, np.nan

        decomp_results[label] = {
            'beta_perm': beta_perm, 'r2_perm': r2_perm,
            'beta_temp': beta_temp, 'r2_temp': r2_temp,
            'vdf': vdf,
        }

    # ── Compute per-day beta (Section 5 from NB 180) ─────────────

    perday_results = OrderedDict()
    for label, r in R.items():
        pc = r['pc']
        if pc.empty:
            continue
        sdm = SAMPLE_DAY_MAP[['sample_id', 'day']].drop_duplicates()
        pc_day = pc.merge(sdm, on='sample_id', how='left')
        day_betas = []
        for day_val, grp in pc_day.groupby('day'):
            bstat = compute_global_beta(grp)
            if np.isfinite(bstat['beta']):
                day_betas.append({'day': day_val, 'beta': bstat['beta'],
                                  'r2': bstat['r2'], 'n': bstat['n']})
        perday_results[label] = pd.DataFrame(day_betas)

    # ==============================================================
    # FIGURE 1: Per-Day Beta boxplot
    # ==============================================================

    print('\n── Figure 1: Per-Day Beta ──')
    fig = go.Figure()
    for label, ddf in perday_results.items():
        if ddf.empty:
            continue
        fig.add_trace(go.Box(
            y=ddf['beta'], name=label,
            marker_color=SCENARIOS[label]['color'],
            line_color=SCENARIOS[label]['color'],
            boxpoints='all', jitter=0.3, pointpos=-1.5,
            marker=dict(size=5, opacity=0.7),
            line_width=1.5))

    fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1.5,
                  annotation_text='\u03b2 = 0.5', annotation_font_size=11,
                  annotation_position='bottom right')

    pub_layout(fig, width=SINGLE_W, height=380, legend_pos='none')
    fig.update_xaxes(title_text='')
    fig.update_yaxes(title_text='\u03b2 (per day)')
    save_fig(fig, 'perday_beta.png')

    # ==============================================================
    # FIGURE 2: Perm/Temp Decomposition (2-panel)
    # ==============================================================

    print('\n── Figure 2: Perm/Temp Decomposition ──')
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['<b>Permanent: I<sub>final</sub>/\u03c3</b>',
                        '<b>Temporary: (I<sub>peak</sub>-I<sub>final</sub>)/\u03c3</b>'],
        horizontal_spacing=0.14)

    for label, dr in decomp_results.items():
        vdf = dr['vdf']
        color = SCENARIOS[label]['color']

        ok_p = (vdf['I_perm'] > 0)
        if ok_p.sum() > 0:
            fig.add_trace(go.Scatter(
                x=vdf.loc[ok_p, 'ln_QV'], y=vdf.loc[ok_p, 'ln_Iperm_s'],
                mode='markers', marker=dict(size=6, color=color, opacity=0.7),
                name=f"{label} (\u03b2={dr['beta_perm']:.2f})"), row=1, col=1)
            x_r = np.array([vdf['ln_QV'].min(), vdf['ln_QV'].max()])
            sl_p = linregress(vdf.loc[ok_p, 'ln_QV'], vdf.loc[ok_p, 'ln_Iperm_s'])
            fig.add_trace(go.Scatter(
                x=x_r, y=sl_p.slope * x_r + sl_p.intercept, mode='lines',
                line=dict(color=color, width=1.5), showlegend=False), row=1, col=1)

        ok_t = (vdf['I_temp'] > 0)
        if ok_t.sum() > 0:
            fig.add_trace(go.Scatter(
                x=vdf.loc[ok_t, 'ln_QV'], y=vdf.loc[ok_t, 'ln_Itemp_s'],
                mode='markers', marker=dict(size=6, color=color, opacity=0.7),
                showlegend=False), row=1, col=2)
            sl_t = linregress(vdf.loc[ok_t, 'ln_QV'], vdf.loc[ok_t, 'ln_Itemp_s'])
            fig.add_trace(go.Scatter(
                x=x_r, y=sl_t.slope * x_r + sl_t.intercept, mode='lines',
                line=dict(color=color, width=1.5), showlegend=False), row=1, col=2)

    fig.update_layout(
        width=FULL_W, height=420,
        template='plotly_white',
        font=dict(family='Times New Roman, DejaVu Serif, serif', size=12, color='black'),
        margin=dict(l=55, r=15, t=35, b=55),
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.85)', bordercolor='black', borderwidth=1, font_size=10),
    )
    fig.update_xaxes(**_AX, title_text='ln(Q / V)', title_font_size=11)
    fig.update_yaxes(**_AX, title_font_size=11)
    fig.update_yaxes(title_text='ln(I<sub>perm</sub> / \u03c3)', row=1, col=1)
    fig.update_yaxes(title_text='ln(I<sub>temp</sub> / \u03c3)', row=1, col=2)
    save_fig(fig, 'perm_temp_decomp.png')

    # ==============================================================
    # FIGURE 3: 2D No-Arbitrage Scatter (beta_perm vs r)
    # ==============================================================

    print('\n── Figure 3: No-Arbitrage 2D Scatter ──')

    # Collect (beta_perm, r) for each model
    scatter_data = {}
    for label in R:
        dr = decomp_results.get(label, {})
        bp = dr.get('beta_perm', np.nan)
        rdf = R[label]['relax_df']
        r_val = rdf['ratio'].median() if not rdf.empty else np.nan
        scatter_data[label] = (bp, r_val)
        print(f'  {label}: beta_perm={bp:.3f}, r={r_val:.3f}')

    fig = go.Figure()

    # Acceptable zone
    fig.add_shape(type='rect',
        x0=0.7, x1=1.3, y0=0.5, y1=1.0,
        fillcolor='rgba(144,238,144,0.15)', line=dict(color='green', width=1, dash='dot'))

    # Target point
    fig.add_trace(go.Scatter(
        x=[1.0], y=[2/3], mode='markers',
        marker=dict(size=18, symbol='star', color='black', line=dict(width=1.5, color='black')),
        name='Theory (1.0, 2/3)', showlegend=True))

    # Model points
    for label, (bp, r_val) in scatter_data.items():
        if not np.isfinite(bp) or not np.isfinite(r_val):
            continue
        fig.add_trace(go.Scatter(
            x=[bp], y=[r_val], mode='markers+text',
            marker=dict(size=14, color=SCENARIOS[label]['color'],
                        line=dict(width=1.5, color='black')),
            text=[label], textposition='top center',
            textfont=dict(size=11, color=SCENARIOS[label]['color']),
            name=f"{label} ({bp:.2f}, {r_val:.2f})"))

    # Reference lines
    fig.add_hline(y=2/3, line_dash='dash', line_color='grey', line_width=1)
    fig.add_vline(x=1.0, line_dash='dash', line_color='grey', line_width=1)

    pub_layout(fig, width=SINGLE_W, height=SINGLE_W, legend_pos='bl')
    fig.update_xaxes(title_text='\u03b2<sub>perm</sub> (permanent impact exponent)',
                     range=[-0.2, 1.8])
    fig.update_yaxes(title_text='r = I<sub>final</sub> / I<sub>peak</sub>',
                     range=[-0.1, 1.7])
    save_fig(fig, 'noarb_scatter.png')

    # ==============================================================
    # SUMMARY TABLE (for LaTeX)
    # ==============================================================

    print('\n' + '='*80)
    print('SUMMARY TABLE — paste into LaTeX (verify against V2 data!)')
    print('='*80)
    print(f'{"Model":12s}  {"beta":>6s}  {"beta_perm":>9s}  {"r":>6s}  '
          f'{"perday_mean":>11s}  {"perday_std":>10s}  {"perday_min":>10s}  {"perday_max":>10s}')
    print('-'*80)
    for label in R:
        beta = R[label]['beta']['beta']
        dr = decomp_results.get(label, {})
        bp = dr.get('beta_perm', np.nan)
        rdf = R[label]['relax_df']
        r_val = rdf['ratio'].median() if not rdf.empty else np.nan
        pd_df = perday_results.get(label, pd.DataFrame())
        if not pd_df.empty:
            pd_mean = pd_df['beta'].mean()
            pd_std = pd_df['beta'].std()
            pd_min = pd_df['beta'].min()
            pd_max = pd_df['beta'].max()
        else:
            pd_mean = pd_std = pd_min = pd_max = np.nan
        print(f'{label:12s}  {beta:6.3f}  {bp:9.3f}  {r_val:6.3f}  '
              f'{pd_mean:11.3f}  {pd_std:10.3f}  {pd_min:10.3f}  {pd_max:10.3f}')
    print('='*80)


if __name__ == '__main__':
    main()
