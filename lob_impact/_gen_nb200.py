#!/usr/bin/env python3
"""Generate lob_impact/200.v3_checkpoint_comparison.ipynb — comprehensive 8-model analysis."""
import json
from pathlib import Path


def _src(text):
    """Convert text block to ipynb source list."""
    if text.endswith('\n'):
        text = text[:-1]
    lines = text.split('\n')
    if not lines:
        return ['']
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {
        "cell_type": "code", "metadata": {},
        "source": _src(text), "outputs": [], "execution_count": None,
    }


cells = []

# ══════════════════════════════════════════════════════════════════
# Cell 0: Title
# ══════════════════════════════════════════════════════════════════
cells.append(md(r"""# 200 · Comprehensive 8-Model Market Impact Analysis

Eight models on the **c10x_v2** grid (30 configs, context=500):

| Model | Type | Params | Encoding |
|-------|------|--------|----------|
| Historic | Baseline (replay) | — | — |
| Heuristic | Baseline (price-shift) | — | — |
| CST | Parametric (Stoikov-Talreja) | — | — |
| LobS5 (v2) | Neural S5 | 45M | 22tok |
| CGAN | Neural GAN | — | — |
| S5-120M | Neural S5 v3 | 120M | 24tok |
| S5-360M | Neural S5 v3 | 360M | 24tok |
| S5-4K | Neural S5 v3 (4K ctx) | 55M | 24tok |

**Part A** — Core: β, master curves, relaxation, stability, γ (from NB 150)
**Part B** — Extended: participation rate, decay, perm/temp, no-arbitrage, per-day β (from NB 180)
**Part C** — Model-size comparison (new)"""))

# ══════════════════════════════════════════════════════════════════
# Cell 1: Imports
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""import numpy as np
import pandas as pd
import re, gc, math, json
from pathlib import Path
from collections import OrderedDict
from scipy.stats import linregress
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')"""))

# ══════════════════════════════════════════════════════════════════
# Cell 2: Publication style
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""# ── Publication figure style ──────────────────────────────────────
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
    'Apply publication styling to a plotly figure.'
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
    'Save figure at publication DPI.'
    fig.write_image(str(FIG_DIR / name), scale=IMG_SCALE)
    print(f"  Saved: {name}")

print("Publication style loaded.")"""))

# ══════════════════════════════════════════════════════════════════
# Cell 3: Config
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""TICK_SIZE = 100
MAX_SAMPLES = 2048
MIDPRICE_MAX = None
N_BOOTSTRAP = 1000
N_COND_MSGS = 500

GRID = 'c10x_v2'

ENABLED = [
    'Historic',
    'Heuristic',
    'CST',
    'LobS5',
    'CGAN',
    'S5-120M',
    'S5-360M',
    'S5-4K',
]

_ALL_SCENARIOS = OrderedDict([
    ('Historic',  {'key': 'historic_scenario',                        'color': '#8F939A', 'dash': 'dash'}),
    ('Heuristic', {'key': 'heuristic_scenario',                       'color': '#2F5DA3', 'dash': 'dot'}),
    ('CST',       {'key': 'cst_scenario',                             'color': '#5B4B8A', 'dash': 'dashdot'}),
    ('LobS5',    {'key': 'aggressive_scenario',                       'color': '#D09A3C', 'dash': 'solid'}),
    ('CGAN',      {'key': 'cgan_aggressive_scenario',                 'color': '#7B4F9E', 'dash': 'longdash'}),
    ('S5-120M',  {'key': 'aggressive_scenario_v3/j2514440',           'color': '#E04040', 'dash': 'solid',       'no_grid_subdir': True}),
    ('S5-360M',  {'key': 'aggressive_scenario_v3/j2504227',           'color': '#2CA02C', 'dash': 'solid',       'no_grid_subdir': True}),
    ('S5-4K',    {'key': 'aggressive_scenario_v3/j2504167',           'color': '#FF7F0E', 'dash': 'solid',       'no_grid_subdir': True}),
])
SCENARIOS = OrderedDict((k, v) for k, v in _ALL_SCENARIOS.items() if k in ENABLED)
print(f"GRID      : {GRID}")
print(f"ENABLED   : {list(SCENARIOS.keys())}  ({len(SCENARIOS)}/{len(_ALL_SCENARIOS)})")

MODEL_META = {
    'LobS5':   {'params': 45e6,  'encoding': '22tok', 'context': 500},
    'S5-120M': {'params': 120e6, 'encoding': '24tok', 'context': 500},
    'S5-360M': {'params': 360e6, 'encoding': '24tok', 'context': 500},
    'S5-4K':   {'params': 55e6,  'encoding': '24tok', 'context': 4096},
}

_GRID_DIRS = [GRID] if GRID != 'all' else ['c10x_v2', 'v3', 'v4']

_BASE = [Path("/app/output/evalsequences"),
         Path("/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences")]
EVAL_BASE = next((p for p in _BASE if p.exists()), _BASE[-1])

_SDM = [Path("/app/lob_impact/sample_day_map.csv"),
        Path("/scratch/local/homes/80/georgenigm/LOBS5/lob_impact/sample_day_map.csv")]
SDM_PATH = next((p for p in _SDM if p.exists()), _SDM[-1])
SAMPLE_DAY_MAP = pd.read_csv(SDM_PATH)

_FIG = [Path("/app/pics_for_200_v3_ckpt_comparison"),
        Path("/homes/80/georgenigm/LOBS5/pics_for_200_v3_ckpt_comparison")]
FIG_DIR = next((p for p in _FIG if p.exists() or p.parent.exists()), _FIG[0])
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"EVAL_BASE : {EVAL_BASE}")
print(f"SDM       : {len(SAMPLE_DAY_MAP)} rows")
print(f"FIG_DIR   : {FIG_DIR}")"""))

# ══════════════════════════════════════════════════════════════════
# Cell 4: Data I/O helpers
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""# ── Data I/O helpers ──────────────────────────────────────────────

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


def compute_midprice_returns(books, min_len):
    'Returns array (n_samples, min_len) of midprice - midprice[0].'
    returns = []
    for sid, book_array in books.items():
        midprice = compute_midprice(book_array[:min_len])
        returns.append(midprice - midprice[0])
    return np.stack(returns, axis=0)


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
    cond_dir = data_path / "data_cond"
    pat = re.compile(r"^(.+?)_(\d{4}-\d{2}-\d{2})_orderbook_real_id_(\d+)\.csv$")
    samples = []
    for f in cond_dir.glob("*_orderbook_real_id_*.csv"):
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
        cond_bp = data_path / f"data_cond/{ticker}_{date}_orderbook_real_id_{sid}.csv"
        gen_bp  = data_path / f"data_gen/{ticker}_{date}_orderbook_real_id_{sid}_gen_id_0.csv"
        gen_mp  = data_path / f"data_gen/{ticker}_{date}_message_real_id_{sid}_gen_id_0.csv"
        if not gen_bp.exists():
            continue
        cond_book = np.loadtxt(cond_bp, delimiter=',')
        gen_book  = np.loadtxt(gen_bp, delimiter=',')
        full_book = np.vstack([cond_book, gen_book])
        if max_midprice and is_midprice_outlier(full_book, max_midprice):
            continue
        gen_msg  = np.loadtxt(gen_mp, delimiter=',')
        cond_mp  = data_path / f"data_cond/{ticker}_{date}_message_real_id_{sid}.csv"
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
            print(f"  ERR {row['folder']}: {e}")
    return all_data"""))

# ══════════════════════════════════════════════════════════════════
# Cell 5: Beta functions (NB 180 version with insertion_idx)
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""# ── Beta (square-root law) — NB 180 version with insertion_idx ─────

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
    return betas"""))

# ══════════════════════════════════════════════════════════════════
# Cell 6: Master curves, relaxation, gamma
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""# ── Master curves, relaxation, gamma ──────────────────────────────

def compute_master_curve(buy_data, sell_data, folder, aggr_gen,
                         u_max=11.0, n_pts=500):
    'Sigma-normalized volume-time impact curve for one folder.'
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


def compute_raw_curve(buy_data, sell_data, folder, aggr_gen,
                      u_max=11.0, n_pts=500):
    'Raw (absolute) volume-time impact curve.'
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


def compute_combined_impact(buy_data, sell_data, folder):
    'Absolute (buy-sell)/2 midprice returns.'
    bb, sb = buy_data['books'], sell_data['books']
    if not bb or not sb:
        return None
    min_len = min(min(b.shape[0] for b in bb.values()),
                  min(b.shape[0] for b in sb.values()))
    buy_returns = compute_midprice_returns(bb, min_len)
    sell_returns = compute_midprice_returns(sb, min_len)
    combined = (buy_returns.mean(axis=0) - sell_returns.mean(axis=0)) / 2
    junction = list(buy_data['cond_lens'].values())[0]
    post = combined[junction:]
    if len(post) == 0:
        return None
    pk_idx = junction + np.argmax(post)
    return {'mean': combined, 'junction': junction,
            'peak': float(combined[pk_idx]), 'final': float(combined[-1]),
            'peak_idx': pk_idx,
            'n_buy': buy_returns.shape[0], 'n_sell': sell_returns.shape[0],
            'min_len': min_len}"""))

# ══════════════════════════════════════════════════════════════════
# Cell 7: Stability (3-method vote)
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""# ── Stability (3-method vote) ─────────────────────────────────────

TAIL_FRAC   = 0.20
WINDOW_FRAC = 0.15
SLOPE_THRESH = 0.05
TWIN_THRESH  = 0.03
CONV_THRESH  = 95.0


def stability_for_folder(stats, row):
    '3-method stability test for one folder.'
    if stats is None:
        return {'folder': row['folder'], 'stabilized': False, 'votes': 0}
    mean_curve = stats['mean']
    junction   = stats['junction']
    post = mean_curve[junction:]
    pk_local = np.argmax(post)
    peak_val = post[pk_local]
    post_peak = post[pk_local:]
    final_val = post_peak[-1]
    n = len(post_peak)
    if n < 3 or abs(peak_val) < 1e-12:
        return {'folder': row['folder'], 'stabilized': False, 'votes': 0}
    # M1: trailing slope
    tl = max(int(n * TAIL_FRAC), 3)
    tail = post_peak[-tl:]
    sl = linregress(np.arange(tl, dtype=float), tail)
    slope_ok = abs(sl.slope * tl / peak_val) < SLOPE_THRESH
    # M2: two-window
    w = max(int(n * WINDOW_FRAC), 3)
    if 2 * w <= n:
        twin_ok = abs((np.mean(post_peak[-w:]) - np.mean(post_peak[-2*w:-w])) / peak_val) < TWIN_THRESH
    else:
        twin_ok = False
    # M3: exponential fit
    exp_ok = False
    try:
        t = np.arange(n, dtype=float)
        def ef(t, A, tau, C):
            return A * np.exp(-t / tau) + C
        popt, _ = curve_fit(ef, t, post_peak,
                            p0=[float(peak_val - final_val), n / 3.0, float(final_val)],
                            maxfev=10000)
        conv = (1.0 - np.exp(-n / popt[1])) * 100 if popt[1] > 0 else 100.0
        exp_ok = conv > CONV_THRESH
    except Exception:
        pass
    votes = slope_ok + twin_ok + exp_ok
    return {'folder': row['folder'], 'stabilized': votes >= 2, 'votes': int(votes)}"""))

# ══════════════════════════════════════════════════════════════════
# Cell 8: fit_decay (from NB 180)
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""# ── Decay function fitting (from NB 180) ──────────────────────────

def fit_decay(u_grid, mean_curve, u_peak=1.0):
    'Fit power-law and exponential decay to post-peak segment.'
    mask_post = u_grid > u_peak
    if mask_post.sum() < 5:
        return None
    u_post = u_grid[mask_post]
    I_post = mean_curve[mask_post]
    I_final = I_post[-1]
    I_temp = I_post - I_final

    result = {'u_post': u_post, 'I_post': I_post, 'I_final': I_final}

    # Power-law fit: I_temp(u) = A * (u - 1)^(-gamma)
    du = u_post - u_peak
    ok_pl = (du > 0.01) & (I_temp > 1e-12)
    if ok_pl.sum() >= 3:
        try:
            ln_du = np.log(du[ok_pl])
            ln_It = np.log(I_temp[ok_pl])
            sl = linregress(ln_du, ln_It)
            gamma = -sl.slope
            A_pl = np.exp(sl.intercept)
            I_fit_pl = A_pl * du**(-gamma) + I_final
            ss_res_pl = np.sum((I_post[ok_pl] - (A_pl * du[ok_pl]**(-gamma) + I_final))**2)
            ss_tot_pl = np.sum((I_post[ok_pl] - np.mean(I_post[ok_pl]))**2)
            r2_pl = 1 - ss_res_pl / ss_tot_pl if ss_tot_pl > 0 else 0
            k_pl = 2
            n_pl = int(ok_pl.sum())
            aic_pl = n_pl * np.log(ss_res_pl / n_pl + 1e-30) + 2 * k_pl
            result['gamma'] = gamma
            result['A_pl'] = A_pl
            result['r2_pl'] = r2_pl
            result['aic_pl'] = aic_pl
            result['I_fit_pl'] = I_fit_pl
        except Exception:
            pass

    # Exponential fit: I(u) = A * exp(-(u-1)/tau) + C
    try:
        def exp_decay(u, A, tau, C):
            return A * np.exp(-(u - u_peak) / tau) + C
        p0 = [float(I_post[0] - I_final), 0.5, float(I_final)]
        popt, _ = curve_fit(exp_decay, u_post, I_post, p0=p0, maxfev=10000)
        I_fit_exp = exp_decay(u_post, *popt)
        ss_res_exp = np.sum((I_post - I_fit_exp)**2)
        ss_tot_exp = np.sum((I_post - np.mean(I_post))**2)
        r2_exp = 1 - ss_res_exp / ss_tot_exp if ss_tot_exp > 0 else 0
        k_exp = 3
        n_exp = len(u_post)
        aic_exp = n_exp * np.log(ss_res_exp / n_exp + 1e-30) + 2 * k_exp
        result['tau'] = popt[1]
        result['r2_exp'] = r2_exp
        result['aic_exp'] = aic_exp
        result['I_fit_exp'] = I_fit_exp
    except Exception:
        pass

    return result"""))

# ══════════════════════════════════════════════════════════════════
# Cell 9: Loading loop
# ══════════════════════════════════════════════════════════════════
cells.append(code(r"""# ── Load & process all 8 scenarios ────────────────────────────────

R = OrderedDict()

for label, cfg in SCENARIOS.items():
    # ── Discover folders (handle no_grid_subdir for v3 checkpoints) ──
    grid_frames = []
    if cfg.get('no_grid_subdir'):
        buy_p  = EVAL_BASE / cfg['key'] / 'context_500_buy'
        sell_p = EVAL_BASE / cfg['key'] / 'context_500_sell'
        if buy_p.exists() and sell_p.exists():
            gf = discover_v2_folders(buy_p, sell_p)
            if not gf.empty:
                gf['grid_version'] = 'v3_ckpt'
                grid_frames.append(gf)
    else:
        for gdir in _GRID_DIRS:
            buy_p  = EVAL_BASE / cfg['key'] / gdir / 'context_500_buy'
            sell_p = EVAL_BASE / cfg['key'] / gdir / 'context_500_sell'
            if buy_p.exists() and sell_p.exists():
                gf = discover_v2_folders(buy_p, sell_p)
                if not gf.empty:
                    gf['grid_version'] = gdir
                    grid_frames.append(gf)
    if not grid_frames:
        print(f"SKIP {label}: no folders found")
        continue
    grid = pd.concat(grid_frames, ignore_index=True)
    print(f"\n{'='*60}\n  {label}: {len(grid)} configs (grids: {grid['grid_version'].unique().tolist()})")
    data = load_all_v2(grid)
    print(f"  Loaded {len(data)} folders, keys sample: {list(data.keys())[:3]}")

    # Debug: check one folder's structure
    if data:
        _f0 = next(iter(data))
        _d0 = data[_f0]
        print(f"  Folder '{_f0}':")
        print(f"    buy  books={len(_d0['buy']['books'])}  msgs={len(_d0['buy']['msgs'])}")
        print(f"    sell books={len(_d0['sell']['books'])} msgs={len(_d0['sell']['msgs'])}")
        _aggr = load_aggressive_indices(grid.iloc[0]['buy_path'])
        print(f"    aggressive_indices: {_aggr} (len={len(_aggr)})")

    # ── Beta (exclude mb=20) ──
    pc_all = extract_point_cloud(data, grid)
    pc = pc_all[pc_all['mb'] != 20] if len(pc_all) > 0 else pc_all
    print(f"  Point cloud: {len(pc_all)} total, {len(pc)} after mb!=20 filter")
    bstat = compute_global_beta(pc)
    bbetas = bootstrap_beta(pc, N_BOOTSTRAP)

    # ── Sigma-normalized master curves ──
    curves = {}
    for _, row in grid.iterrows():
        f = row['folder']
        if f not in data:
            continue
        aggr = load_aggressive_indices(row['buy_path'])
        cv = compute_master_curve(data[f]['buy'], data[f]['sell'], f, aggr)
        if cv is not None:
            curves[f] = cv

    # ── Raw curves ──
    raw_curves = {}
    for _, row in grid.iterrows():
        f = row['folder']
        if f not in data:
            continue
        aggr = load_aggressive_indices(row['buy_path'])
        cv = compute_raw_curve(data[f]['buy'], data[f]['sell'], f, aggr)
        if cv is not None:
            raw_curves[f] = cv

    # ── Combined impact → stability + gamma ──
    stab_rows, metrics_rows = [], []
    for _, row in grid.iterrows():
        f = row['folder']
        if f not in data:
            continue
        st = compute_combined_impact(data[f]['buy'], data[f]['sell'], f)
        stab_rows.append(stability_for_folder(st, row))
        if st is not None:
            metrics_rows.append({'folder': f, 'i': row['i'], 'mb': row['mb'],
                                 'V': row['V'], 'peak': st['peak']})
    stab_df = pd.DataFrame(stab_rows)
    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()

    # ── Gamma (V-scaling) ──
    gamma_rows = []
    if not metrics_df.empty:
        for (iv, mbv), grp in metrics_df.groupby(['i', 'mb']):
            grp = grp.sort_values('V')
            if len(grp) < 3:
                continue
            Vs = grp['V'].values.astype(float)
            pks = grp['peak'].values
            if np.any(pks <= 0) or np.any(Vs <= 0):
                continue
            sl = linregress(np.log(Vs), np.log(pks))
            gamma_rows.append({'i': iv, 'mb': mbv, 'gamma': sl.slope,
                               'r2': sl.rvalue**2, 'se': sl.stderr})
    gamma_df = pd.DataFrame(gamma_rows) if gamma_rows else pd.DataFrame()

    # ── Relaxation ratios (from RAW curves) ──
    relax_rows = []
    for f, cv in raw_curves.items():
        u, m = cv['u_grid'], cv['combined_mean']
        I_peak = float(np.interp(1.0, u, m))
        if abs(I_peak) < 1e-12:
            continue
        I_final = float(m[-1])
        relax_rows.append({'folder': f, 'I_peak': I_peak, 'I_final': I_final,
                           'ratio': I_final / I_peak,
                           'mb': cv['mb'], 'V': cv['V'],
                           'i': cv['i'], 'Q': cv['Q']})
    relax_df = pd.DataFrame(relax_rows) if relax_rows else pd.DataFrame()

    R[label] = {
        'grid': grid, 'pc': pc, 'beta': bstat, 'boot': bbetas,
        'curves': curves, 'raw_curves': raw_curves, 'relax_df': relax_df,
        'stab_df': stab_df, 'gamma_df': gamma_df,
    }
    stable_frac = stab_df['stabilized'].mean() if not stab_df.empty else 0
    relax_med = relax_df['ratio'].median() if not relax_df.empty else np.nan
    print(f"  beta={bstat['beta']:.4f}  R2={bstat['r2']:.4f}  n={bstat['n']:,}")
    print(f"  Curves: {len(curves)} sigma-norm, {len(raw_curves)} raw")
    print(f"  Relax median: {relax_med:.3f}, Stable: {stable_frac:.0%}, Gamma: {len(gamma_rows)}")

    del data
    gc.collect()

print(f"\n{'='*60}\nLoaded {len(R)} scenarios: {list(R.keys())}")"""))

# ══════════════════════════════════════════════════════════════════
# Part A: Core Impact Analysis
# ══════════════════════════════════════════════════════════════════

# Cell 10: Section 1 header
cells.append(md(r"""---
## 1. Square-Root Law ($\beta$)

**Theory**: $\Delta p / \sigma \sim (Q/V)^\beta$ with $\beta = 0.5$ (Kyle, 1985; Toth et al., 2011)."""))

# Cell 11: Table 1
cells.append(code(r"""# ── Table 1: Global Beta Comparison ──

rows = []
for label, r in R.items():
    b = r['beta']
    ci = np.percentile(r['boot'], [2.5, 97.5]) if len(r['boot']) > 0 else [np.nan, np.nan]
    rows.append({'Model': label, 'beta': f"{b['beta']:.3f}",
                 'R2': f"{b['r2']:.3f}", 'N': f"{b['n']:,}",
                 '95% CI': f"[{ci[0]:.3f}, {ci[1]:.3f}]"})
table1 = pd.DataFrame(rows)
print("\n── Table 1: Global Beta Comparison ──")
print(table1.to_string(index=False))

# LaTeX
print("\n── LaTeX ──")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Model & $\\beta$ & $R^2$ & $N$ & 95\\% CI \\\\")
print("\\midrule")
for _, r in table1.iterrows():
    print(f"{r['Model']} & {r['beta']} & {r['R2']} & {r['N']} & {r['95% CI']} \\\\")
print("\\bottomrule")
print("\\end{tabular}")"""))

# Cell 12: Figure 3 (regression lines — FULL_W for 8 models)
cells.append(code(r"""# ── Figure 3: Beta Regression Lines (full width for 8 models) ──

fig = go.Figure()
x_range = np.array([-16, -4])

for label, r in R.items():
    pc = r['pc']
    if pc.empty:
        continue
    color = SCENARIOS[label]['color']
    dash  = SCENARIOS[label]['dash']
    beta  = r['beta']['beta']

    n_show = min(5000, len(pc))
    idx = np.random.RandomState(42).choice(len(pc), n_show, replace=False)
    sub = pc.iloc[idx]
    fig.add_trace(go.Scatter(
        x=sub['x'], y=sub['y'] - sub['alpha'], mode='markers',
        marker=dict(size=2.5, color=color, opacity=0.12),
        name=label, showlegend=False))
    fig.add_trace(go.Scatter(
        x=x_range, y=beta * x_range, mode='lines',
        line=dict(color=color, width=2.5, dash=dash),
        name=f"{label} (\u03b2={beta:.3f})"))

fig.add_trace(go.Scatter(
    x=x_range, y=0.5 * x_range, mode='lines',
    line=dict(color='black', width=1.5, dash='dash'),
    name='Theory (\u03b2=0.5)'))

pub_layout(fig, width=FULL_W, height=480, legend_pos='br')
fig.update_xaxes(title_text='ln(Q / V)')
fig.update_yaxes(title_text='ln(I / \u03c3)')
save_fig(fig, '3. Beta Regression Lines.png')
fig.show()"""))

# Cell 13: Figure 4 (bootstrap — FULL_W)
cells.append(code(r"""# ── Figure 4: Bootstrap Beta Distributions (full width) ──

fig = go.Figure()
for label, r in R.items():
    bb = r['boot']
    if len(bb) == 0:
        continue
    fig.add_trace(go.Histogram(
        x=bb, nbinsx=50, name=label, opacity=0.55,
        marker_color=SCENARIOS[label]['color'],
        marker_line_color='black', marker_line_width=0.5))

fig.add_vline(x=0.5, line_dash='dash', line_color='black', line_width=1.5,
              annotation_text='\u03b2 = 0.5', annotation_font_size=11,
              annotation_position='top left')

pub_layout(fig, width=FULL_W, height=400, legend_pos='tr',
           barmode='overlay')
fig.update_xaxes(title_text='\u03b2 (bootstrap)')
fig.update_yaxes(title_text='Count')
save_fig(fig, '4. Bootstrap Beta Distributions.png')
fig.show()"""))

# Cell 14: Section 2 header
cells.append(md(r"""---
## 2. Master Curves & Impact Relaxation

**Theory**: after rescaling by $Q^\beta \sigma$, individual impact curves should collapse
onto a universal master curve $\mathcal{F}(u)$ with relaxation ratio
$\mathcal{F}(\infty)/\mathcal{F}(1) \approx 2/3$ (Bouchaud et al., 2004)."""))

# Cell 15: Figure 1 (master curves grid — 8 models)
cells.append(code(r"""# ── Figure 1: Master Curves (full width, 4x2 grid for 8 models) ──

_u_all = []
for label, r in R.items():
    for f, c in r['curves'].items():
        _u_all.append(c['u_grid'][-1])
U_MASTER = min(3.0, np.percentile(_u_all, 10)) if _u_all else 3.0
print(f"  Master curves plot limit: u <= {U_MASTER:.2f}")

n_scn = len(R)
n_cols = 2
n_rows = math.ceil(n_scn / n_cols)
fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f'<b>{l}</b>' for l in R.keys()],
    horizontal_spacing=0.10, vertical_spacing=0.08)

palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2

for idx, (label, r) in enumerate(R.items()):
    row, col = idx // n_cols + 1, idx % n_cols + 1
    sorted_f = sorted(r['curves'].keys(),
        key=lambda f: (parse_folder_params_v2(f)[2], parse_folder_params_v2(f)[0]))
    for fi, folder in enumerate(sorted_f):
        c = r['curves'][folder]
        u, m = c['u_grid'], c['combined_mean']
        mask = u <= U_MASTER
        fig.add_trace(go.Scatter(
            x=u[mask], y=m[mask], mode='lines',
            line=dict(color=palette[fi % len(palette)], width=1.2),
            showlegend=False), row=row, col=col)
    fig.add_vline(x=1.0, line_dash='dot', line_color='rgba(0,0,0,0.35)',
                  line_width=1, row=row, col=col)

fig.update_layout(
    width=FULL_W, height=int(FULL_W * 0.36 * n_rows),
    template='plotly_white',
    font=dict(family='Times New Roman, DejaVu Serif, serif', size=12, color='black'),
    title=None,
    margin=dict(l=55, r=15, t=35, b=50),
)
fig.update_xaxes(**_AX, title_text='u = n / L', title_font_size=12, tickfont_size=10)
fig.update_yaxes(**_AX, title_text='I<sub>norm</sub>(u)', title_font_size=12, tickfont_size=10)
fig.update_annotations(font_size=13)
save_fig(fig, '1. Master Curves.png')
fig.show()"""))

# Cell 16: Figure 2 (average master curve)
cells.append(code(r"""# ── Figure 2: Average Master Curve ──

fig = go.Figure()

u_max_all = []
for label, r in R.items():
    for f, c in r['curves'].items():
        u_max_all.append(c['u_grid'][-1])
U_PLOT = min(3.0, np.percentile(u_max_all, 10)) if u_max_all else 3.0
u_common = np.linspace(0, U_PLOT, 300)
print(f"  Average master curve u_max = {U_PLOT:.2f}  (10th percentile of {len(u_max_all)} curves)")

for label, r in R.items():
    interps = []
    for f, c in r['curves'].items():
        if c['u_grid'][-1] >= U_PLOT:
            interps.append(np.interp(u_common, c['u_grid'], c['combined_mean']))
    if not interps:
        print(f"  WARNING: {label} -- no curves reach u={U_PLOT:.2f}, skipping")
        continue
    avg = np.mean(interps, axis=0)
    std = np.std(interps, axis=0)
    color = SCENARIOS[label]['color']

    rc, gc_, bc = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    fill_rgba = f'rgba({rc},{gc_},{bc},0.12)'

    fig.add_trace(go.Scatter(
        x=np.concatenate([u_common, u_common[::-1]]),
        y=np.concatenate([avg + std, (avg - std)[::-1]]),
        fill='toself', fillcolor=fill_rgba,
        line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=u_common, y=avg, mode='lines',
        line=dict(color=color, width=2.5), name=label))

fig.add_vline(x=1.0, line_dash='dot', line_color='rgba(0,0,0,0.35)', line_width=1)

pub_layout(fig, width=FULL_W, height=420, legend_pos='tr')
fig.update_xaxes(title_text='u = n / L')
fig.update_yaxes(title_text='I<sub>norm</sub>(u)')
save_fig(fig, '2. Average Master Curve.png')
fig.show()"""))

# Cell 17: Table 2 + Figure 5 (relaxation)
cells.append(code(r"""# ── Table 2: Relaxation Ratio ──

rows = []
for label, r in R.items():
    rdf = r['relax_df']
    if rdf.empty:
        continue
    med   = rdf['ratio'].median()
    mean  = rdf['ratio'].mean()
    std   = rdf['ratio'].std()
    cv    = std / abs(mean) if abs(mean) > 1e-12 else np.nan
    rows.append({'Model': label,
                 'Median ratio': f"{med:.3f}",
                 'Mean +/- std': f"{mean:.3f} +/- {std:.3f}",
                 'CV': f"{cv:.3f}",
                 'Delta from 2/3': f"{abs(med - 2/3):.3f}"})
table2 = pd.DataFrame(rows)
print("\n── Table 2: Relaxation Ratio (I_final / I_peak) ──")
print("  Bouchaud theoretical: 2/3 = 0.667")
print(table2.to_string(index=False))

# LaTeX
print("\n── LaTeX ──")
print("\\begin{tabular}{lcccr}")
print("\\toprule")
print("Model & Median & Mean $\\pm$ std & CV & $|\\Delta|$ from $\\frac{2}{3}$ \\\\")
print("\\midrule")
for _, r in table2.iterrows():
    print(f"{r['Model']} & {r['Median ratio']} & {r['Mean +/- std']} & {r['CV']} & {r['Delta from 2/3']} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# ── Figure 5: Relaxation Ratio box plot ──
fig = go.Figure()
for label, r in R.items():
    rdf = r['relax_df']
    if rdf.empty:
        continue
    fig.add_trace(go.Box(
        y=rdf['ratio'], name=label,
        marker_color=SCENARIOS[label]['color'],
        line_color=SCENARIOS[label]['color'],
        boxpoints='all', jitter=0.3, pointpos=-1.5,
        marker=dict(size=4, opacity=0.5),
        line_width=1.5))

fig.add_hline(y=2/3, line_dash='dash', line_color='black', line_width=1.5,
              annotation_text='2/3', annotation_font_size=11,
              annotation_position='bottom right')

pub_layout(fig, width=FULL_W, height=400, legend_pos='none')
fig.update_xaxes(title_text='', tickangle=-30)
fig.update_yaxes(title_text='I<sub>final</sub> / I<sub>peak</sub>')
save_fig(fig, '5. Relaxation Ratio.png')
fig.show()"""))

# Cell 18: Section 3 header
cells.append(md(r"""---
## 3. Stability

A configuration is **stable** if $\geq 2$ of 3 methods agree: trailing slope, two-window mean,
exponential convergence. Fraction stable = share of configs that stabilize."""))

# Cell 19: Table 3 + Figure 6
cells.append(code(r"""# ── Table 3: Fraction Stable ──

rows = []
for label, r in R.items():
    sdf = r['stab_df']
    frac = sdf['stabilized'].mean() if not sdf.empty else 0.0
    n_stable = int(sdf['stabilized'].sum()) if not sdf.empty else 0
    n_total  = len(sdf) if not sdf.empty else 0
    rows.append({'Model': label,
                 'Stable': n_stable, 'Total': n_total,
                 'Fraction': f"{frac:.0%}"})
table3 = pd.DataFrame(rows)
print("\n── Table 3: Fraction Stable (2+/3 votes) ──")
print(table3.to_string(index=False))

# LaTeX
print("\n── LaTeX ──")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("Model & Stable / Total & Fraction \\\\")
print("\\midrule")
for _, r in table3.iterrows():
    print(f"{r['Model']} & {r['Stable']}/{r['Total']} & {r['Fraction']} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# ── Figure 6: Fraction Stable bar chart ──
labels = [r['Model'] for _, r in table3.iterrows()]
fracs  = [float(r['Fraction'].strip('%')) / 100 for _, r in table3.iterrows()]
colors = [SCENARIOS[l]['color'] for l in labels]

fig = go.Figure(go.Bar(
    x=labels, y=fracs, marker_color=colors, width=0.55,
    marker_line_color='black', marker_line_width=1))

fig.add_hline(y=0.5, line_dash='dot', line_color='rgba(0,0,0,0.35)', line_width=1)

pub_layout(fig, width=FULL_W, height=400, legend_pos='none',
           yaxis_range=[0, 1.05])
fig.update_xaxes(title_text='', tickangle=-30)
fig.update_yaxes(title_text='Fraction stable (>=2/3 votes)')
save_fig(fig, '6. Fraction Stable.png')
fig.show()"""))

# Cell 20: Section 4 header
cells.append(md(r"""---
## 4. Volume Scaling ($\gamma$)

$\text{Peak impact} \propto V^\gamma$. Square-root law predicts $\gamma = \beta \approx 0.5$."""))

# Cell 21: Figure 99
cells.append(code(r"""# ── Figure 99: Gamma Distribution ──

fig = go.Figure()
for label, r in R.items():
    gdf = r['gamma_df']
    if gdf.empty:
        continue
    fig.add_trace(go.Box(
        y=gdf['gamma'], name=label,
        marker_color=SCENARIOS[label]['color'],
        line_color=SCENARIOS[label]['color'],
        boxpoints='all', jitter=0.3, pointpos=-1.5,
        marker=dict(size=4, opacity=0.5),
        line_width=1.5))

fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1.5,
              annotation_text='\u03b3 = 0.5', annotation_font_size=11,
              annotation_position='bottom right')

pub_layout(fig, width=FULL_W, height=400, legend_pos='none')
fig.update_xaxes(title_text='', tickangle=-30)
fig.update_yaxes(title_text='\u03b3 (volume scaling exponent)')
save_fig(fig, '99. Gamma Distribution.png')
fig.show()

for label, r in R.items():
    gdf = r['gamma_df']
    if not gdf.empty:
        print(f"{label:15s}  gamma = {gdf['gamma'].mean():.3f} +/- {gdf['gamma'].std():.3f}  (n={len(gdf)})")"""))

# ══════════════════════════════════════════════════════════════════
# Part B: Extended Impact Analysis
# ══════════════════════════════════════════════════════════════════

# Cell 22: Section 5 header
cells.append(md(r"""---
## 5. Participation Rate

**Theory**: Zarinelli et al. (2015) argue that participation rate $\rho = Q / (V \cdot T)$
is a better predictor than simple volume fraction $Q/V$.
If $\Delta p / \sigma \sim \rho^\beta$, the exponent may differ from the Q/V-based estimate."""))

# Cell 23: Participation rate compute + table
cells.append(code(r"""# ── Section 5: Participation Rate ─────────────────────────────────

participation_results = OrderedDict()

for label, r in R.items():
    pc = r['pc'].copy()
    if pc.empty:
        continue

    folder_params = {}
    for f in pc['folder'].unique():
        i_val, c_val, mb_val, V_val = parse_folder_params_v2(f)
        folder_params[f] = {'i': i_val, 'c': c_val, 'mb': mb_val, 'V': V_val}

    i_vals = pc['folder'].map(lambda f: folder_params[f]['i'])
    mb_vals = pc['folder'].map(lambda f: folder_params[f]['mb'])

    # T_frac = fraction of context used for execution
    T_frac = (pc['insertion_idx'] + 1) * mb_vals / N_COND_MSGS
    T_frac = T_frac.clip(lower=1e-6)

    # ln(rho) = ln(Q/V) - ln(T)
    ln_rho = pc['x'] - np.log(T_frac)
    y_adj = pc['y'] - pc['alpha']

    # Standard regression: y_adj = beta_qv * x
    ok_std = np.isfinite(pc['x']) & np.isfinite(y_adj) & (pc['x'] != 0)
    x_std = pc['x'].values[ok_std]
    y_std = y_adj.values[ok_std]
    beta_qv = float(np.dot(x_std, y_std) / np.dot(x_std, x_std))
    ss_res_qv = np.sum((y_std - beta_qv * x_std) ** 2)
    ss_tot_qv = np.sum(y_std ** 2)
    r2_qv = 1 - ss_res_qv / ss_tot_qv if ss_tot_qv > 0 else 0

    # Participation rate regression: y_adj = beta_rho * ln(rho)
    ok_rho = np.isfinite(ln_rho) & np.isfinite(y_adj) & (ln_rho != 0)
    x_rho = ln_rho.values[ok_rho]
    y_rho = y_adj.values[ok_rho]
    beta_rho = float(np.dot(x_rho, y_rho) / np.dot(x_rho, x_rho))
    ss_res_rho = np.sum((y_rho - beta_rho * x_rho) ** 2)
    ss_tot_rho = np.sum(y_rho ** 2)
    r2_rho = 1 - ss_res_rho / ss_tot_rho if ss_tot_rho > 0 else 0

    participation_results[label] = {
        'beta_qv': beta_qv, 'r2_qv': r2_qv,
        'beta_rho': beta_rho, 'r2_rho': r2_rho,
        'n': int(ok_std.sum()),
        'ln_rho': ln_rho, 'y_adj': y_adj, 'ok_rho': ok_rho,
    }

# ── Table ──
rows = []
for label, pr in participation_results.items():
    rows.append({
        'Model': label,
        'beta_QV': f"{pr['beta_qv']:.3f}",
        'R2_QV': f"{pr['r2_qv']:.4f}",
        'beta_rho': f"{pr['beta_rho']:.3f}",
        'R2_rho': f"{pr['r2_rho']:.4f}",
        'N': f"{pr['n']:,}",
    })
table_pr = pd.DataFrame(rows)
print('\n── Participation Rate: Q/V vs rho = Q/(V*T) ──')
print(table_pr.to_string(index=False))"""))

# Cell 24: Participation rate plot
cells.append(code(r"""# ── Figure: Participation Rate scatter (2-panel) ──

fig = make_subplots(rows=1, cols=2,
    subplot_titles=['<b>Standard: ln(Q/V)</b>', '<b>Participation: ln(\u03c1)</b>'],
    horizontal_spacing=0.14)

x_range_qv = np.array([-16, -4])
x_range_rho = np.array([-12, 2])

for label, pr in participation_results.items():
    pc = R[label]['pc']
    color = SCENARIOS[label]['color']
    dash = SCENARIOS[label]['dash']

    n_show = min(3000, len(pc))
    idx = np.random.RandomState(42).choice(len(pc), n_show, replace=False)
    sub_y = (pc['y'] - pc['alpha']).values[idx]

    # Left panel: Q/V
    fig.add_trace(go.Scatter(
        x=pc['x'].values[idx], y=sub_y, mode='markers',
        marker=dict(size=2, color=color, opacity=0.08),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_range_qv, y=pr['beta_qv'] * x_range_qv, mode='lines',
        line=dict(color=color, width=2, dash=dash),
        name=f"{label} (R2={pr['r2_qv']:.3f})"), row=1, col=1)

    # Right panel: rho
    ln_rho_sub = pr['ln_rho'].values[idx]
    fig.add_trace(go.Scatter(
        x=ln_rho_sub, y=sub_y, mode='markers',
        marker=dict(size=2, color=color, opacity=0.08),
        showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_range_rho, y=pr['beta_rho'] * x_range_rho, mode='lines',
        line=dict(color=color, width=2, dash=dash),
        showlegend=False), row=1, col=2)

# Theory
fig.add_trace(go.Scatter(x=x_range_qv, y=0.5*x_range_qv, mode='lines',
    line=dict(color='black', width=1.5, dash='dash'),
    name='\u03b2=0.5'), row=1, col=1)
fig.add_trace(go.Scatter(x=x_range_rho, y=0.5*x_range_rho, mode='lines',
    line=dict(color='black', width=1.5, dash='dash'),
    showlegend=False), row=1, col=2)

fig.update_layout(
    width=FULL_W, height=420,
    template='plotly_white',
    font=dict(family='Times New Roman, DejaVu Serif, serif', size=12, color='black'),
    margin=dict(l=55, r=15, t=35, b=55),
    legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom',
                bgcolor='rgba(255,255,255,0.85)', bordercolor='black', borderwidth=1, font_size=10),
)
fig.update_xaxes(**_AX)
fig.update_yaxes(**_AX)
fig.update_xaxes(title_text='ln(Q / V)', row=1, col=1)
fig.update_xaxes(title_text='ln(\u03c1)', row=1, col=2)
fig.update_yaxes(title_text='ln(I / \u03c3)', row=1, col=1)
fig.update_yaxes(title_text='ln(I / \u03c3)', row=1, col=2)
save_fig(fig, '10. Participation Rate.png')
fig.show()"""))

# Cell 25: Section 6 header
cells.append(md(r"""---
## 6. Decay Function Fitting

**Theory**: Post-peak impact decays as $I(u) \sim (u-1)^{-\gamma}$ with $\gamma \in [0.5, 0.8]$
(Bouchaud et al., 2004; Brokmann et al., 2015)."""))

# Cell 26: Decay compute + table
cells.append(code(r"""# ── Section 6: Decay Function Fitting ─────────────────────────────

decay_results = OrderedDict()

for label, r in R.items():
    fits = []
    for f, cv in r['curves'].items():
        dr = fit_decay(cv['u_grid'], cv['combined_mean'])
        if dr is not None and 'gamma' in dr:
            fits.append({
                'folder': f, 'gamma': dr['gamma'],
                'r2_pl': dr.get('r2_pl', np.nan),
                'r2_exp': dr.get('r2_exp', np.nan),
                'aic_pl': dr.get('aic_pl', np.nan),
                'aic_exp': dr.get('aic_exp', np.nan),
                'tau': dr.get('tau', np.nan),
            })
    decay_df = pd.DataFrame(fits) if fits else pd.DataFrame()
    decay_results[label] = decay_df

# ── Table ──
rows = []
for label, ddf in decay_results.items():
    if ddf.empty:
        continue
    g = ddf['gamma']
    rows.append({
        'Model': label,
        'gamma_mean': f'{g.mean():.3f}',
        'gamma_std': f'{g.std():.3f}',
        'gamma_med': f'{g.median():.3f}',
        'R2_PL': f"{ddf['r2_pl'].mean():.3f}",
        'R2_Exp': f"{ddf['r2_exp'].mean():.3f}",
        'AIC_PL<Exp': f"{(ddf['aic_pl'] < ddf['aic_exp']).sum()}/{len(ddf)}",
        'n_configs': len(ddf),
    })
table_decay = pd.DataFrame(rows)
print('\n── Decay Fitting: Power-Law gamma ──')
print('  Expected: gamma in [0.5, 0.8] (Brokmann 2015)')
print(table_decay.to_string(index=False))"""))

# Cell 27: Decay box plot
cells.append(code(r"""# ── Figure: Decay exponent distribution ──

fig = go.Figure()
for label, ddf in decay_results.items():
    if ddf.empty:
        continue
    fig.add_trace(go.Box(
        y=ddf['gamma'], name=label,
        marker_color=SCENARIOS[label]['color'],
        line_color=SCENARIOS[label]['color'],
        boxpoints='all', jitter=0.3, pointpos=-1.5,
        marker=dict(size=4, opacity=0.5),
        line_width=1.5))

fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1,
              annotation_text='\u03b3=0.5', annotation_font_size=10,
              annotation_position='bottom right')
fig.add_hline(y=0.8, line_dash='dash', line_color='grey', line_width=1,
              annotation_text='\u03b3=0.8', annotation_font_size=10,
              annotation_position='top right')

pub_layout(fig, width=FULL_W, height=400, legend_pos='none')
fig.update_xaxes(title_text='', tickangle=-30)
fig.update_yaxes(title_text='\u03b3 (decay exponent)')
save_fig(fig, '11. Decay Exponent.png')
fig.show()"""))

# Cell 28: Decay example fits
cells.append(code(r"""# ── Figure: Example decay fits (one config per model) ──

n_scn = len(R)
n_cols = min(n_scn, 4)
n_rows = math.ceil(n_scn / n_cols)
fig = make_subplots(rows=n_rows, cols=n_cols,
    subplot_titles=[f'<b>{l}</b>' for l in R.keys()],
    horizontal_spacing=0.10, vertical_spacing=0.12)

for idx, (label, r) in enumerate(R.items()):
    row, col = idx // n_cols + 1, idx % n_cols + 1
    ddf = decay_results[label]
    if ddf.empty:
        continue
    med_idx = (ddf['gamma'] - ddf['gamma'].median()).abs().idxmin()
    folder = ddf.loc[med_idx, 'folder']
    cv = r['curves'][folder]
    dr = fit_decay(cv['u_grid'], cv['combined_mean'])
    if dr is None:
        continue

    u_post = dr['u_post']
    fig.add_trace(go.Scatter(x=u_post, y=dr['I_post'], mode='lines',
        line=dict(color=SCENARIOS[label]['color'], width=2),
        name='Data', showlegend=(idx==0)), row=row, col=col)
    if 'I_fit_pl' in dr:
        fig.add_trace(go.Scatter(x=u_post, y=dr['I_fit_pl'], mode='lines',
            line=dict(color='red', width=1.5, dash='dash'),
            name='Power-law', showlegend=(idx==0)), row=row, col=col)
    if 'I_fit_exp' in dr:
        fig.add_trace(go.Scatter(x=u_post, y=dr['I_fit_exp'], mode='lines',
            line=dict(color='blue', width=1.5, dash='dot'),
            name='Exponential', showlegend=(idx==0)), row=row, col=col)

fig.update_layout(
    width=FULL_W, height=int(FULL_W * 0.35 * n_rows),
    template='plotly_white',
    font=dict(family='Times New Roman, DejaVu Serif, serif', size=12, color='black'),
    margin=dict(l=55, r=15, t=35, b=50),
    legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.85)', bordercolor='black', borderwidth=1, font_size=10),
)
fig.update_xaxes(**_AX, title_text='u', title_font_size=11)
fig.update_yaxes(**_AX, title_text='I<sub>norm</sub>(u)', title_font_size=11)
save_fig(fig, '11b. Decay Fits Example.png')
fig.show()"""))

# Cell 29: Section 7 header
cells.append(md(r"""---
## 7. Permanent & Temporary Decomposition

**Theory**: Almgren & Chriss (2001) decompose impact into permanent ($I_\infty$) and temporary
($I_\text{peak} - I_\infty$) components. Huberman-Stanzl no-arbitrage requires $\beta_\text{perm} \approx 1$;
temporary impact scales as $\beta_\text{temp} \approx 0.5$ (Bershova & Rakhlin, 2013)."""))

# Cell 30: Decomposition compute + table
cells.append(code(r"""# ── Section 7: Permanent / Temporary Decomposition ────────────────

decomp_results = OrderedDict()

for label, r_data in R.items():
    rdf = r_data['relax_df'].copy()
    if rdf.empty or len(rdf) < 3:
        continue

    rdf['I_perm'] = rdf['I_final']
    rdf['I_temp'] = rdf['I_peak'] - rdf['I_final']

    pc = r_data['pc']

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

# ── Table ──
rows = []
for label, dr in decomp_results.items():
    rows.append({
        'Model': label,
        'beta_perm': f"{dr['beta_perm']:.3f}",
        'R2_perm': f"{dr['r2_perm']:.3f}",
        'beta_temp': f"{dr['beta_temp']:.3f}",
        'R2_temp': f"{dr['r2_temp']:.3f}",
    })
table_decomp = pd.DataFrame(rows)
print('\n── Permanent/Temporary Decomposition ──')
print('  Theory: beta_perm ~ 1.0 (Huberman-Stanzl), beta_temp ~ 0.5')
print(table_decomp.to_string(index=False))"""))

# Cell 31: Decomposition plot
cells.append(code(r"""# ── Figure: Perm/Temp decomposition log-log ──

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
save_fig(fig, '12. Perm Temp Decomposition.png')
fig.show()"""))

# Cell 32: Section 8 header
cells.append(md(r"""---
## 8. No-Arbitrage Consistency Check

Five tests (Gatheral, 2010; Huberman & Stanzl, 2004):
- **A**: Concavity ($\delta < 1$)
- **B**: Permanent impact linearity ($\beta_\text{perm} \in [0.7, 1.3]$)
- **C**: Decay kernel ($\gamma \in [0.3, 1.0]$)
- **D**: Relaxation bounds ($r \in [0.5, 1.0]$)
- **E**: Gatheral condition ($\delta \leq 1/(1 + 2\gamma)$)"""))

# Cell 33: No-arbitrage scorecard
cells.append(code(r"""# ── Section 8: No-Arbitrage Consistency ───────────────────────────

arb_rows = []

for label, r_data in R.items():
    delta = r_data['beta']['beta']
    relax_med = r_data['relax_df']['ratio'].median() if not r_data['relax_df'].empty else np.nan

    ddf = decay_results.get(label, pd.DataFrame())
    gamma_med = ddf['gamma'].median() if not ddf.empty else np.nan

    dr = decomp_results.get(label, {})
    beta_perm = dr.get('beta_perm', np.nan)

    # Test A: Concavity (delta < 1)
    test_A = delta < 1.0 if np.isfinite(delta) else False
    # Test B: Permanent impact linearity
    test_B = (0.7 <= beta_perm <= 1.3) if np.isfinite(beta_perm) else False
    # Test C: Decay kernel consistency
    test_C = (0.3 <= gamma_med <= 1.0) if np.isfinite(gamma_med) else False
    # Test D: Relaxation ratio bounds
    test_D = (0.5 <= relax_med <= 1.0) if np.isfinite(relax_med) else False
    # Test E: Gatheral condition
    if np.isfinite(delta) and np.isfinite(gamma_med) and gamma_med > 0:
        gatheral_bound = 1.0 / (1.0 + 2.0 * gamma_med)
        test_E = delta <= gatheral_bound
    else:
        gatheral_bound = np.nan
        test_E = False

    n_pass = sum([test_A, test_B, test_C, test_D, test_E])

    arb_rows.append({
        'Model': label,
        'delta': f'{delta:.3f}',
        'beta_perm': f'{beta_perm:.3f}' if np.isfinite(beta_perm) else '---',
        'gamma': f'{gamma_med:.3f}' if np.isfinite(gamma_med) else '---',
        'relax_r': f'{relax_med:.3f}' if np.isfinite(relax_med) else '---',
        'A:Concav': 'PASS' if test_A else 'FAIL',
        'B:Perm~1': 'PASS' if test_B else 'FAIL',
        'C:Decay': 'PASS' if test_C else 'FAIL',
        'D:Relax': 'PASS' if test_D else 'FAIL',
        'E:Gather': 'PASS' if test_E else 'FAIL',
        'Score': f'{n_pass}/5',
    })

arb_table = pd.DataFrame(arb_rows)
print('\n── No-Arbitrage Consistency Check ──')
print(arb_table.to_string(index=False))

# LaTeX
print('\n── LaTeX ──')
print('\\begin{tabular}{lccccccccc}')
print('\\toprule')
print('Model & $\\delta$ & $\\beta_p$ & $\\gamma$ & $r$ & A & B & C & D & E \\\\')
print('\\midrule')
for _, row in arb_table.iterrows():
    vals = ' & '.join([row['Model'], row['delta'], row['beta_perm'], row['gamma'],
                       row['relax_r'], row['A:Concav'], row['B:Perm~1'],
                       row['C:Decay'], row['D:Relax'], row['E:Gather']])
    print(f'{vals} \\\\')
print('\\bottomrule')
print('\\end{tabular}')"""))

# Cell 34: Section 9 header
cells.append(md(r"""---
## 9. Per-Day Beta Stability

Estimate $\beta$ separately for each trading day to assess temporal stability."""))

# Cell 35: Per-day beta compute + table
cells.append(code(r"""# ── Section 9: Per-Day Beta ───────────────────────────────────────

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
    day_df = pd.DataFrame(day_betas)
    perday_results[label] = day_df

# ── Table ──
rows = []
for label, ddf in perday_results.items():
    if ddf.empty:
        continue
    b = ddf['beta']
    rows.append({
        'Model': label,
        'Mean beta': f'{b.mean():.3f}',
        'Std': f'{b.std():.3f}',
        'Min': f'{b.min():.3f}',
        'Max': f'{b.max():.3f}',
        'N_days': len(ddf),
    })
table_perday = pd.DataFrame(rows)
print('\n── Per-Day Beta ──')
print(table_perday.to_string(index=False))"""))

# Cell 36: Per-day beta plot
cells.append(code(r"""# ── Figure: Per-day beta distributions ──

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

pub_layout(fig, width=FULL_W, height=400, legend_pos='none')
fig.update_xaxes(title_text='', tickangle=-30)
fig.update_yaxes(title_text='\u03b2 (per day)')
save_fig(fig, '13. Per-Day Beta.png')
fig.show()"""))

# ══════════════════════════════════════════════════════════════════
# Part C: Model-Size Comparison
# ══════════════════════════════════════════════════════════════════

# Cell 37: Section 10 header
cells.append(md(r"""---
## 10. Model-Size Comparison

Compare S5 variants by parameter count and encoding:
- **LobS5** (v2): 45M params, 22-token encoding, 500 context
- **S5-120M** (v3): 120M params, 24-token encoding, 500 context
- **S5-360M** (v3): 360M params, 24-token encoding, 500 context
- **S5-4K** (v3): 55M params, 24-token encoding, 4096 context"""))

# Cell 38: Beta vs params
cells.append(code(r"""# ── Figure 7: Beta vs Model Size ──

s5_labels = [l for l in R if l in MODEL_META]
if len(s5_labels) >= 2:
    params = [MODEL_META[l]['params'] for l in s5_labels]
    betas  = [R[l]['beta']['beta'] for l in s5_labels]
    boots  = [R[l]['boot'] for l in s5_labels]
    ci_lo  = [np.percentile(b, 2.5) if len(b) > 0 else np.nan for b in boots]
    ci_hi  = [np.percentile(b, 97.5) if len(b) > 0 else np.nan for b in boots]
    colors = [SCENARIOS[l]['color'] for l in s5_labels]

    fig = go.Figure()
    for i, l in enumerate(s5_labels):
        fig.add_trace(go.Scatter(
            x=[params[i]], y=[betas[i]],
            error_y=dict(type='data',
                         array=[ci_hi[i] - betas[i]],
                         arrayminus=[betas[i] - ci_lo[i]],
                         visible=True, thickness=2, width=6),
            mode='markers+text',
            marker=dict(size=14, color=colors[i], line=dict(width=1.5, color='black')),
            text=[l], textposition='top center', textfont=dict(size=11),
            name=l, showlegend=False))

    fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1.5,
                  annotation_text='\u03b2=0.5', annotation_font_size=11,
                  annotation_position='bottom right')

    pub_layout(fig, width=SINGLE_W, height=400, legend_pos='none')
    fig.update_xaxes(title_text='Parameters', type='log',
                     tickvals=[45e6, 55e6, 120e6, 360e6],
                     ticktext=['45M', '55M', '120M', '360M'])
    fig.update_yaxes(title_text='\u03b2')
    save_fig(fig, '20. Beta vs Model Size.png')
    fig.show()
else:
    print("Not enough S5 variants for model-size comparison.")"""))

# Cell 39: Multi-metric bar chart
cells.append(code(r"""# ── Figure 8: S5 Multi-Metric Comparison ──

s5_labels = [l for l in R if l in MODEL_META]
if len(s5_labels) >= 2:
    metrics_data = []
    for l in s5_labels:
        beta = R[l]['beta']['beta']
        relax = R[l]['relax_df']['ratio'].median() if not R[l]['relax_df'].empty else np.nan
        gamma_m = R[l]['gamma_df']['gamma'].mean() if not R[l]['gamma_df'].empty else np.nan
        stable = R[l]['stab_df']['stabilized'].mean() if not R[l]['stab_df'].empty else 0
        metrics_data.append({
            'Model': l,
            'Params': f"{MODEL_META[l]['params']/1e6:.0f}M",
            'Encoding': MODEL_META[l]['encoding'],
            'Context': MODEL_META[l]['context'],
            '|beta-0.5|': abs(beta - 0.5),
            '|relax-2/3|': abs(relax - 2/3) if np.isfinite(relax) else np.nan,
            'R2': R[l]['beta']['r2'],
            'Stable%': stable,
            '|gamma-0.5|': abs(gamma_m - 0.5) if np.isfinite(gamma_m) else np.nan,
        })
    mdf = pd.DataFrame(metrics_data)
    print(mdf.to_string(index=False))

    # Grouped bar chart: smaller = better for |beta-0.5|, |relax-2/3|; larger = better for R2
    fig = go.Figure()
    bar_metrics = ['|beta-0.5|', '|relax-2/3|', '|gamma-0.5|']
    bar_colors = ['#E04040', '#2CA02C', '#FF7F0E']
    for mi, metric in enumerate(bar_metrics):
        fig.add_trace(go.Bar(
            x=mdf['Model'], y=mdf[metric], name=metric,
            marker_color=bar_colors[mi], opacity=0.85,
            marker_line_color='black', marker_line_width=1))

    pub_layout(fig, width=SINGLE_W, height=400, legend_pos='tr',
               barmode='group')
    fig.update_xaxes(title_text='', tickangle=-30)
    fig.update_yaxes(title_text='Distance from theory')
    save_fig(fig, '21. S5 Metrics Comparison.png')
    fig.show()
else:
    print("Not enough S5 variants for metrics comparison.")"""))

# Cell 40: S5 master curves overlay
cells.append(code(r"""# ── Figure 9: S5 Variant Master Curves Overlay ──

s5_labels = [l for l in R if l in MODEL_META]
if len(s5_labels) >= 2:
    fig = go.Figure()

    u_max_all = []
    for l in s5_labels:
        for f, c in R[l]['curves'].items():
            u_max_all.append(c['u_grid'][-1])
    U_PLOT_S5 = min(3.0, np.percentile(u_max_all, 10)) if u_max_all else 3.0
    u_common = np.linspace(0, U_PLOT_S5, 300)

    for l in s5_labels:
        interps = []
        for f, c in R[l]['curves'].items():
            if c['u_grid'][-1] >= U_PLOT_S5:
                interps.append(np.interp(u_common, c['u_grid'], c['combined_mean']))
        if not interps:
            continue
        avg = np.mean(interps, axis=0)
        std = np.std(interps, axis=0)
        color = SCENARIOS[l]['color']
        rc, gc_, bc = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fill_rgba = f'rgba({rc},{gc_},{bc},0.12)'

        fig.add_trace(go.Scatter(
            x=np.concatenate([u_common, u_common[::-1]]),
            y=np.concatenate([avg + std, (avg - std)[::-1]]),
            fill='toself', fillcolor=fill_rgba,
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=u_common, y=avg, mode='lines',
            line=dict(color=color, width=2.5), name=l))

    fig.add_vline(x=1.0, line_dash='dot', line_color='rgba(0,0,0,0.35)', line_width=1)

    pub_layout(fig, width=SINGLE_W, height=400, legend_pos='tr')
    fig.update_xaxes(title_text='u = n / L')
    fig.update_yaxes(title_text='I<sub>norm</sub>(u)')
    save_fig(fig, '22. S5 Master Curves.png')
    fig.show()
else:
    print("Not enough S5 variants for master curve overlay.")"""))

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

# Cell 41: Summary header
cells.append(md(r"""---
## Grand Summary"""))

# Cell 42: Grand summary table
cells.append(code(r"""# ── Grand Summary ─────────────────────────────────────────────────

print("=" * 110)
hdr = (f"{'Model':15s}  {'Params':>7s}  {'beta':>6s}  {'R2':>6s}  {'Relax':>6s}  "
       f"{'Stable':>7s}  {'gamma':>6s}  {'b_rho':>6s}  {'decay':>6s}  "
       f"{'b_perm':>6s}  {'b_temp':>6s}  {'Arb':>5s}")
print(hdr)
print("-" * 110)
for label in R:
    beta = R[label]['beta']['beta']
    r2 = R[label]['beta']['r2']
    relax_med = R[label]['relax_df']['ratio'].median() if not R[label]['relax_df'].empty else np.nan
    stable = R[label]['stab_df']['stabilized'].mean() if not R[label]['stab_df'].empty else 0
    gamma_m = R[label]['gamma_df']['gamma'].mean() if not R[label]['gamma_df'].empty else np.nan

    pr = participation_results.get(label, {})
    beta_rho = pr.get('beta_rho', np.nan)

    ddf = decay_results.get(label, pd.DataFrame())
    gamma_dec = ddf['gamma'].median() if not ddf.empty else np.nan

    dr = decomp_results.get(label, {})
    bp = dr.get('beta_perm', np.nan)
    bt = dr.get('beta_temp', np.nan)

    arb = [r for r in arb_rows if r['Model'] == label]
    score = arb[0]['Score'] if arb else '---'

    mm = MODEL_META.get(label, {})
    params_str = f"{mm['params']/1e6:.0f}M" if 'params' in mm else '---'

    print(f"{label:15s}  {params_str:>7s}  {beta:6.3f}  {r2:6.3f}  {relax_med:6.3f}  "
          f"{stable:6.0%}  {gamma_m:6.3f}  {beta_rho:6.3f}  {gamma_dec:6.3f}  "
          f"{bp:6.3f}  {bt:6.3f}  {score:>5s}")
print("=" * 110)
print(f"{'Theory':15s}  {'':>7s}  {'0.500':>6s}  {'':>6s}  {'0.667':>6s}  "
      f"{'':>7s}  {'0.500':>6s}  {'0.500':>6s}  {'0.5-8':>6s}  "
      f"{'1.000':>6s}  {'0.500':>6s}  {'5/5':>5s}")"""))

# ══════════════════════════════════════════════════════════════════
# Build notebook
# ══════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "200.v3_checkpoint_comparison.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written {len(cells)} cells to {out_path}")
