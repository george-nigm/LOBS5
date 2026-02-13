#!/usr/bin/env python3
"""Generate notebook 120: Cross-Scenario Market Impact Analysis."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source,
            "outputs": [], "execution_count": None, "id": None}

cells = []

# ═══════════════════════════════════════════════════════════════════
# SECTION 0: Setup & Configuration
# ═══════════════════════════════════════════════════════════════════

cells.append(md("""# Cross-Scenario Market Impact Analysis (Notebook 120)

**Compares all 4 ready scenarios** (S5, Historic, Heuristic, CST) with placeholders for RWKV and Coletta.

**Goal**: determine which model best reproduces empirical market impact laws:
- Square-root law: beta ~ 0.5
- Decay ratio: ~ 2/3 (Bouchaud)
- Volume scaling: gamma ~ 0.5

**Structure**:
- **Part A**: Per-scenario full analysis (reproduces 110-style detail for each model)
- **Part B**: Cross-scenario comparison & model ranking"""))

cells.append(code("""import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.stats import linregress
from scipy.optimize import curve_fit
from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm
from IPython.display import display, Markdown
import re, gc, pickle, warnings
warnings.filterwarnings('ignore')"""))

cells.append(code("""# ── Constants ──
MAX_SAMPLES = 10  # SMOKE TEST — set to 2048 for full run
MIDPRICE_MAX = 2_000_000
CONTEXT = 500
TICK_SIZE = 100

# ── Scenario registry ──
SCENARIO_REGISTRY = OrderedDict([
    ('S5 (Neural)',   {'key': 'aggressive_scenario',        'color': '#1f77b4', 'marker': 'circle',       'dash': 'solid'}),
    ('Historic',      {'key': 'historic_scenario',          'color': '#ff7f0e', 'marker': 'square',       'dash': 'dash'}),
    ('Heuristic',     {'key': 'heuristic_scenario',         'color': '#2ca02c', 'marker': 'diamond',      'dash': 'dot'}),
    ('CST',           {'key': 'cst_scenario',               'color': '#d62728', 'marker': 'cross',        'dash': 'dashdot'}),
    ('RWKV (Neural)', {'key': 'rwkv_aggressive_scenario',   'color': '#9467bd', 'marker': 'triangle-up',  'dash': 'longdash'}),
    ('Coletta',       {'key': 'coletta_aggressive_scenario','color': '#8c564b', 'marker': 'star',         'dash': 'longdashdot'}),
])

VOLUME_COLORS = {75: '#e41a1c', 300: '#377eb8', 485: '#4daf4a'}
MB_COLORS = {
    5:  ('#e41a1c', 'rgba(228,26,28,0.12)'),
    10: ('#377eb8', 'rgba(55,126,184,0.12)'),
    15: ('#4daf4a', 'rgba(77,175,74,0.12)'),
    20: ('#984ea3', 'rgba(152,78,163,0.12)'),
}
QUALITY_MAP = {'excellent': 4, 'good': 3, 'acceptable': 2, 'poor': 1}"""))

cells.append(code("""# ── Scenario auto-discovery ──
_BASE_CANDIDATES = [
    Path("/app/output/evalsequences"),
    Path("/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences"),
]
EVAL_BASE = None
for p in _BASE_CANDIDATES:
    if p.exists():
        EVAL_BASE = p
        break
assert EVAL_BASE is not None, "Cannot find evalsequences directory"

_SDM_CANDIDATES = [
    Path("/app/lob_impact/sample_day_map.csv"),
    Path("/scratch/local/homes/80/georgenigm/LOBS5/lob_impact/sample_day_map.csv"),
]
SAMPLE_DAY_MAP = None
for p in _SDM_CANDIDATES:
    if p.exists():
        SAMPLE_DAY_MAP = pd.read_csv(p)
        break
assert SAMPLE_DAY_MAP is not None, "Cannot find sample_day_map.csv"

# Discover active scenarios
ACTIVE_SCENARIOS = OrderedDict()
for label, cfg in SCENARIO_REGISTRY.items():
    scn_base = EVAL_BASE / cfg['key']
    buy_p = scn_base / 'context_500_buy'
    sell_p = scn_base / 'context_500_sell'
    if buy_p.exists() and sell_p.exists():
        buy_dirs = [d for d in buy_p.iterdir() if d.is_dir()]
        sell_dirs = [d for d in sell_p.iterdir() if d.is_dir()]
        if len(buy_dirs) > 0 and len(sell_dirs) > 0:
            ACTIVE_SCENARIOS[label] = {**cfg, 'buy_path': buy_p, 'sell_path': sell_p,
                                        'n_buy': len(buy_dirs), 'n_sell': len(sell_dirs)}
            print(f"  [OK] {label}: {len(buy_dirs)} buy, {len(sell_dirs)} sell folders")
        else:
            print(f"  [SKIP] {label}: empty directories")
    else:
        print(f"  [SKIP] {label}: directory not found")

print(f"\\nActive scenarios: {len(ACTIVE_SCENARIOS)} / {len(SCENARIO_REGISTRY)}")
print(f"Scenarios: {list(ACTIVE_SCENARIOS.keys())}")"""))

# ═══════════════════════════════════════════════════════════════════
# Cell 0.5: Full function library (parameterized from 110)
# ═══════════════════════════════════════════════════════════════════

cells.append(code("""# ═══════════════════════════════════════════════════════════════════
# Full function library (parameterized from notebook 110)
# ═══════════════════════════════════════════════════════════════════

def discover_v2_folders(buy_path, sell_path):
    \"\"\"Auto-discover V2 experiment folders via regex.\"\"\"
    pattern = re.compile(r'^i(\\d+)_c(\\d+)_mb(\\d+)_v(\\d+)_cntxt(.+)$')
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
        rows.append({
            'folder': p.name, 'i': i, 'c': c, 'mb': mb, 'V': V,
            'Q_total': i * V, 'buy_path': p, 'sell_path': sell_p,
        })
    return pd.DataFrame(rows)


def discover_data_params(data_path, max_samples=None):
    cond_dir = data_path / "data_cond"
    pattern = re.compile(r"^(.+?)_(\\d{4}-\\d{2}-\\d{2})_orderbook_real_id_(\\d+)\\.csv$")
    samples = []
    for f in cond_dir.glob("*_orderbook_real_id_*.csv"):
        match = pattern.match(f.name)
        if match:
            samples.append((match.group(1), match.group(2), int(match.group(3))))
    if not samples:
        raise ValueError(f"No orderbook files found in {cond_dir}")
    samples.sort()
    if max_samples is not None and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    return samples


def is_midprice_outlier(book_array, max_midprice):
    midprice = (book_array[:, 0] + book_array[:, 2]) / 2
    return np.any(midprice > max_midprice) or np.any(midprice <= 0)


def load_folder_data(data_path, max_samples=None, max_midprice=None):
    samples = discover_data_params(data_path, max_samples=max_samples)
    gen_books, gen_msgs, cond_lens = {}, {}, {}
    n_outliers = 0
    for ticker, date, sid in samples:
        cond_book_path = data_path / f"data_cond/{ticker}_{date}_orderbook_real_id_{sid}.csv"
        gen_book_path = data_path / f"data_gen/{ticker}_{date}_orderbook_real_id_{sid}_gen_id_0.csv"
        gen_msg_path = data_path / f"data_gen/{ticker}_{date}_message_real_id_{sid}_gen_id_0.csv"
        if not gen_book_path.exists():
            continue
        cond_book = np.loadtxt(cond_book_path, delimiter=',')
        gen_book = np.loadtxt(gen_book_path, delimiter=',')
        full_book = np.vstack([cond_book, gen_book])
        if max_midprice is not None and is_midprice_outlier(full_book, max_midprice):
            n_outliers += 1
            continue
        gen_msg = np.loadtxt(gen_msg_path, delimiter=',')
        cond_msg_path = data_path / f"data_cond/{ticker}_{date}_message_real_id_{sid}.csv"
        cond_msg = np.loadtxt(cond_msg_path, delimiter=',')
        key = (date, sid)
        cond_lens[key] = cond_book.shape[0]
        gen_books[key] = full_book
        gen_msgs[key] = np.vstack([cond_msg, gen_msg])
    if not gen_books:
        raise ValueError(f"No complete sample pairs found in {data_path}")
    return gen_books, gen_msgs, cond_lens


def load_aggressive_indices(data_path):
    aggr_file = data_path / 'aggressive_indices.csv'
    if not aggr_file.exists():
        return np.array([], dtype=int)
    indices = np.loadtxt(aggr_file, dtype=int)
    if indices.ndim == 0:
        indices = np.array([int(indices)])
    return indices


def compute_midprice(book_array):
    return (book_array[:, 0] + book_array[:, 2]) / 2


def parse_folder_params_v2(folder_name):
    match = re.match(r'i(\\d+)_c(\\d+)_mb(\\d+)_v(\\d+)_cntxt(.+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return None, None, None, None


def load_all_v2(grid_df):
    all_data = {}
    for _, row in tqdm(grid_df.iterrows(), total=len(grid_df), desc='Loading'):
        folder = row['folder']
        try:
            buy_books, buy_msgs, buy_cond = load_folder_data(
                row['buy_path'], max_samples=MAX_SAMPLES, max_midprice=MIDPRICE_MAX)
            sell_books, sell_msgs, sell_cond = load_folder_data(
                row['sell_path'], max_samples=MAX_SAMPLES, max_midprice=MIDPRICE_MAX)
            all_data[folder] = {
                'buy': {'books': buy_books, 'msgs': buy_msgs, 'cond_lens': buy_cond},
                'sell': {'books': sell_books, 'msgs': sell_msgs, 'cond_lens': sell_cond},
            }
        except Exception as e:
            print(f"ERROR: {folder} - {e}")
    return all_data


def compute_midprice_returns(books, min_len):
    returns = []
    for sid, book_array in books.items():
        midprice = compute_midprice(book_array[:min_len])
        returns.append(midprice - midprice[0])
    return np.stack(returns, axis=0)


def compute_combined_impact(buy_data, sell_data, folder_name):
    i_val, c_val, mb_val, V = parse_folder_params_v2(folder_name)
    buy_books = buy_data['books']
    sell_books = sell_data['books']
    min_len = min(
        min(b.shape[0] for b in buy_books.values()),
        min(b.shape[0] for b in sell_books.values()))
    buy_returns = compute_midprice_returns(buy_books, min_len)
    sell_returns = compute_midprice_returns(sell_books, min_len)
    combined = (buy_returns.mean(axis=0) - sell_returns.mean(axis=0)) / 2
    combined_std = np.sqrt(buy_returns.std(axis=0)**2 + sell_returns.std(axis=0)**2) / 2
    junction = list(buy_data['cond_lens'].values())[0]
    cooling_start = CONTEXT + i_val * mb_val if i_val else None
    return {
        'steps': np.arange(min_len), 'mean': combined, 'std': combined_std,
        'junction': junction, 'cooling_start': cooling_start,
        'iterations': i_val, 'coolings': c_val, 'metablock': mb_val, 'V': V,
        'n_buy': buy_returns.shape[0], 'n_sell': sell_returns.shape[0], 'min_len': min_len,
    }


def compute_beta_3modes(buy_data, sell_data, folder, buy_path, sell_path):
    \"\"\"Compute beta in 3 modes: @a (exact), :a (cumul<=a), a: (cumul>=a).\"\"\"
    eps = 1e-12
    aggr_buy = load_aggressive_indices(buy_path)
    aggr_sell = load_aggressive_indices(sell_path)
    if len(aggr_buy) == 0 and len(aggr_sell) == 0:
        return {}, {}, {}, None, None
    all_points = []
    for direction, side_data, aggr_indices_gen in [
        ('BUY', buy_data, aggr_buy), ('SELL', sell_data, aggr_sell),
    ]:
        if len(aggr_indices_gen) == 0:
            continue
        books = side_data['books']
        msgs = side_data['msgs']
        cond_lens = side_data['cond_lens']
        for sid in books:
            msg_arr = msgs[sid]
            book_arr = books[sid]
            junction = cond_lens[sid]
            sample_id = sid[1]
            day_row = SAMPLE_DAY_MAP[SAMPLE_DAY_MAP['sample_id'] == sample_id]
            if day_row.empty:
                continue
            H = float(day_row.iloc[0]['highest_price']) / TICK_SIZE
            L = float(day_row.iloc[0]['lowest_price']) / TICK_SIZE
            execution_sum = float(day_row.iloc[0]['execution_sum'])
            if H <= L or L <= 0:
                continue
            eta_day = np.log(H / L) / 0.8325546
            alpha_sample = np.log(max(eta_day, eps))
            aggr_indices = junction + aggr_indices_gen
            aggr_indices = aggr_indices[aggr_indices < len(msg_arr)]
            if len(aggr_indices) < 2:
                continue
            sizes = msg_arr[aggr_indices, 3].astype(float)
            prices = msg_arr[aggr_indices, 4].astype(float)
            first_idx = aggr_indices[0]
            ref_price = (book_arr[first_idx, 0] + book_arr[first_idx, 2]) / 2
            if ref_price <= 0:
                continue
            Q_cum = np.cumsum(sizes)
            notional_cum = np.cumsum(sizes * prices)
            vwap = notional_cum / np.maximum(Q_cum, eps)
            if direction == 'BUY':
                impact = (vwap - ref_price) / ref_price
            else:
                impact = (ref_price - vwap) / ref_price
            impact = np.abs(impact)
            for a_idx in range(len(aggr_indices)):
                a = a_idx + 1
                impact_a = impact[a_idx]
                Q_a = Q_cum[a_idx]
                if impact_a > eps and execution_sum > eps:
                    x = np.log(Q_a / execution_sum)
                    y = np.log(impact_a)
                    all_points.append((x, y, a, alpha_sample, sample_id, sid[0]))
    if not all_points:
        return {}, {}, {}, None, None
    all_X = np.array([p[0] for p in all_points])
    all_Y = np.array([p[1] for p in all_points])
    all_iters = np.array([p[2] for p in all_points])
    all_alphas = np.array([p[3] for p in all_points])
    def beta_for_mask(mask):
        X = all_X[mask]; alphas_m = all_alphas[mask]
        y_adj = all_Y[mask] - alphas_m
        valid = np.isfinite(X) & np.isfinite(y_adj) & (X != 0)
        if np.sum(valid) < 2: return None
        return float(np.dot(X[valid], y_adj[valid]) / np.dot(X[valid], X[valid]))
    beta_exact, beta_upto, beta_from = {}, {}, {}
    for it in sorted(set(all_iters)):
        b = beta_for_mask(all_iters == it)
        if b is not None: beta_exact[it] = b
        b = beta_for_mask(all_iters <= it)
        if b is not None: beta_upto[it] = b
        b = beta_for_mask(all_iters >= it)
        if b is not None: beta_from[it] = b
    final_beta_exact = beta_exact[max(beta_exact)] if beta_exact else None
    final_beta_upto = beta_upto[max(beta_upto)] if beta_upto else None
    return beta_exact, beta_upto, beta_from, final_beta_exact, final_beta_upto


def extract_global_point_cloud_v2(data, grid_df, buy_base, sell_base):
    \"\"\"Extract all (x, y) points from ALL V2 experiments.\"\"\"
    eps = 1e-12
    rows = []
    for _, grow in grid_df.iterrows():
        folder = grow['folder']
        if folder not in data:
            continue
        d = data[folder]
        i_val, c_val, mb_val, V = grow['i'], grow['c'], grow['mb'], grow['V']
        aggr_buy = load_aggressive_indices(grow['buy_path'])
        aggr_sell = load_aggressive_indices(grow['sell_path'])
        for direction, side_data, aggr_indices_gen in [
            ('BUY', d['buy'], aggr_buy), ('SELL', d['sell'], aggr_sell),
        ]:
            if len(aggr_indices_gen) == 0:
                continue
            books = side_data['books']
            msgs = side_data['msgs']
            cond_lens = side_data['cond_lens']
            for sid in books:
                msg_arr = msgs[sid]
                book_arr = books[sid]
                junction = cond_lens[sid]
                sample_id = sid[1]
                day_row = SAMPLE_DAY_MAP[SAMPLE_DAY_MAP['sample_id'] == sample_id]
                if day_row.empty:
                    continue
                H = float(day_row.iloc[0]['highest_price']) / TICK_SIZE
                L = float(day_row.iloc[0]['lowest_price']) / TICK_SIZE
                execution_sum = float(day_row.iloc[0]['execution_sum'])
                if H <= L or L <= 0:
                    continue
                eta_day = np.log(H / L) / 0.8325546
                alpha_sample = np.log(max(eta_day, eps))
                sigma_day = eta_day
                aggr_indices = junction + aggr_indices_gen
                aggr_indices = aggr_indices[aggr_indices < len(msg_arr)]
                if len(aggr_indices) < 2:
                    continue
                sizes = msg_arr[aggr_indices, 3].astype(float)
                prices = msg_arr[aggr_indices, 4].astype(float)
                first_idx = aggr_indices[0]
                ref_price = (book_arr[first_idx, 0] + book_arr[first_idx, 2]) / 2
                if ref_price <= 0:
                    continue
                Q_cum = np.cumsum(sizes)
                notional_cum = np.cumsum(sizes * prices)
                vwap = notional_cum / np.maximum(Q_cum, eps)
                if direction == 'BUY':
                    impact_arr = (vwap - ref_price) / ref_price
                else:
                    impact_arr = (ref_price - vwap) / ref_price
                impact_abs = np.abs(impact_arr)
                for a_idx in range(len(aggr_indices)):
                    impact_a = impact_abs[a_idx]
                    Q_a = Q_cum[a_idx]
                    if impact_a <= eps or execution_sum <= eps:
                        continue
                    rows.append({
                        'x': np.log(Q_a / execution_sum),
                        'y': np.log(impact_a),
                        'alpha': alpha_sample, 'sigma': sigma_day,
                        'Q_cum': Q_a, 'V_exp': execution_sum,
                        'impact': impact_a, 'iteration': a_idx + 1,
                        'folder': folder, 'mb': mb_val, 'i_val': i_val, 'V': V,
                        'direction': direction, 'sample_id': sample_id, 'date': sid[0],
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_global_beta(df):
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    valid = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    x_v, y_v = x[valid], y_adj[valid]
    if len(x_v) < 2:
        return {'beta_origin': np.nan, 'r2_origin': np.nan, 'beta_ols': np.nan,
                'intercept_ols': np.nan, 'r2_ols': np.nan, 'se_ols': np.nan, 'n_points': 0}
    beta_origin = float(np.dot(x_v, y_v) / np.dot(x_v, x_v))
    sl = linregress(x_v, y_v)
    y_pred = beta_origin * x_v
    ss_res = np.sum((y_v - y_pred)**2)
    ss_tot = np.sum(y_v**2)
    r2_origin = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'beta_origin': beta_origin, 'r2_origin': r2_origin,
            'beta_ols': sl.slope, 'intercept_ols': sl.intercept,
            'r2_ols': sl.rvalue**2, 'se_ols': sl.stderr, 'n_points': int(np.sum(valid))}


def compute_beta_for_subset(df):
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    valid = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    x_v, y_v = x[valid], y_adj[valid]
    n = int(np.sum(valid))
    if n < 2:
        return {'beta': np.nan, 'se': np.nan, 'r2': np.nan, 'n': n}
    beta = float(np.dot(x_v, y_v) / np.dot(x_v, x_v))
    y_pred = beta * x_v
    ss_res = np.sum((y_v - y_pred)**2)
    ss_tot = np.sum(y_v**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se = np.sqrt(ss_res / max(n-1, 1)) / np.sqrt(np.dot(x_v, x_v))
    return {'beta': beta, 'se': se, 'r2': r2, 'n': n}


def compute_volume_time_curves(buy_data, sell_data, folder, aggr_indices_gen,
                                u_max=3.0, n_u_points=300):
    i_val, c_val, mb_val, V = parse_folder_params_v2(folder)
    if len(aggr_indices_gen) < 2:
        return None
    s_gen = int(aggr_indices_gen[0])
    e_gen = int(aggr_indices_gen[-1])
    L = e_gen - s_gen
    if L == 0:
        return None
    buy_books = buy_data['books']
    sell_books = sell_data['books']
    min_len = min(
        min(b.shape[0] for b in buy_books.values()),
        min(b.shape[0] for b in sell_books.values()))
    junction_ex = list(buy_data['cond_lens'].values())[0]
    u_data_max = (min_len - 1 - junction_ex - s_gen) / L
    u_cap = min(u_max, u_data_max)
    u_grid = np.linspace(0, u_cap, n_u_points)
    def process_side(books, cond_lens):
        impacts = []
        for sid, book_arr in books.items():
            junction = cond_lens[sid]
            s_abs = junction + s_gen
            if s_abs < 1 or s_abs >= min_len:
                continue
            midprice = compute_midprice(book_arr[:min_len])
            p_ref = midprice[s_abs - 1]
            n_steps = min_len - s_abs
            impact_raw = midprice[s_abs:min_len] - p_ref
            u_raw = np.arange(n_steps) / L
            impact_interp = np.interp(u_grid, u_raw, impact_raw)
            impacts.append(impact_interp)
        return np.array(impacts) if impacts else None
    buy_impacts = process_side(buy_books, buy_data['cond_lens'])
    sell_impacts = process_side(sell_books, sell_data['cond_lens'])
    if buy_impacts is None or sell_impacts is None:
        return None
    combined_mean = (np.mean(buy_impacts, axis=0) - np.mean(sell_impacts, axis=0)) / 2
    combined_std = np.sqrt(np.std(buy_impacts, axis=0)**2 + np.std(sell_impacts, axis=0)**2) / 2
    return {
        'u_grid': u_grid, 'combined_mean': combined_mean, 'combined_std': combined_std,
        'L': L, 'i': i_val, 'c': c_val, 'mb': mb_val, 'V': V,
        'Q': i_val * V, 'gamma': c_val / i_val,
        'n_buy': buy_impacts.shape[0], 'n_sell': sell_impacts.shape[0],
    }


def compute_volume_time_curves_sigma_norm(buy_data, sell_data, folder, aggr_indices_gen,
                                           u_max=3.0, n_u_points=300):
    i_val, c_val, mb_val, V = parse_folder_params_v2(folder)
    if len(aggr_indices_gen) < 2:
        return None
    s_gen = int(aggr_indices_gen[0])
    e_gen = int(aggr_indices_gen[-1])
    L = e_gen - s_gen
    if L == 0:
        return None
    buy_books = buy_data['books']
    sell_books = sell_data['books']
    min_len = min(
        min(b.shape[0] for b in buy_books.values()),
        min(b.shape[0] for b in sell_books.values()))
    junction_ex = list(buy_data['cond_lens'].values())[0]
    u_data_max = (min_len - 1 - junction_ex - s_gen) / L
    u_cap = min(u_max, u_data_max)
    u_grid = np.linspace(0, u_cap, n_u_points)
    def process_side(books, cond_lens):
        impacts = []
        for sid, book_arr in books.items():
            sample_id = sid[1]
            day_row = SAMPLE_DAY_MAP[SAMPLE_DAY_MAP['sample_id'] == sample_id]
            if day_row.empty:
                continue
            H = float(day_row.iloc[0]['highest_price']) / TICK_SIZE
            L_price = float(day_row.iloc[0]['lowest_price']) / TICK_SIZE
            if H <= L_price or L_price <= 0:
                continue
            sigma_day = np.log(H / L_price) / 0.8325546
            if sigma_day <= 0:
                continue
            junction = cond_lens[sid]
            s_abs = junction + s_gen
            if s_abs < 1 or s_abs >= min_len:
                continue
            midprice = compute_midprice(book_arr[:min_len])
            p_ref = midprice[s_abs - 1]
            if p_ref <= 0:
                continue
            n_steps = min_len - s_abs
            impact_raw = (midprice[s_abs:min_len] - p_ref) / (p_ref * sigma_day)
            u_raw = np.arange(n_steps) / L
            impact_interp = np.interp(u_grid, u_raw, impact_raw)
            impacts.append(impact_interp)
        return np.array(impacts) if impacts else None
    buy_impacts = process_side(buy_books, buy_data['cond_lens'])
    sell_impacts = process_side(sell_books, sell_data['cond_lens'])
    if buy_impacts is None or sell_impacts is None:
        return None
    combined_mean = (np.mean(buy_impacts, axis=0) - np.mean(sell_impacts, axis=0)) / 2
    combined_std = np.sqrt(np.std(buy_impacts, axis=0)**2 + np.std(sell_impacts, axis=0)**2) / 2
    return {
        'u_grid': u_grid, 'combined_mean': combined_mean, 'combined_std': combined_std,
        'L': L, 'i': i_val, 'c': c_val, 'mb': mb_val, 'V': V,
        'Q': i_val * V, 'n_buy': buy_impacts.shape[0], 'n_sell': sell_impacts.shape[0],
    }


def style_decay_ratio(val):
    target = 2/3
    dev = abs(val - target)
    if dev < 0.05: return 'background-color: #2e7d32; color: white'
    if dev < 0.10: return 'background-color: #8bc34a'
    if dev < 0.15: return 'background-color: #fff176'
    if dev < 0.25: return 'background-color: #ff9800'
    return 'background-color: #d32f2f; color: white'


def quality_score(row):
    \"\"\"Combined quality: decay + beta + stability (if available).\"\"\"
    dr = row['decay_ratio']
    beta = row['beta_[:a]']
    # Decay quality (0-2)
    if 0.58 <= dr <= 0.82: d_score = 2
    elif 0.45 <= dr <= 0.90: d_score = 1
    else: d_score = 0
    # Beta quality (0-2)
    if pd.isna(beta): b_score = 0
    elif 0.45 <= beta <= 0.55: b_score = 2
    elif 0.40 <= beta <= 0.60: b_score = 1
    else: b_score = 0
    # Stability quality (0-2)
    votes = row.get('votes_num', np.nan)
    if pd.isna(votes): s_score = 0
    elif votes >= 3: s_score = 2
    elif votes >= 2: s_score = 1
    else: s_score = 0
    total = d_score + b_score + s_score  # max 6
    if total >= 5: return 'excellent'
    if total >= 4: return 'good'
    if total >= 2: return 'acceptable'
    return 'poor'


print("All functions defined.")"""))


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Per-Scenario Processing Loop
# ═══════════════════════════════════════════════════════════════════

cells.append(md("""---
# Section 1: Per-Scenario Data Processing

Process each scenario one at a time to manage memory (~28 GB per scenario).
After processing, keep only lightweight summaries."""))

cells.append(code("""def process_scenario(label, scn_cfg):
    \"\"\"Process one scenario end-to-end. Returns lightweight summaries only.\"\"\"
    buy_path = scn_cfg['buy_path']
    sell_path = scn_cfg['sell_path']

    # 1. Discover folders
    grid_df = discover_v2_folders(buy_path, sell_path)
    grid_df = grid_df.sort_values(['mb', 'i', 'V']).reset_index(drop=True)
    print(f"  {label}: {len(grid_df)} configs, V={sorted(grid_df['V'].unique())}")

    # 2. Load data
    data = load_all_v2(grid_df)
    print(f"  Loaded {len(data)}/{len(grid_df)} experiments")

    # 3. Compute impact curves + metrics
    impact_curves = {}
    results = []
    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in data:
            continue
        d = data[folder]
        stats = compute_combined_impact(d['buy'], d['sell'], folder)
        beta_exact, beta_upto, beta_from, final_beta_exact, final_beta_upto = compute_beta_3modes(
            d['buy'], d['sell'], folder, row['buy_path'], row['sell_path'])
        mean_curve = stats['mean']
        junction = stats['junction']
        post_junction = mean_curve[junction:]
        peak_idx = junction + np.argmax(post_junction)
        peak_val = mean_curve[peak_idx]
        final_val = mean_curve[-1]
        decay_ratio = final_val / peak_val if abs(peak_val) > 1e-12 else 0.0
        results.append({
            'folder': folder, 'i': row['i'], 'c': row['c'], 'mb': row['mb'], 'V': row['V'],
            'Q_total': row['Q_total'], 'total_gen': (row['i'] + row['c']) * row['mb'],
            'peak': peak_val, 'final': final_val,
            'decay_ratio': decay_ratio, 'decay_%': decay_ratio * 100,
            'beta_[:a]': final_beta_upto, 'beta_exact_final': final_beta_exact,
            'beta_exact': beta_exact, 'beta_upto': beta_upto, 'beta_from': beta_from,
            'n_buy': stats['n_buy'], 'n_sell': stats['n_sell'],
        })
        impact_curves[folder] = {
            'steps': stats['steps'], 'mean': stats['mean'].copy(), 'std': stats['std'].copy(),
            'junction': stats['junction'], 'cooling_start': stats['cooling_start'],
            'i': row['i'], 'mb': row['mb'], 'V': row['V'],
        }
    metrics_df = pd.DataFrame(results).sort_values(['mb', 'i', 'V']).reset_index(drop=True)
    # NOTE: quality is computed AFTER stability (step 7) so stability votes are available

    # 4. Point cloud
    pc = extract_global_point_cloud_v2(data, grid_df, buy_path, sell_path)
    pc_no20 = pc[pc['mb'] != 20] if len(pc) > 0 else pc
    beta_stats = compute_global_beta(pc_no20) if len(pc_no20) > 0 else {}

    # 5. Gamma (V-scaling)
    gamma_rows = []
    for (i_val, mb_val), grp in metrics_df.groupby(['i', 'mb']):
        grp_v = grp.sort_values('V')
        if len(grp_v) < 3:
            continue
        Vs = grp_v['V'].values.astype(float)
        peaks = grp_v['peak'].values
        if np.any(peaks <= 0) or np.any(Vs <= 0):
            continue
        sl = linregress(np.log(Vs), np.log(peaks))
        gamma_rows.append({
            'i': i_val, 'mb': mb_val,
            'gamma': sl.slope, 'A': np.exp(sl.intercept),
            'r2': sl.rvalue**2, 'se': sl.stderr,
        })
    gamma_df = pd.DataFrame(gamma_rows).sort_values(['mb', 'i']).reset_index(drop=True) if gamma_rows else pd.DataFrame()

    # 6. Volume-time curves
    all_curves = {}
    all_curves_norm = {}
    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in data:
            continue
        aggr = load_aggressive_indices(row['buy_path'])
        if len(aggr) < 2:
            continue
        curve = compute_volume_time_curves(
            data[folder]['buy'], data[folder]['sell'], folder, aggr, u_max=11.0, n_u_points=500)
        if curve:
            all_curves[folder] = curve
        curve_norm = compute_volume_time_curves_sigma_norm(
            data[folder]['buy'], data[folder]['sell'], folder, aggr, u_max=11.0, n_u_points=500)
        if curve_norm:
            all_curves_norm[folder] = curve_norm

    # 7. Stability analysis
    TAIL_FRAC, WINDOW_FRAC, SLOPE_THRESH, TWIN_THRESH, CONV_THRESH = 0.20, 0.15, 0.05, 0.03, 95.0
    stability_rows = []
    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in data:
            continue
        d = data[folder]
        stats = compute_combined_impact(d['buy'], d['sell'], folder)
        mean_curve = stats['mean']
        junction = stats['junction']
        post_junction = mean_curve[junction:]
        peak_idx_local = np.argmax(post_junction)
        peak_val = post_junction[peak_idx_local]
        post_peak = post_junction[peak_idx_local:]
        final_val = post_peak[-1]
        n = len(post_peak)
        srow = {'folder': folder, 'i': row['i'], 'mb': row['mb'], 'V': row['V'], 'n_post': n}
        if n < 3:
            srow.update({'norm_slope': np.nan, 'slope_stable': False, 'twin_diff': np.nan,
                         'twin_stable': False, 'tau': np.nan, 'conv_%': np.nan, 'exp_stable': False,
                         'votes': '0/3', 'votes_num': 0, 'stabilized': False})
            stability_rows.append(srow)
            continue
        # Method 1: Trailing slope
        tail_len = max(int(n * TAIL_FRAC), 3)
        tail_len = min(tail_len, n)  # guard: can't exceed post_peak length
        tail = post_peak[-tail_len:]
        sl = linregress(np.arange(tail_len, dtype=float), tail)
        norm_slope = sl.slope * tail_len / peak_val if abs(peak_val) > 1e-12 else 0.0
        srow['norm_slope'] = norm_slope
        srow['slope_stable'] = abs(norm_slope) < SLOPE_THRESH
        # Method 2: Two-window
        w = max(int(n * WINDOW_FRAC), 3)
        if 2 * w <= n:
            twin_diff = (np.mean(post_peak[-w:]) - np.mean(post_peak[-2*w:-w])) / peak_val if abs(peak_val) > 1e-12 else 0.0
        else:
            twin_diff = np.nan
        srow['twin_diff'] = twin_diff
        srow['twin_stable'] = abs(twin_diff) < TWIN_THRESH if not np.isnan(twin_diff) else False
        # Method 3: Exponential fit
        t = np.arange(n, dtype=float)
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C
        try:
            popt, _ = curve_fit(exp_decay, t, post_peak,
                                p0=[float(peak_val - final_val), n/3.0, float(final_val)], maxfev=10000)
            tau_fit = popt[1]
            conv_pct = (1.0 - np.exp(-n / tau_fit)) * 100 if tau_fit > 0 else 100.0
        except Exception:
            tau_fit, conv_pct = np.nan, np.nan
        srow['tau'] = tau_fit
        srow['conv_%'] = conv_pct
        srow['exp_stable'] = conv_pct > CONV_THRESH if not np.isnan(conv_pct) else False
        votes = sum([srow['slope_stable'], srow['twin_stable'], srow['exp_stable']])
        srow['votes'] = f"{votes}/3"
        srow['votes_num'] = votes
        srow['stabilized'] = votes >= 2
        stability_rows.append(srow)
    stab_df = pd.DataFrame(stability_rows).sort_values(['mb','i','V']).reset_index(drop=True) if stability_rows else pd.DataFrame()

    # 7b. Compute quality (AFTER stability so votes_num is available)
    if len(stab_df) > 0:
        metrics_df = metrics_df.merge(stab_df[['folder', 'votes_num', 'stabilized']], on='folder', how='left')
        metrics_df['votes_num'] = metrics_df['votes_num'].fillna(0)
        metrics_df['stabilized'] = metrics_df['stabilized'].fillna(False)
    else:
        metrics_df['votes_num'] = 0
        metrics_df['stabilized'] = False
    metrics_df['quality'] = metrics_df.apply(quality_score, axis=1)
    metrics_df['quality_num'] = metrics_df['quality'].map(QUALITY_MAP)

    # 8. Relaxation ratios from volume-time
    relax_rows = []
    for folder, c in all_curves.items():
        u = c['u_grid']
        mean = c['combined_mean']
        I_peak = float(np.interp(1.0, u, mean))
        if abs(I_peak) < 1e-12:
            continue
        I_u2 = float(np.interp(2.0, u, mean)) if u[-1] >= 2.0 else np.nan
        I_final = float(mean[-1])
        relax_rows.append({
            'folder': folder, 'Q': c['Q'], 'mb': c['mb'], 'V': c['V'], 'L': c['L'],
            'I_peak': I_peak, 'I_u2': I_u2, 'I_final': I_final,
            'ratio_u2': I_u2/I_peak if not np.isnan(I_u2) else np.nan,
            'ratio_final': I_final/I_peak,
        })
    relax_df = pd.DataFrame(relax_rows) if relax_rows else pd.DataFrame()

    # 9. Bootstrap CI
    ci_lo, ci_hi, median_beta = np.nan, np.nan, np.nan
    boot_betas = np.array([])
    if len(pc_no20) > 0:
        N_BOOTSTRAP = 1000
        rng = np.random.RandomState(42)
        pc_boot = pc_no20[['x','y','alpha','sample_id']].copy()
        pc_boot['y_adj'] = pc_boot['y'] - pc_boot['alpha']
        valid_mask = np.isfinite(pc_boot['x']) & np.isfinite(pc_boot['y_adj']) & (pc_boot['x'] != 0)
        pc_boot = pc_boot[valid_mask].copy()
        sample_groups = {sid: grp[['x','y_adj']].values for sid, grp in pc_boot.groupby('sample_id')}
        sample_ids_arr = np.array(list(sample_groups.keys()))
        n_s = len(sample_ids_arr)
        if n_s > 0:
            boot_betas = np.zeros(N_BOOTSTRAP)
            for b in range(N_BOOTSTRAP):
                chosen = rng.choice(sample_ids_arr, size=n_s, replace=True)
                pooled = np.vstack([sample_groups[sid] for sid in chosen])
                x_b, y_b = pooled[:,0], pooled[:,1]
                boot_betas[b] = np.dot(x_b, y_b) / np.dot(x_b, x_b)
            ci_lo, ci_hi = np.percentile(boot_betas, [2.5, 97.5])
            median_beta = np.median(boot_betas)

    # 10. CV metric for master curve collapse
    cv_data = {}
    u_eval = [0.5, 1.0, 1.5, 2.0, 2.5]
    for u_pt in u_eval:
        vals = []
        for folder in all_curves_norm:
            c = all_curves_norm[folder]
            if u_pt <= c['u_grid'][-1]:
                vals.append(float(np.interp(u_pt, c['u_grid'], c['combined_mean'])))
        if len(vals) > 1:
            cv_data[u_pt] = {'mean': np.mean(vals), 'std': np.std(vals),
                             'cv': np.std(vals) / abs(np.mean(vals)) if abs(np.mean(vals)) > 1e-12 else np.nan}

    # Clean up raw data
    del data
    gc.collect()

    return {
        'grid_df': grid_df, 'metrics_df': metrics_df, 'pc': pc, 'pc_no20': pc_no20,
        'beta_stats': beta_stats, 'gamma_df': gamma_df,
        'impact_curves': impact_curves, 'all_curves': all_curves,
        'all_curves_norm': all_curves_norm, 'stab_df': stab_df, 'relax_df': relax_df,
        'boot_betas': boot_betas, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'median_beta': median_beta,
        'cv_data': cv_data,
    }

print("process_scenario() defined.")"""))

cells.append(code("""# ── Main processing loop ──
all_results = OrderedDict()

for label, scn_cfg in ACTIVE_SCENARIOS.items():
    print(f"\\n{'='*80}")
    print(f"Processing: {label}")
    print(f"{'='*80}")
    result = process_scenario(label, scn_cfg)
    all_results[label] = result
    print(f"  -> {len(result['metrics_df'])} metrics, {len(result['pc']):,} points, "
          f"{len(result['gamma_df'])} gamma rows")

# Combine lightweight DataFrames with scenario column
combined_metrics = pd.concat([
    r['metrics_df'].assign(scenario=label)
    for label, r in all_results.items()
], ignore_index=True)

combined_stab = pd.concat([
    r['stab_df'].assign(scenario=label)
    for label, r in all_results.items() if len(r['stab_df']) > 0
], ignore_index=True)

combined_gamma = pd.concat([
    r['gamma_df'].assign(scenario=label)
    for label, r in all_results.items() if len(r['gamma_df']) > 0
], ignore_index=True)

combined_relax = pd.concat([
    r['relax_df'].assign(scenario=label)
    for label, r in all_results.items() if len(r['relax_df']) > 0
], ignore_index=True)

print(f"\\n{'='*80}")
print(f"Combined: {len(combined_metrics)} metrics, {len(combined_stab)} stability, "
      f"{len(combined_gamma)} gamma, {len(combined_relax)} relaxation")
print(f"Active scenarios: {list(all_results.keys())}")"""))


# ═══════════════════════════════════════════════════════════════════
# PART A: Per-Scenario Full Analysis
# ═══════════════════════════════════════════════════════════════════

cells.append(md("""---
# PART A: Per-Scenario Full Analysis

For each active scenario, we reproduce the full 110-style analysis."""))

# Section 2: Decay Curves (per scenario)
cells.append(md("""## Section 2: Decay Curves"""))

cells.append(code("""# ── Per-scenario decay curves: faceted grid + overlay ──
for label, result in all_results.items():
    scn_color = ACTIVE_SCENARIOS[label]['color']
    grid_df = result['grid_df']
    impact_curves = result['impact_curves']
    metrics_df = result['metrics_df']

    V_levels = sorted(grid_df['V'].unique())
    imb_pairs = sorted(grid_df.groupby(['i','mb']).first().index.tolist())
    n_cols = len(imb_pairs)
    n_rows = len(V_levels)

    display(Markdown(f"### {label}: Faceted Decay Curves"))

    # Faceted grid
    fig = make_subplots(rows=n_rows, cols=n_cols,
        subplot_titles=[f'i={i},mb={mb}' for i,mb in imb_pairs] * n_rows,
        vertical_spacing=0.06, horizontal_spacing=0.03)

    for r_idx, V in enumerate(V_levels):
        for c_idx, (i_val, mb_val) in enumerate(imb_pairs):
            row_match = grid_df[(grid_df['i']==i_val)&(grid_df['mb']==mb_val)&(grid_df['V']==V)]
            if row_match.empty:
                continue
            folder = row_match.iloc[0]['folder']
            if folder not in impact_curves:
                continue
            ic = impact_curves[folder]
            fig.add_trace(go.Scatter(
                x=ic['steps'], y=ic['mean'], mode='lines',
                line=dict(color=VOLUME_COLORS[V], width=1.5), showlegend=False,
            ), row=r_idx+1, col=c_idx+1)
            fig.add_vline(x=ic['junction'], line_dash='dash', line_color='gray',
                          line_width=1, row=r_idx+1, col=c_idx+1)
            if c_idx == 0:
                fig.update_yaxes(title_text=f'V={V}', row=r_idx+1, col=1)

    fig.update_layout(title_text=f'<b>{label}: Decay Curves</b>', width=2400,
                      height=250*n_rows, template='plotly_white')
    fig.show()

    # Overlay view
    display(Markdown(f"### {label}: Overlay by V"))
    ov_cols = 5
    ov_rows = (len(imb_pairs) + ov_cols - 1) // ov_cols
    fig = make_subplots(rows=ov_rows, cols=ov_cols,
        subplot_titles=[f'i={i},mb={mb}' for i,mb in imb_pairs],
        vertical_spacing=0.08, horizontal_spacing=0.05)
    for idx, (i_val, mb_val) in enumerate(imb_pairs):
        r = idx // ov_cols + 1
        c = idx % ov_cols + 1
        for V in V_levels:
            row_match = grid_df[(grid_df['i']==i_val)&(grid_df['mb']==mb_val)&(grid_df['V']==V)]
            if row_match.empty:
                continue
            folder = row_match.iloc[0]['folder']
            if folder not in impact_curves:
                continue
            ic = impact_curves[folder]
            fig.add_trace(go.Scatter(
                x=ic['steps'], y=ic['mean'], mode='lines',
                line=dict(color=VOLUME_COLORS[V], width=2),
                name=f'V={V}', legendgroup=f'V{V}', showlegend=(idx==0),
            ), row=r, col=c)
        fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1, row=r, col=c)
    fig.update_layout(title_text=f'<b>{label}: Overlay by V</b>', width=1600,
                      height=350*ov_rows, template='plotly_white')
    fig.show()

    # Decay ratio pivot
    display(Markdown(f"### {label}: Decay Ratio Table"))
    pivot_decay = metrics_df.pivot_table(index=['i','mb'], columns='V',
                                          values='decay_ratio', aggfunc='first').sort_index()
    styled_decay = pivot_decay.style.map(style_decay_ratio).format('{:.3f}')
    display(styled_decay)"""))

# Section 3: Beta Analysis (per scenario)
cells.append(md("""## Section 3: Square-Root Law (Beta)"""))

cells.append(code("""# ── Per-scenario beta analysis ──
for label, result in all_results.items():
    pc = result['pc']
    pc_no20 = result['pc_no20']
    beta_stats = result['beta_stats']
    metrics_df = result['metrics_df']
    grid_df = result['grid_df']
    boot_betas = result['boot_betas']
    ci_lo, ci_hi = result['ci_lo'], result['ci_hi']
    median_beta = result['median_beta']

    display(Markdown(f"### {label}: Global Beta"))
    if beta_stats:
        print(f"  beta_origin = {beta_stats.get('beta_origin', 'N/A'):.4f}, "
              f"R2 = {beta_stats.get('r2_origin', 'N/A'):.4f}, "
              f"n = {beta_stats.get('n_points', 0):,}")

    if len(pc_no20) == 0:
        print(f"  No point cloud data for {label}")
        continue

    # Scatter plot
    PLOT_MAX = 20_000
    pc_plot = pc_no20.sample(n=min(PLOT_MAX, len(pc_no20)), random_state=42)
    beta_o = beta_stats.get('beta_origin', 0.5)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Raw: log(Impact) vs log(Q/V)',
                        f'Adjusted: (y-alpha) vs x  [beta={beta_o:.4f}]'],
        horizontal_spacing=0.08)
    for V in sorted(pc_plot['V'].unique()):
        sub = pc_plot[pc_plot['V'] == V]
        color = VOLUME_COLORS.get(V, 'gray')
        fig.add_trace(go.Scattergl(x=sub['x'], y=sub['y'], mode='markers',
            marker=dict(size=2, color=color, opacity=0.3),
            name=f'V={V}', legendgroup=f'V{V}'), row=1, col=1)
        fig.add_trace(go.Scattergl(x=sub['x'], y=sub['y']-sub['alpha'], mode='markers',
            marker=dict(size=2, color=color, opacity=0.3),
            name=f'V={V}', legendgroup=f'V{V}', showlegend=False), row=1, col=2)
    x_range = np.array([pc_plot['x'].min(), pc_plot['x'].max()])
    fig.add_trace(go.Scatter(x=x_range, y=beta_o*x_range, mode='lines',
        line=dict(color='black', width=3), name=f'Fit: {beta_o:.3f}'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_range, y=0.5*x_range, mode='lines',
        line=dict(color='red', width=2, dash='dash'), name='Theory: 0.5'), row=1, col=2)
    fig.update_layout(title_text=f'<b>{label}: Square-Root Law</b>', width=1500,
                      height=550, template='plotly_white')
    fig.show()

    # Beta by V
    v_rows = []
    for V in sorted(pc_no20['V'].unique()):
        sub = pc_no20[pc_no20['V'] == V]
        s = compute_beta_for_subset(sub)
        s['V'] = V
        v_rows.append(s)
    v_df = pd.DataFrame(v_rows)
    fig = go.Figure(go.Bar(
        x=[f'V={V}' for V in v_df['V']], y=v_df['beta'],
        error_y=dict(type='data', array=v_df['se']*1.96, visible=True),
        marker_color=[VOLUME_COLORS[V] for V in v_df['V']]))
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2)
    fig.add_hrect(y0=0.4, y1=0.6, fillcolor='rgba(0,200,0,0.07)', line_width=0)
    fig.update_layout(title_text=f'<b>{label}: Beta by V</b>', width=600, height=400,
                      template='plotly_white', yaxis_title='Beta')
    fig.show()

    # Beta by mb
    mb_rows = []
    for mb in sorted(pc['mb'].unique()):
        sub = pc[pc['mb'] == mb]
        s = compute_beta_for_subset(sub)
        s['mb'] = mb
        mb_rows.append(s)
    mb_df = pd.DataFrame(mb_rows)
    fig = go.Figure(go.Bar(
        x=[f'mb={mb}' for mb in mb_df['mb']], y=mb_df['beta'],
        error_y=dict(type='data', array=mb_df['se']*1.96, visible=True),
        marker_color=[MB_COLORS.get(mb, ('#999','rgba(0,0,0,0.1)'))[0] for mb in mb_df['mb']]))
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2)
    fig.update_layout(title_text=f'<b>{label}: Beta by mb</b>', width=600, height=400,
                      template='plotly_white', yaxis_title='Beta')
    fig.show()

    # Bootstrap CI
    if len(boot_betas) > 0:
        fig = go.Figure(go.Histogram(x=boot_betas, nbinsx=50, marker_color='steelblue', opacity=0.7))
        fig.add_vline(x=0.5, line_dash='dash', line_color='red', line_width=3)
        fig.add_vline(x=median_beta, line_dash='solid', line_color='black', line_width=2)
        fig.update_layout(title_text=f'<b>{label}: Bootstrap CI</b> [{ci_lo:.4f}, {ci_hi:.4f}]',
                          width=900, height=400, template='plotly_white')
        fig.show()

    # Per-day consistency
    day_rows = []
    for date in sorted(pc_no20['date'].unique()):
        sub = pc_no20[pc_no20['date'] == date]
        s = compute_beta_for_subset(sub)
        s['date'] = date
        day_rows.append(s)
    day_df = pd.DataFrame(day_rows)
    fig = go.Figure(go.Bar(x=day_df['date'], y=day_df['beta'],
        error_y=dict(type='data', array=day_df['se']*1.96, visible=True),
        marker_color='steelblue'))
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2)
    fig.update_layout(title_text=f'<b>{label}: Per-Day Beta</b>', width=1000, height=450,
                      template='plotly_white', yaxis_title='Beta')
    fig.show()

    # Beta 3 modes
    beta_by_mode = {'@a': defaultdict(list), '[:a]': defaultdict(list), '[a:]': defaultdict(list)}
    for _, row in metrics_df[metrics_df['mb'] != 20].iterrows():
        if row['i'] <= 1:
            continue
        for a, b in row['beta_exact'].items():
            beta_by_mode['@a'][a].append(b)
        for a, b in row['beta_upto'].items():
            beta_by_mode['[:a]'][a].append(b)
        for a, b in row['beta_from'].items():
            beta_by_mode['[a:]'][a].append(b)
    mode_colors = {'@a': '#e41a1c', '[:a]': '#377eb8', '[a:]': '#4daf4a'}
    fig = go.Figure()
    for mode_name in ['@a', '[:a]', '[a:]']:
        iters_sorted = sorted(beta_by_mode[mode_name].keys())
        if not iters_sorted:
            continue
        means = [np.mean(beta_by_mode[mode_name][a]) for a in iters_sorted]
        stds = [np.std(beta_by_mode[mode_name][a]) for a in iters_sorted]
        fig.add_trace(go.Scatter(x=iters_sorted, y=means, mode='lines+markers',
            error_y=dict(type='data', array=stds, visible=True),
            line=dict(color=mode_colors[mode_name], width=2),
            marker=dict(size=6), name=mode_name))
    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', line_width=1)
    fig.update_layout(title_text=f'<b>{label}: Beta 3 Modes</b>', width=900, height=450,
                      template='plotly_white', xaxis_title='Iteration a', yaxis_title='Beta')
    fig.show()"""))

# Section 4: Volume Scaling (per scenario)
cells.append(md("""## Section 4: Volume Scaling (Gamma)"""))

cells.append(code("""# ── Per-scenario gamma analysis ──
for label, result in all_results.items():
    gamma_df = result['gamma_df']
    metrics_df = result['metrics_df']
    grid_df = result['grid_df']

    if len(gamma_df) == 0:
        display(Markdown(f"### {label}: No gamma data"))
        continue

    display(Markdown(f"### {label}: V-Scaling"))
    print(gamma_df[['i','mb','gamma','A','r2','se']].to_string(index=False))
    print(f"Mean gamma (excl mb=20): {gamma_df[gamma_df['mb']!=20]['gamma'].mean():.3f}")

    imb_pairs = sorted(grid_df.groupby(['i','mb']).first().index.tolist())
    ov_cols = 5
    ov_rows = (len(imb_pairs) + ov_cols - 1) // ov_cols
    fig = make_subplots(rows=ov_rows, cols=ov_cols,
        subplot_titles=[f'i={i},mb={mb}' for i,mb in imb_pairs],
        vertical_spacing=0.08, horizontal_spacing=0.05)
    for idx, (i_val, mb_val) in enumerate(imb_pairs):
        r = idx // ov_cols + 1
        c = idx % ov_cols + 1
        grp = metrics_df[(metrics_df['i']==i_val) & (metrics_df['mb']==mb_val)].sort_values('V')
        if grp.empty or np.any(grp['peak'].values <= 0):
            continue
        fig.add_trace(go.Scatter(
            x=np.log(grp['V'].values.astype(float)), y=np.log(grp['peak'].values),
            mode='markers+lines', marker=dict(size=8, color='steelblue'),
            showlegend=False), row=r, col=c)
        gm = gamma_df[(gamma_df['i']==i_val)&(gamma_df['mb']==mb_val)]
        if not gm.empty:
            gam = gm.iloc[0]['gamma']
            A = gm.iloc[0]['A']
            v_fit = np.linspace(np.log(70), np.log(500), 50)
            fig.add_trace(go.Scatter(x=v_fit, y=np.log(A) + gam * v_fit,
                mode='lines', line=dict(color='black', width=1, dash='dash'),
                showlegend=False), row=r, col=c)
            fig.add_trace(go.Scatter(x=v_fit, y=np.log(A) + 0.5 * v_fit,
                mode='lines', line=dict(color='red', width=1, dash='dot'),
                showlegend=False), row=r, col=c)
    fig.update_layout(title_text=f'<b>{label}: V-Scaling</b>', width=1600,
                      height=350*ov_rows, template='plotly_white')
    fig.show()"""))

# Section 5: Quality Heatmaps (per scenario)
cells.append(md("""## Section 5: Quality Heatmaps"""))

cells.append(code("""# ── Per-scenario quality heatmaps ──
for label, result in all_results.items():
    metrics_df = result['metrics_df'].copy()
    metrics_df['label'] = metrics_df.apply(lambda r: f"i{r['i']}_mb{r['mb']}", axis=1)
    label_order = metrics_df.groupby('label')['Q_total'].first().sort_values().index.tolist()

    display(Markdown(f"### {label}: Quality Heatmaps"))

    # Decay ratio heatmap
    pivot_dr = metrics_df.pivot_table(index='label', columns='V',
                                       values='decay_ratio', aggfunc='first').reindex(label_order)
    fig = go.Figure(go.Heatmap(
        z=pivot_dr.values, x=[str(v) for v in pivot_dr.columns], y=pivot_dr.index.tolist(),
        colorscale=[[0,'#d32f2f'],[0.417,'#ff9800'],[0.517,'#fff176'],
                     [0.617,'#8bc34a'],[0.667,'#2e7d32'],[0.717,'#8bc34a'],
                     [0.817,'#fff176'],[0.917,'#ff9800'],[1,'#d32f2f']],
        zmin=0, zmax=1,
        text=np.round(pivot_dr.values, 3).astype(str), texttemplate='%{text}',
        colorbar=dict(title='decay_ratio')))
    fig.update_layout(title_text=f'<b>{label}: Decay Ratio</b>', width=600, height=600,
                      template='plotly_white', xaxis_title='V', yaxis_title='(i, mb)')
    fig.show()

    # Beta heatmap
    pivot_beta = metrics_df.pivot_table(index='label', columns='V',
                                         values='beta_[:a]', aggfunc='first').reindex(label_order)
    fig = go.Figure(go.Heatmap(
        z=pivot_beta.values, x=[str(v) for v in pivot_beta.columns], y=pivot_beta.index.tolist(),
        colorscale=[[0,'red'],[0.3,'yellow'],[0.45,'green'],[0.55,'green'],[0.7,'yellow'],[1,'red']],
        zmin=0.3, zmax=0.7,
        text=np.where(np.isnan(pivot_beta.values), 'N/A', np.round(pivot_beta.values, 3).astype(str)),
        texttemplate='%{text}', colorbar=dict(title='beta[:a]')))
    fig.update_layout(title_text=f'<b>{label}: Beta</b>', width=600, height=600,
                      template='plotly_white', xaxis_title='V', yaxis_title='(i, mb)')
    fig.show()

    # Combined quality
    pivot_q = metrics_df.pivot_table(index='label', columns='V',
                                      values='quality_num', aggfunc='first').reindex(label_order)
    pivot_ql = metrics_df.pivot_table(index='label', columns='V',
                                       values='quality', aggfunc='first').reindex(label_order)
    fig = go.Figure(go.Heatmap(
        z=pivot_q.values, x=[str(v) for v in pivot_q.columns], y=pivot_q.index.tolist(),
        colorscale=[[0,'#d32f2f'],[0.33,'#ff9800'],[0.66,'#8bc34a'],[1,'#2e7d32']],
        zmin=1, zmax=4,
        text=pivot_ql.values, texttemplate='%{text}',
        colorbar=dict(title='Quality', tickvals=[1,2,3,4],
                      ticktext=['poor','acceptable','good','excellent'])))
    fig.update_layout(title_text=f'<b>{label}: Combined Quality</b>', width=600, height=600,
                      template='plotly_white', xaxis_title='V', yaxis_title='(i, mb)')
    fig.show()
    print(f"Quality distribution: {metrics_df['quality'].value_counts().to_dict()}")"""))

# Section 6: Volume-Time (per scenario)
cells.append(md("""## Section 6: Volume-Time Dynamics"""))

cells.append(code("""# ── Per-scenario volume-time analysis ──
for label, result in all_results.items():
    all_curves = result['all_curves']
    all_curves_norm = result['all_curves_norm']
    relax_df = result['relax_df']

    display(Markdown(f"### {label}: Volume-Time"))

    if not all_curves:
        print(f"  No volume-time data for {label}")
        continue

    # Master curve collapse
    U_SHOW = 3.0
    sorted_folders = sorted(all_curves_norm.keys(),
        key=lambda f: (parse_folder_params_v2(f)[2], parse_folder_params_v2(f)[0]))
    colors_all = px.colors.qualitative.D3 + px.colors.qualitative.Set2

    fig = go.Figure()
    for idx, folder in enumerate(sorted_folders):
        c = all_curves_norm[folder]
        u = c['u_grid']
        mask = u <= U_SHOW
        color = colors_all[idx % len(colors_all)]
        i_val, _, mb_val, V = parse_folder_params_v2(folder)
        fig.add_trace(go.Scatter(x=u[mask], y=c['combined_mean'][mask], mode='lines',
            line=dict(color=color, width=1.5), name=f'i{i_val}mb{mb_val}V{V}'))
    fig.add_vline(x=1.0, line_dash='dash', line_color='gray', line_width=1.5)
    fig.update_layout(title_text=f'<b>{label}: Master Curve Collapse</b>', width=1200,
                      height=500, template='plotly_white',
                      xaxis_title='u', yaxis_title='I_norm(u)')
    fig.show()

    # Relaxation ratio
    if len(relax_df) > 0:
        fig = go.Figure()
        for V in sorted(relax_df['V'].unique()):
            sub = relax_df[relax_df['V']==V]
            fig.add_trace(go.Scatter(x=sub['Q'], y=sub['ratio_final'], mode='markers',
                marker=dict(size=8, color=VOLUME_COLORS.get(V, 'gray')), name=f'V={V}'))
        fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2)
        fig.update_layout(title_text=f'<b>{label}: Relaxation Ratio</b>', width=900,
                          height=450, template='plotly_white',
                          xaxis_title='Q', yaxis_title='I_final / I_peak')
        fig.show()

    # CV
    cv_data = result['cv_data']
    if cv_data:
        print(f"  CV at key u-points:")
        for u_pt, vals in cv_data.items():
            print(f"    u={u_pt}: mean={vals['mean']:.4f}, CV={vals['cv']:.3f}")"""))

# Section 7: Stability (per scenario)
cells.append(md("""## Section 7: Stability"""))

cells.append(code("""# ── Per-scenario stability ──
for label, result in all_results.items():
    stab_df = result['stab_df'].copy()
    metrics_df = result['metrics_df'].copy()

    display(Markdown(f"### {label}: Stability"))

    if len(stab_df) == 0:
        print(f"  No stability data for {label}")
        continue

    stab_df['label'] = stab_df.apply(lambda r: f"i{r['i']}_mb{r['mb']}", axis=1)
    label_order = stab_df.groupby('label').first().sort_index().index.tolist()

    pivot_stab = stab_df.pivot_table(index='label', columns='V',
                                      values='votes_num', aggfunc='first').reindex(label_order)
    fig = go.Figure(go.Heatmap(
        z=pivot_stab.values, x=[str(v) for v in pivot_stab.columns], y=pivot_stab.index.tolist(),
        colorscale=[[0,'#d32f2f'],[0.33,'#ff9800'],[0.66,'#8bc34a'],[1,'#2e7d32']],
        zmin=0, zmax=3,
        text=pivot_stab.values.astype(str) + '/3', texttemplate='%{text}',
        colorbar=dict(title='Votes /3')))
    fig.update_layout(title_text=f'<b>{label}: Stability</b>', width=600, height=600,
                      template='plotly_white', xaxis_title='V', yaxis_title='(i, mb)')
    fig.show()

    stable_frac = stab_df['stabilized'].mean()
    print(f"  Fraction stable (2+/3): {stable_frac:.1%}")"""))


# ═══════════════════════════════════════════════════════════════════
# PART B: Cross-Scenario Comparison
# ═══════════════════════════════════════════════════════════════════

cells.append(md("""---
# PART B: Cross-Scenario Comparison"""))

# Section 8: Cross-scenario decay
cells.append(md("""## Section 8: Cross-Scenario Decay"""))

cells.append(code("""# ── Side-by-side decay ratio heatmaps ──
n_scn = len(all_results)
fig = make_subplots(rows=1, cols=n_scn,
    subplot_titles=list(all_results.keys()),
    horizontal_spacing=0.06)

for col_idx, (label, result) in enumerate(all_results.items(), 1):
    mdf = result['metrics_df'].copy()
    mdf['label'] = mdf.apply(lambda r: f"i{r['i']}_mb{r['mb']}", axis=1)
    label_order = mdf.groupby('label')['Q_total'].first().sort_values().index.tolist()
    pivot = mdf.pivot_table(index='label', columns='V', values='decay_ratio',
                             aggfunc='first').reindex(label_order)
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=[str(v) for v in pivot.columns], y=pivot.index.tolist(),
        colorscale=[[0,'#d32f2f'],[0.417,'#ff9800'],[0.517,'#fff176'],
                     [0.617,'#8bc34a'],[0.667,'#2e7d32'],[0.717,'#8bc34a'],
                     [0.817,'#fff176'],[0.917,'#ff9800'],[1,'#d32f2f']],
        zmin=0, zmax=1, showscale=(col_idx==n_scn),
        text=np.round(pivot.values, 2).astype(str), texttemplate='%{text}',
    ), row=1, col=col_idx)

fig.update_layout(title_text='<b>Cross-Scenario: Decay Ratio Heatmaps</b>',
                  width=500*n_scn, height=600, template='plotly_white')
fig.show()"""))

cells.append(code("""# ── Decay ratio distribution by scenario ──
fig = go.Figure()
for label in all_results:
    mdf = all_results[label]['metrics_df']
    fig.add_trace(go.Box(y=mdf['decay_ratio'], name=label,
        marker_color=ACTIVE_SCENARIOS[label]['color'],
        boxpoints='all', jitter=0.3, pointpos=-1.5))
fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2,
              annotation_text='Bouchaud 2/3')
fig.update_layout(title_text='<b>Cross-Scenario: Decay Ratio Distribution</b>',
                  width=900, height=500, template='plotly_white', yaxis_title='Decay Ratio')
fig.show()"""))

cells.append(code("""# ── Decay ratio by mb-group: grouped bar chart ──
# Reveals whether some scenarios look good only at low mb (easy configs)
mb_levels = sorted(combined_metrics['mb'].unique())
fig = go.Figure()
for label in all_results:
    mdf = all_results[label]['metrics_df']
    means, stds = [], []
    for mb in mb_levels:
        sub = mdf[mdf['mb'] == mb]
        means.append(sub['decay_ratio'].mean())
        stds.append(sub['decay_ratio'].std())
    fig.add_trace(go.Bar(
        x=[f'mb={mb}' for mb in mb_levels], y=means,
        error_y=dict(type='data', array=stds, visible=True),
        name=label, marker_color=ACTIVE_SCENARIOS[label]['color']))
fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2,
              annotation_text='Bouchaud 2/3')
fig.update_layout(title_text='<b>Cross-Scenario: Decay Ratio by mb Group</b>',
                  barmode='group', width=1000, height=500, template='plotly_white',
                  yaxis_title='Mean Decay Ratio')
fig.show()

# ── Decay ratio by mb: box plot with individual points ──
fig = go.Figure()
for label in all_results:
    mdf = all_results[label]['metrics_df']
    fig.add_trace(go.Box(
        x=mdf['mb'].astype(str), y=mdf['decay_ratio'], name=label,
        marker_color=ACTIVE_SCENARIOS[label]['color'],
        boxpoints='all', jitter=0.3, pointpos=-1.5,
        offsetgroup=label))
fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2)
fig.update_layout(
    title_text='<b>Cross-Scenario: Decay Ratio Distribution by mb</b>',
    boxmode='group', width=1200, height=500, template='plotly_white',
    xaxis_title='mb (messages between insertions)', yaxis_title='Decay Ratio')
fig.show()"""))

cells.append(code("""# ── Decay ratio by mb: line plot showing trend ──
# If a scenario degrades with increasing mb, this reveals the trend
fig = go.Figure()
for label in all_results:
    mdf = all_results[label]['metrics_df']
    grp = mdf.groupby('mb')['decay_ratio'].agg(['mean','std']).sort_index()
    fig.add_trace(go.Scatter(
        x=grp.index, y=grp['mean'], mode='lines+markers',
        error_y=dict(type='data', array=grp['std'], visible=True),
        line=dict(color=ACTIVE_SCENARIOS[label]['color'], width=2,
                  dash=ACTIVE_SCENARIOS[label]['dash']),
        marker=dict(symbol=ACTIVE_SCENARIOS[label]['marker'], size=10),
        name=label))
fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2)
fig.update_layout(
    title_text='<b>Cross-Scenario: Decay Ratio Trend vs mb</b>',
    width=1000, height=500, template='plotly_white',
    xaxis_title='mb', yaxis_title='Mean Decay Ratio')
fig.show()

# Print numeric summary
print("Decay Ratio by mb × Scenario:")
decay_by_mb = combined_metrics.pivot_table(index='scenario', columns='mb',
    values='decay_ratio', aggfunc='mean')
print(decay_by_mb.round(3).to_string())
print("\\n|target - actual|:")
print((decay_by_mb - 2/3).abs().round(3).to_string())"""))

# Section 9: Cross-scenario beta
cells.append(md("""## Section 9: Cross-Scenario Beta"""))

cells.append(code("""# ── Global beta comparison table ──
beta_table = []
for label, result in all_results.items():
    bs = result['beta_stats']
    row = {'Scenario': label, 'beta_origin': bs.get('beta_origin', np.nan),
           'R2': bs.get('r2_origin', np.nan), 'n_points': bs.get('n_points', 0),
           'CI_lo': result['ci_lo'], 'CI_hi': result['ci_hi'],
           'beta_ols': bs.get('beta_ols', np.nan), 'se_ols': bs.get('se_ols', np.nan)}
    beta_table.append(row)
beta_comparison = pd.DataFrame(beta_table)
print("Global Beta Comparison:")
print(beta_comparison.to_string(index=False))

# Highlight closest to 0.5
beta_comparison['|beta-0.5|'] = abs(beta_comparison['beta_origin'] - 0.5)
winner = beta_comparison.loc[beta_comparison['|beta-0.5|'].idxmin(), 'Scenario']
print(f"\\nClosest to theory (0.5): {winner}")"""))

cells.append(code("""# ── Overlay scatter: 4 regression lines ──
fig = go.Figure()
for label, result in all_results.items():
    pc_no20 = result['pc_no20']
    if len(pc_no20) == 0:
        continue
    beta_o = result['beta_stats'].get('beta_origin', 0.5)
    color = ACTIVE_SCENARIOS[label]['color']
    dash = ACTIVE_SCENARIOS[label]['dash']
    # Sample points
    pc_s = pc_no20.sample(n=min(3000, len(pc_no20)), random_state=42)
    fig.add_trace(go.Scattergl(x=pc_s['x'], y=pc_s['y']-pc_s['alpha'], mode='markers',
        marker=dict(size=2, color=color, opacity=0.15),
        name=f'{label} points', legendgroup=label))
    # Fit line
    x_range = np.array([pc_no20['x'].min(), pc_no20['x'].max()])
    fig.add_trace(go.Scatter(x=x_range, y=beta_o*x_range, mode='lines',
        line=dict(color=color, width=3, dash=dash),
        name=f'{label}: {beta_o:.3f}', legendgroup=label))

fig.add_trace(go.Scatter(x=[-8, 0], y=[-4, 0], mode='lines',
    line=dict(color='black', width=2, dash='dash'), name='Theory: 0.5'))
fig.update_layout(title_text='<b>Cross-Scenario: Beta Regression Lines</b>',
                  width=1200, height=600, template='plotly_white',
                  xaxis_title='log(Q/V)', yaxis_title='log(Impact) - alpha')
fig.show()"""))

cells.append(code("""# ── Beta by V-group: grouped bar chart ──
V_levels = sorted(combined_metrics['V'].unique())
fig = go.Figure()
for label, result in all_results.items():
    pc_no20 = result['pc_no20']
    if len(pc_no20) == 0:
        continue
    betas, ses = [], []
    for V in V_levels:
        sub = pc_no20[pc_no20['V'] == V]
        s = compute_beta_for_subset(sub)
        betas.append(s['beta'])
        ses.append(s['se'])
    fig.add_trace(go.Bar(
        x=[f'V={V}' for V in V_levels], y=betas,
        error_y=dict(type='data', array=[s*1.96 for s in ses], visible=True),
        name=label, marker_color=ACTIVE_SCENARIOS[label]['color']))
fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2)
fig.update_layout(title_text='<b>Cross-Scenario: Beta by V</b>', barmode='group',
                  width=1000, height=500, template='plotly_white', yaxis_title='Beta')
fig.show()"""))

cells.append(code("""# ── Beta by mb-group: grouped bar chart ──
mb_levels = sorted(combined_metrics['mb'].unique())
fig = go.Figure()
for label, result in all_results.items():
    pc = result['pc']
    if len(pc) == 0:
        continue
    betas, ses = [], []
    for mb in mb_levels:
        sub = pc[pc['mb'] == mb]
        s = compute_beta_for_subset(sub)
        betas.append(s['beta'])
        ses.append(s['se'])
    fig.add_trace(go.Bar(
        x=[f'mb={mb}' for mb in mb_levels], y=betas,
        error_y=dict(type='data', array=[s*1.96 for s in ses], visible=True),
        name=label, marker_color=ACTIVE_SCENARIOS[label]['color']))
fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2)
fig.update_layout(title_text='<b>Cross-Scenario: Beta by mb</b>', barmode='group',
                  width=1000, height=500, template='plotly_white', yaxis_title='Beta')
fig.show()"""))

cells.append(code("""# ── Bootstrap CI comparison ──
fig = go.Figure()
for label, result in all_results.items():
    bb = result['boot_betas']
    if len(bb) == 0:
        continue
    fig.add_trace(go.Histogram(x=bb, nbinsx=50, name=label, opacity=0.5,
        marker_color=ACTIVE_SCENARIOS[label]['color']))
fig.add_vline(x=0.5, line_dash='dash', line_color='red', line_width=3)
fig.update_layout(title_text='<b>Cross-Scenario: Bootstrap Beta Distributions</b>',
                  barmode='overlay', width=1000, height=500, template='plotly_white',
                  xaxis_title='Beta')
fig.show()"""))

cells.append(code("""# ── Beta 3 modes cross-scenario ──
fig = make_subplots(rows=1, cols=3,
    subplot_titles=['@a (exact)', '[:a] (cumul <=a)', '[a:] (cumul >=a)'],
    horizontal_spacing=0.08)
mode_keys = ['beta_exact', 'beta_upto', 'beta_from']
mode_names = ['@a', '[:a]', '[a:]']
for col_idx, (mkey, mname) in enumerate(zip(mode_keys, mode_names), 1):
    for label, result in all_results.items():
        mdf = result['metrics_df']
        beta_by_a = defaultdict(list)
        for _, row in mdf[mdf['mb'] != 20].iterrows():
            if row['i'] <= 1:
                continue
            bdict = row[mkey]
            if isinstance(bdict, dict):
                for a, b in bdict.items():
                    beta_by_a[a].append(b)
        iters_sorted = sorted(beta_by_a.keys())
        if not iters_sorted:
            continue
        means = [np.mean(beta_by_a[a]) for a in iters_sorted]
        fig.add_trace(go.Scatter(x=iters_sorted, y=means, mode='lines+markers',
            line=dict(color=ACTIVE_SCENARIOS[label]['color'], width=2,
                      dash=ACTIVE_SCENARIOS[label]['dash']),
            name=label, legendgroup=label, showlegend=(col_idx==1)),
            row=1, col=col_idx)
    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', line_width=1, row=1, col=col_idx)
fig.update_layout(title_text='<b>Cross-Scenario: Beta 3 Modes</b>', width=1500,
                  height=500, template='plotly_white')
fig.show()"""))

# Section 10: Cross-scenario gamma
cells.append(md("""## Section 10: Cross-Scenario Gamma"""))

cells.append(code("""# ── Gamma dot plot ──
if len(combined_gamma) > 0:
    combined_gamma['pair'] = combined_gamma.apply(lambda r: f"i{r['i']}_mb{r['mb']}", axis=1)
    fig = go.Figure()
    for label in all_results:
        sub = combined_gamma[combined_gamma['scenario']==label]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(x=sub['pair'], y=sub['gamma'], mode='markers',
            marker=dict(size=10, color=ACTIVE_SCENARIOS[label]['color'],
                        symbol=ACTIVE_SCENARIOS[label]['marker']),
            name=label))
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2)
    fig.update_layout(title_text='<b>Cross-Scenario: Gamma by (i,mb)</b>', width=1200,
                      height=500, template='plotly_white', yaxis_title='Gamma')
    fig.show()

    # Summary table
    gamma_summary = combined_gamma.groupby('scenario')['gamma'].agg(['mean','std','min','max'])
    print("Gamma Summary:")
    print(gamma_summary.to_string())"""))

# Section 11: Cross-scenario quality
cells.append(md("""## Section 11: Cross-Scenario Quality"""))

cells.append(code("""# ── Side-by-side quality heatmaps ──
fig = make_subplots(rows=1, cols=n_scn, subplot_titles=list(all_results.keys()),
                    horizontal_spacing=0.06)
for col_idx, (label, result) in enumerate(all_results.items(), 1):
    mdf = result['metrics_df'].copy()
    mdf['label'] = mdf.apply(lambda r: f"i{r['i']}_mb{r['mb']}", axis=1)
    label_order = mdf.groupby('label')['Q_total'].first().sort_values().index.tolist()
    pivot = mdf.pivot_table(index='label', columns='V', values='quality_num',
                             aggfunc='first').reindex(label_order)
    pivot_txt = mdf.pivot_table(index='label', columns='V', values='quality',
                                 aggfunc='first').reindex(label_order)
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=[str(v) for v in pivot.columns], y=pivot.index.tolist(),
        colorscale=[[0,'#d32f2f'],[0.33,'#ff9800'],[0.66,'#8bc34a'],[1,'#2e7d32']],
        zmin=1, zmax=4, showscale=(col_idx==n_scn),
        text=pivot_txt.values, texttemplate='%{text}',
    ), row=1, col=col_idx)
fig.update_layout(title_text='<b>Cross-Scenario: Quality Heatmaps</b>',
                  width=500*n_scn, height=600, template='plotly_white')
fig.show()

# Quality distribution stacked bar
q_dist = combined_metrics.groupby(['scenario','quality']).size().unstack(fill_value=0)
for q in ['excellent','good','acceptable','poor']:
    if q not in q_dist.columns:
        q_dist[q] = 0
q_colors = {'excellent': '#2e7d32', 'good': '#8bc34a', 'acceptable': '#ff9800', 'poor': '#d32f2f'}
fig = go.Figure()
for q in ['excellent','good','acceptable','poor']:
    fig.add_trace(go.Bar(x=q_dist.index, y=q_dist[q], name=q, marker_color=q_colors[q]))
fig.update_layout(title_text='<b>Cross-Scenario: Quality Distribution</b>', barmode='stack',
                  width=900, height=500, template='plotly_white', yaxis_title='Count')
fig.show()"""))

cells.append(code("""# ── Quality by mb: grouped bar (fraction good+excellent) ──
mb_levels = sorted(combined_metrics['mb'].unique())
fig = go.Figure()
for label in all_results:
    mdf = all_results[label]['metrics_df']
    fracs = []
    for mb in mb_levels:
        sub = mdf[mdf['mb'] == mb]
        frac = sub['quality'].isin(['excellent','good']).mean() if len(sub) > 0 else 0
        fracs.append(frac)
    fig.add_trace(go.Bar(
        x=[f'mb={mb}' for mb in mb_levels], y=fracs,
        name=label, marker_color=ACTIVE_SCENARIOS[label]['color']))
fig.update_layout(title_text='<b>Cross-Scenario: Quality (good+excellent fraction) by mb</b>',
                  barmode='group', width=1000, height=500, template='plotly_white',
                  yaxis_title='Fraction good+excellent', yaxis_range=[0,1])
fig.show()

# ── Quality distribution by mb × scenario ──
fig = make_subplots(rows=1, cols=len(mb_levels),
    subplot_titles=[f'mb={mb}' for mb in mb_levels],
    horizontal_spacing=0.06)
for col_idx, mb in enumerate(mb_levels, 1):
    for label in all_results:
        mdf = all_results[label]['metrics_df']
        sub = mdf[mdf['mb'] == mb]
        q_counts = sub['quality'].value_counts()
        for q in ['excellent','good','acceptable','poor']:
            if q not in q_counts.index:
                q_counts[q] = 0
        fig.add_trace(go.Bar(
            x=[label], y=[q_counts.get('excellent',0) + q_counts.get('good',0)],
            marker_color=ACTIVE_SCENARIOS[label]['color'],
            showlegend=(col_idx==1), name=label), row=1, col=col_idx)
fig.update_layout(title_text='<b>Quality: Good+Excellent Configs per mb</b>',
                  width=300*len(mb_levels), height=400, template='plotly_white')
fig.show()"""))

cells.append(code("""# ── Quality trend by mb: line plot ──
fig = go.Figure()
for label in all_results:
    mdf = all_results[label]['metrics_df']
    grp = mdf.groupby('mb')['quality_num'].mean().sort_index()
    fig.add_trace(go.Scatter(
        x=grp.index, y=grp.values, mode='lines+markers',
        line=dict(color=ACTIVE_SCENARIOS[label]['color'], width=2,
                  dash=ACTIVE_SCENARIOS[label]['dash']),
        marker=dict(symbol=ACTIVE_SCENARIOS[label]['marker'], size=10),
        name=label))
fig.update_layout(
    title_text='<b>Cross-Scenario: Mean Quality Score vs mb</b>',
    width=1000, height=500, template='plotly_white',
    xaxis_title='mb', yaxis_title='Mean Quality (1=poor, 4=excellent)')
fig.show()

# Numeric table
print("Quality (mean quality_num) by mb × Scenario:")
q_by_mb = combined_metrics.pivot_table(index='scenario', columns='mb',
    values='quality_num', aggfunc='mean')
print(q_by_mb.round(2).to_string())

# Also show decay + beta + stability combined breakdown by mb
print("\\nDecay Ratio by mb × Scenario:")
print(combined_metrics.pivot_table(index='scenario', columns='mb',
    values='decay_ratio', aggfunc='mean').round(3).to_string())
print("\\nBeta [:a] by mb × Scenario:")
print(combined_metrics.pivot_table(index='scenario', columns='mb',
    values='beta_[:a]', aggfunc='mean').round(3).to_string())
if 'votes_num' in combined_metrics.columns:
    print("\\nStability (mean votes) by mb × Scenario:")
    print(combined_metrics.pivot_table(index='scenario', columns='mb',
        values='votes_num', aggfunc='mean').round(2).to_string())"""))

# Section 12: Cross-scenario volume-time
cells.append(md("""## Section 12: Cross-Scenario Volume-Time"""))

cells.append(code("""# ── Master curves per scenario: panels ──
fig = make_subplots(rows=1, cols=n_scn, subplot_titles=list(all_results.keys()),
                    horizontal_spacing=0.06)
U_SHOW = 3.0
colors_all = px.colors.qualitative.D3 + px.colors.qualitative.Set2
for col_idx, (label, result) in enumerate(all_results.items(), 1):
    sorted_folders = sorted(result['all_curves_norm'].keys(),
        key=lambda f: (parse_folder_params_v2(f)[2], parse_folder_params_v2(f)[0]))
    for idx, folder in enumerate(sorted_folders[:15]):
        c = result['all_curves_norm'][folder]
        u = c['u_grid']
        mask = u <= U_SHOW
        color = colors_all[idx % len(colors_all)]
        fig.add_trace(go.Scatter(x=u[mask], y=c['combined_mean'][mask], mode='lines',
            line=dict(color=color, width=1), showlegend=False), row=1, col=col_idx)
    fig.add_vline(x=1.0, line_dash='dash', line_color='gray', line_width=1, row=1, col=col_idx)
fig.update_layout(title_text='<b>Cross-Scenario: Master Curves</b>',
                  width=400*n_scn, height=450, template='plotly_white')
fig.show()"""))

cells.append(code("""# ── Cross-scenario overlay: average curve per scenario ──
fig = go.Figure()
u_common = np.linspace(0, 3.0, 300)
for label, result in all_results.items():
    all_interp = []
    for folder, c in result['all_curves_norm'].items():
        u = c['u_grid']
        mean = c['combined_mean']
        if u[-1] >= 3.0:
            interp = np.interp(u_common, u, mean)
            all_interp.append(interp)
    if all_interp:
        avg = np.mean(all_interp, axis=0)
        std = np.std(all_interp, axis=0)
        color = ACTIVE_SCENARIOS[label]['color']
        fig.add_trace(go.Scatter(x=u_common, y=avg, mode='lines',
            line=dict(color=color, width=3, dash=ACTIVE_SCENARIOS[label]['dash']),
            name=label))
        fig.add_trace(go.Scatter(
            x=np.concatenate([u_common, u_common[::-1]]),
            y=np.concatenate([avg+std, (avg-std)[::-1]]),
            fill='toself', fillcolor=color.replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in color else f'rgba(128,128,128,0.1)',
            line=dict(width=0), showlegend=False, name=f'{label} std'))
fig.add_vline(x=1.0, line_dash='dash', line_color='gray', line_width=1.5)
fig.update_layout(title_text='<b>Cross-Scenario: Average Master Curve</b>',
                  width=1200, height=500, template='plotly_white',
                  xaxis_title='u', yaxis_title='I_norm(u)')
fig.show()"""))

cells.append(code("""# ── Relaxation ratio comparison ──
if len(combined_relax) > 0:
    fig = go.Figure()
    for label in all_results:
        sub = combined_relax[combined_relax['scenario']==label]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Box(y=sub['ratio_final'], name=label,
            marker_color=ACTIVE_SCENARIOS[label]['color'],
            boxpoints='all', jitter=0.3))
    fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2)
    fig.update_layout(title_text='<b>Cross-Scenario: Relaxation Ratio</b>',
                      width=900, height=500, template='plotly_white',
                      yaxis_title='I_final / I_peak')
    fig.show()

# CV comparison
cv_table = []
for label, result in all_results.items():
    cv_data = result['cv_data']
    row = {'Scenario': label}
    for u_pt, vals in cv_data.items():
        row[f'CV@u={u_pt}'] = vals['cv']
    cv_table.append(row)
if cv_table:
    cv_df = pd.DataFrame(cv_table)
    print("CV Comparison (lower = better collapse):")
    print(cv_df.to_string(index=False))"""))

# Section 13: Cross-scenario stability
cells.append(md("""## Section 13: Cross-Scenario Stability"""))

cells.append(code("""# ── Side-by-side stability heatmaps ──
fig = make_subplots(rows=1, cols=n_scn, subplot_titles=list(all_results.keys()),
                    horizontal_spacing=0.06)
for col_idx, (label, result) in enumerate(all_results.items(), 1):
    sdf = result['stab_df'].copy()
    if len(sdf) == 0:
        continue
    sdf['label'] = sdf.apply(lambda r: f"i{r['i']}_mb{r['mb']}", axis=1)
    label_order = sdf.groupby('label').first().sort_index().index.tolist()
    pivot = sdf.pivot_table(index='label', columns='V', values='votes_num',
                             aggfunc='first').reindex(label_order)
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=[str(v) for v in pivot.columns], y=pivot.index.tolist(),
        colorscale=[[0,'#d32f2f'],[0.33,'#ff9800'],[0.66,'#8bc34a'],[1,'#2e7d32']],
        zmin=0, zmax=3, showscale=(col_idx==n_scn),
        text=pivot.values.astype(str) + '/3', texttemplate='%{text}',
    ), row=1, col=col_idx)
fig.update_layout(title_text='<b>Cross-Scenario: Stability Heatmaps</b>',
                  width=500*n_scn, height=600, template='plotly_white')
fig.show()

# Stability bar chart
stab_summary = combined_stab.groupby('scenario')['stabilized'].mean()
fig = go.Figure(go.Bar(
    x=stab_summary.index, y=stab_summary.values,
    marker_color=[ACTIVE_SCENARIOS[s]['color'] for s in stab_summary.index]))
fig.add_hline(y=0.5, line_dash='dash', line_color='gray', line_width=1)
fig.update_layout(title_text='<b>Cross-Scenario: Fraction Stable</b>',
                  width=700, height=400, template='plotly_white',
                  yaxis_title='Fraction (2+/3 votes)')
fig.show()"""))

# ═══════════════════════════════════════════════════════════════════
# Section 14: Model Ranking
# ═══════════════════════════════════════════════════════════════════

cells.append(md("""---
## Section 14: Model Ranking & Final Assessment"""))

cells.append(code("""# ── Comprehensive scorecard ──
scorecard_rows = []
for label, result in all_results.items():
    mdf = result['metrics_df']
    sdf = result['stab_df']
    bs = result['beta_stats']
    gdf = result['gamma_df']
    cv_data = result['cv_data']

    beta_val = bs.get('beta_origin', np.nan)
    beta_accuracy = 1 - abs(beta_val - 0.5) / 0.5 if not np.isnan(beta_val) else 0
    decay_mean = mdf['decay_ratio'].mean()
    decay_accuracy = 1 - abs(decay_mean - 2/3) / (2/3) if not np.isnan(decay_mean) else 0
    gamma_mean = gdf['gamma'].mean() if len(gdf) > 0 else np.nan
    gamma_accuracy = 1 - abs(gamma_mean - 0.5) / 0.5 if not np.isnan(gamma_mean) else 0
    stability_frac = sdf['stabilized'].mean() if len(sdf) > 0 else 0
    quality_frac = (mdf['quality'].isin(['excellent', 'good'])).mean()
    cv_at_1 = cv_data.get(1.0, {}).get('cv', np.nan)
    collapse_score = 1 - min(cv_at_1, 1.0) if not np.isnan(cv_at_1) else 0

    scorecard_rows.append({
        'Scenario': label,
        'beta': beta_val, 'beta_CI': f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]",
        'beta_accuracy': beta_accuracy,
        'decay_ratio': decay_mean, 'decay_accuracy': decay_accuracy,
        'gamma': gamma_mean, 'gamma_accuracy': gamma_accuracy,
        'stability_%': stability_frac * 100,
        'quality_%': quality_frac * 100,
        'CV@u=1': cv_at_1, 'collapse_score': collapse_score,
    })

scorecard = pd.DataFrame(scorecard_rows)
print("="*100)
print("COMPREHENSIVE SCORECARD")
print("="*100)
print(scorecard.to_string(index=False))"""))

cells.append(code("""# ── Radar chart ──
import plotly.graph_objects as go

categories = ['Beta\\nAccuracy', 'Decay\\nAccuracy', 'Gamma\\nAccuracy',
              'Stability', 'Quality', 'Collapse']

fig = go.Figure()
for _, row in scorecard.iterrows():
    vals = [max(row['beta_accuracy'],0), max(row['decay_accuracy'],0),
            max(row['gamma_accuracy'],0), row['stability_%']/100,
            row['quality_%']/100, max(row['collapse_score'],0)]
    vals.append(vals[0])  # close the polygon
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=categories + [categories[0]], fill='toself',
        name=row['Scenario'],
        line=dict(color=ACTIVE_SCENARIOS[row['Scenario']]['color'], width=2),
        opacity=0.7))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title_text='<b>Model Comparison Radar</b>',
    width=800, height=600, template='plotly_white')
fig.show()"""))

cells.append(code("""# ── Composite score ──
WEIGHTS = {'beta_accuracy': 0.25, 'decay_accuracy': 0.20, 'gamma_accuracy': 0.15,
           'stability_%': 0.15, 'quality_%': 0.15, 'collapse_score': 0.10}

scorecard['composite'] = sum(
    scorecard[k] * w if k != 'stability_%' and k != 'quality_%'
    else scorecard[k] / 100 * w
    for k, w in WEIGHTS.items()
)
scorecard_sorted = scorecard.sort_values('composite', ascending=False)

fig = go.Figure(go.Bar(
    x=scorecard_sorted['Scenario'], y=scorecard_sorted['composite'],
    marker_color=[ACTIVE_SCENARIOS[s]['color'] for s in scorecard_sorted['Scenario']],
    text=scorecard_sorted['composite'].round(3).astype(str),
    textposition='outside'))
fig.update_layout(title_text='<b>Composite Score (weighted)</b>',
                  width=700, height=450, template='plotly_white',
                  yaxis_title='Score', yaxis_range=[0, 1])
fig.show()

print("\\nRanking:")
for rank, (_, row) in enumerate(scorecard_sorted.iterrows(), 1):
    print(f"  #{rank}: {row['Scenario']} (composite={row['composite']:.4f})")"""))

cells.append(code("""# ── Per-config winner map ──
# For each config, which scenario has best quality?
configs = combined_metrics[['i','mb','V']].drop_duplicates().sort_values(['mb','i','V'])
winner_rows = []
for _, cfg in configs.iterrows():
    row = {'config': f"i{cfg['i']}_mb{cfg['mb']}_V{cfg['V']}"}
    best_q, best_scn = -1, 'N/A'
    for label in all_results:
        sub = combined_metrics[(combined_metrics['scenario']==label) &
                               (combined_metrics['i']==cfg['i']) &
                               (combined_metrics['mb']==cfg['mb']) &
                               (combined_metrics['V']==cfg['V'])]
        if len(sub) > 0:
            q = sub.iloc[0]['quality_num']
            row[label] = q
            if q > best_q:
                best_q = q
                best_scn = label
    row['winner'] = best_scn
    winner_rows.append(row)
winner_df = pd.DataFrame(winner_rows)
print("Per-config winner (quality_num: 1=poor, 2=acceptable, 3=good, 4=excellent):")
print(winner_df.to_string(index=False))

# Winner counts
print("\\nWinner counts:")
print(winner_df['winner'].value_counts().to_string())"""))

cells.append(code("""# ── Paired bootstrap significance ──
print("="*80)
print("Paired Bootstrap: Is beta difference statistically significant?")
print("="*80)

scn_names = list(all_results.keys())
for i in range(len(scn_names)):
    for j in range(i+1, len(scn_names)):
        a_name, b_name = scn_names[i], scn_names[j]
        a_betas = all_results[a_name]['boot_betas']
        b_betas = all_results[b_name]['boot_betas']
        if len(a_betas) == 0 or len(b_betas) == 0:
            continue
        diff = a_betas - b_betas
        p_val = np.mean(diff > 0)
        sig = '*' if p_val < 0.05 or p_val > 0.95 else ''
        print(f"  {a_name} vs {b_name}: "
              f"mean_diff={np.mean(diff):.4f}, "
              f"P(A>B)={p_val:.3f} {sig}")"""))

cells.append(md("""### Discussion

**Q1: Does the model reproduce market impact?**
Compare beta values across scenarios. Theory predicts beta~0.5.

**Q2: Is the decay consistent with Bouchaud's 2/3 rule?**
Compare decay ratios. Target: permanent impact = 2/3 of peak.

**Q3: How does volume scaling compare?**
Gamma~0.5 expected (square-root). Compare across models.

**Q4: Which model produces the most stable results?**
Stability votes (2+/3 methods) across all configurations.

**Q5: Overall winner?**
Composite score combines all metrics with interpretable weights."""))

# ═══════════════════════════════════════════════════════════════════
# Section 15: Dashboard & Export
# ═══════════════════════════════════════════════════════════════════

cells.append(md("""## Section 15: Dashboard & Export"""))

cells.append(code("""# ── Cross-scenario 2x3 summary dashboard ──
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=['Beta Comparison', 'Decay Ratio', 'Gamma',
                    'Stability', 'Quality Distribution', 'Composite Score'],
    vertical_spacing=0.12, horizontal_spacing=0.08,
    specs=[[{'type':'bar'},{'type':'box'},{'type':'scatter'}],
           [{'type':'bar'},{'type':'bar'},{'type':'bar'}]])

# (1,1) Beta bars
for label, result in all_results.items():
    beta = result['beta_stats'].get('beta_origin', np.nan)
    fig.add_trace(go.Bar(x=[label], y=[beta],
        marker_color=ACTIVE_SCENARIOS[label]['color'], showlegend=False), row=1, col=1)
fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2, row=1, col=1)

# (1,2) Decay ratio box
for label in all_results:
    mdf = all_results[label]['metrics_df']
    fig.add_trace(go.Box(y=mdf['decay_ratio'], name=label,
        marker_color=ACTIVE_SCENARIOS[label]['color'], showlegend=False), row=1, col=2)
fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=1, row=1, col=2)

# (1,3) Gamma scatter
if len(combined_gamma) > 0:
    for label in all_results:
        sub = combined_gamma[combined_gamma['scenario']==label]
        if len(sub) > 0:
            fig.add_trace(go.Scatter(x=sub.apply(lambda r: f"i{r['i']}mb{r['mb']}", axis=1),
                y=sub['gamma'], mode='markers',
                marker=dict(size=6, color=ACTIVE_SCENARIOS[label]['color']),
                showlegend=False), row=1, col=3)
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=1, row=1, col=3)

# (2,1) Stability
stab_summary = combined_stab.groupby('scenario')['stabilized'].mean()
fig.add_trace(go.Bar(x=stab_summary.index, y=stab_summary.values,
    marker_color=[ACTIVE_SCENARIOS[s]['color'] for s in stab_summary.index],
    showlegend=False), row=2, col=1)

# (2,2) Quality
for label in all_results:
    mdf = all_results[label]['metrics_df']
    good_frac = mdf['quality'].isin(['excellent','good']).mean()
    fig.add_trace(go.Bar(x=[label], y=[good_frac],
        marker_color=ACTIVE_SCENARIOS[label]['color'], showlegend=False), row=2, col=2)

# (2,3) Composite
fig.add_trace(go.Bar(x=scorecard_sorted['Scenario'], y=scorecard_sorted['composite'],
    marker_color=[ACTIVE_SCENARIOS[s]['color'] for s in scorecard_sorted['Scenario']],
    showlegend=False), row=2, col=3)

fig.update_layout(title_text='<b>Cross-Scenario Summary Dashboard</b>',
                  width=1600, height=800, template='plotly_white')
fig.show()"""))

cells.append(code("""# ── Export results ──
_LOB_IMPACT = Path("/app/lob_impact")
if not _LOB_IMPACT.exists():
    _LOB_IMPACT = Path("/scratch/local/homes/80/georgenigm/LOBS5/lob_impact")

# Combined metrics
export_cols = ['scenario','folder','i','mb','V','Q_total','total_gen','peak','final',
               'decay_ratio','beta_[:a]','votes_num','quality']
export_df = combined_metrics[export_cols].sort_values(['scenario','mb','i','V']).reset_index(drop=True)
out_path = _LOB_IMPACT / "market_impact_all_scenarios_results.csv"
export_df.to_csv(out_path, index=False)
print(f"Exported {len(export_df)} rows to {out_path}")

# Scorecard
sc_path = _LOB_IMPACT / "market_impact_all_scenarios_scorecard.csv"
scorecard_sorted.to_csv(sc_path, index=False)
print(f"Exported scorecard to {sc_path}")"""))

cells.append(md("""### Conclusions & Executive Summary

This notebook compared **all active scenarios** for market impact reproduction.

**Key metrics** (theoretical targets):
- Beta (square-root law): target = 0.5
- Decay ratio: target = 2/3 (Bouchaud)
- Gamma (volume scaling): target = 0.5
- Stability: fraction of configs with 2+/3 stability votes
- Quality: fraction of configs rated good or excellent

See the **scorecard** and **composite score** above for the final ranking.

**Notebook 120 replaces per-scenario analyses (100-105) with a unified cross-scenario comparison framework.**"""))


# ═══════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════

# Fix cell sources - convert string to list of lines
for i, cell in enumerate(cells):
    if isinstance(cell['source'], str):
        lines = cell['source'].split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    cell['id'] = f'cell-{i:03d}'

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "/homes/80/georgenigm/LOBS5/lob_impact/120.market_impact_all_scenarios.ipynb"
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Written {len(cells)} cells to {out_path}")
