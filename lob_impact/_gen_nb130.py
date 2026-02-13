#!/usr/bin/env python3
"""Generate 130.paper_results.ipynb — focused paper figures & tables.

Four models × three analyses: β (square-root law), relaxation, stability.
Seven figures + three LaTeX-ready tables.
"""
import json, textwrap
from pathlib import Path


def cc(src):
    """Code cell."""
    lines = textwrap.dedent(src).strip("\n").split("\n")
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def mc(src):
    """Markdown cell."""
    lines = textwrap.dedent(src).strip("\n").split("\n")
    return {
        "cell_type": "markdown", "metadata": {},
        "source": [l + "\n" for l in lines],
    }


cells = []

# ═══════════════════════════════════════════════════════════════════
# CELL 0 — Title
# ═══════════════════════════════════════════════════════════════════
cells.append(mc("""\
# 130 · Paper Figures & Tables

Four-model market impact comparison: **S5 (Neural)**, **Historic replay**, **Heuristic price-shift**, **CST**.

Three analyses: **Square-root law (β)**, **Impact relaxation**, **Stability**.

Outputs: 7 figures (PNG) + 3 LaTeX tables.
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 1 — Imports
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
import numpy as np
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
warnings.filterwarnings('ignore')
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 2 — Configuration
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
TICK_SIZE = 100
MAX_SAMPLES = None          # use all available
MIDPRICE_MAX = 200_000      # outlier filter (ticks)
N_BOOTSTRAP = 1000

SCENARIOS = OrderedDict([
    ('S5 (Neural)', {'key': 'aggressive_scenario', 'color': '#1f77b4', 'dash': 'solid'}),
    ('Historic',    {'key': 'historic_scenario',    'color': '#ff7f0e', 'dash': 'dash'}),
    ('Heuristic',   {'key': 'heuristic_scenario',   'color': '#2ca02c', 'dash': 'dot'}),
    ('CST',         {'key': 'cst_scenario',         'color': '#d62728', 'dash': 'dashdot'}),
])

# ── Paths (auto-detect Docker vs host) ──
_BASE = [Path("/app/output/evalsequences"),
         Path("/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences")]
EVAL_BASE = next((p for p in _BASE if p.exists()), _BASE[-1])

_SDM = [Path("/app/lob_impact/sample_day_map.csv"),
        Path("/scratch/local/homes/80/georgenigm/LOBS5/lob_impact/sample_day_map.csv")]
SDM_PATH = next((p for p in _SDM if p.exists()), _SDM[-1])
SAMPLE_DAY_MAP = pd.read_csv(SDM_PATH)

_FIG = [Path("/app/pics_for_transfer_4_methods"),
        Path("/homes/80/georgenigm/LOBS5/pics_for_transfer_4_methods")]
FIG_DIR = next((p for p in _FIG if p.exists()), _FIG[-1])
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"EVAL_BASE : {EVAL_BASE}")
print(f"SDM       : {len(SAMPLE_DAY_MAP)} rows")
print(f"FIG_DIR   : {FIG_DIR}")
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 3 — Data I/O functions
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Data I/O helpers ──────────────────────────────────────────────

def discover_v2_folders(buy_path, sell_path):
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
        rows.append({'folder': p.name, 'i': i, 'c': c, 'mb': mb, 'V': V,
                     'Q_total': i * V, 'buy_path': p, 'sell_path': sell_p})
    return pd.DataFrame(rows)


def parse_folder_params_v2(folder_name):
    m = re.match(r'i(\\d+)_c(\\d+)_mb(\\d+)_v(\\d+)_cntxt(.+)', folder_name)
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
    cond_dir = data_path / "data_cond"
    pat = re.compile(r"^(.+?)_(\\d{4}-\\d{2}-\\d{2})_orderbook_real_id_(\\d+)\\.csv$")
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
    return all_data
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 4 — Beta analysis functions
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Beta (square-root law) ────────────────────────────────────────

def extract_point_cloud(data, grid_df):
    eps = 1e-12
    rows = []
    for _, grow in grid_df.iterrows():
        folder = grow['folder']
        if folder not in data:
            continue
        d = data[folder]
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
                                     'folder': folder, 'direction': direction})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_global_beta(df):
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
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 5 — Master curves, relaxation, gamma functions
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Master curves, relaxation, gamma ──────────────────────────────

def compute_master_curve(buy_data, sell_data, folder, aggr_gen,
                         u_max=3.0, n_pts=300):
    \"\"\"Sigma-normalized volume-time impact curve for one folder.\"\"\"
    i, c, mb, V = parse_folder_params_v2(folder)
    if len(aggr_gen) < 2:
        return None
    s_gen = int(aggr_gen[0])
    e_gen = int(aggr_gen[-1])
    L = e_gen - s_gen
    if L == 0:
        return None
    bb, sb = buy_data['books'], sell_data['books']
    min_len = min(min(b.shape[0] for b in bb.values()),
                  min(b.shape[0] for b in sb.values()))
    junction = list(buy_data['cond_lens'].values())[0]
    u_cap = min(u_max, (min_len - 1 - junction - s_gen) / L)
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


def compute_combined_impact(buy_data, sell_data, folder):
    \"\"\"Raw (buy−sell)/2 mid-price impact curve for stability analysis.\"\"\"
    bb, sb = buy_data['books'], sell_data['books']
    bc, sc = buy_data['cond_lens'], sell_data['cond_lens']
    common = set(bb.keys()) & set(sb.keys())
    if not common:
        return None
    min_len = min(min(bb[s].shape[0] for s in common),
                  min(sb[s].shape[0] for s in common))
    impacts = []
    for sid in common:
        j = bc[sid]
        bm = compute_midprice(bb[sid][:min_len])
        sm = compute_midprice(sb[sid][:min_len])
        ref = (bm[j - 1] + sm[j - 1]) / 2
        impacts.append((bm[:min_len] - sm[:min_len]) / (2 * ref))
    impacts = np.array(impacts)
    mean = np.mean(impacts, axis=0)
    junction = list(bc.values())[0]
    post = mean[junction:]
    pk_idx = junction + np.argmax(post)
    return {'mean': mean, 'junction': junction,
            'peak': float(mean[pk_idx]), 'final': float(mean[-1]),
            'peak_idx': pk_idx, 'n_samples': len(impacts), 'min_len': min_len}
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 6 — Stability functions
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Stability (3-method vote) ─────────────────────────────────────

TAIL_FRAC   = 0.20
WINDOW_FRAC = 0.15
SLOPE_THRESH = 0.05
TWIN_THRESH  = 0.03
CONV_THRESH  = 95.0


def stability_for_folder(stats, row):
    \"\"\"3-method stability test for one folder.\"\"\"
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
    return {'folder': row['folder'], 'stabilized': votes >= 2, 'votes': int(votes)}
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 7 — Load + process all scenarios
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Load & process all 4 scenarios ────────────────────────────────

R = OrderedDict()  # results dict

for label, cfg in SCENARIOS.items():
    buy_p  = EVAL_BASE / cfg['key'] / 'context_500_buy'
    sell_p = EVAL_BASE / cfg['key'] / 'context_500_sell'
    if not buy_p.exists() or not sell_p.exists():
        print(f"SKIP {label}: {buy_p} not found")
        continue
    grid = discover_v2_folders(buy_p, sell_p)
    if grid.empty:
        print(f"SKIP {label}: no folders")
        continue
    print(f"\\n{'='*60}\\n  {label}: {len(grid)} configs")
    data = load_all_v2(grid)

    # ── Beta ──
    pc = extract_point_cloud(data, grid)
    bstat = compute_global_beta(pc)
    bbetas = bootstrap_beta(pc, N_BOOTSTRAP)

    # ── Master curves + relaxation ──
    curves = {}
    for _, row in grid.iterrows():
        f = row['folder']
        if f not in data:
            continue
        aggr = load_aggressive_indices(row['buy_path'])
        c = compute_master_curve(data[f]['buy'], data[f]['sell'], f, aggr)
        if c is not None:
            curves[f] = c

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

    # ── Relaxation ratios ──
    relax_rows = []
    for f, cv in curves.items():
        u, m = cv['u_grid'], cv['combined_mean']
        I_peak = float(np.interp(1.0, u, m))
        if abs(I_peak) < 1e-12:
            continue
        I_final = float(m[-1])
        relax_rows.append({'folder': f, 'I_peak': I_peak, 'I_final': I_final,
                           'ratio': I_final / I_peak,
                           'mb': cv['mb'], 'V': cv['V']})
    relax_df = pd.DataFrame(relax_rows) if relax_rows else pd.DataFrame()

    R[label] = {
        'grid': grid, 'pc': pc, 'beta': bstat, 'boot': bbetas,
        'curves': curves, 'relax_df': relax_df,
        'stab_df': stab_df, 'gamma_df': gamma_df,
    }
    stable_frac = stab_df['stabilized'].mean() if not stab_df.empty else 0
    print(f"  β={bstat['beta']:.4f}  R²={bstat['r2']:.4f}  n={bstat['n']:,}")
    print(f"  Curves: {len(curves)}, Stable: {stable_frac:.0%}, Gamma configs: {len(gamma_rows)}")

    del data
    gc.collect()

print(f"\\n{'='*60}\\nLoaded {len(R)} scenarios: {list(R.keys())}")
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 8 — Section: Beta
# ═══════════════════════════════════════════════════════════════════
cells.append(mc("""\
---
## 1. Square-Root Law (β)

**Theory**: $\\Delta p / \\sigma \\sim (Q/V)^\\beta$ with $\\beta = 0.5$ (Kyle, 1985; Tóth et al., 2011).
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 9 — Table 1: Global Beta Comparison
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Table 1: Global Beta Comparison ──

rows = []
for label, r in R.items():
    b = r['beta']
    ci = np.percentile(r['boot'], [2.5, 97.5]) if len(r['boot']) > 0 else [np.nan, np.nan]
    rows.append({'Model': label, 'β': f"{b['beta']:.3f}",
                 'R²': f"{b['r2']:.3f}", 'N': f"{b['n']:,}",
                 '95% CI': f"[{ci[0]:.3f}, {ci[1]:.3f}]"})
table1 = pd.DataFrame(rows)
print("\\n── Table 1: Global Beta Comparison ──")
print(table1.to_string(index=False))

# LaTeX
print("\\n── LaTeX ──")
print("\\\\begin{tabular}{lcccc}")
print("\\\\toprule")
print("Model & $\\\\beta$ & $R^2$ & $N$ & 95\\\\% CI \\\\\\\\")
print("\\\\midrule")
for _, r in table1.iterrows():
    print(f"{r['Model']} & {r['β']} & {r['R²']} & {r['N']} & {r['95% CI']} \\\\\\\\")
print("\\\\bottomrule")
print("\\\\end{tabular}")
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 10 — Figure 3: Beta Regression Lines
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Figure 3: Beta Regression Lines ──

fig = go.Figure()
x_range = np.array([-16, -4])

for label, r in R.items():
    pc = r['pc']
    if pc.empty:
        continue
    color = SCENARIOS[label]['color']
    dash  = SCENARIOS[label]['dash']
    beta  = r['beta']['beta']

    # scatter (subsample for speed)
    n_show = min(5000, len(pc))
    idx = np.random.RandomState(42).choice(len(pc), n_show, replace=False)
    sub = pc.iloc[idx]
    fig.add_trace(go.Scatter(
        x=sub['x'], y=sub['y'] - sub['alpha'], mode='markers',
        marker=dict(size=2, color=color, opacity=0.15),
        name=f"{label} points", showlegend=True))
    # regression line
    y_line = beta * x_range
    fig.add_trace(go.Scatter(
        x=x_range, y=y_line, mode='lines',
        line=dict(color=color, width=3, dash=dash),
        name=f"{label}: {beta:.3f}"))

# theory line β=0.5
fig.add_trace(go.Scatter(
    x=x_range, y=0.5 * x_range, mode='lines',
    line=dict(color='black', width=2, dash='dash'), name='Theory: 0.5'))

fig.update_layout(
    title='<b>Cross-Scenario: Beta Regression Lines</b>',
    xaxis_title='log(Q/V)', yaxis_title='log(Impact) − α',
    width=900, height=700, template='plotly_white',
    legend=dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom'))
fig.write_image(str(FIG_DIR / '3. Beta Regression Lines.png'), scale=2)
fig.show()
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 11 — Figure 4: Bootstrap Beta Distributions
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Figure 4: Bootstrap Beta Distributions ──

fig = go.Figure()
for label, r in R.items():
    bb = r['boot']
    if len(bb) == 0:
        continue
    fig.add_trace(go.Histogram(
        x=bb, nbinsx=50, name=label, opacity=0.5,
        marker_color=SCENARIOS[label]['color']))

fig.add_vline(x=0.5, line_dash='dash', line_color='red', line_width=3,
              annotation_text='β=0.5', annotation_position='top left')

fig.update_layout(
    title='<b>Cross-Scenario: Bootstrap Beta Distributions</b>',
    xaxis_title='Beta', yaxis_title='Count',
    barmode='overlay', width=900, height=500, template='plotly_white')
fig.write_image(str(FIG_DIR / '4. Bootstrap Beta Distributions.png'), scale=2)
fig.show()
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 12 — Section: Master Curves & Relaxation
# ═══════════════════════════════════════════════════════════════════
cells.append(mc("""\
---
## 2. Master Curves & Impact Relaxation

**Theory**: after rescaling by $Q^\\beta \\sigma$, individual impact curves should collapse
onto a universal master curve $\\mathcal{F}(u)$ with relaxation ratio
$\\mathcal{F}(\\infty)/\\mathcal{F}(1) \\approx 2/3$ (Bouchaud et al., 2004).
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 13 — Figure 1: Master Curves (2×2)
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Figure 1: Master Curves (2×2 panels) ──

n_scn = len(R)
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=list(R.keys()),
                    horizontal_spacing=0.08, vertical_spacing=0.12)
palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2

for idx, (label, r) in enumerate(R.items()):
    row, col = idx // 2 + 1, idx % 2 + 1
    sorted_f = sorted(r['curves'].keys(),
        key=lambda f: (parse_folder_params_v2(f)[2], parse_folder_params_v2(f)[0]))
    for fi, folder in enumerate(sorted_f):
        c = r['curves'][folder]
        u, m = c['u_grid'], c['combined_mean']
        mask = u <= 3.0
        fig.add_trace(go.Scatter(
            x=u[mask], y=m[mask], mode='lines',
            line=dict(color=palette[fi % len(palette)], width=1),
            showlegend=False), row=row, col=col)
    fig.add_vline(x=1.0, line_dash='dash', line_color='gray',
                  line_width=1, row=row, col=col)

fig.update_layout(title_text='<b>Cross-Scenario: Master Curves</b>',
                  width=1300, height=900, template='plotly_white')
fig.write_image(str(FIG_DIR / '1. Master Curves.png'), scale=2)
fig.show()
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 14 — Figure 2: Average Master Curve
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Figure 2: Average Master Curve (overlay) ──

fig = go.Figure()
u_common = np.linspace(0, 3.0, 300)

for label, r in R.items():
    interps = []
    for f, c in r['curves'].items():
        if c['u_grid'][-1] >= 3.0:
            interps.append(np.interp(u_common, c['u_grid'], c['combined_mean']))
    if not interps:
        continue
    avg = np.mean(interps, axis=0)
    std = np.std(interps, axis=0)
    color = SCENARIOS[label]['color']
    dash  = SCENARIOS[label]['dash']
    fig.add_trace(go.Scatter(x=u_common, y=avg, mode='lines',
        line=dict(color=color, width=3, dash=dash), name=label))
    # confidence band
    rgba = color.replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in color else 'rgba(128,128,128,0.1)'
    fig.add_trace(go.Scatter(
        x=np.concatenate([u_common, u_common[::-1]]),
        y=np.concatenate([avg + std, (avg - std)[::-1]]),
        fill='toself', fillcolor=rgba,
        line=dict(width=0), showlegend=False))

fig.add_vline(x=1.0, line_dash='dash', line_color='gray', line_width=1.5)
fig.update_layout(
    title_text='<b>Cross-Scenario: Average Master Curve</b>',
    xaxis_title='u', yaxis_title='I_norm(u)',
    width=1100, height=500, template='plotly_white')
fig.write_image(str(FIG_DIR / '2. Average Master Curve.png'), scale=2)
fig.show()
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 15 — Table 2 + Figure 5: Relaxation Ratio
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Table 2: Relaxation Ratio ──
# I_final / I_peak  from sigma-normalized master curves
# Bouchaud theoretical ≈ 2/3

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
                 'Mean ± std': f"{mean:.3f} ± {std:.3f}",
                 'CV': f"{cv:.3f}",
                 'Δ from 2/3': f"{abs(med - 2/3):.3f}"})
table2 = pd.DataFrame(rows)
print("\\n── Table 2: Relaxation Ratio (I_final / I_peak) ──")
print("  Bouchaud theoretical: 2/3 ≈ 0.667")
print(table2.to_string(index=False))

# LaTeX
print("\\n── LaTeX ──")
print("\\\\begin{tabular}{lcccr}")
print("\\\\toprule")
print("Model & Median & Mean $\\\\pm$ std & CV & $|\\\\Delta|$ from $\\\\frac{2}{3}$ \\\\\\\\")
print("\\\\midrule")
for _, r in table2.iterrows():
    print(f"{r['Model']} & {r['Median ratio']} & {r['Mean ± std']} & {r['CV']} & {r['Δ from 2/3']} \\\\\\\\")
print("\\\\bottomrule")
print("\\\\end{tabular}")

# ── Figure 5: Relaxation Ratio box plot ──
fig = go.Figure()
for label, r in R.items():
    rdf = r['relax_df']
    if rdf.empty:
        continue
    fig.add_trace(go.Box(
        y=rdf['ratio'], name=label,
        marker_color=SCENARIOS[label]['color'],
        boxpoints='all', jitter=0.3, pointpos=-1.5))

fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2,
              annotation_text='Bouchaud 2/3', annotation_position='bottom right')
fig.update_layout(
    title='<b>Cross-Scenario: Relaxation Ratio</b>',
    yaxis_title='I_final / I_peak',
    width=900, height=500, template='plotly_white')
fig.write_image(str(FIG_DIR / '5. Relaxation Ratio.png'), scale=2)
fig.show()
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 16 — Section: Stability
# ═══════════════════════════════════════════════════════════════════
cells.append(mc("""\
---
## 3. Stability

A configuration is **stable** if ≥2 of 3 methods agree: trailing slope, two-window mean,
exponential convergence. Fraction stable = share of 30 configs that stabilize.
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 17 — Table 3 + Figure 6: Fraction Stable
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Table 3: Fraction Stable ──

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
print("\\n── Table 3: Fraction Stable (2+/3 votes) ──")
print(table3.to_string(index=False))

# LaTeX
print("\\n── LaTeX ──")
print("\\\\begin{tabular}{lccc}")
print("\\\\toprule")
print("Model & Stable / Total & Fraction \\\\\\\\")
print("\\\\midrule")
for _, r in table3.iterrows():
    print(f"{r['Model']} & {r['Stable']}/{r['Total']} & {r['Fraction']} \\\\\\\\")
print("\\\\bottomrule")
print("\\\\end{tabular}")

# ── Figure 6: Fraction Stable bar chart ──
labels = [r['Model'] for _, r in table3.iterrows()]
fracs  = [float(r['Fraction'].strip('%')) / 100 for _, r in table3.iterrows()]
colors = [SCENARIOS[l]['color'] for l in labels]

fig = go.Figure(go.Bar(x=labels, y=fracs, marker_color=colors, width=0.5))
fig.add_hline(y=0.5, line_dash='dash', line_color='gray', line_width=1.5)
fig.update_layout(
    title='<b>Cross-Scenario: Fraction Stable</b>',
    yaxis_title='Fraction (2+/3 votes)', yaxis_range=[0, 1.05],
    width=700, height=450, template='plotly_white')
fig.write_image(str(FIG_DIR / '6. Fraction Stable.png'), scale=2)
fig.show()
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 18 — Section: Supplementary
# ═══════════════════════════════════════════════════════════════════
cells.append(mc("""\
---
## 4. Supplementary: Volume Scaling (γ)

$\\text{Peak impact} \\propto V^\\gamma$. Square-root law predicts $\\gamma = \\beta \\approx 0.5$.
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 19 — Figure 99: Gamma Distribution
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Figure 99: Gamma Distribution ──

fig = go.Figure()
for label, r in R.items():
    gdf = r['gamma_df']
    if gdf.empty:
        continue
    fig.add_trace(go.Box(
        y=gdf['gamma'], name=label,
        marker_color=SCENARIOS[label]['color'],
        boxpoints='all', jitter=0.3, pointpos=-1.5))

fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2,
              annotation_text='Square-root (0.5)', annotation_position='bottom right')
fig.update_layout(
    title='<b>Cross-Scenario: Gamma Distribution</b>',
    yaxis_title='Gamma', width=900, height=500, template='plotly_white')
fig.write_image(str(FIG_DIR / '99. Gamma Distribution.png'), scale=2)
fig.show()

# Summary
for label, r in R.items():
    gdf = r['gamma_df']
    if not gdf.empty:
        print(f"{label:15s}  γ = {gdf['gamma'].mean():.3f} ± {gdf['gamma'].std():.3f}  (n={len(gdf)})")
"""))

# ═══════════════════════════════════════════════════════════════════
# CELL 20 — Summary
# ═══════════════════════════════════════════════════════════════════
cells.append(cc("""\
# ── Summary ──

print("=" * 70)
print(f"{'Model':15s}  {'β':>6s}  {'Relax':>6s}  {'Stable':>7s}  {'γ':>6s}")
print("-" * 70)
for label, r in R.items():
    beta = r['beta']['beta']
    relax_med = r['relax_df']['ratio'].median() if not r['relax_df'].empty else np.nan
    stable = r['stab_df']['stabilized'].mean() if not r['stab_df'].empty else 0
    gamma = r['gamma_df']['gamma'].mean() if not r['gamma_df'].empty else np.nan
    print(f"{label:15s}  {beta:6.3f}  {relax_med:6.3f}  {stable:6.0%}  {gamma:6.3f}")
print("=" * 70)
print(f"{'Theory':15s}  {'0.500':>6s}  {'0.667':>6s}  {'—':>7s}  {'0.500':>6s}")
"""))


# ═══════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "130.paper_results.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Generated: {out}  ({len(cells)} cells)")
