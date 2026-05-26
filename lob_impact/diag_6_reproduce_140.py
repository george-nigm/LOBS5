#!/usr/bin/env python3
"""
diag_6_reproduce_140.py — Reproduce exact beta computation from notebook 140
on current v4 data, and compare with run_300 logic.

Goal: Determine whether beta difference (0.5 vs 0.82) comes from code
differences or data differences.

Two approaches:
  A) "140-style": load cond+gen, concatenate, shift aggressive indices by junction
  B) "300-style": load gen-only, use aggressive indices directly

If (x, y) coordinates are identical, the code is equivalent and data is the cause.

Usage:
  python lob_impact/diag_6_reproduce_140.py --stock GOOG
  python lob_impact/diag_6_reproduce_140.py --stock GOOG --model Historic
  python lob_impact/diag_6_reproduce_140.py --stock GOOG --max_samples 20
"""
import argparse
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
TICK_SIZE = 100
EPS = 1e-12

V4_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v4')

# Paths for sample_day_map and daily_h_l (try multiple locations)
PROJECT_DIR = Path(__file__).resolve().parent.parent
SDM_CANDIDATES = [
    PROJECT_DIR / 'lob_impact' / 'sample_day_map_{stock}.csv',
]


def find_file(candidates, stock):
    for c in candidates:
        p = Path(str(c).format(stock=stock))
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════
# Data discovery (v4 layout)
# ═══════════════════════════════════════════════════════════════════════
def discover_v4_folders(buy_root, sell_root):
    """Discover config folders for v4 data layout.

    v4 layout: {model}/context_500_{buy,sell}/{stock}/{folder}/exp_*/{data_cond,data_gen}
    """
    rows = []
    pattern = re.compile(r'^i(\d+)_c(\d+)_mb(\d+)_v(\d+)_cntxt(\d+)%$')
    buy_root, sell_root = Path(buy_root), Path(sell_root)
    if not buy_root.exists():
        return pd.DataFrame()
    for bp in sorted(buy_root.iterdir()):
        if not bp.is_dir():
            continue
        m = pattern.match(bp.name)
        if not m:
            continue
        sp = sell_root / bp.name
        if not sp.exists():
            continue
        # Find latest exp_ subfolder
        buy_exp = _find_latest_exp(bp)
        sell_exp = _find_latest_exp(sp)
        rows.append(dict(
            folder=bp.name,
            i=int(m.group(1)), c=int(m.group(2)),
            mb=int(m.group(3)), vol=int(m.group(4)),
            cntxt_pct=int(m.group(5)),
            buy_path=str(buy_exp), sell_path=str(sell_exp),
        ))
    return pd.DataFrame(rows)


def _find_latest_exp(folder):
    """Find the latest exp_* subfolder, or return the folder itself."""
    p = Path(folder)
    exps = sorted(p.glob('exp_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    return exps[0] if exps else p


def load_aggressive_indices(data_path):
    """Load aggressive order indices from aggressive_indices.csv."""
    f = Path(data_path) / 'aggressive_indices.csv'
    if not f.exists():
        return np.array([], dtype=int)
    idx = np.loadtxt(f, dtype=int)
    return np.atleast_1d(idx)


# ═══════════════════════════════════════════════════════════════════════
# Sample discovery
# ═══════════════════════════════════════════════════════════════════════
def discover_samples(data_path, max_samples=None):
    """Discover samples in data_cond by parsing filenames.

    Returns list of (ticker, date, sample_id) tuples.
    """
    cond_dir = Path(data_path) / 'data_cond'
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
        samples = [samples[j] for j in sorted(idx)]
    return samples


# ═══════════════════════════════════════════════════════════════════════
# Load daily params
# ═══════════════════════════════════════════════════════════════════════
def load_sample_day_map(path):
    """Load sample_day_map_{stock}.csv: maps sample_id -> daily params."""
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        sid = int(row['sample_id'])
        H = float(row['highest_price'])
        L = float(row['lowest_price'])
        V = float(row['execution_sum'])
        sigma = float(row['sigma']) if 'sigma' in row.index else (
            np.log(H / L) / 0.8325546 if H > 0 and L > 0 else 1.0
        )
        lookup[sid] = dict(H=H, L=L, V=V, sigma=sigma, day=row['day'])
    return lookup


# ═══════════════════════════════════════════════════════════════════════
# Approach A: 140-style (cond+gen concatenated, junction offset)
# ═══════════════════════════════════════════════════════════════════════
def approach_140(exp_path, aggr_gen, sdm_lookup, max_samples=None):
    """Reproduce exact point cloud extraction from notebook 140.

    1. Load BOTH data_cond and data_gen, concatenate: full_book = vstack([cond, gen])
    2. Store junction = cond_book.shape[0]
    3. Shift indices: aggr_idx = junction + aggr_gen
    4. For each sample: VWAP impact, normalize by sigma and V_day from sample_day_map
    """
    samples = discover_samples(exp_path, max_samples)
    points = []

    for ticker, date, sid in samples:
        cond_bp = Path(exp_path) / f'data_cond/{ticker}_{date}_orderbook_real_id_{sid}.csv'
        gen_bp = Path(exp_path) / f'data_gen/{ticker}_{date}_orderbook_real_id_{sid}_gen_id_0.csv'
        gen_mp = Path(exp_path) / f'data_gen/{ticker}_{date}_message_real_id_{sid}_gen_id_0.csv'
        cond_mp = Path(exp_path) / f'data_cond/{ticker}_{date}_message_real_id_{sid}.csv'

        if not gen_bp.exists() or not cond_bp.exists():
            continue

        cond_book = np.loadtxt(cond_bp, delimiter=',')
        gen_book = np.loadtxt(gen_bp, delimiter=',')
        full_book = np.vstack([cond_book, gen_book])

        cond_msg = np.loadtxt(cond_mp, delimiter=',')
        gen_msg = np.loadtxt(gen_mp, delimiter=',')
        full_msg = np.vstack([cond_msg, gen_msg])

        junction = cond_book.shape[0]

        if sid not in sdm_lookup:
            continue
        dp = sdm_lookup[sid]
        H = dp['H'] / TICK_SIZE
        L = dp['L'] / TICK_SIZE
        V_day = dp['V']
        if H <= L or L <= 0 or V_day <= EPS:
            continue
        sigma = np.log(H / L) / 0.8325546
        alpha = np.log(max(sigma, EPS))

        # Shift aggressive indices by junction (key difference from 300-style)
        aggr_idx = junction + aggr_gen
        aggr_idx = aggr_idx[aggr_idx < len(full_msg)]
        aggr_idx = aggr_idx[aggr_idx < len(full_book)]
        if len(aggr_idx) < 2:
            continue

        sizes = full_msg[aggr_idx, 3].astype(float)
        prices = full_msg[aggr_idx, 4].astype(float)
        ref = (full_book[aggr_idx[0], 0] + full_book[aggr_idx[0], 2]) / 2.0
        if ref <= 0:
            continue

        Q_cum = np.cumsum(sizes)
        vwap = np.cumsum(sizes * prices) / np.maximum(Q_cum, EPS)
        imp = np.abs((vwap - ref) / ref)

        for a in range(len(aggr_idx)):
            if imp[a] > EPS:
                points.append(dict(
                    x=np.log(Q_cum[a] / V_day),
                    y=np.log(imp[a]),
                    alpha=alpha,
                    sigma=sigma,
                    V_day=V_day,
                    Q=float(Q_cum[a]),
                    I=float(imp[a]),
                    sample_id=sid,
                    k=a + 1,
                    date=date,
                    junction=junction,
                    aggr_idx_abs=int(aggr_idx[a]),
                    aggr_idx_gen=int(aggr_gen[a]),
                    size=float(sizes[a]),
                    price=float(prices[a]),
                    ref_mid=float(ref),
                ))
    return pd.DataFrame(points)


# ═══════════════════════════════════════════════════════════════════════
# Approach B: 300-style (gen-only, no junction offset)
# ═══════════════════════════════════════════════════════════════════════
def approach_300(exp_path, aggr_idx, sdm_lookup, max_samples=None):
    """Reproduce run_300 point cloud extraction.

    1. Load only data_gen (no data_cond)
    2. Use aggressive_indices directly (no junction offset)
    3. Same VWAP impact, same normalization
    """
    samples = discover_samples(exp_path, max_samples)
    points = []

    for ticker, date, sid in samples:
        gen_bp = Path(exp_path) / f'data_gen/{ticker}_{date}_orderbook_real_id_{sid}_gen_id_0.csv'
        gen_mp = Path(exp_path) / f'data_gen/{ticker}_{date}_message_real_id_{sid}_gen_id_0.csv'

        if not gen_bp.exists() or not gen_mp.exists():
            continue

        book = np.loadtxt(gen_bp, delimiter=',')
        msg = np.loadtxt(gen_mp, delimiter=',')

        valid_idx = aggr_idx[aggr_idx < len(msg)]
        valid_idx = valid_idx[valid_idx < len(book)]
        if len(valid_idx) < 2:
            continue

        if sid not in sdm_lookup:
            continue
        dp = sdm_lookup[sid]
        H = dp['H'] / TICK_SIZE
        L = dp['L'] / TICK_SIZE
        V_day = dp['V']
        if H <= L or L <= 0 or V_day <= EPS:
            continue
        sigma = np.log(H / L) / 0.8325546
        alpha = np.log(max(sigma, EPS))

        sizes = msg[valid_idx, 3].astype(float)
        prices = msg[valid_idx, 4].astype(float)
        ref = (book[valid_idx[0], 0] + book[valid_idx[0], 2]) / 2.0
        if ref <= 0:
            continue

        Q_cum = np.cumsum(sizes)
        vwap = np.cumsum(sizes * prices) / np.maximum(Q_cum, EPS)
        imp = np.abs((vwap - ref) / ref)

        for a in range(len(valid_idx)):
            if imp[a] > EPS:
                points.append(dict(
                    x=np.log(Q_cum[a] / V_day),
                    y=np.log(imp[a]),
                    alpha=alpha,
                    sigma=sigma,
                    V_day=V_day,
                    Q=float(Q_cum[a]),
                    I=float(imp[a]),
                    sample_id=sid,
                    k=a + 1,
                    date=date,
                    junction=0,  # no junction in 300-style
                    aggr_idx_abs=int(valid_idx[a]),
                    aggr_idx_gen=int(valid_idx[a]),
                    size=float(sizes[a]),
                    price=float(prices[a]),
                    ref_mid=float(ref),
                ))
    return pd.DataFrame(points)


# ═══════════════════════════════════════════════════════════════════════
# Beta computation (both 140-style and 300-style)
# ═══════════════════════════════════════════════════════════════════════
def compute_beta_140(df):
    """Exact 140 beta: y_adj = y - alpha, beta = dot(x, y_adj) / dot(x, x)."""
    if df.empty:
        return dict(beta=np.nan, r2=np.nan, n=0)
    y_adj = df['y'].values - df['alpha'].values
    x = df['x'].values
    ok = np.isfinite(x) & np.isfinite(y_adj) & (x != 0)
    xv, yv = x[ok], y_adj[ok]
    if len(xv) < 2:
        return dict(beta=np.nan, r2=np.nan, n=0)
    beta = float(np.dot(xv, yv) / np.dot(xv, xv))
    ss_res = np.sum((yv - beta * xv) ** 2)
    ss_tot = np.sum(yv ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(beta=beta, r2=r2, n=int(ok.sum()))


def compute_beta_300(df):
    """300-style: three estimators (origin, intercept, ratio).

    Uses the same columns (x, y, alpha) as 140-style for fair comparison.
    """
    if df.empty:
        return dict(beta_origin=np.nan, beta_intercept=np.nan,
                    beta_ratio=np.nan, alpha_coeff=np.nan, r2_origin=np.nan,
                    r2_intercept=np.nan, n=0)

    x = df['x'].values
    y_adj = df['y'].values - df['alpha'].values  # log(I/sigma)
    y_raw = df['y'].values                        # log(I)

    ok = np.isfinite(x) & np.isfinite(y_adj) & np.isfinite(y_raw) & (x != 0)
    xv, yv_adj, yv_raw = x[ok], y_adj[ok], y_raw[ok]
    if len(xv) < 2:
        return dict(beta_origin=np.nan, beta_intercept=np.nan,
                    beta_ratio=np.nan, alpha_coeff=np.nan,
                    r2_origin=np.nan, r2_intercept=np.nan, n=0)

    # 1. Origin: log(I/sigma) = beta * log(Q/V)
    beta_origin = float(np.dot(xv, yv_adj) / np.dot(xv, xv))
    ss_res_o = np.sum((yv_adj - beta_origin * xv) ** 2)
    ss_tot_o = np.sum(yv_adj ** 2)
    r2_origin = 1 - ss_res_o / ss_tot_o if ss_tot_o > 0 else 0.0

    # 2. Intercept: log(I) = alpha + beta * log(Q/V)
    coeffs = np.polyfit(xv, yv_raw, 1)
    beta_intercept = float(coeffs[0])
    alpha_coeff = float(coeffs[1])
    yhat = beta_intercept * xv + alpha_coeff
    ss_res_i = np.sum((yv_raw - yhat) ** 2)
    ss_tot_i = np.sum((yv_raw - np.mean(yv_raw)) ** 2)
    r2_intercept = 1 - ss_res_i / ss_tot_i if ss_tot_i > 0 else 0.0

    # 3. Ratio: mean(log(I/sigma) / log(Q/V))
    beta_ratio = float(np.mean(yv_adj / xv))

    return dict(
        beta_origin=beta_origin,
        beta_intercept=beta_intercept,
        beta_ratio=beta_ratio,
        alpha_coeff=alpha_coeff,
        r2_origin=r2_origin,
        r2_intercept=r2_intercept,
        n=int(ok.sum()),
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def run_diagnostic(stock, model, max_samples, out_dir):
    print(f'diag_6_reproduce_140: stock={stock}, model={model}, max_samples={max_samples}')
    print(f'v4 base: {V4_BASE}')

    # Load sample_day_map
    sdm_path = find_file(SDM_CANDIDATES, stock)
    if sdm_path is None:
        print(f'ERROR: sample_day_map_{stock}.csv not found')
        sys.exit(1)
    sdm_lookup = load_sample_day_map(sdm_path)
    print(f'sample_day_map: {len(sdm_lookup)} entries from {sdm_path}')

    # Discover v4 folders
    buy_root = V4_BASE / model / 'context_500_buy' / stock
    sell_root = V4_BASE / model / 'context_500_sell' / stock

    grid_df = discover_v4_folders(buy_root, sell_root)
    if grid_df.empty:
        print(f'ERROR: no v4 folders found for {model}/{stock}')
        sys.exit(1)
    print(f'Discovered {len(grid_df)} config folders')

    # Filter to configs with i >= 2 (need cumulative variation)
    grid_df = grid_df[grid_df['i'] >= 2].reset_index(drop=True)
    print(f'After filtering i>=2: {len(grid_df)} configs')

    # Process all folders, both directions
    all_140_points = []
    all_300_points = []

    for idx, row in grid_df.iterrows():
        folder = row['folder']
        bp = row['buy_path']
        sp = row['sell_path']

        aggr_gen = load_aggressive_indices(bp)
        aggr_sell = load_aggressive_indices(sp)
        if len(aggr_gen) < 2 or len(aggr_sell) < 2:
            continue

        for direction, exp_path, aggr in [
            ('buy', bp, aggr_gen),
            ('sell', sp, aggr_sell),
        ]:
            # Approach A: 140-style
            pc_140 = approach_140(exp_path, aggr, sdm_lookup, max_samples)
            if not pc_140.empty:
                pc_140['direction'] = direction
                pc_140['folder'] = folder
                all_140_points.append(pc_140)

            # Approach B: 300-style
            pc_300 = approach_300(exp_path, aggr, sdm_lookup, max_samples)
            if not pc_300.empty:
                pc_300['direction'] = direction
                pc_300['folder'] = folder
                all_300_points.append(pc_300)

        if (idx + 1) % 5 == 0:
            print(f'  Processed {idx + 1}/{len(grid_df)} folders...')

    # Combine
    df_140 = pd.concat(all_140_points, ignore_index=True) if all_140_points else pd.DataFrame()
    df_300 = pd.concat(all_300_points, ignore_index=True) if all_300_points else pd.DataFrame()

    print(f'\n140-style points: {len(df_140)}')
    print(f'300-style points: {len(df_300)}')

    # Compute betas
    beta_140 = compute_beta_140(df_140)
    beta_300_full = compute_beta_300(df_300)
    beta_300_origin_from_140 = compute_beta_140(df_300)  # 140 estimator on 300 data

    # Also compute 300-style betas on 140 data
    beta_140_full = compute_beta_300(df_140)

    # ── Output report ──
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'diag_6_reproduce_140.txt'

    lines = []
    lines.append('=' * 72)
    lines.append(f'diag_6_reproduce_140 — stock={stock}, model={model}')
    lines.append(f'max_samples={max_samples}')
    lines.append('=' * 72)

    lines.append('')
    lines.append('── Beta Results ──')
    lines.append(f'{"Approach":<30} {"n_pts":>8} {"beta_origin":>12} {"beta_intcpt":>12} '
                 f'{"beta_ratio":>12} {"alpha":>10} {"R2_orig":>10} {"R2_intcpt":>10}')
    lines.append('-' * 120)

    # 140-style with 140 estimator
    lines.append(f'{"A) 140-style, 140-est":<30} {beta_140["n"]:>8} {beta_140["beta"]:>12.6f} '
                 f'{"n/a":>12} {"n/a":>12} {"n/a":>10} {beta_140["r2"]:>10.4f} {"n/a":>10}')

    # 140-style with all estimators
    lines.append(f'{"A) 140-style, all-est":<30} {beta_140_full["n"]:>8} '
                 f'{beta_140_full["beta_origin"]:>12.6f} {beta_140_full["beta_intercept"]:>12.6f} '
                 f'{beta_140_full["beta_ratio"]:>12.6f} {beta_140_full["alpha_coeff"]:>10.4f} '
                 f'{beta_140_full["r2_origin"]:>10.4f} {beta_140_full["r2_intercept"]:>10.4f}')

    # 300-style with 140 estimator
    lines.append(f'{"B) 300-style, 140-est":<30} {beta_300_origin_from_140["n"]:>8} '
                 f'{beta_300_origin_from_140["beta"]:>12.6f} {"n/a":>12} {"n/a":>12} '
                 f'{"n/a":>10} {beta_300_origin_from_140["r2"]:>10.4f} {"n/a":>10}')

    # 300-style with all estimators
    lines.append(f'{"B) 300-style, all-est":<30} {beta_300_full["n"]:>8} '
                 f'{beta_300_full["beta_origin"]:>12.6f} {beta_300_full["beta_intercept"]:>12.6f} '
                 f'{beta_300_full["beta_ratio"]:>12.6f} {beta_300_full["alpha_coeff"]:>10.4f} '
                 f'{beta_300_full["r2_origin"]:>10.4f} {beta_300_full["r2_intercept"]:>10.4f}')

    # ── Point-by-point comparison ──
    lines.append('')
    lines.append('── Point-by-point comparison (first 10 matching points) ──')
    lines.append(f'{"sid":>6} {"k":>3} {"x_140":>12} {"x_300":>12} {"dx":>12} '
                 f'{"y_140":>12} {"y_300":>12} {"dy":>12} {"same":>6}')
    lines.append('-' * 100)

    # Merge once and reuse for both point-by-point comparison and conclusion
    merged = None
    dx_all = dy_all = None
    if not df_140.empty and not df_300.empty:
        merge_cols = ['sample_id', 'k', 'folder', 'direction']
        merged = df_140.merge(df_300, on=merge_cols, suffixes=('_140', '_300'), how='inner')
        lines.append(f'Matched points: {len(merged)} (out of 140={len(df_140)}, 300={len(df_300)})')
        lines.append('')

        if len(merged) > 0:
            dx_all = merged['x_140'] - merged['x_300']
            dy_all = merged['y_140'] - merged['y_300']

            n_show = min(10, len(merged))
            for j in range(n_show):
                r = merged.iloc[j]
                dx = dx_all.iloc[j]
                dy = dy_all.iloc[j]
                same = 'YES' if abs(dx) < 1e-10 and abs(dy) < 1e-10 else 'NO'
                lines.append(f'{r["sample_id"]:>6} {r["k"]:>3} '
                             f'{r["x_140"]:>12.6f} {r["x_300"]:>12.6f} {dx:>12.2e} '
                             f'{r["y_140"]:>12.6f} {r["y_300"]:>12.6f} {dy:>12.2e} '
                             f'{same:>6}')

            lines.append('')
            lines.append(f'dx stats: mean={dx_all.mean():.2e}, std={dx_all.std():.2e}, '
                         f'max_abs={dx_all.abs().max():.2e}')
            lines.append(f'dy stats: mean={dy_all.mean():.2e}, std={dy_all.std():.2e}, '
                         f'max_abs={dy_all.abs().max():.2e}')

            lines.append('')
            lines.append('── Junction analysis ──')
            lines.append(f'140 junction values: {sorted(merged["junction_140"].unique()[:5])}')
            lines.append(f'140 aggr_idx_abs values (first 5): '
                         f'{merged["aggr_idx_abs_140"].values[:5].tolist()}')
            lines.append(f'300 aggr_idx_abs values (first 5): '
                         f'{merged["aggr_idx_abs_300"].values[:5].tolist()}')
            lines.append(f'140 aggr_idx_gen values (first 5): '
                         f'{merged["aggr_idx_gen_140"].values[:5].tolist()}')
            lines.append(f'300 aggr_idx_gen values (first 5): '
                         f'{merged["aggr_idx_gen_300"].values[:5].tolist()}')

            lines.append('')
            lines.append('── Size/price comparison ──')
            d_size = merged['size_140'] - merged['size_300']
            d_price = merged['price_140'] - merged['price_300']
            d_ref = merged['ref_mid_140'] - merged['ref_mid_300']
            lines.append(f'd_size:  mean={d_size.mean():.2e}, max_abs={d_size.abs().max():.2e}')
            lines.append(f'd_price: mean={d_price.mean():.2e}, max_abs={d_price.abs().max():.2e}')
            lines.append(f'd_ref:   mean={d_ref.mean():.2e}, max_abs={d_ref.abs().max():.2e}')

            exact_match = (dx_all.abs() < 1e-10) & (dy_all.abs() < 1e-10)
            lines.append(f'\nExact match: {exact_match.sum()}/{len(merged)} '
                         f'({exact_match.mean()*100:.1f}%)')
    else:
        lines.append('Could not compare: one or both approaches returned empty results.')

    # ── Conclusion ──
    lines.append('')
    lines.append('── Conclusion ──')
    if merged is not None and len(merged) > 0:
        if dx_all.abs().max() < 1e-6 and dy_all.abs().max() < 1e-6:
            lines.append('SAME: 140 and 300 produce identical (x, y) coordinates.')
            lines.append('=> Beta difference is due to DATA (2023 vs 2026), not code.')
        else:
            lines.append('DIFFERENT: 140 and 300 produce different (x, y) coordinates.')
            lines.append('=> Code logic difference exists between approaches.')
            if dx_all.abs().max() > 1e-6:
                lines.append(f'   x differs by up to {dx_all.abs().max():.6f}')
            if dy_all.abs().max() > 1e-6:
                lines.append(f'   y differs by up to {dy_all.abs().max():.6f}')
    elif merged is not None:
        lines.append('NO MATCHING POINTS: cannot compare approaches.')
    else:
        lines.append('INCOMPLETE: one or both approaches returned no data.')

    lines.append(f'\nbeta_140_origin   = {beta_140["beta"]:.6f}')
    lines.append(f'beta_300_origin   = {beta_300_origin_from_140["beta"]:.6f}')
    lines.append(f'beta_300_intercept = {beta_300_full["beta_intercept"]:.6f}')

    report = '\n'.join(lines)
    print(report)

    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='diag_6: Reproduce 140-style beta on v4 data, compare with 300-style')
    parser.add_argument('--stock', type=str, default='GOOG')
    parser.add_argument('--model', type=str, default='Historic',
                        help='Model to analyze (default: Historic)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per config folder (default: all)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: pics_for_investigation/)')
    args = parser.parse_args()

    out_dir = args.out_dir or str(PROJECT_DIR / 'pics_for_investigation')
    run_diagnostic(args.stock, args.model, args.max_samples, out_dir)
