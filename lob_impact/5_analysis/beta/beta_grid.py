#!/usr/bin/env python3
"""
Big beta grid: 5 sigma-methods (columns) x 3 fit views (rows), per *-beta experiment.

Rows:
  1. all points   — the full per-(sample, insertion) cloud; free-intercept OLS  (beta, R2)
  2. by-k origin  — aggregate to one point per insertion index k (master curve in
                    (log Q/V, log I/sigma) space: median over samples at each k),
                    then OLS THROUGH ORIGIN                                       (beta_origin)
  3. by-k intercept — same by-k master-curve points, free-intercept OLS          (beta, alpha)

Columns: Parkinson, Garman-Klass, Rogers-Satchell, close-to-close, Yang-Zhang.

I  = signed relative mid impact since just before the metaorder (buy +, sell -)
Q_k= cumulative metaorder executed volume up to insertion k;  V = daily exec volume.

  python 5_analysis/beta/beta_grid.py --grid 3_scenarios/results/grid \
         --daily 2_daily_stats/results/<run>/daily_h_l_all.csv [--exp EA-Mamba3-beta]
"""
import os, glob, csv, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vol_estimators import daily_sigmas, METHODS

SENTINEL = 2147483647
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
LABELS = {'parkinson': 'Parkinson', 'garman_klass': 'Garman-Klass',
          'rogers_satchell': 'Rogers-Satchell', 'close_to_close': 'close-to-close',
          'yang_zhang': 'Yang-Zhang'}


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def collect(side_dir, ticker, sign):
    """Per (sample, insertion): (Q_cum, I_rel, day, k).  sigma/V applied later per method."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    ai = glob.glob(os.path.join(side_dir, '**', 'aggressive_indices.csv'), recursive=True)
    if not obs or not ai:
        return []
    aggr = np.loadtxt(ai[0], dtype=int, ndmin=1)
    rows = []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        day = m.group(1)
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask_p, bid_p = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask_p >= SENTINEL) | (bid_p >= SENTINEL) | (ask_p <= 0) | (bid_p <= 0)
        mid = (ask_p + bid_p) / 2.0
        mid[bad] = np.nan
        L = len(mid)
        idx = aggr[aggr < L]
        if len(idx) < 2 or idx[0] < 1:
            continue
        ref = mid[idx[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        mf = ob.replace('orderbook', 'message')
        if not os.path.exists(mf):
            continue
        mm = _read(mf)
        if mm.ndim != 2 or mm.shape[1] < 6:
            continue
        Q = np.cumsum(mm[idx, 3])
        for k, step in enumerate(idx):
            I = sign * (mid[step] - ref) / ref
            if np.isfinite(I) and I > 0 and Q[k] > 0:
                rows.append((Q[k], I, day, k))
    return rows


def ols_origin(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    beta = np.dot(x, y) / np.dot(x, x)
    r2 = 1 - np.sum((y - beta * x) ** 2) / np.sum(y ** 2)
    return float(beta), float(r2), len(x)


def ols_intercept(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    beta, alpha = np.polyfit(x, y, 1)
    yhat = beta * x + alpha
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(beta), float(alpha), float(r2), len(x)


def plot_experiment(grid, exp, sig, out_dir):
    ticker = exp.split('-')[0]
    raw = collect(os.path.join(grid, exp, 'buy'), ticker, +1) + \
          collect(os.path.join(grid, exp, 'sell'), ticker, -1)
    if not raw:
        print(f'{exp}: no data'); return None
    Q = np.array([r[0] for r in raw], float)
    I = np.array([r[1] for r in raw], float)
    days = [r[2] for r in raw]
    kk = np.array([r[3] for r in raw], int)
    V = np.array([sig.get((ticker, d), {}).get('V', np.nan) for d in days], float)
    x_all = np.log(Q / V)

    fig, axes = plt.subplots(3, len(METHODS), figsize=(4 * len(METHODS), 11), squeeze=False)
    fig.suptitle(f'{exp} — bet-composition β grid   (rows: all-points | by-k origin | by-k intercept;'
                 f'  cols: σ-estimators;  buy+sell, n={len(Q)})', fontsize=13)
    summ = {}
    for c, method in enumerate(METHODS):
        sg = np.array([sig.get((ticker, d), {}).get(method, np.nan) for d in days], float)
        y_all = np.log(I / sg)

        # --- Row 1: all points, free-intercept fit ---
        b1, a1, r1, n1 = ols_intercept(x_all, y_all)
        ax = axes[0][c]
        ax.scatter(x_all, y_all, s=3, alpha=0.08, color='#5B7FB5')
        if np.isfinite(b1):
            xx = np.linspace(np.nanmin(x_all), np.nanmax(x_all), 30)
            ax.plot(xx, b1 * xx + a1, 'k', lw=1.8,
                    label=f'β={b1:.3f}\nα={a1:.2f} R²={r1:.2f}')
            ax.legend(fontsize=7, loc='lower right')
        ax.set_title(LABELS[method], fontsize=10)
        if c == 0: ax.set_ylabel('all points\nlog(I/σ)', fontsize=9)

        # --- by-k aggregation: one master-curve point per insertion index k ---
        ku = np.unique(kk[np.isfinite(x_all) & np.isfinite(y_all)])
        xk = np.array([np.median(x_all[(kk == k) & np.isfinite(x_all)]) for k in ku])
        yk = np.array([np.median(y_all[(kk == k) & np.isfinite(y_all)]) for k in ku])

        # --- Row 2: by-k, OLS through origin ---
        bo, ro, no = ols_origin(xk, yk)
        ax = axes[1][c]
        ax.scatter(xk, yk, s=10, color='#2E7D52')
        if np.isfinite(bo):
            xx = np.linspace(np.nanmin(xk), np.nanmax(xk), 30)
            ax.plot(xx, bo * xx, 'k', lw=1.8, label=f'β₀={bo:.3f}\nR²={ro:.2f} k={no}')
            ax.legend(fontsize=7, loc='lower right')
        if c == 0: ax.set_ylabel('by-k · origin\nlog(I/σ)', fontsize=9)

        # --- Row 3: by-k, free intercept ---
        bi, ai_, ri, ni = ols_intercept(xk, yk)
        ax = axes[2][c]
        ax.scatter(xk, yk, s=10, color='#2E7D52')
        if np.isfinite(bi):
            xx = np.linspace(np.nanmin(xk), np.nanmax(xk), 30)
            ax.plot(xx, bi * xx + ai_, 'k', lw=1.8, label=f'β={bi:.3f}\nα={ai_:.2f} R²={ri:.2f}')
            ax.legend(fontsize=7, loc='lower right')
        ax.set_xlabel('log(Q / V)', fontsize=9)
        if c == 0: ax.set_ylabel('by-k · intercept\nlog(I/σ)', fontsize=9)

        summ[method] = dict(all=b1, origin=bo, intercept=bi)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'betagrid__{exp}.png')
    fig.savefig(out, dpi=115); plt.close(fig)
    line = '  '.join(f'{m[:4]}:all={summ[m]["all"]:.2f}/o={summ[m]["origin"]:.2f}/i={summ[m]["intercept"]:.2f}'
                     for m in METHODS)
    print(f'{exp:22s} {line}')
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--exp', default=None)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'results', 'beta_grid')
    exps = [args.exp] if args.exp else sorted(
        e for e in os.listdir(args.grid)
        if e.endswith('-beta') and os.path.isdir(os.path.join(args.grid, e)))
    for e in exps:
        plot_experiment(args.grid, e, sig, out_dir)


if __name__ == '__main__':
    main()
