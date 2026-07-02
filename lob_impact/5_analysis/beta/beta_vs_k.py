#!/usr/bin/env python3
"""
β as a function of insertion index k — beta_k for k = 1..N_insertions.

At each insertion k we take ALL samples (64 buy + 64 sell) at THAT insertion and fit

    log(I / sigma) = [alpha +] beta_k * log(Q / V)

across that cross-section (the spread in Q/V at fixed k comes from the different days'
daily volume V and the small per-sample size capping). This yields one beta per k -> a
curve beta_k vs k, expected to sit in [0,1] (0.5 = square-root law).

Both estimators per panel:  origin (through 0, dashed)  and  free-intercept (solid).
5 panels = 5 sigma-estimators (Parkinson, Garman-Klass, Rogers-Satchell, c2c, Yang-Zhang).

  python 5_analysis/beta/beta_vs_k.py --grid 3_scenarios/results/grid \
         --daily 2_daily_stats/results/<run>/daily_h_l_all.csv [--exp EA-Mamba3-beta]
"""
import os, glob, csv, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vol_estimators import daily_sigmas, METHODS

SENTINEL = 2147483647
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
MIN_PTS = 8          # min cross-section points to fit a beta at a given k
MIN_XSPREAD = 0.05   # min log(Q/V) spread, else slope is ill-conditioned
LABELS = {'parkinson': 'Parkinson', 'garman_klass': 'Garman-Klass',
          'rogers_satchell': 'Rogers-Satchell', 'close_to_close': 'close-to-close',
          'yang_zhang': 'Yang-Zhang'}


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def _aggr_by_day(side_dir):
    """day -> PER-DAY aggressive_indices_<day>.csv (NOT the summary file = last day only;
    mb varies per day so the summary mis-aligns the insertion rows on other days)."""
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def collect(side_dir, ticker, sign):
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr_by_day = _aggr_by_day(side_dir)
    if not obs or not aggr_by_day:
        return []
    rows = []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        day = m.group(1)
        aggr = aggr_by_day.get(day)
        if aggr is None:
            continue
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


def beta_at(x, y, intercept):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < MIN_PTS or (x.max() - x.min()) < MIN_XSPREAD:
        return np.nan
    if intercept:
        return float(np.polyfit(x, y, 1)[0])
    return float(np.dot(x, y) / np.dot(x, x))


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
    ks = np.arange(kk.max() + 1)

    fig, axes = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 4.6), squeeze=False)
    fig.suptitle(f'{exp} — β as a function of insertion index k   '
                 f'(cross-section of all samples at each k; origin vs intercept)', fontsize=12)
    means = {}
    for c, method in enumerate(METHODS):
        sg = np.array([sig.get((ticker, d), {}).get(method, np.nan) for d in days], float)
        y_all = np.log(I / sg)
        b_int, b_org = [], []
        for k in ks:
            sel = kk == k
            b_int.append(beta_at(x_all[sel], y_all[sel], True))
            b_org.append(beta_at(x_all[sel], y_all[sel], False))
        b_int = np.array(b_int); b_org = np.array(b_org)
        ax = axes[0][c]
        ax.axhspan(0, 1, color='#EEEEEE', zorder=0)
        ax.plot(ks, b_int, color='#2F5DA3', lw=1.3, label='intercept')
        ax.plot(ks, b_org, color='#C0392B', lw=1.0, ls='--', label='origin')
        ax.axhline(0.5, color='#2E7D52', ls=':', lw=1.0)
        ax.set_title(f'{LABELS[method]}\nβ̄={np.nanmean(b_int):.2f} (int)', fontsize=10)
        ax.set_xlabel('insertion k');
        if c == 0: ax.set_ylabel('β_k')
        ax.set_ylim(-0.3, 1.5); ax.legend(fontsize=7)
        means[method] = (np.nanmean(b_int), np.nanmean(b_org))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'beta_vs_k__{exp}.png')
    fig.savefig(out, dpi=120); plt.close(fig)
    line = '  '.join(f'{m[:4]}:int={means[m][0]:.2f}/org={means[m][1]:.2f}' for m in METHODS)
    print(f'{exp:22s} {line}')
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--exp', default=None)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'results', 'beta_vs_k')
    exps = [args.exp] if args.exp else sorted(
        e for e in os.listdir(args.grid)
        if e.endswith('-beta') and os.path.isdir(os.path.join(args.grid, e)))
    for e in exps:
        plot_experiment(args.grid, e, sig, out_dir)


if __name__ == '__main__':
    main()
