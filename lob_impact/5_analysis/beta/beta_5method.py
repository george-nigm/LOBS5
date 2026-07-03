#!/usr/bin/env python3
"""
Big beta plot — bet-composition β under 5 volatility normalizations + free intercept.

For each *-beta experiment we pool the per-(sample, insertion) impact cloud and fit, in log-log,

    log(I / sigma) = alpha + beta * log(Q / V)

once for EACH of the 5 daily volatility estimators (Parkinson, Garman-Klass,
Rogers-Satchell, close-to-close, Yang-Zhang). The slope β is the impact exponent
(0.5 = square-root law); α is the intercept (level). The 5-method panel shows how
robust β is to the σ choice (slope barely moves; intercept absorbs the σ scale).

Big figure per experiment: 5 scatter+fit panels (one per σ method) + 1 summary panel
(β ± bootstrap 95% CI across methods, with the √-law reference at 0.5).

  python 5_analysis/beta/beta_5method.py --grid 3_scenarios/results/grid \
         --daily 2_daily_stats/results/<run>/daily_h_l_all.csv [--exp EA-Mamba3-beta]
"""
import os, glob, csv, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vol_estimators import daily_sigmas, METHODS

SENTINEL = 2147483647
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
N_BOOT = 200
LABELS = {'parkinson': 'Parkinson', 'garman_klass': 'Garman-Klass',
          'rogers_satchell': 'Rogers-Satchell', 'close_to_close': 'close-to-close',
          'yang_zhang': 'Yang-Zhang'}


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def _aggr_by_day(side_dir):
    """day -> PER-DAY aggressive_indices_<day>.csv (NOT the summary file = last day only)."""
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def collect_raw(side_dir, ticker, sign):
    """Pool (Q_cum, I_rel, day) per (sample, insertion); σ/V applied later per method."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr_by_day = _aggr_by_day(side_dir)
    if not obs or not aggr_by_day:
        return np.empty((0, 3))
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
        # fail loudly on misaligned aggressive indices: every row must be an execution (event 4)
        # with positive size (an empty book side records a non-positive-size no-op message)
        ev, sz = mm[idx, 1], mm[idx, 3]
        if not np.all(ev == 4) or np.any(sz <= 0):
            print(f'  [WARN] skip {os.path.basename(mf)}: aggressive rows misaligned '
                  f'({np.sum(ev != 4)} non-exec, {np.sum(sz <= 0)} size<=0)')
            continue
        Q = np.cumsum(mm[idx, 3])
        for k, step in enumerate(idx):
            I = sign * (mid[step] - ref) / ref
            if np.isfinite(I) and I > 0 and Q[k] > 0:
                rows.append((Q[k], I, day))
    if not rows:
        return np.empty((0, 3), dtype=object)
    return rows


def fit(x, y, days=None):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if days is not None:
        days = np.asarray(days, dtype=object)[ok]
    if len(x) < 10:
        return np.nan, np.nan, np.nan, len(x), (np.nan, np.nan)
    beta, alpha = np.polyfit(x, y, 1)
    yhat = beta * x + alpha
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    # bootstrap CI on slope — CLUSTERED by day when day labels are available: the ~100
    # insertions within a sample share ref/trajectory/day, so an i.i.d. point bootstrap
    # understates the CI by ~an order of magnitude (N_eff ≈ n/points-per-day).
    bs = np.empty(N_BOOT)
    n = len(x)
    rs = np.random.RandomState(0)
    uniq = np.unique(days) if days is not None else np.array([])
    if len(uniq) >= 3:
        groups = [np.flatnonzero(days == d) for d in uniq]
        for b in range(N_BOOT):
            pick = rs.randint(0, len(groups), len(groups))
            j = np.concatenate([groups[g] for g in pick])
            bs[b] = np.polyfit(x[j], y[j], 1)[0]
    else:
        for b in range(N_BOOT):
            j = rs.randint(0, n, n)
            bs[b] = np.polyfit(x[j], y[j], 1)[0]
    return float(beta), float(alpha), float(r2), n, (float(np.percentile(bs, 2.5)),
                                                     float(np.percentile(bs, 97.5)))


def plot_experiment(grid, exp, sig, out_dir):
    ticker = exp.split('-')[0]
    # aggregated buy + sell (both signed positive) — the richest cloud
    raw = collect_raw(os.path.join(grid, exp, 'buy'), ticker, +1) + \
          collect_raw(os.path.join(grid, exp, 'sell'), ticker, -1)
    if not raw:
        print(f'{exp}: no data'); return None
    Q = np.array([r[0] for r in raw], float)
    I = np.array([r[1] for r in raw], float)
    days = [r[2] for r in raw]
    V = np.array([sig.get((ticker, d), {}).get('V', np.nan) for d in days], float)
    x = np.log(Q / V)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{exp} — bet-composition β under 5 σ-estimators   '
                 f'(log I/σ = α + β·log Q/V,  buy+sell, n={len(Q)})', fontsize=13)
    res = {}
    for ax, method in zip(axes.ravel()[:5], METHODS):
        sg = np.array([sig.get((ticker, d), {}).get(method, np.nan) for d in days], float)
        y = np.log(I / sg)
        beta, alpha, r2, n, ci = fit(x, y, days)
        res[method] = (beta, alpha, r2, n, ci)
        ax.scatter(x, y, s=3, alpha=0.10, color='#5B7FB5')
        if np.isfinite(beta):
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            ax.plot(xx, beta * xx + alpha, 'k', lw=2,
                    label=f'β={beta:.3f} [{ci[0]:.2f},{ci[1]:.2f}]\nα={alpha:.2f}  R²={r2:.2f}')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(LABELS[method], fontsize=11)
        ax.set_xlabel('log(Q / V)'); ax.set_ylabel('log(I / σ)')

    # summary panel: β ± CI across methods
    axs = axes.ravel()[5]
    xs = np.arange(len(METHODS))
    betas = [res[m][0] for m in METHODS]
    los = [res[m][0] - res[m][4][0] for m in METHODS]
    his = [res[m][4][1] - res[m][0] for m in METHODS]
    axs.bar(xs, betas, color='#2F5DA3', yerr=[los, his], capsize=4)
    axs.axhline(0.5, color='#C0392B', ls='--', lw=1.2, label='√-law (β=0.5)')
    axs.set_xticks(xs); axs.set_xticklabels([LABELS[m] for m in METHODS], rotation=20, fontsize=8)
    axs.set_ylabel('β (slope)'); axs.set_title('β across σ-estimators (95% CI)')
    axs.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'beta5__{exp}.png')
    fig.savefig(out, dpi=120); plt.close(fig)
    bs = '  '.join(f'{m[:4]}={res[m][0]:.3f}' for m in METHODS)
    print(f'{exp:22s} {bs}  (n={len(Q)})  -> {os.path.basename(out)}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--exp', default=None)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'results', 'beta_5method')
    exps = [args.exp] if args.exp else sorted(
        e for e in os.listdir(args.grid)
        if e.endswith('-beta') and os.path.isdir(os.path.join(args.grid, e)))
    for e in exps:
        plot_experiment(args.grid, e, sig, out_dir)


if __name__ == '__main__':
    main()
