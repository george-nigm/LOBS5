#!/usr/bin/env python3
"""
Bet-composition beta: the master-curve impact exponent within one metaorder.

For the bet_composition shape (100 insertions, 0 cooling) the metaorder accumulates
volume Q_k = sum of executed child sizes up to insertion k. The price impact grows as

    I(Q) ~ Q^beta          (beta ~ 0.5 = square-root law)

We fit, in log-log, the per-(sample,insertion) cloud:

    y = log(I / sigma_day)   vs   x = log(Q_k / V_day)
    y = alpha + beta * x

I       = signed relative mid impact since just before the metaorder (buy: +, sell: -)
sigma   = Parkinson daily vol  ln(H/L)/1.6651092   (the ONLY estimator our daily data supports;
          Garman-Klass / Rogers-Satchell / Yang-Zhang need OHLC we don't yet store)
V       = daily executed volume (execution_sum)
Q_k     = cumulative metaorder executed volume up to insertion k

Empty-book-collapse rows (sentinel ask/bid) -> NaN, excluded from the fit.

  python 5_analysis/beta/beta_master_curve.py --grid 3_scenarios/results/grid \
         --daily 2_daily_stats/results/daily_20260603-213810/daily_h_l_all.csv --exp EA-Mamba3-beta
  (omit --exp to do every *-beta experiment)
"""
import os, glob, csv, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICK = 100
SENTINEL = 2147483647
PARKINSON = 1.6651092
EXECS = (4, 5)
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def load_daily(daily_csv):
    """(ticker, day) -> (sigma_parkinson, V)."""
    out = {}
    with open(daily_csv) as fh:
        for row in csv.DictReader(fh):
            H, L = float(row['highest_price']), float(row['lowest_price'])
            sig = np.log(H / L) / PARKINSON if H > 0 and L > 0 and H > L else np.nan
            out[(row['ticker'], row['day'])] = (sig, float(row['execution_sum']))
    return out


def collect_points(side_dir, ticker, daily, sign):
    """Pool (x=log(Q/V), y=log(I/sigma)) points over all samples in a side."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr_by_day = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):  # PER-DAY indices, not the last-day summary file
            aggr_by_day[d] = np.loadtxt(f, dtype=int, ndmin=1)
    if not obs or not aggr_by_day:
        return np.array([]), np.array([])
    xs, ys = [], []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        day = m.group(1)
        key = (ticker, day)
        if key not in daily:
            continue
        aggr = aggr_by_day.get(day)
        if aggr is None:
            continue
        sigma, V = daily[key]
        if not np.isfinite(sigma) or sigma <= 0 or V <= 0:
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
        if len(idx) < 2:
            continue
        ref = mid[idx[0] - 1] if idx[0] >= 1 else mid[0]
        if not np.isfinite(ref) or ref <= 0:
            continue
        # cumulative metaorder volume: read sizes from the paired message file at aggr positions
        mf = ob.replace('orderbook', 'message')
        if not os.path.exists(mf):
            continue
        mm = _read(mf)
        if mm.ndim != 2 or mm.shape[1] < 6:
            continue
        meta_sz = mm[idx, 3]
        Q_cum = np.cumsum(meta_sz)
        for k, step in enumerate(idx):
            I = sign * (mid[step] - ref) / ref       # signed relative impact
            if not np.isfinite(I) or I <= 0 or Q_cum[k] <= 0:
                continue
            xs.append(np.log(Q_cum[k] / V))
            ys.append(np.log(I / sigma))
    return np.array(xs), np.array(ys)


def fit_beta(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return dict(beta=np.nan, alpha=np.nan, r2=np.nan, n=len(x))
    beta, alpha = np.polyfit(x, y, 1)
    yhat = beta * x + alpha
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(beta=float(beta), alpha=float(alpha), r2=float(r2), n=len(x))


def plot_experiment(grid, exp, daily, out_dir):
    ticker = exp.split('-')[0]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f'{exp} — bet-composition β   (log I/σ  vs  log Q/V, Parkinson σ)', fontsize=12)
    summary = {}
    for j, (d, sign, color) in enumerate([('buy', +1, '#2F5DA3'), ('sell', -1, '#C0392B')]):
        x, y = collect_points(os.path.join(grid, exp, d), ticker, daily, sign)
        r = fit_beta(x, y)
        summary[d] = r
        ax[j].scatter(x, y, s=4, alpha=0.15, color=color)
        if np.isfinite(r['beta']):
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            ax[j].plot(xx, r['beta'] * xx + r['alpha'], color='k', lw=2,
                       label=f"β={r['beta']:.3f}  α={r['alpha']:.2f}  R²={r['r2']:.2f}  n={r['n']}")
        ax[j].set_title(f'{d}'); ax[j].set_xlabel('log(Q / V)'); ax[j].set_ylabel('log(I / σ)')
        ax[j].legend(fontsize=8, loc='lower right')
    # combined buy+sell (both signed positive -> one cloud)
    xb, yb = collect_points(os.path.join(grid, exp, 'buy'), ticker, daily, +1)
    xs2, ys2 = collect_points(os.path.join(grid, exp, 'sell'), ticker, daily, -1)
    x, y = np.concatenate([xb, xs2]), np.concatenate([yb, ys2])
    r = fit_beta(x, y)
    summary['agg'] = r
    ax[2].scatter(xb, yb, s=4, alpha=0.12, color='#2F5DA3', label='buy')
    ax[2].scatter(xs2, ys2, s=4, alpha=0.12, color='#C0392B', label='sell')
    if np.isfinite(r['beta']):
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax[2].plot(xx, r['beta'] * xx + r['alpha'], color='k', lw=2.5,
                   label=f"β={r['beta']:.3f}  α={r['alpha']:.2f}  R²={r['r2']:.2f}  n={r['n']}")
    ax[2].set_title('buy + sell (aggregated)'); ax[2].set_xlabel('log(Q / V)'); ax[2].set_ylabel('log(I / σ)')
    ax[2].legend(fontsize=8, loc='lower right')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'beta__{exp}.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f'{exp:24s} buy β={summary["buy"]["beta"]:.3f} | sell β={summary["sell"]["beta"]:.3f} '
          f'| agg β={summary["agg"]["beta"]:.3f} (R²={summary["agg"]["r2"]:.2f}, n={summary["agg"]["n"]})')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--exp', default=None)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    daily = load_daily(args.daily)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'results', 'beta_master')
    if args.exp:
        exps = [args.exp]
    else:
        exps = sorted(e for e in os.listdir(args.grid)
                      if e.endswith('-beta') and os.path.isdir(os.path.join(args.grid, e)))
    for e in exps:
        plot_experiment(args.grid, e, daily, out_dir)


if __name__ == '__main__':
    main()
