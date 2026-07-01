#!/usr/bin/env python3
"""
β_intercept(k) with σ=1 — one panel per stock, one curve per model.

Reproduces the paper-style "<STOCK>, sigma=1, intercept" figure: at each insertion
index k we pool ALL impact points from insertions 1..k (across every sample & day) and
fit the FREE-INTERCEPT slope

    log(I) = α + β_k · log(Q / V)        (σ = 1 — no volatility normalisation)

This is the CUMULATIVE β (pool k'≤k), which converges smoothly to the asymptotic
β as k grows — unlike the per-k cross-section (beta_vs_k.py), which is noisy and, for
the constant-σ estimators (close-to-close / yang-zhang), collapses onto exactly this
σ=1 slope anyway (their per-ticker-constant σ is absorbed by the intercept). So σ=1 IS
the honest, non-degenerate version of those panels.

  V (daily volume) still normalises x; only σ is dropped from y.
  The legend shows each model's final (k=max) β.

  python 5_analysis/beta/beta_sigma1_by_model.py --grid 3_scenarios/results/grid \
         --daily 2_daily_stats/results/<run>/daily_h_l_all.csv
"""
import os, glob, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vol_estimators import daily_sigmas
from beta_vs_k import collect, MIN_PTS, MIN_XSPREAD

# stable per-model styling; unknown models fall back to the cycle
MODEL_STYLE = {
    'Historic':  dict(color='#C0392B', ls='-',  lw=1.6),
    'Heuristic': dict(color='#7F8C8D', ls=':',  lw=1.4),
    'CST':       dict(color='#27AE60', ls='-.', lw=1.4),
    'Mamba3':    dict(color='#2F5DA3', ls='-',  lw=1.6),
}
_FALLBACK = ['#8E44AD', '#E67E22', '#16A085', '#2C3E50', '#D35400']


def _style(model, i):
    return MODEL_STYLE.get(model, dict(color=_FALLBACK[i % len(_FALLBACK)], ls='-', lw=1.4))


def fit_intercept(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < MIN_PTS or (x.max() - x.min()) < MIN_XSPREAD:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def cumulative_beta(grid, exp, sig):
    """Return (ks, beta_cum) for one experiment under σ=1, intercept, pooled k'≤k."""
    ticker = exp.split('-')[0]
    raw = collect(os.path.join(grid, exp, 'buy'), ticker, +1) + \
          collect(os.path.join(grid, exp, 'sell'), ticker, -1)
    if not raw:
        return None
    Q = np.array([r[0] for r in raw], float)
    I = np.array([r[1] for r in raw], float)
    days = [r[2] for r in raw]
    kk = np.array([r[3] for r in raw], int)
    V = np.array([sig.get((ticker, d), {}).get('V', np.nan) for d in days], float)
    x = np.log(Q / V)        # participation-rate normalised
    y = np.log(I)            # σ = 1
    ks = np.arange(kk.max() + 1)
    beta = np.array([fit_intercept(x[kk <= k], y[kk <= k]) for k in ks])
    return ks, beta


def discover(grid, stock):
    """Models with a beta experiment for this stock: '<stock>-<Model>-beta'."""
    out = []
    for d in sorted(os.listdir(grid)):
        m = re.fullmatch(rf'{re.escape(stock)}-(.+)-beta', d)
        if m and os.path.isdir(os.path.join(grid, d)):
            out.append(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stocks', default='EA,NVDA,AMD')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    stocks = [s for s in args.stocks.split(',') if s]
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'results', 'beta_sigma1')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(stocks), figsize=(5 * len(stocks), 4.6), squeeze=False)
    for c, stock in enumerate(stocks):
        ax = axes[0][c]
        models = discover(args.grid, stock)
        for i, model in enumerate(models):
            res = cumulative_beta(args.grid, f'{stock}-{model}-beta', sig)
            if res is None:
                print(f'{stock}-{model}: no data'); continue
            ks, beta = res
            fin = beta[np.isfinite(beta)]
            final = fin[-1] if fin.size else np.nan
            st = _style(model, i)
            ax.plot(ks, beta, label=f'{model} ({final:.2f})', **st)
            print(f'{stock:5s} {model:10s} final β={final:.3f} mean β={np.nanmean(beta):.3f}')
        ax.axhline(0.5, color='red', ls='--', lw=1.0)
        ax.set_ylim(0.0, 1.0); ax.set_xlabel('insertion k')
        if c == 0:
            ax.set_ylabel(r'$\beta_{intercept}$')
        ax.set_title(f'{stock}, sigma=1, intercept', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    out = os.path.join(out_dir, 'beta_sigma1_intercept_by_stock.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print('->', out)


if __name__ == '__main__':
    main()
