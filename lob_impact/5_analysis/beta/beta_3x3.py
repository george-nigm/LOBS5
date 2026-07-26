#!/usr/bin/env python3
"""
3x3 exponent dynamics: THREE honest estimators x THREE cross-sections, delta(k) per model.

Rows (methods):
  binned — signed quantile-bin means then log-fit over positive bins (headline)
  L2     — direct fit y = a*(Q/V)^d on signed raw points, grid over d, closed-form a
  L1     — same grid, weighted-median amplitude (robust twin)
Cols (cross-sections of the k-th insertion):
  <=k  (cumulative pool),   ==k  (exact slice),   >=k  (reverse-cumulative)

x = ln(Q/V), y = I/sigma_parkinson. No cloud panels (see beta_3views for those).
All curves cached to npz next to the png.

  python beta_3x3.py --grid <root> --daily <daily.csv> --stock NVDA \
      [--models ...] [--ks 5:100:5]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beta_grid import collect
from beta_binned import bin_means
from vol_estimators import daily_sigmas

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E',
          'OW': '#827717', 'QR': '#00ACC1'}
DGRID = np.arange(0.05, 1.51, 0.01)
NBINS = 18
MIN_PTS = 200
CAP_L2, CAP_L1 = 200_000, 60_000


def delta_binned(x, y):
    xc, ym, _ = bin_means(x, y, NBINS)
    pos = ym > 0
    if pos.sum() < 5 or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan
    return float(np.polyfit(xc[pos], np.log(ym[pos]), 1)[0])


def _sub(x, y, cap, rng):
    if x.size <= cap:
        return x, y
    i = rng.choice(x.size, size=cap, replace=False)
    return x[i], y[i]


def delta_l2(x, y, rng):
    x, y = _sub(x, y, CAP_L2, rng)
    best = (np.inf, np.nan)
    for d in DGRID:
        t = np.exp(d * x)
        a = float(np.dot(y, t) / np.dot(t, t))
        s = float(np.sum((y - a * t) ** 2))
        if s < best[0]:
            best = (s, float(d))
    return best[1]


def delta_l1(x, y, rng):
    x, y = _sub(x, y, CAP_L1, rng)
    best = (np.inf, np.nan)
    for d in DGRID:
        t = np.exp(d * x)
        r = y / t
        idx = np.argsort(r)
        cw = np.cumsum(t[idx])
        a = float(r[idx][np.searchsorted(cw, 0.5 * cw[-1])])
        s = float(np.sum(np.abs(y - a * t)))
        if s < best[0]:
            best = (s, float(d))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', required=True)
    ap.add_argument('--models', default='Historic,Heuristic,Propagator,OW,CST,NMZI,Hawkes,QR,'
                                        'Mamba3,GDN,S5_120M,S5,Mamba3_4k,S5_4k')
    ap.add_argument('--ks', default='5:100:5')
    ap.add_argument('--trim', type=float, default=0.0,
                    help='top-fraction of |y| to drop per fitted pool (outlier experiment)')
    ap.add_argument('--tag', default='', help='output filename suffix, e.g. trim95')
    args = ap.parse_args()
    lo, hi, st = (int(v) for v in args.ks.split(':'))
    ks = np.arange(lo, hi + 1, st)
    sig = daily_sigmas(args.daily)
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', 'beta_3x3')
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    methods = [('binned', 'signed binned (headline)', lambda x, y: delta_binned(x, y)),
               ('l2', r'direct $L_2$ on signed points', lambda x, y: delta_l2(x, y, rng)),
               ('l1', r'direct $L_1$ on signed points', lambda x, y: delta_l1(x, y, rng))]
    views = [('le', r'cumulative  $\leq k$', lambda kc, k: kc <= k),
             ('eq', r'exact  $=k$', lambda kc, k: kc == k),
             ('ge', r'reverse  $\geq k$', lambda kc, k: kc >= k)]

    cache = {'ks': ks}
    curves = {}
    for model in [m for m in args.models.split(',') if m]:
        raw = collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'sell'), args.stock, -1)
        if not raw:
            print(f'{model}: no data', flush=True); continue
        Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
        days = np.array([r[2] for r in raw]); kc = np.array([r[3] for r in raw], int) + 1
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        sp = np.array([sig.get((args.stock, d), {}).get('parkinson', np.nan) for d in days], float)
        x_all = np.log(Q / V); y_all = I / sp
        ok = np.isfinite(x_all) & np.isfinite(y_all)
        x_all, y_all, kc = x_all[ok], y_all[ok], kc[ok]
        print(f'{model}: {x_all.size:,} points', flush=True)
        for mkey, _, fn in methods:
            for vkey, _, sel in views:
                vals = np.full(len(ks), np.nan)
                for i, k in enumerate(ks):
                    m = sel(kc, int(k))
                    if m.sum() >= MIN_PTS:
                        xx, yy = x_all[m], y_all[m]
                        if args.trim > 0:
                            keep = np.abs(yy) <= np.quantile(np.abs(yy), 1.0 - args.trim)
                            xx, yy = xx[keep], yy[keep]
                        vals[i] = fn(xx, yy)
                curves[(model, mkey, vkey)] = vals
                cache[f'{model}_{mkey}_{vkey}'] = vals
        b = curves[(model, 'binned', 'le')][-1]
        print(f'  binned<=100={b:+.2f}  L2={curves[(model, "l2", "le")][-1]:+.2f}  '
              f'L1={curves[(model, "l1", "le")][-1]:+.2f}', flush=True)

    models_done = sorted({m for m, _, _ in curves})
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True, sharey=True)
    for ri, (mkey, mlabel, _) in enumerate(methods):
        for ci, (vkey, vlabel, _) in enumerate(views):
            ax = axes[ri][ci]
            for model in models_done:
                v = curves[(model, mkey, vkey)]
                ax.plot(ks, v, color=COLORS.get(model, '#444444'), lw=1.6,
                        label=model if (ri == 0 and ci == 0) else None)
            ax.axhline(0.5, color='#555555', lw=1.0, ls='--')
            ax.axhline(0.0, color='#bbbbbb', lw=0.8)
            if ri == 0:
                ax.set_title(vlabel, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f'{mlabel}\n' + r'$\delta(k)$', fontsize=9.5)
            if ri == 2:
                ax.set_xlabel('insertion index  $k$')
            ax.set_ylim(-0.6, 1.7)
    axes[0][0].legend(fontsize=7.2, ncol=2, loc='lower right')
    fig.suptitle(f'{args.stock}: exponent dynamics under three estimators (rows) '
                 f'and three cross-sections (cols); dashed = 0.5, sigma = Parkinson', y=0.995)
    fig.tight_layout()
    suff = f'_{args.tag}' if args.tag else ''
    png = os.path.join(outdir, f'beta_3x3_{args.stock}{suff}.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), **cache)
    print(f'BETA_3X3_DONE -> {png} (+npz)', flush=True)


if __name__ == '__main__':
    main()
