#!/usr/bin/env python3
"""
Estimator x volatility grid with amplitudes: delta(k) under the 3 honest estimators
for ALL SIX sigma normalisations, plus the amplitude/intercept dynamics.

Rows:  binned delta | direct-L2 delta | direct-L1 delta | amplitude row
Cols:  sigma = 1 (none), parkinson, garman_klass, rogers_satchell,
       close_to_close, yang_zhang
View:  --view le|eq|ge (cumulative <=k / exact =k / reverse >=k) — one figure per view.

The two window estimators (close_to_close, yang_zhang) are one value per ticker,
i.e. a constant rescale of y -> delta must be IDENTICAL to the sigma=1 column and
only the amplitude moves; the per-day range estimators (parkinson/gk/rs) are the
only ones that can move delta. This figure makes that visible instead of assumed.

Amplitude row: direct-L2 amplitude a(k) (solid) and the binned intercept
exp(alpha)(k) (dotted), log scale.

  python beta_sigma_grid.py --grid <root> --daily <daily.csv> --stock NVDA
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beta_grid import collect
from beta_binned import bin_means
from vol_estimators import daily_sigmas, METHODS

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
SIGMAS = ['none'] + METHODS
DGRID = np.arange(0.05, 1.51, 0.01)
NBINS = 18
MIN_PTS = 200
CAP_L2, CAP_L1 = 200_000, 60_000


def fit_binned(x, y):
    xc, ym, _ = bin_means(x, y, NBINS)
    pos = ym > 0
    if pos.sum() < 5 or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan, np.nan
    d, b = np.polyfit(xc[pos], np.log(ym[pos]), 1)
    return float(d), float(b)


def _sub(x, y, cap, rng):
    if x.size <= cap:
        return x, y
    i = rng.choice(x.size, size=cap, replace=False)
    return x[i], y[i]


def fit_l2(x, y, rng):
    x, y = _sub(x, y, CAP_L2, rng)
    best = (np.inf, np.nan, np.nan)
    for d in DGRID:
        t = np.exp(d * x)
        a = float(np.dot(y, t) / np.dot(t, t))
        s = float(np.sum((y - a * t) ** 2))
        if s < best[0]:
            best = (s, float(d), a)
    return best[1], best[2]


def fit_l1(x, y, rng):
    x, y = _sub(x, y, CAP_L1, rng)
    best = (np.inf, np.nan, np.nan)
    for d in DGRID:
        t = np.exp(d * x)
        r = y / t
        idx = np.argsort(r)
        cw = np.cumsum(t[idx])
        a = float(r[idx][np.searchsorted(cw, 0.5 * cw[-1])])
        s = float(np.sum(np.abs(y - a * t)))
        if s < best[0]:
            best = (s, float(d), a)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', required=True)
    ap.add_argument('--models', default='Historic,Heuristic,Propagator,Hawkes,CST,NMZI,'
                                        'Mamba3,GDN,S5_120M,S5,Mamba3_4k,S5_4k')
    ap.add_argument('--ks', default='5:100:5')
    ap.add_argument('--view', default='le', choices=['le', 'eq', 'ge'])
    ap.add_argument('--points_cache', default=None,
                    help='npz with pre-collected raw points; created if absent '
                         '(collect() is ~5h on the GOOG grid — cache it once)')
    args = ap.parse_args()
    SEL = {'le': lambda kk, k: kk <= k, 'eq': lambda kk, k: kk == k,
           'ge': lambda kk, k: kk >= k}[args.view]
    VLBL = {'le': r'cumulative $\leq k$', 'eq': r'exact $=k$', 'ge': r'reverse $\geq k$'}[args.view]
    lo, hi, st = (int(v) for v in args.ks.split(':'))
    ks = np.arange(lo, hi + 1, st)
    sig = daily_sigmas(args.daily)
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', 'beta_sigma_grid')
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    cache = {'ks': ks}
    curves = {}
    pts = None
    if args.points_cache and os.path.exists(args.points_cache):
        pts = dict(np.load(args.points_cache, allow_pickle=True))
        print(f'points cache loaded: {args.points_cache}', flush=True)
    pts_out = {}
    for model in [m for m in args.models.split(',') if m]:
        if pts is not None and f'{model}_Q' in pts:
            Q = pts[f'{model}_Q']; I = pts[f'{model}_I']
            days = pts[f'{model}_days']; kc = pts[f'{model}_kc']
        else:
            raw = collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy'), args.stock, +1) + \
                  collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'sell'), args.stock, -1)
            if not raw:
                print(f'{model}: no data', flush=True); continue
            Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
            days = np.array([r[2] for r in raw]); kc = np.array([r[3] for r in raw], int) + 1
            if args.points_cache:
                pts_out[f'{model}_Q'] = Q; pts_out[f'{model}_I'] = I
                pts_out[f'{model}_days'] = days; pts_out[f'{model}_kc'] = kc
                np.savez_compressed(args.points_cache, **pts_out)
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        x0 = np.log(Q / V)
        print(f'{model}: {len(Q):,} points', flush=True)
        for sm in SIGMAS:
            sv = np.ones(len(days)) if sm == 'none' else \
                np.array([sig.get((args.stock, d), {}).get(sm, np.nan) for d in days], float)
            y0 = I / sv
            ok = np.isfinite(x0) & np.isfinite(y0)
            x, y, kk = x0[ok], y0[ok], kc[ok]
            db = np.full(len(ks), np.nan); ib = np.full(len(ks), np.nan)
            d2 = np.full(len(ks), np.nan); a2 = np.full(len(ks), np.nan)
            d1 = np.full(len(ks), np.nan); a1 = np.full(len(ks), np.nan)
            for i, k in enumerate(ks):
                m = SEL(kk, int(k))
                if m.sum() < MIN_PTS:
                    continue
                db[i], ib[i] = fit_binned(x[m], y[m])
                d2[i], a2[i] = fit_l2(x[m], y[m], rng)
                d1[i], a1[i] = fit_l1(x[m], y[m], rng)
            for tag, arr in (('dbin', db), ('ibin', ib), ('dl2', d2), ('al2', a2),
                             ('dl1', d1), ('al1', a1)):
                curves[(model, sm, tag)] = arr
                cache[f'{model}_{sm}_{tag}'] = arr
            print(f'  {sm:16s} bin={db[-1]:+.2f}  L2={d2[-1]:+.2f}  L1={d1[-1]:+.2f}', flush=True)

    models_done = sorted({m for m, _, _ in curves})
    rows = [('dbin', 'signed binned  $\\delta(k)$'),
            ('dl2', 'direct $L_2$  $\\delta(k)$'),
            ('dl1', 'direct $L_1$  $\\delta(k)$'),
            ('amp', 'amplitude:  $a_{L_2}$ (solid), $e^{\\alpha_{bin}}$ (dotted)')]
    fig, axes = plt.subplots(4, len(SIGMAS), figsize=(3.1 * len(SIGMAS), 12.0),
                             sharex=True)
    for ci, sm in enumerate(SIGMAS):
        for ri, (tag, rlabel) in enumerate(rows):
            ax = axes[ri][ci]
            for model in models_done:
                c = COLORS.get(model, '#444444')
                if tag == 'amp':
                    ax.plot(ks, np.abs(curves[(model, sm, 'al2')]), color=c, lw=1.3)
                    ax.plot(ks, np.exp(curves[(model, sm, 'ibin')]), color=c, lw=1.1, ls=':')
                else:
                    ax.plot(ks, curves[(model, sm, tag)], color=c, lw=1.5,
                            label=model if (ri == 0 and ci == 0) else None)
            if tag == 'amp':
                ax.set_yscale('log')
            else:
                ax.axhline(0.5, color='#555555', lw=0.9, ls='--')
                ax.axhline(0.0, color='#bbbbbb', lw=0.7)
                ax.set_ylim(-0.6, 1.7)
            if ri == 0:
                ax.set_title(sm, fontsize=10.5)
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=8.5)
            if ri == 3:
                ax.set_xlabel('$k$')
    axes[0][0].legend(fontsize=6.6, ncol=2, loc='lower right')
    fig.suptitle(f'{args.stock}: three estimators x six volatility normalisations '
                 f'({VLBL}; window sigmas = per-ticker constants -> '
                 f'delta must match the sigma=1 column, only amplitude moves)', y=0.995)
    fig.tight_layout()
    png = os.path.join(outdir, f'beta_sigma_grid_{args.stock}_{args.view}.png')
    fig.savefig(png, dpi=145, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), **cache)
    print(f'BETA_SIGMA_GRID_DONE -> {png} (+npz)', flush=True)


if __name__ == '__main__':
    main()
