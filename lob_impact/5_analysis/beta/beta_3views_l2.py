#!/usr/bin/env python3
"""
Figure-10 replacement: three-view exponent dynamics computed by the DIRECT L2
estimator on UNFILTERED signed points (no I>0 selection, no logs on y), plus a
bottom row that shows exactly how the L2 error is computed.

Top row    delta_L2(k) for all models under the three cross-sections of the
           k-th insertion: cumulative <=k, exact =k, reverse >=k.
           Fit: y = a * (Q/V)^delta on signed y = I/sigma, grid over delta,
           closed-form amplitude a = <y,t>/<t,t> with t = (Q/V)^delta.
Bottom row hero model at the cursor k: the signed cloud (x = ln(Q/V), y = I/sigma)
           with the view-active points highlighted, the fitted curve a*e^{delta x},
           and an INSET with the normalised misfit profile S(delta)/S_min --
           the quantity the estimator minimises. A sharp minimum = identified
           exponent; a flat profile (exact =k for weak-signal models) = not
           identifiable.

  python beta_3views_l2.py --grid <root> --daily <daily.csv> --stock NVDA
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beta_grid import collect
from vol_estimators import daily_sigmas

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
DGRID = np.arange(0.05, 1.51, 0.01)
MIN_PTS = 200
CAP_L2 = 200_000
CLOUD_CAP = 4_000

VIEWS = [('le', r'cumulative  $\leq k$', lambda kc, k: kc <= k),
         ('eq', r'exact  $=k$', lambda kc, k: kc == k),
         ('ge', r'reverse  $\geq k$', lambda kc, k: kc >= k)]


def l2_profile(x, y, rng=None):
    """Return (S(delta) over DGRID, a(delta)) for the closed-form-amplitude L2 fit."""
    if rng is not None and x.size > CAP_L2:
        i = rng.choice(x.size, size=CAP_L2, replace=False)
        x, y = x[i], y[i]
    S = np.empty(len(DGRID)); A = np.empty(len(DGRID))
    for j, d in enumerate(DGRID):
        t = np.exp(d * x)
        a = float(np.dot(y, t) / np.dot(t, t))
        S[j] = float(np.sum((y - a * t) ** 2))
        A[j] = a
    return S, A


def delta_l2(x, y, rng):
    S, A = l2_profile(x, y, rng)
    j = int(np.argmin(S))
    return float(DGRID[j]), float(A[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', required=True)
    ap.add_argument('--models', default='Historic,Heuristic,Propagator,Hawkes,CST,NMZI,'
                                        'Mamba3,GDN,S5_120M,S5,Mamba3_4k,S5_4k')
    ap.add_argument('--method', default='parkinson')
    ap.add_argument('--ks', default='5:100:5')
    ap.add_argument('--hero', default='Mamba3')
    ap.add_argument('--kcursor', type=int, default=70)
    args = ap.parse_args()
    lo, hi, st = (int(v) for v in args.ks.split(':'))
    ks = np.arange(lo, hi + 1, st)
    sig = daily_sigmas(args.daily)
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', 'beta_3views_l2')
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    cache = {'ks': ks, 'dgrid': DGRID, 'kcursor': args.kcursor, 'hero': args.hero}
    curves = {}
    hero_pts = None
    for model in [m for m in args.models.split(',') if m]:
        raw = collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'sell'), args.stock, -1)
        if not raw:
            print(f'{model}: no data', flush=True); continue
        Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
        days = np.array([r[2] for r in raw]); kc = np.array([r[3] for r in raw], int) + 1
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        sp = np.array([sig.get((args.stock, d), {}).get(args.method, np.nan) for d in days], float)
        x_all = np.log(Q / V); y_all = I / sp
        ok = np.isfinite(x_all) & np.isfinite(y_all)
        x_all, y_all, kc = x_all[ok], y_all[ok], kc[ok]
        if x_all.size == 0:
            print(f'{model}: all points non-finite (skipped)', flush=True); continue
        print(f'{model}: {x_all.size:,} signed points', flush=True)
        for vkey, _, selfn in VIEWS:
            vals = np.full(len(ks), np.nan)
            for i, k in enumerate(ks):
                m = selfn(kc, int(k))
                if m.sum() >= MIN_PTS:
                    vals[i], _ = delta_l2(x_all[m], y_all[m], rng)
            curves[(model, vkey)] = vals
            cache[f'{model}_{vkey}'] = vals
        if model == args.hero:
            hero_pts = (x_all, y_all, kc)
        print(f'  dL2 <=100={curves[(model, "le")][-1]:+.2f}  '
              f'=k mean={np.nanmean(curves[(model, "eq")]):+.2f}  '
              f'>=k(5)={curves[(model, "ge")][0]:+.2f}', flush=True)

    if not curves:
        print('nothing collected'); return
    models_done = sorted({m for m, _ in curves})
    if hero_pts is None:
        args.hero = models_done[0]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for ci, (vkey, vlabel, _) in enumerate(VIEWS):
        ax = axes[0][ci]
        for model in models_done:
            ax.plot(ks, curves[(model, vkey)], color=COLORS.get(model, '#444444'), lw=1.6,
                    label=model if ci == 0 else None)
        ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0)
        ax.axvline(args.kcursor, color='#555555', ls=':', lw=1.0)
        ax.set_ylim(-0.05, 1.55); ax.set_xlim(ks[0], ks[-1])
        ax.set_title(vlabel, fontsize=11)
        ax.set_xlabel('insertion index  $k$')
        if ci == 0:
            ax.set_ylabel(r'$\delta_{L_2}(k)$  (signed, unfiltered)')
            ax.legend(fontsize=6.8, ncol=2, loc='upper left')

    # bottom row: hero cloud + fitted curve + misfit-profile inset, per view
    x_all, y_all, kc = hero_pts
    hc = COLORS.get(args.hero, '#2F5DA3')
    xmin, xmax = np.nanpercentile(x_all, [0.5, 99.5])
    ymin, ymax = np.nanpercentile(y_all, [0.5, 99.5])
    for ci, (vkey, _, selfn) in enumerate(VIEWS):
        ax = axes[1][ci]
        m = selfn(kc, args.kcursor)
        # cloud subsamples for readability (fits below use all masked points)
        iia = np.flatnonzero(~m); iib = np.flatnonzero(m)
        if iia.size > CLOUD_CAP:
            iia = rng.choice(iia, CLOUD_CAP, replace=False)
        if iib.size > CLOUD_CAP:
            iib = rng.choice(iib, CLOUD_CAP, replace=False)
        ax.scatter(x_all[iia], y_all[iia], s=4, alpha=0.05, color='#9a9a9a', lw=0, zorder=1)
        ax.scatter(x_all[iib], y_all[iib], s=5, alpha=0.30, color=hc, lw=0, zorder=2)
        ax.axhline(0.0, color='#bbbbbb', lw=0.8, zorder=1)
        if m.sum() >= MIN_PTS:
            S, A = l2_profile(x_all[m], y_all[m], rng)
            j = int(np.argmin(S)); d, a = float(DGRID[j]), float(A[j])
            xx = np.linspace(x_all[m].min(), x_all[m].max(), 100)
            ax.plot(xx, a * np.exp(d * xx), color='#1a1a1a', lw=1.9, zorder=4,
                    label=rf'$y = a\,(Q/V)^{{\delta}}$:  $\delta={d:.2f}$, $n={int(m.sum()):,}$')
            ax.legend(fontsize=8, loc='upper left')
            # inset: the misfit profile the estimator minimises
            ins = ax.inset_axes([0.62, 0.08, 0.35, 0.38])
            ins.plot(DGRID, S / S.min(), color=hc, lw=1.3)
            ins.axvline(d, color='#1a1a1a', lw=0.9, ls=':')
            ins.set_ylim(bottom=0.99)
            ins.set_title(r'$S(\delta)/S_{\min}$', fontsize=7)
            ins.tick_params(labelsize=6)
            cache[f'hero_{vkey}_S'] = S
            cache[f'hero_{vkey}_delta'] = d
            cache[f'hero_{vkey}_a'] = a
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$\ln(Q/V)$')
        if ci == 0:
            ax.set_ylabel(rf'$I/\sigma$  (signed)   [{args.hero}]')
    cache['hero_cloud_x'] = x_all[:CLOUD_CAP]; cache['hero_cloud_y'] = y_all[:CLOUD_CAP]

    fig.suptitle(f'{args.stock}: direct-$L_2$ exponent on unfiltered signed points, three views (top); '
                 f'{args.hero} cloud, fitted power law and misfit profile at $k={args.kcursor}$ (bottom). '
                 r'$\sigma$ = ' + args.method, fontsize=11.5, y=0.995)
    fig.tight_layout()
    png = os.path.join(outdir, f'beta_3views_l2_{args.stock}.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), **cache)
    print(f'B3L2_DONE -> {png} (+npz)', flush=True)


if __name__ == '__main__':
    main()
