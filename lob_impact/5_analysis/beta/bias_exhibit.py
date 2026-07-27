#!/usr/bin/env python3
"""
ONE-LOOK BIAS EXHIBIT: everything on a single static figure per stock.

Per model panel (x = ln(Q/V), y = signed I/sigma):
  - grey cloud of raw per-(sample,insertion) points (subsampled);
  - SIGNED quantile-bin means +- 1.96 SE  (the unbiased headline of beta_binned.py);
  - three fitted power laws drawn through the data:
      red dashed   = naive per-point log-log fit on the I>0 subset (the BIASED estimator:
                     on impact-free replay it reads ~0.5 from pure diffusion |I| ~ sigma*sqrt(k*mb));
      solid        = fit over the signed bin means (headline delta);
      green dashdot= direct L2 fit of y = a*(Q/V)^d on the signed raw points (no log, no selection);
  - dotted grey reference with slope 0.5 anchored at the top bin.
Rows: sigma = Parkinson (normalised) and sigma = 1 (raw impact) — bias visible with and
without the volatility normalisation at once.

All plotted arrays are dumped to an npz next to the png (figure-data caching rule).

  python bias_exhibit.py --grid <root> --stock EA --daily <daily_h_l_all.csv> \
      [--models Historic,Hawkes,Mamba3,GDN,Mamba3_4k,S5_4k] [--nbins 18]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beta_grid import collect
from beta_binned import bin_means
from vol_estimators import daily_sigmas


def _canon_palette():
    """Canonical palette from lob_impact/core/model_style.py — see the note there on why the
    per-script dicts drifted. Falls back to the local dict if that file is not reachable."""
    import os, importlib.util
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        p = os.path.join(d, 'core', 'model_style.py')
        if os.path.exists(p):
            sp = importlib.util.spec_from_file_location('_lob_model_style', p)
            m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
            return dict(m.COLORS)
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return {}

_LOCAL_COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'OW': '#6A1B9A', 'QR': '#00ACC1'}
COLORS = {**_LOCAL_COLORS, **_canon_palette()}
CLOUD_MAX = 4000


def fit_perpoint(x, y):
    """Biased naive estimator: OLS of ln y on x over the I>0 subset."""
    pos = np.isfinite(y) & (y > 0)
    if pos.sum() < 30:
        return np.nan, np.nan
    d, b = np.polyfit(x[pos], np.log(y[pos]), 1)
    return float(d), float(b)


def fit_bins(xc, ym):
    pos = ym > 0
    if pos.sum() < 5 or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan, np.nan
    d, b = np.polyfit(xc[pos], np.log(ym[pos]), 1)
    return float(d), float(b)


def fit_l2(x, y):
    """Direct L2 on signed raw data: y = a*exp(d*x), a closed-form per d on a grid."""
    best = (np.inf, np.nan, np.nan)
    for d in np.arange(0.05, 1.51, 0.01):
        qd = np.exp(d * x)
        a = float(np.dot(y, qd) / np.dot(qd, qd))
        l2 = float(np.sum((y - a * qd) ** 2))
        if l2 < best[0]:
            best = (l2, float(d), a)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,OW,Hawkes,QR,S5_4k,Mamba3,Mamba3_4k,GDN')
    ap.add_argument('--nbins', type=int, default=18)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', 'bias_exhibit')
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    data = {}
    for model in models:
        raw = collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'sell'), args.stock, -1)
        if not raw:
            print(f'{model}: no data', flush=True); continue
        Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
        days = np.array([r[2] for r in raw])
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        sp = np.array([sig.get((args.stock, d), {}).get('parkinson', np.nan) for d in days], float)
        x = np.log(Q / V)
        data[model] = (x, I, sp)
        print(f'{model}: {len(raw):,} points', flush=True)

    rows = [('parkinson', r'signed $I/\sigma_{\rm Parkinson}$'), ('none', 'signed impact $I$ (raw)')]
    fig, axes = plt.subplots(len(rows), len(data), figsize=(3.4 * len(data), 7.2),
                             squeeze=False, sharex='col')
    cache = {}
    for ci, (model, (x_all, I, sp)) in enumerate(data.items()):
        for ri, (method, ylabel) in enumerate(rows):
            y_all = I / sp if method == 'parkinson' else I.copy()
            ok = np.isfinite(x_all) & np.isfinite(y_all)
            x, y = x_all[ok], y_all[ok]
            ax = axes[ri][ci]
            if x.size < 50:
                ax.set_title(f'{model} (no data)'); continue
            sub = rng.choice(x.size, size=min(CLOUD_MAX, x.size), replace=False)
            ax.scatter(x[sub], y[sub], s=3, color='#999999', alpha=0.15, lw=0, rasterized=True)
            xc, ym, nb = bin_means(x, y, args.nbins)
            yse = np.array([1.96 * np.std(y[np.argsort(x)][c]) / np.sqrt(len(c))
                            for c in np.array_split(np.arange(x.size), len(xc))])
            col = COLORS.get(model, '#333333')
            ax.errorbar(xc, ym, yerr=yse, fmt='o', ms=5, color=col, ecolor=col,
                        elinewidth=1.2, capsize=2, zorder=5, label='signed bin means')
            xs = np.linspace(x.min(), x.max(), 100)
            d_pp, b_pp = fit_perpoint(x, y)
            d_bin, b_bin = fit_bins(xc, ym)
            d_l2, a_l2 = fit_l2(x, y)
            if np.isfinite(d_pp):
                ax.plot(xs, np.exp(b_pp + d_pp * xs), '--', color='#C0392B', lw=1.6,
                        label=f'per-point $I{{>}}0$ (biased): {d_pp:+.2f}')
            if np.isfinite(d_bin):
                ax.plot(xs, np.exp(b_bin + d_bin * xs), '-', color='k', lw=1.8,
                        label=f'binned (headline): {d_bin:+.2f}')
            if np.isfinite(d_l2):
                ax.plot(xs, a_l2 * np.exp(d_l2 * xs), '-.', color='#27AE60', lw=1.6,
                        label=f'direct $L_2$: {d_l2:+.2f}')
            top = np.argmax(xc)
            anchor = ym[top] if ym[top] > 0 else np.abs(ym).max()
            ax.plot(xs, anchor * np.exp(0.5 * (xs - xc[top])), ':', color='#555555', lw=1.4,
                    label=r'$\sqrt{\cdot}$ reference (0.5)')
            ax.axhline(0, color='#bbbbbb', lw=0.8)
            lo, hi = np.nanpercentile(y, [2, 98])
            pad = 0.15 * (hi - lo)
            ax.set_ylim(min(lo - pad, ym.min() - pad), max(hi + pad, ym.max() + pad))
            ax.legend(fontsize=6.4, loc='upper left', framealpha=0.85)
            if ri == 0:
                ax.set_title(model, color=col, fontweight='bold')
            if ri == len(rows) - 1:
                ax.set_xlabel(r'$\ln(Q/V)$')
            if ci == 0:
                ax.set_ylabel(ylabel)
            tag = f'{model}_{method}'
            cache[f'{tag}_cloud_x'] = x[sub]; cache[f'{tag}_cloud_y'] = y[sub]
            cache[f'{tag}_xc'] = xc; cache[f'{tag}_ym'] = ym; cache[f'{tag}_yse'] = yse
            cache[f'{tag}_fits'] = np.array([d_pp, b_pp, d_bin, b_bin, d_l2, a_l2])
            print(f'{model:10s} {method:9s} per-point={d_pp:+.3f} binned={d_bin:+.3f} '
                  f'L2={d_l2:+.3f} (N={x.size:,})', flush=True)

    fig.suptitle(f'{args.stock}: biased per-point vs unbiased signed-binned vs direct $L_2$ '
                 f'(rows: with / without $\\sigma$ normalisation)', y=1.0)
    fig.tight_layout()
    png = os.path.join(outdir, f'bias_exhibit_{args.stock}.png')
    fig.savefig(png, dpi=170, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), **cache)
    print(f'BIAS_EXHIBIT_DONE -> {png} (+npz)', flush=True)


if __name__ == '__main__':
    main()
