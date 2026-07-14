#!/usr/bin/env python3
"""
EVERY MANIPULATION ON THE WAY TO beta, ONE PANEL PER STEP (per model, per stock).

Column 1  raw signed data           (x=ln(Q/V), y=I/sigma, linear)  -- what we actually have
Column 2  the naive move            drop y<=0, take logs, OLS       -- the biased estimator
Column 3  no logs: direct L2 grid   SSE(delta) profile + minimum    -- negatives kept, a closed-form
Column 4  no logs: direct L1 grid   sum|resid|(delta) profile       -- robust twin of L2
Column 5  average then log: bins    signed bin means +- SE, log-fit -- the literature convention

Data source: the bias_exhibit npz caches (figure-data-caching rule) -- NO grid re-read.
The cloud and the L1/L2 profiles use the cached 4000-point subsample; the quoted
delta_pp / delta_bin / delta_L2 in panel titles are the EXACT full-grid fits stored in the
npz (fits = [d_pp, b_pp, d_bin, b_bin, d_l2, a_l2]); subsample estimates are marked '~'.

  python beta_pipeline_steps.py --stock GOOG [--method parkinson] [--models ...]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22'}
C_BIASED, C_BINS, C_L2, C_L1 = '#C0392B', 'k', '#27AE60', '#7B1FA2'
DGRID = np.arange(0.05, 1.51, 0.01)


def l2_profile(x, y):
    """SSE(delta) with closed-form amplitude per delta (identical to bias_exhibit fit_l2)."""
    sse, amps = [], []
    for d in DGRID:
        t = np.exp(d * x)
        a = float(np.dot(y, t) / np.dot(t, t))
        sse.append(float(np.sum((y - a * t) ** 2))); amps.append(a)
    return np.array(sse), np.array(amps)


def l1_amp(y, t):
    """argmin_a sum|y - a t| = weighted median of y/t with weights t (t>0)."""
    r = y / t
    idx = np.argsort(r)
    cw = np.cumsum(t[idx])
    return float(r[idx][np.searchsorted(cw, 0.5 * cw[-1])])


def l1_profile(x, y):
    dev, amps = [], []
    for d in DGRID:
        t = np.exp(d * x)
        a = l1_amp(y, t)
        dev.append(float(np.sum(np.abs(y - a * t)))); amps.append(a)
    return np.array(dev), np.array(amps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', default='GOOG')
    ap.add_argument('--method', default='parkinson', choices=['parkinson', 'none'])
    ap.add_argument('--models', default='')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', 'bias_exhibit')
    z = np.load(os.path.join(outdir, f'bias_exhibit_{args.stock}.npz'))
    models = ([m for m in args.models.split(',') if m] or
              sorted({k[:-len('_parkinson_cloud_x')] for k in z.files
                      if k.endswith('_parkinson_cloud_x')},
                     key=lambda m: (m != 'Historic', m)))

    nrow, ncol = len(models), 5
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.7 * nrow), squeeze=False)
    cache = {}
    for ri, model in enumerate(models):
        tag = f'{model}_{args.method}'
        x, y = z[f'{tag}_cloud_x'], z[f'{tag}_cloud_y']
        xc, ym, yse = z[f'{tag}_xc'], z[f'{tag}_ym'], z[f'{tag}_yse']
        d_pp, b_pp, d_bin, b_bin, d_l2, a_l2 = z[f'{tag}_fits']
        col = COLORS.get(model, '#333333')
        neg = float((y <= 0).mean())
        ylo, yhi = np.nanpercentile(y, [2, 98]); pad = 0.15 * (yhi - ylo)

        # -- 1. raw signed data ------------------------------------------------
        ax = axes[ri][0]
        ax.scatter(x, y, s=3, color='#999999', alpha=0.2, lw=0, rasterized=True)
        ax.axhline(0, color='#bbbbbb', lw=0.8)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f'1. raw signed points  ({neg:.0%} are $\\leq 0$)', fontsize=8.5)
        ax.set_ylabel(model, color=col, fontweight='bold')

        # -- 2. the naive move: drop negatives, log-log ------------------------
        ax = axes[ri][1]
        pos = y > 0
        ax.scatter(x[pos], np.log(y[pos]), s=3, color='#999999', alpha=0.2, lw=0, rasterized=True)
        xs = np.linspace(x.min(), x.max(), 50)
        if np.isfinite(d_pp):
            ax.plot(xs, b_pp + d_pp * xs, '--', color=C_BIASED, lw=1.8)
        ax.set_title(f'2. drop the {neg:.0%}, log $y$: '
                     f'$\\delta_{{pp}}$={d_pp:+.2f} (biased)', fontsize=8.5, color=C_BIASED)

        # -- 3. direct L2 grid --------------------------------------------------
        ax = axes[ri][2]
        sse, amps2 = l2_profile(x, y)
        i2 = int(np.argmin(sse))
        ax.plot(DGRID, sse / sse.min(), color=C_L2, lw=1.6)
        ax.axvline(DGRID[i2], color=C_L2, lw=1.0, ls=':')
        ax.axvline(0.5, color='#555555', lw=0.8, ls=':')
        ax.set_ylim(0.995, min(1.2, float((sse / sse.min()).max()) * 1.02 + 1e-3))
        ax.set_title(f'3. no logs, $L_2(\\delta)$: ~{DGRID[i2]:.2f} '
                     f'(exact {d_l2:+.2f}, a={a_l2:+.3f})', fontsize=8.5, color=C_L2)

        # -- 4. direct L1 grid --------------------------------------------------
        ax = axes[ri][3]
        dev, amps1 = l1_profile(x, y)
        i1 = int(np.argmin(dev))
        ax.plot(DGRID, dev / dev.min(), color=C_L1, lw=1.6)
        ax.axvline(DGRID[i1], color=C_L1, lw=1.0, ls=':')
        ax.axvline(0.5, color='#555555', lw=0.8, ls=':')
        ax.set_ylim(0.9995, min(1.05, float((dev / dev.min()).max()) * 1.005))
        ax.set_title(f'4. no logs, $L_1(\\delta)$: ~{DGRID[i1]:.2f} '
                     f'(a={amps1[i1]:+.3f})', fontsize=8.5, color=C_L1)

        # -- 5. average then log: signed bins ----------------------------------
        ax = axes[ri][4]
        ax.scatter(x, y, s=3, color='#dddddd', alpha=0.2, lw=0, rasterized=True)
        ax.errorbar(xc, ym, yerr=yse, fmt='o', ms=4.5, color=col, ecolor=col,
                    elinewidth=1.1, capsize=2, zorder=5)
        if np.isfinite(d_bin):
            ax.plot(xs, np.exp(b_bin + d_bin * xs), '-', color=C_BINS, lw=1.6)
        top = int(np.argmax(xc))
        anchor = ym[top] if ym[top] > 0 else np.abs(ym).max()
        ax.plot(xs, anchor * np.exp(0.5 * (xs - xc[top])), ':', color='#555555', lw=1.2)
        ax.axhline(0, color='#bbbbbb', lw=0.8)
        ax.set_ylim(min(ylo - pad, ym.min() - pad), max(yhi + pad, ym.max() + pad))
        ax.set_title(f'5. signed bin means, then log-fit: '
                     f'$\\delta_{{bin}}$={d_bin:+.2f}', fontsize=8.5)

        for ci in range(ncol):
            axes[ri][ci].tick_params(labelsize=7)
            if ri == nrow - 1:
                axes[ri][ci].set_xlabel(r'$\ln(Q/V)$' if ci not in (2, 3) else r'$\delta$', fontsize=8)
        cache[f'{tag}_l2_profile'] = sse; cache[f'{tag}_l1_profile'] = dev
        cache[f'{tag}_l2_amps'] = amps2; cache[f'{tag}_l1_amps'] = amps1
        cache[f'{tag}_dgrid'] = DGRID
        print(f'{model:10s} neg={neg:.0%}  pp={d_pp:+.2f}  L2~{DGRID[i2]:.2f} (exact {d_l2:+.2f})  '
              f'L1~{DGRID[i1]:.2f}  bin={d_bin:+.2f}', flush=True)

    fig.suptitle(f'{args.stock}: every manipulation on the way to $\\beta$  '
                 f'(sigma = {args.method}; cloud & $L_1$/$L_2$ profiles on the cached 4,000-pt '
                 f'subsample, quoted $\\delta_{{pp}}/\\delta_{{bin}}$/exact-$L_2$ are full-grid fits)',
                 y=1.0, fontsize=11)
    fig.tight_layout()
    png = os.path.join(outdir, f'beta_pipeline_{args.stock}.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), **cache)
    print(f'BETA_PIPELINE_DONE -> {png} (+npz)', flush=True)


if __name__ == '__main__':
    main()
