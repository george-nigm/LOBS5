#!/usr/bin/env python3
"""
Publication re-plot of the mid-price impact trajectory, read straight from the cached
mid_trajectory .npz (per-model x / mid / band, plus sqrt & propagator overlays).

No grid access: this only touches the tiny cache, so it is login-node safe and fast.
Style comes from ../pubstyle.py so it matches every other paper figure.

  python3.11 5_analysis/beta/pub_mid_trajectory.py \
      --npz results/mid_impact/mid_trajectory_EA_beta.npz --shape beta \
      --out results/mid_impact/pub_midprice_EA_beta.png
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pubstyle import apply, style, ref_style, legend_box, savefig_pub

# order models are drawn (hero neural model last so it sits on top)
DRAW_ORDER = ['CST', 'Historic', 'Heuristic', 'Hawkes', 'Mamba3_4k', 'Mamba3', 'OW', 'QR']


def models_in(npz):
    present = {k[:-4] for k in npz.files if k.endswith('_mid')}
    ordered = [m for m in DRAW_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--shape', default='beta', choices=['beta', 'decay', 'relaxation'])
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--title', default=None, help='override the short title; "" for none')
    ap.add_argument('--loc', default='upper left', help='legend location (matplotlib loc string)')
    ap.add_argument('--ncol', type=int, default=1, help='number of legend columns')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    apply()
    d = np.load(args.npz)
    build_up = (args.shape == 'beta')

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for m in models_in(d):
        x = d[f'{m}_x']; y = d[f'{m}_mid']; band = d.get(f'{m}_band')
        ok = np.isfinite(y)
        st = style(m)
        ax.plot(x[ok], y[ok], color=st['color'], ls=st['ls'], lw=1.7, label=st['label'], zorder=3)
        if band is not None:
            b = band.copy()
            ax.fill_between(x, y - 1.96 * b, y + 1.96 * b, color=st['color'],
                            alpha=0.10, lw=0, zorder=1)

    # --- theory overlays ---
    if 'sqrt_x' in d.files:
        rs = ref_style('sqrt')
        lbl = rs['label'] if build_up else r'$\sqrt{\cdot}$-law peak (permanent ref.)'
        ax.plot(d['sqrt_x'], d['sqrt_y'], color=rs['color'], ls=rs['ls'], lw=rs['lw'],
                label=lbl, zorder=4)
    if 'prop_x' in d.files:
        rp = ref_style('propagator')
        ax.plot(d['prop_x'], d['prop_y'], color=rp['color'], ls=rp['ls'], lw=rp['lw'],
                label='propagator (transient)', zorder=4)

    ax.axhline(0.0, color='#1a1a1a', lw=0.7, zorder=2)

    # x-limit: last step any model is finite
    xmax = max(float(d[f'{m}_x'][np.isfinite(d[f'{m}_mid'])].max())
               for m in models_in(d) if np.isfinite(d[f'{m}_mid']).any())
    ax.set_xlim(0, xmax)

    ax.set_xlabel('message step  (generation time)')
    ax.set_ylabel(r'mean signed mid-price change  (bps)')
    if args.title is None:
        phase = 'build-up' if build_up else 'relaxation'
        args.title = f'{args.stock} — mid-price impact ({phase})'
    if args.title:
        ax.set_title(args.title)
    legend_box(ax, loc=args.loc, ncol=args.ncol, columnspacing=1.1, handlelength=1.6)
    ax.margins(x=0)

    fig.tight_layout()
    out = savefig_pub(fig, os.path.abspath(args.out))
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
