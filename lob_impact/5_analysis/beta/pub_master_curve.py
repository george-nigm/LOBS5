#!/usr/bin/env python3
"""
Publication re-plot of the impact MASTER CURVE straight from the cached master_curve_*.npz
(vgrid + per-model normalised master curve).  No grid access -> login-node safe & instant;
use this to tweak styling/legend placement without re-reading the Lustre grid.

  python3.11 5_analysis/beta/pub_master_curve.py \
      --npz results/master_curve/master_curve_EA_beta.npz --shape beta \
      --out results/master_curve/master_curve_EA_beta.png
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pubstyle import apply, style, ref_style, legend_box, savefig_pub

DRAW_ORDER = ['Historic', 'Heuristic', 'Hawkes', 'CST', 'Mamba3_4k', 'Mamba3', 'OW', 'QR']


def models_in(d):
    present = {k[:-7] for k in d.files if k.endswith('_master')}
    return [m for m in DRAW_ORDER if m in present] + sorted(present - set(DRAW_ORDER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--shape', default='beta', choices=['beta', 'relaxation', 'decay'])
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--loc', default=None, help='legend location override (matplotlib loc string)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    apply()
    d = np.load(args.npz)
    v = d['vgrid']
    build_up = (args.shape == 'beta')

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in models_in(d):
        st = style(m)
        ax.plot(v, d[f'{m}_master'], color=st['color'], ls=st['ls'], lw=1.8, label=st['label'], zorder=3)

    rs = ref_style('sqrt')
    vb = np.linspace(0.01, 1.0, 60)
    ax.plot(vb, vb ** 0.5, color=rs['color'], ls=rs['ls'], lw=rs['lw'],
            label=r'$\sqrt{\cdot}$-law build-up  $v^{0.5}$', zorder=4)
    ax.axvline(1.0, color='#9a9a9a', lw=1.0, ls=':', zorder=2)
    ax.axhline(1.0, color='#9a9a9a', lw=0.7, zorder=2)
    if not build_up:
        rt = ref_style('twothirds')
        ax.axhline(2 / 3, color=rt['color'], ls=rt['ls'], lw=rt['lw'], alpha=0.8,
                   label=r'permanent $\approx \frac{2}{3}$', zorder=2)

    ax.set_xlabel(r'$v$ = fraction of metaorder executed   ($v=1$: execution end)')
    ax.set_ylabel(r'$\mathrm{master}(v)=\langle I(v)\rangle/\langle I(1)\rangle$')
    phase = 'build-up' if build_up else 'build-up + relaxation'
    ax.set_title(f'{args.stock} — impact master curve ({phase})')
    ax.margins(x=0)
    # legends in the empty corner (override with --loc):
    # build-up -> upper-right;  relaxation -> upper-left (top is empty until CST climbs late).
    loc = args.loc or ('upper right' if build_up else 'upper left')
    legend_box(ax, loc=loc, ncol=1)

    savefig_pub(fig, os.path.abspath(args.out))
    plt.close(fig)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
