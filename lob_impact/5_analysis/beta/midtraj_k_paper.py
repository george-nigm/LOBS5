#!/usr/bin/env python3
"""
Paper Figure "Response to the Full Metaorder": the k-clock (executed-volume) mid-price impact,
re-rendered in pubstyle from the mid_trajectory.py npz caches — no grid access, runs in seconds.

  python 5_analysis/beta/midtraj_k_paper.py --shape beta
  python 5_analysis/beta/midtraj_k_paper.py --shape decay
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(B, '..'))
sys.path.insert(0, B)
from pubstyle import apply, style, ref_style, legend_box, savefig_pub   # noqa: E402
from mid_trajectory import propagator_curve                            # noqa: E402
apply()

MODELS = ['Historic', 'Heuristic', 'Hawkes', 'CST', 'NMZI', 'Propagator', 'Mamba3', 'Mamba3_4k', 'S5_4k', 'GDN']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', default='beta', choices=['beta', 'decay'])
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--npz', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    npz = args.npz or os.path.join(B, 'results', 'mid_impact',
                                   f'mid_trajectory_{args.stock}_{args.shape}.npz')
    d = np.load(npz, allow_pickle=True)
    n_ins = 100 if args.shape == 'beta' else 10

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    nb = None
    for m in MODELS:
        if f'{m}_k_mean' not in d.files:
            continue
        km, kse = d[f'{m}_k_mean'], d[f'{m}_k_se']
        nb = len(km) - 1
        x = np.arange(len(km))
        st = style(m)
        n = int(np.nanmax(d[f'{m}_cnt'])) if f'{m}_cnt' in d.files else 0
        end = km[np.isfinite(km)][-1]
        ax.fill_between(x, km - 1.96 * kse, km + 1.96 * kse,
                        color=st['color'], alpha=0.12, lw=0)
        ax.plot(x, km, color=st['color'], ls=st['ls'], lw=1.8,
                label=f"{st['label']} — {end:.1f} bps (n={n})", zorder=3)

    # √-law reference on the k-clock (per-day child volume, x = k directly)
    ys = d['sqrt_y']
    rs = ref_style('sqrt')
    xk = np.arange(1, n_ins + 1, dtype=float)
    yk = ys[:n_ins].copy()
    if args.shape != 'beta':
        xk = np.append(xk, float(nb)); yk = np.append(yk, yk[-1])   # permanent-ref continuation
    ax.plot(xk, yk, color=rs['color'], ls=rs['ls'], lw=rs['lw'],
            label=rf"$\sqrt{{\cdot}}$-law $\beta=0.5$ — {yk[-1]:.2f} bps", zorder=4)
    if args.shape != 'beta':
        ps = ref_style('propagator')
        pxk, pyk = propagator_curve(float(yk[n_ins - 1]), float(n_ins), None, 0.5, nb)
        ax.plot(pxk, pyk, color=ps['color'], ls=ps['ls'], lw=ps['lw'],
                label=rf"propagator $\beta=0.5$ — peak {yk[n_ins - 1]:.2f}"
                      rf"$\,\to\,${pyk[-1]:.2f} bps", zorder=4)
        ax.axvline(n_ins, color='#9a9a9a', lw=1.0, ls=':', zorder=2)

    ax.axhline(0, color='#9a9a9a', lw=0.7, zorder=1)
    ax.margins(x=0)
    if args.shape == 'beta':
        ax.set_xlabel(r'insertion index $k$  (children executed)')
        ax.set_title(f'{args.stock} — metaorder build-up on the executed-volume clock')
    else:
        ax.set_xlabel(r'boundary index $k$  (insertions $1$--$10$, then cooling windows)')
        ax.set_title(f'{args.stock} — build-up and relaxation on the executed-volume clock')
    ax.set_ylabel(r'mean signed mid-price change  (bps)')
    legend_box(ax, loc='upper left')
    out = args.out or os.path.join(B, 'results', 'mid_impact',
                                   f'pub_midtraj_k_{args.stock}_{args.shape}.png')
    fig.tight_layout()
    savefig_pub(fig, os.path.abspath(out))
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
