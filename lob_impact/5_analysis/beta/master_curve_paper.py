#!/usr/bin/env python3
"""
Paper Figure "master curve": pubstyle re-render from the master_curve.py npz cache with a COMPACT
legend — the four impact-blind baselines (⟨I(1)⟩ ≈ 0, not normalisable) collapse into one entry
instead of four boxed lines that cover half the plot. No grid access, runs in seconds.

  python 5_analysis/beta/master_curve_paper.py --shape beta
  python 5_analysis/beta/master_curve_paper.py --shape relaxation
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
from mid_trajectory import propagator_curve                             # noqa: E402
apply()

MODELS = ['Historic', 'Heuristic', 'Hawkes', 'CST', 'Propagator', 'Mamba3', 'Mamba3_4k', 'S5_4k', 'GDN']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', default='beta', choices=['beta', 'relaxation'])
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--npz', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    npz = args.npz or os.path.join(B, 'results', 'master_curve',
                                   f'master_curve_{args.stock}_{args.shape}_v2gated.npz')
    d = np.load(npz, allow_pickle=True)
    vgrid = d['vgrid']
    build_up = (args.shape == 'beta')

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    blind = []
    sig_masters = []
    for m in MODELS:
        if f'{m}_master' not in d.files:
            continue
        st = style(m)
        n = int(d[f'{m}_n'])
        if bool(d[f'{m}_sig']):
            master = d[f'{m}_master']
            ax.plot(vgrid, master, color=st['color'], ls=st['ls'], lw=1.8,
                    label=f"{st['label']} (n={n})", zorder=3)
            sig_masters.append(master)
        else:
            blind.append(st['label'])
    if blind:
        # one compact entry for all non-normalisable baselines (⟨I(1)⟩ ~ noise)
        ax.plot([], [], color='#8a8a8a', lw=1.2, alpha=0.7,
                label=f"{', '.join(blind)}:\n" + r"$\langle I(1)\rangle \approx 0$ — not normalisable")

    if sig_masters:
        smax = np.nanmax([np.nanmax(s) for s in sig_masters])
        smin = np.nanmin([np.nanmin(s) for s in sig_masters])
        pad = 0.15 * max(smax - smin, 1.0)
        ax.set_ylim(min(smin, 0) - pad, max(smax, 1.0) + pad)

    rs = ref_style('sqrt')
    vb = np.linspace(0.01, 1.0, 60)
    ax.plot(vb, vb ** 0.5, color=rs['color'], ls=rs['ls'], lw=rs['lw'],
            label=r'$\sqrt{\cdot}$-law build-up  $v^{0.5}$', zorder=4)
    ax.axvline(1.0, color='#9a9a9a', lw=1.0, ls=':', zorder=2)
    ax.axhline(1.0, color='#9a9a9a', lw=0.7, zorder=2)
    if not build_up:
        # THE theoretical relaxation: smooth propagator-shaped descent from the peak toward the
        # permanent level ≈ 2/3 of peak (fair pricing) — one curve, ref(v) = 2/3 + 1/3·prop(v),
        # where prop(v) = v^0.5 - (v-1)^0.5 is the constant-rate transient (G(l)~l^-0.5, ->0).
        rt = ref_style('twothirds')
        px, prop = propagator_curve(1.0, 1.0, None, 0.5, float(vgrid[-1]))
        py = 2 / 3 + prop / 3
        ax.plot(px, py, color=rt['color'], ls=rt['ls'], lw=2.0,
                label=r'theory: relax to permanent $\approx \frac{2}{3}$ of peak', zorder=4)
    ax.set_xlabel(r'$v$ = fraction of metaorder executed   ($v=1$: execution end)')
    ax.set_ylabel(r'$\mathrm{master}(v)=\langle I(v)\rangle/\langle I(1)\rangle$')
    phase = 'build-up' if build_up else 'build-up + relaxation'
    ax.set_title(f'{args.stock} — impact master curve ({phase})')
    ax.margins(x=0)
    legend_box(ax, loc='upper left' if build_up else 'lower right', ncol=1)
    out = args.out or os.path.join(B, 'results', 'master_curve',
                                   f'pub_master_{args.stock}_{args.shape}.png')
    fig.tight_layout()
    savefig_pub(fig, os.path.abspath(out))
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
