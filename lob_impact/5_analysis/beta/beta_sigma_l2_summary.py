#!/usr/bin/env python3
"""
L2 lens across the robustness cube: delta_L2(k) with rows = cross-sections
(<=k / =k / >=k) and columns = the six sigma normalisations. Rendered from the
beta_sigma_grid npz caches — no grid access.

  python beta_sigma_l2_summary.py --stock NVDA
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'Historic': '#C0392B', 'Hawkes': '#D4AC0D', 'Mamba3': '#2F5DA3',
          'GDN': '#D81B60', 'S5_120M': '#F06292', 'Mamba3_4k': '#16A085',
          'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
SIGMAS = ['none', 'parkinson', 'garman_klass', 'rogers_satchell', 'close_to_close', 'yang_zhang']
VIEWS = [('le', r'cumulative $\leq k$'), ('eq', r'exact $=k$'), ('ge', r'reverse $\geq k$')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, 'results', 'beta_sigma_grid')
    Z = {v: np.load(os.path.join(d, f'beta_sigma_grid_{args.stock}_{v}.npz')) for v, _ in VIEWS}
    ks = Z['le']['ks']
    models = sorted({k.split('_' + SIGMAS[0] + '_')[0] for k in Z['le'].files
                     if f'_{SIGMAS[0]}_dl2' in k})

    fig, axes = plt.subplots(3, len(SIGMAS), figsize=(3.0 * len(SIGMAS), 9.0),
                             sharex=True, sharey=True)
    for ri, (v, vlabel) in enumerate(VIEWS):
        for ci, sm in enumerate(SIGMAS):
            ax = axes[ri][ci]
            for m in models:
                key = f'{m}_{sm}_dl2'
                if key in Z[v].files:
                    ax.plot(ks, Z[v][key], color=COLORS.get(m, '#444444'), lw=1.5,
                            label=m if (ri == 0 and ci == 0) else None)
            ax.axhline(0.5, color='#555555', lw=0.9, ls='--')
            ax.set_ylim(-0.1, 1.6)
            if ri == 0:
                ax.set_title(sm, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f'{vlabel}\n' + r'$\delta_{L_2}(k)$', fontsize=9)
            if ri == 2:
                ax.set_xlabel('$k$')
    axes[0][0].legend(fontsize=6.6, ncol=2, loc='upper right')
    fig.suptitle(f'{args.stock}: direct-$L_2$ exponent across the robustness cube '
                 f'(rows = cross-sections, cols = sigma normalisations)', y=0.995)
    fig.tight_layout()
    png = os.path.join(d, f'beta_sigma_l2_{args.stock}.png')
    fig.savefig(png, dpi=145, bbox_inches='tight')
    print(f'SIGMA_L2_DONE -> {png}', flush=True)


if __name__ == '__main__':
    main()
