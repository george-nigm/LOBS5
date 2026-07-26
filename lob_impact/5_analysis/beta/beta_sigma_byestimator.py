#!/usr/bin/env python3
"""
One-estimator lens on the sigma grid: rows = cross-sections (<=k / =k / >=k),
cols = all six sigma normalisations, delta(k) curves per model — for ONE estimator.

Pure re-render from the beta_sigma_grid npz caches (no grid access).

  python beta_sigma_byestimator.py --stock NVDA --estimator l2   # dl2 | dbin | dl1
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E', 'OW': '#6A1B9A', 'QR': '#00ACC1'}
SIGMAS = ['none', 'parkinson', 'garman_klass', 'rogers_satchell', 'close_to_close', 'yang_zhang']
VIEWS = [('le', r'cumulative $\leq k$'), ('eq', r'exact $=k$'), ('ge', r'reverse $\geq k$')]
EST = {'l2': ('dl2', 'direct $L_2$'), 'l1': ('dl1', 'direct $L_1$'), 'bin': ('dbin', 'signed binned')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--estimator', default='l2', choices=list(EST))
    args = ap.parse_args()
    tag, elabel = EST[args.estimator]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', 'beta_sigma_grid')

    fig, axes = plt.subplots(3, len(SIGMAS), figsize=(3.0 * len(SIGMAS), 8.6),
                             sharex=True, sharey=True)
    for ri, (view, vlabel) in enumerate(VIEWS):
        z = np.load(os.path.join(outdir, f'beta_sigma_grid_{args.stock}_{view}.npz'))
        ks = z['ks']
        models = sorted({k.split('_' + SIGMAS[0] + '_')[0] for k in z.files
                         if f'_{SIGMAS[0]}_{tag}' in k})
        for ci, sm in enumerate(SIGMAS):
            ax = axes[ri][ci]
            for m in models:
                key = f'{m}_{sm}_{tag}'
                if key in z.files:
                    ax.plot(ks, z[key], color=COLORS.get(m, '#444444'), lw=1.5,
                            label=m if (ri == 0 and ci == 0) else None)
            ax.axhline(0.5, color='#555555', lw=0.9, ls='--')
            ax.axhline(0.0, color='#bbbbbb', lw=0.7)
            ax.set_ylim(-0.6, 1.7)
            if ri == 0:
                ax.set_title(sm, fontsize=10)
            if ci == 0:
                ax.set_ylabel(vlabel + '\n' + r'$\delta(k)$', fontsize=9)
            if ri == 2:
                ax.set_xlabel('$k$')
    axes[0][0].legend(fontsize=6.6, ncol=2, loc='lower right')
    fig.suptitle(f'{args.stock}: {elabel} exponent across three cross-sections (rows) '
                 f'and six sigma normalisations (cols); dashed = 0.5', y=0.995)
    fig.tight_layout()
    out = os.path.join(outdir, f'beta_sigma_{args.estimator}_{args.stock}.png')
    fig.savefig(out, dpi=145, bbox_inches='tight')
    print(f'BY_ESTIMATOR_DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
