#!/usr/bin/env python3
"""
Body figure: the FULL estimator x cross-section grid with sample counts.
Rows: signed binned | direct L2 | direct L1 | n(k) points entering each fit.
Cols: cumulative <=k | exact =k | reverse >=k.
One shared legend under the figure; Hawkes drawn thick (the instability story).
Pure cache re-render: beta_3x3_<ST>.npz (curves) + mid_trajectory K (counts).

  python beta_3x3_body.py --stock NVDA [--copy_to <dir>]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ORDER = ['Historic', 'Heuristic', 'Propagator', 'Hawkes', 'CST', 'NMZI',
         'Mamba3', 'GDN', 'S5_120M', 'S5', 'Mamba3_4k', 'S5_4k']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
ROWS = [('binned', 'signed binned'), ('l2', r'direct $L_2$ (canon)'), ('l1', r'direct $L_1$')]
VIEWS = [('le', r'cumulative $\leq k$'), ('eq', r'exact $=k$'), ('ge', r'reverse $\geq k$')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    z3 = np.load(os.path.join(here, 'results', 'beta_3x3', f'beta_3x3_{args.stock}.npz'))
    zk = np.load(os.path.join(here, 'results', 'mid_impact', f'mid_trajectory_{args.stock}_beta.npz'),
                 allow_pickle=True)
    ks = z3['ks']
    models = [m for m in ORDER if f'{m}_l2_le' in z3.files]

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11.5})
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 13.0), sharex=True)

    for ri, (est, rlabel) in enumerate(ROWS):
        for ci, (view, vlabel) in enumerate(VIEWS):
            ax = axes[ri][ci]
            for m in models:
                ax.plot(ks, z3[f'{m}_{est}_{view}'], color=COLORS.get(m, '#444444'),
                        lw=2.6 if m == 'Hawkes' else 1.6,
                        alpha=1.0 if m == 'Hawkes' else 0.9)
            ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0)
            ax.axhline(0.0, color='#cccccc', lw=0.8)
            ax.set_ylim(-0.6, 1.6)
            if ri == 0:
                ax.set_title(vlabel, fontsize=13)
            if ci == 0:
                ax.set_ylabel(f'{rlabel}\n' + r'$\delta(k)$', fontsize=11)

    # bottom row: n(k) entering each fit, per model (K matrices carry BOTH sides)
    fins = {}
    for m in models:
        if f'{m}_K' in zk.files:
            fins[m] = np.isfinite(zk[f'{m}_K'][:, 1:]).sum(axis=0)  # per insertion k=1..100
    for ci, (view, _) in enumerate(VIEWS):
        ax = axes[3][ci]
        for m, fin in fins.items():
            kk = np.arange(1, len(fin) + 1)
            if view == 'le':
                n = np.cumsum(fin)
            elif view == 'eq':
                n = fin.astype(float)
            else:
                n = np.cumsum(fin[::-1])[::-1]
            ax.plot(kk, n, color=COLORS.get(m, '#444444'),
                    lw=2.4 if m == 'Hawkes' else 1.4)
        ax.set_yscale('log')
        ax.axhline(200, color='#555555', lw=1.0, ls=':')
        ax.set_xlabel('insertion index $k$')
        if ci == 0:
            ax.set_ylabel('points entering the fit\n$n(k)$ (log; dotted: MIN\\_PTS)', fontsize=10.5)

    handles = [Line2D([], [], color=COLORS.get(m, '#444444'), lw=2.8, label=m.replace('_', '-'))
               for m in models]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6),
               fontsize=11.5, frameon=False, bbox_to_anchor=(0.5, -0.002))
    fig.suptitle(f'{args.stock}: exponent dynamics under three estimators (rows) and three '
                 f'cross-sections (columns), with the sample counts behind every fit (bottom row)',
                 fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    out = os.path.join(here, 'results', 'beta_3x3', f'beta_3x3_body_{args.stock}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('B3X3BODY ->', out)
    if args.copy_to:
        import shutil; shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)


if __name__ == '__main__':
    main()
