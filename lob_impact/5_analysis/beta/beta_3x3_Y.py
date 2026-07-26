#!/usr/bin/env python3
"""
Companion to the 3x3 exponent grid: the AMPLITUDE side of every fit, expressed
as implied Y(k) — the square-root-law prefactor the model's fitted curve implies
at the data-edge volume q_ref, raw-range convention:

    Y(k) = amp(k) * q_ref^(delta(k) - 0.5) / 1.665

Rows: signed binned (amp = e^alpha, the OLS intercept), direct L2 (closed-form a),
direct L1 (weighted-median a); columns: cumulative <=k / exact =k / reverse >=k.
Log y-scale; green line = empirical Y ~ 0.5 (Toth 2011, Zarinelli 2015); grey
line = 0.05 identifiability floor (an order below empirical: below it ANY
exponent fits equally well and delta is suspended). Negative fitted amplitude
(adverse drift: CST/NMZI) is drawn dashed at |Y|.

Pure cache re-render: beta_sigma_grid_<ST>_{le,eq,ge}.npz + bias_exhibit q_ref.

  python beta_3x3_Y.py --stock NVDA [--copy_to <dir>]
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
ROWS = [('binned', r'signed binned: $e^{\alpha}$', 'ibin', 'dbin'),
        ('l2', r'direct $L_2$: closed-form $a$', 'al2', 'dl2'),
        ('l1', r'direct $L_1$: weighted-median $a$', 'al1', 'dl1')]
VIEWS = [('le', r'cumulative $\leq k$'), ('eq', r'exact $=k$'), ('ge', r'reverse $\geq k$')]
P2R = np.sqrt(4 * np.log(2.0))     # 1.665: sigma_range = P2R * sigma_parkinson
Y_EMP, Y_MIN = 0.5, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    zb = np.load(os.path.join(here, 'results', 'bias_exhibit',
                              f'bias_exhibit_{args.stock}.npz'))
    rm = 'Mamba3' if 'Mamba3_parkinson_xc' in zb.files else \
        sorted(k for k in zb.files if k.endswith('_parkinson_xc'))[0][:-len('_parkinson_xc')]
    q_ref = float(np.exp(zb[f'{rm}_parkinson_xc'][-1]))
    zsg = {v: np.load(os.path.join(here, 'results', 'beta_sigma_grid',
                                   f'beta_sigma_grid_{args.stock}_{v}.npz'))
           for v, _ in VIEWS}
    ks = None
    models = [m for m in ORDER if f'{m}_parkinson_dl2' in zsg['le'].files]

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11.5})
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.2), sharex=True)

    for ri, (est, rlabel, akey, dkey) in enumerate(ROWS):
        for ci, (view, vlabel) in enumerate(VIEWS):
            ax = axes[ri][ci]
            zz = zsg[view]
            for m in models:
                a = np.asarray(zz[f'{m}_parkinson_{akey}'], float)
                d = np.asarray(zz[f'{m}_parkinson_{dkey}'], float)
                if ks is None:
                    ks = np.arange(5, 5 * len(d) + 1, 5)
                amp = np.exp(a) if est == 'binned' else a
                Y = amp * q_ref ** (d - 0.5) / P2R
                pos = np.where(Y > 0, np.abs(Y), np.nan)
                neg = np.where(Y < 0, np.abs(Y), np.nan)
                ax.plot(ks, pos, color=COLORS.get(m, '#444444'),
                        lw=2.6 if m == 'Hawkes' else 1.6,
                        alpha=1.0 if m == 'Hawkes' else 0.9)
                ax.plot(ks, neg, color=COLORS.get(m, '#444444'), lw=1.3, ls='--', alpha=0.8)
            ax.axhline(Y_EMP, color='#1B5E20', lw=1.4, ls='-', alpha=0.8)
            ax.axhline(Y_MIN, color='#888888', lw=1.1, ls=':')
            ax.set_yscale('log')
            ax.set_ylim(1e-3, 80)
            if ri == 0:
                ax.set_title(vlabel, fontsize=13)
            if ci == 0:
                ax.set_ylabel(f'{rlabel}\n' + r'implied $|Y|(k)$ at $q_{\rm ref}$', fontsize=10.5)
            if ri == 2:
                ax.set_xlabel('insertion index $k$')
    axes[0][0].annotate(r'empirical $Y \approx 0.5$', xy=(0.02, 0.5), xycoords=('axes fraction', 'data'),
                        fontsize=8.6, color='#1B5E20', va='bottom')
    axes[0][0].annotate(r'identifiability floor $0.05$', xy=(0.02, 0.05), xycoords=('axes fraction', 'data'),
                        fontsize=8.6, color='#666666', va='bottom')

    # n in the legend = points in the model's (size, impact) cloud, i.e. how much data
    # the amplitude is read off; the identifiability floor is only meaningful next to it.
    def _n(m):
        k = f'{m}_parkinson_cloud_x'
        return int(np.size(zb[k])) if k in zb.files else 0
    handles = [Line2D([], [], color=COLORS.get(m, '#444444'), lw=2.8,
                      label=f"{m.replace('_', '-')} ($n{{=}}{_n(m)}$)" if _n(m) else m.replace('_', '-'))
               for m in models]
    handles.append(Line2D([], [], color='#444444', lw=1.3, ls='--', label='negative amplitude (adverse drift)'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 4),
               fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.075))
    fig.suptitle(f'{args.stock}: implied $Y(k)$ at the data edge '
                 f'$q_{{\\rm ref}} = {q_ref:.1e}$ (raw-range convention) — the amplitude '
                 f'companion of the exponent grid', fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0.085, 1, 0.975])
    out = os.path.join(here, 'results', 'beta_3x3', f'beta_3x3_Y_{args.stock}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    np.savez_compressed(out.replace('.png', '.npz'), ks=ks, q_ref=q_ref)
    print('B3X3Y ->', out)
    if args.copy_to:
        import shutil; shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)


if __name__ == '__main__':
    main()
