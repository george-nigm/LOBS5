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

# OW (Obizhaeva-Wang exponential-resilience kernel) and QR (queue-reactive) sit with the
# other mechanical baselines, between Propagator and the point-process models.
ORDER = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR', 'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'OW': '#6A1B9A', 'QR': '#455A64',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
ROWS = [('binned', 'signed binned'), ('l2', r'direct $L_2$ (canon)'), ('l1', r'direct $L_1$')]
VIEWS = [('le', r'cumulative $\leq k$'), ('eq', r'exact $=k$'), ('ge', r'reverse $\geq k$')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--copy_to', default=None)
    ap.add_argument('--mask', action='store_true', help='identifiability mask (OFF by default pending decision)')
    ap.add_argument('--mask_style', default='zero', choices=['zero', 'dots'],
                    help="zero: non-identified segments pinned at delta-axis zero (suspended); dots: faint dotted raw values")
    ap.add_argument('--suffix', default='', help='output filename suffix')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    z3 = np.load(os.path.join(here, 'results', 'beta_3x3', f'beta_3x3_{args.stock}.npz'))
    zk = np.load(os.path.join(here, 'results', 'mid_impact', f'mid_trajectory_{args.stock}_beta.npz'),
                 allow_pickle=True)
    ks = z3['ks']
    models = [m for m in ORDER if f'{m}_l2_le' in z3.files]
    # identifiability mask: suppress delta where the fitted amplitude is on zero
    # (implied |Y| at the data edge below 0.05 — an order under the empirical
    # band). A flat misfit profile has no bottom; its argmin is a grid-bound
    # artefact (the 0.05<->1.5 teleport), not an estimate.
    P2R = np.sqrt(4 * np.log(2.0))
    Y_MIN = 0.05
    try:
        if not args.mask:
            raise RuntimeError('mask disabled by flag')
        zb = np.load(os.path.join(here, 'results', 'bias_exhibit', f'bias_exhibit_{args.stock}.npz'))
        rm = 'Mamba3' if 'Mamba3_parkinson_xc' in zb.files else sorted(k for k in zb.files if k.endswith('_parkinson_xc'))[0][:-len('_parkinson_xc')]
        q_ref = float(np.exp(zb[f'{rm}_parkinson_xc'][-1]))
        zsg = {v: np.load(os.path.join(here, 'results', 'beta_sigma_grid',
                                       f'beta_sigma_grid_{args.stock}_{v}.npz')) for v in ('le', 'eq', 'ge')}
        AMP = {'binned': 'ibin', 'l2': 'al2', 'l1': 'al1'}
        def _Y(m, est, view):
            zz = zsg[view]
            key_a = f'{m}_parkinson_{AMP[est]}'; key_d = f'{m}_parkinson_d{"bin" if est == "binned" else est}'
            if key_a not in zz.files:
                return None
            a = zz[key_a]; d = zz[key_d]
            amp = np.exp(a) if est == 'binned' else np.abs(a)
            return np.abs(amp * q_ref ** (d - 0.5) / P2R)
        def ok_mask(m, est, view):
            # the row's own amplitude AND the L2 arbiter must both clear the
            # floor: amplitude-on-zero is estimator-independent physics
            Yo = _Y(m, est, view); Y2 = _Y(m, 'l2', view)
            if Yo is None or Y2 is None:
                return np.ones(len(ks), bool)
            return (Yo >= Y_MIN) & (Y2 >= Y_MIN)
    except Exception as e:
        print('mask disabled:', e)
        ok_mask = lambda m, est, view: np.ones(len(ks), bool)

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11.5})
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.2), sharex=True)

    for ri, (est, rlabel) in enumerate(ROWS):
        for ci, (view, vlabel) in enumerate(VIEWS):
            ax = axes[ri][ci]
            for m in models:
                v = z3[f'{m}_{est}_{view}'].astype(float).copy()
                msk = ok_mask(m, est, view)
                shown = v.copy(); shown[~msk] = np.nan
                ax.plot(ks, shown, color=COLORS.get(m, '#444444'),
                        lw=2.6 if m == 'Hawkes' else 1.6,
                        alpha=1.0 if m == 'Hawkes' else 0.9)
                if (~msk).any():
                    hidden = v.copy(); hidden[msk] = np.nan
                    if args.mask_style == 'zero':
                        # SUSPENDED: no law identified -> pinned at zero (tiny stagger so
                        # overlapping suspended models stay distinguishable)
                        off = -0.022 * models.index(m)
                        zline = np.where(np.isfinite(hidden), off, np.nan)
                        ax.plot(ks, zline, color=COLORS.get(m, '#444444'), lw=1.8, ls='-', alpha=0.9)
                    else:
                        ax.plot(ks, hidden, color=COLORS.get(m, '#444444'), lw=0.8, ls=':', alpha=0.35)
            ax.axhline(0.5, color='#C0392B', ls='--', lw=1.0)
            ax.axhline(0.0, color='#cccccc', lw=0.8)
            ax.set_ylim(-0.6, 1.6)
            if ri == 0:
                ax.set_title(vlabel, fontsize=13)
            if ci == 0:
                ax.set_ylabel(f'{rlabel}\n' + r'$\delta(k)$', fontsize=11)
            if ri == 2:
                ax.set_xlabel('insertion index $k$')

    # counts go into the legend labels (K matrices carry both sides)
    fins = {}
    for m_ in models:
        if f'{m_}_K' in zk.files:
            fins[m_] = int(np.isfinite(zk[f'{m_}_K'][:, 1:]).sum())

    handles = [Line2D([], [], color=COLORS.get(m, '#444444'), lw=2.8,
                  label=f"{m.replace('_', '-')} ({fins.get(m, 0)//1000}k pts)")
               for m in models]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 4),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.065))
    fig.suptitle(f'{args.stock}: exponent dynamics under three estimators (rows) and three '
                 f'cross-sections (columns); per-model point counts in the legend',
                 fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0.075, 1, 0.975])
    suff = f'_{args.suffix}' if args.suffix else ''
    out = os.path.join(here, 'results', 'beta_3x3', f'beta_3x3_body_{args.stock}{suff}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('B3X3BODY ->', out)
    if args.copy_to:
        import shutil; shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)


if __name__ == '__main__':
    main()
