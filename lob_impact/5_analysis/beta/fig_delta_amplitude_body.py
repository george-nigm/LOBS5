#!/usr/bin/env python3
"""
Body figure: the headline estimator only — signed binned, cumulative <= k — for both quantities.

The 3x3 grid (beta_3x3_body.py) exists to show that the exponent is not an artefact of one
estimator or one cross-section. That is a robustness argument and it belongs in the appendix; the
body needs the single row a reader actually reads: the literature-standard signed-binned estimator
on the cumulative pool.

  left   delta(<=k): how impact grows with size. Target 0.5 (square-root law).
  right  Y(<=k):     what one pays per unit at the data edge, same fit, log scale.
                     Empirical Y ~ 0.5 (Toth 2011, Zarinelli 2015); below the 0.05 floor ANY
                     exponent fits equally well, so delta there is not identified.

Both panels carry the identifiability mask of beta_3x3_body.py: a delta whose amplitude sits on
zero, or which ran to the fit's own search bounds, is a non-estimate and is suspended rather than
drawn as a measurement. Suspended stretches appear as bars below the axis, per model.

Pure cache re-render — beta_3x3_<ST>.npz + beta_sigma_grid_<ST>_le.npz + bias_exhibit_<ST>.npz.

  python fig_delta_amplitude_body.py --stock AMD [--copy_to <dir>]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _legend_below(fig, handles, labels, y=-0.04):
    """Shared legend rule (lob_impact/core/model_style.py): 3 entries per column, one size for
    every figure. Falls back to a plain bottom legend if the module is unreachable."""
    import os, importlib.util, math
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        p = os.path.join(d, 'core', 'model_style.py')
        if os.path.exists(p):
            sp = importlib.util.spec_from_file_location('_lob_model_style', p)
            m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
            return m.legend_below(fig, handles, labels, y=y)
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return fig.legend(handles, labels, loc='lower center', frameon=False, fontsize=9.5,
                      ncol=max(1, math.ceil(len(labels) / 3)), bbox_to_anchor=(0.5, y))



def _canon_palette():
    import os as _os
    import importlib.util
    d = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(4):
        p = _os.path.join(d, 'core', 'model_style.py')
        if _os.path.exists(p):
            sp = importlib.util.spec_from_file_location('_lob_model_style', p)
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            return dict(m.COLORS), list(m.CANON)
        nd = _os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise SystemExit('core/model_style.py не найден — палитра должна быть общей')


COLORS, ORDER = _canon_palette()
P2R = np.sqrt(4 * np.log(2.0))       # 1.665: sigma_range = P2R * sigma_parkinson
Y_EMP, Y_MIN = 0.5, 0.05
D_LO, D_HI = 0.05, 1.50              # the fit's own search bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--copy_to', default=None)
    ap.add_argument('--est', default='l2', choices=['l2', 'binned', 'l1'],
                    help="estimator. Default l2 — the paper fixes 'direct L2 on unfiltered signed "
                         "points, cumulative <=k' as the canonical configuration, and the scorecard "
                         "quotes it, so the body figure must plot the same thing.")
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    z3 = np.load(os.path.join(here, 'results', 'beta_3x3', f'beta_3x3_{args.stock}.npz'))
    zsg = np.load(os.path.join(here, 'results', 'beta_sigma_grid',
                               f'beta_sigma_grid_{args.stock}_le.npz'))
    zb = np.load(os.path.join(here, 'results', 'bias_exhibit', f'bias_exhibit_{args.stock}.npz'))
    rm = ('Mamba3' if 'Mamba3_parkinson_xc' in zb.files
          else sorted(k for k in zb.files if k.endswith('_parkinson_xc'))[0][:-len('_parkinson_xc')])
    q_ref = float(np.exp(zb[f'{rm}_parkinson_xc'][-1]))

    ks = z3['ks']
    models = [m for m in ORDER if f'{m}_{args.est}_le' in z3.files]
    if not models:
        print(f'нет моделей в beta_3x3_{args.stock}.npz')
        return 1

    AMP = {'binned': ('ibin', 'dbin'), 'l2': ('al2', 'dl2'), 'l1': ('al1', 'dl1')}
    akey, dkey = AMP[args.est]

    def amplitude(m):
        ka, kd = f'{m}_parkinson_{akey}', f'{m}_parkinson_{dkey}'
        if ka not in zsg.files or kd not in zsg.files:
            return None
        a = np.asarray(zsg[ka], float)
        d = np.asarray(zsg[kd], float)
        amp = np.exp(a) if args.est == 'binned' else np.abs(a)
        return amp * q_ref ** (d - 0.5) / P2R

    def identified(m):
        """Same two criteria as the 3x3 body panel: amplitude off zero, and delta off the bounds."""
        Y = amplitude(m)
        d = np.asarray(z3[f'{m}_{args.est}_le'], float)
        ok = np.ones(len(d), bool)
        if Y is not None:
            n = min(len(Y), len(ok))
            ok[:n] &= np.abs(Y[:n]) >= Y_MIN
        ok &= np.isfinite(d) & (d > D_LO + 1e-6) & (d < D_HI - 1e-6)
        return ok

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})
    fig, (axd, axy) = plt.subplots(1, 2, figsize=(13.6, 5.4))

    handles, labels = [], []
    for m in models:
        c = COLORS.get(m, '#444444')
        ok = identified(m)
        d = np.asarray(z3[f'{m}_{args.est}_le'], float)
        shown = np.where(ok, d, np.nan)
        ln, = axd.plot(ks, shown, color=c, lw=2.0)
        handles.append(ln)
        labels.append(m.replace('_', '-'))
        if (~ok).any():                     # suspended stretch, staggered so they stay separable
            off = -0.055 - 0.030 * models.index(m)
            axd.plot(ks, np.where(~ok, off, np.nan), color=c, lw=2.2, solid_capstyle='butt')

        Y = amplitude(m)
        if Y is None:
            continue
        n = min(len(Y), len(ks), len(ok))
        Yv = np.where(ok[:n], Y[:n], np.nan)
        axy.plot(ks[:n], np.abs(np.where(Yv > 0, Yv, np.nan)), color=c, lw=2.0)
        axy.plot(ks[:n], np.abs(np.where(Yv < 0, Yv, np.nan)), color=c, lw=1.6, ls='--')

    axd.axhline(0.5, color='#B03A2E', ls='--', lw=1.6)
    axd.text(0.985, 0.5, r'$\delta=0.5$', transform=axd.get_yaxis_transform(),
             color='#B03A2E', va='bottom', ha='right', fontsize=10)
    axd.axhline(0.0, color='#cccccc', lw=0.8)
    axd.set_ylim(-0.06 - 0.030 * len(models), 1.55)
    axd.set_xlabel('children executed $k$ (cumulative pool $\\leq k$)')
    axd.set_ylabel(r'build-up exponent $\delta(k)$')
    axd.set_title('how impact grows with size', fontsize=12.5)
    axd.grid(alpha=0.25)
    axd.text(0.015, 0.015, 'bars below the axis: no law identified at that $k$',
             transform=axd.transAxes, fontsize=8.5, color='#777777')

    axy.set_yscale('log')
    axy.axhline(Y_EMP, color='#1E8449', ls='--', lw=1.6)
    axy.text(0.985, Y_EMP, r'empirical $Y\approx0.5$', transform=axy.get_yaxis_transform(),
             color='#1E8449', va='bottom', ha='right', fontsize=10)
    axy.axhline(Y_MIN, color='#999999', ls=':', lw=1.4)
    axy.text(0.015, Y_MIN, 'identifiability floor', transform=axy.get_yaxis_transform(),
             color='#999999', va='bottom', ha='left', fontsize=9)
    axy.set_xlabel('children executed $k$ (cumulative pool $\\leq k$)')
    axy.set_ylabel(r'implied amplitude $Y(k)$  (log)')
    axy.set_title('what one pays per unit at the data edge', fontsize=12.5)
    axy.grid(alpha=0.25, which='both')
    axy.text(0.985, 0.015, 'dashed: fitted amplitude negative (adverse drift), $|Y|$ shown',
             transform=axy.transAxes, fontsize=8.5, color='#777777', ha='right')

    _ESTLAB = {'l2': r'direct $L_2$ (canonical)', 'binned': 'signed binned', 'l1': r'direct $L_1$'}
    fig.suptitle(f'{args.stock}: {_ESTLAB[args.est]}, cumulative $\\leq k$ — exponent and amplitude',
                 fontsize=13.5)
    _legend_below(fig, handles, labels, y=-0.02)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))

    outdir = os.path.join(here, 'results', 'beta_3x3')
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f'delta_amplitude_body_{args.stock}{args.suffix}.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), ks=ks,
                        **{f'{m}_delta': np.asarray(z3[f'{m}_{args.est}_le'], float) for m in models},
                        **{f'{m}_ok': identified(m) for m in models},
                        **{f'{m}_Y': amplitude(m) for m in models if amplitude(m) is not None})
    print(f'DELTA_AMP_BODY -> {png}  ({len(models)} моделей)')
    if args.copy_to:
        import shutil
        shutil.copy(png, args.copy_to)
        print('copied ->', args.copy_to)
    return 0


if __name__ == '__main__':
    sys.exit(main())
