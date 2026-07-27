#!/usr/bin/env python
"""
Redraw the flow-balance figure from its cache, with one shared legend and the numbers in a table.

flow_balance.py recomputes from the grid, which takes far longer than a layout change deserves.
Everything the figure needs is already in flow_balance_<ST>.npz (<model>_C, _V, _R, _gamma, _H,
_H_ci), so this redraws from there in seconds.

What changes: the three panels used to carry a legend each, 14 entries apiece, with the fitted
numbers inside the labels — three boxes sitting on top of the curves they described. Now there is
one legend under the figure with names only, and gamma_flow / H go to a markdown table written next
to the png, ready to paste as a table rather than squinted at inside a plot.

  python redraw_flow_balance.py --stock AMD [--copy_to <dir>]
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



def _canon():
    import importlib.util
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        p = os.path.join(d, 'core', 'model_style.py')
        if os.path.exists(p):
            sp = importlib.util.spec_from_file_location('_ms', p)
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            return dict(m.COLORS), list(m.CANON), m.REAL
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise SystemExit('core/model_style.py не найден')


COLORS, CANON, C_REAL = _canon()
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()

    npz = args.npz or os.path.join(HERE, 'results', f'causal_{args.stock}',
                                   f'flow_balance_{args.stock}.npz')
    if not os.path.exists(npz):
        print('нет кэша:', npz)
        return 1
    z = np.load(npz, allow_pickle=True)

    present = [m for m in ['Real'] + CANON if f'{m}_C' in z.files]
    if not present:
        print('в кэше нет кривых')
        return 1

    plt.rcParams.update({'font.size': 10.5})
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    rows = []
    for m in present:
        c = C_REAL if m == 'Real' else COLORS.get(m, '#444444')
        lw = 2.4 if m == 'Real' else 1.5
        C = np.asarray(z[f'{m}_C'], float)
        V = np.asarray(z[f'{m}_V'], float)
        R = np.asarray(z[f'{m}_R'], float)
        lab = 'real stream' if m == 'Real' else m.replace('_', '-')
        ok = C > 0
        axes[0].loglog(np.arange(1, len(C) + 1)[ok], C[ok], color=c, lw=lw, label=lab)
        axes[1].loglog(np.arange(1, len(V) + 1), V, color=c, lw=lw)
        axes[2].plot(np.arange(1, len(R) + 1), R, color=c, lw=lw)
        ci = np.asarray(z[f'{m}_H_ci'], float) if f'{m}_H_ci' in z.files else (np.nan, np.nan)
        rows.append((lab, float(z[f'{m}_gamma']), float(z[f'{m}_H']), float(ci[0]), float(ci[1])))

    axes[0].set_title(r'trade-sign autocorrelation $C(\ell)$' '\n' '(long memory: slow power-law decay)',
                      fontsize=10.5)
    axes[1].set_title('mid displacement variance vs lag\n'
                      r'(slope $=2H$; efficiency $\Rightarrow H \approx 0.5$)', fontsize=10.5)
    axes[2].set_title(r'response $R(\ell) = E[\epsilon_t (m_{t+\ell}-m_t)]$, ticks', fontsize=10.5)
    for a in axes:
        a.set_xlabel(r'lag $\ell$ (trades)')
        a.grid(alpha=0.22, which='both')
    axes[2].axhline(0, color='#cccccc', lw=0.8)

    fig.suptitle(f'{args.stock}: flow persistence vs price diffusivity '
                 r'(balance requires $\beta \approx (1-\gamma)/2$)', fontsize=12.5)
    h, l = axes[0].get_legend_handles_labels()
    _legend_below(fig, h, l, y=-0.06)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))

    out = os.path.join(os.path.dirname(npz), f'flow_balance_{args.stock}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')

    tbl = out.replace('.png', '_stats.md')
    real = next((r for r in rows if r[0] == 'real stream'), None)
    with open(tbl, 'w') as fh:
        fh.write(f'Flow balance — {args.stock}\n\n')
        fh.write('| model | gamma_flow | H | 95% CI on H |\n|---|---|---|---|\n')
        for name, g, H, lo, hi in rows:
            fh.write(f'| {name} | {g:.2f} | {H:.2f} | [{lo:.2f}, {hi:.2f}] |\n')
    print('saved ->', out)
    print('stats ->', tbl, f'({len(rows)} моделей)')
    if real:
        print(f'  цель (реальный поток): gamma={real[1]:.2f}, H={real[2]:.2f}')
    if args.copy_to:
        import shutil
        shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)
    return 0


if __name__ == '__main__':
    sys.exit(main())
