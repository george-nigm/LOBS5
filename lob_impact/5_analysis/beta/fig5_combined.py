#!/usr/bin/env python3
"""
Figure 5 combined: 2x2 panel (rows = build-up / relaxation shapes; cols = k-clock
impact in bps / gated master curve) with ONE shared legend UNDER all four panels.
Pure cache re-render: mid_trajectory_<ST>_{beta,decay}.npz + master_curve_<ST>_
{beta,relaxation}_v2gated.npz. No grid access, nothing recomputed.

  python fig5_combined.py --stock NVDA [--copy_to <paper/Figures dir>]
"""
import os, argparse, json
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
GREY = '#666666'


def models_in(z, suffix):
    ms = {k[:-len(suffix)] for k in z.files if k.endswith(suffix)}
    return [m for m in ORDER if m in ms]


def draw_mid(ax, z, n_ins=None):
    for m in models_in(z, '_k_mean'):
        y = z[f'{m}_k_mean']; se = z[f'{m}_k_se']
        x = np.arange(len(y))
        c = COLORS.get(m, '#444444')
        ax.plot(x, y, color=c, lw=1.6)
        ax.fill_between(x, y - 2 * se, y + 2 * se, color=c, alpha=0.10, lw=0)
    if 'sqrt_x' in z.files:
        ax.plot(z['sqrt_x'], z['sqrt_y'], color=GREY, lw=1.4, ls='--')
    if n_ins is not None:
        ax.axvline(n_ins, color='#999999', lw=1.0, ls=':')
    ax.axhline(0, color='#cccccc', lw=0.8)
    ax.margins(x=0)


def draw_master(ax, z, relax):
    gated_out = []
    for m in models_in(z, '_master'):
        if not bool(z[f'{m}_sig']):
            gated_out.append(m); continue
        v = z['vgrid']; y = z[f'{m}_master']
        ax.plot(v, y, color=COLORS.get(m, '#444444'), lw=1.6)
    v = z['vgrid']
    if relax:
        ax.axhline(2 / 3, color=GREY, lw=1.4, ls='--')
        ax.axvline(1.0, color='#999999', lw=1.0, ls=':')
    else:
        ax.plot(v, np.sqrt(np.clip(v, 0, None)), color=GREY, lw=1.4, ls='--')
    if gated_out:
        ax.text(0.02, 0.97, 'gate-failed: ' + ', '.join(gated_out), transform=ax.transAxes,
                fontsize=6.6, color='#888888', va='top')
    ax.axhline(0, color='#cccccc', lw=0.8)
    ax.margins(x=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, 'results')
    src = {
        'mid_beta': os.path.join(res, 'mid_impact', f'mid_trajectory_{args.stock}_beta.npz'),
        'mid_decay': os.path.join(res, 'mid_impact', f'mid_trajectory_{args.stock}_decay.npz'),
        'ms_beta': os.path.join(res, 'master_curve', f'master_curve_{args.stock}_beta_v2gated.npz'),
        'ms_relax': os.path.join(res, 'master_curve', f'master_curve_{args.stock}_relaxation_v2gated.npz'),
    }
    Z = {k: np.load(v, allow_pickle=True) for k, v in src.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    draw_mid(axes[0][0], Z['mid_beta'])
    axes[0][0].set_title('build-up: mid-price impact $I(k)$ (bps, antisymmetrised)', fontsize=10)
    axes[0][0].set_ylabel('build-up shape\n$I$, bps')
    draw_master(axes[0][1], Z['ms_beta'], relax=False)
    axes[0][1].set_title(r'build-up: master curve $\langle I(v)\rangle/\langle I(1)\rangle$ (gated)', fontsize=10)
    draw_mid(axes[1][0], Z['mid_decay'], n_ins=10)
    axes[1][0].set_title('relaxation: $I(k)$ through execution end (dotted)', fontsize=10)
    axes[1][0].set_ylabel('relaxation shape\n$I$, bps')
    axes[1][0].set_xlabel('children executed $k$ (then cooling blocks)')
    draw_master(axes[1][1], Z['ms_relax'], relax=True)
    axes[1][1].set_title(r'relaxation: master curve, $v>1$ = cooling (dashed: $2/3$ level)', fontsize=10)
    axes[1][1].set_xlabel('metaorder fraction executed $v$')

    present = [m for m in ORDER if any(f'{m}_k_mean' in Z[k].files for k in ('mid_beta', 'mid_decay'))]
    handles = [Line2D([], [], color=COLORS.get(m, '#444444'), lw=2.2, label=m.replace('_', '-'))
               for m in present]
    handles.append(Line2D([], [], color=GREY, lw=1.6, ls='--',
                          label=r'reference ($\sqrt{\cdot}$-law / $2/3$ level)'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 7),
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f'{args.stock}: metaorder impact — build-up and relaxation, '
                 'raw $k$-clock (left) and normalised master curves (right)',
                 fontsize=11.5, y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(res, 'mid_impact', f'fig5_combined_{args.stock}.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    with open(out.replace('.png', '_sources.json'), 'w') as f:
        json.dump(src, f, indent=1)
    print(f'FIG5_DONE -> {out}', flush=True)
    if args.copy_to:
        import shutil
        shutil.copy(out, os.path.join(args.copy_to, os.path.basename(out)))
        print(f'copied -> {args.copy_to}', flush=True)


if __name__ == '__main__':
    main()
