#!/usr/bin/env python3
"""
Body figure: THREE estimators side by side on the canonical cumulative pool,
plus the evidence that Hawkes' noise is signal-weakness, not sample-shortage.
Pure cache re-render (beta_3x3 npz + b3l2 npz + mid_trajectory K matrices).

Top row     delta(k) on <=k under signed binned | direct L2 | direct L1.
Bottom row  (1) L2 misfit profiles S(delta)/S_min at the cursor k;
            (2) pooled signed points per model (identical bars => Hawkes'
                wiggle is not a sample-count artefact);
            (3) estimator self-consistency: the three <=100 endpoints per
                model — a tight cluster = a law, a wide spread = grey zone.
One shared legend under all panels.

  python beta_3est_body.py --stock NVDA [--copy_to <dir>]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# OW (Obizhaeva-Wang exponential-resilience kernel) and QR (queue-reactive) sit with the
# other mechanical baselines, between Propagator and the point-process models.
ORDER = ['Historic', 'Heuristic', 'Propagator', 'OW', 'QR', 'Hawkes', 'CST', 'NMZI',
         'Mamba3', 'GDN', 'S5_120M', 'S5', 'Mamba3_4k', 'S5_4k']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'OW': '#6A1B9A', 'QR': '#455A64',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
ESTS = [('binned', 'signed binned (headline convention)'),
        ('l2', r'direct $L_2$ on signed points (canon)'),
        ('l1', r'direct $L_1$ on signed points')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    z3 = np.load(os.path.join(here, 'results', 'beta_3x3', f'beta_3x3_{args.stock}.npz'))
    zl = np.load(os.path.join(here, 'results', 'beta_3views_l2', f'beta_3views_l2_{args.stock}.npz'))
    zk = np.load(os.path.join(here, 'results', 'mid_impact', f'mid_trajectory_{args.stock}_beta.npz'),
                 allow_pickle=True)
    ks = z3['ks']
    models = [m for m in ORDER if f'{m}_l2_le' in z3.files]

    plt.rcParams.update({'font.size': 11.5, 'axes.labelsize': 12})
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))

    for ci, (est, title) in enumerate(ESTS):
        ax = axes[0][ci]
        for m in models:
            ax.plot(ks, z3[f'{m}_{est}_le'], color=COLORS.get(m, '#444444'),
                    lw=2.4 if m == 'Hawkes' else 1.7)
        ax.axhline(0.5, color='#C0392B', ls='--', lw=1.1)
        ax.axhline(0.0, color='#cccccc', lw=0.8)
        ax.set_ylim(-0.6, 1.6); ax.set_xlim(ks[0], ks[-1])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('insertion index $k$')
        if ci == 0:
            ax.set_ylabel(r'$\delta(k)$ on the cumulative pool $\leq k$')

    # (1) L2 misfit profiles
    ax = axes[1][0]
    dg = zl['dgrid']; kcur = int(zl['kcursor'])
    for m in models:
        key = f'{m}_le_S'
        if key not in zl.files:
            continue
        S = zl[key]
        j = int(np.argmin(S))
        ax.plot(dg, S / S.min(), color=COLORS.get(m, '#444444'),
                lw=2.4 if m == 'Hawkes' else 1.6)
        ax.axvline(dg[j], color=COLORS.get(m, '#444444'), lw=0.8, ls=':', alpha=0.5)
    ax.set_yscale('log')
    ax.set_title(rf'$L_2$ misfit $S(\delta)/S_{{\min}}$ at $k={kcur}$' + '\n(flat = no law)', fontsize=11)
    ax.set_xlabel(r'$\delta$ candidate')

    # (2) pooled point counts
    ax = axes[1][1]
    names, counts = [], []
    for m in models:
        if f'{m}_K' in zk.files:
            names.append(m); counts.append(int(np.isfinite(zk[f'{m}_K'][:, 1:]).sum()))
    ypos = np.arange(len(names))
    ax.barh(ypos, counts, color=[COLORS.get(m, '#444444') for m in names], alpha=0.85)
    ax.set_yticks(ypos); ax.set_yticklabels([m.replace('_', '-') for m in names], fontsize=9)
    ax.invert_yaxis()
    for y, c in zip(ypos, counts):
        ax.text(c, y, f' {c/1000:.0f}k', va='center', fontsize=8.5)
    ax.set_title('pooled signed points per model\n(Hawkes has as many as everyone)', fontsize=11)
    ax.set_xlabel('points entering the fits (buy+sell)')

    # (3) estimator self-consistency
    ax = axes[1][2]
    for i, m in enumerate(models):
        vals = [z3[f'{m}_{est}_le'][-1] for est, _ in ESTS]
        c = COLORS.get(m, '#444444')
        ax.plot([i] * 3, vals, 'o', color=c, ms=6, alpha=0.9)
        ax.plot([i, i], [min(vals), max(vals)], color=c, lw=2.2, alpha=0.7)
        ax.annotate(f'{max(vals)-min(vals):.2f}', (i, max(vals)), fontsize=7.5,
                    ha='center', va='bottom', color=c)
    ax.axhline(0.5, color='#C0392B', ls='--', lw=1.1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('_', '-') for m in models], rotation=45, ha='right', fontsize=8.5)
    ax.set_title('self-consistency: the three $\\delta(\\leq 100)$ per model\n(spread number = grey-zone signature)',
                 fontsize=11)
    ax.set_ylabel(r'$\delta$')

    # n in the legend: rollouts behind each model (mid-impact cache `_cnt`), so a wide
    # estimator spread can be read as a real grey zone rather than a thin sample.
    def _n(m):
        c = zk[f'{m}_cnt'] if f'{m}_cnt' in zk.files else None
        return int(np.atleast_1d(c).ravel()[0]) if c is not None and np.size(c) else 0
    handles = [Line2D([], [], color=COLORS.get(m, '#444444'), lw=2.6,
                      label=f"{m.replace('_', '-')} ($n{{=}}{_n(m)}$)" if _n(m) else m.replace('_', '-'))
               for m in models]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f'{args.stock}: the exponent under three estimators on the canonical pool '
                 f'--- agreement is the verdict', fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(here, 'results', 'beta_3views_l2', f'beta_3est_{args.stock}.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print('B3EST ->', out)
    if args.copy_to:
        import shutil; shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)


if __name__ == '__main__':
    main()
