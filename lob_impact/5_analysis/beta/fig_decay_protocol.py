#!/usr/bin/env python3
"""
Stage-2 protocol figure: WHAT PASSING LOOKS LIKE on the relaxation shape.
Theory curves (power-law decay to the 2/3 permanent level; the forbidden
exponential kernel) with the four numbered tests annotated, drawn OVER the
actual gated model master curves from the cache. Pure cache re-render.

  python fig_decay_protocol.py --stock NVDA [--copy_to <dir>]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['Historic', 'Heuristic', 'Propagator', 'Hawkes', 'CST', 'NMZI',
         'Mamba3', 'GDN', 'S5_120M', 'S5', 'Mamba3_4k', 'S5_4k']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--vmax', type=float, default=4.0)
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    z = np.load(os.path.join(here, 'results', 'master_curve',
                             f'master_curve_{args.stock}_relaxation_v2gated.npz'),
                allow_pickle=True)
    v = z['vgrid']
    sel = v <= args.vmax

    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13})
    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=170)

    # measured model curves (gated), thin
    for m in ORDER:
        if f'{m}_master' not in z.files or not bool(z[f'{m}_sig']):
            continue
        # n in the legend: rollouts behind the master curve, so a jagged line
        # (Hawkes here) is attributable to the model, not to a thin sample.
        nm = int(np.atleast_1d(z[f'{m}_n']).ravel()[0]) if f'{m}_n' in z.files else 0
        lab = f"{m.replace('_', '-')} ($n{{=}}{nm}$)" if nm else m.replace('_', '-')
        ax.plot(v[sel], z[f'{m}_master'][sel], color=COLORS.get(m, '#444444'),
                lw=1.6, alpha=0.85, label=lab)

    # THEORY: sqrt build-up, power-law decay to the 2/3 permanent level
    vv = np.linspace(0.01, args.vmax, 400)
    up = vv <= 1
    theory = np.where(up, np.sqrt(vv), 2/3 + (1/3) * (np.sqrt(vv) - np.sqrt(np.clip(vv - 1, 0, None))))
    ax.plot(vv, theory, color='#1B5E20', lw=3.4,
            label=r'THEORY: power-law decay to $\frac{2}{3}$ (passing)')
    # forbidden exponential kernel
    expo = np.where(up, np.sqrt(vv), 2/3 + (1/3) * np.exp(-(vv - 1) / 0.35))
    ax.plot(vv, expo, color='#B71C1C', lw=2.2, ls='--',
            label=r'exponential kernel --- admits arbitrage (failing form)')

    ax.axhline(2/3, color='#1B5E20', lw=1.2, ls=':')
    ax.axvline(1.0, color='#777777', lw=1.2, ls=':')
    ax.text(1.02, 0.06, 'execution ends', rotation=90, fontsize=10, color='#555555', va='bottom')

    # the four numbered tests
    ax.annotate('TEST 1 — level: fall to $\\approx\\frac{2}{3}$ of peak\nwithin 1–2 execution durations\n(uninformed metaorder: band $[0,\\frac{2}{3}]$)',
                xy=(2.15, 2/3), xytext=(1.30, 0.16), fontsize=10.5, color='#1B5E20',
                arrowprops=dict(arrowstyle='->', color='#1B5E20'))
    ax.annotate('TEST 2 — exponent: $\\gamma \\approx 0.5$\n(slope of the decay segment)',
                xy=(1.45, float(2/3 + (1/3)*(np.sqrt(1.45)-np.sqrt(0.45)))), xytext=(0.014, 0.44),
                fontsize=10.5, color='#1B5E20', arrowprops=dict(arrowstyle='->', color='#1B5E20'))
    ax.annotate('TEST 3 — kernel form: power law, NOT exponential\n(exponential + nonlinear impact = dynamic arbitrage)',
                xy=(1.35, float(2/3 + (1/3)*np.exp(-0.35/0.35))), xytext=(1.55, 1.32),
                fontsize=10.5, color='#B71C1C', arrowprops=dict(arrowstyle='->', color='#B71C1C'))
    ax.text(2.05, 1.52, 'TEST 4 — consistency: $\\gamma + \\delta \\geq 1$\n(couples Stage 2 to the build-up exponent)',
            fontsize=10.5, color='#333333',
            bbox=dict(boxstyle='round,pad=0.35', fc='#F5F5F5', ec='#999999'))
    ax.text(1.25, 1.16, 'current generators:\nflat or rising — all four tests fail',
            fontsize=10.5, color='#37474F', style='italic')

    # inset: HOW gamma is fitted — log-log of the excess above the permanent
    # level vs time since execution end; power law = straight line, slope -gamma
    ins = ax.inset_axes([0.66, 0.06, 0.32, 0.30])
    zz = vv[vv > 1.02] - 1.0
    exc_t = (theory[vv > 1.02] - 2/3) / (1/3)
    exc_e = (expo[vv > 1.02] - 2/3) / (1/3)
    ins.loglog(zz, exc_t, color='#1B5E20', lw=2.2)
    ins.loglog(zz, exc_e, color='#B71C1C', lw=1.6, ls='--')
    ins.set_title('the fit: $\\log$ excess vs $\\log$ time\nslope $= -\\gamma$ (straight = power law)', fontsize=8.2)
    ins.set_xlabel('$v-1$', fontsize=8); ins.tick_params(labelsize=7)
    ax.set_xlim(0, args.vmax); ax.set_ylim(0, 1.75)
    ax.set_xlabel('metaorder fraction executed  $v$   ($v>1$ = cooling, in execution durations)')
    ax.set_ylabel(r'master curve  $\langle I(v)\rangle / \langle I(1)\rangle$')
    ax.set_title(f'{args.stock}: the Stage-2 scoring protocol — what passing looks like',
                 fontsize=13.5, loc='left')
    ax.legend(fontsize=9, ncol=2, loc='upper left', framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(here, 'results', 'master_curve', f'decay_protocol_{args.stock}.png')
    fig.savefig(out, bbox_inches='tight')
    np.savez_compressed(out.replace('.png', '.npz'), vgrid=vv, theory=theory, expo=expo)
    print('DECAY_PROTOCOL ->', out)
    if args.copy_to:
        import shutil; shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)


if __name__ == '__main__':
    main()
