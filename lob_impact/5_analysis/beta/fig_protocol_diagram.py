#!/usr/bin/env python3
"""Figure 1 (v3): the staged evaluation system — observational screening on free
generation gates interventional certification under metaorder injection.
Pure-matplotlib schematic, no data. Output: framework_pipeline_v3.png."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_CAL = ('#ECEFF1', '#607D8B')
C_SCR = ('#E3F2FD', '#2F5DA3')
C_INT = ('#FFF3E0', '#E67E22')
C_OUT = ('#E8F5E9', '#27AE60')
C_STOP = '#C0392B'
C_CTRL = ('#F3E5F5', '#8E24AA')


def box(ax, x, y, w, h, face, edge, title, lines, title_fs=11.5, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.10',
                                facecolor=face, edgecolor=edge, lw=2.0, zorder=2))
    ax.text(x + w / 2, y + h - 0.34, title, ha='center', va='top',
            fontsize=title_fs, fontweight='bold', color=edge, zorder=3)
    ax.text(x + w / 2, y + h - 0.78, '\n'.join(lines), ha='center', va='top',
            fontsize=fs, color='#263238', zorder=3, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, color='#455A64', lw=2.4, style='-|>'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=22, color=color, lw=lw, zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(16.2, 6.0), dpi=200)
    ax.set_xlim(0, 16.2); ax.set_ylim(0, 6.6); ax.axis('off')

    Y0, H = 1.85, 3.3
    # A. calibrate
    box(ax, 0.25, Y0, 2.55, H, *C_CAL, 'A · CALIBRATE',
        ['per stock and day:', r'$m_b$ at 10% participation', 'child = median MO size',
         r'daily $\sigma$, $V$'])
    # 0. screen
    box(ax, 3.30, Y0, 3.75, H, *C_SCR, '0 · SCREEN — free flow',
        ['no injections · per checkpoint', 'distributions: KS / L1 / WS',
         'drift at protocol horizon',
         r'flow balance: $\gamma_{\rm flow}$, $H$',
         r'own-trade $R(\ell)$ vs real'])
    # 1. build-up
    box(ax, 8.30, Y0, 3.35, H, *C_INT, '1 · CERTIFY — build-up',
        ['inject 100 children @ 10% POV', r'$\delta \to 0.5$  ($L_2$, $\leq k$)',
         r'$Y \to 0.5$  (raw-range)', r'$R(m) \leftrightarrow R(\ell)$ validity'])
    # 2. relaxation
    box(ax, 11.95, Y0, 3.10, H, *C_INT, '2 · CERTIFY — relaxation',
        ['10 children + cooling', r'end/peak $\to \frac{2}{3}$,  $\gamma \to 0.5$',
         'power-law kernel (no-arb)', r'$\gamma + \delta \geq 1$'])
    # scorecard
    box(ax, 12.55, 0.12, 3.4, 1.28, *C_OUT, 'SCORECARD', [], title_fs=11)
    ax.text(14.25, 0.62, 'one row per model:\nKS/L1/WS | $\\gamma_{\\rm flow}$,$H$ | $\\delta$,$Y$ | e/p,$\\gamma$',
            ha='center', va='center', fontsize=8.6, color='#1B5E20')

    # controls rail
    box(ax, 3.30, 0.12, 8.35, 1.28, *C_CTRL, 'CONTROLS (attribution & instrument validity)', [],
        title_fs=10.5)
    ax.text(7.47, 0.66, 'visible / invisible / no-insertion triangle   ·   placebo   ·   pre-trend',
            ha='center', va='center', fontsize=9.6, color='#4A148C')

    # main flow arrows
    ym = Y0 + H / 2
    arrow(ax, 2.80, ym, 3.30, ym)
    arrow(ax, 7.05, ym, 8.30, ym)
    arrow(ax, 11.65, ym, 11.95, ym)
    arrow(ax, 15.05, ym, 15.75, ym); ax.text(15.42, ym + 0.28, '', fontsize=9)
    arrow(ax, 14.60, Y0, 14.60, 1.40, color=C_OUT[1])

    # gate between screen and certify
    ax.text(7.67, ym + 0.62, 'gate', ha='center', fontsize=10, color=C_STOP, fontweight='bold')
    ax.text(7.67, Y0 + H + 0.42, r'STOP if $H > 0.5$ or drift $\gg$ signal:'
            '\ndisqualified before any fleet is spent',
            ha='center', va='bottom', fontsize=9.6, color=C_STOP)
    arrow(ax, 7.67, ym + 0.5, 7.67, Y0 + H + 0.25, color=C_STOP, lw=2.0)

    # controls feed both certification stages
    arrow(ax, 9.9, 1.40, 9.9, Y0, color=C_CTRL[1], lw=1.8)
    arrow(ax, 13.4, 1.40, 13.4, Y0, color=C_CTRL[1], lw=1.8)

    # layer labels (top, clear of the controls rail)
    ax.text(4.05, Y0 + H + 0.42, 'OBSERVATIONAL — cheap screening\n(continues the LOB-Bench programme)',
            ha='center', va='bottom', fontsize=9.8, color=C_SCR[1], style='italic')
    ax.text(12.6, Y0 + H + 0.42, 'INTERVENTIONAL — the counterfactual step\n(certification, once per release)',
            ha='center', va='bottom', fontsize=9.8, color=C_INT[1], style='italic')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                       'framework_pipeline_v3.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print('DIAGRAM ->', out)


if __name__ == '__main__':
    main()
