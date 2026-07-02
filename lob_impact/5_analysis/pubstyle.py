#!/usr/bin/env python3
"""
Shared publication figure style for the market-impact paper (Action 5 analysis figures).

Goal: every paper figure reads as ONE system, matching the clean look of the older paper:
serif (Times-like via STIX, consistent with the .tex mathptmx body), a full box frame
(all four spines), a subtle grey grid, a bordered legend, muted per-model colours, and a
high-DPI export with NO debug titles.

Usage (scripts run as files from 5_analysis/<sub>/):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from pubstyle import apply, MODEL_STYLE, ref_style, savefig_pub
    apply()
    ...
    st = MODEL_STYLE['Mamba3']; ax.plot(x, y, color=st['color'], ls=st['ls'], label=st['label'])
"""
import matplotlib


# ---- per-model visual identity (colour kept consistent with the interactive explorers) ----
# label = the display name used in the paper (Mamba-3, not Mamba3).
MODEL_STYLE = {
    'Historic':  dict(color='#C0392B', ls='-',  label='Historic'),
    'Heuristic': dict(color='#7F8C8D', ls='-',  label='Heuristic'),
    'Hawkes':    dict(color='#D4A017', ls='-',  label='Hawkes'),
    'CST':       dict(color='#27AE60', ls='-',  label='CST'),
    'Mamba3':    dict(color='#2F5DA3', ls='-',  label='Mamba-3'),
    'Mamba3_4k': dict(color='#16A085', ls='-',  label='Mamba-3-4k'),
    'S5':        dict(color='#E67E22', ls='-',  label='LobS5'),
    'S5_4k':     dict(color='#E67E22', ls='-',  label='S5-4k'),
}

# reference-curve styles (theory overlays)
REF = {
    'sqrt':       dict(color='k',       ls='--', lw=2.0, label=r'$\sqrt{\cdot}$-law  ($\beta=0.5$)'),
    'propagator': dict(color='#8E44AD', ls=':',  lw=2.2, label='propagator'),
    'twothirds':  dict(color='#8E44AD', ls='--', lw=1.6, label=r'permanent $\approx \frac{2}{3}$'),
    'theory':     dict(color='k',       ls='--', lw=2.0, label=r'theory ($\beta=0.5$)'),
}


def style(name):
    """MODEL_STYLE lookup with a sane fallback for unknown model keys."""
    return MODEL_STYLE.get(name, dict(color='#444444', ls='-', label=name))


def ref_style(name):
    return REF[name]


def apply():
    """Install the publication rcParams globally (call once, before creating figures)."""
    matplotlib.use('Agg')
    rc = matplotlib.rcParams
    # --- fonts: Times-like serif to match the .tex body (mathptmx) ---
    rc['font.family'] = 'serif'
    rc['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    rc['mathtext.fontset'] = 'stix'
    rc['axes.unicode_minus'] = True
    # --- box frame: all four spines, thin black ---
    for s in ('top', 'right', 'bottom', 'left'):
        rc[f'axes.spines.{s}'] = True
    rc['axes.edgecolor'] = '#1a1a1a'
    rc['axes.linewidth'] = 1.0
    # --- subtle grid ---
    rc['axes.grid'] = True
    rc['grid.color'] = '#d9d9d9'
    rc['grid.linewidth'] = 0.6
    rc['grid.alpha'] = 1.0
    rc['axes.axisbelow'] = True
    # --- ticks ---
    rc['xtick.direction'] = 'out'
    rc['ytick.direction'] = 'out'
    rc['xtick.major.size'] = 4.0
    rc['ytick.major.size'] = 4.0
    rc['xtick.labelsize'] = 11
    rc['ytick.labelsize'] = 11
    # --- text sizes ---
    rc['axes.labelsize'] = 13
    rc['axes.titlesize'] = 13
    rc['axes.titleweight'] = 'bold'
    rc['legend.fontsize'] = 9.5
    # --- bordered legend (square corners, opaque, like the old paper) ---
    rc['legend.frameon'] = True
    rc['legend.framealpha'] = 1.0
    rc['legend.edgecolor'] = '#1a1a1a'
    rc['legend.fancybox'] = False
    rc['legend.borderpad'] = 0.5
    # --- lines / export ---
    rc['lines.linewidth'] = 1.7
    rc['lines.solid_capstyle'] = 'round'
    rc['figure.dpi'] = 150
    rc['savefig.dpi'] = 220
    rc['savefig.bbox'] = 'tight'
    rc['savefig.pad_inches'] = 0.03


def legend_box(ax, **kw):
    """A bordered legend consistent across figures."""
    lg = ax.legend(**kw)
    fr = lg.get_frame()
    fr.set_edgecolor('#1a1a1a')
    fr.set_linewidth(0.9)
    return lg


def savefig_pub(fig, out):
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    return out
