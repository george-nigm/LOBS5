#!/usr/bin/env python3
"""
220 · Figures from Cache (Plotly)

Reads pre-computed pickle cache from run_210_analysis.py and generates
12 interactive Plotly figures.  Much faster than Matplotlib DPI=300.

Usage:
  python lob_impact/220.figures_from_cache.py --stock GOOG           # all 12 figs
  python lob_impact/220.figures_from_cache.py --stock INTC           # all 12 figs
  python lob_impact/220.figures_from_cache.py --stock GOOG --figs 2,3
  python lob_impact/220.figures_from_cache.py --list                 # show catalog
"""
import argparse, pickle, sys, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
TEMPLATE = 'plotly_white'
FONT = dict(family='Times New Roman, serif', size=14)
FULL_W = 1080
SINGLE_W = 520
FIG_H = 400


# ═══════════════════════════════════════════════════════════════════════
# Model palette  (Plotly dash/marker names)
# ═══════════════════════════════════════════════════════════════════════
MODEL_META = OrderedDict([
    ('ZeroInsertions',  dict(color='#7CAE7A', dash='dot',     marker='diamond-open')),
    ('Historic',        dict(color='#90939C', dash='dot',     marker='x')),
    ('Heuristic',       dict(color='#546884', dash='dashdot', marker='diamond')),
    ('CST',             dict(color='#213552', dash='dash',    marker='triangle-up')),
    ('CGAN',            dict(color='#7B4F9E', dash='longdash',marker='square')),
    ('LobS5',           dict(color='#C88A3A', dash='solid',   marker='circle')),
    ('S5-120M',         dict(color='#D95F02', dash='solid',   marker='triangle-down')),
    ('S5-4K',           dict(color='#5B7BBF', dash='solid',   marker='star')),
    ('S5-360M',         dict(color='#B5446E', dash='solid',   marker='hexagon')),
    ('LobS5-v2',        dict(color='#2CA02C', dash='solid',   marker='cross')),
])

def _c(m):  return MODEL_META[m]['color']
def _d(m):  return MODEL_META[m]['dash']
def _m(m):  return MODEL_META[m]['marker']

# ═══════════════════════════════════════════════════════════════════════
# Figure catalog
# ═══════════════════════════════════════════════════════════════════════
FIGURE_CATALOG = OrderedDict([
    (0,  'Null Baseline Drift'),
    (1,  'Master Curves'),
    (2,  'Average Master Curve'),
    (3,  'Beta Regression Lines'),
    (4,  'Bootstrap Beta Distributions'),
    (5,  'Relaxation Ratio'),
    (6,  'Fraction Stable'),
    (7,  'Hurst Exponent'),
    (8,  'Propagator G(l)'),
    (9,  'Spread Dynamics'),
    (10, 'Beta vs mb'),
    (11, 'Beta vs Volume'),
])

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
SAVE_DIR = None  # set by main()

def _plotly_to_mpl_png(fig, path, w, h):
    """Fallback: render Plotly figure to PNG via matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False,
        'axes.spines.right': False,
    })
    # Extract data from Plotly figure
    mpl_fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=200)
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ''
    ax.set_title(title, fontsize=13, fontweight='bold')
    if fig.layout.xaxis and fig.layout.xaxis.title:
        ax.set_xlabel(fig.layout.xaxis.title.text or '')
    if fig.layout.yaxis and fig.layout.yaxis.title:
        ax.set_ylabel(fig.layout.yaxis.title.text or '')

    import re as _re
    import colorsys as _colorsys
    def _parse_color(c):
        """Convert rgba/hsl(...) to matplotlib tuple, pass hex/named through."""
        if not c:
            return None
        s = str(c)
        m = _re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)', s)
        if m:
            r, g, b = int(m.group(1))/255, int(m.group(2))/255, int(m.group(3))/255
            a = float(m.group(4)) if m.group(4) else 1.0
            return (r, g, b, a)
        m = _re.match(r'hsl\((\d+),\s*(\d+)%?,\s*(\d+)%?\)', s)
        if m:
            h, s_val, l = int(m.group(1))/360, int(m.group(2))/100, int(m.group(3))/100
            r, g, b = _colorsys.hls_to_rgb(h, l, s_val)
            return (r, g, b, 1.0)
        return c

    for trace in fig.data:
        x = np.asarray(trace.x) if trace.x is not None else np.array([])
        y = np.asarray(trace.y) if trace.y is not None else np.array([])
        if len(x) == 0 or len(y) == 0:
            continue
        color = None
        if hasattr(trace, 'line') and trace.line and trace.line.color:
            color = _parse_color(trace.line.color)
        elif hasattr(trace, 'marker') and trace.marker and trace.marker.color:
            color = _parse_color(trace.marker.color)
        name = trace.name or ''

        ttype = trace.type
        if ttype in ('scatter', 'scattergl'):
            mode = getattr(trace, 'mode', 'lines') or 'lines'
            if hasattr(trace, 'fill') and trace.fill == 'toself':
                fc = _parse_color(trace.fillcolor) if trace.fillcolor else (color or '#ccc')
                ax.fill(x, y, alpha=0.12, color=fc, label='')
            elif 'markers' in mode and 'lines' in mode:
                ax.plot(x, y, '-o', color=color, ms=3, lw=1.5, label=name)
            elif 'markers' in mode:
                ax.scatter(x, y, s=4, color=color, alpha=0.3, label=name)
            else:
                ax.plot(x, y, color=color, lw=2, label=name)
        elif ttype == 'histogram':
            ax.hist(x, bins=100, color=color, edgecolor='white', linewidth=0.3)
        elif ttype == 'bar':
            mc = _parse_color(trace.marker.color) if trace.marker and trace.marker.color else color
            # x might be strings — use range positions
            x_pos = np.arange(len(y))
            ax.bar(x_pos, y, color=mc if not isinstance(mc, list) else [_parse_color(c) for c in mc],
                   width=0.55, edgecolor='white', tick_label=list(x))
            ax.tick_params(axis='x', rotation=45)
        elif ttype in ('box', 'violin'):
            mc = _parse_color(trace.marker_color if hasattr(trace, 'marker_color') and trace.marker_color
                              else (trace.line_color if hasattr(trace, 'line_color') and trace.line_color
                              else (trace.marker.color if trace.marker and trace.marker.color else None)))
            y_clean = y[~np.isnan(y.astype(float))] if len(y) > 0 else np.array([])
            if len(y_clean) > 0:
                pos = [len([t for t in fig.data[:fig.data.index(trace)] if t.type in ('box','violin')])]
                bp = ax.boxplot([y_clean], positions=pos, patch_artist=True, widths=0.5)
                if bp['boxes']:
                    bp['boxes'][0].set_facecolor(mc or '#888')
                    bp['boxes'][0].set_alpha(0.6)
                ax.set_xticks(pos)
                # Will be overwritten by next trace — accumulate labels
                ticks = ax.get_xticks()
                existing = [t.get_text() for t in ax.get_xticklabels()]
                while len(existing) < len(ticks):
                    existing.append('')
                existing[pos[0]] = name
                ax.set_xticklabels(existing, rotation=45, ha='right', fontsize=9)

    # Legend only if there are labeled traces
    handles, labels = ax.get_legend_handles_labels()
    if any(l for l in labels):
        ax.legend(fontsize=7, ncol=2, loc='best')
    mpl_fig.tight_layout()
    mpl_fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(mpl_fig)


def save_fig(fig, name, w=FULL_W, h=FIG_H):
    """Save figure: HTML always, PNG via kaleido or matplotlib fallback."""
    fig.write_html(SAVE_DIR / f"{name}.html", include_plotlyjs='cdn')
    png_path = SAVE_DIR / f"{name}.png"
    try:
        fig.write_image(png_path, width=w, height=h, scale=3)
        fig.write_image(SAVE_DIR / f"{name}.pdf", width=w, height=h)
        print(f"  Saved {name} (.html + .png + .pdf)")
    except Exception:
        try:
            _plotly_to_mpl_png(fig, png_path, w, h)
            print(f"  Saved {name} (.html + .png via mpl)")
        except Exception as e:
            print(f"  Saved {name} (.html only, mpl err: {e})")


def load_cache(stock):
    cache_file = Path(f'pics_for_210_{stock}') / 'results_cache.pkl'
    if not cache_file.exists():
        print(f'ERROR: {cache_file} not found. Run "run_210_analysis.py compute --stock {stock}" first.')
        sys.exit(1)
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    print(f'Loaded cache: {cache_file} ({cache_file.stat().st_size/1024:.0f} KB)')
    print(f'  Models: {cache["models"]}')
    print(f'  Impact models: {cache["impact_models"]}')
    return cache



def _apply_layout(fig, title, xaxis_title='', yaxis_title='', w=FULL_W, h=FIG_H):
    fig.update_layout(
        title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        template=TEMPLATE, font=FONT, width=w, height=h,
    )
    return fig


def _hex_to_rgba(hex_color, alpha=0.08):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ═══════════════════════════════════════════════════════════════════════
# Figure functions
# ═══════════════════════════════════════════════════════════════════════

def fig_0_null_drift(cache):
    """Null Baseline Drift histogram."""
    if 'null_drifts' not in cache:
        print('  [skip] no null_drifts in cache')
        return None
    drifts = cache['null_drifts']
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=drifts, nbinsx=100,
        marker_color=_c('ZeroInsertions'),
        marker_line_color='white', marker_line_width=0.3,
    ))
    fig.add_vline(x=0, line_dash='dash', line_color='black', line_width=1)
    _apply_layout(fig, 'Null Baseline: Mid-Price Drift (ZeroInsertions)',
                  'Drift (ticks)', 'Count', w=SINGLE_W*1.4, h=FIG_H)
    save_fig(fig, '0. Null Baseline Drift', w=int(SINGLE_W*1.4), h=FIG_H)
    return fig


def fig_1_master_curves(cache):
    """Master Curves — panel grid, one subplot per impact model."""
    imp = cache['impact_models']
    available = [m for m in imp if m in cache['master_curves'] and cache['master_curves'][m]]
    if not available:
        print('  [skip] no master curves')
        return None

    n_m = len(available)
    nc = 2
    nr = math.ceil(n_m / nc)
    import colorsys
    colors = ['#' + ''.join(f'{int(c*255):02x}' for c in colorsys.hls_to_rgb(i/30, 0.5, 0.7))
              for i in range(30)]

    fig = make_subplots(rows=nr, cols=nc, subplot_titles=available,
                        horizontal_spacing=0.06, vertical_spacing=0.08)

    for idx, label in enumerate(available):
        r, c = idx // nc + 1, idx % nc + 1
        fi = 0
        for mc in cache['master_curves'][label].values():
            mask = mc['u'] <= 3.0
            fig.add_trace(go.Scatter(
                x=mc['u'][mask], y=mc['mean'][mask],
                mode='lines', line=dict(color=colors[fi % len(colors)], width=1),
                opacity=0.6, showlegend=False,
            ), row=r, col=c)
            fi += 1
        fig.add_vline(x=1.0, line_dash='dash', line_color='gray',
                      line_width=0.8, row=r, col=c)

    # Hide empty subplots
    for idx in range(n_m, nr * nc):
        r, c = idx // nc + 1, idx % nc + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    h = max(300 * nr, FIG_H)
    fig.update_layout(
        title='Cross-Model: Master Curves',
        template=TEMPLATE, font=FONT,
        width=FULL_W, height=h, showlegend=False,
    )
    fig.update_yaxes(title_text='I_norm(u)', col=1)
    fig.update_xaxes(title_text='u', row=nr)
    save_fig(fig, '1. Master Curves', w=FULL_W, h=h)
    return fig


def fig_2_avg_master_curve(cache):
    """Average Master Curve — one line per model with fill band."""
    imp = cache['impact_models']
    fig = go.Figure()
    for label in imp:
        mcs = cache['master_curves'].get(label, {})
        if not mcs:
            continue
        all_m = [mc['mean'] for mc in mcs.values()]
        u = list(mcs.values())[0]['u']
        avg = np.nanmean(all_m, axis=0)
        std = np.nanstd(all_m, axis=0)

        fig.add_trace(go.Scatter(
            x=u, y=avg, mode='lines',
            line=dict(color=_c(label), dash=_d(label), width=3),
            name=label,
        ))
        # Fill band
        fig.add_trace(go.Scatter(
            x=np.concatenate([u, u[::-1]]),
            y=np.concatenate([avg + std, (avg - std)[::-1]]),
            fill='toself', fillcolor=_hex_to_rgba(_c(label), 0.08),
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))

    fig.add_vline(x=1.0, line_dash='dash', line_color='gray', line_width=1.5)
    _apply_layout(fig, 'Cross-Model: Average Master Curve', 'u', 'I_norm(u)',
                  w=FULL_W, h=FIG_H+50)
    fig.update_layout(legend=dict(font_size=10))
    save_fig(fig, '2. Average Master Curve', w=FULL_W, h=FIG_H+50)
    return fig


def fig_3_beta_regression(cache):
    """Beta Regression Lines — scatter + fitted lines."""
    imp = cache['impact_models']
    beta = cache['beta']
    fig = go.Figure()

    for label in imp:
        if label not in beta:
            continue
        br = beta[label]
        x, y = br['pc_x'], br['pc_y']
        if len(x) == 0:
            continue
        # Point cloud (subsample for speed)
        n_pts = len(x)
        step = max(1, n_pts // 2000)
        fig.add_trace(go.Scattergl(
            x=x[::step], y=y[::step], mode='markers',
            marker=dict(size=2, color=_c(label), opacity=0.12),
            showlegend=False, hoverinfo='skip',
        ))
        # Regression line
        xl = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=xl, y=br['beta'] * xl, mode='lines',
            line=dict(color=_c(label), dash=_d(label), width=3),
            name=f"{label}: {br['beta']:.3f}",
        ))

    # Theory line
    fig.add_trace(go.Scatter(
        x=[-15, -5], y=[0.5*-15, 0.5*-5], mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='Theory: 0.5',
    ))
    _apply_layout(fig, 'Cross-Model: Beta Regression Lines',
                  'log(Q/V)', 'log(I/sigma)', w=FULL_W, h=FIG_H+150)
    fig.update_layout(legend=dict(font_size=9, x=0.01, y=0.01, yanchor='bottom'))
    save_fig(fig, '3. Beta Regression Lines', w=FULL_W, h=FIG_H+150)

    # LaTeX table
    t1 = []
    for label in imp:
        if label not in beta:
            continue
        br = beta[label]
        ci = f"[{br['ci_lo']:.3f}, {br['ci_hi']:.3f}]" if not np.isnan(br.get('ci_lo', np.nan)) else '---'
        t1.append([label, f"{br['beta']:.3f}", f"{br['r2']:.3f}", f"{br['n']:,}", ci])
    latex_table(['Model', r'$\beta$', r'$R^2$', '$N$', '95\\% CI'], t1, 'Table 1: Global Beta')

    return fig


def fig_4_bootstrap_beta(cache):
    """Bootstrap Beta — violin plots (one per model)."""
    imp = cache['impact_models']
    beta = cache['beta']
    fig = go.Figure()

    for label in imp:
        if label not in beta:
            continue
        boots = beta[label].get('boots', np.array([]))
        if len(boots) == 0:
            continue
        fig.add_trace(go.Violin(
            y=boots, name=label,
            line_color=_c(label), fillcolor=_hex_to_rgba(_c(label), 0.45),
            meanline_visible=True, box_visible=True,
        ))

    fig.add_hline(y=0.5, line_dash='dash', line_color='red', line_width=2.5)
    _apply_layout(fig, 'Cross-Model: Bootstrap Beta Distributions',
                  '', 'beta', w=FULL_W, h=FIG_H+50)
    save_fig(fig, '4. Bootstrap Beta Distributions', w=FULL_W, h=FIG_H+50)
    return fig


def fig_5_relaxation(cache):
    """Relaxation Ratio — box plots."""
    imp = cache['impact_models']
    fig = go.Figure()
    has_data = False

    for label in imp:
        ratios = cache['relax'].get(label, [])
        if not ratios:
            continue
        has_data = True
        fig.add_trace(go.Box(
            y=ratios, name=label,
            marker_color=_c(label), fillcolor=_hex_to_rgba(_c(label), 0.65),
            line_color=_c(label),
        ))

    if not has_data:
        print('  [skip] no relaxation data')
        return None

    fig.add_hline(y=2/3, line_dash='dash', line_color='red', line_width=2,
                  annotation_text='Bouchaud 2/3', annotation_position='top right')
    _apply_layout(fig, 'Cross-Model: Relaxation Ratio',
                  '', 'I_final / I_peak', w=FULL_W, h=FIG_H+50)
    save_fig(fig, '5. Relaxation Ratio', w=FULL_W, h=FIG_H+50)

    # LaTeX table
    t2 = []
    for label in imp:
        arr = cache['relax'].get(label, [])
        if not arr:
            continue
        arr = np.array(arr)
        med = np.median(arr)
        t2.append([label, f"{med:.3f}",
                   f"{np.mean(arr):.3f} $\\pm$ {np.std(arr):.3f}",
                   f"{np.std(arr)/(abs(np.mean(arr))+1e-10):.3f}",
                   f"{abs(med-2/3):.3f}"])
    latex_table(['Model', 'Median', r'Mean $\pm$ std', 'CV', r'$|\Delta|$ from 2/3'],
                t2, 'Table 2: Relaxation')

    return fig


def fig_6_fraction_stable(cache):
    """Fraction Stable — bar chart."""
    stab = cache['stability']
    labels_s = [m for m in stab if m in MODEL_META]
    if not labels_s:
        print('  [skip] no stability data')
        return None

    vals = [stab[l]['frac'] for l in labels_s]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels_s, y=vals,
        marker_color=[_c(l) for l in labels_s],
        width=0.55,
    ))
    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', line_width=1.5)
    _apply_layout(fig, 'Cross-Model: Fraction Stable',
                  '', 'Fraction (2+/3 votes)', w=SINGLE_W*1.4, h=FIG_H)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(tickangle=45)
    save_fig(fig, '6. Fraction Stable', w=int(SINGLE_W*1.4), h=FIG_H)

    # LaTeX table
    t3 = [[l, f"{stab[l]['stable']}/{stab[l]['total']}",
           f"{stab[l]['frac']:.0%}"] for l in labels_s]
    latex_table(['Model', 'Stable/Total', 'Fraction'], t3, 'Table 3: Stability')

    return fig


def fig_7_hurst(cache):
    """Hurst Exponent — bar chart."""
    hurst = cache.get('hurst', {})
    models_h = [m for m in hurst if m in MODEL_META and not np.isnan(hurst[m])]
    if not models_h:
        print('  [skip] Hurst all NaN or empty')
        return None

    vals = [hurst[m] for m in models_h]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models_h, y=vals,
        marker_color=[_c(m) for m in models_h],
        width=0.55,
    ))
    fig.add_hline(y=0.7, line_dash='dash', line_color='black', line_width=1.5)
    fig.add_hline(y=0.5, line_dash='dot', line_color='gray', line_width=1)
    _apply_layout(fig, 'Cross-Model: Hurst Exponent',
                  '', 'H (DFA)', w=SINGLE_W*1.4, h=FIG_H)
    fig.update_xaxes(tickangle=45)
    save_fig(fig, '7. Hurst Exponent', w=int(SINGLE_W*1.4), h=FIG_H)
    return fig


def fig_8_propagator(cache):
    """Propagator G(l) — log-log lines."""
    prop = cache.get('propagator', {})
    if not prop:
        print('  [skip] no propagator data')
        return None

    fig = go.Figure()
    for label, pr in prop.items():
        if label not in MODEL_META:
            continue
        G, lags = pr['G'], pr['lags']
        mask = (lags > 0) & (G > 0)
        if mask.sum() < 3:
            continue
        fig.add_trace(go.Scatter(
            x=np.log10(lags[mask]), y=np.log10(G[mask]),
            mode='lines',
            line=dict(color=_c(label), dash=_d(label), width=2.5),
            name=label,
        ))

    # Theory line l^{-0.5}
    l_th = np.logspace(0, 2.3, 50)
    fig.add_trace(go.Scatter(
        x=np.log10(l_th), y=np.log10(l_th**(-0.5) * 0.1),
        mode='lines', line=dict(color='black', dash='dash', width=1.5),
        name='l^{-0.5}',
    ))
    _apply_layout(fig, 'Cross-Model: Propagator G(l)',
                  'log10(l)', 'log10(G)', w=FULL_W, h=FIG_H+50)
    fig.update_layout(legend=dict(font_size=9))
    save_fig(fig, '8. Propagator G(l)', w=FULL_W, h=FIG_H+50)
    return fig


def fig_9_spread(cache):
    """Spread Dynamics — lines per model."""
    spread = cache.get('spread', {})
    if not spread:
        print('  [skip] no spread data')
        return None

    fig = go.Figure()
    for label, sr in spread.items():
        if label not in MODEL_META:
            continue
        fig.add_trace(go.Scatter(
            x=sr['u'], y=sr['mean'], mode='lines',
            line=dict(color=_c(label), dash=_d(label), width=2.5),
            name=label,
        ))

    fig.add_vline(x=1.0, line_dash='dash', line_color='gray', line_width=1)
    _apply_layout(fig, 'Cross-Model: Spread Dynamics',
                  'Volume time u', 'Spread (ticks)', w=FULL_W, h=FIG_H+50)
    fig.update_layout(legend=dict(font_size=9))
    save_fig(fig, '9. Spread Dynamics', w=FULL_W, h=FIG_H+50)
    return fig


def fig_10_beta_vs_mb(cache):
    """Beta vs messages-between — lines + markers."""
    imp = cache['impact_models']
    mb_betas = cache.get('mb_betas', {})
    fig = go.Figure()
    has_data = False

    for label in imp:
        d = mb_betas.get(label, {})
        if not d:
            continue
        has_data = True
        mbs = sorted(d.keys())
        fig.add_trace(go.Scatter(
            x=mbs, y=[d[m] for m in mbs], mode='lines+markers',
            line=dict(color=_c(label), dash=_d(label), width=2),
            marker=dict(symbol=_m(label), size=8),
            name=label,
        ))

    if not has_data:
        print('  [skip] no mb_betas data')
        return None

    fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1)
    _apply_layout(fig, 'beta vs messages-between',
                  'mb', 'beta', w=SINGLE_W*1.4, h=FIG_H)
    fig.update_layout(legend=dict(font_size=9))
    save_fig(fig, '10. Beta vs mb', w=int(SINGLE_W*1.4), h=FIG_H)
    return fig


def fig_11_beta_vs_volume(cache):
    """Beta vs order volume — lines + markers."""
    imp = cache['impact_models']
    vol_betas = cache.get('vol_betas', {})
    fig = go.Figure()
    has_data = False

    for label in imp:
        d = vol_betas.get(label, {})
        if not d:
            continue
        has_data = True
        vols = sorted(d.keys())
        fig.add_trace(go.Scatter(
            x=vols, y=[d[v] for v in vols], mode='lines+markers',
            line=dict(color=_c(label), dash=_d(label), width=2),
            marker=dict(symbol=_m(label), size=8),
            name=label,
        ))

    if not has_data:
        print('  [skip] no vol_betas data')
        return None

    fig.add_hline(y=0.5, line_dash='dash', line_color='black', line_width=1)
    _apply_layout(fig, 'beta vs order volume',
                  'Order volume', 'beta', w=SINGLE_W*1.4, h=FIG_H)
    fig.update_layout(legend=dict(font_size=9))
    save_fig(fig, '11. Beta vs Volume', w=int(SINGLE_W*1.4), h=FIG_H)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# LaTeX table + summary CSV
# ═══════════════════════════════════════════════════════════════════════

def latex_table(headers, rows, caption=""):
    fmt = 'l' + 'c' * (len(headers) - 1)
    lines = [f"\n-- LaTeX: {caption} --",
             f"\\begin{{tabular}}{{{fmt}}}", "\\toprule",
             " & ".join(headers) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(str(v) for v in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    print('\n'.join(lines))


def export_summary_csv(cache):
    imp = cache['impact_models']
    summary = []
    for label in imp:
        br = cache['beta'].get(label, {})
        summary.append(dict(
            Model=label,
            beta=br.get('beta', np.nan),
            CI_lo=br.get('ci_lo', np.nan),
            CI_hi=br.get('ci_hi', np.nan),
            R2=br.get('r2', np.nan),
            N=br.get('n', 0),
            relaxation=np.median(cache.get('relax', {}).get(label, [np.nan])),
            stable_frac=cache.get('stability', {}).get(label, {}).get('frac', np.nan),
            Hurst=cache.get('hurst', {}).get(label, np.nan),
            no_arb_score=cache.get('scorecard', {}).get(label, {}).get('score', np.nan),
        ))
    df = pd.DataFrame(summary)
    out = SAVE_DIR / 'summary_statistics.csv'
    df.to_csv(out, index=False)
    print(f'\n  Summary CSV -> {out}')
    return df


# ═══════════════════════════════════════════════════════════════════════
# Figure dispatch
# ═══════════════════════════════════════════════════════════════════════
FIGURE_FUNCS = {
    0:  fig_0_null_drift,
    1:  fig_1_master_curves,
    2:  fig_2_avg_master_curve,
    3:  fig_3_beta_regression,
    4:  fig_4_bootstrap_beta,
    5:  fig_5_relaxation,
    6:  fig_6_fraction_stable,
    7:  fig_7_hurst,
    8:  fig_8_propagator,
    9:  fig_9_spread,
    10: fig_10_beta_vs_mb,
    11: fig_11_beta_vs_volume,
}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    global SAVE_DIR

    parser = argparse.ArgumentParser(
        description='220 · Generate Plotly figures from 210 cache')
    parser.add_argument('--stock', type=str, default='GOOG',
                        help='Stock ticker (GOOG, INTC)')
    parser.add_argument('--figs', type=str, default=None,
                        help='Comma-separated figure numbers, e.g. 1,3,5')
    parser.add_argument('--list', action='store_true',
                        help='List available figures and exit')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory override')
    args = parser.parse_args()

    if args.list:
        print('\nAvailable figures:')
        for num, name in FIGURE_CATALOG.items():
            print(f'  [{num:>2}] {name}')
        sys.exit(0)

    stock = args.stock.upper()
    SAVE_DIR = Path(args.out) if args.out else Path(f'pics_for_220_{stock}')
    SAVE_DIR.mkdir(exist_ok=True)

    cache = load_cache(stock)

    want = set(int(x) for x in args.figs.split(',')) if args.figs else set(FIGURE_CATALOG.keys())
    n_total = len(want)
    print('=' * 60)
    print(f'  220 · Plotly Figures — {stock}, generating {n_total} figures')
    print('=' * 60)

    n_done = 0
    for num in sorted(want):
        if num not in FIGURE_FUNCS:
            print(f'  [warn] Unknown figure #{num}')
            continue
        name = FIGURE_CATALOG[num]
        print(f'\n  [{num:>2}] {name}')
        fig = FIGURE_FUNCS[num](cache)
        if fig is not None:
            n_done += 1

    # Summary CSV (always)
    export_summary_csv(cache)

    print(f'\n  Done: {n_done}/{n_total} figures in {SAVE_DIR}/')


if __name__ == '__main__':
    main()
