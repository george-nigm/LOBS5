#!/usr/bin/env python3
"""
Interactive (plotly) versions of the paper's model-comparison figures, built from the SAME cached
data as the static PNGs (npz/json caches — no grid access). One self-contained HTML per figure in
a STABLE folder, so the user can toggle models via the legend (click = hide, double-click =
isolate) and zoom/pan freely.

  python 5_analysis/beta/paper_html_explorers.py
  -> results/paper_html/fig5_event_response.html
     results/paper_html/fig6_midtraj_beta.html          (k-clock; event-time variant included)
     results/paper_html/fig6_midtraj_decay.html
     results/paper_html/fig7_master_beta.html           (incl. non-normalisable models, hidden by default)
     results/paper_html/fig7_master_relaxation.html
     results/paper_html/fig8_beta_3views.html
"""
import os, sys, re, json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(B, '..'))
sys.path.insert(0, os.path.join(B, '..', '..', '4_diagnostics'))
from pubstyle import MODEL_STYLE                                   # noqa: E402
from triangle_compare_report import latest_numbers                 # noqa: E402
from mid_trajectory import propagator_curve                        # noqa: E402

OUT = os.path.join(B, 'results', 'paper_html')
RESULTS_4D = os.path.abspath(os.path.join(B, '..', '..', '4_diagnostics', 'results'))
MID = os.path.join(B, 'results', 'mid_impact')
MASTER = os.path.join(B, 'results', 'master_curve')
B3V = os.path.join(B, 'results', 'beta_vs_k_3views')

NEURAL = ['Mamba3', 'Mamba3_4k', 'S5_4k', 'GDN']
BASELINES = ['Historic', 'Heuristic', 'Hawkes', 'CST', 'Propagator']
ALL = BASELINES + NEURAL
C_REAL = '#008300'


def st(m):
    return MODEL_STYLE.get(m, dict(color='#444444', label=m))


def rgba(hexc, a):
    h = hexc.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


def band(fig, x, lo, hi, color, group, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], fill='toself',
                             fillcolor=rgba(color, 0.12), line=dict(width=0),
                             legendgroup=group, showlegend=False, hoverinfo='skip'), **kw)


def layout(fig, title, xt, yt):
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt,
                      template='plotly_white', hovermode='x unified',
                      legend=dict(groupclick='togglegroup'),
                      margin=dict(l=60, r=20, t=60, b=50))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.write_html(p, include_plotlyjs='embed')
    print('saved ->', p)


# ---------------- Fig 5: single-order response ----------------
def fig5():
    fig = go.Figure()
    emp = None
    for m in ['Mamba3', 'Mamba3_4k', 'S5_4k']:
        try:
            N = json.load(open(latest_numbers(RESULTS_4D, m)))
        except Exception as e:
            print(f'fig5: {m} skipped ({e})'); continue
        emp = emp or N['empirical']
        r = N['visible-buy']
        rm = np.array(r['resp_mean'], float); rs = np.array(r['resp_se'], float)
        x = np.arange(len(rm)); ok = np.isfinite(rm)
        s = st(m)
        band(fig, x[ok], (rm - 2 * rs)[ok], (rm + 2 * rs)[ok], s['color'], m)
        fig.add_trace(go.Scatter(x=x[ok], y=rm[ok], name=f"{s['label']} ({r['resp_n_events']//1000}k events)",
                                 line=dict(color=s['color'], width=2.2), legendgroup=m))
    bl = os.path.join(RESULTS_4D, 'baseline_resp.json')
    if os.path.isfile(bl):
        for m, r in json.load(open(bl)).items():
            rm = np.array(r['resp_mean'], float); x = np.arange(len(rm)); ok = np.isfinite(rm)
            s = st(m)
            fig.add_trace(go.Scatter(x=x[ok], y=rm[ok], name=f"{s['label']} ({r['resp_n_events']//1000}k events)",
                                     line=dict(color=s['color'], width=1.2), legendgroup=m))
    if emp:
        hs = sorted(int(k.split('_')[1]) for k in emp if re.fullmatch(r'R_\d+', k))
        rv = [emp[f'R_{h}'] for h in hs]
        se = [2 * emp.get(f'R_{h}_se', 0) for h in hs]
        fig.add_trace(go.Scatter(x=hs, y=rv, name=f"real EA ({emp['n_events']:,} executions)",
                                 mode='lines+markers', line=dict(color=C_REAL, width=2.2, dash='dash'),
                                 error_y=dict(type='data', array=se)))
    layout(fig, 'EA — response to a single child order R(m)',
           'messages after the execution event m', 'mean mid response R(m) (ticks)')
    save(fig, 'fig7_event_response.html')


# ---------------- Fig 6: midtraj (k-clock + event-time) ----------------
def fig6(shape):
    npz = os.path.join(MID, f'mid_trajectory_EA_{shape}.npz')
    d = np.load(npz, allow_pickle=True)
    n_ins = 100 if shape == 'beta' else 10
    fig = go.Figure()
    for m in ALL:
        if f'{m}_k_mean' not in d.files:
            continue
        km, kse = d[f'{m}_k_mean'], d[f'{m}_k_se']
        x = np.arange(len(km)); s = st(m)
        n = int(np.nanmax(d[f'{m}_cnt'])) if f'{m}_cnt' in d.files else 0
        band(fig, x, km - 1.96 * kse, km + 1.96 * kse, s['color'], m)
        fig.add_trace(go.Scatter(x=x, y=km, name=f"{s['label']} (n={n})",
                                 line=dict(color=s['color'], width=2.2), legendgroup=m))
    ys = d['sqrt_y']
    xk = np.arange(1, n_ins + 1, dtype=float); yk = ys[:n_ins].copy()
    nb = len(d[[k for k in d.files if k.endswith('_k_mean')][0]]) - 1
    if shape != 'beta':
        xk = np.append(xk, float(nb)); yk = np.append(yk, yk[-1])
    fig.add_trace(go.Scatter(x=xk, y=yk, name='sqrt-law beta=0.5',
                             line=dict(color='black', width=2.4, dash='dash')))
    if shape != 'beta':
        px, py = propagator_curve(float(yk[n_ins - 1]), float(n_ins), None, 0.5, nb)
        fig.add_trace(go.Scatter(x=px, y=py, name='propagator beta=0.5',
                                 line=dict(color='#8E44AD', width=2.2, dash='dot')))
        fig.add_vline(x=n_ins, line=dict(color='#9a9a9a', width=1, dash='dot'))
    xt = ('insertion index k (children executed)' if shape == 'beta'
          else 'boundary index k (insertions 1-10, then cooling windows)')
    layout(fig, f'EA — mid-price impact on the executed-volume clock ({shape})',
           xt, 'mean signed mid-price change (bps)')
    save(fig, f'fig5_midtraj_{shape}.html')

    # event-time variant (raw message clock — lengths differ across days)
    fig2 = go.Figure()
    for m in ALL:
        if f'{m}_mid' not in d.files:
            continue
        mid, bd, x = d[f'{m}_mid'], d[f'{m}_band'], d[f'{m}_x']
        s = st(m)
        band(fig2, x, mid - 1.96 * bd, mid + 1.96 * bd, s['color'], m)
        fig2.add_trace(go.Scatter(x=x, y=mid, name=s['label'],
                                  line=dict(color=s['color'], width=1.8), legendgroup=m))
    if 'sqrt_x' in d.files:
        fig2.add_trace(go.Scatter(x=d['sqrt_x'], y=d['sqrt_y'], name='sqrt-law beta=0.5',
                                  line=dict(color='black', width=2.4, dash='dash')))
    if 'prop_x' in d.files:
        fig2.add_trace(go.Scatter(x=d['prop_x'], y=d['prop_y'], name='propagator beta=0.5',
                                  line=dict(color='#8E44AD', width=2.2, dash='dot')))
    layout(fig2, f'EA — mid-price impact in EVENT TIME ({shape}; composition artifacts visible)',
           'message step (generation time)', 'mean signed mid-price change (bps)')
    save(fig2, f'fig5_midtraj_{shape}_eventtime.html')


# ---------------- Fig 7: master curves ----------------
def fig7(shape):
    npz = os.path.join(MASTER, f'master_curve_EA_{shape}_v2gated.npz')
    d = np.load(npz, allow_pickle=True)
    vgrid = d['vgrid']
    fig = go.Figure()
    for m in ALL:
        if f'{m}_master' not in d.files:
            continue
        s = st(m)
        sig = bool(d[f'{m}_sig'])
        n = int(d[f'{m}_n']); pk = float(d[f'{m}_peak']); pse = float(d[f'{m}_peak_se'])
        nm = f"{s['label']} (n={n}, I(1)={pk:.2f}±{pse:.2f} bps{'' if sig else ' — NOISE/NOISE'})"
        fig.add_trace(go.Scatter(x=vgrid, y=d[f'{m}_master'], name=nm,
                                 line=dict(color=s['color'], width=2.2 if sig else 1.1),
                                 opacity=1.0 if sig else 0.55, legendgroup=m))
    vb = np.linspace(0.01, 1.0, 60)
    fig.add_trace(go.Scatter(x=vb, y=vb ** 0.5, name='sqrt-law build-up v^0.5',
                             line=dict(color='black', width=2.4, dash='dash')))
    if shape != 'beta':
        px, prop = propagator_curve(1.0, 1.0, None, 0.5, float(vgrid[-1]))
        fig.add_trace(go.Scatter(x=px, y=2 / 3 + prop / 3,
                                 name='theory: relax to permanent ~2/3 of peak',
                                 line=dict(color='#8E44AD', width=2.2, dash='dash')))
    fig.add_vline(x=1.0, line=dict(color='#9a9a9a', width=1, dash='dot'))
    fig.add_hline(y=1.0, line=dict(color='#9a9a9a', width=0.7))
    layout(fig, f'EA — impact master curve ({shape}); all models drawn '
                '(NOISE/NOISE = noise-level denominator, curve not interpretable)',
           'v = fraction of metaorder executed', 'master(v) = <I(v)>/<I(1)>')
    save(fig, f'fig6_master_{shape}.html')


# ---------------- Fig 8: beta(k) three views ----------------
def fig8():
    # paper uses the sigma=1 variant
    npz = os.path.join(B3V, 'beta_vs_k_3views_EA_sigma1.npz')
    d = np.load(npz, allow_pickle=True)
    views = [('cumul', '<=k (cumulative)'), ('exact', '=k (exact)'), ('reverse', '>=k (reverse-cumulative)')]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for _, t in views], shared_yaxes=True)
    for ci, (key, _) in enumerate(views, start=1):
        for m in ALL:
            if f'{m}_{key}' not in d.files:
                continue
            s = st(m)
            ks = d[f'{m}_ks']
            fig.add_trace(go.Scatter(x=ks, y=d[f'{m}_{key}'], name=s['label'],
                                     line=dict(color=s['color'],
                                               width=2.0 if key != 'exact' else 1.3),
                                     legendgroup=m, showlegend=(ci == 1)),
                          row=1, col=ci)
        fig.add_hline(y=0.5, line=dict(color='#C0392B', width=1.2, dash='dash'), row=1, col=ci)
    fig.update_yaxes(range=[-0.4, 1.05], title_text='beta(k)', row=1, col=1)
    for ci in (1, 2, 3):
        fig.update_xaxes(title_text='insertion index k', row=1, col=ci)
    fig.update_layout(title='EA — square-root exponent beta(k) in three views (sigma=1, free intercept)',
                      template='plotly_white', hovermode='x unified',
                      legend=dict(groupclick='togglegroup'),
                      margin=dict(l=60, r=20, t=80, b=50))
    save(fig, 'fig8_beta_3views.html')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig5()
    for sh in ('beta', 'decay'):
        fig6(sh)
    for sh in ('beta', 'relaxation'):
        fig7(sh)
    fig8()
