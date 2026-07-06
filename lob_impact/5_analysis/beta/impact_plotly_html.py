#!/usr/bin/env python3
"""
ALL-MODELS interactive mid-price impact TRAJECTORY as one self-contained Plotly HTML.

Same physics as mid_trajectory.py (the PNG version) — reuses its collectors and overlays:
  collect_traj, sqrt_law_curve, propagator_curve, COLORS — imported here, NOT re-derived.

For every model folder present in the grid (skip missing/empty gracefully) we plot one central
trajectory line + a 95% confidence band (trimmed-mean per step over buy+sell samples, in bps),
plus a √-law (β=0.5) overlay (beta shape: rising peak; decay/relaxation shape: permanent ref) and
a Bouchaud propagator overlay (decay/relaxation ONLY). x = message step (generation time),
y = mean signed mid-price change in bps (buy + (−1)·sell).

The HTML inlines plotly.js so it opens offline. Legend interaction is Plotly-native:
  • single-click a legend entry  -> toggle that model's line+band on/off (legendgroup togglegroup);
  • double-click a legend entry   -> isolate it (hide all others); double-click again -> restore.

  python 5_analysis/beta/impact_plotly_html.py --grid <root> --stock EA --shape beta \
         --daily <daily_h_l_all.csv> --per_day_params <per_day_params_EA.csv> \
         --models Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k \
         --out impact_trajectory_EA_beta.html
"""
import os, argparse
import numpy as np
import plotly.graph_objects as go

# reuse the verified trajectory logic from the PNG script (sibling import; run with cwd in beta/)
from mid_trajectory import collect_traj, sqrt_law_curve, propagator_curve, COLORS

# extend COLORS with a distinct hue for Hawkes (PNG script lacks it; #8E44AD is taken by propagator)
COLORS = dict(COLORS)
COLORS.setdefault('Hawkes', '#D4AC0D')


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def central_line(trajs, estimator='tmean', trim=0.10, min_frac=0.5):
    """Trimmed-mean (default) per-step central trajectory + SEM band, padded to the longest sample.
       Extracted verbatim from mid_trajectory.py lines 145-178 so the HTML and PNG agree.
       Returns (x, mid, band, Lkeep, nsamp). 95% band = mid ± 1.96*band."""
    Lm = max(len(t) for t in trajs)
    M = np.full((len(trajs), Lm), np.nan)
    for i, t in enumerate(trajs):
        M[i, :len(t)] = t
    cnt = np.sum(np.isfinite(M), axis=0)
    if estimator == 'tmean' and trim > 0:
        lo = np.nanpercentile(M, trim * 100, axis=0)
        hi = np.nanpercentile(M, (1 - trim) * 100, axis=0)
        Mk = np.where((M >= lo) & (M <= hi), M, np.nan)
        cntk = np.sum(np.isfinite(Mk), axis=0)
        mid = np.nanmean(Mk, axis=0)
        band = np.nanstd(Mk, axis=0) / np.sqrt(np.maximum(1, cntk))
    elif estimator == 'median':
        q1, mid, q3 = (np.nanpercentile(M, q, axis=0) for q in (25, 50, 75))
        band = 1.57 * (q3 - q1) / np.sqrt(np.maximum(1, cnt))
    else:
        mid = np.nanmean(M, axis=0)
        band = np.nanstd(M, axis=0) / np.sqrt(np.maximum(1, cnt))
    keep = cnt >= max(min_frac * len(trajs), 30)
    Lkeep = int(np.max(np.where(keep)[0])) + 1 if keep.any() else Lm
    mid = mid.copy(); band = band.copy()
    mid[~keep] = np.nan; band[~keep] = np.nan
    return np.arange(Lm), mid, band, Lkeep, len(trajs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--shape', default='beta', help='grid exp suffix: beta | relaxation')
    ap.add_argument('--models', default='Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k')
    ap.add_argument('--daily', default=None, help='daily H/L CSV; with --per_day_params -> √-law overlay')
    ap.add_argument('--per_day_params', default=None, help='per_day_params_<STOCK>.csv (child/mb/V per day)')
    ap.add_argument('--Y', type=float, default=1.0, help='√-law prefactor  I=Y*sigma*sqrt(Q/V)')
    ap.add_argument('--sigma_method', default='parkinson')
    ap.add_argument('--estimator', default='tmean', choices=['tmean', 'median', 'mean'])
    ap.add_argument('--beta_prop', type=float, default=0.5, help='propagator decay exponent (decay overlay)')
    ap.add_argument('--trim', type=float, default=0.10)
    ap.add_argument('--min_frac', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    is_beta = (args.shape == 'beta')
    shape_label = 'beta' if is_beta else 'decay'
    models = [m for m in args.models.split(',') if m]

    fig = go.Figure()
    ins_steps = None
    Lmax = 0

    for model in models:
        exp = f'{args.stock}-{model}-{args.shape}'
        tb, ab = collect_traj(os.path.join(args.grid, exp, 'buy'), +1)
        ts, asl = collect_traj(os.path.join(args.grid, exp, 'sell'), -1)
        trajs = tb + ts
        if not trajs:
            print(f'{exp}: no data (skipped)'); continue
        x, mid, band, Lkeep, n = central_line(trajs, args.estimator, args.trim, args.min_frac)
        finite = np.isfinite(mid)
        if not finite.any():
            print(f'{exp}: all-nan central line (skipped)'); continue
        end = mid[finite][-1]
        c = COLORS.get(model, '#444444')

        xf = x[finite]
        mid_f = mid[finite]
        band_f = band[finite]
        # 95% band (drawn first so the line sits on top), grouped with the line for joint toggle
        fig.add_trace(go.Scatter(
            x=np.concatenate([xf, xf[::-1]]),
            y=np.concatenate([mid_f + 1.96 * band_f, (mid_f - 1.96 * band_f)[::-1]]),
            fill='toself', fillcolor=_hex_to_rgba(c, 0.12), line=dict(width=0),
            legendgroup=model, showlegend=False, hoverinfo='skip', name=f'{model} band'))
        # central line
        fig.add_trace(go.Scatter(
            x=x, y=mid, mode='lines',
            line=dict(color=c, width=1.8), legendgroup=model,
            name=f'{model} (end {end:.1f} bps, n={n})',
            hovertemplate='step %{x}<br>%{y:.2f} bps<extra>' + model + '</extra>'))

        if ins_steps is None:
            ins_steps = ab if ab is not None else asl
        Lmax = max(Lmax, Lkeep)
        print(f'{exp}: n={n}, Lkeep={Lkeep}, end={end:.2f} bps')

    if not Lmax:
        print(f'{args.stock} [{args.shape}]: no models had data; nothing to plot'); return

    # --- √-law (β=0.5) overlay, per-day child/mb/V ---
    if args.daily and args.per_day_params and ins_steps is not None:
        n_ins = len(ins_steps)
        n_blocks = 100 if is_beta else 110
        xs, ys = sqrt_law_curve(args.daily, args.per_day_params, args.stock,
                                n_ins, n_blocks, Lmax, args.Y, args.sigma_method)
        if xs is not None:
            keep = xs <= Lmax
            lbl = '√-law β=0.5 peak' + (f' (Y={args.Y:g})' if is_beta else ' (permanent ref)')
            fig.add_trace(go.Scatter(
                x=xs[keep], y=ys[keep], mode='lines',
                line=dict(color='black', width=2.0, dash='dash'),
                name=f'{lbl}, end {ys[keep][-1]:.2f} bps'))
            print(f'√-law: end={ys[keep][-1]:.3f} bps (σ={args.sigma_method}, Y={args.Y:g})')
            # propagator (Bouchaud transient impact) — DECAY/relaxation shape only
            if not is_beta and n_ins >= 1:
                peak = float(ys[n_ins - 1]); T = float(xs[n_ins - 1])
                px, py = propagator_curve(peak, T, xs, args.beta_prop, Lmax)
                fig.add_trace(go.Scatter(
                    x=px, y=py, mode='lines',
                    line=dict(color='#8E44AD', width=2.2, dash='dot'),
                    name=f'propagator β={args.beta_prop:g} (peak {peak:.2f}→{py[-1]:.2f} bps)'))
                print(f'propagator: peak={peak:.3f} @T={T:.0f}, end={py[-1]:.3f} bps')

    # --- insertion ticks (faint, non-toggling layout shapes) ---
    if ins_steps is not None:
        for s in ins_steps[ins_steps < Lmax]:
            fig.add_shape(type='line', x0=float(s), x1=float(s), y0=0, y1=1, yref='paper',
                          line=dict(color='black', width=0.3), opacity=0.12, layer='below')

    fig.add_hline(y=0, line=dict(color='black', width=0.6))
    fig.update_layout(
        title=f'{args.stock} — mid-price trajectory by model  [{shape_label}]'
              + ('   (thin ticks = aggressive insertions)' if ins_steps is not None else ''),
        xaxis=dict(title='message step  (generation time)', range=[0, Lmax]),
        yaxis=dict(title='mean signed mid-price change  (bps)   buy + (−1)·sell'),
        legend=dict(groupclick='togglegroup', itemclick='toggle', itemdoubleclick='toggleothers'),
        template='plotly_white', hovermode='closest', width=1100, height=640)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'mid_impact', f'impact_trajectory_{args.stock}_{shape_label}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs='inline', full_html=True)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
