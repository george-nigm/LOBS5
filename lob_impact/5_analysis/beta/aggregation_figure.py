#!/usr/bin/env python3
"""
Two outputs around the k-units aggregation story:

1. PAPER figure (pubstyle): raw trajectories of very different lengths (event time)
   -> the same samples aligned on insertion index k with the plain mean.
   "Guided by duration independence (Bouchaud et al. 2018) we average in executed-volume units."

2. DURATION-INDEPENDENCE check: per-day mean I(k=100) vs the day's m_b, per model.
   If impact depends on executed volume only, the scatter is flat. (Caveat: our m_b is
   calibrated to the day's trade rate, so this is an across-day, partially confounded test.)

  python 5_analysis/beta/aggregation_figure.py --grid <root> --stock EA \
         --models Mamba3,Mamba3_4k,S5_4k --out_dir results/aggregation
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(B, '..'))                      # pubstyle
sys.path.insert(0, os.path.join(B, '..', '..', '4_diagnostics'))
from pubstyle import apply, style, legend_box, savefig_pub     # noqa: E402
from control_triangle_report import discover_exp                # noqa: E402
from make_fig4_averaging_docx import collect_raw_and_k          # noqa: E402
apply()

HI = ['#2a78d6', '#1baf7a', '#eb6834']   # short / median / long m_b highlights


def fig_paper(out, data, n_show=300):
    raw, atk, mbs, _ = data
    n = len(raw)
    rng = np.random.default_rng(3)
    show = rng.choice(n, size=min(n_show, n), replace=False)
    order = np.argsort(mbs)
    hi_idx = [order[0], order[len(order) // 2], order[-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for i in show:
        t = np.arange(len(raw[i]))
        ax1.plot(t[::20], raw[i][::20], color='0.15', lw=0.4, alpha=0.05, zorder=2)
    offs = [(-900, 14), (300, -4), (-900, 14)]
    for (i, col), (dx, dy) in zip(zip(hi_idx, HI), offs):
        t = np.arange(len(raw[i]))
        end = raw[i][np.isfinite(raw[i])][-1]
        ax1.plot(t[::10], raw[i][::10], color=col, lw=1.5, zorder=3)
        ax1.plot(len(raw[i]) - 1, end, 'o', ms=5, color=col, zorder=4)
        ax1.annotate(f'$m_b$={mbs[i]}', xy=(len(raw[i]), end),
                     xytext=(len(raw[i]) + dx, end + dy), color=col,
                     fontsize=10, fontweight='bold',
                     ha='right' if dx < 0 else 'left')
    ax1.set_xlabel('message step (event time)')
    ax1.set_ylabel('signed mid move (bps)')
    ax1.set_title(f'(a) {min(n_show, n)} of $n$={n} raw trajectories', loc='left')

    ks = np.arange(atk.shape[1])
    for i in show:
        ax2.plot(ks, atk[i], color='0.15', lw=0.4, alpha=0.05, zorder=2)
    for i, col in zip(hi_idx, HI):
        ax2.plot(ks, atk[i], color=col, lw=1.5, zorder=3)
    mean = np.nanmean(atk, axis=0)
    se = np.nanstd(atk, axis=0) / np.sqrt(len(atk))
    ax2.fill_between(ks, mean - 2 * se, mean + 2 * se, color='#d03b3b', alpha=0.3,
                     lw=0, zorder=4)
    ax2.plot(ks, mean, color='#d03b3b', lw=2.4, zorder=5,
             label=f'mean over $n$={n}  ($\\pm 2$ s.e.)')
    ax2.set_xlabel('insertion index $k$ (children executed)')
    ax2.set_title('(b) aligned on $k$: equal length by construction', loc='left')
    legend_box(ax2, loc='upper left')
    lo, hi = np.nanpercentile(atk, [1, 99])
    pad = 0.12 * (hi - lo)
    for ax in (ax1, ax2):
        ax.set_ylim(min(lo, 0) - pad, hi + pad)
        ax.axhline(0, color='#9a9a9a', lw=0.7, zorder=1)
    fig.tight_layout()
    savefig_pub(fig, out)
    plt.close(fig)
    print(f'paper fig -> {out}  (n={n})')


def fig_duration(out, per_model):
    """Per-day mean I(k=100) vs the day's m_b — duration-independence check."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m, (atk, mbs, dates) in per_model.items():
        st = style(m)
        days = sorted(set(dates))
        x = np.array([mbs[dates == d].max() for d in days], float)
        y = np.array([np.nanmean(atk[dates == d, -1]) for d in days])
        ax.scatter(x, y, s=34, color=st['color'], label=None, zorder=3)
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        b1, b0 = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, b0 + b1 * xs, color=st['color'], lw=1.4, ls=st['ls'],
                label=f"{st['label']}: slope {b1:+.2f} bps per m_b, r={r:+.2f} "
                      f"({len(days)} days)", zorder=2)
    ax.set_xlabel(r'day $m_b$ (messages between children $\propto$ execution duration)')
    ax.set_ylabel(r'day mean $I(k{=}100)$ (bps, buy)')
    ax.set_title('Duration-independence check: final impact vs execution speed, per day')
    legend_box(ax, loc='upper left')
    fig.tight_layout()
    savefig_pub(fig, out)
    plt.close(fig)
    print(f'duration fig -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2')
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='S5_4k,Mamba3,Mamba3_4k')
    ap.add_argument('--paper_model', default='Mamba3')
    ap.add_argument('--out_dir', default=os.path.join(B, 'results', 'aggregation'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_model = {}
    for m in [x for x in args.models.split(',') if x]:
        exp = discover_exp(os.path.join(args.grid, f'{args.stock}-{m}-beta', 'buy'))
        if not exp:
            print(f'{m}: no data'); continue
        data = collect_raw_and_k(exp)
        raw, atk, mbs, dates = data
        if m == args.paper_model:
            fig_paper(os.path.join(args.out_dir, f'aggregation_{args.stock}.png'), data)
        per_model[m] = (atk, mbs, dates)
        print(f'{m}: n={len(atk)}, days={len(set(dates))}')
    fig_duration(os.path.join(args.out_dir, f'duration_independence_{args.stock}.png'),
                 per_model)
    print(f'AGGFIG_DONE -> {args.out_dir}')


if __name__ == '__main__':
    main()
