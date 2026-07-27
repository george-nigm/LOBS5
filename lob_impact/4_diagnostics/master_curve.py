#!/usr/bin/env python3
"""
Buy/sell asymmetry overlay for ONE experiment (stock-model-shape).

Plots the master curve (mean mid-impact in ticks vs step) for buy and for sell,
with sell NEGATED (-1 * sell) so both point the same way — the gap between the
two lines IS the buy/sell asymmetry. Insertions marked.

  python 4_diagnostics/master_curve.py --grid <results/grid> --exp EA-Mamba3-beta
"""
import os, csv, glob, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICK = 100
SENTINEL = 2147483647   # empty book side -> not a price


def _read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def master_curve(side_dir):
    """Mean/SE mid-impact (ticks, vs step 0) over all gen trajectories under side_dir.
    Empty-book sentinel rows -> NaN so they don't poison the mean (nanmean)."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    mids, collapsed = [], []
    for ob in obs:
        a = _read_csv(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask_p, bid_p = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask_p >= SENTINEL) | (bid_p >= SENTINEL) | (ask_p <= 0) | (bid_p <= 0)
        mid = (ask_p + bid_p) / 2.0
        mid[bad] = np.nan
        mids.append(mid)
        collapsed.append(bad)
    if not mids:
        raise RuntimeError(f'no trajectories under {side_dir}')
    L = min(len(t) for t in mids)
    M = np.stack([t[:L] for t in mids])
    C = np.stack([c[:L] for c in collapsed])
    impact = (M - M[:, :1]) / TICK
    # Book-collapse guard, same rule as mid_trajectory.py. One rollout out of 2080 whose price ran
    # to +8668 bps lifted a single point of the AMD GDN curve from 7.07 to 11.23 bps and put a
    # matching spike in the master curve at the same k. A displacement past COLLAPSE_TICKS is not a
    # market move at any horizon; drop those trajectories and say how many.
    COLLAPSE_TICKS = 2000.0
    _bad = np.nanmax(np.abs(impact), axis=1) > COLLAPSE_TICKS
    if _bad.any():
        print(f'  [collapse] dropped {int(_bad.sum())}/{len(impact)} trajectories '
              f'(|I| > {COLLAPSE_TICKS:.0f} ticks; worst {np.nanmax(np.abs(impact[_bad])):.0f})', flush=True)
        impact = impact[~_bad]
        M = M[~_bad]
        C = C[~_bad]
    n = M.shape[0]
    aggr = np.loadtxt(glob.glob(os.path.join(side_dir, '**', 'aggressive_indices.csv'),
                                recursive=True)[0], dtype=int, ndmin=1)
    return dict(mean=np.nanmean(impact, 0), se=np.nanstd(impact, 0) / np.sqrt(n),
                collapse=C.mean(0), aggr=aggr[aggr < L], n=n, L=L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True, help='results/grid root')
    ap.add_argument('--exp', required=True, help='e.g. EA-Mamba3-beta')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    buy = master_curve(os.path.join(args.grid, args.exp, 'buy'))
    sell = master_curve(os.path.join(args.grid, args.exp, 'sell'))
    L = min(buy['L'], sell['L'])
    u = np.arange(L)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [2.2, 1]})
    fig.suptitle(f'{args.exp} — buy vs (-1)·sell master curve', fontsize=12)

    b = buy['mean'][:L]
    s = -sell['mean'][:L]
    avg = (b + s) / 2.0                      # average master curve (asymmetry removed)
    ax[0].plot(u, b, color='#2F5DA3', lw=1.6, alpha=0.85, label='buy')
    ax[0].fill_between(u, (buy['mean'] - buy['se'])[:L], (buy['mean'] + buy['se'])[:L],
                       color='#2F5DA3', alpha=0.15)
    ax[0].plot(u, s, color='#C0392B', lw=1.6, alpha=0.85, label='(-1)·sell')
    ax[0].fill_between(u, (-sell['mean'] - sell['se'])[:L], (-sell['mean'] + sell['se'])[:L],
                       color='#C0392B', alpha=0.15)
    ax[0].plot(u, avg, color='#1A1A1A', lw=2.4, label='average = (buy + (-1)·sell)/2')
    for k in buy['aggr']:
        ax[0].axvline(k, color='grey', ls='--', lw=0.5, alpha=0.4)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('step'); ax[0].set_ylabel('|mid impact| (ticks)')
    ax[0].set_title('Master curve (sell negated)'); ax[0].legend()

    # collapse fraction (book emptied) — honest marker that a side ran dry
    ax[1].plot(u, buy['collapse'][:L] * 100, color='#2F5DA3', lw=1.5, label='buy')
    ax[1].plot(u, sell['collapse'][:L] * 100, color='#C0392B', lw=1.5, label='sell')
    ax[1].set_xlabel('step'); ax[1].set_ylabel('% samples with empty book side')
    ax[1].set_title('Book collapse'); ax[1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'results', 'master_curve')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'master_curve__{args.exp}.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f'buy     final {buy["mean"][L-1]:+.2f} ticks (n={buy["n"]})')
    print(f'sell    final {sell["mean"][L-1]:+.2f} ticks (n={sell["n"]}) -> negated {-sell["mean"][L-1]:+.2f}')
    print(f'average final {avg[L-1]:+.2f} ticks')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
