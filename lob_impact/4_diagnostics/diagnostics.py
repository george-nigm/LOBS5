#!/usr/bin/env python3
"""
Action 4 — diagnostics for one impact experiment.

Reads an experiment's generated trajectories (data_gen/) and plots the three sanity views
from the framework: (1) master curve (mean mid-price impact vs normalized time, insertions
marked), (2) book update (spread + touch depth over time), (3) participation rate
(metaorder executed volume / total executed volume).

  python 4_diagnostics/diagnostics.py --exp_dir <.../EA-Mamba3-beta/buy/mb5/exp_*>
  python 4_diagnostics/diagnostics.py --run_dir <.../results/run_*>   # all experiments in a run

Generated L10 orderbook CSV layout (40 cols): [ask1_p, ask1_v, bid1_p, bid1_v, ask2_p, ...].
Generated message CSV (LOBSTER): time, event_type, order_id, size, price, direction.
"""
import os, csv, glob, argparse, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICK = 100
EXEC = 4  # event_type == execution
SENTINEL = 2147483647  # empty book side: ask saved as +int32max, bid as -int32max


def _read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def load_experiment(exp_dir):
    """Return aligned arrays: mids (n,L), spreads (n,L), touch_depth (n,L), and per-sample exec sums."""
    obs = sorted(glob.glob(os.path.join(exp_dir, 'data_gen', '*orderbook*gen*.csv')))
    mids, spreads, depths, exec_meta, exec_total = [], [], [], [], []
    for ob in obs:
        a = _read_csv(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask_p, ask_v, bid_p, bid_v = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        # sentinel mask (same as master_curve.py): one collapsed row would shift means by ~1e9
        bad = (ask_p >= SENTINEL) | (bid_p >= SENTINEL) | (ask_p <= 0) | (bid_p <= 0)
        mid = (ask_p + bid_p) / 2.0
        spread = (ask_p - bid_p) / TICK
        depth = ask_v + bid_v
        mid[bad] = np.nan
        spread[bad] = np.nan
        depth[bad] = np.nan
        mids.append(mid)
        spreads.append(spread)
        depths.append(depth)
        # participation: from the paired message file
        mf = ob.replace('orderbook', 'message')
        if os.path.exists(mf):
            m = _read_csv(mf)
            if m.ndim == 2 and m.shape[1] >= 4:
                ex = m[m[:, 1] == EXEC]
                exec_total.append(float(ex[:, 3].sum()) if ex.size else 0.0)
    if not mids:
        raise RuntimeError(f'no usable orderbook trajectories in {exp_dir}')
    L = min(len(t) for t in mids)
    mids = np.stack([t[:L] for t in mids])
    spreads = np.stack([t[:L] for t in spreads])
    depths = np.stack([t[:L] for t in depths])
    aggr = np.loadtxt(os.path.join(exp_dir, 'aggressive_indices.csv'), dtype=int, ndmin=1)
    return dict(mids=mids, spreads=spreads, depths=depths,
                aggr=aggr[aggr < L], exec_total=np.array(exec_total), n=mids.shape[0], L=L)


def diagnostics_figure(exp_dir, out_dir):
    d = load_experiment(exp_dir)
    mids, L, n = d['mids'], d['L'], d['n']
    u = np.arange(L)

    # (1) master curve: mid impact in ticks, relative to the pre-metaorder mid (step 0)
    impact = (mids - mids[:, :1]) / TICK          # (n, L) in ticks; NaN where the book collapsed
    imp_mean, imp_std = np.nanmean(impact, 0), np.nanstd(impact, 0)

    name = os.path.relpath(exp_dir).split('/results/')[-1]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Diagnostics — {name}  (n={n} samples)', fontsize=11)

    ax[0].plot(u, imp_mean, color='#2F5DA3', lw=2, label='mean impact')
    ax[0].fill_between(u, imp_mean - imp_std / np.sqrt(n), imp_mean + imp_std / np.sqrt(n),
                       color='#2F5DA3', alpha=0.2, label='±SE')
    for k in d['aggr']:
        ax[0].axvline(k, color='#C0392B', ls='--', lw=0.8, alpha=0.6)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_title('Master curve (mid impact)'); ax[0].set_xlabel('step'); ax[0].set_ylabel('impact (ticks)')
    ax[0].legend(fontsize=8)

    # (2) book update: spread + touch depth over time
    ax[1].plot(u, np.nanmean(d['spreads'], 0), color='#2E7D52', lw=2, label='spread (ticks)')
    ax[1].set_title('Book update'); ax[1].set_xlabel('step'); ax[1].set_ylabel('spread (ticks)')
    axb = ax[1].twinx()
    axb.plot(u, np.nanmean(d['depths'], 0), color='#E08E0B', lw=1.5, alpha=0.7, label='touch depth')
    axb.set_ylabel('touch depth (shares)', color='#E08E0B')
    for k in d['aggr']:
        ax[1].axvline(k, color='#C0392B', ls='--', lw=0.8, alpha=0.5)

    # (3) participation rate: metaorder vol / total executed vol (per sample distribution)
    if d['exec_total'].size:
        # metaorder volume = order_volume * n_insertions (from config not loaded here -> infer от aggr count)
        # report total executed volume distribution as the participation denominator proxy
        ax[2].hist(d['exec_total'], bins=30, color='#7D5BA6', edgecolor='white')
        ax[2].axvline(np.median(d['exec_total']), color='#C0392B', ls='--',
                      label=f'median={np.median(d["exec_total"]):.0f}')
        ax[2].set_title('Total executed volume / sample'); ax[2].set_xlabel('shares'); ax[2].set_ylabel('# samples')
        ax[2].legend(fontsize=8)
    else:
        ax[2].text(0.5, 0.5, 'no message files', ha='center')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    safe = name.replace('/', '__')
    out = os.path.join(out_dir, f'diag__{safe}.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f'  final impact = {imp_mean[-1]:+.2f} ticks | spread {d["spreads"].mean():.2f} | wrote {out}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--exp_dir', help='a single experiment dir (contains data_gen/ + aggressive_indices.csv)')
    g.add_argument('--run_dir', help='a run dir; plots every experiment found under it')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    if args.exp_dir:
        exps = [args.exp_dir]
        out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                                                'diag_' + os.path.basename(args.exp_dir.rstrip('/')))
    else:
        exps = sorted(glob.glob(os.path.join(args.run_dir, '**', 'exp_*'), recursive=True))
        exps = [e for e in exps if os.path.isdir(os.path.join(e, 'data_gen'))]
        out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                                                'diag_' + os.path.basename(args.run_dir.rstrip('/')))
    print(f'{len(exps)} experiment(s) -> {out_dir}')
    for e in exps:
        try:
            diagnostics_figure(e, out_dir)
        except Exception as ex:
            print(f'  SKIP {e}: {ex}')


if __name__ == '__main__':
    main()
