#!/usr/bin/env python
"""
Error accumulation over the generation horizon (LOB-Bench divergence family).

LOB-Bench slices each generated rollout into fixed-length buckets and scores every bucket
against the real distribution, so the score-vs-bucket curve measures how fast a model drifts
away from reality as it generates. That is the benchmark's own "model derailment" diagnostic
(Nagy et al. 2025, Fig. 5 right / Fig. 18) — but their protocol generates 500 messages, giving
5 buckets of 100. Our free rollouts are 25,001 messages, giving ~250 buckets: the same curve,
50x further out, in the regime a metaorder actually lives in.

Input: results/lobbench_noins/scores_<STOCK>_*/scores/scores_div_<STOCK>_<MODEL>_<H>_<ts>.pkl
  gzip+pickle, 2-tuple; [0] = {metric: {divergence: [ (point, CI, bootstrap), ... per bucket ]}}
The curve plotted is the MEAN over the 21 metrics for a given divergence, per bucket.
A `REAL` model entry (real data scored against itself) is drawn as the noise floor.

  python fig_divergence_horizon.py --stock NVDA [--div l1] [--out <png>]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'lobbench_noins')
NAME_RE = re.compile(r'scores_div_(?P<stock>[A-Z0-9]+)_(?P<model>.+?)_(?P<h>\d+)_(?P<ts>\d{8}_\d{6})\.pkl$')

CANON = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR',
         'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'OW': '#6A1B9A', 'CST': '#27AE60', 'NMZI': '#117864', 'Hawkes': '#D4AC0D',
          'QR': '#00ACC1', 'S5': '#5D6D7E', 'S5_120M': '#F06292', 'S5_4k': '#E67E22',
          'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'GDN': '#D81B60'}
NEURAL = {'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN'}


def curve(path, div):
    """-> (values per bucket, n_metrics contributing) averaged over the 21 metrics."""
    with gzip.open(path, 'rb') as fh:
        d = pickle.load(fh)[0]
    per_metric = []
    for metric, divs in d.items():
        seq = divs.get(div)
        if not seq:
            continue
        vals = []
        for entry in seq:
            try:
                v = float(np.atleast_1d(entry[0]).ravel()[0])
            except Exception:
                v = np.nan
            vals.append(v)
        per_metric.append(vals)
    if not per_metric:
        return None, 0
    n = max(len(v) for v in per_metric)
    arr = np.full((len(per_metric), n), np.nan)
    for i, v in enumerate(per_metric):
        arr[i, :len(v)] = v
    with np.errstate(all='ignore'):
        return np.nanmean(arr, axis=0), arr.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--div', default='l1', choices=['l1', 'ks', 'wasserstein'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--copy_to', default=None)
    ap.add_argument('--tag', default='*', help="restrict to scores_<STOCK>_<tag>/. A long-mode "
                    "divergence pickle is ~370 MB gzipped and expands to several GB — read those "
                    "in a batch job only; the windowed ones are <=51 MB.")
    args = ap.parse_args()

    latest = {}
    for p in glob.glob(os.path.join(RES, f'scores_{args.stock}_{args.tag}', 'scores', 'scores_div_*.pkl')):
        mo = NAME_RE.search(os.path.basename(p))
        if not mo or mo.group('stock') != args.stock:
            continue
        m, ts = mo.group('model'), mo.group('ts')
        if m not in latest or ts > latest[m][0]:
            latest[m] = (ts, p, int(mo.group('h')))
    if not latest:
        print(f'нет divergence-результатов для {args.stock} в {RES}')
        return 1

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=200)
    plotted, horizon, cached = [], 100, {}
    order = [m for m in CANON if m in latest] + [m for m in latest if m not in CANON and m != 'REAL']
    for m in order:
        ts, path, h = latest[m]
        horizon = h
        y, nmet = curve(path, args.div)
        cached[m] = y
        if y is None or not np.isfinite(y).any():
            continue
        x = (np.arange(len(y)) + 1) * h
        c = COLORS.get(m, '#444444')
        ax.plot(x, y, color=c, lw=2.0 if m in NEURAL else 1.4,
                alpha=1.0 if m in NEURAL else 0.85,
                label=f"{m.replace('_', '-')} ({len(y)} buckets)")
        plotted.append(m)

    if 'REAL' in latest:            # benchmark's own real-vs-real noise floor
        y, _ = curve(latest['REAL'][1], args.div)
        if y is not None:
            ax.plot((np.arange(len(y)) + 1) * horizon, y, color='#333333', ls=':', lw=1.8,
                    label='real vs real (noise floor)')

    # the regime LOB-Bench itself validated
    ax.axvspan(0, 500, color='#dddddd', alpha=0.55, lw=0, zorder=0)
    ax.text(520, ax.get_ylim()[1] * 0.96,
            "LOB-Bench's own protocol stops here (500 msgs)",
            fontsize=9, color='#666666', va='top')

    ax.set_xscale('log')
    ax.set_xlabel('generation horizon [messages]')
    ax.set_ylabel(f'{args.div.upper()} divergence from real (mean over 21 metrics)')
    ax.set_title(f'{args.stock}: error accumulation over the generated sequence '
                 f'(buckets of {horizon} messages)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, ncol=2, frameon=True, framealpha=0.8)
    fig.tight_layout()

    out = args.out or os.path.join(RES, f'divergence_horizon_{args.stock}_{args.div}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    np.savez_compressed(out.replace('.png', '.npz'),
                        **{f'{m}_y': cached[m] for m in plotted},
                        horizon=horizon, models=np.array(plotted))
    print('saved ->', out, '| models:', ', '.join(plotted))
    if args.copy_to:
        import shutil
        shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)
    return 0


if __name__ == '__main__':
    sys.exit(main())
