#!/usr/bin/env python
"""
Free-drift column of the scorecard: mean end displacement of the no-insertion rollouts, in ticks.

This is Stage 0's direct measurement of unconditional drift — the number the visible response is
read against. It existed for NVDA (-4 / -6 / -27 / -9 / -14 for the neural panel) but was never
computed for AMD, which left the drift column all-AWAIT in the AMD edition.

Per model: read the noins orderbook files, mid = (ask+bid)/2 with the empty-book sentinel dropped,
drift = (last finite mid - first finite mid) / TICK, averaged over rollouts. Both sides (buy/sell)
pooled — no injection happens, so the label carries no meaning beyond which folder the rollout
landed in.

  python compute_drift.py --stock AMD [--n_files 64]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

NOINS = '/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2/noins'
TICK = 100.0
SENT = 2_000_000_000.0
CANON = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR',
         'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']


def read_ob(f):
    rows = []
    for r in csv.reader(open(f)):
        if not r:
            continue
        try:
            rows.append([float(x) for x in r[:4]])
        except ValueError:
            continue          # corrupted sentinel rows: same defect placebo_pretrend guards against
    return np.array(rows, float) if rows else np.empty((0, 4))


def drift_one(f):
    ob = read_ob(f)
    if ob.ndim != 2 or ob.shape[0] < 100:
        return np.nan
    ask, bid = ob[:, 0], ob[:, 2]
    mid = (ask + bid) / 2.0
    mid[(ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)] = np.nan
    fin = np.flatnonzero(np.isfinite(mid))
    if len(fin) < 100:
        return np.nan
    return (mid[fin[-1]] - mid[fin[0]]) / TICK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--n_files', type=int, default=64, help='per model per side')
    args = ap.parse_args()

    out_lines = [f'# Free drift — {args.stock} (mean end displacement of noins rollouts, ticks)', '',
                 '| model | drift, ticks | s.e. | n rollouts |', '|---|---|---|---|']
    for m in CANON:
        vals = []
        for side in ('buy', 'sell'):
            fs = sorted(glob.glob(os.path.join(
                NOINS, f'{args.stock}-{m}-beta', side, '**', 'data_gen', '*orderbook*gen*.csv'),
                recursive=True))[:args.n_files]
            for f in fs:
                vals.append(drift_one(f))
        v = np.array(vals, float)
        v = v[np.isfinite(v)]
        # Book-collapse guard, same rule as the impact curves: a broken book drifts by thousands of
        # ticks and owns the mean of 64 rollouts (Mamba3-4k first read -335 +- 86 from this).
        bad = np.abs(v) > 5000.0
        if bad.any():
            print(f'  [collapse] {m}: {int(bad.sum())}/{len(v)} прогонов отброшено '
                  f'(экстремум {v[np.argmax(np.abs(v))]:+.0f} тиков)', flush=True)
            v = v[~bad]
        if not len(v):
            print(f'{m:11s} нет данных')
            out_lines.append(f'| {m.replace("_", "-")} | n/a | | 0 |')
            continue
        mu, se = float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))
        print(f'{m:11s} drift {mu:+8.1f} тиков  (s.e. {se:.1f}, n={len(v)})', flush=True)
        out_lines.append(f'| {m.replace("_", "-")} | {mu:+.0f} | {se:.1f} | {len(v)} |')

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, 'results', f'drift_{args.stock}.md')
    open(out, 'w').write('\n'.join(out_lines) + '\n')
    print('\nsaved ->', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
