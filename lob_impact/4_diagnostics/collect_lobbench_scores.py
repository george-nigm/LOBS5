#!/usr/bin/env python
"""
Collect the LOB-Bench noins scores into one table (the KS / L1 / Wasserstein
columns of Tables 7-8).

The scorer writes one gzipped pickle per (model, metric family) under
  results/lobbench_noins/scores_<STOCK>_<tag>/scores/scores_<family>_<STOCK>_<MODEL>_<ts>.pkl
Each pickle is a 2-tuple; element 0 maps 21 distributional metrics -> {'wasserstein','ks','l1'}
-> (point estimate, CI, bootstrap draws). The paper reports the MEAN over metrics, so that is
what we take; the per-metric detail stays in the pickles.

`unconditional` is the headline family (gen vs real on the same stock/month). Later timestamps
win when a model was scored more than once.

  python collect_lobbench_scores.py --stock NVDA [--family uncond] [--out scores_NVDA.md]
"""

import argparse
import glob
import gzip
import os
import pickle
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'lobbench_noins')
NAME_RE = re.compile(r'scores_(?P<fam>[a-z_]+)_(?P<stock>[A-Z0-9]+)_(?P<model>.+?)_(?P<ts>\d{8}_\d{6})\.pkl$')


def mean_scores(path):
    """-> dict divergence -> mean over the metrics that produced a finite estimate."""
    with gzip.open(path, 'rb') as fh:
        d = pickle.load(fh)[0]
    out = {}
    for div in ('ks', 'l1', 'wasserstein'):
        vals = []
        for m in d:
            v = d[m].get(div)
            if v is None:
                continue
            x = float(np.atleast_1d(v[0]).ravel()[0])
            if np.isfinite(x):
                vals.append(x)
        if vals:
            out[div] = (float(np.mean(vals)), len(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--family', default='uncond', help='uncond | cond | div | context | time_lagged')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    latest = {}
    for p in glob.glob(os.path.join(RES, f'scores_{args.stock}_*', 'scores', '*.pkl')):
        mo = NAME_RE.search(os.path.basename(p))
        if not mo or mo.group('stock') != args.stock or mo.group('fam') != args.family:
            continue
        model = mo.group('model')
        if model not in latest or mo.group('ts') > latest[model][0]:
            latest[model] = (mo.group('ts'), p)

    if not latest:
        print(f'нет результатов для {args.stock} (family={args.family}) в {RES}')
        return

    rows = []
    for model, (ts, p) in sorted(latest.items()):
        try:
            sc = mean_scores(p)
        except Exception as e:
            print(f'  [WARN] {model}: {type(e).__name__} {e}')
            continue
        rows.append((model, sc, ts))

    hdr = f'| Model | KS | L1 | Wasserstein | metrics | scored |'
    sep = '|---|---|---|---|---|---|'
    lines = [f'LOB-Bench ({args.family}) — {args.stock}', '', hdr, sep]
    for model, sc, ts in sorted(rows, key=lambda r: r[1].get('wasserstein', (9e9,))[0]):
        ks = sc.get('ks', (float('nan'), 0))
        l1 = sc.get('l1', (float('nan'), 0))
        ws = sc.get('wasserstein', (float('nan'), 0))
        lines.append('| %s | %.3f | %.3f | %.3f | %d | %s |'
                     % (model, ks[0], l1[0], ws[0], max(ks[1], l1[1], ws[1]),
                        ts[:8] + ' ' + ts[9:11] + ':' + ts[11:13]))
    txt = '\n'.join(lines)
    print(txt)
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write(txt + '\n')
        print('\nsaved ->', args.out)


if __name__ == '__main__':
    main()
