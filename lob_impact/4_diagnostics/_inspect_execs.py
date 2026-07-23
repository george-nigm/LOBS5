#!/usr/bin/env python3
"""One-off: how is the aggressive child logged? event-type distribution + rows AT/around the
first few aggressive indices, for EA-Historic and EA-Mamba3 (beta shape). Resolves whether the
participation-floor 50% degeneracy is a child-exec double-count.  Run on a compute node."""
import os, glob, csv
import numpy as np

GRID = os.environ.get('GRID', '/lus/lfs1aip2/projects/u6gb/lob_impact_grid')
EXECS = (4, 5)


def _read(f):
    return np.array([[float(x) if x not in ('', 'nan') else np.nan for x in r]
                     for r in csv.reader(open(f)) if r], dtype=float)


def inspect(exp):
    side = os.path.join(GRID, exp, 'buy')
    ai = glob.glob(os.path.join(side, '**', 'aggressive_indices.csv'), recursive=True)
    mfs = sorted(glob.glob(os.path.join(side, '**', 'data_gen', '*message*gen*.csv'), recursive=True))
    print(f'\n===== {exp} =====')
    if not ai or not mfs:
        print('  no data'); return
    idx = np.loadtxt(ai[0], dtype=int, ndmin=1)
    print(f'  aggressive_indices (first 8): {idx[:8].tolist()}  (total {len(idx)})')
    m = _read(mfs[0])
    print(f'  message file: {os.path.basename(mfs[0])}  shape={m.shape}')
    et = m[:, 1].astype(int)
    vals, cnts = np.unique(et, return_counts=True)
    print(f'  event_type distribution: {dict(zip(vals.tolist(), cnts.tolist()))}')
    exec_frac = np.isin(et, EXECS).mean()
    print(f'  exec fraction (type in {EXECS}): {exec_frac:.3%}')
    # rows around the first 3 insertions
    for k in idx[:3]:
        lo, hi = max(0, k - 1), min(len(m), k + 3)
        print(f'  -- around insertion idx={k}: rows[{lo}:{hi}] (col1=event_type, col3=size, col5=dir):')
        for r in range(lo, hi):
            tag = ' <== insertion' if r == k else ''
            print(f'       row {r}: et={int(m[r,1])} size={m[r,3]:.0f} dir={int(m[r,5])}{tag}')
    # how many execs strictly BETWEEN insertion k and k+1 (organic), first 5 windows
    org = []
    for a, b in zip(idx[:6], idx[1:7]):
        seg = et[a + 1:b]              # exclude the insertion row itself
        org.append(int(np.isin(seg, EXECS).sum()))
    print(f'  organic execs per window (first windows, excl. insertion row): {org}')


for exp in ('EA-Historic-beta', 'EA-Mamba3-beta'):
    inspect(exp)
