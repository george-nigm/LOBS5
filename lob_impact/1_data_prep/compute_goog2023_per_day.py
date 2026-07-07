#!/usr/bin/env python3
"""
Per-day scenario params for the LEGACY twilight-sound-77 runs: GOOG 2023_Jan (flair06 old-proc
format). Same design as compute_per_day_params.py (eta=10%): child_d = p50 market-order size,
mb_d = int(9 / trade_frac_d). Validates the column convention (event col 1 in {1..4}, size col 5)
before trusting it — old and new proc share the preproc.py lineage but this asserts it.

Run via sbatch (reads 18 x ~200MB npy): emits per_day_params_GOOG2023.csv (day,mult,child,mb,...).
"""
import glob
import os
import re

import numpy as np

SRC = ('/lus/lfs1aip2/projects/public/u6gb/projects_public_s5e_quant_team_quant/'
       'AlphaTrade/LOBS5/flair06_data/GOOG/2023_Jan')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'per_day_params',
                   'per_day_params_GOOG2023.csv')

rows = []
for f in sorted(glob.glob(os.path.join(SRC, '*message*_proc.npy'))):
    day = re.search(r'GOOG_(\d{4}-\d{2}-\d{2})_', os.path.basename(f)).group(1)
    m = np.load(f, mmap_mode='r')
    ev = np.asarray(m[:, 1])
    uniq = np.unique(ev)
    assert set(uniq.tolist()) <= {1, 2, 3, 4}, f'{day}: event col 1 has {uniq} — column convention differs!'
    is_mo = ev == 4
    n, ntr = len(ev), int(is_mo.sum())
    trade_frac = ntr / n
    size = np.asarray(m[is_mo, 5])
    assert (size > 0).all(), f'{day}: size col 5 has non-positive values — column convention differs!'
    child = int(np.median(size))
    mb = int(9 / trade_frac)
    rows.append((day, 1.0, child, mb, float(np.median(size)), trade_frac, mb, ntr))
    print(f'{day}: n={n} trades={ntr} trade_frac={trade_frac:.4f} child={child} mb={mb}')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write('day,mult,child,mb,p50_mo_volume,trade_frac,msgs_btw,n_trades\n')
    for r in rows:
        fh.write(','.join(str(x) for x in r) + '\n')
print(f'{len(rows)} days -> {OUT}')
