#!/usr/bin/env python3
"""
Daily OHLC for GOOG 2023_Jan from the flair06 old-proc npy (for the LobS5/twilight beta:
Parkinson sigma needs H/L of the SAME days the legacy grid was generated on; the Action-2
table only covers Jan-2026). Same definition as compute_daily_stats.py: executions only
(event col 1 == 4), abs price col 3; open/close = first/last execution price.
Emits results/per_day_params/daily_h_l_GOOG2023.csv with the Action-2 schema
(ticker,day,open_price,highest_price,lowest_price,close_price,execution_sum).
"""
import glob
import os
import re

import numpy as np

SRC = ('/lus/lfs1aip2/projects/public/u6gb/projects_public_s5e_quant_team_quant/'
       'AlphaTrade/LOBS5/flair06_data/GOOG/2023_Jan')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'per_day_params',
                   'daily_h_l_GOOG2023.csv')

rows = []
for f in sorted(glob.glob(os.path.join(SRC, '*message*_proc.npy'))):
    day = re.search(r'GOOG_(\d{4}-\d{2}-\d{2})_', os.path.basename(f)).group(1)
    m = np.load(f, mmap_mode='r')
    ev = np.asarray(m[:, 1])
    is_mo = ev == 4
    px = np.asarray(m[is_mo, 3], dtype=np.int64)
    sz = np.asarray(m[is_mo, 5], dtype=np.int64)
    assert (px > 0).all(), f'{day}: price col 3 has non-positive values'
    rows.append((day, int(px[0]), int(px.max()), int(px.min()), int(px[-1]), int(sz.sum())))
    print(f'{day}: O={px[0]} H={px.max()} L={px.min()} C={px[-1]} vol={sz.sum()}')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write('ticker,day,open_price,highest_price,lowest_price,close_price,execution_sum\n')
    for r in rows:
        fh.write('GOOG,' + ','.join(str(x) for x in r) + '\n')
print(f'{len(rows)} days -> {OUT}')
