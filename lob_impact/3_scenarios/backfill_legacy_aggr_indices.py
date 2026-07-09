#!/usr/bin/env python3
"""
Backfill per-day aggressive_indices_<day>.csv in the twilight legacy grid.
lobs5_scenario.py writes only the summary aggressive_indices.csv, but the beta collector
(beta_grid.collect/_aggr_by_day) requires aggressive_indices_<YYYY-MM-DD>.csv. Every legacy
exp folder is a SINGLE day (one-day per_day CSV per job), so the summary IS that day's
indices — copy it under the dated name. Sanity: first index must equal that day's mb
(Shape I: idx_i = (i+1)*mb+i -> idx_0 = mb) read from the folder's config.yaml.
"""
import glob
import os
import re
import shutil

import yaml

ROOT = '/lus/lfs1aip2/projects/u6gb/lob_impact_legacy'
n_done = n_skip = 0
for exp in sorted(glob.glob(os.path.join(ROOT, 'GOOG-LobS5-*', '*', 'exp_*'))):
    src = os.path.join(exp, 'aggressive_indices.csv')
    if not os.path.exists(src):
        print(f'[skip] no summary: {exp}'); n_skip += 1; continue
    gen = glob.glob(os.path.join(exp, 'data_gen', '*message*gen*.csv'))
    if not gen:
        print(f'[skip] no data_gen: {exp}'); n_skip += 1; continue
    days = {re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(g)).group(1) for g in gen}
    if len(days) != 1:
        print(f'[skip] multi-day exp (unexpected): {exp} days={sorted(days)}'); n_skip += 1; continue
    day = days.pop()
    with open(src) as fh:
        first = int(fh.readline().strip())
    cfg = yaml.safe_load(open(os.path.join(exp, 'config.yaml')))
    mb = int(cfg['n_gen_msgs'])
    # per-day mode: config.yaml keeps the template mb; the run's real mb comes from the 1-day
    # CSV, so validate against it when present
    pdc = cfg.get('per_day_params')
    if pdc and os.path.exists(pdc):
        import csv as _csv
        with open(pdc) as fh:
            row = next(_csv.DictReader(fh))
        mb = int(row['mb'])
    assert first == mb, f'{exp}: first index {first} != mb {mb}'
    dst = os.path.join(exp, f'aggressive_indices_{day}.csv')
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
        n_done += 1
print(f'backfilled {n_done}, skipped {n_skip}')
