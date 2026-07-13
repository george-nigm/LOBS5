import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from participation_rate import sample_curves
GRID = '/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2'
for st in ['GOOG', 'NVDA']:
    for side in ['buy', 'sell']:
        r = sample_curves(os.path.join(GRID, f'{st}-Mamba3-beta', side))
        if r is None:
            print(f'{st} {side}: NO DATA'); continue
        cum, k0 = r['cum'], (int(r['aggr'][0]) if len(r['aggr']) else 0)
        c = cum[k0:]
        pk, pkat = float(np.nanmax(c)), int(np.nanargmax(c)) + k0
        print(f"{st} {side}: cum peak={100*pk:.2f}% @step {pkat} | final={100*cum[-1]:.2f}% "
              f"| floor={100*r['floor']:.2f}% | n={r['n']} L={r['L']}")
