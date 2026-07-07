#!/usr/bin/env python3
"""
Per-event response R(m) for the BASELINE models (visible regime, buy) from grid_v2 —
so the single-order-response figure can show every model in the zoo. Caches to JSON
next to the triangle results.

  python 4_diagnostics/baseline_resp.py --out results/baseline_resp.json
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from control_triangle_report import load_regime, discover_exp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2')
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,Hawkes,CST')
    ap.add_argument('--n_samples', type=int, default=4096)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'baseline_resp.json'))
    args = ap.parse_args()

    out = {}
    for m in [x for x in args.models.split(',') if x]:
        exp = discover_exp(os.path.join(args.grid, f'{args.stock}-{m}-beta', 'buy'))
        if not exp:
            print(f'{m}: no data'); continue
        print(f'loading {m} <- {exp}')
        r = load_regime(exp, +1, args.n_samples, True)
        out[m] = {'n': r['n'], 'resp_n_events': r.get('resp_n_events', 0),
                  'resp_mean': np.asarray(r['resp_mean']).tolist(),
                  'resp_se': np.asarray(r['resp_se']).tolist()}
        rm = np.asarray(r['resp_mean'])
        last = rm[np.isfinite(rm)][-1] if np.isfinite(rm).any() else float('nan')
        print(f'  n={r["n"]} events={r.get("resp_n_events", 0)} R(end)={last:+.3f} ticks')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fo:
        json.dump(out, fo)
    print(f'BASERESP_DONE -> {args.out}')


if __name__ == '__main__':
    main()
