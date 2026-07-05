#!/usr/bin/env python3
"""
Action 4 — empirical event-response from REAL data (calibration anchor).

From the real conditioning streams (data_cond of any grid_v2 experiment: message + orderbook
CSV pairs, 500 rows each), measure the average mid-price response to an aggressive execution:

    R(m) = E[ sign * (mid_{t+m} - mid_{t-1}) | execution at t ]     in ticks

with sign=+1 for buy MOs (LOBSTER direction == -1: resting sell hit) and -1 for sell MOs.
Reported for horizons m in HORIZONS, split by trade size tercile. This is what a CALIBRATED
generator should reproduce per insertion; compare with the model's marginal per-insertion
impact (~1.2-1.7 ticks for Mamba3 at ~14 shares).

  python 4_diagnostics/empirical_response.py --exp <.../EA-Mamba3-beta/buy/exp_*> \
      --n_files 200 --out_dir 4_diagnostics/results/empresp_<ts>
"""
import os, csv, glob, json, argparse
import numpy as np

TICK = 100
SENT = 2147483647
HORIZONS = [1, 5, 10, 30, 60, 100, 131]


def read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='experiment dir containing data_cond/')
    ap.add_argument('--n_files', type=int, default=200)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    msgs = sorted(glob.glob(os.path.join(args.exp, 'data_cond', '*message*.csv')))
    step = max(len(msgs) // args.n_files, 1)
    resp = {m: [] for m in HORIZONS}
    sizes = []
    n_ev = 0
    for mf in msgs[::step][:args.n_files]:
        bf = mf.replace('message', 'orderbook')
        if not os.path.exists(bf):
            continue
        m = read_csv(mf)
        b = read_csv(bf)
        L = min(len(m), len(b))
        m, b = m[:L], b[:L]
        ask, bid = b[:, 0], b[:, 2]
        bad = (ask >= SENT) | (bid <= 0) | (ask <= 0)
        mid = (ask + bid) / 2.0
        mid[bad] = np.nan
        ex = np.where(m[:, 1] == 4)[0]
        ex = ex[(ex > 0) & (ex < L - max(HORIZONS))]
        for t in ex:
            sign = 1.0 if m[t, 5] == -1 else -1.0   # -1: sell side hit => buy MO
            base = mid[t - 1]
            if np.isnan(base):
                continue
            n_ev += 1
            sizes.append(m[t, 3])
            for h in HORIZONS:
                v = mid[t + h]
                resp[h].append(sign * (v - base) / TICK if not np.isnan(v) else np.nan)

    sizes = np.array(sizes)
    out = {'n_events': n_ev, 'size_mean': float(sizes.mean()) if n_ev else np.nan,
           'size_median': float(np.median(sizes)) if n_ev else np.nan}
    print(f'{n_ev} execution events | size mean {out["size_mean"]:.1f} median {out["size_median"]:.1f}')
    print(f'{"horizon":>8} {"R(m) ticks":>12} {"SE":>8}   (small/mid/large size tercile)')
    if n_ev:
        terc = np.quantile(sizes, [1 / 3, 2 / 3])
        for h in HORIZONS:
            v = np.array(resp[h])
            ok = ~np.isnan(v)
            r, se = float(np.nanmean(v)), float(np.nanstd(v) / max(np.sqrt(ok.sum()), 1))
            by_t = [float(np.nanmean(v[(sizes < terc[0])])),
                    float(np.nanmean(v[(sizes >= terc[0]) & (sizes < terc[1])])),
                    float(np.nanmean(v[(sizes >= terc[1])]))]
            out[f'R_{h}'] = r
            out[f'R_{h}_se'] = se
            out[f'R_{h}_terciles'] = by_t
            print(f'{h:>8} {r:>12.3f} {se:>8.3f}   ({by_t[0]:+.3f} / {by_t[1]:+.3f} / {by_t[2]:+.3f})')
    with open(os.path.join(args.out_dir, 'empirical_response.json'), 'w') as fo:
        json.dump(out, fo, indent=1)
    print(f'wrote {args.out_dir}/empirical_response.json')


if __name__ == '__main__':
    main()
