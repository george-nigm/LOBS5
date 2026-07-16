#!/usr/bin/env python3
"""
Model per-event response R(m) to m_max, from the grid (visible-buy rollouts).

Replaces the 131-capped triangle-cache curves so Figure 8 v2 has an even x-axis
against the 250-message validated real anchor. For every injected child order
(aggressive insertion) at t: R(m) = mean over events of (mid[t+m] - mid[t-1])/TICK,
window capped at the NEXT insertion (NaN beyond) — identical convention to the
old figure, longer horizon. Running sums only (no per-event storage).

  python model_response_curve.py --grid <root> --stock NVDA \
      [--models ...] [--m_max 250] [--n_files 300] --out_dir results/mresp_NVDA
"""
import os, re, csv, glob, argparse
import numpy as np

TICK = 100
SENT = 2147483647
DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def aggr_by_day(side_dir):
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def model_response(side_dir, m_max, n_files):
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr = aggr_by_day(side_dir)
    if not obs or not aggr:
        return None
    step = max(len(obs) // n_files, 1)
    s1 = np.zeros(m_max); s2 = np.zeros(m_max); cnt = np.zeros(m_max, int)
    n_ev = n_run = 0
    for ob in obs[::step][:n_files]:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        idx = aggr.get(m.group(1))
        if idx is None or len(idx) < 2 or idx[0] < 1:
            continue
        a = read_csv(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask, bid = a[:, 0], a[:, 2]
        mid = (ask + bid) / 2.0
        mid[(ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)] = np.nan
        L = len(mid)
        n_run += 1
        ii = idx[idx < L]
        for j, t in enumerate(ii):
            base = mid[t - 1]
            if not np.isfinite(base):
                continue
            nxt = ii[j + 1] if j + 1 < len(ii) else L
            hi = min(m_max, nxt - t, L - t)
            if hi < 1:
                continue
            n_ev += 1
            r = (mid[t + 1:t + hi + 1] - base) / TICK
            ok = np.isfinite(r)
            s1[:hi][ok] += r[ok]; s2[:hi][ok] += r[ok] ** 2; cnt[:hi][ok] += 1
    if n_ev == 0:
        return None
    mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    se = np.sqrt(np.clip(var, 0, None) / np.maximum(cnt, 1))
    return mean, se, cnt, n_ev, n_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', required=True)
    ap.add_argument('--models', default='Historic,Heuristic,Propagator,Hawkes,CST,NMZI,'
                                        'Mamba3,GDN,S5_120M,S5,Mamba3_4k,S5_4k')
    ap.add_argument('--m_max', type=int, default=250)
    ap.add_argument('--n_files', type=int, default=300)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache = {'m_max': np.array([args.m_max])}
    for model in [m for m in args.models.split(',') if m]:
        side = os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy')
        r = model_response(side, args.m_max, args.n_files)
        if r is None:
            print(f'{model}: no data', flush=True); continue
        mean, se, cnt, n_ev, n_run = r
        cache[f'{model}_R'] = mean; cache[f'{model}_Rse'] = se; cache[f'{model}_cnt'] = cnt
        cache[f'{model}_n'] = np.array([n_ev, n_run])
        edge = mean[np.isfinite(mean)]
        print(f'{model:11s} events={n_ev:,} runs={n_run}  R(10)={mean[9]:+.3f}  '
              f'R(60)={mean[59]:+.3f}  R(edge)={edge[-1]:+.3f}', flush=True)
    out = os.path.join(args.out_dir, f'model_response_{args.stock}.npz')
    np.savez_compressed(out, **cache)
    print(f'MODEL_RESP_DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
