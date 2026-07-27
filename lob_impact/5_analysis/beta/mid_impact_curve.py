#!/usr/bin/env python3
"""
Mean mid-price impact curve  mean(buy + (-1)*sell)  vs insertion index k, overlaid per model.

At each aggressive order k we take the SIGNED relative mid move since just before the metaorder
  I_k = sign * (mid[aggr_k] - mid_ref) / mid_ref         (buy: sign=+1, sell: sign=-1)
so buy and sell point the same way; pooling buy+sell over all samples and averaging gives the
combined mean impact trajectory. One curve per model (Historic/Heuristic/CST/Mamba3), in bps.
(NO I>0 filter here — this is a mean, not the β log-log cloud.)

  python 5_analysis/beta/mid_impact_curve.py --grid <root> --stock EA \
         --models Historic,Heuristic,CST,Mamba3 --out mid_impact_EA.png
"""
import os, glob, csv, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENTINEL = 2147483647
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')

def _canon_palette():
    """Canonical palette from lob_impact/core/model_style.py — see the note there on why the
    per-script dicts drifted. Falls back to the local dict if that file is not reachable."""
    import os, importlib.util
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        p = os.path.join(d, 'core', 'model_style.py')
        if os.path.exists(p):
            sp = importlib.util.spec_from_file_location('_lob_model_style', p)
            m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
            return dict(m.COLORS)
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return {}

_LOCAL_COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'CST': '#27AE60',
          'Mamba3': '#2F5DA3', 'S5': '#E67E22', 'OW': '#6A1B9A', 'QR': '#00ACC1'}


COLORS = {**_LOCAL_COLORS, **_canon_palette()}
def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def _aggr_by_day(side_dir):
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def collect_signed(side_dir, sign, kmax):
    """rows of (k, I_signed) for every sample/insertion (no positivity filter)."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'), recursive=True))
    aggr_by_day = _aggr_by_day(side_dir)
    rows = []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        aggr = aggr_by_day.get(m.group(1))
        if aggr is None:
            continue
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask_p, bid_p = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask_p >= SENTINEL) | (bid_p >= SENTINEL) | (ask_p <= 0) | (bid_p <= 0)
        mid = (ask_p + bid_p) / 2.0
        mid[bad] = np.nan
        L = len(mid)
        idx = aggr[aggr < L]
        if len(idx) < 2 or idx[0] < 1:
            continue
        ref = mid[idx[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        for k, step in enumerate(idx):
            if k + 1 > kmax:
                break
            I = sign * (mid[step] - ref) / ref
            if np.isfinite(I):
                rows.append((k + 1, I))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,OW,CST,QR,Mamba3')
    ap.add_argument('--kmax', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    models = [m for m in args.models.split(',') if m]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for model in models:
        exp = f'{args.stock}-{model}-beta'
        rows = collect_signed(os.path.join(args.grid, exp, 'buy'), +1, args.kmax) + \
               collect_signed(os.path.join(args.grid, exp, 'sell'), -1, args.kmax)
        if not rows:
            print(f'{exp}: no data'); continue
        A = np.array(rows)
        K = A[:, 0].astype(int); I = A[:, 1] * 1e4  # bps
        ks = np.arange(1, args.kmax + 1)
        mean = np.array([I[K == k].mean() if (K == k).any() else np.nan for k in ks])
        sem = np.array([I[K == k].std() / max(1, np.sqrt((K == k).sum())) if (K == k).any() else np.nan for k in ks])
        c = COLORS.get(model, '#444')
        ax.plot(ks, mean, '-', color=c, lw=1.8, label=f'{model} (k=100: {mean[-1]:.1f} bps)')
        ax.fill_between(ks, mean - 1.96 * sem, mean + 1.96 * sem, color=c, alpha=0.12)
        print(f'{exp}: {len(rows)} points, mean impact @k=100 = {mean[-1]:.2f} bps')

    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('insertion k  (number of child orders)')
    ax.set_ylabel('mean signed mid impact  (bps)   buy + (−1)·sell')
    ax.set_title(f'{args.stock} — combined mid-price impact vs metaorder size, by model')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'mid_impact', f'mid_impact_{args.stock}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
