#!/usr/bin/env python3
"""
Slide-ready cross-model β(k) figure: two panels side by side per stock, σ=1, free intercept.
  LEFT  : ≤k  CUMULATIVE β(k)  (pool k'≤k) — the smooth "paper" curve
  RIGHT : ==k EXACT     β(k)  (cross-section at exactly k) — the noisy drill-down
One coloured curve per model + the 0.5 √-law line + final-β in the legend.

  python beta_cumul_vs_exact.py --grid <root> --daily <csv> --stock EA \
         --models Historic,Heuristic,CST,Mamba3 --out beta_cumul_vs_exact_EA.png
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beta_grid import collect
from vol_estimators import daily_sigmas

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'CST': '#27AE60',
          'Mamba3': '#2F5DA3', 'S5': '#E67E22'}
MIN_PTS = 8


def fit_int(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < MIN_PTS or (x.max() - x.min()) < 1e-6:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,CST,Mamba3')
    ap.add_argument('--kmax', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]

    fig, (axc, axe) = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for model in models:
        exp = f'{args.stock}-{model}-beta'
        raw = collect(os.path.join(args.grid, exp, 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, exp, 'sell'), args.stock, -1)
        if not raw:
            print(f'{exp}: no data'); continue
        Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
        days = [r[2] for r in raw]; kc = np.array([r[3] for r in raw], int) + 1
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        x = np.log(Q / V); y = np.log(I)            # σ=1
        ok = np.isfinite(x) & np.isfinite(y); x, y, kc = x[ok], y[ok], kc[ok]
        kmax = int(min(args.kmax, kc.max()))
        ks = np.arange(1, kmax + 1)
        bc = np.array([fit_int(x[kc <= k], y[kc <= k]) for k in ks])
        be = np.array([fit_int(x[kc == k], y[kc == k]) for k in ks])
        c = COLORS.get(model, '#444')
        fc = bc[np.isfinite(bc)]; fe = be[np.isfinite(be)]
        axc.plot(ks, bc, '-', color=c, lw=1.9, label=f'{model} ({fc[-1]:.2f})' if fc.size else model)
        axe.plot(ks, be, '-', color=c, lw=1.3, alpha=0.85,
                 label=f'{model} (mean {np.nanmean(be):.2f})')
        print(f'{exp}: cumul β(100)={fc[-1] if fc.size else np.nan:.3f}  exact mean β={np.nanmean(be):.3f}')

    for ax in (axc, axe):
        ax.axhline(0.5, color='red', ls='--', lw=1)
        ax.set_ylim(-0.2, 1.0); ax.set_xlabel('insertion k'); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    axc.set_ylabel(r'$\beta_{intercept}$  (σ=1)')
    axc.set_title(f'{args.stock} — ≤k  CUMULATIVE β(k)  (smooth)')
    axe.set_title(f'{args.stock} — ==k  EXACT β(k)  (drill-down, noisy)')
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'beta_cumul_vs_exact', f'beta_cumul_vs_exact_{args.stock}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
