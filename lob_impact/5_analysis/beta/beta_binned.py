#!/usr/bin/env python3
"""
LITERATURE-STANDARD market-impact β (square-root law) via BINNING + CONDITIONAL MEAN — the way
Tóth+ 2011, Almgren+ 2005, Bouchaud-Bonart-Donier-Gould 2018 estimate it.

NO I>0 selection (impact is SIGNED, negatives kept) and NO outlier trim.  Instead:
  1. pool every (sample, insertion) point  (Q_k cumulative metaorder volume, I signed impact);
  2. bin by  log(Q/V)  into equal-count (quantile) bins;
  3. per bin take the CONDITIONAL MEAN  ⟨I/σ⟩  (signed — averaging negatives IN keeps it unbiased,
     and the bin mean stays positive for a real metaorder, so log of it is well defined);
  4. fit  log⟨I/σ⟩ = α + δ·log(Q/V)  over the bin means  →  δ  (= our β; √-law ⇔ δ=0.5).

This removes BOTH problems of the per-point log-OLS: (a) the I>0 selection bias (worst at small k),
(b) the heavy-tail noise that made ==k jitter — bin means average it out.

Two figures per stock (× σ=parkinson and σ=1):
  A. beta_binned_law_<stock>.png/html  — ⟨I/σ⟩ vs Q/V log-log bin-means + δ fit, all models, 0.5 slope.
  B. beta_binned_vsk_<stock>.png/html  — δ(k) under ≤k / ==k / ≥k poolings, each computed by the SAME
     bin+conditional-mean recipe (the methodologically-correct version of beta_vs_k_3views).

  python beta_binned.py --grid <root> --stock EA --daily <daily_h_l_all.csv> \
         --models Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k [--method none] [--nbins 18]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beta_grid import collect
from vol_estimators import daily_sigmas

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Hawkes': '#D4AC0D',
          'CST': '#27AE60', 'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'S5_4k': '#8E44AD', 'OW': '#6A1B9A', 'QR': '#00ACC1'}
MIN_BINS = 5


def bin_means(x, y, nbins):
    """Equal-count (quantile) bins of x. Return per-bin (xc=mean log Q/V, ym=⟨I/σ⟩ signed, n)."""
    n = x.size
    if n < nbins * 3:
        nbins = max(MIN_BINS, n // 3)
    order = np.argsort(x)
    xc, ym, nb = [], [], []
    for chunk in np.array_split(order, nbins):
        if chunk.size == 0:
            continue
        xc.append(float(np.mean(x[chunk]))); ym.append(float(np.mean(y[chunk]))); nb.append(int(chunk.size))
    return np.array(xc), np.array(ym), np.array(nb)


def delta_from_bins(x, y, nbins):
    """δ = OLS slope of log⟨I/σ⟩ vs log(Q/V) over the (positive) bin means."""
    xc, ym, nb = bin_means(x, y, nbins)
    pos = ym > 0
    if pos.sum() < MIN_BINS or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan, xc, ym, nb
    d = float(np.polyfit(xc[pos], np.log(ym[pos]), 1)[0])
    return d, xc, ym, nb


def load(grid, stock, model, sig, method):
    raw = collect(os.path.join(grid, f'{stock}-{model}-beta', 'buy'), stock, +1) + \
          collect(os.path.join(grid, f'{stock}-{model}-beta', 'sell'), stock, -1)
    if not raw:
        return None
    Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
    days = [r[2] for r in raw]; kc = np.array([r[3] for r in raw], int) + 1
    V = np.array([sig.get((stock, d), {}).get('V', np.nan) for d in days], float)
    x = np.log(Q / V)
    if method == 'none':
        y = I                                   # σ = 1
    else:
        sg = np.array([sig.get((stock, d), {}).get(method, np.nan) for d in days], float)
        y = I / sg
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], kc[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,OW,Hawkes,QR,CST,Mamba3,Mamba3_4k')
    ap.add_argument('--method', default='parkinson', help='σ estimator, or "none" for σ=1')
    ap.add_argument('--nbins', type=int, default=18)
    ap.add_argument('--kmax', type=int, default=100)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]
    sigtag = 'σ=1' if args.method == 'none' else f'σ={args.method}'
    stag = 'sigma1' if args.method == 'none' else args.method
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'beta_binned')
    os.makedirs(outdir, exist_ok=True)

    data = {}
    for m in models:
        d = load(args.grid, args.stock, m, sig, args.method)
        if d is None:
            print(f'{m}: no data'); continue
        data[m] = d
        print(f'{m}: {d[0].size} signed points (NO I>0 filter)')

    # ---------- FIGURE A: binned √-law (full pool) ----------
    figA, ax = plt.subplots(figsize=(8.5, 6.5))
    print(f'\n=== {args.stock} binned √-law δ ({sigtag}, {args.nbins} quantile bins, conditional mean, signed) ===')
    for m, (x, y, kc) in data.items():
        d, xc, ym, nb = delta_from_bins(x, y, args.nbins)
        c = COLORS.get(m, '#444'); pos = ym > 0
        ax.plot(xc[pos], ym[pos], 'o', color=c, ms=6, label=f'{m}  δ={d:.3f}  (N={x.size:,})')
        ax.plot(xc[~pos], -ym[~pos], 'x', color=c, ms=5, alpha=0.4)        # negative bin-means (shown reflected, faint)
        if np.isfinite(d):
            xs = np.array([xc[pos].min(), xc[pos].max()])
            a = np.polyfit(xc[pos], np.log(ym[pos]), 1)[1]
            ax.plot(xs, np.exp(a + d * xs), '-', color=c, lw=1.6)
        print(f'  {m:10s}: δ={d:.3f}  (bins={nb.size}, N={x.size})')
    # √-law reference slope 0.5 anchored at the cloud
    allx = np.concatenate([d_[0] for d_ in data.values()]); allmid = np.median(allx)
    ax.set_yscale('log'); ax.set_xlabel('log(Q / V)   (participation)')
    ax.set_ylabel(f'⟨I/σ⟩  conditional mean   ({sigtag})')
    ax.set_title(f'{args.stock} — market-impact √-law (binned, conditional mean, NO I>0 filter)\n'
                 f'δ = slope of log⟨I/σ⟩ vs log(Q/V);  √-law ⇔ δ=0.5')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(True, which='both', alpha=0.25)
    outA = os.path.join(outdir, f'beta_binned_law_{args.stock}_{stag}.png')
    figA.tight_layout(); figA.savefig(outA, dpi=150); plt.close(figA)
    print(f'saved -> {outA}')

    # ---------- FIGURE B: δ(k) under 3 poolings, binned ----------
    ks = np.arange(1, args.kmax + 1)
    figB, axes = plt.subplots(1, 3, figsize=(20, 5.4), sharey=True)
    for axp, (key, title) in zip(axes, [('le', '≤k CUMULATIVE'), ('eq', '==k EXACT'), ('ge', '≥k REVERSE-CUMUL')]):
        for m, (x, y, kc) in data.items():
            curve = []
            for k in ks:
                mask = {'le': kc <= k, 'eq': kc == k, 'ge': kc >= k}[key]
                d, *_ = delta_from_bins(x[mask], y[mask], args.nbins)
                curve.append(d)
            c = COLORS.get(m, '#444')
            rep = np.nanmean(curve) if key == 'eq' else (curve[-1] if key == 'le' else curve[0])
            axp.plot(ks, curve, '-', color=c, lw=1.6, label=f'{m} ({rep:.2f})')
        axp.axhline(0.5, color='red', ls='--', lw=1); axp.set_ylim(-0.2, 1.0)
        axp.set_xlabel('insertion k'); axp.grid(True, alpha=0.3)
        axp.set_title(f'{args.stock} — {title}  [binned δ, conditional mean]', fontsize=10)
        axp.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel(f'δ  (binned, {sigtag})')
    outB = os.path.join(outdir, f'beta_binned_vsk_{args.stock}_{stag}.png')
    figB.tight_layout(); figB.savefig(outB, dpi=150); plt.close(figB)
    print(f'saved -> {outB}')


if __name__ == '__main__':
    main()
