#!/usr/bin/env python3
"""
Figure-6 transparency exhibit: the EXACT k-clock curve that goes into the paper
(mean signed I(k), buy + flipped sell), drawn on top of ALL its raw samples.

One panel per model: grey spaghetti = every rollout's per-sample I(k) (buy signed +,
sell flipped so both push up under real impact), coloured curve = the plain mean
+-2 SE — identical to the midprice_traj_<STOCK>_beta.png / Figure 6 curve.
Legends carry the sample counts (buy / sell / total).

Needs the npz produced by mid_trajectory.py --dump_samples (keys <model>_K,
<model>_K_side, <model>_K_day). No grid access — cache only.

  python fig6_spaghetti.py --npz results/mid_impact/mid_trajectory_GOOG_beta.npz --stock GOOG
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5': '#E67E22', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'OW': '#6A1B9A', 'QR': '#00ACC1'}
MAX_SPAGHETTI = 400   # per panel; a uniform random subset if more (seeded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--stock', default='GOOG')
    ap.add_argument('--models', default='OW,QR,')
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    models = ([m for m in args.models.split(',') if m] or
              sorted({k[:-2] for k in z.files if k.endswith('_K')}))
    models = [m for m in models if f'{m}_K' in z.files]
    assert models, 'npz has no <model>_K keys — rerun mid_trajectory.py --dump_samples'
    rng = np.random.default_rng(42)

    ncol = min(4, len(models)); nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow),
                             squeeze=False, sharex=True)
    for i, m in enumerate(models):
        ax = axes[i // ncol][i % ncol]
        K = z[f'{m}_K']; side = z[f'{m}_K_side']
        km, kse = z[f'{m}_k_mean'], z[f'{m}_k_se']
        ks = np.arange(K.shape[1])
        sub = rng.choice(len(K), size=min(MAX_SPAGHETTI, len(K)), replace=False)
        for r in sub:
            ax.plot(ks, K[r], color='#999999', lw=0.3, alpha=0.12, zorder=1)
        c = COLORS.get(m, '#333333')
        ax.fill_between(ks, km - 1.96 * kse, km + 1.96 * kse, color=c, alpha=0.25, zorder=3, lw=0)
        ax.plot(ks, km, color=c, lw=2.0, zorder=4)
        lo, hi = np.nanpercentile(K, [1, 99])
        ax.set_ylim(lo, hi)
        ax.axhline(0, color='#bbbbbb', lw=0.8)
        nb, nsl = int((side > 0).sum()), int((side < 0).sum())
        ax.set_title(f'{m} — end {km[-1]:+.1f} bps  (n={len(K)}: {nb} buy / {nsl} sell)',
                     fontsize=9, color=c)
        if i % ncol == 0:
            ax.set_ylabel('signed impact (bps)')
        if i // ncol == nrow - 1:
            ax.set_xlabel('k (children executed)')
        print(f'{m:10s} n={len(K)} ({nb}b/{nsl}s)  mean_end={km[-1]:+.2f}  '
              f'sample_end p1/p50/p99 = {lo:+.1f}/{np.nanpercentile(K[:, -1], 50):+.1f}/{hi:+.1f}',
              flush=True)
    for j in range(len(models), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'{args.stock}: the Figure-6 curve on top of ALL its samples '
                 f'(grey = individual rollouts, buy signed / sell flipped; '
                 f'spaghetti subset {MAX_SPAGHETTI}/panel, mean over ALL)', y=1.0)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(args.npz), f'fig6_spaghetti_{args.stock}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'FIG6_SPAGHETTI_DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
