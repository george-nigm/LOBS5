#!/usr/bin/env python3
"""
The other half of the law: AMPLITUDE analysis + exact-slice sample counts.
Pure cache re-render (sigma-grid npz + mid_trajectory npz), no grid access.

Panel 1  a_L2(k), cumulative <=k, sigma=parkinson (log y): for a true power law the
         amplitude must be CONSTANT in k — drift in a(k) means the power form bends.
         Legend carries the max/min ratio over k in [20,100] as a constancy score.
Panel 2  implied Y(k) evaluated AT THE TOP-DECADE EDGE OF THE DATA (q_ref = the
         top quantile-bin centre of the actual Q/V cloud, from the bias-exhibit
         cache — NOT an out-of-sample extrapolation), RAW-RANGE convention
         (a_park / 1.665): Y_impl = a * q_ref^(delta-0.5).
         Literature target Y ~ 0.5 (band 0.2-1.0) drawn for comparison.
Panel 3  n(k) finite points on the exact =k slice per model (from the per-sample
         K matrices) — separates "hidden by ylim/overlap" from "dropped by MIN_PTS"
         in the =k sigma-grid panels.

  python amplitude_stability.py --stock NVDA
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E', 'OW': '#6A1B9A', 'QR': '#00ACC1'}
PARK2RANGE = np.sqrt(4 * np.log(2.0))      # 1.665: sigma_range = PARK2RANGE * sigma_parkinson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    zg = np.load(os.path.join(here, 'results', 'beta_sigma_grid',
                              f'beta_sigma_grid_{args.stock}_le.npz'))
    zk = np.load(os.path.join(here, 'results', 'mid_impact',
                              f'mid_trajectory_{args.stock}_beta.npz'), allow_pickle=True)
    zb = np.load(os.path.join(here, 'results', 'bias_exhibit',
                              f'bias_exhibit_{args.stock}.npz'))
    ref_model = 'Mamba3' if 'Mamba3_parkinson_xc' in zb.files else \
        sorted(k for k in zb.files if k.endswith('_parkinson_xc'))[0][:-len('_parkinson_xc')]
    q_ref = float(np.exp(zb[f'{ref_model}_parkinson_xc'][-1]))   # top-bin centre of the DATA
    ks = zg['ks']
    models = sorted({k.split('_parkinson_')[0] for k in zg.files if '_parkinson_dl2' in k})

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    sel = (ks >= 20)
    for m in models:
        a = np.abs(zg[f'{m}_parkinson_al2'])
        c = COLORS.get(m, '#444444')
        with np.errstate(all='ignore'):
            ratio = np.nanmax(a[sel]) / max(np.nanmin(a[sel]), 1e-300)
        ax.plot(ks, a, color=c, lw=1.6,
                label=f'{m} (max/min {ratio:.1f}x)' if np.isfinite(ratio) else m)
    ax.set_yscale('log')
    ax.set_title(r'$a_{L_2}(k)$, $\leq k$, $\sigma$=Parkinson' + '\n(power law $\\Rightarrow$ constant in $k$)',
                 fontsize=10)
    ax.set_xlabel('$k$'); ax.set_ylabel('|amplitude|')
    ax.legend(fontsize=6.2, ncol=1, loc='lower right')

    ax = axes[1]
    for m in models:
        a = np.abs(zg[f'{m}_parkinson_al2']) / PARK2RANGE     # raw-range convention
        d = zg[f'{m}_parkinson_dl2']
        Y = a * q_ref ** (d - 0.5)
        ax.plot(ks, Y, color=COLORS.get(m, '#444444'), lw=1.6)
    ax.axhspan(0.2, 1.0, color='#27AE60', alpha=0.10, lw=0)
    ax.axhline(0.5, color='#27AE60', lw=1.2, ls='--')
    ax.set_yscale('log')
    ax.set_title(f'implied $Y$ at the data edge $Q/V = {q_ref:.1e}$, raw-range convention\n'
                 '(green: empirical $Y \\approx 0.5$, band $0.2$--$1$)', fontsize=10)
    ax.set_xlabel('$k$'); ax.set_ylabel('$Y_{\\mathrm{impl}}$')

    ax = axes[2]
    for m in models:
        if f'{m}_K' not in zk.files:
            continue
        K = zk[f'{m}_K']
        kk = np.arange(1, K.shape[1])
        n = np.isfinite(K[:, 1:]).sum(0)
        ax.plot(kk, n, color=COLORS.get(m, '#444444'), lw=1.4)
    ax.axhline(200, color='#555555', lw=1.0, ls='--')
    ax.set_title('exact $=k$ slice: finite points $n(k)$ per model\n(dashed: MIN\\_PTS $=200$ fit gate)',
                 fontsize=10)
    ax.set_xlabel('$k$'); ax.set_ylabel('$n(k)$')
    ax.set_yscale('log')
    fig.suptitle(f'{args.stock}: amplitude stability, literature-anchored level, and exact-slice counts',
                 y=1.02)
    fig.tight_layout()
    out = os.path.join(here, 'results', 'beta_sigma_grid', f'amplitude_stability_{args.stock}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    np.savez_compressed(out.replace('.png', '.npz'), ks=ks)
    print(f'AMP_STAB_DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
