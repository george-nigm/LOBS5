#!/usr/bin/env python3
"""Binning, drawn step by step (cache-only, bias_exhibit cloud + bin means):
(1) the signed cloud with bin boundaries; one bin highlighted, its points
    averaged into a single signed mean;
(2) all bin means +-1.96 SE on a LINEAR y axis -- averaging first lets the
    noise cancel while the y-axis still admits negatives;
(3) only now take logs: ln(mean) vs ln q is LINEAR IN THE PARAMETERS
    (ln I = alpha + delta ln q), so plain OLS gives both numbers in closed
    form -- no scan. The price of logs is positivity, paid by binning first.
  python fig_binning_steps.py --stock NVDA --model Mamba3 [--copy_to <dir>]"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True); ap.add_argument('--model', default='Mamba3')
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    z = np.load(os.path.join(here, 'results', 'bias_exhibit', f'bias_exhibit_{args.stock}.npz'))
    m = args.model
    x = z[f'{m}_parkinson_cloud_x']; y = z[f'{m}_parkinson_cloud_y']
    xc = z[f'{m}_parkinson_xc']; ym = z[f'{m}_parkinson_ym']; yse = z[f'{m}_parkinson_yse']
    edges = np.concatenate([[x.min()], (xc[1:] + xc[:-1]) / 2, [x.max()]])
    plt.rcParams.update({'font.size': 11.5})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=150)
    # (1) cloud + bins, one highlighted
    ax = axes[0]
    ax.scatter(x, y, s=4, alpha=0.15, color='#9a9a9a', lw=0)
    for e in edges: ax.axvline(e, color='#cccccc', lw=0.5)
    hb = len(xc) * 2 // 3
    sel = (x >= edges[hb]) & (x < edges[hb + 1])
    ax.scatter(x[sel], y[sel], s=6, alpha=0.5, color='#2F5DA3', lw=0)
    ax.plot(xc[hb], ym[hb], 'o', color='#C0392B', ms=11, zorder=5)
    ax.annotate('one bin: average its points\n(noise cancels; the mean stays signed)',
                xy=(xc[hb], ym[hb]), xytext=(x.min() + 0.4, np.nanpercentile(y, 99) * 0.75),
                fontsize=10, color='#C0392B', arrowprops=dict(arrowstyle='->', color='#C0392B'))
    ax.axhline(0, color='#bbbbbb', lw=0.8)
    ax.set_title('STEP 1 — equal-count bins over $\\ln(Q/V)$', fontsize=11.5)
    ax.set_xlabel(r'$\ln(Q/V)$'); ax.set_ylabel(r'$I/\sigma$ (signed)')
    ax.set_ylim(np.nanpercentile(y, 0.5), np.nanpercentile(y, 99.5))
    # (2) bin means, linear y
    ax = axes[1]
    ax.errorbar(xc, ym, yerr=1.96 * yse, fmt='o-', ms=5, color='#2F5DA3', lw=1.4, capsize=2)
    ax.axhline(0, color='#bbbbbb', lw=0.8)
    ax.set_title('STEP 2 — the 18 bin means $\\pm 1.96$ SE\n(linear $y$: negatives still allowed)', fontsize=11.5)
    ax.set_xlabel(r'$\ln(Q/V)$')
    # (3) logs + OLS
    ax = axes[2]
    pos = ym > 0
    lx, ly = xc[pos], np.log(ym[pos])
    d, al = np.polyfit(lx, ly, 1)
    ax.plot(lx, ly, 'o', ms=6, color='#2F5DA3')
    xx = np.linspace(lx.min(), lx.max(), 50)
    ax.plot(xx, al + d * xx, color='#C0392B', lw=2.0, label=rf'OLS: $\ln I = \alpha + \delta \ln q$,  $\delta = {d:.2f}$')
    ax.set_title('STEP 3 — only now take logs:\nlinear in $(\\alpha, \\delta)$ $\\Rightarrow$ closed-form OLS, no scan', fontsize=11.5)
    ax.set_xlabel(r'$\ln(Q/V)$'); ax.set_ylabel(r'$\ln(\mathrm{bin\ mean})$')
    ax.legend(fontsize=9.5)
    fig.suptitle(f'{args.stock} / {m}: the binning convention, step by step --- logs are legal only after '
                 f'averaging has made the means positive', fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(here, 'results', 'bias_exhibit', f'binning_steps_{args.stock}.png')
    fig.savefig(out, bbox_inches='tight')
    print('BINNING ->', out)
    if args.copy_to:
        import shutil; shutil.copy(out, args.copy_to); print('copied')

if __name__ == '__main__':
    main()
