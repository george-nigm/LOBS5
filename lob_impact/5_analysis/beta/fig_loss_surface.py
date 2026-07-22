#!/usr/bin/env python3
"""
The (a, delta) loss surface, drawn: heatmap of log10 S(a, delta) on the cached
4k-point clouds. Two panels: an identified law (Mamba3: a diagonal banana
valley with one bottom) vs no law (Historic: a flat basin along a ~ 0).
Overlays: the profiled path a*(delta) (closed form -- the 'valley floor' the
estimator walks instead of scanning a), the global minimum, and one dashed
iso-line of constant fitted value AT THE DATA EDGE (a * q_ref^delta = const):
it runs parallel to the valley -- the geometric reason the data-edge value
(and hence Y) is pinned while a and delta individually trade off.

  python fig_loss_surface.py --stock NVDA [--copy_to <dir>]
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--models', default='Mamba3,Historic')
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    zb = np.load(os.path.join(here, 'results', 'bias_exhibit', f'bias_exhibit_{args.stock}.npz'))
    q_ref = float(np.exp(zb['Mamba3_parkinson_xc'][-1]))
    dg = np.linspace(0.05, 1.5, 140)

    models = args.models.split(',')
    fig, axes = plt.subplots(1, len(models), figsize=(7.2 * len(models), 5.6), dpi=160)
    for ax, m in zip(np.atleast_1d(axes), models):
        x = zb[f'{m}_parkinson_cloud_x']; y = zb[f'{m}_parkinson_cloud_y']
        ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
        # profiled closed-form a*(delta) and its scale, to set the a-grid
        a_star = np.array([np.dot(y, np.exp(d * x)) / np.dot(np.exp(d * x), np.exp(d * x)) for d in dg])
        j05 = int(np.argmin(np.abs(dg - 0.5)))
        s05 = abs(a_star[j05]) + 1e-12
        if m == 'Historic':
            ag = np.linspace(-6.0 * s05, 6.0 * s05, 160)
        else:
            ref = np.nanmedian(np.abs(a_star)) + 1e-12
            ag = np.sign(np.nanmedian(a_star)) * np.logspace(np.log10(ref / 30), np.log10(ref * 30), 160)
        S = np.empty((len(ag), len(dg)))
        for j, d in enumerate(dg):
            t = np.exp(d * x)
            # S(a) is a parabola in a: expand once, evaluate for all a at once
            yy = float(np.dot(y, y)); yt = float(np.dot(y, t)); tt = float(np.dot(t, t))
            S[:, j] = yy - 2 * ag * yt + ag ** 2 * tt
        im = ax.pcolormesh(dg, ag, np.log10(S / S.min()), cmap='viridis', shading='auto', vmax=3.0)
        ax.set_ylim(ag.min() if m == 'Historic' else abs(ag).min(), ag.max())
        plt.colorbar(im, ax=ax, label=r'$\log_{10} S(a,\delta)/S_{\min}$')
        ax.plot(dg, a_star, color='white', lw=2.2, label=r'profiled $a^{*}(\delta)$ (closed form)')
        jmin = int(np.argmin([S[np.argmin(np.abs(ag - a_star[j])), j] for j in range(len(dg))]))
        ax.plot(dg[jmin], a_star[jmin], 'r*', ms=16, label='global minimum')
        # iso-line of constant value at the data edge, through the minimum
        C = a_star[jmin] * q_ref ** dg[jmin]
        iso = C * q_ref ** (-dg)
        ax.plot(dg, iso, 'w--', lw=1.4, alpha=0.85, label=r'iso-line: $a\,q_{\rm ref}^{\delta}$ = const (data-edge value)')
        if m != 'Historic':
            ax.set_yscale('log')
        ax.set_xlabel(r'$\delta$'); ax.set_ylabel('$a$')
        ax.set_title(f'{m}: ' + ('one bottom — identified' if m != 'Historic' else 'flat basin along $a\\approx 0$ — no law'),
                     fontsize=11.5)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    fig.suptitle(f'{args.stock}: the loss surface behind the fit — the estimator scans $\\delta$ and '
                 f'jumps to the valley floor $a^{{*}}(\\delta)$ analytically', fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(here, 'results', 'bias_exhibit', f'loss_surface_{args.stock}.png')
    fig.savefig(out, bbox_inches='tight')
    print('LOSS_SURFACE ->', out)
    if args.copy_to:
        import shutil; shutil.copy(out, args.copy_to); print('copied ->', args.copy_to)

if __name__ == '__main__':
    main()
