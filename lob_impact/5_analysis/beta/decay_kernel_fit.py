#!/usr/bin/env python3
"""
Stage-2 kernel fit, computed TODAY (no dashes): for each model's relaxation
master curve, fit the post-execution segment (z = v-1 >= z0) with three
candidates and compare by AIC:
  M0 flat        I(z) = c                      (k=1)
  M1 power law   I(z) = I_inf + A * z^(-gamma) (k=3; grid gamma, OLS I_inf,A)
  M2 exponential I(z) = I_inf + B * e^(-z/tau) (k=3; grid tau,  OLS I_inf,B)
Verdict: decay requires the best decay model to beat flat by dAIC <= -2 AND
A (or B) > 0; a significantly positive linear slope reads 'rising'.
Pure cache re-render (master_curve_<ST>_relaxation_v2gated.npz).

  python decay_kernel_fit.py --stock NVDA
"""
import os, argparse
import numpy as np

GGRID = np.arange(0.05, 1.51, 0.05)
TGRID = np.arange(0.1, 5.01, 0.1)


def aic(rss, n, k):
    return n * np.log(max(rss, 1e-12) / n) + 2 * k


def fit_two_param(z, y, regressor):
    X = np.column_stack([np.ones_like(z), regressor])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ coef) ** 2))
    return coef, rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--z0', type=float, default=0.2)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    z = np.load(os.path.join(here, 'results', 'master_curve',
                             f'master_curve_{args.stock}_relaxation_v2gated.npz'),
                allow_pickle=True)
    v = z['vgrid']
    models = sorted({k[:-len('_master')] for k in z.files if k.endswith('_master')})
    lines = ['| model | sig | slope/z | gamma_hat | dAIC(PL-flat) | dAIC(EXP-flat) | verdict |',
             '|---|---|---|---|---|---|---|']
    out = {}
    for m in models:
        y_full = z[f'{m}_master']
        sig = bool(z[f'{m}_sig'])
        sel = v - 1.0 >= args.z0
        zz, y = (v[sel] - 1.0), y_full[sel]
        ok = np.isfinite(y)
        zz, y = zz[ok], y[ok]
        if len(y) < 10:
            continue
        n = len(y)
        # M0 flat
        rss0 = float(np.sum((y - y.mean()) ** 2)); a0 = aic(rss0, n, 1)
        # linear trend (diagnostic)
        slope = float(np.polyfit(zz, y, 1)[0])
        # M1 power law
        best1 = (np.inf, np.nan, None)
        for g in GGRID:
            coef, rss = fit_two_param(zz, y, zz ** (-g))
            if rss < best1[0]:
                best1 = (rss, float(g), coef)
        a1 = aic(best1[0], n, 3)
        # M2 exponential
        best2 = (np.inf, np.nan, None)
        for t in TGRID:
            coef, rss = fit_two_param(zz, y, np.exp(-zz / t))
            if rss < best2[0]:
                best2 = (rss, float(t), coef)
        a2 = aic(best2[0], n, 3)
        d1, d2 = a1 - a0, a2 - a0
        A_pl = best1[2][1]; B_ex = best2[2][1]
        # material total move of the FITTED curve over the window, in peak units
        # (master is normalised to I(1)=1). Naive AIC on autocorrelated residuals
        # is overconfident, so a decay verdict additionally requires a material
        # decline and a kernel parameter away from its grid edge.
        rise_total = slope * (zz[-1] - zz[0])
        if d1 <= d2:
            c0, cA = best1[2]
            decline = float(cA * (zz[0] ** (-best1[1]) - zz[-1] ** (-best1[1])))
            edge = best1[1] >= GGRID[-1] - 1e-9 or best1[1] <= GGRID[0] + 1e-9
        else:
            c0, cB = best2[2]
            decline = float(cB * (np.exp(-zz[0] / best2[1]) - np.exp(-zz[-1] / best2[1])))
            edge = best2[1] >= TGRID[-1] - 1e-9
        if rise_total > 0.10:
            verdict = f'RISING (+{slope:.3f}/z)'
        elif min(d1, d2) <= -10 and decline >= 0.15 and not edge:
            form = 'PL' if d1 <= d2 else 'EXP'
            verdict = (f'DECAYS (PL, gamma={best1[1]:.2f}, drop {decline:.2f})' if form == 'PL'
                       else f'DECAYS (EXP, tau={best2[1]:.2f}, drop {decline:.2f})')
        elif decline >= 0.08:
            verdict = f'mild drift down ({slope:+.3f}/z, drop {decline:.2f} — immaterial vs gamma=0.5 target)'
        else:
            verdict = f'FLAT ({slope:+.3f}/z)'
        lines.append(f'| {m} | {"y" if sig else "n"} | {slope:+.3f} | {best1[1]:.2f} | {d1:+.1f} | {d2:+.1f} | {verdict} |')
        out[m] = dict(slope=slope, gamma=best1[1], dAIC_pl=d1, dAIC_exp=d2, verdict=verdict)
    txt = '\n'.join(lines)
    print(txt)
    dst = os.path.join(here, 'results', 'master_curve', f'decay_kernel_fit_{args.stock}.md')
    with open(dst, 'w') as f:
        f.write(txt + '\n')
    np.savez_compressed(dst.replace('.md', '.npz'),
                        **{f'{m}_{k}': val for m, d in out.items() for k, val in d.items()
                           if not isinstance(val, str)})
    print('KERNEL_FIT ->', dst)


if __name__ == '__main__':
    main()
