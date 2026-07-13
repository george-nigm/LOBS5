#!/usr/bin/env python3
"""
Methodology-audit follow-ups that run from the master-curve npz CACHES (no grid access).

1. TRANSIENT exponent (Bacry et al 2015 reference: ~0.64 mean, ~0.80 short/fast, 'almost
   linear' at the fastest): slope of ln<I(v)> vs ln v on the build-up master curve,
   v in [0.05, 1]. The literature says the within-execution trajectory is a STEEPER object
   than the cross-metaorder law — this quantifies where each generator sits.
2. RELAXATION decay: on the relaxation-shape master curve past v=1, fit the decay of
   (I(v) - I_end_extrapolated) and report end/peak plus the tail slope — scored against
   the information-free target (relaxation toward ~0, slowly) rather than the informed 2/3.

  python transient_audit.py            # all three stocks from results/master_curve/*.npz
"""
import os
import numpy as np

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'master_curve')
STOCKS = ['EA', 'GOOG', 'NVDA']


def vgrid(z):
    for k in ('v', 'vgrid', 'grid', 'x'):
        if k in z.files:
            return np.asarray(z[k], float)
    return None


def main():
    out = {}
    for st in STOCKS:
        fb = os.path.join(B, f'master_curve_{st}_beta_v2gated.npz')
        fr = os.path.join(B, f'master_curve_{st}_relaxation_v2gated.npz')
        if not os.path.exists(fb):
            print(f'{st}: no build-up cache'); continue
        zb = np.load(fb, allow_pickle=True)
        models = sorted({k[:-7] for k in zb.files if k.endswith('_master')})
        vb = vgrid(zb)
        zr = np.load(fr, allow_pickle=True) if os.path.exists(fr) else None
        vr = vgrid(zr) if zr is not None else None
        print(f'\n=== {st} (v grid: build {"cached" if vb is not None else "implied"}, '
              f'relax {"cached" if vr is not None else "implied"}) ===')
        for m in models:
            cur = np.asarray(zb[f'{m}_master'], float)
            sig = bool(np.atleast_1d(zb.get(f'{m}_sig', np.array(True)))[0])
            v = vb if vb is not None and len(vb) == len(cur) else np.linspace(0, 1, len(cur))
            ok = np.isfinite(cur) & (v >= 0.05) & (v <= 1.0) & (cur > 0)
            te = np.nan
            if ok.sum() >= 10:
                te = float(np.polyfit(np.log(v[ok]), np.log(cur[ok]), 1)[0])
            line = f'{m:10s} transient_exp={te:+.3f}' + ('' if sig else ' (GATED)')
            if zr is not None and f'{m}_master' in zr.files:
                cr = np.asarray(zr[f'{m}_master'], float)
                w = vr if vr is not None and len(vr) == len(cr) else np.linspace(0, 2, len(cr))
                post = np.isfinite(cr) & (w > 1.0)
                if post.sum() >= 8:
                    pk = np.nanmax(cr[np.isfinite(cr) & (w <= 1.05)])
                    endv = float(np.nanmean(cr[post][-max(3, post.sum() // 10):]))
                    # tail slope in units of peak per unit v — sign tells building vs decaying
                    seg_w, seg_c = w[post], cr[post]
                    sl = float(np.polyfit(seg_w, seg_c, 1)[0]) if len(seg_w) >= 4 else np.nan
                    line += (f' | relax: end/peak={endv / pk:+.2f} tail_slope={sl:+.3f}/v '
                             f'(target: decay toward 0 for information-free flow)')
            print(line, flush=True)
            out[f'{st}_{m}'] = te
    np.savez(os.path.join(B, 'transient_audit.npz'), **{k: v for k, v in out.items()})
    print('\nTRANSIENT_AUDIT_DONE (Bacry 2015 refs: mean 0.64, short/fast 0.80, cross-metaorder ~0.45)')


if __name__ == '__main__':
    main()
