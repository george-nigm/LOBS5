#!/usr/bin/env python3
"""
Figure 8 v2: per-event response R(m), models vs the VALIDATED real anchor.

Replaces the old event_response figure after the anchor anatomy:
  - real anchor drawn to m=250 from the empresp_curve npz (saturation now visible;
    the old 131-point anchor cut the NVDA climb mid-rise and read "still rising");
  - model curves from the triangle numbers.json caches (visible-buy resp_mean,
    window capped at the next insertion) — no grid re-read;
  - right panel: TRADE-TIME view — real R at the k-th following execution
    (the only clock shared across stocks: 10% participation => next insertion
    ~ 10 executions), with each model's instant plateau drawn as a flat line
    (the models' defect IS that their response does not depend on the clock).

  python event_response_v2.py --stock NVDA \
      --triangle_glob 'results/triangle_NVDA_*_20260713-161454' \
      --anchor results/empresp_curve_NVDA/empresp_curve_NVDA.npz \
      --out results/empresp_curve_NVDA/event_response_v2_NVDA.png
"""
import os, glob, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_COLOR = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
               'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
               'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
               'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22'}
C_REAL = '#111111'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--triangle_glob', required=True)
    ap.add_argument('--anchor', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    A = np.load(args.anchor, allow_pickle=True)
    mg, R, Rse = A['mgrid'], A['R'], A['R_se']
    Rtr, Rtr_se = A['Rtr'], A['Rtr_se']

    fig, (ax, axt) = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=200,
                                  gridspec_kw={'width_ratios': [2.1, 1]})
    plateaus = {}
    for d in sorted(glob.glob(args.triangle_glob)):
        f = os.path.join(d, 'numbers.json')
        if not os.path.exists(f):
            continue
        nn = json.load(open(f))
        m = nn.get('model') or os.path.basename(d).split('_', 2)[2].rsplit('_', 1)[0]
        r = nn.get('visible-buy')
        if not r or 'resp_mean' not in r:
            continue
        rm = np.array(r['resp_mean'], float); rs = np.array(r['resp_se'], float)
        x = np.arange(len(rm)); ok = np.isfinite(rm)
        c = MODEL_COLOR.get(m, '#444444')
        ax.fill_between(x[ok], (rm - 2 * rs)[ok], (rm + 2 * rs)[ok], color=c, alpha=0.12, lw=0)
        ax.plot(x[ok], rm[ok], color=c, lw=1.9,
                label=f"{m} — {r['resp_n_events']:,} child events (n={r['n']})")
        tail = rm[ok][len(rm[ok]) // 2:]
        plateaus[m] = float(np.nanmean(tail))
    ax.plot(mg, R, color=C_REAL, lw=2.2, ls=(0, (4, 2)),
            label=f'Real {args.stock} anchor (validated, m to {int(mg[-1])})')
    ax.fill_between(mg, R - 2 * Rse, R + 2 * Rse, color=C_REAL, alpha=0.15, lw=0)
    ax.axhline(0, color='#bbbbbb', lw=0.8)
    ax.set_xlim(0, mg[-1] * 1.02)
    ax.set_xlabel('messages after the execution  m  (model windows capped at the next insertion)')
    ax.set_ylabel('mean mid response R(m), ticks')
    ax.set_title(f'{args.stock}: per-event response, models vs the validated real anchor',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=7.6)

    ks = np.arange(1, len(Rtr) + 1)
    axt.errorbar(ks, Rtr, yerr=2 * Rtr_se, fmt='o-', ms=4, color=C_REAL, lw=1.6,
                 capsize=2, label='real, trade time')
    for m, p in sorted(plateaus.items(), key=lambda kv: -kv[1]):
        axt.axhline(p, color=MODEL_COLOR.get(m, '#444444'), lw=1.4, ls=':',
                    alpha=0.9)
        axt.annotate(m, (ks[-1], p), fontsize=6.6, color=MODEL_COLOR.get(m, '#444444'),
                     va='bottom', ha='right')
    axt.axvline(10, color='#888888', lw=1.0, ls='--')
    axt.annotate('next insertion\n(10% participation)', (10, axt.get_ylim()[0]),
                 fontsize=7, color='#666666', ha='center', va='bottom')
    axt.axhline(0, color='#bbbbbb', lw=0.8)
    axt.set_xlabel('k-th execution after the event (trade time)')
    axt.set_title('trade-time: real builds (and relaxes);\nmodel plateaus are clock-independent',
                  fontsize=9.5, loc='left')
    axt.legend(fontsize=7.6, loc='upper left')
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    np.savez_compressed(os.path.splitext(args.out)[0] + '.npz',
                        mgrid=mg, R=R, R_se=Rse, Rtr=Rtr, Rtr_se=Rtr_se,
                        plateau_models=np.array(list(plateaus)),
                        plateau_vals=np.array(list(plateaus.values())))
    print('plateaus:', {k: round(v, 2) for k, v in plateaus.items()})
    print(f'EVRESP_V2_DONE -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
