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
    ~ 10 executions). When the model npz carries <m>_Rtr keys (trade-time model
    curves computed by model_response_curve.py) the ACTUAL model curves are
    drawn — short by construction (window capped at the next insertion) and
    flat, visible to the eye rather than asserted; without those keys it falls
    back to the old plateau hlines.

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


NEURAL = ['S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']
# non-neural models are drawn too (they were silently absent from Fig. 7: only NEURAL was
# iterated). Canonical order: replay -> mechanical/parametric by complexity -> neural.
BASELINES = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--triangle_glob', default=None)
    ap.add_argument('--model_npz', default=None,
                    help='model_response_<ST>.npz from model_response_curve.py (250-msg, even axis)')
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
    if args.model_npz:
        M = np.load(args.model_npz, allow_pickle=True)
        all_models = sorted({k[:-2] for k in M.files if k.endswith('_R') and not k.endswith('_Rse')})
        # canonical order: replay baselines -> mechanical/parametric by complexity -> neural
        canon = BASELINES + NEURAL
        for m in sorted(all_models, key=lambda x: canon.index(x) if x in canon else 99):
            rm, rs = M[f'{m}_R'], M[f'{m}_Rse']
            n_ev, n_run = (int(v) for v in M[f'{m}_n'])
            x = np.arange(1, len(rm) + 1); ok = np.isfinite(rm) & (M[f'{m}_cnt'] >= 100)
            c = MODEL_COLOR.get(m, '#444444')
            neural = m in NEURAL
            ax.fill_between(x[ok], (rm - 2 * rs)[ok], (rm + 2 * rs)[ok], color=c, alpha=0.12 if neural else 0.07, lw=0)
            ax.plot(x[ok], rm[ok], color=c, lw=1.9 if neural else 1.1, alpha=1.0 if neural else 0.85,
                    label=f'{m} — {n_ev:,} child events (n={n_run})' if neural else m)
            tail = rm[ok][len(rm[ok]) // 2:]
            plateaus[m] = float(np.nanmean(tail))
    for d in sorted(glob.glob(args.triangle_glob)) if args.triangle_glob else []:
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

    ks = np.arange(1, len(Rtr) + 1)
    axt.errorbar(ks, Rtr, yerr=2 * Rtr_se, fmt='o-', ms=4, color=C_REAL, lw=1.6,
                 capsize=2, label='real, trade time')
    model_tr = {}
    if args.model_npz:
        M = np.load(args.model_npz, allow_pickle=True)
        for m in NEURAL:
            if f'{m}_Rtr' not in M.files:
                continue
            rt, rs, rc = M[f'{m}_Rtr'], M[f'{m}_Rtr_se'], M[f'{m}_Rtr_cnt']
            kk = np.arange(1, len(rt) + 1)
            okt = np.isfinite(rt) & (rc >= 100)
            if not okt.any():
                continue
            c = MODEL_COLOR.get(m, '#444444')
            axt.errorbar(kk[okt], rt[okt], yerr=2 * rs[okt], fmt='o-', ms=2.6,
                         color=c, lw=1.2, capsize=0, alpha=0.95)
            axt.annotate(m, (kk[okt][-1], rt[okt][-1]), fontsize=6.4, color=c,
                         va='center', ha='left', xytext=(3, 0), textcoords='offset points')
            model_tr[m] = rt
    if not model_tr:
        for m, p in sorted(plateaus.items(), key=lambda kv: -kv[1]):
            axt.axhline(p, color=MODEL_COLOR.get(m, '#444444'), lw=1.4, ls=':', alpha=0.9)
            axt.annotate(m, (ks[-1], p), fontsize=6.6, color=MODEL_COLOR.get(m, '#444444'),
                         va='bottom', ha='right')
    axt.axvline(10, color='#888888', lw=1.0, ls='--')
    axt.annotate('next insertion\n(10% participation)', (10, axt.get_ylim()[0]),
                 fontsize=7, color='#666666', ha='center', va='bottom')
    axt.axhline(0, color='#bbbbbb', lw=0.8)
    axt.set_xlabel('k-th execution after the event (trade time)')
    axt.set_title('trade-time: real builds (and relaxes); model curves are\n'
                  'short (capped at the next insertion) and flat',
                  fontsize=9.5, loc='left')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axt.get_legend_handles_labels()
    seen = set(); H = []; L = []
    for h, l in zip(h1 + h2, l1 + l2):
        base = l.split(' —')[0]
        if base in seen:
            continue
        seen.add(base); H.append(h); L.append(l)
    fig.legend(H, L, loc='lower center', ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(args.out, bbox_inches='tight')
    np.savez_compressed(os.path.splitext(args.out)[0] + '.npz',
                        mgrid=mg, R=R, R_se=Rse, Rtr=Rtr, Rtr_se=Rtr_se,
                        plateau_models=np.array(list(plateaus)),
                        plateau_vals=np.array(list(plateaus.values())),
                        **{f'{m}_Rtr_fig': v for m, v in model_tr.items()})
    print('plateaus:', {k: round(v, 2) for k, v in plateaus.items()})
    print(f'EVRESP_V2_DONE -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
