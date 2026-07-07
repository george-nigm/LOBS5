#!/usr/bin/env python3
"""
Paper Figure "Response to a Single Child Order": R(m) for the neural generators vs the
real-EA anchor, rendered in pubstyle from the CACHED full-sample triangle numbers.json
(no grid access — runs in seconds).

  python 5_analysis/beta/event_response_paper.py --out results/aggregation/event_response_EA.png
"""
import os, sys, re, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(B, '..'))
sys.path.insert(0, os.path.join(B, '..', '..', '4_diagnostics'))
from pubstyle import apply, style, legend_box, savefig_pub          # noqa: E402
from triangle_compare_report import latest_numbers                  # noqa: E402
apply()

C_REAL = '#008300'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='Mamba3,Mamba3_4k,S5_4k')
    ap.add_argument('--results_dir', default=os.path.join(B, '..', '..',
                                                          '4_diagnostics', 'results'))
    ap.add_argument('--baselines', default=os.path.join(B, '..', '..', '4_diagnostics',
                                                        'results', 'baseline_resp.json'),
                    help='cached R(m) for the impact-blind baselines (baseline_resp.py); "" to skip')
    ap.add_argument('--out', default=os.path.join(B, 'results', 'aggregation',
                                                  'event_response_EA.png'))
    args = ap.parse_args()
    models = [m for m in args.models.split(',') if m]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    emp = None
    ends = []
    for m in models:
        N = json.load(open(latest_numbers(os.path.abspath(args.results_dir), m)))
        emp = emp or N['empirical']
        r = N['visible-buy']
        rm = np.array(r['resp_mean'], float)
        rs = np.array(r['resp_se'], float)
        x = np.arange(len(rm))
        ok = np.isfinite(rm)
        st = style(m)
        ax.fill_between(x[ok], (rm - 2 * rs)[ok], (rm + 2 * rs)[ok],
                        color=st['color'], alpha=0.15, lw=0)
        ax.plot(x[ok], rm[ok], color=st['color'], ls=st['ls'], lw=1.8,
                label=f"{st['label']} ({r['resp_n_events'] // 1000}k child events, "
                      f"n={r['n']} runs)", zorder=3)
        ends.append((float(rm[ok][-1]), st['label'], st['color']))

    # impact-blind baselines: thin muted lines hugging zero (replay/stationary models cannot
    # react to the inserted child order) — the P3 discrimination in one glance
    if args.baselines and os.path.isfile(args.baselines):
        BL = json.load(open(args.baselines))
        for m, r in BL.items():
            rm = np.array(r['resp_mean'], float)
            x = np.arange(len(rm))
            ok = np.isfinite(rm)
            st = style(m)
            ax.plot(x[ok], rm[ok], color=st['color'], ls=st['ls'], lw=1.1, alpha=0.75,
                    label=f"{st['label']} ({r['resp_n_events'] // 1000}k events) — no response",
                    zorder=2)

    hs = sorted(int(k.split('_')[1]) for k in emp if re.fullmatch(r'R_\d+', k))
    rv = [emp[f'R_{h}'] for h in hs]
    se = [emp.get(f'R_{h}_se', 0) for h in hs]
    ax.errorbar(hs, rv, yerr=[2 * s for s in se], color=C_REAL, lw=1.9,
                ls=(0, (4, 2)), marker='o', ms=5.5, capsize=3,
                label=f'real EA data ({emp["n_events"]:,} executions)', zorder=4)
    # saturation guide: real impact stops growing past m ~ 60
    ax.axhline(rv[-1], color=C_REAL, lw=0.8, ls=':', alpha=0.7, zorder=1)
    ax.annotate(f'real response saturates at {rv[-1]:+.2f} ticks',
                xy=(131, rv[-1]), xytext=(66, rv[-1] - 0.14), color=C_REAL,
                fontsize=10, fontweight='bold')
    ends.append((rv[-1], 'real EA', C_REAL))

    ax.set_xlim(0, 135)
    ax.set_ylim(bottom=-0.1)     # keep the ≈0 baselines visible (they dip a few hundredths negative)
    ax.axhline(0, color='#9a9a9a', lw=0.7, zorder=1)
    ax.set_xlabel(r'messages after the execution event  $m$')
    ax.set_ylabel(r'mean mid response $R(m)$  (ticks)')
    ax.set_title('EA — response to a single child order vs real data')
    legend_box(ax, loc='upper left')
    fig.tight_layout()
    savefig_pub(fig, os.path.abspath(args.out))
    plt.close(fig)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
