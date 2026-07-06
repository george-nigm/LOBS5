#!/usr/bin/env python3
"""
Duration-(in)dependence from EXISTING grid data — no new generation.

Three in-data probes, one figure:
  (A) deconfounded across-day test: per-day I(k=100)/sigma_d vs m_b(d)
      (dividing by day volatility removes the vol confound of the raw scatter);
  (B) within-day window-tail test: R_d(m_b-1) / R_d(60) vs m_b(d)
      (per-child contribution at its own window end over the m=60 level; day vol
       cancels in the ratio; ratio>1 growing with m_b => longer windows add impact);
  (C) Shape II statement: at FIXED Q, letting 10x more time pass grows impact by
      end/peak-1 (read from the cached master-curve npz).

  python 5_analysis/beta/duration_indata.py --grid <v2root> --stock EA \
         --models Mamba3,Mamba3_4k,S5_4k
"""
import os, sys, glob, re, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(B, '..'))
sys.path.insert(0, os.path.join(B, '..', '..', '4_diagnostics'))
from pubstyle import apply, style, legend_box, savefig_pub          # noqa: E402
from control_triangle_report import (read_csv_np, discover_exp,      # noqa: E402
                                     aggr_for, book_mid_spread)
from vol_estimators import daily_sigmas                              # noqa: E402
apply()

DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')


def collect_day_stats(exp, n_ins=100, m_max=160):
    """One pass over buy-side books: per-day I(k=100) list + per-day streaming R(m)."""
    days = {}   # date -> dict(i100=[], mb, r_sum, r_cnt)
    for bf in sorted(glob.glob(os.path.join(exp, 'data_gen', '*orderbook*gen*.csv'))):
        m = DATE_RE.search(os.path.basename(bf))
        if not m:
            continue
        date = m.group(1)
        mid, _ = book_mid_spread(read_csv_np(bf))
        ai = aggr_for(exp, date, len(mid))
        ai = ai[ai > 0]
        if len(ai) < n_ins:
            continue
        ref = mid[ai[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        mb = int(np.diff(ai).min())
        d = days.setdefault(date, dict(mb=mb, i100=[],
                                       r_sum=np.zeros(m_max + 1),
                                       r_cnt=np.zeros(m_max + 1, dtype=np.int64)))
        if ai[n_ins - 1] < len(mid) and np.isfinite(mid[ai[n_ins - 1]]):
            d['i100'].append((mid[ai[n_ins - 1]] - ref) / ref * 1e4)   # bps
        nxt = np.append(ai[1:], len(mid))
        for i, nx in zip(ai[:n_ins], nxt[:n_ins]):
            base = mid[i - 1]
            if np.isnan(base):
                continue
            hi = min(i - 1 + m_max, nx - 1, len(mid) - 1)
            seg = (mid[i:hi + 1] - base) / 100.0                        # ticks (TICK=100)
            ok = ~np.isnan(seg)
            mm = np.arange(1, len(seg) + 1)[ok]
            d['r_sum'][mm] += seg[ok]
            d['r_cnt'][mm] += 1
    return days


def day_table(days):
    rows = []
    for date, d in sorted(days.items()):
        if not d['i100']:
            continue
        R = np.where(d['r_cnt'] > 10, d['r_sum'] / np.maximum(d['r_cnt'], 1), np.nan)
        m_end = min(d['mb'] - 1, len(R) - 1)
        rows.append(dict(date=date, mb=d['mb'], i100=float(np.mean(d['i100'])),
                         n=len(d['i100']), r60=float(R[60]) if len(R) > 60 else np.nan,
                         r_end=float(R[m_end])))
    return rows


def fit_line(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    b1, b0 = np.polyfit(x[ok], y[ok], 1)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return b0, b1, r, ok.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2')
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Mamba3,Mamba3_4k,S5_4k')
    ap.add_argument('--daily', default=os.path.join(B, '..', '..',
                    '2_daily_stats/results/daily_20260618-215656/daily_h_l_all.csv'))
    ap.add_argument('--relax_npz', default=os.path.join(B, 'results', 'master_curve',
                    'master_curve_EA_relaxation_v2gated.npz'))
    ap.add_argument('--out', default=os.path.join(B, 'results', 'aggregation',
                    'duration_indata_EA.png'))
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.6, 4.3),
                                        width_ratios=[1.15, 1.15, 0.7])
    print('=== duration (in)dependence from existing data ===')
    models = [m for m in args.models.split(',') if m]
    endpeak = {}
    z = np.load(args.relax_npz) if os.path.exists(args.relax_npz) else None
    for m in models:
        exp = discover_exp(os.path.join(args.grid, f'{args.stock}-{m}-beta', 'buy'))
        if not exp:
            print(f'{m}: no data'); continue
        rows = day_table(collect_day_stats(exp))
        st = style(m)
        mb = np.array([r['mb'] for r in rows], float)
        # (A) sigma-normalised I(100), rescaled to its own mean so models share an axis
        sd = np.array([sig.get((args.stock, r['date']), {}).get('parkinson', np.nan)
                       for r in rows])
        in_ = np.array([r['i100'] for r in rows]) / sd
        in_ /= np.nanmean(in_)
        _, b1, r1, n1 = fit_line(mb, in_)
        axA.scatter(mb, in_, s=26, color=st['color'], zorder=3)
        xs = np.linspace(mb.min(), mb.max(), 10)
        axA.plot(xs, np.polyval(np.polyfit(mb[np.isfinite(in_)], in_[np.isfinite(in_)], 1), xs),
                 color=st['color'], ls=st['ls'], lw=1.4,
                 label=f"{st['label']}: {100 * b1 * (mb.max() - mb.min()):+.0f}% over the m_b "
                       f"range, r={r1:+.2f}")
        # (B) window-tail ratio
        ratio = np.array([r['r_end'] / r['r60'] if r['r60'] else np.nan for r in rows])
        _, b2, r2, n2 = fit_line(mb, ratio)
        axB.scatter(mb, ratio, s=26, color=st['color'], zorder=3)
        axB.plot(xs, np.polyval(np.polyfit(mb[np.isfinite(ratio)],
                                           ratio[np.isfinite(ratio)], 1), xs),
                 color=st['color'], ls=st['ls'], lw=1.4,
                 label=f"{st['label']}: mean {np.nanmean(ratio):.2f}, r={r2:+.2f}")
        # (C) Shape II end/peak
        if z is not None and f'{m}_master' in z.files:
            mast = z[f'{m}_master']
            endpeak[m] = float(mast[np.isfinite(mast)][-1])
        print(f'{m}: A slope={b1:+.4f}/mb (r={r1:+.2f}, n={n1}) | '
              f'B ratio mean={np.nanmean(ratio):.3f} slope={b2:+.4f}/mb (r={r2:+.2f}) | '
              f'C end/peak={endpeak.get(m, float("nan")):.2f}')

    axA.set_xlabel(r'day $m_b$')
    axA.set_ylabel(r'$I(k{=}100)\,/\,\sigma_d$  (norm.)')
    axA.set_title(r'(A) across days, vol-deconfounded', loc='left')
    axA.axhline(1.0, color='#9a9a9a', lw=0.8)
    legend_box(axA, loc='upper left', fontsize=7.5)

    axB.set_xlabel(r'day $m_b$')
    axB.set_ylabel(r'$R_d(m_b{-}1)\,/\,R_d(60)$')
    axB.set_title(r'(B) within day: window-tail growth', loc='left')
    axB.axhline(1.0, color='#9a9a9a', lw=0.8)
    legend_box(axB, loc='upper left', fontsize=7.5)

    ms = [m for m in models if m in endpeak]
    y = [100 * (endpeak[m] - 1) for m in ms]
    axC.bar(range(len(ms)), y, color=[style(m)['color'] for m in ms], width=0.6)
    for i, v in enumerate(y):
        axC.text(i, v + 1, f'+{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
    axC.set_xticks(range(len(ms)), [style(m)['label'] for m in ms], fontsize=8)
    axC.set_ylabel(r'impact growth at fixed $Q$ (%)')
    axC.set_title('(C) Shape II: +10× time', loc='left')
    axC.axhline(0, color='#9a9a9a', lw=0.8)

    fig.tight_layout()
    savefig_pub(fig, args.out)
    plt.close(fig)
    print(f'DURATION_INDATA_DONE -> {args.out}')


if __name__ == '__main__':
    main()
