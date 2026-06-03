#!/usr/bin/env python3.11
"""Post-process sp500 per-day stats: integer day-mean volume & msgs_btw,
clean 3-panel histogram (volume, trade_frac, msgs_btw). Light: safe on login."""
import csv, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/sp500_universe'

byt = {}
for r in csv.DictReader(open(f'{D}/msgs_btw_sp500_perday.csv')):
    byt.setdefault(r['ticker'], []).append(r)

rows = []
for tk, days in byt.items():
    mb = [int(d['msgs_btw']) for d in days]
    vol = [float(d['p50_mo_volume']) for d in days]
    tf = [float(d['trade_frac']) for d in days]
    rows.append(dict(
        ticker=tk, n_days=len(days),
        vol_day_mean=int(round(st.mean(vol))),           # whole shares
        msgs_btw_day_mean=int(round(st.mean(mb))),         # whole messages
        vol_day_std=int(round(st.pstdev(vol))) if len(vol) > 1 else 0,
        msgs_btw_day_std=int(round(st.pstdev(mb))) if len(mb) > 1 else 0,
        trade_frac_day_mean=round(st.mean(tf) * 100, 2),
        gen_100ins=100 * int(round(st.mean(mb))),
    ))
rows.sort(key=lambda r: r['msgs_btw_day_mean'])

out = f'{D}/msgs_btw_sp500_daymean.csv'
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['ticker', 'n_days', 'vol_day_mean',
        'msgs_btw_day_mean', 'vol_day_std', 'msgs_btw_day_std',
        'trade_frac_day_mean', 'gen_100ins'])
    w.writeheader(); w.writerows(rows)
print(f'wrote {out} ({len(rows)} tickers)')

# ---------- clean 3-panel histogram ----------
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 17, 'axes.labelsize': 14,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12})
vol = np.array([r['vol_day_mean'] for r in rows])
tf  = np.array([r['trade_frac_day_mean'] for r in rows])
mb  = np.array([r['msgs_btw_day_mean'] for r in rows])
d   = {r['ticker']: r for r in rows}
SEL = [('EA', '#00897B'), ('NVDA', '#8E24AA'), ('AMD', '#E53935')]
BAR = {'vol': '#F4A93644', 'tf': '#5BB87044', 'mb': '#4A86C544'}
EDGE = {'vol': '#E69500', 'tf': '#2E8B57', 'mb': '#2F5DA3'}

fig, ax = plt.subplots(1, 3, figsize=(24, 7))

def panel(a, data, key, title, xlabel, selval, fmt='.0f'):
    q1, q2, q3 = np.quantile(data, [.25, .5, .75])
    a.hist(data, bins=45, color=BAR[key], edgecolor=EDGE[key], linewidth=1.2)
    a.axvline(q2, color='#333333', ls='--', lw=2.2)            # median only
    a.set_title(f'{title}\nQ25={q1:{fmt}}  median={q2:{fmt}}  Q75={q3:{fmt}}', fontweight='bold')
    a.set_xlabel(xlabel); a.set_ylabel('# stocks')
    a.grid(axis='y', alpha=0.25)
    for tk, col in SEL:
        v = selval(d[tk])
        a.axvline(v, color=col, lw=3.2, label=f'{tk} = {v:g}')
    a.legend(fontsize=13, framealpha=0.95, loc='upper right')

panel(ax[0], vol, 'vol', 'MO volume  (median size, shares)', 'shares per market order',
      lambda r: r['vol_day_mean'])
panel(ax[1], tf, 'tf', 'trade_frac  (% messages = MO)', '% of messages that are MO',
      lambda r: r['trade_frac_day_mean'], fmt='.1f')
panel(ax[2], mb, 'mb', 'msgs_btw  (eta=10%)', 'msgs_btw = 9 / trade_frac',
      lambda r: r['msgs_btw_day_mean'])

fig.suptitle('S&P500 universe (488 stocks, Jan-2026, day-mean)', fontsize=20, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
hp = f'{D}/msgs_btw_sp500_hist3.png'
fig.savefig(hp, dpi=140); print(f'wrote {hp}')

# ---------- cohort + pick tables (printed for markdown) ----------
bands = [('Liquid', 0, 200), ('Mid-liquid', 200, 280), ('Core', 280, 360),
         ('Heavy', 360, 460), ('Thin', 460, 99999)]
print('\n=== COHORTS ===')
for name, lo, hi in bands:
    g = [r for r in rows if lo <= r['msgs_btw_day_mean'] < hi]
    if not g: continue
    ex = ', '.join(r['ticker'] for r in sorted(g, key=lambda r: r['msgs_btw_day_mean'])[:5])
    mbmin = min(r['msgs_btw_day_mean'] for r in g); mbmax = max(r['msgs_btw_day_mean'] for r in g)
    print(f"{name:11} mb[{lo}-{hi if hi<9999 else '+'}) n={len(g):3}  mb={mbmin}-{mbmax}  gen100={mbmin*100//1000}-{mbmax*100//1000}k  ex: {ex}")
print('\n=== PICK ===')
for tk, _ in SEL:
    r = d[tk]
    print(f"{tk:5} vol={r['vol_day_mean']:3}  mb={r['msgs_btw_day_mean']:3} (+/-{r['msgs_btw_day_std']})  tf={r['trade_frac_day_mean']}%  gen100={r['gen_100ins']//1000}k")
