#!/usr/bin/env python3
"""
Paper Figure "The S&P 500 universe by liquidity": distribution of the day-mean
messages-between-trades statistic m_b across all names, with the three selected
stocks (EA / NVDA / AMD) and the liquidity-cohort bands marked. Rendered in
pubstyle from the Action-1 day-mean CSV — no data mount, runs in seconds.

  python3 1_data_prep/sp500_hist_paper.py --csv <results>/msgs_btw_sp500_daymean.csv
  (default: newest 1_data_prep/results/msgs_btw_*/msgs_btw_sp500_daymean.csv)
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(B, '..', '5_analysis'))
from pubstyle import apply, legend_box, savefig_pub    # noqa: E402
apply()

SEL = [('EA', '#00897B'), ('NVDA', '#8E24AA'), ('AMD', '#E53935')]
# cohort bands from the paper's Table "S&P500 liquidity cohorts"
BANDS = [('Liquid', 0, 200, '#4A86C51A'), ('Mid', 200, 350, '#5BB8701A'),
         ('Thin', 350, 10_000, '#F4A9361A')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    ap.add_argument('--out', default=os.path.join(B, 'results', 'sp500_msgs_btw_hist.png'))
    args = ap.parse_args()

    path = args.csv
    if path is None:
        cands = sorted(glob.glob(os.path.join(B, 'results', 'msgs_btw_*', 'msgs_btw_sp500_daymean.csv')))
        if not cands:
            sys.exit('no msgs_btw_sp500_daymean.csv under 1_data_prep/results/ — run run_msgs_btw.sh first')
        path = cands[-1]
    rows = list(csv.DictReader(open(path)))
    mb = np.array([int(r['msgs_btw_day_mean']) for r in rows])
    d = {r['ticker']: r for r in rows}
    print(f'{path}: {len(rows)} tickers, m_b range {mb.min()}-{mb.max()}, median {np.median(mb):.0f}')

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for name, lo, hi, col in BANDS:
        ax.axvspan(lo, min(hi, mb.max() * 1.04), color=col, zorder=0)
    ax.hist(mb, bins=48, color='#4A86C566', edgecolor='#2F5DA3', linewidth=0.9, zorder=2)
    med = np.median(mb)
    ax.axvline(med, color='#333333', ls='--', lw=1.6, zorder=3,
               label=f'universe median = {med:.0f}')
    for tk, col in SEL:
        if tk not in d:
            print(f'WARN: {tk} not in table'); continue
        v = int(d[tk]['msgs_btw_day_mean'])
        ax.axvline(v, color=col, lw=2.6, zorder=4, label=f'{tk}  ($m_b$ = {v})')
    # cohort labels along the top
    ymax = ax.get_ylim()[1]
    for name, lo, hi, _ in BANDS:
        xc = (lo + min(hi, mb.max())) / 2
        ax.text(xc, ymax * 0.97, name, ha='center', va='top',
                fontsize=10, color='#555555', style='italic')
    ax.set_xlim(0, mb.max() * 1.04)
    ax.set_xlabel(r'$m_b$ — messages between consecutive trades (day-mean, $\eta = 10\%$ calibration)')
    ax.set_ylabel('number of stocks')
    legend_box(ax, loc='center right')
    fig.tight_layout()
    print('wrote', savefig_pub(fig, args.out))


if __name__ == '__main__':
    main()
