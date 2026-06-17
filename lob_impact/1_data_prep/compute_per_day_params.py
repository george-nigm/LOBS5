#!/usr/bin/env python3
"""
Per-day scenario parameters (child = order_volume, mb = msgs_between) per stock.

Rebuilds the per-day calibration chain that was dropped in the 5-action refactor
(old lob_impact/compute_per_day_params.py), adapted to the CURRENT Action-1 output
`msgs_btw_sp500_perday.csv` (columns: ticker, day, n_msgs, n_trades, trade_frac,
p50_mo_volume, msgs_btw).

Design (eta = 10% participation, the value msgs_btw was derived under):
    child_d = p50_mo_volume(d)          # daily-median market-order size
    mb_d    = msgs_btw(d) = int(9 / trade_frac(d))   # already eta-calibrated for child = p50
The "child = trade_p50 cancels" identity (see compute_sp500_msgs_btw.py) means
order_volume MUST equal p50 for the realized participation to hit eta — which is
exactly why the hardcoded order_volume=75 broke it.

Emits per_day_params_<STOCK>.csv with the columns the scenario per-day branch reads
(historic_scenario.py / mamba3_scenario.py: day, mult, child, mb), plus provenance.

  python 1_data_prep/compute_per_day_params.py --stocks EA,NVDA,AMD [--mults 1.0]
"""
import os, glob, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def latest_perday_csv():
    c = sorted(glob.glob(os.path.join(HERE, 'results', 'msgs_btw_*', 'msgs_btw_sp500_perday.csv')))
    if not c:
        raise SystemExit('no msgs_btw_sp500_perday.csv under 1_data_prep/results/ (run Action 1 first)')
    return c[-1]


def build(stock, df, mults):
    sub = df[df['ticker'] == stock].copy()
    if sub.empty:
        print(f'  {stock}: no rows'); return None
    sub = sub.sort_values('day')
    rows = []
    for _, r in sub.iterrows():
        child_base = max(int(round(float(r['p50_mo_volume']))), 1)   # daily-median MO size
        mb = max(int(r['msgs_btw']), 1)                              # eta=10% spacing for child=p50
        for mult in mults:
            rows.append(dict(
                day=r['day'], mult=mult, child=max(int(round(child_base * mult)), 1), mb=mb,
                p50_mo_volume=float(r['p50_mo_volume']), trade_frac=float(r['trade_frac']),
                msgs_btw=int(r['msgs_btw']), n_trades=int(r['n_trades'])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stocks', default='EA,NVDA,AMD')
    ap.add_argument('--mults', default='1.0', help='child multipliers (1.0 = median MO)')
    ap.add_argument('--perday_csv', default=None)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    src = args.perday_csv or latest_perday_csv()
    df = pd.read_csv(src)
    mults = [float(x) for x in args.mults.split(',')]
    out_dir = args.out_dir or os.path.join(HERE, 'results', 'per_day_params')
    os.makedirs(out_dir, exist_ok=True)
    print(f'source: {src}\nmults: {mults}\nout:    {out_dir}')
    for stock in args.stocks.split(','):
        d = build(stock, df, mults)
        if d is None:
            continue
        out = os.path.join(out_dir, f'per_day_params_{stock}.csv')
        d.to_csv(out, index=False)
        one = d[d['mult'] == 1.0]
        print(f'  {stock}: {d["day"].nunique()} days  child(p50) [{one["child"].min()}–{one["child"].max()}] '
              f'median {int(one["child"].median())}  |  mb [{one["mb"].min()}–{one["mb"].max()}]  -> {out}')


if __name__ == '__main__':
    main()
