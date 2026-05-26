#!/usr/bin/env python3
"""
Compute per-day experiment parameters for V10 (per-day calibrated) experiments.

Reads lobster_stats_{STOCK}.csv + daily_h_l_{STOCK}.csv and produces a
per_day_params_{STOCK}.csv with day-specific child, mb, and volume multipliers.

Design:
  child_base_d = trade_size_p50 on day d (median market order size)
  mb_d = child_base_d × (1/η − 1) / (trade_frac_d × trade_p50_d), capped at mb_max
  child_d(mult) = child_base_d × mult, for mult ∈ {0.5, 1.0, 2.0}

  mb_d is fixed per day (does NOT scale with multiplier).
  η varies with multiplier: lower mult → lower η (smaller player).

Usage:
    python lob_impact/compute_per_day_params.py --stock AAPL
    python lob_impact/compute_per_day_params.py --stock AAPL --eta 0.10 --mb_max 500 --mults 0.5,1.0,2.0
    python lob_impact/compute_per_day_params.py --all
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_ETA = 0.10
DEFAULT_MB_MAX = 500
DEFAULT_MULTS = [0.5, 1.0, 2.0]
DEFAULT_N_INS = 10
DEFAULT_N_COOL = 0

ALL_STOCKS = ['AAPL', 'META', 'MSFT', 'NVDA', 'TSLA', 'GOOG', 'INTC', 'AMZN',
              'AMD', 'MU', 'NFLX']


def compute_params(stock, eta=DEFAULT_ETA, mb_max=DEFAULT_MB_MAX,
                   mults=None, n_ins=DEFAULT_N_INS):
    """Compute per-day parameters for one stock."""
    if mults is None:
        mults = DEFAULT_MULTS

    ls_path = SCRIPT_DIR / f'lobster_stats_{stock}.csv'
    hl_path = SCRIPT_DIR / f'daily_h_l_{stock}.csv'

    if not ls_path.exists():
        print(f'  SKIP {stock}: {ls_path} not found')
        return None
    if not hl_path.exists():
        print(f'  SKIP {stock}: {hl_path} not found')
        return None

    ls = pd.read_csv(ls_path)
    hl = pd.read_csv(hl_path)

    # Build daily volume map
    if 'day' not in hl.columns:
        hl['day'] = hl['filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    hl_map = dict(zip(hl['day'], hl['execution_sum']))

    # Map lobster_stats days to daily_h_l days (may have date mismatch 2022↔2026)
    ls_days = sorted(ls['day'].unique())
    hl_days = sorted(hl_map.keys())
    if ls_days[0] != hl_days[0] and len(ls_days) == len(hl_days):
        day_map = dict(zip(ls_days, hl_days))
    else:
        day_map = {d: d for d in ls_days}

    rows = []
    for _, r in ls.iterrows():
        day = r['day']
        hl_day = day_map.get(day, day)
        V_d = hl_map.get(hl_day, np.nan)
        if np.isnan(V_d) or V_d <= 0:
            continue

        child_base = int(r['trade_size_p50'])
        if child_base < 1:
            child_base = 1

        trade_frac = r['trade_fraction']
        trade_p50 = r['trade_size_p50']

        # mb calibrated for η at child=child_base
        denom = trade_frac * trade_p50
        if denom <= 0:
            continue
        mb_d = int(child_base * (1.0 / eta - 1) / denom)
        mb_d = min(mb_d, mb_max)
        mb_d = max(mb_d, 1)

        total_gen = n_ins * mb_d

        # Parkinson sigma
        H = hl.loc[hl['day'] == hl_day, 'highest_price']
        L = hl.loc[hl['day'] == hl_day, 'lowest_price']
        if len(H) > 0 and len(L) > 0 and float(H.iloc[0]) > 0 and float(L.iloc[0]) > 0:
            sigma_d = np.log(float(H.iloc[0]) / float(L.iloc[0])) / 1.6651092
        else:
            sigma_d = np.nan

        for mult in mults:
            child_d = max(int(child_base * mult), 1)
            Q_d = n_ins * child_d

            # Actual η for this child (mb unchanged)
            mkt_exec = mb_d * trade_frac * trade_p50
            eta_actual = child_d / (child_d + mkt_exec) if (child_d + mkt_exec) > 0 else 0

            rows.append(dict(
                day=day,
                hl_day=hl_day,
                mult=mult,
                child=child_d,
                child_base=child_base,
                mb=mb_d,
                n_ins=n_ins,
                Q=Q_d,
                total_gen=total_gen,
                V_daily=V_d,
                sigma_daily=sigma_d,
                eta_actual=eta_actual,
                eta_target=eta,
                trade_frac=trade_frac,
                trade_p50=trade_p50,
                depth_p50=r.get('depth_best_p50', np.nan),
                msg_rate=r.get('msg_rate_per_sec', np.nan),
            ))

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description='Compute per-day V10 parameters')
    parser.add_argument('--stock', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--eta', type=float, default=DEFAULT_ETA)
    parser.add_argument('--mb_max', type=int, default=DEFAULT_MB_MAX)
    parser.add_argument('--mults', type=str, default='0.5,1.0,2.0',
                        help='Comma-separated child multipliers')
    parser.add_argument('--n_ins', type=int, default=DEFAULT_N_INS)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    mults = [float(x) for x in args.mults.split(',')]
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR

    stocks = ALL_STOCKS if args.all else ([args.stock] if args.stock else [])
    if not stocks:
        print('Specify --stock AAPL or --all')
        sys.exit(1)

    for stock in stocks:
        print(f'\n{"="*60}')
        print(f'  {stock}: η_target={args.eta}, mb_max={args.mb_max}, mults={mults}')
        print(f'{"="*60}')

        df = compute_params(stock, eta=args.eta, mb_max=args.mb_max,
                            mults=mults, n_ins=args.n_ins)
        if df is None:
            continue

        out_path = out_dir / f'per_day_params_{stock}.csv'
        df.to_csv(out_path, index=False)

        n_days = df['day'].nunique()
        n_mults = df['mult'].nunique()

        print(f'  {n_days} days × {n_mults} multipliers = {len(df)} rows')
        print(f'  child range: [{df["child"].min()}, {df["child"].max()}]')
        print(f'  mb range: [{df["mb"].min()}, {df["mb"].max()}]')
        print(f'  Q range: [{df["Q"].min()}, {df["Q"].max()}]')
        print(f'  total_gen range: [{df["total_gen"].min()}, {df["total_gen"].max()}]')
        print(f'  η actual range: [{df["eta_actual"].min():.1%}, {df["eta_actual"].max():.1%}]')

        x = np.log(df['Q'] / df['V_daily'])
        print(f'  x=log(Q/V) range: [{x.min():+.3f}, {x.max():+.3f}] = {x.max()-x.min():.2f} log-units')
        print(f'  Saved: {out_path}')


if __name__ == '__main__':
    main()
