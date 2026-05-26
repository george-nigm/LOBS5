#!/usr/bin/env python3
"""
Compute comprehensive LOBSTER statistics for framework calibration.

Reads preprocessed .npy files (message + orderbook) for GOOG and INTC,
computes per-day statistics and aggregates across all trading days.

Preprocessed format:
  Message (N, 14): OrderID, EventType, Direction, Price, RelPrice, Size, Flag, ...
    - EventType: 1=add, 2=modify, 3=cancel, 4=execution
    - Price: centidollars (e.g. 3177800 = $317.78)
    - Size: shares
  Orderbook (N, 43): Flag, Time, Counter, then 10 levels of
    [ask_price, ask_vol, bid_price, bid_vol]

Usage:
    python lob_impact/compute_lobster_stats.py
    python lob_impact/compute_lobster_stats.py --stock GOOG
    python lob_impact/compute_lobster_stats.py --stock INTC --n_days 10
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_BASE = '/lus/lfs1aip2/projects/s5e/lob_pipeline/data'

# ── Constants ───────────────────────────────────────────────────────────────
TICK_SIZE = 100          # centidollars
TRADING_SECONDS = 23400  # 6.5 hours (9:30 AM – 4:00 PM)

# Message column indices (preprocessed .npy)
COL_ORDERID    = 0
COL_EVENT_TYPE = 1
COL_DIRECTION  = 2
COL_PRICE      = 3
COL_SIZE       = 5
COL_TIME_SEC   = 8

# Orderbook column indices (preprocessed .npy)
# First 3 cols are metadata; then 10 × [ask_price, ask_vol, bid_price, bid_vol]
BOOK_ASK_PRICE_L1 = 3
BOOK_ASK_VOL_L1   = 4
BOOK_BID_PRICE_L1 = 5
BOOK_BID_VOL_L1   = 6

# Book depth sampling: every N-th event to reduce autocorrelation
DEPTH_SAMPLE_STEP = 100

# Mid-price return windows
RETURN_WINDOWS = [100, 500, 1000]


def compute_day_stats(msg_file: str, book_file: str, tick_size: int = TICK_SIZE) -> dict:
    """Compute statistics for a single day."""
    basename = os.path.basename(msg_file)
    m = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
    day = m.group(1) if m else basename

    msg = np.load(msg_file)
    book = np.load(book_file)

    n_total = len(msg)
    n_book = len(book)
    n_aligned = min(n_total, n_book)

    # ── 1. Events ───────────────────────────────────────────────────────
    event_types = msg[:, COL_EVENT_TYPE]
    trades_mask = event_types == 4  # only type 4 in preprocessed data
    n_trades = int(trades_mask.sum())
    r_trade_frac = n_trades / n_total if n_total > 0 else 0.0

    # Per event-type counts
    n_add    = int((event_types == 1).sum())
    n_modify = int((event_types == 2).sum())
    n_cancel = int((event_types == 3).sum())

    # ── 2. Trade volumes ────────────────────────────────────────────────
    trade_sizes = msg[trades_mask, COL_SIZE].astype(float)
    adv = float(trade_sizes.sum()) if len(trade_sizes) > 0 else 0.0

    size_pctiles = {}
    if len(trade_sizes) > 0:
        for p in [25, 50, 75, 90, 95, 99]:
            size_pctiles[f'trade_size_p{p}'] = float(np.percentile(trade_sizes, p))
        size_pctiles['trade_size_mean'] = float(np.mean(trade_sizes))
    else:
        for p in [25, 50, 75, 90, 95, 99]:
            size_pctiles[f'trade_size_p{p}'] = np.nan
        size_pctiles['trade_size_mean'] = np.nan

    # ── 3. Book depth (sampled every DEPTH_SAMPLE_STEP events) ──────────
    idx = np.arange(0, n_aligned, DEPTH_SAMPLE_STEP)
    ask_vol = book[idx, BOOK_ASK_VOL_L1].astype(float)
    bid_vol = book[idx, BOOK_BID_VOL_L1].astype(float)
    depth_best = ask_vol + bid_vol

    depth_pctiles = {}
    valid_depth = depth_best[depth_best > 0]
    if len(valid_depth) > 0:
        for p in [25, 50, 75, 90, 95]:
            depth_pctiles[f'depth_best_p{p}'] = float(np.percentile(valid_depth, p))
        depth_pctiles['depth_best_mean'] = float(np.mean(valid_depth))
        depth_pctiles['ask_vol_L1_median'] = float(np.median(ask_vol[ask_vol > 0]))
        depth_pctiles['bid_vol_L1_median'] = float(np.median(bid_vol[bid_vol > 0]))
    else:
        for p in [25, 50, 75, 90, 95]:
            depth_pctiles[f'depth_best_p{p}'] = np.nan
        depth_pctiles['depth_best_mean'] = np.nan
        depth_pctiles['ask_vol_L1_median'] = np.nan
        depth_pctiles['bid_vol_L1_median'] = np.nan

    # ── 4. Spread ───────────────────────────────────────────────────────
    ask_prices = book[idx, BOOK_ASK_PRICE_L1].astype(float)
    bid_prices = book[idx, BOOK_BID_PRICE_L1].astype(float)
    valid_spread_mask = (ask_prices > 0) & (bid_prices > 0) & (ask_prices > bid_prices)
    if valid_spread_mask.sum() > 0:
        spreads = ask_prices[valid_spread_mask] - bid_prices[valid_spread_mask]
        mids = (ask_prices[valid_spread_mask] + bid_prices[valid_spread_mask]) / 2.0
        spread_ticks_arr = spreads / tick_size
        spread_bps_arr = spreads / mids * 10000

        spread_ticks_mean = float(np.mean(spread_ticks_arr))
        spread_ticks_median = float(np.median(spread_ticks_arr))
        spread_bps_mean = float(np.mean(spread_bps_arr))
        spread_bps_median = float(np.median(spread_bps_arr))
    else:
        spread_ticks_mean = spread_ticks_median = np.nan
        spread_bps_mean = spread_bps_median = np.nan

    # ── 5. Mid-price volatility ─────────────────────────────────────────
    all_ask = book[:n_aligned, BOOK_ASK_PRICE_L1].astype(float)
    all_bid = book[:n_aligned, BOOK_BID_PRICE_L1].astype(float)
    valid_mid_mask = (all_ask > 0) & (all_bid > 0) & (all_ask > all_bid)
    all_mid = np.where(valid_mid_mask, (all_ask + all_bid) / 2.0, np.nan)

    sigma = {}
    for w in RETURN_WINDOWS:
        if n_aligned > w:
            mid_w = all_mid[::w]  # sample at every w-th event
            mid_w = mid_w[~np.isnan(mid_w)]
            if len(mid_w) > 2:
                log_ret = np.diff(np.log(mid_w))
                sigma[f'sigma_W{w}'] = float(np.std(log_ret))
            else:
                sigma[f'sigma_W{w}'] = np.nan
        else:
            sigma[f'sigma_W{w}'] = np.nan

    # ── 6. Message rate ─────────────────────────────────────────────────
    times = msg[:, COL_TIME_SEC].astype(float)
    t_range = float(times[-1] - times[0])
    if t_range <= 0:
        t_range = TRADING_SECONDS
    msg_rate = n_total / t_range

    # ── Assemble record ────────────────────────────────────────────────
    record = {
        'day': day,
        'n_total': n_total,
        'n_trades': n_trades,
        'n_add': n_add,
        'n_modify': n_modify,
        'n_cancel': n_cancel,
        'trade_fraction': r_trade_frac,
        'ADV': adv,
        'msg_rate_per_sec': msg_rate,
        't_range_sec': t_range,
        'spread_ticks_mean': spread_ticks_mean,
        'spread_ticks_median': spread_ticks_median,
        'spread_bps_mean': spread_bps_mean,
        'spread_bps_median': spread_bps_median,
    }
    record.update(size_pctiles)
    record.update(depth_pctiles)
    record.update(sigma)

    return record


def run(stock: str, n_days: int = 20, tick_size: int = TICK_SIZE):
    data_dir = os.path.join(DATA_BASE, f'{stock}_jan2026')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    msg_files = sorted(glob.glob(os.path.join(data_dir, f'{stock}_*_message_10_proc.npy')))
    if not msg_files:
        raise FileNotFoundError(f'No message .npy files in {data_dir}')

    msg_files = msg_files[:n_days]
    print(f'=== LOBSTER Statistics: {stock} ({len(msg_files)} days) ===')
    print(f'Data dir: {data_dir}\n')

    records = []
    for msg_file in msg_files:
        book_file = msg_file.replace('_message_', '_orderbook_')
        if not os.path.exists(book_file):
            print(f'  SKIP {os.path.basename(msg_file)} — no orderbook file')
            continue

        day_label = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(msg_file))
        day_label = day_label.group(1) if day_label else '?'
        print(f'  Processing {day_label} ...', end=' ', flush=True)

        rec = compute_day_stats(msg_file, book_file, tick_size)
        records.append(rec)

        print(f'n={rec["n_total"]:,}  trades={rec["n_trades"]:,}  '
              f'ADV={rec["ADV"]:,.0f}  spread={rec["spread_ticks_median"]:.1f}t')

    df = pd.DataFrame(records)

    # ── Print aggregate summary ─────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  AGGREGATE SUMMARY: {stock} ({len(df)} days)')
    print(f'{"="*70}')

    def fmt(label, series, fmt_str='.0f'):
        vals = series.dropna()
        if len(vals) == 0:
            print(f'  {label}: no data')
            return
        print(f'  {label}:  mean={vals.mean():{fmt_str}}  '
              f'median={vals.median():{fmt_str}}  '
              f'[{vals.min():{fmt_str}}, {vals.max():{fmt_str}}]')

    print(f'\n  EVENTS:')
    fmt('n_total (msgs/day)', df['n_total'], ',')
    fmt('n_trades (execs/day)', df['n_trades'], ',')
    fmt('trade_fraction', df['trade_fraction'], '.3%')
    fmt('msg_rate (msg/sec)', df['msg_rate_per_sec'], '.1f')

    print(f'\n  TRADE SIZES:')
    for p in [25, 50, 75, 90, 95, 99]:
        col = f'trade_size_p{p}'
        if col in df.columns:
            fmt(f'  p{p}', df[col], '.0f')
    fmt('  mean', df['trade_size_mean'], '.1f')
    fmt('ADV (daily volume)', df['ADV'], ',')

    print(f'\n  BOOK DEPTH (at best, sampled every {DEPTH_SAMPLE_STEP} events):')
    for p in [25, 50, 75, 90, 95]:
        col = f'depth_best_p{p}'
        if col in df.columns:
            fmt(f'  depth p{p}', df[col], '.0f')
    fmt('  depth mean', df['depth_best_mean'], '.0f')
    fmt('  ask_vol_L1 median', df['ask_vol_L1_median'], '.0f')
    fmt('  bid_vol_L1 median', df['bid_vol_L1_median'], '.0f')

    print(f'\n  SPREAD:')
    fmt('spread (ticks) mean', df['spread_ticks_mean'], '.2f')
    fmt('spread (ticks) median', df['spread_ticks_median'], '.2f')
    fmt('spread (bps) mean', df['spread_bps_mean'], '.2f')
    fmt('spread (bps) median', df['spread_bps_median'], '.2f')

    print(f'\n  MID-PRICE VOLATILITY (std of log-returns):')
    for w in RETURN_WINDOWS:
        col = f'sigma_W{w}'
        if col in df.columns:
            fmt(f'  sigma(W={w})', df[col], '.6f')

    # ── Save ────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), f'lobster_stats_{stock}.csv')
    df.to_csv(out_path, index=False)
    print(f'\nSaved per-day stats: {out_path} ({len(df)} rows)')

    return df


def main():
    parser = argparse.ArgumentParser(description='Compute LOBSTER data statistics')
    parser.add_argument('--stock', type=str, default=None,
                        help='Stock ticker (GOOG, INTC). If not specified, runs both.')
    parser.add_argument('--n_days', type=int, default=20,
                        help='Number of trading days to process (default: 20)')
    parser.add_argument('--tick_size', type=int, default=TICK_SIZE,
                        help='Tick size in centidollars (default: 100)')
    args = parser.parse_args()

    stocks = [args.stock.upper()] if args.stock else ['GOOG', 'INTC']

    for stock in stocks:
        try:
            run(stock, args.n_days, args.tick_size)
        except FileNotFoundError as e:
            print(f'ERROR: {e}')
        print()


if __name__ == '__main__':
    main()
