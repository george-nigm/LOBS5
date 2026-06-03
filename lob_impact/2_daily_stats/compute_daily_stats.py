#!/usr/bin/env python3
"""
Action 2 — daily statistics for market-impact normalization.

Reads a mounted squashfs shard (same --mnt convention as 1_data_prep/compute_sp500_msgs_btw.py:
one subdir per ticker, files <ticker>/*message*proc.npy) and emits daily High/Low + execution
volume, used downstream for Parkinson volatility (sigma = ln(H/L)/1.6651092) and participation rate.

  # ALL tickers -> one combined file (default):
  python 2_daily_stats/compute_daily_stats.py --mnt <root> --out_dir <dir>        # daily_h_l_all.csv
  # a single ticker -> per-stock file (for the analysis loader):
  python 2_daily_stats/compute_daily_stats.py --mnt <root> --stock EA --out_dir <dir>

Combined output  daily_h_l_all.csv  : columns  ticker, day, highest_price, lowest_price, execution_sum
Per-stock output daily_h_l_<STOCK>.csv: columns  day, highest_price, lowest_price, execution_sum

PROC-FILE COLUMN LAYOUT (verified on the Jan-2026 squashfs proc .npy, shape [N, 14]):
    COL_EVENT_TYPE = 1     # event_type (4 == execution/trade)
    COL_PRICE      = 3     # ABSOLUTE price (col 4 is price-relative-to-mid)
    COL_SIZE       = 5     # order size
Daily H/L are taken from EXECUTION prices (event_type == 4) only -> true traded high/low,
robust to far-away resting limit orders.
"""
import os, re, csv, glob, argparse
from multiprocessing import Pool
import numpy as np

COL_EVENT_TYPE = 1
COL_PRICE      = 3
COL_SIZE       = 5
EXECUTION_EVENT_TYPE = 4
DAY_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def stats_one_file(f, col_price):
    """(day, highest_price, lowest_price, execution_sum) for one proc .npy, or None."""
    try:
        a = np.load(f, mmap_mode='r')
        block = np.asarray(a[:, [COL_EVENT_TYPE, col_price, COL_SIZE]])   # single disk pass
        ev, price, size = block[:, 0], block[:, 1].astype(np.float64), block[:, 2]
        is_exec = ev == EXECUTION_EVENT_TYPE
        exec_px = price[is_exec]; exec_px = exec_px[exec_px > 0]
        if exec_px.size == 0:
            return None
        m = DAY_RE.search(os.path.basename(f))
        return dict(day=(m.group(1) if m else os.path.basename(f)),
                    highest_price=float(exec_px.max()), lowest_price=float(exec_px.min()),
                    execution_sum=int(size[is_exec].sum()))
    except Exception as e:
        print(f'  WARN {os.path.basename(f)}: {e}', flush=True)
        return None


def rows_for_ticker(args):
    """All per-day rows for one ticker (used by the parallel --all path)."""
    mnt, ticker, col_price = args
    out = []
    for f in sorted(glob.glob(os.path.join(mnt, ticker, '*message*proc.npy'))):
        r = stats_one_file(f, col_price)
        if r is not None:
            out.append(dict(ticker=ticker, **r))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mnt', required=True, help='mounted squashfs root (one subdir per ticker)')
    ap.add_argument('--stock', default=None, help='single ticker; omit for ALL tickers (combined file)')
    ap.add_argument('--out_dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--price-col', type=int, default=COL_PRICE)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- single ticker -> per-stock file (for the analysis loader) ----
    if args.stock:
        rows = [r for r in rows_for_ticker((args.mnt, args.stock, args.price_col))]
        rows.sort(key=lambda r: r['day'])
        out = os.path.join(args.out_dir, f'daily_h_l_{args.stock}.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['day', 'highest_price', 'lowest_price', 'execution_sum'])
            w.writeheader()
            w.writerows({k: r[k] for k in ('day', 'highest_price', 'lowest_price', 'execution_sum')} for r in rows)
        print(f'Wrote {out} ({len(rows)} days)', flush=True)
        return

    # ---- ALL tickers -> one combined file (parallel) ----
    tickers = sorted(d for d in os.listdir(args.mnt) if os.path.isdir(os.path.join(args.mnt, d)))
    print(f'{len(tickers)} tickers, {args.workers} workers', flush=True)
    all_rows = []
    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(
                rows_for_ticker, [(args.mnt, t, args.price_col) for t in tickers], chunksize=1)):
            all_rows.extend(rows)
            if (i + 1) % 25 == 0:
                print(f'  {i+1}/{len(tickers)} tickers done', flush=True)
    all_rows.sort(key=lambda r: (r['ticker'], r['day']))

    out = os.path.join(args.out_dir, 'daily_h_l_all.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['ticker', 'day', 'highest_price', 'lowest_price', 'execution_sum'])
        w.writeheader(); w.writerows(all_rows)
    print(f'Wrote {out} ({len(all_rows)} rows, {len(tickers)} tickers)', flush=True)


if __name__ == '__main__':
    main()
