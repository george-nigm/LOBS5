#!/usr/bin/env python3
"""
Action 2 — daily statistics for market-impact normalization.

Reads a mounted squashfs shard (same --mnt convention as 1_data_prep/compute_sp500_msgs_btw.py:
one subdir per ticker, files <ticker>/*message*proc.npy) and emits, per stock:

    daily_h_l_<STOCK>.csv   with columns:  day, highest_price, lowest_price, execution_sum

consumed downstream by the impact analysis (Parkinson volatility
sigma = ln(highest/lowest) / 1.6651092, and execution_sum for participation rate).

  python 2_daily_stats/compute_daily_stats.py --mnt <mounted_root> --stock EA --out_dir <dir>
  python 2_daily_stats/compute_daily_stats.py --mnt <mounted_root> --stock EA --aggregate

Modes:
  (default) per-day : one row per trading day (what the analysis loader expects).
  --aggregate       : a single pooled row (overall H/L, total execution_sum).

PROC-FILE COLUMN LAYOUT (verified against compute_sp500_msgs_btw.py):
    COL_EVENT_TYPE = 1     # event_type (4 == execution/trade)
    COL_SIZE       = 5     # order size  (NOT col 3 — proc format differs from raw LOBSTER CSV)
    COL_PRICE      = 4     # <-- NOT yet verified for the proc .npy. The raw LOBSTER CSV had
                          #     Price at col 4, but the proc npy is [N, 14] and reorders columns.
                          #     ON FIRST RUN: sanity-check the printed H/L against known prices for
                          #     the stock; if wrong (e.g. normalized/relative), set --price-col or
                          #     confirm the true price column before trusting Parkinson sigma.
"""
import os, re, csv, glob, argparse
import numpy as np

COL_EVENT_TYPE = 1
COL_SIZE       = 5
COL_PRICE      = 4          # see header NOTE — verify on first run
EXECUTION_EVENT_TYPE = 4
PRICE_MIN = 500_000         # LOBSTER price units (1/10000 $); guards padding/sentinel values
PRICE_MAX = 20_000_000
DAY_RE    = re.compile(r'(\d{4}-\d{2}-\d{2})')


def stats_one_file(f, col_price):
    """Return dict(day, highest_price, lowest_price, execution_sum) for one proc .npy."""
    a = np.load(f, mmap_mode='r')
    ev    = np.asarray(a[:, COL_EVENT_TYPE])
    size  = np.asarray(a[:, COL_SIZE])
    price = np.asarray(a[:, col_price]).astype(np.float64)

    valid = price[(price >= PRICE_MIN) & (price <= PRICE_MAX)]
    hi = float(valid.max()) if valid.size else float('nan')
    lo = float(valid.min()) if valid.size else float('nan')
    exec_sum = int(size[ev == EXECUTION_EVENT_TYPE].sum())

    m = DAY_RE.search(os.path.basename(f))
    return dict(day=(m.group(1) if m else os.path.basename(f)),
                highest_price=hi, lowest_price=lo, execution_sum=exec_sum)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mnt', required=True, help='mounted squashfs root (one subdir per ticker)')
    ap.add_argument('--stock', required=True, help='ticker, e.g. EA / NVDA / AMD')
    ap.add_argument('--out_dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--price-col', type=int, default=COL_PRICE,
                    help=f'price column in proc .npy (default {COL_PRICE}; verify on first run)')
    ap.add_argument('--aggregate', action='store_true',
                    help='emit a single pooled row instead of one row per day')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.mnt, args.stock, '*message*proc.npy')))
    if not files:
        raise FileNotFoundError(f'no *message*proc.npy under {os.path.join(args.mnt, args.stock)}')
    print(f'{args.stock}: {len(files)} message files', flush=True)

    rows = []
    for f in files:
        try:
            rows.append(stats_one_file(f, args.price_col))
        except Exception as e:
            print(f'  WARN {os.path.basename(f)}: {e}', flush=True)
    rows.sort(key=lambda r: r['day'])

    os.makedirs(args.out_dir, exist_ok=True)
    if args.aggregate:
        his = [r['highest_price'] for r in rows if not np.isnan(r['highest_price'])]
        los = [r['lowest_price']  for r in rows if not np.isnan(r['lowest_price'])]
        rows = [dict(day='ALL', highest_price=max(his) if his else float('nan'),
                     lowest_price=min(los) if los else float('nan'),
                     execution_sum=sum(r['execution_sum'] for r in rows))]

    out = os.path.join(args.out_dir, f'daily_h_l_{args.stock}.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['day', 'highest_price', 'lowest_price', 'execution_sum'])
        w.writeheader(); w.writerows(rows)

    # Parkinson sigma summary (sanity-check the price column here!)
    valid = [r for r in rows if not np.isnan(r['highest_price']) and r['lowest_price'] > 0]
    if valid:
        ln_hl = np.log([r['highest_price'] / r['lowest_price'] for r in valid])
        print(f'  H/L sample: hi={valid[0]["highest_price"]:.0f} lo={valid[0]["lowest_price"]:.0f} '
              f'| mean Parkinson sigma = {float(np.mean(ln_hl) / 1.6651092):.5f}')
        print('  ^ verify these prices look like real prices for this stock (see header NOTE).')
    print(f'Wrote {out} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    main()
