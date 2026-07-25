#!/usr/bin/env python
"""
Export proc .npy day files back to standard LOBSTER csv (message 6-col +
orderbook 40-col, no headers) in the directory layout DeepMarket's training
pipeline expects:

  <out>/<STOCK>/<STOCK>_<first>_<last>/<STOCK>_<date>_34200000_57600000_message_10.csv
                                       <STOCK>_<date>_34200000_57600000_orderbook_10.csv

Message columns: time("s.nnnnnnnnn"), event_type, order_id, size, price, direction(+-1).
Used for the v4 external-baseline retrains (TRADES / DeepMarket-CGAN on AAPL Jan-2026).

Usage: python export_lobster_csv.py --mnt <proc_dir_with_npy> --stock AAPL --out <dir> [--max_days N]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as onp
import pandas as pd

DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def export_day(msg_path: Path, book_path: Path, out_dir: Path, stock: str, date: str):
    m = onp.load(msg_path, mmap_mode='r')
    b = onp.load(book_path, mmap_mode='r')
    n = min(m.shape[0], b.shape[0])
    m = onp.asarray(m[:n])
    b = onp.asarray(b[:n])

    # 14-col decoded -> LOBSTER 6-col (same field mapping as msg_to_lobster_format)
    time_s = m[:, 8].astype(onp.int64)
    time_ns = m[:, 9].astype(onp.int64)
    time_str = pd.Series(time_s).astype(str) + '.' + pd.Series(time_ns).astype(str).str.zfill(9)
    df = pd.DataFrame({
        'time': time_str,
        'event_type': m[:, 1].astype(onp.int64),
        'order_id': m[:, 0].astype(onp.int64),
        'size': m[:, 5].astype(onp.int64),
        'price': m[:, 3].astype(onp.int64),
        'direction': 2 * m[:, 2].astype(onp.int64) - 1,
    })
    base = f'{stock}_{date}_34200000_57600000'
    df.to_csv(out_dir / f'{base}_message_10.csv', index=False, header=False)
    pd.DataFrame(b[:, 3:43].astype(onp.int64)).to_csv(
        out_dir / f'{base}_orderbook_10.csv', index=False, header=False)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mnt', required=True, help='dir with <stock> proc npy day files')
    ap.add_argument('--stock', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_days', type=int, default=0, help='0 = all days')
    args = ap.parse_args()

    src = Path(args.mnt)
    msg_files = sorted(src.glob('*message*.npy'))
    book_files = sorted(src.glob('*book*.npy'))
    if not msg_files:
        print(f'ERROR: no message npy in {src}', file=sys.stderr)
        sys.exit(1)
    pairs = []
    books_by_date = {DATE_RE.search(f.name).group(1): f for f in book_files if DATE_RE.search(f.name)}
    for mf in msg_files:
        mm = DATE_RE.search(mf.name)
        if mm and mm.group(1) in books_by_date:
            pairs.append((mm.group(1), mf, books_by_date[mm.group(1)]))
    if args.max_days:
        pairs = pairs[:args.max_days]

    first, last = pairs[0][0], pairs[-1][0]
    out_dir = Path(args.out) / args.stock / f'{args.stock}_{first}_{last}'
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for date, mf, bf in pairs:
        n = export_day(mf, bf, out_dir, args.stock, date)
        total += n
        print(f'{date}: {n} rows', flush=True)
    print(f'EXPORT_DONE {args.stock}: {len(pairs)} days, {total} messages -> {out_dir}')


if __name__ == '__main__':
    main()
