#!/usr/bin/env python3.11
"""
Compute msgs_btw for the whole S&P500 universe from a mounted s5e squashfs shard.

msgs_btw = int( (1/eta - 1) / trade_frac )  with eta=0.10  -> int(9/trade_frac)
  (child = trade_p50 cancels, see lob_impact/compute_per_day_params.py)

trade_frac = #(event_type==4) / #messages     (event_type col=1, size col=5)

Outputs (to --out_dir):
  msgs_btw_sp500_perday.csv      one row per (ticker, day)
  msgs_btw_sp500_aggregated.csv  one row per ticker (pooled over all days)
  msgs_btw_sp500_hist.png        distribution histograms for clustering
"""
import os, re, csv, glob, argparse
from multiprocessing import Pool
import numpy as np

COL_EVENT_TYPE = 1
COL_SIZE       = 5
ETA            = 0.10
NUM            = (1.0 / ETA - 1.0)   # = 9 for eta=10%
DAY_RE         = re.compile(r'(\d{4}-\d{2}-\d{2})')


def stats_one_file(f):
    """Return (day, n_msgs, n_trades, p50) for one message .npy, or None."""
    try:
        a = np.load(f, mmap_mode='r')
        block = np.asarray(a[:, [COL_EVENT_TYPE, COL_SIZE]])  # single disk pass
        ev = block[:, 0]; sz = block[:, 1]
        n_msgs = int(ev.shape[0])
        mask = ev == 4
        n_tr = int(mask.sum())
        p50 = float(np.percentile(sz[mask].astype(np.float64), 50)) if n_tr > 0 else float('nan')
        m = DAY_RE.search(os.path.basename(f))
        return (m.group(1) if m else '?', n_msgs, n_tr, p50)
    except Exception as e:
        print(f'  WARN {os.path.basename(f)}: {e}', flush=True)
        return None


def process_ticker(args):
    mnt, ticker = args
    files = sorted(glob.glob(os.path.join(mnt, ticker, '*message*proc.npy')))
    per_day = []
    for f in files:
        r = stats_one_file(f)
        if r is None:
            continue
        day, n_msgs, n_tr, p50 = r
        if n_msgs == 0 or n_tr == 0:
            continue
        tf = n_tr / n_msgs
        mb = int(NUM / tf)
        per_day.append(dict(ticker=ticker, day=day, n_msgs=n_msgs, n_trades=n_tr,
                            trade_frac=tf, p50_mo_volume=p50, msgs_btw=mb))
    if not per_day:
        return ticker, [], None
    n_msgs_tot = sum(d['n_msgs'] for d in per_day)
    n_tr_tot   = sum(d['n_trades'] for d in per_day)
    tf_pool    = n_tr_tot / n_msgs_tot
    mb_agg     = int(NUM / tf_pool)
    mb_days    = np.array([d['msgs_btw'] for d in per_day], float)
    p50s       = np.array([d['p50_mo_volume'] for d in per_day], float)
    agg = dict(
        ticker=ticker, n_days=len(per_day), n_msgs=n_msgs_tot, n_trades=n_tr_tot,
        trade_frac=tf_pool, p50_mo_volume=float(np.median(p50s)),
        msgs_btw=mb_agg, gen_100ins=100 * mb_agg,
        msgs_btw_day_mean=float(mb_days.mean()), msgs_btw_day_std=float(mb_days.std()),
    )
    return ticker, per_day, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mnt', required=True, help='mounted squashfs root')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--workers', type=int, default=96)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tickers = sorted(d for d in os.listdir(args.mnt)
                     if os.path.isdir(os.path.join(args.mnt, d)))
    print(f'{len(tickers)} tickers, {args.workers} workers', flush=True)

    perday_rows, agg_rows = [], []
    with Pool(args.workers) as pool:
        for i, (tk, pd, ag) in enumerate(
                pool.imap_unordered(process_ticker,
                                    [(args.mnt, t) for t in tickers], chunksize=1)):
            perday_rows.extend(pd)
            if ag is not None:
                agg_rows.append(ag)
            if (i + 1) % 25 == 0:
                print(f'  {i+1}/{len(tickers)} done', flush=True)

    perday_rows.sort(key=lambda r: (r['ticker'], r['day']))
    agg_rows.sort(key=lambda r: r['msgs_btw'])  # sort by msgs_btw -> easy to read clusters

    pf = os.path.join(args.out_dir, 'msgs_btw_sp500_perday.csv')
    with open(pf, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['ticker', 'day', 'n_msgs', 'n_trades',
                                           'trade_frac', 'p50_mo_volume', 'msgs_btw'])
        w.writeheader(); w.writerows(perday_rows)

    af = os.path.join(args.out_dir, 'msgs_btw_sp500_aggregated.csv')
    with open(af, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['ticker', 'n_days', 'n_msgs', 'n_trades',
                                           'trade_frac', 'p50_mo_volume', 'msgs_btw',
                                           'gen_100ins', 'msgs_btw_day_mean',
                                           'msgs_btw_day_std'])
        w.writeheader(); w.writerows(agg_rows)

    print(f'\nWrote {pf} ({len(perday_rows)} rows)\nWrote {af} ({len(agg_rows)} tickers)', flush=True)

    # ---- histograms for clustering ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mb = np.array([r['msgs_btw'] for r in agg_rows], float)
        tf = np.array([r['trade_frac'] * 100 for r in agg_rows], float)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        ax[0].hist(mb, bins=50, color='#2F5DA3', edgecolor='white')
        ax[0].set_title(f'msgs_btw distribution (S&P, n={len(mb)}, eta=10%)')
        ax[0].set_xlabel('msgs_btw = 9 / trade_frac'); ax[0].set_ylabel('# stocks')
        for q in (0.25, 0.5, 0.75):
            v = np.quantile(mb, q)
            ax[0].axvline(v, color='#C0392B', ls='--', lw=1)
            ax[0].text(v, ax[0].get_ylim()[1]*0.95, f'Q{int(q*100)}={v:.0f}',
                       rotation=90, va='top', fontsize=8, color='#C0392B')
        ax[1].hist(tf, bins=50, color='#2E7D52', edgecolor='white')
        ax[1].set_title('trade_frac distribution (%)')
        ax[1].set_xlabel('trade_frac (% of messages that are MO)'); ax[1].set_ylabel('# stocks')
        fig.tight_layout()
        hp = os.path.join(args.out_dir, 'msgs_btw_sp500_hist.png')
        fig.savefig(hp, dpi=130); print(f'Wrote {hp}', flush=True)
    except Exception as e:
        print(f'  histogram skipped: {e}', flush=True)


if __name__ == '__main__':
    main()
