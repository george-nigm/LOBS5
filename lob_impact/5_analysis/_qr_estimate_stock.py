"""
Estimate queue-reactive (Huang-Lehalle-Rosenbaum) params for one stock from grid
conditioning windows (data_cond), mirroring _hawkes_estimate_stock.py.

Usage:
    JAX_PLATFORMS=cpu python _qr_estimate_stock.py <STOCK> <GRID> [N] [TICK]

Reads up to N (message, orderbook) conditioning windows from
  <GRID>/<STOCK>-Historic-beta/{buy,sell}/exp_*/data_cond/*orderbook*.csv
estimates QR params (queue_reactive.estimate), and writes
  <REPO>/lob_impact/3_scenarios/qr_params/qr_params_<STOCK>.pkl
"""
import sys, glob, os

REPO = '/home/u6gb/georgenigm.u6gb/LOBS5'
for p in (REPO, REPO + '/Alphatrade', REPO + '/lob_bench/hawkes_model'):
    sys.path.insert(0, p)

import numpy as np
import queue_reactive as qr
from lob_bench import data_loading as dl

STOCK = sys.argv[1]
GRID = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
TICK = int(sys.argv[4]) if len(sys.argv) > 4 else 100

# data_cond is the REAL LOBSTER window every scenario conditions on, so it is identical
# whichever model's folder it sits in. Prefer Historic (always present for the established
# stocks); for a stock whose Historic fleet has not run yet (AAPL) fall back to any
# <STOCK>-*-beta folder that already staged cond windows, so estimation is not blocked.
books = []
for side in ('buy', 'sell'):
    books += sorted(glob.glob(f'{GRID}/{STOCK}-Historic-beta/{side}/exp_*/data_cond/*orderbook*.csv'))
if not books:
    for side in ('buy', 'sell'):
        books += sorted(glob.glob(f'{GRID}/{STOCK}-*-beta/{side}/exp_*/data_cond/*orderbook*.csv'))
    print(f'{STOCK}: no Historic cond windows; falling back to any model folder', flush=True)
books = books[:N]
print(f'{STOCK}: estimating QR params from {len(books)} conditioning windows', flush=True)

windows = []
for i, b in enumerate(books):
    try:
        m = b.replace('orderbook', 'message')
        msg_df = dl.load_message_df(m)
        book_df = dl.load_book_df(b)
        windows.append((msg_df, book_df))
    except Exception as e:
        print('skip', os.path.basename(b), repr(e)[:80])
    if (i + 1) % 100 == 0:
        print(f'  loaded {i + 1}/{len(books)}', flush=True)

print(f'{STOCK}: {len(windows)} windows loaded; estimating...', flush=True)
params = qr.estimate(windows, tick_size=TICK)

out = f'{REPO}/lob_impact/3_scenarios/qr_params/qr_params_{STOCK}.pkl'
qr.save_params(params, out)

lam = np.asarray(params['lam'])
print(f'=== {STOCK} QR: n_windows={params["n_windows"]} total_time={params["total_time"]:.4g}s '
      f'aes={params["aes"]:.1f} p_inside={params["p_inside"]:.3f}')
print(f'    event_counts(MO,LO,CO)={np.asarray(params["event_counts"]).tolist()}')
print(f'    deep_rate(LO,CO)={np.round(np.asarray(params["deep_rate"]), 4).tolist()}')
for g, name in [(0, 'MO'), (1, 'LO'), (2, 'CO')]:
    print(f'    lam[{name}] level1 by queue-bin: {np.round(lam[g, 0], 3).tolist()}')
print('saved', out)
