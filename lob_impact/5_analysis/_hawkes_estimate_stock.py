"""
Estimate multivariate Hawkes order-flow params for one stock from grid conditioning
windows (data_cond), mirroring _cst_estimate_stock.py.

Usage:
    JAX_PLATFORMS=cpu python _hawkes_estimate_stock.py <STOCK> <GRID> [N] [TICK]

Reads up to N (message, orderbook) conditioning windows from
  <GRID>/<STOCK>-Historic-beta/{buy,sell}/exp_*/data_cond/*orderbook*.csv
estimates Hawkes params (hawkes.estimate), and writes
  <REPO>/lob_impact/3_scenarios/hawkes_params/hawkes_params_<STOCK>.pkl
"""
import sys, glob, os

REPO = '/home/u6gb/georgenigm.u6gb/LOBS5'
for p in (REPO, REPO + '/Alphatrade', REPO + '/lob_bench/hawkes_model'):
    sys.path.insert(0, p)

import numpy as np
import hawkes
from lob_bench import data_loading as dl

STOCK = sys.argv[1]
GRID = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
TICK = int(sys.argv[4]) if len(sys.argv) > 4 else 100

books = []
for side in ('buy', 'sell'):
    books += sorted(glob.glob(f'{GRID}/{STOCK}-Historic-beta/{side}/exp_*/data_cond/*orderbook*.csv'))
books = books[:N]
print(f'{STOCK}: estimating Hawkes params from {len(books)} conditioning windows', flush=True)

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
params = hawkes.estimate(windows, tick_size=TICK)

out = f'{REPO}/lob_impact/3_scenarios/hawkes_params/hawkes_params_{STOCK}.pkl'
hawkes.save_params(params, out)

mu = np.asarray(params['mu'])
beta = np.asarray(params['beta'])
G = np.asarray(params['G'])
print(f'=== {STOCK} Hawkes: n_windows={params["n_windows"]} '
      f'spectral_radius={params["spectral_radius"]:.4g} total_time={params["total_time"]:.4g}s')
print(f'    event_counts(MOa,MOb,LOa,LOb,COa,COb)={np.asarray(params["event_counts"]).tolist()}')
print(f'    mu={np.round(mu, 5).tolist()}')
print(f'    beta={np.round(beta, 4).tolist()}')
print(f'    diag(G)={np.round(np.diag(G), 4).tolist()}')
print('saved', out)
