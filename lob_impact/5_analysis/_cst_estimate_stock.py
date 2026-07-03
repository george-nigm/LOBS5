import sys, glob, os
REPO='/home/u6gb/georgenigm.u6gb/LOBS5'
for p in (REPO, REPO+'/Alphatrade', REPO+'/lob_bench/cst_model'): sys.path.insert(0, p)
import numpy as np, param_estimation as pe
STOCK, GRID = sys.argv[1], sys.argv[2]
# N=None (default) -> use EVERY distinct conditioning window. The OLD code capped at 300
# windows of ~500 msgs each and only looked at one model/shape/side, so it pooled barely
# ~340 s of trading time -> per-second rates (mo_mu / lo_lambda / co_theta) were noisy and
# burst-biased -> mis-scaled MO rate + under-estimated LO replenishment -> book can't refill
# -> the depleted side collapses to the get_best_ask()=-1 sentinel -> NEGATIVE impact. Widening
# to all REAL (replay) conditioning windows multiplies Sigma(total_time) by ~10-40x.
N = int(sys.argv[3]) if len(sys.argv) > 3 else None  # None = use everything
# Pool every REAL conditioning window (replay models only; their data_cond is genuine
# historical context). aggregate_params sums counts and times, so this is statistically valid.
books = []
for model in ('Historic', 'Heuristic'):
    for shape in ('beta', 'relaxation'):
        for side in ('buy', 'sell'):
            books += sorted(glob.glob(
                f'{GRID}/{STOCK}-{model}-{shape}/{side}/exp_*/data_cond/*orderbook*.csv'))
# Dedup by filename: the SAME (date, sample_idx) window can appear under several
# model/shape/side folders. Keeping distinct windows avoids re-reading identical files
# (rates are count/time, so duplicates don't change them, but dedup cuts Lustre I/O).
_seen, _uniq = set(), []
for b in books:
    bn = os.path.basename(b)
    if bn not in _seen:
        _seen.add(bn); _uniq.append(b)
books = sorted(_uniq)
books = books if N is None else books[:N]
print(f'{STOCK}: estimating CST params from {len(books)} distinct conditioning windows', flush=True)
params = []
for i, b in enumerate(books):
    try:
        params.append(pe.estimate_data_file(b, save_dir=None, tick_size=100, num_ticks=500, save=False))
    except Exception as e:
        print('skip', os.path.basename(b), repr(e)[:80])
    if (i+1) % 100 == 0: print(f'  {i+1}/{len(books)}', flush=True)
assert len(params) > 0, f'no windows estimated for {STOCK} (all {len(books)} skipped or globs empty) -> cannot aggregate'
out = f'{REPO}/lob_impact/3_scenarios/cst_params/cst_params_{STOCK}.pkl'
os.makedirs(os.path.dirname(out), exist_ok=True)
aggr = pe.aggregate_params(params, save_path=out)
print(f'=== {STOCK} aggregated: mo_mu={float(aggr["mo_mu"]):.5g} lo_alpha={float(aggr["lo_alpha"]):.4g} '
      f'mean_spread={float(aggr["mean_spread"]):.3g} mo_count={int(aggr["mo_count"])} '
      f'lo_size={float(aggr["lo_size"]):.4g} co_theta_finite={bool(np.isfinite(np.asarray(aggr["co_theta"])).all())}')
print('saved', out)
