import sys, glob
REPO='/home/u6gb/georgenigm.u6gb/LOBS5'
for p in (REPO, REPO+'/Alphatrade', REPO+'/lob_bench/cst_model'): sys.path.insert(0, p)
import numpy as np
import param_estimation as pe
DC=sys.argv[1]
books=sorted(glob.glob(DC+'/*orderbook*.csv'))[:3]
print('CST param estimation on', len(books), 'EA orderbook files (L10, tick=100)', flush=True)
b=books[0]
p = pe.estimate_data_file(b, save_dir=None, tick_size=100, num_ticks=500, save=False)
print('--- params for', b.rsplit('/',1)[1], '---')
for k,v in p.items():
    a=np.asarray(v)
    if a.ndim>0 and a.size>1: print(f'  {k}: array{a.shape} mean={np.nanmean(a):.4g} nonzero={int((a!=0).sum())}')
    else: print(f'  {k}: {float(a):.5g}')
print('OK')
