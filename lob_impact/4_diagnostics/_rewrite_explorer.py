"""One-shot editor: re-point interactive_explorer.ipynb at the new-pipeline grid
(3_scenarios/results/grid/<STOCK>-<MODEL>-<beta|relaxation>/<buy|sell>/exp_*).
Edits cells 2-5 (loader) + light patches to 9 and 12. Inner file format unchanged."""
import json, sys
from pathlib import Path

NB = Path(__file__).with_name('interactive_explorer.ipynb')
nb = json.load(open(NB))


def L(s):
    return [l + '\n' for l in s.strip('\n').split('\n')]


CELL2 = r'''
TICK_SIZE = 100
MAX_SAMPLES = 2048

# ── New-pipeline grid (Action 3) ──
# results/grid/<STOCK>-<MODEL>-<beta|relaxation>/<buy|sell>/exp_*/{data_cond,data_gen,aggressive_indices.csv}
def _find_lob_impact(start):
    for q in [start, *start.parents]:
        if (q / '3_scenarios' / 'results' / 'grid').exists():
            return q
    return start

LOB_IMPACT = _find_lob_impact(Path.cwd())
GRID_ROOT = LOB_IMPACT / '3_scenarios' / 'results' / 'grid'

# Daily H/L table → Parkinson sigma (newest run wins); keyed by (ticker, day).
_dhl = sorted((LOB_IMPACT / '2_daily_stats' / 'results').glob('daily_*/daily_h_l_all.csv'))
DHL_PATH = _dhl[-1] if _dhl else None
DAILY_HL = pd.read_csv(DHL_PATH) if DHL_PATH else pd.DataFrame()
_HL = {(str(r['ticker']), str(r['day'])): (float(r['highest_price']), float(r['lowest_price']))
       for _, r in DAILY_HL.iterrows()}

COLOR = '#D09A3C'
print(f'LOB_IMPACT : {LOB_IMPACT}')
print(f'GRID_ROOT  : {GRID_ROOT}  exists={GRID_ROOT.exists()}')
print(f'DAILY_HL   : {DHL_PATH}  ({len(_HL)} ticker-days)')
'''

CELL3 = r'''
# ── Data I/O helpers (new-pipeline grid layout) ──
LEAF_RE = re.compile(r'^([A-Za-z0-9]+)-([A-Za-z0-9]+)-(beta|relaxation)$')
FOLDER_PARAMS = {}   # folder name -> dict(stock, model, shape, i, c, mb, V)

def _latest_exp(side_dir):
    exps = sorted(p for p in side_dir.glob('exp_*') if p.is_dir())
    return exps[-1] if exps else None

def _read_cfg(exp_dir):
    import yaml
    f = exp_dir / 'config.yaml'
    return (yaml.safe_load(open(f)) or {}) if f.exists() else {}

def discover_grid(grid_root, stocks=None, models=None, shapes=None):
    rows = []
    for leaf in sorted(grid_root.iterdir()):
        if not leaf.is_dir() or leaf.name == '_configs':
            continue
        m = LEAF_RE.match(leaf.name)
        if not m:
            continue
        stock, model, shape = m.group(1), m.group(2), m.group(3)
        if (stocks and stock not in stocks) or (models and model not in models) \
           or (shapes and shape not in shapes):
            continue
        buy_exp, sell_exp = _latest_exp(leaf / 'buy'), _latest_exp(leaf / 'sell')
        if buy_exp is None or sell_exp is None:
            continue
        cfg = _read_cfg(buy_exp)
        # Map new identity onto the old (i, c, mb, V) schema so downstream cells just work.
        ins  = int(cfg.get('num_insertions', 0))
        cool = int(cfg.get('num_coolings', 0))
        mb   = int(cfg.get('n_gen_msgs', 0))
        vol  = int(cfg.get('order_volume', 0))
        rows.append({'folder': leaf.name, 'stock': stock, 'model': model, 'shape': shape,
                     'i': ins, 'c': cool, 'mb': mb, 'V': vol, 'Q_total': ins * vol,
                     'buy_path': buy_exp, 'sell_path': sell_exp})
        FOLDER_PARAMS[leaf.name] = {'stock': stock, 'model': model, 'shape': shape,
                                    'i': ins, 'c': cool, 'mb': mb, 'V': vol}
    return pd.DataFrame(rows)

def parse_folder_params(folder_name):
    p = FOLDER_PARAMS.get(folder_name)
    return (p['i'], p['c'], p['mb'], p['V']) if p else (None, None, None, None)

def compute_midprice(book_array):
    return (book_array[:, 0] + book_array[:, 2]) / 2

def load_aggressive_indices(data_path):
    f = data_path / 'aggressive_indices.csv'
    if not f.exists():
        return np.array([], dtype=int)
    return np.atleast_1d(np.loadtxt(f, dtype=int))

def discover_data_params(data_path, max_samples=None):
    cond_dir = data_path / 'data_cond'
    pat = re.compile(r'^(.+?)_(\d{4}-\d{2}-\d{2})_orderbook_real_id_(\d+)\.csv$')
    samples = []
    for f in cond_dir.glob('*_orderbook_real_id_*.csv'):
        m = pat.match(f.name)
        if m:
            samples.append((m.group(1), m.group(2), int(m.group(3))))
    samples.sort()
    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    return samples

def load_folder_data(data_path, max_samples=None):
    samples = discover_data_params(data_path, max_samples)
    gen_books, gen_msgs, cond_lens = {}, {}, {}
    for ticker, date, sid in samples:
        cond_bp = data_path / f'data_cond/{ticker}_{date}_orderbook_real_id_{sid}.csv'
        gen_bp  = data_path / f'data_gen/{ticker}_{date}_orderbook_real_id_{sid}_gen_id_0.csv'
        gen_mp  = data_path / f'data_gen/{ticker}_{date}_message_real_id_{sid}_gen_id_0.csv'
        if not gen_bp.exists():
            continue
        cond_book = np.loadtxt(cond_bp, delimiter=',')
        gen_book  = np.loadtxt(gen_bp, delimiter=',')
        gen_msg   = np.loadtxt(gen_mp, delimiter=',')
        cond_mp   = data_path / f'data_cond/{ticker}_{date}_message_real_id_{sid}.csv'
        cond_msg  = np.loadtxt(cond_mp, delimiter=',')
        key = (date, sid)
        cond_lens[key] = cond_book.shape[0]
        gen_books[key] = np.vstack([cond_book, gen_book])
        gen_msgs[key]  = np.vstack([cond_msg, gen_msg])
    return gen_books, gen_msgs, cond_lens

def load_all(grid_df):
    all_data = {}
    for _, row in tqdm(grid_df.iterrows(), total=len(grid_df), desc='Loading grid'):
        try:
            bb, bm, bc = load_folder_data(row['buy_path'],  MAX_SAMPLES)
            sb, sm, sc = load_folder_data(row['sell_path'], MAX_SAMPLES)
            all_data[row['folder']] = {
                'buy':  {'books': bb, 'msgs': bm, 'cond_lens': bc},
                'sell': {'books': sb, 'msgs': sm, 'cond_lens': sc},
            }
        except Exception as e:
            print(f'  ERR {row["folder"]}: {e}')
    return all_data

print('Helpers loaded.')
'''

CELL4 = r'''
# ── Discover the new grid & load ──
# Filters default to ALL 12 leaves. Mamba3 = neural model, Historic = replay baseline.
STOCKS = None    # e.g. ['EA', 'NVDA', 'AMD']
MODELS = None    # e.g. ['Mamba3', 'Historic']
SHAPES = None    # e.g. ['beta', 'relaxation']

grid = discover_grid(GRID_ROOT, STOCKS, MODELS, SHAPES).reset_index(drop=True)
print(f'Discovered {len(grid)} configs')
display(grid[['folder', 'stock', 'model', 'shape', 'i', 'c', 'mb', 'V', 'Q_total']])

data = load_all(grid)
print(f'Loaded {len(data)} folders')
'''

# Cell 5: only the sigma source changes (sample_day_map -> daily H/L by (ticker, date)).
CELL5 = r'''
# ── Sigma-normalized master curves (volume time) ──

def compute_master_curve(buy_data, sell_data, folder, aggr_gen,
                         u_max=11.0, n_pts=500):
    i, c, mb, V = parse_folder_params(folder)
    stock = FOLDER_PARAMS.get(folder, {}).get('stock')
    if len(aggr_gen) < 2:
        return None
    s_gen = int(aggr_gen[0])
    e_gen = int(aggr_gen[-1])
    L = e_gen - s_gen
    if L == 0:
        return None
    bb, sb = buy_data['books'], sell_data['books']
    if not bb or not sb:
        return None
    min_len = min(min(b.shape[0] for b in bb.values()),
                  min(b.shape[0] for b in sb.values()))
    junction = list(buy_data['cond_lens'].values())[0]
    u_cap = min(u_max, (min_len - 1 - junction - s_gen) / L)
    if u_cap <= 0:
        return None
    u_grid = np.linspace(0, u_cap, n_pts)

    def side_impacts(books, conds):
        imps = []
        for sid, bk in books.items():
            date = sid[0]                       # date is embedded in the filename
            hl = _HL.get((stock, date))         # per-day H/L from daily_h_l_all.csv
            if hl is None:
                continue
            H = hl[0] / TICK_SIZE
            Lp = hl[1] / TICK_SIZE
            if H <= Lp or Lp <= 0:
                continue
            sigma = np.log(H / Lp) / 0.8325546   # Parkinson sigma
            if sigma <= 0:
                continue
            j = conds[sid]
            s_abs = j + s_gen
            if s_abs < 1 or s_abs >= min_len:
                continue
            mid = compute_midprice(bk[:min_len])
            ref = mid[s_abs - 1]
            if ref <= 0:
                continue
            raw = (mid[s_abs:min_len] - ref) / (ref * sigma)
            u_raw = np.arange(len(raw)) / L
            imps.append(np.interp(u_grid, u_raw, raw))
        return np.array(imps) if imps else None

    bi = side_impacts(bb, buy_data['cond_lens'])
    si = side_impacts(sb, sell_data['cond_lens'])
    if bi is None or si is None:
        return None
    mean = (np.mean(bi, axis=0) - np.mean(si, axis=0)) / 2
    std  = np.sqrt(np.std(bi, axis=0)**2 + np.std(si, axis=0)**2) / 2
    return {'u_grid': u_grid, 'combined_mean': mean, 'combined_std': std,
            'L': L, 'i': i, 'c': c, 'mb': mb, 'V': V, 'Q': i * V,
            'n_buy': bi.shape[0], 'n_sell': si.shape[0]}


# Compute all curves
curves = {}
for _, row in grid.iterrows():
    f = row['folder']
    if f not in data:
        continue
    aggr = load_aggressive_indices(row['buy_path'])
    cc = compute_master_curve(data[f]['buy'], data[f]['sell'], f, aggr)
    if cc is not None:
        curves[f] = cc

print(f'Computed {len(curves)} master curves')
for f, cc in sorted(curves.items(), key=lambda x: x[0]):
    print(f"  {f:28s}  i={cc['i']:3d} c={cc['c']:3d} mb={cc['mb']:4d} V={cc['V']:3d}  "
          f"L={cc['L']:4d}  n_buy={cc['n_buy']}  n_sell={cc['n_sell']}")
'''

new_src = {2: CELL2, 3: CELL3, 4: CELL4, 5: CELL5}
for idx, src in new_src.items():
    cell = nb['cells'][idx]
    assert cell['cell_type'] == 'code', f'cell {idx} not code'
    cell['source'] = L(src)
    cell['outputs'] = []
    cell['execution_count'] = None

# Cell 9: richer config table
c9 = ''.join(nb['cells'][9]['source'])
c9 = c9.replace(
    "configs = grid[['folder','i','c','mb','V','Q_total']].sort_values(['i','mb','V'])",
    "configs = grid[['folder','stock','model','shape','i','c','mb','V','Q_total']].sort_values(['stock','model','shape'])")
nb['cells'][9]['source'] = c9.splitlines(keepends=True)
nb['cells'][9]['outputs'] = []
nb['cells'][9]['execution_count'] = None

# Cell 12: hard-coded index [20] is out of range for 12 configs
c12 = ''.join(nb['cells'][12]['source'])
c12 = c12.replace("FOLDER_KEY = configs['folder'][20]",
                  "FOLDER_KEY = configs['folder'].iloc[len(configs) // 2]")
nb['cells'][12]['source'] = c12.splitlines(keepends=True)
nb['cells'][12]['outputs'] = []
nb['cells'][12]['execution_count'] = None

json.dump(nb, open(NB, 'w'), indent=1, ensure_ascii=False)
print('Rewrote cells 2,3,4,5 + patched 9,12 ->', NB)
