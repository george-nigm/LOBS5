#!/usr/bin/env python3
"""
Decay / relaxation analysis (Shape II: num_insertions=10, num_coolings=100) for the new grid.

Builds the master impact curve in normalized event-time u (u=1 at the metaorder peak = last
insertion, u>1 = relaxation), combining buy and sell on the SAME windows:
    combined_j(t) = ( buy_ret_j(t) - sell_ret_j(t) ) / 2 ,   ret = (mid_t - mid_0)/mid_0
then averages over samples and fits the post-peak decay
    m(u)/m(1) = c * (1+(u-1))^(-gamma)        gamma = decay exponent
plus the relaxation ratio m(u=3)/m(u=1) (permanent component). Overlays all models.

  python 5_analysis/decay/decay_master_curve.py --grid <root> --stock EA \
         --models Historic,Heuristic,CST,Mamba3 --out decay_EA.png
"""
import os, glob, csv, re, argparse
import numpy as np
from scipy import optimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENTINEL = 2147483647
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
RID_RE = re.compile(r'_real_id_(\d+)_')
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'CST': '#27AE60',
          'Mamba3': '#2F5DA3', 'S5': '#E67E22', 'OW': '#6A1B9A', 'QR': '#00ACC1'}
N_U = 200


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def _aggr_by_day(side_dir):
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def _mid_ret(ob):
    a = _read(ob)
    if a.ndim != 2 or a.shape[1] < 4:
        return None
    ask, bid = a[:, 0].astype(float), a[:, 2].astype(float)
    bad = (ask >= SENTINEL) | (bid >= SENTINEL) | (ask <= 0) | (bid <= 0)
    mid = (ask + bid) / 2.0
    mid[bad] = np.nan
    m0 = mid[0] if np.isfinite(mid[0]) and mid[0] > 0 else np.nanmedian(mid[:5])
    if not np.isfinite(m0) or m0 <= 0:
        return None
    return (mid - m0) / m0


def _side_traj(side_dir):
    """(date,rid) -> (mid_ret trajectory, peak_step L)."""
    aggr = _aggr_by_day(side_dir)
    out = {}
    for ob in sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'), recursive=True)):
        bn = os.path.basename(ob)
        md, mr = DATE_RE.search(bn), RID_RE.search(bn)
        if not md or not mr:
            continue
        day, rid = md.group(1), mr.group(1)
        idx = aggr.get(day)
        if idx is None or len(idx) < 2:
            continue
        ret = _mid_ret(ob)
        if ret is None:
            continue
        L = int(idx[-1])  # last insertion = metaorder peak
        if L < 2 or L >= len(ret):
            continue
        out[(day, rid)] = (ret, L)
    return out


def master_curve(grid, exp):
    buy = _side_traj(os.path.join(grid, exp, 'buy'))
    sell = _side_traj(os.path.join(grid, exp, 'sell'))
    keys = sorted(set(buy) & set(sell))
    if not keys:
        return None
    umax = 0.0
    for k in keys:
        (br, Lb), (sr, Ls) = buy[k], sell[k]
        ml = min(len(br), len(sr))
        umax = max(umax, (ml - 1) / Lb)
    umax = min(umax, 6.0)
    u_grid = np.linspace(0, umax, N_U)
    curves = []
    for k in keys:
        (br, L), (sr, _) = buy[k], sell[k]
        ml = min(len(br), len(sr))
        combined = (br[:ml] - sr[:ml]) / 2.0
        curves.append(np.interp(u_grid, np.arange(ml) / L, combined))
    arr = np.array(curves)
    return dict(u=u_grid, mean=np.nanmean(arr, axis=0), n=len(keys))


def relaxation_ratio(mc, u_peak=1.0, u_final=3.0):
    u, m = mc['u'], mc['mean']
    pi, fi = np.argmin(np.abs(u - u_peak)), np.argmin(np.abs(u - u_final))
    return m[fi] / m[pi] if abs(m[pi]) > 1e-10 else np.nan


def fit_decay(mc, u_peak=1.0):
    u, m = mc['u'], mc['mean']
    pi = np.argmin(np.abs(u - u_peak))
    pu, py = u[pi:] - u_peak, m[pi:]
    if len(py) < 5 or abs(py[0]) < 1e-10:
        return np.nan
    yn = py / py[0]
    try:
        popt, _ = optimize.curve_fit(lambda u, g, c: c * (1 + u) ** (-g), pu, yn, p0=[0.5, 1.0], maxfev=5000)
        return float(popt[0])
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,OW,CST,Mamba3,QR')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    models = [m for m in args.models.split(',') if m]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for model in models:
        exp = f'{args.stock}-{model}-relaxation'
        mc = master_curve(args.grid, exp)
        if mc is None:
            print(f'{exp}: no data'); continue
        g = fit_decay(mc); rr = relaxation_ratio(mc)
        c = COLORS.get(model, '#444')
        ax.plot(mc['u'], mc['mean'] * 1e4, '-', color=c, lw=1.8,
                label=f'{model}: γ={g:.2f}, perm={rr:.2f} (n={mc["n"]})')
        print(f'{exp}: peak={mc["mean"][np.argmin(np.abs(mc["u"]-1))]*1e4:.2f}bps  γ={g:.3f}  relax_ratio={rr:.3f}  n={mc["n"]}')
    ax.axvline(1.0, color='k', ls=':', lw=1, alpha=0.6)
    ax.text(1.02, ax.get_ylim()[1]*0.92, 'peak (end of metaorder)', fontsize=8)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('normalized event time  u   (u=1 at peak, u>1 = relaxation)')
    ax.set_ylabel('mean combined mid impact  (bps)')
    ax.set_title(f'{args.stock} — impact build-up & decay (Shape II), by model')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', f'decay_{args.stock}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
