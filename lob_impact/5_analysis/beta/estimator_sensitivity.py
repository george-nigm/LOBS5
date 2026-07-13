#!/usr/bin/env python3
"""
Methodology-audit items 5-7 (one grid pass per stock; results to json + printed tables):

5. ESTIMATOR SENSITIVITY (Bacry et al 2015 pattern log-log 0.54 / L2 0.45 / L1 0.40):
   delta fitted three ways on the same signed per-point data y = I/sigma vs q = Q/V —
   per-point log-log on I>0 (biased), direct L2 on signed raw (a(d) closed-form, grid over d),
   direct L1 (a via median of y/q^d).
6. ALMGREN COST DECOMPOSITION (J = I/2 + temporary): per rollout, realized VWAP slippage
   J = sign*(VWAP - p0)/p0 from the child executions vs end impact I = sign*(mid_end - p0)/p0;
   report <J>/<I> per model (Almgren: J - I/2 = temporary cost, exponent 0.6 in rate).
7. DURATION COVARIATE (Bacry: impact ~ sqrt(participation) / T^0.25): per-day regression of
   ln<|I_end|> on ln(eta_day) and ln(T_day) (2-covariate OLS); report the T coefficient.

  python estimator_sensitivity.py --grid <root> --stock EA --daily <daily.csv> --models ...
"""
import os, re, glob, json, argparse
import numpy as np
from beta_grid import _read, _aggr_by_day, DATE_RE
from vol_estimators import daily_sigmas

SENTINEL = 2147483647


def scan(side_dir, sign, sig, stock):
    """Per rollout: (day, I_end, J_vwap, eta_actual, T_msgs) + per-insertion (q=Q/V, y=I/sigma, k)."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr_by_day = _aggr_by_day(side_dir)
    roll, pts = [], []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        day = m.group(1)
        aggr = aggr_by_day.get(day)
        sg = sig.get((stock, day), {})
        V, s_park = sg.get('V', np.nan), sg.get('parkinson', np.nan)
        if aggr is None or not np.isfinite(V) or not np.isfinite(s_park):
            continue
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask, bid = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask >= SENTINEL) | (bid >= SENTINEL) | (ask <= 0) | (bid <= 0)
        mid = (ask + bid) / 2.0
        mid[bad] = np.nan
        L = len(mid)
        idx = aggr[aggr < L]
        if len(idx) < 2 or idx[0] < 1:
            continue
        ref = mid[idx[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        mf = ob.replace('orderbook', 'message')
        if not os.path.exists(mf):
            continue
        mm = _read(mf)
        if mm.ndim != 2 or mm.shape[1] < 6 or not np.all(mm[idx, 1] == 4) or np.any(mm[idx, 3] <= 0):
            continue
        szs, prs = mm[idx, 3], mm[idx, 4]
        Q = np.cumsum(szs)
        for k, step in enumerate(idx):
            I = sign * (mid[step] - ref) / ref
            if np.isfinite(I) and Q[k] > 0:
                pts.append((Q[k] / V, I / s_park, k + 1))
        Iend = sign * (mid[idx[-1]] - ref) / ref
        vwap = float(np.dot(szs, prs) / szs.sum())
        J = sign * (vwap - ref) / ref
        is_ex = np.isin(mm[:idx[-1] + 1, 1], (4, 5))
        tot_vol = float(mm[:idx[-1] + 1, 3][is_ex].sum())
        eta = float(Q[-1] / tot_vol) if tot_vol > 0 else np.nan
        T = int(idx[-1] - idx[0] + 1)
        if np.isfinite(Iend):
            roll.append((day, Iend / s_park, J / s_park, eta, T))
    return roll, pts


def fit_three_ways(q, y):
    """(log-log I>0, direct L2 signed, direct L1 signed) over delta grid."""
    pos = y > 0
    ll = np.nan
    if pos.sum() > 30:
        ll = float(np.polyfit(np.log(q[pos]), np.log(y[pos]), 1)[0])
    grid = np.arange(0.05, 1.51, 0.01)
    best2, best1 = (np.inf, np.nan), (np.inf, np.nan)
    for d in grid:
        qd = q ** d
        a2 = float(np.dot(y, qd) / np.dot(qd, qd))
        l2 = float(np.sum((y - a2 * qd) ** 2))
        if l2 < best2[0]:
            best2 = (l2, d)
        a1 = float(np.median(y / qd))
        l1 = float(np.sum(np.abs(y - a1 * qd)))
        if l1 < best1[0]:
            best1 = (l1, d)
    return ll, best2[1], best1[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,Propagator,Hawkes,'
                                        'Mamba3,GDN,Mamba3_4k,S5_4k')
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    out = {}
    for m in [x for x in args.models.split(',') if x]:
        roll, pts = [], []
        for side, sign in (('buy', +1), ('sell', -1)):
            r, p = scan(os.path.join(args.grid, f'{args.stock}-{m}-beta', side), sign, sig, args.stock)
            roll += r; pts += p
        if not pts:
            print(f'{m}: no data', flush=True); continue
        q = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        ok = np.isfinite(q) & np.isfinite(y) & (q > 0)
        ll, l2, l1 = fit_three_ways(q[ok], y[ok])
        # Almgren: per-model means
        Ie = np.array([r[1] for r in roll]); J = np.array([r[2] for r in roll])
        eta = np.array([r[3] for r in roll]); T = np.array([r[4] for r in roll], float)
        days = np.array([r[0] for r in roll])
        JI = float(np.nanmean(J) / np.nanmean(Ie)) if np.nanmean(Ie) != 0 else np.nan
        # duration covariate: per-day means, ln|I| ~ b0 + b_eta ln eta + b_T ln T
        b_eta = b_T = np.nan
        du = np.unique(days)
        if len(du) >= 8:
            Id = np.array([np.nanmean(Ie[days == d]) for d in du])
            ed = np.array([np.nanmean(eta[days == d]) for d in du])
            Td = np.array([np.nanmean(T[days == d]) for d in du])
            keep = np.isfinite(Id) & (Id > 0) & np.isfinite(ed) & (ed > 0)
            if keep.sum() >= 6:
                X = np.column_stack([np.ones(keep.sum()), np.log(ed[keep]), np.log(Td[keep])])
                coef, *_ = np.linalg.lstsq(X, np.log(Id[keep]), rcond=None)
                b_eta, b_T = float(coef[1]), float(coef[2])
        out[m] = dict(loglog=ll, L2=l2, L1=l1, J_over_I=JI,
                      J_mean=float(np.nanmean(J)), I_mean=float(np.nanmean(Ie)),
                      eta_mean=float(np.nanmean(eta)), b_eta=b_eta, b_T=b_T,
                      n_roll=len(roll), n_pts=int(ok.sum()))
        print(f'{m:10s} delta: loglog={ll:+.3f} L2={l2:+.3f} L1={l1:+.3f} | '
              f'J/I={JI:+.2f} (J={np.nanmean(J):+.5f} I={np.nanmean(Ie):+.5f}) | '
              f'eta={np.nanmean(eta):.3f} b_eta={b_eta:+.2f} b_T={b_T:+.2f} '
              f'(Bacry refs: ll 0.54/L2 0.45/L1 0.40; b_T -0.25) n={len(roll)}', flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    od = os.path.join(here, 'results', 'estimator_sensitivity')
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f'estimator_sensitivity_{args.stock}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f'ESTIMATOR_SENSITIVITY_DONE -> {od}/estimator_sensitivity_{args.stock}.json')


if __name__ == '__main__':
    main()
