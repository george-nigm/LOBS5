#!/usr/bin/env python3
"""
Full mid-price TRAJECTORY (every message step, not sampled at insertions), averaged over samples,
overlaid per model.  This is "just the mid-price change" the user asked for.

For every generated orderbook we take the signed relative mid move vs the pre-metaorder reference
  m_t = sign * (mid[t] - mid_ref) / mid_ref          (buy: sign=+1, sell: sign=-1, so they add)
where mid_ref = mid just before the first aggressive insertion. We stack all samples (buy+sell),
truncate to the common length, and average per step -> one mean trajectory per model, in bps.
Vertical ticks mark the aggressive-insertion steps.

  python 5_analysis/beta/mid_trajectory.py --grid <root> --stock EA \
         --models Historic,Heuristic,CST,Mamba3 --out mid_trajectory_EA.png
"""
import os, glob, csv, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENTINEL = 2147483647
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'CST': '#27AE60',
          'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'S5': '#E67E22', 'S5_4k': '#E67E22', 'OW': '#6A1B9A', 'QR': '#00ACC1'}


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def _aggr_by_day(side_dir):
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def sqrt_law_curve(daily, per_day_csv, stock, n_ins, n_total_blocks, Lmax, Y, method):
    """Square-root-law expected mid impact (β=0.5), with PER-DAY child(order_volume), mb and V.
       The grid was generated in per-day mode: each day d has its own child_d (order_volume),
       mb_d (messages between insertions) and daily volume V_d. So:
         insertion k lands at step ~ k*mb_d  (day-specific) -> we average the step over days
         Q_k(d) = k * child_d ,   I_exp(k) = mean_d[ Y * sigma_d * sqrt(Q_k(d)/V_d) ] * 1e4  (bps)
       For the decay shape Q peaks after n_ins insertions then stays flat through the coolings
       (√-law gives PEAK/permanent impact, no resilience) -> rise then flat to Lmax.
       Returns (x_steps, y_bps)."""
    import csv as _csv
    from vol_estimators import daily_sigmas
    sig = daily_sigmas(daily)
    pd_rows = {r['day']: r for r in _csv.DictReader(open(per_day_csv))}
    days = []
    for day, r in pd_rows.items():
        s = sig.get((stock, day), {})
        sd, V = s.get(method, np.nan), s.get('V', np.nan)
        try:
            child, mb = float(r['child']), float(r['mb'])
        except (KeyError, ValueError):
            continue
        if np.isfinite(sd) and np.isfinite(V) and V > 0 and child > 0 and mb > 0:
            days.append((sd, V, child, mb))
    if not days or n_ins < 1:
        return None, None
    ks = np.arange(1, n_ins + 1)
    per_day_I = np.array([Y * sd * np.sqrt(ks * child / V) for (sd, V, child, mb) in days])  # [nd, n_ins]
    y = np.nanmean(per_day_I, axis=0) * 1e4                       # bps, rising part
    mb_mean = float(np.mean([mb for (_, _, _, mb) in days]))
    x = ks * mb_mean                                             # mean insertion step across days
    if n_total_blocks > n_ins and Lmax:                         # decay: hold the peak flat to the end
        x = np.append(x, float(Lmax)); y = np.append(y, y[-1])
    return x, y


def propagator_curve(peak, T, x_steps, beta, Lmax):
    """Bouchaud transient-impact (propagator) prediction for a constant-rate metaorder of duration T.
       Kernel G(ℓ)~ℓ^(-β). Convolving a box of trades [0,T] gives
         I(t) ∝ t^(1-β) - max(t-T,0)^(1-β) ,  normalized so I(T)=peak.
       t<=T: builds as t^(1-β) (β=½ -> √t, the √-law build); t>T: power-law decay (relaxation).
       Returns (x, y_bps) on a fine grid out to Lmax."""
    g = 1.0 - beta
    x = np.linspace(1.0, float(Lmax), 1200)
    raw = x**g - np.maximum(x - T, 0.0)**g
    y = peak * raw / (T**g)                                   # normalize: at t=T, raw=T^g -> y=peak
    return x, y


def collect_traj(side_dir, sign):
    """list of (signed-relative mid trajectory in bps, aggr indices, mb_d) + a sample aggr array."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'), recursive=True))
    aggr_by_day = _aggr_by_day(side_dir)
    trajs, any_aggr = [], None
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        aggr = aggr_by_day.get(m.group(1))
        if aggr is None or len(aggr) < 1 or aggr[0] < 1:
            continue
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask_p, bid_p = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask_p >= SENTINEL) | (bid_p >= SENTINEL) | (ask_p <= 0) | (bid_p <= 0)
        mid = (ask_p + bid_p) / 2.0
        mid[bad] = np.nan
        ref = mid[aggr[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        mb = int(np.diff(aggr).min()) if len(aggr) > 1 else 0
        trajs.append((sign * (mid - ref) / ref * 1e4, aggr, mb, m.group(1)))   # bps, +day
        if any_aggr is None:
            any_aggr = aggr
    return trajs, any_aggr


def at_boundaries(I, aggr, mb, n_ins, n_cool):
    """Sample a trajectory at executed-volume boundaries: insertions k=1..n_ins, then
    cooling-window ends j=1..n_cool (spaced mb). Returns nb+1 values, index 0 = 0."""
    nb = n_ins + n_cool
    vals = np.full(nb + 1, np.nan)
    vals[0] = 0.0
    if len(aggr) < 1:
        return vals
    idx = list(aggr[:n_ins])
    last = aggr[min(n_ins, len(aggr)) - 1]
    if n_cool and mb > 0:
        idx += [int(last + j * mb) for j in range(1, n_cool + 1)]
    for j, ix in enumerate(idx, start=1):
        if ix >= len(I) and mb > 0 and ix - (len(I) - 1) <= mb:
            ix = len(I) - 1          # final window boundary lands exactly at the array end
        if ix < len(I):
            vals[j] = I[ix]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,OW,CST,Mamba3,QR')
    ap.add_argument('--shape', default='beta', help='grid exp suffix: beta | relaxation')
    ap.add_argument('--daily', default=None, help='daily H/L CSV; if given (with --per_day_params), overlay √-law')
    ap.add_argument('--per_day_params', default=None, help='per_day_params_<STOCK>.csv (child/mb/V per day)')
    ap.add_argument('--Y', type=float, default=0.8325546,
                help='√-law prefactor I=Y*sigma*sqrt(Q/V) in the SIGMA UNITS USED (default = empirical Y=0.5 of the raw-range convention converted to Parkinson: 0.5*1.665)')
    ap.add_argument('--sigma_method', default='parkinson')
    ap.add_argument('--estimator', default='tmean', choices=['tmean', 'median', 'mean'], help='central trajectory estimator')
    ap.add_argument('--beta_prop', type=float, default=0.5, help='propagator decay exponent (decay shape overlay)')
    ap.add_argument('--trim', type=float, default=0.10, help='trimmed-mean: drop this fraction from EACH tail per step')
    ap.add_argument('--min_frac', type=float, default=0.5, help='only plot steps where >=this fraction of samples reach')
    ap.add_argument('--smooth', type=int, default=0, help='rolling-mean window (steps) on the mean line; 0=off')
    ap.add_argument('--dump_samples', action='store_true',
                    help='also cache the per-sample k-clock matrix <model>_K (+_K_side/_K_day) for spaghetti exhibits')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    models = [m for m in args.models.split(',') if m]
    n_ins_k = 100 if args.shape == 'beta' else 10             # k-clock geometry
    n_cool_k = 0 if args.shape == 'beta' else 100
    nb = n_ins_k + n_cool_k
    fig, ax = plt.subplots(figsize=(11, 5.8))
    figk, axk = plt.subplots(figsize=(11, 5.8))
    ins_steps = None
    Lmax = 0
    cache = {}
    for model in models:
        exp = f'{args.stock}-{model}-{args.shape}'
        tb, ab = collect_traj(os.path.join(args.grid, exp, 'buy'), +1)
        ts, asl = collect_traj(os.path.join(args.grid, exp, 'sell'), -1)
        items = tb + ts
        if not items:
            print(f'{exp}: no data'); continue
        trajs = [x[0] for x in items]
        K = np.vstack([at_boundaries(I, aggr, mb, n_ins_k, n_cool_k)
                       for I, aggr, mb, _day in items])
        smp_side = np.array([+1] * len(tb) + [-1] * len(ts))
        smp_day = np.array([x[3] for x in items])
        Lm = max(len(t) for t in trajs)                       # PAD to longest (was: truncate to shortest = bug)
        M = np.full((len(trajs), Lm), np.nan)
        for i, t in enumerate(trajs):
            M[i, :len(t)] = t
        cnt = np.sum(np.isfinite(M), axis=0)                  # samples reaching each step
        if args.estimator == 'tmean' and args.trim > 0:       # TRIMMED MEAN: drop tail outliers, average the rest
            lo = np.nanpercentile(M, args.trim * 100, axis=0)        # -> smooth (unlike median) + robust to 353-bps blow-ups
            hi = np.nanpercentile(M, (1 - args.trim) * 100, axis=0)
            Mk = np.where((M >= lo) & (M <= hi), M, np.nan)
            cntk = np.sum(np.isfinite(Mk), axis=0)
            mid = np.nanmean(Mk, axis=0)
            band = np.nanstd(Mk, axis=0) / np.sqrt(np.maximum(1, cntk))
        elif args.estimator == 'median':                      # robust but staircases on discrete tick prices
            q1, mid, q3 = (np.nanpercentile(M, q, axis=0) for q in (25, 50, 75))
            band = 1.57 * (q3 - q1) / np.sqrt(np.maximum(1, cnt))
        else:                                                 # plain mean (tail-distorted; for reference)
            mid = np.nanmean(M, axis=0)
            band = np.nanstd(M, axis=0) / np.sqrt(np.maximum(1, cnt))
        keep = cnt >= max(args.min_frac * len(trajs), 30)     # drop ragged thin tail (consistent across models)
        Lkeep = int(np.max(np.where(keep)[0])) + 1 if keep.any() else Lm
        mid[~keep] = np.nan; band[~keep] = np.nan
        if args.smooth > 1:                                   # optional cosmetic smoothing of the central line
            k = np.ones(args.smooth) / args.smooth
            valid = np.isfinite(mid)
            mid[valid] = np.convolve(mid[valid], k, mode='same')
        x = np.arange(Lm)
        c = COLORS.get(model, '#444')
        end = mid[keep][-1] if keep.any() else np.nan
        ax.plot(x, mid, '-', color=c, lw=1.6, label=f'{model} (end: {end:.1f} bps, n={len(trajs)})')
        ax.fill_between(x, mid - 1.96 * band, mid + 1.96 * band, color=c, alpha=0.10)
        if ins_steps is None:
            ins_steps = ab if ab is not None else asl
        Lmax = max(Lmax, Lkeep)
        cache[f'{model}_x'] = x; cache[f'{model}_mid'] = mid; cache[f'{model}_band'] = band; cache[f'{model}_cnt'] = cnt
        # quantify blow-ups: spread of per-sample END impact
        ends = np.array([t[np.isfinite(t)][-1] if np.isfinite(t).any() else np.nan for t in trajs])
        p = np.nanpercentile(ends, [1, 50, 99])
        print(f'{exp}: n={len(trajs)}, Lkeep={Lkeep}, end={end:.2f} bps | per-sample end p1/p50/p99 = {p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f}')
        # --- k-clock (executed-volume) aggregation: equal length by construction, plain mean ---
        kcnt = np.sum(np.isfinite(K), axis=0)
        kmean = np.nanmean(K, axis=0)
        kse = np.nanstd(K, axis=0) / np.sqrt(np.maximum(1, kcnt))
        ks = np.arange(nb + 1)
        axk.plot(ks, kmean, '-', color=c, lw=1.7,
                 label=f'{model} (end: {kmean[-1]:.1f} bps, n={len(items)})')
        axk.fill_between(ks, kmean - 1.96 * kse, kmean + 1.96 * kse, color=c, alpha=0.10)
        cache[f'{model}_k_mean'] = kmean
        cache[f'{model}_k_se'] = kse
        if args.dump_samples:            # per-sample k-clock matrix for spaghetti exhibits
            cache[f'{model}_K'] = K.astype(np.float32)
            cache[f'{model}_K_side'] = smp_side
            cache[f'{model}_K_day'] = smp_day
        print(f'  k-clock: end={kmean[-1]:.2f}±{kse[-1]:.2f} bps (k={nb})')

    if ins_steps is not None and Lmax:
        for s in ins_steps[ins_steps < Lmax]:
            ax.axvline(s, color='k', lw=0.3, alpha=0.12)

    # --- theoretical square-root law (β=0.5) overlay, per-day child/mb/V ---
    if args.daily and args.per_day_params:
        n_ins = len(ins_steps) if ins_steps is not None else 0
        n_blocks = 100 if args.shape == 'beta' else 110   # beta:100 ins; decay:10 ins + 100 cool
        xs, ys = sqrt_law_curve(args.daily, args.per_day_params, args.stock,
                                n_ins, n_blocks, Lmax, args.Y, args.sigma_method)
        if xs is not None:
            keep = xs <= Lmax if Lmax else np.ones_like(xs, bool)
            lbl = '√-law β=0.5 peak' + (' (permanent ref)' if args.shape != 'beta' else f' (Y={args.Y:g})')
            ax.plot(xs[keep], ys[keep], '--', color='k', lw=2.0,
                    label=f'{lbl}, end {ys[keep][-1]:.2f} bps')
            cache['sqrt_x'] = xs; cache['sqrt_y'] = ys
            print(f'√-law: end={ys[keep][-1]:.3f} bps (per-day child, σ={args.sigma_method}, Y={args.Y:g})')
            # same reference on the k-clock: x = k directly (per-day mean, no mb involved)
            xk = np.arange(1, n_ins_k + 1, dtype=float)
            yk = ys[:n_ins_k].copy()
            if args.shape != 'beta':
                xk = np.append(xk, float(nb)); yk = np.append(yk, yk[-1])
            axk.plot(xk, yk, '--', color='k', lw=2.0, label=f'{lbl}, end {yk[-1]:.2f} bps')
            # propagator (Bouchaud transient impact): rise to peak then power-law decay — DECAY shape only
            if args.shape != 'beta' and n_ins >= 1:
                peak = float(ys[n_ins - 1]); T = float(xs[n_ins - 1])
                px, py = propagator_curve(peak, T, xs, args.beta_prop, Lmax)
                ax.plot(px, py, ':', color='#8E44AD', lw=2.2,
                        label=f'propagator β={args.beta_prop:g} (peak {peak:.2f}→{py[-1]:.2f} bps)')
                cache['prop_x'] = px; cache['prop_y'] = py
                print(f'propagator: peak={peak:.3f} @T={T:.0f}, end={py[-1]:.3f} bps (β={args.beta_prop:g})')
                pxk, pyk = propagator_curve(float(yk[n_ins_k - 1]), float(n_ins_k), None,
                                            args.beta_prop, nb)
                axk.plot(pxk, pyk, ':', color='#8E44AD', lw=2.2,
                         label=f'propagator β={args.beta_prop:g} (peak {yk[n_ins_k - 1]:.2f}→{pyk[-1]:.2f} bps)')

    if Lmax:
        ax.set_xlim(0, Lmax)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('message step  (generation time)')
    ax.set_ylabel('mean signed mid-price change  (bps)   buy + (−1)·sell')
    ax.set_title(f'{args.stock} — mid-price trajectory over generation, by model'
                 + ('  [thin ticks = aggressive insertions]' if ins_steps is not None else ''))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'mid_impact', f'mid_trajectory_{args.stock}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)

    # --- the k-clock figure ---
    if args.shape != 'beta':
        axk.axvline(n_ins_k, color='k', lw=0.8, ls=':', alpha=0.6)
    axk.axhline(0, color='k', lw=0.6)
    axk.set_xlim(0, nb)
    axk.set_xlabel('insertion / cooling-window index  k  (children executed — executed-volume units)'
                   if args.shape != 'beta' else
                   'insertion index  k  (children executed — executed-volume units)')
    axk.set_ylabel('mean signed mid-price change  (bps)   buy + (−1)·sell')
    axk.set_title(f'{args.stock} — mid-price impact in executed-volume units, by model'
                  + ('  [dotted vline = execution end]' if args.shape != 'beta' else ''))
    axk.legend(loc='upper left', fontsize=9)
    axk.grid(True, alpha=0.3)
    outk = os.path.splitext(out)[0] + '_k.png'
    figk.tight_layout(); figk.savefig(outk, dpi=150)
    print(f'saved -> {outk}')

    np.savez(os.path.splitext(out)[0] + '.npz', **cache)   # cache mean curves for instant re-plot
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
