#!/usr/bin/env python3
"""
β vs k in THREE views (one panel each), cross-model, σ-normalized (default Parkinson; σ=1 with
--method none).  For every model we pool the per-(sample, insertion) cloud and, for each k in
1..kmax, slope-fit (free intercept) y = log(I/σ)  vs  x = log(Q/V) over three poolings of the
1-based insertion index kc:

  (a) ≤k  CUMULATIVE         mask = kc <= k   — smooth "paper" curve; k=kmax == full pooled β.
  (b) ==k EXACT              mask = kc == k   — noisy cross-section drill-down at exactly k.
  (c) ≥k  REVERSE-CUMULATIVE mask = kc >= k   — k=1 == full pooled β; drills into the high-k tail.

Reuses beta_grid.collect (per-day aggressive-index handling already correct: insertion steps come
from per-day aggressive_indices_<day>.csv matched to each orderbook by date in the filename; reads
the message CSV for cumulative executed volume Q; filters I>0 & Q>0) and vol_estimators.daily_sigmas.

  python 5_analysis/beta/beta_vs_k_3views.py --grid <root> --daily <daily_h_l_all.csv> \
         --stock EA --models Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k \
         --method parkinson --kmax 100 --out beta_vs_k_3views_EA.png [--html]

CAVEATS surfaced in output:
  * Hawkes folder may be absent from the grid -> collect() returns empty -> model skipped (no error).
  * CST currently has wrong-sign (negative) buy impact; collect()'s I>0 filter silently DROPS those
    rows, so CST β here is sparse / biased -> printed with a [CST: I>0 filter -> biased] flag.
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from beta_grid import collect
from vol_estimators import daily_sigmas

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pubstyle import apply, style, legend_box, savefig_pub
apply()

MIN_PTS = 8

VIEWS = [
    ('cumul',   '<=', r'$\leq k$  (cumulative)'),
    ('exact',   '==', r'$=k$  (exact)'),
    ('reverse', '>=', r'$\geq k$  (reverse-cumulative)'),
]


def fit_int(x, y, I=None, trim=0.0):
    """Free-intercept OLS slope. With trim>0 and I given, drop samples whose impact I (mid-price
    change) is outside the [trim, 1-trim] percentiles BEFORE fitting — removes the heavy-tail
    outliers that dominate the small ==k cross-section (verified: trim=0.05 cuts ==k std ~10% AND
    de-biases ==k β up onto the robust ≤k value; Theil-Sen smooths more but biases β downward, so
    we keep OLS). nan if <MIN_PTS points or no x-spread."""
    ok = np.isfinite(x) & np.isfinite(y)
    if I is not None:
        ok &= np.isfinite(I)
    x, y = x[ok], y[ok]
    if trim > 0 and I is not None:
        Ii = I[ok]
        if len(Ii) >= 20:
            lo, hi = np.percentile(Ii, [trim * 100, (1 - trim) * 100])
            keep = (Ii >= lo) & (Ii <= hi)
            x, y = x[keep], y[keep]
    if len(x) < MIN_PTS or (x.max() - x.min()) < 1e-6:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def model_betas(grid, stock, model, sig, method, kmax, trim=0.05):
    """Return (ks, b_cumul, b_exact, b_reverse, n_rows, flag) for one model, or None if no data."""
    exp = f'{stock}-{model}-beta'
    raw = collect(os.path.join(grid, exp, 'buy'),  stock, +1) + \
          collect(os.path.join(grid, exp, 'sell'), stock, -1)
    if not raw:
        return None
    Q = np.array([r[0] for r in raw], float)
    I = np.array([r[1] for r in raw], float)
    days = [r[2] for r in raw]
    kc = np.array([r[3] for r in raw], int) + 1            # 1-based insertion index
    V = np.array([sig.get((stock, d), {}).get('V', np.nan) for d in days], float)
    x = np.log(Q / V)
    if method == 'none':
        y = np.log(I)                                       # σ=1
    else:
        sg = np.array([sig.get((stock, d), {}).get(method, np.nan) for d in days], float)
        y = np.log(I / sg)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, kc, I = x[ok], y[ok], kc[ok], I[ok]
    if x.size == 0:
        return None
    kmax = int(min(kmax, kc.max()))
    ks = np.arange(1, kmax + 1)
    # trim outliers by impact I per-pooling-subset before each slope fit (esp. helps ==k)
    bc = np.array([fit_int(x[kc <= k], y[kc <= k], I[kc <= k], trim) for k in ks])
    be = np.array([fit_int(x[kc == k], y[kc == k], I[kc == k], trim) for k in ks])
    br = np.array([fit_int(x[kc >= k], y[kc >= k], I[kc >= k], trim) for k in ks])
    flag = ' [CST: I>0 filter -> biased]' if model == 'CST' else ''
    nk = np.array([int((kc == k).sum()) for k in ks])     # REAL points per exact k (==k fit size)
    return ks, bc, be, br, int(x.size), flag, nk, x, y, kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k')
    ap.add_argument('--method', default='parkinson',
                    help="σ estimator: parkinson|garman_klass|rogers_satchell|"
                         "close_to_close|yang_zhang|none (σ=1)")
    ap.add_argument('--kmax', type=int, default=100)
    ap.add_argument('--trim', type=float, default=0.0,
                    help='(SUPERSEDED by beta_binned.py — the literature binned/conditional-mean estimator). '
                         'per-point log-OLS; 0 = no trim. Kept for reference only.')
    ap.add_argument('--hero', default='Mamba3', help='model whose impact cloud is drawn on the bottom row')
    ap.add_argument('--kcursor', type=int, default=None, help='cursor insertion k for the cloud row (default 0.7*kmax)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]
    results = {}
    clouds = {}                                     # model -> (x, y, kc)
    for model in models:
        res = model_betas(args.grid, args.stock, model, sig, args.method, args.kmax, args.trim)
        if res is None:
            print(f'{args.stock}-{model}-beta: no data (skipped)')
            continue
        ks, bc, be, br, n, flag, nk, cx, cy, ckc = res
        results[model] = (ks, bc, be, br, n, flag, nk)
        clouds[model] = (cx, cy, ckc)
        fc = bc[np.isfinite(bc)]; fr = br[np.isfinite(br)]
        print(f'{args.stock}-{model}-beta (n={n}){flag}:  '
              f'cumul β(kmax)={fc[-1] if fc.size else np.nan:.3f}  '
              f'exact mean β={np.nanmean(be):.3f}  '
              f'reverse β(1)={fr[0] if fr.size else np.nan:.3f}')
    if not results:
        print('no models with data — nothing to plot'); return

    kmax = max(int(r[0][-1]) for r in results.values())
    kcur = args.kcursor if args.kcursor else max(1, int(round(0.7 * kmax)))
    hero = args.hero if args.hero in clouds else next(iter(clouds))
    sigtag = r'$\sigma{=}1$' if args.method == 'none' else f'$\\sigma$: {args.method}'

    # ---------- 2 x 3 publication figure: β(k) top, hero impact-cloud bottom ----------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    cache = {'kcursor': kcur, 'hero': hero}

    # --- TOP ROW: β(k) in three views, all models ---
    for col, (key, _, title) in enumerate(VIEWS):
        ax = axes[0][col]
        for model, (ks, bc, be, br, n, _, nk) in results.items():
            b = {'cumul': bc, 'exact': be, 'reverse': br}[key]
            st = style(model)
            fb = b[np.isfinite(b)]
            rep = (fb[-1] if key == 'cumul' else (np.nanmean(b) if key == 'exact' else fb[0])) if fb.size else np.nan
            lw = 1.2 if key == 'exact' else 1.8
            al = 0.8 if key == 'exact' else 1.0
            ax.plot(ks, b, color=st['color'], ls=st['ls'], lw=lw, alpha=al,
                    label=f"{st['label']}  ({rep:.2f})", zorder=3)
            cache[f'{model}_{key}'] = b
            cache[f'{model}_ks'] = ks
        ax.axhline(0.5, color='#C0392B', ls='--', lw=1.1, zorder=2)
        ax.axvline(kcur, color='#555', ls=':', lw=1.0, zorder=2)
        ax.set_ylim(-0.2, 1.0); ax.set_xlim(1, kmax); ax.margins(x=0)
        ax.set_xlabel('insertion index  $k$')
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel(r'$\beta(k)$  (free-intercept)')
            legend_box(ax, loc='lower right', ncol=1, fontsize=8)

    # --- BOTTOM ROW: hero impact cloud, active points per view highlighted at k=kcur ---
    cx, cy, ckc = clouds[hero]
    hs = style(hero)
    xmin, xmax = np.nanpercentile(cx, [0.5, 99.5])
    ymin, ymax = np.nanpercentile(cy, [0.5, 99.5])
    for col, (key, _, _) in enumerate(VIEWS):
        ax = axes[1][col]
        mask = {'cumul': ckc <= kcur, 'exact': ckc == kcur, 'reverse': ckc >= kcur}[key]
        # inactive points faint grey, active points in the hero colour
        ax.scatter(cx[~mask], cy[~mask], s=4, alpha=0.06, color='#9a9a9a', lw=0, zorder=1)
        ax.scatter(cx[mask], cy[mask], s=5, alpha=0.35, color=hs['color'], lw=0, zorder=2)
        # subset free-intercept fit line
        ok = mask & np.isfinite(cx) & np.isfinite(cy)
        if ok.sum() >= MIN_PTS and (cx[ok].max() - cx[ok].min()) > 1e-6:
            b1, a1 = np.polyfit(cx[ok], cy[ok], 1)
            xx = np.linspace(xmin, xmax, 40)
            ax.plot(xx, b1 * xx + a1, color='#1a1a1a', lw=1.8, zorder=4,
                    label=rf'$\beta={b1:.2f}$  ($n={int(ok.sum()):,}$)')
            legend_box(ax, loc='lower right', fontsize=8)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$\ln(Q/V)$')
        if col == 0:
            ax.set_ylabel(rf'$\ln(I/\sigma)$   [{hs["label"]}]')

    fig.suptitle(rf'{args.stock} — square-root exponent $\beta(k)$ in three views '
                 rf'(top) and {hs["label"]} impact cloud at $k={kcur}$ (bottom).  {sigtag}',
                 fontsize=12, fontweight='bold')

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'beta_vs_k_3views',
                                   f'beta_vs_k_3views_{args.stock}.png')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig_pub(fig, os.path.abspath(out))
    np.savez(os.path.splitext(out)[0] + '.npz', **cache)
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
