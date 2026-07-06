#!/usr/bin/env python3
"""
MASTER CURVE of market impact — the literature object that lives between the raw mid-price trajectory
and the β exponent (Bacry+ 2015 "life cycle of investor orders"; Gomes-Waelbroeck; Bouchaud-Bonart-
Donier-Gould 2018).

For every sample (metaorder) we take the SIGNED relative mid impact path I(step) and rescale TIME by the
metaorder's own execution duration:  v = step / T_exec ,  where T_exec = step of the LAST aggressive
insertion (end of execution).  So v=1 marks "execution finished".  We then AVERAGE I across samples on a
common v-grid and normalise by the average impact at v=1:

      master(v) = ⟨I(v)⟩ / ⟨I(v=1)⟩

  * beta shape   (100 insertions, 0 cooling):  v ∈ [0,1] — the concave BUILD-UP  (√-law ⇒ master ~ v^δ).
  * decay shape  (10 insertions + 100 cooling): v ∈ [0, ~11] — build-up to v=1 then RELAXATION
        (permanent fraction = where it plateaus; √-law peak vs Bouchaud-propagator decay).

"Average then normalise" (not per-sample I/I_peak) avoids blow-ups when a sample's own peak ≈ 0.

  python master_curve.py --grid <root> --stock EA --shape beta|relaxation \
         --models Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k --out master_curve_EA_beta.png
"""
import os, sys, glob, csv, re, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pubstyle import apply, style, ref_style, legend_box, savefig_pub
apply()

SENTINEL = 2147483647
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def _aggr_by_day(side_dir):
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def collect_paths(side_dir, sign, vgrid):
    """Per sample: SIGNED impact path interpolated onto the shared v-grid (v = step / T_exec)."""
    ab = _aggr_by_day(side_dir)
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'), recursive=True))
    out = []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        aggr = ab.get(m.group(1))
        if aggr is None or len(aggr) < 1 or aggr[0] < 1:
            continue
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask, bid = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask >= SENTINEL) | (bid >= SENTINEL) | (ask <= 0) | (bid <= 0)
        mid = (ask + bid) / 2.0; mid[bad] = np.nan
        ref = mid[aggr[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        T = int(aggr[-1])                       # last insertion = end of execution -> v=1
        if T < 2 or T >= len(mid):
            continue
        steps = np.arange(len(mid), dtype=float)
        v = steps / T
        I = sign * (mid - ref) / ref * 1e4      # bps
        ok = np.isfinite(I)
        if ok.sum() < 5:
            continue
        out.append(np.interp(vgrid, v[ok], I[ok], left=np.nan, right=np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--shape', default='beta', help='beta (build-up) | relaxation (build-up+decay)')
    ap.add_argument('--models', default='Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    models = [m for m in args.models.split(',') if m]
    vmax = 1.05 if args.shape == 'beta' else 11.0
    vgrid = np.linspace(0.0, vmax, 400)
    i1 = int(np.argmin(np.abs(vgrid - 1.0)))    # index of v=1 (execution end)

    build_up = (args.shape == 'beta')
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    print(f'=== {args.stock} {args.shape} master curve ===')
    cache = {'vgrid': vgrid}
    sig_masters = []
    for m in models:
        paths = collect_paths(os.path.join(args.grid, f'{args.stock}-{m}-{args.shape}', 'buy'), +1, vgrid) + \
                collect_paths(os.path.join(args.grid, f'{args.stock}-{m}-{args.shape}', 'sell'), -1, vgrid)
        if not paths:
            print(f'{m}: no data'); continue
        M = np.vstack(paths)
        mean = np.nanmean(M, axis=0)
        cnt = np.sum(np.isfinite(M), axis=0)
        mean[cnt < max(30, 0.3 * len(paths))] = np.nan
        peak = mean[i1]
        # master(v) divides by ⟨I(1)⟩: for impact-blind models that denominator is pure noise
        # (framework P3 fail) and the normalised curve is meaningless -> significance gate.
        peak_se = float(np.nanstd(M[:, i1]) / max(np.sqrt(cnt[i1]), 1.0))
        sig = bool(np.isfinite(peak) and abs(peak) >= 3 * peak_se and abs(peak) > 1e-9)
        if not np.isfinite(peak) or abs(peak) < 1e-9:
            print(f'{m}: peak~0, skipping normalise'); continue
        master = mean / peak
        st = style(m)
        n = len(paths)
        if sig:
            ax.plot(vgrid, master, color=st['color'], ls=st['ls'], lw=1.8,
                    label=f"{st['label']} (n={n})", zorder=3)
            sig_masters.append(master)
        else:
            ax.plot(vgrid, master, color=st['color'], ls=st['ls'], lw=1.0, alpha=0.25,
                    label=f"{st['label']} (n={n}; ⟨I(1)⟩={peak:.2f}±{peak_se:.2f} bps ≈ 0)",
                    zorder=1)
        cache[f'{m}_master'] = master
        cache[f'{m}_peak'] = peak
        cache[f'{m}_peak_se'] = peak_se
        cache[f'{m}_n'] = n
        cache[f'{m}_sig'] = sig
        print(f'  {m:10s}: n={n}, peak⟨I(v=1)⟩={peak:.3f}±{peak_se:.3f} bps '
              f'{"SIG" if sig else "NOT significant (curve greyed)"}, '
              f'end/peak={master[np.isfinite(master)][-1]:.2f}')
    # y-limits from SIGNIFICANT curves only, so noise/noise baselines can't blow up the axis
    if sig_masters:
        smax = np.nanmax([np.nanmax(s) for s in sig_masters])
        smin = np.nanmin([np.nanmin(s) for s in sig_masters])
        pad = 0.15 * max(smax - smin, 1.0)
        ax.set_ylim(min(smin, 0) - pad, max(smax, 1.0) + pad)

    # references
    rs = ref_style('sqrt')
    vb = np.linspace(0.01, 1.0, 60)
    ax.plot(vb, vb ** 0.5, color=rs['color'], ls=rs['ls'], lw=rs['lw'],
            label=r'$\sqrt{\cdot}$-law build-up  $v^{0.5}$', zorder=4)
    ax.axvline(1.0, color='#9a9a9a', lw=1.0, ls=':', zorder=2)
    ax.axhline(1.0, color='#9a9a9a', lw=0.7, zorder=2)
    if not build_up:
        rt = ref_style('twothirds')
        ax.axhline(2 / 3, color=rt['color'], ls=rt['ls'], lw=rt['lw'], alpha=0.8,
                   label=r'permanent $\approx \frac{2}{3}$', zorder=2)
    ax.set_xlabel(r'$v$ = fraction of metaorder executed   ($v=1$: execution end)')
    ax.set_ylabel(r'$\mathrm{master}(v)=\langle I(v)\rangle/\langle I(1)\rangle$')
    phase = 'build-up' if build_up else 'build-up + relaxation'
    ax.set_title(f'{args.stock} — impact master curve ({phase})')
    ax.margins(x=0)
    legend_box(ax, loc='upper left' if build_up else 'lower left', ncol=1)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'master_curve', f'master_curve_{args.stock}_{args.shape}.png')
    fig.tight_layout()
    savefig_pub(fig, os.path.abspath(out))
    np.savez(os.path.splitext(out)[0] + '.npz', **cache)   # cache for instant pub re-plot
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
