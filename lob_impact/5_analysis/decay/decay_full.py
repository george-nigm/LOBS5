#!/usr/bin/env python3
"""
Full Step-5 (II) dynamic-relaxation analysis over ALL models (framework paper).

Scenario Shape II (relaxation): num_insertions=10, num_coolings=100. The metaorder is executed
over the first L messages (last insertion = peak), then a long cooling tail exposes how the
mid-price impact relaxes toward a permanent level.

For every model present under <grid>/<stock>-<model>-relaxation/{buy,sell} it computes the four
dynamic statistics named in the paper's Step 5(II) + no-arbitrage scorecard:

  * master relaxation curve  m(u)  in normalized event-time u (u=1 at peak), buy/sell antisymmetrised
  * relaxation ratio         r = m(u=3)/m(u=1)                 target 2/3   (Bouchaud 2018)
  * post-peak decay exponent gamma :  m(u)/m(1) = c (1+(u-1))^{-gamma}
  * impact propagator        G(l) = <dp_{t+l} . eps_t>          ~ l^{-1/2}  (memory kernel)
  * Hurst exponent           H of the order-sign series via DFA  target 0.7 (persistent flow)
  * asymptotic-stability vote (3-of-checks on the cooling tail)

Metric implementations mirror `5_analysis/run_300_analyze_one.py` (compute_relaxation_ratio,
fit_decay, stability_vote, compute_hurst_dfa, compute_propagator) so the framework reports one
consistent set of definitions, adapted to the raw grid CSV layout.

Data layout (per model):
  <grid>/<stock>-<model>-relaxation/<side>/exp_*/data_gen/<STOCK>_<DATE>_{message,orderbook}_real_id_<R>_gen_id_<G>.csv
  <grid>/<stock>-<model>-relaxation/<side>/exp_*/aggressive_indices_<DATE>.csv   (insertion rows; last = peak L)
Message cols (LOBSTER 6): time, event_type(4=exec), order_id, size, price, direction(0=buy,1=sell)
Orderbook cols: ask_px(0), ask_sz(1), bid_px(2), bid_sz(3), ...

Outputs (to results/, override with --outdir):
  decay_master_<stock>.png     7-model master relaxation curves (unit-peak, 2/3 ref)
  decay_propagator_<stock>.png 7-model propagator G(l) log-log vs l^{-1/2}
  decay_scorecard_<stock>.png  r / gamma / H bars vs targets
  decay_scorecard_<stock>.csv  per-model table (n, r, gamma, H, stable, pass flags)

Cluster-safe: heavy Lustre globbing -> run under sbatch (run_decay.sh), never on a login node.

  python 5_analysis/decay/decay_full.py --grid <root> --stock EA \
         --models Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k,S5_4k [--max_samples N]
"""
import os, glob, csv, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENTINEL = 2147483647
TICK = 100.0                       # price units per tick (EA/GOOG default)
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
RID_RE = re.compile(r'_real_id_(\d+)_')
N_U = 200                          # master-curve resolution in u
MAX_LAG = 200                      # propagator / Hurst lag horizon
TWO_THIRDS = 2.0 / 3.0
H_TARGET = 0.70

ALL_MODELS = ['Historic', 'Heuristic', 'OW', 'CST', 'Hawkes', 'QR', 'S5_4k', 'Mamba3', 'Mamba3_4k']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Hawkes': '#8E44AD', 'CST': '#27AE60',
          'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'OW': '#6A1B9A', 'QR': '#00ACC1'}


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers (tolerant of the corrupt CST cells that crash a plain float parse)
# ─────────────────────────────────────────────────────────────────────────────
def _read(f):
    out = []
    for r in csv.reader(open(f)):
        if not r:
            continue
        row = []
        for x in r:
            try:
                row.append(float(x))
            except ValueError:
                row.append(np.nan)
        out.append(row)
    return np.array(out, dtype=float) if out else None


def _mid_ret(ob):
    """mid-price return series (mid_t - mid_0)/mid_0 from an orderbook csv path."""
    a = _read(ob)
    if a is None or a.ndim != 2 or a.shape[1] < 4:
        return None
    ask, bid = a[:, 0], a[:, 2]
    bad = (ask >= SENTINEL) | (bid >= SENTINEL) | (ask <= 0) | (bid <= 0) | ~np.isfinite(ask) | ~np.isfinite(bid)
    mid = (ask + bid) / 2.0
    mid[bad] = np.nan
    m0 = mid[0] if np.isfinite(mid[0]) and mid[0] > 0 else np.nanmedian(mid[:5])
    if not np.isfinite(m0) or m0 <= 0:
        return None
    return (mid - m0) / m0


def _mid_ticks(ob):
    """mid-price in ticks (for the propagator dp), nan on bad rows."""
    a = _read(ob)
    if a is None or a.ndim != 2 or a.shape[1] < 4:
        return None
    ask, bid = a[:, 0], a[:, 2]
    bad = (ask >= SENTINEL) | (bid >= SENTINEL) | (ask <= 0) | (bid <= 0) | ~np.isfinite(ask) | ~np.isfinite(bid)
    mid = (ask + bid) / 2.0
    mid[bad] = np.nan
    return mid / TICK


def _signs(msg):
    """trade-sign series eps for executions (event_type==4). Generated messages use the native
    LOBSTER direction convention dir in {-1,+1}: dir=-1 means a resting ASK was consumed -> a BUY
    market order -> buyer-initiated -> +1; dir=+1 (resting BID consumed) -> seller-initiated -> -1.
    Returns full-length eps aligned to message rows (0 for non-executions) + a compact exec-only
    sign series for the Hurst DFA."""
    a = _read(msg)
    if a is None or a.ndim != 2 or a.shape[1] < 6:
        return None, None
    et, dr = a[:, 1], a[:, 5]
    eps = np.zeros(len(a))
    is_exec = (et == 4)
    eps[is_exec & (dr < 0)] = 1.0    # buyer-initiated
    eps[is_exec & (dr > 0)] = -1.0   # seller-initiated
    exec_signs = eps[is_exec]
    exec_signs = exec_signs[exec_signs != 0]
    return eps, exec_signs


def _aggr_by_day(side_dir):
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            try:
                out[d] = np.loadtxt(f, dtype=int, ndmin=1)
            except Exception:
                pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-model aggregation
# ─────────────────────────────────────────────────────────────────────────────
def _gen_files(side_dir):
    """yield (day, rid, gen_id, orderbook_path, message_path) for every generated sample."""
    for ob in sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'), recursive=True)):
        bn = os.path.basename(ob)
        md, mr = DATE_RE.search(bn), RID_RE.search(bn)
        if not md or not mr:
            continue
        msg = ob.replace('orderbook', 'message')
        yield md.group(1), mr.group(1), bn, ob, msg


def _side_traj(side_dir, aggr, max_samples=None):
    """(day,rid,gen) -> (mid_ret trajectory, peak_step L)."""
    out = {}
    for day, rid, key, ob, _msg in _gen_files(side_dir):
        idx = aggr.get(day)
        if idx is None or len(idx) < 2:
            continue
        ret = _mid_ret(ob)
        if ret is None:
            continue
        L = int(idx[-1])
        if L < 2 or L >= len(ret):
            continue
        out[(day, rid, key)] = (ret, L)
        if max_samples and len(out) >= max_samples:
            break
    return out


def master_curve(exp_dir, max_samples=None):
    """buy/sell antisymmetrised master curve on shared (day,rid) windows, in u=n/L."""
    buy_dir, sell_dir = os.path.join(exp_dir, 'buy'), os.path.join(exp_dir, 'sell')
    if not (os.path.isdir(buy_dir) and os.path.isdir(sell_dir)):
        return None
    aggr_b, aggr_s = _aggr_by_day(buy_dir), _aggr_by_day(sell_dir)
    buy = _side_traj(buy_dir, aggr_b, max_samples)
    sell = _side_traj(sell_dir, aggr_s, max_samples)
    # match on (day, rid) regardless of gen-id key (buy/sell drew independent gen streams)
    def _bykey(d):
        m = {}
        for (day, rid, _k), v in d.items():
            m.setdefault((day, rid), []).append(v)
        return m
    bm, sm = _bykey(buy), _bykey(sell)
    keys = sorted(set(bm) & set(sm))
    if not keys:
        return None
    curves = []
    for k in keys:
        # pair the i-th buy sample with the i-th sell sample on this window
        bs, ss = bm[k], sm[k]
        for (br, L), (sr, _Ls) in zip(bs, ss):
            ml = min(len(br), len(sr))
            if ml < 3 or L < 2:
                continue
            combined = (br[:ml] - sr[:ml]) / 2.0
            u = np.arange(ml) / L
            umax_local = u[-1]
            curves.append((u, combined, umax_local))
    if not curves:
        return None
    umax = min(6.0, max(c[2] for c in curves))
    u_grid = np.linspace(0, umax, N_U)
    interp = np.array([np.interp(u_grid, u, c) for (u, c, _) in curves])
    return dict(u=u_grid, mean=np.nanmean(interp, axis=0), std=np.nanstd(interp, axis=0), n=len(curves))


def propagator(exp_dir, max_samples=None):
    """G(l) = <dp_{t+l} . eps_t> averaged over samples/sides."""
    G_sum, G_cnt = np.zeros(MAX_LAG), np.zeros(MAX_LAG)
    n_used = 0
    for side in ('buy', 'sell'):
        side_dir = os.path.join(exp_dir, side)
        if not os.path.isdir(side_dir):
            continue
        cnt = 0
        for _day, _rid, _key, ob, msg in _gen_files(side_dir):
            if not os.path.exists(msg):
                continue
            eps, _ = _signs(msg)
            mid = _mid_ticks(ob)
            if eps is None or mid is None:
                continue
            n = min(len(eps), len(mid))
            if n < MAX_LAG + 10:
                continue
            mid = mid[:n]
            eps = eps[:n]
            dp = np.diff(mid)                      # dp[t] = mid[t+1]-mid[t], len n-1
            valid_dp = np.isfinite(dp)
            m = len(dp)
            for lag in range(min(MAX_LAG, m)):
                # dp[t+lag] paired with eps[t], t in [0, m-lag)
                a = dp[lag:m]
                b = eps[0:m - lag]
                ok = valid_dp[lag:m] & (b != 0)
                if ok.any():
                    G_sum[lag] += np.sum(a[ok] * b[ok])
                    G_cnt[lag] += int(ok.sum())
            n_used += 1
            cnt += 1
            if max_samples and cnt >= max_samples:
                break
    G = np.where(G_cnt > 0, G_sum / np.maximum(G_cnt, 1), np.nan)
    return G, n_used


def hurst(exp_dir, max_samples=None):
    """mean Hurst (DFA) of the per-sample execution-sign series."""
    Hs = []
    for side in ('buy', 'sell'):
        side_dir = os.path.join(exp_dir, side)
        if not os.path.isdir(side_dir):
            continue
        cnt = 0
        for _day, _rid, _key, _ob, msg in _gen_files(side_dir):
            if not os.path.exists(msg):
                continue
            _, exec_signs = _signs(msg)
            if exec_signs is None:
                continue
            h = _hurst_dfa(exec_signs)
            if np.isfinite(h):
                Hs.append(h)
            cnt += 1
            if max_samples and cnt >= max_samples:
                break
    return (float(np.mean(Hs)), float(np.std(Hs)), len(Hs)) if Hs else (np.nan, np.nan, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations (mirror run_300_analyze_one.py)
# ─────────────────────────────────────────────────────────────────────────────
def relaxation_ratio(mc, u_peak=1.0, u_final=3.0):
    if mc is None:
        return np.nan
    u, m = mc['u'], mc['mean']
    pi, fi = np.argmin(np.abs(u - u_peak)), np.argmin(np.abs(u - u_final))
    if abs(m[pi]) < 1e-12:
        return np.nan
    return float(m[fi] / m[pi])


def fit_decay(mc, u_peak=1.0):
    """post-peak power-law exponent gamma: m(u)/m(1) = c (1+(u-1))^{-gamma}.
    Fit in log-log space (log yn = log c - gamma log(1+pu)); numpy-only, no scipy."""
    if mc is None:
        return np.nan
    u, m = mc['u'], mc['mean']
    pi = np.argmin(np.abs(u - u_peak))
    pu, py = u[pi:] - u_peak, m[pi:]
    if len(py) < 5 or abs(py[0]) < 1e-12:
        return np.nan
    yn = py / py[0]
    mask = (pu > 0) & np.isfinite(yn) & (yn > 1e-6)
    if mask.sum() < 3:
        return np.nan
    x = np.log1p(pu[mask])
    y = np.log(yn[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(-slope)


def stability_vote(mc, u_peak=1.0):
    """asymptotic-stability of the cooling tail: (1) flat tail slope and (2) mid-level ~= end-level.
    Both numpy checks must hold. (Mirrors run_300_analyze_one's m1/m2; the scipy exp-fit m3 is dropped.)"""
    if mc is None:
        return False
    u, m = mc['u'], mc['mean']
    post = m[np.argmin(np.abs(u - u_peak)):]
    if len(post) < 10:
        return False
    n = len(post)
    tail = post[int(n * 0.80):]
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) > 1 else 1.0
    m1 = abs(slope) / (abs(np.mean(tail)) + 1e-10) < 0.05
    mid = n // 2
    m2 = abs(np.mean(post[-max(n // 8, 3):]) - np.mean(post[max(0, mid - n // 8):mid + n // 8])) / (abs(np.mean(tail)) + 1e-10) < 0.03
    return bool(m1 and m2)


def _hurst_dfa(signs, max_lag=MAX_LAG):
    signs = np.asarray(signs, dtype=float)
    if len(signs) < 60:
        return np.nan
    lag = min(max_lag, len(signs) // 3)
    cumsum = np.cumsum(signs - np.mean(signs))
    scales = np.unique(np.logspace(1, np.log10(lag), 20).astype(int))
    scales = scales[scales >= 4]
    flucts, used = [], []
    for sc in scales:
        n_seg = len(cumsum) // sc
        if n_seg < 1:
            continue
        F2 = 0.0
        for s in range(n_seg):
            seg = cumsum[s * sc:(s + 1) * sc]
            trend = np.polyval(np.polyfit(np.arange(sc), seg, 1), np.arange(sc))
            F2 += np.mean((seg - trend) ** 2)
        flucts.append(np.sqrt(F2 / n_seg))
        used.append(sc)
    # drop degenerate scales whose fluctuation is 0 (else log(0)=-inf poisons polyfit -> nan)
    flucts, used = np.asarray(flucts), np.asarray(used)
    pos = flucts > 0
    flucts, used = flucts[pos], used[pos]
    if len(flucts) < 3:
        return np.nan
    H, _ = np.polyfit(np.log(used), np.log(flucts), 1)
    return float(H)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_master(rows, stock, out):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for r in rows:
        mc = r['mc']
        if mc is None:
            continue
        u, m = mc['u'], mc['mean']
        pi = np.argmin(np.abs(u - 1.0))
        peak = m[pi]
        if abs(peak) < 1e-12:
            continue
        c = COLORS.get(r['model'], '#444')
        ax.plot(u, m / peak, '-', color=c, lw=1.9,
                label=f"{r['model']}: r={r['r']:.2f}, γ={r['gamma']:.2f} (n={mc['n']})")
    ax.axhline(TWO_THIRDS, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.7, TWO_THIRDS + 0.03, 'permanent 2/3 (Bouchaud)', fontsize=8.5)
    ax.axvline(1.0, color='k', ls=':', lw=1, alpha=0.6)
    ax.text(1.02, ax.get_ylim()[0] + 0.05, 'peak (execution end)', fontsize=8, rotation=90, va='bottom')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('normalized event time  u   (u=1 at peak, u>1 = relaxation)')
    ax.set_ylabel('impact  m(u) / m(peak)   (unit-peak normalised)')
    ax.set_title(f'{stock} — impact build-up & relaxation master curve (Shape II)')
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'saved -> {out}')


def fig_propagator(rows, stock, out):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    lags = np.arange(1, MAX_LAG)
    plotted = False
    for r in rows:
        G = r['G']
        if G is None or not np.isfinite(G[1:]).any():
            continue
        g = G[1:MAX_LAG]
        g0 = g[0] if np.isfinite(g[0]) and abs(g[0]) > 1e-12 else np.nanmax(np.abs(g))
        if not np.isfinite(g0) or abs(g0) < 1e-12:
            continue
        c = COLORS.get(r['model'], '#444')
        ax.plot(lags, np.abs(g / g0), '-', color=c, lw=1.6, alpha=0.9, label=f"{r['model']} (n={r['G_n']})")
        plotted = True
    if plotted:
        ref = lags.astype(float) ** (-0.5)
        ax.plot(lags, ref, 'k--', lw=1.3, alpha=0.8, label=r'$\ell^{-1/2}$ reference')
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('lag  ℓ  (messages)')
    ax.set_ylabel(r'normalised propagator  $|G(\ell)/G(1)|$')
    ax.set_title(f'{stock} — impact propagator  G(ℓ) = ⟨Δp_{{t+ℓ}}·ε_t⟩')
    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'saved -> {out}')


def fig_scorecard(rows, stock, out):
    models = [r['model'] for r in rows]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    cols = [COLORS.get(m, '#444') for m in models]
    # relaxation ratio vs 2/3
    r_vals = [r['r'] for r in rows]
    axes[0].bar(x, r_vals, color=cols)
    axes[0].axhline(TWO_THIRDS, color='k', ls='--', lw=1.3); axes[0].axhline(0, color='k', lw=0.6)
    axes[0].text(len(models) - 0.5, TWO_THIRDS + 0.02, '2/3', fontsize=9, ha='right')
    axes[0].set_title('Relaxation ratio  r = m(3)/m(1)\n(target 2/3)'); axes[0].set_ylabel('r')
    # decay exponent gamma
    g_vals = [r['gamma'] for r in rows]
    axes[1].bar(x, g_vals, color=cols)
    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].set_title('Post-peak decay exponent γ\n(m∝(1+u)^{-γ}; γ>0 = decays)'); axes[1].set_ylabel('γ')
    # Hurst vs 0.7
    h_vals = [r['H'] for r in rows]
    axes[2].bar(x, h_vals, color=cols)
    axes[2].axhline(H_TARGET, color='k', ls='--', lw=1.3); axes[2].axhline(0.5, color='gray', ls=':', lw=1)
    axes[2].text(len(models) - 0.5, H_TARGET + 0.01, '0.7', fontsize=9, ha='right')
    axes[2].text(len(models) - 0.5, 0.5 + 0.01, '0.5 (random)', fontsize=8, ha='right', color='gray')
    axes[2].set_title('Hurst exponent H of order signs\n(target 0.7 = persistent flow)'); axes[2].set_ylabel('H')
    for a in axes:
        a.set_xticks(x); a.set_xticklabels(models, rotation=40, ha='right', fontsize=8)
        a.grid(True, axis='y', alpha=0.3)
    fig.suptitle(f'{stock} — Step 5(II) dynamic-relaxation scorecard', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'saved -> {out}')


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='OW,QR,,'.join(ALL_MODELS))
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--max_samples', type=int, default=None,
                    help='cap generated samples read PER SIDE per stat (smoke); default = all')
    args = ap.parse_args()
    models = [m for m in args.models.split(',') if m]
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for model in models:
        exp_dir = os.path.join(args.grid, f'{args.stock}-{model}-relaxation')
        if not os.path.isdir(exp_dir):
            print(f'[skip] {model}: no dir {exp_dir}')
            continue
        print(f'[{model}] reading {exp_dir} ...', flush=True)
        mc = master_curve(exp_dir, args.max_samples)
        G, G_n = propagator(exp_dir, args.max_samples)
        H, H_std, H_n = hurst(exp_dir, args.max_samples)
        r = relaxation_ratio(mc)
        gamma = fit_decay(mc)
        stable = stability_vote(mc)
        peak_bps = (mc['mean'][np.argmin(np.abs(mc['u'] - 1.0))] * 1e4) if mc else np.nan
        n = mc['n'] if mc else 0
        rows.append(dict(model=model, mc=mc, G=G, G_n=G_n, r=r, gamma=gamma, H=H, H_std=H_std,
                         H_n=H_n, stable=stable, peak_bps=peak_bps, n=n))
        print(f'   n={n} peak={peak_bps:.2f}bps  r={r:.3f}  γ={gamma:.3f}  '
              f'H={H:.3f}±{H_std:.3f}(n={H_n})  G_n={G_n}  stable={stable}', flush=True)

    if not rows:
        print('no models found — nothing to do'); return

    fig_master(rows, args.stock, os.path.join(outdir, f'decay_master_{args.stock}.png'))
    fig_propagator(rows, args.stock, os.path.join(outdir, f'decay_propagator_{args.stock}.png'))
    fig_scorecard(rows, args.stock, os.path.join(outdir, f'decay_scorecard_{args.stock}.png'))

    # scorecard csv (+ pass flags against the paper's target bands)
    csv_path = os.path.join(outdir, f'decay_scorecard_{args.stock}.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'n_curves', 'peak_bps', 'relax_ratio_r', 'r_pass_2/3',
                    'decay_gamma', 'gamma_pass', 'hurst_H', 'H_n', 'H_pass_0.7', 'stable'])
        for r in rows:
            r_pass = np.isfinite(r['r']) and 0.5 <= r['r'] <= 0.85
            g_pass = np.isfinite(r['gamma']) and r['gamma'] > 0
            h_pass = np.isfinite(r['H']) and 0.6 <= r['H'] <= 0.8
            w.writerow([r['model'], r['n'], f"{r['peak_bps']:.3f}", f"{r['r']:.4f}", int(bool(r_pass)),
                        f"{r['gamma']:.4f}", int(bool(g_pass)), f"{r['H']:.4f}", r['H_n'],
                        int(bool(h_pass)), int(bool(r['stable']))])
    print(f'saved -> {csv_path}')

    # printed table
    print('\n' + '=' * 92)
    print(f'{args.stock} — Step 5(II) dynamic-relaxation scorecard   (targets: r≈2/3, H≈0.7, γ>0)')
    print('=' * 92)
    print(f"{'model':<12}{'n':>6}{'peak_bps':>10}{'r(2/3)':>9}{'gamma':>8}{'Hurst':>8}{'H_n':>6}{'stable':>8}")
    print('-' * 92)
    for r in rows:
        print(f"{r['model']:<12}{r['n']:>6}{r['peak_bps']:>10.2f}{r['r']:>9.3f}"
              f"{r['gamma']:>8.3f}{r['H']:>8.3f}{r['H_n']:>6}{str(r['stable']):>8}")
    print('=' * 92)


if __name__ == '__main__':
    main()
