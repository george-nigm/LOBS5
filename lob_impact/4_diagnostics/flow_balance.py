#!/usr/bin/env python3
"""
Flow-balance test: does the generated market carry long-memory order flow
WITHOUT the compensating resilience? (Bouchaud et al. 2004: diffusive prices
require the propagator to decay as G ~ l^-beta with beta ~ (1-gamma)/2 when
trade signs decay as C(l) ~ l^-gamma.)

Per model (clean no-insertion rollouts) and for the real conditioning stream:
  1. C(l): autocorrelation of trade signs in trade time (l = 1..200); log-log
     fit on l in [5,60] -> gamma_flow. Real markets: gamma ~ 0.5--0.7.
  2. Var(mid_{t+l} - mid_t) in trade time; slope/2 -> H_price. Efficiency
     (balance) means H ~ 0.5 even though gamma < 1.
  3. Response R(l) = E[eps_t (mid_{t+l} - mid_t)] in ticks (integrated
     propagator shape).
Verdict per model: long-memory flow + H > 0.5 => persistence not compensated
=> the reaction/replenishment component is missing (the mechanism behind
delta ~ 1 and no relaxation).

  python flow_balance.py --controls <controls_v2> --grid <grid_v2> --stock NVDA
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENT = 2147483647
TICK = 100
DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')
COLORS = {'Real': '#111111', 'Historic': '#C0392B', 'Heuristic': '#7F8C8D',
          'Propagator': '#8B5E3C', 'OW': '#6A1B9A', 'CST': '#27AE60', 'NMZI': '#117864',
          'Hawkes': '#D4AC0D', 'QR': '#00ACC1', 'S5': '#5D6D7E', 'S5_120M': '#F06292',
          'S5_4k': '#E67E22', 'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'GDN': '#D81B60'}
LMAX = 300
FIT_LO, FIT_HI = 2, 40


def read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def signs_and_mid(msg_file, ob_file):
    mm = read_csv(msg_file)
    ob = read_csv(ob_file)
    if mm.ndim != 2 or ob.ndim != 2 or mm.shape[1] < 6 or ob.shape[1] < 4:
        return None
    n = min(len(mm), len(ob))
    mm, ob = mm[:n], ob[:n]
    ask, bid = ob[:, 0], ob[:, 2]
    mid = (ask + bid) / 2.0
    mid[(ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)] = np.nan
    ex = np.flatnonzero((mm[:, 1] == 4) & (mm[:, 3] > 0))
    if len(ex) < 30:
        return None
    # LOBSTER: direction is the side of the standing limit order; the trade is
    # initiated by the opposite side -> trade sign = -direction.
    eps = -np.sign(mm[ex, 5])
    # literature convention: the sign series is per AGGRESSIVE ORDER, not per
    # fill — one market order eating several resting orders produces a burst of
    # same-timestamp same-sign rows (52% of consecutive fills here). Collapse
    # each burst into one trade; mid taken at the last fill of the burst.
    ts = mm[ex, 0]
    keep = np.ones(len(ex), bool)
    keep[1:] = ~((np.diff(ts) == 0) & (eps[1:] == eps[:-1]))
    # take the LAST row of each burst: shift the keep mask
    last = np.ones(len(ex), bool)
    last[:-1] = keep[1:]
    return eps[last], mid[ex][last]


def accum(files_msg, per_run_cap=None):
    """Return per-run lists of (eps, mid_at_trades)."""
    runs = []
    for mf in files_msg:
        obf = mf.replace('message', 'orderbook')
        if not os.path.exists(obf):
            continue
        r = signs_and_mid(mf, obf)
        if r is not None:
            runs.append(r)
        if per_run_cap and len(runs) >= per_run_cap:
            break
    return runs


def acf_gamma(runs):
    """Pooled sign autocorrelation C(l) and power-law fit -> gamma."""
    num = np.zeros(LMAX); den = 0.0; cnt = np.zeros(LMAX)
    for eps, _ in runs:
        e = eps - eps.mean()
        v = float(np.dot(e, e))
        den += v
        for l in range(1, min(LMAX, len(e) - 1) + 1):
            num[l - 1] += float(np.dot(e[:-l], e[l:]))
            cnt[l - 1] += 1
    C = num / max(den, 1e-12)
    C = C / max(C[0], 1e-12) if C[0] > 0 else C
    ls = np.arange(1, LMAX + 1)
    m = (ls >= FIT_LO) & (ls <= FIT_HI) & (C > 0)
    gamma = np.nan
    if m.sum() > 10:
        gamma = -float(np.polyfit(np.log(ls[m]), np.log(C[m]), 1)[0])
    return ls, C, gamma


def var_H(runs):
    """Var of mid displacement over l trades -> Hurst from slope/2."""
    ls = np.unique(np.round(np.logspace(0, np.log10(LMAX), 24)).astype(int))
    s = np.zeros(len(ls)); c = np.zeros(len(ls))
    for _, mid in runs:
        ok = np.isfinite(mid)
        m = mid[ok]
        for i, l in enumerate(ls):
            if len(m) > l + 10:
                d = m[l:] - m[:-l]
                s[i] += float(np.nansum(d * d)); c[i] += len(d)
    V = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    m = np.isfinite(V) & (V > 0) & (ls >= 2) & (ls <= 100)
    H = np.nan
    if m.sum() > 6:
        H = float(np.polyfit(np.log(ls[m]), np.log(V[m]), 1)[0]) / 2.0
    return ls, V, H


def resp(runs):
    ls = np.arange(1, 201)
    s = np.zeros(len(ls)); c = np.zeros(len(ls))
    for eps, mid in runs:
        for i, l in enumerate(ls):
            if len(mid) > l + 5:
                d = (mid[l:] - mid[:-l]) * eps[:-l]
                ok = np.isfinite(d)
                s[i] += float(np.sum(d[ok])); c[i] += int(ok.sum())
    return ls, np.where(c > 0, s / np.maximum(c, 1) / TICK, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controls', required=True)
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', required=True)
    ap.add_argument('--models', default='Historic,Heuristic,Propagator,OW,CST,NMZI,Hawkes,QR,S5_120M,S5_4k,Mamba3,Mamba3_4k,GDN')
    ap.add_argument('--n_files', type=int, default=64)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', f'causal_{args.stock}')
    os.makedirs(outdir, exist_ok=True)

    # Real = the Historic no-insertion replay: pure real message flow, ~13k msgs
    # per rollout (the 500-msg data_cond windows are too short for gamma/H fits).
    sources = {'Real': sorted(glob.glob(os.path.join(
        args.controls, 'noins', f'{args.stock}-Historic-beta', 'buy', '**', 'data_gen',
        '*message*gen*.csv'), recursive=True))[:args.n_files]}
    for m in [x for x in args.models.split(',') if x]:
        sources[m] = sorted(glob.glob(os.path.join(
            args.controls, 'noins', f'{args.stock}-{m}-beta', 'buy', '**', 'data_gen',
            '*message*gen*.csv'), recursive=True))[:args.n_files]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    cache = {}
    stats_rows = []   # (модель, gamma_flow, H, H_lo, H_hi) -> таблица рядом с фигурой
    lines = ['| source | runs | trades/run | pairs@100 | buy_share | gamma_flow | H_price | verdict |',
             '|---|---|---|---|---|---|---|---|']
    for name, files in sources.items():
        if not files:
            print(f'{name}: no files', flush=True); continue
        runs = accum(files)
        if not runs:
            print(f'{name}: no usable runs', flush=True); continue
        n_tr = np.array([len(e) for e, _ in runs])
        buy_share = float(np.mean(np.concatenate([e for e, _ in runs]) > 0))
        pairs100 = int(np.sum(np.maximum(n_tr - 100, 0)))
        print(f'{name}: runs={len(runs)} trades/run median={int(np.median(n_tr))} pairs@l=100={pairs100:,} buy_share={buy_share:.3f}', flush=True)
        ls, C, gamma = acf_gamma(runs)
        lv, V, H = var_H(runs)
        lr, R = resp(runs)
        # bootstrap CIs over runs (B=200): resample rollouts with replacement
        bs_rng = np.random.default_rng(7)
        gs, hs = [], []
        for _ in range(200):
            idx = bs_rng.integers(0, len(runs), len(runs))
            sub = [runs[i] for i in idx]
            gs.append(acf_gamma(sub)[2]); hs.append(var_H(sub)[2])
        g_lo, g_hi = np.nanpercentile(gs, [2.5, 97.5])
        h_lo, h_hi = np.nanpercentile(hs, [2.5, 97.5])
        c = COLORS.get(name, '#444444')
        ok = C > 0
        # The per-panel legends used to carry the fitted numbers, which made three stacked
        # 14-entry boxes covering the curves. Numbers now go to a table written next to the figure;
        # the panels get one shared legend at the bottom with names only.
        axes[0].loglog(ls[ok], C[ok], color=c, lw=1.5, label=name)
        axes[1].loglog(lv, V, color=c, lw=1.5)
        axes[2].plot(lr, R, color=c, lw=1.5)
        stats_rows.append((name, gamma, H, h_lo, h_hi))
        cache[f'{name}_C'] = C; cache[f'{name}_V'] = V; cache[f'{name}_R'] = R
        cache[f'{name}_gamma'] = gamma; cache[f'{name}_H'] = H
        cache[f'{name}_H_ci'] = np.array([h_lo, h_hi]); cache[f'{name}_gamma_ci'] = np.array([g_lo, g_hi])
        bal = (1 - gamma) / 2 if np.isfinite(gamma) else np.nan
        verdict = ('balanced' if np.isfinite(H) and abs(H - 0.5) < 0.07 else
                   'SUPERDIFFUSIVE (persistence uncompensated)' if H > 0.57 else
                   'subdiffusive')
        lines.append(f'| {name} | {len(runs)} | {int(np.median(n_tr))} | {pairs100:,} | {buy_share:.3f} | {gamma:.2f} [{g_lo:.2f},{g_hi:.2f}] | {H:.2f} [{h_lo:.2f},{h_hi:.2f}] | {verdict} |')
        print(f'{name}: gamma_flow={gamma:.2f} H_price={H:.2f} runs={len(runs)}', flush=True)

    axes[0].set_title('trade-sign autocorrelation $C(\\ell)$\n(long memory: slow power-law decay)', fontsize=10)
    axes[0].set_xlabel(r'lag $\ell$ (trades)')
    axes[1].set_title('mid displacement variance vs lag\n(slope $=2H$; efficiency $\\Rightarrow H \\approx 0.5$)', fontsize=10)
    axes[1].set_xlabel(r'lag $\ell$ (trades)')
    axes[2].set_title('response $R(\\ell) = E[\\varepsilon_t\\,(m_{t+\\ell}-m_t)]$, ticks', fontsize=10)
    axes[2].set_xlabel(r'lag $\ell$ (trades)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(7, max(4, (len(l) + 1) // 2)),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    axes[2].axhline(0, color='#cccccc', lw=0.8)
    fig.suptitle(f'{args.stock}: flow persistence vs price diffusivity '
                 r'(balance requires $\beta \approx (1-\gamma)/2$)', y=1.03)
    fig.tight_layout()
    png = os.path.join(outdir, f'flow_balance_{args.stock}.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), **cache)
    # The two Stage-0 numbers as a table instead of crammed into legend labels: three 14-entry
    # boxes used to sit on top of the curves they described.
    tbl = png.replace('.png', '_stats.md')
    real = next((r for r in stats_rows if r[0].lower().startswith('real')), None)
    with open(tbl, 'w') as fh:
        fh.write(f'| model | $\\gamma_{{flow}}$ | $H$ | 95% CI on $H$ |\n|---|---|---|---|\n')
        for name, g, H, lo, hi in stats_rows:
            fh.write(f'| {name} | {g:.2f} | {H:.2f} | [{lo:.2f}, {hi:.2f}] |\n')
    print('STATS ->', tbl)
    if real:
        print(f'  (real stream: gamma={real[1]:.2f}, H={real[2]:.2f} — the target both gates aim at)')
    with open(os.path.join(outdir, f'flow_balance_{args.stock}.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'FLOW_BALANCE_DONE -> {png}', flush=True)


if __name__ == '__main__':
    main()
