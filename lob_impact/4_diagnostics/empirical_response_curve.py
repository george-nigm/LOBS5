#!/usr/bin/env python3
"""
Diagnose the SHAPE of the real-data event response R(m) (the Figure-8 anchor).

The coarse empirical_response.py showed NVDA R(60)=0.10 -> R(100)=0.40 -> R(131)=0.74:
a local exponent ~2.6, far too convex for any standard response function. Candidate
explanations: (a) genuine large-tick queue-depletion two-phase microstructure;
(b) drift leak through buy/sell order-flow imbalance on trending days;
(c) a per-day mixture (few volatile days dominating late horizons);
(d) NaN-mid selection at long horizons.

This script measures all of it on the same events (fixed event set, ex < L - m_max):
  - full pooled R(m), m = 1..m_max, +-2 SE
  - unsigned drift D(m) = E[mid_{t+m} - mid_{t-1}]  (trend leak shows here)
  - buy-only / sell-only responses (leak diverges them; real impact is symmetric)
  - per-day mean curves (mixture shows as heterogeneous knees)
  - sign-shuffled null band (100 reps, 2.5/97.5 pct)
  - trade-time response R_tr(k) at the k-th following execution (activity-normalised
    clock: answers "different days have different messages-per-trade")
  - CONTAMINATION CHECK (the "inside the window there are MORE market orders" worry):
    mean number of intervening executions within m messages, and the CLEAN response
    R_clean(m) = E[response | no other execution in (t, t+m]] vs the full R(m).
    R(m) - R_clean(m) is the follow-on-flow contribution. Caveat: conditioning on
    "no MO for m messages" selects quiet periods, so R_clean is a LOWER bound of the
    bare single-order impact, not an unbiased propagator.

  python empirical_response_curve.py --data_cond <.../buy/data_cond> --stock NVDA \
      --n_files 400 --m_max 250 --out_dir results/empresp_curve_NVDA
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICK = 100
SENT = 2147483647
K_TRADES = 20


def read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_cond', required=True)
    ap.add_argument('--stock', default='NVDA')
    ap.add_argument('--n_files', type=int, default=400)
    ap.add_argument('--m_max', type=int, default=250)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    M = args.m_max

    files = sorted(glob.glob(os.path.join(args.data_cond, '*message*.csv')))
    assert files, f'no message files under {args.data_cond}'
    by_day = {}
    for f in files:
        mday = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(f))
        by_day.setdefault(mday.group(1) if mday else '?', []).append(f)
    days = sorted(by_day)
    per_day_quota = max(args.n_files // max(len(days), 1), 1)
    picked = [f for d in days for f in by_day[d][:per_day_quota]]
    print(f'{len(files)} files, {len(days)} days -> using {len(picked)} '
          f'({per_day_quota}/day)', flush=True)

    ev_day, ev_sign, ev_resp, ev_tr, ev_gap = [], [], [], [], []
    msgs_per_ex = {d: [] for d in days}
    for f in picked:
        bf = f.replace('message', 'orderbook')
        if not os.path.exists(bf):
            continue
        day = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(f)).group(1)
        m = read_csv(f); b = read_csv(bf)
        L = min(len(m), len(b)); m, b = m[:L], b[:L]
        ask, bid = b[:, 0], b[:, 2]
        mid = (ask + bid) / 2.0
        mid[(ask >= SENT) | (bid <= 0) | (ask <= 0)] = np.nan
        ex_all = np.where(m[:, 1] == 4)[0]
        if len(ex_all) > 1:
            msgs_per_ex[day].append(float(np.mean(np.diff(ex_all))))
        ex = ex_all[(ex_all > 0) & (ex_all < L - M)]
        for t in ex:
            base = mid[t - 1]
            if np.isnan(base):
                continue
            sign = 1.0 if m[t, 5] == -1 else -1.0
            ev_day.append(day); ev_sign.append(sign)
            ev_resp.append((mid[t + 1:t + M + 1] - base) / TICK)      # UNSIGNED, len M
            nxt = ex_all[ex_all > t][:K_TRADES]
            tr = np.full(K_TRADES, np.nan); dist = np.full(K_TRADES, np.inf)
            tr[:len(nxt)] = (mid[nxt] - base) / TICK
            dist[:len(nxt)] = nxt - t                                 # msgs to k-th next MO
            ev_tr.append(tr); ev_gap.append(dist)

    ev_day = np.array(ev_day); sign = np.array(ev_sign)
    U = np.stack(ev_resp)                    # (N, M) unsigned responses
    TR = np.stack(ev_tr)                     # (N, K) unsigned at k-th next execution
    S = U * sign[:, None]                    # signed
    N = len(sign)
    print(f'{N} events  (buy {int((sign > 0).sum())} / sell {int((sign < 0).sum())})', flush=True)

    mgrid = np.arange(1, M + 1)
    R = np.nanmean(S, 0); R_se = np.nanstd(S, 0) / np.sqrt(np.sum(np.isfinite(S), 0))
    D = np.nanmean(U, 0)                                     # drift leak
    Rb = np.nanmean(U[sign > 0], 0); Rs = -np.nanmean(U[sign < 0], 0)
    Rday = np.stack([np.nanmean(S[ev_day == d], 0) for d in days])
    nday = np.array([(ev_day == d).sum() for d in days])
    rng = np.random.default_rng(0)
    null = np.stack([np.nanmean(U * rng.choice([-1.0, 1.0], N)[:, None], 0)
                     for _ in range(100)])
    nlo, nhi = np.nanpercentile(null, [2.5, 97.5], axis=0)
    Rtr = np.nanmean(TR * sign[:, None], 0)
    Rtr_se = np.nanstd(TR * sign[:, None], 0) / np.sqrt(np.sum(np.isfinite(TR), 0))
    mpe = {d: float(np.mean(v)) if v else np.nan for d, v in msgs_per_ex.items()}

    # contamination: intervening executions inside the m-window, and the clean response
    DIST = np.stack(ev_gap)                  # (N, K) msgs to k-th next MO (inf-padded)
    GAP = DIST[:, 0]                         # msgs to the FIRST next MO
    kmean = np.array([np.mean(np.sum(DIST <= m, 1)) for m in mgrid])
    R_clean = np.array([np.nanmean(S[GAP > m, m - 1]) if (GAP > m).sum() >= 30 else np.nan
                        for m in mgrid])
    n_clean = np.array([(GAP > m).sum() for m in mgrid])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax = axes[0][0]
    ax.plot(mgrid, R, color='#2F5DA3', lw=1.8, label='signed R(m)')
    ax.fill_between(mgrid, R - 2 * R_se, R + 2 * R_se, color='#2F5DA3', alpha=0.18, lw=0)
    ax.fill_between(mgrid, nlo, nhi, color='#999999', alpha=0.3, lw=0, label='sign-shuffled null')
    ax.plot(mgrid, D, color='#C0392B', lw=1.4, ls='--', label='unsigned drift D(m)')
    ax.axhline(0, color='#bbbbbb', lw=0.8); ax.legend(fontsize=8)
    ax.set_title('pooled response vs drift leak'); ax.set_xlabel('m (messages)'); ax.set_ylabel('ticks')
    ax = axes[0][1]
    for i, d in enumerate(days):
        ax.plot(mgrid, Rday[i], color='#aaaaaa', lw=0.7, alpha=0.8)
    ax.plot(mgrid, R, color='#2F5DA3', lw=2.0)
    ax.axhline(0, color='#bbbbbb', lw=0.8)
    ax.set_title(f'per-day curves ({len(days)} days, n/day median {int(np.median(nday))})')
    ax.set_xlabel('m (messages)')
    ax = axes[1][0]
    ax.plot(mgrid, Rb, color='#27AE60', lw=1.6, label='buy MOs: mid rise')
    ax.plot(mgrid, Rs, color='#C0392B', lw=1.6, label='sell MOs: mid fall (flipped)')
    ax.plot(mgrid, R, color='#2F5DA3', lw=1.2, ls=':', label='antisym (=pooled)')
    ax.axhline(0, color='#bbbbbb', lw=0.8); ax.legend(fontsize=8)
    ax.set_title('buy vs sell (leak diverges them)'); ax.set_xlabel('m (messages)'); ax.set_ylabel('ticks')
    ax = axes[1][1]
    ax.errorbar(np.arange(1, K_TRADES + 1), Rtr, yerr=2 * Rtr_se, fmt='o-', ms=4,
                color='#7B1FA2', lw=1.4, capsize=2)
    ax.axhline(0, color='#bbbbbb', lw=0.8)
    ax.set_title('trade-time: R at k-th next execution')
    ax.set_xlabel('k (executions after the event)'); ax.set_ylabel('ticks')
    ax = axes[0][2]
    ax.plot(mgrid, R, color='#2F5DA3', lw=1.8, label='full R(m) (with follow-on MOs)')
    ax.plot(mgrid, R_clean, color='#E67E22', lw=1.6,
            label='clean: no other MO in (t, t+m]')
    ax.axhline(0, color='#bbbbbb', lw=0.8); ax.legend(fontsize=8)
    for probe in (30, 60, 100, 150, 200):
        if probe <= M and np.isfinite(R_clean[probe - 1]):
            ax.annotate(f'n={n_clean[probe - 1]}', (probe, R_clean[probe - 1]),
                        fontsize=6.5, color='#E67E22', textcoords='offset points',
                        xytext=(0, -11))
    ax.set_title('contamination check: full vs clean-window response')
    ax.set_xlabel('m (messages)'); ax.set_ylabel('ticks')
    ax = axes[1][2]
    ax.plot(mgrid, kmean, color='#333333', lw=1.6)
    ax.set_title('mean # of OTHER executions inside the m-window')
    ax.set_xlabel('m (messages)'); ax.set_ylabel('count')
    fig.suptitle(f'{args.stock} real-data event response anatomy '
                 f'({N} events, msgs/exec day-mean {np.nanmean(list(mpe.values())):.0f})')
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'empresp_curve_{args.stock}.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'),
                        mgrid=mgrid, R=R, R_se=R_se, D=D, Rb=Rb, Rs=Rs,
                        Rday=Rday, days=np.array(days), nday=nday,
                        null_lo=nlo, null_hi=nhi, Rtr=Rtr, Rtr_se=Rtr_se,
                        R_clean=R_clean, n_clean=n_clean, kmean=kmean,
                        msgs_per_exec=np.array([mpe[d] for d in days]))
    for probe in (10, 30, 60, 100, 131, 200, 250):
        if probe <= M:
            print(f'R({probe})={R[probe - 1]:+.3f}  clean={R_clean[probe - 1]:+.3f} '
                  f'(n={n_clean[probe - 1]})  #MOs_in={kmean[probe - 1]:.1f}  '
                  f'drift D={D[probe - 1]:+.3f}  '
                  f'buy={Rb[probe - 1]:+.3f} sell={Rs[probe - 1]:+.3f}', flush=True)
    print(f'EMPRESP_CURVE_DONE -> {png} (+npz)', flush=True)


if __name__ == '__main__':
    main()
