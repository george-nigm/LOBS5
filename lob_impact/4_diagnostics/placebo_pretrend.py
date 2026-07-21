#!/usr/bin/env python3
"""
Two cheap causal tests that close the identification argument.

PLACEBO   On no-insertion rollouts, impose the day's real injection schedule as
          PSEUDO-insertions and run the full impact pipeline: mean I(k) curve
          and the direct-L2 misfit profile. A healthy pipeline returns a flat
          near-zero curve and a flat profile (amplitude ~ 0, "no law") --- any
          signal here would mean the estimator manufactures impact.
PRE-TREND On visible rollouts, measure the mid drift in the window BEFORE each
          insertion, R_pre(m) = (mid[t-1] - mid[t-1-m])/TICK, capped at the
          previous insertion. The schedule is exogenous, so the pre-window must
          be flat; the post-window R(m) is the treatment response.

  python placebo_pretrend.py --grid <grid_v2> --controls <controls_v2> \
      --daily <daily.csv> --stock NVDA [--models ...] [--n_files 200]
"""
import os, re, csv, glob, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICK = 100
SENT = 2147483647
DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')
DGRID = np.arange(0.05, 1.51, 0.01)
COLORS = {'Historic': '#C0392B', 'Mamba3': '#2F5DA3', 'GDN': '#D81B60',
          'S5_120M': '#F06292', 'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22'}


def read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def mid_of(ob):
    ask, bid = ob[:, 0], ob[:, 2]
    mid = (ask + bid) / 2.0
    mid[(ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)] = np.nan
    return mid


def aggr_by_day(side_dir):
    out = {}
    for f in glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'), recursive=True):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def daily_sigmas(path):
    import csv as _csv
    out = {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            out[(r['stock'], r['date'])] = {'V': float(r['V']) if r.get('V') else np.nan,
                                            'parkinson': float(r['parkinson']) if r.get('parkinson') else np.nan}
    return out


def l2_profile(x, y):
    S = np.empty(len(DGRID)); A = np.empty(len(DGRID))
    for j, d in enumerate(DGRID):
        t = np.exp(d * x)
        a = float(np.dot(y, t) / np.dot(t, t))
        S[j] = float(np.sum((y - a * t) ** 2)); A[j] = a
    return S, A


def placebo_one(noins_dir, sched, sig, stock, n_files):
    obs = sorted(glob.glob(os.path.join(noins_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))[:n_files]
    K = []
    xs, ys = [], []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        day = m.group(1)
        idx = sched.get(day)
        if idx is None or len(idx) < 2:
            continue
        a = read_csv(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        mid = mid_of(a)
        L = len(mid)
        ii = idx[idx < L]
        if len(ii) < 5 or ii[0] < 1:
            continue
        ref = mid[ii[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        mf = ob.replace('orderbook', 'message')
        v_child = np.nan
        if os.path.exists(mf):
            mm = read_csv(mf)
            if mm.ndim == 2 and mm.shape[1] > 3:
                ex = mm[mm[:, 1] == 4]
                if len(ex):
                    v_child = float(np.median(ex[:, 3]))
        d = sig.get((stock, day), {})
        V, sp = d.get('V', np.nan), d.get('parkinson', np.nan)
        row = np.full(101, np.nan, np.float32)
        for k, step in enumerate(ii[:101]):
            I = (mid[step] - ref) / ref
            if np.isfinite(I):
                row[k] = I * 1e4
                if k > 0 and np.isfinite(v_child) and np.isfinite(V) and np.isfinite(sp) and V > 0:
                    xs.append(np.log(k * v_child / V)); ys.append(I / sp)
        K.append(row)
    if not K:
        return None
    K = np.array(K)
    mean = np.nanmean(K, 0)
    se = np.nanstd(K, 0) / np.sqrt(np.maximum((np.isfinite(K)).sum(0), 1))
    return mean, se, len(K), np.array(xs), np.array(ys)


def pretrend_one(side_dir, m_max, n_files):
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr = aggr_by_day(side_dir)
    step = max(len(obs) // n_files, 1)
    pre1 = np.zeros(m_max); pre_c = np.zeros(m_max, int)
    post1 = np.zeros(m_max); post_c = np.zeros(m_max, int)
    for ob in obs[::step][:n_files]:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        idx = aggr.get(m.group(1))
        if idx is None or len(idx) < 2:
            continue
        a = read_csv(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        mid = mid_of(a)
        L = len(mid)
        ii = idx[idx < L]
        for j, t in enumerate(ii):
            base = mid[t - 1] if t >= 1 else np.nan
            if not np.isfinite(base):
                continue
            prev = ii[j - 1] if j > 0 else 0
            lo = min(m_max, t - 1 - prev)
            if lo >= 1:
                r = base - mid[t - 1 - np.arange(1, lo + 1)]
                ok = np.isfinite(r)
                pre1[:lo][ok] += r[ok] / TICK; pre_c[:lo][ok] += 1
            nxt = ii[j + 1] if j + 1 < len(ii) else L
            hi = min(m_max, nxt - t, L - 1 - t)
            if hi >= 1:
                r = (mid[t + 1:t + hi + 1] - base) / TICK
                ok = np.isfinite(r)
                post1[:hi][ok] += r[ok]; post_c[:hi][ok] += 1
    pre = np.where(pre_c > 0, pre1 / np.maximum(pre_c, 1), np.nan)
    post = np.where(post_c > 0, post1 / np.maximum(post_c, 1), np.nan)
    return pre, post, int(pre_c[0]), int(post_c[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--controls', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', required=True)
    ap.add_argument('--models', default='Historic,Mamba3,GDN,S5_120M,Mamba3_4k,S5_4k')
    ap.add_argument('--m_max', type=int, default=60)
    ap.add_argument('--n_files', type=int, default=200)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', f'causal_{args.stock}')
    os.makedirs(outdir, exist_ok=True)
    sig = daily_sigmas(args.daily)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    cache = {}
    for model in [m for m in args.models.split(',') if m]:
        vis = os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy')
        sched = aggr_by_day(vis)
        noins = os.path.join(args.controls, 'noins', f'{args.stock}-{model}-beta', 'buy')
        c = COLORS.get(model, '#444444')
        r = placebo_one(noins, sched, sig, args.stock, args.n_files)
        if r is not None:
            mean, se, n, xs, ys = r
            kk = np.arange(len(mean))
            axes[0].plot(kk, mean, color=c, lw=1.5, label=f'{model} (n={n})')
            axes[0].fill_between(kk, mean - 2 * se, mean + 2 * se, color=c, alpha=0.12, lw=0)
            cache[f'{model}_placebo_mean'] = mean; cache[f'{model}_placebo_se'] = se
            if xs.size > 500:
                S, A = l2_profile(xs, ys)
                j = int(np.argmin(S))
                axes[1].plot(DGRID, S / S.min(), color=c, lw=1.5,
                             label=f'{model}: $a$={A[j]:+.2e}')
                cache[f'{model}_placebo_S'] = S
            print(f'{model}: placebo n={n} mean(k=100)={mean[-1]:+.2f} bps', flush=True)
        pre, post, n_pre, n_post = pretrend_one(vis, args.m_max, args.n_files)
        mm_ = np.arange(1, args.m_max + 1)
        axes[2].plot(mm_, post, color=c, lw=1.6)
        axes[2].plot(mm_, pre, color=c, lw=1.3, ls='--', alpha=0.8)
        cache[f'{model}_pre'] = pre; cache[f'{model}_post'] = post
        print(f'{model}: pretrend |pre|max={np.nanmax(np.abs(pre)):.3f} '
              f'post(60)={post[-1]:+.3f} ticks (n_pre={n_pre:,})', flush=True)

    axes[0].axhline(0, color='#bbbbbb', lw=0.8)
    axes[0].set_title('PLACEBO: pseudo-metaorder on no-insertion rollouts\n(pipeline must return ~0)', fontsize=10)
    axes[0].set_xlabel('pseudo-insertion index $k$'); axes[0].set_ylabel('mean $I(k)$, bps')
    axes[0].legend(fontsize=7)
    axes[1].set_title(r'placebo direct-$L_2$ profile $S(\delta)/S_{\min}$' + '\n(flat = no law, as it must be)', fontsize=10)
    axes[1].set_xlabel(r'$\delta$'); axes[1].legend(fontsize=6.5)
    axes[2].axhline(0, color='#bbbbbb', lw=0.8)
    axes[2].set_title('PRE-TREND: drift before insertion (dashed)\nvs response after (solid)', fontsize=10)
    axes[2].set_xlabel('messages before / after the insertion')
    axes[2].set_ylabel('mean mid move, ticks')
    fig.suptitle(f'{args.stock}: placebo and pre-trend causal checks', y=1.02)
    fig.tight_layout()
    png = os.path.join(outdir, f'placebo_pretrend_{args.stock}.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    np.savez_compressed(png.replace('.png', '.npz'), dgrid=DGRID, **cache)
    print(f'PLACEBO_PRETREND_DONE -> {png}', flush=True)


if __name__ == '__main__':
    main()
