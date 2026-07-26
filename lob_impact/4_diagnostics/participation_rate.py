#!/usr/bin/env python3
"""
Participation rate per step — windowed (sawtooth) AND cumulative, ONE FILE PER MODEL.

Two definitions, side by side, for one model across the 3 stocks (EA, NVDA, AMD):

  WINDOWED (reset at each insertion):  eta_win(t) = child_k / (exec vol since insertion k)
      right after an insertion the denominator is just the child  -> eta ~ 100%;
      the next ~mb organic executions grow the denominator -> eta decays to a FLOOR.
      That floor (value just before the next insertion) is the *effective* participation,
      designed to sit at eta = 10%. -> the sawtooth the metaorder actually experiences.

  CUMULATIVE:  eta_cum(t) = sum(meta <= t) / sum(exec <= t)
      metaorder share of ALL volume so far; converges to the mean of the window floors.

One figure per model: 3 rows (stocks) x 2 cols (windowed sawtooth | cumulative).
buy+sell pooled (they're symmetric). Collapse-corrupted samples dropped.

  python 4_diagnostics/participation_rate.py --grid 3_scenarios/results/grid --shape beta
"""
import os, glob, csv, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXECS = (4, 5)
TARGET = 10.0
STOCKS = os.environ.get('PART_STOCKS', 'EA,NVDA,AMD').split(',')
MODELS = ['Mamba3', 'Historic']
METRIC = 'count'    # 'count' (trades, matches eta=10% design) | 'volume' (shares); set in main()


def _read(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def sample_curves(side_dir):
    """Per-sample windowed + cumulative participation curves (aligned to min length)."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*message*gen*.csv'),
                           recursive=True))
    ai = glob.glob(os.path.join(side_dir, '**', 'aggressive_indices.csv'), recursive=True)
    if not obs or not ai:
        return None
    aggr = np.loadtxt(ai[0], dtype=int, ndmin=1)
    win, cum, sample_floors, dropped = [], [], [], 0
    for mf in obs:
        m = _read(mf)
        if m.ndim != 2 or m.shape[1] < 6:
            continue
        L = m.shape[0]
        size, etype = m[:, 3], m[:, 1]
        idx = aggr[aggr < L]
        if len(idx) < 2:
            continue
        is_exec = np.isin(etype, EXECS)
        if np.any(size < 0):                              # collapse corruption
            dropped += 1; continue
        # METRIC: count (1 per trade) — matches the eta=10% design (mb = 9/trade_frac);
        #         volume (shares) — the alternative, share-weighted view.
        if METRIC == 'count':
            exec_w = is_exec.astype(float)
            meta_w = np.zeros(L); meta_w[idx] = 1.0
        else:
            exec_w = np.where(is_exec, size, 0.0)
            meta_w = np.zeros(L); meta_w[idx] = size[idx]
        # child = the metaorder's ACTUALLY-LOGGED execution weight at the insertion row. We do NOT
        # fabricate it: neural scenarios inject a market order that decodes to a type-4 execution at
        # idx (child>0, measurable); replay scenarios (Historic/Heuristic/CST) inject an order that
        # is logged as a submission/cancel (type 1/3), so child=0 there -> that window is N/A, not 0
        # or a fabricated 50%. Participation is only defined where the child actually traded.
        cumexec = np.cumsum(exec_w)
        is_meta = np.zeros(L, bool); is_meta[idx] = True
        meta_cum = np.cumsum(meta_w)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.where(cumexec > 0, meta_cum / cumexec, np.nan)
        # windowed: reset at each insertion
        steps = np.arange(L)
        wpos = np.searchsorted(idx, steps, side='right') - 1     # window index per step
        valid = wpos >= 0
        start = np.where(valid, idx[np.clip(wpos, 0, len(idx) - 1)], 0)   # window start step
        child = np.where(valid, exec_w[start], np.nan)                    # metaorder weight of window
        base = np.where(start > 0, cumexec[np.clip(start - 1, 0, L - 1)], 0.0)
        denom = cumexec - base                                            # exec vol since window start
        with np.errstate(divide='ignore', invalid='ignore'):
            # child>0 required: windows whose injected order did not execute (replay) -> N/A, not 0
            w = np.where(valid & (denom > 0) & (child > 0), child / denom, np.nan)
        # window floor = eta at the END of each window (step just before the next insertion).
        # Pool EVERY window-end value across all samples (robust; the old per-sample nanmedian
        # collapsed to ~0 when short samples had a single insertion -> empty ends -> nan).
        if len(idx) >= 2:
            ends = np.clip(idx[1:] - 1, 0, L - 1)
            fw = w[ends]
            sample_floors.extend(fw[np.isfinite(fw)].tolist())
        win.append(w); cum.append(c)
    if not win:
        return None
    Lmin = min(len(x) for x in win)
    W = np.stack([x[:Lmin] for x in win]); C = np.stack([x[:Lmin] for x in cum])
    floor = float(np.median(sample_floors)) if sample_floors else np.nan
    return dict(win=np.nanmean(W, 0), cum=np.nanmean(C, 0), floor=floor,
                aggr=aggr[aggr < Lmin], n=W.shape[0], drop=dropped, L=Lmin)


def model_figure(grid, model, shape, out_dir):
    fig, axes = plt.subplots(len(STOCKS), 2, figsize=(15, 3.4 * len(STOCKS)), squeeze=False)
    unit = 'trade count' if METRIC == 'count' else 'volume (shares)'
    fig.suptitle(f'Participation rate [{unit}] — {model} / {shape}   '
                 f'(windowed sawtooth | cumulative,  buy+sell pooled)', fontsize=13)
    any_data = False
    for r, stock in enumerate(STOCKS):
        exp = f'{stock}-{model}-{shape}'
        curves = []
        for d in ('buy', 'sell'):
            c = sample_curves(os.path.join(grid, exp, d))
            if c:
                curves.append(c)
        axw, axc = axes[r]
        if not curves:
            axw.text(0.5, 0.5, f'no data: {exp}', ha='center'); axw.axis('off'); axc.axis('off')
            continue
        any_data = True
        Lmin = min(c['L'] for c in curves)
        win = np.nanmean(np.stack([c['win'][:Lmin] for c in curves]), 0) * 100
        cum = np.nanmean(np.stack([c['cum'][:Lmin] for c in curves]), 0) * 100
        aggr = curves[0]['aggr']; aggr = aggr[aggr < Lmin]
        n = sum(c['n'] for c in curves); drop = sum(c['drop'] for c in curves)
        u = np.arange(Lmin)
        # window floor: pooled median of per-window troughs (NOT read off the averaged curve).
        # nan when the injected order never executes (replay) -> report N/A, do not plot a floor.
        floor_med = float(np.nanmedian([c['floor'] for c in curves])) * 100
        floor_ok = np.isfinite(floor_med)
        floor_str = f'{floor_med:.1f}%' if floor_ok else 'N/A (child not executed)'
        print(f'  {model}/{shape} {stock}: floor={floor_str}  cum_final={cum[-1]:.2f}%  n={n}')

        axw.plot(u, win, color='#2F5DA3', lw=0.7)
        axw.axhline(TARGET, color='#2E7D52', ls='--', lw=1.2, label=f'η target {TARGET:.0f}%')
        if floor_ok:
            axw.axhline(floor_med, color='#C0392B', ls=':', lw=1.2, label=f'median floor {floor_med:.1f}%')
        else:
            axw.plot([], [], ' ', label='floor: N/A (child not executed)')
        axw.set_title(f'{stock} — windowed (n={n}{", −%d bad" % drop if drop else ""})', fontsize=10)
        axw.set_xlabel('step'); axw.set_ylabel('participation (%)'); axw.legend(fontsize=8)
        axw.set_ylim(0, 105)

        axc.plot(u, cum, color='#1A1A1A', lw=1.6)
        axc.axhline(TARGET, color='#2E7D52', ls='--', lw=1.2)
        axc.set_title(f'{stock} — cumulative (final {cum[-1]:.1f}%)', fontsize=10)
        axc.set_xlabel('step'); axc.set_ylabel('participation (%)')

    if not any_data:
        plt.close(fig); print(f'{model}/{shape}: no data'); return
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'participation_{model}_{shape}_{METRIC}.png')
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--shape', default='beta', choices=['beta', 'relaxation'])
    ap.add_argument('--metric', default='count', choices=['count', 'volume'])
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    global METRIC; METRIC = args.metric
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'results', 'participation_rate')
    for model in [m for m in args.models.split(',') if m]:
        model_figure(args.grid, model, args.shape, out_dir)


if __name__ == '__main__':
    main()
