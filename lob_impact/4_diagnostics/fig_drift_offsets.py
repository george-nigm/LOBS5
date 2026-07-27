#!/usr/bin/env python
"""
Error accumulation over the generation horizon, measured with the reference held fixed.

The obvious way to ask "does a generator drift?" is to score its whole rollout and compare with a
short one. That answer is wrong: our 500-vs-25,001-message comparison shows every model degrading by
1.3-2.5x INCLUDING the Historic replay, which is real data and cannot accumulate generation error.
That degradation is a property of the real reference (a 500-message conditioning window is not
distributionally representative of the 25,001-message stretch of the day a rollout spans), not of
the models.

This figure removes that confound. A fixed 500-message window is cut from each rollout at a series
of offsets and scored against the SAME real reference every time, so the only thing that varies is
how much generation preceded the window. Historic is the control: whatever it does is the floor, and
only a model's excess over that floor is drift.

Input: one LOB-Bench run per offset, staged by run_lobbench_noins.sbatch with
  GEN_WINDOW=500 GEN_OFFSET=<off> TAG=w500_off<off> SCORE_FLAGS=--unconditional
reading results/lobbench_noins/scores_<STOCK>_w500_off<off>/scores/scores_uncond_*.pkl

  python fig_drift_offsets.py --stock NVDA [--div wasserstein]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'lobbench_noins')
DIR_RE = re.compile(r'scores_(?P<stock>[A-Z0-9]+)_w500_off(?P<off>\d+)$')
FILE_RE = re.compile(r'scores_uncond_(?P<stock>[A-Z0-9]+)_(?P<model>.+?)_(?P<ts>\d{8}_\d{6})\.pkl$')

CANON = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR',
         'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'OW': '#6A1B9A', 'CST': '#27AE60', 'NMZI': '#117864', 'Hawkes': '#D4AC0D',
          'QR': '#00ACC1', 'S5': '#5D6D7E', 'S5_120M': '#F06292', 'S5_4k': '#E67E22',
          'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'GDN': '#D81B60'}
NEURAL = {'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN'}


def mean_over_metrics(path, div):
    """Mean of the 21 distributional metrics for one divergence; nan if none are finite."""
    with gzip.open(path, 'rb') as fh:
        d = pickle.load(fh)[0]
    vals = []
    for m in d:
        v = d[m].get(div)
        if v is None:
            continue
        x = float(np.atleast_1d(v[0]).ravel()[0])
        if np.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else np.nan


def collect(stock, div):
    """-> {model: {offset: score}}, keeping the newest timestamp per (model, offset)."""
    out, seen = {}, {}
    for d in sorted(glob.glob(os.path.join(RES, f'scores_{stock}_w500_off*'))):
        mo = DIR_RE.search(os.path.basename(d))
        if not mo or mo.group('stock') != stock:
            continue
        off = int(mo.group('off'))
        for p in glob.glob(os.path.join(d, 'scores', 'scores_uncond_*.pkl')):
            fm = FILE_RE.search(os.path.basename(p))
            if not fm or fm.group('stock') != stock:
                continue
            model, ts = fm.group('model'), fm.group('ts')
            if seen.get((model, off), '') > ts:
                continue
            seen[(model, off)] = ts
            out.setdefault(model, {})[off] = mean_over_metrics(p, div)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--div', default='wasserstein', choices=['wasserstein', 'ks', 'l1'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--copy_to', default=None)
    args = ap.parse_args()

    data = collect(args.stock, args.div)
    if not data:
        print(f'нет прогонов w500_off* для {args.stock} в {RES}')
        return 1
    offs = sorted({o for v in data.values() for o in v})
    if len(offs) < 2:
        print(f'найдено смещений: {offs} — нужно минимум два, жду остальные джобы')
        return 1

    # Only the models actually swept belong on this figure. The rest were scored at offset 0 only
    # (the sweep ran ONLY_MODELS to keep each job at ~9 min), and drawing them as lone dots made the
    # right panel read as if a large drift had been MEASURED for them when nothing was.
    swept = {m for m, v in data.items() if sum(np.isfinite(list(v.values()))) >= 2}
    skipped = sorted(set(data) - swept)
    order = [m for m in CANON if m in swept] + [m for m in swept if m not in CANON]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=200)
    ctrl = data.get('Historic', {})

    for m in order:
        x = np.array([o for o in offs if o in data[m]], dtype=float)
        y = np.array([data[m][o] for o in x])
        if not np.isfinite(y).any():
            continue
        c = COLORS.get(m, '#444444')
        # Historic is the control the whole argument rests on, and the Heuristic (same replay plus a
        # price shift) sits exactly on top of it — so draw the control dashed, thick and last-on-top,
        # or it is invisible under its own twin.
        ctrl_style = (m == 'Historic')
        kw = dict(color=c, marker='o', ms=4,
                  lw=3.0 if ctrl_style else (2.2 if m in NEURAL else 1.5),
                  ls='--' if ctrl_style else '-',
                  zorder=5 if ctrl_style else 3,
                  alpha=1.0 if (ctrl_style or m in NEURAL) else 0.85)
        ax[0].plot(x, y, label=(m.replace('_', '-') + ' (control)') if ctrl_style else m.replace('_', '-'), **kw)
        # right panel: excess over the replay control at the same offset -- the actual drift
        if ctrl and m != 'Historic':
            base = np.array([ctrl.get(o, np.nan) for o in x])
            with np.errstate(all='ignore'):
                ax[1].plot(x, y - base, label=m.replace('_', '-'), **kw)

    ax[0].set_title(f'{args.stock}: distance from real, 500-message window cut at each offset')
    ax[0].set_ylabel(f'{args.div} (mean over 21 metrics)')
    ax[1].axhline(0.0, color='#C0392B', ls='--', lw=1.6)
    ax[1].set_title('excess over the Historic replay control = drift')
    ax[1].set_ylabel(f'$\\Delta$ {args.div} vs replay')
    for a in ax:
        a.set_xlabel('messages generated before the scored window')
        a.grid(alpha=0.25)
        a.legend(fontsize=8, ncol=2, frameon=True, framealpha=0.8)
    if skipped:
        ax[0].text(0.02, 0.02, 'single-offset models not swept: ' + ', '.join(
            m.replace('_', '-') for m in skipped), transform=ax[0].transAxes,
            fontsize=7.5, color='#777777', va='bottom')
    fig.tight_layout()

    out = args.out or os.path.join(RES, f'drift_offsets_{args.stock}_{args.div}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    np.savez_compressed(out.replace('.png', '.npz'),
                        offsets=np.array(offs),
                        **{f'{m}_y': np.array([data[m].get(o, np.nan) for o in offs])
                           for m in data})
    print('saved ->', out, '| смещения:', offs, '| в развёртке:', len(swept),
          '| только смещение 0 (не рисуются):', ', '.join(skipped) if skipped else '—')
    for m in order:
        v = [data[m].get(o, np.nan) for o in offs]
        print(f'  {m:12s} ' + ' '.join(f'{x:7.3f}' for x in v))
    if args.copy_to:
        import shutil
        shutil.copy(out, args.copy_to)
        print('copied ->', args.copy_to)
    return 0


if __name__ == '__main__':
    sys.exit(main())
