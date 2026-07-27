#!/usr/bin/env python
"""
Recompute the k-clock mean of a mid_trajectory cache with collapsed rollouts dropped.

The aggregate in mid_trajectory.py was a plain mean over rollouts. The paper's protocol says runs
that fail the well-formedness checks are excluded before analysis, but this aggregate never enforced
it, so a single collapsed book could own the curve: on AMD, two NMZI rollouts out of 4054 (0.05%,
both 2026-01-15) reached -9.3e6 bps and dragged the mean from -0.09 bps to -4592 bps. That is the
wild line in the relaxation panel — not a model result, a broken run.

mid_trajectory.py now drops them at source. This script repairs caches already on disk, which is
possible because --dump_samples stored the per-rollout matrix <model>_K next to the mean. It
recomputes <model>_k_mean and <model>_k_se from that matrix and rewrites the npz in place.

This is NOT tail trimming. Only trajectories whose |displacement| exceeds COLLAPSE_BPS anywhere are
removed — a move no metaorder produces at any horizon — so both tails of the real distribution
survive (AMD NMZI keeps p1/p99 = -16/+16 bps) and no exponent is biased.

  python fix_collapsed_runs.py results/mid_impact/mid_trajectory_AMD_decay.npz [...]
  python fix_collapsed_runs.py --all [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLLAPSE_BPS = 2000.0


def fix_one(path, thr, dry):
    z = dict(np.load(path, allow_pickle=True))
    models = sorted(k[:-2] for k in z if k.endswith('_K') and f'{k[:-2]}_k_mean' in z)
    if not models:
        return None
    report, changed = [], False
    for m in models:
        K = np.asarray(z[f'{m}_K'], float)
        bad = np.nanmax(np.abs(K), axis=1) > thr
        if not bad.any():
            continue
        before = float(np.nanmean(K, axis=0)[-1])
        Kg = K[~bad]
        kcnt = np.sum(np.isfinite(Kg), axis=0)
        kmean = np.nanmean(Kg, axis=0)
        kse = np.nanstd(Kg, axis=0) / np.sqrt(np.maximum(1, kcnt))
        # критерий по МОДУЛЮ, поэтому и показывать надо экстремум по модулю: печать min давала
        # бессмыслицу вида «худший -1 б.п.» у прогона, улетевшего в плюс
        ext = K[bad].ravel()
        ext = ext[np.isfinite(ext)]
        worst = ext[np.argmax(np.abs(ext))]
        report.append(f'    {m}: убрано {int(bad.sum())}/{len(K)} прогонов '
                      f'({bad.mean()*100:.2f}%, экстремум {worst:+.0f} б.п.); '
                      f'конец {before:+.1f} -> {kmean[-1]:+.2f} б.п.')
        if not dry:
            z[f'{m}_k_mean'] = kmean
            z[f'{m}_k_se'] = kse
            z[f'{m}_K'] = Kg.astype(np.float32)
            for side in ('_K_side', '_K_day'):
                if f'{m}{side}' in z:
                    arr = np.asarray(z[f'{m}{side}'])
                    if arr.shape[:1] == bad.shape:
                        z[f'{m}{side}'] = arr[~bad]
        changed = True
    if changed and not dry:
        np.savez_compressed(path, **z)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--all', action='store_true', help='every mid_trajectory_*.npz under results/mid_impact')
    ap.add_argument('--thr', type=float, default=COLLAPSE_BPS)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    paths = list(args.paths)
    if args.all:
        paths += sorted(glob.glob(os.path.join(HERE, 'results', 'mid_impact', 'mid_trajectory_*.npz')))
    if not paths:
        print('нечего чинить: укажите файлы или --all')
        return 1

    total = 0
    for p in paths:
        rep = fix_one(p, args.thr, args.dry_run)
        if rep is None:
            print(f'{os.path.basename(p)}: нет посэмпловых матриц (_K) — пропуск')
        elif rep:
            print(f'{os.path.basename(p)}:')
            print('\n'.join(rep))
            total += len(rep)
        else:
            print(f'{os.path.basename(p)}: развалившихся прогонов нет')
    print(f'\nисправлено кривых: {total}' + ('  (пробный прогон, файлы не тронуты)' if args.dry_run else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
