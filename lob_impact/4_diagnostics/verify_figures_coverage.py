#!/usr/bin/env python
"""
Coverage gate: for every stock and every figure/table cache, which models are actually in it?

Every paper figure is drawn from a cached .npz. A model missing from that cache is silently
absent from the figure — the failure mode that hid OW and QR in the left panels of Fig. 6, in
Fig. 5 and Fig. 7, and behind the dots in the scorecard. This script reads only the KEY NAMES of
each cache (npz is lazy, so no array is loaded) and prints a stock x figure matrix of what is
present, so "is everything there on every stock?" is answered by a command instead of by eye.

  python verify_figures_coverage.py [--stocks EA,NVDA,GOOG,AMD,MSFT] [--quiet-ok]

Exit code 1 if anything is missing, so it can gate a push.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BETA = os.path.join(os.path.dirname(HERE), '5_analysis', 'beta', 'results')
DIAG = os.path.join(HERE, 'results')

# canonical order: replay baselines -> mechanical/parametric by increasing complexity -> neural
CANON = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR',
         'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']
# S5 (single-stock LobS5) exists only for GOOG; don't flag it elsewhere
OPTIONAL = {'S5'}

# figure family -> (path template, key suffix identifying a model's presence)
FAMILIES = [
    ('Fig6 left  (mid beta)',   f'{BETA}/mid_impact/mid_trajectory_{{s}}_beta.npz',            '_k_mean'),
    ('Fig6 left  (mid decay)',  f'{BETA}/mid_impact/mid_trajectory_{{s}}_decay.npz',           '_k_mean'),
    ('Fig6 right (master b)',   f'{BETA}/master_curve/master_curve_{{s}}_beta_v2gated.npz',    '_master'),
    ('Fig6 right (master r)',   f'{BETA}/master_curve/master_curve_{{s}}_relaxation_v2gated.npz', '_master'),
    ('Fig8 3x3 estimators',     f'{BETA}/beta_3x3/beta_3x3_{{s}}.npz',                         '_binned_le'),
    ('bias exhibit',            f'{BETA}/bias_exhibit/bias_exhibit_{{s}}.npz',                 '_parkinson_ym'),
    ('Fig5 flow balance',       f'{DIAG}/causal_{{s}}/flow_balance_{{s}}.npz',                 '_gamma'),
    ('Fig7 event response',     f'{DIAG}/empresp_curve_{{s}}/event_response_v2_{{s}}.npz',     '_Rtr_fig'),
    # audited 2026-07-27: the paper includes 25 stock-specific figures but this gate covered only the
    # 8 above, so the same "model silently absent from the legend" bug could hide in any of the rest.
    # These four are the remaining caches that carry per-model arrays.
    ('beta 3views L2',          f'{BETA}/beta_3views_l2/beta_3views_l2_{{s}}.npz',             '_le'),
    ('beta pipeline steps',     f'{BETA}/bias_exhibit/beta_pipeline_{{s}}.npz',                '_parkinson_l2_profile'),
    ('sigma grid (eq)',         f'{BETA}/beta_sigma_grid/beta_sigma_grid_{{s}}_eq.npz',        '_none_dbin'),
    ('placebo pretrend',        f'{DIAG}/causal_{{s}}/placebo_pretrend_{{s}}.npz',             '_placebo_mean'),
]

# Figures whose npz holds only axes/theory curves -- no per-model arrays at all, so a redesign has to
# re-read the grid instead of loading the cache. Reported separately: there is nothing to check per
# model, and silence here would read as "covered".
NO_MODEL_CACHE = [
    ('amplitude stability', f'{BETA}/beta_sigma_grid/amplitude_stability_{{s}}.npz'),
    ('beta 3x3 Y',          f'{BETA}/beta_3x3/beta_3x3_Y_{{s}}.npz'),
    ('decay protocol',      f'{BETA}/master_curve/decay_protocol_{{s}}.npz'),
]


def models_in(path, suffix):
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        return {k[:-len(suffix)] for k in z.files if k.endswith(suffix)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stocks', default='EA,NVDA,GOOG,AMD,MSFT')
    ap.add_argument('--quiet-ok', action='store_true', help='print only rows with gaps')
    args = ap.parse_args()
    stocks = args.stocks.split(',')

    bad = 0
    print(f'{"figure":26s} {"stock":6s} {"have":>5s}  missing')
    print('-' * 78)
    for label, tmpl, suffix in FAMILIES:
        for s in stocks:
            path = tmpl.format(s=s)
            # some caches are per-run directories — take the newest match
            if '*' in path:
                cands = sorted(glob.glob(path))
                path = cands[-1] if cands else path
            got = models_in(path, suffix)
            if got is None:
                print(f'{label:26s} {s:6s} {"-":>5s}  КЭША НЕТ ({os.path.basename(path)})')
                bad += 1
                continue
            miss = [m for m in CANON if m not in got and m not in OPTIONAL]
            if miss:
                print(f'{label:26s} {s:6s} {len(got):>5d}  {", ".join(miss)}')
                bad += 1
            elif not args.quiet_ok:
                print(f'{label:26s} {s:6s} {len(got):>5d}  OK')
    print('-' * 78)
    print('пробелов:', bad)

    thin = [(lbl, s) for lbl, tmpl in NO_MODEL_CACHE for s in stocks
            if models_in(tmpl.format(s=s), '_XXX') is not None
            and not any(k.split('_')[0] in CANON for k in np.load(tmpl.format(s=s), allow_pickle=True).files)]
    if thin:
        print()
        print('кэш без модельных массивов (перерисовка потребует чтения сетки):')
        for lbl, s in thin:
            print(f'  {lbl:22s} {s}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
