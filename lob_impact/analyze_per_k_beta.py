#!/usr/bin/env python3
"""
Per-k Beta Analysis: compute β at each insertion k=1..k_max separately.

Tests whether β(k=1) (pristine book, single insertion) is closer to 0.5
than β(k=10) (depleted book, cumulative metaorder).

Usage:
    python lob_impact/analyze_per_k_beta.py \
        --pickle_base /path/to/pickles --stock GOOG \
        --daily_hl lob_impact/daily_h_l_GOOG.csv \
        --out_dir pics_for_per_k_beta
"""
import argparse, pickle, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, extract_point_cloud, compute_global_beta,
    bootstrap_beta, load_daily_params, collect_days,
    TICK_SIZE, N_BOOTSTRAP,
)


def main():
    parser = argparse.ArgumentParser(description='Per-k beta analysis')
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--stock', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--out_dir', default='pics_for_per_k_beta')
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle_base = Path(args.pickle_base)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    COLORS = {
        'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
        'CGAN': '#7B4F9E', 'LobS5': '#C88A3A', 'S5-120M': '#D95F02',
        'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E', 'LobS5-v2': '#2CA02C',
    }

    pkl_dir = pickle_base / args.stock
    if args.models:
        models = args.models
    else:
        models = sorted([f.stem for f in pkl_dir.glob('*.pkl')
                         if f.stem != 'ZeroInsertions'])

    all_results = {}

    for model in models:
        pkl_path = pkl_dir / f'{model}.pkl'
        if not pkl_path.exists():
            print(f'  SKIP {model}')
            continue

        print(f'\n{"="*60}')
        print(f'  {model} / {args.stock}')
        print(f'{"="*60}')
        print(f'Loading ({pkl_path.stat().st_size/1e9:.1f} GB)...')

        with open(pkl_path, 'rb') as f:
            md = pickle.load(f)

        exp_days = collect_days(md)
        daily_params = load_daily_params(args.daily_hl, exp_days)
        filtered, n_total, n_skip = filter_model(md)
        print(f'  Filtered: {n_skip}/{n_total}, Daily: {len(daily_params)} days')

        pc = extract_point_cloud(filtered, md['grid'], daily_params)
        if pc.empty:
            continue

        k_max = int(pc['k'].max())
        print(f'  Points: {len(pc)}, k_max={k_max}')

        # Per-k beta (VWAP cumulative)
        per_k = {}
        for k in range(1, k_max + 1):
            pc_k = pc[pc['k'] == k].copy()
            if len(pc_k) < 20:
                continue
            res = compute_global_beta(pc_k)
            # Quick bootstrap (fewer resamples for speed)
            boot = bootstrap_beta(pc_k, n_boot=500)
            boots = boot.get('boots', np.array([]))
            ci = (np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)) \
                if len(boots) > 10 else (np.nan, np.nan)

            per_k[k] = dict(
                beta=res['beta'], r2=res['r2'], n=res['n'], alpha=res['alpha'],
                ci_lo=ci[0], ci_hi=ci[1],
                beta_origin=res.get('beta_origin', np.nan),
            )

        # Per-k beta (midprice I_mid)
        per_k_mid = {}
        for k in range(1, k_max + 1):
            pc_k = pc[pc['k'] == k].copy()
            if len(pc_k) < 20 or 'I_mid' not in pc_k.columns:
                continue
            pc_k_valid = pc_k[pc_k['I_mid'].notna() & (pc_k['I_mid'] > 1e-12)].copy()
            if len(pc_k_valid) < 20:
                continue
            res = compute_global_beta(pc_k_valid, impact_col='I_mid')
            per_k_mid[k] = dict(
                beta=res['beta'], r2=res['r2'], n=res['n'],
            )

        # Per-k beta (instantaneous I_inst)
        per_k_inst = {}
        for k in range(1, k_max + 1):
            pc_k = pc[pc['k'] == k].copy()
            if len(pc_k) < 20 or 'I_inst' not in pc_k.columns:
                continue
            pc_k_valid = pc_k[pc_k['I_inst'].notna() & (pc_k['I_inst'] > 1e-12)].copy()
            if len(pc_k_valid) < 20:
                continue
            # For instantaneous: Q = size_k (not cumulative)
            pc_k_valid = pc_k_valid.copy()
            pc_k_valid['Q'] = pc_k_valid['size_k']
            pc_k_valid['I'] = pc_k_valid['I_inst']
            res = compute_global_beta(pc_k_valid)
            per_k_inst[k] = dict(
                beta=res['beta'], r2=res['r2'], n=res['n'],
            )

        all_results[model] = dict(per_k=per_k, per_k_mid=per_k_mid, per_k_inst=per_k_inst)

        # Print table
        print(f'\n  {"k":>3s}  {"β_vwap":>8s}  {"CI":>18s}  {"R²":>8s}  {"β_mid":>8s}  {"β_inst":>8s}  {"N":>8s}')
        for k in range(1, k_max + 1):
            vwap = per_k.get(k, {})
            mid = per_k_mid.get(k, {})
            inst = per_k_inst.get(k, {})
            bv = vwap.get('beta', np.nan)
            ci_l = vwap.get('ci_lo', np.nan)
            ci_h = vwap.get('ci_hi', np.nan)
            r2 = vwap.get('r2', np.nan)
            bm = mid.get('beta', np.nan)
            bi = inst.get('beta', np.nan)
            n = vwap.get('n', 0)
            print(f'  {k:>3d}  {bv:>8.4f}  [{ci_l:.3f},{ci_h:.3f}]  {r2:>8.4f}  {bm:>8.4f}  {bi:>8.4f}  {n:>8d}')

    if not all_results:
        print('No results')
        sys.exit(1)

    models_done = list(all_results.keys())

    # ═══════════════════════════════════════════════════════
    # PDF
    # ═══════════════════════════════════════════════════════
    pdf_path = out_dir / f'per_k_beta_{args.stock}.pdf'
    with PdfPages(str(pdf_path)) as pdf:

        # Fig 1: β(k) trajectory — VWAP
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in models_done:
            per_k = all_results[model]['per_k']
            ks = sorted(per_k.keys())
            betas = [per_k[k]['beta'] for k in ks]
            ci_lo = [per_k[k]['ci_lo'] for k in ks]
            ci_hi = [per_k[k]['ci_hi'] for k in ks]
            c = COLORS.get(model, 'gray')
            ax.plot(ks, betas, 'o-', color=c, lw=2, markersize=6, label=model)
            ax.fill_between(ks, ci_lo, ci_hi, color=c, alpha=0.1)
        ax.axhline(0.5, ls='--', color='red', lw=1.5, label='Theory β=0.5')
        ax.axhline(0, ls='-', color='gray', lw=0.5)
        ax.set(xlabel='Insertion k', ylabel='β (intercept estimator)',
               title=f'{args.stock} — β per insertion k (VWAP cumulative impact)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Fig 2: β(k) — midprice
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in models_done:
            per_k = all_results[model]['per_k_mid']
            if not per_k: continue
            ks = sorted(per_k.keys())
            betas = [per_k[k]['beta'] for k in ks]
            c = COLORS.get(model, 'gray')
            ax.plot(ks, betas, 'o-', color=c, lw=2, markersize=6, label=model)
        ax.axhline(0.5, ls='--', color='red', lw=1.5, label='Theory β=0.5')
        ax.axhline(0, ls='-', color='gray', lw=0.5)
        ax.set(xlabel='Insertion k', ylabel='β (I_mid)',
               title=f'{args.stock} — β per insertion k (Midprice response)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Fig 3: β(k) — instantaneous
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in models_done:
            per_k = all_results[model]['per_k_inst']
            if not per_k: continue
            ks = sorted(per_k.keys())
            betas = [per_k[k]['beta'] for k in ks]
            c = COLORS.get(model, 'gray')
            ax.plot(ks, betas, 'o-', color=c, lw=2, markersize=6, label=model)
        ax.axhline(0.5, ls='--', color='red', lw=1.5, label='Theory β=0.5')
        ax.axhline(0, ls='-', color='gray', lw=0.5)
        ax.set(xlabel='Insertion k', ylabel='β (I_inst, Q=size_k)',
               title=f'{args.stock} — β per insertion k (Instantaneous impact)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Fig 4: Summary table
        fig = plt.figure(figsize=(11, 8))
        y = 0.92
        fig.text(0.5, 0.96, f'Per-k Beta Summary — {args.stock}', ha='center',
                 fontsize=16, fontweight='bold')
        for model in models_done:
            fig.text(0.05, y, f'{model}:', fontsize=11, fontweight='bold')
            y -= 0.025
            fig.text(0.05, y, f'  {"k":>3s}  {"β_vwap":>8s}  {"β_mid":>8s}  {"β_inst":>8s}  {"R²":>8s}  {"N":>8s}',
                     fontsize=8, fontfamily='monospace')
            y -= 0.018
            per_k = all_results[model]['per_k']
            per_k_mid = all_results[model]['per_k_mid']
            per_k_inst = all_results[model]['per_k_inst']
            for k in sorted(per_k.keys()):
                bv = per_k[k]['beta']
                bm = per_k_mid.get(k, {}).get('beta', np.nan)
                bi = per_k_inst.get(k, {}).get('beta', np.nan)
                r2 = per_k[k]['r2']
                n = per_k[k]['n']
                fig.text(0.05, y, f'  {k:>3d}  {bv:>8.4f}  {bm:>8.4f}  {bi:>8.4f}  {r2:>8.4f}  {n:>8d}',
                         fontsize=8, fontfamily='monospace')
                y -= 0.016
            y -= 0.010
            if y < 0.05:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(11, 8)); y = 0.94
        pdf.savefig(fig); plt.close(fig)

    print(f'\nSaved: {pdf_path}')


if __name__ == '__main__':
    main()
