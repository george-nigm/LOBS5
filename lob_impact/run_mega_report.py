#!/usr/bin/env python3
"""
MEGA REPORT: Complete Market Impact Experiment Log (V1 → V5).

All versions, all models, all stocks, all findings, all figures.
For supervisor review.
"""
import pickle, numpy as np, pandas as pd, math, glob, os
from pathlib import Path
from collections import OrderedDict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings; warnings.filterwarnings('ignore')

OUT = Path('pics_for_mega_report')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def text_page(pdf, lines, fontsize=10, title=None):
    fig = plt.figure(figsize=(11, 8.5))
    y = 0.93
    if title:
        fig.text(0.5, 0.97, title, ha='center', fontsize=15, fontweight='bold')
        y = 0.90
    for line in lines:
        if line.startswith('###'):
            fig.text(0.05, y, line[3:].strip(), fontsize=10, fontweight='bold', style='italic'); y -= 0.025
        elif line.startswith('##'):
            fig.text(0.05, y, line[2:].strip(), fontsize=12, fontweight='bold'); y -= 0.030
        elif line.startswith('#'):
            fig.text(0.05, y, line[1:].strip(), fontsize=14, fontweight='bold'); y -= 0.040
        elif line == '':
            y -= 0.010
        elif line.startswith('  '):
            fig.text(0.06, y, line, fontsize=8, fontfamily='monospace'); y -= 0.018
        else:
            fig.text(0.05, y, line, fontsize=fontsize); y -= 0.022
        if y < 0.03:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5)); y = 0.95
    pdf.savefig(fig); plt.close(fig)

def image_page(pdf, path, title=None):
    p = Path(path)
    if not p.exists():
        print(f'  SKIP: {path}')
        return
    img = plt.imread(str(p))
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.imshow(img); ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else [0, 0, 1, 1])
    pdf.savefig(fig); plt.close(fig)

def load_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None

def fmt_beta_table(df, extra_cols=None):
    if df is None: return ['  (no data)']
    lines = []
    cols = ['Model', 'beta_intercept', 'beta_origin', 'CI_lo', 'CI_hi', 'R2', 'N']
    if extra_cols:
        cols += [c for c in extra_cols if c in df.columns]
    hdr_map = {'beta_intercept': 'b_int', 'beta_origin': 'b_orig', 'CI_lo': 'CI_lo', 'CI_hi': 'CI_hi',
               'R2': 'R2', 'N': 'N', 'relaxation': 'relax', 'stable_frac': 'stab',
               'Hurst': 'Hurst', 'no_arb_score': 'noarb', 'kyle_lambda_k1': 'lam_k1',
               'kyle_lambda_kmax': 'lam_kN', 'beta_perm': 'b_perm'}
    hdr = '  ' + '  '.join(f'{hdr_map.get(c,c):>8s}' for c in cols)
    lines.append(hdr)
    for _, row in df.iterrows():
        m = str(row.get('Model', ''))
        if m == 'ZeroInsertions': continue
        vals = []
        for c in cols:
            v = row.get(c, np.nan)
            if c == 'Model': vals.append(f'{str(v):>8s}')
            elif c == 'N': vals.append(f'{int(v):>8d}' if pd.notna(v) else f'{"":>8s}')
            elif c == 'no_arb_score': vals.append(f'{int(v):>8d}' if pd.notna(v) else f'{"":>8s}')
            else: vals.append(f'{float(v):>8.3f}' if pd.notna(v) else f'{"---":>8s}')
        lines.append('  ' + '  '.join(vals))
    return lines

def load_lobster_stats(stock):
    path = f'lob_impact/lobster_stats_{stock}.csv'
    if not Path(path).exists(): return {}
    df = pd.read_csv(path)
    return {k: df[k].median() for k in df.columns if k != 'day'}


def main():
    pdf_path = OUT / 'mega_experiment_report.pdf'
    print(f'Generating: {pdf_path}')

    # Load all data
    v3_goog = load_csv('pics_for_300_GOOG/summary_statistics.csv')
    v3_intc = load_csv('pics_for_300_INTC/summary_statistics.csv')
    v4_goog = load_csv('pics_for_v4_300_GOOG/summary_statistics.csv')
    v4_intc = load_csv('pics_for_v4_300_INTC/summary_statistics.csv')
    v5_aapl = load_csv('pics_for_v5_Q90_AAPL/summary_statistics.csv')
    v5_amzn = load_csv('pics_for_v5_Q40_AMZN/summary_statistics.csv')
    beta_grid = load_csv('pics_for_beta_grid/beta_grid_results.csv')
    lifecycle = load_csv('pics_for_lifecycle/lifecycle_GOOG.csv')

    with PdfPages(str(pdf_path)) as pdf:

        # ════════════════════════════════════════════════════════
        # TITLE PAGE
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '', '', '',
            '# Market Impact Evaluation Framework',
            '# Complete Experiment Log: V1 through V5',
            '',
            'George Nigmatulin — University of Oxford',
            'April 2026',
            '', '',
            '## Contents',
            '',
            '1. Executive Summary & Timeline',
            '2. V1/V2: First Grid Experiments (GOOG, INTC, uncalibrated volumes)',
            '3. V3: Methodology Corrections (intercept estimator, per-metaorder)',
            '4. V4: Production Grid (calibrated volumes, 10 models, GOOG + INTC)',
            '5. Diagnostics: Estimator Bias, Stratification, Beta Grid',
            '6. V5 Round 1: Low Participation Rate (phi=5-20%, AAPL + AMZN)',
            '7. V5 Round 2: Trade-Size Child Orders (phi=20-72%)',
            '8. Midprice Impact Analysis (VWAP vs Midprice)',
            '9. Kyle Lambda: Universal Model Differentiator',
            '10. Stock Universe Analysis (9 tickers)',
            '11. Model Training Data & Architecture',
            '12. Key Findings (8 discoveries)',
            '13. Conclusions & Next Steps',
        ], title='MEGA REPORT\nMarket Impact Evaluation Framework')

        # ════════════════════════════════════════════════════════
        # 1. EXECUTIVE SUMMARY
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 1. Executive Summary',
            '',
            '## Timeline',
            '  V1/V2 (Mar 19-20): First experiments, uncalibrated volumes (75/300/485 sh)',
            '  V3 (Mar 21-24): Intercept estimator, per-metaorder collapse, volume calibration',
            '  V4 (Mar 24-28): Production grid, 10 models, GOOG + INTC, 82K samples/model',
            '  V5r1 (Mar 30): Low phi (5-20%), AAPL + AMZN, child=2-9 shares → beta=0',
            '  V5r2 (Mar 30-Apr 1): Trade-size child (p25/p50/p75), phi=20-72%',
            '',
            '## Key Results Across All Versions',
            '',
            '  Version  Stock  Models  Estimator   beta range     R2 range     phi',
            '  V1/V2    GOOG   9       origin      0.80-0.83      0.08-0.14    ~99%',
            '  V3       GOOG   9       intercept   0.08-0.23      0.02-0.14    ~99%',
            '  V4       GOOG   9       intercept   0.21-0.34      0.11-0.28    ~99%',
            '  V4       INTC   9       intercept   0.14-0.43      0.11-0.42    ~99%',
            '  V5r1     AAPL   4       intercept   ~0 (noise)     ~0           5-20%',
            '  V5r2     AAPL   6       intercept   0.02-0.15      0.001-0.028  20-72%',
            '  V5r2     AMZN   6       intercept   0.02-0.14      0.001-0.034  20-84%',
            '',
            '## Core Findings',
            '1. Origin estimator biased by +0.51 — masks all model differentiation',
            '2. S5 models cluster at beta~0.33 (GOOG) / 0.40 (INTC) with intercept',
            '3. Kyle lambda is the strongest model differentiator at ANY phi',
            '4. S5-4K (4K context) > S5-360M (360M params) for dynamics',
            '5. At realistic phi (10-20%), VWAP impact unmeasurable',
            '6. Midprice cumulative impact shows signal at phi~50%',
            '7. Square-root law (beta=0.5) requires multi-agent market dynamics',
        ])

        # ════════════════════════════════════════════════════════
        # 2. V1/V2: FIRST EXPERIMENTS
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 2. V1/V2: First Grid Experiments',
            '',
            '## Setup',
            '  Stock: GOOG (and later INTC)',
            '  Volumes: 75 / 300 / 485 shares (arbitrary, uncalibrated)',
            '  Grid: 10 (i,mb) pairs x 3 vol x 2 dir = 60 configs',
            '  Samples: 2048 per config',
            '  Estimator: through-origin (biased)',
            '  Models: 9 (Historic, Heuristic, CST, CGAN, LobS5, S5-120M, S5-4K,',
            '          S5-360M, LobS5-v2)',
            '',
            '## V1/V2 GOOG Results (through-origin estimator)',
            '',
        ] + fmt_beta_table(v3_goog, ['relaxation']) + [
            '',
            '## Key Problem: No Model Differentiation',
            'All models yield beta_origin ~ 0.80-0.83. No separation.',
            'beta_intercept ranges 0.08-0.23 but we did not use it yet.',
            '',
            'Volume calibration issue discovered later:',
            '  GOOG 75 shares = p38 of depth at best (marginal)',
            '  INTC 75 shares = p6 of depth at best (broken!)',
        ])

        # V1/V2 figures
        for fig in ['2. Average Master Curve.png', '3. Beta Regression Lines.png',
                     '5. Relaxation Ratio.png', '17. Kyle Lambda per Insertion.png']:
            image_page(pdf, f'pics_for_300_GOOG/{fig}', title=f'V1/V2 GOOG — {fig[3:-4]}')

        # V1/V2 INTC
        text_page(pdf, [
            '## V1/V2 INTC Results',
            '',
        ] + fmt_beta_table(v3_intc, ['relaxation']))

        for fig in ['2. Average Master Curve.png', '5. Relaxation Ratio.png']:
            image_page(pdf, f'pics_for_300_INTC/{fig}', title=f'V1/V2 INTC — {fig[3:-4]}')

        # ════════════════════════════════════════════════════════
        # 3. V3: METHODOLOGY CORRECTIONS
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 3. V3: Methodology Corrections',
            '',
            '## Three Critical Discoveries',
            '',
            '### Finding 1: Origin Estimator Bias (+0.51)',
            'Through-origin: log(I/sigma) = beta * log(Q/V)',
            'When true model has nonzero intercept alpha:',
            '  bias = (alpha - E[log sigma]) * E[x] / E[x^2]',
            '  With alpha~-9, E[log sigma]~-3.5: bias ~ +0.51',
            '',
            'Verified on 80,000 synthetic data points:',
            '  Intercept estimator: recovered beta within +/-0.003',
            '  Origin estimator: biased by +0.13 to +0.66',
            '',
            '### Finding 2: One Metaorder = One Data Point',
            'Per-insertion points are autocorrelated (same book trajectory).',
            'Correct: collapse to one (Q, I) per (config, sample, direction).',
            'beta improved from 0.33 to 0.46 for S5 models.',
            '',
            '### Finding 3: Volume Calibration',
            'Fixed volumes (75/300/485) are arbitrary.',
            'INTC: 75 shares = p6 of depth — order never eats first level!',
            'Solution: calibrate from depth-at-best percentiles per stock.',
            '  GOOG: {105, 165, 325} = p50/p75/p95',
            '  INTC: {590, 1110, 3120} = p50/p75/p95',
        ])

        # Estimator comparison figure
        image_page(pdf, 'pics_for_v4_300_GOOG/15. Beta Estimator Comparison.png',
                   title='Finding 1: Estimator Comparison')

        # Beta report figures
        for fig in ['R01_synthetic_calibration.png', 'R04_beta_vs_k.png',
                     'R05_beta_vs_mb.png', 'R10_kyle_lambda.png']:
            p = f'pics_for_beta_report/{fig}'
            if Path(p).exists():
                image_page(pdf, p, title=f'V3 Diagnostic — {fig[4:-4]}')

        # ════════════════════════════════════════════════════════
        # 4. V4: PRODUCTION GRID
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 4. V4: Production Grid (Calibrated Volumes)',
            '',
            '## Setup',
            '  Volumes: GOOG {105,165,325}, INTC {590,1110,3120} (calibrated)',
            '  Grid: 10 (i,mb) pairs x 3 vol x 2 dir = 60 configs',
            '  Samples: 2048 per config (~82,000 metaorders per model)',
            '  Estimator: intercept (primary) + origin (comparison)',
            '  Models: 9 + ZeroInsertions null baseline',
            '',
            '## V4 GOOG — All 9 Models (sorted by beta_intercept)',
            '',
        ] + fmt_beta_table(v4_goog, ['relaxation', 'stable_frac', 'Hurst', 'no_arb_score']) + [
            '',
            'S5 models cluster at beta~0.33. CGAN lowest (0.205).',
            'S5-4K best relaxation (0.870, closest to 2/3).',
            'Hurst H~0.50 for all S5 (no long memory — negative finding).',
        ])

        for fig in ['2. Average Master Curve.png', '1. Master Curves.png',
                     '3. Beta Regression Lines.png', '4. Bootstrap Beta Distributions.png',
                     '5. Relaxation Ratio.png', '6. Fraction Stable.png',
                     '17. Kyle Lambda per Insertion.png', '14. No-Arb Scatter.png',
                     '13. Perm Temp Decomposition.png', '12. Per-Day Beta.png',
                     '7. Hurst Exponent.png', '8. Propagator G(l).png',
                     '9. Spread Dynamics.png', '16. Per-Insertion Midprice Response.png',
                     '0. Null Baseline Drift.png']:
            image_page(pdf, f'pics_for_v4_300_GOOG/{fig}', title=f'V4 GOOG — {fig[3:-4]}')

        # V4 INTC
        text_page(pdf, [
            '## V4 INTC — Cross-Stock Validation',
            '',
        ] + fmt_beta_table(v4_intc, ['relaxation', 'stable_frac', 'Hurst', 'no_arb_score']) + [
            '',
            'S5 beta~0.40 (closer to 0.5 than GOOG 0.33).',
            'S5-4K INTC: relaxation=0.763 (closest to 2/3 of any model/stock).',
            'INTC beta_perm~0.92-0.99 (close to 1.0 target).',
        ])

        for fig in ['2. Average Master Curve.png', '3. Beta Regression Lines.png',
                     '4. Bootstrap Beta Distributions.png', '5. Relaxation Ratio.png',
                     '17. Kyle Lambda per Insertion.png', '14. No-Arb Scatter.png',
                     '12. Per-Day Beta.png']:
            image_page(pdf, f'pics_for_v4_300_INTC/{fig}', title=f'V4 INTC — {fig[3:-4]}')

        # ════════════════════════════════════════════════════════
        # 5. DIAGNOSTICS
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 5. Diagnostics & Stratification',
            '',
            '## Beta Grid Validation',
            'Tested beta stability across different (i, mb) subsets.',
            '',
        ] + (['Grid results:'] + [f'  {r}' for r in (beta_grid.head(10).to_string(index=False).split('\n') if beta_grid is not None else ['no data'])] if beta_grid is not None else ['  (no grid data)']) + [
            '',
            '## Lifecycle Analysis',
            'Per-model diagnostic checks:',
        ] + ([f'  {r}' for r in (lifecycle.head(10).to_string(index=False).split('\n') if lifecycle is not None else ['no data'])] if lifecycle is not None else ['  (no lifecycle data)']),
        )

        # Diagnostic figures
        for fig in ['diag_1_synthetic.png', 'diag_3_scatter_GOOG.png']:
            p = f'pics_for_investigation/{fig}'
            if Path(p).exists():
                image_page(pdf, p, title=f'Diagnostic — {fig[:-4]}')

        # Beta report stratification
        for fig in ['R03_depth_calibration.png', 'R06_beta_vs_depth.png',
                     'R07_beta_per_day_boxplot.png', 'R08_normalizations.png',
                     'R09_impact_definitions.png', 'R12_beta_incremental.png',
                     'R13_cross_stock.png', 'R14_cross_sectional_beta.png',
                     'R15_penetration_split.png']:
            p = f'pics_for_beta_report/{fig}'
            if Path(p).exists():
                image_page(pdf, p, title=f'Stratification — {fig[4:-4]}')

        # Beta test early
        for fig in sorted(glob.glob('pics_for_beta_test_GOOG/*.png')):
            image_page(pdf, fig, title=f'Early Beta Test — {Path(fig).stem}')

        # ════════════════════════════════════════════════════════
        # 6. V5 ROUND 1: LOW PARTICIPATION RATE
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 6. V5 Round 1: Low Participation Rate (phi=5-20%)',
            '',
            '## Motivation',
            'V4 operates at phi~99% (stress test). Real markets: phi=5-20%.',
            'Can we measure beta at realistic participation rates?',
            '',
            '## Design',
            '  i=10, c=100, mb=36, budget=3960 messages (S5-4K 4096 ctx)',
            '  child = floor(eta/(1-eta) * mb * r * q_med)',
            '  Stocks: AAPL (r=2.73%, q_med=40), AMZN (r=2.17%, q_med=25)',
            '',
            '  eta=5%:  AAPL child=2,  AMZN child=1  (3-7x < p25 trade!)',
            '  eta=10%: AAPL child=4,  AMZN child=2',
            '  eta=20%: AAPL child=9,  AMZN child=4',
            '',
            '## Result: beta = 0, impact unmeasurable',
            '  child/depth = 0.4-3.0% (vs V4: 63-196%)',
            '  mean_I ~ 6e-5 (0.006% of price = microstructure noise)',
            '  Conclusion: phi-based child sizing produces unrealistic orders',
            '',
            '## V5r1 AAPL (eta=20%, Q=90)',
            '',
        ] + (fmt_beta_table(v5_aapl) if v5_aapl is not None else ['  (no data)']))

        for fig in ['1. Master Curves.png', '2. Average Master Curve.png']:
            image_page(pdf, f'pics_for_v5_Q90_AAPL/{fig}', title=f'V5r1 AAPL — {fig[3:-4]}')

        for fig in ['1. Scatter log(I) vs log(QV).png', '4. Kyle Lambda per Insertion.png',
                     '5. Beta Comparison.png']:
            image_page(pdf, f'pics_for_v5_single_AAPL/{fig}', title=f'V5r1 AAPL single — {fig[3:-4]}')

        # V5r1 AMZN
        for fig in ['2. Average Master Curve.png']:
            image_page(pdf, f'pics_for_v5_Q40_AMZN/{fig}', title=f'V5r1 AMZN — {fig[3:-4]}')

        for fig in ['4. Kyle Lambda per Insertion.png']:
            image_page(pdf, f'pics_for_v5_single_AMZN/{fig}', title=f'V5r1 AMZN single — {fig[3:-4]}')

        # ════════════════════════════════════════════════════════
        # 7. V5 ROUND 2: TRADE-SIZE CHILD
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 7. V5 Round 2: Trade-Size Child Orders (phi=20-72%)',
            '',
            '## Design Change',
            'Instead of deriving child from target phi, use REAL trade sizes:',
            '  child = p25/p50/p75 of actual trade size distribution',
            '  AAPL: child = 10/40/100 shares → phi = 20%/50%/72%',
            '  AMZN: child = 7/25/100 shares → phi = 26%/56%/84%',
            '',
            '## V5r2 AAPL — Beta (VWAP, intercept estimator)',
            '',
            f'  {"Model":>12s}  {"b_int":>7s}  {"R2":>7s}  {"N":>8s}',
            f'  {"Historic":>12s}  {"0.027":>7s}  {"0.001":>7s}  {"240K":>8s}',
            f'  {"Heuristic":>12s}  {"0.145":>7s}  {"0.021":>7s}  {"240K":>8s}',
            f'  {"CST":>12s}  {"0.087":>7s}  {"0.008":>7s}  {"241K":>8s}',
            f'  {"S5-4K":>12s}  {"0.047":>7s}  {"0.002":>7s}  {"240K":>8s}',
            f'  {"S5-120M":>12s}  {"0.077":>7s}  {"0.005":>7s}  {"240K":>8s}',
            f'  {"S5-360M":>12s}  {"0.094":>7s}  {"0.008":>7s}  {"241K":>8s}',
            '',
            '## V5r2 AMZN — Beta (VWAP)',
            '',
            f'  {"Model":>12s}  {"b_int":>7s}  {"R2":>7s}  {"N":>8s}',
            f'  {"Historic":>12s}  {"0.029":>7s}  {"0.002":>7s}  {"240K":>8s}',
            f'  {"Heuristic":>12s}  {"0.141":>7s}  {"0.028":>7s}  {"241K":>8s}',
            f'  {"CST":>12s}  {"0.089":>7s}  {"0.010":>7s}  {"241K":>8s}',
            f'  {"S5-4K":>12s}  {"0.075":>7s}  {"0.006":>7s}  {"240K":>8s}',
            f'  {"S5-120M":>12s}  {"0.090":>7s}  {"0.008":>7s}  {"241K":>8s}',
            f'  {"S5-360M":>12s}  {"0.109":>7s}  {"0.012":>7s}  {"241K":>8s}',
            '',
            'Beta improved vs V5r1 (0→0.03-0.15) but still << 0.5.',
            'VWAP impact limited: child < depth → execution at best price.',
        ])

        # ════════════════════════════════════════════════════════
        # 8. MIDPRICE IMPACT
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 8. Midprice Cumulative Impact Analysis',
            '',
            '## Why VWAP Impact Fails at Low phi',
            'When child < depth, order fills entirely at best ask/bid.',
            'VWAP = best_ask = constant → I_vwap = half_spread.',
            'No dependence on Q → beta = 0.',
            '',
            '## Midprice Impact: Model Response Metric',
            'I_mid = |mid_at_k10 - mid_at_k1| / mid_at_k1',
            'Measures how the MODEL shifts midprice through its generation',
            'after seeing 10 aggressive orders.',
            '',
            '## V5r2 AAPL — Midprice vs VWAP Comparison',
            '',
            f'  {"Model":>12s}  {"b_vwap":>7s}  {"b_mid":>7s}  {"CI":>15s}  {"R2_mid":>7s}  {"lam1->10":>12s}',
            f'  {"Historic":>12s}  {"0.027":>7s}  {"0.025":>7s}  {"[0.018,0.034]":>15s}  {"0.002":>7s}  {"0.211->0.228":>12s}',
            f'  {"Heuristic":>12s}  {"0.145":>7s}  {"0.092":>7s}  {"[0.084,0.101]":>15s}  {"0.021":>7s}  {"0.211->0.220":>12s}',
            f'  {"CST":>12s}  {"0.087":>7s}  {"0.091":>7s}  {"[0.083,0.098]":>15s}  {"0.028":>7s}  {"0.176->0.195":>12s}',
            f'  {"S5-4K":>12s}  {"0.047":>7s}  {"0.039":>7s}  {"[0.031,0.048]":>15s}  {"0.004":>7s}  {"0.221->0.198":>12s}',
            f'  {"S5-120M":>12s}  {"0.077":>7s}  {"0.041":>7s}  {"[0.033,0.050]":>15s}  {"0.004":>7s}  {"0.217->0.204":>12s}',
            f'  {"S5-360M":>12s}  {"0.094":>7s}  {"0.057":>7s}  {"[0.048,0.065]":>15s}  {"0.008":>7s}  {"0.217->0.218":>12s}',
            '',
            '## V5r2 AMZN — Midprice',
            '',
            f'  {"Model":>12s}  {"b_vwap":>7s}  {"b_mid":>7s}  {"CI":>15s}  {"R2_mid":>7s}  {"lam1->10":>12s}',
            f'  {"Historic":>12s}  {"0.029":>7s}  {"0.016":>7s}  {"[0.009,0.024]":>15s}  {"0.001":>7s}  {"0.444->0.467":>12s}',
            f'  {"Heuristic":>12s}  {"0.141":>7s}  {"0.085":>7s}  {"[0.078,0.093]":>15s}  {"0.024":>7s}  {"0.444->0.455":>12s}',
            f'  {"CST":>12s}  {"0.089":>7s}  {"0.088":>7s}  {"[0.080,0.094]":>15s}  {"0.034":>7s}  {"0.357->0.369":>12s}',
            f'  {"S5-4K":>12s}  {"0.075":>7s}  {"0.034":>7s}  {"[0.027,0.042]":>15s}  {"0.004":>7s}  {"0.446->0.414":>12s}',
            f'  {"S5-120M":>12s}  {"0.090":>7s}  {"0.045":>7s}  {"[0.038,0.053]":>15s}  {"0.007":>7s}  {"0.451->0.435":>12s}',
            f'  {"S5-360M":>12s}  {"0.109":>7s}  {"0.060":>7s}  {"[0.053,0.068]":>15s}  {"0.012":>7s}  {"0.452->0.450":>12s}',
            '',
            'KEY: S5-4K and S5-120M Kyle lambda FALLS (book restoration).',
            'All baselines: lambda RISES (depletion without recovery).',
            'S5-360M (500 ctx, 8x extrapolation): lambda FLAT.',
        ])

        # V5r2 midprice figures
        for fig in ['1. Scatter VWAP vs Midprice.png', '2. Average Master Curve.png',
                     '4. Kyle Lambda per Insertion.png', '5. Beta Comparison.png',
                     '3. Impact Distribution.png']:
            image_page(pdf, f'pics_for_v5r2_mid_AAPL/{fig}', title=f'V5r2 AAPL Midprice — {fig[3:-4]}')
        for fig in ['1. Scatter VWAP vs Midprice.png', '2. Average Master Curve.png',
                     '4. Kyle Lambda per Insertion.png']:
            image_page(pdf, f'pics_for_v5r2_mid_AMZN/{fig}', title=f'V5r2 AMZN Midprice — {fig[3:-4]}')

        # ════════════════════════════════════════════════════════
        # 9. KYLE LAMBDA
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 9. Kyle Lambda: Universal Model Differentiator',
            '',
            'lambda(k) = |exec_price - mid_before| / (tick * size)',
            'Measures per-insertion book response. At k=1: all identical.',
            '',
            '## Works at ANY participation rate:',
            '',
            'V4 (phi~99%, GOOG, child=105-325):',
            '  S5:       lambda rises 0.06 → 0.15-0.30 (depletion + recovery)',
            '  CST/CGAN: lambda flat 0.06-0.09 (over-replenishment)',
            '  Historic: lambda rises → 0.36 (full depletion)',
            '',
            'V5r1 (phi~20%, AAPL, child=9):',
            '  S5-4K: lambda FALLS 0.149 → 0.136 (recovery dominates!)',
            '  Baselines: lambda RISES',
            '',
            'V5r2 (phi~50%, AAPL, child=10/40/100):',
            '  S5-4K: lambda FALLS 0.221 → 0.198',
            '  S5-120M: lambda FALLS 0.217 → 0.204',
            '  S5-360M: lambda FLAT 0.217 → 0.218 (500 ctx extrapolation)',
            '',
            'Mechanism: S5 hidden state "sees" aggressive order →',
            'generates compensatory limit orders → restores depth.',
            'This is the CORRECT behavior for a realistic model.',
        ])

        # Kyle lambda comparison across versions
        image_page(pdf, 'pics_for_v4_300_GOOG/17. Kyle Lambda per Insertion.png',
                   title='Kyle Lambda V4 GOOG (phi~99%, 9 models)')
        image_page(pdf, 'pics_for_v5r2_mid_AAPL/4. Kyle Lambda per Insertion.png',
                   title='Kyle Lambda V5r2 AAPL (phi~50%, 6 models)')
        image_page(pdf, 'pics_for_v5r2_mid_AMZN/4. Kyle Lambda per Insertion.png',
                   title='Kyle Lambda V5r2 AMZN (phi~56%, 6 models)')

        # ════════════════════════════════════════════════════════
        # 10. STOCK UNIVERSE
        # ════════════════════════════════════════════════════════
        tickers = ['GOOG', 'AAPL', 'NVDA', 'AMZN', 'META', 'TSLA', 'MSFT', 'AMD', 'INTC']
        stock_lines = [
            '# 10. Stock Universe (9 Tickers)',
            '',
            'Computed from 20 trading days, January 2026, LOBSTER L2.',
            'S5-4K trained on: GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, AMD.',
            'INTC: NOT in training (out-of-distribution).',
            '',
            f'  {"Ticker":>6s}  {"r%":>6s}  {"q_med":>5s}  {"depth":>6s}  {"spr":>4s}  {"ADV_M":>6s}  {"r*q":>6s}  {"in_train":>8s}',
        ]
        for t in tickers:
            s = load_lobster_stats(t)
            if not s: continue
            r = s.get('trade_fraction', 0) * 100
            q = s.get('trade_size_p50', 0)
            d = s.get('depth_best_p50', 0)
            sp = s.get('spread_ticks_median', 0)
            adv = s.get('ADV', 0) / 1e6
            rq = r * q / 100
            tr = 'Yes' if t != 'INTC' else 'No (OOD)'
            stock_lines.append(f'  {t:>6s}  {r:>6.2f}  {q:>5.0f}  {d:>6.0f}  {sp:>4.0f}  {adv:>6.1f}  {rq:>6.2f}  {tr:>8s}')
        text_page(pdf, stock_lines)

        # ════════════════════════════════════════════════════════
        # 11. MODEL TRAINING DATA
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 11. Model Architecture & Training Data',
            '',
            f'  {"Model":>12s}  {"Params":>8s}  {"Ctx":>6s}  {"Training Stocks":>30s}  {"Years":>10s}',
            f'  {"LobS5":>12s}  {"75M":>8s}  {"500":>6s}  {"GOOG only":>30s}  {"2025":>10s}',
            f'  {"LobS5-v2":>12s}  {"75M":>8s}  {"500":>6s}  {"GOOG only":>30s}  {"2022":>10s}',
            f'  {"S5-120M":>12s}  {"120M":>8s}  {"500":>6s}  {"8 stocks":>30s}  {"2022-2025":>10s}',
            f'  {"S5-360M":>12s}  {"360M":>8s}  {"500":>6s}  {"8 stocks":>30s}  {"2022-2025":>10s}',
            f'  {"S5-4K":>12s}  {"55M":>8s}  {"4096":>6s}  {"8 stocks":>30s}  {"2022-2025":>10s}',
            f'  {"CGAN":>12s}  {"~5M":>8s}  {"snap":>6s}  {"GOOG (3 days)":>30s}  {"2025":>10s}',
            f'  {"CST":>12s}  {"param":>8s}  {"inf":>6s}  {"per-stock calibration":>30s}  {"Dec 2025":>10s}',
            f'  {"Historic":>12s}  {"---":>8s}  {"inf":>6s}  {"replay (no model)":>30s}  {"---":>10s}',
            f'  {"Heuristic":>12s}  {"---":>8s}  {"inf":>6s}  {"replay + shift":>30s}  {"---":>10s}',
            '',
            'All tested on January 2026 data (out-of-sample).',
            '',
            '## Key Insight: Context Length > Model Size',
            'S5-4K (55M, 4K ctx) consistently outperforms S5-360M (360M, 500 ctx)',
            'on dynamic metrics (relaxation, Kyle lambda).',
            'S5-360M at budget=4000 generates 8x beyond training context → degraded.',
        ])

        # ════════════════════════════════════════════════════════
        # 12. KEY FINDINGS
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 12. Eight Key Findings',
            '',
            '## F1: Origin Estimator Bias (+0.51)',
            'All models gave beta~0.84. Intercept estimator reveals true range.',
            '',
            '## F2: One Metaorder = One Data Point',
            'Per-insertion points are autocorrelated. Collapse to per-metaorder.',
            '',
            '## F3: Volume Calibration Critical',
            'Fixed volumes broken for INTC. Calibrate from depth percentiles.',
            '',
            '## F4: Participation Rate phi~99%',
            'Our injections dominate market. Stress test, not realistic execution.',
            '',
            '## F5: mb Controls Book Resilience',
            'More messages between insertions → more book recovery → lower beta.',
            '',
            '## F6: Cross-Sectional Beta Differentiates Models',
            'S5: beta~0.33 (GOOG), 0.40 (INTC). CST: 0.26. CGAN: 0.20.',
            '',
            '## F7: Kyle Lambda = Strongest Differentiator',
            'At k=1 all identical. At k>5: S5 diverges. Works at any phi.',
            '',
            '## F8: VWAP Impact Fails at Low phi',
            'When child < depth, VWAP = best price (constant). Use midprice.',
        ])

        # ════════════════════════════════════════════════════════
        # 13. CONCLUSIONS
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# 13. Conclusions & Next Steps',
            '',
            '## What We Established',
            '',
            '1. A complete evaluation framework for generative LOB models:',
            '   counterfactual injection → impact measurement → beta + dynamics.',
            '',
            '2. The intercept estimator is essential (origin biased by +0.51).',
            '',
            '3. S5 models with hidden state produce the most realistic dynamics:',
            '   best relaxation (r~0.76-0.87), best Kyle lambda trajectory.',
            '',
            '4. Context length matters more than model size:',
            '   S5-4K (55M, 4K ctx) > S5-360M (360M, 500 ctx).',
            '',
            '5. beta=0.5 (square-root law) is NOT achievable in single-model',
            '   simulation: requires multi-agent market, persistent order flow',
            '   (H~0.7), and institutional Q/V scale (1-10% of daily volume).',
            '',
            '## What Remains Open',
            '',
            '1. Trade-size grid (child = p50/p75/p95) gives beta~0.04-0.15',
            '   with trade-size child — measurable but far from 0.5.',
            '',
            '2. Midprice cumulative impact shows signal (beta_mid~0.03-0.09)',
            '   but does not differentiate S5 from baselines.',
            '',
            '3. Kyle lambda differentiates models at ANY phi — this is the',
            '   recommended primary metric for model evaluation.',
            '',
            '## Recommended Paper Framing',
            '',
            '"We propose a dynamic response evaluation framework for generative',
            'LOB models based on controlled aggressive flow injection. While the',
            'square-root scaling exponent requires multi-agent market dynamics',
            'beyond the scope of single-model simulation, our framework reveals',
            'that S5 models with continuous hidden state uniquely exhibit',
            'realistic book resilience (Kyle lambda decrease) and impact',
            'relaxation, properties invisible to distributional benchmarks."',
        ])

        # ════════════════════════════════════════════════════════
        # REMAINING V4 FIGURES
        # ════════════════════════════════════════════════════════
        text_page(pdf, [
            '# Appendix: Additional V4 INTC Figures',
        ])
        for fig in ['1. Master Curves.png', '13. Perm Temp Decomposition.png',
                     '7. Hurst Exponent.png', '8. Propagator G(l).png',
                     '9. Spread Dynamics.png']:
            image_page(pdf, f'pics_for_v4_300_INTC/{fig}', title=f'V4 INTC — {fig[3:-4]}')

        # V5r2 standard pipeline AAPL
        text_page(pdf, ['# Appendix: V5r2 Standard Pipeline (AAPL)'])
        # Generate figures from metrics if they exist but no PNGs
        # For now just note they exist as pickles
        text_page(pdf, [
            '## V5r2 Standard Pipeline Metrics Available',
            '',
            'Metrics pickles computed for all 6 models x 2 stocks:',
            '  pics_for_v5r2_AAPL/{Historic,Heuristic,CST,S5-4K,S5-120M,S5-360M}.metrics.pkl',
            '  pics_for_v5r2_AMZN/{Historic,Heuristic,CST,S5-4K,S5-120M,S5-360M}.metrics.pkl',
            '',
            'Figures can be generated with:',
            '  python lob_impact/run_300_figures.py --stock AAPL --out pics_for_v5r2_AAPL',
        ])

    size_mb = pdf_path.stat().st_size / 1e6
    print(f'\nSaved: {pdf_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
