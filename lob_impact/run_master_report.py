#!/usr/bin/env python3
"""
Master Experiment Report — Full Journey from v3 to v5.

Compiles ALL existing results into one comprehensive PDF.
No recomputation — reads only existing PNGs, CSVs, pickles.

Usage:
    python lob_impact/run_master_report.py
"""
import pickle, numpy as np, pandas as pd, math
from pathlib import Path
from collections import OrderedDict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings; warnings.filterwarnings('ignore')

OUT = Path('pics_for_master_report')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def text_page(pdf, lines, fontsize=11, title=None):
    fig = plt.figure(figsize=(11, 8.5))
    y = 0.92
    if title:
        fig.text(0.5, 0.96, title, ha='center', fontsize=16, fontweight='bold')
        y = 0.89
    for line in lines:
        if line.startswith('###'):
            fig.text(0.06, y, line[3:].strip(), fontsize=11, fontweight='bold', style='italic')
            y -= 0.030
        elif line.startswith('##'):
            fig.text(0.06, y, line[2:].strip(), fontsize=13, fontweight='bold')
            y -= 0.035
        elif line.startswith('#'):
            fig.text(0.06, y, line[1:].strip(), fontsize=15, fontweight='bold')
            y -= 0.045
        elif line == '':
            y -= 0.012
        elif line.startswith('  '):
            fig.text(0.07, y, line, fontsize=9, fontfamily='monospace')
            y -= 0.020
        else:
            fig.text(0.06, y, line, fontsize=fontsize)
            y -= 0.026
        if y < 0.04:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            y = 0.94
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, img_path, title=None):
    p = Path(img_path)
    if not p.exists():
        print(f'  SKIP: {img_path}')
        return
    img = plt.imread(str(p))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img); ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else [0, 0, 1, 1])
    pdf.savefig(fig); plt.close(fig)


def load_summary(path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def load_lobster_stats(stock):
    path = f'lob_impact/lobster_stats_{stock}.csv'
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return {
        'r': df['trade_fraction'].median() * 100,
        'q_med': df['trade_size_p50'].median(),
        'q_p25': df['trade_size_p25'].median(),
        'q_p75': df['trade_size_p75'].median(),
        'q_p95': df['trade_size_p95'].median(),
        'depth': df['depth_best_p50'].median(),
        'spread': df['spread_ticks_median'].median(),
        'ADV': df['ADV'].median() / 1e6,
    }


def fmt_summary_table(df, cols=None):
    """Format summary_statistics.csv into text lines."""
    if df is None:
        return ['  (no data)']
    if cols is None:
        cols = ['Model', 'beta_intercept', 'beta_origin', 'CI_lo', 'CI_hi', 'R2', 'N', 'relaxation', 'no_arb_score']
    lines = []
    hdr = f'  {"Model":>12s}  {"b_int":>7s}  {"b_orig":>7s}  {"CI_lo":>7s}  {"CI_hi":>7s}  {"R2":>6s}  {"N":>7s}  {"relax":>6s}  {"noarb":>5s}'
    lines.append(hdr)
    for _, row in df.iterrows():
        m = str(row.get('Model', ''))
        if m == 'ZeroInsertions':
            continue
        bi = row.get('beta_intercept', np.nan)
        bo = row.get('beta_origin', np.nan)
        cl = row.get('CI_lo', np.nan)
        ch = row.get('CI_hi', np.nan)
        r2 = row.get('R2', np.nan)
        n = row.get('N', 0)
        rx = row.get('relaxation', np.nan)
        na = row.get('no_arb_score', np.nan)
        lines.append(f'  {m:>12s}  {bi:>7.3f}  {bo:>7.3f}  {cl:>7.3f}  {ch:>7.3f}  {r2:>6.3f}  {int(n):>7d}  {rx:>6.3f}  {int(na):>5d}')
    return lines


def main():
    pdf_path = OUT / 'master_experiment_report.pdf'
    print(f'Generating: {pdf_path}')

    # Load data
    goog_v4 = load_summary('pics_for_v4_300_GOOG/summary_statistics.csv')
    intc_v4 = load_summary('pics_for_v4_300_INTC/summary_statistics.csv')
    stats = {t: load_lobster_stats(t) for t in ['GOOG', 'AAPL', 'NVDA', 'AMZN', 'META', 'TSLA', 'MSFT', 'AMD', 'INTC']}

    with PdfPages(str(pdf_path)) as pdf:

        # ═══════════════════════════════════════════════════════
        # SECTION 1: TITLE + INTRODUCTION
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '',
            '',
            '',
            '# Market Impact Evaluation Framework',
            '# Complete Experiment Log',
            '',
            '',
            'George Nigmatulin — University of Oxford',
            'March 2026',
            '',
            '',
            '## Timeline',
            '',
            'v3-v4 (March 19-26): Stress-test experiments on GOOG + INTC',
            '  - 10 models, calibrated volumes, phi ~ 99%',
            '  - Key discovery: intercept estimator, Kyle lambda differentiator',
            '',
            'v5 (March 30-31): Low-participation-rate experiments on AAPL + AMZN',
            '  - 6 models, phi = 5-20% derived from stock statistics',
            '  - Key discovery: child orders too small for measurable VWAP impact',
            '',
            '',
            '## Models Tested',
            '  S5-4K (55M, 4K ctx) | S5-120M (120M) | S5-360M (360M)',
            '  LobS5 (75M) | LobS5-v2 (75M) | CGAN | CST',
            '  Historic (replay) | Heuristic (replay+shift) | ZeroInsertions',
            '',
            '## Stocks',
            '  v4: GOOG, INTC (calibrated volumes from depth percentiles)',
            '  v5: AAPL, AMZN (child sizes from participation rate formula)',
            '  Stats computed: GOOG, INTC, AAPL, AMZN, NVDA, TSLA, MSFT, AMD, META',
        ], title='Market Impact Evaluation Framework\nComplete Experiment Log')

        # ═══════════════════════════════════════════════════════
        # SECTION 2: v4 STRESS TEST — GOOG
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Section 2: v4 Stress Test (phi ~ 99%)',
            '',
            '## GOOG — Experiment Setup',
            '',
            'Injection protocol: i insertions of Q_step shares, mb messages between,',
            'c = 10*i cooling periods. Constraint: 11*i*mb <= 500.',
            '',
            '  10 (i, mb) pairs x 3 volumes x 2 directions = 60 configs',
            '  Volumes: 105 / 165 / 325 shares (p50/p75/p95 of depth at best)',
            '  Samples: 2,048 per config',
            '  Total: ~82,000 metaorders per model',
            '',
            '## GOOG Results — 9 Models (sorted by beta_intercept)',
            '',
        ] + fmt_summary_table(goog_v4) + [
            '',
            'Key observations:',
            '  - S5 models cluster at beta ~ 0.33 (intercept estimator)',
            '  - Origin estimator gives beta ~ 0.84 for ALL models (biased +0.51)',
            '  - CGAN: lowest beta (0.205), worst relaxation (1.001)',
            '  - S5-4K: best relaxation (0.870), closest to theoretical 2/3',
            '  - Hurst H ~ 0.50 for all S5 (no long memory — negative finding)',
        ])

        # v4 GOOG figures
        for fig_name in [
            '2. Average Master Curve.png',
            '3. Beta Regression Lines.png',
            '4. Bootstrap Beta Distributions.png',
            '5. Relaxation Ratio.png',
            '17. Kyle Lambda per Insertion.png',
            '14. No-Arb Scatter.png',
        ]:
            image_page(pdf, f'pics_for_v4_300_GOOG/{fig_name}', title=f'GOOG v4 — {fig_name[3:-4]}')

        # v4 INTC
        text_page(pdf, [
            '## INTC — Cross-Stock Validation',
            '',
            'Same protocol, calibrated volumes: 590 / 1110 / 3120 shares.',
            'INTC is 10x more liquid than GOOG (depth_p50 = 590 vs 105).',
            '',
        ] + fmt_summary_table(intc_v4) + [
            '',
            'Key observations:',
            '  - S5 models: beta ~ 0.40 (higher than GOOG 0.33, closer to 0.5)',
            '  - S5-4K: best relaxation in BOTH stocks (INTC: 0.763, GOOG: 0.870)',
            '  - INTC beta_perm ~ 0.92-0.99 (close to 1.0 target), GOOG beta_perm ~ 1.4-1.5',
            '  - Higher liquidity -> better relaxation -> closer to theory',
        ])

        for fig_name in ['2. Average Master Curve.png', '5. Relaxation Ratio.png', '17. Kyle Lambda per Insertion.png']:
            image_page(pdf, f'pics_for_v4_300_INTC/{fig_name}', title=f'INTC v4 — {fig_name[3:-4]}')

        # ═══════════════════════════════════════════════════════
        # SECTION 3: METHODOLOGICAL DISCOVERIES
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Section 3: Key Methodological Discoveries',
            '',
            '## Finding 1: Origin Estimator Bias (+0.51)',
            '',
            'The through-origin estimator log(I/sigma) = beta * log(Q/V) introduces',
            'a systematic positive bias when the true model has nonzero intercept alpha:',
            '',
            '  bias = (alpha - E[log sigma]) * E[x] / E[x^2]',
            '  With alpha ~ -9, E[log sigma] ~ -3.5, E[x] ~ -6.8: bias ~ +0.51',
            '',
            'This explains why ALL models gave beta_origin ~ 0.84 with no differentiation.',
            'The intercept estimator log(I) = alpha + beta * log(Q/V) is unbiased.',
            '',
            'Verified on 80,000 synthetic data points with known beta_true.',
            'Intercept estimator: recovered beta within +/- 0.003 across 20 scenarios.',
            'Origin estimator: biased by +0.13 to +0.66 depending on alpha.',
        ])

        image_page(pdf, 'pics_for_v4_300_GOOG/15. Beta Estimator Comparison.png',
                   title='Finding 1: Estimator Comparison (GOOG)')

        text_page(pdf, [
            '## Finding 3: Volume Calibration is Critical',
            '',
            'Original experiments used fixed volumes (75/300/485 shares) for all stocks.',
            '',
            '  GOOG: 75 shares = p38 of depth (marginal)',
            '  INTC: 75 shares = p6 of depth (broken — order never eats first level)',
            '',
            'Solution: calibrate from depth-at-best distribution per stock.',
            '  GOOG: {105, 165, 325} = p50/p75/p95',
            '  INTC: {590, 1110, 3120} = p50/p75/p95',
            '',
            '## Finding 4: Participation Rate phi ~ 99%',
            '',
            'Our aggressive orders dominate the market:',
            '  phi = Q_step / (Q_step + mb * r * q_med)',
            '  GOOG, mb=5, Q=105: phi = 105 / (105 + 1.4) = 98.6%',
            '',
            'This is a STRESS TEST, not realistic institutional execution (phi = 5-20%).',
            'The relative model ranking is valid regardless of phi.',
            '',
            '## Finding 7: Kyle Lambda — Strongest Differentiator',
            '',
            'Kyle lambda(k) = |exec_price - mid_before| / (tick * size) per insertion.',
            '  - At k=1: all models identical (same conditioning book)',
            '  - At k>5: models diverge sharply',
            '  - S5: lambda rises to 0.15-0.30 (realistic depletion + partial recovery)',
            '  - CST/CGAN: lambda stays flat 0.06-0.09 (over-replenish)',
            '  - Historic: lambda rises to 0.36 (no recovery at all)',
        ])

        # ═══════════════════════════════════════════════════════
        # SECTION 4: STOCK UNIVERSE
        # ═══════════════════════════════════════════════════════
        tickers = ['GOOG', 'AAPL', 'NVDA', 'AMZN', 'META', 'TSLA', 'MSFT', 'AMD', 'INTC']
        stock_lines = [
            '# Section 4: Stock Statistics — All 9 Tickers',
            '',
            'Computed from 20 trading days of LOBSTER L2 data, January 2026.',
            'S5-4K trained on: GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, AMD.',
            'INTC: NOT in training data (out-of-distribution test).',
            '',
            f'  {"Ticker":>6s}  {"r%":>6s}  {"q_med":>5s}  {"q_p75":>5s}  {"depth":>6s}  {"spr":>4s}  {"ADV_M":>6s}  {"r*q":>6s}  {"child@10%":>9s}',
        ]
        for t in tickers:
            s = stats.get(t, {})
            r = s.get('r', 0)
            q = s.get('q_med', 0)
            q75 = s.get('q_p75', 0)
            d = s.get('depth', 0)
            sp = s.get('spread', 0)
            adv = s.get('ADV', 0)
            rq = r * q / 100
            mb = 36
            c10 = int(math.floor(0.1/0.9 * mb * r/100 * q)) if r > 0 else 0
            stock_lines.append(
                f'  {t:>6s}  {r:>6.2f}  {q:>5.0f}  {q75:>5.0f}  {d:>6.0f}  {sp:>4.0f}  {adv:>6.1f}  {rq:>6.2f}  {c10:>9d}'
            )
        stock_lines += [
            '',
            'Key metric: r * q_med (execution fraction * median trade size).',
            'Higher r*q -> larger child at target phi -> more measurable impact.',
            '',
            'Best for low-phi experiments: INTC (r*q=3.60, but OOD),',
            'AAPL (r*q=1.09), NVDA (r*q=1.04). GOOG (r*q=0.30) is worst.',
        ]
        text_page(pdf, stock_lines)

        # ═══════════════════════════════════════════════════════
        # SECTION 5: v5 LOW PARTICIPATION RATE
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Section 5: v5 — Low Participation Rate (phi = 5-20%)',
            '',
            '## The Question',
            'v4 showed beta ~ 0.33 at phi ~ 99%. Can we measure beta at realistic',
            'phi = 5-20%? This would match institutional metaorder execution.',
            '',
            '## Design',
            '  i = 10 insertions (fixed)',
            '  c = 100 cooling periods',
            '  mb = 36 messages between',
            '  budget = 3,960 messages (within S5-4K 4,096 context)',
            '  child = floor(eta/(1-eta) * mb * r * q_med)',
            '',
            '## Child Sizes',
            '',
            f'  {"eta":>5s}  {"AAPL":>8s} {"depth%":>7s}  {"AMZN":>8s} {"depth%":>7s}',
            f'  {"5%":>5s}  {"2 sh":>8s} {"0.7%":>7s}  {"1 sh":>8s} {"0.4%":>7s}',
            f'  {"10%":>5s}  {"4 sh":>8s} {"1.3%":>7s}  {"2 sh":>8s} {"0.9%":>7s}',
            f'  {"20%":>5s}  {"9 sh":>8s} {"3.0%":>7s}  {"4 sh":>8s} {"1.8%":>7s}',
            '',
            'Compare to v4: GOOG child = 105-325 sh, child/depth = 63-196%.',
            'v5 child/depth = 0.4-3.0% — orders barely scratch the first level.',
            '',
            '## Models',
            '  Historic, Heuristic, CST — baselines (no context limit)',
            '  S5-4K — primary neural (4096 context, 8 stocks training)',
            '  S5-120M, S5-360M — also run (500 context, 8 stocks training)',
            '  LobS5 — cancelled (trained on GOOG only, OOD for AAPL/AMZN)',
        ])

        text_page(pdf, [
            '## v5 Results: Beta at All Participation Rates',
            '',
            'AAPL (all 3 eta pooled, all k):',
            '',
            f'  {"Model":>12s}  {"beta_int":>10s}  {"R2":>8s}  {"N":>8s}',
            f'  {"Historic":>12s}  {"-0.001":>10s}  {"0.000":>8s}  {"120108":>8s}',
            f'  {"Heuristic":>12s}  {"0.011":>10s}  {"0.000":>8s}  {"120202":>8s}',
            f'  {"CST":>12s}  {"0.011":>10s}  {"0.000":>8s}  {"120741":>8s}',
            f'  {"S5-4K":>12s}  {"-0.011":>10s}  {"0.000":>8s}  {"119771":>8s}',
            '',
            'AMZN (all 3 eta pooled, all k):',
            '',
            f'  {"Model":>12s}  {"beta_int":>10s}  {"R2":>8s}  {"N":>8s}',
            f'  {"Historic":>12s}  {"0.024":>10s}  {"0.001":>8s}  {"120282":>8s}',
            f'  {"Heuristic":>12s}  {"0.033":>10s}  {"0.001":>8s}  {"120359":>8s}',
            f'  {"CST":>12s}  {"0.019":>10s}  {"0.001":>8s}  {"120773":>8s}',
            f'  {"S5-4K":>12s}  {"0.015":>10s}  {"0.000":>8s}  {"120007":>8s}',
            '',
            'Beta ~ 0, R^2 ~ 0 for ALL models. VWAP impact is noise.',
            '',
            '## eta=20% Full Metaorder (k=10 only)',
            '',
            'AAPL (child=9, Q=90):',
            f'  {"Model":>12s}  {"beta":>8s}  {"CI":>18s}  {"mean_I":>10s}  {"lam1->10":>12s}',
            f'  {"Historic":>12s}  {"-0.217":>8s}  {"[-0.33, -0.10]":>18s}  {"6.4e-05":>10s}  {"0.138->0.157":>12s}',
            f'  {"Heuristic":>12s}  {"-0.197":>8s}  {"[-0.30, -0.08]":>18s}  {"6.7e-05":>10s}  {"0.138->0.152":>12s}',
            f'  {"CST":>12s}  {"-0.057":>8s}  {"[-0.15,  0.04]":>18s}  {"4.8e-05":>10s}  {"0.115->0.130":>12s}',
            f'  {"S5-4K":>12s}  {"-0.199":>8s}  {"[-0.32, -0.09]":>18s}  {"6.4e-05":>10s}  {"0.149->0.136":>12s}',
            '',
            'S5-4K: lambda FALLS (0.149 -> 0.136). All others: lambda RISES.',
        ])

        # v5 figures
        for fig_name in ['1. Master Curves.png', '2. Average Master Curve.png']:
            image_page(pdf, f'pics_for_v5_Q90_AAPL/{fig_name}', title=f'AAPL v5 (eta=20%) — {fig_name[3:-4]}')

        image_page(pdf, 'pics_for_v5_single_AAPL/1. Scatter log(I) vs log(QV).png',
                   title='AAPL v5 — Scatter (full metaorder, k=10)')
        image_page(pdf, 'pics_for_v5_single_AAPL/4. Kyle Lambda per Insertion.png',
                   title='AAPL v5 — Kyle Lambda per Insertion')
        image_page(pdf, 'pics_for_v5_single_AAPL/5. Beta Comparison.png',
                   title='AAPL v5 — Beta Comparison (full metaorder)')

        for fig_name in ['2. Average Master Curve.png']:
            image_page(pdf, f'pics_for_v5_Q40_AMZN/{fig_name}', title=f'AMZN v5 (eta=20%) — {fig_name[3:-4]}')

        image_page(pdf, 'pics_for_v5_single_AMZN/1. Scatter log(I) vs log(QV).png',
                   title='AMZN v5 — Scatter (full metaorder, k=10)')
        image_page(pdf, 'pics_for_v5_single_AMZN/4. Kyle Lambda per Insertion.png',
                   title='AMZN v5 — Kyle Lambda per Insertion')

        # ═══════════════════════════════════════════════════════
        # SECTION 6: KYLE LAMBDA COMPARISON
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Section 6: Kyle Lambda — Works at Any Scale',
            '',
            'Kyle lambda differentiates models regardless of participation rate:',
            '',
            '## v4 (phi ~ 99%, GOOG, child=105-325 shares):',
            '  S5 models: lambda rises to 0.15-0.30 (realistic depletion + recovery)',
            '  CST/CGAN:  lambda stays flat 0.06-0.09 (over-replenishment)',
            '  Historic:  lambda rises to 0.36 (full depletion, no recovery)',
            '',
            '## v5 (phi ~ 20%, AAPL, child=9 shares):',
            '  S5-4K:     lambda FALLS 0.149 -> 0.136 (book restoration!)',
            '  Baselines: lambda RISES (depletion without adequate recovery)',
            '',
            'The mechanism: S5-4K hidden state "sees" the aggressive order and',
            'generates compensatory limit orders that restore depth.',
            'At phi~99% depth is overwhelmed; at phi~20% restoration dominates.',
            '',
            'This makes Kyle lambda the most robust model differentiator:',
            '  - Does not require beta regression (works at single Q)',
            '  - Does not require measurable VWAP impact',
            '  - Reveals model mechanism (hidden state -> book restoration)',
        ])

        image_page(pdf, 'pics_for_v4_300_GOOG/17. Kyle Lambda per Insertion.png',
                   title='Kyle Lambda: v4 GOOG (phi~99%, 10 models)')
        image_page(pdf, 'pics_for_v5_single_AAPL/4. Kyle Lambda per Insertion.png',
                   title='Kyle Lambda: v5 AAPL (phi~20%, 4 models)')

        # ═══════════════════════════════════════════════════════
        # SECTION 7: ADDITIONAL v4 DIAGNOSTICS
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Section 7: Additional v4 Diagnostics (GOOG)',
            '',
            'Selected figures from the standard pipeline (v4, phi~99%):',
        ])

        for fig_name in [
            '1. Master Curves.png',
            '13. Perm Temp Decomposition.png',
            '7. Hurst Exponent.png',
            '8. Propagator G(l).png',
            '9. Spread Dynamics.png',
            '12. Per-Day Beta.png',
            '16. Per-Insertion Midprice Response.png',
        ]:
            image_page(pdf, f'pics_for_v4_300_GOOG/{fig_name}', title=f'GOOG v4 — {fig_name[3:-4]}')

        # ═══════════════════════════════════════════════════════
        # SECTION 8: TRADE SIZE ANALYSIS
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Section 8: Trade Size Analysis — What Next?',
            '',
            '## Why phi-based child sizing fails',
            '',
            'Fixing phi determines child. At phi=10%:',
            '  child ~ 2-4 shares (AAPL/AMZN)',
            '  p25 trade size = 7-10 shares',
            '  -> child is 3-7x SMALLER than the smallest real trade',
            '',
            '## Proposed: use trade size percentiles as child',
            '',
            'AAPL:',
            f'  {"child":>10s}  {"Q=10*child":>10s}  {"child/depth":>12s}  {"phi":>8s}',
            f'  {"p50=40":>10s}  {"400":>10s}  {"13%":>12s}  {"50%":>8s}',
            f'  {"p75=100":>10s}  {"1000":>10s}  {"33%":>12s}  {"72%":>8s}',
            f'  {"p95=140":>10s}  {"1400":>10s}  {"47%":>12s}  {"78%":>8s}',
            '',
            'AMZN:',
            f'  {"child":>10s}  {"Q=10*child":>10s}  {"child/depth":>12s}  {"phi":>8s}',
            f'  {"p50=25":>10s}  {"250":>10s}  {"11%":>12s}  {"56%":>8s}',
            f'  {"p75=100":>10s}  {"1000":>10s}  {"45%":>12s}  {"84%":>8s}',
            f'  {"p95=131":>10s}  {"1310":>10s}  {"58%":>12s}  {"87%":>8s}',
            '',
            'phi = 50-87% is between v4 stress test (99%) and failed v5 (10-20%).',
            'child/depth = 11-58% — should produce measurable impact.',
            'All child sizes are REAL trade sizes from the data.',
        ])

        # ═══════════════════════════════════════════════════════
        # SECTION 9: CONCLUSIONS
        # ═══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Section 9: Conclusions',
            '',
            '## What We Know (from v4)',
            '',
            '1. S5 models cluster at beta ~ 0.33 (GOOG) / 0.40 (INTC)',
            '   with intercept estimator. Origin estimator masks differences.',
            '',
            '2. S5-4K achieves best relaxation (GOOG 0.870, INTC 0.763),',
            '   closest to theoretical 2/3. Context length > model size.',
            '',
            '3. Kyle lambda trajectory is the strongest model differentiator.',
            '   S5 models show realistic book depletion + recovery.',
            '',
            '4. All S5 models: Hurst H ~ 0.50 (no long memory). Negative finding.',
            '',
            '## What We Learned (from v5)',
            '',
            '5. At realistic phi (5-20%), child orders are too small for',
            '   measurable VWAP impact. Beta ~ 0, R^2 ~ 0.',
            '',
            '6. Fundamental tension: realistic phi requires tiny child,',
            '   but tiny child cannot move the price. phi and impact are',
            '   inversely coupled through book depth.',
            '',
            '7. Kyle lambda STILL differentiates models at low phi.',
            '   S5-4K uniquely shows lambda decrease (book restoration).',
            '',
            '8. The v4 stress-test approach (phi~99%) remains the correct',
            '   methodology for beta estimation in LOB simulation.',
            '',
            '## Next Steps',
            '',
            'A. Trade-size grid: child = p50/p75/p95 trade size (phi 50-87%)',
            'B. Model response study: single child = p50, focus on dynamics',
            'C. Both A + B for the paper',
        ])

    print(f'\nSaved: {pdf_path} ({pdf_path.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
