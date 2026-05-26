#!/usr/bin/env python3
"""
v5 Low-Participation-Rate Experiment Report — Single PDF.

Compiles all results, figures, and analysis into one comprehensive document.

Usage:
    python lob_impact/run_v5_report.py
"""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import OrderedDict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings; warnings.filterwarnings('ignore')

OUT = Path('pics_for_v5_report')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = OrderedDict([
    ('Historic',  '#90939C'),
    ('Heuristic', '#546884'),
    ('CST',       '#213552'),
    ('S5-4K',     '#D95F02'),
    ('S5-120M',   '#5B7BBF'),
    ('S5-360M',   '#B5446E'),
])


def text_page(pdf, lines, fontsize=11, title=None):
    """Create a text-only page."""
    fig = plt.figure(figsize=(11, 8.5))
    y = 0.92
    if title:
        fig.text(0.5, 0.96, title, ha='center', fontsize=16, fontweight='bold')
        y = 0.90
    for line in lines:
        if line.startswith('##'):
            fig.text(0.06, y, line[2:].strip(), fontsize=13, fontweight='bold')
            y -= 0.035
        elif line.startswith('#'):
            fig.text(0.06, y, line[1:].strip(), fontsize=14, fontweight='bold')
            y -= 0.04
        elif line == '':
            y -= 0.015
        elif line.startswith('  '):
            fig.text(0.08, y, line, fontsize=9, fontfamily='monospace')
            y -= 0.022
        else:
            fig.text(0.06, y, line, fontsize=fontsize)
            y -= 0.028
        if y < 0.04:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            y = 0.94
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, img_path, title=None):
    """Insert a PNG image as a full page."""
    if not Path(img_path).exists():
        print(f'  SKIP (not found): {img_path}')
        return
    img = plt.imread(str(img_path))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img)
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    pdf.savefig(fig); plt.close(fig)


def load_metrics(directory, models=None):
    """Load all .metrics.pkl from a directory."""
    d = Path(directory)
    if not d.exists():
        return {}
    result = {}
    for f in sorted(d.glob('*.metrics.pkl')):
        model = f.stem.replace('.metrics', '')
        if models and model not in models:
            continue
        with open(f, 'rb') as fh:
            result[model] = pickle.load(fh)
    return result


def load_lobster_stats(stock):
    """Load lobster_stats CSV and compute medians."""
    path = f'lob_impact/lobster_stats_{stock}.csv'
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    return {
        'trade_fraction': df['trade_fraction'].median(),
        'trade_size_p25': df['trade_size_p25'].median(),
        'trade_size_p50': df['trade_size_p50'].median(),
        'trade_size_p75': df['trade_size_p75'].median(),
        'trade_size_p95': df['trade_size_p95'].median(),
        'depth_best_p50': df['depth_best_p50'].median(),
        'spread_ticks_median': df['spread_ticks_median'].median(),
        'ADV': df['ADV'].median(),
    }


def main():
    pdf_path = OUT / 'v5_experiment_report.pdf'
    print(f'Generating report: {pdf_path}')

    # Load stock stats
    aapl = load_lobster_stats('AAPL')
    amzn = load_lobster_stats('AMZN')

    # Load metrics
    metrics_q90 = load_metrics('pics_for_v5_Q90_AAPL')
    metrics_q40 = load_metrics('pics_for_v5_Q40_AMZN')
    metrics_300_aapl = load_metrics('pics_for_v5_300_AAPL')
    metrics_300_amzn = load_metrics('pics_for_v5_300_AMZN')

    with PdfPages(str(pdf_path)) as pdf:

        # ══════════════════════════════════════════════════════════
        # PAGE 1: Title + Executive Summary
        # ══════════════════════════════════════════════════════════
        text_page(pdf, [
            '# v5 Low-Participation-Rate Experiments',
            '',
            'Date: March 30, 2026',
            'Stocks: AAPL, AMZN (both in S5-4K training data)',
            'Models: Historic, Heuristic, CST, S5-4K',
            'Samples: 2,048 per (config, direction)',
            '',
            '## Goal',
            'Test market impact at realistic participation rates (phi = 5-20%)',
            'instead of v4 stress-test regime (phi ~ 99%).',
            'All parameters derived from stock statistics (no arbitrary values).',
            '',
            '## Key Findings',
            '',
            '1. Child orders derived from target phi are 3-7x SMALLER than',
            '   real market trades (p25). Impact is unmeasurable (VWAP I ~ 6e-5).',
            '',
            '2. Beta ~ 0 with R^2 ~ 0 for ALL models at phi = 5-20%.',
            '   Origin estimator gives beta ~ 0.5 but this is the known bias.',
            '',
            '3. Kyle lambda is the ONLY working differentiator:',
            '   - S5-4K: lambda FALLS from k=1 to k=10 (book restoration)',
            '   - All baselines: lambda RISES (book depletion without recovery)',
            '',
            '4. Fundamental tension: realistic phi requires tiny orders that',
            '   cannot move the price. Measurable impact requires phi >> 20%.',
            '',
            '## Implication for Paper',
            'The v4 stress-test approach (phi ~ 99%) remains the correct',
            'methodology for measuring market impact in LOB simulation.',
            'At realistic phi, the metaorder is too small relative to book depth.',
        ], title='v5 Experiment Report')

        # ══════════════════════════════════════════════════════════
        # PAGE 2: Experimental Setup
        # ══════════════════════════════════════════════════════════
        r_aapl = aapl.get('trade_fraction', 0.0273) * 100
        r_amzn = amzn.get('trade_fraction', 0.0217) * 100
        q_aapl = aapl.get('trade_size_p50', 40)
        q_amzn = amzn.get('trade_size_p50', 25)
        d_aapl = aapl.get('depth_best_p50', 300)
        d_amzn = amzn.get('depth_best_p50', 224)

        text_page(pdf, [
            '# Experimental Setup',
            '',
            '## Stock Statistics (Jan 2026, 20 trading days)',
            '',
            f'  {"":>20s}  {"AAPL":>10s}  {"AMZN":>10s}',
            f'  {"Exec fraction (r)":>20s}  {r_aapl:>9.2f}%  {r_amzn:>9.2f}%',
            f'  {"Median trade size":>20s}  {q_aapl:>10.0f}  {q_amzn:>10.0f}',
            f'  {"Depth at best p50":>20s}  {d_aapl:>10.0f}  {d_amzn:>10.0f}',
            f'  {"Spread (ticks)":>20s}  {aapl.get("spread_ticks_median",2):>10.0f}  {amzn.get("spread_ticks_median",2):>10.0f}',
            f'  {"ADV (M shares)":>20s}  {aapl.get("ADV",6.4e6)/1e6:>10.1f}  {amzn.get("ADV",5.2e6)/1e6:>10.1f}',
            '',
            '## Fixed Parameters',
            '  i = 10 (child orders per metaorder)',
            '  c = 100 (cooling periods)',
            '  mb = 36 (messages between insertions)',
            '  budget = 3,960 messages (within S5-4K 4,096 context)',
            '  n_samples = 2,048 per (config, direction)',
            '',
            '## Child Sizes at Target Participation Rates',
            '',
            '  Formula: child = floor(eta/(1-eta) * mb * r * q_med)',
            '',
            f'  {"eta":>5s}  {"AAPL child":>11s}  {"child/depth":>11s}  {"AMZN child":>11s}  {"child/depth":>11s}',
            f'  {"5%":>5s}  {"2":>11s}  {"0.7%":>11s}  {"1":>11s}  {"0.4%":>11s}',
            f'  {"10%":>5s}  {"4":>11s}  {"1.3%":>11s}  {"2":>11s}  {"0.9%":>11s}',
            f'  {"20%":>5s}  {"9":>11s}  {"3.0%":>11s}  {"4":>11s}  {"1.8%":>11s}',
            '',
            '## Models',
            '  Historic   — replay historical messages (no adaptation)',
            '  Heuristic  — replay + price shift when level consumed',
            '  CST        — parametric (Cont-Stoikov-Talreja, memoryless)',
            '  S5-4K      — neural autoregressive, 55M params, 4096 context',
            '               Trained on 8 stocks (incl. AAPL, AMZN), 2022-2025',
        ])

        # ══════════════════════════════════════════════════════════
        # PAGE 3: Why phi-Based Child Sizing Fails
        # ══════════════════════════════════════════════════════════
        text_page(pdf, [
            '# Why phi-Based Child Sizing Fails',
            '',
            '## Child vs Real Trade Sizes',
            '',
            f'  {"":>15s}  {"AAPL":>30s}  {"AMZN":>30s}',
            f'  {"":>15s}  {"child":>8s} {"p25 trade":>10s} {"ratio":>10s}  {"child":>8s} {"p25 trade":>10s} {"ratio":>10s}',
            f'  {"eta=5%":>15s}  {"2":>8s} {"10":>10s} {"5x small":>10s}  {"1":>8s} {"7":>10s} {"7x small":>10s}',
            f'  {"eta=10%":>15s}  {"4":>8s} {"10":>10s} {"2.5x small":>10s}  {"2":>8s} {"7":>10s} {"3.5x small":>10s}',
            f'  {"eta=20%":>15s}  {"9":>8s} {"10":>10s} {"~OK":>10s}  {"4":>8s} {"7":>10s} {"1.8x small":>10s}',
            '',
            'At eta=5% and eta=10%, child orders are 3-7x SMALLER than the',
            'smallest typical trade (p25). Such orders barely exist in real markets.',
            '',
            '## child/depth Comparison: v4 vs v5',
            '',
            f'  {"Experiment":>15s}  {"child":>8s}  {"depth":>8s}  {"child/depth":>12s}  {"Impact?":>10s}',
            f'  {"v4 GOOG p50":>15s}  {"105":>8s}  {"166":>8s}  {"63%":>12s}  {"YES":>10s}',
            f'  {"v4 GOOG p95":>15s}  {"325":>8s}  {"166":>8s}  {"196%":>12s}  {"YES":>10s}',
            f'  {"v5 AAPL 20%":>15s}  {"9":>8s}  {"300":>8s}  {"3%":>12s}  {"NO":>10s}',
            f'  {"v5 AMZN 20%":>15s}  {"4":>8s}  {"224":>8s}  {"1.8%":>12s}  {"NO":>10s}',
            '',
            'v4 orders consumed 63-196% of the best level -> price moved.',
            'v5 orders consume 1-3% of the best level -> price stays flat.',
            '',
            '## Measured Impact',
            '  AAPL mean_I = 6.4e-5 (0.006% of price = 0.014 cents on $230)',
            '  AMZN mean_I = 7.3e-5 (0.007% of price = 0.015 cents on $215)',
            '  This is indistinguishable from microstructure noise.',
        ])

        # ══════════════════════════════════════════════════════════
        # PAGE 4: All-eta Results (from 300 pipeline)
        # ══════════════════════════════════════════════════════════
        text_page(pdf, [
            '# Results: All Three Participation Rates (eta=5%/10%/20%)',
            '',
            '## Beta (intercept estimator, all k pooled)',
            '',
            'AAPL (all 3 configs combined):',
            '',
            f'  {"Model":>12s}  {"beta_int":>10s}  {"R2":>8s}  {"N":>8s}',
            f'  {"Historic":>12s}  {"-0.001":>10s}  {"0.000":>8s}  {"120108":>8s}',
            f'  {"Heuristic":>12s}  {"0.011":>10s}  {"0.000":>8s}  {"120202":>8s}',
            f'  {"CST":>12s}  {"0.011":>10s}  {"0.000":>8s}  {"120741":>8s}',
            f'  {"S5-4K":>12s}  {"-0.011":>10s}  {"0.000":>8s}  {"119771":>8s}',
            '',
            'AMZN (all 3 configs combined):',
            '',
            f'  {"Model":>12s}  {"beta_int":>10s}  {"R2":>8s}  {"N":>8s}',
            f'  {"Historic":>12s}  {"0.024":>10s}  {"0.001":>8s}  {"120282":>8s}',
            f'  {"Heuristic":>12s}  {"0.033":>10s}  {"0.001":>8s}  {"120359":>8s}',
            f'  {"CST":>12s}  {"0.019":>10s}  {"0.001":>8s}  {"120773":>8s}',
            f'  {"S5-4K":>12s}  {"0.015":>10s}  {"0.000":>8s}  {"120007":>8s}',
            '',
            'Beta ~ 0, R^2 ~ 0 for ALL models. No measurable impact scaling.',
        ])

        # ══════════════════════════════════════════════════════════
        # PAGE 5: eta=20% Full Metaorder Results
        # ══════════════════════════════════════════════════════════
        text_page(pdf, [
            '# Results: eta=20% Full Metaorder (k=10 only)',
            '',
            'Each sample = one complete metaorder (10 insertions).',
            'Impact measured at the LAST insertion (cumulative VWAP).',
            '',
            'AAPL (child=9, Q=90, phi~19%):',
            '',
            f'  {"Model":>12s}  {"beta":>8s}  {"CI_lo":>8s}  {"CI_hi":>8s}  {"R2":>8s}  {"N":>6s}  {"mean_I":>10s}  {"lam1":>8s}  {"lam10":>8s}',
            f'  {"Historic":>12s}  {"-0.217":>8s}  {"-0.331":>8s}  {"-0.102":>8s}  {"0.004":>8s}  {"4028":>6s}  {"6.4e-05":>10s}  {"0.138":>8s}  {"0.157":>8s}',
            f'  {"Heuristic":>12s}  {"-0.197":>8s}  {"-0.304":>8s}  {"-0.083":>8s}  {"0.003":>8s}  {"4028":>6s}  {"6.7e-05":>10s}  {"0.138":>8s}  {"0.152":>8s}',
            f'  {"CST":>12s}  {"-0.057":>8s}  {"-0.153":>8s}  {"0.039":>8s}  {"0.000":>8s}  {"4020":>6s}  {"4.8e-05":>10s}  {"0.115":>8s}  {"0.130":>8s}',
            f'  {"S5-4K":>12s}  {"-0.199":>8s}  {"-0.317":>8s}  {"-0.089":>8s}  {"0.003":>8s}  {"4018":>6s}  {"6.4e-05":>10s}  {"0.149":>8s}  {"0.136":>8s}',
            '',
            'AMZN (child=4, Q=40, phi~17%):',
            '',
            f'  {"Model":>12s}  {"beta":>8s}  {"CI_lo":>8s}  {"CI_hi":>8s}  {"R2":>8s}  {"N":>6s}  {"mean_I":>10s}  {"lam1":>8s}  {"lam10":>8s}',
            f'  {"Historic":>12s}  {"0.114":>8s}  {"-0.016":>8s}  {"0.252":>8s}  {"0.001":>8s}  {"4028":>6s}  {"7.4e-05":>10s}  {"0.334":>8s}  {"0.361":>8s}',
            f'  {"Heuristic":>12s}  {"0.070":>8s}  {"-0.069":>8s}  {"0.215":>8s}  {"0.000":>8s}  {"4031":>6s}  {"7.6e-05":>10s}  {"0.334":>8s}  {"0.352":>8s}',
            f'  {"CST":>12s}  {"0.044":>8s}  {"-0.076":>8s}  {"0.164":>8s}  {"0.000":>8s}  {"4030":>6s}  {"5.3e-05":>10s}  {"0.267":>8s}  {"0.286":>8s}',
            f'  {"S5-4K":>12s}  {"0.071":>8s}  {"-0.070":>8s}  {"0.212":>8s}  {"0.000":>8s}  {"4030":>6s}  {"7.3e-05":>10s}  {"0.336":>8s}  {"0.318":>8s}',
            '',
            'Note: S5-4K is the ONLY model where Kyle lambda DECREASES',
            '(0.149->0.136 AAPL, 0.336->0.318 AMZN). All baselines increase.',
            'This indicates S5-4K restores the book between injections.',
        ])

        # ══════════════════════════════════════════════════════════
        # FIGURE PAGES
        # ══════════════════════════════════════════════════════════

        # Single-config figures (k=10 analysis)
        for stock in ['AAPL', 'AMZN']:
            q = '90' if stock == 'AAPL' else '40'
            d = f'pics_for_v5_single_{stock}'
            for fig_file in [
                '1. Scatter log(I) vs log(QV).png',
                '2. Average Master Curve.png',
                '3. Impact Distribution.png',
                '4. Kyle Lambda per Insertion.png',
                '5. Beta Comparison.png',
            ]:
                image_page(pdf, f'{d}/{fig_file}',
                          title=f'{stock} (child={9 if stock=="AAPL" else 4}, Q={q})')

        # Standard pipeline figures (eta=20% only)
        for stock, d in [('AAPL', 'pics_for_v5_Q90_AAPL'), ('AMZN', 'pics_for_v5_Q40_AMZN')]:
            for fig_file in [
                '1. Master Curves.png',
                '2. Average Master Curve.png',
                '5. Relaxation Ratio.png',
                '6. Fraction Stable.png',
                '9. Spread Dynamics.png',
                '8. Propagator G(l).png',
                '17. Kyle Lambda per Insertion.png',
                '16. Per-Insertion Midprice Response.png',
            ]:
                image_page(pdf, f'{d}/{fig_file}', title=f'{stock} (standard pipeline)')

        # ══════════════════════════════════════════════════════════
        # TRADE SIZE ANALYSIS PAGE
        # ══════════════════════════════════════════════════════════
        text_page(pdf, [
            '# Trade Size Analysis: Proposed Next Steps',
            '',
            '## Actual Trade Size Percentiles (Jan 2026)',
            '',
            f'  {"":>10s}  {"AAPL":>10s}  {"AMZN":>10s}',
            f'  {"p25":>10s}  {"10":>10s}  {"7":>10s}',
            f'  {"p50":>10s}  {"40":>10s}  {"25":>10s}',
            f'  {"p75":>10s}  {"100":>10s}  {"100":>10s}',
            f'  {"p95":>10s}  {"140":>10s}  {"131":>10s}',
            '',
            '## Proposed New Grid: child = trade size percentiles',
            '',
            f'  {"child":>10s}  {"AAPL Q":>10s}  {"child/depth":>12s}  {"phi":>8s}',
            f'  {"p50=40":>10s}  {"400":>10s}  {"13%":>12s}  {"50%":>8s}',
            f'  {"p75=100":>10s}  {"1000":>10s}  {"33%":>12s}  {"72%":>8s}',
            f'  {"p95=140":>10s}  {"1400":>10s}  {"47%":>12s}  {"78%":>8s}',
            '',
            f'  {"child":>10s}  {"AMZN Q":>10s}  {"child/depth":>12s}  {"phi":>8s}',
            f'  {"p50=25":>10s}  {"250":>10s}  {"11%":>12s}  {"56%":>8s}',
            f'  {"p75=100":>10s}  {"1000":>10s}  {"45%":>12s}  {"84%":>8s}',
            f'  {"p95=131":>10s}  {"1310":>10s}  {"58%":>12s}  {"87%":>8s}',
            '',
            'These child sizes are REAL trade sizes from the data.',
            'phi = 50-87% is between v4 stress-test (99%) and failed v5 (10-20%).',
            'child/depth = 11-58% should produce measurable impact.',
        ])

        # ══════════════════════════════════════════════════════════
        # CONCLUSIONS
        # ══════════════════════════════════════════════════════════
        text_page(pdf, [
            '# Conclusions and Next Steps',
            '',
            '## What We Learned',
            '',
            '1. Participation rate phi cannot be independently controlled in LOB',
            '   simulation. Fixing phi determines child size, and at phi < 20%',
            '   the child is too small to produce measurable VWAP impact.',
            '',
            '2. The fundamental constraint:',
            '   phi = child / (child + mb * r * q_med)',
            '   At phi=10%: child ~ 4 shares (AAPL), 2 shares (AMZN)',
            '   But depth ~ 224-300 shares -> child/depth < 2% -> no impact',
            '',
            '3. Kyle lambda trajectory differentiates models REGARDLESS of',
            '   child size. S5-4K uniquely shows lambda DECREASE (book',
            '   restoration via hidden state), while all baselines show increase.',
            '',
            '4. The v4 stress-test approach (phi~99%, child > depth) remains',
            '   the correct methodology for beta estimation.',
            '',
            '## Recommended Next Steps',
            '',
            'Option A: Trade-size-based grid',
            '  child = p50/p75/p95 of actual trade sizes',
            '  phi = 50-87% (between v4 and v5)',
            '  3 Q-points for beta regression',
            '  Same framework, just larger orders',
            '',
            'Option B: Reframe as "model response to realistic trades"',
            '  Keep child = median trade size (p50)',
            '  Focus on Kyle lambda, master curves, relaxation',
            '  Not beta regression (single Q, no variation)',
            '  Story: "how does each model respond to a normal-sized trade?"',
            '',
            'Option C: Combined approach',
            '  Run trade-size grid (A) for beta estimation',
            '  Add single-config analysis (B) for model response narrative',
            '  Present both in the paper',
        ])

    print(f'\nReport saved: {pdf_path}')
    print(f'Pages: ~{14 + 10 + 16} (text + single figs + pipeline figs)')


if __name__ == '__main__':
    main()
