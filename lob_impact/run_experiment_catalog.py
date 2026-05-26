#!/usr/bin/env python3
"""
EXPERIMENT CATALOG: Every grid, every scheme, every β calculation.

Separate document with complete tables for every experiment version.
For each: grid parameters, volume calibration, β formula, β results,
master curves, Kyle λ, relaxation.
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings; warnings.filterwarnings('ignore')

A4_W, A4_H = 8.27, 11.69
OUT = Path('pics_for_experiment_catalog')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.2})

def text_page(pdf, lines, title=None, fs=9):
    fig = plt.figure(figsize=(A4_W, A4_H))
    y = 0.95
    if title:
        fig.text(0.5, 0.97, title, ha='center', fontsize=14, fontweight='bold')
        y = 0.93
    for line in lines:
        if line.startswith('##'):
            fig.text(0.04, y, line[2:].strip(), fontsize=11, fontweight='bold'); y -= 0.025
        elif line.startswith('#'):
            fig.text(0.04, y, line[1:].strip(), fontsize=13, fontweight='bold'); y -= 0.032
        elif line == '':
            y -= 0.008
        elif line.startswith('  '):
            fig.text(0.05, y, line, fontsize=7.5, fontfamily='monospace'); y -= 0.013
        else:
            fig.text(0.04, y, line, fontsize=fs); y -= 0.016
        if y < 0.03:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(A4_W, A4_H)); y = 0.96
    pdf.savefig(fig); plt.close(fig)

def img_page(pdf, path, title=None):
    p = Path(path)
    if not p.exists(): print(f'  SKIP: {path}'); return
    img = plt.imread(str(p))
    fig, ax = plt.subplots(figsize=(A4_W, A4_H * 0.8))
    ax.imshow(img); ax.axis('off')
    if title: fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else [0, 0, 1, 1])
    pdf.savefig(fig); plt.close(fig)

def load_csv(p):
    return pd.read_csv(p) if Path(p).exists() else None

def fmt_table(df, cols):
    if df is None: return ['  (no data)']
    lines = []
    hdr = '  ' + '  '.join(f'{c[:8]:>8s}' for c in cols)
    lines.append(hdr)
    lines.append('  ' + '-' * (len(cols) * 10))
    for _, row in df.iterrows():
        m = str(row.get('Model', ''))
        if m == 'ZeroInsertions': continue
        vals = []
        for c in cols:
            v = row.get(c, np.nan)
            if c == 'Model': vals.append(f'{str(v):>8s}')
            elif c == 'N': vals.append(f'{int(v):>8d}' if pd.notna(v) else f'{"":>8s}')
            elif c in ('no_arb_score',): vals.append(f'{int(v):>8d}' if pd.notna(v) else f'{"":>8s}')
            else: vals.append(f'{float(v):>8.4f}' if pd.notna(v) else f'{"---":>8s}')
        lines.append('  ' + '  '.join(vals))
    return lines


def main():
    pdf_path = OUT / 'experiment_catalog.pdf'
    print(f'Generating: {pdf_path}')

    v1g = load_csv('pics_for_300_GOOG/summary_statistics.csv')
    v1i = load_csv('pics_for_300_INTC/summary_statistics.csv')
    v4g = load_csv('pics_for_v4_300_GOOG/summary_statistics.csv')
    v4i = load_csv('pics_for_v4_300_INTC/summary_statistics.csv')
    v5r1a = load_csv('pics_for_v5_Q90_AAPL/summary_statistics.csv')
    v5r1z = load_csv('pics_for_v5_Q40_AMZN/summary_statistics.csv')
    v5r2a = load_csv('pics_for_v5r2_AAPL/summary_statistics.csv')
    v5r2z = load_csv('pics_for_v5r2_AMZN/summary_statistics.csv')
    beta_phi = load_csv('pics_for_beta_phi/beta_per_phi.csv')

    with PdfPages(str(pdf_path)) as pdf:

        # ═══════════ TITLE ═══════════
        text_page(pdf, [
            '', '', '',
            '# EXPERIMENT CATALOG',
            '# Every Grid, Every Scheme, Every Beta',
            '', '',
            'Complete reference of all market impact experiments.',
            'For each version: grid parameters, β formula, β results,',
            'master curves, Kyle λ, relaxation.',
            '', '',
            '## Contents',
            '',
            '1. V1/V2: Uncalibrated (GOOG, INTC) — origin estimator',
            '2. V4: Calibrated Production Grid (GOOG, INTC) — intercept estimator',
            '3. V5r1: Low Participation Rate (AAPL, AMZN) — η-derived child',
            '4. V5r2: Trade-Size Grid (AAPL, AMZN) — p25/p50/p75 child',
            '5. β(φ) Per-Volume Stratification — all versions combined',
            '6. Per-k Beta Analysis — β at each insertion',
            '7. Stock Statistics — all 9 tickers',
        ], title='EXPERIMENT CATALOG')

        # ═══════════ 1. V1/V2 ═══════════
        text_page(pdf, [
            '# 1. V1/V2: Uncalibrated Experiments',
            '',
            '## Grid Parameters',
            '',
            '  Stock:       GOOG (primary), INTC (validation)',
            '  Volumes:     75 / 300 / 485 shares (ARBITRARY, not calibrated)',
            '  (i, mb) pairs: (3,5) (5,5) (9,5) (2,10) (3,10) (4,10) (2,15) (3,15) (1,20) (2,20)',
            '  Cooling:     c = 10 × i',
            '  Constraint:  11 × i × mb ≤ 500',
            '  Directions:  buy, sell (antisymmetrised)',
            '  Samples:     2,048 per config',
            '  Total:       10 pairs × 3 vol × 2 dir = 60 configs',
            '',
            '  GOOG depth_p50 = 166 shares',
            '    vol=75:  child/depth = 45% → sometimes penetrates first level',
            '    vol=300: child/depth = 181% → always penetrates',
            '    vol=485: child/depth = 292% → always penetrates multiple levels',
            '',
            '  INTC depth_p50 = 1248 shares',
            '    vol=75:  child/depth = 6% → NEVER penetrates (BROKEN!)',
            '    vol=300: child/depth = 24% → rarely penetrates',
            '    vol=485: child/depth = 39% → sometimes penetrates',
            '',
            '## β Estimation Formula',
            '',
            '  ORIGIN (biased):    ln(I/σ) = β · ln(Q/V)',
            '  INTERCEPT (correct): ln(I) = α + β · ln(Q/V)',
            '  where I = |VWAP - mid_arrival| / mid_arrival',
            '        Q = cumulative injected volume (per-metaorder)',
            '        V = daily traded volume',
            '        σ = Parkinson volatility',
            '',
            '  Bias: β_origin ≈ β_true + 0.51',
            '',
            '  Participation rate φ ≈ 99% (stress test regime)',
            '',
            '## β Results — V1/V2 GOOG (9 models)',
            '',
        ] + fmt_table(v1g, ['Model', 'beta_intercept', 'beta_origin', 'CI_lo', 'CI_hi', 'R2', 'N', 'relaxation']) + [
            '',
            '  KEY: β_origin ≈ 0.80 for ALL models → no differentiation',
            '       β_intercept ≈ 0.08-0.23 → models separated but low',
            '       Volume miscalibrated → V3/V4 needed',
        ])

        img_page(pdf, 'pics_for_300_GOOG/2. Average Master Curve.png', 'V1/V2 GOOG — Master Curve')
        img_page(pdf, 'pics_for_300_GOOG/5. Relaxation Ratio.png', 'V1/V2 GOOG — Relaxation')
        img_page(pdf, 'pics_for_300_GOOG/17. Kyle Lambda per Insertion.png', 'V1/V2 GOOG — Kyle λ')

        # V1/V2 INTC
        text_page(pdf, [
            '## V1/V2 INTC Results',
            '',
        ] + fmt_table(v1i, ['Model', 'beta_intercept', 'beta_origin', 'R2', 'N', 'relaxation']))

        # ═══════════ 2. V4 ═══════════
        text_page(pdf, [
            '# 2. V4: Calibrated Production Grid',
            '',
            '## Grid Parameters',
            '',
            '  Stock:       GOOG, INTC',
            '  Volumes:     CALIBRATED from depth-at-best percentiles (122,880 obs)',
            '    GOOG: 105 (p50) / 165 (p75) / 325 (p95)',
            '    INTC: 590 (p50) / 1110 (p75) / 3120 (p95)',
            '  (i, mb) pairs: same 10 as V1/V2',
            '  Cooling:     c = 10 × i',
            '  Samples:     2,048 per config',
            '  Total:       60 configs per stock',
            '',
            '  GOOG: child/depth = 63% (p50), 99% (p75), 196% (p95)',
            '  INTC: child/depth = 47% (p50), 89% (p75), 250% (p95)',
            '  → Orders consistently penetrate first level → measurable impact',
            '',
            '  Participation rate φ ≈ 95-99%',
            '',
            '## β Estimation Formula',
            '  INTERCEPT (primary): ln(I) = α + β · ln(Q/V)',
            '  ORIGIN (comparison):  ln(I/σ) = β · ln(Q/V)',
            '  Bootstrap: 2,000 resamples, sample-level',
            '',
            '## β Results — V4 GOOG (9 models, ~82K metaorders each)',
            '',
        ] + fmt_table(v4g, ['Model', 'beta_intercept', 'beta_origin', 'CI_lo', 'CI_hi', 'R2', 'N', 'relaxation', 'Hurst', 'no_arb_score']) + [
            '',
            '  KEY: S5 cluster at β ≈ 0.33, CGAN 0.205, CST 0.256',
            '       S5-4K best relaxation (0.870, closest to 2/3)',
            '       Hurst H ≈ 0.50 (no long memory — negative finding)',
        ])

        img_page(pdf, 'pics_for_v4_300_GOOG/2. Average Master Curve.png', 'V4 GOOG — Master Curve')
        img_page(pdf, 'pics_for_v4_300_GOOG/3. Beta Regression Lines.png', 'V4 GOOG — β Regression')
        img_page(pdf, 'pics_for_v4_300_GOOG/4. Bootstrap Beta Distributions.png', 'V4 GOOG — Bootstrap β')
        img_page(pdf, 'pics_for_v4_300_GOOG/5. Relaxation Ratio.png', 'V4 GOOG — Relaxation')
        img_page(pdf, 'pics_for_v4_300_GOOG/17. Kyle Lambda per Insertion.png', 'V4 GOOG — Kyle λ')
        img_page(pdf, 'pics_for_v4_300_GOOG/15. Beta Estimator Comparison.png', 'V4 GOOG — Estimator Comparison')

        # V4 INTC
        text_page(pdf, [
            '## V4 INTC Results (9 models, ~72K each)',
            '',
        ] + fmt_table(v4i, ['Model', 'beta_intercept', 'CI_lo', 'CI_hi', 'R2', 'N', 'relaxation', 'Hurst', 'no_arb_score']) + [
            '',
            '  KEY: S5 β ≈ 0.40 (closer to 0.5 than GOOG 0.33)',
            '       S5-4K relaxation = 0.763 (closest to 2/3 across all)',
        ])

        img_page(pdf, 'pics_for_v4_300_INTC/2. Average Master Curve.png', 'V4 INTC — Master Curve')
        img_page(pdf, 'pics_for_v4_300_INTC/5. Relaxation Ratio.png', 'V4 INTC — Relaxation')
        img_page(pdf, 'pics_for_v4_300_INTC/17. Kyle Lambda per Insertion.png', 'V4 INTC — Kyle λ')

        # ═══════════ 3. V5r1 ═══════════
        text_page(pdf, [
            '# 3. V5r1: Low Participation Rate (η-derived child)',
            '',
            '## Grid Parameters',
            '',
            '  Stock:       AAPL, AMZN (in S5-4K training data)',
            '  i = 10 insertions (FIXED)',
            '  c = 100 cooling periods',
            '  mb = 36 messages between (≈ natural interarrival for AAPL)',
            '  Budget: 110 × 36 = 3,960 messages (within S5-4K 4,096 ctx)',
            '',
            '  Child derived from target participation rate η:',
            '    child = floor(η/(1-η) × mb × r × q_med)',
            '',
            '  AAPL (r=2.73%, q_med=40, depth=300):',
            f'    {"η":>5s}  {"child":>6s}  {"Q=10×c":>8s}  {"c/depth":>8s}  {"φ actual":>10s}',
            f'    {"5%":>5s}  {"2":>6s}  {"20":>8s}  {"0.7%":>8s}  {"4.8%":>10s}',
            f'    {"10%":>5s}  {"4":>6s}  {"40":>8s}  {"1.3%":>8s}  {"9.2%":>10s}',
            f'    {"20%":>5s}  {"9":>6s}  {"90":>8s}  {"3.0%":>8s}  {"18.6%":>10s}',
            '',
            '  AMZN (r=2.17%, q_med=25, depth=224): similar but smaller',
            '',
            '  PROBLEM: child = 2-9 shares vs depth = 224-300',
            '           child/depth < 3% → order NEVER penetrates first level',
            '           VWAP = best_ask (constant) → impact ≡ half_spread',
            '',
            '  Models: Historic, Heuristic, CST, S5-4K (4 only)',
            '  Samples: 2,048 per config',
            '',
            '## β Results — V5r1 AAPL (η=20%, Q=90)',
            '',
        ] + fmt_table(v5r1a, ['Model', 'beta_intercept', 'CI_lo', 'CI_hi', 'R2', 'N']) + [
            '',
            '  KEY: β ≈ 0 for ALL models. Impact unmeasurable.',
            '       mean_I ≈ 6e-5 (0.006% of price = microstructure noise)',
            '',
            '  BUT: Kyle λ STILL differentiates:',
            '       S5-4K λ(1)=0.149 → λ(10)=0.136 (FALLS — book restoration)',
            '       All baselines: λ RISES',
        ])

        # ═══════════ 4. V5r2 ═══════════
        text_page(pdf, [
            '# 4. V5r2: Trade-Size Grid (p25/p50/p75 child)',
            '',
            '## Grid Parameters',
            '',
            '  Same as V5r1 except child = trade size percentiles:',
            '',
            '  AAPL:',
            f'    {"child":>8s}  {"source":>8s}  {"Q":>8s}  {"c/depth":>8s}  {"φ":>8s}',
            f'    {"10":>8s}  {"p25":>8s}  {"100":>8s}  {"3.3%":>8s}  {"20%":>8s}',
            f'    {"40":>8s}  {"p50":>8s}  {"400":>8s}  {"13.3%":>8s}  {"50%":>8s}',
            f'    {"100":>8s}  {"p75":>8s}  {"1000":>8s}  {"33.3%":>8s}  {"72%":>8s}',
            '',
            '  AMZN:',
            f'    {"7":>8s}  {"p25":>8s}  {"70":>8s}  {"3.1%":>8s}  {"26%":>8s}',
            f'    {"25":>8s}  {"p50":>8s}  {"250":>8s}  {"11.2%":>8s}  {"56%":>8s}',
            f'    {"100":>8s}  {"p75":>8s}  {"1000":>8s}  {"44.6%":>8s}  {"84%":>8s}',
            '',
            '  Models: Historic, Heuristic, CST, S5-4K, S5-120M, S5-360M (6)',
            '  Samples: 2,048 per config',
            '',
            '## β Results — V5r2 AAPL (pooled across 3 child sizes)',
            '',
        ] + fmt_table(v5r2a, ['Model', 'beta_intercept', 'CI_lo', 'CI_hi', 'R2', 'N', 'relaxation']) + [
            '',
            '## V5r2 AMZN',
            '',
        ] + fmt_table(v5r2z, ['Model', 'beta_intercept', 'CI_lo', 'CI_hi', 'R2', 'N', 'relaxation']) + [
            '',
            '  KEY: β = 0.02-0.09 (measurable but far from 0.5)',
            '       S5 β LOWER than Heuristic (S5 compensates impact)',
            '       Kyle λ: S5-4K ↓, S5-120M ↓, S5-360M →, baselines ↑',
        ])

        img_page(pdf, 'pics_for_v5r2_mid_AAPL/2. Average Master Curve.png', 'V5r2 AAPL — Master Curve')
        img_page(pdf, 'pics_for_v5r2_mid_AAPL/4. Kyle Lambda per Insertion.png', 'V5r2 AAPL — Kyle λ')
        img_page(pdf, 'pics_for_v5r2_mid_AMZN/4. Kyle Lambda per Insertion.png', 'V5r2 AMZN — Kyle λ')

        # ═══════════ 5. β(φ) STRATIFICATION ═══════════
        text_page(pdf, [
            '# 5. β(φ) Per-Volume Stratification',
            '',
            '## Method',
            'For each (model, stock), split point cloud by volume level.',
            'Compute β separately at each φ. This gives β as function of φ.',
            '',
            '## Data: 126 unique (model, stock, vol) points',
            '',
        ])

        if beta_phi is not None:
            # S5-4K table
            s5 = beta_phi[beta_phi['model'] == 'S5-4K'].sort_values(['stock', 'phi'])
            lines = [
                '## S5-4K β(φ) across all stocks',
                '',
                f'  {"Stock":>5s}  {"vol":>6s}  {"φ%":>5s}  {"β_int":>8s}  {"R²":>8s}  {"λ(1)":>7s}  {"λ(k)":>7s}  {"λ":>2s}  {"N":>8s}',
            ]
            for _, r in s5.iterrows():
                phi_s = f'{r["phi"]:.0f}' if pd.notna(r['phi']) else '?'
                lines.append(
                    f'  {r["stock"]:>5s}  {r["vol"]:>6.0f}  {phi_s:>5s}  {r["beta_int"]:>8.4f}  {r["r2"]:>8.4f}  '
                    f'{r["lam_k1"]:>7.4f}  {r["lam_kmax"]:>7.4f}  {r["lam_trend"]:>2s}  {r["n"]:>8.0f}'
                )
            lines += ['', '  KEY: β increases with φ. λ trend crosses from ↓ to ↑ near φ≈90%.']
            text_page(pdf, lines)

            # All models comparison at similar φ
            for stock in ['GOOG', 'INTC', 'AAPL', 'AMZN']:
                ds = beta_phi[beta_phi['stock'] == stock].sort_values(['vol', 'model'])
                if ds.empty: continue
                lines = [f'## {stock} — all models per volume level', '']
                lines.append(f'  {"Model":>12s}  {"vol":>6s}  {"φ%":>5s}  {"β_int":>8s}  {"R²":>8s}  {"λ(1)":>7s}  {"λ(k)":>7s}  {"N":>7s}')
                for _, r in ds.iterrows():
                    phi_s = f'{r["phi"]:.0f}' if pd.notna(r['phi']) else '?'
                    lines.append(
                        f'  {r["model"]:>12s}  {r["vol"]:>6.0f}  {phi_s:>5s}  {r["beta_int"]:>8.4f}  {r["r2"]:>8.4f}  '
                        f'{r["lam_k1"]:>7.4f}  {r["lam_kmax"]:>7.4f}  {r["n"]:>7.0f}'
                    )
                text_page(pdf, lines)

        # β(φ) figures
        img_page(pdf, 'pics_for_beta_phi/beta_vs_phi_curve.png', 'β(φ) Curve — All Models')
        img_page(pdf, 'pics_for_beta_phi/beta_vs_phi_per_stock.png', 'β(φ) Per Stock')
        img_page(pdf, 'pics_for_beta_phi/kyle_lambda_vs_phi.png', 'Kyle λ vs φ')
        img_page(pdf, 'pics_for_beta_phi/r2_vs_phi.png', 'R² vs φ (Signal Strength)')

        # ═══════════ 6. STOCK STATS ═══════════
        tickers = ['GOOG', 'AAPL', 'NVDA', 'AMZN', 'META', 'TSLA', 'MSFT', 'AMD', 'INTC']
        lines = [
            '# 6. Stock Statistics (9 Tickers)',
            '',
            '  All from 20 trading days, January 2026, LOBSTER L2.',
            '',
            f'  {"Ticker":>6s}  {"r%":>6s}  {"q_p25":>5s}  {"q_med":>5s}  {"q_p75":>5s}  {"depth":>6s}  {"spr_t":>5s}  {"ADV_M":>6s}  {"train":>5s}',
        ]
        for t in tickers:
            p = f'lob_impact/lobster_stats_{t}.csv'
            if not Path(p).exists(): continue
            df = pd.read_csv(p)
            r = df['trade_fraction'].median() * 100
            q25 = df['trade_size_p25'].median()
            q50 = df['trade_size_p50'].median()
            q75 = df['trade_size_p75'].median()
            d = df['depth_best_p50'].median()
            s = df['spread_ticks_median'].median()
            adv = df['ADV'].median() / 1e6
            tr = 'Yes' if t != 'INTC' else 'No'
            lines.append(f'  {t:>6s}  {r:>6.2f}  {q25:>5.0f}  {q50:>5.0f}  {q75:>5.0f}  {d:>6.0f}  {s:>5.0f}  {adv:>6.1f}  {tr:>5s}')
        lines += ['', '  S5-4K trained on: GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, AMD']
        text_page(pdf, lines)

        # ═══════════ 7. SUMMARY ═══════════
        text_page(pdf, [
            '# 7. Summary: β Across All Versions',
            '',
            f'  {"Version":>8s}  {"Stock":>5s}  {"Volumes":>20s}  {"Estimator":>10s}  {"β (S5)":>10s}  {"φ":>8s}  {"R²":>6s}',
            f'  {"V1/V2":>8s}  {"GOOG":>5s}  {"75/300/485":>20s}  {"origin":>10s}  {"0.80-0.83":>10s}  {"~99%":>8s}  {"0.08":>6s}',
            f'  {"V1/V2":>8s}  {"GOOG":>5s}  {"75/300/485":>20s}  {"intercept":>10s}  {"0.08-0.23":>10s}  {"~99%":>8s}  {"0.08":>6s}',
            f'  {"V4":>8s}  {"GOOG":>5s}  {"105/165/325":>20s}  {"intercept":>10s}  {"0.33":>10s}  {"~99%":>8s}  {"0.25":>6s}',
            f'  {"V4":>8s}  {"INTC":>5s}  {"590/1110/3120":>20s}  {"intercept":>10s}  {"0.40":>10s}  {"~99%":>8s}  {"0.34":>6s}',
            f'  {"V5r1":>8s}  {"AAPL":>5s}  {"2/4/9 (η)":>20s}  {"intercept":>10s}  {"≈0":>10s}  {"5-20%":>8s}  {"0.00":>6s}',
            f'  {"V5r2":>8s}  {"AAPL":>5s}  {"10/40/100 (p25/50/75)":>20s}  {"intercept":>10s}  {"0.03-0.05":>10s}  {"20-72%":>8s}  {"0.01":>6s}',
            f'  {"V5r2":>8s}  {"AMZN":>5s}  {"7/25/100 (p25/50/75)":>20s}  {"intercept":>10s}  {"0.04-0.06":>10s}  {"26-84%":>8s}  {"0.01":>6s}',
            '',
            '## Key Conclusion',
            '',
            'β increases monotonically with φ but never reaches 0.5.',
            'Root causes: H=0.50 (no persistent order flow), single-model paradigm.',
            'Kyle λ is the universal differentiator at ALL φ levels.',
            'S5-4K outperforms all models on dynamics (context > model size).',
        ])

    print(f'\nSaved: {pdf_path} ({pdf_path.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
