#!/usr/bin/env python3
"""
Article v4: Comprehensive A4 report with V4 as primary + all experiments in appendix.

Generates a complete publication-ready PDF with:
- Main body: V4 GOOG results (best grid) with full theoretical justification
- Cross-stock: V4 INTC validation
- Appendix: V1/V2, V5, per-k, fixed-β, stock universe

Usage:
    python lob_impact/run_article_v4.py
"""
import pickle, numpy as np, pandas as pd, math, glob
from pathlib import Path
from collections import OrderedDict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings; warnings.filterwarnings('ignore')

OUT = Path('pics_for_article_v4')
OUT.mkdir(exist_ok=True)

# A4 dimensions in inches
A4_W, A4_H = 8.27, 11.69

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def text_page(pdf, lines, fontsize=10, title=None):
    fig = plt.figure(figsize=(A4_W, A4_H))
    y = 0.94
    if title:
        fig.text(0.5, 0.97, title, ha='center', fontsize=14, fontweight='bold')
        y = 0.92
    for line in lines:
        if line.startswith('####'):
            fig.text(0.05, y, line[4:].strip(), fontsize=9, fontweight='bold', style='italic'); y -= 0.020
        elif line.startswith('###'):
            fig.text(0.05, y, line[3:].strip(), fontsize=10, fontweight='bold', style='italic'); y -= 0.022
        elif line.startswith('##'):
            fig.text(0.05, y, line[2:].strip(), fontsize=12, fontweight='bold'); y -= 0.028
        elif line.startswith('#'):
            fig.text(0.05, y, line[1:].strip(), fontsize=13, fontweight='bold'); y -= 0.032
        elif line == '':
            y -= 0.008
        elif line.startswith('  '):
            fig.text(0.06, y, line, fontsize=8, fontfamily='monospace'); y -= 0.015
        else:
            fig.text(0.05, y, line, fontsize=fontsize); y -= 0.019
        if y < 0.03:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(A4_W, A4_H)); y = 0.96
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, path, title=None):
    p = Path(path)
    if not p.exists():
        print(f'  SKIP: {path}')
        return
    img = plt.imread(str(p))
    fig, ax = plt.subplots(figsize=(A4_W, A4_H * 0.85))
    ax.imshow(img); ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else [0, 0, 1, 1])
    pdf.savefig(fig); plt.close(fig)


def load_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def fmt_beta_table(df, cols_extra=None):
    if df is None: return ['  (no data)']
    lines = []
    hdr = f'  {"Model":>12s}  {"β_int":>7s}  {"β_orig":>7s}  {"CI_lo":>7s}  {"CI_hi":>7s}  {"R²":>6s}  {"N":>7s}'
    if cols_extra:
        for c in cols_extra:
            hdr += f'  {c[:6]:>7s}'
    lines.append(hdr)
    for _, row in df.iterrows():
        m = str(row.get('Model', ''))
        if m == 'ZeroInsertions': continue
        s = f'  {m:>12s}  {row.get("beta_intercept",0):>7.3f}  {row.get("beta_origin",0):>7.3f}  {row.get("CI_lo",0):>7.3f}  {row.get("CI_hi",0):>7.3f}  {row.get("R2",0):>6.3f}  {int(row.get("N",0)):>7d}'
        if cols_extra:
            for c in cols_extra:
                v = row.get(c, np.nan)
                if c == 'no_arb_score':
                    s += f'  {int(v):>7d}' if pd.notna(v) else f'  {"":>7s}'
                else:
                    s += f'  {float(v):>7.3f}' if pd.notna(v) else f'  {"---":>7s}'
        lines.append(s)
    return lines


def load_lobster(stock):
    p = f'lob_impact/lobster_stats_{stock}.csv'
    if not Path(p).exists(): return {}
    df = pd.read_csv(p)
    return {k: df[k].median() for k in df.columns if k != 'day'}


def main():
    pdf_path = OUT / 'article_v4.pdf'
    print(f'Generating: {pdf_path}')

    v4g = load_csv('pics_for_v4_300_GOOG/summary_statistics.csv')
    v4i = load_csv('pics_for_v4_300_INTC/summary_statistics.csv')
    v3g = load_csv('pics_for_300_GOOG/summary_statistics.csv')

    with PdfPages(str(pdf_path)) as pdf:

        # ══════════════════════════════════════════════════════
        # TITLE
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '', '', '', '',
            '# Emergent Macroscopic Market Impact Analysis',
            '# on AI-Generated Limit Order Book Data',
            '', '',
            'George Nigmatulin, Sascha Frey, Valentin Mohl, Kang Li,',
            'Bidipta Sarkar, Mihai Cucuringu, Anisoara Calinescu,',
            'Jakob Foerster, Stefan Zohren',
            '',
            'University of Oxford',
            '', '',
            '## Abstract',
            '',
            'Generative AI models can now synthesise realistic limit order book (LOB)',
            'message streams, but no prior work has tested whether they reproduce the',
            'square-root law of market impact. We present a counterfactual injection',
            'methodology with per-stock volume calibration and apply it to nine model',
            'classes across two NASDAQ stocks. Using an unbiased intercept estimator,',
            'we find β ∈ [0.21, 0.34] for GOOG and β ∈ [0.14, 0.43] for INTC.',
            'The models diverge sharply in dynamic properties: S5-4K achieves the',
            'closest relaxation to the theoretical 2/3 in both stocks. Kyle lambda',
            'trajectory emerges as the strongest model differentiator, revealing',
            'that S5 models uniquely restore book depth after aggressive injection.',
        ], title='Article v4 — Comprehensive Report')

        # ══════════════════════════════════════════════════════
        # §1 INTRODUCTION
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# §1 Introduction',
            '',
            'The square-root law of market impact, I(Q) ∝ σ·(Q/V)^β with β ≈ 0.5,',
            'is one of the most robust stylised facts in market microstructure',
            '(Bouchaud 2018, Tóth 2011, Sato & Kanazawa 2025).',
            '',
            'Despite its importance, NO prior work has tested whether generative',
            'LOB models reproduce this law at the meta-order level:',
            '',
            '  Paper              Impact validated?   β reported?',
            '  MarS (Li 2024)     Fits α, not β       NOT explicitly',
            '  TRADES (2025)      "Potential app"      NO',
            '  LOBERT (2025)      NO                   NO',
            '  LOB-Bench          Event-level only     NOT meta-order',
            '  CGAN, CST, ABIDES  NO                   NO',
            '',
            'We close this gap with four contributions:',
            '',
            '1. Counterfactual injection protocol with per-stock volume calibration',
            '2. Nine-model comparison on two stocks (GOOG, INTC)',
            '3. Unbiased intercept estimator (origin biased by +0.51)',
            '4. Dynamic evaluation hierarchy: β → relaxation → Kyle λ',
        ])

        # ══════════════════════════════════════════════════════
        # §2 BACKGROUND (brief)
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# §2 Background',
            '',
            '## 2.1 Square-Root Law',
            'I(Q) = Y · σ · (Q/V)^β, β ≈ 0.5',
            'Observed across equities, futures, FX over decades (Torre 1997,',
            'Almgren 2005, Lillo 2003, Tóth 2011, Sato & Kanazawa 2025).',
            '',
            '## 2.2 Impact Dynamics',
            'Relaxation ratio r = I_final/I_peak ≈ 2/3 (Bouchaud 2018).',
            'Permanent impact linear in volume (Huberman & Stanzl 2004).',
            'Temporary component decays as power law γ ≈ 0.5-0.8 (Brokmann 2015).',
            '',
            '## 2.3 Participation Rate',
            'Bucci et al. (2019): β transitions from ~1 (linear, φ<1%) to',
            '~0.5 (square-root, φ>2%). Institutional metaorders: φ = 5-20%.',
            '',
            '## 2.4 Order Flow Persistence',
            'Hurst exponent H ≈ 0.7 in real markets (Lillo 2004).',
            'This persistence drives the square-root law via the propagator',
            'framework (Bouchaud 2004, Eisler 2012).',
            '',
            '## 2.5 Evaluation Gap',
            'LOB-Bench measures event-level price response, not meta-order impact.',
            'A model could pass all LOB-Bench tests while failing the square-root law.',
        ])

        # ══════════════════════════════════════════════════════
        # §3 METHODOLOGY
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# §3 Methodology',
            '',
            '## 3.1 Counterfactual Injection Protocol',
            'Three phases: conditioning (500 msgs) → insertion (i aggressive orders,',
            'mb messages between) → cooling (c = 10×i blocks).',
            'Antisymmetrisation: I_combined = (I_buy - I_sell) / 2.',
            'Constraint: 11·i·mb ≤ 500 (within training context).',
            '',
            '## 3.2 Nine Models',
            '',
            f'  {"Model":>12s}  {"Params":>8s}  {"Ctx":>6s}  {"Training":>25s}',
            f'  {"LobS5":>12s}  {"75M":>8s}  {"500":>6s}  {"GOOG, 2025":>25s}',
            f'  {"LobS5-v2":>12s}  {"75M":>8s}  {"500":>6s}  {"GOOG, 2022":>25s}',
            f'  {"S5-120M":>12s}  {"120M":>8s}  {"500":>6s}  {"8 stocks, 2022-2025":>25s}',
            f'  {"S5-360M":>12s}  {"360M":>8s}  {"500":>6s}  {"8 stocks, 2022-2025":>25s}',
            f'  {"S5-4K":>12s}  {"55M":>8s}  {"4096":>6s}  {"8 stocks, 2022-2025":>25s}',
            f'  {"CGAN":>12s}  {"~5M":>8s}  {"snap":>6s}  {"GOOG, 3 days":>25s}',
            f'  {"CST":>12s}  {"param":>8s}  {"∞":>6s}  {"calibrated":>25s}',
            f'  {"Historic":>12s}  {"—":>8s}  {"∞":>6s}  {"replay":>25s}',
            f'  {"Heuristic":>12s}  {"—":>8s}  {"∞":>6s}  {"replay + shift":>25s}',
            '',
            '## 3.3 Per-Stock Volume Calibration',
            'From 122,880 depth-at-best observations:',
            '  GOOG: {105, 165, 325} shares (p50/p75/p95)',
            '  INTC: {590, 1110, 3120} shares (p50/p75/p95)',
        ])

        text_page(pdf, [
            '## 3.4 Impact Measurement',
            'Implementation shortfall: I = |VWAP_fills - mid_arrival| / mid_arrival',
            'One metaorder = one data point (not per-insertion).',
            'Q = total injected volume, V = daily traded volume.',
            '',
            '## 3.5 Intercept Estimator',
            'Through-origin: log(I/σ) = β·log(Q/V) → biased by +0.51',
            'Intercept: log(I) = α + β·log(Q/V) → unbiased',
            '',
            'Bias = (α - E[log σ]) · E[x] / E[x²]',
            'With α ≈ -9, E[log σ] ≈ -3.5: bias ≈ +0.51',
            '',
            'Verified on 80,000 synthetic points:',
            '  Intercept: recovered β within ±0.003 (20 scenarios)',
            '  Origin: biased by +0.13 to +0.66',
            '',
            '## 3.6 Dynamic Metrics',
            'Relaxation ratio: r = I_final / I_peak (target: 2/3)',
            'Stability: 3-method vote (trailing slope, window mean, exp fit)',
            'Kyle λ(k): per-insertion book response',
            'No-arb scorecard: 5 tests (concavity, perm linearity, decay, relax, Gatheral)',
        ])

        # Protocol figure
        image_page(pdf, 'overleaf_article_report/overleaf/overleaf_project_article_v3/Figures/scheme-6.png',
                   title='Figure 1: Counterfactual Injection Protocol')

        # ══════════════════════════════════════════════════════
        # §4 EXPERIMENTAL SETUP
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# §4 Experimental Setup',
            '',
            'Data: LOBSTER L2, GOOG + INTC, 20 days January 2026 (out-of-sample).',
            'Grid: 10 valid (i, mb) pairs × 3 volumes × 2 directions = 60 configs.',
            'Samples: 2,048 per config → ~82,000 metaorders per model (GOOG).',
            '',
            '## Grid Parameters',
            '',
            f'  {"mb":>4s}  {"i":>3s}  {"c":>4s}  {"total":>6s}  {"ctx%":>5s}',
            f'  {"5":>4s}  {"3":>3s}  {"30":>4s}  {"165":>6s}  {"33%":>5s}',
            f'  {"5":>4s}  {"5":>3s}  {"50":>4s}  {"275":>6s}  {"55%":>5s}',
            f'  {"5":>4s}  {"9":>3s}  {"90":>4s}  {"495":>6s}  {"99%":>5s}',
            f'  {"10":>4s}  {"2":>3s}  {"20":>4s}  {"220":>6s}  {"44%":>5s}',
            f'  {"10":>4s}  {"3":>3s}  {"30":>4s}  {"330":>6s}  {"66%":>5s}',
            f'  {"10":>4s}  {"4":>3s}  {"40":>4s}  {"440":>6s}  {"88%":>5s}',
            f'  {"15":>4s}  {"2":>3s}  {"20":>4s}  {"330":>6s}  {"66%":>5s}',
            f'  {"15":>4s}  {"3":>3s}  {"30":>4s}  {"495":>6s}  {"99%":>5s}',
            f'  {"20":>4s}  {"1":>3s}  {"10":>4s}  {"220":>6s}  {"44%":>5s}',
            f'  {"20":>4s}  {"2":>3s}  {"20":>4s}  {"440":>6s}  {"88%":>5s}',
        ])

        # ══════════════════════════════════════════════════════
        # §5 RESULTS — GOOG
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# §5 Results',
            '',
            '## 5.1 GOOG — Square-Root Law (9 models)',
            '',
        ] + fmt_beta_table(v4g, ['relaxation', 'Hurst', 'no_arb_score']) + [
            '',
            'S5 models cluster at β ≈ 0.33. CGAN lowest (0.205).',
            'Origin estimator gives β ≈ 0.84 for ALL models (biased).',
        ])

        # V4 GOOG figures — main body
        for fig in [
            '2. Average Master Curve.png',
            '3. Beta Regression Lines.png',
            '4. Bootstrap Beta Distributions.png',
            '15. Beta Estimator Comparison.png',
            '12. Per-Day Beta.png',
            '5. Relaxation Ratio.png',
            '6. Fraction Stable.png',
            '17. Kyle Lambda per Insertion.png',
            '14. No-Arb Scatter.png',
            '13. Perm Temp Decomposition.png',
            '16. Per-Insertion Midprice Response.png',
            '1. Master Curves.png',
            '0. Null Baseline Drift.png',
            '7. Hurst Exponent.png',
            '8. Propagator G(l).png',
            '9. Spread Dynamics.png',
        ]:
            image_page(pdf, f'pics_for_v4_300_GOOG/{fig}', title=f'GOOG — {fig[3:-4]}')

        # ══════════════════════════════════════════════════════
        # §5 cont — INTC
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '## 5.2 INTC — Cross-Stock Validation',
            '',
        ] + fmt_beta_table(v4i, ['relaxation', 'Hurst', 'no_arb_score']) + [
            '',
            'S5 β ≈ 0.40 (closer to 0.5 than GOOG 0.33).',
            'S5-4K relaxation = 0.763 (closest to 2/3).',
            'INTC β_perm ≈ 0.92-0.99 (near 1.0 target).',
        ])

        for fig in ['2. Average Master Curve.png', '3. Beta Regression Lines.png',
                     '4. Bootstrap Beta Distributions.png', '5. Relaxation Ratio.png',
                     '17. Kyle Lambda per Insertion.png', '14. No-Arb Scatter.png',
                     '12. Per-Day Beta.png', '1. Master Curves.png']:
            image_page(pdf, f'pics_for_v4_300_INTC/{fig}', title=f'INTC — {fig[3:-4]}')

        # ══════════════════════════════════════════════════════
        # §6 DISCUSSION
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# §6 Discussion',
            '',
            '## Static vs Dynamic',
            'β differentiates models only with intercept estimator.',
            'Dynamic properties (relaxation, Kyle λ) are the true differentiators.',
            '',
            '## Context Length > Model Size',
            'S5-4K (55M, 4K ctx): best relaxation in BOTH stocks.',
            'S5-360M (360M, 500 ctx): weaker dynamics despite 6.5× more parameters.',
            '',
            '## Kyle Lambda: Universal Differentiator',
            'At k=1: all models identical (same conditioning book).',
            'At k>5: S5 models diverge — realistic depletion + recovery.',
            'CST/CGAN: flat (over-replenishment). Historic: rises (no recovery).',
            'Works at ANY participation rate (confirmed in V5 experiments).',
            '',
            '## Why β < 0.5',
            '1. Single-model simulation (no multi-agent equilibrium)',
            '2. Generated order flow H ≈ 0.50 (not 0.70 as in real markets)',
            '3. Extreme participation rate φ ≈ 99% (stress test regime)',
            '4. Q/V ≈ 10^-5 to 10^-3 (vs institutional 10^-2 to 10^-1)',
            '',
            '## Comparison with MarS',
            'MarS (Li 2024) uses TWAP agent, fixes β=0.5, fits α.',
            'Our approach: estimate β directly, test 10 models, 2 stocks.',
            'We are the FIRST to report β values for generative LOB models.',
            '',
            '## Evaluation Hierarchy',
            '1. Distributional fidelity (LOB-Bench)',
            '2. Static impact scaling (β) — now informative with intercept',
            '3. Dynamic impact fidelity (relaxation ≈ 2/3) — strongest test',
        ])

        # ══════════════════════════════════════════════════════
        # §7 CONCLUSION
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# §7 Conclusion',
            '',
            'We present the first systematic evaluation of the square-root law',
            'of market impact on AI-generated limit order book data.',
            '',
            'Key findings:',
            '1. S5 models cluster at β ≈ 0.33 (GOOG) / 0.40 (INTC)',
            '2. S5-4K achieves best relaxation (0.76-0.87, closest to 2/3)',
            '3. Context length matters more than model size for dynamics',
            '4. Kyle λ is the strongest and most universal model differentiator',
            '5. Origin estimator biased by +0.51 — intercept essential',
            '6. Hurst H ≈ 0.50 (negative finding: no long-range memory)',
            '',
            'The scaling exponent alone is not discriminative — dynamic properties',
            '(relaxation, book restoration via Kyle λ) are the true differentiators.',
        ])

        # ══════════════════════════════════════════════════════
        # APPENDIX A: INTC additional figures
        # ══════════════════════════════════════════════════════
        text_page(pdf, ['# Appendix A: INTC Additional Figures'])
        for fig in ['13. Perm Temp Decomposition.png', '7. Hurst Exponent.png',
                     '8. Propagator G(l).png', '9. Spread Dynamics.png',
                     '6. Fraction Stable.png', '15. Beta Estimator Comparison.png']:
            image_page(pdf, f'pics_for_v4_300_INTC/{fig}', title=f'INTC — {fig[3:-4]}')

        # ══════════════════════════════════════════════════════
        # APPENDIX B: V1/V2 (uncalibrated)
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Appendix B: V1/V2 Experiments (Uncalibrated Volumes)',
            '',
            'Initial experiments used fixed volumes 75/300/485 shares.',
            'Through-origin estimator gave β ≈ 0.80 for ALL models (no differentiation).',
            '',
            '## V1/V2 GOOG Results',
            '',
        ] + fmt_beta_table(v3g, ['relaxation']))

        for fig in ['2. Average Master Curve.png', '3. Beta Regression Lines.png',
                     '5. Relaxation Ratio.png']:
            image_page(pdf, f'pics_for_300_GOOG/{fig}', title=f'V1/V2 GOOG — {fig[3:-4]}')

        # ══════════════════════════════════════════════════════
        # APPENDIX C: Estimator Bias
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Appendix C: Estimator Bias Derivation',
            '',
            'Through-origin: β̂_origin = Σ(xi·yi) / Σ(xi²)',
            'When true model: yi = α + β·xi + εi:',
            '  E[β̂_origin] = β + α · x̄ / (s²_x + x̄²)',
            '',
            'For our data: α ≈ -9, x̄ ≈ -6.8, s²_x ≈ 3.1',
            '  bias ≈ (-9) × (-6.8) / (3.1 + 46.2) ≈ +1.24',
            '  (empirically: +0.51 due to σ normalization)',
            '',
            'Verified on 80,000 synthetic data points with β_true ∈ {0.3,...,0.7}:',
            '  Intercept: recovered within ±0.003',
            '  Origin: biased by +0.13 to +0.66',
        ])
        for fig in ['diag_1_synthetic.png']:
            image_page(pdf, f'pics_for_investigation/{fig}', title='Synthetic Calibration')

        # ══════════════════════════════════════════════════════
        # APPENDIX D: V5 Participation Rate
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Appendix D: V5 Participation Rate Experiments',
            '',
            '## D.1 V5r1: Low φ (5-20%), AAPL + AMZN',
            'child = floor(η/(1-η) × mb × r × q_med)',
            'Result: β ≈ 0, impact unmeasurable. child/depth < 3%.',
            '',
            '## D.2 V5r2: Trade-Size Child (φ = 20-72%)',
            'child = p25/p50/p75 of actual trade sizes',
            'AAPL: 10/40/100 shares, AMZN: 7/25/100 shares',
            '',
            '  β_vwap ≈ 0.03-0.15 (measurable but low)',
            '  β_mid ≈ 0.03-0.09 (midprice cumulative impact)',
            '',
            '## D.3 Key Finding: VWAP Fails at Low φ',
            'When child < depth: exec price = best ask (constant).',
            'VWAP impact = half_spread (no Q dependence) → β ≈ 0.',
            '',
            '## D.4 Kyle λ Still Differentiates',
            'S5-4K λ FALLS (book restoration) at ALL φ levels.',
            'Baselines: λ RISES (depletion without recovery).',
        ])

        for fig in ['4. Kyle Lambda per Insertion.png', '2. Average Master Curve.png']:
            image_page(pdf, f'pics_for_v5r2_mid_AAPL/{fig}', title=f'V5r2 AAPL Midprice — {fig[3:-4]}')
        for fig in ['4. Kyle Lambda per Insertion.png']:
            image_page(pdf, f'pics_for_v5r2_mid_AMZN/{fig}', title=f'V5r2 AMZN Midprice — {fig[3:-4]}')

        # ══════════════════════════════════════════════════════
        # APPENDIX E: Per-k Beta
        # ══════════════════════════════════════════════════════
        text_page(pdf, [
            '# Appendix E: Per-k Beta Analysis',
            '',
            'β computed at each insertion k=1..9 separately.',
            'β INCREASES with k (more Q variation at higher k).',
            '',
            'β(k=1) ≈ 0.01-0.07 (single insertion, 3 Q-points from volumes)',
            'β(k=9) ≈ 0.08-0.24 (full metaorder)',
            'β(all k pooled) ≈ 0.21-0.34 (maximum Q variation)',
            '',
            'β(k=1) is NOT closer to 0.5 — the low value reflects',
            'insufficient Q variation at a single k (only 3 volume levels).',
        ])

        # ══════════════════════════════════════════════════════
        # APPENDIX F: Stock Universe
        # ══════════════════════════════════════════════════════
        tickers = ['GOOG', 'AAPL', 'NVDA', 'AMZN', 'META', 'TSLA', 'MSFT', 'AMD', 'INTC']
        lines = [
            '# Appendix F: Stock Universe (9 Tickers)',
            '',
            'Lobster stats from 20 trading days, January 2026.',
            '',
            f'  {"Ticker":>6s}  {"r%":>6s}  {"q_med":>5s}  {"depth":>6s}  {"spr":>4s}  {"ADV_M":>6s}',
        ]
        for t in tickers:
            s = load_lobster(t)
            if not s: continue
            r = s.get('trade_fraction', 0) * 100
            q = s.get('trade_size_p50', 0)
            d = s.get('depth_best_p50', 0)
            sp = s.get('spread_ticks_median', 0)
            adv = s.get('ADV', 0) / 1e6
            lines.append(f'  {t:>6s}  {r:>6.2f}  {q:>5.0f}  {d:>6.0f}  {sp:>4.0f}  {adv:>6.1f}')
        text_page(pdf, lines)

        # ══════════════════════════════════════════════════════
        # APPENDIX G: Stratification
        # ══════════════════════════════════════════════════════
        text_page(pdf, ['# Appendix G: Stratification & Diagnostics'])
        for fig in ['R04_beta_vs_k.png', 'R05_beta_vs_mb.png', 'R06_beta_vs_depth.png',
                     'R07_beta_per_day_boxplot.png', 'R08_normalizations.png',
                     'R09_impact_definitions.png', 'R10_kyle_lambda.png',
                     'R12_beta_incremental.png', 'R13_cross_stock.png',
                     'R14_cross_sectional_beta.png', 'R15_penetration_split.png']:
            p = f'pics_for_beta_report/{fig}'
            if Path(p).exists():
                image_page(pdf, p, title=f'Diagnostic — {fig[4:-4]}')

    size_mb = pdf_path.stat().st_size / 1e6
    print(f'\nSaved: {pdf_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
