#!/usr/bin/env python3
"""Diagnostic 1: Synthetic Calibration Test.

Generate fake data with known beta, run all estimators, verify recovery.
Proves whether the estimator or the data is the problem.

True data-generating process:
    log(I) = alpha + beta_true * log(Q/V) + epsilon

We generate N~80000 points grouped into ~2000 samples (~40 points each),
with per-sample sigma (Parkinson volatility) and daily_vol (V).
Parameters calibrated from real v4 GOOG data.

Usage:
    python lob_impact/diag_1_synthetic_calibration.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# Output directory
# ═══════════════════════════════════════════════════════════════════════
OUT_DIR = Path('pics_for_investigation')

# ═══════════════════════════════════════════════════════════════════════
# Calibration from real v4 GOOG data
# ═══════════════════════════════════════════════════════════════════════
# From summary_statistics.csv:
#   beta_intercept ≈ 0.33, alpha ≈ -9, N ≈ 82000
# From real point cloud distributions:
#   x = log(Q/V) ~ N(-11, 2), sigma ≈ 0.03, noise σ_eps ≈ 0.8
N_POINTS = 80000
N_SAMPLES = 2000
POINTS_PER_SAMPLE = N_POINTS // N_SAMPLES  # 40

# x = log(Q/V) distribution
X_MEAN = -11.0
X_STD = 2.0

# Parkinson volatility per sample
SIGMA_MEAN = 0.03
SIGMA_STD = 0.005

# Daily volume per sample
V_MEAN = 1e6
V_STD = 3e5

# Noise
EPS_STD = 0.8


# ═══════════════════════════════════════════════════════════════════════
# Three estimators (ported exactly from run_300_analyze_one.py)
# ═══════════════════════════════════════════════════════════════════════
def compute_betas(x, y_adj, y_raw):
    """Run all three beta estimators.

    Args:
        x:     log(Q/V) array
        y_adj: log(I/sigma) array (for origin + ratio estimators)
        y_raw: log(I) array       (for intercept estimator)

    Returns:
        dict with beta_origin, beta_intercept, beta_ratio, alpha
    """
    ok = np.isfinite(x) & np.isfinite(y_adj) & np.isfinite(y_raw) & (x != 0)
    xv, yv_adj, yv_raw = x[ok], y_adj[ok], y_raw[ok]
    if len(xv) < 2:
        return dict(beta_origin=np.nan, beta_intercept=np.nan,
                    beta_ratio=np.nan, alpha=np.nan)

    # 1. OLS through origin: log(I/sigma) = beta * log(Q/V)
    beta_origin = float(np.dot(xv, yv_adj) / np.dot(xv, xv))

    # 2. OLS with free intercept: log(I) = alpha + beta * log(Q/V)
    coeffs = np.polyfit(xv, yv_raw, 1)
    beta_intercept = float(coeffs[0])
    alpha = float(coeffs[1])

    # 3. Ratio estimator: beta = mean(log(I/sigma) / log(Q/V))
    beta_ratio = float(np.mean(yv_adj / xv))

    return dict(beta_origin=beta_origin, beta_intercept=beta_intercept,
                beta_ratio=beta_ratio, alpha=alpha)


# ═══════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════
def generate_synthetic_data(beta_true, alpha_true, rng, sigma_eps=EPS_STD):
    """Generate synthetic point cloud with known beta.

    Model: log(I) = alpha + beta_true * log(Q/V) + epsilon
    where epsilon ~ N(0, sigma_eps^2).

    Returns DataFrame with columns matching real pipeline:
        Q, I, daily_vol, daily_sigma, sample_id
    """
    n_total = N_SAMPLES * POINTS_PER_SAMPLE

    # Per-sample parameters, repeated for each point in the sample
    daily_vol_per_sample = np.maximum(rng.normal(V_MEAN, V_STD, N_SAMPLES), 1e3)
    daily_sigma_per_sample = np.maximum(rng.normal(SIGMA_MEAN, SIGMA_STD, N_SAMPLES), 1e-4)
    daily_vol = np.repeat(daily_vol_per_sample, POINTS_PER_SAMPLE)
    daily_sigma = np.repeat(daily_sigma_per_sample, POINTS_PER_SAMPLE)
    sample_ids = np.repeat(np.arange(N_SAMPLES), POINTS_PER_SAMPLE)

    # x = log(Q/V), so Q = V * exp(x)
    x = rng.normal(X_MEAN, X_STD, n_total)
    Q = daily_vol * np.exp(x)

    # log(I) = alpha + beta_true * x + eps
    eps = rng.normal(0, sigma_eps, n_total)
    log_I = alpha_true + beta_true * x + eps
    I_val = np.exp(log_I)

    # Filter invalid points
    ok = (Q > 0) & np.isfinite(I_val) & (I_val > 0)
    sample_id_strs = np.array([f'sample_{s}' for s in sample_ids[ok]])

    return pd.DataFrame(dict(
        Q=Q[ok], I=I_val[ok],
        daily_vol=daily_vol[ok],
        daily_sigma=daily_sigma[ok],
        sample_id=sample_id_strs,
    ))


def run_estimators_on_df(df):
    """Run the three estimators on a synthetic DataFrame,
    matching exactly how run_300_analyze_one.py processes real data."""
    x = np.log(df['Q'].values / df['daily_vol'].values)
    y_adj = np.log(df['I'].values / df['daily_sigma'].values)
    y_raw = np.log(df['I'].values)
    return compute_betas(x, y_adj, y_raw)


# ═══════════════════════════════════════════════════════════════════════
# Parameter sweep
# ═══════════════════════════════════════════════════════════════════════
def run_sweep():
    """Sweep beta_true x alpha, run all estimators, record bias."""
    beta_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    alpha_values = [-5.0, -7.0, -9.0, -11.0]

    results = []
    rng = np.random.default_rng(42)

    for beta_true in beta_values:
        for alpha_true in alpha_values:
            df = generate_synthetic_data(beta_true, alpha_true, rng)
            est = run_estimators_on_df(df)

            bias_origin = est['beta_origin'] - beta_true
            bias_intercept = est['beta_intercept'] - beta_true
            bias_ratio = est['beta_ratio'] - beta_true

            results.append(dict(
                beta_true=beta_true,
                alpha=alpha_true,
                N=len(df),
                beta_origin=est['beta_origin'],
                beta_intercept=est['beta_intercept'],
                beta_ratio=est['beta_ratio'],
                alpha_est=est['alpha'],
                bias_origin=bias_origin,
                bias_intercept=bias_intercept,
                bias_ratio=bias_ratio,
            ))

            print(f'  beta_true={beta_true:.1f}  alpha={alpha_true:6.1f}  '
                  f'origin={est["beta_origin"]:.4f} (bias={bias_origin:+.4f})  '
                  f'intercept={est["beta_intercept"]:.4f} (bias={bias_intercept:+.4f})  '
                  f'ratio={est["beta_ratio"]:.4f} (bias={bias_ratio:+.4f})')

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic scatter plot
# ═══════════════════════════════════════════════════════════════════════
def make_scatter_plot(out_dir):
    """Generate scatter of synthetic data (beta_true=0.5, alpha=-9)
    with all three regression lines overlaid."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available, skipping scatter plot')
        return

    rng = np.random.default_rng(123)
    beta_true = 0.5
    alpha_true = -9.0
    df = generate_synthetic_data(beta_true, alpha_true, rng)
    est = run_estimators_on_df(df)

    x = np.log(df['Q'].values / df['daily_vol'].values)
    y_adj = np.log(df['I'].values / df['daily_sigma'].values)
    y_raw = np.log(df['I'].values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: origin estimator space (y_adj vs x)
    ax = axes[0]
    ax.scatter(x, y_adj, s=1, alpha=0.05, color='grey', rasterized=True)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, beta_true * x_line, 'g-', lw=2,
            label=f'True: slope={beta_true:.2f}')
    ax.plot(x_line, est['beta_origin'] * x_line, 'r--', lw=2,
            label=f'Origin: slope={est["beta_origin"]:.4f}')
    ax.plot(x_line, est['beta_ratio'] * x_line, 'b:', lw=2,
            label=f'Ratio: slope={est["beta_ratio"]:.4f}')
    ax.set_xlabel('log(Q/V)')
    ax.set_ylabel('log(I/sigma)')
    ax.set_title('Origin estimator space\n'
                 'log(I/sigma) = beta * log(Q/V)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: intercept estimator space (y_raw vs x)
    ax = axes[1]
    ax.scatter(x, y_raw, s=1, alpha=0.05, color='grey', rasterized=True)
    ax.plot(x_line, alpha_true + beta_true * x_line, 'g-', lw=2,
            label=f'True: beta={beta_true:.2f}, alpha={alpha_true:.1f}')
    ax.plot(x_line, est['alpha'] + est['beta_intercept'] * x_line, 'r--', lw=2,
            label=f'Intercept: beta={est["beta_intercept"]:.4f}, '
                  f'alpha={est["alpha"]:.2f}')
    ax.set_xlabel('log(Q/V)')
    ax.set_ylabel('log(I)')
    ax.set_title('Intercept estimator space\n'
                 'log(I) = alpha + beta * log(Q/V)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Synthetic Calibration: beta_true={beta_true}, '
                 f'alpha={alpha_true}, N={len(df)}, eps_std={EPS_STD}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    png_path = out_dir / 'diag_1_synthetic.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved scatter plot: {png_path}')


# ═══════════════════════════════════════════════════════════════════════
# Analytical bias explanation
# ═══════════════════════════════════════════════════════════════════════
def print_analytical_explanation(results_df):
    """Print explanation of why each estimator is biased or unbiased."""
    lines = [
        '',
        '=' * 70,
        'ANALYTICAL EXPLANATION',
        '=' * 70,
        '',
        'True DGP: log(I) = alpha + beta_true * log(Q/V) + eps',
        '',
        'INTERCEPT estimator: log(I) = a + b * log(Q/V)',
        '  -> Correctly matches the DGP. Should recover beta_true.',
        '',
        'ORIGIN estimator: log(I/sigma) = b * log(Q/V)  [no intercept]',
        '  -> Transforms y to log(I/sigma) = log(I) - log(sigma)',
        '     = alpha + beta_true*x + eps - log(sigma)',
        '  -> Forced through origin, so the omitted intercept',
        '     (alpha - log(sigma)) is absorbed into the slope.',
        '  -> Bias = (alpha - E[log(sigma)]) * E[x] / E[x^2]',
        '     When alpha < 0 and x < 0: bias is POSITIVE.',
        '',
        'RATIO estimator: mean(log(I/sigma) / log(Q/V))',
        '  -> E[y_adj/x] != E[y_adj]/E[x] (Jensen inequality)',
        '  -> Especially biased when x has high variance or is near zero.',
        '',
    ]

    # Compute theoretical bias for origin estimator
    # y_adj = (alpha - log(sigma)) + beta_true * x + eps
    # Forcing through origin: beta_hat = dot(x, y_adj) / dot(x, x)
    #   = beta_true + (alpha - E[log(sigma)]) * sum(x) / sum(x^2)
    # For x ~ N(mu, s^2): E[sum(x)] = N*mu, E[sum(x^2)] = N*(mu^2+s^2)
    # So bias ≈ (alpha - log(sigma_mean)) * mu / (mu^2 + s^2)
    log_sigma = np.log(SIGMA_MEAN)
    for _, row in results_df.iterrows():
        c = row['alpha'] - log_sigma
        theoretical_bias = c * X_MEAN / (X_MEAN**2 + X_STD**2)
        lines.append(
            f'  alpha={row["alpha"]:6.1f}: '
            f'theoretical origin bias = {theoretical_bias:+.4f}, '
            f'actual = {row["bias_origin"]:+.4f}'
        )

    lines.append('')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('DIAGNOSTIC 1: Synthetic Calibration Test')
    print('=' * 70)
    print(f'N_points={N_POINTS}, N_samples={N_SAMPLES}, '
          f'x~N({X_MEAN},{X_STD}), eps_std={EPS_STD}')
    print(f'sigma~N({SIGMA_MEAN},{SIGMA_STD}), V~N({V_MEAN:.0f},{V_STD:.0f})')
    print()

    # --- Run parameter sweep ---
    print('Parameter sweep: beta_true x alpha')
    print('-' * 70)
    results_df = run_sweep()
    print()

    # --- Summary table ---
    txt_path = OUT_DIR / 'diag_1_synthetic.txt'
    lines = []
    lines.append('DIAGNOSTIC 1: Synthetic Calibration Test')
    lines.append(f'N_points={N_POINTS}, N_samples={N_SAMPLES}, '
                 f'x~N({X_MEAN},{X_STD}), eps_std={EPS_STD}')
    lines.append(f'sigma~N({SIGMA_MEAN},{SIGMA_STD}), V~N({V_MEAN:.0f},{V_STD:.0f})')
    lines.append('')
    lines.append('=' * 100)
    header = (f'{"beta_true":>9s}  {"alpha":>6s}  {"N":>6s}  '
              f'{"b_origin":>9s}  {"b_intcpt":>9s}  {"b_ratio":>9s}  '
              f'{"bias_orig":>10s}  {"bias_intcpt":>11s}  {"bias_ratio":>10s}')
    lines.append(header)
    lines.append('-' * 100)

    for _, row in results_df.iterrows():
        line = (f'{row["beta_true"]:9.2f}  {row["alpha"]:6.1f}  {row["N"]:6.0f}  '
                f'{row["beta_origin"]:9.4f}  {row["beta_intercept"]:9.4f}  '
                f'{row["beta_ratio"]:9.4f}  '
                f'{row["bias_origin"]:+10.4f}  {row["bias_intercept"]:+11.4f}  '
                f'{row["bias_ratio"]:+10.4f}')
        lines.append(line)
    lines.append('=' * 100)

    # --- Verdict ---
    avg_bias = results_df.groupby('beta_true')[
        ['bias_origin', 'bias_intercept', 'bias_ratio']
    ].mean()

    lines.append('')
    lines.append('AVERAGE ABSOLUTE BIAS (across alpha values):')
    lines.append('-' * 60)
    for bt in avg_bias.index:
        row = avg_bias.loc[bt]
        lines.append(
            f'  beta_true={bt:.1f}: '
            f'|bias_origin|={abs(row["bias_origin"]):.4f}  '
            f'|bias_intercept|={abs(row["bias_intercept"]):.4f}  '
            f'|bias_ratio|={abs(row["bias_ratio"]):.4f}'
        )

    overall_abs_bias = results_df[
        ['bias_origin', 'bias_intercept', 'bias_ratio']
    ].abs().mean()
    lines.append('')
    lines.append('OVERALL MEAN |BIAS| (across all beta_true x alpha):')
    lines.append(f'  Origin:    {overall_abs_bias["bias_origin"]:.4f}')
    lines.append(f'  Intercept: {overall_abs_bias["bias_intercept"]:.4f}')
    lines.append(f'  Ratio:     {overall_abs_bias["bias_ratio"]:.4f}')
    lines.append('')

    best = overall_abs_bias.idxmin().replace('bias_', '')
    lines.append(f'VERDICT: {best.upper()} estimator has lowest bias on synthetic data.')

    max_intcpt_bias = results_df['bias_intercept'].abs().max()
    if max_intcpt_bias < 0.02:
        lines.append('  Intercept estimator recovers beta_true to within 0.02 '
                      'across all settings.')
        lines.append('  -> Real data beta_intercept ~0.33 likely reflects TRUE '
                      'beta (not estimator bias).')
    else:
        lines.append(f'  WARNING: max intercept bias = {max_intcpt_bias:.4f} '
                      '(> 0.02), investigate further.')

    max_origin_bias = results_df['bias_origin'].abs().max()
    lines.append(f'  Origin estimator max bias = {max_origin_bias:.4f} '
                 '(omitted-intercept bias).')

    # --- Analytical explanation ---
    # Filter to beta_true=0.5 for the explanation section
    subset = results_df[results_df['beta_true'] == 0.5]
    explanation = print_analytical_explanation(subset)
    lines.append(explanation)

    report = '\n'.join(lines)
    txt_path.write_text(report)
    print(report)
    print(f'\nSaved report: {txt_path}')

    # --- Scatter plot ---
    make_scatter_plot(OUT_DIR)

    # --- Save CSV for further analysis ---
    csv_path = OUT_DIR / 'diag_1_synthetic.csv'
    results_df.to_csv(csv_path, index=False)
    print(f'Saved CSV: {csv_path}')


if __name__ == '__main__':
    main()
