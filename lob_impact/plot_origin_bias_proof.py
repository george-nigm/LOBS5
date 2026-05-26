#!/usr/bin/env python3
"""
Synthetic proof: origin estimator gives β ≈ 0.47 even when I = const (zero impact).

Generates a figure with two panels:
  Left:  Scatter of synthetic data (I=const) + origin line (β≈0.47) + intercept line (β≈0)
  Right: Same with realistic I ∝ (Q/V)^0.5 data, showing both estimators agree

Usage:
    python lob_impact/plot_origin_bias_proof.py --out_dir pics_for_origin_proof
"""
import argparse
import numpy as np
from pathlib import Path


def fit_origin(x, y):
    """β = dot(x,y) / dot(x,x)"""
    ok = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[ok], y[ok]
    return float(np.dot(xv, yv) / np.dot(xv, xv))


def fit_intercept(x, y):
    """OLS: y = α + β x"""
    ok = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[ok], y[ok]
    coeffs = np.polyfit(xv, yv, 1)
    return float(coeffs[0]), float(coeffs[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='pics_for_origin_proof')
    parser.add_argument('--n_points', type=int, default=50000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.random.seed(42)
    N = args.n_points

    # Realistic parameter ranges (from actual AAPL/AMZN data)
    # Q/V ~ 10^{-6} to 10^{-3}  →  log(Q/V) ~ -14 to -7
    # σ ~ 0.002 to 0.02          →  log(σ) ~ -6.2 to -3.9
    log_QV = np.random.uniform(-13.5, -7.5, N)
    log_sigma = np.random.normal(-4.8, 0.3, N)

    # ── Panel 1: I = const (ZERO impact scaling) ──
    I_const = 5e-5  # typical impact magnitude
    noise1 = np.random.normal(0, 0.15, N)
    log_I_const = np.log(I_const) + noise1

    x1 = log_QV
    y1 = log_I_const - log_sigma  # log(I/σ)

    beta_orig_1 = fit_origin(x1, y1)
    beta_int_1, alpha_int_1 = fit_intercept(x1, y1)

    # ── Panel 2: I ∝ σ (Q/V)^0.5 (PERFECT square-root law) ──
    Y = 0.3  # prefactor
    noise2 = np.random.normal(0, 0.15, N)
    log_I_sqrt = np.log(Y) + log_sigma + 0.5 * log_QV + noise2

    x2 = log_QV
    y2 = log_I_sqrt - log_sigma  # log(I/σ) = log(Y) + 0.5*log(Q/V) + noise

    beta_orig_2 = fit_origin(x2, y2)
    beta_int_2, alpha_int_2 = fit_intercept(x2, y2)

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, x, y, beta_o, beta_i, alpha_i, title, true_beta in [
        (axes[0], x1, y1, beta_orig_1, beta_int_1, alpha_int_1,
         f'$I = \\mathrm{{const}}$ (zero impact scaling)\nTrue $\\beta = 0$', 0.0),
        (axes[1], x2, y2, beta_orig_2, beta_int_2, alpha_int_2,
         f'$I \\propto \\sigma (Q/V)^{{0.5}}$ (perfect square-root)\nTrue $\\beta = 0.5$', 0.5),
    ]:
        ax.scatter(x, y, s=0.5, alpha=0.03, color='#555555', rasterized=True)

        x_range = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)

        # Origin line
        ax.plot(x_range, beta_o * x_range, 'r-', lw=2.5,
                label=f'Origin: $\\beta = {beta_o:.3f}$')

        # Intercept line
        ax.plot(x_range, beta_i * x_range + alpha_i, 'b--', lw=2.5,
                label=f'Intercept: $\\beta = {beta_i:.3f}$')

        # Theory
        if true_beta > 0:
            y_theory = true_beta * x_range + np.log(Y)
            ax.plot(x_range, y_theory, 'g:', lw=2, alpha=0.7,
                    label=f'Theory: $\\beta = {true_beta}$')

        # Mark origin
        ax.plot(0, 0, 'k+', markersize=15, mew=2, zorder=10)
        ax.annotate('Origin (0,0)', (0, 0), fontsize=9,
                    xytext=(0.5, 0.5), textcoords='offset fontsize')

        ax.set_xlabel('$\\ln(Q/V)$', fontsize=13)
        ax.set_ylabel('$\\ln(I/\\sigma)$', fontsize=13)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Origin Estimator Ratio Bias: Proof by Construction\n'
                 'N = {:,} synthetic points, realistic parameter ranges'.format(N),
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = out_dir / 'origin_bias_proof.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {png_path}')
    print(f'\nPanel 1 (I=const):     origin β = {beta_orig_1:.4f},  intercept β = {beta_int_1:.4f}')
    print(f'Panel 2 (I∝√(Q/V)):   origin β = {beta_orig_2:.4f},  intercept β = {beta_int_2:.4f}')


if __name__ == '__main__':
    main()
