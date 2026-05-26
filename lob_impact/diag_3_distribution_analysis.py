#!/usr/bin/env python3
"""Diagnostic 3: Distribution Analysis of x=log(Q/V) and y=log(I/σ).

Explains WHY the origin estimator gives β≈0.82:
  β_origin = β_true + α·Σ(x)/Σ(x²)

When α≈-9 and mean(x)≈-11, the bias ≈ +0.5.

Usage:
    python lob_impact/diag_3_distribution_analysis.py
    python lob_impact/diag_3_distribution_analysis.py --stock INTC
"""
import argparse, pickle
import numpy as np
from pathlib import Path

OUT_DIR = Path('pics_for_investigation')
MODELS_SKIP = {'ZeroInsertions'}


def run(stock):
    OUT_DIR.mkdir(exist_ok=True)
    src = Path(f'pics_for_v4_300_{stock}')
    lines = [f'=== Diagnostic 3: Distribution Analysis ({stock}) ===\n']

    rows = []
    scatter_model = None
    scatter_x, scatter_y_adj, scatter_y_raw = None, None, None

    for pkl in sorted(src.glob('*.metrics.pkl')):
        with open(pkl, 'rb') as f:
            mc = pickle.load(f)
        model = mc.get('model', pkl.stem.replace('.metrics', ''))
        if model in MODELS_SKIP:
            continue
        beta_dict = mc.get('beta', {})
        pc_x = beta_dict.get('pc_x', np.array([]))
        pc_y = beta_dict.get('pc_y', np.array([]))        # log(I/σ)
        pc_y_raw = beta_dict.get('pc_y_raw', np.array([]))  # log(I)
        alpha = beta_dict.get('alpha', np.nan)
        beta_int = beta_dict.get('beta', np.nan)
        beta_orig = beta_dict.get('beta_origin', np.nan)

        if len(pc_x) < 10:
            continue

        ok = np.isfinite(pc_x) & np.isfinite(pc_y) & np.isfinite(pc_y_raw) & (pc_x != 0)
        x = pc_x[ok]
        y_adj = pc_y[ok]
        y_raw = pc_y_raw[ok]

        # Exact bias formula for origin estimator:
        # β_origin = Σ(x·y_adj) / Σ(x²)
        # If true model: y_raw = α + β_true·x + ε, and y_adj = y_raw - log(σ)
        # Then y_adj = (α - log(σ)) + β_true·x + ε = a + β_true·x + ε
        # β_origin = β_true + a·Σ(x)/Σ(x²)
        # where a = mean(y_adj) - β_origin·mean(x) ... but simpler:
        # Predicted bias = (β_origin - β_intercept) should match α·Σx/Σx²
        # Note: the exact formula uses the intercept of y_adj vs x, not y_raw vs x

        sum_x = np.sum(x)
        sum_x2 = np.sum(x ** 2)
        n = len(x)
        mean_x = np.mean(x)
        mean_x2 = np.mean(x ** 2)

        # Intercept of y_adj vs x (the "a" in y_adj = a + β·x)
        coeffs_adj = np.polyfit(x, y_adj, 1)
        beta_adj_intercept = float(coeffs_adj[0])
        a_adj = float(coeffs_adj[1])

        # Predicted origin bias
        predicted_bias = a_adj * sum_x / sum_x2
        actual_bias = beta_orig - beta_int

        # What α (of y_adj) would make β_origin = 0.5?
        # 0.5 = β_adj_intercept + a_target · sum_x / sum_x2
        # a_target = (0.5 - β_adj_intercept) · sum_x2 / sum_x
        a_target_05 = (0.5 - beta_adj_intercept) * sum_x2 / sum_x if abs(sum_x) > 1e-10 else np.nan

        rows.append(dict(
            model=model, n=n,
            mean_x=mean_x, std_x=np.std(x), min_x=np.min(x), max_x=np.max(x),
            mean_y_adj=np.mean(y_adj), mean_y_raw=np.mean(y_raw),
            alpha_raw=alpha,  # intercept from log(I) = α + β·log(Q/V)
            a_adj=a_adj,       # intercept from log(I/σ) = a + β·log(Q/V)
            beta_int=beta_int, beta_orig=beta_orig,
            beta_adj_int=beta_adj_intercept,
            predicted_bias=predicted_bias, actual_bias=actual_bias,
            a_target_05=a_target_05,
        ))

        # Save scatter data for one model
        if model in ('LobS5', 'Historic') and scatter_model is None:
            scatter_model = model
            scatter_x = x
            scatter_y_adj = y_adj
            scatter_y_raw = y_raw

    # Print results table
    lines.append(f'\n{"Model":<12} {"n":>7} {"mean(x)":>8} {"mean(y_adj)":>11} {"α_raw":>7} '
                 f'{"a_adj":>7} {"β_int":>7} {"β_orig":>7} {"pred_bias":>10} {"act_bias":>10} {"a_for_0.5":>10}')
    lines.append('-' * 110)
    for r in rows:
        lines.append(f'{r["model"]:<12} {r["n"]:7d} {r["mean_x"]:8.2f} {r["mean_y_adj"]:11.2f} '
                     f'{r["alpha_raw"]:7.2f} {r["a_adj"]:7.2f} {r["beta_int"]:7.4f} {r["beta_orig"]:7.4f} '
                     f'{r["predicted_bias"]:10.4f} {r["actual_bias"]:10.4f} {r["a_target_05"]:10.2f}')

    # Summary
    lines.append(f'\n=== Analysis ===')
    if rows:
        mean_pred = np.mean([r['predicted_bias'] for r in rows])
        mean_act = np.mean([r['actual_bias'] for r in rows])
        lines.append(f'Mean predicted bias: {mean_pred:.4f}')
        lines.append(f'Mean actual bias:    {mean_act:.4f}')
        lines.append(f'Match: {"YES" if abs(mean_pred - mean_act) < 0.05 else "NO"} '
                     f'(Δ={abs(mean_pred - mean_act):.4f})')
        lines.append(f'\nTo get β_origin=0.5, need a_adj ≈ {rows[0]["a_target_05"]:.2f} '
                     f'(currently {rows[0]["a_adj"]:.2f})')
        lines.append(f'This means log(I/σ) intercept must change by '
                     f'{rows[0]["a_target_05"] - rows[0]["a_adj"]:.2f}')
        lines.append(f'\nData cloud center: ({np.mean([r["mean_x"] for r in rows]):.2f}, '
                     f'{np.mean([r["mean_y_adj"] for r in rows]):.2f})')
        lines.append(f'Origin (0, 0) is FAR from data — origin estimator is heavily biased.')

    # Sensitivity: what V would shift mean(x) enough?
    if rows:
        r0 = rows[0]
        # If we scale V by factor k, mean(x) shifts by -log(k)
        # We need: a_adj * sum_x_new / sum_x2_new gives β_origin = 0.5
        # Approximately: need mean(x) to be a_target_05 / a_adj * mean(x)
        if abs(r0['a_adj']) > 0.01 and abs(r0['mean_x']) > 0.01:
            ratio = r0['a_target_05'] / r0['a_adj']
            lines.append(f'\nV sensitivity: to get β_origin=0.5, need mean(x) × {ratio:.2f}')
            lines.append(f'  i.e., use V = V_daily × {np.exp(r0["mean_x"] * (1 - ratio)):.1f}')

    report = '\n'.join(lines)
    print(report)

    out_file = OUT_DIR / f'diag_3_distribution_{stock}.txt'
    with open(out_file, 'w') as f:
        f.write(report)
    print(f'\nSaved: {out_file}')

    # Scatter plot
    if scatter_x is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Subsample for visibility
            n_show = min(2000, len(scatter_x))
            idx = np.random.default_rng(42).choice(len(scatter_x), n_show, replace=False)

            # Left: y_adj vs x (origin estimator space)
            ax1.scatter(scatter_x[idx], scatter_y_adj[idx], s=2, alpha=0.2, color='steelblue')
            xl = np.array([scatter_x.min(), scatter_x.max()])
            r0 = rows[0]
            ax1.plot(xl, r0['beta_orig'] * xl, 'r-', lw=2,
                     label=f'Origin: β={r0["beta_orig"]:.3f}')
            ax1.plot(xl, r0['beta_adj_int'] * xl + r0['a_adj'], 'g--', lw=2,
                     label=f'Intercept: β={r0["beta_adj_int"]:.3f}')
            ax1.plot(xl, 0.5 * xl, 'k:', lw=1.5, label='β=0.5')
            ax1.scatter(0, 0, s=200, marker='*', color='gold', edgecolors='black',
                        zorder=10, label='Origin (0,0)')
            ax1.set(title=f'{scatter_model}: log(I/σ) vs log(Q/V)',
                    xlabel='x = log(Q/V)', ylabel='y_adj = log(I/σ)')
            ax1.legend(fontsize=8)

            # Right: y_raw vs x (intercept estimator space)
            ax2.scatter(scatter_x[idx], scatter_y_raw[idx], s=2, alpha=0.2, color='coral')
            ax2.plot(xl, r0['beta_int'] * xl + r0['alpha_raw'], 'g--', lw=2,
                     label=f'Intercept: β={r0["beta_int"]:.3f}, α={r0["alpha_raw"]:.1f}')
            ax2.plot(xl, 0.5 * xl + (-0.5 * np.mean(xl) + np.mean(scatter_y_raw)), 'k:', lw=1.5,
                     label='β=0.5 (shifted)')
            ax2.set(title=f'{scatter_model}: log(I) vs log(Q/V)',
                    xlabel='x = log(Q/V)', ylabel='y_raw = log(I)')
            ax2.legend(fontsize=8)

            fig.suptitle(f'Diagnostic 3: Origin Estimator Bias ({stock})', fontsize=14, fontweight='bold')
            fig.tight_layout()
            png = OUT_DIR / f'diag_3_scatter_{stock}.png'
            fig.savefig(png, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {png}')
        except Exception as e:
            print(f'Scatter plot failed: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', default='GOOG')
    args = parser.parse_args()
    run(args.stock.upper())
