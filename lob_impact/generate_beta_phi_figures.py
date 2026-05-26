#!/usr/bin/env python3
"""Generate β(φ) curve and Kyle λ crossover figures from beta_per_phi.csv."""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('pics_for_beta_phi')
OUT.mkdir(exist_ok=True)

COLORS = {
    'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
    'CGAN': '#7B4F9E', 'LobS5': '#C88A3A', 'S5-120M': '#D95F02',
    'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E', 'LobS5-v2': '#2CA02C',
}
MARKERS = {
    'Historic': 'x', 'Heuristic': 'd', 'CST': '^', 'CGAN': 's',
    'LobS5': 'o', 'S5-120M': 'v', 'S5-4K': '*', 'S5-360M': 'H', 'LobS5-v2': 'P',
}

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

df = pd.read_csv(OUT / 'beta_per_phi.csv')

# Compute φ for rows where it's missing
# V4: mb≈12.5 (average of 5,10,15,20), market_vol depends on stock
# Approximate φ from child/depth ratio
PHI_APPROX = {
    ('GOOG', 75): 97, ('GOOG', 300): 99, ('GOOG', 485): 99,
    ('INTC', 75): 40, ('INTC', 300): 80, ('INTC', 485): 90,
    # V4 volumes against depth: GOOG depth≈166, INTC depth≈1248
    # φ = vol / (vol + mb * r * q_med)
    # GOOG: r=1.35%, q_med=22, mb=12.5 → mkt=3.7
    # INTC: r=3.6%, q_med=100, mb=12.5 → mkt=45
}
# Better: compute from actual grid data
# GOOG mb average ≈ 10 (median of 5,10,15,20), r=0.0135, q_med=22
# mkt_vol_goog = 10 * 0.0135 * 22 = 2.97
# INTC r=0.036, q_med=100 → mkt_vol_intc = 10 * 0.036 * 100 = 36

for idx, row in df.iterrows():
    if pd.isna(row['phi']):
        vol = row['vol']
        stock = row['stock']
        mb = row['mb']
        if stock == 'GOOG':
            mkt = mb * 0.0135 * 22
        elif stock == 'INTC':
            mkt = mb * 0.036 * 100
        elif stock == 'AAPL':
            mkt = mb * 0.0273 * 40
        elif stock == 'AMZN':
            mkt = mb * 0.0217 * 25
        else:
            mkt = 10
        phi = vol / (vol + mkt) * 100
        df.at[idx, 'phi'] = phi

df['phi'] = df['phi'].astype(float)
df = df.sort_values(['model', 'stock', 'phi'])

# Save updated CSV
df.to_csv(OUT / 'beta_per_phi.csv', index=False)

# ═══════════════════════════════════════════════════════
# Figure 1: β(φ) curve — all models, all stocks combined
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))

# Average β across stocks at each φ level for cleaner curves
for model in ['S5-4K', 'S5-120M', 'S5-360M', 'LobS5', 'Historic', 'Heuristic', 'CST', 'CGAN']:
    md = df[df['model'] == model]
    if md.empty:
        continue
    # Group by approximate φ bins
    md = md.copy()
    md['phi_bin'] = pd.cut(md['phi'], bins=[0, 25, 45, 55, 65, 75, 85, 92, 97, 100],
                           labels=[15, 35, 50, 60, 70, 80, 88, 95, 99])
    grouped = md.groupby('phi_bin').agg({'beta_int': 'mean', 'phi': 'mean'}).dropna()

    if len(grouped) < 2:
        # Plot individual points
        ax.scatter(md['phi'], md['beta_int'], color=COLORS.get(model, 'gray'),
                   marker=MARKERS.get(model, 'o'), s=40, alpha=0.5, label=model)
    else:
        ax.plot(grouped['phi'], grouped['beta_int'], 'o-',
                color=COLORS.get(model, 'gray'), marker=MARKERS.get(model, 'o'),
                lw=2, markersize=8, label=model, alpha=0.8)

ax.axhline(0.5, ls='--', color='red', lw=2, label='Theory β=0.5')
ax.axhline(0, ls='-', color='gray', lw=0.5)
ax.set_xlabel('Participation Rate φ (%)', fontsize=13)
ax.set_ylabel('β (intercept estimator)', fontsize=13)
ax.set_title('β(φ) Curve: Scaling Exponent vs Participation Rate', fontsize=15, fontweight='bold')
ax.legend(fontsize=9, ncol=2, loc='upper left')
ax.set_xlim(0, 100)
ax.set_ylim(-0.1, 0.55)
fig.tight_layout()
fig.savefig(OUT / 'beta_vs_phi_curve.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: beta_vs_phi_curve.png')

# ═══════════════════════════════════════════════════════
# Figure 2: β(φ) per stock (4 panels)
# ═══════════════════════════════════════════════════════
stocks = ['GOOG', 'INTC', 'AAPL', 'AMZN']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, stock in enumerate(stocks):
    ax = axes[idx // 2][idx % 2]
    ds = df[df['stock'] == stock]
    for model in ds['model'].unique():
        md = ds[ds['model'] == model].sort_values('phi')
        c = COLORS.get(model, 'gray')
        m = MARKERS.get(model, 'o')
        ax.plot(md['phi'], md['beta_int'], 'o-', color=c, marker=m, lw=1.5, markersize=6, label=model, alpha=0.7)
    ax.axhline(0.5, ls='--', color='red', lw=1.5)
    ax.axhline(0, ls='-', color='gray', lw=0.5)
    ax.set_title(f'{stock}', fontsize=13, fontweight='bold')
    ax.set_xlabel('φ (%)')
    ax.set_ylabel('β')
    ax.set_ylim(-0.1, 0.55)
    if idx == 0:
        ax.legend(fontsize=7, ncol=2)
fig.suptitle('β(φ) per Stock', fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUT / 'beta_vs_phi_per_stock.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: beta_vs_phi_per_stock.png')

# ═══════════════════════════════════════════════════════
# Figure 3: Kyle λ crossover — λ_k1 and λ_kmax vs φ (S5-4K)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, stock in enumerate(['AAPL', 'GOOG']):
    ax = axes[idx]
    for model in ['S5-4K', 'Historic', 'CST']:
        md = df[(df['model'] == model) & (df['stock'] == stock)].sort_values('phi')
        if md.empty:
            continue
        c = COLORS.get(model, 'gray')
        ax.plot(md['phi'], md['lam_k1'], 'o--', color=c, lw=1.5, markersize=5, alpha=0.6, label=f'{model} λ(k=1)')
        ax.plot(md['phi'], md['lam_kmax'], 's-', color=c, lw=2, markersize=6, label=f'{model} λ(k=max)')
    ax.set_title(f'{stock}', fontsize=13, fontweight='bold')
    ax.set_xlabel('φ (%)')
    ax.set_ylabel('Kyle λ')
    ax.legend(fontsize=8)

fig.suptitle('Kyle λ vs Participation Rate: λ(k=1) vs λ(k=max)', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'kyle_lambda_vs_phi.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: kyle_lambda_vs_phi.png')

# ═══════════════════════════════════════════════════════
# Figure 4: R² vs φ (signal strength)
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for model in ['S5-4K', 'Historic', 'Heuristic', 'CST']:
    md = df[df['model'] == model].sort_values('phi')
    if md.empty:
        continue
    c = COLORS.get(model, 'gray')
    ax.scatter(md['phi'], md['r2'], color=c, marker=MARKERS.get(model, 'o'), s=50, alpha=0.6, label=model)
ax.set_xlabel('Participation Rate φ (%)', fontsize=12)
ax.set_ylabel('R² (goodness of fit)', fontsize=12)
ax.set_title('Signal Strength vs Participation Rate', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'r2_vs_phi.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: r2_vs_phi.png')

print(f'\nAll figures saved to {OUT}/')
print(f'Total: {len(list(OUT.glob("*.png")))} PNGs')
