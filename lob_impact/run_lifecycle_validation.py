#!/usr/bin/env python3
"""
Lifecycle Validation — 6 empirical checks on existing v4 metrics.

Reads metrics pickles from pics_for_v4_300_{STOCK}/, performs 6 checks
from the market impact literature, generates PDF report + CSV summary.

Checks:
  1. Execution Profile (concavity): δ_exec < 1 via kyle_lambda I(k) ~ k^δ
  2. Fair Pricing (2/3 rule): relaxation ratio ≈ 0.667
  3. Duration Independence: β invariant to mb (execution duration)
  4. No-Arbitrage: γ_decay ≤ 1 − δ_impact
  5. Propagator Consistency: propagator decay + order flow autocorrelation ≈ 1
  6. Benchmark Comparison Table

Usage:
  python lob_impact/run_lifecycle_validation.py --stock GOOG
  python lob_impact/run_lifecycle_validation.py --stock GOOG --stock INTC
"""
import argparse, pickle, sys, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from scipy import optimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings; warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
DPI = 200

MODEL_META = OrderedDict([
    ('ZeroInsertions',  dict(color='#7CAE7A', marker='D', group='Null')),
    ('Historic',        dict(color='#90939C', marker='x', group='Baseline')),
    ('Heuristic',       dict(color='#546884', marker='d', group='Baseline')),
    ('CST',             dict(color='#213552', marker='^', group='Parametric')),
    ('CGAN',            dict(color='#7B4F9E', marker='s', group='Parametric')),
    ('LobS5',           dict(color='#C88A3A', marker='o', group='S5 Neural')),
    ('S5-120M',         dict(color='#D95F02', marker='v', group='S5 Neural')),
    ('S5-4K',           dict(color='#5B7BBF', marker='*', group='S5 Neural')),
    ('S5-360M',         dict(color='#B5446E', marker='H', group='S5 Neural')),
    ('LobS5-v2',        dict(color='#2CA02C', marker='P', group='S5 Neural')),
])

IMPACT_MODELS = [m for m in MODEL_META if m != 'ZeroInsertions']

def _c(m):  return MODEL_META.get(m, {}).get('color', '#888')
def _mk(m): return MODEL_META.get(m, {}).get('marker', 'o')
def _grp(m): return MODEL_META.get(m, {}).get('group', '?')

# Empirical targets from literature
EMPIRICAL = dict(
    delta=0.5,          # Kyle (1985), Toth (2011): I ~ Q^0.5
    relaxation=0.667,   # Bouchaud (2010): perm/peak ≈ 2/3
    gamma_max=0.5,      # No-arb: γ ≤ 1 - δ at δ=0.5
    hurst=0.5,          # Efficient market: H ≈ 0.5
)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ═══════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════
def load_metrics(stock, src_dir=None):
    src = Path(src_dir) if src_dir else Path(f'pics_for_v4_300_{stock}')
    if not src.exists():
        print(f'ERROR: {src} not found'); return None
    data = {}
    for model in MODEL_META:
        pkl = src / f'{model}.metrics.pkl'
        if not pkl.exists():
            print(f'  [skip] {model}'); continue
        with open(pkl, 'rb') as f:
            data[model] = pickle.load(f)
    print(f'Loaded {len(data)}/{len(MODEL_META)} models from {src}/')
    return data


# ═══════════════════════════════════════════════════════════════════════
# Check 1: Execution Profile (concavity)
# ═══════════════════════════════════════════════════════════════════════
def _parse_config_name(name):
    """Parse config name like 'i10_c100_mb4_v105_cntxt88%' → dict."""
    m = re.match(r'i(\d+)_c(\d+)_mb(\d+)_v(\d+)_cntxt(\d+)%', name)
    if not m:
        return None
    return dict(i=int(m.group(1)), c=int(m.group(2)),
                mb=int(m.group(3)), vol=int(m.group(4)),
                cntxt=int(m.group(5)))


def check_execution_profile(data):
    """Extract I(k) from master curves at injection points.

    For each config, the k-th injection happens at normalized u = k/i.
    Read midprice impact at these points.
    Fit: I(k/i) = a * (k/i)^δ_exec. Concavity requires δ_exec < 1.
    """
    results = {}
    for model in IMPACT_MODELS:
        if model not in data:
            continue
        mcs = data[model].get('master_curves', {})
        if not mcs:
            continue

        # Collect I(k/i) across configs, normalized to I(1) = I at final insertion
        # Use fractional progress f = k/i ∈ (0, 1]
        frac_impact = {}  # f → list of normalized impacts

        for cfg_name, mc in mcs.items():
            cfg = _parse_config_name(cfg_name)
            if cfg is None or cfg['i'] < 3:
                continue
            u = mc['u']
            mean = mc['mean']
            i_val = cfg['i']

            # Injection points at u_k = k/i for k = 1..i
            # But u is normalized by L = i*(mb+1), so u for injection k is:
            # u_k = k * (mb+1) / L = k/i
            impacts = []
            for k in range(1, i_val + 1):
                u_k = k / i_val
                I_k = float(np.interp(u_k, u, mean))
                impacts.append(I_k)

            # Normalize by I at final injection (k=i, u=1)
            I_final = impacts[-1]
            if I_final <= 1e-12:
                continue

            for k in range(1, i_val + 1):
                f = k / i_val
                f_round = round(f, 3)
                frac_impact.setdefault(f_round, []).append(impacts[k-1] / I_final)

        # Average I_norm(f) across configs
        fracs = sorted(frac_impact.keys())
        if len(fracs) < 3:
            continue
        f_arr = np.array(fracs)
        I_arr = np.array([np.median(frac_impact[f]) for f in fracs])

        # Filter: only positive and finite
        ok = (f_arr > 0) & (I_arr > 0) & np.isfinite(I_arr)
        if ok.sum() < 3:
            continue

        # Fit: log(I_norm) = δ * log(f) + log(a)
        log_f = np.log(f_arr[ok])
        log_I = np.log(I_arr[ok])
        coeffs = np.polyfit(log_f, log_I, 1)
        delta_exec = float(coeffs[0])
        a = float(np.exp(coeffs[1]))
        yhat = coeffs[0] * log_f + coeffs[1]
        ss_res = np.sum((log_I - yhat)**2)
        ss_tot = np.sum((log_I - np.mean(log_I))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results[model] = dict(
            delta_exec=delta_exec, a=a, r2=r2,
            f=f_arr[ok].tolist(), I_norm=I_arr[ok].tolist(),
            concave=delta_exec < 1.0,
        )
    return results


def fig_execution_profile(results, stock, pdf):
    """Figure: normalized execution profile I(k/i) with power-law fits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: I_norm vs f = k/i
    ax = axes[0]
    for model, r in results.items():
        f = np.array(r['f'])
        ax.plot(f, r['I_norm'], 'o-', color=_c(model), ms=4, lw=1.5,
                label=f'{model} (δ={r["delta_exec"]:.2f})')
    # Reference curves
    f_ref = np.linspace(0.05, 1, 100)
    ax.plot(f_ref, f_ref**0.5, 'r--', lw=2, alpha=0.5, label='√(k/i) concave')
    ax.plot(f_ref, f_ref, ':', color='gray', lw=1.5, alpha=0.5, label='linear')
    ax.set(xlabel='k / i (fractional progress)', ylabel='I(k) / I(i) (normalized)',
           title=f'{stock}: Execution Profile I(k)')
    ax.legend(fontsize=7, ncol=2)

    # Right: log-log with fits
    ax = axes[1]
    for model, r in results.items():
        f = np.array(r['f'])
        I = np.array(r['I_norm'])
        ok = (f > 0) & (I > 0)
        ax.scatter(np.log(f[ok]), np.log(I[ok]),
                   color=_c(model), s=20, alpha=0.6, marker=_mk(model))
        f_fit = np.linspace(np.log(f[ok].min()), 0, 50)
        ax.plot(f_fit, r['delta_exec'] * f_fit + np.log(r['a']),
                color=_c(model), lw=1.5, label=f'{model}: δ={r["delta_exec"]:.3f}')
    ax.axline((0, 0), slope=1.0, ls=':', color='gray', lw=1, label='linear (δ=1)')
    ax.axline((0, 0), slope=0.5, ls='--', color='red', lw=1.5, label='√k (δ=0.5)')
    ax.set(xlabel='log(k/i)', ylabel='log(I_norm)', title=f'{stock}: Log-Log Execution Profile')
    ax.legend(fontsize=6, ncol=2)

    fig.suptitle('Check 1: Execution Profile — Concavity Test', fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Check 2: Fair Pricing (2/3 rule)
# ═══════════════════════════════════════════════════════════════════════
def check_fair_pricing(data):
    """Relaxation ratio = I_permanent / I_peak. Target ≈ 2/3."""
    results = {}
    for model in IMPACT_MODELS:
        if model not in data:
            continue
        relax = data[model].get('relax', [])
        if not relax:
            continue
        arr = np.array(relax)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr))
        target = EMPIRICAL['relaxation']
        dist = abs(mean_r - target)
        results[model] = dict(
            mean=mean_r, std=std_r, n=len(arr),
            target=target, distance=dist,
            pass_strict=dist < 0.10,    # within 10% of 2/3
            pass_loose=dist < 0.20,     # within 20% of 2/3
        )
    return results


def fig_fair_pricing(results, stock, pdf):
    """Figure: relaxation ratio per model vs 2/3 target."""
    models = [m for m in IMPACT_MODELS if m in results]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    vals = [results[m]['mean'] for m in models]
    errs = [results[m]['std'] for m in models]
    colors = [_c(m) for m in models]
    bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor='gray', alpha=0.8)
    ax.errorbar(x, vals, yerr=errs, fmt='none', ecolor='black', capsize=4, lw=1.5)

    # 2/3 target band
    ax.axhline(EMPIRICAL['relaxation'], ls='--', color='red', lw=2, label='Target: 2/3 = 0.667')
    ax.axhspan(EMPIRICAL['relaxation'] - 0.10, EMPIRICAL['relaxation'] + 0.10,
               alpha=0.08, color='red', label='±0.10 band')

    for i, (m, v) in enumerate(zip(models, vals)):
        ax.text(i, v + errs[i] + 0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set(ylabel='Relaxation Ratio (perm/peak)', ylim=(0, max(vals) * 1.3 + 0.1),
           title=f'{stock}: Fair Pricing — Relaxation Ratio vs 2/3 Rule')
    ax.legend(fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Check 3: Duration Independence
# ═══════════════════════════════════════════════════════════════════════
def check_duration_independence(data):
    """β should be invariant to mb (execution duration).

    Fixed Q but different mb = different execution speed.
    Duration-independent impact means β(mb) ≈ const.
    """
    results = {}
    for model in IMPACT_MODELS:
        if model not in data:
            continue
        mb_betas = data[model].get('mb_betas', {})
        if len(mb_betas) < 3:
            continue
        mbs = sorted(mb_betas.keys())
        betas = np.array([mb_betas[mb] for mb in mbs])
        mean_b = float(np.mean(betas))
        std_b = float(np.std(betas))
        cv = std_b / abs(mean_b) if abs(mean_b) > 1e-10 else np.inf
        # Range
        beta_range = float(np.max(betas) - np.min(betas))
        # Slope (linear trend)
        if len(mbs) >= 3:
            slope = float(np.polyfit(mbs, betas, 1)[0])
        else:
            slope = np.nan

        results[model] = dict(
            mbs=mbs, betas=betas.tolist(),
            mean=mean_b, std=std_b, cv=cv,
            range=beta_range, slope=slope,
            # Duration-independent if CV < 0.25 and range < 0.15
            pass_cv=cv < 0.25,
            pass_range=beta_range < 0.15,
            independent=cv < 0.25 and beta_range < 0.15,
        )
    return results


def fig_duration_independence(results, stock, pdf):
    """Figure: β vs mb for each model."""
    models = [m for m in IMPACT_MODELS if m in results]
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: β vs mb per model
    ax = axes[0]
    for model in models:
        r = results[model]
        ax.plot(r['mbs'], r['betas'], 'o-', color=_c(model), ms=5, lw=1.5,
                marker=_mk(model), label=f'{model} (CV={r["cv"]:.2f})')
    ax.axhline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
    ax.set(xlabel='mb (cooling messages = execution duration)',
           ylabel=r'$\beta_{intercept}$',
           title=f'{stock}: β vs Execution Duration (mb)')
    ax.legend(fontsize=6, ncol=2)

    # Right: CV and range summary
    ax = axes[1]
    cvs = [results[m]['cv'] for m in models]
    ranges = [results[m]['range'] for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, cvs, w, color=[_c(m) for m in models], alpha=0.7, label='CV(β)')
    ax.bar(x + w/2, ranges, w, color=[_c(m) for m in models], edgecolor='black',
           lw=0.5, alpha=0.4, label='Range(β)')
    ax.axhline(0.25, ls=':', color='blue', lw=1, label='CV threshold (0.25)')
    ax.axhline(0.15, ls=':', color='green', lw=1, label='Range threshold (0.15)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set(ylabel='CV / Range', title=f'{stock}: Duration Independence Metrics')
    ax.legend(fontsize=8)

    fig.suptitle('Check 3: Duration Independence — β Invariant to mb?',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Check 4: No-Arbitrage Conditions
# ═══════════════════════════════════════════════════════════════════════
def check_no_arbitrage(data):
    """γ_decay ≤ 1 − δ_impact.

    If violated, the model permits round-trip price manipulation.
    δ = impact exponent (our β from cross-sectional regression).
    γ = decay exponent from master curve post-peak.
    """
    results = {}
    for model in IMPACT_MODELS:
        if model not in data:
            continue
        sc = data[model].get('scorecard', {})
        beta_d = data[model].get('beta', {})
        gamma = sc.get('gamma', np.nan)
        delta = beta_d.get('beta', np.nan)  # impact exponent = β_intercept

        if np.isnan(gamma) or np.isnan(delta):
            continue

        threshold = 1.0 - delta
        margin = threshold - gamma
        results[model] = dict(
            gamma=float(gamma), delta=float(delta),
            threshold=float(threshold), margin=float(margin),
            no_arb=gamma <= threshold,
        )
    return results


def fig_no_arbitrage(results, stock, pdf):
    """Figure: γ vs δ scatter with no-arb boundary."""
    models = [m for m in IMPACT_MODELS if m in results]
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter γ vs δ with boundary
    ax = axes[0]
    for model in models:
        r = results[model]
        color = 'green' if r['no_arb'] else 'red'
        ax.scatter(r['delta'], r['gamma'], color=_c(model), s=80, marker=_mk(model),
                   edgecolors=color, linewidths=2, zorder=5,
                   label=f'{model} (δ={r["delta"]:.3f}, γ={r["gamma"]:.3f})')
    # Boundary: γ = 1 - δ
    d_range = np.linspace(0, 1, 100)
    ax.plot(d_range, 1 - d_range, 'r--', lw=2, label='γ = 1 − δ (boundary)')
    ax.fill_between(d_range, 1 - d_range, 1.5, alpha=0.05, color='red')
    ax.fill_between(d_range, -0.5, 1 - d_range, alpha=0.05, color='green')
    ax.set(xlabel=r'$\delta$ (impact exponent = $\beta$)',
           ylabel=r'$\gamma$ (decay exponent)',
           xlim=(-0.05, 0.8), ylim=(-0.3, 1.2),
           title=f'{stock}: No-Arbitrage Boundary')
    ax.text(0.15, 0.3, 'NO-ARB\n(safe)', fontsize=12, color='green', alpha=0.5, ha='center')
    ax.text(0.6, 0.8, 'ARBITRAGE\n(violated)', fontsize=12, color='red', alpha=0.5, ha='center')
    ax.legend(fontsize=6, ncol=1, loc='upper left')

    # Right: margin (distance from boundary)
    ax = axes[1]
    x = np.arange(len(models))
    margins = [results[m]['margin'] for m in models]
    colors_bar = ['green' if results[m]['no_arb'] else 'red' for m in models]
    ax.bar(x, margins, color=colors_bar, alpha=0.7, edgecolor='gray')
    ax.axhline(0, ls='-', color='black', lw=1)
    for i, (m, v) in enumerate(zip(models, margins)):
        ax.text(i, v + (0.02 if v >= 0 else -0.04), f'{v:.3f}',
                ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set(ylabel='Margin (1 − δ − γ)', title=f'{stock}: No-Arb Margin (>0 = safe)')

    fig.suptitle('Check 4: No-Arbitrage Conditions — γ ≤ 1 − δ',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Check 5: Propagator Consistency
# ═══════════════════════════════════════════════════════════════════════
def fit_propagator_decay(G, max_lag=100):
    """Fit G(l) ~ l^{-β_prop} from propagator response.

    Returns β_prop (propagator decay exponent).
    """
    lags = np.arange(1, min(len(G), max_lag + 1))
    g_vals = G[1:min(len(G), max_lag + 1)]
    ok = (g_vals > 0) & np.isfinite(g_vals)
    if ok.sum() < 5:
        return np.nan, np.nan
    log_l = np.log(lags[ok])
    log_g = np.log(g_vals[ok])
    coeffs = np.polyfit(log_l, log_g, 1)
    beta_prop = -float(coeffs[0])  # negative because G decays
    # R²
    yhat = coeffs[0] * log_l + coeffs[1]
    ss_res = np.sum((log_g - yhat)**2)
    ss_tot = np.sum((log_g - np.mean(log_g))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta_prop, r2


def check_propagator_consistency(data):
    """Propagator decay β_prop + autocorrelation decay γ_flow ≈ 1.

    - β_prop from propagator G(l) power-law decay
    - γ_flow = 2 - 2H from Hurst exponent (fBm autocorrelation decay)
    - Consistency: β_prop + γ_flow ≈ 1 (Bouchaud 2004)

    Note: baselines without Hurst are marked applicable=False (N/A, not FAIL).
    """
    results = {}
    for model in IMPACT_MODELS:
        if model not in data:
            continue
        prop = data[model].get('propagator', {})
        hurst = data[model].get('hurst', np.nan)
        G = prop.get('G', np.array([]))

        # If no Hurst exponent → check is not applicable (replay baselines)
        if not np.isfinite(hurst) or hurst == 0:
            results[model] = dict(
                beta_prop=np.nan, r2_prop=np.nan,
                hurst=np.nan, gamma_flow=np.nan, total=np.nan,
                consistent=False, applicable=False,
            )
            continue

        if len(G) < 10:
            continue

        beta_prop, r2_prop = fit_propagator_decay(G)

        # γ_flow from Hurst: for fractional Brownian motion,
        # C(l) ~ l^{2H-2} → γ_flow = 2 - 2H
        gamma_flow = 2.0 - 2.0 * hurst if np.isfinite(hurst) else np.nan

        # Consistency check: β_prop + γ_flow ≈ 1
        if np.isfinite(beta_prop) and np.isfinite(gamma_flow):
            total = beta_prop + gamma_flow
            consistent = abs(total - 1.0) < 0.3
        else:
            total = np.nan
            consistent = False

        results[model] = dict(
            beta_prop=float(beta_prop) if np.isfinite(beta_prop) else np.nan,
            r2_prop=float(r2_prop) if np.isfinite(r2_prop) else np.nan,
            hurst=float(hurst) if np.isfinite(hurst) else np.nan,
            gamma_flow=float(gamma_flow) if np.isfinite(gamma_flow) else np.nan,
            total=float(total) if np.isfinite(total) else np.nan,
            consistent=consistent,
            applicable=True,
        )
    return results


def fig_propagator_consistency(results, stock, pdf):
    """Figure: propagator decay and consistency relation."""
    models = [m for m in IMPACT_MODELS if m in results]
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: β_prop vs γ_flow with β+γ=1 line
    ax = axes[0]
    for model in models:
        r = results[model]
        if np.isnan(r['beta_prop']) or np.isnan(r['gamma_flow']):
            continue
        color = 'green' if r['consistent'] else 'orange'
        ax.scatter(r['gamma_flow'], r['beta_prop'], color=_c(model), s=80,
                   marker=_mk(model), edgecolors=color, linewidths=2, zorder=5,
                   label=f'{model} (Σ={r["total"]:.2f})')
    # β + γ = 1 line
    g_range = np.linspace(-0.5, 1.5, 100)
    ax.plot(g_range, 1 - g_range, 'r--', lw=2, label='β + γ = 1')
    ax.set(xlabel=r'$\gamma_{flow} = 2 - 2H$',
           ylabel=r'$\beta_{prop}$ (propagator decay)',
           title=f'{stock}: Propagator Consistency')
    ax.legend(fontsize=7, ncol=1)

    # Right: bar chart of total = β_prop + γ_flow
    ax = axes[1]
    models_with = [m for m in models if np.isfinite(results[m]['total'])]
    x = np.arange(len(models_with))
    totals = [results[m]['total'] for m in models_with]
    colors_bar = ['green' if abs(t - 1) < 0.3 else 'orange' for t in totals]
    ax.bar(x, totals, color=colors_bar, alpha=0.7, edgecolor='gray')
    ax.axhline(1.0, ls='--', color='red', lw=2, label='Target: 1.0')
    ax.axhspan(0.7, 1.3, alpha=0.08, color='green', label='±0.3 band')
    for i, (m, v) in enumerate(zip(models_with, totals)):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_with, rotation=45, ha='right', fontsize=8)
    ax.set(ylabel=r'$\beta_{prop} + \gamma_{flow}$',
           title=f'{stock}: β_prop + γ_flow (should ≈ 1)')
    ax.legend(fontsize=9)

    fig.suptitle('Check 5: Propagator Consistency — Bouchaud (2004) Relation',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Check 6: Benchmark Comparison Table
# ═══════════════════════════════════════════════════════════════════════
def build_benchmark_table(data, exec_res, fair_res, dur_res, arb_res, prop_res):
    """Build comprehensive comparison table."""
    rows = []
    for model in IMPACT_MODELS:
        if model not in data:
            continue
        beta_d = data[model].get('beta', {})
        sc = data[model].get('scorecard', {})
        row = dict(
            Model=model,
            Group=_grp(model),
            # Impact exponent
            delta=beta_d.get('beta', np.nan),
            delta_CI=f"[{beta_d.get('ci_lo', np.nan):.3f}, {beta_d.get('ci_hi', np.nan):.3f}]",
            R2=beta_d.get('r2', np.nan),
            N=beta_d.get('n', 0),
        )
        # Check 1: execution profile
        if model in exec_res:
            row['delta_exec'] = exec_res[model]['delta_exec']
            row['exec_concave'] = exec_res[model]['concave']
        else:
            row['delta_exec'] = np.nan
            row['exec_concave'] = False
        # Check 2: fair pricing
        if model in fair_res:
            row['relaxation'] = fair_res[model]['mean']
            row['fair_pass'] = fair_res[model]['pass_loose']
        else:
            row['relaxation'] = np.nan
            row['fair_pass'] = False
        # Check 3: duration independence
        if model in dur_res:
            row['dur_cv'] = dur_res[model]['cv']
            row['dur_pass'] = dur_res[model]['independent']
        else:
            row['dur_cv'] = np.nan
            row['dur_pass'] = False
        # Check 4: no-arb
        if model in arb_res:
            row['gamma'] = arb_res[model]['gamma']
            row['no_arb'] = arb_res[model]['no_arb']
        else:
            row['gamma'] = np.nan
            row['no_arb'] = False
        # Check 5: propagator (N/A for baselines without Hurst)
        if model in prop_res:
            r5 = prop_res[model]
            row['beta_prop'] = r5['beta_prop']
            row['prop_consistent'] = r5['consistent']
            row['prop_applicable'] = r5.get('applicable', True)
        else:
            row['beta_prop'] = np.nan
            row['prop_consistent'] = False
            row['prop_applicable'] = False
        # Total checks passed (out of applicable checks)
        applicable = [True, True, True, True, row.get('prop_applicable', False)]
        passed = [
            row.get('exec_concave', False),
            row.get('fair_pass', False),
            row.get('dur_pass', False),
            row.get('no_arb', False),
            row.get('prop_consistent', False),
        ]
        n_applicable = sum(applicable)
        n_passed = sum(p for p, a in zip(passed, applicable) if a)
        row['checks_passed'] = n_passed
        row['checks_applicable'] = n_applicable
        rows.append(row)
    return pd.DataFrame(rows)


def fig_benchmark_table(df, stock, pdf):
    """Figure: comprehensive benchmark comparison table."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    ax.text(0.5, 0.98, f'Check 6: Benchmark Comparison — {stock}',
            fontsize=14, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    # Build table data
    cell_data = []
    for _, row in df.iterrows():
        def pass_str(v, applicable=True):
            if not applicable:
                return 'N/A'
            return 'PASS' if v else 'FAIL'
        n_app = row.get('checks_applicable', 5)
        cell_data.append([
            row['Model'], row['Group'],
            f"{row['delta']:.3f}" if np.isfinite(row['delta']) else '—',
            f"{row['delta_exec']:.2f}" if np.isfinite(row['delta_exec']) else '—',
            pass_str(row.get('exec_concave', False)),
            f"{row['relaxation']:.3f}" if np.isfinite(row['relaxation']) else '—',
            pass_str(row.get('fair_pass', False)),
            f"{row['dur_cv']:.2f}" if np.isfinite(row['dur_cv']) else '—',
            pass_str(row.get('dur_pass', False)),
            f"{row['gamma']:.3f}" if np.isfinite(row['gamma']) else '—',
            pass_str(row.get('no_arb', False)),
            pass_str(row.get('prop_consistent', False), row.get('prop_applicable', False)),
            f"{row.get('checks_passed', 0)}/{n_app}",
        ])

    col_labels = ['Model', 'Group', 'δ(β)', 'δ_exec', 'C1', 'Y', 'C2', 'CV(β)', 'C3', 'γ', 'C4', 'C5', 'Score']
    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellLoc='center', loc='upper center', bbox=[0.01, 0.02, 0.98, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Color header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=7)
        elif row % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        # Color PASS/FAIL/N/A cells
        text = cell.get_text().get_text()
        if text == 'PASS':
            cell.set_facecolor('#d5f5e3')
        elif text == 'FAIL':
            cell.set_facecolor('#fadbd8')
        elif text == 'N/A':
            cell.set_facecolor('#eee')

    # Empirical targets row
    ax.text(0.5, 0.01,
            r'Empirical targets: δ ≈ 0.5, δ_exec < 1, Y ≈ 0.667, CV(β) < 0.25, γ ≤ 1−δ',
            fontsize=9, ha='center', va='bottom', transform=ax.transAxes, style='italic')

    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Summary scorecard figure
# ═══════════════════════════════════════════════════════════════════════
def fig_summary_heatmap(all_tables, pdf):
    """Heatmap: pass/fail across models × checks × stocks."""
    check_names = ['C1: Concavity', 'C2: Fair Pricing', 'C3: Duration Indep.',
                   'C4: No-Arb', 'C5: Propagator']
    check_cols = ['exec_concave', 'fair_pass', 'dur_pass', 'no_arb', 'prop_consistent']
    check_applicable = [None, None, None, None, 'prop_applicable']

    stocks = list(all_tables.keys())
    if not stocks:
        return

    n_stocks = len(stocks)
    fig, axes = plt.subplots(1, n_stocks, figsize=(8 * n_stocks, 5), squeeze=False)

    for si, stock in enumerate(stocks):
        ax = axes[0][si]
        df = all_tables[stock]
        models = df['Model'].values
        # 0=FAIL, 0.5=N/A, 1=PASS
        matrix = np.zeros((len(models), len(check_cols)))
        for ci, col in enumerate(check_cols):
            app_col = check_applicable[ci]
            for mi, _ in enumerate(models):
                applicable = df.iloc[mi].get(app_col, True) if app_col else True
                if not applicable:
                    matrix[mi, ci] = 0.5  # N/A
                elif df.iloc[mi].get(col, False):
                    matrix[mi, ci] = 1.0  # PASS
                else:
                    matrix[mi, ci] = 0.0  # FAIL

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#fadbd8', '#eeeeee', '#d5f5e3'])
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(check_names)))
        ax.set_xticklabels(check_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(stock, fontsize=12, fontweight='bold')

        # Annotate cells
        for mi in range(len(models)):
            for ci in range(len(check_cols)):
                if matrix[mi, ci] > 0.7:
                    txt = 'P'
                elif matrix[mi, ci] > 0.3:
                    txt = '—'
                else:
                    txt = 'F'
                ax.text(ci, mi, txt, ha='center', va='center', fontsize=9,
                        fontweight='bold', color='black')

    fig.suptitle('Lifecycle Validation Summary — Pass/Fail Heatmap',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig, dpi=DPI); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Text pages
# ═══════════════════════════════════════════════════════════════════════
def text_page(pdf, title, body, fontsize=11):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top')
    ax.text(0.05, 0.88, body, transform=ax.transAxes, fontsize=fontsize,
            va='top', linespacing=1.5, family='monospace')
    fig.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close(fig)


def title_page(pdf, stocks, total_models):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.65,
            'Market Impact Lifecycle Validation\nFramework v2',
            transform=ax.transAxes, fontsize=26, fontweight='bold',
            ha='center', linespacing=1.4)
    ax.text(0.5, 0.42,
            '6 empirical checks on existing v4 grid data\n\n'
            f'Stocks: {", ".join(stocks)}\n'
            f'Models: {total_models} (Null + Baselines + Parametric + S5 Neural)\n\n'
            'C1: Execution Profile (concavity)\n'
            'C2: Fair Pricing (2/3 rule)\n'
            'C3: Duration Independence\n'
            'C4: No-Arbitrage Conditions\n'
            'C5: Propagator Consistency\n'
            'C6: Benchmark Comparison Table',
            transform=ax.transAxes, fontsize=13, ha='center', linespacing=1.5)
    pdf.savefig(fig, dpi=150); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def run_stock(stock, data, pdf):
    """Run all 6 checks for one stock, adding pages to PDF."""
    print(f'\n=== {stock}: Running 6 checks ===')

    # Check 1
    print('  Check 1: Execution Profile...')
    exec_res = check_execution_profile(data)
    for m, r in exec_res.items():
        status = 'PASS' if r['concave'] else 'FAIL'
        print(f'    {m:15s} delta_exec={r["delta_exec"]:.3f} R2={r["r2"]:.3f} [{status}]')
    fig_execution_profile(exec_res, stock, pdf)

    # Check 2
    print('  Check 2: Fair Pricing...')
    fair_res = check_fair_pricing(data)
    for m, r in fair_res.items():
        status = 'PASS' if r['pass_loose'] else 'FAIL'
        print(f'    {m:15s} relax={r["mean"]:.3f} +/- {r["std"]:.3f} dist={r["distance"]:.3f} [{status}]')
    fig_fair_pricing(fair_res, stock, pdf)

    # Check 3
    print('  Check 3: Duration Independence...')
    dur_res = check_duration_independence(data)
    for m, r in dur_res.items():
        status = 'PASS' if r['independent'] else 'FAIL'
        print(f'    {m:15s} CV={r["cv"]:.3f} range={r["range"]:.3f} [{status}]')
    fig_duration_independence(dur_res, stock, pdf)

    # Check 4
    print('  Check 4: No-Arbitrage...')
    arb_res = check_no_arbitrage(data)
    for m, r in arb_res.items():
        status = 'PASS' if r['no_arb'] else 'FAIL'
        print(f'    {m:15s} gamma={r["gamma"]:.3f} delta={r["delta"]:.3f} '
              f'margin={r["margin"]:.3f} [{status}]')
    fig_no_arbitrage(arb_res, stock, pdf)

    # Check 5
    print('  Check 5: Propagator Consistency...')
    prop_res = check_propagator_consistency(data)
    for m, r in prop_res.items():
        if not r.get('applicable', True):
            status = 'N/A'
        elif r['consistent']:
            status = 'PASS'
        else:
            status = 'FAIL'
        h_str = f'H={r["hurst"]:.3f}' if np.isfinite(r['hurst']) else 'H=—'
        bp_str = f'beta_prop={r["beta_prop"]:.3f}' if np.isfinite(r['beta_prop']) else 'beta_prop=—'
        t_str = f'total={r["total"]:.3f}' if np.isfinite(r['total']) else 'total=—'
        print(f'    {m:15s} {h_str} {bp_str} {t_str} [{status}]')
    fig_propagator_consistency(prop_res, stock, pdf)

    # Check 6
    print('  Check 6: Benchmark Table...')
    table_df = build_benchmark_table(data, exec_res, fair_res, dur_res, arb_res, prop_res)
    fig_benchmark_table(table_df, stock, pdf)

    return table_df


def main():
    parser = argparse.ArgumentParser(description='Lifecycle Validation (6 checks)')
    parser.add_argument('--stock', type=str, nargs='+', default=['GOOG', 'INTC'],
                        help='Stock(s) to analyze')
    parser.add_argument('--src', type=str, default=None,
                        help='Override source directory pattern (default: pics_for_v4_300_{STOCK})')
    parser.add_argument('--out', type=str, default='pics_for_lifecycle',
                        help='Output directory')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # Load all stocks
    all_data = {}
    for stock in args.stock:
        src_dir = args.src.format(STOCK=stock) if args.src else None
        data = load_metrics(stock, src_dir)
        if data:
            all_data[stock] = data

    if not all_data:
        print('ERROR: No data loaded.')
        sys.exit(1)

    total_models = max(len(d) for d in all_data.values())

    # Generate PDF
    pdf_path = out / 'lifecycle_report.pdf'
    all_tables = {}

    with PdfPages(pdf_path) as pdf:
        title_page(pdf, list(all_data.keys()), total_models)

        for stock, data in all_data.items():
            # Section header
            text_page(pdf, f'Stock: {stock}',
                      f'Models loaded: {len(data)}\n'
                      f'Impact models: {len([m for m in IMPACT_MODELS if m in data])}\n\n'
                      'Running 6 lifecycle checks...')

            table_df = run_stock(stock, data, pdf)
            all_tables[stock] = table_df

        # Summary heatmap across stocks
        fig_summary_heatmap(all_tables, pdf)

        # Final summary text
        summary_lines = []
        for stock, df in all_tables.items():
            summary_lines.append(f'\n{stock}:')
            for _, row in df.iterrows():
                n_app = row.get('checks_applicable', 5)
                summary_lines.append(f'  {row["Model"]:15s}  {row["checks_passed"]}/{n_app} checks passed')
        text_page(pdf, 'Summary',
                  'Lifecycle Validation Results\n' + '\n'.join(summary_lines))

    print(f'\nPDF: {pdf_path}')

    # Save CSV
    for stock, df in all_tables.items():
        csv_path = out / f'lifecycle_{stock}.csv'
        df.to_csv(csv_path, index=False)
        print(f'CSV: {csv_path}')

    # Combined CSV
    if len(all_tables) > 1:
        combined = pd.concat([df.assign(Stock=s) for s, df in all_tables.items()])
        combined_path = out / 'lifecycle_combined.csv'
        combined.to_csv(combined_path, index=False)
        print(f'CSV: {combined_path}')

    # Print overall pass rates
    print('\n=== Overall Pass Rates ===')
    for stock, df in all_tables.items():
        total_applicable = df['checks_applicable'].sum()
        passed = df['checks_passed'].sum()
        print(f'  {stock}: {passed}/{total_applicable} ({100*passed/total_applicable:.0f}%)')


if __name__ == '__main__':
    main()
