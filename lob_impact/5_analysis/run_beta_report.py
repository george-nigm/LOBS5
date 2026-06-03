#!/usr/bin/env python3
"""
Beta Report: Complete analysis for paper section "Market Impact Scaling".

Generates ~15 figures + ~5 LaTeX tables examining why β ≠ 0.5:
  §1 Estimator bias (origin vs intercept, synthetic proof)
  §2 Volume calibration (depth-at-best analysis)
  §3 Impact scaling dynamics (β vs k, mb, daily liquidity)
  §4 Alternative normalizations
  §5 Impact definitions
  §6 Model differentiation (Kyle λ, incremental β)
  §7 Cross-stock comparison
  §8 New analyses (cross-sectional β, penetration split)

Usage:
    python lob_impact/run_beta_report.py --stock GOOG
    sbatch lob_impact/run_beta_report_slurm.sh GOOG
"""
import argparse, pickle, re, textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
DPI = 200
TICK_SIZE = 100
PICKLE_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v4/pickles')

MODEL_META = OrderedDict([
    ('Historic',   dict(color='#90939C', ls=':',  marker='x', ms=6)),
    ('Heuristic',  dict(color='#546884', ls='-.', marker='d', ms=5)),
    ('CST',        dict(color='#213552', ls='--', marker='^', ms=5)),
    ('CGAN',       dict(color='#7B4F9E', ls=(0,(8,4)), marker='s', ms=5)),
    ('LobS5',      dict(color='#C88A3A', ls='-',  marker='o', ms=5)),
    ('S5-120M',    dict(color='#D95F02', ls='-',  marker='v', ms=5)),
    ('S5-4K',      dict(color='#5B7BBF', ls='-',  marker='*', ms=7)),
    ('S5-360M',    dict(color='#B5446E', ls='-',  marker='H', ms=6)),
    ('LobS5-v2',   dict(color='#2CA02C', ls='-',  marker='P', ms=6)),
])

def _c(m): return MODEL_META.get(m, {}).get('color', '#888888')
def _ls(m): return MODEL_META.get(m, {}).get('ls', '-')
def _mk(m): return MODEL_META.get(m, {}).get('marker', 'o')
def _ms(m): return MODEL_META.get(m, {}).get('ms', 5)

OUT = None  # set in main()

def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
        'axes.labelsize': 12, 'legend.fontsize': 9,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    })
    return plt

def save(plt, fig, name, w=10.8, h=5.5):
    fig.set_size_inches(w, h); fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [{name}]')

def save_tex(name, content):
    with open(OUT / f'{name}.tex', 'w') as f:
        f.write(content)

# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════
def load_all_data(stock):
    D = {}
    # Metrics pickles
    src = Path(f'pics_for_v4_300_{stock}')
    D['metrics'] = {}
    D['models'] = []
    for pkl in sorted(src.glob('*.metrics.pkl')):
        with open(pkl, 'rb') as f:
            mc = pickle.load(f)
        model = mc.get('model', pkl.stem.replace('.metrics', ''))
        if model == 'ZeroInsertions': continue
        D['metrics'][model] = mc
        D['models'].append(model)

    # Summary CSV
    csv_path = src / 'summary_statistics.csv'
    D['summary'] = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    # Synthetic calibration
    syn_path = Path('pics_for_investigation/diag_1_synthetic.csv')
    D['synthetic'] = pd.read_csv(syn_path) if syn_path.exists() else pd.DataFrame()

    # Depth stats
    for s in ['GOOG', 'INTC']:
        p = Path(f'lob_impact/depth_stats_{s}.csv')
        if p.exists():
            D[f'depth_{s}'] = pd.read_csv(p)

    # Daily H/L
    hl = Path(f'lob_impact/daily_h_l_{stock}.csv')
    D['daily_hl'] = pd.read_csv(hl) if hl.exists() else pd.DataFrame()

    # Other stock summary (for cross-stock)
    other = 'INTC' if stock == 'GOOG' else 'GOOG'
    other_csv = Path(f'pics_for_v4_300_{other}/summary_statistics.csv')
    D['summary_other'] = pd.read_csv(other_csv) if other_csv.exists() else pd.DataFrame()
    D['other_stock'] = other

    # Diag texts (parse later as needed)
    for d in [4, 5, 7]:
        p = Path(f'pics_for_investigation/diag_{d}_*.txt')
        matches = list(Path('pics_for_investigation').glob(f'diag_{d}_*{stock}*.txt'))
        if matches:
            D[f'diag_{d}'] = matches[0].read_text()

    return D


# ═══════════════════════════════════════════════════════════════════════
# §1: Estimator Analysis
# ═══════════════════════════════════════════════════════════════════════
def section_1_estimators(plt, D, stock):
    print('§1 Estimator Analysis')

    # Fig R1: Synthetic Calibration Heatmap
    syn = D.get('synthetic', pd.DataFrame())
    if not syn.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        alphas = sorted(syn['alpha'].unique())
        betas = sorted(syn['beta_true'].unique())
        for ax, col, title, cmap in [
            (ax1, 'bias_origin', r'Origin Estimator Bias ($\beta_{origin} - \beta_{true}$)', 'Reds'),
            (ax2, 'bias_intercept', r'Intercept Estimator Bias ($\beta_{intercept} - \beta_{true}$)', 'RdYlGn_r'),
        ]:
            matrix = np.full((len(alphas), len(betas)), np.nan)
            for _, r in syn.iterrows():
                ai = alphas.index(r['alpha'])
                bi = betas.index(r['beta_true'])
                matrix[ai, bi] = r[col]
            im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                          vmin=-0.1 if 'intercept' in col else 0,
                          vmax=0.1 if 'intercept' in col else 0.7)
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels([f'{b:.1f}' for b in betas])
            ax.set_yticks(range(len(alphas)))
            ax.set_yticklabels([f'{a:.0f}' for a in alphas])
            ax.set_xlabel(r'$\beta_{true}$'); ax.set_ylabel(r'$\alpha$')
            ax.set_title(title, fontsize=11)
            for i in range(len(alphas)):
                for j in range(len(betas)):
                    ax.text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center', fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle('R1: Synthetic Calibration — Estimator Bias', fontsize=14, fontweight='bold')
        save(plt, fig, 'R01_synthetic_calibration', w=14, h=5.5)

    # Fig R2: Point Cloud + Regression Lines
    # Pick LobS5 or first available model
    target = 'LobS5' if 'LobS5' in D['metrics'] else D['models'][0]
    mc = D['metrics'][target]
    br = mc.get('beta', {})
    pc_x = br.get('pc_x', np.array([]))
    pc_y = br.get('pc_y', np.array([]))
    pc_y_raw = br.get('pc_y_raw', np.array([]))
    if len(pc_x) > 100:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        rng = np.random.default_rng(42)
        n_show = min(2000, len(pc_x))
        idx = rng.choice(len(pc_x), n_show, replace=False)
        xl = np.array([pc_x.min(), pc_x.max()])

        # Left: log(I/σ) space — origin estimator
        ax1.scatter(pc_x[idx], pc_y[idx], s=2, alpha=0.15, color='steelblue', rasterized=True)
        bo = br.get('beta_origin', 0.84)
        bi = br.get('beta', 0.33)
        # Intercept of y_adj vs x
        ok = np.isfinite(pc_x) & np.isfinite(pc_y)
        if ok.sum() > 2:
            c_adj = np.polyfit(pc_x[ok], pc_y[ok], 1)
            ax1.plot(xl, c_adj[0]*xl + c_adj[1], color='green', ls='--', lw=2.5,
                     label=f'Intercept: β={c_adj[0]:.3f}')
        ax1.plot(xl, bo*xl, color='red', lw=2.5, label=f'Origin: β={bo:.3f}')
        ax1.plot(xl, 0.5*xl, 'k:', lw=1.5, label='Theory: β=0.5')
        ax1.scatter(0, 0, s=250, marker='*', color='gold', edgecolors='black', zorder=10)
        ax1.annotate('Origin (0,0)', (0, 0), fontsize=9, xytext=(10, 10),
                    textcoords='offset points', fontweight='bold')
        ax1.set(xlabel='x = log(Q/V)', ylabel='y = log(I/σ)',
                title=f'{target}: Origin Estimator Space')
        ax1.legend(fontsize=9)

        # Right: log(I) space — intercept estimator
        ax2.scatter(pc_x[idx], pc_y_raw[idx], s=2, alpha=0.15, color='coral', rasterized=True)
        alpha_val = br.get('alpha', -9)
        ax2.plot(xl, bi*xl + alpha_val, color='green', ls='--', lw=2.5,
                 label=f'Intercept: β={bi:.3f}, α={alpha_val:.1f}')
        ax2.plot(xl, 0.5*xl + (0.5*np.mean(pc_x[ok]) + np.mean(pc_y_raw[ok]) - 0.5*np.mean(pc_x[ok])),
                 'k:', lw=1.5, label='β=0.5 (shifted)')
        ax2.set(xlabel='x = log(Q/V)', ylabel='y = log(I)',
                title=f'{target}: Intercept Estimator Space')
        ax2.legend(fontsize=9)

        fig.suptitle('R2: Regression Lines — Origin vs Intercept', fontsize=14, fontweight='bold')
        save(plt, fig, 'R02_point_cloud_regression', w=16, h=6)

    # Table R1
    if not D['summary'].empty:
        df = D['summary']
        rows_tex = []
        for _, r in df.iterrows():
            rows_tex.append(
                f"  {r['Model']:<12} & {r['beta_origin']:.3f} & {r['beta_intercept']:.3f} & "
                f"{r['beta_ratio']:.3f} & [{r['CI_lo']:.3f}, {r['CI_hi']:.3f}] & "
                f"{r['R2']:.3f} & {int(r['N']):,} & {r.get('alpha_incremental', np.nan):.1f} \\\\"
            )
        tex = textwrap.dedent(f"""\
        \\begin{{tabular}}{{lccccccc}}
        \\toprule
        Model & $\\beta_{{origin}}$ & $\\beta_{{intercept}}$ & $\\beta_{{ratio}}$ & 95\\% CI & $R^2$ & $N$ & $\\alpha$ \\\\
        \\midrule
        {chr(10).join(rows_tex)}
        \\bottomrule
        \\end{{tabular}}""")
        save_tex('R01_table_estimators', tex)


# ═══════════════════════════════════════════════════════════════════════
# §2: Volume Calibration
# ═══════════════════════════════════════════════════════════════════════
def section_2_volume_calibration(plt, D, stock):
    print('§2 Volume Calibration')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, s in zip(axes, ['GOOG', 'INTC']):
        key = f'depth_{s}'
        if key not in D: continue
        depth = D[key]
        vals = depth['depth_relevant'].dropna().values
        vals = vals[vals > 0]
        if len(vals) == 0: continue

        ax.hist(vals, bins=100, color='steelblue', edgecolor='white', lw=0.3, alpha=0.7)
        # Old volumes
        for v, ls, c in [(75, ':', 'red'), (300, '--', 'red'), (485, '-.', 'red')]:
            pct = int(np.searchsorted(np.sort(vals), v) * 100 // len(vals))
            ax.axvline(v, ls=ls, color=c, lw=1.5, alpha=0.7, label=f'old v={v} (p{pct})')
        # New volumes
        new_vols = [105, 165, 325] if s == 'GOOG' else [590, 1110, 3120]
        for v in new_vols:
            pct = int(np.searchsorted(np.sort(vals), v) * 100 // len(vals))
            ax.axvline(v, ls='-', color='green', lw=1.5, alpha=0.7, label=f'new v={v} (p{pct})')
        ax.set(title=f'{s}: Depth at Best (n={len(vals):,})',
               xlabel='Volume (shares)', ylabel='Count')
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlim(0, np.percentile(vals, 99.5))

    fig.suptitle('R3: Depth at Best — Volume Calibration', fontsize=14, fontweight='bold')
    save(plt, fig, 'R03_depth_calibration', w=14, h=5)


# ═══════════════════════════════════════════════════════════════════════
# §3: Impact Scaling Dynamics
# ═══════════════════════════════════════════════════════════════════════
def section_3_scaling_dynamics(plt, D, stock):
    print('§3 Scaling Dynamics')

    # Load raw pickle for per-k analysis (Historic only for speed)
    raw_pkl = PICKLE_BASE / stock / 'Historic.pkl'
    pc_k_data = None
    if raw_pkl.exists():
        print('  Loading raw pickle for per-k analysis...')
        with open(raw_pkl, 'rb') as f:
            md = pickle.load(f)
        # Quick point cloud extraction
        from lob_impact.analysis.run_300_analyze_one import (
            extract_point_cloud, filter_model, load_daily_params, collect_days
        )
        daily_params = None
        hl_path = f'lob_impact/daily_h_l_{stock}.csv'
        if Path(hl_path).exists():
            exp_days = collect_days(md)
            daily_params = load_daily_params(hl_path, exp_days)
        filtered, _, _ = filter_model(md)
        pc_k_data = extract_point_cloud(filtered, md['grid'], daily_params)
        del md, filtered

    # Fig R4: β vs k
    if pc_k_data is not None and not pc_k_data.empty and 'k' in pc_k_data.columns:
        from lob_impact.analysis.run_300_analyze_one import compute_global_beta

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Per-k individual
        k_vals = sorted(pc_k_data['k'].unique())
        k_betas_ind, k_ns = [], []
        for k in k_vals:
            sub = pc_k_data[pc_k_data['k'] == k]
            if len(sub) < 10: continue
            res = compute_global_beta(sub)
            k_betas_ind.append((k, res['beta'], res['r2'], len(sub)))
        if k_betas_ind:
            ks, bs, r2s, ns = zip(*k_betas_ind)
            ax1.plot(ks, bs, 'o-', color='steelblue', lw=2, ms=6)
            ax1.axhline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
            ax1.set(title=r'β at Individual k', xlabel='Insertion k',
                    ylabel=r'$\beta_{intercept}$')
            ax1.legend()

        # Cumulative k≥threshold
        thresholds = [1, 2, 3, 5, 7, 10]
        cum_betas = []
        for t in thresholds:
            sub = pc_k_data[pc_k_data['k'] >= t]
            if len(sub) < 10: continue
            res = compute_global_beta(sub)
            cum_betas.append((t, res['beta'], res['r2'], len(sub)))
        if cum_betas:
            ts, bs, r2s, ns = zip(*cum_betas)
            ax2.plot(ts, bs, 's-', color='coral', lw=2, ms=6)
            ax2.axhline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
            ax2.set(title=r'β Cumulative (k ≥ threshold)', xlabel='k threshold',
                    ylabel=r'$\beta_{intercept}$')
            ax2.legend()

        fig.suptitle('R4: Impact Saturation — β Decreases with k', fontsize=14, fontweight='bold')
        save(plt, fig, 'R04_beta_vs_k', w=14, h=5.5)

    # Fig R5: β vs mb (from metrics pickles)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in D['models']:
        mc = D['metrics'].get(model, {})
        mb_b = mc.get('mb_betas', {})
        if not mb_b: continue
        mbs = sorted(mb_b.keys())
        ax.plot(mbs, [mb_b[m] for m in mbs], color=_c(model), ls=_ls(model),
                lw=2, marker=_mk(model), ms=_ms(model), label=model)
    ax.axhline(0.5, ls='--', color='red', lw=1.5, label='β=0.5')
    ax.set(title=r'R5: $\beta$ vs Cooling Messages (mb)', xlabel='mb (messages between insertions)',
           ylabel=r'$\beta_{intercept}$')
    ax.legend(fontsize=8, ncol=2)
    save(plt, fig, 'R05_beta_vs_mb', w=10, h=5.5)

    # Fig R7: Per-day β boxplot
    bdata, blabels = [], []
    for model in D['models']:
        mc = D['metrics'].get(model, {})
        db = mc.get('day_betas', {})
        if db:
            bdata.append(list(db.values()))
            blabels.append(model)
    if bdata:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        bp = ax.boxplot(bdata, patch_artist=True, widths=0.55,
                        flierprops=dict(markersize=3, alpha=0.4))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(_c(blabels[i])); patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(blabels)+1))
        ax.set_xticklabels(blabels, rotation=45, ha='right', fontsize=9)
        ax.axhline(0.5, ls='--', color='red', lw=2, label='β=0.5')
        ax.axhline(0.33, ls=':', color='blue', lw=1.5, label='β=0.33 (aggregate)')
        ax.set(title=r'R7: Per-Day $\beta$ Distribution', ylabel=r'$\beta_{intercept}$')
        ax.legend(fontsize=9)
        save(plt, fig, 'R07_beta_per_day_boxplot', w=11, h=5.5)

    # Fig R6: β vs daily depth (scatter)
    if pc_k_data is not None and 'day' in pc_k_data.columns and 'depth_at_best' in pc_k_data.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        for model in D['models']:
            mc = D['metrics'].get(model, {})
            db = mc.get('day_betas', {})
            ds = mc.get('depth_stats', {})
            if not db: continue
            # Get per-day depth from the raw pickle (Historic only for now)
            # Use aggregate depth_p50 as proxy
            p50 = ds.get('p50', np.nan)
            if np.isnan(p50): continue
            for day, beta_val in db.items():
                ax.scatter(p50, beta_val, color=_c(model), s=20, alpha=0.5,
                          marker=_mk(model))
        # Add model labels via legend
        for model in D['models']:
            ax.scatter([], [], color=_c(model), marker=_mk(model), s=40, label=model)
        ax.axhline(0.5, ls='--', color='red', lw=1.5)
        ax.set(title=r'R6: $\beta$ vs Book Depth', xlabel='Depth at Best (p50, shares)',
               ylabel=r'Per-day $\beta$')
        ax.legend(fontsize=7, ncol=2)
        save(plt, fig, 'R06_beta_vs_depth', w=9, h=6)


# ═══════════════════════════════════════════════════════════════════════
# §4: Alternative Normalizations
# ═══════════════════════════════════════════════════════════════════════
def section_4_normalizations(plt, D, stock):
    print('§4 Normalizations')

    # Parse diag_4 output
    diag4 = D.get('diag_4', '')
    if not diag4: return

    rows = []
    for line in diag4.split('\n'):
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('Norm') or line.startswith('Close'): continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                rows.append(dict(norm=parts[0], model=parts[1],
                                beta_int=float(parts[2]), beta_orig=float(parts[3]),
                                r2=float(parts[4]), n=int(parts[5])))
            except (ValueError, IndexError): pass

    if not rows: return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    norms = df['norm'].unique()
    models_show = [m for m in ['Historic', 'LobS5', 'CST'] if m in df['model'].values]
    n_norms = len(norms)
    n_models = len(models_show)
    w = 0.8 / n_models

    for mi, model in enumerate(models_show):
        sub = df[df['model'] == model]
        x_pos = np.arange(n_norms)
        vals = [sub[sub['norm'] == n]['beta_int'].values[0] if len(sub[sub['norm'] == n]) > 0 else np.nan
                for n in norms]
        ax.bar(x_pos + mi * w - (n_models-1)*w/2, vals, w * 0.9,
               label=model, color=_c(model), edgecolor='gray', linewidth=0.5, alpha=0.8)

    ax.set_xticks(np.arange(n_norms))
    ax.set_xticklabels(norms, fontsize=10)
    ax.axhline(0.5, ls='--', color='red', lw=2, label='β=0.5')
    ax.set(title=r'R8: $\beta$ Under Different Volume Normalizations',
           ylabel=r'$\beta_{intercept}$')
    ax.legend(fontsize=9, ncol=2)
    save(plt, fig, 'R08_normalizations', w=12, h=5.5)


# ═══════════════════════════════════════════════════════════════════════
# §5: Impact Definitions
# ═══════════════════════════════════════════════════════════════════════
def section_5_impact_definitions(plt, D, stock):
    print('§5 Impact Definitions')

    diag5 = D.get('diag_5', '')
    if not diag5: return

    rows = []
    for line in diag5.split('\n'):
        line = line.strip()
        parts = line.split()
        if len(parts) >= 7 and parts[0].startswith('I_'):
            try:
                rows.append(dict(impact=parts[0], scope=parts[1],
                                model=parts[2], beta=float(parts[3]),
                                r2=float(parts[4]), n=int(parts[5])))
            except (ValueError, IndexError): pass

    if not rows: return
    df = pd.DataFrame(rows)
    impacts = ['I_vwap', 'I_inst', 'I_mid', 'I_perm', 'I_terminal']
    scopes = df['scope'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, scope in zip(axes, scopes):
        sub = df[df['scope'] == scope]
        models_show = sub['model'].unique()
        n_imp = len(impacts)
        n_mod = len(models_show)
        w = 0.8 / max(n_mod, 1)
        for mi, model in enumerate(models_show):
            ms = sub[sub['model'] == model]
            vals = [ms[ms['impact'] == imp]['beta'].values[0] if len(ms[ms['impact'] == imp]) > 0 else np.nan
                    for imp in impacts]
            x_pos = np.arange(n_imp)
            ax.bar(x_pos + mi*w - (n_mod-1)*w/2, vals, w*0.9,
                   label=model, color=_c(model), edgecolor='gray', linewidth=0.5, alpha=0.8)
        ax.set_xticks(np.arange(n_imp))
        ax.set_xticklabels([i.replace('I_', '') for i in impacts], fontsize=10)
        ax.axhline(0.5, ls='--', color='red', lw=1.5)
        ax.set(title=scope, ylabel=r'$\beta$')
        ax.legend(fontsize=8)

    fig.suptitle(r'R9: $\beta$ by Impact Definition', fontsize=14, fontweight='bold')
    save(plt, fig, 'R09_impact_definitions', w=16, h=5.5)


# ═══════════════════════════════════════════════════════════════════════
# §6: Model Differentiation
# ═══════════════════════════════════════════════════════════════════════
def section_6_model_differentiation(plt, D, stock):
    print('§6 Model Differentiation')

    # Fig R10: Kyle λ (from metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    has_data = False
    for model in D['models']:
        kl = D['metrics'].get(model, {}).get('kyle_lambda', {})
        if not kl: continue
        has_data = True
        ks = sorted(kl.keys())
        means = [kl[k]['mean'] for k in ks]
        stds = [kl[k]['std'] for k in ks]
        medians = [kl[k]['median'] for k in ks]
        ax1.errorbar(ks, means, yerr=stds, color=_c(model), ls=_ls(model),
                     lw=2, marker=_mk(model), ms=_ms(model), label=model, capsize=3, alpha=0.8)
        ax2.plot(ks, medians, color=_c(model), ls=_ls(model),
                 lw=2, marker=_mk(model), ms=_ms(model), label=model)
    if has_data:
        for ax, t in [(ax1, r'Mean $\lambda_k$ ($\pm$ std)'), (ax2, r'Median $\lambda_k$')]:
            ax.set(xlabel='Insertion k', ylabel=r'$\lambda_k$ (ticks/share)', title=t)
            ax.legend(fontsize=7, ncol=2)
        fig.suptitle(r'R10: Kyle $\lambda$ — Price Impact per Share per Insertion',
                     fontsize=14, fontweight='bold')
        save(plt, fig, 'R10_kyle_lambda', w=16, h=5.5)
    else:
        plt.close(fig)

    # Fig R12: β_incremental bar chart
    if not D['summary'].empty:
        df = D['summary']
        models = df['Model'].values
        beta_inc = df['beta_incremental'].values if 'beta_incremental' in df.columns else []
        if len(beta_inc) > 0:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            x_pos = np.arange(len(models))
            ax.bar(x_pos, beta_inc, color=[_c(m) for m in models],
                   width=0.6, edgecolor='gray', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.axhline(0, ls='-', color='gray', lw=0.5)
            ax.set(title=r'R12: $\beta_{incremental}$ (per-insertion, k$\geq$3)',
                   ylabel=r'$\beta$')
            save(plt, fig, 'R12_beta_incremental', w=10, h=5.5)


# ═══════════════════════════════════════════════════════════════════════
# §7: Cross-Stock
# ═══════════════════════════════════════════════════════════════════════
def section_7_cross_stock(plt, D, stock):
    print('§7 Cross-Stock')

    df1 = D['summary']
    df2 = D['summary_other']
    if df1.empty or df2.empty: return

    other = D['other_stock']
    fig, ax = plt.subplots(figsize=(12, 5.5))
    merged = df1.merge(df2, on='Model', suffixes=(f'_{stock}', f'_{other}'), how='inner')
    if merged.empty: return

    models = merged['Model'].values
    n = len(models)
    x_pos = np.arange(n)
    w = 0.35
    b1 = merged[f'beta_intercept_{stock}'].values
    b2 = merged[f'beta_intercept_{other}'].values

    ax.bar(x_pos - w/2, b1, w*0.9, label=stock, color='steelblue', edgecolor='gray')
    ax.bar(x_pos + w/2, b2, w*0.9, label=other, color='coral', edgecolor='gray')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.axhline(0.5, ls='--', color='red', lw=2, label='β=0.5')
    ax.set(title=f'R13: Cross-Stock β Comparison ({stock} vs {other})',
           ylabel=r'$\beta_{intercept}$')
    ax.legend(fontsize=10)
    save(plt, fig, 'R13_cross_stock', w=12, h=5.5)


# ═══════════════════════════════════════════════════════════════════════
# §8: New Analyses (cross-sectional β, penetration split)
# ═══════════════════════════════════════════════════════════════════════
def section_8_new_analyses(plt, D, stock):
    print('§8 New Analyses')

    raw_pkl = PICKLE_BASE / stock / 'Historic.pkl'
    if not raw_pkl.exists():
        print('  SKIP: raw pickle not found')
        return

    with open(raw_pkl, 'rb') as f:
        md = pickle.load(f)

    from lob_impact.analysis.run_300_analyze_one import (
        extract_point_cloud, filter_model, load_daily_params, collect_days, compute_global_beta
    )
    daily_params = None
    hl_path = f'lob_impact/daily_h_l_{stock}.csv'
    if Path(hl_path).exists():
        exp_days = collect_days(md)
        daily_params = load_daily_params(hl_path, exp_days)
    filtered, _, _ = filter_model(md)
    pc = extract_point_cloud(filtered, md['grid'], daily_params)
    del md, filtered

    if pc.empty: return

    # Fig R14: Cross-sectional β at fixed k
    # For each k, use different volume configs as Q variation
    k_vals = sorted(pc['k'].unique())
    cross_betas = []
    for k in k_vals:
        sub = pc[pc['k'] == k]
        if len(sub) < 20: continue
        vols = sub['vol'].unique()
        if len(vols) < 2: continue
        res = compute_global_beta(sub)
        cross_betas.append((k, res['beta'], res['r2'], len(sub), len(vols)))

    if cross_betas:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ks, bs, r2s, ns, nvs = zip(*cross_betas)
        ax.plot(ks, bs, 'o-', color='steelblue', lw=2, ms=6)
        ax.axhline(0.5, ls='--', color='red', lw=2, label='β=0.5')
        ax.axhline(0.33, ls=':', color='blue', lw=1.5, label='β=0.33 (aggregate)')
        ax.set(title=r'R14: Cross-Sectional $\beta$ at Fixed k (Historic)',
               xlabel='Insertion k', ylabel=r'$\beta_{intercept}$')
        ax.legend(fontsize=10)
        # Annotate n_volumes
        for k, b, nv in zip(ks, bs, nvs):
            ax.annotate(f'n_vol={nv}', (k, b), fontsize=7, xytext=(5, 5),
                       textcoords='offset points')
        save(plt, fig, 'R14_cross_sectional_beta', w=10, h=5.5)

    # Fig R15: β split by Q/depth penetration
    if 'depth_at_best' in pc.columns:
        pc_valid = pc[(pc['depth_at_best'] > 0) & pc['depth_at_best'].notna()].copy()
        if len(pc_valid) > 100:
            # size_k is the individual order size for this insertion
            penetrate = pc_valid[pc_valid['size_k'] >= pc_valid['depth_at_best']]
            no_penetrate = pc_valid[pc_valid['size_k'] < pc_valid['depth_at_best']]

            fig, ax = plt.subplots(figsize=(9, 5.5))
            labels_p = []
            betas_p = []
            for name, sub, color in [
                ('Q < depth\n(no penetration)', no_penetrate, 'steelblue'),
                ('Q ≥ depth\n(penetration)', penetrate, 'coral'),
                ('All', pc_valid, 'gray'),
            ]:
                if len(sub) < 20: continue
                res = compute_global_beta(sub)
                labels_p.append(name)
                betas_p.append(res['beta'])

            if labels_p:
                x_pos = np.arange(len(labels_p))
                colors = ['steelblue', 'coral', 'gray'][:len(labels_p)]
                ax.bar(x_pos, betas_p, color=colors, width=0.5, edgecolor='gray')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels_p, fontsize=11)
                ax.axhline(0.5, ls='--', color='red', lw=2, label='β=0.5')
                for i, b in enumerate(betas_p):
                    ax.text(i, b + 0.01, f'{b:.3f}', ha='center', fontsize=11, fontweight='bold')
                ax.set(title=r'R15: $\beta$ Split by First-Level Penetration (Historic)',
                       ylabel=r'$\beta_{intercept}$')
                n_pen = len(penetrate)
                n_no = len(no_penetrate)
                ax.text(0.02, 0.98, f'No penetration: {n_no:,}\nPenetration: {n_pen:,}',
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                ax.legend()
                save(plt, fig, 'R15_penetration_split', w=9, h=5.5)

    # Table R5: Summary of all β estimates
    if not D['summary'].empty:
        df = D['summary']
        df2 = D['summary_other']
        lines = []
        lines.append(r'\begin{tabular}{lcc}')
        lines.append(r'\toprule')
        lines.append(f'Estimator / Condition & {stock} & {D["other_stock"]} \\\\')
        lines.append(r'\midrule')

        def avg_col(frame, col):
            if col in frame.columns:
                v = frame[col].dropna()
                return f'{v.mean():.3f}' if len(v) > 0 else '---'
            return '---'

        pairs = [
            (r'$\beta_{origin}$ (all k)', 'beta_origin'),
            (r'$\beta_{intercept}$ (all k)', 'beta_intercept'),
            (r'$\beta_{intercept}$ (k$\geq$3)', 'beta_k3plus'),
            (r'$\beta_{incremental}$ (k$\geq$3)', 'beta_incremental'),
            (r'$\beta_{V_{local}}$', 'beta_Vlocal'),
        ]
        for label, col in pairs:
            v1 = avg_col(df, col)
            v2 = avg_col(df2, col)
            lines.append(f'  {label} & {v1} & {v2} \\\\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        save_tex('R05_table_summary', '\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    global OUT

    parser = argparse.ArgumentParser(description='Beta Report for paper')
    parser.add_argument('--stock', type=str, default='GOOG')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    stock = args.stock.upper()
    OUT = Path(f'pics_for_beta_report')
    OUT.mkdir(exist_ok=True)

    print(f'{"="*60}')
    print(f'  Beta Report: {stock}')
    print(f'{"="*60}')

    D = load_all_data(stock)
    print(f'Loaded: {len(D["models"])} models, {len(D["summary"])} summary rows')

    plt = setup_mpl()

    section_1_estimators(plt, D, stock)
    section_2_volume_calibration(plt, D, stock)
    section_3_scaling_dynamics(plt, D, stock)
    section_4_normalizations(plt, D, stock)
    section_5_impact_definitions(plt, D, stock)
    section_6_model_differentiation(plt, D, stock)
    section_7_cross_stock(plt, D, stock)
    section_8_new_analyses(plt, D, stock)

    n_figs = len(list(OUT.glob('*.png')))
    n_tex = len(list(OUT.glob('*.tex')))
    print(f'\n{"="*60}')
    print(f'  Done: {n_figs} PNGs + {n_tex} LaTeX tables in {OUT}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
