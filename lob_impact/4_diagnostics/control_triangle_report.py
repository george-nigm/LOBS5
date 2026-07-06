#!/usr/bin/env python3
"""
Action 4 — "control triangle" figures + numbers for the explanatory report.

Three generation regimes, identical except for two switches (child MOs applied to
the book? / child MOs visible to the model?):
  visible   : grid_v2 EA-Mamba3-beta  (book YES, model YES)  -> full impact
  invisible : controls_v2/invisible   (book YES, model NO)   -> pure mechanics
  noins     : controls_v2/noins       (book NO,  model NO)   -> unconditional drift

Outputs (to --out_dir):
  fig1_regimes.png      schematic of the three regimes
  fig2_trajectories.png mean signed mid-price trajectory per regime
  fig3_decomposition.png mechanical vs model-generated ticks per regime
  fig4_event_response.png model per-child response vs real-data R(m) anchor
  fig5_spread.png       spread evolution per regime
  numbers.json          all aggregates used in the report text

Run on a COMPUTE node (Lustre CSV reads):
  python 4_diagnostics/control_triangle_report.py --out_dir results/triangle_<ts>
"""
import os, re, csv, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TICK = 100
SENT = 2147483647
GRID01 = np.linspace(0.0, 1.0, 241)
MODEL_LABEL = 'Mamba3'

# --- palette (validated reference set, fixed slot order; light surface) ---
C_VIS, C_INV, C_NOI = '#2a78d6', '#1baf7a', '#eda100'   # regimes: blue / aqua / yellow
C_REAL = '#008300'                                       # real-data anchor: green
C_MECH, C_DRIFT = '#4a3aa7', '#eb6834'                   # components: violet / orange
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRIDC, AXISC = '#e1e0d9', '#c3c2b7'
GOOD, CRIT = '#0ca30c', '#d03b3b'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'text.color': INK, 'axes.edgecolor': AXISC, 'axes.labelcolor': INK2,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRIDC, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False,
})


try:
    import pandas as _pd
except ImportError:
    _pd = None


def read_csv_np(f):
    if _pd is not None:
        return _pd.read_csv(f, header=None, dtype=float).to_numpy()
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def discover_exp(side_dir):
    e = sorted(glob.glob(os.path.join(side_dir, 'exp_*')))
    if e:
        return e[0]
    if os.path.isdir(os.path.join(side_dir, 'data_gen')):
        return side_dir
    return None


def aggr_for(exp, date, L):
    f = os.path.join(exp, f'aggressive_indices_{date}.csv')
    if not os.path.exists(f):
        f = os.path.join(exp, 'aggressive_indices.csv')
    if not os.path.exists(f):
        return np.array([], dtype=int)
    a = np.loadtxt(f, dtype=int, ndmin=1)
    return a[a < L]


def book_mid_spread(b):
    ask, bid = b[:, 0], b[:, 2]
    bad = (ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)
    mid = (ask + bid) / 2.0
    spr = (ask - bid) / TICK
    mid[bad] = np.nan
    spr[bad] = np.nan
    return mid, spr


def load_regime(exp, sign, n_samples, with_insertions, m_max=131):
    """Per-sample trajectories (signed ticks, on GRID01), spreads, decomposition,
    and pooled per-insertion event response."""
    gens = sorted(glob.glob(os.path.join(exp, 'data_gen', '*message*gen*.csv')))
    step = max(len(gens) // n_samples, 1)
    trajs, sprs, mechs, drifts, finals = [], [], [], [], []
    # streaming accumulators for R(m): full-sample runs would not fit as lists
    r_sum = np.zeros(m_max + 1)
    r_sumsq = np.zeros(m_max + 1)
    r_cnt = np.zeros(m_max + 1, dtype=np.int64)
    min_gap = 10 ** 9
    for mf in gens[::step][:n_samples]:
        bf = mf.replace('message', 'orderbook')
        if not os.path.exists(bf):
            continue
        m_date = re.search(r'_(\d{4}-\d{2}-\d{2})_', os.path.basename(mf))
        date = m_date.group(1) if m_date else ''
        msg = read_csv_np(mf)
        book = read_csv_np(bf)
        L = min(len(msg), len(book))
        book = book[:L]
        mid, spr = book_mid_spread(book)
        if np.isnan(mid[0]):
            continue
        x = np.arange(L) / max(L - 1, 1)
        ok = ~np.isnan(mid)
        trajs.append(np.interp(GRID01, x[ok], (sign * (mid[ok] - mid[0]) / TICK)))
        oks = ~np.isnan(spr)
        sprs.append(np.interp(GRID01, x[oks], spr[oks]))
        ai = aggr_for(exp, date, L)
        ai = ai[ai > 0]
        if with_insertions and len(ai) > 1:
            min_gap = min(min_gap, int(np.diff(ai).min()))
            mech = np.nansum(mid[ai] - mid[ai - 1]) / TICK
            segs = np.concatenate([[mid[ai[0] - 1] - mid[0]],
                                   mid[ai[1:] - 1] - mid[ai[:-1]]])
            mechs.append(sign * mech)
            drifts.append(sign * np.nansum(segs) / TICK)
            nxt = np.append(ai[1:], L)
            for i, nx in zip(ai, nxt):
                base = mid[i - 1]
                if np.isnan(base):
                    continue
                # stop before the NEXT insertion so its impact never leaks into R(m)
                hi = min(i - 1 + m_max, nx - 1, L - 1)
                seg = sign * (mid[i:hi + 1] - base) / TICK        # values at m = 1..len(seg)
                ok = ~np.isnan(seg)
                mm = np.arange(1, len(seg) + 1)[ok]
                v = seg[ok]
                r_sum[mm] += v
                r_sumsq[mm] += v * v
                r_cnt[mm] += 1
        end = mid[-1] if not np.isnan(mid[-1]) else np.nanmean(mid[-20:])
        finals.append(sign * (end - mid[0]) / TICK)
    trajs, sprs = np.array(trajs), np.array(sprs)
    n = len(trajs)
    out = {
        'n': n,
        'traj_mean': trajs.mean(0), 'traj_se': trajs.std(0) / max(np.sqrt(n), 1),
        'spr_mean': sprs.mean(0),
        'final_mean': float(np.mean(finals)), 'final_se': float(np.std(finals) / max(np.sqrt(n), 1)),
        'mech_mean': float(np.mean(mechs)) if mechs else 0.0,
        'drift_mean': float(np.mean(drifts)) if drifts else float(np.mean(finals)),
        'min_gap': min_gap if min_gap < 10 ** 9 else None,
    }
    if with_insertions:
        cnt = np.maximum(r_cnt, 1)
        mean = r_sum / cnt
        var = np.maximum(r_sumsq / cnt - mean ** 2, 0.0)
        enough = r_cnt > 10
        r_mean = np.where(enough, mean, np.nan)
        r_se = np.where(enough, np.sqrt(var / cnt), np.nan)
        r_mean[0] = r_se[0] = np.nan
        out['resp_mean'], out['resp_se'] = r_mean, r_se
        out['resp_n_events'] = int(r_cnt[1])
    return out


def load_hist_spread(exp, n=64):
    """Mean spread of the REAL conditioning streams (data_cond) — the historical anchor."""
    books = sorted(glob.glob(os.path.join(exp, 'data_cond', '*orderbook*.csv')))
    step = max(len(books) // n, 1)
    vals = []
    for bf in books[::step][:n]:
        _, spr = book_mid_spread(read_csv_np(bf))
        vals.append(np.nanmean(spr))
    return float(np.nanmean(vals)) if vals else float('nan')


def style_ax(ax):
    ax.grid(axis='x', visible=False)
    ax.axhline(0, color=AXISC, lw=1.0, zorder=1)


def end_labels(ax, items, min_sep_frac=0.045):
    """Right-edge direct labels without collisions. items: [(y, text, color)]."""
    lo, hi = ax.get_ylim()
    sep = (hi - lo) * min_sep_frac
    order = sorted(range(len(items)), key=lambda k: items[k][0])
    ys = [items[k][0] for k in order]
    for j in range(1, len(ys)):          # push up
        ys[j] = max(ys[j], ys[j - 1] + sep)
    over = ys[-1] - max(hi, items[order[-1]][0])
    if over > 0:                          # keep inside the axis
        ys = [y - over for y in ys]
        for j in range(len(ys) - 2, -1, -1):
            ys[j] = min(ys[j], ys[j + 1] - sep)
    for k, y in zip(order, ys):
        _, text, col = items[k]
        ax.annotate(text, xy=(1.005, y), xycoords=('axes fraction', 'data'),
                    color=col, fontsize=8.5, fontweight='bold', va='center',
                    annotation_clip=False)


def fig_schematic(path, vals):
    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=200)
    ax.set_axis_off()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    cols = [('Visible metaorder', C_VIS, True, True, vals['visible']),
            ('Invisible metaorder', C_INV, True, False, vals['invisible']),
            ('No insertions', C_NOI, False, False, vals['noins'])]
    rows = ['Child MOs applied\nto the order book', 'Child MOs visible\nto the model',
            'What it isolates']
    iso = ['mechanics + model reaction', 'book mechanics only', 'unconditional model drift']
    for c, (name, col, hit, seen, val) in enumerate(cols):
        x = c + 0.06
        ax.add_patch(FancyBboxPatch((x, 0.86), 0.88, 0.11,
                     boxstyle='round,pad=0.012', fc=col, ec='none'))
        ax.text(x + 0.44, 0.915, name, ha='center', va='center', color='white',
                fontsize=11, fontweight='bold')
        for rI, flag in enumerate([hit, seen]):
            y = 0.72 - rI * 0.17
            ax.text(x + 0.03, y, rows[rI], fontsize=8.5, color=INK2, va='center')
            mark, mcol = ('✓ yes', GOOD) if flag else ('✗ no', CRIT)
            ax.text(x + 0.66, y, mark, fontsize=10.5, color=mcol, va='center',
                    fontweight='bold')
        ax.text(x + 0.03, 0.345, rows[2], fontsize=8.5, color=INK2, va='center')
        ax.text(x + 0.44, 0.255, iso[c], fontsize=8.5, color=INK, ha='center',
                va='center', style='italic')
        ax.add_patch(FancyBboxPatch((x + 0.10, 0.03), 0.68, 0.13,
                     boxstyle='round,pad=0.012', fc='white', ec=col, lw=2))
        ax.text(x + 0.44, 0.115, val[0], ha='center', va='center', fontsize=11,
                color=INK, fontweight='bold')
        ax.text(x + 0.44, 0.055, val[1], ha='center', va='center', fontsize=8,
                color=MUTED)
        ax.annotate('', xy=(x + 0.44, 0.17), xytext=(x + 0.44, 0.22),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4))
    fig.suptitle(f'{MODEL_LABEL}: three generation regimes — identical run, two switches flipped',
                 fontsize=12, fontweight='bold', color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_trajectories(path, R):
    fig, ax = plt.subplots(figsize=(9.4, 4.9), dpi=200)
    style_ax(ax)
    series = [('visible-buy', C_VIS, '-', 'Visible, buy'),
              ('visible-sell', C_VIS, (0, (5, 2)), 'Visible, sell'),
              ('invisible-buy', C_INV, '-', 'Invisible, buy'),
              ('invisible-sell', C_INV, (0, (5, 2)), 'Invisible, sell'),
              ('noins', C_NOI, '-', 'No insertions')]
    labels = []
    for key, col, ls, lab in series:
        r = R[key]
        ax.fill_between(GRID01 * 100, r['traj_mean'] - 2 * r['traj_se'],
                        r['traj_mean'] + 2 * r['traj_se'], color=col, alpha=0.13, lw=0)
        ax.plot(GRID01 * 100, r['traj_mean'], color=col, ls=ls, lw=2.0,
                label=f"{lab} (n={r['n']})", solid_capstyle='round')
        labels.append((r['traj_mean'][-1], f"{lab}  {r['traj_mean'][-1]:+.0f}", col))
    ax.set_xlim(0, 100)
    end_labels(ax, labels)
    ax.set_xlabel('Rollout progress, %   (visible/invisible: ≈ child order 1 → 100)')
    ax.set_ylabel('Mid-price move in trade direction, ticks')
    ax.set_title(f'{MODEL_LABEL} — cumulative impact by regime (does the model need to SEE the metaorder?)',
                 fontsize=11.5, fontweight='bold', loc='left', color=INK)
    ax.legend(loc='upper left', fontsize=8.5, ncols=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_decomposition(path, R):
    keys = ['visible-buy', 'visible-sell', 'invisible-buy', 'invisible-sell', 'noins']
    base = ['Visible, buy', 'Visible, sell', 'Invisible, buy', 'Invisible, sell',
            'No insertions']
    labels = [f"{b}\n(n={R[k]['n']})" for b, k in zip(base, keys)]
    mech = [R[k]['mech_mean'] for k in keys]
    drift = [R[k]['drift_mean'] for k in keys]
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=200)
    ax.grid(axis='y', visible=False)
    ax.axvline(0, color=AXISC, lw=1.0)
    h = 0.34
    ax.barh(y + h / 2 + 0.02, mech, height=h, color=C_MECH, label='Mechanical (book eating at insertion)')
    ax.barh(y - h / 2 - 0.02, drift, height=h, color=C_DRIFT, label='Model-generated (between insertions)')
    for yy, v in zip(y + h / 2 + 0.02, mech):
        ax.text(v + np.sign(v) * 1.5 + 0.5, yy, f'{v:+.1f}', va='center', fontsize=8.5, color=INK2)
    for yy, v in zip(y - h / 2 - 0.02, drift):
        ax.text(v + np.sign(v) * 1.5 + 0.5, yy, f'{v:+.1f}', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlabel('Contribution to total mid move (trade direction), ticks')
    ax.set_title(f'{MODEL_LABEL} — decomposition: mechanical book-eating vs model-generated drift',
                 fontsize=11.5, fontweight='bold', loc='left', color=INK)
    ax.legend(loc='lower right', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_event_response(path, R, emp):
    fig, ax = plt.subplots(figsize=(9.4, 4.9), dpi=200)
    style_ax(ax)
    for key, col, lab in [('visible-buy', C_VIS, 'Model, visible (buy)'),
                          ('invisible-buy', C_INV, 'Model, invisible (buy)')]:
        r = R[key]
        if 'resp_mean' not in r:
            continue
        m = np.arange(len(r['resp_mean']))
        ok = ~np.isnan(r['resp_mean'])
        ax.fill_between(m[ok], (r['resp_mean'] - 2 * r['resp_se'])[ok],
                        (r['resp_mean'] + 2 * r['resp_se'])[ok], color=col, alpha=0.13, lw=0)
        ax.plot(m[ok], r['resp_mean'][ok], color=col, lw=2.0,
                label=f"{lab} — {r['resp_n_events']} child events, n={r['n']} runs")
        last = np.where(ok)[0][-1]
        ax.annotate(f"{lab}  {r['resp_mean'][last]:+.2f}", xy=(last, r['resp_mean'][last]),
                    xytext=(last + 1.5, r['resp_mean'][last]), color=col, fontsize=8.5,
                    fontweight='bold', va='center', annotation_clip=False)
    hs = sorted(int(k.split('_')[1]) for k in emp if re.fullmatch(r'R_\d+', k))
    rv = [emp[f'R_{h}'] for h in hs]
    se = [emp.get(f'R_{h}_se', 0) for h in hs]
    ax.errorbar(hs, rv, yerr=[2 * s for s in se], color=C_REAL, lw=2.0, marker='o',
                ms=6, capsize=3, label=f'Real EA data ({emp["n_events"]} executions)')
    ax.annotate(f'real data saturates  {rv[-1]:+.2f}', xy=(hs[-1], rv[-1]),
                xytext=(hs[-1] - 38, rv[-1] - 0.22), color=C_REAL, fontsize=8.5,
                fontweight='bold')
    ax.set_xlim(0, 135)
    ax.set_xlabel('Messages after the execution event  m')
    ax.set_ylabel('Mean mid response R(m), ticks')
    ax.set_title(f'{MODEL_LABEL} — per-event response R(m) vs real-data anchor',
                 fontsize=11.5, fontweight='bold', loc='left', color=INK)
    ax.legend(loc='upper left', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_spread(path, R, hist_spread):
    fig, ax = plt.subplots(figsize=(9.4, 4.4), dpi=200)
    ax.grid(axis='x', visible=False)
    if hist_spread == hist_spread:
        ax.axhline(hist_spread, color=C_REAL, lw=1.8, ls=(0, (2, 2)), zorder=1)
    series = [('visible-buy', C_VIS, '-', 'Visible, buy'),
              ('visible-sell', C_VIS, (0, (5, 2)), 'Visible, sell'),
              ('invisible-buy', C_INV, '-', 'Invisible, buy'),
              ('invisible-sell', C_INV, (0, (5, 2)), 'Invisible, sell'),
              ('noins', C_NOI, '-', 'No insertions')]
    labels = []
    for key, col, ls, lab in series:
        r = R[key]
        ax.plot(GRID01 * 100, r['spr_mean'], color=col, ls=ls, lw=2.0,
                label=f"{lab} (n={r['n']})")
        labels.append((r['spr_mean'][-1], f"{lab}  {r['spr_mean'][-1]:.1f}", col))
    if hist_spread == hist_spread:
        labels.append((hist_spread, f'Real EA (historical)  {hist_spread:.1f}', C_REAL))
        ax.plot([], [], color=C_REAL, lw=1.8, ls=(0, (2, 2)), label='Real EA (historical)')
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    end_labels(ax, labels)
    ax.set_xlabel('Rollout progress, %')
    ax.set_ylabel('Mean bid–ask spread, ticks')
    ax.set_title(f'{MODEL_LABEL} — book health: mean spread along the rollout by regime',
                 fontsize=11.5, fontweight='bold', loc='left', color=INK)
    ax.legend(loc='upper left', fontsize=8.5, ncols=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2')
    ap.add_argument('--controls', default='/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2')
    ap.add_argument('--emp_json', default=os.path.join(os.path.dirname(__file__),
                    'results/empresp_20260705-142822/empirical_response.json'))
    ap.add_argument('--n_samples', type=int, default=64)
    ap.add_argument('--model', default='Mamba3', help='grid model label (Mamba3, Mamba3_4k, S5_4k)')
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    global MODEL_LABEL
    MODEL_LABEL = args.model
    scen = f'{args.stock}-{args.model}-beta'
    spec = {
        'visible-buy': (os.path.join(args.grid, scen, 'buy'), +1, True),
        'visible-sell': (os.path.join(args.grid, scen, 'sell'), -1, True),
        'invisible-buy': (os.path.join(args.controls, 'invisible', scen, 'buy'), +1, True),
        'invisible-sell': (os.path.join(args.controls, 'invisible', scen, 'sell'), -1, True),
        'noins': (os.path.join(args.controls, 'noins', scen, 'buy'), +1, False),
    }
    R = {}
    for key, (d, sign, ins) in spec.items():
        exp = discover_exp(d)
        if not exp:
            raise SystemExit(f'missing regime dir: {d}')
        print(f'loading {key} <- {exp}')
        R[key] = load_regime(exp, sign, args.n_samples, ins)
        print(f"  n={R[key]['n']} final={R[key]['final_mean']:+.1f}±{R[key]['final_se']:.1f} "
              f"mech={R[key]['mech_mean']:+.1f} drift={R[key]['drift_mean']:+.1f}")

    emp = json.load(open(args.emp_json))
    hist_spread = load_hist_spread(discover_exp(spec['visible-buy'][0]), args.n_samples)
    print(f'historical (data_cond) mean spread: {hist_spread:.2f} ticks')

    vals = {
        'visible': (f"{R['visible-buy']['final_mean']:+.0f} / {R['visible-sell']['final_mean']:+.0f} ticks",
                    f"mean over n={R['visible-buy']['n']}+{R['visible-sell']['n']} runs (buy+sell)"),
        'invisible': (f"{R['invisible-buy']['final_mean']:+.0f} / {R['invisible-sell']['final_mean']:+.0f} ticks",
                      f"mean over n={R['invisible-buy']['n']}+{R['invisible-sell']['n']} runs (buy+sell)"),
        'noins': (f"{R['noins']['final_mean']:+.0f} ticks",
                  f"mean over n={R['noins']['n']} runs"),
    }
    fig_schematic(os.path.join(args.out_dir, 'fig1_regimes.png'), vals)
    fig_trajectories(os.path.join(args.out_dir, 'fig2_trajectories.png'), R)
    fig_decomposition(os.path.join(args.out_dir, 'fig3_decomposition.png'), R)
    fig_event_response(os.path.join(args.out_dir, 'fig4_event_response.png'), R, emp)
    fig_spread(os.path.join(args.out_dir, 'fig5_spread.png'), R, hist_spread)

    num = {'empirical': emp, 'model': args.model, 'stock': args.stock,
           'hist_spread': hist_spread}
    for k, r in R.items():
        num[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                  for kk, vv in r.items()}
        num[k]['spread_first'] = float(np.mean(r['spr_mean'][:24]))
        num[k]['spread_last'] = float(np.mean(r['spr_mean'][-24:]))
    with open(os.path.join(args.out_dir, 'numbers.json'), 'w') as fo:
        json.dump(num, fo, indent=1)
    print(f'FIGURES_DONE -> {args.out_dir}')


if __name__ == '__main__':
    main()
