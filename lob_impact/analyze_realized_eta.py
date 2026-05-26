#!/usr/bin/env python3
"""
Realized participation rate analysis.

For each experiment: compute ACTUAL η during insertion phase.
η_realized = Q_injected / (Q_injected + Q_model_generated)

Where Q_model_generated = total executed volume of model-generated trades
during the insertion phase (between first and last injection).

Usage:
    python lob_impact/analyze_realized_eta.py --stock AAPL \
        --pickle_base /path/to/pickles --version v6 \
        --out_dir pics_for_realized_eta
"""
import argparse, pickle, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob_impact.run_300_analyze_one import (
    filter_model, collect_days, load_aggressive_indices,
)

MODELS_ORDER = ['Historic', 'Heuristic', 'CST', 'S5-120M', 'S5-360M', 'S5-4K']

ETA_MAP = {
    ('AAPL', 20): 5, ('AAPL', 40): 9, ('AAPL', 90): 19,
    ('AAPL', 100): 22, ('AAPL', 400): 50, ('AAPL', 1000): 72,
    ('AMZN', 10): 5, ('AMZN', 20): 9, ('AMZN', 40): 17,
    ('AMZN', 70): 26, ('AMZN', 250): 56, ('AMZN', 1000): 84,
    ('AAPL', 3570): 90, ('AAPL', 390): 50,
    ('AMZN', 190): 50, ('AMZN', 1770): 90,
}


def analyze_one_model(pkl_path, model_name, stock):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f'  SKIP {model_name}')
        return []

    print(f'\n  Loading {model_name} ({pkl_path.stat().st_size / 1e9:.1f} GB)...')
    with open(pkl_path, 'rb') as f:
        md = pickle.load(f)

    grid_df = md['grid']
    filtered, n_total, n_skip = filter_model(md)
    del md
    gc.collect()

    results = []
    _aggr_cache = {}

    for _, row in grid_df.iterrows():
        folder = row['folder']
        if folder not in filtered:
            continue

        fd = filtered[folder]
        vol = int(row['vol'])
        eta_target = ETA_MAP.get((stock, vol), -1)

        bp, sp = row['buy_path'], row['sell_path']
        if bp not in _aggr_cache:
            _aggr_cache[bp] = load_aggressive_indices(bp)
        if sp not in _aggr_cache:
            _aggr_cache[sp] = load_aggressive_indices(sp)
        aggr_buy = _aggr_cache[bp]
        aggr_sell = _aggr_cache[sp]
        if aggr_buy is None or aggr_sell is None:
            continue

        for direction, aggr_idx, msgs_list in [
            ('buy',  aggr_buy,  fd['buy']['msgs']),
            ('sell', aggr_sell, fd['sell']['msgs']),
        ]:
            n_aggr = len(aggr_idx)
            if n_aggr < 2:
                continue

            for j, msg in enumerate(msgs_list):
                if len(msg) == 0:
                    continue
                if aggr_idx.max() >= len(msg):
                    continue

                # Our injected sizes
                our_sizes = msg[aggr_idx, 3].astype(float)
                Q_injected = float(our_sizes.sum())

                # Insertion phase: from first injection to last injection
                start_idx = int(aggr_idx[0])
                end_idx = int(aggr_idx[-1])

                # All messages in insertion phase
                phase_msgs = msg[start_idx:end_idx+1]

                # Total executed volume in insertion phase (event_type == 4 = executed)
                exec_mask = (phase_msgs[:, 1] == 4)
                V_total_phase = float(phase_msgs[exec_mask, 3].astype(float).sum()) if exec_mask.any() else 0

                # Model-generated volume = total - our injections
                V_model = V_total_phase - Q_injected

                # Also count: number of model trades, number of our trades
                # Our injections are at aggr_idx positions
                n_model_trades = int(exec_mask.sum()) - n_aggr
                n_our_trades = n_aggr

                # Realized η
                if V_total_phase > 0:
                    eta_realized = Q_injected / V_total_phase
                else:
                    eta_realized = 1.0  # we're the only volume

                # Also: per-interval stats
                # Between each pair of consecutive injections, how much volume?
                interval_model_vols = []
                for k in range(n_aggr - 1):
                    a, b = int(aggr_idx[k]) + 1, int(aggr_idx[k + 1])
                    if b > a:
                        interval = msg[a:b]
                        iv_exec = (interval[:, 1] == 4)
                        iv_vol = float(interval[iv_exec, 3].astype(float).sum()) if iv_exec.any() else 0
                        interval_model_vols.append(iv_vol)

                avg_model_vol_per_interval = np.mean(interval_model_vols) if interval_model_vols else 0
                child_size = float(our_sizes[0])

                # Per-interval η
                if child_size + avg_model_vol_per_interval > 0:
                    eta_per_interval = child_size / (child_size + avg_model_vol_per_interval)
                else:
                    eta_per_interval = 1.0

                results.append(dict(
                    model=model_name, stock=stock, vol=vol,
                    eta_target=eta_target, child=int(child_size),
                    direction=direction, sample=j,
                    Q_injected=Q_injected, V_model=V_model,
                    V_total=V_total_phase,
                    eta_realized=eta_realized,
                    eta_per_interval=eta_per_interval,
                    avg_model_vol_per_interval=avg_model_vol_per_interval,
                    n_model_trades=n_model_trades,
                    n_our_trades=n_our_trades,
                    n_intervals=len(interval_model_vols),
                ))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', required=True)
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--version', default='v6')
    parser.add_argument('--out_dir', default='pics_for_realized_eta')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = Path(args.pickle_base) / args.stock
    models = [m for m in MODELS_ORDER if (pkl_dir / f'{m}.pkl').exists()]

    print(f'Realized η analysis — {args.stock} ({args.version})')
    all_results = []

    for model in models:
        results = analyze_one_model(
            str(pkl_dir / f'{model}.pkl'), model, args.stock)
        all_results.extend(results)
        gc.collect()

    if not all_results:
        print('ERROR: no results')
        sys.exit(1)

    df = pd.DataFrame(all_results)
    csv_path = out_dir / f'realized_eta_{args.stock}_{args.version}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path} ({len(df)} rows)')

    # ── Summary table ──
    print(f'\n{"=" * 100}')
    print(f'  Realized η — {args.stock} ({args.version})')
    print(f'  η_target vs η_realized (median ± IQR)')
    print(f'{"=" * 100}')

    for model in models:
        mdf = df[df['model'] == model]
        print(f'\n  {model}:')
        for eta_t in sorted(mdf['eta_target'].unique()):
            if eta_t <= 0:
                continue
            edf = mdf[mdf['eta_target'] == eta_t]
            child = int(edf['child'].iloc[0])
            eta_r = edf['eta_per_interval']
            med = eta_r.median() * 100
            q25, q75 = eta_r.quantile(0.25) * 100, eta_r.quantile(0.75) * 100
            avg_mv = edf['avg_model_vol_per_interval'].median()
            n_mt = edf['n_model_trades'].median()
            print(f'    η_target={eta_t:>3d}%  child={child:>4d}  '
                  f'η_realized={med:6.1f}% [{q25:5.1f}%, {q75:5.1f}%]  '
                  f'model_vol/interval={avg_mv:7.1f}  model_trades/interval={n_mt:.0f}')

    # ── Plot ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    COLORS = {
        'Historic': '#90939C', 'Heuristic': '#546884', 'CST': '#213552',
        'S5-120M': '#D95F02', 'S5-4K': '#5B7BBF', 'S5-360M': '#B5446E',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: η_target vs η_realized (per-interval)
    ax = axes[0]
    for model in models:
        mdf = df[df['model'] == model]
        etas = sorted(mdf['eta_target'].unique())
        etas = [e for e in etas if e > 0]
        meds = [mdf[mdf['eta_target'] == e]['eta_per_interval'].median() * 100 for e in etas]
        ax.plot(etas, meds, 'o-', color=COLORS.get(model, 'gray'), lw=2, markersize=7, label=model)
    ax.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Target η (%)', fontsize=13)
    ax.set_ylabel('Realized η (%, per-interval median)', fontsize=13)
    ax.set_title('Target vs Realized Participation Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: model volume per interval
    ax = axes[1]
    for model in models:
        mdf = df[df['model'] == model]
        etas = sorted(mdf['eta_target'].unique())
        etas = [e for e in etas if e > 0]
        meds = [mdf[mdf['eta_target'] == e]['avg_model_vol_per_interval'].median() for e in etas]
        ax.plot(etas, meds, 'o-', color=COLORS.get(model, 'gray'), lw=2, markersize=7, label=model)
    ax.set_xlabel('Target η (%)', fontsize=13)
    ax.set_ylabel('Model volume per interval (shares)', fontsize=13)
    ax.set_title('Model-Generated Volume Between Injections', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.stock} ({args.version}) — Realized Participation Rate Analysis',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = out_dir / f'realized_eta_{args.stock}_{args.version}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {png_path}')


if __name__ == '__main__':
    main()
