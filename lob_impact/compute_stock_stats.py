#!/usr/bin/env python3
"""
Compute per-stock statistics from LOBSTER conditioning data.
Derives recommended mb (messages between insertions) from participation rate targets.

Usage:
    python lob_impact/compute_stock_stats.py --stock GOOG
    python lob_impact/compute_stock_stats.py --stock INTC
"""
import argparse, re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/data')
# Also check v4 experiment conditioning data
EXP_BASE = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v4')

TRADING_SECONDS = 23400  # 6.5 hours


def compute_from_lobster_raw(stock):
    """Compute stats from raw LOBSTER message files."""
    data_dir = DATA_BASE / f'{stock}_jan2026'
    if not data_dir.exists():
        print(f'  Raw data dir not found: {data_dir}')
        return None

    msg_files = sorted(data_dir.glob('*message*.npy'))
    if not msg_files:
        msg_files = sorted(data_dir.glob('*message*.csv'))
    if not msg_files:
        print(f'  No message files in {data_dir}')
        return None

    all_stats = []
    for f in msg_files[:20]:  # first 20 days
        try:
            if f.suffix == '.npy':
                msgs = np.load(f)
            else:
                msgs = pd.read_csv(f, header=None).values
        except:
            continue

        n_msgs = len(msgs)
        if n_msgs < 100:
            continue

        # Time range (column 0 = seconds from midnight, or nanoseconds)
        times = msgs[:, 0].astype(float)
        t_range = times[-1] - times[0]
        if t_range <= 0:
            t_range = TRADING_SECONDS  # fallback

        # Message rate
        R = n_msgs / t_range if t_range > 0 else 0

        # Executions (event type 4 or 5)
        exec_mask = np.isin(msgs[:, 1].astype(int), [4, 5])
        n_exec = exec_mask.sum()
        exec_sizes = msgs[exec_mask, 3].astype(float) if n_exec > 0 else np.array([0])
        exec_prices = msgs[exec_mask, 4].astype(float) if n_exec > 0 else np.array([0])

        avg_trade_size = float(np.mean(exec_sizes)) if n_exec > 0 else 0
        median_trade_size = float(np.median(exec_sizes)) if n_exec > 0 else 0
        total_volume = float(np.sum(exec_sizes))

        # Day from filename
        m = re.search(r'(\d{4}-\d{2}-\d{2})', f.name)
        day = m.group(1) if m else f.stem

        all_stats.append(dict(
            day=day, n_msgs=n_msgs, t_range_sec=t_range,
            R_msg_per_sec=R, n_executions=n_exec,
            avg_trade_size=avg_trade_size,
            median_trade_size=median_trade_size,
            total_volume=total_volume,
            exec_rate=n_exec / t_range if t_range > 0 else 0,
        ))

    return pd.DataFrame(all_stats) if all_stats else None


def compute_from_conditioning(stock):
    """Compute stats from experiment conditioning data (data_cond CSVs)."""
    # Find one model's conditioning data (Historic — same for all)
    cond_base = EXP_BASE / 'Historic' / 'context_500_buy' / stock
    if not cond_base.exists():
        print(f'  Conditioning dir not found: {cond_base}')
        return None

    # Find first config folder
    config_dirs = sorted([d for d in cond_base.iterdir() if d.is_dir()])
    if not config_dirs:
        return None

    # Find exp folder
    exp_dirs = sorted(config_dirs[0].glob('exp_*'))
    if not exp_dirs:
        return None

    cond_dir = exp_dirs[0] / 'data_cond'
    if not cond_dir.exists():
        return None

    msg_files = sorted(cond_dir.glob('*_message_*.csv'))[:50]  # 50 samples
    all_stats = []

    for f in msg_files:
        try:
            msgs = pd.read_csv(f, header=None).values
        except:
            continue

        n_msgs = len(msgs)
        if n_msgs < 10:
            continue

        # LOBSTER format: Time(0), EventType(1), OrderID(2), Size(3), Price(4), Direction(5)
        times = msgs[:, 0].astype(float)
        t_range = times[-1] - times[0]

        # Message rate (from conditioning segment)
        R = n_msgs / t_range if t_range > 1 else n_msgs / TRADING_SECONDS

        # Executions
        exec_mask = np.isin(msgs[:, 1].astype(int), [4, 5])
        n_exec = exec_mask.sum()
        exec_sizes = msgs[exec_mask, 3].astype(float) if n_exec > 0 else np.array([0])

        avg_trade_size = float(np.mean(exec_sizes)) if n_exec > 0 else 0
        median_trade_size = float(np.median(exec_sizes)) if n_exec > 0 else 0

        all_stats.append(dict(
            n_msgs=n_msgs, t_range_sec=t_range,
            R_msg_per_sec=R, n_executions=n_exec,
            avg_trade_size=avg_trade_size,
            median_trade_size=median_trade_size,
        ))

    return pd.DataFrame(all_stats) if all_stats else None


def derive_mb(R, avg_trade, vol, phi_target):
    """Derive mb from participation rate target.

    phi = vol / (R × mb × avg_trade)
    mb = vol / (R × avg_trade × phi)
    """
    denom = R * avg_trade * phi_target
    if denom <= 0:
        return np.nan
    return vol / denom


def run(stock):
    print(f'=== Stock Statistics: {stock} ===\n')

    # Try conditioning data first (always available for experiments)
    stats = compute_from_conditioning(stock)
    source = 'conditioning (data_cond)'

    if stats is None or stats.empty:
        stats = compute_from_lobster_raw(stock)
        source = 'raw LOBSTER'

    if stats is None or stats.empty:
        print('ERROR: No data found')
        return

    print(f'Source: {source}')
    print(f'Samples: {len(stats)}\n')

    # Aggregate
    R = stats['R_msg_per_sec'].median()
    avg_trade = stats['avg_trade_size'].median()
    median_trade = stats['median_trade_size'].median()
    n_exec_frac = stats['n_executions'].median() / stats['n_msgs'].median()

    print(f'MESSAGE RATE:')
    print(f'  R (median): {R:.1f} msg/sec')
    print(f'  R range: [{stats["R_msg_per_sec"].min():.1f}, {stats["R_msg_per_sec"].max():.1f}]')
    print(f'  Execution fraction: {n_exec_frac:.1%} of all messages are executions')
    print()
    print(f'TRADE SIZE:')
    print(f'  Mean: {avg_trade:.0f} shares')
    print(f'  Median: {median_trade:.0f} shares')
    print()

    # Load depth stats for vol recommendations
    depth_file = Path(f'lob_impact/depth_stats_{stock}.csv')
    if depth_file.exists():
        depth = pd.read_csv(depth_file)
        d = depth['depth_relevant'].dropna()
        d = d[d > 0]
        vols = {
            'p25': np.percentile(d, 25),
            'p50': np.percentile(d, 50),
            'p75': np.percentile(d, 75),
            'p95': np.percentile(d, 95),
        }
        print(f'DEPTH AT BEST (from depth_stats):')
        for k, v in vols.items():
            print(f'  {k}: {v:.0f} shares')
        print()
    else:
        vols = {'p50': 100, 'p75': 200}

    # Daily volume
    hl_file = Path(f'lob_impact/daily_h_l_{stock}.csv')
    V_daily = 300000  # fallback
    if hl_file.exists():
        hl = pd.read_csv(hl_file)
        V_daily = hl['execution_sum'].median()
        print(f'DAILY VOLUME (median): {V_daily:.0f} shares')
        print()

    # Derive mb for different participation rates
    print(f'RECOMMENDED mb (messages between insertions):')
    print(f'  φ = participation rate = vol / (R × mb × avg_trade)')
    print(f'  mb = vol / (R × avg_trade × φ)')
    print()

    phi_targets = [0.01, 0.02, 0.05, 0.10, 0.20]
    vol_ref = vols.get('p50', 100)

    print(f'  Using vol = {vol_ref:.0f} (p50 depth), R = {R:.1f}, avg_trade = {avg_trade:.0f}:')
    print(f'  {"φ":>6} {"mb":>6} {"regime":>20} {"total_gen(i=10)":>16}')
    print(f'  {"-"*55}')
    for phi in phi_targets:
        mb = derive_mb(R, avg_trade, vol_ref, phi)
        regime = 'LINEAR (β≈1)' if phi < 0.01 else \
                 'CROSSOVER' if phi < 0.02 else \
                 'SQUARE-ROOT (β≈0.5)' if phi < 0.10 else 'VERY AGGRESSIVE'
        total_gen = 10 * (int(mb) + 1) + int(mb) if not np.isnan(mb) else np.nan
        fits = '✓' if not np.isnan(total_gen) and total_gen <= 600 else '✗ (>600)'
        print(f'  {phi:6.1%} {mb:6.1f} {regime:>20} {total_gen:>10.0f} {fits}')

    # Table for all vol levels
    print(f'\n  FULL TABLE (mb for each vol × φ):')
    print(f'  {"vol":>6}', end='')
    for phi in phi_targets:
        print(f'  φ={phi:.0%}', end='')
    print()
    print(f'  {"-"*50}')
    for vol_name, vol_val in sorted(vols.items(), key=lambda x: x[1]):
        print(f'  {vol_val:6.0f}', end='')
        for phi in phi_targets:
            mb = derive_mb(R, avg_trade, vol_val, phi)
            print(f'  {mb:6.1f}', end='')
        print(f'  ({vol_name})')

    # Recommendation
    print(f'\n{"="*60}')
    print(f'RECOMMENDATION for {stock}:')
    print(f'{"="*60}')
    mb_rec = derive_mb(R, avg_trade, vol_ref, 0.05)  # φ=5%
    print(f'  Target φ = 5% (solidly in square-root regime)')
    print(f'  vol = {vol_ref:.0f} (p50 depth)')
    print(f'  → mb = {mb_rec:.0f}')
    print(f'  → With i=10: total_gen = {10*(int(mb_rec)+1)+int(mb_rec):.0f} messages')
    gen = 10 * (int(mb_rec) + 1) + int(mb_rec)
    print(f'  → Fits in gen≤600: {"YES" if gen <= 600 else "NO"}')
    print(f'  → Participation rate check: φ = {vol_ref/(R*int(mb_rec)*avg_trade):.1%}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', default='GOOG')
    args = parser.parse_args()
    run(args.stock.upper())
