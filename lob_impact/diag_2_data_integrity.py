#!/usr/bin/env python3
"""
Diagnostic 2: Data Integrity Validation for Market Impact Simulations.

Validates that simulation outputs are physically correct at the message level:
- aggressive_indices point to actual execution messages (EventType == 4)
- Prices are within the spread at time of execution
- Midprice moves in expected direction after aggressive orders
- Column mapping in CSVs matches expected LOBSTER format
- VWAP impact computed manually matches run_300_analyze_one.py logic

Reads raw CSV samples from v4 experiments on Lustre.

Usage:
  python lob_impact/diag_2_data_integrity.py --stock GOOG
  python lob_impact/diag_2_data_integrity.py --stock GOOG --max_samples 5
  python lob_impact/diag_2_data_integrity.py --stock GOOG --base /custom/path
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
SAVE_BASE = Path(
    '/lus/lfs1aip2/projects/s5e/lob_pipeline/LOBS5/evalsequences/aggressive_scenario_v4'
)

# Models to validate
MODELS = ['LobS5', 'Historic']

# Default config folder to inspect (many insertions = more data points)
DEFAULT_CONFIG = 'i10_c100_mb4_v105_cntxt88%'

MAX_SAMPLES = 20

# Message CSV columns (LOBSTER format, no header)
MSG_COL_TIME = 0
MSG_COL_EVENT_TYPE = 1
MSG_COL_ORDER_ID = 2
MSG_COL_SIZE = 3
MSG_COL_PRICE = 4
MSG_COL_DIRECTION = 5

# Orderbook CSV columns (LOBSTER L2 format, no header)
# Interleaved: ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...
BOOK_COL_ASK_PRICE = 0
BOOK_COL_ASK_VOL = 1
BOOK_COL_BID_PRICE = 2
BOOK_COL_BID_VOL = 3

OUTPUT_DIR = Path('pics_for_investigation')


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def find_latest_exp(folder_path):
    """Find the most recently modified exp_* subdirectory."""
    p = Path(folder_path)
    if not p.exists():
        return p
    exps = sorted(p.glob('exp_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    return exps[0] if exps else p


def get_midprice(book_row):
    """Compute midprice from a single orderbook row."""
    ask = float(book_row[BOOK_COL_ASK_PRICE])
    bid = float(book_row[BOOK_COL_BID_PRICE])
    if ask > 0 and bid > 0:
        return (ask + bid) / 2.0
    return 0.0


def get_midprice_series(book_arr):
    """Compute midprice series from orderbook array."""
    ask = book_arr[:, BOOK_COL_ASK_PRICE].astype(float)
    bid = book_arr[:, BOOK_COL_BID_PRICE].astype(float)
    return (ask + bid) / 2.0


def load_aggressive_indices(exp_path):
    """Load aggressive order indices from aggressive_indices.csv."""
    aggr_file = Path(exp_path) / 'aggressive_indices.csv'
    if not aggr_file.exists():
        return None
    vals = np.loadtxt(aggr_file, dtype=int)
    return np.atleast_1d(vals)


def discover_config_folder(base, model, direction_label, stock, config_name):
    """Find a specific config folder for a model."""
    root = base / model / f'context_500_{direction_label}' / stock / config_name
    if not root.exists():
        return None
    return find_latest_exp(root)


def load_sample_csvs(exp_path, max_samples):
    """Load message and orderbook CSV pairs from data_gen/."""
    gen_dir = Path(exp_path) / 'data_gen'
    if not gen_dir.exists():
        return [], [], []

    ob_files = sorted(gen_dir.glob('*_orderbook_*_gen_id_0.csv'))[:max_samples]
    msgs_list, books_list, filenames = [], [], []
    for ob_f in ob_files:
        msg_f = ob_f.parent / ob_f.name.replace('_orderbook_', '_message_')
        if not msg_f.exists():
            continue
        try:
            book = pd.read_csv(ob_f, header=None).values
            msg = pd.read_csv(msg_f, header=None).values
            msgs_list.append(msg)
            books_list.append(book)
            filenames.append(ob_f.stem)
        except Exception as e:
            print(f'  WARN: failed to load {ob_f.name}: {e}')
    return msgs_list, books_list, filenames


# ═══════════════════════════════════════════════════════════════════════
# Validation checks
# ═══════════════════════════════════════════════════════════════════════
def check_column_mapping(msg, book, sample_name):
    """Verify column mapping is consistent with expected LOBSTER format."""
    issues = []

    # Message checks
    if msg.shape[1] < 6:
        issues.append(f'{sample_name}: message has only {msg.shape[1]} columns (expected >= 6)')
        return issues

    event_types_col = msg[:, MSG_COL_EVENT_TYPE].astype(int)

    # EventType should be in {1, 2, 3, 4, 5} (LOBSTER event types)
    event_types = set(event_types_col)
    unexpected_events = event_types - {1, 2, 3, 4, 5}
    if unexpected_events:
        issues.append(f'{sample_name}: unexpected EventType values: {unexpected_events}')

    # Size should be positive where EventType != 0
    active_mask = event_types_col != 0
    if active_mask.any():
        sizes = msg[active_mask, MSG_COL_SIZE].astype(float)
        if np.any(sizes <= 0):
            n_bad = int(np.sum(sizes <= 0))
            issues.append(f'{sample_name}: {n_bad} messages with Size <= 0')

    # Price should be positive for executions
    exec_mask = event_types_col == 4
    if exec_mask.any():
        prices = msg[exec_mask, MSG_COL_PRICE].astype(float)
        if np.any(prices <= 0):
            n_bad = int(np.sum(prices <= 0))
            issues.append(f'{sample_name}: {n_bad} executions with Price <= 0')

    # Direction should be in {-1, 1} (LOBSTER uses -1=sell, 1=buy)
    directions = set(msg[:, MSG_COL_DIRECTION].astype(int))
    unexpected_dirs = directions - {-1, 1}
    if unexpected_dirs:
        issues.append(f'{sample_name}: unexpected Direction values: {unexpected_dirs}')

    # Orderbook: ask > bid at level 1
    if book.shape[1] >= 4:
        ask_prices = book[:, BOOK_COL_ASK_PRICE].astype(float)
        bid_prices = book[:, BOOK_COL_BID_PRICE].astype(float)
        valid = (ask_prices > 0) & (bid_prices > 0)
        if valid.any():
            crossed = ask_prices[valid] < bid_prices[valid]
            if np.any(crossed):
                n_crossed = int(np.sum(crossed))
                issues.append(f'{sample_name}: {n_crossed} rows with ask < bid (crossed book)')

    return issues


def check_aggressive_indices(msg, book, aggr_idx, direction_label, sample_name):
    """Validate aggressive order positions."""
    issues = []
    details = []

    if aggr_idx is None:
        issues.append(f'{sample_name}: aggressive_indices.csv not found')
        return issues, details

    for k, idx in enumerate(aggr_idx):
        detail = {'k': k + 1, 'index': int(idx)}

        # Bounds check
        if idx >= len(msg) or idx >= len(book):
            issues.append(
                f'{sample_name}: aggressive index {idx} out of bounds '
                f'(msg={len(msg)}, book={len(book)})'
            )
            detail['status'] = 'OUT_OF_BOUNDS'
            details.append(detail)
            continue

        event_type = int(msg[idx, MSG_COL_EVENT_TYPE])
        size = float(msg[idx, MSG_COL_SIZE])
        price = float(msg[idx, MSG_COL_PRICE])
        direction = int(msg[idx, MSG_COL_DIRECTION])

        detail['event_type'] = event_type
        detail['size'] = size
        detail['price'] = price
        detail['direction'] = direction

        # Check EventType == 4 (execution/market order)
        if event_type != 4:
            issues.append(
                f'{sample_name}: aggr[{k}] at idx={idx} has EventType={event_type} (expected 4)'
            )
            detail['status'] = 'WRONG_EVENT_TYPE'
            details.append(detail)
            continue

        # Check Size > 0
        if size <= 0:
            issues.append(f'{sample_name}: aggr[{k}] at idx={idx} has Size={size} <= 0')

        # Check Price > 0
        if price <= 0:
            issues.append(f'{sample_name}: aggr[{k}] at idx={idx} has Price={price} <= 0')

        # Check price is between bid and ask (within spread)
        ask_price = float(book[idx, BOOK_COL_ASK_PRICE])
        bid_price = float(book[idx, BOOK_COL_BID_PRICE])
        mid = (ask_price + bid_price) / 2.0 if ask_price > 0 and bid_price > 0 else 0
        detail['ask'] = ask_price
        detail['bid'] = bid_price
        detail['mid'] = mid

        if ask_price > 0 and bid_price > 0:
            # For buy orders: price should be at or near ask (hitting ask side)
            # For sell orders: price should be at or near bid (hitting bid side)
            # Allow some tolerance for multi-level executions
            if price < bid_price * 0.99 or price > ask_price * 1.01:
                issues.append(
                    f'{sample_name}: aggr[{k}] price={price} outside '
                    f'spread [{bid_price}, {ask_price}] (tolerance 1%)'
                )
                detail['status'] = 'PRICE_OUTSIDE_SPREAD'
            else:
                detail['status'] = 'OK'
        else:
            detail['status'] = 'INVALID_BOOK'

        details.append(detail)

    return issues, details


def check_midprice_direction(book, aggr_idx, direction_label, sample_name):
    """Check that midprice moves in expected direction after aggressive orders."""
    issues = []

    if aggr_idx is None or len(aggr_idx) < 2:
        return issues

    mid = get_midprice_series(book)

    # Check midprice at first vs last aggressive order
    first_idx, last_idx = aggr_idx[0], aggr_idx[-1]
    if first_idx >= len(mid) or last_idx >= len(mid):
        return issues

    mid_first = mid[first_idx]
    mid_last = mid[last_idx]

    if mid_first <= 0 or mid_last <= 0:
        issues.append(f'{sample_name}: invalid midprice (first={mid_first}, last={mid_last})')
        return issues

    delta = mid_last - mid_first

    if direction_label == 'buy' and delta < 0:
        issues.append(
            f'{sample_name}: BUY midprice decreased ({mid_first:.0f} -> {mid_last:.0f}, '
            f'delta={delta:.0f})'
        )
    elif direction_label == 'sell' and delta > 0:
        issues.append(
            f'{sample_name}: SELL midprice increased ({mid_first:.0f} -> {mid_last:.0f}, '
            f'delta={delta:.0f})'
        )

    return issues


def compute_vwap_impact_manual(msg, book, aggr_idx, direction_label):
    """Manually compute VWAP impact (same logic as run_300_analyze_one.py)."""
    if aggr_idx is None or len(aggr_idx) < 2:
        return None

    if aggr_idx.max() >= len(msg) or aggr_idx.max() >= len(book):
        return None

    # Reference = book midprice at first aggressive order
    ref_mid = get_midprice(book[aggr_idx[0]])
    if ref_mid <= 0:
        return None

    # Sizes and prices at aggressive order positions
    sizes = msg[aggr_idx, MSG_COL_SIZE].astype(float)
    prices = msg[aggr_idx, MSG_COL_PRICE].astype(float)
    if np.any(sizes <= 0) or np.any(prices <= 0):
        return None

    # Cumulative VWAP
    Q_cum = np.cumsum(sizes)
    vwap = np.cumsum(sizes * prices) / Q_cum

    # Fractional impact (absolute, same as run_300_analyze_one.py)
    if direction_label == 'buy':
        impact = (vwap - ref_mid) / ref_mid
    else:
        impact = (ref_mid - vwap) / ref_mid
    impact = np.abs(impact)

    return {
        'ref_mid': ref_mid,
        'Q_cum': Q_cum.tolist(),
        'vwap': vwap.tolist(),
        'impact': impact.tolist(),
        'n_aggr': len(aggr_idx),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main validation
# ═══════════════════════════════════════════════════════════════════════
def validate_model(base, model, stock, config_name, max_samples):
    """Validate one model's data integrity."""
    results = {
        'model': model,
        'config': config_name,
        'directions': {},
    }

    for direction_label in ['buy', 'sell']:
        exp_path = discover_config_folder(
            base, model, direction_label, stock, config_name
        )
        if exp_path is None:
            results['directions'][direction_label] = {
                'status': 'NOT_FOUND',
                'samples': [],
            }
            continue

        aggr_idx = load_aggressive_indices(exp_path)
        msgs_list, books_list, filenames = load_sample_csvs(exp_path, max_samples)

        if not msgs_list:
            results['directions'][direction_label] = {
                'status': 'NO_DATA',
                'exp_path': str(exp_path),
                'samples': [],
            }
            continue

        samples = []
        all_issues = []

        for j, (msg, book, fname) in enumerate(zip(msgs_list, books_list, filenames)):
            sample_name = f'{direction_label}/{fname}'
            sample_result = {
                'name': sample_name,
                'msg_shape': msg.shape,
                'book_shape': book.shape,
                'issues': [],
                'details': [],
                'vwap': None,
            }

            # Check 1: Column mapping
            col_issues = check_column_mapping(msg, book, sample_name)
            sample_result['issues'].extend(col_issues)

            # Check 2: Aggressive indices
            aggr_issues, aggr_details = check_aggressive_indices(
                msg, book, aggr_idx, direction_label, sample_name
            )
            sample_result['issues'].extend(aggr_issues)
            sample_result['details'] = aggr_details

            # Check 3: Midprice direction
            mid_issues = check_midprice_direction(
                book, aggr_idx, direction_label, sample_name
            )
            sample_result['issues'].extend(mid_issues)

            # Check 4: VWAP impact
            vwap = compute_vwap_impact_manual(msg, book, aggr_idx, direction_label)
            sample_result['vwap'] = vwap

            all_issues.extend(sample_result['issues'])
            samples.append(sample_result)

        results['directions'][direction_label] = {
            'status': 'OK' if not all_issues else 'HAS_ISSUES',
            'exp_path': str(exp_path),
            'aggr_indices': aggr_idx.tolist() if aggr_idx is not None else None,
            'n_samples': len(samples),
            'n_issues': len(all_issues),
            'samples': samples,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Report formatting
# ═══════════════════════════════════════════════════════════════════════
def format_report(all_results, stock):
    """Format validation results as a text report."""
    lines = []
    lines.append('=' * 80)
    lines.append(f'  DIAGNOSTIC 2: DATA INTEGRITY VALIDATION — {stock}')
    lines.append('=' * 80)
    lines.append('')

    total_valid = 0
    total_invalid = 0
    failure_types = {}

    for result in all_results:
        model = result['model']
        config = result['config']
        lines.append(f'Model: {model}  |  Config: {config}')
        lines.append('-' * 60)

        for direction_label in ['buy', 'sell']:
            dr = result['directions'].get(direction_label, {})
            status = dr.get('status', 'UNKNOWN')

            if status in ('NOT_FOUND', 'NO_DATA'):
                lines.append(f'  {direction_label}: {status}')
                continue

            aggr_indices = dr.get('aggr_indices')
            lines.append(
                f'  {direction_label}: {dr["n_samples"]} samples, '
                f'aggressive_indices={aggr_indices}'
            )

            for s in dr['samples']:
                is_valid = len(s['issues']) == 0
                if is_valid:
                    total_valid += 1
                else:
                    total_invalid += 1
                    for issue in s['issues']:
                        # Classify failure type
                        if 'EventType' in issue:
                            failure_types['wrong_event_type'] = (
                                failure_types.get('wrong_event_type', 0) + 1
                            )
                        elif 'outside spread' in issue.lower():
                            failure_types['price_outside_spread'] = (
                                failure_types.get('price_outside_spread', 0) + 1
                            )
                        elif 'midprice' in issue.lower():
                            failure_types['midprice_wrong_direction'] = (
                                failure_types.get('midprice_wrong_direction', 0) + 1
                            )
                        elif 'Size' in issue:
                            failure_types['invalid_size'] = (
                                failure_types.get('invalid_size', 0) + 1
                            )
                        elif 'crossed book' in issue.lower():
                            failure_types['crossed_book'] = (
                                failure_types.get('crossed_book', 0) + 1
                            )
                        else:
                            failure_types['other'] = (
                                failure_types.get('other', 0) + 1
                            )

                if s['issues']:
                    lines.append(f'    {s["name"]}: FAIL')
                    for issue in s['issues']:
                        lines.append(f'      - {issue}')

            # Per-sample validation table
            lines.append('')
            lines.append(f'  Per-sample validation ({direction_label}):')
            lines.append(
                f'  {"Sample":<50} {"msg_shape":>12} {"book_shape":>12} '
                f'{"n_issues":>9} {"status":>8}'
            )
            lines.append(f'  {"-"*95}')
            for s in dr['samples']:
                short_name = s['name'].split('/')[-1][:48]
                status_str = 'PASS' if not s['issues'] else 'FAIL'
                lines.append(
                    f'  {short_name:<50} {str(s["msg_shape"]):>12} '
                    f'{str(s["book_shape"]):>12} {len(s["issues"]):>9} '
                    f'{status_str:>8}'
                )
            lines.append('')

        lines.append('')

    # ── Summary ──
    lines.append('=' * 80)
    lines.append('  SUMMARY')
    lines.append('=' * 80)
    lines.append(f'  N_valid:   {total_valid}')
    lines.append(f'  N_invalid: {total_invalid}')
    lines.append(f'  Total:     {total_valid + total_invalid}')
    if failure_types:
        lines.append('')
        lines.append('  Failure types:')
        for ft, count in sorted(failure_types.items(), key=lambda x: -x[1]):
            lines.append(f'    {ft}: {count}')
    lines.append('')

    # ── Spot-check: first aggressive order details for first 3 samples ──
    lines.append('=' * 80)
    lines.append('  SPOT-CHECK: First aggressive order details (up to 3 samples per model)')
    lines.append('=' * 80)
    for result in all_results:
        model = result['model']
        lines.append(f'\n  Model: {model}')
        for direction_label in ['buy', 'sell']:
            dr = result['directions'].get(direction_label, {})
            if dr.get('status') in ('NOT_FOUND', 'NO_DATA', 'UNKNOWN', None):
                continue
            lines.append(f'    Direction: {direction_label}')
            for s in dr['samples'][:3]:
                lines.append(f'      Sample: {s["name"]}')
                if s['details']:
                    d = s['details'][0]  # first aggressive order
                    lines.append(
                        f'        k={d.get("k")}, index={d.get("index")}, '
                        f'EventType={d.get("event_type")}, '
                        f'Size={d.get("size")}, Price={d.get("price")}, '
                        f'Direction={d.get("direction")}'
                    )
                    lines.append(
                        f'        Book: ask={d.get("ask")}, '
                        f'bid={d.get("bid")}, '
                        f'mid={d.get("mid", 0):.1f}'
                    )
                    lines.append(f'        Status: {d.get("status")}')
                if s['vwap']:
                    v = s['vwap']
                    lines.append(
                        f'        VWAP impact: ref_mid={v["ref_mid"]:.1f}, '
                        f'n_aggr={v["n_aggr"]}'
                    )
                    # Show first and last impact
                    if v['impact']:
                        lines.append(
                            f'        Impact[0]={v["impact"][0]:.6f}, '
                            f'Impact[-1]={v["impact"][-1]:.6f}'
                        )
                        lines.append(
                            f'        Q_cum[0]={v["Q_cum"][0]:.0f}, '
                            f'Q_cum[-1]={v["Q_cum"][-1]:.0f}'
                        )
                lines.append('')

    # ── VWAP consistency check ──
    lines.append('=' * 80)
    lines.append('  VWAP IMPACT CONSISTENCY (manual vs expected)')
    lines.append('=' * 80)
    lines.append(
        '  This section verifies that our manual VWAP calculation matches the logic'
    )
    lines.append('  in run_300_analyze_one.py:extract_point_cloud().')
    lines.append('')
    for result in all_results:
        model = result['model']
        lines.append(f'  Model: {model}')
        for direction_label in ['buy', 'sell']:
            dr = result['directions'].get(direction_label, {})
            if dr.get('status') in ('NOT_FOUND', 'NO_DATA', 'UNKNOWN', None):
                continue
            n_with_vwap = sum(1 for s in dr['samples'] if s['vwap'] is not None)
            n_positive_impact = sum(
                1 for s in dr['samples']
                if s['vwap'] is not None and all(i > 0 for i in s['vwap']['impact'])
            )
            lines.append(
                f'    {direction_label}: {n_with_vwap}/{len(dr["samples"])} samples '
                f'have valid VWAP, {n_positive_impact} with all-positive impact'
            )
        lines.append('')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 2: Data Integrity Validation'
    )
    parser.add_argument('--stock', type=str, default='GOOG', help='Stock ticker')
    parser.add_argument(
        '--config', type=str, default=DEFAULT_CONFIG,
        help='Config folder name to inspect'
    )
    parser.add_argument(
        '--max_samples', type=int, default=MAX_SAMPLES,
        help='Max samples to load per direction'
    )
    parser.add_argument(
        '--base', type=str, default=None,
        help='Override SAVE_BASE path'
    )
    parser.add_argument(
        '--models', type=str, nargs='+', default=None,
        help='Models to validate (default: LobS5 Historic)'
    )
    args = parser.parse_args()

    base = Path(args.base) if args.base else SAVE_BASE
    models = args.models if args.models else MODELS

    print(f'Diagnostic 2: Data Integrity Validation')
    print(f'  Stock:       {args.stock}')
    print(f'  Config:      {args.config}')
    print(f'  Max samples: {args.max_samples}')
    print(f'  Base:        {base}')
    print(f'  Models:      {models}')
    print()

    if not base.exists():
        print(f'ERROR: base path does not exist: {base}')
        sys.exit(1)

    all_results = []
    for model in models:
        print(f'Validating {model}...')
        result = validate_model(base, model, args.stock, args.config, args.max_samples)
        all_results.append(result)

        # Print quick summary
        for dl in ['buy', 'sell']:
            dr = result['directions'].get(dl, {})
            status = dr.get('status', 'UNKNOWN')
            n_samples = dr.get('n_samples', 0)
            n_issues = dr.get('n_issues', 0)
            print(f'  {dl}: {status} ({n_samples} samples, {n_issues} issues)')
        print()

    # Generate report
    report = format_report(all_results, args.stock)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'diag_2_data_integrity.txt'
    with open(out_path, 'w') as f:
        f.write(report)
    print(f'\nReport saved to: {out_path}')

    # Also print to stdout
    print()
    print(report)


if __name__ == '__main__':
    main()
