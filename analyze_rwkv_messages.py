#!/usr/bin/env python3
"""
Comprehensive analysis of RWKV experiment message CSV files.
Checks for:
1. Negative order IDs
2. Duplicate countdown IDs for type 1 messages
3. Cancel message validity (type 3)
4. Aggressive order IDs at positions 5 and 11 (type 4)
5. Event type counts
"""

import os
import csv
from collections import defaultdict, Counter
from pathlib import Path

data_dir = "/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences/rwkv_goog2022_benchmark/exp_4_20260217_112951/data_gen/"

# Find all message CSV files
message_files = sorted([f for f in os.listdir(data_dir) if '_message_' in f and f.endswith('.csv')])

print(f"Found {len(message_files)} message CSV files")
print("=" * 100)

# Global statistics
global_event_counts = Counter()
global_negative_ids = []
global_cancel_issues = []
global_aggressive_orders = []

for file_idx, filename in enumerate(message_files, 1):
    filepath = os.path.join(data_dir, filename)

    print(f"\n{'=' * 100}")
    print(f"FILE {file_idx}/16: {filename}")
    print(f"{'=' * 100}")

    # Per-file statistics
    event_counts = Counter()
    negative_ids = []
    type1_countdown_ids = []  # Track countdown IDs for type 1
    type1_order_ids = {}  # Map countdown ID to actual order ID
    cancel_issues = []
    aggressive_orders = []

    # Read and parse the CSV
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse each line
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Parse the message format: time,event_type,order_id,size,price,direction
        parts = line.split(',')
        if len(parts) < 6:
            continue

        try:
            time = float(parts[0])
            event_type = int(parts[1])
            order_id = int(parts[2])
            size = int(parts[3])
            price = int(parts[4])
            direction = int(parts[5])
        except (ValueError, IndexError):
            continue

        # Count event types
        event_counts[event_type] += 1
        global_event_counts[event_type] += 1

        # Check for negative order IDs
        if order_id < 0:
            neg_info = {
                'file': filename,
                'line': line_num,
                'event_type': event_type,
                'order_id': order_id,
                'size': size,
                'price': price,
                'direction': direction
            }
            negative_ids.append(neg_info)
            global_negative_ids.append(neg_info)

        # Track type 1 (limit order submission) messages
        if event_type == 1:
            type1_countdown_ids.append(order_id)
            type1_order_ids[order_id] = line_num

        # Check type 3 (cancel) messages
        if event_type == 3:
            # Check if this is a conditioning-era ID (large number like 111xxxxx or similar)
            is_conditioning_id = order_id > 100000  # Heuristic for historical IDs
            is_negative = order_id < 0
            was_generated = order_id in type1_order_ids

            if not is_conditioning_id and not is_negative and not was_generated:
                cancel_info = {
                    'file': filename,
                    'line': line_num,
                    'order_id': order_id,
                    'was_generated': was_generated,
                    'is_conditioning': is_conditioning_id,
                    'is_negative': is_negative
                }
                cancel_issues.append(cancel_info)
                global_cancel_issues.append(cancel_info)

        # Check type 4 (aggressive/market order) at positions 6 and 12
        if event_type == 4 and line_num in [6, 12]:
            agg_info = {
                'file': filename,
                'line': line_num,
                'order_id': order_id,
                'size': size,
                'price': price,
                'direction': direction
            }
            aggressive_orders.append(agg_info)
            global_aggressive_orders.append(agg_info)

    # Check for duplicate countdown IDs
    countdown_counts = Counter(type1_countdown_ids)
    duplicates = {k: v for k, v in countdown_counts.items() if v > 1}

    # Print per-file results
    print(f"\n--- EVENT TYPE COUNTS ---")
    for et in sorted(event_counts.keys()):
        print(f"  Type {et}: {event_counts[et]}")

    print(f"\n--- NEGATIVE ORDER IDs ---")
    if negative_ids:
        for neg in negative_ids:
            print(f"  Line {neg['line']}: type={neg['event_type']}, id={neg['order_id']}, size={neg['size']}, price={neg['price']}, dir={neg['direction']}")
    else:
        print("  None found")

    print(f"\n--- DUPLICATE COUNTDOWN IDs (Type 1) ---")
    if duplicates:
        for oid, count in duplicates.items():
            print(f"  ID {oid}: appears {count} times")
    else:
        print("  None found")

    print(f"\n--- CANCEL MESSAGE VALIDITY (Type 3) ---")
    print(f"  Total type 3 messages: {event_counts[3]}")
    print(f"  Issues found: {len(cancel_issues)}")
    if cancel_issues:
        for issue in cancel_issues[:5]:  # Show first 5
            print(f"    Line {issue['line']}: ID {issue['order_id']} (generated={issue['was_generated']}, conditioning={issue['is_conditioning']}, negative={issue['is_negative']})")
        if len(cancel_issues) > 5:
            print(f"    ... and {len(cancel_issues) - 5} more")

    print(f"\n--- AGGRESSIVE ORDERS (Type 4 at lines 6, 12) ---")
    if aggressive_orders:
        for agg in aggressive_orders:
            print(f"  Line {agg['line']}: ID={agg['order_id']}, size={agg['size']}, price={agg['price']}, dir={agg['direction']}")
    else:
        print("  None found at expected positions")

# Global summary
print(f"\n\n{'=' * 100}")
print(f"GLOBAL SUMMARY (ALL 16 FILES)")
print(f"{'=' * 100}")

print(f"\n--- TOTAL EVENT TYPE COUNTS ---")
for et in sorted(global_event_counts.keys()):
    print(f"  Type {et}: {global_event_counts[et]}")

print(f"\n--- TOTAL NEGATIVE ORDER IDs: {len(global_negative_ids)} ---")
if global_negative_ids:
    # Group by order_id value
    neg_grouped = defaultdict(list)
    for neg in global_negative_ids:
        neg_grouped[neg['order_id']].append(neg)

    for oid in sorted(neg_grouped.keys()):
        occurrences = neg_grouped[oid]
        print(f"  ID {oid}: {len(occurrences)} occurrence(s)")
        for occ in occurrences[:3]:
            print(f"    {occ['file']}, line {occ['line']}, type {occ['event_type']}")

print(f"\n--- TOTAL CANCEL ISSUES: {len(global_cancel_issues)} ---")

print(f"\n--- AGGRESSIVE ORDERS AT EXPECTED POSITIONS ---")
print(f"  Total found: {len(global_aggressive_orders)}")
if global_aggressive_orders:
    # Group by position
    pos_6 = [a for a in global_aggressive_orders if a['line'] == 6]
    pos_12 = [a for a in global_aggressive_orders if a['line'] == 12]

    print(f"  Position 6: {len(pos_6)} orders")
    print(f"  Position 12: {len(pos_12)} orders")

    # Check if all have ID 17 and 11
    ids_6 = set(a['order_id'] for a in pos_6)
    ids_12 = set(a['order_id'] for a in pos_12)

    print(f"  IDs at position 6: {ids_6}")
    print(f"  IDs at position 12: {ids_12}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
