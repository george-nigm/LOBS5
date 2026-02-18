#!/usr/bin/env python3
"""
Detailed analysis of RWKV experiment message CSV files.
Provides comprehensive report on negative IDs and their context.
"""

import os
from collections import defaultdict, Counter

data_dir = "/scratch/local/homes/80/georgenigm/LOBS5/output/evalsequences/rwkv_goog2022_benchmark/exp_4_20260217_112951/data_gen/"

# Find all message CSV files
message_files = sorted([f for f in os.listdir(data_dir) if '_message_' in f and f.endswith('.csv')])

print("=" * 100)
print("DETAILED ANALYSIS: NEGATIVE ORDER IDs")
print("=" * 100)

all_negative_details = []

for file_idx, filename in enumerate(message_files, 1):
    filepath = os.path.join(data_dir, filename)

    # Read and parse the CSV
    with open(filepath, 'r') as f:
        lines = f.readlines()

    messages = []
    type1_ids = set()

    # Parse each line
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

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

        msg = {
            'line': line_num,
            'time': time,
            'event_type': event_type,
            'order_id': order_id,
            'size': size,
            'price': price,
            'direction': direction
        }
        messages.append(msg)

        if event_type == 1:
            type1_ids.add(order_id)

    # Find negative IDs and their context
    for i, msg in enumerate(messages):
        if msg['order_id'] < 0:
            # Get context (2 messages before and after)
            context_start = max(0, i - 2)
            context_end = min(len(messages), i + 3)
            context = messages[context_start:context_end]

            detail = {
                'file': filename,
                'negative_msg': msg,
                'context': context,
                'all_type1_ids': sorted(type1_ids)
            }
            all_negative_details.append(detail)

# Print detailed report
for idx, detail in enumerate(all_negative_details, 1):
    print(f"\n{'=' * 100}")
    print(f"NEGATIVE ID #{idx}: {detail['file']}")
    print(f"{'=' * 100}")

    neg_msg = detail['negative_msg']
    print(f"\nNEGATIVE ORDER MESSAGE:")
    print(f"  Line {neg_msg['line']}: time={neg_msg['time']:.2f}, type={neg_msg['event_type']}, " +
          f"id={neg_msg['order_id']}, size={neg_msg['size']}, price={neg_msg['price']}, dir={neg_msg['direction']}")

    print(f"\nCONTEXT (2 before, 2 after):")
    for ctx_msg in detail['context']:
        marker = ">>> " if ctx_msg['order_id'] < 0 else "    "
        print(f"{marker}Line {ctx_msg['line']}: time={ctx_msg['time']:.2f}, type={ctx_msg['event_type']}, " +
              f"id={ctx_msg['order_id']}, size={ctx_msg['size']}, price={ctx_msg['price']}, dir={ctx_msg['direction']}")

    print(f"\nALL TYPE 1 IDs IN THIS FILE:")
    print(f"  {detail['all_type1_ids']}")

# Summary statistics
print(f"\n\n{'=' * 100}")
print("SUMMARY STATISTICS")
print(f"{'=' * 100}")

event_type_counts = Counter()
negative_id_values = Counter()

for detail in all_negative_details:
    neg_msg = detail['negative_msg']
    event_type_counts[neg_msg['event_type']] += 1
    negative_id_values[neg_msg['order_id']] += 1

print(f"\nTotal negative IDs found: {len(all_negative_details)}")

print(f"\nBy event type:")
for et in sorted(event_type_counts.keys()):
    print(f"  Type {et}: {event_type_counts[et]}")

print(f"\nBy order ID value:")
for oid in sorted(negative_id_values.keys()):
    print(f"  ID {oid}: {negative_id_values[oid]} occurrence(s)")

print(f"\n{'=' * 100}")
