#!/usr/bin/env python3
"""
Migrate market impact experiment results into grid-specific subdirectories.

Before:
    evalsequences/{scenario}/context_500_buy/i1_c10_mb20_v75_cntxt44%/
    evalsequences/{scenario}/context_500_buy/i83_c5_mb5_v75_cntxt99%/

After:
    evalsequences/{scenario}/c10x_v2/context_500_buy/i1_c10_mb20_v75_cntxt44%/
    evalsequences/{scenario}/v3/context_500_buy/i83_c5_mb5_v75_cntxt99%/

Usage:
    python lob_impact/migrate_grids.py --eval-base /app/output/evalsequences --dry-run
    python lob_impact/migrate_grids.py --eval-base /app/output/evalsequences --execute
"""
import argparse
import re
import shutil
from pathlib import Path

# ── Known (i, c, mb) triples per grid ────────────────────────────
# Derived from configs_context_500_c10x_v2, configs_s5_v3, configs_s5_v4

C10X_V2_TRIPLES = {
    (1, 10, 20), (2, 20, 10), (2, 20, 15), (2, 20, 20),
    (3, 30, 5), (3, 30, 10), (3, 30, 15),
    (4, 40, 10), (5, 50, 5), (9, 90, 5),
}

V3_TRIPLES = {
    (4, 5, 100), (6, 5, 75), (9, 5, 50), (19, 5, 25),
    (23, 5, 20), (45, 5, 10), (83, 5, 5),
}

V4_TRIPLES = {
    (4, 40, 100), (6, 60, 75), (9, 90, 50), (19, 190, 25),
    (23, 230, 20), (45, 450, 10), (83, 830, 5),
}

SCENARIOS = [
    'aggressive_scenario',
    'cgan_aggressive_scenario',
    'cst_scenario',
    'historic_scenario',
    'heuristic_scenario',
]

FOLDER_RE = re.compile(r'^i(\d+)_c(\d+)_mb(\d+)(_v\d+)?_cntxt.+$')


def classify_folder(name: str) -> str | None:
    """Return grid name ('c10x_v2', 'v3', 'v4') or None if unrecognized."""
    m = FOLDER_RE.match(name)
    if not m:
        return None
    i, c, mb = int(m.group(1)), int(m.group(2)), int(m.group(3))
    triple = (i, c, mb)
    if triple in C10X_V2_TRIPLES:
        return 'c10x_v2'
    if triple in V3_TRIPLES:
        return 'v3'
    if triple in V4_TRIPLES:
        return 'v4'
    return None


def migrate(eval_base: Path, execute: bool):
    stats = {'moved': 0, 'skipped': 0, 'unknown': 0, 'already': 0}

    for scenario in SCENARIOS:
        scenario_dir = eval_base / scenario
        if not scenario_dir.exists():
            continue

        for context_dir in sorted(scenario_dir.iterdir()):
            if not context_dir.is_dir() or not context_dir.name.startswith('context_'):
                continue

            context_name = context_dir.name  # e.g. "context_500_buy"
            folders = [f for f in sorted(context_dir.iterdir()) if f.is_dir()]

            for folder in folders:
                grid = classify_folder(folder.name)
                if grid is None:
                    print(f"  UNKNOWN: {scenario}/{context_name}/{folder.name}")
                    stats['unknown'] += 1
                    continue

                dest_parent = scenario_dir / grid / context_name
                dest = dest_parent / folder.name

                if dest.exists():
                    stats['already'] += 1
                    continue

                action = "MOVE" if execute else "WOULD MOVE"
                print(f"  {action}: {scenario}/{context_name}/{folder.name} -> {scenario}/{grid}/{context_name}/{folder.name}")

                if execute:
                    dest_parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(folder), str(dest))

                stats['moved'] += 1

        # Create empty v4 dirs for future experiments
        for grid in ('c10x_v2', 'v3', 'v4'):
            for suffix in ('buy', 'sell'):
                d = scenario_dir / grid / f'context_500_{suffix}'
                if execute:
                    d.mkdir(parents=True, exist_ok=True)

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-base', type=Path, required=True, help='Path to evalsequences/')
    parser.add_argument('--execute', action='store_true', help='Actually move folders (default: dry-run)')
    args = parser.parse_args()

    if not args.eval_base.exists():
        print(f"ERROR: {args.eval_base} does not exist")
        return

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"=== Migration: {mode} ===")
    print(f"Base: {args.eval_base}\n")

    stats = migrate(args.eval_base, args.execute)

    print(f"\n=== Summary ===")
    print(f"  {'Moved' if args.execute else 'Would move'}: {stats['moved']}")
    print(f"  Already in place: {stats['already']}")
    print(f"  Unknown (skipped): {stats['unknown']}")

    if not args.execute and stats['moved'] > 0:
        print(f"\nRe-run with --execute to apply.")


if __name__ == '__main__':
    main()
