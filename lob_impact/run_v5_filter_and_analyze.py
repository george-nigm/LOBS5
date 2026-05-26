#!/usr/bin/env python3
"""Filter v5 pickle to specific Q value and run standard analysis."""
import argparse, pickle, sys, os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--stock', required=True)
    parser.add_argument('--q_filter', type=int, required=True, help='Keep only folders with this Q value')
    parser.add_argument('--pickle_base', required=True)
    parser.add_argument('--daily_hl', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    pickle_base = Path(args.pickle_base)
    raw_pkl = pickle_base / args.stock / f'{args.model}.pkl'

    print(f'Loading {raw_pkl} ...')
    with open(raw_pkl, 'rb') as f:
        md = pickle.load(f)

    grid = md['grid']
    data = md['data']
    q_tag = f'Q{args.q_filter}'
    print(f'  Original: {len(grid)} folders: {list(grid["folder"])}')

    # Filter
    mask = grid['folder'].str.contains(q_tag)
    grid_f = grid[mask].reset_index(drop=True)
    data_f = {k: v for k, v in data.items() if q_tag in k}
    print(f'  Filtered to {q_tag}: {len(grid_f)} folders: {list(grid_f["folder"])}')

    if len(grid_f) == 0:
        print('ERROR: no folders match filter')
        sys.exit(1)

    # Save filtered pickle
    filt_dir = pickle_base.parent / f'pickles_Q{args.q_filter}' / args.stock
    filt_dir.mkdir(parents=True, exist_ok=True)
    filt_pkl = filt_dir / f'{args.model}.pkl'
    md_f = dict(model=args.model, stock=args.stock, grid=grid_f, data=data_f)
    with open(filt_pkl, 'wb') as f:
        pickle.dump(md_f, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Saved filtered pickle: {filt_pkl} ({filt_pkl.stat().st_size/1e6:.0f} MB)')

    # Run standard analysis
    print(f'\nRunning analysis...')
    from lob_impact.run_300_analyze_one import run
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run(args.model, args.stock,
        daily_hl_path=args.daily_hl,
        pickle_base=filt_dir.parent,
        out_dir=Path(args.out_dir))

if __name__ == '__main__':
    main()
