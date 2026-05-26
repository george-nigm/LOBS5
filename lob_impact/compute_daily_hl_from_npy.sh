#!/bin/bash
#SBATCH --job-name=daily_hl
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=10G
#SBATCH --output=logs/daily_hl_%j.out
#SBATCH --error=logs/daily_hl_%j.err

set -euo pipefail
cd /home/s5e/georgenigm.s5e/LOBS5_11_march
source ~/.bashrc
conda activate lob

python3 -c "
import numpy as np, pandas as pd, re
from pathlib import Path

data_base = Path('/lus/lfs1aip2/projects/s5e/lob_pipeline/data')

for stock in ['META', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'MU', 'NFLX']:
    stock_dir = data_base / f'{stock}_jan2026'
    if not stock_dir.exists():
        print(f'SKIP {stock}: no data dir')
        continue

    msg_files = sorted(stock_dir.glob(f'{stock}_*_message_*_proc.npy'))
    rows = []
    for f in msg_files:
        msgs = np.load(f)
        # col1=event_type, col3=price (LOBSTER ×10000), col5=abs_size
        exec_mask = (msgs[:, 1] == 4) | (msgs[:, 1] == 5)
        if exec_mask.sum() == 0:
            continue
        exec_prices = msgs[exec_mask, 3].astype(float)
        exec_sizes = msgs[exec_mask, 5].astype(float)

        # Remove outliers (IQR)
        q1, q3 = np.percentile(exec_prices, [25, 75])
        iqr = q3 - q1
        mask = (exec_prices >= q1 - 1.5*iqr) & (exec_prices <= q3 + 1.5*iqr)
        clean_prices = exec_prices[mask]

        highest = int(clean_prices.max()) if len(clean_prices) > 0 else 0
        lowest = int(clean_prices.min()) if len(clean_prices) > 0 else 0
        vol_sum = int(exec_sizes.sum())

        # Extract day
        m = re.search(r'(\d{4}-\d{2}-\d{2})', f.name)
        day = m.group(1) if m else 'unknown'

        rows.append(dict(
            filename=f.name.replace('_proc.npy', '.csv'),
            highest_price=highest, lowest_price=lowest,
            execution_sum=vol_sum, day=day))

    df = pd.DataFrame(rows)
    out = f'lob_impact/daily_h_l_{stock}.csv'
    df.to_csv(out, index=False)
    print(f'{stock}: {len(df)} days → {out}')
"
echo "Done: $(date)"
