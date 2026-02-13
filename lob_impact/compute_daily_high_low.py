"""Compute daily High/Low prices and execution volume from raw LOBSTER message files.

Usage:
    python lob_impact/compute_daily_high_low.py --data_dir /homes/groups/finance/data/rawLOBSTER/GOOG/JAN2023
    python lob_impact/compute_daily_high_low.py --data_dir /homes/groups/finance/data/rawLOBSTER/GOOG/JAN2023 --output daily_h_l.csv

Reads LOBSTER message CSVs, computes per-day:
  - highest_price, lowest_price (from all orders with valid prices)
  - execution_sum (total executed volume, Event_Type == 4)
  - all Order_IDs (for sample-to-day matching)
"""

import argparse
import glob
import os
import pandas as pd
import numpy as np


PRICE_MIN = 500_000
PRICE_MAX = 20_000_000
EXECUTION_EVENT_TYPE = 4

LOBSTER_COLUMNS = [
    "Time", "Event_Type", "Order_ID", "Size", "Price", "Direction", "Extra"
]


def compute_daily_stats(data_dir: str) -> pd.DataFrame:
    message_files = sorted(glob.glob(os.path.join(data_dir, "*message*.csv")))

    if not message_files:
        raise FileNotFoundError(f"No message CSV files found in {data_dir}")

    print(f"Found {len(message_files)} message files in {data_dir}")

    records = []
    for file_path in message_files:
        filename = os.path.basename(file_path)
        print(f"  Processing {filename} ...", end=" ")

        df = pd.read_csv(
            file_path, header=None, names=LOBSTER_COLUMNS, low_memory=False,
        )

        valid_prices = df.loc[
            (df["Price"] >= PRICE_MIN) & (df["Price"] <= PRICE_MAX), "Price"
        ]

        if len(valid_prices) > 0:
            highest_price = int(valid_prices.max())
            lowest_price = int(valid_prices.min())
        else:
            highest_price = float("nan")
            lowest_price = float("nan")

        execution_sum = int(
            df.loc[df["Event_Type"] == EXECUTION_EVENT_TYPE, "Size"].sum()
        )

        all_message_ids = df["Order_ID"].unique().astype(int).tolist()

        records.append({
            "filename": filename,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "all_message_ids": all_message_ids,
            "execution_sum": execution_sum,
        })

        print(f"H={highest_price}  L={lowest_price}  exec_vol={execution_sum}")

    result = pd.DataFrame(records)

    # Summary
    valid = result.dropna(subset=["highest_price", "lowest_price"])
    ln_hl = np.log(valid["highest_price"] / valid["lowest_price"])
    print(f"\n--- Summary ({len(result)} days) ---")
    print(f"  Mean(ln(H/L)) = {ln_hl.mean():.6f}")
    print(f"  Std(ln(H/L))  = {ln_hl.std():.6f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute daily High/Low from LOBSTER data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/homes/groups/finance/data/rawLOBSTER/GOOG/JAN2023",
        help="Path to folder with raw LOBSTER message CSVs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: daily_h_l.csv next to this script)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "daily_h_l.csv")

    df = compute_daily_stats(args.data_dir)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
