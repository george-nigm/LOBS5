"""Match experiment samples to daily High/Low from daily_h_l.csv.

Parses the day from sample filenames (e.g. GOOG_2023-01-03_message_real_id_10.csv)
and joins with daily_h_l.csv to produce sample_day_map.csv.

Usage:
    python lob_impact/create_sample_day_map.py
    python lob_impact/create_sample_day_map.py --experiments_dir /path/to/context_500_buy --daily_hl /path/to/daily_h_l.csv
"""

import argparse
import os
import re
import glob
import pandas as pd


def parse_samples(experiment_dir: str) -> pd.DataFrame:
    """Extract sample_id and day from filenames in data_cond/."""
    cond_dir = os.path.join(experiment_dir, "data_cond")
    if not os.path.isdir(cond_dir):
        raise FileNotFoundError(f"data_cond not found in {experiment_dir}")

    pattern = re.compile(
        r"^(?P<stock>[A-Z]+)_(?P<day>\d{4}-\d{2}-\d{2})_message_real_id_(?P<sample_id>\d+)\.csv$"
    )

    records = []
    for fname in os.listdir(cond_dir):
        m = pattern.match(fname)
        if m:
            records.append({
                "sample_id": int(m.group("sample_id")),
                "day": m.group("day"),
                "stock": m.group("stock"),
            })

    if not records:
        raise ValueError(f"No matching message files in {cond_dir}")

    return pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)


def load_daily_hl(path: str) -> pd.DataFrame:
    """Load daily_h_l.csv and extract day from filename."""
    df = pd.read_csv(path)
    # Extract day from filename like GOOG_2023-01-03_34200000_57600000_message_10.csv
    df["day"] = df["filename"].str.extract(r"(\d{4}-\d{2}-\d{2})")
    return df[["day", "highest_price", "lowest_price", "execution_sum"]]


def main():
    script_dir = os.path.dirname(__file__)
    default_base = os.path.join(
        os.path.dirname(script_dir),
        "output", "evalsequences", "aggressive_scenario", "context_500_buy",
    )

    parser = argparse.ArgumentParser(description="Create sample-to-day mapping with daily H/L")
    parser.add_argument(
        "--experiments_dir", type=str, default=default_base,
        help="Path to folder containing experiment subfolders (default: context_500_buy)",
    )
    parser.add_argument(
        "--daily_hl", type=str,
        default=os.path.join(script_dir, "daily_h_l.csv"),
        help="Path to daily_h_l.csv",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(script_dir, "sample_day_map.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Load daily H/L
    daily = load_daily_hl(args.daily_hl)
    print(f"Loaded daily H/L for {len(daily)} days")
    print(daily.to_string(index=False))

    # Pick any experiment folder to read sample filenames (all have same samples)
    exp_folders = sorted([
        d for d in os.listdir(args.experiments_dir)
        if os.path.isdir(os.path.join(args.experiments_dir, d))
    ])
    if not exp_folders:
        raise FileNotFoundError(f"No experiment folders in {args.experiments_dir}")

    first_exp = os.path.join(args.experiments_dir, exp_folders[0])
    print(f"\nParsing samples from: {exp_folders[0]}")

    samples = parse_samples(first_exp)
    print(f"Found {len(samples)} samples across {samples['day'].nunique()} days")

    # Join
    result = samples.merge(daily, on="day", how="left")

    missing = result["highest_price"].isna().sum()
    if missing > 0:
        print(f"\nWARNING: {missing} samples have no daily H/L match!")

    # Summary
    print(f"\nSamples per day:")
    print(result.groupby("day").size().to_string())

    print(f"\nDaily H/L per day:")
    summary = result.groupby("day")[["highest_price", "lowest_price", "execution_sum"]].first()
    print(summary.to_string())

    # Save
    result[["sample_id", "day", "stock", "highest_price", "lowest_price", "execution_sum"]].to_csv(
        args.output, index=False,
    )
    print(f"\nSaved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
