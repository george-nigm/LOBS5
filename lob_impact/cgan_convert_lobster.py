#!/usr/bin/env python
"""
Convert LOBSTER CSV data to ABIDES pickle format for CGAN training.

Wrapper around abides_worldmodel_offline lobster_converted.py that works
with our data layout and Docker volume mounts.

Usage (inside Docker container):
    python lob_impact/cgan_convert_lobster.py \
        --lobster_dir /home/myuser/data/raw_lobster/GOOG/ \
        --save_dir /home/myuser/data/cgan/GOOG/converted/ \
        --ticker GOOG \
        --dates 20230103 20230104 20230105

The script expects LOBSTER CSV files in the format:
    {lobster_dir}/{TICKER}_{DATE}_34200000_57600000_message_10.csv
    {lobster_dir}/{TICKER}_{DATE}_34200000_57600000_orderbook_10.csv

Output: {save_dir}/{date}/{TICKER}.pickle
"""

import argparse
import os
import sys
import pickle
import itertools

import pandas as pd
import numpy as np
from tqdm import tqdm

# LOBSTER file format constants
LOB_LEVEL = 10
MGS_LOB_FILE = "{}_{}_34200000_57600000_message_" + str(LOB_LEVEL) + ".csv"
ORBK_FILE = "{}_{}_34200000_57600000_orderbook_" + str(LOB_LEVEL) + ".csv"
LOB_MSG_COLUMNS = ["Time", "Type", "ID", "SIZE", "Price", "Direction", "Ukwn"]


def load_lob_data(ticker, date, lob_path, only_message=False, price_div=100):
    """Load LOBSTER message and orderbook CSVs."""
    # load messages
    msg_filename = os.path.join(lob_path, MGS_LOB_FILE.format(ticker, date))
    msg_data = pd.read_csv(msg_filename)
    msg_data.columns = LOB_MSG_COLUMNS[:len(msg_data.columns)]
    msg_data["Price"] = (msg_data["Price"] / price_div).astype(int)

    if only_message:
        return msg_data

    # load orderbook
    lob_filename = os.path.join(lob_path, ORBK_FILE.format(ticker, date))
    orderbook_data = pd.read_csv(lob_filename)
    orderbook_data.columns = list(
        itertools.chain(
            *[
                [f"pask_{i}", f"vask_{i}", f"pbid_{i}", f"vbid_{i}"]
                for i in range(LOB_LEVEL)
            ]
        )
    )
    price_cols = [x for x in orderbook_data.columns if "p" in x]
    orderbook_data[price_cols] = (orderbook_data[price_cols] / price_div).astype(int)

    return msg_data, orderbook_data


def lobster_to_replay_data_format(ticker, msg, orderbook, date):
    """
    Convert LOBSTER message + orderbook DataFrames to ABIDES L3 replay format.

    This is a standalone port of lobster_converted.py:lobster_to_replay_data_format()
    that does not require ABIDES imports (Side enum, datetime_str_to_ns, etc.).
    """
    data = {}

    msg_lob_data = msg.copy()
    msg_lob_data.reset_index(drop=True, inplace=True)

    # Fix hidden execution IDs (ID=0 → negative IDs) — vectorized
    mask = msg_lob_data["ID"] == 0
    msg_lob_data.loc[mask, "ID"] = -np.arange(1, mask.sum() + 1)

    # Convert timestamp to nanoseconds from midnight
    msg_lob_data["timestamp"] = (msg_lob_data["Time"] * 1e9).astype(int)
    # Add date offset (nanoseconds from epoch to midnight of date)
    date_ns = int(pd.Timestamp(date).timestamp() * 1e9)
    msg_lob_data["timestamp"] += date_ns
    msg_lob_data["stock"] = ticker

    # Fill TYPE
    msg_lob_data["TYPE"] = np.nan
    msg_lob_data.loc[msg_lob_data["Type"] == 1, "TYPE"] = "ADD_LIMIT_ORDER"
    msg_lob_data.loc[msg_lob_data["Type"] == 2, "TYPE"] = "CANCEL_PARTIAL_ORDER"
    msg_lob_data.loc[msg_lob_data["Type"] == 3, "TYPE"] = "CANCEL_FULL_ORDER"
    msg_lob_data.loc[msg_lob_data["Type"] == 4, "TYPE"] = "MARKET_ORDER"

    # Keep only types 1,2,3,4 (remove hidden orders type 5)
    msg_lob_data = msg_lob_data[msg_lob_data["Type"].isin([1, 2, 3, 4])]

    # Reference ID
    msg_lob_data["ORDER_ID"] = msg_lob_data["ID"]
    msg_lob_data["NEW_ORDER_ID"] = np.nan

    # SIDE: LOBSTER Direction -1=sell(ask), 1=buy(bid)
    # ABIDES: Side.BID="BID", Side.ASK="ASK"
    msg_lob_data["BUY_SELL_FLAG"] = np.where(
        msg_lob_data["Direction"] == -1, "ASK", "BID"
    )
    # Fix cancel: no side/size in LOBSTER
    msg_lob_data.loc[msg_lob_data["Type"] == 3, ["BUY_SELL_FLAG", "SIZE"]] = np.nan
    msg_lob_data["PRICE"] = msg_lob_data["Price"]

    # Extra columns
    msg_lob_data["is_auction_exec"] = np.nan
    msg_lob_data["is_ptc"] = np.nan
    msg_lob_data["printable"] = np.nan
    msg_lob_data["force_increasing_id"] = False
    msg_lob_data["reference_old"] = np.nan
    msg_lob_data["is_post_only"] = np.nan

    data["original_L3"] = msg_lob_data

    # Store stream
    stream_df = msg_lob_data[
        [
            "stock", "timestamp", "ORDER_ID", "NEW_ORDER_ID", "TYPE",
            "BUY_SELL_FLAG", "SIZE", "PRICE", "is_auction_exec", "is_ptc",
            "printable", "force_increasing_id", "reference_old", "is_post_only",
        ]
    ]
    stream_list = [
        g.to_dict(orient="records") for k, g in stream_df.groupby("timestamp")
    ]
    data["stream"] = {"stream": stream_list, "stream_df": stream_df, "excluded_ids": []}

    # Orderbook
    orderbook.reset_index(drop=True, inplace=True)
    orderbook = orderbook.loc[msg_lob_data.index]
    timesteps = msg_lob_data["timestamp"].values

    # L1
    l1_data = {"best_bids": None, "best_asks": None}
    l1_data["best_bids"] = np.stack(
        [timesteps, orderbook["pbid_0"].values, orderbook["vbid_0"].values], axis=-1
    )
    l1_data["best_asks"] = np.stack(
        [timesteps, orderbook["pask_0"].values, orderbook["vask_0"].values], axis=-1
    )
    data["L1"] = l1_data

    # L2
    l2_data = {"bids": None, "asks": None, "times": timesteps}
    l2_data["asks"] = np.stack(
        [
            orderbook[[f"pask_{i}", f"vask_{i}"]].values
            for i in range(len(orderbook.columns) // 4)
        ],
        axis=1,
    )
    l2_data["bids"] = np.stack(
        [
            orderbook[[f"pbid_{i}", f"vbid_{i}"]].values
            for i in range(len(orderbook.columns) // 4)
        ],
        axis=1,
    )
    data["L2"] = l2_data

    data["cancellation_queue_position"] = None

    return {"data": data}


def convert_stocks(tickers, dates, lobster_data_path, save_path):
    """Load, convert, and save LOBSTER data to ABIDES pickle format."""
    for date in tqdm(dates, desc="Converting dates"):
        for ticker in tickers:
            # Format date for LOBSTER filenames (YYYY-MM-DD)
            datetime_obj = pd.to_datetime(date, format="%Y%m%d")
            lob_date_format = datetime_obj.strftime("%Y-%m-%d")

            msg_file = os.path.join(
                lobster_data_path,
                MGS_LOB_FILE.format(ticker, lob_date_format)
            )
            if not os.path.exists(msg_file):
                print(f"  Skipping {ticker} {date}: {msg_file} not found")
                continue

            print(f"  Converting {ticker} {date}...")
            msg, orb = load_lob_data(ticker, lob_date_format, lobster_data_path)
            out_data = lobster_to_replay_data_format(ticker, msg, orb, date)

            outdir = os.path.join(save_path, date)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"{ticker}.pickle")
            with open(outfile, "wb") as f:
                pickle.dump(out_data, f)

            n_msgs = len(out_data["data"]["original_L3"])
            print(f"    Saved {outfile} ({n_msgs} messages)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LOBSTER CSV to ABIDES pickle for CGAN training"
    )
    parser.add_argument(
        '--lobster_dir', type=str, required=True,
        help='Directory with raw LOBSTER CSV files'
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Output directory for ABIDES pickle files'
    )
    parser.add_argument(
        '--ticker', type=str, default='GOOG',
        help='Stock ticker'
    )
    parser.add_argument(
        '--dates', type=str, nargs='+', required=True,
        help='Dates to convert (format: YYYYMMDD)'
    )
    parser.add_argument(
        '--price_div', type=int, default=100,
        help='Price divisor for LOBSTER data (default: 100 for cents→dollars)'
    )

    args = parser.parse_args()

    print(f"Converting LOBSTER data for {args.ticker}")
    print(f"  Source: {args.lobster_dir}")
    print(f"  Output: {args.save_dir}")
    print(f"  Dates: {args.dates}")

    convert_stocks(
        tickers=[args.ticker],
        dates=args.dates,
        lobster_data_path=args.lobster_dir,
        save_path=args.save_dir,
    )

    print("Conversion complete!")


if __name__ == "__main__":
    main()
