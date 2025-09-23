#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

# Prevent XLA pre-allocation if using TPU/GPU backends
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Torch spawn safety (avoid "context already set")
import torch
try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import json
import yaml
import numpy as np
import pandas as pd  # not used directly, but kept per original snippet
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState

# --- Project imports (as in your environment) ---
from lob.encoding import Vocab, Message_Tokenizer
from lob import inference_no_errcorr as inference
from lob.init_train import init_train_state, load_checkpoint, load_metadata
import lob.encoding as encoding
import preproc as preproc

import historical_scenario  # for create_next_experiment_folder
from lob.lobster_dataloader import LOBSTER_Dataset


# ----------------------------
# Utilities
# ----------------------------
def parse_args(default_config: str = "historical_scenario.yaml") -> argparse.Namespace:
    """
    Parse CLI args with a default YAML config fallback.
    Uses parse_known_args to ignore unknown flags passed by launchers.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to your YAML config file",
    )
    args, _ = parser.parse_known_args()
    return args


def safe_deduplicate_trainstate(state: TrainState) -> TrainState:
    """
    Replace init_train.deduplicate_trainstate with a version that selects device[0],
    works on CPU if no GPU available.
    """
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = jax.devices("cpu")
        print("[INFO] GPU not available. Falling back to CPU.")
    else:
        if len(devices) > 0:
            print("[INFO] Running on GPU.")
        else:
            print("[INFO] No GPU devices found. Falling back to CPU.")
            devices = jax.devices("cpu")

    return jax.device_put(jax.tree.map(lambda x: x[0], state), device=devices[0])


# ----------------------------
# Message helpers
# ----------------------------
def construct_custom_msg(
    last_msg: jnp.ndarray,
    current_book: jnp.ndarray,
    sim_init,
    sim_state,
    DIRECTION_i: int,
    EVENT_TYPE_i: int,
    order_volume: int,
    use_relative_volume: bool,
    order_volume_ratio: float,
) -> jnp.ndarray:
    """
    Construct a custom message (e.g., a large market order) based on the last message and current order book.
    Assumes message layout consistent with your pipeline (PRICE at index 3, TIME_s at 8, TIME_ns at 9).
    """
    batch_size = last_msg.shape[0]
    ORDER_ID_i = 77777777

    EVENT_TYPE = jnp.full((batch_size,), EVENT_TYPE_i, dtype=jnp.int32)
    SIDE = jnp.full((batch_size,), DIRECTION_i, dtype=jnp.int32)
    # Initialize with last price, replaced below with best bid/ask
    PRICE = last_msg[:, 3]
    DISPLAY_FLAG = jnp.ones((batch_size,), dtype=jnp.int32)

    # Per-state best bid/ask
    best_bid_info, best_ask_info = jax.vmap(
        lambda state: sim_init.get_best_bid_and_ask_inclQuants(state)
    )(sim_state)

    PRICE = jnp.where(
        DIRECTION_i == 0, best_ask_info[:, 0], best_bid_info[:, 0]
    )  # 0 = sell -> hit ask, 1 = buy -> hit bid

    # pick available volume on the level we will consume
    avail = jnp.where(
        DIRECTION_i == 0,
        best_ask_info[:, 1],  # ask vol for a buy
        best_bid_info[:, 1],  # bid vol for a sell
    ).astype(jnp.int32)

    # === SIZING RULE (CONFIG-DRIVEN) ===
    # If use_relative_volume: consume a fraction of available L1 volume.
    #   SIZE = floor(avail * order_volume_ratio); if avail>0 and result==0 -> 1
    # Else: fixed-size capped by avail (do not cross levels here).
    if use_relative_volume:
        ratio = jnp.clip(jnp.float32(order_volume_ratio), 0.0, 1.0)
        SIZE = jnp.floor(ratio * jnp.float32(avail)).astype(jnp.int32)
        # if there's some liquidity but rounding gave 0, take 1; if avail==0, keep 0
        SIZE = jnp.where((avail > 0) & (SIZE == 0), 1, SIZE)
    else:
        SIZE = jnp.minimum(jnp.int32(order_volume), avail)

    SIZE = jnp.clip(SIZE, 1, 999_999)

    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    TIME_s = last_msg[:, 8]
    TIME_ns = last_msg[:, 9]

    custom_msg = jnp.stack(
        [
            jnp.full((batch_size,), ORDER_ID_i, dtype=jnp.int32),  # ORDER_ID
            EVENT_TYPE,  # EVENT_TYPE
            SIDE,        # DIRECTION
            PRICE,       # PRICE_ABS
            DISPLAY_FLAG,
            SIZE,
            zeros, zeros,
            TIME_s,
            TIME_ns,
            zeros, zeros, zeros, zeros,
        ],
        axis=1,
    )

    return custom_msg.astype(jnp.int32)


def shift_sell_prices(
    msg: jnp.ndarray,
    tick_size: int,
    PRICE_ABS_i: int,
    DIRECTION_i: int,
    EVENT_TYPE_i: int,
) -> jnp.ndarray:
    """
    Shift prices for SELL (ask) messages by +tick_size for limit/execution events.
    """
    msg_shifted = msg.copy()
    is_relevant = (msg[..., EVENT_TYPE_i] == 1) | (msg[..., EVENT_TYPE_i] == 4)  # Limit or Execution
    is_sell = (msg[..., 2] == 0)
    mask = is_relevant & is_sell

    msg_shifted = msg_shifted.at[..., PRICE_ABS_i].set(
        jnp.where(mask, msg[..., PRICE_ABS_i] + (1 - 2 * DIRECTION_i) * tick_size, msg[..., PRICE_ABS_i])
    )
    return msg_shifted


def shift_buy_prices(
    msg: jnp.ndarray,
    tick_size: int,
    PRICE_ABS_i: int,
    DIRECTION_i: int,
    EVENT_TYPE_i: int,
) -> jnp.ndarray:
    """
    Shift prices for BUY (bid) messages by +tick_size for limit/execution events.
    """
    msg_shifted = msg.copy()
    is_relevant = (msg[..., EVENT_TYPE_i] == 1) | (msg[..., EVENT_TYPE_i] == 4)  # Limit or Execution
    is_buy = (msg[..., 2] == 1)
    mask = is_relevant & is_buy

    msg_shifted = msg_shifted.at[..., PRICE_ABS_i].set(
        jnp.where(mask, msg[..., PRICE_ABS_i] + (1 - 2 * DIRECTION_i) * tick_size, msg[..., PRICE_ABS_i])
    )
    return msg_shifted


# ----------------------------
# Core scenario
# ----------------------------
def run_historical_scenario(
    n_samples: int,
    batch_size: int,
    ds: LOBSTER_Dataset,
    rng: Any,
    seq_len: int,
    n_msgs: int,
    n_gen_msgs: int,
    train_state: TrainState,
    model: nn.Module,
    batchnorm: bool,
    encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    stock_symbol: str,
    n_vol_series: int = 500,
    save_folder: str = "./data_saved/",
    tick_size: int = 100,
    sample_top_n: int = -1,
    sample_all: bool = False,
    num_insertions: int = 2,
    num_coolings: int = 2,
    midprice_step_size: int = 100,
    EVENT_TYPE_i: int = 4,
    DIRECTION_i: int = 0,
    order_volume: int = 75,
    use_sample_file: bool = False,
    sample_file_path: Optional[str] = None,
    start_batch: int = 0,
    end_batch: int = -1,
    use_relative_volume: bool = False,
    order_volume_ratio: float = 1.0,
):
    rng, rng_ = jax.random.split(rng)

    # Build sample indices
    if use_sample_file:
        assert sample_file_path is not None, "Path to sample file not provided"
        with open(sample_file_path, "r") as f:
            sample_i_full = json.load(f)
        sample_i = sample_i_full[start_batch : (None if end_batch == -1 else end_batch)]
        for i, batch in enumerate(sample_i):
            assert len(batch) == batch_size, f"Batch {i} has incorrect size {len(batch)}, expected {batch_size}"
    else:
        if sample_all:
            sample_i = (
                jnp.arange(len(ds) // batch_size * batch_size, dtype=jnp.int32)
                .reshape(-1, batch_size)
                .tolist()
            )
        else:
            assert n_samples % batch_size == 0, "n_samples must be divisible by batch_size"
            sample_i = jax.random.choice(
                rng_,
                jnp.arange(len(ds), dtype=jnp.int32),
                shape=(n_samples // batch_size, batch_size),
                replace=False,
            ).tolist()

    rng, rng_ = jax.random.split(rng)

    save_folder = Path(save_folder)
    (save_folder / "msgs_decoded_doubled").mkdir(exist_ok=True, parents=True)
    (save_folder / "b_seq_gen_doubled").mkdir(exist_ok=True, parents=True)
    (save_folder / "mid_price").mkdir(exist_ok=True, parents=True)

    # JIT for L2 transform
    transform_L2_state_batch = jax.jit(
        jax.vmap(preproc.transform_L2_state, in_axes=(0, None, None)),
        static_argnums=(1, 2),
    )

    for batch_i in tqdm(sample_i):
        print("BATCH", batch_i)
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = jnp.array(book_l2_init)
        current_book = book_l2_init

        # insertion points after historical segments (as you had)
        insertion_points = [n_msgs + (i + 1) * n_gen_msgs + i for i in range(num_insertions)]
        insertion_points = sorted([p for p in insertion_points if p <= msg_seq_raw.shape[1]])
        print(f"[BATCH {batch_i}] Planned insertion points at indices: {insertion_points}")

        books: List[jnp.ndarray] = []
        messages: List[jnp.ndarray] = []
        midprices: List[jnp.ndarray] = []

        original_idx = 0
        steps_count = 0
        insertion_iter = 0
        shift_ticks = 0

        while original_idx < msg_seq_raw.shape[1] or insertion_iter < len(insertion_points):
            # Perform insertion
            if insertion_iter < len(insertion_points) and steps_count == insertion_points[insertion_iter]:
                last_msg = msg_seq_raw[:, 0, :] if steps_count == 0 else messages[-1]

                sim_init, sim_state = inference.get_sims_vmap(current_book, last_msg[:, None, :])
                custom_msg = construct_custom_msg(
                    last_msg,
                    current_book,
                    sim_init,
                    sim_state,
                    DIRECTION_i=DIRECTION_i,
                    EVENT_TYPE_i=EVENT_TYPE_i,
                    order_volume=order_volume,
                    use_relative_volume=use_relative_volume,
                    order_volume_ratio=order_volume_ratio,
                )

                sim_init_ins, sim_state_ins = inference.get_sims_vmap(current_book, custom_msg[:, None, :])
                mid_price_ins = inference.batched_get_safe_mid_price(sim_init_ins, sim_state_ins, tick_size)
                full_l2_state_ins = jax.vmap(sim_init_ins.get_L2_state, in_axes=(0, None))(
                    sim_state_ins, current_book.shape[1]
                )
                current_book = full_l2_state_ins[:, : current_book.shape[1]]

                messages.append(custom_msg)
                books.append(current_book)
                midprices.append(mid_price_ins)

                shift_ticks += 1
                steps_count += 1
                insertion_iter += 1
                continue

            # Consume next historical message
            if original_idx < msg_seq_raw.shape[1]:
                msg = msg_seq_raw[:, original_idx : original_idx + 1, :]

                # shift sell then buy prices (heuristic) - comment when just historical without shifting
                msg = shift_sell_prices(
                    msg,
                    tick_size * shift_ticks,
                    PRICE_ABS_i=3,
                    DIRECTION_i=DIRECTION_i,
                    EVENT_TYPE_i=1,
                )
                msg = shift_buy_prices(
                    msg,
                    tick_size * shift_ticks,
                    PRICE_ABS_i=3,
                    DIRECTION_i=DIRECTION_i,
                    EVENT_TYPE_i=1,
                )

                sim_init_hist, sim_state_hist = inference.get_sims_vmap(current_book, msg)
                mid_price_hist = inference.batched_get_safe_mid_price(sim_init_hist, sim_state_hist, tick_size)
                full_l2_state_hist = jax.vmap(sim_init_hist.get_L2_state, in_axes=(0, None))(
                    sim_state_hist, current_book.shape[1]
                )
                current_book = full_l2_state_hist[:, : current_book.shape[1]]

                messages.append(msg[:, 0, :])
                books.append(current_book)
                midprices.append(mid_price_hist)

                original_idx += 1
                steps_count += 1
            else:
                print("No more historical messages. Awaiting remaining insertions...")
                break

        # Stack & save
        messages_arr = jnp.concatenate([m.reshape(m.shape[0], 1, m.shape[1]) for m in messages], axis=1)  # (B, T, 14)
        books_arr = jnp.stack(books, axis=1)  # (B, T, book_dim)
        midprices_arr = jnp.stack(midprices, axis=0)  # (T, B)

        np.save(
            save_folder / "msgs_decoded_doubled" / f"msgs_decoded_doubled_batch_{batch_i}_iter_0.npy",
            jax.device_get(messages_arr),
        )
        np.save(
            save_folder / "mid_price" / f"mid_price_batch_{batch_i}_iter_0.npy",
            jax.device_get(midprices_arr),
        )

        # Transform L2 and save
        midprices_batched = midprices_arr.T[:, :, None]  # (B, T, 1)
        books_with_mid = jnp.concatenate([midprices_batched, books_arr], axis=-1)
        books_transformed = transform_L2_state_batch(books_with_mid, n_vol_series, tick_size)
        np.save(
            save_folder / "b_seq_gen_doubled" / f"b_seq_gen_doubled_batch_{batch_i}_iter_0.npy",
            jax.device_get(books_transformed),
        )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args("heuristic_historical_scenario_run.yaml")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Unpack config
    save_folder       = cfg["save_folder"]
    batch_size        = cfg["batch_size"]
    n_samples         = cfg["n_samples"]
    n_gen_msgs        = cfg["n_gen_msgs"]
    midprice_step_size= cfg["midprice_step_size"]
    num_insertions    = cfg["num_insertions"]
    num_coolings      = cfg["num_coolings"]
    EVENT_TYPE_i      = cfg["EVENT_TYPE_i"]
    DIRECTION_i       = cfg["DIRECTION_i"]
    order_volume      = cfg["order_volume"]
    bsz               = cfg["bsz"]
    n_messages        = cfg["n_messages"]
    book_dim          = cfg["book_dim"]
    n_vol_series      = cfg["n_vol_series"]
    sample_top_n      = cfg["sample_top_n"]
    model_size        = cfg["model_size"]
    data_dir          = cfg["data_dir"]
    sample_all        = cfg["sample_all"]
    stock             = cfg["stock"]
    tick_size         = cfg["tick_size"]
    rng_seed          = cfg["rng_seed"]
    ckpt_path         = cfg["ckpt_path"]

    use_sample_file   = cfg["use_sample_file"]
    sample_file_path  = cfg["sample_file_path"]
    start_batch       = cfg["start_batch"]
    end_batch         = cfg["end_batch"]

    # Optional relative volume controls
    order_volume_ratio  = cfg.get("order_volume_ratio", 1.0)
    use_relative_volume = cfg.get("use_relative_volume", False)

    num_devices = jax.local_device_count()
    print(f"num_devices: {num_devices}")

    # Load metadata and model
    print("Loading metadata from", ckpt_path)
    args_ckpt = load_metadata(ckpt_path)

    print("Initializing model...")
    train_state, model_cls = init_train_state(
        args_ckpt,
        n_classes=len(Vocab()),
        seq_len=n_messages * Message_Tokenizer.MSG_LEN,
        book_dim=book_dim,
        book_seq_len=n_messages,
    )

    # Monkey-patch deduplicate_trainstate
    from lob import init_train
    init_train.deduplicate_trainstate = safe_deduplicate_trainstate

    print("Loading checkpoint...")
    ckpt = load_checkpoint(train_state, ckpt_path, train=False)
    state = ckpt["model"]
    model = model_cls(training=False, step_rescale=1.0)

    # RNG
    rng = jax.random.PRNGKey(rng_seed)

    # Data directory per stock
    data_path = Path(data_dir) / stock
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_path} ({len(list(data_path.iterdir()))} files)")

    # Experiment folder
    exp_folder = historical_scenario.create_next_experiment_folder(save_folder)
    print("Experiment dir:", exp_folder)
    with open(exp_folder / "used_config.yaml", "w") as f_out:
        yaml.dump(cfg, f_out)
    
    # Setup logging to experiment folder
    log_file_path = exp_folder / "job.log"
    print(f"Redirecting all output to: {log_file_path}")
    
    # Redirect stdout and stderr to log file
    import sys
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    # Print initial info to log
    print(f"Experiment started at: {datetime.now()}")
    print(f"Experiment folder: {exp_folder}")
    print(f"Configuration: {cfg}")
    print("=" * 80)

    # Dataset (historical)
    ds = inference.get_dataset(
        data_path, n_messages, (num_insertions + num_coolings) * n_gen_msgs
    )

    # Optionally pre-sample indices and store them on disk (even if using sample_file later)
    rng, rng_ = jax.random.split(rng)
    if sample_all:
        sample_i = (
            jnp.arange(len(ds) // batch_size * batch_size, dtype=jnp.int32)
            .reshape(-1, batch_size)
            .tolist()
        )
    else:
        assert n_samples % batch_size == 0, "n_samples must be divisible by batch_size"
        sample_i = jax.random.choice(
            rng_,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(n_samples // batch_size, batch_size),
            replace=False,
        ).tolist()
    rng, _ = jax.random.split(rng)

    print(len(sample_i))
    filename = f"random_sample_indices_b{len(sample_i)}_bs{batch_size}_ins{num_insertions}_cool{num_coolings}.json"
    with open(filename, "w") as f:
        json.dump(sample_i, f)

    # Run scenario
    run_historical_scenario(
        n_samples=n_samples,
        batch_size=batch_size,
        ds=ds,
        rng=rng,
        seq_len=n_messages * Message_Tokenizer.MSG_LEN,
        n_msgs=n_messages,
        n_gen_msgs=n_gen_msgs,
        train_state=state,
        model=model,
        batchnorm=args_ckpt.batchnorm,
        encoder=Vocab().ENCODING,
        stock_symbol=stock,
        n_vol_series=n_vol_series,
        save_folder=exp_folder,
        tick_size=tick_size,
        sample_top_n=sample_top_n,
        sample_all=sample_all,
        num_insertions=num_insertions,
        num_coolings=num_coolings,
        midprice_step_size=midprice_step_size,
        EVENT_TYPE_i=EVENT_TYPE_i,
        DIRECTION_i=DIRECTION_i,
        order_volume=order_volume,
        use_sample_file=use_sample_file,
        sample_file_path=sample_file_path,
        start_batch=start_batch,
        end_batch=end_batch,
        use_relative_volume=use_relative_volume,
        order_volume_ratio=order_volume_ratio,
    )
    
    # Close log file and restore stdout/stderr
    print(f"Experiment completed at: {datetime.now()}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Logs saved to: {log_file_path}")


if __name__ == "__main__":
    main()