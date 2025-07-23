#!/usr/bin/env python
import os
import sys
import argparse
import yaml

# prevent XLA pre-allocation if using TPU/GPU backends
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import wandb
import jax
from pathlib import Path
from datetime import datetime

from lob.encoding import Vocab, Message_Tokenizer
from lob import inference_no_errcorr as inference
from lob.init_train import init_train_state, load_checkpoint, load_metadata

import os
import numpy as np  # or use onp if preferred

from datetime import datetime
import functools
from glob import glob
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.nn import one_hot
import flax.linen as nn
from flax.training.train_state import TrainState
from lob import train_helpers
import numpy as onp
import os
import sys
import pandas as pd
import pickle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
from utils import debug, info
import os

import lob.validation_helpers as valh
import lob.evaluation as eval
import preproc as preproc
from preproc import transform_L2_state
import lob.encoding as encoding
from lob.encoding import Message_Tokenizer, Vocab
from lob.lobster_dataloader import LOBSTER_Dataset
from jax import lax


# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint


def track_midprices_during_messages(
        m_seq_raw_inp: jax.Array,
        book_l2_init: jax.Array,
        tick_size: int,
        step_size: int,
    ) -> jax.Array:
    """
    Sequential version that computes mid‑price after every message.
    Returns mid‑price after each message, not every step_size.

    Args:
        m_seq_raw_inp:  (batch, T, msg_dim) – raw decoded messages
        book_l2_init:   (batch, book_dim) – initial L2 state of the orderbook
        tick_size:      Tick size of the instrument
        step_size:      Unused in this variant

    Returns
    -------
    midprices : jax.Array
        Shape (T, batch). mid‑price after each message.
    """
    batch_size, T, msg_dim = m_seq_raw_inp.shape
    print(f"[INFO] Starting tracking midprices for batch size {batch_size}, {T} messages, msg_dim={msg_dim}")
    print(f"[INFO] Initial book shape: {book_l2_init.shape}")

    midprices = []
    current_book = book_l2_init

    for t in range(T):
        msg = m_seq_raw_inp[:, t:t+1, :]  # shape (batch, 1, msg_dim)
        print(f"[STEP {t}] Processing message {t}, shape: {msg.shape}")

        sim_init, sim_states = inference.get_sims_vmap(current_book, msg)
        mid_price = inference.batched_get_safe_mid_price(sim_init, sim_states, tick_size)
        print(f"[STEP {t}] Midprice: {mid_price}")

        full_l2_state = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(
            sim_states, current_book.shape[1]
        )
        current_book = full_l2_state[:, : current_book.shape[1]]
        print(f"[STEP {t}] Updated book shape: {current_book.shape}")

        midprices.append(mid_price)

    midprices = jnp.stack(midprices, axis=0)  # (T, batch)
    print(f"[INFO] Finished. Final midprices shape: {midprices.shape}")
    return midprices



def insert_custom_end(m_seq_gen_doubled, b_seq_gen_doubled, msgs_decoded_doubled,
                        l2_book_states_halved, encoder, mid_price, tick_size = 100, 
                        EVENT_TYPE_i = 4, DIRECTION_i = 0, order_volume = 75):
    
    ORDER_ID_i = 77777777
    # sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved[:,-2], msgs_decoded_doubled[:,-1:])
    sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved[:, -2, :], msgs_decoded_doubled[:, -1:]
)
    
    if DIRECTION_i == 0:
        PRICE_i = jax.vmap(sim_init.get_best_ask)(sim_states_init)
    if DIRECTION_i == 1:
        PRICE_i = jax.vmap(sim_init.get_best_bid)(sim_states_init)

    PRICE_i = jnp.expand_dims(PRICE_i, axis=-1)
    mid_price = jnp.expand_dims(mid_price, axis=-1)

    TIMEs_i = msgs_decoded_doubled[:, -1:, 8].astype(jnp.int32)
    TIMEns_i = msgs_decoded_doubled[:, -1:, 9].astype(jnp.int32)

    batch_size = TIMEns_i.shape[0]

    best_bid_ask = jax.vmap(sim_init.get_best_bid_and_ask_inclQuants)(sim_states_init)
    best_ask_volume = best_bid_ask[1][:, 1]

    SIZE_i = jnp.minimum(best_ask_volume, order_volume)
    batched_quantity = SIZE_i

    batched_new_order_id = jnp.array([ORDER_ID_i] * batch_size, dtype=jnp.int32)
    batched_EVENT_TYPE = jnp.array([EVENT_TYPE_i] * batch_size, dtype=jnp.int32)
    batched_side = jnp.array([DIRECTION_i] * batch_size, dtype=jnp.int32)

    batched_p_abs = PRICE_i.squeeze(-1)    
    batched_time_s = TIMEs_i.squeeze(-1)
    batched_time_ns = TIMEns_i.squeeze(-1)

    batched_construct_sim_msg = jax.vmap(inference.construct_sim_msg)
    batched_sim_msg = batched_construct_sim_msg(
        batched_EVENT_TYPE,
        batched_side,
        batched_quantity,
        batched_p_abs,
        batched_new_order_id,
        batched_time_s,
        batched_time_ns,
    )

    new_sim_state = jax.vmap(sim_init.process_order_array)(sim_states_init, batched_sim_msg)
    p_mid_new = inference.batched_get_safe_mid_price(sim_init, new_sim_state, tick_size)
    p_mid_new = p_mid_new[:, None]
    p_change = ((p_mid_new - mid_price) // tick_size).astype(jnp.int32)
    book_l2 = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(new_sim_state, 20)
    new_l2_book_states_halved = jnp.concatenate([l2_book_states_halved, book_l2[:, None, :]], axis=1)
    new_book_raw = jnp.concatenate([p_change, book_l2], axis=1)
    new_book_raw = new_book_raw[:, None, :]

    transform_L2_state_batch = jax.jit(jax.vmap(inference.transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2))
    new_book = transform_L2_state_batch(new_book_raw, 500, 100)

    b_seq_gen_doubled = jnp.concatenate([b_seq_gen_doubled, new_book], axis=1)

    ins_msg = jnp.concatenate([
        batched_new_order_id.reshape(-1, 1),
        batched_EVENT_TYPE.reshape(-1, 1),
        batched_side.reshape(-1, 1),
        batched_p_abs.reshape(-1, 1),
        jnp.full((batch_size, 1), 1, dtype=jnp.int32),
        batched_quantity.reshape(-1, 1),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        batched_time_s.reshape(-1, 1),
        batched_time_ns.reshape(-1, 1),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
    ], axis=1)

    new_batched_sim_msg = ins_msg[:, None, :]
    UPDATED_msgs_decoded_doubled = jnp.concatenate([msgs_decoded_doubled, new_batched_sim_msg], axis=1)

    msg_encoded = jax.vmap(lambda m: encoding.encode_msg(m, encoder))(ins_msg)
    UPDATED_m_seq_gen_doubled = jnp.concatenate([m_seq_gen_doubled, msg_encoded], axis=1)

    return UPDATED_m_seq_gen_doubled, b_seq_gen_doubled, UPDATED_msgs_decoded_doubled, new_l2_book_states_halved, p_mid_new

def run_historical_scenario(
        n_samples: int,
        batch_size: int,
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        stock_symbol: str,
        n_vol_series: int = 500,
        save_folder: str = './data_saved/',
        tick_size: int = 100,
        sample_top_n: int = -1,
        sample_all: bool = False,
        num_insertions: int = 2,
        num_coolings: int = 2,
        midprice_step_size=100,
        EVENT_TYPE_i = 4,
        DIRECTION_i = 0,
        order_volume = 75
    ):
    """
    Manual, step-by-step scenario runner: for each batch, processes messages one at a time,
    updating the orderbook and tracking midprices, mimicking track_midprices_during_messages.
    Saves processed messages, books, and midprices for each batch.

    
    """
    

    rng, rng_ = jax.random.split(rng)
    if sample_all:
        sample_i = jnp.arange(
            len(ds) // batch_size * batch_size,
            dtype=jnp.int32
        ).reshape(-1, batch_size).tolist()
    else:
        assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'
        sample_i = jax.random.choice(
            rng_,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(n_samples // batch_size, batch_size),
            replace=False
        ).tolist()
    rng, rng_ = jax.random.split(rng)

    save_folder = Path(save_folder)
    save_folder.joinpath('msgs_decoded_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('l2_book_states_halved').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('b_seq_gen_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('mid_price').mkdir(exist_ok=True, parents=True)
    base_save_folder = save_folder

    for batch_i in tqdm(sample_i):
        print('BATCH', batch_i)
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        m_seq = jnp.array(m_seq)
        b_seq_pv = jnp.array(b_seq_pv)
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = jnp.array(book_l2_init)

        #=============#
        # Step 1: Prepare positions where to insert messages (accounting for prior insertions)
        insertion_points = [n_msgs + (i + 1) * n_gen_msgs + i for i in range(num_insertions)]
        insertion_points = [p for p in insertion_points if p <= msg_seq_raw.shape[1]]
        print(f"[BATCH {batch_i}] Inserting custom messages at: {insertion_points}")

        # Step 2: Generate placeholder message using same logic as insert_custom_end (just 14-dim msg, no book logic yet)
        def construct_custom_msg(last_msg):
            ORDER_ID_i = 77777777
            EVENT_TYPE = jnp.full((batch_size,), EVENT_TYPE_i)
            SIDE = jnp.full((batch_size,), DIRECTION_i)
            PRICE = last_msg[:, 3]  # or use fixed jnp.full((batch_size,), 123456)
            DISPLAY_FLAG = jnp.ones((batch_size,), dtype=jnp.int32)
            SIZE = jnp.full((batch_size,), order_volume)
            zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
            TIME_s = last_msg[:, 8]
            TIME_ns = last_msg[:, 9]

            msg = jnp.stack([
                jnp.full((batch_size,), ORDER_ID_i),
                EVENT_TYPE,
                SIDE,
                PRICE,
                DISPLAY_FLAG,
                SIZE,
                zeros, zeros,
                TIME_s, TIME_ns,
                zeros, zeros, zeros, zeros,
            ], axis=1)

            return msg.astype(jnp.int32)

        # Step 3: Loop and insert messages
        for i, idx in enumerate(insertion_points):
            custom_msg = construct_custom_msg(msg_seq_raw[:, idx - 1])
            msg_seq_raw = jnp.concatenate([
                msg_seq_raw[:, :idx, :],
                custom_msg[:, None, :],  # shape (B, 1, 14)
                msg_seq_raw[:, idx:, :]
            ], axis=1)

        print(f"[BATCH {batch_i}] msg_seq_raw shape after insertions: {msg_seq_raw.shape}")
        #=============#

        batch_size, T, msg_dim = msg_seq_raw.shape
        current_book = book_l2_init

        books = []
        messages = []
        midprices = []

        for t in range(T):
            msg = msg_seq_raw[:, t:t+1, :]
            sim_init, sim_states = inference.get_sims_vmap(current_book, msg)
            mid_price = inference.batched_get_safe_mid_price(sim_init, sim_states, tick_size)
            full_l2_state = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(sim_states, current_book.shape[1])
            current_book = full_l2_state[:, : current_book.shape[1]]
            books.append(current_book)
            messages.append(msg)
            midprices.append(mid_price)

        books = jnp.stack(books, axis=1)             # (batch, T, book_dim)
        messages = jnp.concatenate(messages, axis=1) # (batch, T, msg_dim)
        midprices = jnp.stack(midprices, axis=0)     # (T, batch)

        print(f"[BATCH {batch_i}] Finished all {T} steps")
        print(f"[BATCH {batch_i}] Final messages shape: {messages.shape}")
        print(f"[BATCH {batch_i}] Final books shape: {books.shape}")
        print(f"[BATCH {batch_i}] Final midprices shape: {midprices.shape}")

        np.save(os.path.join(base_save_folder, 'msgs_decoded_doubled', f'msgs_decoded_doubled_batch_{batch_i}_iter_0.npy'), np.array(jax.device_get(messages)))
        np.save(os.path.join(base_save_folder, 'l2_book_states_halved', f'l2_book_states_halved_batch_{batch_i}_iter_0.npy'), np.array(jax.device_get(books)))
        np.save(os.path.join(base_save_folder, 'mid_price', f'mid_price_batch_{batch_i}_iter_0.npy'), np.array(jax.device_get(midprices)))

        # ========================
        transform_L2_state_batch = jax.jit(jax.vmap(preproc.transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2))

        # Get midprices for each step: (T, B) → (B, T)
        midprices_batched = midprices.T  # (B, T)
        p_mid = midprices_batched[:, :, None]  # (B, T, 1)

        # Add midprice as first column to each book state
        books_with_mid = jnp.concatenate([p_mid, books], axis=-1)  # (B, T, 41)
        print(f"[BATCH {batch_i}] books_with_mid.shape: {books_with_mid.shape}")

        # Transform each book+midprice into model input format
        books_transformed = transform_L2_state_batch(books_with_mid, n_vol_series, tick_size)  # (B, T, D)
        print(f"[BATCH {batch_i}] books_transformed.shape: {books_transformed.shape}")

        # Save transformed books
        np.save(os.path.join(base_save_folder, 'b_seq_gen_doubled', f'b_seq_gen_doubled_batch_{batch_i}_iter_0.npy'), np.array(jax.device_get(books_transformed)))


def create_next_experiment_folder(save_folder: str) -> Path:
    base = Path(save_folder)
    if not base.exists():
        raise FileNotFoundError(f"Directory {save_folder!r} does not exist")
    max_idx = 0
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith("exp_"):
            parts = entry.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
    next_idx = max_idx + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder = base / f"exp_{next_idx}_{timestamp}"
    new_folder.mkdir(parents=True, exist_ok=False)
    return new_folder

def parse_args(config_file):
    p = argparse.ArgumentParser(description="Run LOB inference scenario")
    p.add_argument(
        "--config", "-c", 
        type=str, 
        default=f"/app/{config_file}.yaml",
        help="Path to your YAML config file"
    )
    return p.parse_args()

def main():
    print(f"JAX backend platform: {jax.lib.xla_bridge.get_backend().platform}")

    args = parse_args(config_file = "historical_scenario")
    # load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Check JAX devices and log to console and file
    jax_devices = jax.devices()
    accelerator_types = set(d.device_kind for d in jax_devices)
    device_summary = [f"{d.id}: {d.device_kind}" for d in jax_devices]
    device_message = f"JAX devices detected: {device_summary}"

    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        log_message = f"✅ Running on GPU(s): {device_message}"
    else:
        log_message = f"⚠️ Running on CPU only: {device_message}"

    print(log_message)
    logger.info(log_message)

    # Initialize WandB
    wandb.init(
        project="Aggressive_Scenario",    # ← fill in
        entity="george-nigm",     # ← fill in
        config=cfg
    )
    
    # Multi-host / multi-node JAX setup (expects env vars WORLD_SIZE, RANK, COORD_ADDR)
    if "WORLD_SIZE" in os.environ:
        import jax.distributed as jdist
        jdist.initialize(
            coordinator_address=os.environ["COORD_ADDR"],
            num_processes=int(os.environ["WORLD_SIZE"]),
            process_id=int(os.environ["RANK"])
        )

    # unpack config
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
    # num_devices       = cfg["num_devices"]
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

    # num_devices  = jax.local_device_count() 
    # print(f'num_devices: ', num_devices)
    
    num_devices = jax.local_device_count()  # 6
    print(f'num_devices: ', num_devices)
    # assert bsz % num_devices == 0

    
    # Experiment folder
    exp_folder = create_next_experiment_folder(save_folder)
    print("Experiment dir:", exp_folder)
    wandb.run.summary["experiment_dir"] = str(exp_folder)

    with open(exp_folder / "used_config.yaml", "w") as f_out:
        yaml.dump(cfg, f_out)

    # Log config file as artifact
    artifact = wandb.Artifact(name="used_config", type="config")
    artifact.add_file(str(exp_folder / "used_config.yaml"))
    wandb.log_artifact(artifact)

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
    print("Loading checkpoint...")
    ckpt = load_checkpoint(train_state, ckpt_path, train=False)
    state = ckpt["model"]
    model = model_cls(training=False, step_rescale=1.0)

    # override devices & batch
    args_ckpt.bsz = bsz
    args_ckpt.num_devices = num_devices

    # prepare RNG
    rng = jax.random.PRNGKey(rng_seed)

    # data directory
    data_path = Path(data_dir) / stock
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_path} ({len(list(data_path.iterdir()))} files)")

    # get dataset
    ds = inference.get_dataset(data_path, n_messages, (num_insertions + num_coolings) * n_gen_msgs)
    wandb.log({"dataset_size": len(list(data_path.iterdir()))})

    # run generation
    results = run_historical_scenario(
        n_samples,
        batch_size,
        ds,
        rng,
        n_messages * Message_Tokenizer.MSG_LEN,
        n_messages,
        n_gen_msgs,
        state,
        model,
        args_ckpt.batchnorm,
        Vocab().ENCODING,
        stock,
        n_vol_series,
        exp_folder,
        tick_size,
        sample_top_n,
        sample_all,
        num_insertions,
        num_coolings,
        midprice_step_size,
        EVENT_TYPE_i,
        DIRECTION_i,
        order_volume,
    )

    # log any returned metrics/artifacts
    wandb.log({"finished": True})
    wandb.save(str(exp_folder / "*"))
    wandb.finish()

if __name__ == "__main__":
    main()