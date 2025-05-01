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


# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint

generate_batched = jax.jit(
    jax.vmap(
        inference.generate,
        in_axes=(
            None, None, None, None,
            None, None, None,    0,
               0, None,    0,    0
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9)
)


def track_midprices_during_messages(midprices, proc_msgs_numb, m_seq_raw_inp, book_l2_init, tick_size, step_size):
    for idx, i in enumerate(range(step_size, m_seq_raw_inp.shape[1] + 1, step_size)):
        print(f'Im using m_seq_raw_inp[:, :{i}, :]')
        sim_init, sim_states_init = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp[:, :i, :])
        mid_price = inference.batched_get_safe_mid_price(sim_init, sim_states_init, tick_size)
        midprices.append(mid_price)
        print(f'midprice after processing {proc_msgs_numb+(idx+1)*step_size} messages =', mid_price)
    
    proc_msgs_numb = proc_msgs_numb+m_seq_raw_inp.shape[1]

    return midprices, proc_msgs_numb

def insert_custom_end(m_seq_gen_doubled, b_seq_gen_doubled, msgs_decoded_doubled,
                        l2_book_states_halved, encoder, mid_price, tick_size = 100, 
                        EVENT_TYPE_i = 4, DIRECTION_i = 0, order_volume = 75):
    
    ORDER_ID_i = 77777777
    sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved[:,-2], msgs_decoded_doubled[:,-1:])
    PRICE_i = jax.vmap(sim_init.get_best_ask)(sim_states_init)
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

def run_generation_scenario(
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
        DIRECTION_i =0, 
        order_volume = 75
        ):

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

    save_folder.joinpath('data_scenario_cond').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('data_scenario_real').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('data_scenario_gen').mkdir(exist_ok=True, parents=True)


    save_folder.joinpath('m_seq_gen_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('b_seq_gen_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('msgs_decoded_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('l2_book_states_halved').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('mid_price').mkdir(exist_ok=True, parents=True)

    base_save_folder = save_folder

    transform_L2_state_batch = jax.jit(
        jax.vmap(
            transform_L2_state,
            in_axes=(0, None, None)
        ),
        static_argnums=(1, 2)
    )

    num_iterations = num_insertions + num_coolings

    for batch_i in tqdm(sample_i):
        print('BATCH', batch_i)
        proc_msgs_numb = -n_msgs

        for iteration in range(1,num_iterations+1):
            print('\nITERATION ', iteration)
            midprices = []
            
            if iteration == 1:
                m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
                m_seq = jnp.array(m_seq)
                b_seq_pv = jnp.array(b_seq_pv)
                msg_seq_raw = jnp.array(msg_seq_raw)
                book_l2_init = jnp.array(book_l2_init)
                b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)

                m_seq_inp = m_seq[:, : seq_len]
                b_seq_inp = b_seq[: , : n_msgs]
                m_seq_raw_inp = msg_seq_raw[:, : n_msgs]

                m_seq_np = np.array(jax.device_get(m_seq_inp))
                b_seq_np = np.array(jax.device_get(b_seq_inp))
                msgs_decoded_np = np.array(jax.device_get(m_seq_raw_inp))
                l2_book_states_np = np.array(jax.device_get(book_l2_init))

                np.save(os.path.join(base_save_folder, 'm_seq_gen_doubled', f'm_seq_inp_{batch_i}.npy'), m_seq_inp)
                np.save(os.path.join(base_save_folder, 'b_seq_gen_doubled', f'b_seq_inp_{batch_i}.npy'), b_seq_inp)
                np.save(os.path.join(base_save_folder, 'msgs_decoded_doubled', f'm_seq_raw_inp_{batch_i}.npy'), m_seq_raw_inp)
                np.save(os.path.join(base_save_folder, 'l2_book_states_halved', f'book_l2_init_{batch_i}.npy'), book_l2_init)

                sim_init, sim_states_init = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp)
                
                midprices, proc_msgs_numb = track_midprices_during_messages(midprices, proc_msgs_numb, m_seq_raw_inp, book_l2_init, tick_size, midprice_step_size)

            else: 
                m_seq = m_seq_gen_doubled
                b_seq = b_seq_gen_doubled
                msg_seq_raw = msgs_decoded_doubled
                m_seq_inp = m_seq
                b_seq_inp = b_seq
                m_seq_raw_inp = msg_seq_raw
                sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved[:,-2], m_seq_raw_inp[:,-1:])
            

            m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = generate_batched(
                sim_init,
                train_state,
                model,
                batchnorm,
                encoder,
                sample_top_n,
                tick_size,
                m_seq_inp,
                b_seq_inp,
                n_gen_msgs,
                sim_states_init,
                jax.random.split(rng_, batch_size),
                )
            
            m_seq_gen_doubled = m_seq_gen
            b_seq_gen_doubled = b_seq_gen
            msgs_decoded_doubled = msgs_decoded
            l2_book_states_halved = l2_book_states[:, -1, :40]

            print(f'\n\nsuccessfully generated iteration no. {iteration}')

            midprices, proc_msgs_numb = track_midprices_during_messages(midprices, proc_msgs_numb, msgs_decoded_doubled, l2_book_states[:, 0, :40], tick_size, midprice_step_size)
 
            if iteration <= num_insertions:
                m_seq_gen_doubled, b_seq_gen_doubled, msgs_decoded_doubled, l2_book_states_halved, p_mid_new = insert_custom_end(m_seq_gen_doubled, 
                                                                                                                      b_seq_gen_doubled, 
                                                                                                                      msgs_decoded_doubled, 
                                                                                                                      l2_book_states,
                                                                                                                      encoder,
                                                                                                                      midprices[-1],
                                                                                                                      tick_size,
                                                                                                                      EVENT_TYPE_i, 
                                                                                                                      DIRECTION_i, 
                                                                                                                      order_volume)
                midprices.append(jnp.squeeze(p_mid_new, axis=-1))
                print(f'\nAGGRESSIVE MESSAGE INSERTED - {iteration}\n\n')

                m_seq_np = np.array(jax.device_get(m_seq_gen_doubled))
                b_seq_np = np.array(jax.device_get(b_seq_gen_doubled))
                msgs_decoded_np = np.array(jax.device_get(msgs_decoded_doubled))
                mid_price_np = np.array(jax.device_get(midprices))
                l2_book_states_np = np.array(jax.device_get(l2_book_states_halved))

                m_seq_gen_doubled = m_seq_gen_doubled[:, Message_Tokenizer.MSG_LEN:]
                msgs_decoded_doubled = msgs_decoded_doubled[:, 1:, :]
                b_seq_gen_doubled = b_seq_gen_doubled[:, 1:, :]

            if iteration > num_insertions:
                print(f'\nGENERATE FORWARD WITHOUT AGGRESSIVE ORDER - {iteration}\n\n')
                l2_book_states_halved = l2_book_states

                m_seq_np = np.array(jax.device_get(m_seq_gen_doubled))
                b_seq_np = np.array(jax.device_get(b_seq_gen_doubled))
                msgs_decoded_np = np.array(jax.device_get(msgs_decoded_doubled))
                l2_book_states_np = np.array(jax.device_get(l2_book_states_halved))
                mid_price_np = np.array(jax.device_get(midprices))
            
            np.save(os.path.join(base_save_folder, 'm_seq_gen_doubled', f'm_seq_gen_doubled_batch_{batch_i}_iter_{iteration}.npy'), m_seq_np)
            np.save(os.path.join(base_save_folder, 'b_seq_gen_doubled', f'b_seq_gen_doubled_batch_{batch_i}_iter_{iteration}.npy'), b_seq_np)
            np.save(os.path.join(base_save_folder, 'msgs_decoded_doubled', f'msgs_decoded_doubled_batch_{batch_i}_iter_{iteration}.npy'), msgs_decoded_np)
            np.save(os.path.join(base_save_folder, 'l2_book_states_halved', f'l2_book_states_halved_batch_{batch_i}_iter_{iteration}.npy'), l2_book_states_np)
            np.save(os.path.join(base_save_folder, 'mid_price', f'mid_price_batch_{batch_i}_iter_{iteration}.npy'), mid_price_np)

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

def parse_args():
    p = argparse.ArgumentParser(description="Run LOB inference scenario")
    p.add_argument(
        "--config", "-c", 
        type=str, 
        default="/app/1_run_exp_aggresive_scenario.yaml",
        help="Path to your YAML config file"
    )
    return p.parse_args()

def main():
    args = parse_args()
    # load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

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
    num_devices       = cfg["num_devices"]
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

    # Experiment folder
    exp_folder = create_next_experiment_folder(save_folder)
    print("Experiment dir:", exp_folder)
    wandb.run.summary["experiment_dir"] = str(exp_folder)

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
    ds = inference.get_dataset(data_path, n_messages, n_gen_msgs)
    wandb.log({"dataset_size": len(list(data_path.iterdir()))})

    # run generation
    results = run_generation_scenario(
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

if __name__ == "__main__":
    main()