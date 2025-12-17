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

import os
import numpy as np  # or use onp if preferred

from datetime import datetime
import functools
from glob import glob
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.nn import one_hot
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
import json

# RWKV imports
# Add lobgen parent directory to path so we can import jax_rwkv as a package
lobgen_parent = os.path.dirname(os.path.abspath(__file__))
if lobgen_parent not in sys.path:
    sys.path.insert(0, lobgen_parent)

# Mock huggingface_hub only (if needed for local checkpoints)
def mock_hf_hub_download(*args, **kwargs):
    raise NotImplementedError("hf_hub_download not available - use local checkpoint")

# Only mock huggingface_hub if it's not available
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.modules['huggingface_hub'] = type('Module', (), {'hf_hub_download': mock_hf_hub_download})()

# Now import jax_rwkv from lobgen
try:
    from lobgen.jax_rwkv.auto import get_model, load
    from lobgen.jax_rwkv.utils import simple_sampler
except ImportError as e:
    print(f"Warning: Could not import from lobgen.jax_rwkv: {e}")
    # Try alternative import path
    try:
        import sys
        import importlib.util
        
        # Import auto module
        auto_file = os.path.join(lobgen_parent, "lobgen", "jax_rwkv", "auto.py")
        if os.path.exists(auto_file):
            # Create package structure
            if 'lobgen' not in sys.modules:
                sys.modules['lobgen'] = type('Module', (), {})()
            if 'lobgen.jax_rwkv' not in sys.modules:
                sys.modules['lobgen.jax_rwkv'] = type('Module', (), {})()
            
            spec = importlib.util.spec_from_file_location("lobgen.jax_rwkv.auto", auto_file)
            auto_mod = importlib.util.module_from_spec(spec)
            
            # Mock tokenizer
            class MockGptTokenizer: pass
            class MockWorldTokenizer: pass
            sys.modules['lobgen.jax_rwkv.tokenizer'] = type('Module', (), {
                'GptTokenizer': MockGptTokenizer,
                'WorldTokenizer': MockWorldTokenizer
            })()
            
            spec.loader.exec_module(auto_mod)
            get_model = auto_mod.get_model
            load = auto_mod.load
        else:
            raise ImportError(f"Could not find auto.py at {auto_file}")
        
        # Import utils
        utils_file = os.path.join(lobgen_parent, "lobgen", "jax_rwkv", "utils.py")
        if os.path.exists(utils_file):
            spec = importlib.util.spec_from_file_location("lobgen.jax_rwkv.utils", utils_file)
            utils_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_mod)
            simple_sampler = utils_mod.simple_sampler
        else:
            # Define simple_sampler manually
            def simple_sampler(key, logits, temperature=1.0):
                if temperature == 0.0:
                    return jnp.argmax(logits)
                elif temperature != 1.0:
                    logits = logits / temperature
                return jax.random.categorical(key, logits)
    except Exception as e2:
        print(f"Failed to import jax_rwkv: {e2}")
        raise

from transformers import PreTrainedTokenizerFast

# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
import gymnax_exchange.jaxob.JaxOrderBookArrays as job


# ===== Helpers for pmap sharding =====
def _shard_for_pmap(x, num_devices):
    """Shard array along the leading axis for pmap."""
    if x is None:
        return None
    
    # Check if x is a JAX array or numpy array
    if hasattr(x, 'shape'):
        b = x.shape[0]
        if b % num_devices != 0:
            raise ValueError(
                f"Leading axis {b} must be divisible by num_devices={num_devices} for pmap sharding"
            )
        
        # Reshape to (num_devices, batch_per_device, ...)
        new_shape = (num_devices, b // num_devices) + x.shape[1:]
        print(f"   Sharding {x.shape} -> {new_shape} (batch_per_device: {b // num_devices})")
        return x.reshape(new_shape)
    
    # Check if x is a structured object (like LobState) with array attributes
    elif hasattr(x, '__dict__') or hasattr(x, '_fields'):
        print(f"   Structured object: {type(x).__name__}")
        
        # Check if this is a JAX pytree (like NamedTuple or dataclass)
        try:
            import jax
            tree_structure = jax.tree_structure(x)
            flat_values, tree_def = jax.tree_flatten(x)
            
            print(f"   Found {len(flat_values)} arrays in structured object")
            
            # Check if any arrays have batch dimensions that need sharding
            needs_sharding = False
            for i, val in enumerate(flat_values):
                if hasattr(val, 'shape') and len(val.shape) > 0:
                    print(f"     Array {i}: {val.shape}")
                    if val.shape[0] == num_devices * (val.shape[0] // num_devices):
                        needs_sharding = True
            
            if needs_sharding:
                # Shard each array in the structure
                sharded_values = []
                for val in flat_values:
                    if hasattr(val, 'shape') and len(val.shape) > 0 and val.shape[0] % num_devices == 0:
                        b = val.shape[0]
                        new_shape = (num_devices, b // num_devices) + val.shape[1:]
                        sharded_val = val.reshape(new_shape)
                        print(f"     Sharding array: {val.shape} -> {new_shape}")
                        sharded_values.append(sharded_val)
                    else:
                        sharded_values.append(val)
                
                # Reconstruct the structured object
                return jax.tree_unflatten(tree_def, sharded_values)
            else:
                print(f"   No arrays need sharding, keeping as-is")
                return x
                
        except Exception as e:
            print(f"   Could not process as JAX pytree: {e}")
            print(f"   Keeping as-is: {type(x).__name__}")
            return x
    
    else:
        # For other non-array objects, return as-is
        print(f"   Skipping sharding for non-array object: {type(x).__name__}")
        return x

def _unshard_from_pmap(x):
    """Unshard array from pmap output."""
    if x is None:
        return None
    
    # Check if x is a JAX array or numpy array
    if hasattr(x, 'shape'):
        # Reshape from (num_devices, batch_per_device, ...) back to (batch, ...)
        new_shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
        print(f"   Unsharding {x.shape} -> {new_shape}")
        return x.reshape(new_shape)
    
    # Check if x is a structured object (like LobState) with array attributes
    elif hasattr(x, '__dict__') or hasattr(x, '_fields'):
        print(f"   Structured object: {type(x).__name__}")
        
        try:
            import jax
            flat_values, tree_def = jax.tree_flatten(x)
            
            # Unshard each array in the structure
            unsharded_values = []
            for val in flat_values:
                if hasattr(val, 'shape') and len(val.shape) > 2:
                    # This looks like a sharded array: (num_devices, batch_per_device, ...)
                    new_shape = (val.shape[0] * val.shape[1],) + val.shape[2:]
                    unsharded_val = val.reshape(new_shape)
                    print(f"     Unsharding array: {val.shape} -> {new_shape}")
                    unsharded_values.append(unsharded_val)
                else:
                    unsharded_values.append(val)
            
            # Reconstruct the structured object
            return jax.tree_unflatten(tree_def, unsharded_values)
            
        except Exception as e:
            print(f"   Could not process as JAX pytree: {e}")
            print(f"   Keeping as-is: {type(x).__name__}")
            return x
    
    else:
        # For other non-array objects, return as-is
        print(f"   Skipping unsharding for non-array object: {type(x).__name__}")
        return x


def _shard_encoder_for_pmap(encoder, num_devices):
    """Shard encoder dictionary if it contains batched arrays."""
    if not isinstance(encoder, dict):
        return encoder
    
    sharded_encoder = {}
    for key, value in encoder.items():
        if hasattr(value, 'shape') and value.shape[0] > 1:
            # This is a batched array, shard it
            sharded_encoder[key] = _shard_for_pmap(value, num_devices)
        else:
            # This is not batched, keep as-is
            sharded_encoder[key] = value
    
    return sharded_encoder


def track_midprices_during_messages(
        m_seq_raw_inp: jax.Array,
        book_l2_init: jax.Array,
        tick_size: int,
        step_size: int,
    ) -> jax.Array:
    """
    JIT‑friendly, loop‑free version that computes mid‑prices every
    ``step_size`` messages using `jax.lax.scan`.

    Args:
        m_seq_raw_inp:  (batch, T, msg_dim) – raw decoded messages
        book_l2_init:   (batch, book_dim) – initial L2 state of the orderbook
        tick_size:      Tick size of the instrument
        step_size:      Interval (in number of messages) between successive mid‑price samples

    Returns
    -------
    midprices : jax.Array
        Shape (num_steps, batch).  mid‑price after each `step_size` messages.
    """
    # Number of *complete* chunks of length `step_size`
    num_steps = m_seq_raw_inp.shape[1] // step_size
    if num_steps == 0:
        return jnp.empty((0, m_seq_raw_inp.shape[0]), dtype=jnp.float32)

    # Reshape messages into (num_steps, batch, step_size, msg_dim)
    msgs = m_seq_raw_inp[:, : num_steps * step_size, :]
    msgs = msgs.reshape(
        m_seq_raw_inp.shape[0],          # batch
        num_steps,
        step_size,
        m_seq_raw_inp.shape[2],          # msg_dim
    )
    msgs = jnp.swapaxes(msgs, 0, 1)      # → (num_steps, batch, step_size, msg_dim)

    def scan_step(carry, msg_chunk):
        # carry: current book L2 state  (batch, book_dim)
        sim_init, sim_states = inference.get_sims_vmap(carry, msg_chunk)
        mid_price = inference.batched_get_safe_mid_price(
            sim_init, sim_states, tick_size
        )
        # Extract fresh L2 state to feed next step
        full_l2_state = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(
            sim_states, carry.shape[1]
        )
        # keep only the leading slice so the carry's shape matches the input
        new_l2_state = full_l2_state[:, : carry.shape[1]]
        return new_l2_state, mid_price

    _, midprices = lax.scan(scan_step, book_l2_init, msgs)  # (num_steps, batch)
    return midprices



# def insert_custom_end(
#         m_seq_gen_doubled,
#         b_seq_gen_doubled,
#         msgs_decoded_doubled,
#         l2_book_states_halved,
#         encoder,
#         mid_price,
#         tick_size=100,
#         EVENT_TYPE_i=4,
#         DIRECTION_i=0,
#         order_volume=75,
#         use_relative_volume: bool = False,
#         order_volume_ratio: float = 1.0,
#     ):
    
#     ORDER_ID_i = 77777777
#     sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved[:,-2], msgs_decoded_doubled[:,-1:])
#     if DIRECTION_i == 0:
#         PRICE_i = jax.vmap(sim_init.get_best_ask)(sim_states_init)
#     if DIRECTION_i == 1:
#         PRICE_i = jax.vmap(sim_init.get_best_bid)(sim_states_init)

#     PRICE_i = jnp.expand_dims(PRICE_i, axis=-1)
#     mid_price = jnp.expand_dims(mid_price, axis=-1)

#     TIMEs_i = msgs_decoded_doubled[:, -1:, 8].astype(jnp.int32)
#     TIMEns_i = msgs_decoded_doubled[:, -1:, 9].astype(jnp.int32)

#     batch_size = TIMEns_i.shape[0]

#     best_bid_ask = jax.vmap(sim_init.get_best_bid_and_ask_inclQuants)(sim_states_init)
#     # Insert custom logic for order sizing
#     if use_relative_volume:
#         if DIRECTION_i == 0:
#             SIZE_i = (best_bid_ask[1][:, 1] * order_volume_ratio).astype(jnp.int32)
#         else:
#             SIZE_i = (best_bid_ask[0][:, 1] * order_volume_ratio).astype(jnp.int32)
#     else:
#         if DIRECTION_i == 0:
#             SIZE_i = jnp.minimum(order_volume, best_bid_ask[1][:, 1])
#         else:
#             SIZE_i = jnp.minimum(order_volume, best_bid_ask[0][:, 1])
#     batched_quantity = SIZE_i

#     batched_new_order_id = jnp.array([ORDER_ID_i] * batch_size, dtype=jnp.int32)
#     batched_EVENT_TYPE = jnp.array([EVENT_TYPE_i] * batch_size, dtype=jnp.int32)
#     batched_side = jnp.array([DIRECTION_i] * batch_size, dtype=jnp.int32)
#     batched_p_abs = PRICE_i.squeeze(-1)    
#     batched_time_s = TIMEs_i.squeeze(-1)
#     batched_time_ns = TIMEns_i.squeeze(-1)

#     batched_construct_sim_msg = jax.vmap(inference.construct_sim_msg)
#     batched_sim_msg = batched_construct_sim_msg(
#         batched_EVENT_TYPE,
#         batched_side,
#         batched_quantity,
#         batched_p_abs,
#         batched_new_order_id,
#         batched_time_s,
#         batched_time_ns,
#     )

#     new_sim_state = jax.vmap(sim_init.process_order_array)(sim_states_init, batched_sim_msg)
#     p_mid_new = inference.batched_get_safe_mid_price(sim_init, new_sim_state, tick_size)
#     p_mid_new = p_mid_new[:, None]
#     p_change = ((p_mid_new - mid_price) // tick_size).astype(jnp.int32)
#     book_l2 = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(new_sim_state, 20)
#     new_l2_book_states_halved = jnp.concatenate([l2_book_states_halved, book_l2[:, None, :]], axis=1)
#     new_book_raw = jnp.concatenate([p_change, book_l2], axis=1)
#     new_book_raw = new_book_raw[:, None, :]

#     transform_L2_state_batch = jax.jit(jax.vmap(inference.transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2))
#     new_book = transform_L2_state_batch(new_book_raw, 500, 100)

#     b_seq_gen_doubled = jnp.concatenate([b_seq_gen_doubled, new_book], axis=1)

#     ins_msg = jnp.concatenate([
#         batched_new_order_id.reshape(-1, 1),
#         batched_EVENT_TYPE.reshape(-1, 1),
#         batched_side.reshape(-1, 1),
#         batched_p_abs.reshape(-1, 1),
#         jnp.full((batch_size, 1), 1, dtype=jnp.int32),
#         batched_quantity.reshape(-1, 1),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         batched_time_s.reshape(-1, 1),
#         batched_time_ns.reshape(-1, 1),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#     ], axis=1)
 
#     new_batched_sim_msg = ins_msg[:, None, :]
#     UPDATED_msgs_decoded_doubled = jnp.concatenate([msgs_decoded_doubled, new_batched_sim_msg], axis=1)

#     msg_encoded = jax.vmap(lambda m: encoding.encode_msg(m, encoder))(ins_msg)
#     UPDATED_m_seq_gen_doubled = jnp.concatenate([m_seq_gen_doubled, msg_encoded], axis=1)

#     return UPDATED_m_seq_gen_doubled, b_seq_gen_doubled, UPDATED_msgs_decoded_doubled, new_l2_book_states_halved, p_mid_new

def insert_custom_end(
        m_seq_gen_doubled,
        b_seq_gen_doubled,
        msgs_decoded_doubled,
        l2_book_states_halved,
        encoder,
        mid_price,
        tick_size=100,
        EVENT_TYPE_i=4,
        DIRECTION_i=0,          # 0 = buy (hit ask), 1 = sell (hit bid)
        order_volume=75,        # kept for compat, not used by sizing rule
        use_relative_volume=False,
        order_volume_ratio=1.0,
    ):
    ORDER_ID_i = 77777777
    
    # Fix: l2_book_states_halved is 2D (batch_size, book_dim), not 3D
    # We need to use the last valid L2 state
    print(f"[DEBUG insert_custom_end] l2_book_states_halved.shape: {l2_book_states_halved.shape}")
    if len(l2_book_states_halved.shape) == 2:
        # Shape is (batch_size, book_dim) - use directly
        base_l2 = l2_book_states_halved
    elif len(l2_book_states_halved.shape) == 3:
        # Shape is (batch_size, seq_len, book_dim) - use last element
        base_l2 = l2_book_states_halved[:, -1, :]
    else:
        raise ValueError(f"Unexpected l2_book_states_halved shape: {l2_book_states_halved.shape}")
    
    print(f"[DEBUG insert_custom_end] base_l2.shape: {base_l2.shape}")
    print(f"[DEBUG insert_custom_end] msgs_decoded_doubled.shape: {msgs_decoded_doubled.shape}")
    
    # Get last message or use empty if no messages
    if msgs_decoded_doubled.shape[1] > 0:
        last_msgs = msgs_decoded_doubled[:, -1:, :]
    else:
        # Create empty message placeholder - but we still need valid L2 state
        print("[DEBUG insert_custom_end] No messages, using empty placeholder")
        last_msgs = jnp.zeros((msgs_decoded_doubled.shape[0], 1, msgs_decoded_doubled.shape[2]), dtype=jnp.int32)
    
    print(f"[DEBUG insert_custom_end] base_l2.shape before get_sims: {base_l2.shape}")
    sim_init, sim_states_init = inference.get_sims_vmap(
        base_l2, last_msgs
    )
    print(f"[DEBUG insert_custom_end] sim_states_init type: {type(sim_states_init)}")

    # best price on the side we will aggress
    if DIRECTION_i == 0:
        PRICE_i = jax.vmap(sim_init.get_best_ask)(sim_states_init)
    else:
        PRICE_i = jax.vmap(sim_init.get_best_bid)(sim_states_init)

    PRICE_i   = jnp.expand_dims(PRICE_i, axis=-1)
    mid_price = jnp.expand_dims(mid_price, axis=-1)

    TIMEs_i  = msgs_decoded_doubled[:, -1:, 8].astype(jnp.int32)
    TIMEns_i = msgs_decoded_doubled[:, -1:, 9].astype(jnp.int32)
    batch_size = TIMEns_i.shape[0]

    # === volumes at best levels ===
    # best_bid_ask structure from your code: [0] = bid info, [1] = ask info; [:, 1] = quantity
    best_bid_ask = jax.vmap(sim_init.get_best_bid_and_ask_inclQuants)(sim_states_init)

    # pick available volume on the level we will consume
    avail = jnp.where(
        DIRECTION_i == 0,
        best_bid_ask[1][:, 1],  # ask vol for a buy
        best_bid_ask[0][:, 1],  # bid vol for a sell
    ).astype(jnp.int32)

    # === SIZING RULE (CONFIG-DRIVEN) ===
    # If use_relative_volume: consume a fraction of available L1 volume.
    #   SIZE = floor(avail * order_volume_ratio); if avail>0 and result==0 -> 1
    # Else: fixed-size capped by avail (do not cross levels here).
    if use_relative_volume:
        ratio = jnp.clip(jnp.float32(order_volume_ratio), 0.0, 1.0)
        SIZE_i = jnp.floor(ratio * jnp.float32(avail)).astype(jnp.int32)
        # if there's some liquidity but rounding gave 0, take 1; if avail==0, keep 0
        SIZE_i = jnp.where((avail > 0) & (SIZE_i == 0), 1, SIZE_i)
    else:
        SIZE_i = jnp.minimum(jnp.int32(order_volume), avail)

    batched_quantity   = SIZE_i  # sized by relative fraction or fixed cap
    batched_new_order_id = jnp.full((batch_size,), ORDER_ID_i, dtype=jnp.int32)
    batched_EVENT_TYPE   = jnp.full((batch_size,), EVENT_TYPE_i, dtype=jnp.int32)
    batched_side         = jnp.full((batch_size,), DIRECTION_i, dtype=jnp.int32)
    batched_p_abs        = PRICE_i.squeeze(-1)
    batched_time_s       = TIMEs_i.squeeze(-1)
    batched_time_ns      = TIMEns_i.squeeze(-1)

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
    # book_l2 shape: (batch_size, 80) - 20 levels * 4 = 80
    
    # Fix: l2_book_states_halved is 2D (batch_size, 40) based on debug output
    # book_l2 is (batch_size, 80), we need to take first 40 elements
    book_l2_40 = book_l2[:, :40]  # (batch_size, 40)
    
    # In original code, this would concatenate along sequence dimension
    # But since we're using 2D, just replace the state
    new_l2_book_states_halved = book_l2_40
    
    new_book_raw = jnp.concatenate([p_change, book_l2], axis=1)
    new_book_raw = new_book_raw[:, None, :]

    transform_L2_state_batch = jax.jit(
        jax.vmap(inference.transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2)
    )
    new_book = transform_L2_state_batch(new_book_raw, 500, 100)
    b_seq_gen_doubled = jnp.concatenate([b_seq_gen_doubled, new_book], axis=1)

    ins_msg = jnp.concatenate([
        batched_new_order_id.reshape(-1, 1),
        batched_EVENT_TYPE.reshape(-1, 1),
        batched_side.reshape(-1, 1),
        batched_p_abs.reshape(-1, 1),
        jnp.full((batch_size, 1), 1, dtype=jnp.int32),       # visible flag
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

    new_batched_sim_msg     = ins_msg[:, None, :]
    UPDATED_msgs_decoded_doubled = jnp.concatenate([msgs_decoded_doubled, new_batched_sim_msg], axis=1)

    msg_encoded = jax.vmap(lambda m: encoding.encode_msg(m, encoder))(ins_msg)
    UPDATED_m_seq_gen_doubled = jnp.concatenate([m_seq_gen_doubled, msg_encoded], axis=1)

    return UPDATED_m_seq_gen_doubled, b_seq_gen_doubled, UPDATED_msgs_decoded_doubled, new_l2_book_states_halved, p_mid_new


# ===== RWKV Generation Functions =====

def encode_messages_to_rwkv_tokens(msgs_decoded, tokenizer):
    """
    Convert decoded messages to RWKV token format.
    Each message is converted to CSV format: <time>,value,<event_type>,value,...\n
    Then tokenized.
    """
    try:
        from lobgen.constants import MESSAGE_TOKEN_TYPES
    except ImportError:
        # Fallback if constants not available
        MESSAGE_TOKEN_TYPES = ["<time>", "<event_type>", "<order_id>", "<size>", "<price>", "<direction>"]
    
    batch_size = msgs_decoded.shape[0]
    num_messages = msgs_decoded.shape[1]
    
    # Convert messages to CSV string format
    all_row_strings = []
    for b in range(batch_size):
        batch_strings = []
        for m in range(num_messages):
            msg = msgs_decoded[b, m]
            # Convert message to CSV format: <col>,value,<col>,value,...
            # Format matches evaluate.py exactly: ','.join([f"{col},{val}" for col, val in zip(columns, row) if not pd.isna(val)])
            parts = []
            for i, col in enumerate(MESSAGE_TOKEN_TYPES):
                if i < len(msg):
                    val = int(msg[i])
                    # Include all values (evaluate.py filters NaN, but we have ints, so include all)
                    parts.append(f"{col},{val}")
            if parts:
                batch_strings.append(','.join(parts) + "\n")
        if batch_strings:
            all_row_strings.append(batch_strings)
        else:
            all_row_strings.append(["\n"])  # Empty message
    
    # Tokenize all strings
    tokenized_batch = []
    for batch_strings in all_row_strings:
        try:
            tokenized = tokenizer(batch_strings, return_attention_mask=False, return_tensors="np")["input_ids"]
            # Flatten tokenized sequences
            flat_tokens = np.concatenate([seq for seq in tokenized])
            tokenized_batch.append(flat_tokens)
        except Exception as e:
            print(f"Error tokenizing batch: {e}")
            # Fallback: create empty token sequence
            tokenized_batch.append(np.array([], dtype=np.int32))
    
    # Pad to same length
    if tokenized_batch:
        max_len = max(len(t) for t in tokenized_batch) if tokenized_batch else 1000
        padded = []
        for tokens in tokenized_batch:
            if len(tokens) < max_len:
                padded.append(np.concatenate([tokens, np.zeros(max_len - len(tokens), dtype=np.int32)]))
            else:
                padded.append(tokens[:max_len])
        
        return jnp.array(padded)
    else:
        # Return empty array with proper shape
        return jnp.zeros((batch_size, 1000), dtype=jnp.int32)

def csv_decode_tokens(tokens, tokenizer):
    """
    Decode tokens to CSV string format (similar to evaluate.py).
    Returns CSV string with messages.
    Uses the same logic as evaluate.py for consistency.
    """
    # Convert to numpy if JAX array
    if hasattr(tokens, '__array__'):
        tokens = np.array(tokens)
    
    partial_ans = ""
    ans = ""
    num_newlines = 0
    had_newline = True
    was_time = False
    time_ans = ""
    num_tags = 0
    
    for t in tokens:
        t_int = int(t)
        if t_int == 0:  # padding token
            continue
        
        # Check for newline token (may be 36 or tokenizer-specific)
        # Try to decode to see if it's a newline
        try:
            decoded_char = tokenizer.decode([t_int], skip_special_tokens=False).strip()
            if decoded_char == '\n' or t_int == 36:  # newline token
                partial_ans += ",null\n"
                if num_tags == 6:
                    ans += partial_ans
                    num_newlines += 1
                else:
                    # Skip broken messages but don't print every time
                    if num_tags > 0:
                        pass  # Message had some tags but not complete
                had_newline = True
                num_tags = 0
                partial_ans = ""
                time_ans = ""
                was_time = False
                if num_newlines >= 500:  # Limit messages
                    break
                continue
        except:
            pass
        
        # Decode token
        try:
            true_token = tokenizer.decode([t_int], skip_special_tokens=False).strip()
        except:
            continue
        
        is_tag = ("<" in true_token)
        if is_tag:
            num_tags += 1
            if was_time:
                time_str = str(time_ans)
                last_time = time_str[-9:]
                while len(last_time) > 0 and last_time[-1] == "0":
                    last_time = last_time[:-1]
                partial_ans += time_str[:-9] + "." + last_time
                time_ans = ""
                was_time = False
            if "time" in true_token.lower():
                was_time = True
            if not had_newline:
                partial_ans += ","
        else:
            if was_time:
                time_ans += true_token
            else:
                partial_ans += true_token
        had_newline = False
    
    # Add last message if valid
    if num_tags >= 5 and partial_ans:
        ans += partial_ans
        if not ans.endswith('\n'):
            ans += "\n"
    
    return ans

def parse_csv_to_messages(csv_string, max_messages=100):
    """
    Parse CSV string to message arrays.
    CSV format (without tags): time,event_type,order_id,size,price,direction,null
    Example: 19135.59139624,3,78233502,50,892100,-1,null
    Returns: numpy array of shape (num_messages, 14) with message fields
    """
    if not csv_string or csv_string.strip() == "":
        return np.zeros((0, 14), dtype=np.int32)
    
    try:
        # Split by newlines to get individual messages
        lines = [l.strip() for l in csv_string.strip().split('\n') if l.strip()]
        messages = []
        
        for line in lines:
            if not line:
                continue
            
            # Parse comma-separated values
            # Format: time,event_type,order_id,size,price,direction,null
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) < 6:  # Need at least 6 values (time, event_type, order_id, size, price, direction)
                continue
            
            msg = np.zeros(14, dtype=np.int32)
            
            try:
                # Parse time (first field) - can be float like 19135.59139624
                time_str = parts[0]
                if '.' in time_str:
                    time_parts = time_str.split('.')
                    msg[8] = int(float(time_parts[0]))  # TIMEs_i (seconds)
                    ns_str = time_parts[1].ljust(9, '0')[:9]  # Pad to 9 digits for nanoseconds
                    msg[9] = int(ns_str)  # TIMEns_i (nanoseconds)
                else:
                    msg[8] = int(float(time_str))
                    msg[9] = 0
                
                # Parse event_type (second field)
                if len(parts) > 1 and parts[1] not in ['null', '']:
                    msg[1] = int(float(parts[1]))  # EVENT_TYPE_i
                
                # Parse order_id (third field)
                if len(parts) > 2 and parts[2] not in ['null', '']:
                    msg[0] = int(float(parts[2]))  # ORDER_ID_i
                
                # Parse size (fourth field)
                if len(parts) > 3 and parts[3] not in ['null', '']:
                    msg[5] = int(float(parts[3]))  # SIZE_i
                
                # Parse price (fifth field)
                if len(parts) > 4 and parts[4] not in ['null', '']:
                    msg[4] = int(float(parts[4]))  # PRICE_i
                
                # Parse direction (sixth field)
                if len(parts) > 5 and parts[5] not in ['null', '']:
                    msg[2] = int(float(parts[5]))  # DIRECTION_i
                
                # Only add message if it has essential fields
                if msg[1] != 0 or msg[0] != 0:  # EVENT_TYPE or ORDER_ID
                    messages.append(msg)
                    
            except (ValueError, IndexError, OverflowError) as e:
                # Skip invalid lines
                continue
            
            if len(messages) >= max_messages:
                break
        
        if len(messages) == 0:
            # Debug: print first 500 chars of CSV to see what we got
            print(f"Warning: No messages parsed from CSV. First 500 chars: {csv_string[:500]}")
            return np.zeros((0, 14), dtype=np.int32)
        
        return np.array(messages, dtype=np.int32)
        
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((0, 14), dtype=np.int32)

def decode_rwkv_tokens_to_messages(tokens, tokenizer, encoder_dict=None, max_messages_per_batch=100):
    """
    Decode RWKV tokens back to message format.
    
    Args:
        tokens: Token array, shape (batch_size, seq_len) or (seq_len,)
        tokenizer: Tokenizer for decoding
        encoder_dict: Not used for RWKV, kept for compatibility
        max_messages_per_batch: Maximum messages to extract per batch
    
    Returns:
        msgs_decoded: Array of shape (batch_size, num_messages, 14)
    """
    # Handle single sequence
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    
    batch_size = tokens.shape[0]
    all_messages = []
    
    for b in range(batch_size):
        # Decode tokens to CSV string
        csv_string = csv_decode_tokens(tokens[b], tokenizer)
        
        # Parse CSV to messages
        messages = parse_csv_to_messages(csv_string, max_messages=max_messages_per_batch)
        all_messages.append(messages)
        
        # Debug first batch
        if b == 0:
            print(f"Batch 0: CSV length={len(csv_string)}, parsed {len(messages)} messages")
            if len(messages) > 0:
                print(f"  First message: {messages[0]}")
            elif len(csv_string) > 0:
                print(f"  CSV sample (first 300 chars): {csv_string[:300]}")
    
    # Pad to same length
    max_len = max(len(m) for m in all_messages) if all_messages else 0
    if max_len == 0:
        print(f"Warning: No messages decoded from any batch!")
        return jnp.zeros((batch_size, 0, 14), dtype=jnp.int32)
    
    padded = []
    for messages in all_messages:
        if len(messages) < max_len:
            padding = np.zeros((max_len - len(messages), 14), dtype=np.int32)
            padded.append(np.concatenate([messages, padding], axis=0))
        else:
            padded.append(messages[:max_len])
    
    return jnp.array(padded, dtype=jnp.int32)

def process_long_seq_rwkv(tokens, state, length, forward_jit, params, padding=128):
    """Process long sequence in chunks for RWKV."""
    full_instruction_length = tokens.shape[-1]
    instruction_len = length
    x = (jnp.zeros_like(params['emb']['weight'][:, 0]), state)
    
    def inner_loop(x, i):
        (true_out, state) = x
        cur_len = jnp.minimum(padding, instruction_len - i)
        out, new_state = forward_jit(
            jax.lax.dynamic_slice_in_dim(tokens, i, padding), 
            state, params, cur_len
        )
        state = jax.lax.cond(cur_len <= 0, lambda: state, lambda: new_state)
        true_out = jax.lax.cond(cur_len <= 0, lambda: true_out, lambda: out[cur_len-1])
        return (true_out, state), 0

    (true_out, state), _ = jax.lax.scan(
        inner_loop, x, jnp.arange(0, full_instruction_length, padding)
    )
    return true_out, state

def sample_tokens_rwkv(out, state, key, forward_jit, params):
    """Sample next token from RWKV output."""
    key, _key = jax.random.split(key)
    token = simple_sampler(key, out)
    out, state = forward_jit([token], state, params, 1)
    return out[0], state, token, _key

# Vectorized versions
v_process_long_seq_rwkv = None  # Will be set up after model loading
v_sample_tokens_rwkv = None  # Will be set up after model loading

def run_generation_scenario(
        n_samples: int,
        batch_size: int,
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state,  # rwkv_params for RWKV
        model,  # None for RWKV
        batchnorm: bool,  # Not used for RWKV
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
        EVENT_TYPE_i=4,
        DIRECTION_i=0,
        order_volume=75,
        use_relative_volume: bool = False,
        order_volume_ratio: float = 1.0,
        use_sample_file: bool = False,
        sample_file_path: Optional[str] = None,
        start_batch: int = 0,
        end_batch: int = -1,
        # RWKV-specific parameters
        RWKV_model=None,
        rwkv_params=None,
        tokenizer=None,
        forward_jit=None,
        v_process_long_seq_rwkv=None,
        v_sample_tokens_rwkv=None,
    ):

    # rng, rng_ = jax.random.split(rng)
    # if sample_all:
    #     sample_i = jnp.arange(
    #         len(ds) // batch_size * batch_size,
    #         dtype=jnp.int32
    #     ).reshape(-1, batch_size).tolist()
    # else:
    #     assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'

    #     sample_i = jax.random.choice(
    #         rng_,
    #         jnp.arange(len(ds), dtype=jnp.int32),
    #         shape=(n_samples // batch_size, batch_size),
    #         replace=False
    #     ).tolist()
    # rng, rng_ = jax.random.split(rng)

    
    rng, rng_ = jax.random.split(rng)

    if use_sample_file:
        assert sample_file_path is not None, "Path to sample file not provided"
        with open(sample_file_path, "r") as f:
            sample_i_full = json.load(f)
        # режем по start/end
        sample_i = sample_i_full[start_batch: end_batch if end_batch != -1 else None]
        # проверяем, что каждый батч нужного размера
        for i, batch in enumerate(sample_i):
            assert len(batch) == batch_size, (
                f"Batch {i} has incorrect size {len(batch)}, expected {batch_size}"
            )
    else:
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

    # Align saved folders with heuristic_historical_scenario_run_quantile_fixing.py
    save_folder.joinpath('msgs_decoded_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('b_seq_gen_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('mid_price').mkdir(exist_ok=True, parents=True)

    base_save_folder = save_folder

    transform_L2_state_batch = jax.jit(jax.vmap(transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2))

    num_iterations = num_insertions + num_coolings

    print('sample_i:', sample_i)

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
                m_seq_raw_inp = msg_seq_raw[:, : n_msgs]  # n_msgs = n_messages = 500 (context messages)

                m_seq_np = np.array(jax.device_get(m_seq_inp))
                b_seq_np = np.array(jax.device_get(b_seq_inp))
                msgs_decoded_np = np.array(jax.device_get(m_seq_raw_inp))
                l2_book_states_np = np.array(jax.device_get(book_l2_init))

                # Do not save intermediate inputs or initial states

                sim_init, sim_states_init = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp)

                # Track midprices for initial context messages (500 messages)
                midprices_batch = track_midprices_during_messages(
                    m_seq_raw_inp,
                    book_l2_init,
                    tick_size,
                    midprice_step_size,
                )
                midprices = list(midprices_batch)            # start new history
                proc_msgs_numb += m_seq_raw_inp.shape[1]     # advance counter
                print(f"[ITERATION 1] Tracked {len(midprices_batch)} midprice steps from {m_seq_raw_inp.shape[1]} context messages (step_size={midprice_step_size})")

            else: 
                m_seq = m_seq_gen_doubled
                b_seq = b_seq_gen_doubled
                msg_seq_raw = msgs_decoded_doubled
                m_seq_inp = m_seq
                b_seq_inp = b_seq
                m_seq_raw_inp = msg_seq_raw
                # l2_book_states_halved is 2D (batch, book_dim), not 3D, so use it directly
                # In original code it was 3D (batch, seq, book_dim) and [:,-2] got second-to-last sequence element
                sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved, m_seq_raw_inp[:,-1:])
            
            # === PMAP SHARDING ===
            # Get number of devices for sharding
            num_devices = jax.local_device_count()
            print(f"\n=== PMAP SHARDING (batch_size={batch_size}, num_devices={num_devices}) ===")
            
            # shard arrays along the leading batch axis for pmap
            print("Sharding inputs for pmap:")
            m_seq_inp_sharded       = _shard_for_pmap(m_seq_inp, num_devices)
            b_seq_inp_sharded       = _shard_for_pmap(b_seq_inp, num_devices)
            sim_states_init_sharded = _shard_for_pmap(sim_states_init, num_devices)
            
            # Create RNG keys for the batch and shard them
            # generate_batched expects a batch of keys, one per sample
            rng_keys = jax.random.split(rng_, batch_size)  # Shape: (batch_size, 2)
            rng_keys_sharded = _shard_for_pmap(rng_keys, num_devices)
            print(f"   RNG keys: {rng_keys.shape} -> {rng_keys_sharded.shape}")
            
            # The issue is that encoder contains batched arrays that need to be broadcast
            # For pmap to work, all batched arrays must either be sharded consistently 
            # or be single values that can be broadcast
            
            # The encoder might contain nested structures. Let's examine it more thoroughly
            def process_encoder_recursive(obj, depth=0):
                indent = "  " * depth
                if isinstance(obj, dict):
                    processed = {}
                    print(f"{indent}Dict with keys: {list(obj.keys())}")
                    for key, value in obj.items():
                        print(f"{indent}{key}:")
                        processed[key] = process_encoder_recursive(value, depth + 1)
                    return processed
                elif isinstance(obj, (list, tuple)):
                    print(f"{indent}{type(obj).__name__} with {len(obj)} elements")
                    if hasattr(obj, '__len__') and len(obj) > 0:
                        first_elem = obj[0]
                        if hasattr(first_elem, 'shape'):
                            print(f"{indent}  First element shape: {first_elem.shape}")
                            if len(first_elem.shape) > 0 and first_elem.shape[0] == batch_size:
                                print(f"{indent}  -> Extracting first sample: {first_elem.shape} -> {first_elem[0].shape}")
                                return tuple(elem[0] if hasattr(elem, 'shape') and len(elem.shape) > 0 and elem.shape[0] == batch_size else elem for elem in obj)
                    return obj
                elif hasattr(obj, 'shape'):
                    print(f"{indent}Array: {obj.shape}")
                    if len(obj.shape) > 0 and obj.shape[0] == batch_size:
                        print(f"{indent}  -> Extracting first sample: {obj.shape} -> {obj[0].shape}")
                        return obj[0]
                    return obj
                else:
                    print(f"{indent}{type(obj)}: {obj}")
                    return obj
            
            print("Processing encoder structure:")
            encoder_broadcast = process_encoder_recursive(encoder)
            
            print("Using prepared encoder for broadcasting")
            
            # === RWKV GENERATION ===
            # Convert decoded messages to RWKV token format
            print("Converting messages to RWKV token format...")
            m_seq_inp_tokens = encode_messages_to_rwkv_tokens(m_seq_raw_inp, tokenizer)
            m_seq_inp_tokens = jnp.array(m_seq_inp_tokens)
            
            # Calculate actual tokens per message for context
            actual_tokens = m_seq_inp_tokens.shape[1]
            actual_messages = m_seq_raw_inp.shape[1]
            tokens_per_msg_context = actual_tokens / actual_messages if actual_messages > 0 else 0
            print(f"Context: {actual_messages} messages → {actual_tokens} tokens ({tokens_per_msg_context:.1f} tokens/message)")
            
            # Initialize RWKV states
            initial_state = RWKV_model.default_state(rwkv_params)
            states = jnp.repeat(initial_state[None], batch_size, axis=0)
            
            # Process input sequence to get initial state
            input_lengths = jnp.array([m_seq_inp_tokens.shape[1]] * batch_size)
            
            # Process long sequences using vmap
            def process_single_seq(tokens, state, length):
                return process_long_seq_rwkv(tokens, state, length, forward_jit, rwkv_params, padding=128)
            
            v_process_seq = jax.vmap(process_single_seq, in_axes=(0, 0, 0))
            outs, states = v_process_seq(m_seq_inp_tokens, states, input_lengths)
            
            # Generate tokens
            print(f"Generating {n_gen_msgs} messages with RWKV...")
            all_tokens = []
            current_outs = outs
            current_states = states
            current_keys = rng_keys
            
            # Calculate tokens per message based on actual context tokenization
            # Use the same ratio as context messages (more accurate than hardcoded 50)
            if actual_messages > 0:
                tokens_per_msg = max(1, int(actual_tokens / actual_messages))
            else:
                tokens_per_msg = 50  # Empirical fallback if no context
            total_tokens_to_gen = n_gen_msgs * tokens_per_msg
            print(f"Generating {n_gen_msgs} messages × {tokens_per_msg} tokens/message = {total_tokens_to_gen} total tokens")
            
            # Sample tokens function
            def sample_single_token(out, state, key):
                return sample_tokens_rwkv(out, state, key, forward_jit, rwkv_params)
            
            v_sample_token = jax.vmap(sample_single_token, in_axes=(0, 0, 0))
            
            for t in tqdm(range(total_tokens_to_gen), desc="Generating tokens"):
                current_outs, current_states, tokens, current_keys = v_sample_token(
                    current_outs, current_states, current_keys
                )
                all_tokens.append(tokens)
            
            # Stack all generated tokens
            m_seq_gen_tokens = jnp.stack(all_tokens, axis=1)  # (batch_size, n_tokens)
            
            # Decode tokens back to messages
            print("Decoding tokens to messages...")
            print(f"Generated tokens shape: {m_seq_gen_tokens.shape}")
            print(f"Sample tokens (first 50): {m_seq_gen_tokens[0, :50]}")
            
            msgs_decoded = decode_rwkv_tokens_to_messages(
                np.array(m_seq_gen_tokens),  # Convert to numpy for processing
                tokenizer, 
                encoder,
                max_messages_per_batch=n_gen_msgs
            )
            msgs_decoded = jnp.array(msgs_decoded)  # Convert back to JAX array
            
            print(f"Decoded messages shape: {msgs_decoded.shape}")
            if msgs_decoded.shape[1] > 0:
                print(f"Sample decoded message (first): {msgs_decoded[0, 0]}")
            else:
                print("Warning: No messages were decoded from tokens!")
                # Try to decode a sample to see what we get
                sample_tokens = np.array(m_seq_gen_tokens[0, :100])
                sample_csv = csv_decode_tokens(sample_tokens, tokenizer)
                print(f"Sample CSV (first 200 chars): {sample_csv[:200]}")
            
            # Update book states based on generated messages
            # IMPORTANT: Save the book state BEFORE processing current messages for midprice tracking
            # This matches S5 version: l2_book_states[:, 0, :40] - initial state before processing msgs_decoded_doubled
            if iteration == 1:
                # In iteration 1, we need the state AFTER processing context messages (m_seq_raw_inp)
                # This is the state before processing generated messages (msgs_decoded)
                # We need to process context messages to get the final state
                sim_init_context, sim_states_context = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp)
                # Get L2 state after processing context
                l2_after_context = jax.vmap(sim_init_context.get_L2_state, in_axes=(0, None))(
                    sim_states_context, book_l2_init.shape[1]
                )
                book_l2_before_current_msgs = l2_after_context[:, :book_l2_init.shape[1]]  # State after context, before generated
            else:
                # Use final state from previous iteration (state BEFORE current iteration's messages)
                book_l2_before_current_msgs = l2_book_states_halved
            
            if msgs_decoded.shape[1] > 0:
                
                # Process messages through simulator to get book states
                # Use book_l2_before_current_msgs as initial state
                sim_init, sim_states = inference.get_sims_vmap(
                    book_l2_before_current_msgs, 
                    msgs_decoded
                )
                
                # Get L2 states after processing all messages
                # sim_states is a sequence, we need to get L2 state for each step
                # For now, get final state only
                final_sim_states = sim_states  # This is the final state after all messages
                l2_book_states_full = jax.vmap(
                    sim_init.get_L2_state, 
                    in_axes=(0, None)
                )(final_sim_states, 20)  # 20 levels -> (batch, 80)
                
                # Get final state - take first 40 elements to match expected shape
                # Original code uses [:40] to get first 40 elements
                l2_book_states_halved = l2_book_states_full[:, :40]  # (batch, 40)
                
                # Transform to book sequence format
                # Calculate mid prices
                mid_prices_gen = inference.batched_get_safe_mid_price(
                    sim_init, sim_states, tick_size
                )
                
                # Get initial mid price (before processing msgs_decoded)
                # Use book_l2_before_current_msgs - the state BEFORE processing current messages
                empty_msgs = jnp.zeros((batch_size, 0, 14), dtype=jnp.int32)
                init_sim, init_sim_states = inference.get_sims_vmap(book_l2_before_current_msgs, empty_msgs)
                init_mid = inference.batched_get_safe_mid_price(
                    init_sim, init_sim_states, tick_size
                )
                
                # Calculate price changes
                if len(mid_prices_gen.shape) == 1:
                    mid_prices_gen = mid_prices_gen[:, None]
                if len(init_mid.shape) == 1:
                    init_mid = init_mid[:, None]
                
                p_changes = ((mid_prices_gen - init_mid) // tick_size).astype(jnp.int32)
                
                # Combine price changes with L2 states
                book_raw = jnp.concatenate([p_changes, l2_book_states_full], axis=1)
                book_raw = book_raw[:, None, :]  # Add sequence dimension
                
                # Transform L2 state
                transform_L2_state_batch = jax.jit(
                    jax.vmap(transform_L2_state, in_axes=(0, None, None)), 
                    static_argnums=(1, 2)
                )
                b_seq_gen = transform_L2_state_batch(book_raw, n_vol_series, tick_size)
            else:
                # No messages generated, use placeholders
                print("Warning: No messages decoded, using placeholders")
                b_seq_gen = jnp.zeros((batch_size, 1, b_seq_inp.shape[-1]), dtype=jnp.float32)
                # Keep previous l2_book_states_halved or use initial
                if iteration == 1:
                    l2_book_states_halved = book_l2_init
                # else: keep previous l2_book_states_halved
            
            num_errors = jnp.array(0)  # TODO: Calculate actual errors
            
            # Set outputs
            m_seq_gen = m_seq_gen_tokens
            m_seq_gen_doubled = m_seq_gen_tokens  # Tokenized output
            b_seq_gen_doubled = b_seq_gen
            msgs_decoded_doubled = msgs_decoded

            print(f'\n\nsuccessfully generated iteration no. {iteration}')
            
            # Track midprices if we have messages
            # IMPORTANT: Use book_l2_before_current_msgs (state BEFORE processing msgs_decoded_doubled)
            # This matches S5 version: l2_book_states[:, 0, :40] - initial state before processing messages
            # NOTE: In iteration 1, midprices already contains context messages, so we extend with generated messages
            if msgs_decoded_doubled.shape[1] > 0:
                # Use the book state BEFORE processing current messages
                # This ensures continuous tracking: state after previous iterations, before current messages
                midprices_batch = track_midprices_during_messages(
                    msgs_decoded_doubled,
                    book_l2_before_current_msgs,  # State BEFORE processing msgs_decoded_doubled
                    tick_size,
                    midprice_step_size,
                )
                # midprices_batch shape: (num_steps, batch) where num_steps = msgs_decoded_doubled.shape[1] // midprice_step_size
                # With step_size=1: num_steps = msgs_decoded_doubled.shape[1] (one midprice per message)
                midprices.extend(list(midprices_batch))
                proc_msgs_numb += msgs_decoded_doubled.shape[1]
                expected_steps = msgs_decoded_doubled.shape[1] // midprice_step_size
                print(f"[ITERATION {iteration}] Tracked {len(midprices_batch)} midprice steps from {msgs_decoded_doubled.shape[1]} generated messages (step_size={midprice_step_size}, expected={expected_steps})")
                print(f"[ITERATION {iteration}] Total midprices so far: {len(midprices)}")
            else:
                print("Warning: No messages to track midprices for")

 
            if iteration <= num_insertions:
                print(">> INSERTING CUSTOM ORDER")
                # Check if we have messages and valid L2 state
                if msgs_decoded_doubled.shape[1] == 0:
                    print("Warning: No messages to insert custom order after. Using initial book state.")
                    # Use initial book state if no messages generated
                    if iteration == 1:
                        base_l2_for_insert = book_l2_init
                    else:
                        base_l2_for_insert = l2_book_states_halved
                    # Create empty message array for insert_custom_end
                    empty_msgs = jnp.zeros((batch_size, 1, msgs_decoded_doubled.shape[2]), dtype=jnp.int32)
                    msgs_decoded_doubled = empty_msgs
                
                # Ensure l2_book_states_halved has correct shape
                if len(l2_book_states_halved.shape) == 2:
                    base_l2_for_insert = l2_book_states_halved
                elif len(l2_book_states_halved.shape) == 3:
                    base_l2_for_insert = l2_book_states_halved[:, -1, :]
                else:
                    base_l2_for_insert = l2_book_states_halved
                
                m_seq_gen_doubled, b_seq_gen_doubled, msgs_decoded_doubled, l2_book_states_halved, p_mid_new = insert_custom_end(
                    m_seq_gen_doubled,
                    b_seq_gen_doubled,
                    msgs_decoded_doubled,
                    base_l2_for_insert,  # Use properly shaped L2 state
                    encoder,
                    midprices[-1] if len(midprices) > 0 else jnp.array([858200.0] * batch_size),  # Fallback mid price
                    tick_size,
                    EVENT_TYPE_i,
                    DIRECTION_i,
                    order_volume,
                    use_relative_volume,
                    order_volume_ratio,
                )
                midprices.append(jnp.squeeze(p_mid_new, axis=-1))
                proc_msgs_numb += 1
                
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
                # l2_book_states_halved is already defined from previous iterations

                m_seq_np = np.array(jax.device_get(m_seq_gen_doubled))
                b_seq_np = np.array(jax.device_get(b_seq_gen_doubled))
                msgs_decoded_np = np.array(jax.device_get(msgs_decoded_doubled))
                l2_book_states_np = np.array(jax.device_get(l2_book_states_halved))
                mid_price_np = np.array(jax.device_get(midprices))

            # Save only the same artifacts as heuristic_historical_scenario_run_quantile_fixing.py
            np.save(os.path.join(base_save_folder, 'msgs_decoded_doubled', f'msgs_decoded_doubled_batch_{batch_i}_iter_{iteration}.npy'), msgs_decoded_np)
            np.save(os.path.join(base_save_folder, 'mid_price', f'mid_price_batch_{batch_i}_iter_{iteration}.npy'), mid_price_np)
            np.save(os.path.join(base_save_folder, 'b_seq_gen_doubled', f'b_seq_gen_doubled_batch_{batch_i}_iter_{iteration}.npy'), b_seq_np)

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

# def parse_args():
#     p = argparse.ArgumentParser(description="Run LOB inference scenario")
#     p.add_argument(
#         "--config", "-c", 
#         type=str, 
#         default="/app/1_run_exp_aggresive_scenario.yaml",
#         help="Path to your YAML config file"
#     )
#     return p.parse_args()

# def main():
#     print(f"JAX backend platform: {jax.lib.xla_bridge.get_backend().platform}")
    
#     args = parse_args()
#     # load YAML config
#     with open(args.config, "r") as f:
#         cfg = yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser(description="Run LOB inference scenario")
    p.add_argument(
        "--config", "-c", 
        type=str, 
        default="1_run_exp_aggresive_scenario",
        help="Name of the config file (without .yaml extension) or full path to YAML config file"
    )
    return p.parse_args()

def main():
    print(f"JAX backend platform: {jax.lib.xla_bridge.get_backend().platform}")

    args = parse_args()
    
    # Determine config file path
    if args.config.endswith('.yaml'):
        # Full path provided
        config_path = args.config
    else:
        # Just the name provided, construct path
        # Try current directory first, then /app
        if os.path.exists(f"{args.config}.yaml"):
            config_path = f"{args.config}.yaml"
        else:
            config_path = f"/app/{args.config}.yaml"
    
    print(f"Loading config from: {config_path}")
    
    # load YAML config
    with open(config_path, "r") as f:
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
    use_relative_volume = cfg["use_relative_volume"]
    order_volume_ratio = cfg["order_volume_ratio"]

    # num_devices  = jax.local_device_count() 
    # print(f'num_devices: ', num_devices)

    use_sample_file  = cfg["use_sample_file"]
    sample_file_path = cfg["sample_file_path"]
    start_batch      = cfg["start_batch"]
    end_batch        = cfg["end_batch"]
    
    num_devices = jax.local_device_count()  # 6
    print(f'num_devices: ', num_devices)
    
    # Validate that batch_size is compatible with num_devices for pmap sharding
    if batch_size % num_devices != 0:
        print(f"Warning: batch_size ({batch_size}) is not divisible by num_devices ({num_devices})")
        print(f"This will cause pmap sharding to fail. Consider using a batch_size that's divisible by {num_devices}")
        print(f"Valid batch_sizes: {[i * num_devices for i in range(1, 11)]}")
        raise ValueError(f"batch_size ({batch_size}) must be divisible by num_devices ({num_devices}) for pmap sharding")
    
    print(f"✅ batch_size ({batch_size}) is compatible with num_devices ({num_devices})")
    print(f"   Each GPU will process {batch_size // num_devices} samples")

    
    # Experiment folder
    exp_folder = create_next_experiment_folder(save_folder)
    print("Experiment dir:", exp_folder)
    wandb.run.summary["experiment_dir"] = str(exp_folder)

    with open(exp_folder / "used_config.yaml", "w") as f_out:
        yaml.dump(cfg, f_out)
    
    # Setup logging to experiment folder
    log_file_path = exp_folder / "job.log"
    print(f"Redirecting all output to: {log_file_path}")
    
    # Redirect stdout and stderr to log file
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    # Print initial info to log
    print(f"Experiment started at: {datetime.now()}")
    print(f"Experiment folder: {exp_folder}")
    print(f"Configuration: {cfg}")
    print("=" * 80)

    # Log config file as artifact
    artifact = wandb.Artifact(name="used_config", type="config")
    artifact.add_file(str(exp_folder / "used_config.yaml"))
    wandb.log_artifact(artifact)

    # Load RWKV model
    print("Loading RWKV model from", ckpt_path)
    
    # Get RWKV config from YAML
    model_choice = cfg.get("model_choice", "6g0.1B")
    rwkv_type = cfg.get("rwkv_type", "AssociativeScanRWKV")
    tokenizer_file = cfg.get("tokenizer_file", "/app/lobgen/tokenizers/lob_tok.json")
    dtype_str = cfg.get("dtype", None)
    
    print(f"Model choice: {model_choice}, RWKV type: {rwkv_type}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_file}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        clean_up_tokenization_spaces=False
    )
    
    # Get RWKV model class first (without loading weights)
    print("Initializing RWKV model class...")
    # Import RWKV module directly
    from lobgen.jax_rwkv import jax_rwkv as rwkv6
    if rwkv_type == "AssociativeScanRWKV":
        RWKV = rwkv6.base_rwkv.AssociativeScanRWKV
    elif rwkv_type == "ScanRWKV":
        RWKV = rwkv6.base_rwkv.ScanRWKV
    else:
        RWKV = rwkv6.base_rwkv.BaseRWKV
    
    # Load checkpoint directly (bypass get_model to avoid device parameter issue)
    if ckpt_path and ckpt_path != "":
        print(f"Loading RWKV checkpoint from {ckpt_path}")
        ckpt_file = Path(ckpt_path)
        
        # Try to find params.model file
        if ckpt_file.is_file() and ckpt_file.suffix == '.model':
            rwkv_params = load(str(ckpt_file))
            print(f"Loaded checkpoint from {ckpt_file}")
        elif ckpt_file.is_dir():
            # Look for params.model in subdirectories (common structure: step/params.model)
            params_files = list(ckpt_file.rglob("params.model"))
            if params_files:
                # Use the latest one (highest step number)
                params_files.sort(key=lambda x: int(x.parent.name) if x.parent.name.isdigit() else 0, reverse=True)
                rwkv_params = load(str(params_files[0]))
                print(f"Loaded checkpoint from {params_files[0]}")
            else:
                # Try direct .model files
                model_files = list(ckpt_file.glob("*.model"))
                if model_files:
                    rwkv_params = load(str(model_files[0]))
                    print(f"Loaded from {model_files[0]}")
                else:
                    raise FileNotFoundError(f"No .model files found in {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint path {ckpt_path} is not valid")
    else:
        # Try to get model using get_model (may fail due to device parameter issue)
        print("Warning: No ckpt_path provided, trying to load model using get_model...")
        try:
            RWKV, rwkv_params, _ = get_model(model_choice, dtype_str, rwkv_type=rwkv_type)
        except (TypeError, NotImplementedError) as e:
            if "device" in str(e) or "hf_hub_download" in str(e):
                raise RuntimeError(
                    f"Could not load model using get_model: {e}. "
                    f"Please provide ckpt_path in config to load checkpoint directly."
                ) from e
            raise
    
    # JIT compile forward function
    forward_jit = jax.jit(RWKV.forward)
    
    # Create vectorized functions
    v_forward = jax.jit(jax.vmap(forward_jit, in_axes=(0, 0, None, 0)))
    
    # Create process_long_seq function with proper closure
    def make_process_long_seq(forward_fn, params):
        def process_long_seq(tokens, state, length, padding=128):
            return process_long_seq_rwkv(tokens, state, length, forward_fn, params, padding)
        return process_long_seq
    
    v_process_long_seq_rwkv = jax.jit(jax.vmap(make_process_long_seq(forward_jit, rwkv_params)))
    
    # Create sample_tokens function with proper closure
    def make_sample_tokens(forward_fn, params):
        def sample_tokens(out, state, key):
            return sample_tokens_rwkv(out, state, key, forward_fn, params)
        return sample_tokens
    
    v_sample_tokens_rwkv = jax.jit(jax.vmap(make_sample_tokens(forward_jit, rwkv_params)))
    
    print("RWKV model loaded successfully")
    
    # For compatibility, create placeholder state and model objects
    state = rwkv_params  # Use params as state for compatibility
    model = None  # Not used for RWKV

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
        state,  # rwkv_params
        model,  # None for RWKV
        None,  # batchnorm not used for RWKV
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
        use_relative_volume,
        order_volume_ratio,
        use_sample_file,
        sample_file_path,
        start_batch,
        end_batch,
        # RWKV-specific parameters
        RWKV_model=RWKV,
        rwkv_params=rwkv_params,
        tokenizer=tokenizer,
        forward_jit=forward_jit,
        v_process_long_seq_rwkv=v_process_long_seq_rwkv,
        v_sample_tokens_rwkv=v_sample_tokens_rwkv,
    )

    # log any returned metrics/artifacts
    wandb.log({"finished": True})
    wandb.save(str(exp_folder / "*"))
    
    # Close log file and restore stdout/stderr
    print(f"Experiment completed at: {datetime.now()}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Logs saved to: {log_file_path}")

if __name__ == "__main__":
    main()