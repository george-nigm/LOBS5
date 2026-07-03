"""
inference_w_insertions.py  (lob_impact.core)

Impact-specific variant of lob.inference_no_errcorr with insertion_schedule support.
Allows inserting aggressive orders at specific steps during single-call generation.
Moved here from lob/ so the LOB-Impact submodule owns its own impact-generation logic.
"""

from jax import config
config.update("jax_disable_jit", False)
#config.update("jax_disable_jit", True)

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
import math
import time

import lob.validation_helpers as valh
import lob.evaluation as eval
import preproc as preproc
# from preproc import transform_L2_state_gpu
import lob.encoding as encoding
from lob.encoding import Message_Tokenizer, Vocab
from lob.lobster_dataloader import LOBSTER_Dataset
import chex

# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
# from gym_exchange.environment.base_env.assets.action import OrderIdGenerator

REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

# indices for DECODED message fields
ORDER_ID_i = 0
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7
TIMEs_i = 8
TIMEns_i = 9
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13

# Generated ids count DOWN from n_msg_todo; insertion steps decrement twice, so without an
# offset the tail ids underflow to 0..-num_insertions and collide with simulator sentinels
# (-1 empty slot, INITID -2, NEGATIVE_RETURN_ID -99). Offset keeps them strictly positive.
GEN_ORDER_ID_OFFSET = 1000

l2_state_n = 10

# ENCODED TOKEN INDICES
# time tokens aren't generated but calculated using delta_t
# hence, skip generation from TIME_START_I (inclusive) to TIME_END_I (exclusive)
TIME_START_I, _ = valh.get_idx_from_field('time_s')
_, TIME_END_I = valh.get_idx_from_field('time_ns')


def df_msgs_to_jnp(m_df: pd.DataFrame) -> jnp.ndarray:
    """"""
    m_df = m_df.copy()
    cols = ['Time', 'Type', 'OrderID', 'Quantity', 'Price', 'Side']
    if m_df.shape[1] == 7:
        cols += ["TradeID"]
    m_df.columns = cols
    m_df['TradeID'] = 0  #  TODO: should be TraderID for multi-agent support
    col_order=['Type','Side','Quantity','Price','TradeID','OrderID','Time']
    m_df = m_df[col_order]
    m_df = m_df[(m_df['Type'] != 6) & (m_df['Type'] != 7) & (m_df['Type'] != 5)]
    time = m_df["Time"].astype('string').str.split('.',expand=True)
    m_df[["TimeWhole","TimeDec"]] = time.astype('int32')
    m_df = m_df.drop("Time", axis=1)
    mJNP = jnp.array(m_df)
    return mJNP

@jax.jit
def msg_to_jnp(
        m_raw: jax.Array,
    ) -> jax.Array:
    """ Select only the relevant columns from the raw messages
        and rearrange for simulator.
    """
    m = m_raw.copy()

    return jnp.array([
        m[EVENT_TYPE_i],
        (m[DIRECTION_i] * 2) - 1,
        m[SIZE_i],
        m[PRICE_ABS_i],
        m[ORDER_ID_i],
        0,  # TraderID
        m[TIMEs_i],
        m[TIMEns_i],
    ])

msgs_to_jnp = jax.jit(jax.vmap(msg_to_jnp))


def copy_orderbook(
        b: OrderBook
    ) -> OrderBook:
    b_copy = OrderBook(cfg=b.cfg)
    b_copy.bids = b.bids.copy()
    b_copy.asks = b.asks.copy()
    b_copy.trades = b.trades.copy()
    return b_copy

def get_sim(
        init_l2_book: jax.Array,
        replay_msgs_raw: jax.Array,
        start_time: jax.Array,
        sim: OrderBook,
    ) -> Tuple[OrderBook, jax.Array]:
    """
    """

    # reset simulator : args are (nOrders, nTrades)
    #Set the ns component of the start time to 0 to ensure that init messages are before first message.
    # Only edge case is if first message and init are 0 ns - unlikely.
    start_time=start_time.at[1].set(0)
    # init simulator at the start of the sequence
    sim_state = sim.reset(init_l2_book,start_time)
    # return sim, sim_state
    # replay sequence in simulator (actual)
    # so that sim is at the same state as the model
    replay = msgs_to_jnp(replay_msgs_raw)
    sim_state = sim.process_orders_array(sim_state, replay)
    return sim_state

get_sims_vmap = jax.jit(
    jax.vmap(
        get_sim,
        in_axes=(0, 0,0,None),
        out_axes=(0),
    ),
    static_argnums=(3,)
)

def get_dataset(
        data_dir: str,
        n_messages: int,
        n_eval_messages: int,
        *,
        n_cache_files: int = 500,
        seed: int = 42,
        book_depth: int = 500,
        day_indeces: Optional[List[int]] = None,
        limit_seq: int = math.inf,
        test_split: float = 0.1,
    ):
    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    if day_indeces is not None:
        #restricts the data to only include certain days.
        msg_files=[msg_files[i] for i in day_indeces]
        book_files=[book_files[i] for i in day_indeces]
    if test_split>0:
        n_test_files = max(1, int(len(msg_files) * test_split))
        msg_files = msg_files[-n_test_files:]
        book_files = book_files[-n_test_files:]

    ds = LOBSTER_Dataset(
        msg_files,
        n_messages=n_messages + n_eval_messages,
        mask_fn=LOBSTER_Dataset.inference_mask,
        seed=seed,
        n_cache_files=n_cache_files,
        randomize_offset=False,
        book_files=book_files,
        use_simple_book=True,
        book_transform=False,
        book_depth=book_depth,
        return_raw_msgs=True,
        inference=True, #this flag shifts the book to exclude the very first state b4 the 1st message.
        limit_seq_per_file=limit_seq,
    )
    return ds

def switch(
        condlist: Sequence[jax.Array],
        funclist: Sequence[Callable],
        operands: Any = None,
        *args, **kw
    ) -> Any:
    """ Convenience function for jax.lax.switch, assuming conditions in condlist are
        mutually exclusive cases. If an extra function is given in funclist,
        this will be applied if no condition is true.
    """
    # unroll the loop over a few args
    switch_i = sum([(i+1) * condlist[i] for i in range(len(condlist))])
    # last funclist element is the default function if no condition is true
    if len(condlist) == len(funclist) - 1:
        return jax.lax.switch(
            switch_i,
            (funclist[-1],
            *funclist[:-1]),
            *operands
        )
    elif len(condlist) == len(funclist):
        return jax.lax.switch(
            switch_i - 1,
            funclist,
            *operands
        )
    else:
        raise ValueError(f'Invalid number of conditions and functions, got {len(condlist)} and {len(funclist)}')


def get_sim_msg(
        pred_msg_enc: jax.Array,
        sim: OrderBook,
        sim_state: LobState,
        mid_price: int,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Dict[str, Any]:
    """"""
    # decoded predicted message
    # pred_msg = tok.decode(pred_msg_enc, v).squeeze()
    msg_decoded = encoding.decode_msg(pred_msg_enc, encoder)
    # jax.debug.print('decoded predicted message: \n {}', msg_decoded)
    new_part = msg_decoded[: Message_Tokenizer.N_NEW_FIELDS]
    # ref part is not needed for the simulator logic
    # ref_part = pred_msg[Message_Tokenizer.N_NEW_FIELDS: ]

    event_type = msg_decoded[EVENT_TYPE_i]
    quantity = msg_decoded[SIZE_i]
    side = msg_decoded[DIRECTION_i]
    rel_price = msg_decoded[PRICE_i]
    delta_t_s = msg_decoded[DTs_i]
    delta_t_ns = msg_decoded[DTns_i]
    time_s = msg_decoded[TIMEs_i]
    time_ns = msg_decoded[TIMEns_i]

    rel_price_ref = msg_decoded[PRICE_REF_i]
    quantity_ref = msg_decoded[SIZE_REF_i]
    time_s_ref = msg_decoded[TIMEs_REF_i]
    time_ns_ref = msg_decoded[TIMEns_REF_i]

    p_abs = mid_price + rel_price * tick_size

    orig_order = sim.get_order_at_time(sim_state, side, time_s_ref, time_ns_ref)

    order_id_ref = orig_order[2]

    order_id = jax.lax.cond(
        (event_type == 2) | (event_type == 3),
        lambda new_id, ref_id: ref_id,
        lambda new_id, ref_id: new_id,
        new_order_id, order_id_ref
    )

    sim_msg = construct_sim_msg(
        event_type,  # type: execution
        side,  # side of execution
        quantity,
        p_abs,
        order_id,
        time_s,
        time_ns,
    )

    msg_decoded = msg_decoded.at[PRICE_ABS_i].set(p_abs) \
                             .at[ORDER_ID_i].set(order_id)

    # return dummy message instead if new_part contains NaNs
    return jax.lax.cond(
        jnp.isnan(new_part).any(),
        lambda sim_msg, msg_decoded: (construct_dummy_sim_msg(), msg_decoded),
        lambda sim_msg, msg_decoded: (sim_msg, msg_decoded),
        sim_msg, msg_decoded
    )

# event_type, side, quantity, price,order_id,trade(r)_id, time_s, time_ns
@jax.jit
def construct_sim_msg(
        event_type: int,
        side: int,
        quantity: int,
        price: int,
        order_id: int,
        time_s: int,
        time_ns: int,
    ):
    """ NOTE: trader ID is set to 0
    """
    return jnp.array([
        event_type,
        (side * 2) - 1,
        quantity,
        price,
        order_id, # order_id
        -88,
        time_s,
        time_ns,
    ], dtype=jnp.int32)

@jax.jit
def construct_dummy_sim_msg(*args) -> jax.Array:
    return jnp.ones((8,), dtype=jnp.int32) * (-1)

@jax.jit
def construct_raw_msg(
        oid: Optional[int] = encoding.NA_VAL,
        event_type: Optional[int] = encoding.NA_VAL,
        direction: Optional[int] = encoding.NA_VAL,
        price_abs: Optional[int] = encoding.NA_VAL,
        price: Optional[int] = encoding.NA_VAL,
        size: Optional[int] = encoding.NA_VAL,
        delta_t_s: Optional[int] = encoding.NA_VAL,
        delta_t_ns: Optional[int] = encoding.NA_VAL,
        time_s: Optional[int] = encoding.NA_VAL,
        time_ns: Optional[int] = encoding.NA_VAL,
        price_ref: Optional[int] = encoding.NA_VAL,
        size_ref: Optional[int] = encoding.NA_VAL,
        time_s_ref: Optional[int] = encoding.NA_VAL,
        time_ns_ref: Optional[int] = encoding.NA_VAL,
    ):
    msg_raw = jnp.array([
        oid,
        event_type,
        direction,
        price_abs,
        price,
        size,
        delta_t_s,
        delta_t_ns,
        time_s,
        time_ns,
        price_ref,
        size_ref,
        time_s_ref,
        time_ns_ref,
    ])
    return msg_raw

@jax.jit
def rel_to_abs_price(
        p_rel: jax.Array,
        best_bid: jax.Array,
        best_ask: jax.Array,
        tick_size: int = 100,
    ) -> jax.Array:

    p_ref = (best_bid + best_ask) / 2
    p_ref = ((p_ref // tick_size) * tick_size).astype(jnp.int32)
    return p_ref + p_rel * tick_size

@jax.jit
def construct_orig_msg_enc(
        pred_msg_enc: jax.Array,
        #v: Vocab,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> jax.Array:
    """ Reconstructs encoded original message WITHOUT Delta t
        from encoded message string --> delta_t field is filled with NA_TOK
    """
    return jnp.concatenate([
        encoding.encode(jnp.array([1]), *encoder['event_type']),
        pred_msg_enc[slice(*valh.get_idx_from_field('direction'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('price_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size_ref'))],
        # NOTE: no delta_t here
        jnp.full(
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_s']] + \
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_ns']],
            Vocab.NA_TOK
        ),
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns_ref'))],
    ])

@jax.jit
def convert_msg_to_ref(
        pred_msg_enc: jax.Array,
    ) -> jax.Array:
    """ Converts encoded message to reference message part,
        i.e. (price, size, time) tokens
    """
    return jnp.concatenate([
        pred_msg_enc[slice(*valh.get_idx_from_field('price'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns'))],
    ])

def search_orig_msg(
        sim, sim_state, side, p_mod_raw, m_seq, pred_msg_enc, encoder, m_seq_raw
    ):
    vol = sim.get_volume_at_price(sim_state, side, p_mod_raw)
    ret_none = (vol==0)

    m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
    # ref part is only needed to match to an order ID
    # find original msg index location in the sequence (if it exists)
    orig_enc = construct_orig_msg_enc(pred_msg_enc, encoder)
    debug('reconstruct. orig_enc \n', orig_enc)

    sim_ids = sim.get_side_ids(sim_state, side)
    debug('sim IDs', sim_ids[sim_ids > 1])
    mask = get_invalid_ref_mask(m_seq_raw, p_mod_raw, sim_ids)
    orig_i, n_fields_removed = valh.try_find_msg(orig_enc, m_seq, mask)

    # didn't find matching original message
    if orig_i is None:
        if sim.get_volume_at_price(sim_state, side, p_mod_raw, True) == 0:
            debug('No init volume found', side, p_mod_raw)
            return None, None, None
        order_id = job.INITID
        # keep generated ref part, which we cannot validate
        orig_msg_found = orig_enc[-REF_LEN: ]

    # found matching original message
    else:
        # get order ID from raw data for simulator
        ORDER_ID_i = 0
        order_id = m_seq_raw[orig_i, ORDER_ID_i]
        # found original message: convert to ref part
        EVENT_TYPE_i = 1
        if m_seq_raw[orig_i, EVENT_TYPE_i] == 1:
            orig_msg_found = convert_msg_to_ref(m_seq[orig_i])
        # found reference to original message
        else:
            # take ref fields from matching message
            orig_msg_found = jnp.array(m_seq[orig_i, -REF_LEN: ])

@jax.jit
def get_invalid_ref_mask(
        m_seq_raw: jax.Array,
        p_mod_raw: int,
        sim_ids: jax.Array
    ):
    """
    """
    PRICE_ABS_i = 3
    # filter sequence to prices matching the correct price level
    wrong_price_mask = (m_seq_raw[:, PRICE_ABS_i] != p_mod_raw)
    # filter to orders still in the book: order IDs from sim
    ORDER_ID_i = 0
    not_in_book_mask = jnp.isin(m_seq_raw[:, ORDER_ID_i], sim_ids, invert=True)
    mask = not_in_book_mask | wrong_price_mask
    return mask

@jax.jit
def add_times(
        a_s: jax.Array,
        a_ns: jax.Array,
        b_s: jax.Array,
        b_ns: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
    """ Adds two timestamps given as seconds and nanoseconds each (both fit in int32)
        and returns new timestamp, split into time_s and time_ns
    """
    a_ns = b_ns + a_ns
    extra_s = a_ns // 1000000000
    a_ns = a_ns % 1000000000
    a_s = a_s + b_s + extra_s
    return a_s, a_ns

def _get_safe_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        tick_size: int,
    ) -> int:
    """
    """
    # get current mid price from simulator
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)

    # both negative: 0 ~> (ask + bid) / 2
    # ask negative:  1 ~> bid + tick_size
    # bid negative:  2 ~> ask - tick_size
    # both negative: 3 ~> 0
    case_i = (ask <= 0) * 1 + (bid <= 0) * 2

    p_mid = jax.lax.switch(
        case_i,
        (
            lambda ask, bid: (ask + bid) // 2,
            lambda ask, bid: bid + tick_size,
            lambda ask, bid: ask - tick_size,
            lambda ask, bid: 0,
        ),
        ask, bid
    )
    # round down to next valid tick
    p_mid = (p_mid // tick_size) * tick_size
    return p_mid

@partial(jax.jit, static_argnums=(0,))
def _get_new_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        p_mid_old: jax.Array,
        tick_size: int,
    ) -> jax.Array:
    """
    """
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)
    mid = ((((ask + bid) // 2) // tick_size) * tick_size)
    return jax.lax.cond(
        (ask <= 0) | (bid <= 0),
        lambda new, old: old,
        lambda new, old: new,
        mid, p_mid_old
    )

def _add_time_tokens(
        tok_seq_A: jax.Array,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        time_init_s: int,
        time_init_ns: int,
        delta_t_s_start_i: int,
        delta_t_s_end_i: int,
        delta_t_ns_start_i: int,
        delta_t_ns_end_i: int,
    ):
    """
    """
    # TODO: simplify --> separate function
    delta_t_s_toks = tok_seq_A[delta_t_s_start_i: delta_t_s_end_i]
    delta_t_ns_toks = tok_seq_A[delta_t_ns_start_i: delta_t_ns_end_i]
    # debug('delta_t_toks', delta_t_s_toks, delta_t_ns_toks)
    delta_t_s = encoding.decode(delta_t_s_toks, *encoder['time'])
    delta_t_s = encoding.combine_field(delta_t_s, 3)
    delta_t_ns = encoding.decode(delta_t_ns_toks, *encoder['time'])
    delta_t_ns = encoding.combine_field(delta_t_ns, 3)

    # debug('delta_t', delta_t_s, delta_t_ns)
    time_s_ret, time_ns_ret = add_times(time_init_s, time_init_ns, delta_t_s, delta_t_ns)
    # debug('time', time_s, time_ns)

    # encode time and add to sequence
    time_s = encoding.split_field(time_s_ret, 2, 3)
    time_s_toks = encoding.encode(time_s, *encoder['time'])
    time_ns = encoding.split_field(time_ns_ret, 3, 3)
    time_ns_toks = encoding.encode(time_ns, *encoder['time'])

    # debug('time_toks', time_s_toks, time_ns_toks)
    time_tokens=jnp.hstack([time_s_toks, time_ns_toks])
    return time_tokens, time_s_ret, time_ns_ret


def _generate_token(
        train_state : TrainState,
        model : nn.module,
        batchnorm : bool,
        valid_mask_array : jax.Array ,
        sample_top_n : int,

        m_tok: jax.Array ,
        b_tok: jax.Array ,
        hidden: Tuple,
        token_index : int,
        rng,
    ):
    # syntactically valid tokens for current message position
    valid_mask = valh.get_valid_mask(valid_mask_array, token_index)

    hidden, logits = valh.apply_model(hidden,
                              m_tok,
                              b_tok,
                              train_state,
                              model,
                              batchnorm,
                              False)
    logits=logits[0]
    argsortedlogits=jnp.argsort(logits,descending=True)

    # filter out (syntactically) invalid tokens for current position
    if valid_mask is not None:
        logits = valh.filter_valid_pred(logits, valid_mask)

    # update sequence
    # NOTE: rng arg expects one element per batch element
    rng, rng_ = jax.random.split(rng)
    m_tok = valh.fill_predicted_tok( logits, sample_top_n, jnp.array([rng_]))
    return m_tok, hidden, token_index + 1, rng

def _make_generate_token_scannable(
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        valid_mask_array: jax.Array,
        sample_top_n: int,
    ):
    """
    """
    __generate_token = jax.jit(functools.partial(
        _generate_token, train_state, model, batchnorm, valid_mask_array, sample_top_n
    ))

    def _generate_token_scannable(carry, xs):
        # m_seq, b_tok, mask_i, rng = carry
        m_tok, hidden, mask_i, rng = __generate_token(*carry)
        return (m_tok, carry[1], hidden, mask_i, rng), m_tok

    return _generate_token_scannable


def _generate_msg(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
        # NEW: insertion_schedule parameters
        insertion_schedule: Optional[jax.Array],  # shape (total_msgs, 9) or None
        total_n_msg_todo: int,  # original n_msg_todo for calculating current_step
        metaorder_visible: bool,  # static: roll agg msg through hidden + feed post-agg book/p_change

        m_init: jax.Array, #last token from prev message, or start tok.
        b_init: jax.Array, #last book state after prev message, or start book.
        n_msg_todo: int,
        p_mid: jax.Array,
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        hidden: Tuple,
        time_i: jax.Array,
        # ema_start:bool,
        b_seq_real:Optional[jax.Array]=None,
    ) -> Tuple[jax.Array, LobState, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """
    """
    rng, rng_ = jax.random.split(rng)
    # treat as compile time constants
    with jax.ensure_compile_time_eval():
        l = Message_Tokenizer.MSG_LEN
        time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
        delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
        delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    #
    time_init_s = time_i[0]
    time_init_ns = time_i[1]

    generate_token_scannable = _make_generate_token_scannable(
        train_state, model, batchnorm, valid_mask_array, sample_top_n
    )

    if debug_book:
        b_init=jnp.expand_dims(b_seq_real,0)

    # get next message: generate l tokens:
    # generate tokens until time is reached
    token_idx = 0
    # Pass the first token & book (last from prev msg or START)
    #Generate tokens up to the last delta t (before first abs time)
    gen_token_carry = (m_init, b_init,hidden, token_idx, rng_)
    (m_inter, b_inter, hidden, token_idx, rng_), tok_seq_A = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=time_s_start_i
    )
    tok_seq_A=jnp.squeeze(tok_seq_A)
    # fill the time tokens, retain the actual times, to generate the next message.
    tok_seq_T, time_s, time_ns= _add_time_tokens(
        tok_seq_A,
        encoder,
        time_init_s,
        time_init_ns,
        delta_t_s_start_i,
        delta_t_s_end_i,
        delta_t_ns_start_i,
        delta_t_ns_end_i,
    )
    time_f=jnp.array([time_s, time_ns])

    tok_seq_roll_thru_hidden=jnp.concatenate([tok_seq_A[-1:],tok_seq_T[:-1]])

    hidden,_=valh.apply_model(hidden,
                            tok_seq_roll_thru_hidden,
                            b_init,
                            train_state,
                            model,
                            batchnorm,
                            False)

    # update mask index to skip time token positions
    token_idx = time_ns_end_i
    gen_token_carry = (tok_seq_T[-1:], b_init, hidden, token_idx, rng_)

    # finish message generation
    (m_final, b_final, hidden, token_idx, rng_), tok_seq_B = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=l-time_ns_end_i
    )

    tok_seq_B=jnp.squeeze(tok_seq_B)


    # Fully generated message.
    tok_seq_gen=jnp.concatenate([tok_seq_A,tok_seq_T,tok_seq_B])
    # order_id = id_gen.step()  # no order ID generator any more in v3 sim?
    order_id = n_msg_todo + GEN_ORDER_ID_OFFSET

    sim_msg, msg_decoded = get_sim_msg(
        tok_seq_gen,  # the generated message
        sim,
        sim_state,
        mid_price = p_mid,
        new_order_id = order_id,
        tick_size = tick_size,
        encoder = encoder,
    )

    # feed message to simulator, updating book state
    sim_state = sim.process_order_array(sim_state, sim_msg)

    # === CAPTURE BOOK STATE AFTER REGULAR MESSAGE ===
    p_mid_after_regular = _get_new_mid_price(sim, sim_state, p_mid, tick_size)
    p_change_regular = ((p_mid_after_regular - p_mid) // tick_size)
    book_l2_regular = sim.get_L2_state(sim_state, l2_state_n)

    # === INSERTION SCHEDULE LOGIC ===
    aggressive_msg_decoded = jnp.zeros(14, dtype=jnp.int32)
    # Initialize book_l2_after_aggressive as copy of regular (will be overwritten if insertion)
    book_l2_after_aggressive = book_l2_regular
    p_mid_after_aggressive = p_mid_after_regular

    # model book input built from the post-regular state; also the conditioning book when the
    # aggressive message is rolled through the hidden state (metaorder_visible)
    new_book_raw_regular = jnp.concatenate([jnp.array([p_change_regular]),time_f, book_l2_regular[0:40]]).reshape(1,-1)
    b_regular_transformed = preproc.transform_L2_state_gpu(new_book_raw_regular, 500, 100)

    if insertion_schedule is not None:
        current_step = total_n_msg_todo - n_msg_todo
        should_insert = insertion_schedule[current_step, 0] == 1

        def do_insert(operands):
            sim_state_in, n_msg_todo_in, time_f_in, p_mid_in, hidden_in, m_carry_in = operands

            # Get dynamic price and available volume from book
            best_bid_ask = sim.get_best_bid_and_ask_inclQuants(sim_state_in)
            # best_bid_ask[0] = (ask_price, ask_qty), best_bid_ask[1] = (bid_price, bid_qty)
            direction = insertion_schedule[current_step, 2]
            price = jax.lax.cond(
                direction == 0,
                lambda: best_bid_ask[0][0],  # BUY hits ask price
                lambda: best_bid_ask[1][0]   # SELL hits bid price
            )

            # Get available volume at best level and cap order size
            avail = jax.lax.cond(
                direction == 0,
                lambda: best_bid_ask[0][1],  # ask volume for buy
                lambda: best_bid_ask[1][1]   # bid volume for sell
            ).astype(jnp.int32)
            order_volume = insertion_schedule[current_step, 3]
            quantity = jnp.minimum(order_volume, avail)

            # Aggressive order gets NEXT order_id (same decreasing sequence)
            aggressive_order_id = n_msg_todo_in - 1 + GEN_ORDER_ID_OFFSET

            aggressive_msg = construct_sim_msg(
                event_type=insertion_schedule[current_step, 1],
                side=direction,
                quantity=quantity,
                price=price,
                order_id=aggressive_order_id,
                time_s=time_f_in[0],
                time_ns=time_f_in[1] + 1,
            )
            new_sim_state = sim.process_order_array(sim_state_in, aggressive_msg)
            new_n_msg_todo = n_msg_todo_in - 1  # Decrement for aggressive order

            # Create decoded representation of aggressive order.
            # Fields 4 (rel price), 6/7 (delta_t) and 10-13 (refs) are needed by
            # encoding.encode_msg when the message is made visible to the model; ref fields
            # use the touch price / executed qty / current time as a proxy for the resting
            # order (exact resting-order lookup would need a book scan).
            mid_r = (p_mid_in // tick_size) * tick_size
            rel_price = ((price - mid_r) // tick_size).astype(jnp.int32)
            agg_msg_decoded = jnp.zeros(14, dtype=jnp.int32)
            agg_msg_decoded = agg_msg_decoded.at[ORDER_ID_i].set(aggressive_order_id)
            agg_msg_decoded = agg_msg_decoded.at[EVENT_TYPE_i].set(insertion_schedule[current_step, 1])
            agg_msg_decoded = agg_msg_decoded.at[DIRECTION_i].set(direction)
            agg_msg_decoded = agg_msg_decoded.at[PRICE_ABS_i].set(price)
            agg_msg_decoded = agg_msg_decoded.at[PRICE_i].set(rel_price)
            agg_msg_decoded = agg_msg_decoded.at[SIZE_i].set(quantity)
            agg_msg_decoded = agg_msg_decoded.at[DTs_i].set(0)
            agg_msg_decoded = agg_msg_decoded.at[DTns_i].set(1)
            agg_msg_decoded = agg_msg_decoded.at[TIMEs_i].set(time_f_in[0])
            agg_msg_decoded = agg_msg_decoded.at[TIMEns_i].set(time_f_in[1] + 1)
            agg_msg_decoded = agg_msg_decoded.at[PRICE_REF_i].set(rel_price)
            agg_msg_decoded = agg_msg_decoded.at[SIZE_REF_i].set(quantity)
            agg_msg_decoded = agg_msg_decoded.at[TIMEs_REF_i].set(time_f_in[0])
            agg_msg_decoded = agg_msg_decoded.at[TIMEns_REF_i].set(time_f_in[1] + 1)

            # Capture book state AFTER aggressive order
            book_l2_after = sim.get_L2_state(new_sim_state, l2_state_n)
            p_mid_after = _get_new_mid_price(sim, new_sim_state, p_mid_in, tick_size)

            if metaorder_visible:
                # Roll the aggressive message through the hidden state, mirroring the
                # regular-message convention: the not-yet-consumed carry token goes first,
                # the message's last token becomes the new carry.
                agg_tok = encoding.encode_msg(agg_msg_decoded, encoder).astype(m_carry_in.dtype)
                roll_seq = jnp.concatenate([m_carry_in.reshape(-1), agg_tok[:-1]])
                hidden_out, _ = valh.apply_model(
                    hidden_in,
                    roll_seq,
                    b_regular_transformed,
                    train_state,
                    model,
                    batchnorm,
                    False,
                )
                m_carry_out = agg_tok[-1:].reshape(m_carry_in.shape)
            else:
                hidden_out = hidden_in
                m_carry_out = m_carry_in

            return new_sim_state, new_n_msg_todo, agg_msg_decoded, book_l2_after, p_mid_after, hidden_out, m_carry_out

        def no_insert(operands):
            sim_state_in, n_msg_todo_in, time_f_in, p_mid_in, hidden_in, m_carry_in = operands
            # Return same book state (no change from aggressive)
            book_l2_same = sim.get_L2_state(sim_state_in, l2_state_n)
            return sim_state_in, n_msg_todo_in, jnp.zeros(14, dtype=jnp.int32), book_l2_same, p_mid_in, hidden_in, m_carry_in

        sim_state, n_msg_todo, aggressive_msg_decoded, book_l2_after_aggressive, p_mid_after_aggressive, hidden, m_final = jax.lax.cond(
            should_insert,
            do_insert,
            no_insert,
            (sim_state, n_msg_todo, time_f, p_mid_after_regular, hidden, m_final)
        )

    if (insertion_schedule is not None) and metaorder_visible:
        # Post-aggressive book and price change INCLUDING the insertion jump; on steps
        # without an insertion these equal the regular values by construction.
        p_change_final = ((p_mid_after_aggressive - p_mid) // tick_size)
        new_book_raw = jnp.concatenate([jnp.array([p_change_final]), time_f, book_l2_after_aggressive[0:40]]).reshape(1,-1)
        b_final = preproc.transform_L2_state_gpu(new_book_raw, 500, 100)
    else:
        # legacy: model conditions on the pre-insertion book, jump netted out of p_change
        b_final = b_regular_transformed

    n_msg_todo -= 1

    # Return both book states: regular (for regular msg) and after_aggressive (for aggressive msg)
    return msg_decoded, aggressive_msg_decoded, sim_state, m_final, tok_seq_gen, b_final, book_l2_regular, book_l2_after_aggressive, p_mid_after_aggressive, n_msg_todo, hidden, time_f


def _make_generate_msg_scannable(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
        # NEW: insertion_schedule parameters
        insertion_schedule: Optional[jax.Array],
        total_n_msg_todo: int,
        metaorder_visible: bool,
    ):
    """
    """
    __generate_msg = jax.jit(functools.partial(
        _generate_msg, sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size, debug_book,
        insertion_schedule, total_n_msg_todo, metaorder_visible,  # NEW
    ),device=jax.devices()[0])

    def _generate_msg_scannable(gen_state, input):
        """ Wrapper for _generate_msg to be used with jax.lax.scan
        """
        b_seq_real=input
        m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time= gen_state
        rng, rng_ = jax.random.split(rng)

        msg_decoded, aggressive_msg_decoded, sim_state, m_seq, msg_token, b_seq, book_l2_regular, book_l2_after_agg, p_mid, n_msg_todo, hidden, time = __generate_msg(
            m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng_, hidden, time, b_seq_real
        )
        # Return both book states: regular and after_aggressive
        return (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng,hidden, time), (msg_decoded, aggressive_msg_decoded, book_l2_regular, book_l2_after_agg, msg_token)
    return _generate_msg_scannable

@partial(jax.jit, static_argnums=(0, 2, 3, 5, 6, 9,13,15,18,19),backend='gpu')
def generate(
        sim: OrderBook,  # static
        train_state: TrainState,
        model: nn.Module,  # static
        batchnorm: bool,  # static
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,  # static
        tick_size: int,  # static
        m_seq_cond: jax.Array,
        b_seq_cond: jax.Array,
        n_msg_todo: int,  # static
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        init_hidden : Tuple,
        conditional : bool, # static
        init_time : jax.Array,
        debug_book: bool=False,
        b_seq_real: Optional[jax.Array]=None,
        # NEW: insertion_schedule parameter
        insertion_schedule: Optional[jax.Array] = None,
        chunk_size: int = 1,  # NEW: N for chunking conditional sequence
        metaorder_visible: bool = True,  # static: model sees the inserted metaorder
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:

    print("WARNING: Compiling the generate function, you should only see this once.")

    with jax.ensure_compile_time_eval():
        valid_mask_array = valh.syntax_validation_matrix()

    if not debug_book:
        b_seq_real=None


    if conditional:
        def roll_hidden_scan(carry,xs):
            m_seq,b_seq=xs
            h=carry
            h,log=valh.apply_model(h,
                                m_seq,
                                b_seq,
                                train_state,
                                model,
                                batchnorm,
                                True)
            carry=h
            return carry, None


        print(m_seq_cond[:-1],b_seq_cond[:-1])
        # Split arrays into N chunks along the leading axis
        N = chunk_size
        chex.assert_is_divisible(m_seq_cond[:-1].shape[0], N)
        chex.assert_is_divisible(b_seq_cond[:-1].shape[0], N)
        m_seq_cond_split = m_seq_cond[:-1].reshape((N, -1))
        b_seq_cond_split = b_seq_cond[:-1].reshape((N, -1) + b_seq_cond[:-1].shape[1:])

        hidden_state,_ = jax.lax.scan(roll_hidden_scan, init_hidden, (m_seq_cond_split, b_seq_cond_split))
        init_token=m_seq_cond[-1:]
        init_book=b_seq_cond[-1:]
        init_time=jnp.asarray(valh.get_first_time(m_seq_cond,encoder))
        init_ema=False
    else:
        #If unconditional generation, then the initial token
        #  and book have one less dimension.
        hidden_state=init_hidden
        init_ema=True
        # FIXME: Currently wrong and incomplete.
        # Needs to just be START token and init book state.
        assert (m_seq_cond.ndim==1) & (m_seq_cond.shape[0]==1), "m_seq_cond needs to be a scalar (start tok?)"
        init_token=m_seq_cond
        init_book=b_seq_cond

    # get current mid price from simulator
    p_mid = _get_safe_mid_price(sim, sim_state, tick_size)

    # Store original n_msg_todo for later use (will be decremented during scan)
    original_n_msg_todo = n_msg_todo

    generate_msg_scannable = _make_generate_msg_scannable(
        sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size, debug_book,
        insertion_schedule, n_msg_todo, metaorder_visible,  # NEW: pass schedule and total count
    )
    gen_state, (msgs_decoded, aggressive_msgs_decoded, l2_book_states_regular, l2_book_states_after_agg, msgs_tokens) = jax.lax.scan(
        generate_msg_scannable,
        (init_token, init_book, n_msg_todo, p_mid, sim_state,rng, hidden_state,init_time),
        length=n_msg_todo,
        xs=b_seq_real,
    )
    (final_token, final_book, final_n_msg_todo, p_mid, sim_state, rng, hidden_state, final_time) = gen_state

    # Merge regular and aggressive messages: [reg0, agg0, reg1, agg1, ...]
    merged_msgs = jnp.zeros((original_n_msg_todo * 2, msgs_decoded.shape[1]), dtype=msgs_decoded.dtype)
    merged_msgs = merged_msgs.at[::2].set(msgs_decoded)
    merged_msgs = merged_msgs.at[1::2].set(aggressive_msgs_decoded)

    # Merge book states: regular (for regular msg) and after_agg (for aggressive msg)
    # l2_book_states_regular[i] = book state AFTER regular message i (BEFORE aggressive)
    # l2_book_states_after_agg[i] = book state AFTER aggressive order i (or same as regular if no insertion)
    merged_books = jnp.zeros((original_n_msg_todo * 2, l2_book_states_regular.shape[1]), dtype=l2_book_states_regular.dtype)
    merged_books = merged_books.at[::2].set(l2_book_states_regular)
    merged_books = merged_books.at[1::2].set(l2_book_states_after_agg)

    # count errors when the message does not change the (visible) book state
    num_errors = (merged_books[1:] == merged_books[:-1]).all(axis=1).sum()

    # Return merged messages and book states (caller filters zeros)
    # Shape: (n_msg_todo * 2, ...) - includes placeholder zeros for positions without aggressive orders
    return merged_msgs, merged_books, num_errors, msgs_tokens

generate_batched = jax.jit(
    jax.vmap(
        generate,
        in_axes=(
            None, None, None, None, None,  # sim, train_state, model, batchnorm, encoder
            None, None,    0,    0, None,  # sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo
               0,    0,    0, None,    0,  # sim_state, rng, init_hidden, conditional, init_time
            None,    0, None, None, None,  # debug_book, b_seq_real, insertion_schedule (SHARED, unbatched), chunk_size, metaorder_visible
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9,13,15,18,19),backend='gpu'
)


def msg_to_lobster_format(
        m_seq: jax.Array,
) -> pd.DataFrame:
    """
    message format: [time, event_type, order_id, size, price, direction]
    """
    m_seq_ = onp.array(m_seq)[:, [TIMEs_i, TIMEns_i, EVENT_TYPE_i, ORDER_ID_i, SIZE_i, PRICE_ABS_i, DIRECTION_i]]
    m_seq_ = pd.DataFrame(m_seq_, columns=['time_s', 'time_ns', 'event_type', 'order_id', 'size', 'price', 'direction'])

    # combine time field to single field
    m_seq_.insert(
        column = 'time',
        loc = 0,
        value = m_seq_['time_s'].astype(str) \
              + '.' \
              + m_seq_['time_ns'].astype(str).str.pad(width=9, side='left', fillchar='0')
    )
    m_seq_.drop(columns=['time_s', 'time_ns'], inplace=True)

    # convert direction {0,1} to {-1,1}
    m_seq_['direction'] = m_seq_['direction'].replace({0: -1})
    return m_seq_

def book_to_lobster_format(
        b_seq: jax.Array,
    ) -> pd.DataFrame:
    """
    """
    b_seq_ = pd.DataFrame(b_seq)

    return b_seq_


transform_L2_state_batch = jax.jit(
    jax.vmap(
        preproc.transform_L2_state_gpu,
        in_axes=(0, None, None)
    ),
    static_argnums=(1, 2)
)
