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

import lob.validation_helpers as valh
import lob.evaluation as eval
import preproc as preproc
from preproc import transform_L2_state
import lob.encoding as encoding
from lob.encoding import Message_Tokenizer, Vocab
from lob.lobster_dataloader import LOBSTER_Dataset


# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
# from gym_exchange.environment.base_env.assets.action import OrderIdGenerator

REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

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

l2_state_n = 20

# time tokens aren't generated but calculated using delta_t
# hence, skip generation from TIME_START_I (inclusive) to TIME_END_I (exclusive)
TIME_START_I, _ = valh.get_idx_from_field('time_s')
_, TIME_END_I = valh.get_idx_from_field('time_ns')

# @jax.jit
# def init_msgs_from_l2(book: Union[pd.Series, onp.ndarray]) -> jnp.ndarray:
#     """"""
#     orderbookLevels = len(book) // 4  # price/quantity for bid/ask
#     data = jnp.array(book).reshape(int(orderbookLevels*2),2)
#     newarr = jnp.zeros((int(orderbookLevels*2),8))
#     initOB = newarr \
#         .at[:,3].set(data[:,0]) \
#         .at[:,2].set(data[:,1]) \
#         .at[:,0].set(1) \
#         .at[0:orderbookLevels*4:2,1].set(-1) \
#         .at[1:orderbookLevels*4:2,1].set(1) \
#         .at[:,4].set(0) \
#         .at[:,5].set(job.INITID) \
#         .at[:,6].set(34200) \
#         .at[:,7].set(0).astype('int32')
#     return initOB


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
        0, # TradeID
        m[ORDER_ID_i],
        m[TIMEs_i],
        m[TIMEns_i],
    ])

msgs_to_jnp = jax.jit(jax.vmap(msg_to_jnp))

# # NOTE: cannot jit due to side effects --> resolve later
# @jax.jit
# def reset_orderbook(
#         b: OrderBook,
#         l2_book: Optional[Union[pd.Series, onp.ndarray]] = None,
#     ) -> OrderBook:
#     """"""
#     b.bids = b.bids.at[:].set(-1)
#     b.asks = b.asks.at[:].set(-1)
#     b.trades = b.trades.at[:].set(-1)
#     if l2_book is not None:
#         msgs = init_msgs_from_l2(l2_book)
#         # NOTE: cannot jit due to side effects --> resolve later
#         # CONTINUE HERE....
#         b.process_orders_array(msgs)
#     return b

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
        # nOrders: int = 100,
        # nTrades: int = 100
        # sim_book_levels: int,
        # sim_queue_len: int,
    ) -> Tuple[OrderBook, jax.Array]:
    """
    """
    # reset simulator : args are (nOrders, nTrades)
    sim = OrderBook()
    # init simulator at the start of the sequence
    sim_state = sim.reset(init_l2_book)
    # return sim, sim_state
    # replay sequence in simulator (actual)
    # so that sim is at the same state as the model
    replay = msgs_to_jnp(replay_msgs_raw)
    sim_state = sim.process_orders_array(sim_state, replay)
    return sim, sim_state

get_sims_vmap = jax.jit(
    jax.vmap(
        get_sim,
        in_axes=(0, 0),
        out_axes=(None, 0)
    )
)

def get_dataset(
        data_dir: str,
        n_messages: int,
        n_eval_messages: int,
        *,
        n_cache_files: int = 500,
        seed: int = 42,
        book_depth: int = 500,
    ):
    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    ds = LOBSTER_Dataset(
        msg_files,
        n_messages=n_messages + n_eval_messages,
        mask_fn=lambda X, rng: (X, jnp.array(0)),
        seed=seed,
        n_cache_files=n_cache_files,
        randomize_offset=False,
        book_files=book_files,
        use_simple_book=True,
        book_transform=False,
        book_depth=book_depth,
        return_raw_msgs=True,
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
    debug('decoded predicted message:', msg_decoded)

    new_part = msg_decoded[: Message_Tokenizer.N_NEW_FIELDS]
    # ref part is not needed for the simulator logic
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

    # get message for jax lob simulator
    sim_msg = switch(
        (event_type == 1, (event_type == 2) | (event_type == 3), event_type == 4),
        (get_sim_msg_new, get_sim_msg_mod, get_sim_msg_exec, construct_dummy_sim_msg),
        (event_type, quantity, side, rel_price, time_s, time_ns, 
                rel_price_ref, quantity_ref, time_s_ref, time_ns_ref,
                mid_price, new_order_id,
                sim, sim_state, tick_size
        )
    )

    # return dummy message instead if new_part contains NaNs
    return jax.lax.cond(
        jnp.isnan(new_part).any(),
        lambda: (construct_dummy_sim_msg(), msg_decoded),
        lambda: (sim_msg, msg_decoded)
    )

# event_type, side, quantity, price, trade(r)_id, order_id, time_s, time_ns
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
        0, # trader_id
        order_id,
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
def get_sim_msg_new(
        event_type: int,
        quantity: int,
        side: int,
        rel_price: int,
        time_s: int,
        time_ns: int, 
        rel_price_ref: int,
        quantity_ref: int,
        time_s_ref: int,
        time_ns_ref: int,
        mid_price: int,
        new_order_id: int,
        sim: OrderBook,
        sim_state: LobState,
        tick_size: int
    ) -> jax.Array:
 
        # new limit order
        # debug('NEW LIMIT ORDER')
        # convert relative to absolute price
        price = mid_price + rel_price * tick_size

        sim_msg = construct_sim_msg(
            1,
            side,
            quantity,
            price,
            new_order_id,
            time_s,
            time_ns,
        )

        return sim_msg#, msg_corr, msg_raw

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
    
    # if sim.get_volume_at_price(sim_state, side, p_mod_raw) == 0:
    #     debug('No volume at given price, discarding...')
    #     return None, None, None

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

# TODO: resolve control flow to be able to jit function
#@jax.jit
def get_sim_msg_mod(
        event_type: int,
        removed_quantity: int,
        side: int,
        rel_price: int,
        time_s: int,
        time_ns: int, 
        rel_price_ref: int,
        quantity_ref: int,
        time_s_ref: int,
        time_ns_ref: int,
        mid_price: int,
        new_order_id: int,
        sim: OrderBook,
        sim_state: LobState,
        tick_size: int
    ) -> jax.Array:

    # debug('ORDER CANCEL / DELETE')

    # the actual price of the order to be modified
    p_mod_raw = mid_price + rel_price * tick_size

    # TODO: get order ID from simulator if time matches exactly,
    #       otherwise modify random order at the given price
    # TODO: get message with exact timestamp from simulator instead of relying on raw sequence
    # orig_enc = construct_orig_msg_enc(pred_msg_enc, encoder)

    # get limit order by timestamp
    orig_order = sim.get_order_at_time(sim_state, side, time_s_ref, time_ns_ref)
    order_id = orig_order[ORDER_ID_i]

    # debug(f'(event_type={event_type}) -{removed_quantity} from {remaining_quantity} '
    #       + f'@{p_mod_raw} --> {remaining_quantity-removed_quantity}')

    # if order is not found (-1) cancel random order at the given price
    sim_msg = construct_sim_msg(
        event_type,
        side,
        removed_quantity,
        p_mod_raw,
        order_id, # exact match or -1
        time_s,
        time_ns,
    )

    return sim_msg


def get_sim_msg_exec(
        event_type: int,
        removed_quantity: int,
        side: int,
        rel_price: int,
        time_s: int,
        time_ns: int, 
        rel_price_ref: int,
        quantity_ref: int,
        time_s_ref: int,
        time_ns_ref: int,
        mid_price: int,
        new_order_id: int,
        sim: OrderBook,
        sim_state: LobState,
        tick_size: int
    ) -> jax.Array:

    # debug('ORDER EXECUTION')
    REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

    # the actual price of the order to be modified
    p_mod_raw = mid_price + rel_price * tick_size

    sim_msg = construct_sim_msg(
        4,  # type: execution
        side,  # side of execution
        removed_quantity,
        p_mod_raw,
        new_order_id,
        time_s,
        time_ns,
    )

    return sim_msg

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

def _get_new_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        p_mid_old: int,
        tick_size: int,
    ) -> jax.Array:
    """
    """
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)
    return jax.lax.cond(
        ((ask <= 0) | (bid <= 0)),
        lambda: p_mid_old,
        lambda: (((ask + bid) // 2) // tick_size) * tick_size
    )

def _add_time_tokens(
        m_seq: jax.Array,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        time_init_s: int,
        time_init_ns: int,
        last_start_i: int,
        time_s_start_i: int,
        time_ns_end_i: int,
        delta_t_s_start_i: int,
        delta_t_s_end_i: int,
        delta_t_ns_start_i: int,
        delta_t_ns_end_i: int,
    ):
    """
    """
    # TODO: simplify --> separate function
    delta_t_s_toks = m_seq[last_start_i + delta_t_s_start_i: last_start_i + delta_t_s_end_i]
    delta_t_ns_toks = m_seq[last_start_i + delta_t_ns_start_i: last_start_i + delta_t_ns_end_i]
    # debug('delta_t_toks', delta_t_s_toks, delta_t_ns_toks)
    delta_t_s = encoding.decode(delta_t_s_toks, *encoder['time'])
    delta_t_s = encoding.combine_field(delta_t_s, 3)
    delta_t_ns = encoding.decode(delta_t_ns_toks, *encoder['time'])
    delta_t_ns = encoding.combine_field(delta_t_ns, 3)

    # debug('delta_t', delta_t_s, delta_t_ns)
    time_s, time_ns = add_times(time_init_s, time_init_ns, delta_t_s, delta_t_ns)
    # debug('time', time_s, time_ns)
    
    # encode time and add to sequence
    time_s = encoding.split_field(time_s, 2, 3)
    time_s_toks = encoding.encode(time_s, *encoder['time'])
    time_ns = encoding.split_field(time_ns, 3, 3)
    time_ns_toks = encoding.encode(time_ns, *encoder['time'])

    # debug('time_toks', time_s_toks, time_ns_toks)
    m_seq = m_seq.at[last_start_i + time_s_start_i: last_start_i + time_ns_end_i].set(
        jnp.hstack([time_s_toks, time_ns_toks]))
    return m_seq

def _generate_token(
        train_state,
        model,
        batchnorm,
        valid_mask_array,
        sample_top_n,
        m_seq,
        b_seq,
        mask_i,
        rng 
    ):

    # syntactically valid tokens for current message position
    valid_mask = valh.get_valid_mask(valid_mask_array, mask_i)
    m_seq, _ = valh.mask_last_msg_in_seq(m_seq, mask_i)
    input = (
        jnp.expand_dims(m_seq, axis=0),
        jnp.expand_dims(b_seq, axis=0)
    )
    integration_timesteps = (
        jnp.ones((1, len(m_seq))), 
        jnp.ones((1, len(b_seq)))
    )
    logits = valh.predict(
        input,
        integration_timesteps, train_state, model, batchnorm)
    
    # filter out (syntactically) invalid tokens for current position
    if valid_mask is not None:
        logits = valh.filter_valid_pred(logits, valid_mask)

    # update sequence
    # NOTE: rng arg expects one element per batch element
    rng, rng_ = jax.random.split(rng)
    m_seq = valh.fill_predicted_toks(m_seq, logits, sample_top_n, jnp.array([rng_]))

    return m_seq, mask_i + 1, rng

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
        # m_seq, b_seq, mask_i, rng = carry
        m_seq, mask_i, rng = __generate_token(*carry)
        return (m_seq, carry[1], mask_i, rng), None

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
        
        m_seq: jax.Array,
        b_seq: jax.Array,
        n_msg_todo: int,
        p_mid: int,
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
    ) -> Tuple[jax.Array, LobState, jax.Array, jax.Array, jax.Array, int, int]:
    """
    """
    rng, rng_ = jax.random.split(rng)
    # treat as compile time constants
    with jax.ensure_compile_time_eval():
        l = Message_Tokenizer.MSG_LEN
        last_start_i = m_seq.shape[0] - l

        time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
        delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
        delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    # 
    time_init_s, time_init_ns = encoding.decode_time(
        m_seq[last_start_i + time_s_start_i: last_start_i + time_ns_end_i],
        encoder
    )

    # roll sequence one step forward
    m_seq = valh.append_hid_msg(m_seq)

    # TODO: calculating time in case where generation is not sequentially left to right
    #       --> check if delta_t complete --> calc time once

    generate_token_scannable = _make_generate_token_scannable(
        train_state, model, batchnorm, valid_mask_array, sample_top_n
    )

    # get next message: generate l tokens:
    # generate tokens until time is reached
    mask_i = 0
    gen_token_carry = (m_seq, b_seq, mask_i, rng_)
    (m_seq, b_seq, mask_i, rng_), _ = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=TIME_START_I
    )
    # fill the time tokens
    m_seq = _add_time_tokens(
        m_seq,
        encoder,
        time_init_s,
        time_init_ns,
        last_start_i,
        time_s_start_i,
        time_ns_end_i,
        delta_t_s_start_i,
        delta_t_s_end_i,
        delta_t_ns_start_i,
        delta_t_ns_end_i,
    )
    # update mask index to skip time token positions
    mask_i = TIME_END_I
    gen_token_carry = (m_seq, b_seq, mask_i, rng_)

    # finish message generation
    gen_token_carry, _ = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=l-TIME_END_I
    )
    (m_seq, b_seq, mask_i, rng_) = gen_token_carry

    # order_id = id_gen.step()  # no order ID generator any more in v3 sim?
    order_id = n_msg_todo

    sim_msg, msg_decoded = get_sim_msg(
        m_seq[-l:],  # the generated message
        # m_seq[:-l],  # sequence without generated message
        # m_seq_raw[1:],   # raw data (same length as sequence without generated message)
        # None,
        sim,
        sim_state,
        mid_price = p_mid,
        new_order_id = order_id,
        tick_size = tick_size,
        encoder = encoder,
    )

    # feed message to simulator, updating book state
    sim_state = sim.process_order_array(sim_state, sim_msg)

    # debug('trades', _trades)

    # get current mid price from simulator
    p_mid_new = _get_new_mid_price(sim, sim_state, p_mid, tick_size)

    # price change in ticks
    p_change = ((p_mid_new - p_mid) // tick_size)#.astype(jnp.int32)

    # get new book state
    book_l2 = sim.get_L2_state(sim_state, l2_state_n)
    # l2_book_states.append(book_l2)

    # error if the new message does not change the book state
    # is_error = (book_l2 == b_seq[-1, 1:]).all()
    # num_errors += is_error

    # TODO: count as error if book state does not change

    new_book_raw = jnp.concatenate([jnp.array([p_change]), book_l2]).reshape(1,-1)
    new_book = preproc.transform_L2_state(new_book_raw, 500, 100)
    # update book sequence
    b_seq = jnp.concatenate([b_seq[1:], new_book])

    n_msg_todo -= 1

    return msg_decoded, sim_state, m_seq, b_seq, book_l2, p_mid, n_msg_todo

    
def _make_generate_msg_scannable(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
    ):
    """
    """
    __generate_msg = jax.jit(functools.partial(
        _generate_msg, sim, train_state, model,batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size,
    ))

    def _generate_msg_scannable(gen_state, unused):
        """ Wrapper for _generate_msg to be used with jax.lax.scan
        """
        m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng = gen_state
        
        msg_decoded, sim_state, m_seq, b_seq, book_l2, p_mid, n_msg_todo = __generate_msg(*gen_state)

        return (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng), (msg_decoded, book_l2)
    return _generate_msg_scannable

@partial(jax.jit, static_argnums=(0, 2, 3, 5, 6, 9))
def generate(
        sim: OrderBook,  # static
        train_state: TrainState,
        model: nn.Module,  # static
        batchnorm: bool,  # static
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,  # static
        tick_size: int,  # static
        m_seq: jax.Array,
        b_seq: jax.Array,
        n_msg_todo: int,  # static
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        # if eval_msgs given, also returns loss of predictions
        # e.g. to calculate perplexity
        # m_seq_eval: Optional[jax.Array] = None,  
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    # id_gen = OrderIdGenerator()
    # l = Message_Tokenizer.MSG_LEN
    # v = Vocab()
    # vocab_len = len(v)
    # last_start_i = m_seq.shape[0] - l
    # l2_book_states = []
    m_seq = m_seq.copy()
    b_seq = b_seq.copy()
    # m_seq_raw = m_seq_raw.copy()
    # num_errors = 0

    with jax.ensure_compile_time_eval():
        valid_mask_array = valh.syntax_validation_matrix()

    # get current mid price from simulator
    p_mid = _get_safe_mid_price(sim, sim_state, tick_size)

    # while n_msg_todo > 0:
    gen_state, (msgs_decoded, l2_book_states) = jax.lax.scan(
        _make_generate_msg_scannable(
            sim, train_state, model, batchnorm, 
            encoder, valid_mask_array, sample_top_n, tick_size
        ),
        (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng),
        length=n_msg_todo,
        xs=None,
    )
    (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng) = gen_state

    # count errors when the message does not change the (visible) book state
    num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()

    return m_seq, b_seq, msgs_decoded, l2_book_states, num_errors

generate_batched = jax.jit(
    jax.vmap(
        generate,
        in_axes=(
            None, None, None, None,
            None, None, None,    0, 
               0, None,    0,    0
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9)
)

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def calc_sequence_losses(
        m_seq,
        b_seq,
        state,
        model,
        batchnorm,
        n_inp_msgs,  # length of input sequence in messages
        valid_mask_array
    ):
    """ Takes a sequence of messages, and calculates cross-entropy loss for each message,
        based on the next message in the sequence.
    """
    @partial(jax.jit, static_argnums=(1,2))
    def moving_window(a: jax.Array, size: int, stride: int = 1):
        starts = jnp.arange(0, len(a) - size + 1, stride)
        return jax.vmap(
            lambda start: jax.lax.dynamic_slice(
                a,
                (start, *jnp.zeros(a.ndim-1, dtype=jnp.int32)),
                (size, *a.shape[1:])
            )
        )(starts)
    
    l = Message_Tokenizer.MSG_LEN

    @jax.jit
    def prep_single_inp(
            mask_i,
            na_mask,
            m_seq,
            b_seq,
        ):
        m_seq = m_seq.copy().reshape((-1, l))
        last_msg = jnp.where(
            na_mask,
            Vocab.HIDDEN_TOK,#Vocab.NA_TOK,
            m_seq[-1]
        )
        m_seq = m_seq.at[-1, :].set(last_msg).reshape(-1)
        m_seq, y = valh.mask_last_msg_in_seq(m_seq, mask_i)

        input = (m_seq, b_seq)
        integration_timesteps = (
            jnp.ones(len(m_seq), dtype=jnp.float32), 
            jnp.ones(len(b_seq), dtype=jnp.float32)
        )
        return input, integration_timesteps, y.astype(jnp.float32)
    prep_multi_input = jax.vmap(prep_single_inp, in_axes=(0, 0, None, None))

    @jax.jit
    def single_msg_losses(carry, inp):
        @partial(jax.jit, static_argnums=(0,))
        def na_mask_slice(last_non_masked_i):
            a = jnp.ones((l,), dtype=jnp.bool_)
            a = a.at[: last_non_masked_i+1].set(False)
            return a

        m_seq, b_seq, valid_mask = inp
        mask_idxs = jnp.concatenate([jnp.arange(0, TIME_START_I), jnp.arange(TIME_END_I, l)])
        na_masks = jnp.array([na_mask_slice(i) for i in range(TIME_START_I)] \
            + [na_mask_slice(i) for i in range(TIME_END_I, l)])

        bsz = 10
        assert 2*bsz >= mask_idxs.shape[0], f'bsz:{bsz}; msg len:{mask_idxs.shape[0]}'
        # split inference into two batches to avoid OOM
        input, integration_timesteps, y1 = prep_multi_input(mask_idxs[:bsz], na_masks[:bsz], m_seq, b_seq)
        logits1 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)
        input, integration_timesteps, y2 = prep_multi_input(mask_idxs[-bsz:], na_masks[-bsz:], m_seq, b_seq)
        logits2 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)
        
        logits = jnp.concatenate([logits1, logits2[2*bsz - mask_idxs.shape[0] : ]], axis=0)
        y = jnp.concatenate([y1, y2[2*bsz - mask_idxs.shape[0] : ]], axis=0)
        
        # filter out (syntactically) invalid tokens for current position
        if valid_mask is not None:
            logits = valh.filter_valid_pred(logits, valid_mask)

        losses = train_helpers.cross_entropy_loss(logits, y)
        return carry, losses

    m_seq = m_seq.reshape((-1, l))
    inputs = (
        moving_window(m_seq, n_inp_msgs),
        moving_window(b_seq, n_inp_msgs),
        jnp.repeat(
            jnp.expand_dims(
                jnp.delete(valid_mask_array, slice(TIME_START_I, TIME_END_I), axis=0),
                axis=0
            ),
            m_seq.shape[0] - n_inp_msgs + 1,
            axis=0
        )
    )
    last_i, losses = jax.lax.scan(
        single_msg_losses,
        init=0,
        xs=inputs
    )
    return losses

# def generate_single_rollout(
#         m_seq_inp,
#         b_seq_inp,
#         n_gen_msgs,
#         sim,
#         sim_state,
#         state,
#         model,
#         batchnorm,
#         encoder,
#         rng,
#     ):
    
#     rng, rng_ = jax.random.split(rng)

#     # generate predictions
#     m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = generate(
#         m_seq_inp,
#         b_seq_inp,
#         n_gen_msgs,
#         sim,
#         sim_state,
#         state,
#         model,
#         batchnorm,
#         encoder,
#         rng_,
#         sample_top_n=-1,  # sample from entire distribution
#     )

#     return (
#         m_seq_gen,
#         b_seq_gen,
#         {
#             'num_errors': num_errors,
#             'l2_book_states': l2_book_states,
#         }
#     )

# # sample from distribution of rollouts with same input an different rng keys
# generate_repeated_rollouts = jax.vmap(generate_single_rollout, in_axes=((None,)*9 + (0,)))
# # sample different rollouts with different input sequences (and different rng keys)
# generate_multiple_rollouts = jax.vmap(generate_single_rollout, in_axes=(0, 0, None, None, 0, None, None, None, None, 0))

@partial(jax.jit, static_argnums=(5,))
def calculate_rollout_metrics(
        m_seq_raw_gen: jax.Array,
        m_seq_raw_eval: jax.Array,
        l2_book_states: jax.Array,
        l2_book_states_eval: jax.Array,
        l2_book_state_init: jax.Array,
        data_levels: int,
    ) -> Dict[str, jax.Array]:

    # arrival times
    delta_t_gen = m_seq_raw_gen[..., DTs_i].astype(jnp.float32) \
                + m_seq_raw_gen[..., DTns_i].astype(jnp.float32) / 1e9
    delta_t_gen = jnp.where(
        delta_t_gen > 0,
        delta_t_gen,
        1e-9
    )
    delta_t_eval = m_seq_raw_eval[..., DTs_i].astype(jnp.float32) \
                 + m_seq_raw_eval[..., DTns_i].astype(jnp.float32) / 1e9
    delta_t_eval = jnp.where(
        delta_t_eval > 0,
        delta_t_eval,
        1e-9
    )

    ## MID PRICE EVAL:
    # mid price at start of generation
    mid_t0 = l2_book_state_init[([0, 2],)].mean()

    if l2_book_states.ndim <= 2:
        l2_book_states = jnp.expand_dims(l2_book_states, axis=0)

    # mean mid-price over J iterations
    mid_gen = jnp.mean(
        (l2_book_states[..., 0] + l2_book_states[..., 2]) / 2.,
        axis=0
    )
    # filter out mid-prices where one side is empty
    mid_gen = jnp.where(
        ((l2_book_states[..., 0] < 0) | (l2_book_states[..., 2] < 0)).any(axis=0),
        jnp.nan,
        mid_gen
    )
    rets_gen = mid_gen / mid_t0 - 1
    
    mid_eval = (l2_book_states_eval[..., 0] + l2_book_states_eval[..., 2]) / 2.
    # filter our mid-prices where one side is empty
    mid_eval = jnp.where(
        (l2_book_states_eval[..., 0] < 0) | (l2_book_states_eval[..., 2] < 0),
        jnp.nan,
        mid_eval
    )
    rets_eval = mid_eval / mid_t0 - 1
    
    # shape: (n_eval_messages, )
    mid_ret_errs = eval.mid_price_ret_squ_err(
        mid_gen, mid_eval, mid_t0)
    # compare to squared error from const prediction
    mid_ret_errs_const = jnp.square(rets_eval)
    
    ## BOOK EVAL:
    # get loss sequence using J generations and 1 evaluation
    book_losses_l1 = eval.book_loss_l1_batch(l2_book_states, l2_book_states_eval, data_levels)
    book_losses_wass = eval.book_loss_wass_batch(l2_book_states, l2_book_states_eval, data_levels)
    # compare to loss between fixed book (at t0) and actual book
    # --> as if we were predicting with the most recent observation
    book_losses_l1_const = eval.book_loss_l1(
        jnp.tile(l2_book_state_init, (l2_book_states_eval.shape[0], 1)),
        l2_book_states_eval,
        data_levels
    )
    book_losses_wass_const = eval.book_loss_wass(
        jnp.tile(l2_book_state_init, (l2_book_states_eval.shape[0], 1)),
        l2_book_states_eval,
        data_levels
    )

    metrics = {
        'delta_t_gen': delta_t_gen,
        'delta_t_eval': delta_t_eval,
        'rets_gen': rets_gen,
        'rets_eval': rets_eval,
        'mid_ret_errs': mid_ret_errs,
        'mid_ret_errs_const': mid_ret_errs_const,
        'book_losses_l1': book_losses_l1,
        'book_losses_l1_const': book_losses_l1_const,
        'book_losses_wass': book_losses_wass,
        'book_losses_wass_const': book_losses_wass_const,
    }
    return metrics

def generate_repeated_rollouts_OLD(
        num_repeats: int,
        m_seq: jax.Array,
        b_seq_pv: jax.Array,
        msg_seq_raw: jax.Array,
        book_l2_init: jax.Array,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        rng: jax.dtypes.prng_key,
        n_vol_series: int,
        sim_book_levels: int,
        sim_queue_len: int,
        data_levels: int,
    ):

    l2_book_states = jnp.zeros((num_repeats, n_gen_msgs, sim_book_levels * 4))
    # how many messages had to be discarded
    num_errors = jnp.zeros(num_repeats, dtype=jnp.int32)
    event_types_gen = jnp.zeros((num_repeats, 4))
    event_types_eval = jnp.zeros((num_repeats, 4))
    raw_msgs_gen = jnp.zeros((num_repeats, n_gen_msgs, 14))

    # transform book to volume image representation for model
    b_seq = jnp.array(transform_L2_state(b_seq_pv, n_vol_series, 100))

    # encoded data
    m_seq_inp = m_seq[: seq_len]
    m_seq_eval = m_seq[seq_len: ]
    b_seq_inp = b_seq[: n_msgs]
    b_seq_eval = b_seq[n_msgs: ]
    # true L2 data
    b_seq_pv_eval = jnp.array(b_seq_pv[n_msgs: ])

    # raw LOBSTER data
    m_seq_raw_inp = msg_seq_raw[: n_msgs]
    m_seq_raw_eval = msg_seq_raw[n_msgs: ]

    # initialise simulator
    sim_init, sim_state_init = get_sim(
        book_l2_init,  # book state before any messages
        m_seq_raw_inp, # messages to replay to init sim
        # TODO: consider passing nOrders, nTrades
    )
    # book state after initialisation (replayed messages)
    l2_book_state_init = sim_init.get_L2_state(sim_state_init, l2_state_n)

    # run actual messages on sim_eval (once) to compare
    # convert m_seq_raw_eval to sim_msgs
    msgs_eval = msgs_to_jnp(m_seq_raw_eval[: n_gen_msgs])
    sim_state_eval, l2_book_states_eval, _ = sim_init.process_orders_array_l2(sim_state_init, msgs_eval, l2_state_n)

    # TODO: repeat for multiple scenarios from same input to average over
    #       --> parallelise? loaded data is the same, just different rngs
    for i in range(num_repeats):
        print('ITERATION', i)
        m_seq_gen, b_seq_gen, m_seq_raw_gen, rollout_metrics = generate_single_rollout(
            m_seq_inp,
            b_seq_inp,
            m_seq_raw_inp,
            n_gen_msgs,
            sim_init,
            sim_state_init,
            train_state,
            model,
            batchnorm,
            encoder,
            rng,
            m_seq_eval
        )
        event_types_gen = event_types_gen.at[i].set(rollout_metrics['event_types_gen'])
        event_types_eval = event_types_eval.at[i].set(eval.event_type_count(m_seq_raw_eval[:, 1]))
        num_errors = num_errors.at[i].set(rollout_metrics['num_errors'])
        l2_book_states = l2_book_states.at[i, :, :].set(rollout_metrics['l2_book_states'])
        raw_msgs_gen = raw_msgs_gen.at[i, :, :].set(m_seq_raw_gen)

    metrics = calculate_rollout_metrics(
        m_seq_raw_gen,
        m_seq_raw_eval,
        l2_book_states,
        l2_book_states_eval,
        l2_book_state_init,
        data_levels
    )
    metrics['l2_book_states'] = l2_book_states
    metrics['l2_book_states_eval'] = l2_book_states_eval
    metrics['num_errors'] = num_errors
    metrics['event_types_gen'] = event_types_gen
    metrics['event_types_eval'] = event_types_eval
    metrics['raw_msgs_gen'] = raw_msgs_gen
    metrics['raw_msgs_eval'] = m_seq_raw_eval

    return metrics

def sample_new(
        n_samples: int,  # draw n random samples from dataset for evaluation
        batch_size: int,  # how many samples to process in parallel
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        n_vol_series: int = 500,
        # sim_book_levels: int = 20,
        # sim_queue_len: int = 100,
        # data_levels: int = 10,
        save_folder: str = './data_saved/',
        tick_size: int = 100,
        sample_top_n: int = -1,
    ):
    """
    """
    assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'

    rng, rng_ = jax.random.split(rng)
    sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=False
    ).tolist()
    rng, rng_ = jax.random.split(rng)

    # create folders to save the data if they don't exist yet
    Path(save_folder + f'/data_cond/').mkdir(exist_ok=True, parents=True)
    Path(save_folder + f'/data_real/').mkdir(exist_ok=True, parents=True)
    Path(save_folder + f'/data_gen/').mkdir(exist_ok=True, parents=True)

    transform_L2_state_batch = jax.jit(
        jax.vmap(
            transform_L2_state,
            in_axes=(0, None, None)
        ),
        static_argnums=(1, 2)
    )

    # all_metrics = []
    for batch_i in tqdm(sample_i):
        print('BATCH', batch_i)
        # TODO: check if we can init the dataset without the raw data 
        #       if it's not needed 
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        m_seq = jnp.array(m_seq)
        b_seq_pv = jnp.array(b_seq_pv)
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = jnp.array(book_l2_init)

        # TODO: copy some logic from old repeated rollouts
        #     # transform book to volume image representation for model
        b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)

        # encoded data
        m_seq_inp = m_seq[..., : seq_len]
        m_seq_eval = m_seq[..., seq_len: ]
        b_seq_inp = b_seq[: , : n_msgs]
        b_seq_eval = b_seq[:, n_msgs: ]
        # true L2 data
        b_seq_pv_inp = onp.array(b_seq_pv[..., : n_msgs])
        b_seq_pv_eval = onp.array(b_seq_pv[..., n_msgs: ])

        # raw LOBSTER data
        m_seq_raw_inp = msg_seq_raw[..., : n_msgs]
        m_seq_raw_eval = msg_seq_raw[..., n_msgs: ]

        # initialise simulator
        sim_init, sim_states_init = get_sims_vmap(
            book_l2_init,  # book state before any messages
            m_seq_raw_inp, # messages to replay to init sim
            # TODO: consider passing nOrders, nTrades
        )

        # book state after initialisation (replayed messages)
        # actually, this is already part of the input data --> only needed for comparison
        # l2_book_states_init = sim_init.get_L2_states_vmap(sim_states_init, l2_state_n)

        # run actual messages on sim_eval (once) to compare
        # convert m_seq_raw_eval to sim_msgs
        # msgs_eval = msgs_to_jnp(m_seq_raw_eval[: n_gen_msgs])
        # sim_state_eval, l2_book_states_eval, _ = sim_init.process_orders_array_l2(sim_state_init, msgs_eval, l2_state_n)

        # print('m_seq_inp.shape', m_seq_inp.shape)
        # print('b_seq_inp.shape', b_seq_inp.shape)
        # print('sim_states_init.asks.shape', sim_states_init.asks.shape)
        # print('sim_states_init.bids.shape', sim_states_init.bids.shape)
        # print('sim_states_init.trades.shape', sim_states_init.trades.shape)
        m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = generate_batched(
            sim_init,
            train_state,
            model,
            batchnorm,
            encoder,
            sample_top_n,  # sample from entire distribution
            tick_size,
            m_seq_inp, # in_axis = 0
            b_seq_inp, # in_axis = 0
            n_gen_msgs,
            sim_states_init, # in_axis = 0
            jax.random.split(rng_, batch_size), # in_axis = 0
        )
        rng, rng_ = jax.random.split(rng)
        # TODO: save as metadata
        print('num_errors', num_errors)

        # TODO: remove this - just to test for first batch
        # return m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors

        # only keep actually newly generated messages
        # m_seq_raw_gen = m_seq_raw_gen[-n_gen_msgs:]

        # save data for all elements in the batch
        for i, cond_msg, cond_book, real_msg, real_book, gen_msg, gen_book \
            in zip(
                batch_i,
                m_seq_raw_inp, b_seq_pv_inp,
                m_seq_raw_eval, b_seq_pv_eval, 
                msgs_decoded, l2_book_states
            ):

            # input / cond data
            jnp.save(save_folder + f'/data_cond/message_real_id_{i}.npy', cond_msg)
            jnp.save(save_folder + f'/data_cond/orderbook_real_id_{i}.npy', cond_book)

            # real data
            jnp.save(save_folder + f'/data_real/message_real_id_{i}.npy', real_msg)
            jnp.save(save_folder + f'/data_real/orderbook_real_id_{i}.npy', real_book)
            
            # gen data
            jnp.save(save_folder + f'/data_gen/message_real_id_{i}_gen_id_0.npy', gen_msg)
            jnp.save(save_folder + f'/data_gen/orderbook_real_id_{i}_gen_id_0.npy', gen_book)
            

# TODO: jaxify and jit
def sample_messages(
        n_samples: int,  # draw n random samples from dataset for evaluation
        num_repeats: int,  # how often to repeat generation for each data sample
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        n_vol_series: int = 500,
        sim_book_levels: int = 20,
        sim_queue_len: int = 100,
        data_levels: int = 10,
        save_folder: str = './tmp/'
    ):

    rng, rng_ = jax.random.split(rng)
    # use this to load data (not jitted) and do parallel inference (jitted)
    sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples,),
        replace=False)

    all_metrics = []

    # create folder if it doesn't exist yet
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # TODO: vmap over samples or repeats or both?
    #       as long as there is no batched_vmap in jax, it's probably better to vmap over samples
    #       and then call the function multiple times or scan if more should be done...

    # iterate over random samples
    for i in tqdm(sample_i):

        # check if file already exists
        if os.path.isfile(save_folder + f'tmp_inference_results_dict_{i}.pkl'):
            print(f'Skipping existing sample {i}...')
            continue
        
        print(f'Processing sample {i}...')

        # 0: encoded message sequence
        # 1: prediction targets (dummy 0 here)
        # 2: book sequence (in Price, Volume format)
        # 3: raw message sequence (pandas df from LOBSTER)
        # 4: initial level 2 book state (before start of sequence)
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[int(i)]
        sequence_metrics = generate_repeated_rollouts(
            num_repeats,
            m_seq,
            b_seq_pv,
            msg_seq_raw,
            book_l2_init,
            seq_len,
            n_msgs,
            n_gen_msgs,
            train_state,
            model,
            batchnorm,
            encoder,
            rng,
            n_vol_series,
            sim_book_levels,
            sim_queue_len,
            data_levels
        )
        # save results dict as pickle file
        with open(save_folder + f'/tmp_inference_results_dict_{i}.pkl', 'wb') as f:
            pickle.dump(sequence_metrics, f)
        all_metrics.append(sequence_metrics)
    # combine metrics into single dict
    all_metrics = {
        metric: jnp.array([d[metric] for d in all_metrics])
        for metric in all_metrics[0].keys()
    }
    return all_metrics
