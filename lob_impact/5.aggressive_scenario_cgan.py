#!/usr/bin/env python
"""
CGAN (Coletta) Aggressive Scenario: GAN-based LOB Generation with Aggressive Order Injection

Uses the Conditional GAN model from Coletta et al. to generate LOB messages.
The CGAN takes a lookback window of market state features as input and
generates the next order (limit, market, or cancel).

Unlike S5, CGAN has no hidden state — the tracked market state features
(imbalance, spread, returns, exec imbalance) serve as the full model context.

Output format matches S5/CST scenarios (1.aggressive_scenario_s5.py,
4.aggressive_scenario_cst.py) so existing analysis notebooks (100-110)
work with results from all models.
"""

import argparse
import os
import sys
import yaml
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import jax
import jax.numpy as jnp
import numpy as onp
import torch
from tqdm import tqdm

# Add parent folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(script_dir)
sys.path.insert(0, parent_folder_path)

# Add AlphaTrade submodule
if os.path.exists('/AlphaTrade'):
    sys.path.insert(0, '/AlphaTrade')
else:
    sys.path.insert(0, os.path.join(parent_folder_path, 'Alphatrade'))

# Add CGAN model path
cgan_base = os.path.join(parent_folder_path, 'abides_worldmodel_offline', 'abides-markets')
sys.path.insert(0, cgan_base)

from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.inference_no_errcorr import (
    get_sims_vmap, get_dataset, msg_to_jnp, msgs_to_jnp,
    msg_to_lobster_format, book_to_lobster_format, construct_sim_msg,
)

# CGAN imports — set abides_test before importing ganmodels
from abides_markets.agents.gan.v2_41 import ganmodels
ganmodels.abides_test = True
from abides_markets.agents.gan.v2_41 import gan_utils

# Indices for DECODED message fields (from inference_no_errcorr.py)
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

AGGRESSIVE_ORDER_ID = 77777777
START_OID = 100

# CGAN feature set
FEATURE_SET = gan_utils.FeatureSet.NEW_SET
TIME_SCALER = gan_utils.TIME_SCALER  # 5 seconds


# ============================================================================
# Utilities (shared with CST scenario)
# ============================================================================

class TeeLogger:
    """Duplicates output to both console and log file."""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_logging(save_folder: Path) -> TeeLogger:
    """Set up logging to both console and file."""
    log_file = save_folder / 'experiment.log'
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


def create_experiment_folder(base_dir: str) -> Path:
    """Create experiment folder with sequential number and timestamp."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

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


SIM_CONFIG = JAXLOB_Configuration()


def update_oid(
    msg: jax.Array,
    rng: jax.Array,
    sim: OrderBook,
    sim_state: LobState,
) -> Tuple[jax.Array, jax.Array]:
    """Assign correct order ID for cancel/execution messages using JAX-LOB state."""
    def _leave_oid(msg, rng, *args):
        return msg, rng

    def _get_random_cancel_msg(msg, rng, sim, sim_state):
        side = msg[1]
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            sim_state.asks, sim_state.bids
        )
        rng, _rng = jax.random.split(rng)
        msg_dict = {
            "quantity": msg[2],
            "price": msg[3],
        }
        idx = job.get_random_id_match(SIM_CONFIG, _rng, side_array, msg_dict)
        cancelled_oid = side_array[idx, 2]
        msg = msg.at[5].set(cancelled_oid)
        return msg, rng

    def _get_oid_from_active(msg, rng, sim, sim_state):
        side = msg[1]
        oid = jax.lax.cond(
            side == 1,
            _get_top_bid_order_oid,
            _get_top_ask_order_oid,
            sim_state
        )
        return msg.at[5].set(oid), rng

    def _get_top_bid_order_oid(sim_state):
        idx = job._get_top_bid_order_idx(SIM_CONFIG, sim_state.bids).squeeze()
        return sim_state.bids[idx, 2]

    def _get_top_ask_order_oid(sim_state):
        idx = job._get_top_ask_order_idx(SIM_CONFIG, sim_state.asks).squeeze()
        return sim_state.asks[idx, 2]

    msg, rng = jax.lax.switch(
        # 0 -> leave, 1 -> random cancel, 2 -> get executable oid from active
        (msg[0] == 3) + 2 * (msg[0] == 4),
        (_leave_oid, _get_random_cancel_msg, _get_oid_from_active),
        msg, rng, sim, sim_state
    )
    return msg, rng


@jax.jit
def create_aggressive_order(
    sim: OrderBook,
    sim_state: LobState,
    last_time_s: jax.Array,
    last_time_ns: jax.Array,
    tick_size: int,
    event_type: int,
    direction: int,
    order_volume: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Create an aggressive market order based on current JAX-LOB book state.

    Returns:
        sim_msg: Message for JAX-LOB simulator (8 fields)
        msg_decoded: Decoded message for storage (14 fields)
    """
    # Get best price on the side we will aggress
    price = jax.lax.cond(
        direction == 0,
        lambda: sim.get_best_ask(sim_state),
        lambda: sim.get_best_bid(sim_state)
    )

    # Get available volume at best level
    best_bid_ask = sim.get_best_bid_and_ask_inclQuants(sim_state)
    avail = jax.lax.cond(
        direction == 0,
        lambda: best_bid_ask[1][1],  # ask volume for buy
        lambda: best_bid_ask[0][1],  # bid volume for sell
    ).astype(jnp.int32)

    # Cap order size at available volume
    quantity = jnp.minimum(jnp.int32(order_volume), avail)

    # Use time from last message + small increment
    time_s = last_time_s.astype(jnp.int32)
    time_ns = (last_time_ns + 1).astype(jnp.int32)

    # Build simulator message (8 fields)
    sim_msg = construct_sim_msg(
        event_type, direction, quantity, price,
        AGGRESSIVE_ORDER_ID, time_s, time_ns,
    )

    # Build decoded message for storage (14 fields)
    mid_price = (sim.get_best_ask(sim_state) + sim.get_best_bid(sim_state)) // 2
    mid_price = (mid_price // tick_size) * tick_size
    rel_price = (price - mid_price) // tick_size

    msg_decoded = jnp.array([
        AGGRESSIVE_ORDER_ID,  # order_id
        event_type,           # event_type
        direction,            # direction
        price,                # price_abs
        rel_price,            # price (relative)
        quantity,             # size
        0,                    # delta_t_s
        1,                    # delta_t_ns
        time_s,               # time_s
        time_ns,              # time_ns
        0,                    # price_ref
        0,                    # size_ref
        0,                    # time_s_ref
        0,                    # time_ns_ref
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded


# ============================================================================
# CGAN State Tracker
# ============================================================================

class CGANStateTracker:
    """
    Tracks market state features needed by the CGAN model.

    Maintains a rolling lookback window of state features and computes
    the features that the CGAN expects as input: imbalance, volumes,
    spread, returns, and execution imbalance.

    The features match those in gan_utils.FeatureSet.NEW_SET:
    - Order features (7): depth, del_depth, SIZE, SIZE100s, SIZE_TYPE, TYPE, BUY_SELL_FLAG
    - State features (10): imbalance_1, imbalance_5, qty_exec_imbalance_last_00:01:00,
      qty_exec_imbalance_last_00:05:00, vols_1, vols_5, spread, returns_1, returns_50, midprice
    """

    def __init__(self, scalers: dict, lookback_window: int):
        self.scalers = scalers
        self.lookback_window = lookback_window

        # Semantic features used in the lookback (order + state, minus SIZE100s and SIZE_TYPE)
        self.semantic_features = ["timestamp", "best_bid", "best_ask"] + [
            f for f in FEATURE_SET.all_features()
            if f not in ["SIZE100s", "SIZE_TYPE"]
        ]

        # Rolling state
        self.lookback_raw_state = deque([], maxlen=lookback_window)
        self.historical_mid_price = deque([], maxlen=50)
        self.current_discrete_time = 0

        # Market order tracking for tick-based exec imbalance
        # Each entry: (qty, side_flag) where side_flag: -1=BID, 1=ASK
        self.mo_buffer = deque([], maxlen=256)

        # Current state dict
        self.state = {}

    def init_from_l2_sequence(
        self,
        l2_states: onp.ndarray,
        timestamps_s: onp.ndarray,
        timestamps_ns: onp.ndarray,
        msgs_raw: onp.ndarray,
        tick_size: int,
    ):
        """
        Initialize tracker from conditioning L2 states and messages.

        Args:
            l2_states: (n_cond+1, n_levels*4) L2 book states (ask_p, ask_v, bid_p, bid_v, ...)
            timestamps_s: (n_cond,) seconds for each message
            timestamps_ns: (n_cond,) nanoseconds for each message
            msgs_raw: (n_cond, 14) raw decoded messages
            tick_size: price tick size
        """
        n_msgs = len(timestamps_s)

        # Build initial MO buffer from conditioning messages
        for j in range(n_msgs):
            event_type = int(msgs_raw[j, EVENT_TYPE_i])
            direction = int(msgs_raw[j, DIRECTION_i])
            size = int(msgs_raw[j, SIZE_i])
            if event_type == 4:  # Market order
                side_flag = -1 if direction == 0 else 1  # BID=-1 buy MO, ASK=1 sell MO
                self.mo_buffer.append((size, side_flag))

        # Process L2 states to build lookback
        for j in range(n_msgs):
            l2 = l2_states[j + 1]  # +1 because l2_states[0] is before first msg
            ts_ns = int(timestamps_s[j]) * int(1e9) + int(timestamps_ns[j])

            best_ask = float(l2[0])
            best_bid = float(l2[2])

            if best_ask <= 0 or best_bid <= 0:
                continue

            spread = best_ask - best_bid
            midprice = (best_ask + best_bid) / 2.0

            # Compute features
            state = self._compute_state_from_l2(l2, midprice, spread, ts_ns)
            self.historical_mid_price.append(midprice)

            # Discrete time bucketing
            sec_approx = TIME_SCALER
            discrete_time = int(sec_approx * ((ts_ns / 1e9) // sec_approx))

            if discrete_time > self.current_discrete_time:
                row = [state[f] for f in self.semantic_features]
                self.current_discrete_time = discrete_time
                self.lookback_raw_state.append(row)

            # Update current state
            self.state = state

    def _compute_state_from_l2(
        self,
        l2_flat: onp.ndarray,
        midprice: float,
        spread: float,
        timestamp_ns: int,
    ) -> dict:
        """Compute all state features from a flat L2 array."""
        # L2 format: [askP_0, askV_0, bidP_0, bidV_0, askP_1, askV_1, bidP_1, bidV_1, ...]
        best_ask = float(l2_flat[0])
        best_bid = float(l2_flat[2])

        # Parse levels
        n_levels = len(l2_flat) // 4
        ask_vols = []
        bid_vols = []
        for lv in range(n_levels):
            ask_vols.append(float(l2_flat[4 * lv + 1]))
            bid_vols.append(float(l2_flat[4 * lv + 3]))

        # Imbalance at depth d = sum(bid_vols[:d]) / (sum(bid_vols[:d]) + sum(ask_vols[:d]))
        def imbalance_n(d):
            bv = sum(bid_vols[:d])
            av = sum(ask_vols[:d])
            return bv / (bv + av) if (bv + av) > 0 else 0.5

        def vols_n(d):
            return sum(bid_vols[:d]) + sum(ask_vols[:d])

        imbalance_1 = imbalance_n(1)
        imbalance_5 = imbalance_n(5)
        vols_1 = vols_n(1)
        vols_5 = vols_n(5)

        # Returns
        returns_1 = 0.0
        returns_50 = 0.0
        if len(self.historical_mid_price) > 0:
            returns_1 = midprice / self.historical_mid_price[-1] - 1
            returns_50 = midprice / self.historical_mid_price[0] - 1

        # Tick-based exec imbalance
        exec_imbalance_128 = self._compute_exec_imbalance(128)
        exec_imbalance_256 = self._compute_exec_imbalance(256)

        state = {
            "timestamp": timestamp_ns,
            "best_bid": best_bid,
            "best_ask": best_ask,
            # Order features (placeholders — filled only when generating)
            "depth": 0,
            "del_depth": 0,
            "SIZE": 100,
            "TYPE": 0,
            "BUY_SELL_FLAG": -1,
            # State features
            "imbalance_1": imbalance_1,
            "imbalance_5": imbalance_5,
            "qty_exec_imbalance_last_00:01:00": exec_imbalance_128,
            "qty_exec_imbalance_last_00:05:00": exec_imbalance_256,
            "vols_1": vols_1,
            "vols_5": vols_5,
            "spread": spread,
            "returns_1": returns_1,
            "returns_50": returns_50,
            "midprice": midprice,
        }
        return state

    def _compute_exec_imbalance(self, n_ticks: int) -> float:
        """Compute execution imbalance over last n_ticks market orders."""
        recent = list(self.mo_buffer)[-n_ticks:]
        if len(recent) == 0:
            return 0.5
        bid_vol = sum(qty for qty, side in recent if side == -1)
        ask_vol = sum(qty for qty, side in recent if side == 1)
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def update_after_order(
        self,
        action_type: str,
        side: str,
        qty: int,
        l2_flat: onp.ndarray,
        timestamp_ns: int,
    ):
        """Update tracker state after an order is applied to the book."""
        best_ask = float(l2_flat[0])
        best_bid = float(l2_flat[2])

        if best_ask <= 0 or best_bid <= 0:
            return

        spread = best_ask - best_bid
        midprice = (best_ask + best_bid) / 2.0

        # Update MO buffer
        if action_type == "MARKET_ORDER":
            side_flag = -1 if side == "BUY" else 1
            self.mo_buffer.append((qty, side_flag))

        # Compute new state
        state = self._compute_state_from_l2(l2_flat, midprice, spread, timestamp_ns)
        self.historical_mid_price.append(midprice)

        # Discrete time bucketing
        sec_approx = TIME_SCALER
        discrete_time = int(sec_approx * ((timestamp_ns / 1e9) // sec_approx))

        if discrete_time > self.current_discrete_time:
            row = [state[f] for f in self.semantic_features]
            self.current_discrete_time = discrete_time
            self.lookback_raw_state.append(row)

        self.state = state

    def build_lookback_tensor(self) -> torch.Tensor:
        """
        Build the lookback input tensor for the CGAN model.

        Returns:
            torch.Tensor of shape (1, lookback_window, n_all_features)
        """
        import pandas as pd

        # Build DataFrame from lookback + current state
        rows = list(self.lookback_raw_state) + [
            [self.state[f] for f in self.semantic_features]
        ]
        lookback_df = pd.DataFrame(rows, columns=self.semantic_features)

        # Normalize using scalers
        lookback_df = self._normalize_lookback(lookback_df)

        # Discretize time and forward-fill
        sec_approx = TIME_SCALER
        lookback_df["discrete_time"] = (
            sec_approx * ((lookback_df["timestamp"] / 1e9) // sec_approx)
        ).astype(int)
        lookback_df = lookback_df.drop_duplicates(subset=["discrete_time"], keep="last")
        lookback_df = lookback_df.set_index("discrete_time")

        cur_discrete_time = int(sec_approx * ((self.state["timestamp"] / 1e9) // sec_approx))
        if len(lookback_df) > 0:
            new_index = list(range(lookback_df.index[0], cur_discrete_time + 1, sec_approx))
            lookback_df = lookback_df.reindex(new_index).ffill()

        all_features = FEATURE_SET.all_features()
        lookback_df = lookback_df[-self.lookback_window:][all_features]

        # Convert to tensor
        gan_input = torch.FloatTensor(lookback_df.values).unsqueeze(0)
        return gan_input

    def _normalize_lookback(self, df):
        """
        Normalize the lookback DataFrame for CGAN input.

        Simplified version of gan_utils.normalize_input_data() that works
        without ABIDES-specific dependencies (Side enum, fmt_ts, etc.).
        """
        import pandas as pd

        all_features = FEATURE_SET.all_features()

        # Handle SIZE fields
        df["SIZE_TYPE"] = onp.where(df["SIZE"] % 100 == 0, -1, 1)
        df["SIZE100s"] = onp.where(df["SIZE"] % 100 == 0, df["SIZE"] / 100, onp.nan)
        df["SIZE100s"] = df["SIZE100s"].ffill().bfill().fillna(1.0)
        df["SIZE"] = onp.where(df["SIZE"] % 100 != 0, df["SIZE"], onp.nan)
        df["SIZE"] = df["SIZE"].ffill().bfill().fillna(100.0)

        # Fill depth for market orders: use negative spread
        if "depth" in all_features:
            min_depth = df["depth"].dropna().min()
            if pd.isna(min_depth):
                min_depth = 0
            mo_mask = df["TYPE"] == 0
            df.loc[mo_mask, "depth"] = -df.loc[mo_mask, "spread"].clip(lower=min_depth)
        df["del_depth"] = df["del_depth"].ffill().bfill().fillna(0.0)
        df["depth"] = df["depth"].ffill().bfill().fillna(0.0)

        # TYPE is already numeric: -1=cancel, 0=market, 1=limit
        df["TYPE"] = df["TYPE"].astype(int)

        # BUY_SELL_FLAG: already -1=BUY, 1=SELL
        # (We use the same convention as in worldagent_vgan)

        # Scale features using saved scalers
        features_to_scale = [
            "depth", "spread", "SIZE", "SIZE100s", "del_depth", "vols_1", "vols_5",
        ]
        for col in features_to_scale:
            if col in all_features and col in self.scalers:
                vals = df[col].values.reshape(-1, 1)
                df[col] = self.scalers[col].transform(vals)

        # Fill exec imbalance NaN
        for col in all_features:
            if "qty_exec_imbalance" in col:
                df[col] = df[col].fillna(0.5)

        return df

    def decode_cgan_output(self, y_hat: onp.ndarray) -> dict:
        """
        Decode CGAN output to an action dictionary.

        Port of worldagent_vgan.py:unnormalized_order() + sanitize_action().

        Args:
            y_hat: CGAN output array of shape (n_order_features,)

        Returns:
            dict with action_type, side, qty, limit_price, depth, del_depth
        """
        order_features = FEATURE_SET.order_features()
        output = {}

        # Inverse-scale each order feature
        for i, feature in enumerate(order_features):
            val = y_hat[i]
            if feature in self.scalers:
                val = self.scalers[feature].inverse_transform(
                    onp.array([[val]])
                )[0][0]
            output[feature] = val

        # Un-normalized SIZE
        output["SIZE"] = (
            output["SIZE"] if output["SIZE_TYPE"] > 0 else output["SIZE100s"] * 100
        )

        # Semantic type conversion
        if output["TYPE"] <= -0.33:
            output["action_type"] = "CANCEL_FULL_ORDER"
        elif output["TYPE"] >= 0.33:
            output["action_type"] = "ADD_LIMIT_ORDER"
        else:
            output["action_type"] = "MARKET_ORDER"

        # Side
        output["side"] = "BUY" if output["BUY_SELL_FLAG"] <= 0 else "SELL"

        # Sanitize
        spread = self.state.get("spread", 1)
        if output["action_type"] == "ADD_LIMIT_ORDER":
            output["depth"] = max(int(output["depth"]), -(int(spread) - 1))
        if output["action_type"] == "CANCEL_FULL_ORDER":
            output["del_depth"] = max(output["del_depth"], 0)
        if output["action_type"] in ["ADD_LIMIT_ORDER", "MARKET_ORDER"]:
            if output["SIZE"] <= 0:
                output["SIZE"] = 1

        # Qty
        output["qty"] = max(1, int(output["SIZE"]))

        # Limit price from depth
        near_touch = (
            self.state["best_bid"] if output["side"] == "BUY"
            else self.state["best_ask"]
        )
        limit_price = near_touch + (-1 if output["side"] == "BUY" else 1) * int(output["depth"])
        output["limit_price"] = max(0, int(limit_price))

        # Cancel price from del_depth
        near_touch_cancel = (
            self.state["best_bid"] if output["side"] == "BUY"
            else self.state["best_ask"]
        )
        del_price = near_touch_cancel + (-1 if output["side"] == "BUY" else 1) * int(output.get("del_depth", 0))
        output["cancel_price"] = max(0, int(del_price))

        return output


# ============================================================================
# CGAN order → JAX-LOB simulator message
# ============================================================================

def cgan_order_to_sim_msg(
    action: dict,
    oid: int,
    time_s: int,
    time_ns: int,
    l2_flat: onp.ndarray,
) -> jax.Array:
    """
    Convert a decoded CGAN action dict to a JAX-LOB 8-field simulator message.
    """
    direction = 0 if action["side"] == "BUY" else 1

    if action["action_type"] == "ADD_LIMIT_ORDER":
        event_type = 1
        price = action["limit_price"]
        qty = action["qty"]
    elif action["action_type"] == "MARKET_ORDER":
        event_type = 4
        # Price is the best price on the opposing side
        if direction == 0:  # buy → hit ask
            price = int(l2_flat[0])  # best ask
        else:  # sell → hit bid
            price = int(l2_flat[2])  # best bid
        qty = action["qty"]
    elif action["action_type"] == "CANCEL_FULL_ORDER":
        event_type = 3
        price = action["cancel_price"]
        qty = action["qty"]
    else:
        raise ValueError(f"Unknown action type: {action['action_type']}")

    return construct_sim_msg(
        jnp.int32(event_type),
        jnp.int32(direction),
        jnp.int32(qty),
        jnp.int32(price),
        jnp.int32(oid),
        jnp.int32(time_s),
        jnp.int32(time_ns),
    )


def build_decoded_msg(
    action: dict,
    oid: int,
    time_s: int,
    time_ns: int,
    l2_flat: onp.ndarray,
    tick_size: int,
) -> jax.Array:
    """Build the 14-field decoded message for LOBSTER CSV output."""
    direction = 0 if action["side"] == "BUY" else 1

    if action["action_type"] == "ADD_LIMIT_ORDER":
        event_type = 1
        price = action["limit_price"]
    elif action["action_type"] == "MARKET_ORDER":
        event_type = 4
        price = int(l2_flat[0]) if direction == 0 else int(l2_flat[2])
    elif action["action_type"] == "CANCEL_FULL_ORDER":
        event_type = 3
        price = action["cancel_price"]
    else:
        raise ValueError(f"Unknown action type: {action['action_type']}")

    qty = action["qty"]

    # Compute relative price
    best_ask = float(l2_flat[0])
    best_bid = float(l2_flat[2])
    mid_price = (best_ask + best_bid) / 2.0
    mid_price = int(mid_price // tick_size) * tick_size
    rel_price = (price - mid_price) // tick_size if mid_price > 0 else 0

    return jnp.array([
        oid,            # order_id
        event_type,     # event_type
        direction,      # direction
        price,          # price_abs
        rel_price,      # price (relative)
        qty,            # size
        0,              # delta_t_s
        0,              # delta_t_ns
        time_s,         # time_s
        time_ns,        # time_ns
        0,              # price_ref
        0,              # size_ref
        0,              # time_s_ref
        0,              # time_ns_ref
    ], dtype=jnp.int32)


# ============================================================================
# Main scenario function
# ============================================================================

def run_cgan_scenario(cfg: Dict[str, Any], save_folder: Path):
    """
    Main function for CGAN aggressive scenario.

    Flow:
    1. Load CGAN model & scalers
    2. Load dataset and sample indices
    3. Initialize JAX-LOB simulator with conditioning messages
    4. For each sample: run CGAN generation blocks with aggressive order injections
    5. Save results in LOBSTER CSV format
    """
    # Unpack config
    n_gen_msgs = cfg['n_gen_msgs']
    num_insertions = cfg['num_insertions']
    num_coolings = cfg['num_coolings']
    n_cond_msgs = cfg['n_cond_msgs']
    n_samples = cfg['n_samples']
    batch_size = cfg['batch_size']
    rng_seed = cfg['rng_seed']
    stock = cfg['stock']
    data_dir = cfg['data_dir']
    tick_size = cfg['tick_size']
    n_levels = cfg['n_levels']
    test_split = cfg.get('test_split', 0)
    event_type = cfg['event_type']
    direction = cfg['direction']
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)
    order_volume = cfg['order_volume']
    cgan_checkpoint = cfg['cgan_checkpoint']
    cgan_scalers_path = cfg['cgan_scalers']
    cgan_interarrival_path = cfg.get('cgan_interarrival_times', None)
    lookback_window = cfg.get('lookback_window', 100)

    total_blocks = num_insertions + num_coolings

    # 1. Load CGAN model
    print(f"Loading CGAN model from {cgan_checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cgan_model = ganmodels.LOBGAN.load_from_checkpoint(
        cgan_checkpoint, map_location=device
    )
    cgan_model.eval()
    print(f"CGAN model loaded on {device}")

    # Load scalers
    print(f"Loading scalers from {cgan_scalers_path}")
    with open(cgan_scalers_path, "rb") as f:
        scalers = pickle.load(f)
    print(f"Scalers loaded: {list(scalers.keys())}")

    # Load interarrival times
    interarrival_k = 1.0
    interarrival_theta = 0.05  # default ~50ms
    if cgan_interarrival_path and os.path.exists(cgan_interarrival_path):
        print(f"Loading interarrival times from {cgan_interarrival_path}")
        with open(cgan_interarrival_path, "rb") as f:
            ia_params = pickle.load(f)
        interarrival_k = ia_params["k"]
        interarrival_theta = ia_params["theta"]
        print(f"Interarrival time params: k={interarrival_k:.4f}, theta={interarrival_theta:.4f}")
    else:
        print(f"No interarrival times file found, using defaults: k={interarrival_k}, theta={interarrival_theta}")

    # 2. Initialize
    rng = jax.random.key(rng_seed)
    np_rng = onp.random.RandomState(rng_seed)
    torch.manual_seed(rng_seed)

    # Initialize JAX-LOB simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    # Load dataset
    print(f"Loading dataset from {data_dir}")
    ds = get_dataset(
        data_dir,
        n_cond_msgs,
        n_eval_msgs_dataset,
        test_split=test_split,
    )
    print(f"Dataset length: {len(ds)}")

    # Create output folders
    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    # Sample indices (same RNG pattern as other scenarios for reproducibility)
    assert n_samples % batch_size == 0, f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=False
    ).tolist()

    # Build insertion schedule (0-indexed positions of aggressive orders in output)
    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1
    print(f"Aggressive order positions (0-indexed): {aggressive_positions}")
    print(f"Total messages per sample: {pos}")

    # 3. Process batches
    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')

        # Load data
        jax_device = jax.devices()[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), jax_device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), jax_device)
        b_seq_pv = jnp.array(b_seq_pv)

        # Split conditioning
        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs, :]
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])
        init_time = b_seq_pv[:, 0, 1:3]
        init_time = jax.device_put(jnp.array(init_time), jax_device)

        # Initialize JAX-LOB simulators by replaying conditioning messages
        sim_states = get_sims_vmap(
            book_l2_init, m_seq_raw_cond, init_time, sim,
        )

        # Save conditioning data for each sample in batch
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False
            )
            book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False
            )

        # 4. Process each sample individually
        for i, sample_idx in enumerate(batch_i):
            # Extract per-sample JAX-LOB state
            sim_state_i = jax.tree.map(lambda x: x[i], sim_states)

            # Get post-conditioning L2 from JAX-LOB
            l2_flat = onp.array(sim.get_L2_state(sim_state_i, n_levels))

            # Get conditioning messages and L2 for tracker initialization
            cond_msgs_np = onp.array(m_seq_raw_cond[i])
            cond_l2_np = onp.array(b_seq_pv_cond[i])
            cond_times_s = cond_msgs_np[:, TIMEs_i]
            cond_times_ns = cond_msgs_np[:, TIMEns_i]

            # Initialize CGAN state tracker
            tracker = CGANStateTracker(scalers, lookback_window)
            tracker.init_from_l2_sequence(
                cond_l2_np, cond_times_s, cond_times_ns,
                cond_msgs_np, tick_size,
            )

            # Per-sample RNG
            rng, rng_sample = jax.random.split(rng)

            # Current time (from last conditioning message)
            current_time_s = int(cond_times_s[-1])
            current_time_ns = int(cond_times_ns[-1])
            current_time_total_ns = current_time_s * int(1e9) + current_time_ns

            all_msgs = []
            all_books = []
            oid_counter = jnp.int32(START_OID)

            for block in range(total_blocks):
                for msg_idx in range(n_gen_msgs):
                    # Sample interarrival time
                    dt_seconds = np_rng.gamma(interarrival_k, interarrival_theta)
                    current_time_total_ns += int(dt_seconds * 1e9)
                    time_s = int(current_time_total_ns // int(1e9))
                    time_ns = int(current_time_total_ns % int(1e9))

                    # Build lookback tensor and generate
                    gan_input = tracker.build_lookback_tensor()
                    with torch.no_grad():
                        y_hat = cgan_model((gan_input, None, None, 1)).detach().numpy()[0][0]

                    # Decode CGAN output to action dict
                    action = tracker.decode_cgan_output(y_hat)

                    # Convert to JAX-LOB sim message
                    sim_msg = cgan_order_to_sim_msg(
                        action, int(oid_counter), time_s, time_ns, l2_flat,
                    )

                    # Assign correct OID for cancel/execution messages
                    sim_msg, rng_sample = update_oid(
                        sim_msg, rng_sample, sim, sim_state_i,
                    )

                    # Apply to JAX-LOB simulator
                    sim_state_i = sim.process_order_array(sim_state_i, sim_msg)

                    # Get new L2 state
                    l2_flat = onp.array(sim.get_L2_state(sim_state_i, n_levels))

                    # Build decoded message for output
                    msg_decoded = build_decoded_msg(
                        action, int(sim_msg[4]), time_s, time_ns, l2_flat, tick_size,
                    )

                    # Update tracker
                    tracker.update_after_order(
                        action["action_type"], action["side"], action["qty"],
                        l2_flat, current_time_total_ns,
                    )

                    all_msgs.append(msg_decoded)
                    all_books.append(jnp.array(l2_flat))

                    oid_counter = oid_counter + 1

                # Inject aggressive order after insertion blocks
                if block < num_insertions:
                    last_time_s = jnp.int32(time_s)
                    last_time_ns = jnp.int32(time_ns)

                    sim_msg_aggr, msg_decoded_aggr = create_aggressive_order(
                        sim, sim_state_i,
                        last_time_s, last_time_ns,
                        tick_size, event_type, direction, order_volume,
                    )

                    # Apply to JAX-LOB
                    sim_state_i = sim.process_order_array(sim_state_i, sim_msg_aggr)

                    # Get L2 after aggressive order
                    l2_flat = onp.array(sim.get_L2_state(sim_state_i, n_levels))

                    # Update tracker for aggressive order
                    aggr_qty = int(msg_decoded_aggr[SIZE_i])
                    aggr_dir = "BUY" if direction == 0 else "SELL"
                    tracker.update_after_order(
                        "MARKET_ORDER", aggr_dir, aggr_qty,
                        l2_flat,
                        int(msg_decoded_aggr[TIMEs_i]) * int(1e9) + int(msg_decoded_aggr[TIMEns_i]),
                    )

                    all_msgs.append(msg_decoded_aggr)
                    all_books.append(jnp.array(l2_flat))

            # Stack all messages and books
            all_msgs_arr = jnp.stack(all_msgs, axis=0)
            all_books_arr = jnp.stack(all_books, axis=0)

            # Save generated data
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(all_msgs_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )
            book_to_lobster_format(all_books_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )

            if batch_idx == 0 and i == 0:
                print(f"  First sample: {all_msgs_arr.shape[0]} total messages "
                      f"({total_blocks} blocks x {n_gen_msgs} + {num_insertions} aggressive)")

        # Save aggressive indices once
        if batch_idx == 0:
            aggressive_indices = onp.array(aggressive_positions)
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="CGAN (Coletta) Aggressive Scenario")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lob_impact/5.aggressive_scenario_cgan_config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument('--n_gen_msgs', type=int, default=None, help='Override n_gen_msgs from config')
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1], help='Override direction (0=buy, 1=sell)')
    return parser.parse_args()


def main():
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")

    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs
    if args.direction is not None:
        cfg['direction'] = args.direction

    print(f"Configuration: {cfg}")

    # Create experiment folder
    save_folder = create_experiment_folder(cfg['save_dir'])
    print(f"Experiment folder: {save_folder}")

    # Set up logging
    logger = setup_logging(save_folder)
    print(f"\n{'='*60}")
    print(f"CGAN (Coletta) Aggressive Scenario Experiment")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"Configuration: {cfg}")
    print(f"Experiment folder: {save_folder}")

    # Save config
    with open(save_folder / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    try:
        run_cgan_scenario(cfg, save_folder)

        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {save_folder}")
        print(f"{'='*60}")
    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log saved to: {save_folder / 'experiment.log'}")


if __name__ == "__main__":
    main()
