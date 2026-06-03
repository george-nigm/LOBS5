#!/usr/bin/env python
"""
CGAN (Coletta) Aggressive Scenario v2: Optimized with batched inference.

Key optimizations over v1 (5.aggressive_scenario_cgan.py):
1. Numpy-based state tracker — no pandas per-message overhead (~50x faster lookback)
2. Batched CGAN forward pass — process batch_size samples simultaneously
3. Vectorized decode of CGAN output — numpy instead of per-sample dicts
4. Pre-allocated output arrays — no per-message list appends

Output format is identical to v1 for compatibility with analysis notebooks (100-130).
"""

import argparse
import os
import sys
import yaml
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".40")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

# Add parent folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(os.path.dirname(script_dir))
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

# CGAN imports — mock ABIDES deps, then import ganmodels
import lob_impact.core._cgan_mocks  # noqa: F401
from abides_markets.agents.gan.v2_41 import ganmodels
ganmodels.abides_test = True
from abides_markets.agents.gan.v2_41 import gan_utils

# ============================================================================
# Constants
# ============================================================================
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

FEATURE_SET = gan_utils.FeatureSet.NEW_SET
TIME_SCALER = gan_utils.TIME_SCALER  # 5 seconds
ALL_FEATURES = FEATURE_SET.all_features()
ORDER_FEATURES = FEATURE_SET.order_features()
N_ALL_FEAT = len(ALL_FEATURES)
N_ORDER_FEAT = len(ORDER_FEATURES)

# Feature indices within ALL_FEATURES
FI = {f: i for i, f in enumerate(ALL_FEATURES)}
# Order feature indices within ORDER_FEATURES
OFI = {f: i for i, f in enumerate(ORDER_FEATURES)}

SIM_CONFIG = JAXLOB_Configuration()


# ============================================================================
# Utilities (shared with other scenarios)
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
    log_file = save_folder / 'experiment.log'
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


def create_experiment_folder(base_dir: str) -> Path:
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


# ============================================================================
# JIT+vmap'd JAX-LOB step functions
# ============================================================================
def make_step_fns(sim, n_levels):
    """
    Create JIT+vmap'd step functions for batched JAX-LOB processing.

    Instead of processing N samples sequentially (N calls per message),
    these functions process all N samples in ONE vmapped GPU call.

    Returns:
        step_vmap:  (N,8), (N,key), batched_LobState →
                    (N,8), (N,key), batched_LobState, (N, n_levels*4)
        aggr_vmap:  batched_LobState, (N,), (N,), int, int, int, int →
                    batched_LobState, (N, n_levels*4), (N, 14)
    """
    cfg = SIM_CONFIG

    def _step_one(sim_msg, rng, sim_state):
        """update_oid + process_order + get_L2 for one sample."""
        # --- update_oid (inlined, no sim/OrderBook arg in lax.switch) ---
        def _leave(msg, rng, ss):
            return msg, rng

        def _cancel(msg, rng, ss):
            side = msg[1]
            side_array = jax.lax.cond(
                side == 1, lambda a, b: b, lambda a, b: a,
                ss.asks, ss.bids)
            rng, k = jax.random.split(rng)
            msg_dict = {"quantity": msg[2], "price": msg[3]}
            idx = job.get_random_id_match(cfg, k, side_array, msg_dict)
            oid = side_array[idx, 2]
            return msg.at[5].set(oid), rng

        def _exec(msg, rng, ss):
            side = msg[1]
            def _bid(s):
                idx = job._get_top_bid_order_idx(cfg, s.bids).squeeze()
                return s.bids[idx, 2]
            def _ask(s):
                idx = job._get_top_ask_order_idx(cfg, s.asks).squeeze()
                return s.asks[idx, 2]
            oid = jax.lax.cond(side == 1, _bid, _ask, ss)
            return msg.at[5].set(oid), rng

        sim_msg, rng = jax.lax.switch(
            (sim_msg[0] == 3) + 2 * (sim_msg[0] == 4),
            (_leave, _cancel, _exec),
            sim_msg, rng, sim_state
        )
        # --- process order + extract L2 ---
        sim_state = sim.process_order_array(sim_state, sim_msg)
        l2 = sim.get_L2_state(sim_state, n_levels)
        return sim_msg, rng, sim_state, l2

    step_vmap = jax.jit(jax.vmap(_step_one))

    def _aggr_one(sim_state, last_s, last_ns,
                  tick_size, event_type, direction, order_volume):
        """Create aggressive order + apply + get_L2 for one sample."""
        price = jax.lax.cond(
            direction == 0,
            lambda: sim.get_best_ask(sim_state),
            lambda: sim.get_best_bid(sim_state))
        bba = sim.get_best_bid_and_ask_inclQuants(sim_state)
        avail = jax.lax.cond(
            direction == 0, lambda: bba[1][1], lambda: bba[0][1]
        ).astype(jnp.int32)
        quantity = jnp.minimum(jnp.int32(order_volume), avail)
        time_s = last_s.astype(jnp.int32)
        time_ns = (last_ns + 1).astype(jnp.int32)

        sim_msg = construct_sim_msg(
            event_type, direction, quantity, price,
            AGGRESSIVE_ORDER_ID, time_s, time_ns)

        mid_price = (sim.get_best_ask(sim_state) +
                     sim.get_best_bid(sim_state)) // 2
        mid_price = (mid_price // tick_size) * tick_size
        rel_price = (price - mid_price) // tick_size

        msg_decoded = jnp.array([
            AGGRESSIVE_ORDER_ID, event_type, direction, price, rel_price,
            quantity, 0, 1, time_s, time_ns, 0, 0, 0, 0,
        ], dtype=jnp.int32)

        sim_state = sim.process_order_array(sim_state, sim_msg)
        l2 = sim.get_L2_state(sim_state, n_levels)
        return sim_state, l2, msg_decoded

    aggr_vmap = jax.jit(jax.vmap(
        _aggr_one, in_axes=(0, 0, 0, None, None, None, None)))

    return step_vmap, aggr_vmap


# ============================================================================
# FastCGANTracker — numpy-based batched state tracker
# ============================================================================
class FastCGANTracker:
    """
    Numpy-based CGAN state tracker for N samples processed in parallel.

    Replaces CGANStateTracker (v1) which used pandas DataFrames per-message.
    Maintains pre-normalized lookback buffers updated incrementally.
    """

    # Features that need MinMaxScaler
    SCALED_FEATS = ["depth", "del_depth", "SIZE", "SIZE100s", "vols_1", "vols_5", "spread"]

    def __init__(self, n_samples: int, scalers: dict, lookback_window: int):
        self.N = n_samples
        self.lb_win = lookback_window
        self.scalers = scalers

        # Extract scaler params: transform x_scaled = x * scale_ + min_
        self.sc_scale = {}
        self.sc_min = {}
        for feat in self.SCALED_FEATS:
            if feat in scalers:
                self.sc_scale[feat] = float(scalers[feat].scale_[0])
                self.sc_min[feat] = float(scalers[feat].min_[0])

        # Order-feature inverse-transform params (for decode)
        self.of_scale = np.ones(N_ORDER_FEAT, dtype=np.float64)
        self.of_min = np.zeros(N_ORDER_FEAT, dtype=np.float64)
        for i, feat in enumerate(ORDER_FEATURES):
            if feat in scalers:
                self.of_scale[i] = float(scalers[feat].scale_[0])
                self.of_min[i] = float(scalers[feat].min_[0])

        # Pre-compute default order features (normalized) for conditioning init
        self._default_order = self._make_default_order_row()

        # ---- Per-sample buffers ----
        # Pre-normalized lookback: (N, lookback_window, n_features)
        self.lookback = np.zeros((n_samples, lookback_window, N_ALL_FEAT), dtype=np.float32)
        # Current row (latest state, may overwrite last lookback slot)
        self.current_row = np.zeros((n_samples, N_ALL_FEAT), dtype=np.float32)

        # Discrete time tracking
        self.cur_dtime = np.zeros(n_samples, dtype=np.int64)

        # Book state (raw)
        self.best_bid = np.zeros(n_samples, dtype=np.float64)
        self.best_ask = np.zeros(n_samples, dtype=np.float64)
        self.spread = np.zeros(n_samples, dtype=np.float64)
        self.midprice = np.zeros(n_samples, dtype=np.float64)

        # Midprice history for returns (circular, per-sample)
        self.mid_hist = np.zeros((n_samples, 50), dtype=np.float64)
        self.mid_ptr = np.zeros(n_samples, dtype=np.int32)
        self.mid_cnt = np.zeros(n_samples, dtype=np.int32)

        # MO buffer for exec imbalance (circular, per-sample)
        self.mo_qty = np.zeros((n_samples, 256), dtype=np.float64)
        self.mo_side = np.zeros((n_samples, 256), dtype=np.float64)
        self.mo_ptr = np.zeros(n_samples, dtype=np.int32)
        self.mo_cnt = np.zeros(n_samples, dtype=np.int32)

    # ------------------------------------------------------------------
    # Scaler helpers
    # ------------------------------------------------------------------
    def _scale(self, feat: str, val):
        """Apply MinMaxScaler: x_scaled = x * scale_ + min_"""
        if feat in self.sc_scale:
            return val * self.sc_scale[feat] + self.sc_min[feat]
        return val

    def _make_default_order_row(self) -> np.ndarray:
        """Default order features during conditioning (normalized)."""
        # Raw defaults: depth=0, del_depth=0, SIZE=100 (round), TYPE=0, BSF=-1
        depth_s = self._scale("depth", 0.0)
        del_depth_s = self._scale("del_depth", 0.0)
        # SIZE=100 is round → SIZE_TYPE=-1, SIZE100s=1.0, SIZE=ffill→100.0
        size_s = self._scale("SIZE", 100.0)
        size100s_s = self._scale("SIZE100s", 1.0)
        size_type = -1.0
        type_val = 0.0
        bsf = -1.0
        return np.array([depth_s, del_depth_s, size_s, size100s_s,
                         size_type, type_val, bsf], dtype=np.float32)

    # ------------------------------------------------------------------
    # State feature computation
    # ------------------------------------------------------------------
    def _compute_state_feats(self, idx: int, l2: np.ndarray,
                             midprice: float, spread: float) -> np.ndarray:
        """Compute normalized state features (10,) from L2 for one sample."""
        n_lv = min(len(l2) // 4, 10)
        ask_v = np.array([float(l2[4 * k + 1]) for k in range(n_lv)])
        bid_v = np.array([float(l2[4 * k + 3]) for k in range(n_lv)])

        bv1, av1 = bid_v[0], ask_v[0]
        imb1 = bv1 / (bv1 + av1) if (bv1 + av1) > 0 else 0.5

        d5 = min(5, n_lv)
        bv5, av5 = bid_v[:d5].sum(), ask_v[:d5].sum()
        imb5 = bv5 / (bv5 + av5) if (bv5 + av5) > 0 else 0.5

        vols1 = bv1 + av1
        vols5 = bv5 + av5

        # Returns
        ret1 = ret50 = 0.0
        if self.mid_cnt[idx] > 0:
            last_p = (self.mid_ptr[idx] - 1) % 50
            if self.mid_hist[idx, last_p] > 0:
                ret1 = midprice / self.mid_hist[idx, last_p] - 1
            first_p = (self.mid_ptr[idx] - min(int(self.mid_cnt[idx]), 50)) % 50
            if self.mid_hist[idx, first_p] > 0:
                ret50 = midprice / self.mid_hist[idx, first_p] - 1

        # Exec imbalance
        ei128 = self._exec_imb(idx, 128)
        ei256 = self._exec_imb(idx, 256)

        return np.array([
            imb1, imb5, ei128, ei256,
            self._scale("vols_1", vols1),
            self._scale("vols_5", vols5),
            self._scale("spread", spread),
            ret1, ret50, midprice,
        ], dtype=np.float32)

    def _exec_imb(self, idx: int, n_ticks: int) -> float:
        """Execution imbalance over last n_ticks MOs for one sample."""
        cnt = min(int(self.mo_cnt[idx]), n_ticks)
        if cnt == 0:
            return 0.5
        ptr = int(self.mo_ptr[idx])
        if ptr >= cnt:
            sl = slice(ptr - cnt, ptr)
            q, s = self.mo_qty[idx, sl], self.mo_side[idx, sl]
        else:
            part1 = 256 - (cnt - ptr)
            q = np.concatenate([self.mo_qty[idx, part1:256], self.mo_qty[idx, :ptr]])
            s = np.concatenate([self.mo_side[idx, part1:256], self.mo_side[idx, :ptr]])
        bid_vol = np.sum(q * (s == -1))
        total = np.sum(q)
        return bid_vol / total if total > 0 else 0.5

    def _push_midprice(self, idx: int, mp: float):
        p = self.mid_ptr[idx] % 50
        self.mid_hist[idx, p] = mp
        self.mid_ptr[idx] = (p + 1) % 50
        self.mid_cnt[idx] = min(self.mid_cnt[idx] + 1, 50)

    def _push_mo(self, idx: int, qty: float, side: float):
        p = self.mo_ptr[idx] % 256
        self.mo_qty[idx, p] = qty
        self.mo_side[idx, p] = side
        self.mo_ptr[idx] = (p + 1) % 256
        self.mo_cnt[idx] = min(self.mo_cnt[idx] + 1, 256)

    # ------------------------------------------------------------------
    # Lookback management
    # ------------------------------------------------------------------
    def _advance_lookback(self, idx: int, new_row: np.ndarray, new_dtime: int):
        """Push new_row into lookback for sample idx, handling time gaps."""
        old_dtime = self.cur_dtime[idx]
        if new_dtime <= old_dtime:
            return
        k = max(1, (new_dtime - old_dtime) // TIME_SCALER)
        k = min(k, self.lb_win)

        if k < self.lb_win:
            prev = self.lookback[idx, -1].copy()
            self.lookback[idx, :-k] = self.lookback[idx, k:]
            if k > 1:
                self.lookback[idx, -k:-1] = prev  # forward-fill gaps
            self.lookback[idx, -1] = new_row
        else:
            # Gap larger than window — fill with new_row
            self.lookback[idx, :] = new_row

        self.cur_dtime[idx] = new_dtime

    # ------------------------------------------------------------------
    # Initialization from conditioning data
    # ------------------------------------------------------------------
    def init_single(self, idx: int, l2_states: np.ndarray,
                    timestamps_s: np.ndarray, timestamps_ns: np.ndarray,
                    msgs_raw: np.ndarray, tick_size: int):
        """Initialize tracker for one sample from conditioning messages."""
        n_msgs = len(timestamps_s)

        # Build MO buffer from conditioning
        for j in range(n_msgs):
            if int(msgs_raw[j, EVENT_TYPE_i]) == 4:
                direction = int(msgs_raw[j, DIRECTION_i])
                size = int(msgs_raw[j, SIZE_i])
                self._push_mo(idx, float(size), -1.0 if direction == 0 else 1.0)

        # Process L2 states
        for j in range(n_msgs):
            l2 = l2_states[j + 1]  # l2[0] is before first msg
            ts_ns = int(timestamps_s[j]) * int(1e9) + int(timestamps_ns[j])

            ba = float(l2[0])
            bb = float(l2[2])
            if ba <= 0 or bb <= 0:
                continue

            sp = ba - bb
            mp = (ba + bb) / 2.0

            state_feats = self._compute_state_feats(idx, l2, mp, sp)
            self._push_midprice(idx, mp)

            row = np.concatenate([self._default_order, state_feats])

            self.best_bid[idx] = bb
            self.best_ask[idx] = ba
            self.spread[idx] = sp
            self.midprice[idx] = mp

            dtime = int(TIME_SCALER * ((ts_ns / 1e9) // TIME_SCALER))
            if dtime > self.cur_dtime[idx]:
                self._advance_lookback(idx, row, dtime)

            self.current_row[idx] = row

    # ------------------------------------------------------------------
    # Build CGAN input tensor
    # ------------------------------------------------------------------
    def build_batch_tensor(self) -> torch.Tensor:
        """Build (N, lookback_window, n_features) tensor for batched CGAN."""
        result = self.lookback.copy()
        result[:, -1, :] = self.current_row
        return torch.from_numpy(result).float()

    # ------------------------------------------------------------------
    # Vectorized decode of CGAN output
    # ------------------------------------------------------------------
    def decode_batch(self, y_hat: np.ndarray, l2s: np.ndarray):
        """
        Decode CGAN output for N samples.

        Args:
            y_hat: (N, 7) normalized order features from CGAN
            l2s:   (N, n_levels*4) current L2 book states

        Returns:
            event_types (N,), directions (N,), quantities (N,), prices (N,) — all int32
        """
        N = y_hat.shape[0]

        # Inverse-scale order features
        y_raw = (y_hat - self.of_min[None, :]) / self.of_scale[None, :]

        # SIZE reconstruction
        size_type_raw = y_raw[:, OFI["SIZE_TYPE"]]
        raw_size = np.where(
            size_type_raw > 0,
            y_raw[:, OFI["SIZE"]],
            y_raw[:, OFI["SIZE100s"]] * 100,
        )
        quantities = np.maximum(1, np.round(raw_size)).astype(np.int32)

        # TYPE → event_type: ≤-0.33=cancel(3), ≥0.33=limit(1), else=market(4)
        type_val = y_raw[:, OFI["TYPE"]]
        event_types = np.where(type_val <= -0.33, 3,
                      np.where(type_val >= 0.33, 1, 4)).astype(np.int32)

        # BUY_SELL_FLAG → actor intent (ABIDES convention: ≤0 = buying, >0 = selling)
        is_buy = y_raw[:, OFI["BUY_SELL_FLAG"]] <= 0

        # Direction for JAX-LOB (book-side convention):
        #   Limit/cancel: buy → bid side (1), sell → ask side (0)
        #   Market/exec:  buy → ask side (0), sell → bid side (1) [LOBSTER passive-side]
        directions = np.where(
            event_types == 4,
            np.where(is_buy, 0, 1),   # market: buy→0(ask hit), sell→1(bid hit)
            np.where(is_buy, 1, 0),   # limit/cancel: buy→1(bid), sell→0(ask)
        ).astype(np.int32)

        # Prices (use is_buy for actor-centric price logic)
        best_bid = l2s[:, 2].astype(np.float64)
        best_ask = l2s[:, 0].astype(np.float64)
        near_touch = np.where(is_buy, best_bid, best_ask)
        sign = np.where(is_buy, -1.0, 1.0)

        # Limit price from depth (clamped by spread)
        depth = y_raw[:, OFI["depth"]]
        min_depth = -(self.spread - 1)
        depth_clamped = np.maximum(depth, min_depth)
        limit_price = (near_touch + sign * depth_clamped).astype(np.int64)

        # Market price (opposing side best)
        market_price = np.where(is_buy, best_ask, best_bid).astype(np.int64)

        # Cancel price from del_depth
        del_depth = np.maximum(y_raw[:, OFI["del_depth"]], 0)
        cancel_price = (near_touch + sign * del_depth).astype(np.int64)

        prices = np.where(event_types == 1, limit_price,
                 np.where(event_types == 4, market_price,
                          cancel_price))
        prices = np.maximum(0, prices).astype(np.int32)

        return event_types, directions, quantities, prices

    # ------------------------------------------------------------------
    # Batch update after CGAN-generated orders
    # ------------------------------------------------------------------
    def update_batch_generated(self, event_types: np.ndarray,
                               directions: np.ndarray,
                               quantities: np.ndarray,
                               l2s: np.ndarray,
                               times_ns: np.ndarray,
                               y_hat_order: np.ndarray):
        """
        Update all N trackers after CGAN-generated orders are applied to books.

        Args:
            event_types:  (N,) int — 1=limit, 3=cancel, 4=market
            directions:   (N,) int — JAX-LOB book-side convention:
                          limit/cancel: 1=bid(buy), 0=ask(sell)
                          market: 0=ask-hit(buy), 1=bid-hit(sell)
            quantities:   (N,) int
            l2s:          (N, n_levels*4) — L2 book states AFTER order applied
            times_ns:     (N,) int64 — timestamp in nanoseconds
            y_hat_order:  (N, 7) — raw CGAN output (normalized order features)
        """
        for i in range(self.N):
            ba = float(l2s[i, 0])
            bb = float(l2s[i, 2])
            if ba <= 0 or bb <= 0:
                continue

            sp = ba - bb
            mp = (ba + bb) / 2.0

            # Update MO buffer
            if event_types[i] == 4:
                side = -1.0 if directions[i] == 0 else 1.0
                self._push_mo(i, float(quantities[i]), side)

            # State features
            state_feats = self._compute_state_feats(i, l2s[i], mp, sp)
            self._push_midprice(i, mp)

            # Order features: use CGAN output directly (already in normalized space)
            order_feats = y_hat_order[i].astype(np.float32)

            row = np.concatenate([order_feats, state_feats])

            self.best_bid[i] = bb
            self.best_ask[i] = ba
            self.spread[i] = sp
            self.midprice[i] = mp

            dtime = int(TIME_SCALER * ((times_ns[i] / 1e9) // TIME_SCALER))
            if dtime > self.cur_dtime[i]:
                self._advance_lookback(i, row, dtime)
            self.current_row[i] = row

    def update_batch_aggressive(self, direction: int, l2s: np.ndarray,
                                times_ns: np.ndarray, quantities: np.ndarray):
        """Update all N trackers after aggressive order injection."""
        for i in range(self.N):
            ba = float(l2s[i, 0])
            bb = float(l2s[i, 2])
            if ba <= 0 or bb <= 0:
                continue

            sp = ba - bb
            mp = (ba + bb) / 2.0

            side = -1.0 if direction == 0 else 1.0
            self._push_mo(i, float(quantities[i]), side)

            state_feats = self._compute_state_feats(i, l2s[i], mp, sp)
            self._push_midprice(i, mp)

            # Order features for aggressive MO (normalized)
            depth_s = self._scale("depth", float(-sp))  # MO depth = negative spread
            del_depth_s = self._scale("del_depth", 0.0)
            size_val = float(quantities[i])
            if size_val % 100 == 0:
                size_s = self._scale("SIZE", 100.0)  # ffill default
                size100s_s = self._scale("SIZE100s", size_val / 100.0)
                size_type = -1.0
            else:
                size_s = self._scale("SIZE", size_val)
                size100s_s = self._scale("SIZE100s", 1.0)  # ffill default
                size_type = 1.0
            type_val = 0.0  # market order
            bsf = -1.0 if direction == 0 else 1.0

            order_feats = np.array([depth_s, del_depth_s, size_s, size100s_s,
                                    size_type, type_val, bsf], dtype=np.float32)
            row = np.concatenate([order_feats, state_feats])

            self.best_bid[i] = bb
            self.best_ask[i] = ba
            self.spread[i] = sp
            self.midprice[i] = mp

            dtime = int(TIME_SCALER * ((times_ns[i] / 1e9) // TIME_SCALER))
            if dtime > self.cur_dtime[i]:
                self._advance_lookback(i, row, dtime)
            self.current_row[i] = row


# ============================================================================
# Main scenario function (v2 — batched)
# ============================================================================
def run_cgan_scenario(cfg: Dict[str, Any], save_folder: Path):
    """
    Optimized CGAN aggressive scenario with batched inference.

    vs v1: CGAN forward pass is batched across batch_size samples.
    JAX-LOB operations remain per-sample (sequential but JIT-compiled).
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
    n_levels = cfg.get('n_levels', 10)
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
    total_msgs_per_sample = total_blocks * n_gen_msgs + num_insertions

    # 1. Load CGAN model
    print(f"Loading CGAN model from {cgan_checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})
    cgan_model = ganmodels.LOBGAN.load_from_checkpoint(
        cgan_checkpoint, map_location=device
    )
    torch.load = _orig_torch_load
    cgan_model = cgan_model.to(device)
    cgan_model.eval()
    print(f"CGAN model loaded on {device}")

    # Load scalers
    print(f"Loading scalers from {cgan_scalers_path}")
    with open(cgan_scalers_path, "rb") as f:
        scalers = pickle.load(f)
    print(f"Scalers loaded: {list(scalers.keys())}")

    # Load interarrival times
    interarrival_k = 1.0
    interarrival_theta = 0.05
    if cgan_interarrival_path and os.path.exists(cgan_interarrival_path):
        print(f"Loading interarrival times from {cgan_interarrival_path}")
        with open(cgan_interarrival_path, "rb") as f:
            ia_params = pickle.load(f)
        interarrival_k = ia_params["k"]
        interarrival_theta = ia_params["theta"]
        print(f"Interarrival: k={interarrival_k:.4f}, theta={interarrival_theta:.4f}")

    # 2. Initialize
    rng = jax.random.key(rng_seed)
    np_rng = np.random.RandomState(rng_seed)
    torch.manual_seed(rng_seed)

    sim = OrderBook(cfg=JAXLOB_Configuration(
        cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value
    ))

    # Create vmapped step functions (JIT compilation deferred to first call)
    step_vmap, aggr_vmap = make_step_fns(sim, n_levels)

    print(f"Loading dataset from {data_dir}")
    ds = get_dataset(data_dir, n_cond_msgs, n_eval_msgs_dataset, test_split=test_split)
    print(f"Dataset length: {len(ds)}")

    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    assert n_samples % batch_size == 0, \
        f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    sample_i = jax.random.choice(
        rng_, jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size), replace=False
    ).tolist()

    # Build insertion schedule
    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1
    print(f"Aggressive positions (0-indexed): {aggressive_positions}")
    print(f"Total messages per sample: {total_msgs_per_sample}")
    print(f"Batch size: {batch_size}, n_batches: {n_samples // batch_size}")

    # 3. Process batches
    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        N = len(batch_i)
        print(f'\n=== BATCH {batch_idx}: {N} samples ===')

        # Load data
        jax_device = jax.devices()[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), jax_device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), jax_device)
        b_seq_pv = jnp.array(b_seq_pv)

        # Split conditioning
        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs, :]
        b_seq_pv_cond = np.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])
        init_time = b_seq_pv[:, 0, 1:3]
        init_time = jax.device_put(jnp.array(init_time), jax_device)

        # Initialize JAX-LOB simulators (vmapped)
        sim_states_batched = get_sims_vmap(
            book_l2_init, m_seq_raw_cond, init_time, sim,
        )

        # Save conditioning data
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

        # Get initial L2 states (batched — keep sim_states_batched as pytree)
        l2s = np.array(jax.vmap(
            sim.get_L2_state, in_axes=(0, None)
        )(sim_states_batched, n_levels))

        # Initialize tracker
        tracker = FastCGANTracker(N, scalers, lookback_window)

        cond_msgs_np = np.array(m_seq_raw_cond)
        cond_l2_np = np.array(b_seq_pv_cond)

        for i in range(N):
            tracker.init_single(
                i, cond_l2_np[i],
                cond_msgs_np[i, :, TIMEs_i],
                cond_msgs_np[i, :, TIMEns_i],
                cond_msgs_np[i], tick_size,
            )

        # Per-sample RNGs (stacked array for vmap) and time tracking
        rng, rng_batch = jax.random.split(rng)
        rngs = jax.random.split(rng_batch, N)  # (N,) stacked keys

        times_total_ns = np.array([
            int(cond_msgs_np[i, -1, TIMEs_i]) * int(1e9) +
            int(cond_msgs_np[i, -1, TIMEns_i])
            for i in range(N)
        ], dtype=np.int64)

        oid_counters = np.full(N, START_OID, dtype=np.int32)

        # Pre-allocate output arrays
        all_msgs = np.zeros((N, total_msgs_per_sample, 14), dtype=np.int32)
        all_books = np.zeros((N, total_msgs_per_sample, n_levels * 4), dtype=np.float32)

        msg_out_idx = 0

        for block in range(total_blocks):
            for msg_idx in range(n_gen_msgs):
                # 1. Sample interarrival times (N samples)
                dt = np_rng.gamma(interarrival_k, interarrival_theta, size=N)
                times_total_ns += (dt * 1e9).astype(np.int64)
                times_s = (times_total_ns // int(1e9)).astype(np.int32)
                times_ns = (times_total_ns % int(1e9)).astype(np.int32)

                # 2. Build batched CGAN input
                gan_input = tracker.build_batch_tensor()
                if device.type == 'cuda':
                    gan_input = gan_input.to(device)

                # 3. Batched CGAN forward pass (single call for N samples)
                with torch.no_grad():
                    y_hat = cgan_model((gan_input, None, None, 1))
                    y_hat = y_hat.detach().cpu().numpy()[:, 0, :]  # (N, 7)

                # 4. Decode to actions (vectorized numpy)
                event_types, directions, quantities, prices = \
                    tracker.decode_batch(y_hat, l2s)

                # 5. Batched JAX-LOB (vmapped — all N samples in one GPU call)
                sim_msgs = jnp.array(np.column_stack([
                    event_types,
                    (directions * 2) - 1,
                    quantities,
                    prices,
                    oid_counters,
                    np.full(N, -88, dtype=np.int32),
                    times_s,
                    times_ns,
                ]).astype(np.int32))

                sim_msgs, rngs, sim_states_batched, l2s_jax = step_vmap(
                    sim_msgs, rngs, sim_states_batched
                )
                l2s = np.array(l2s_jax)
                actual_oids = np.array(sim_msgs[:, 4])

                # 6. Build decoded messages (vectorized)
                ba = l2s[:, 0].astype(np.float64)
                bb = l2s[:, 2].astype(np.float64)
                mid = (ba + bb) / 2.0
                mid = (mid // tick_size) * tick_size
                rel_price = np.where(mid > 0,
                    (prices.astype(np.float64) - mid) // tick_size, 0
                ).astype(np.int32)

                all_msgs[:, msg_out_idx, 0] = actual_oids
                all_msgs[:, msg_out_idx, 1] = event_types
                all_msgs[:, msg_out_idx, 2] = directions
                all_msgs[:, msg_out_idx, 3] = prices
                all_msgs[:, msg_out_idx, 4] = rel_price
                all_msgs[:, msg_out_idx, 5] = quantities
                all_msgs[:, msg_out_idx, 8] = times_s
                all_msgs[:, msg_out_idx, 9] = times_ns
                all_books[:, msg_out_idx, :] = l2s

                # 7. Update tracker
                tracker.update_batch_generated(
                    event_types, directions, quantities,
                    l2s, times_total_ns, y_hat,
                )

                oid_counters += 1
                msg_out_idx += 1

            # Inject aggressive order after insertion blocks (vmapped)
            if block < num_insertions:
                last_s_arr = jnp.array(times_s, dtype=jnp.int32)
                last_ns_arr = jnp.array(times_ns, dtype=jnp.int32)

                sim_states_batched, l2s_jax, msgs_decoded = aggr_vmap(
                    sim_states_batched, last_s_arr, last_ns_arr,
                    tick_size, event_type, direction, order_volume,
                )
                l2s = np.array(l2s_jax)
                msgs_decoded_np = np.array(msgs_decoded)

                all_msgs[:, msg_out_idx, :] = msgs_decoded_np
                all_books[:, msg_out_idx, :] = l2s
                times_total_ns = (
                    msgs_decoded_np[:, TIMEs_i].astype(np.int64) * int(1e9) +
                    msgs_decoded_np[:, TIMEns_i].astype(np.int64)
                )

                tracker.update_batch_aggressive(
                    direction, l2s, times_total_ns,
                    msgs_decoded_np[:, SIZE_i].astype(np.int32),
                )
                msg_out_idx += 1

        # Save generated data
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(jnp.array(all_msgs[i])).to_csv(
                save_folder / 'data_gen' /
                f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )
            book_to_lobster_format(jnp.array(all_books[i])).to_csv(
                save_folder / 'data_gen' /
                f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )

        if batch_idx == 0:
            print(f"  First batch: {msg_out_idx} msgs/sample "
                  f"({total_blocks} blocks x {n_gen_msgs} + {num_insertions} aggressive)")
            aggressive_indices = np.array(aggressive_positions)
            np.savetxt(save_folder / 'aggressive_indices.csv',
                       aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="CGAN (Coletta) Aggressive Scenario v2 (optimized)"
    )
    parser.add_argument(
        '--config', '-c', type=str,
        default='lob_impact/3_scenarios/5.aggressive_scenario_cgan_config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument('--n_gen_msgs', type=int, default=None,
                        help='Override n_gen_msgs from config')
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1],
                        help='Override direction (0=buy, 1=sell)')
    return parser.parse_args()


def main():
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")

    args = parse_args()

    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs
    if args.direction is not None:
        cfg['direction'] = args.direction

    print(f"Configuration: {cfg}")

    save_folder = create_experiment_folder(cfg['save_dir'])
    print(f"Experiment folder: {save_folder}")

    logger = setup_logging(save_folder)
    print(f"\n{'='*60}")
    print(f"CGAN (Coletta) Aggressive Scenario v2 (optimized)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"Configuration: {cfg}")
    print(f"Experiment folder: {save_folder}")

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
