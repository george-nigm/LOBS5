#!/usr/bin/env python3
"""
TWAP Scenario: Passive Limit Order Injection with S5 Model.

Instead of aggressive market orders (event_type=4), injects LIMIT ORDERS
(event_type=1) at or near the best price. This simulates a TWAP-style
institutional execution strategy that interacts passively with the book.

The TWAP agent:
1. Places a limit order at (best_ask - tick) for buy, (best_bid + tick) for sell
2. Waits for the model to generate mb messages (some may execute our order)
3. Repeats for i insertions

This tests whether the model's order flow naturally fills our passive orders
and how the price responds to persistent passive flow.

Usage:
    python -u lob_impact/6.twap_scenario_s5.py --config <config.yaml> --direction 0
"""
import argparse
import os
import sys
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from functools import partial

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import jax
import jax.numpy as jnp

# Swap encoding for v3 checkpoints
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(script_dir)
sys.path.insert(0, parent_folder_path)
import lob.encoding_24tok
sys.modules['lob.encoding'] = lob.encoding_24tok

from lob.encoding import Message_Tokenizer, Vocab
from lob import inference
from lob.inference import get_dataset
from lob.init_train import init_train_state, load_checkpoint, load_metadata
from gymnax_exchange.jaxob.JaxOrderBookArrays import OrderBook
from gymnax_exchange.jaxob.jorderbook import JAXLOB_Configuration
from gymnax_exchange.jaxob import constants as cst


def create_limit_order_message(direction, price, size, order_id=999999, tick_size=100):
    """Create a limit order message (event_type=1) for injection.

    Args:
        direction: 0=buy (place on bid side), 1=sell (place on ask side)
        price: absolute price in centidollars
        size: order size in shares
        order_id: unique order ID
        tick_size: tick size (100 = 1 cent)

    Returns:
        message array compatible with JAX-LOB process_order_array
    """
    # LOBSTER message format: [OrderID, EventType, Direction, Price, RelPrice, Size, Flag, ...]
    # EventType: 1=add limit order
    msg = np.zeros(14, dtype=np.int32)
    msg[0] = order_id       # OrderID
    msg[1] = 1              # EventType = add limit order
    msg[2] = direction      # Direction: 0=buy, 1=sell
    msg[3] = int(price)     # Price (centidollars)
    msg[4] = 0              # RelPrice (will be computed)
    msg[5] = int(size)      # Size
    msg[6] = 0              # Flag
    return msg


def get_best_prices(book_state):
    """Extract best ask and best bid from L2 book state."""
    # Book state format: [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ...]
    best_ask = float(book_state[0])
    best_bid = float(book_state[2])
    return best_ask, best_bid


def run(cfg):
    """Main TWAP scenario function."""
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
    ckpt_path = cfg['ckpt_path']
    tick_size = cfg['tick_size']
    n_vol_series = cfg['n_vol_series']
    book_dim = cfg['book_dim']
    test_split = cfg.get('test_split', 0)
    direction = cfg['direction']  # 0=buy, 1=sell
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)
    order_volume = cfg['order_volume']
    checkpoint_step = cfg.get('checkpoint_step', None)
    chunk_size = cfg.get('chunk_size', 1)
    # TWAP-specific
    price_offset_ticks = cfg.get('price_offset_ticks', 1)  # how many ticks inside the spread

    # Derived
    total_gen_msgs = (num_insertions + num_coolings) * n_gen_msgs
    n_msg_todo_total = total_gen_msgs + num_insertions
    cond_seq_len = n_cond_msgs * Message_Tokenizer.MSG_LEN

    v = Vocab()
    n_classes = len(v)
    rng = jax.random.key(rng_seed)

    # Load model
    print(f"Loading model from {ckpt_path}")
    args = load_metadata(ckpt_path)
    args.num_devices = 1
    args.bsz = 1

    new_train_state, model_cls = init_train_state(
        args, n_classes=n_classes, seq_len=cond_seq_len,
        book_dim=book_dim, book_seq_len=n_cond_msgs,
    )

    ckpt = load_checkpoint(new_train_state, ckpt_path, step=checkpoint_step, train=False)
    print(f"Loaded checkpoint step: {ckpt['step']}")
    train_state = ckpt['model']
    model = model_cls(training=False, step_rescale=1.0)

    # Load dataset
    print(f"Loading dataset from {data_dir}")
    ds = get_dataset(data_dir, n_cond_msgs, n_eval_msgs_dataset, test_split=test_split)
    print(f"Dataset length: {len(ds)}")

    # Save dir
    save_dir = Path(cfg['save_dir'])
    save_folder = save_dir / f'exp_1_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    # Log config
    print(f"\n{'='*60}")
    print(f"TWAP Scenario Experiment")
    print(f"Started: {datetime.now()}")
    print(f"{'='*60}")
    print(f"Model: {ckpt_path}")
    print(f"Stock: {stock}, Direction: {'buy' if direction == 0 else 'sell'}")
    print(f"TWAP: {num_insertions} limit orders, {order_volume} shares each")
    print(f"Price offset: {price_offset_ticks} ticks inside spread")
    print(f"Messages between: {n_gen_msgs}, Cooling: {num_coolings} blocks")
    print(f"Samples: {n_samples}, Batch: {batch_size}")

    # Save aggressive indices (same format as other scenarios for compatibility)
    aggr_positions = []
    pos = 0
    for k in range(num_insertions):
        insert_pos = (k + 1) * n_gen_msgs + k  # after k-th block of generated msgs
        aggr_positions.append(insert_pos)
    np.savetxt(save_folder / 'aggressive_indices.csv',
               np.array(aggr_positions), fmt='%d')
    print(f"Insertion positions: {aggr_positions}")

    # Sample indices
    assert n_samples % batch_size == 0
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    sample_i = jax.random.choice(
        rng_, jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size), replace=False
    ).tolist()

    # Initialize simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))
    process_msg_vmap = jax.jit(jax.vmap(sim.process_order_array, in_axes=(0, 0)))
    get_L2_vmap = jax.jit(jax.vmap(sim.get_L2_state, in_axes=(0, None)), static_argnums=(1,))

    # Tokenizer
    tokenizer = Message_Tokenizer()
    encode_msg_jit = jax.jit(jax.vmap(tokenizer.encode_msg))

    # Generation function (reuse from inference)
    from lob.inference_no_errcorr_w_insertions import generate_msgs_w_insertions

    print(f"\nProcessing {len(sample_i)} batches...")
    from tqdm import tqdm

    for batch_idx, batch_samples in enumerate(tqdm(sample_i, desc='Batches')):
        print(f'\n=== BATCH {batch_idx+1}: samples {batch_samples} ===')

        for j, sample_idx in enumerate(batch_samples):
            # Load conditioning data
            cond_data = ds[sample_idx]
            cond_msgs = cond_data['messages'][:n_cond_msgs]
            cond_books = cond_data['book_states'][:n_cond_msgs]

            # Initialize book from conditioning
            init_book = cond_books[0]

            # Process conditioning through model to get hidden state
            cond_tokens = tokenizer.encode_batch(cond_msgs)

            # TODO: Full implementation requires:
            # 1. Process conditioning tokens through S5 model to get hidden state
            # 2. For each insertion:
            #    a. Generate mb messages autoregressively
            #    b. Get current book state (best ask/bid)
            #    c. Create limit order at best_ask - offset (buy) or best_bid + offset (sell)
            #    d. Inject limit order into book and model's hidden state
            #    e. Continue generation
            # 3. Save results

            # For now, use the same generation infrastructure as aggressive scenario
            # but modify the injection to be a limit order instead of market order

            # This is a STUB — the full implementation follows the same pattern
            # as 1.aggressive_scenario_s5.py but with create_limit_order_message()
            # instead of the aggressive market order creation.

            pass

    print(f"\n{'='*60}")
    print(f"TWAP Scenario — STUB COMPLETE")
    print(f"Full implementation requires integrating limit order injection")
    print(f"into the autoregressive generation loop.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='TWAP Scenario (passive limit orders)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n_gen_msgs', type=int, default=None)
    parser.add_argument('--direction', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs
    if args.direction is not None:
        cfg['direction'] = args.direction

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Loading config from: {args.config}")
    print(f"Configuration: {cfg}")

    run(cfg)


if __name__ == "__main__":
    main()
