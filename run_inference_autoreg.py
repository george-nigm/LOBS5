#!/usr/bin/env python
"""
Autoregressive LOB generation with YAML config, WandB logging.
Based on run_inference.py (autoreg generation) + aggressive_scenario.py (config/logging/output format).

No pmap (single GPU), no aggressive insertions, conditional generation only.
"""
import os
import sys
import argparse
import yaml
import json

# Memory management (same as run_inference.py)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import wandb
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from pathlib import Path
from datetime import datetime
from glob import glob
from tqdm import tqdm
import logging

# Suppress Orbax missing metrics warnings
logging.getLogger('absl').setLevel(logging.CRITICAL)

# Add parent folder to path
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(parent_folder_path)

# Add AlphaTrade submodule to path
sys.path.append(os.path.join(parent_folder_path, 'AlphaTrade'))

from gymnax_exchange.jaxob.jorderbook import OrderBook
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.encoding import Vocab, Message_Tokenizer
from lob import inference_no_errcorr as inference
from lob.init_train import init_train_state, load_checkpoint, load_metadata
import lob.validation_helpers as valh
import lob.evaluation as eval
import preproc
from preproc import transform_L2_state

logger = logging.getLogger(__name__)


def create_next_experiment_folder(save_folder: str) -> Path:
    """Create next numbered experiment folder with timestamp."""
    base = Path(save_folder)
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


def batched_get_safe_mid_price(
        sim: OrderBook,
        sim_states,
        tick_size: int,
    ) -> jax.Array:
    """
    Batched version of _get_safe_mid_price.
    Uses vmap to apply _get_safe_mid_price across batch dimension.
    """
    def _get_safe_mid_price_single(sim_state):
        ask = sim.get_best_ask(sim_state)
        bid = sim.get_best_bid(sim_state)
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
        p_mid = (p_mid // tick_size) * tick_size
        return p_mid

    return jax.vmap(_get_safe_mid_price_single)(sim_states)


def track_midprices_during_messages(
        m_seq_raw_inp: jax.Array,
        book_l2_init: jax.Array,
        tick_size: int,
        step_size: int,
    ) -> jax.Array:
    """
    JIT-friendly, loop-free version that computes mid-prices every
    `step_size` messages using jax.lax.scan.

    Args:
        m_seq_raw_inp:  (batch, T, msg_dim) – raw decoded messages
        book_l2_init:   (batch, book_dim) – initial L2 state of the orderbook
        tick_size:      Tick size of the instrument
        step_size:      Interval (in number of messages) between successive mid-price samples

    Returns:
        midprices: (num_steps, batch) array of mid-prices
    """
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
        # carry: current book L2 state (batch, book_dim)
        sim_init, sim_states = inference.get_sims_vmap(carry, msg_chunk)
        mid_price = batched_get_safe_mid_price(
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


def l2_to_volume_image(l2_states, n_vol_series, tick_size):
    """
    Convert raw L2 states (40-dim) to volume image format (n_vol_series-dim).

    L2 format: [ask_p_1, ask_v_1, bid_p_1, bid_v_1, ...] for 10 levels
    Output: volume at each price level, centered on mid price
            ask volumes are NEGATIVE, bid volumes are POSITIVE

    Args:
        l2_states: (batch, n_msgs, 40) - raw L2 states from simulator
        n_vol_series: int - number of price levels in output (e.g., 500)
        tick_size: int - tick size for price discretization

    Returns:
        vol_image: (batch, n_msgs, n_vol_series) - volume image format
    """
    batch, n_msgs, _ = l2_states.shape
    n_levels = 10

    # Reshape to (batch, n_msgs, n_levels, 4)
    l2_reshaped = l2_states.reshape(batch, n_msgs, n_levels, 4)
    ask_prices = l2_reshaped[:, :, :, 0]   # (batch, n_msgs, 10)
    ask_vols = l2_reshaped[:, :, :, 1]
    bid_prices = l2_reshaped[:, :, :, 2]
    bid_vols = l2_reshaped[:, :, :, 3]

    # Compute mid price from best levels
    mid_price = (ask_prices[:, :, 0] + bid_prices[:, :, 0]) // 2
    mid_price = (mid_price // tick_size) * tick_size  # Round to tick

    # Create volume image
    center_idx = n_vol_series // 2
    vol_image = jnp.zeros((batch, n_msgs, n_vol_series), dtype=jnp.float32)

    # Compute price indices relative to mid price
    # ask_idx[b, t, lvl] = center_idx + (ask_price - mid_price) / tick_size
    ask_rel = (ask_prices - mid_price[:, :, None]) // tick_size
    bid_rel = (bid_prices - mid_price[:, :, None]) // tick_size

    ask_idx = (center_idx + ask_rel).astype(jnp.int32)
    bid_idx = (center_idx + bid_rel).astype(jnp.int32)

    # Clip indices to valid range
    ask_idx = jnp.clip(ask_idx, 0, n_vol_series - 1)
    bid_idx = jnp.clip(bid_idx, 0, n_vol_series - 1)

    # Use scatter to place volumes (vectorized per level)
    # For each level, scatter ask volumes (negative) and bid volumes (positive)
    batch_idx = jnp.arange(batch)[:, None, None]  # (batch, 1, 1)
    msg_idx = jnp.arange(n_msgs)[None, :, None]   # (1, n_msgs, 1)

    for lvl in range(n_levels):
        # Broadcast indices for scatter
        b_idx = jnp.broadcast_to(batch_idx, (batch, n_msgs, 1)).squeeze(-1)  # (batch, n_msgs)
        m_idx = jnp.broadcast_to(msg_idx, (batch, n_msgs, 1)).squeeze(-1)    # (batch, n_msgs)

        # Scatter ask volumes (negative)
        vol_image = vol_image.at[b_idx, m_idx, ask_idx[:, :, lvl]].add(
            -ask_vols[:, :, lvl].astype(jnp.float32)
        )
        # Scatter bid volumes (positive)
        vol_image = vol_image.at[b_idx, m_idx, bid_idx[:, :, lvl]].add(
            bid_vols[:, :, lvl].astype(jnp.float32)
        )

    return vol_image


def parse_args():
    p = argparse.ArgumentParser(description="Run autoregressive LOB inference")
    p.add_argument(
        "--config", "-c",
        type=str,
        default="1_run_exp_aggresive_scenario.yaml",
        help="Path to YAML config file"
    )
    # CLI overrides for common parameters
    p.add_argument("--ckpt_path", type=str, default=None, help="Override checkpoint path from config")
    p.add_argument("--data_dir", type=str, default=None, help="Override data directory from config")
    p.add_argument("--save_folder", type=str, default=None, help="Override save folder from config")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    p.add_argument("--n_samples", type=int, default=None, help="Override n_samples from config")
    return p.parse_args()


def main():
    print(f"JAX backend platform: {jax.lib.xla_bridge.get_backend().platform}")

    args = parse_args()

    # Load YAML config
    config_path = args.config
    if not config_path.endswith('.yaml'):
        config_path = f"{config_path}.yaml"

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Check JAX devices
    jax_devices = jax.devices()
    device_summary = [f"{d.id}: {d.device_kind}" for d in jax_devices]

    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        log_message = f"Running on GPU(s): {device_summary}"
    else:
        log_message = f"Running on CPU only: {device_summary}"
    print(log_message)

    # Unpack config with CLI overrides
    # Handle /app/ paths - replace with relative paths for Docker compatibility
    save_folder = (args.save_folder or cfg["save_folder"]).replace("/app/", "./")
    batch_size = args.batch_size or cfg["batch_size"]
    n_samples = args.n_samples or cfg["n_samples"]
    n_gen_msgs = cfg["n_gen_msgs"]
    n_messages = cfg["n_messages"]
    book_dim = cfg["book_dim"]
    n_vol_series = cfg["n_vol_series"]
    sample_top_n = cfg["sample_top_n"]
    data_dir = (args.data_dir or cfg["data_dir"]).replace("/app/", "./")
    sample_all = cfg.get("sample_all", False)

    # JSON sample file support
    use_sample_file = cfg.get("use_sample_file", False)
    sample_file_path = cfg.get("sample_file_path", None)
    start_batch = cfg.get("start_batch", 0)
    end_batch = cfg.get("end_batch", -1)  # -1 means all batches

    stock = cfg["stock"]
    tick_size = cfg["tick_size"]
    midprice_step_size = cfg.get("midprice_step_size", 1)
    rng_seed = cfg["rng_seed"]
    ckpt_path = args.ckpt_path or cfg["ckpt_path"]

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Using data_dir: {data_dir}")
    print(f"Using save_folder: {save_folder}")

    # Conditional generation settings
    # For now hardcoded to 5, later from config
    n_cond_msgs = 5  # TODO: cfg.get("n_cond_msgs", 5)

    # Create experiment folder
    exp_folder = create_next_experiment_folder(save_folder)
    print(f"Experiment dir: {exp_folder}")

    # Initialize WandB (offline mode if no API key, or use WANDB_API_KEY env var)
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb.init(
        project="LOB_Autoreg_Generation",
        entity="george-nigm",
        config=cfg,
        mode=wandb_mode,
    )
    wandb.run.summary["experiment_dir"] = str(exp_folder)

    # Save used config
    with open(exp_folder / "used_config.yaml", "w") as f_out:
        yaml.dump(cfg, f_out)

    # Log config as artifact
    artifact = wandb.Artifact(name="used_config", type="config")
    artifact.add_file(str(exp_folder / "used_config.yaml"))
    wandb.log_artifact(artifact)

    # Setup logging to experiment folder
    log_file_path = exp_folder / "job.log"
    print(f"Redirecting output to: {log_file_path}")

    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"Experiment started at: {datetime.now()}")
    print(f"Experiment folder: {exp_folder}")
    print(f"Configuration: {cfg}")
    print("=" * 80)

    # Create output folders (same structure as aggressive_scenario.py)
    (exp_folder / 'msgs_decoded_doubled').mkdir(exist_ok=True, parents=True)
    (exp_folder / 'b_seq_gen_doubled').mkdir(exist_ok=True, parents=True)
    (exp_folder / 'mid_price').mkdir(exist_ok=True, parents=True)

    # Load metadata and model
    print(f"Loading metadata from {ckpt_path}")
    args_ckpt = load_metadata(ckpt_path)
    args_ckpt.num_devices = 1
    args_ckpt.bsz = 1

    # Get token_mode from checkpoint metadata
    token_mode = getattr(args_ckpt, 'token_mode', 22)
    print(f"Using token_mode={token_mode} from checkpoint metadata")
    v = Vocab(token_mode=token_mode)
    n_classes = len(v)
    print(f"Vocab size: {n_classes}")

    # Override book_dim based on checkpoint type
    # twilight-sound-77 uses book_dim=503 (autoreg), denim-elevator-754 uses book_dim=501
    # book_dim = n_vol_series + 3 (price_change + time_s + time_ns)
    # For autoreg models with book_transform, we need 503
    ckpt_book_dim = getattr(args_ckpt, 'book_dim', None)
    if ckpt_book_dim is None:
        # Compute from n_vol_series if book_dim not in metadata
        ckpt_book_dim = n_vol_series + 3  # 500 + 3 = 503
        print(f"book_dim not in checkpoint metadata, computing from n_vol_series: {ckpt_book_dim}")
    if ckpt_book_dim != book_dim:
        print(f"Overriding book_dim from config ({book_dim}) with computed value ({ckpt_book_dim})")
        book_dim = ckpt_book_dim

    # Calculate sequence lengths
    seq_len = n_messages * Message_Tokenizer.MSG_LEN
    cond_seq_len = n_cond_msgs * Message_Tokenizer.MSG_LEN

    # Initialize model
    print("Initializing model...")
    train_state, model_cls, total_params = init_train_state(
        args_ckpt,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=n_messages,
        train_size=1,  # dummy for inference
    )
    print(f"Model parameters: {total_params:,}")

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = load_checkpoint(train_state, ckpt_path, train=False)
    state = ckpt["model"]
    model = model_cls(training=False, step_rescale=1.0)

    # Prepare RNG
    rng = jax.random.PRNGKey(rng_seed)

    # Load dataset
    data_path = Path(data_dir) / stock
    print(f"Data directory: {data_path}")

    ds = inference.get_dataset(
        data_path,
        n_cond_msgs,
        n_gen_msgs,
        token_mode=token_mode,
    )
    print(f"Dataset size: {len(ds)}")
    wandb.log({"dataset_size": len(ds)})

    # Prepare batch indices (3 modes: sample_file > sample_all > random)
    rng, rng_ = jax.random.split(rng)
    if use_sample_file:
        # Mode 1: Load from JSON file
        assert sample_file_path is not None, "use_sample_file=True but sample_file_path not provided"
        print(f"Loading sample indices from: {sample_file_path}")
        with open(sample_file_path, "r") as f:
            sample_i_full = json.load(f)
        # Slice by start_batch:end_batch
        sample_i = sample_i_full[start_batch: end_batch if end_batch != -1 else None]
        print(f"Using batches {start_batch} to {end_batch if end_batch != -1 else len(sample_i_full)} "
              f"({len(sample_i)} batches of {len(sample_i[0])} samples)")
    elif sample_all:
        # Mode 2: Use all samples sequentially
        sample_i = jnp.arange(
            len(ds) // batch_size * batch_size,
            dtype=jnp.int32
        ).reshape(-1, batch_size).tolist()
    else:
        # Mode 3: Random sampling
        assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'
        sample_i = jax.random.choice(
            rng_,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(n_samples // batch_size, batch_size),
            replace=False
        ).tolist()

    print(f"Processing {len(sample_i)} batches of size {batch_size}")

    # Transform function for book data
    transform_L2_state_batch = jax.jit(
        jax.vmap(transform_L2_state, in_axes=(0, None, None)),
        static_argnums=(1, 2)
    )

    # Initialize hidden state
    init_hidden = model.initialize_carry(
        1,
        hidden_size=(args_ckpt.ssm_size_base // pow(2, int(args_ckpt.conj_sym))),
        n_message_layers=args_ckpt.n_message_layers,
        n_book_pre_layers=args_ckpt.n_book_pre_layers,
        n_book_post_layers=args_ckpt.n_book_post_layers,
        n_fused_layers=args_ckpt.n_layers,
        h_size_ema=args_ckpt.ssm_size_base
    )
    init_hidden_batched = jax.tree_util.tree_map(
        lambda x: jnp.resize(x, (batch_size,) + x.shape),
        init_hidden
    )

    # Initialize order book simulator
    from gymnax_exchange.jaxob.jaxob_config import Configuration as JAXLOB_Configuration
    import gymnax_exchange.jaxob.jaxob_constants as cst
    sim_init = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    # Main generation loop
    for batch_idx, batch_i in enumerate(tqdm(sample_i)):
        print(f'\nBatch {batch_idx}: indices {batch_i}')

        # Load batch data
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        m_seq = jnp.array(m_seq)
        b_seq_pv = jnp.array(b_seq_pv)
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = jnp.array(book_l2_init)

        # Transform book to volume image representation
        b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)

        # Split into conditional and evaluation parts
        m_seq_inp = m_seq[:, :cond_seq_len + 1]
        b_seq_inp = b_seq[:, :n_cond_msgs + 1]
        m_seq_raw_inp = msg_seq_raw[:, :n_cond_msgs + 1]

        # Get initial simulator state
        sim_state = jax.vmap(sim_init.reset)(book_l2_init)

        # Create RNG keys for batch
        rng, rng_ = jax.random.split(rng)
        rng_keys = jax.random.split(rng_, batch_size)

        # Get initial time from book data
        init_time_batched = b_seq_pv[:, 0, 1:3]  # [time_s, time_ns]

        # Run autoregressive generation
        # generate_batched signature: sim, train_state, model, batchnorm, encoder,
        #   sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo,
        #   sim_state, rng, init_hidden, conditional, init_time,
        #   debug_book, b_seq_real, token_mode
        conditional = True  # Always conditional generation
        debug_book = False
        real_book = None  # Not used for generation

        # generate_batched returns: msgs_decoded, l2_book_states, num_errors, msgs_tokens
        msgs_decoded, l2_book_states, num_errors, msgs_tokens = inference.generate_batched(
            sim_init,           # 0: sim (static)
            state,              # 1: train_state
            model,              # 2: model (static)
            args_ckpt.batchnorm,# 3: batchnorm (static)
            v.ENCODING,         # 4: encoder
            sample_top_n,       # 5: sample_top_n (static)
            tick_size,          # 6: tick_size (static)
            m_seq_inp,          # 7: m_seq_cond (batched)
            b_seq_inp,          # 8: b_seq_cond (batched)
            n_gen_msgs,         # 9: n_msg_todo (static)
            sim_state,          # 10: sim_state (batched)
            rng_keys,           # 11: rng (batched)
            init_hidden_batched,# 12: init_hidden (batched)
            conditional,        # 13: conditional (static)
            init_time_batched,  # 14: init_time (batched)
            debug_book,         # 15: debug_book (static)
            real_book,          # 16: b_seq_real (batched, but None)
            token_mode,         # 17: token_mode (static)
        )

        print(f'Generated {msgs_decoded.shape[1]} messages, errors: {num_errors.sum()}')

        # Convert to numpy for saving
        # Note: generate_batched returns (msgs_decoded, l2_book_states, num_errors, msgs_tokens)
        # l2_book_states is the generated book sequence
        msgs_decoded_np = np.array(jax.device_get(msgs_decoded))
        l2_book_states_np = np.array(jax.device_get(l2_book_states))

        # Transform raw L2 states (40-dim) to volume image format (n_vol_series-dim)
        # This is required for visualization which expects indices like [210:293]
        b_seq_gen_transformed = l2_to_volume_image(l2_book_states, n_vol_series, tick_size)
        b_seq_gen_np = np.array(jax.device_get(b_seq_gen_transformed))

        # Calculate mid-prices from L2 book states
        # L2 format: [ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...]
        # Shape of l2_book_states: (batch, n_msgs, L2_dim)
        ask_prices = l2_book_states[:, :, 0]  # Best ask price
        bid_prices = l2_book_states[:, :, 2]  # Best bid price

        # Handle invalid prices (-1 or 0 means no orders on that side)
        # Use the other side + tick_size when one side is invalid
        ask_valid = ask_prices > 0
        bid_valid = bid_prices > 0

        # Compute mid-price with fallbacks for missing sides
        mid_price = jnp.where(
            ask_valid & bid_valid,
            (ask_prices + bid_prices) // 2,  # Normal case
            jnp.where(
                ask_valid,
                ask_prices - tick_size,  # Only ask valid
                jnp.where(
                    bid_valid,
                    bid_prices + tick_size,  # Only bid valid
                    0  # Neither valid
                )
            )
        )
        # Round to tick
        mid_price = (mid_price // tick_size) * tick_size

        # Transpose to (n_msgs, batch) to match aggressive_scenario.py format
        mid_price_np = np.array(jax.device_get(mid_price.T))

        # Save outputs in aggressive_scenario format
        # iteration=1 since we only do one generation pass (no insertions)
        iteration = 1

        np.save(
            exp_folder / 'msgs_decoded_doubled' / f'msgs_decoded_doubled_batch_{batch_i}_iter_{iteration}.npy',
            msgs_decoded_np
        )
        np.save(
            exp_folder / 'b_seq_gen_doubled' / f'b_seq_gen_doubled_batch_{batch_i}_iter_{iteration}.npy',
            b_seq_gen_np
        )
        np.save(
            exp_folder / 'mid_price' / f'mid_price_batch_{batch_i}_iter_{iteration}.npy',
            mid_price_np
        )

        # Log batch metrics to WandB
        wandb.log({
            "batch_idx": batch_idx,
            "num_errors": int(num_errors.sum()),
            "msgs_generated": msgs_decoded.shape[1],
        })

    # Finish
    print(f"\nGeneration completed at: {datetime.now()}")
    print(f"Total batches processed: {len(sample_i)}")

    wandb.log({"finished": True})
    wandb.save(str(exp_folder / "*"))

    # Close log file and restore stdout/stderr
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print(f"Experiment completed. Logs saved to: {log_file_path}")
    print(f"Results saved to: {exp_folder}")


if __name__ == "__main__":
    main()
