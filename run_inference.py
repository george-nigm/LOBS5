import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['XLA_FLAGS'] ='--xla_gpu_deterministic_ops=true'


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import torch
torch.multiprocessing.set_start_method('spawn')

# Add parent folder to path (to run this file from subdirectories)
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(parent_folder_path)

# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(os.path.join(parent_folder_path, submodule_name))

print(sys.path)
from gymnax_exchange.jaxob.jorderbook import OrderBook
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

# from argparse import Namespace
from glob import glob
import numpy as onp
import pandas as pd
# from functools import partial
# from typing import Union, Optional
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# from line_profiler import LineProfiler

import jax
import jax.numpy as jnp
from jax.nn import one_hot
# from jax import random
# from jax.scipy.linalg import block_diag
# from flax import jax_utils
# from flax.training import checkpoints
# import orbax

#from lob.lob_seq_model import BatchLobPredModel
# from lob.train_helpers import create_train_state, eval_step, prep_batch, cross_entropy_loss, compute_accuracy
from s5.ssm import *
# from s5.ssm_init import make_DPLR_HiPPO
# from s5.dataloading import make_data_loader
# from lob_seq_model import LobPredModel
from lob.encoding import Vocab, Message_Tokenizer
# from lobster_dataloader import LOBSTER_Dataset, LOBSTER_Subset, LOBSTER_Sampler, LOBSTER

import preproc
from time import time
# import inference
from lob import inference_no_errcorr as inference
import lob.validation_helpers as valh
from lob.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint
# import lob.encoding as encoding


import lob.evaluation as eval
from preproc import transform_L2_state



##################################################

if __name__ == "__main__":

    # get args from command line to select stock between GOOG, INTC
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str, default='GOOG', help='stock to evaluate')
    parser.add_argument('--checkpoint_step', type=int, default=None, help='Which checkpoint step to load')
    parser.add_argument('--test_split', type=float, default=0.1, help='Which test split to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument("--n_sequences", type=int, default=1024, help="Number of sequences to generate")
    parser.add_argument("--n_cond_msgs", type=int, default=500, help="Number of conditional messages")
    parser.add_argument("--n_gen_msgs", type=int, default=500, help="Number of messages to generate")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--save_dir", type=str, default=None, help="Override save directory")
    parser.add_argument("--sample_indices_file", type=str, default=None, help="File with pre-determined sample indices")
    parser.add_argument("--wide_book_dir", type=str, default=None, help="Directory for wider L2 book files")
    parser.add_argument("--wide_levels", type=int, default=10, help="Number of book levels for simulator init")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank for multi-GPU inference")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs for inference")

    run_args = parser.parse_args()

    overfit_debug = False

    # CLI overrides take priority; fall back to hardcoded stock paths
    if run_args.ckpt_path is not None:
        data_dir = run_args.data_dir
        ckpt_path = run_args.ckpt_path
        save_dir = run_args.save_dir or f'./inference_results/{run_args.stock}'
    elif run_args.stock == 'AMZN':
        data_dir = '/home/myuser/processed_data/AMZN/2024_Dec'
        ckpt_path='/home/myuser/checkpoints/ruby-aardvark-62_98nov1i7'
        save_dir=f'/home/myuser/data/evalsequences/s5v2/AMZN/2024'
    elif run_args.stock == 'GOOG':
        data_dir = '/home/myuser/data/processed_data/GOOG/2023_Jan'
        ckpt_path='/home/myuser/data/checkpoints/lobs5_v2/twilight-sound-77_s42sujip'
        save_dir=f'/home/myuser/data/evalsequences/s5v2/GOOG/2023_Jan'
    elif run_args.stock == 'INTC':
        data_dir = '/home/myuser/data/processed_data/INTC/2023_Jan'
        ckpt_path='/home/myuser/data/checkpoints/lobs5_v2/dazzling-meadow-75_zpp3bf6z'
        save_dir=f'/home/myuser/data/evalsequences/s5v2/INTC/2023_Jan'
    else:
        raise Warning("Saved Model was trained on GOOGLE data. Generating for TSLA")
        data_dir = '/data1/sascha/data/lobster_proc'
        ckpt_path = '/data1/sascha/data/checkpoints/honest-oath-159_3kn3xbd5' # Dummy model trained on just 5 days... for debugging.

    ##################################################

    n_gen_msgs = run_args.n_gen_msgs  # how many messages to generate into the future
    n_messages_conditional = run_args.n_cond_msgs
    n_eval_messages = n_gen_msgs  # how many to load from dataset 
    eval_seq_len = (n_eval_messages-1) * Message_Tokenizer.MSG_LEN
    cond_seq_len = (n_messages_conditional) * Message_Tokenizer.MSG_LEN
    data_levels = 10
    # TODO: deprecated - remove from functions
    sim_book_levels = 20 # 10  # order book simulator levels
    sim_queue_len = 100  # per price in sim, how many orders in queue

    n_vol_series = 500  # how many book volume series model uses as input

    v = Vocab()
    n_classes = len(v)
    book_dim = 503 #b_enc.shape[1]
    eval_book_seq_len = eval_seq_len


    rng = jax.random.key(42)
    rng, rng_ = jax.random.split(rng)
    if overfit_debug:
        sample_top_n = 1
    else:
        sample_top_n = -1
    tick_size = 100

    # load train state from disk

    args = load_metadata(ckpt_path)
    args.num_devices=1
    args.bsz=1


    new_train_state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=eval_seq_len,
        book_dim=book_dim,
        book_seq_len=eval_book_seq_len,
    )


    # jax.tree_util.tree_map(lambda x: x.shape,state)
    ckpt = load_checkpoint(
        new_train_state,
        ckpt_path,
        step=0 if run_args.checkpoint_step is None else run_args.checkpoint_step,
        train=False,
    )
    state = ckpt['model']
    print(state.params['message_encoder']['encoder']['embedding'].shape)


    import chex
    chex.clear_trace_counter()

    model = model_cls(training=False, step_rescale=1.0)

    ##################################################

    import lob.evaluation as eval

    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    ds = inference.get_dataset(data_dir,
                               n_messages_conditional,
                               n_eval_messages,
                               test_split= run_args.test_split,
                               wide_book_dir=run_args.wide_book_dir,
                            #    day_indeces= [0],
                            #    limit_seq=4
                               )

    print("Dataset length: ", len(ds))
    # ds = LOBSTER_Dataset(
    #     msg_files,
    #     n_messages=n_messages + n_eval_messages,
    #     mask_fn=lambda X, rng: (X, jnp.array(0)),
    #     seed=42,
    #     n_cache_files=100,
    #     randomize_offset=False,
    #     book_files=book_files,
    #     use_simple_book=True,
    #     book_transform=False,
    #     book_depth=500,
    #     return_raw_msgs=True,
    # )

    ##################################################

    import logging
    # logging.basicConfig(filename='ar_debug.log', level=logging.DEBUG)
    fhandler = logging.FileHandler(filename='generation_debug.log', mode='w')
    logger = logging.getLogger()
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(fhandler)
    logger.setLevel(logging.WARNING)
    # logger.setLevel(logging.DEBUG)

    ##################################################



    n_samples = run_args.n_sequences
    batch_size = run_args.batch_size

    # Multi-GPU rank splitting: interleaved index assignment
    sample_indices = None
    if run_args.world_size > 1:
        all_indices = list(range(len(ds)))
        rank_indices = all_indices[run_args.rank::run_args.world_size]
        n_samples = min(n_samples, len(rank_indices))
        # Round down to batch_size multiple
        n_samples = (n_samples // batch_size) * batch_size
        sample_indices = rank_indices[:n_samples]
        print(f"[Rank {run_args.rank}/{run_args.world_size}] "
              f"Processing {n_samples} samples (indices {sample_indices[0]}..{sample_indices[-1]})")
    elif run_args.sample_indices_file is not None:
        sample_indices = list(onp.load(run_args.sample_indices_file).astype(int))

    # m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = inference.sample_new(
    # saves data to disk
    start=time()
    inference.sample_new(
        n_samples,
        batch_size,
        ds,
        rng,
        cond_seq_len,
        n_messages_conditional,
        n_gen_msgs,
        state,
        model,
        args.batchnorm,
        v.ENCODING,
        run_args.stock,
        save_folder=save_dir,
        sample_top_n= sample_top_n,
        args=args,
        conditional= True if n_messages_conditional>0 else False,
        overfit_debug=overfit_debug,
        sample_indices=sample_indices,
        wide_levels=run_args.wide_levels,
    )
    print(f"Generation time for {n_samples} sequences across {batch_size} batch size: {time()-start}")