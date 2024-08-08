import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['XLA_FLAGS'] ='--xla_gpu_deterministic_ops=true'


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

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
from encoding import Vocab, Message_Tokenizer
# from lobster_dataloader import LOBSTER_Dataset, LOBSTER_Subset, LOBSTER_Sampler, LOBSTER

import preproc
# import inference
from lob import inference_no_errcorr as inference
import validation_helpers as valh
from lob.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint
# import lob.encoding as encoding


import lob.evaluation as eval
from preproc import transform_L2_state

##################################################

if __name__ == "__main__":

    # get args from command line to select stock between GOOG, INTC
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str, default='GOOG', help='stock to evaluate')
    run_args = parser.parse_args()



    if run_args.stock == 'GOOG':
        data_dir = '/data1/sascha/data/GOOG2017to2019'
        ckpt_path='/data1/sascha/data/checkpoints/honest-oath-159_3kn3xbd5'
    elif run_args.stock == 'INTC':
        raise NotImplementedError("Nothing trained for INTC yet")
    elif run_args.stock == 'TSLA':
        raise Warning("Saved Model was trained on GOOGLE data. Generating for TSLA")
        data_dir = '/data1/sascha/data/lobster_proc'
        ckpt_path = '/data1/sascha/data/checkpoints/honest-oath-159_3kn3xbd5' # Dummy model trained on just 5 days... for debugging. 

    ##################################################

    n_gen_msgs = 1000  #500 # how many messages to generate into the future
    n_messages_conditional = 500
    n_eval_messages = 1000  # how many to load from dataset 
    eval_seq_len = (n_eval_messages-1) * Message_Tokenizer.MSG_LEN
    cond_seq_len = (n_messages_conditional) * Message_Tokenizer.MSG_LEN

    data_levels = 10
    # TODO: deprecated - remove from functions
    sim_book_levels = 20 # 10  # order book simulator levels
    sim_queue_len = 100  # per price in sim, how many orders in queue

    n_vol_series = 500  # how many book volume series model uses as input

    v = Vocab()
    n_classes = len(v)
    book_dim = 501 #b_enc.shape[1]
    eval_book_seq_len = eval_seq_len


    rng = jax.random.key(42)
    rng, rng_ = jax.random.split(rng)
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
        train=False,
    )
    state = ckpt['model']
    print(state.params['message_encoder']['encoder']['embedding'].shape)


    import chex
    chex.clear_trace_counter()

    model = model_cls(training=False, step_rescale=1.0)

    ##################################################

    import lob.evaluation as eval
    from preproc import transform_L2_state

    # n_gen_msgs = 100  #500 # how many messages to generate into the future
    # n_messages = 500
    # n_eval_messages = 100  # how many to load from dataset 
    # eval_seq_len = n_eval_messages * Message_Tokenizer.MSG_LEN

    # data_levels = 10
    # sim_book_levels = 20 # 10  # order book simulator levels
    # sim_queue_len = 100  # per price in sim, how many orders in queue

    # n_vol_series = 500  # how many book volume series model uses as input

    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    ds = inference.get_dataset(data_dir, n_messages_conditional, n_eval_messages)

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
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.WARNING)

    ##################################################



    n_samples = 160
    batch_size = 16

    # m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = inference.sample_new(
    # saves data to disk
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
        save_folder='/data1/sascha/data/GOOG_argmax/',
        sample_top_n=sample_top_n,
        args=args,
        conditional= True,
        init_time = (jnp.array([0]),jnp.array([0])),
    )