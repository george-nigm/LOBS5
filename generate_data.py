import argparse
import os
import sys


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

import torch
torch.multiprocessing.set_start_method('spawn')

import jax
from lob.encoding import Vocab, Message_Tokenizer

from lob import inference_no_errcorr as inference
from lob.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint

print(os.path.abspath(''))

#############################################################################
parser = argparse.ArgumentParser()

parser.add_argument(
    "--stock", type=str)
parser.add_argument(
    "--save_folder", type=str, default='/nfs/home/peern/LOBS5/data_saved/')
parser.add_argument(
    "--n_gen_msgs", type=int,
	help="how many messages to generate following each input sequence")
parser.add_argument(
    "--n_samples", type=int,
	help="how many messages sequences to generate")
parser.add_argument(
    "--batch_size", type=int, default=16,
	help="how many sequences to generate in parallel (vmap)")
parser.add_argument(
    "--model_size", type=str, default='large',)
parser.add_argument(
    "--data_dir", type=str, default='/nfs/home/peern/LOBS5/data/test_set/',)
parser.add_argument(
    "--rng_seed", type=int, default=42,)
parser.add_argument(
    "--sample_all", action='store_true', default=False,
    help="sample all data sequences, instead of randomly sampling")

args = parser.parse_args()

#############################################################################

n_messages = 500  # length of input sequence to model
n_gen_msgs = args.n_gen_msgs # how many messages to generate into the future
n_samples = args.n_samples
batch_size = args.batch_size

v = Vocab()
n_classes = len(v)
seq_len = n_messages * Message_Tokenizer.MSG_LEN
book_dim = 501 #b_enc.shape[1]
book_seq_len = n_messages

n_eval_messages = args.n_gen_msgs  # how many to load from dataset
eval_seq_len = n_eval_messages * Message_Tokenizer.MSG_LEN

rng = jax.random.key(args.rng_seed)
rng, rng_ = jax.random.split(rng)

stock = args.stock # 'GOOG', 'INTC'

if stock == 'GOOG':
    # ckpt_path = './checkpoints/treasured-leaf-149_84yhvzjt/' # 0.5 y GOOG, (full model)
    # ckpt_path = './checkpoints/denim-elevator-754_czg1ss71/' # large model
    ckpt_path = './checkpoints/stilted-deluge-759_8g3vqor4'  # small model
elif stock == 'INTC':
    # ckpt_path = './checkpoints/pleasant-cherry-152_i6h5n74c/' # 0.5 y INTC, (full model)
    ckpt_path = './checkpoints/eager-sea-755_2rw1ofs3/'  # large model
else:
    raise ValueError(f'stock {stock} not recognized')

print('Loading metadata:', ckpt_path)
args_ckpt = load_metadata(ckpt_path)

# scale down to single GPU, single sample inference
args_ckpt.bsz = 1 #1, 10
args_ckpt.num_devices = 1

batchnorm = args_ckpt.batchnorm

# load train state from disk

print('Initializing model...')
new_train_state, model_cls = init_train_state(
    args_ckpt,
    n_classes=n_classes,
    seq_len=seq_len,
    book_dim=book_dim,
    book_seq_len=book_seq_len,
)

print('Loading model checkpoint...')
ckpt = load_checkpoint(
    new_train_state,
    ckpt_path,
    train=False,
)
state = ckpt['model']

model = model_cls(training=False, step_rescale=1.0)

param_count = sum(x.size for x in jax.tree_leaves(state.params))
print('param count:', param_count)

data_dir = args.data_dir + stock
ds = inference.get_dataset(data_dir, n_messages, n_eval_messages)

print('Generating...')
inference.sample_new(
    n_samples,
    batch_size,
    ds,
    rng,
    seq_len,
    n_messages,
    n_gen_msgs,
    state,
    model,
    batchnorm,
    v.ENCODING,
    stock,
    save_folder=args.save_folder + '/' + stock + '/',
    sample_all=args.sample_all,
)
print('DONE.')
