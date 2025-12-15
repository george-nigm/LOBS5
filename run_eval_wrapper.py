#!/usr/bin/env python
"""
Wrapper for run_eval.py that doesn't hardcode CUDA_VISIBLE_DEVICES
"""
import os
import sys

# Don't override SLURM's GPU assignment
# Just use whatever CUDA_VISIBLE_DEVICES SLURM sets

# Now import and run the actual eval
import jax
from jax import random
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
import wandb

from lob.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate
from lob.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import reduce_lr_on_plateau, linear_warmup, \
    cosine_annealing, constant_lr, train_epoch, validate

from lob.init_train import load_metadata, load_args_from_checkpoint


def eval(eval_args):
    """Main function to evaluate a given checkpoint"""

    args = load_metadata(eval_args.restore)
    for arg in vars(eval_args):
        setattr(args, str(arg), getattr(eval_args, str(arg)))

    if args.USE_WANDB:
        run = wandb.init(project=args.wandb_project, job_type='model_eval',
                        config=vars(args), entity=args.wandb_entity)
    else:
        run = wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    print("[*] Eval Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    init_rng, key = random.split(init_rng, num=2)
    mask_fn = None
    if args.masking == 'causal':
        mask_fn = LOBSTER_Dataset.causal_mask
    elif args.masking == 'random':
        mask_fn = LOBSTER_Dataset.random_mask
    elif args.masking == 'last_pos':
        mask_fn = LOBSTER_Dataset.last_pos_mask
    elif args.masking == 'none':
        mask_fn = LOBSTER_Dataset.no_mask
    else:
        raise ValueError('Issue with mask function: logic for ' + args.masking + ' not implemented.')

    (lobster_dataset, trainloader, valloader, testloader, aux_dataloaders,
        n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(
            args.dir_name,
            seed=args.jax_seed,
            mask_fn=mask_fn,
            msg_seq_len=args.msg_seq_len,
            bsz=args.bsz,
            use_book_data=args.use_book_data,
            use_simple_book=args.use_simple_book,
            book_transform=args.book_transform,
            n_data_workers=args.n_data_workers,
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
        )

    print(f"[*] Starting S5 Eval on lobster-prediction =>> Loading the states...")
    state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        print_shapes=True
    )

    dt = [[x] for (x,) in zip([*range(seq_len)])]
    ce_table = wandb.Table(columns=["tok"], data=dt)

    for epoch in range(args.epochs):
        print(f"[*] Starting Val/Test of Epoch {args.restore_step + epoch + 1}...")
        print(f"[*] Restoring weights from {args.restore}")

        ckpt = load_checkpoint(
            state,
            args.restore,
            step=args.restore_step + epoch,
        )
        state = ckpt['model']
        eval_model = model_cls(training=False, step_rescale=1)

        init_hidden = model_cls().initialize_carry(
            batch_size=args.bsz // args.num_devices,
            hidden_size=(ssm_size // pow(2, int(args.conj_sym))),
            n_message_layers=args.n_message_layers,
            n_book_pre_layers=args.n_book_pre_layers,
            n_book_post_layers=args.n_book_post_layers,
            n_fused_layers=args.n_layers,
        )

        print(f"[*] Running Epoch {args.restore_step + epoch + 1} Validation on train set (With call)...")
        val_loss, val_acc, val_ce_by_tok, val_acc_by_tok = validate(
            state,
            eval_model.apply,
            trainloader,
            seq_len,
            in_dim,
            args.batchnorm,
            args.num_devices,
            epoch,
            curtail_epoch=args.curtail_epoch,
            ignore_times=args.ignore_times,
            apply_method='__call_ar__'
        )

        print(f"[*] Running Epoch {args.restore_step + epoch + 1} Test on train set (With Scan RNN)...")
        test_loss, test_acc, test_ce_by_tok, test_acc_by_tok = validate(
            state,
            eval_model.apply,
            trainloader,
            seq_len,
            in_dim,
            args.batchnorm,
            args.num_devices,
            epoch,
            curtail_epoch=args.curtail_epoch,
            ignore_times=args.ignore_times,
            apply_method='__call_rnn__',
            init_hiddens=init_hidden
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f" Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
              f" Val Accuracy: {val_acc:.4f} Test Accuracy: {test_acc:.4f}")

        ce_table.add_column(name="test_ce_" + str(epoch), data=test_ce_by_tok.tolist())
        ce_table.add_column(name="test_acc_" + str(epoch), data=test_acc_by_tok.tolist())
        ce_table = wandb.Table(columns=ce_table.columns, data=ce_table.data)

        wandb.log({
            "'Call' loss": val_loss,
            "'Call' Accuracy": val_acc,
            "'Call_Rnn' Loss": test_loss,
            "'Call_Rnn' Accuracy": test_acc,
            "Training CE by token": ce_table
        })

    wandb.finish()


if __name__ == "__main__":
    import argparse
    from s5.utils.util import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=str, required=True)
    parser.add_argument("--curtail_epoch", type=int, default=None)
    parser.add_argument("--restore_step", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mask_time_losses", type=str2bool, default=False)
    parser.add_argument("--dir_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="LOBS5-Eval")
    parser.add_argument("--wandb_entity", type=str, default="kang-oxford")
    parser.add_argument("--n_data_workers", type=int, default=0)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--USE_WANDB", type=str2bool, default=True)
    parser.add_argument("--ignore_times", type=str2bool, default=True)

    eval_args = parser.parse_args()

    import torch
    torch.multiprocessing.set_start_method('spawn')

    eval(eval_args)
