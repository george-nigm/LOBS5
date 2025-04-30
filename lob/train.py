import os
import jax
from jax import random
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
# import wandb

from lob.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate
from lob.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader#, Datasets
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import reduce_lr_on_plateau, linear_warmup, \
    cosine_annealing, constant_lr, train_epoch, validate


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    # for parameter sweep: get args from wandb server
    # if args is None:
    #     args = wandb.config
    # else:
    #     if args.USE_WANDB:
    #         # Make wandb config dictionary
    #         run = wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    #     else:
    #         run = wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    # wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    ds = 'lobster-prediction'
    #create_dataset_fn =  Datasets[ds]

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    mask_fn = LOBSTER_Dataset.causal_mask if args.masking == 'causal' else LOBSTER_Dataset.random_mask
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
        )

    print(f"[*] Starting S5 Training on {ds} =>> Initializing...")

    state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        print_shapes=True
    )

    if args.restore is not None and args.restore != '':
        print(f"[*] Restoring weights from {args.restore}")
        ckpt = load_checkpoint(
            state,
            args.restore,
            # args.__dict__,
            step=args.restore_step,
        )
        state = ckpt['model']
    
    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size/args.bsz)

    val_model = model_cls(training=False, step_rescale=1)

    mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        create=True,
        max_to_keep=10,
        keep_period=10,
        # step_prefix=f'{run.name}_{run.id}',
        enable_async_checkpointing=False,
    )
    ckpt_mgr = ocp.CheckpointManager(
        os.path.abspath(f'checkpoints/{run.name}_{run.id}/'),
        # ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        # ocp.Checkpointer(ocp.StandardCheckpointHandler()),
        item_names=('state', 'metadata'),
        options=mgr_options,
        metadata=vars(args)
    )

    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        print('Training on', args.num_devices, 'devices.')
        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(state,
                                              skey,
                                              #model_cls,
                                              #train_model,
                                              trainloader,
                                              seq_len,
                                              #in_dim,
                                              args.batchnorm,
                                              lr_params,
                                              args.num_devices)
        # reinit training loader, so that sequences are initialised with
        del trainloader
        # different offsets
        trainloader = create_lobster_train_loader(
            lobster_dataset,
            int(random.randint(skey, (1,), 0, 100000)[0]),
            args.bsz,
            num_workers=args.n_data_workers,
            reset_train_offsets=True)

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(state,
                                         #model_cls,
                                         val_model.apply,
                                         valloader,
                                         seq_len,
                                         in_dim,
                                         args.batchnorm,
                                         args.num_devices)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(state,
                                           #model_cls,
                                           val_model.apply,
                                           testloader,
                                           seq_len,
                                           in_dim,
                                           args.batchnorm,
                                           args.num_devices)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(state,
                                         #model_cls,
                                         val_model.apply,
                                         testloader,
                                         seq_len,
                                         in_dim,
                                         args.batchnorm,
                                         args.num_devices)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # save checkpoint
        ckpt = {
            'model': deduplicate_trainstate(state),
            'config': vars(args),
            'metrics': {
                'loss_train': float(train_loss),
                'loss_val': float(val_loss),
                'loss_test': float(test_loss),
                'acc_val': float(val_acc),
                'acc_test': float(test_acc),
            }
        }
        save_checkpoint(ckpt_mgr, ckpt, epoch)

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        # if valloader is not None:
        #     wandb.log(
        #         {
        #             "Training Loss": train_loss,
        #             "Val loss": val_loss,
        #             "Val Accuracy": val_acc,
        #             "Test Loss": test_loss,
        #             "Test Accuracy": test_acc,
        #             "count": count,
        #             "Learning rate count": lr_count,
        #             "Opt acc": opt_acc,
        #             "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
        #             "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
        #         }
        #     )
        # else:
        #     wandb.log(
        #         {
        #             "Training Loss": train_loss,
        #             "Val loss": val_loss,
        #             "Val Accuracy": val_acc,
        #             "count": count,
        #             "Learning rate count": lr_count,
        #             "Opt acc": opt_acc,
        #             "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
        #             "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
        #         }
        #     )
        # wandb.run.summary["Best Val Loss"] = best_loss
        # wandb.run.summary["Best Val Accuracy"] = best_acc
        # wandb.run.summary["Best Epoch"] = best_epoch
        # wandb.run.summary["Best Test Loss"] = best_test_loss
        # wandb.run.summary["Best Test Accuracy"] = best_test_acc

        if count > args.early_stop_patience:
            break
