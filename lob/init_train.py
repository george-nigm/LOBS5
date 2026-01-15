from __future__ import annotations
import json
import os
from argparse import Namespace
from glob import glob
from functools import partial
from typing import Any, Optional, Tuple, Union
import jax
import jax.numpy as np
from jax import random
import flax
from flax import jax_utils
import orbax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from jax.scipy.linalg import block_diag
from flax.training import checkpoints
from flax import linen as nn
from orbax import checkpoint
from lob.encoding import Vocab
from lob.lob_seq_model import BatchFullLobPredModel, BatchLobPredModel, BatchPaddedLobPredModel, FullLobPredModel, PaddedLobPredModel

#from lob.lob_seq_model import BatchLobPredModel
from lob.train_helpers import create_train_state#, eval_step, prep_batch, cross_entropy_loss, compute_accuracy
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
# from s5.dataloading import make_data_loader
# from lob.lobster_dataloader import LOBSTER_Dataset, LOBSTER

import lob.validation_helpers as valh


def deduplicate_trainstate(
        state: TrainState,
    ) -> TrainState:
    """
    Extract state to single device for checkpoint saving/loading.

    Old (pmap): State had device dimension, used x[0] to extract first device
    New (jit+shardings): State is replicated via sharding (no device dimension),
                        just need to put on single device
    """
    # With jit+shardings, arrays don't have device dimension
    # Just move to single device (no indexing needed)
    return jax.device_put(state, device=jax.devices('gpu')[0])

def load_args_from_checkpoint(
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> Namespace:

    """Load arguments from checkpoint"""
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    raw_restored = checkpoints.restore_checkpoint(
        checkpoint_path,
        None,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    args = Namespace(**raw_restored['config'])
    return args

def save_checkpoint(
        ckpt_mgr: ocp.CheckpointManager,
        ckpt: dict,
        epoch: int,
    ) -> bool:
    """
    """
    return ckpt_mgr.save(
        epoch,
        # args=ocp.args.PyTreeSave(ckpt)
        args=ocp.args.Composite(
            # train state
            state=ocp.args.StandardSave(ckpt['model']),
            # all other dict elements
            metadata=ocp.args.JsonSave({k: v for k, v in ckpt.items() if k != 'model'}),
        )
    )


# def load_checkpoint(
#         state: TrainState,
#         path: str,
#         config_dict: dict,
#         step: Optional[int] = None,
#     ) -> TrainState:
#     ckpt = {
#         'model': state,
#         'config': config_dict,
#         'metrics': {
#             'loss_train': np.nan,
#             'loss_val': np.nan,
#             'loss_test': np.nan,
#             'acc_val': np.nan,
#             'acc_test': np.nan,
#         }
#     }
#     orbax_checkpointer = checkpoint.PyTreeCheckpointer()
#     restored = checkpoints.restore_checkpoint(
#         path,
#         ckpt,
#         step=step,
#         orbax_checkpointer=orbax_checkpointer
#     )
#     return restored

def load_metadata(
        path: str,
    ) -> Namespace:

    # Remove trailing slash if present
    path = path.rstrip('/')

    # Try both possible metadata file names
    json_path = path + '/metadata/_ROOT_METADATA'
    if not os.path.exists(json_path):
        json_path = path + '/metadata/metadata'

    # load json path to dict
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # Handle nested "custom" or "custom_metadata" key in orbax checkpoints
    if 'custom' in metadata and isinstance(metadata['custom'], dict):
        metadata = metadata['custom']
    elif 'custom_metadata' in metadata and isinstance(metadata['custom_metadata'], dict):
        metadata = metadata['custom_metadata']

    # Handle renamed fields for compatibility
    if 'global_bsz' in metadata and 'bsz' not in metadata:
        metadata['bsz'] = metadata['global_bsz']

    # Handle mode field - convert "none" to "last" for compatibility
    if metadata.get('mode') == 'none':
        metadata['mode'] = 'last'

    return Namespace(**metadata)

def load_checkpoint(
        state: TrainState,
        path: str,
        # config_dict: dict,
        step: Optional[int] = None,
        train: bool = True,
    ) -> dict[str, Any]:
    """
    Load checkpoint from OCDBT format using tensorstore directly.
    This avoids orbax version compatibility issues.
    """
    import json as json_mod
    import tensorstore as ts

    abs_path = os.path.abspath(path)

    # Find latest step
    if step is None:
        step_dirs = []
        for item in os.listdir(abs_path):
            item_path = os.path.join(abs_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                step_dirs.append(int(item))
        if not step_dirs:
            raise ValueError(f"No checkpoint steps found in {abs_path}")
        step = max(step_dirs)
        print(f"Found latest step: {step}")

    step_dir = os.path.join(abs_path, str(step), 'state')
    print(f"Loading checkpoint from step {step} using tensorstore...")

    # Read _METADATA to get the structure
    metadata_path = os.path.join(step_dir, '_METADATA')
    with open(metadata_path, 'r') as f:
        ckpt_metadata = json_mod.load(f)

    tree_metadata = ckpt_metadata.get('tree_metadata', {})
    use_zarr3 = ckpt_metadata.get('use_zarr3', False)
    zarr_driver = 'zarr3' if use_zarr3 else 'zarr'

    # Filter to only params and batch_stats keys (skip opt_state)
    keys_to_load = []
    for key_str in tree_metadata.keys():
        key_tuple = eval(key_str)
        if key_tuple[0] in ('params', 'batch_stats', 'step'):
            keys_to_load.append((key_tuple, tree_metadata[key_str]))

    print(f"Loading {len(keys_to_load)} arrays from OCDBT checkpoint...")

    # Load each array from OCDBT using tensorstore
    restored_params = {}
    loaded_count = 0
    failed_keys = []

    for key_tuple, value_meta in keys_to_load:
        path_str = '/'.join(str(k) for k in key_tuple)

        # Try multiple drivers and path formats
        loaded = False

        # Different path formats to try
        path_formats = [
            path_str,                               # params/book_encoder/...
            path_str.replace('/', '.'),             # params.book_encoder....
        ]

        for driver in [zarr_driver, 'zarr', 'zarr3']:
            if loaded:
                break
            for path_fmt in path_formats:
                if loaded:
                    break
                try:
                    spec = {
                        'driver': driver,
                        'kvstore': {
                            'driver': 'ocdbt',
                            'base': f'file://{step_dir}',
                            'path': path_fmt,
                        },
                    }
                    store = ts.open(spec).result()
                    arr = store.read().result()

                    # Put into nested dict
                    current = restored_params
                    for k in key_tuple[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[key_tuple[-1]] = np.array(arr)
                    loaded_count += 1
                    loaded = True
                    if loaded_count == 1:
                        print(f"  Success with driver={driver}, path format: {path_fmt[:50]}")
                except Exception as e:
                    if driver == zarr_driver and path_fmt == path_str:
                        failed_keys.append((path_str, f"[{driver}] {str(e)[:60]}"))

        if not loaded and loaded_count <= 1:
            # Print first few failures for debugging
            print(f"  Failed to load: {path_str[:60]}")

    print(f"Successfully loaded {loaded_count}/{len(keys_to_load)} arrays")
    if failed_keys and len(failed_keys) < 10:
        for fk, err in failed_keys:
            print(f"  Failed: {fk}: {err[:100]}")

    if 'params' not in restored_params or loaded_count == 0:
        raise RuntimeError(f"Failed to load params from {step_dir}. Loaded {loaded_count} arrays.")

    # Build restored_state as a dict (not TrainState - will be converted later if needed)
    restored_state_dict = {
        'params': restored_params.get('params', {}),
    }
    if 'batch_stats' in restored_params:
        restored_state_dict['batch_stats'] = restored_params['batch_stats']
    if 'step' in restored_params:
        restored_state_dict['step'] = restored_params['step']

    # Create TrainState-like object with loaded params
    # Replace params in the provided state template
    target = deduplicate_trainstate(state)
    restored_state = target.replace(params=restored_state_dict['params'])
    if 'batch_stats' in restored_state_dict and hasattr(target, 'batch_stats'):
        restored_state = restored_state.replace(batch_stats=restored_state_dict['batch_stats'])

    # Load metadata JSON
    metadata_dir = os.path.join(abs_path, str(step), 'metadata')
    metadata_file = os.path.join(metadata_dir, 'metadata')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata_dict = json_mod.load(f)
    else:
        metadata_dict = {}

    ckpt = metadata_dict
    # Copy train state back to all devices
    if train:
        ckpt['model'] = jax_utils.replicate(restored_state)
    else:
        ckpt['model'] = restored_state
    return ckpt


def load_checkpoint_legacy(
        state: TrainState,
        path: str,
        # config_dict: dict,
        step: Optional[int] = None,
        train: bool = True,
    ) -> dict[str, Any]:
    """Legacy checkpoint loader for older checkpoint formats."""

    # Create CheckpointManager with item names
    abs_path = os.path.abspath(path)

    if step is None:
        # Find latest step by scanning directories directly
        # (Avoids CheckpointManager issues with file ownership in Docker)
        step_dirs = []
        for item in os.listdir(abs_path):
            item_path = os.path.join(abs_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                step_dirs.append(int(item))
        if not step_dirs:
            raise ValueError(f"No checkpoint steps found in {abs_path}")
        step = max(step_dirs)
        print(f"Found latest step: {step}")

    step_dir = os.path.join(abs_path, str(step), 'state')

    # Check if modern Orbax format with _METADATA file
    has_metadata_file = os.path.exists(os.path.join(step_dir, '_METADATA'))

    if has_metadata_file:
        # Modern Orbax checkpoint format with _METADATA - load using tensorstore
        print(f"Loading checkpoint from step {step} using tensorstore (zarr3 format)...")
        import json
        import tensorstore as ts
        import numpy as np

        target = deduplicate_trainstate(state)

        # Read _METADATA to get the structure
        metadata_path = os.path.join(step_dir, '_METADATA')
        with open(metadata_path, 'r') as f:
            ckpt_metadata = json.load(f)

        tree_metadata = ckpt_metadata.get('tree_metadata', {})
        use_zarr3 = ckpt_metadata.get('use_zarr3', False)

        # Filter to only params and batch_stats keys (skip opt_state)
        keys_to_load = []
        for key_str in tree_metadata.keys():
            key_tuple = eval(key_str)
            if key_tuple[0] in ('params', 'batch_stats', 'step'):
                keys_to_load.append((key_tuple, tree_metadata[key_str]))

        print(f"Loading {len(keys_to_load)} arrays from checkpoint...")

        # Determine zarr driver based on format
        zarr_driver = 'zarr3' if use_zarr3 else 'zarr'

        # Load each array from OCDBT using tensorstore
        restored_params = {}
        loaded_count = 0

        for key_tuple, value_meta in keys_to_load:
            path_str = '/'.join(str(k) for k in key_tuple)

            try:
                spec = {
                    'driver': zarr_driver,
                    'kvstore': {
                        'driver': 'ocdbt',
                        'base': f'file://{step_dir}',
                        'path': path_str,
                    },
                }
                store = ts.open(spec).result()
                arr = store.read().result()

                # Convert to jax array
                arr = jnp.array(arr)

                # Put into nested dict
                current = restored_params
                for k in key_tuple[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[key_tuple[-1]] = arr
                loaded_count += 1
            except Exception as e:
                print(f"Warning: Failed to load {path_str}: {e}")

        print(f"Successfully loaded {loaded_count} arrays")

        # Build restored_state matching target structure
        restored_state = {}
        restored_state['params'] = restored_params.get('params', {})
        if 'batch_stats' in restored_params:
            restored_state['batch_stats'] = restored_params['batch_stats']
        if 'step' in restored_params:
            restored_state['step'] = restored_params['step']

        # Add empty opt_state if not training
        if not train:
            restored_state['opt_state'] = None

        # Load metadata JSON
        metadata_dir = os.path.join(abs_path, str(step), 'metadata')
        metadata_file = os.path.join(metadata_dir, 'metadata')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
        else:
            metadata_dict = {}

        loaded = {
            'state': restored_state,
            'metadata': metadata_dict
        }
    else:
        # Use PyTreeCheckpointHandler directly to load only the params we need
        # This handles the case where checkpoint has opt_state that doesn't match
        print(f"Loading checkpoint from step {step} using direct PyTree handler...")

        target = deduplicate_trainstate(state)

        # Build a restore_args structure that matches only params (not opt_state)
        # and use transformations to load just what we need
        from orbax.checkpoint import type_handlers

        # First, try to load using the standard method
        try:
            mngr = ocp.CheckpointManager(
                abs_path,
                item_names=('state', 'metadata'),
                options=ocp.CheckpointManagerOptions(),
            )

            loaded = mngr.restore(
                step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(target),
                    metadata=ocp.args.JsonRestore()
                )
            )
        except (KeyError, ValueError) as e:
            # Checkpoint structure mismatch (likely opt_state issue)
            # Use direct tensorstore/OCDBT reading to load only params
            print(f"StandardRestore failed ({e}), loading params directly from OCDBT...")
            import json as json_mod
            import tensorstore as ts

            # Read _METADATA to get the structure for params only
            metadata_path = os.path.join(step_dir, '_METADATA')
            with open(metadata_path, 'r') as f:
                ckpt_metadata = json_mod.load(f)

            tree_metadata = ckpt_metadata.get('tree_metadata', {})

            # Filter to only params and batch_stats keys
            params_keys = []
            for key_str in tree_metadata.keys():
                key_tuple = eval(key_str)
                if key_tuple[0] in ('params', 'batch_stats'):
                    params_keys.append(key_tuple)

            print(f"Found {len(params_keys)} params/batch_stats keys to load from OCDBT")

            # Load each array from OCDBT using tensorstore
            restored_params = {}
            loaded_count = 0

            for key_tuple in params_keys:
                path_str = '/'.join(str(k) for k in key_tuple)

                try:
                    spec = {
                        'driver': 'zarr',
                        'kvstore': {
                            'driver': 'ocdbt',
                            'base': f'file://{step_dir}',
                            'path': path_str,
                        },
                    }
                    store = ts.open(spec).result()
                    arr = store.read().result()

                    # Put into nested dict
                    current = restored_params
                    for k in key_tuple[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[key_tuple[-1]] = np.array(arr)
                    loaded_count += 1
                except Exception as arr_e:
                    # Only warn if it's not a known issue
                    if 'ocdbt' not in str(arr_e).lower():
                        print(f"Warning: Could not load {key_tuple}: {arr_e}")

            print(f"Successfully loaded {loaded_count}/{len(params_keys)} arrays from OCDBT")

            if 'params' not in restored_params or loaded_count == 0:
                raise RuntimeError(f"Failed to load params from {step_dir}. Loaded {loaded_count} arrays.")

            # Create restored state
            restored_state = target.replace(params=restored_params['params'])
            if 'batch_stats' in restored_params:
                restored_state = restored_state.replace(batch_stats=restored_params['batch_stats'])

            # Load metadata JSON
            metadata_file = os.path.join(abs_path, str(step), 'metadata', 'metadata')
            with open(metadata_file, 'r') as f:
                metadata_dict = json_mod.load(f)

            loaded = {
                'state': restored_state,
                'metadata': metadata_dict
            }

    ckpt = loaded['metadata']
    # copy train state back to all devices
    if train:
        ckpt['model'] = jax_utils.replicate(loaded['state'])
    else:
        ckpt['model'] = loaded['state']
    return ckpt


def init_train_state(
        args: Namespace,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        print_shapes=False
    ) -> Tuple[TrainState, Union[partial[BatchLobPredModel], partial[FullLobPredModel]]]:

    in_dim = n_classes

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)

    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    if print_shapes:
        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(V.shape))
        print("Vinv.shape={}".format(Vinv.shape))
        print("book_seq_len", book_seq_len)
        print("book_dim", book_dim)

    padded = False
    retrieval = False
    speech = False

    ssm_init_fn = init_S5SSM(
        H=args.d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional
    )
    
    # Also create unbatched model class for use with vmap-based inference
    model_cls_unbatched = None

    if args.use_book_data:
        # Select model based on merging method
        merging = getattr(args, 'merging', 'projected')

        if merging == 'projected':
            model_cls = partial(
                BatchFullLobPredModel,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,  # 2
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )
            model_cls_unbatched = partial(
                FullLobPredModel,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )
        elif merging == 'padded':
            model_cls = partial(
                BatchPaddedLobPredModel,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,  # 2
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )
            model_cls_unbatched = partial(
                PaddedLobPredModel,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )
        else:
            raise ValueError(f"Merge method: {merging} is not valid (check spelling)")
    else:
        if args.num_devices > 1:
            raise NotImplementedError("Message only model not implemented for multi-device training")
        
        model_cls = partial(
            BatchLobPredModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        use_book_data=args.use_book_data,
        in_dim=1, # in_dim,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=args.dt_global,
        num_devices=args.num_devices,
    )

    return state, model_cls, model_cls_unbatched
