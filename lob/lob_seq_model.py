from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from s5.layers import SequenceLayer
from s5.seq_model import StackedEncoderModel, masked_meanpool


class LobPredModel(nn.Module):
    """ S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            mode        (str):      Options: [pool: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                                        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)

# Here we call vmap to parallelize across a batch of input sequences
BatchLobPredModel = nn.vmap(
    LobPredModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')


class LobBookModel(nn.Module):
    ssm: nn.Module
    d_book: int
    d_model: int
    #n_layers: int
    n_pre_layers: int
    n_post_layers: int
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes pre_layers, projection, and post_layers.
        Uses separate attributes (not unified tuple) to match checkpoint structure.
        """
        # Pre-layers: process book at d_book dimension
        self.pre_layers = tuple(
            SequenceLayer(
                # fix ssm init to correct shape (different than other layers)
                ssm=partial(self.ssm, H=self.d_book),
                dropout=self.dropout,
                d_model=self.d_book,  # take book series as is
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
            ) for _ in range(self.n_pre_layers)
        )
        # Projection layer: project from d_book to d_model
        self.projection = nn.Dense(self.d_model)
        # Post-layers: process at d_model dimension
        self.post_layers = tuple(
            SequenceLayer(
                ssm=self.ssm,
                dropout=self.dropout,
                d_model=self.d_model,
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
            )
            for _ in range(self.n_post_layers)
        )

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        for layer in self.pre_layers:
            x = layer(x)
        x = self.projection(x)
        for layer in self.post_layers:
            x = layer(x)
        return x

    def __call_rnn__(self, hiddens, x, d, integration_timesteps):
        """
        Compute the LxH output with hidden state support.

        Args:
            hiddens: tuple of (pre_hiddens, post_hiddens) lists
            x (float32): input sequence (L, d_book)
            d (bool): reset signal
            integration_timesteps: unused
        Returns:
            new_hiddens: updated tuple of hidden states
            output sequence (float32): (L, d_model)
        """
        hidden_pre, hidden_post = hiddens
        new_hiddens_pre = []
        new_hiddens_post = []

        # Pre-layers (SequenceLayers)
        for i, layer in enumerate(self.pre_layers):
            new_h, x = layer.__call_rnn__(hidden_pre[i], x, d)
            new_hiddens_pre.append(new_h)

        # Projection layer (Dense - no hidden state)
        x = self.projection(x)

        # Post-layers (SequenceLayers)
        for i, layer in enumerate(self.post_layers):
            new_h, x = layer.__call_rnn__(hidden_post[i], x, d)
            new_hiddens_post.append(new_h)

        return (new_hiddens_pre, new_hiddens_post), x

    @staticmethod
    def initialize_carry(batch_size, hidden_size, n_layers_pre, n_layers_post):
        """Initialize hidden states for RNN mode."""
        init_hidden = (
            [SequenceLayer.initialize_carry(batch_size, hidden_size)
             for _ in range(n_layers_pre)],
            [SequenceLayer.initialize_carry(batch_size, hidden_size)
             for _ in range(n_layers_post)]
        )
        return init_hidden


class FullLobPredModel(nn.Module):
    ssm: nn.Module
    d_output: int
    d_model: int
    d_book: int
    n_message_layers: int
    n_fused_layers: int
    n_book_pre_layers: int = 1
    n_book_post_layers: int = 1
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.message_encoder = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_message_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=True,
            vocab_size=self.d_output,
        )
        # applied to transposed message output to get seq len for fusion
        self.message_out_proj = nn.Dense(self.d_model)  
        self.book_encoder = LobBookModel(
            ssm=self.ssm,
            d_book=self.d_book,
            d_model=self.d_model,
            n_pre_layers=self.n_book_pre_layers,
            n_post_layers=self.n_book_post_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        # applied to transposed book output to get seq len for fusion
        self.book_out_proj = nn.Dense(self.d_model)
        self.fused_s5 = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_fused_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        (L_m x d_input, L_b x [P+1]) input sequence tuple,
        combining message and book inputs.
        Args:
             x (float32): 2-tuple of input sequences (L_m x d_input, L_b x [P+1])
        Returns:
            output (float32): (d_output)
        """
        #x_m, x_b = x
        # print(x_m.shape, x_b.shape)

        x_m = self.message_encoder(x_m, message_integration_timesteps)
        # TODO: check integration time steps make sense here
        x_b = self.book_encoder(x_b, book_integration_timesteps)

        x_m = self.message_out_proj(x_m.T).T
        x_b = self.book_out_proj(x_b.T).T
        x = jnp.concatenate([x_m, x_b], axis=1)
        # TODO: again, check integration time steps make sense here
        x = self.fused_s5(x, jnp.ones(x.shape[0]))

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)

    def __call_rnn__(self, hiddens_tuple, x_m, x_b, d_m, d_b, d_f,
                     message_integration_timesteps, book_integration_timesteps):
        """
        Compute output with hidden state support for autoregressive inference.

        Args:
            hiddens_tuple: tuple of (message_hiddens, book_hiddens, fused_hiddens, ema_state)
            x_m: message sequence (L_m,) tokens
            x_b: book sequence (L_b, d_book)
            d_m, d_b, d_f: reset signals (unused)
            message_integration_timesteps, book_integration_timesteps: unused
        Returns:
            new_hiddens_tuple: updated hidden states
            logits: output (d_output,)
        """
        hiddens_m, hiddens_b, hiddens_fused, ema_state = hiddens_tuple

        # Message encoder with hidden state
        hiddens_m, x_m = self.message_encoder.__call_rnn__(
            hiddens_m, x_m, d_m, message_integration_timesteps)

        # Book encoder with hidden state
        hiddens_b, x_b = self.book_encoder.__call_rnn__(
            hiddens_b, x_b, d_b, book_integration_timesteps)

        # Project to same dimension
        x_m = self.message_out_proj(x_m.T).T
        x_b = self.book_out_proj(x_b.T).T

        # Concatenate
        x = jnp.concatenate([x_m, x_b], axis=1)

        # Fused S5 with hidden state
        hiddens_fused, x = self.fused_s5.__call_rnn__(
            hiddens_fused, x, d_f, jnp.ones(x.shape[0]))

        # Pool or take last
        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        # Decode
        x = self.decoder(x)
        logits = nn.log_softmax(x, axis=-1)

        new_hiddens_tuple = (hiddens_m, hiddens_b, hiddens_fused, ema_state)
        return new_hiddens_tuple, logits

    @staticmethod
    def initialize_carry(batch_size, hidden_size, n_message_layers,
                         n_book_pre_layers, n_book_post_layers,
                         n_fused_layers, h_size_ema):
        """Initialize hidden states for RNN mode."""
        message_hiddens = StackedEncoderModel.initialize_carry(
            batch_size, hidden_size, n_message_layers)
        book_hiddens = LobBookModel.initialize_carry(
            batch_size, hidden_size, n_book_pre_layers, n_book_post_layers)
        fused_hiddens = StackedEncoderModel.initialize_carry(
            batch_size, hidden_size, n_fused_layers)
        # EMA state: (running_avg, step_count)
        ema_state = (jnp.zeros((batch_size, 1, h_size_ema)),
                     jnp.ones((batch_size, 1, 1)))
        return (message_hiddens, book_hiddens, fused_hiddens, ema_state)


# Variable axes and split_rngs configuration
_variable_axes_args = {"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None}
_split_rngs_args = {"params": False, "dropout": True}

# Here we call vmap to parallelize across a batch of input sequences
# Note: BatchFullLobPredModel supports both __call__ (default) and __call_rnn__ methods
# The methods parameter specifies vmap axes for each method
BatchFullLobPredModel = nn.vmap(
    FullLobPredModel,
    in_axes=(0, 0, 0, 0),
    out_axes=0,
    variable_axes=_variable_axes_args,
    split_rngs=_split_rngs_args,
    axis_name='batch',
    methods={
        '__call__': {
            'in_axes': (0, 0, 0, 0),
            'out_axes': 0,
            'variable_axes': _variable_axes_args,
            'split_rngs': _split_rngs_args,
            'axis_name': 'batch',
        },
        '__call_rnn__': {
            'in_axes': (0, 0, 0, 0, 0, 0, 0, 0),  # hiddens_tuple, x_m, x_b, d_m, d_b, d_f, msg_ts, book_ts
            'out_axes': 0,
            'variable_axes': _variable_axes_args,
            'split_rngs': _split_rngs_args,
            'axis_name': 'batch',
        },
    },
)

## Repeat shorter sequences, instead of linear projection:

class PaddedLobPredModel(nn.Module):
    ssm: nn.Module
    d_output: int
    d_model: int
    d_book: int
    n_message_layers: int
    n_fused_layers: int
    n_book_pre_layers: int = 1
    n_book_post_layers: int = 1
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.message_encoder = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_message_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=True,
            vocab_size=self.d_output,
        )
        # applied to transposed message output to get seq len for fusion
        #self.message_out_proj = nn.Dense(self.d_model)
        self.book_encoder = LobBookModel(
            ssm=self.ssm,
            d_book=self.d_book,
            d_model=self.d_model,
            n_pre_layers=self.n_book_pre_layers,
            n_post_layers=self.n_book_post_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        # applied to transposed book output to get seq len for fusion
        #self.book_out_proj = nn.Dense(self.d_model)
        self.fused_s5 = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_fused_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        (L_m x d_input, L_b x [P+1]) input sequence tuple,
        combining message and book inputs.
        Args:
             x_m: message input sequence (L_m x d_input, 
             x_b: book state (volume series) (L_b x [P+1])
        Returns:
            output (float32): (d_output)
        """

        x_m = self.message_encoder(x_m, message_integration_timesteps)
        x_b = self.book_encoder(x_b, book_integration_timesteps)

        # repeat book input to match message length
        x_b = jnp.repeat(x_b, x_m.shape[0] // x_b.shape[0], axis=0)
        
        x = jnp.concatenate([x_m, x_b], axis=1)
        # TODO: again, check integration time steps make sense here
        x = self.fused_s5(x, jnp.ones(x.shape[0]))

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)

    def __call_rnn__(self, hiddens_tuple, x_m, x_b, d_m, d_b, d_f,
                     message_integration_timesteps, book_integration_timesteps):
        """
        Compute output with hidden state support for autoregressive inference.
        Uses repeat instead of projection for matching sequence lengths.

        Args:
            hiddens_tuple: tuple of (message_hiddens, book_hiddens, fused_hiddens, ema_state)
            x_m: message sequence (L_m,) tokens
            x_b: book sequence (L_b, d_book)
            d_m, d_b, d_f: reset signals (unused)
            message_integration_timesteps, book_integration_timesteps: unused
        Returns:
            new_hiddens_tuple: updated hidden states
            logits: output (d_output,)
        """
        hiddens_m, hiddens_b, hiddens_fused, ema_state = hiddens_tuple

        # Message encoder with hidden state
        hiddens_m, x_m = self.message_encoder.__call_rnn__(
            hiddens_m, x_m, d_m, message_integration_timesteps)

        # Book encoder with hidden state
        hiddens_b, x_b = self.book_encoder.__call_rnn__(
            hiddens_b, x_b, d_b, book_integration_timesteps)

        # Repeat book to match message length (instead of projection)
        x_b = jnp.repeat(x_b, x_m.shape[0] // x_b.shape[0], axis=0)

        # Concatenate
        x = jnp.concatenate([x_m, x_b], axis=1)

        # Fused S5 with hidden state
        hiddens_fused, x = self.fused_s5.__call_rnn__(
            hiddens_fused, x, d_f, jnp.ones(x.shape[0]))

        # Pool or take last
        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        # Decode
        x = self.decoder(x)
        logits = nn.log_softmax(x, axis=-1)

        new_hiddens_tuple = (hiddens_m, hiddens_b, hiddens_fused, ema_state)
        return new_hiddens_tuple, logits

    @staticmethod
    def initialize_carry(batch_size, hidden_size, n_message_layers,
                         n_book_pre_layers, n_book_post_layers,
                         n_fused_layers, h_size_ema):
        """Initialize hidden states for RNN mode."""
        message_hiddens = StackedEncoderModel.initialize_carry(
            batch_size, hidden_size, n_message_layers)
        book_hiddens = LobBookModel.initialize_carry(
            batch_size, hidden_size, n_book_pre_layers, n_book_post_layers)
        fused_hiddens = StackedEncoderModel.initialize_carry(
            batch_size, hidden_size, n_fused_layers)
        # EMA state: (running_avg, step_count)
        ema_state = (jnp.zeros((batch_size, 1, h_size_ema)),
                     jnp.ones((batch_size, 1, 1)))
        return (message_hiddens, book_hiddens, fused_hiddens, ema_state)


# Variable axes and split_rngs for BatchPaddedLobPredModel
_padded_variable_axes_args = {"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None}
_padded_split_rngs_args = {"params": False, "dropout": True}

# Here we call vmap to parallelize across a batch of input sequences
# Note: BatchPaddedLobPredModel supports both __call__ (default) and __call_rnn__ methods
BatchPaddedLobPredModel = nn.vmap(
    PaddedLobPredModel,
    in_axes=(0, 0, 0, 0),
    out_axes=0,
    variable_axes=_padded_variable_axes_args,
    split_rngs=_padded_split_rngs_args,
    axis_name='batch',
    methods={
        '__call__': {
            'in_axes': (0, 0, 0, 0),
            'out_axes': 0,
            'variable_axes': _padded_variable_axes_args,
            'split_rngs': _padded_split_rngs_args,
            'axis_name': 'batch',
        },
        '__call_rnn__': {
            'in_axes': (0, 0, 0, 0, 0, 0, 0, 0),  # hiddens_tuple, x_m, x_b, d_m, d_b, d_f, msg_ts, book_ts
            'out_axes': 0,
            'variable_axes': _padded_variable_axes_args,
            'split_rngs': _padded_split_rngs_args,
            'axis_name': 'batch',
        },
    },
)
