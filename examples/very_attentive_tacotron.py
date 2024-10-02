# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the decoder layer of Very Attentive Tacotron.

Very Attentive Tacotron:
Robust Length Generalization in Transformer-based Text-to-Speech.
"""


from collections import abc
import dataclasses
import functools
import math
from typing import Sequence

import numpy as np
from sequence_layers import tensorflow as sl
from sequence_layers.tensorflow import proto as slp
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


# Value used to bias the attention logits for excluded positions.
ATTENTION_MASK_BIAS = -1e9
# Epsilon value to prevent log underflow.
LOG_EPS = 1e-16
# Constants key for alignment position.
ALIGNMENT_POSITION = 'alignment_position'


@dataclasses.dataclass
class InterpolatedRelativePositionBiasesConfig:
  """Configuration for InterpolatedRelativePositionBiases."""

  # Number of buckets to use for the relative position bias matrix.
  num_buckets: int
  # Maximum relative distance to support. All distances above this value are
  # mapped to the same bucket.
  max_distance: int
  # Scale of value to subtract from the output position biases for relative
  # positions that exceed max_distance. The subtracted value is equal to:
  # `max_distance_penalty * (abs(relative_position) - max_distance)`.
  max_distance_penalty: float
  # pyformat: disable
  # Bias matrix initialization scheme. One of:
  # * 'constant': Initialize all biases to `init_scheme_value`.
  # * 'gaussian_window_stddev': Initialize biases with values drawn from a
  #   Gaussian distribution centered at relative position 0, with standard
  #   deviation `init_scheme_value`.
  # * 'truncated_normal_stddev': Initialize biases with values drawn from a
  #   truncated normal distribution centered at relative position 0, with
  #   standard deviation `init_scheme_value`.
  # pyformat: enable
  init_scheme: str
  # Value to use for the bias matrix initialization scheme.
  init_scheme_value: float


class PreprocessConstants(abc.ABC):
  """sl.SequenceLayer mix-in for preprocess_constants support."""

  @abc.abstractmethod
  def preprocess_constants(self, constants: sl.Constants) -> None:
    """Preprocess constants and stash resulting Tensors internally.

    This method allows a SequenceLayer to precompute and stash constant-derived
    values so they don't need to be recomputed on each call to step.

    Args:
      constants: Constants to preprocess.
    """

  @abc.abstractmethod
  def clear_preprocessed_constants(self) -> None:
    """Clear preprocessed constants.

    This method should remove any internally stashed preprocessed constants when
    they are no longer needed in the current graph (or tf.function). This is to
    prevent Tensors from crossing from one graph to another (which results in
    an error.)
    """


class SequenceLayerBlock(sl.Serial):
  """Small hack to allow underlying layers to share the common namescope."""

  def __init__(self, name: str | None = None):
    super().__init__(layers=[], name=name)

  def _set_layers(self, layers: Sequence[sl.SequenceLayer]):
    """Call this at the end of the child constructor."""
    if self._layers:
      raise ValueError('layers has already been set.')
    # Copy the list.
    self._layers = layers[:]


class GaussianWindowBiasInitializer(tf.keras.initializers.Initializer):
  """Gaussian window initializer for InterpolatedRelativePositionBiases.

  This returns the normalized logits of a Gaussian window.
  """

  def __init__(self, stddev: float, bidirectional: bool):
    """Construct GaussianWindowBiasInitializer.

    Args:
      stddev: The standard deviation of the Gaussian window.
      bidirectional: Whether the position biases are bidirectional. This
        determines where the window is centered.
    """
    self.bidirectional = bidirectional
    self.stddev = stddev

  def __call__(self, shape, dtype=tf.float32, **kwargs):
    num_buckets, num_heads = shape

    # Work around missing bfloat16 support in tf.range.
    if dtype == tf.bfloat16:
      bucket_inds = tf.cast(
          tf.range(num_buckets, dtype=tf.float32), tf.bfloat16
      )
    else:
      bucket_inds = tf.range(num_buckets, dtype=dtype)
    if self.bidirectional:
      offset = num_buckets // 2
    else:
      offset = 0

    logits = -0.5 * tf.square((bucket_inds - offset) / self.stddev)
    logits -= tf.math.reduce_logsumexp(logits)  # Normalize logits.

    biases = tf.tile(logits[:, tf.newaxis], [1, num_heads])
    return biases


class InterpolatedRelativePositionBiases(sl.RelativePositionEmbedding):
  """InterpolatedRelativePositionBiases.

  This is an extension of T5's relative position biases that supports
  non-integer relative positions. This is done by interpolating between
  bias values for adjacent integer bins. Additionally, interpolation is used
  to compute the bias values for the "non-exact" logarithmically spaced bins
  (as an alternative to flooring to the nearest bin).

  The motivation behind this scheme is to support location-relative
  cross-attention where the "query" position is differentiable; and therefore,
  must be continuous.

  This also supports the SequenceLayer RelativePositionEmbedding interface.
  """

  def __init__(
      self,
      num_heads: int,
      num_buckets: int,
      max_distance: int,
      max_distance_penalty: float,
      bidirectional: bool,
      initializer: str,
      name: str | None = None,
  ):
    """Construct InterpolatedRelativePositionBiases.

    Args:
      num_heads: Number of biases to produce for each relative position.
      num_buckets: Number of relative position buckets to use.
      max_distance: Maximum relative distance to support. All distances above
        this value are mapped to the same bucket.
      max_distance_penalty: Scale of value to subtract from the output position
        biases for relative positions that exceed max_distance. The subtracted
        value is equal to: `max_distance_penalty * (abs(relative_position) -
        max_distance)`.
      bidirectional: Whether biases should be produced for both positive and
        negative relative positions. If False, positive relative positions
        (where key_position > query_position) are not supported.
      initializer: Initializer for the biases (embedding) matrix.
      name: Module name.

    If bidirectional is True, the buckets are evenly split between positive and
    negative relative positions; if False, all of the buckets are used for
    negative relative positions (and zero). In either case, half of the buckets
    are used for "exact" relative positions and the other half are used for
    logarithmically-spaced buckets of increasing size.

    Positive relative positions greater than or equal to max_distance are
    mapped to the same bias bucket. Negative relative positions less than or
    equal to -max_distance are mapped to the same bias bucket.
    """
    super().__init__(name=name)
    self.num_heads = num_heads
    self.num_buckets = num_buckets
    self.bidirectional = bidirectional
    self.max_distance = max_distance
    self.max_distance_penalty = max_distance_penalty
    with self.name_scope:
      self.bias_matrix = tf.keras.layers.Embedding(
          input_dim=self.num_buckets,
          output_dim=self.num_heads,
          embeddings_initializer=initializer,
          use_one_hot_matmul=True,  # Avoid gathers on TPU.
          name='bias_matrix',
      )

  @classmethod
  def from_config(
      cls,
      num_heads: int,
      bidirectional: bool,
      config: InterpolatedRelativePositionBiasesConfig,
  ) -> 'InterpolatedRelativePositionBiases':

    if config.init_scheme is None or config.init_scheme == 'constant':
      initializer = tf.keras.initializers.Constant(config.init_scheme_value)
    elif config.init_scheme == 'gaussian_window_stddev':
      initializer = GaussianWindowBiasInitializer(
          config.init_scheme_value, bidirectional=bidirectional
      )
    elif config.init_scheme == 'truncated_normal_stddev':
      initializer = tf.keras.initializers.TruncatedNormal(
          stddev=config.init_scheme_value
      )
    else:
      raise NotImplementedError(f'Unknown {config.init_scheme=}')

    return cls(
        num_heads=num_heads,
        num_buckets=config.num_buckets,
        max_distance=config.max_distance,
        max_distance_penalty=config.max_distance_penalty,
        bidirectional=bidirectional,
        initializer=initializer,
    )

  def _relative_position_bucket(self, relative_position):
    """Translate relative position to a continuous bucket value.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.

    If bidirectional=False, then positive relative positions are invalid.

    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.

    All relative positions >=max_distance map to the same bucket.

    All relative positions <=-max_distance map to the same bucket.

    Args:
      relative_position: a float32 Tensor

    Returns:
      a Tensor with the same shape as relative_position, containing float32
      values in the range [0, num_buckets)
    """
    is_positive = None
    n = -relative_position
    if self.bidirectional:
      buckets_per_side = self.num_buckets // 2
      is_positive = tf.math.less(n, 0)  # n = -relative_position
      n = tf.math.abs(n)
    else:
      buckets_per_side = self.num_buckets
      n = tf.math.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = buckets_per_side // 2
    is_small = tf.math.less(n, max_exact)
    # Note that `(buckets_per_side - 1 - max_exact)` below differs from the
    # reference implementation of T5 relative position biases which uses
    # `(buckets_per_side - max_exact)`, and therefore doesn't produce the
    # max_distance behavior described in the docstring above.
    val_if_large = max_exact + (
        tf.math.log(n / max_exact + LOG_EPS)
        / math.log(self.max_distance / max_exact)
        * (buckets_per_side - 1 - max_exact)
    )
    val_if_large = tf.math.minimum(val_if_large, buckets_per_side - 1)
    val = tf.where(is_small, n, val_if_large)
    if self.bidirectional:
      # Relative position `0` is centered in the bias matrix.
      val = tf.where(
          is_positive, buckets_per_side + val, buckets_per_side - val
      )
    return val

  @tf.Module.with_name_scope
  def __call__(
      self, query_positions: tf.Tensor, key_positions: tf.Tensor
  ) -> tf.Tensor:
    """Get biases from position indices.

    Args:
      query_positions: [batch, query_length] int/float Tensor containing query
        position indices.
      key_positions: [batch, key_length] int/float Tensor containing position
        indices.

    Returns:
      position_biases: [batch, num_heads, query_length, key_length] Tensor
        containing head-wise biases for each query-key pair.
    """
    compute_dtype = utils.compute_dtype()
    # [batch, query_length, 1]
    query_positions = tf.cast(query_positions[:, :, tf.newaxis], compute_dtype)
    # [batch, 1, key_length]
    key_positions = tf.cast(key_positions[:, tf.newaxis, :], compute_dtype)
    # [batch, query_length, key_length]
    relative_positions = key_positions - query_positions
    bucket_val = self._relative_position_bucket(relative_positions)

    # Interpolate between adjacent buckets.
    bucket_ind_low = tf.math.floor(bucket_val)
    bucket_ind_high = tf.math.ceil(bucket_val)
    # [batch, query_length, key_length, 1]
    high_weight = tf.expand_dims(bucket_val - bucket_ind_low, axis=-1)
    # [batch, query_length, key_length, num_heads]
    biases_low = self.bias_matrix(tf.cast(bucket_ind_low, tf.int32))
    biases_high = self.bias_matrix(tf.cast(bucket_ind_high, tf.int32))
    biases = (1.0 - high_weight) * biases_low + high_weight * biases_high

    # Apply optional max_distance_penalty.
    if self.max_distance_penalty != 0.0:
      # [batch, query_length, key_length]
      excess_distance = tf.maximum(
          0.0, tf.abs(relative_positions) - self.max_distance
      )
      penalty_amount = excess_distance * self.max_distance_penalty
      # [batch, query_length, key_length, num_heads]
      biases -= penalty_amount[:, :, :, tf.newaxis]

    # -> [batch, num_heads, query_length, key_length]
    position_biases = tf.transpose(biases, [0, 3, 1, 2])

    return position_biases

  @tf.Module.with_name_scope
  def get_position_bias_raw(
      self,
      queries_position: tf.Tensor,
      queries_length: tf.Tensor | int,
      keys_position: tf.Tensor,
      keys_length: tf.Tensor | int,
  ) -> tf.Tensor:
    """Computes relative self-attention position biases for absolute positions.

    This method computes relative position biases for absolute query / key
    positions and lengths.

    Args:
      queries_position: Scalar integer query absolute position.
      queries_length: Number of query timesteps to produce relative position
        embeddings for.
      keys_position: Scalar integer key absolute position.
      keys_length: Number of key timesteps to produce relative position
        embeddings for.

    Returns:
      A tensor of relative position biases broadcastable to
      [batch, queries_length, num_heads, keys_length].
    """
    queries_positions = (
        queries_position + tf.range(queries_length)[tf.newaxis, :]
    )
    keys_positions = keys_position + tf.range(keys_length)[tf.newaxis, :]

    position_biases = self(queries_positions, keys_positions)
    position_biases = tf.transpose(position_biases, [0, 2, 1, 3])
    return position_biases

  @tf.Module.with_name_scope
  def get_position_bias_streaming(
      self, queries: tf.Tensor, keys: tf.Tensor, queries_position: tf.Tensor
  ) -> tf.Tensor:
    """Computes relative self-attention position biases for streaming queries.

    This method computes relative position biases for a block of queries_time
    timesteps of the overall queries tensor, and the keys/values available are
    always the queries within the current queries_time block, and a fixed
    "max_previous" trailing window of timesteps.

    Args:
      queries: [batch, queries_time, num_heads, units_per_head] queries.
        queries_time is the number of streaming steps we are taking at once.
      keys: [batch, queries_time + max_previous, num_heads, units_per_head]
        keys.
      queries_position: scalar integer indicating the current decode position of
        queries[:, 0, :]. For example, this value is n * queries_time for the
        n'th queries_time block we process.

    Returns:
      A tensor of relative position biases broadcastable to
      [batch, num_heads, queries_time, keys_time].
    """
    queries_time_static = queries.shape.dims[1].value
    keys_time_static = keys.shape.dims[1].value
    queries_time = sl.utils.smart_dimension_size(queries, 1)
    keys_time = sl.utils.smart_dimension_size(keys, 1)

    # Add singleton batch dimension to context/memory_position.
    context_position = queries_position + tf.range(queries_time)[tf.newaxis, :]
    # keys[:, 0, :]'s absolute position is queries_position - max_previous.
    max_previous = keys_time - queries_time
    memory_start_position = queries_position - max_previous
    memory_position = memory_start_position + tf.range(keys_time)[tf.newaxis, :]

    values = self.__call__(context_position, memory_position)
    values.shape.assert_is_compatible_with(
        [1, self.num_heads, queries_time_static, keys_time_static]
    )
    return values

  @tf.Module.with_name_scope
  def get_position_bias(self, queries: tf.Tensor) -> tf.Tensor:
    """Computes relative self-attention position biases for queries.

    Args:
      queries: [batch, queries_time, num_heads, units_per_head] queries.

    Returns:
      A tensor of relative position biases broadcastable to
      [batch, num_heads, queries_time, keys_time].
    """
    queries_time_static = queries.shape.dims[1].value
    queries_time = sl.utils.smart_dimension_size(queries, 1)
    # Add singleton batch dimension.
    context_position = tf.range(queries_time)[tf.newaxis, :]
    values = self.__call__(context_position, context_position)
    values.shape.assert_is_compatible_with(
        [1, self.num_heads, queries_time_static, queries_time_static]
    )
    return values


def _inverse_softplus(x: tf.Tensor) -> tf.Tensor:
  return np.log(np.exp(x) - 1.0)


@dataclasses.dataclass
class AlignmentLayerConfig:
  """Configuration for AlignmentLayer."""

  # Initial delta for the alignment position.
  initial_delta: float = 0.25
  # Number of RNN units for the alignment LSTM sublayer.
  alignment_rnn_units: int = 96
  # Number of heads for the alignment sublayer.
  num_heads: int = 4
  # Number of units per head for the alignment sublayer.
  head_dim: int = 32
  # Configuration for the cross-attention bias.
  cross_attention_bias: InterpolatedRelativePositionBiasesConfig = (
      dataclasses.field(
          default_factory=InterpolatedRelativePositionBiasesConfig(
              num_buckets=32,
              max_distance=128,
              max_distance_penalty=0.0,
              init_scheme='gaussian_window_stddev',
              init_scheme_value=5.0,
          )
      )
  )
  # Scale factor for alignment position output.
  output_scale: float = 1.0


class AlignmentLayer(sl.Emitting, PreprocessConstants):
  """AlignmentLayer.

  Emits a differentiable monotonic alignment position. This can be used to
  provide the "query" position in a cross-attention mechanism that uses relative
  position biases.

  The order of operations is as follows:
  1. The alignment position is initialized to zero.
  2. For the current time step:
    a. A multi-head context vector is computed for the current alignment
      position using a purely location-relative mechanism (using no query-key
      comparisons) based on T5 relative position biases.
    b. The input to this layer and the context vector are concatenated and fed
      into an RNN.
    c. To produce the updated alignment position:
      - The output of the MLP is converted to a positive scalar using a
        softplus layer.
      - The positive float is added to the current alignment position to produce
        the updated alignment position.
  3. Repeat #2 for all time steps in the input sequence.
  4. RNN outputs are returned as layer outputs and the alignment position
    Sequence is returned as layer emits.
  """

  def __init__(
      self,
      source_name: str,
      config: AlignmentLayerConfig,
      dropout_rate: float,
      name: str | None = None,
  ):

    super().__init__(name=name)
    self.config = config
    self.source_name = source_name
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim

    # This is set by preprocess_constants().
    self._source_head_values = None

    with self.name_scope:
      # Source value-projection for cross-attention.
      self.source_value_projection = tf.keras.layers.Dense(
          units=config.num_heads * config.head_dim,
          use_bias=False,
          name='source_value_projection',
      )
      # Position biases for cross-attention.
      self.position_embeddings = InterpolatedRelativePositionBiases.from_config(
          num_heads=config.num_heads,
          bidirectional=True,
          config=config.cross_attention_bias,
      )
      self.sublayer = sl.RNN(
          tf.keras.layers.LSTMCell(config.alignment_rnn_units),
          default_to_identity=True,
      )
      # Compute initial bias from initial_delta using inverse softplus.
      if config.initial_delta <= 0.0:
        raise ValueError('initial_delta must be positive')
      initial_output_bias = _inverse_softplus(config.initial_delta)
      self.delta_output_layer = tf.keras.layers.Dense(
          units=1,
          activation='softplus',  # Keeps the output positive.
          kernel_initializer='zeros',
          bias_initializer=tf.constant_initializer(initial_output_bias),
          name='delta_output_layer',
      )
      self.dropout = tf.keras.layers.Dropout(
          rate=dropout_rate, name='attention_dropout'
      )

  @tf.Module.with_name_scope
  def preprocess_constants(self, constants: sl.Constants) -> None:
    source = self._get_source(constants)
    batch_size, source_len, _ = sl.utils.smart_dimension_size(source.values)
    # Precompute self._source_head_values.
    source_head_values = tf.reshape(
        self.source_value_projection(source.values),
        [batch_size, source_len, self.num_heads, self.head_dim],
    )
    self._source_head_values = sl.Sequence(
        source_head_values, source.mask
    ).mask_invalid()

  @tf.Module.with_name_scope
  def clear_preprocessed_constants(self) -> None:
    self._source_head_values = None

  def _get_source_head_values(self) -> sl.Sequence:
    """Get [bs, source_len, num_heads, head_dim] Sequence."""
    if self._source_head_values is None:
      raise ValueError(
          'source_head_values must be precomputed using preprocess_constants().'
      )
    return self._source_head_values

  def _get_source(self, constants: sl.Constants) -> sl.Sequence:
    if constants is None:
      constants = {}
    source = constants.get(self.source_name)
    if not isinstance(source, sl.Sequence):
      raise ValueError('constants[source_name] must contain an sl.Sequence')
    return source

  def _compute_context(
      self, x: sl.Sequence, alignment_position: tf.Tensor, training: bool
  ) -> sl.Sequence:
    """Compute bias-only attention context (no query-key comparisons)."""
    source_head_values = self._get_source_head_values()

    # Compute position biases.
    batch_size, source_len, _, _ = sl.utils.smart_dimension_size(
        source_head_values.values
    )
    source_position = tf.tile(
        tf.range(source_len)[tf.newaxis, :], [batch_size, 1]
    )
    # [bs, num_heads, query_len=1, source_len]
    position_biases = self.position_embeddings(
        alignment_position, source_position
    )
    # [bs, num_heads, source_len]
    position_biases = tf.squeeze(position_biases, axis=2)

    # Compute attention bias mask.
    # [bs, num_heads=1, source_len] (broadcast across heads).
    attention_mask = source_head_values.mask[:, tf.newaxis, :]
    attention_bias_mask = (1.0 - attention_mask) * ATTENTION_MASK_BIAS
    assert attention_bias_mask.dtype == tf.float32

    compute_dtype = utils.compute_dtype()

    # Compute attention weights in float32, then downcast to compute_dtype.
    # [bs, num_heads, source_len]
    scores = tf.cast(position_biases, tf.float32) + attention_bias_mask
    weights = tf.nn.softmax(scores, axis=-1)
    weights = tf.cast(weights, compute_dtype)
    weights = self.dropout(weights, training=training)

    # Compute context vectors.
    # b=batch_size, s=source_len, n=num_heads, d=head_dim.
    context = tf.einsum('bns,bsnd->bnd', weights, source_head_values.values)
    # Add time dim, flatten heads.
    context = tf.reshape(
        context, [batch_size, 1, self.num_heads * self.head_dim]
    )
    context = sl.Sequence(context, x.mask).mask_invalid()

    return context

  def _compute_alignment_deltas(
      self, sublayer_output: sl.Sequence, training: bool
  ) -> tf.Tensor:
    # Here, T = 1, because we only compute deltas for single steps.
    net = tf.ensure_shape(sublayer_output.values, [None, 1, None])
    # [B, T=1, D]
    net = self.delta_net(net, training=training)
    # [B, T=1, 1]
    net = self.delta_output_layer(net, training=training)
    # [B, T=1]
    net = tf.squeeze(net, axis=2)
    # Set invalid deltas to zero.
    deltas = sl.Sequence(net, sublayer_output.mask).mask_invalid()
    # [B, T=1]
    return deltas.values

  def _single_step_with_emits(
      self,
      x: sl.Sequence,
      state: sl.State,
      training: bool,
      constants: sl.Constants = None,
  ) -> tuple[sl.Sequence, sl.State, sl.Emits]:

    alignment_position, sublayer_state = state

    context = self._compute_context(x, alignment_position, training)
    sublayer_input = sl.Sequence(
        tf.concat([x.values, context.values], axis=-1), x.mask
    )

    outputs, sublayer_state = self.sublayer.step(
        sublayer_input, sublayer_state, training, constants
    )
    alignment_position += self._compute_alignment_deltas(outputs, training)

    state = (alignment_position, sublayer_state)
    emits = sl.Sequence(alignment_position, x.mask).mask_invalid()

    return outputs, state, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: sl.Sequence,
      training: bool,
      initial_state: sl.State = None,
      constants: sl.Constants = None,
  ) -> tuple[sl.Sequence, sl.Emits]:
    if initial_state is None:
      initial_state = self.get_initial_state(x, constants)
    output, _, alignment_position = self.step_with_emits(
        x, initial_state, training, constants, unroll=False
    )
    return output, alignment_position

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: sl.Sequence,
      state: sl.State,
      training: bool,
      constants: sl.Constants = None,
      unroll: bool = True,
  ) -> tuple[sl.Sequence, sl.State, sl.Emits]:

    # block_size = 1
    num_blocks = sl.utils.smart_dimension_size(x.values, 1)

    step_fn = functools.partial(
        self._single_step_with_emits, training=training, constants=constants
    )

    if unroll:
      return sl.utils.step_by_step_fn_static(step_fn, num_blocks, 1, x, state)
    else:
      output_spec = self.sublayer.get_output_spec_for_sequence(x, constants)
      emit_specs = self.get_emit_specs_for_sequence(x, constants)
      return sl.utils.step_by_step_fn_dynamic(
          step_fn,
          num_blocks,
          x.channel_shape,
          output_spec,
          emit_specs,
          num_blocks,
          1,
          1,
          x,
          state,
      )

  @tf.Module.with_name_scope
  def get_emit_specs(
      self, input_spec: tf.TensorSpec, constants: sl.Constants = None
  ) -> sl.Sequence:
    del constants
    # Scalar alignment position Sequence is emitted.
    return sl.Sequence(
        tf.TensorSpec(tf.TensorShape([]), input_spec.dtype),
        tf.TensorSpec(tf.TensorShape([]), sl.MASK_DTYPE),
    )

  @tf.Module.with_name_scope
  def get_initial_state(
      self, x: sl.Sequence, constants: sl.Constants = None
  ) -> sl.State:
    batch_size = sl.utils.smart_dimension_size(x.values, 0)
    compute_dtype = utils.compute_dtype()

    # Initial alignment position is zero.
    alignment_position = tf.zeros([batch_size, 1], compute_dtype)

    # Note: technically the shape is wrong for passing this on to sublayer
    # since we concat with context. In practice, the layer is an RNN
    # which does not care about input size for computing state size.
    x = x.apply_values(tf.cast, compute_dtype)

    sublayer_state = self.sublayer.get_initial_state(x, constants)
    return (alignment_position, sublayer_state)

  @tf.Module.with_name_scope
  def get_output_shape(
      self, input_shape: tf.TensorShape, constants: sl.Constants = None
  ) -> tf.TensorShape:
    return self.sublayer.get_output_shape(input_shape, constants)


class RelativeCrossAttention(sl.DotProductAttention):
  """RelativeCrossAttention.

  This computes: y = cross_attention(x, source, alignment_position).
  Here, cross_attention computes multi-head attention between a query sequence
  and a source sequence. It uses interpolated relative position biases that are
  computed from externally provided alignment positions for the query positions
  and basic time indices for the source positions.

  The constants dict must contain the following:
    constants[source_name] - A [batch, source_len, source_dim] source Sequence
      to attend to.
    constants[ALIGNMENT_POSITION] - A [batch, input_len] query alignment
      position Tensor.

  Note that because constants[ALIGNMENT_POSITION] contains a Tensor that is to
  be used synchronously with the input Sequence, this layer doesn't fully obey
  the SequenceLayer contract and shouldn't be used as a general-purpose
  SequenceLayer.
  """

  def __init__(
      self,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      relative_position_biases: InterpolatedRelativePositionBiases,
      attention_probabilities_dropout_rate: float,
      broadcast_dropout_across_queries: bool = False,
      use_bias: bool = False,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name: str | None = None,
  ):
    """Construct RelativeCrossAttention."""
    super().__init__(
        source_name=source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        attention_probabilities_dropout_rate=attention_probabilities_dropout_rate,
        broadcast_dropout_across_queries=broadcast_dropout_across_queries,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name,
    )
    self._relative_position_biases = relative_position_biases

  def _get_constants(
      self, constants: sl.Constants
  ) -> tuple[sl.Sequence, tf.Tensor]:
    if constants is None:
      constants = {}
    source = constants.get(self._source_name)
    if not isinstance(source, sl.Sequence):
      raise ValueError('constants[source_name] must contain an sl.Sequence')
    alignment_position = constants.get(ALIGNMENT_POSITION)
    if not isinstance(alignment_position, tf.Tensor):
      raise ValueError('constants[ALIGNMENT_POSITION] must contain a tf.Tensor')
    return source, alignment_position

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: sl.Sequence,
      state: sl.State,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> tuple[sl.Sequence, sl.State, sl.Emits]:
    source, alignment_position = self._get_constants(constants)

    # alignment_position is used synchronously with the input.
    alignment_position.shape.assert_is_compatible_with(x.values.shape[:2])

    queries = self._get_queries(x)
    keys, values = self._get_key_values(source)

    if self._query_network:
      queries, state = self._query_network.step(
          queries.mask_invalid(),
          state,
          training=training,
          constants=constants,
      )

    if self._key_network:
      keys = self._key_network.layer(
          keys.mask_invalid(),
          training=training,
          constants=constants,
      )

    if self._value_network:
      values = self._value_network.layer(
          values.mask_invalid(),
          training=training,
          constants=constants,
      )

    y, emits = self._attention(
        queries,
        keys,
        values,
        logit_bias_fn=functools.partial(
            self._logit_bias_fn, alignment_position=alignment_position
        ),
        is_step=True,
        training=training,
    )

    return y, state, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: sl.Sequence,
      training: bool,
      initial_state: sl.State = None,
      constants: sl.Constants = None,
      is_step: bool = False,
  ) -> tuple[sl.Sequence, sl.Emits]:
    del initial_state  # Stateless.
    source, alignment_position = self._get_constants(constants)

    # alignment_position is used synchronously with the input.
    alignment_position.shape.assert_is_compatible_with(x.values.shape[:2])

    queries = self._get_queries(x)
    keys, values = self._get_key_values(source)

    if self._query_network:
      queries = self._query_network.layer(
          queries.mask_invalid(),
          training=training,
          constants=constants,
      )

    if self._key_network:
      keys = self._key_network.layer(
          keys.mask_invalid(),
          training=training,
          constants=constants,
      )

    if self._value_network:
      values = self._value_network.layer(
          values.mask_invalid(),
          training=training,
          constants=constants,
      )

    return self._attention(
        queries,
        keys,
        values,
        logit_bias_fn=functools.partial(
            self._logit_bias_fn, alignment_position=alignment_position
        ),
        is_step=False,
        training=training,
    )

  def _logit_bias_fn(
      self,
      query_time: tf.Tensor,
      query_length: tf.Tensor | int,
      key_time: tf.Tensor,
      key_length: tf.Tensor | int,
      alignment_position: tf.Tensor,
  ) -> tf.Tensor:
    """Returns logit biases [b, q, h, k] for the provided positions."""
    batch_size = sl.utils.smart_dimension_size(alignment_position, 0)
    query_position = tf.slice(
        alignment_position, [0, query_time], [-1, query_length]
    )

    source_position = key_time + tf.range(key_length)[tf.newaxis, :]
    source_position = tf.tile(source_position, [batch_size, 1])

    position_biases = self._relative_position_biases(
        query_position, source_position
    )
    position_biases = tf.transpose(position_biases, [0, 2, 1, 3])
    return position_biases


class AlignmentBlock(SequenceLayerBlock, PreprocessConstants):
  """AlignmentBlock that wraps AlignmentLayer with a residual connection."""

  def __init__(
      self,
      source_name: str,
      config: AlignmentLayerConfig,
      output_dim: int,
      dropout_rate: float,
      name: str | None = None,
  ):
    super().__init__(name=name)
    with self.name_scope:
      self.alignment_layer = AlignmentLayer(
          source_name, config, dropout_rate, name='alignment_layer'
      )
      if config.output_scale is not None:
        maybe_scale = [sl.Scale(config.output_scale, name='output_scale')]
      else:
        maybe_scale = []
      layer = sl.Residual(
          sl.Serial(
              [
                  sl.RMSNormalization(epsilon=1e-6, name='rms_normalization'),
                  self.alignment_layer,
                  sl.Dense(units=output_dim, use_bias=False, name='dense'),
                  sl.Dropout(rate=dropout_rate, noise_shape=[None, 1, None]),
              ]
              + maybe_scale,
              name='serial',
          ),
          name='residual',
      )
    self._set_layers([layer])

  def get_alignment_position(self, emits: sl.Emits) -> sl.Sequence:
    return emits['residual']['serial']['alignment_layer']

  @tf.Module.with_name_scope
  def preprocess_constants(self, constants: sl.Constants) -> None:
    self.alignment_layer.preprocess_constants(constants)

  @tf.Module.with_name_scope
  def clear_preprocessed_constants(self) -> None:
    self.alignment_layer.clear_preprocessed_constants()


class SelfAttentionBlock(SequenceLayerBlock):
  """Causal T5-style self-attention."""

  def __init__(
      self,
      output_dim: int,
      num_heads: int,
      head_dim: int,
      max_horizon: int,
      position_bias_config: InterpolatedRelativePositionBiasesConfig,
      dropout_rate: float,
      name: str | None = None,
  ):
    super().__init__(name=name)

    with self.name_scope:
      layer = sl.Residual(
          sl.Serial([
              sl.RMSNormalization(epsilon=1e-6, name='rms_normalization'),
              sl.DotProductSelfAttention(
                  num_heads=num_heads,
                  units_per_head=head_dim,
                  max_horizon=max_horizon,
                  use_relative_position_embedding=True,
                  relative_position_embedding=InterpolatedRelativePositionBiases.from_config(
                      num_heads=num_heads,
                      bidirectional=False,  # Causal self-attention.
                      config=position_bias_config,
                  ),
                  attention_probabilities_dropout_rate=dropout_rate,
                  broadcast_dropout_across_queries=True,
                  use_bias=False,
                  name='dot_product_self_attention',
              ),
              sl.Flatten(),
              sl.Dense(units=output_dim, use_bias=False, name='dense'),
              sl.Dropout(rate=dropout_rate, noise_shape=[None, 1, None]),
          ])
      )
    self._set_layers([layer])


class CrossAttentionBlock(SequenceLayerBlock):
  """Relative cross-attention block.

  This SequenceLayer is stateless.

  See RelativeCrossAttention for required constants.
  """

  def __init__(
      self,
      source_name: str,
      output_dim: int,
      num_heads: int,
      head_dim: int,
      position_bias_config: InterpolatedRelativePositionBiasesConfig,
      dropout_rate: float,
      output_scale: float | None = None,
      attention_logits_soft_cap: float | None = None,
      name: str | None = None,
  ):
    super().__init__(name=name)
    with self.name_scope:
      relative_position_embedding = (
          InterpolatedRelativePositionBiases.from_config(
              num_heads=num_heads,
              bidirectional=True,  # Non-causal cross-attention.
              config=position_bias_config,
          )
      )

      if output_scale is not None:
        maybe_scale = [sl.Scale(scale=output_scale, name='output_scale')]
      else:
        maybe_scale = []
      layer = sl.Residual(
          sl.Serial(
              [
                  sl.RMSNormalization(epsilon=1e-6, name='rms_normalization'),
                  RelativeCrossAttention(
                      source_name=source_name,
                      num_heads=num_heads,
                      units_per_head=head_dim,
                      relative_position_biases=relative_position_embedding,
                      attention_probabilities_dropout_rate=dropout_rate,
                      broadcast_dropout_across_queries=True,
                      use_bias=False,
                      name='relative_cross_attention',
                  ),
                  sl.Flatten(),
                  sl.Dense(units=output_dim, use_bias=False, name='dense'),
                  sl.Dropout(rate=dropout_rate, noise_shape=[None, 1, None]),
              ]
              + maybe_scale
          )
      )
    self._set_layers([layer])


class ProductOfDense(sl.Stateless):
  """T5-style product-of-dense feed-forward SequenceLayer."""

  def __init__(
      self,
      units: int,
      activations: Sequence[str] | None = None,
      use_bias: bool = False,
      name: str | None = None,
  ):
    """Construct ProductOfDense.

    Args:
      units: Number of output units.
      activations: List of activations to use for each underlying Dense layer.
        The point-wise product of the output of all the Dense layers is returned
        as the output.
      use_bias: Whether the underlying Dense layers use a bias.
      name: Name for this layer.
    """
    super().__init__(name=name)
    if not activations:
      activations = ['relu']
    self.units = units
    self.layers = []
    with self.name_scope:
      for i, activation in enumerate(activations):
        self.layers.append(
            tf.keras.layers.Dense(
                units,
                activation=activation,
                use_bias=use_bias,
                name=f'Dense_{i}',
            )
        )
      self.multiply = tf.keras.layers.Multiply()

  @tf.Module.with_name_scope
  def layer(
      self,
      x: sl.Sequence,
      training: bool,
      initial_state: sl.State | None = None,
      constants: sl.Constants | None = None,
  ) -> sl.Sequence:
    x.channel_shape.with_rank_at_least(1)
    output = self.multiply([layer(x.values) for layer in self.layers])
    return sl.Sequence(output, x.mask).mask_invalid()

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: sl.Constants | None = None,
  ) -> tf.TensorShape:
    input_shape.with_rank_at_least(1)
    return input_shape[:-1].concatenate(self.units)


class FeedForwardBlock(SequenceLayerBlock):
  """T5-style feed-forward block."""

  def __init__(
      self,
      hidden_dim: int,
      output_dim: int,
      activations: Sequence[str],
      dropout_rate: float,
      name: str | None = None,
  ):
    super().__init__(name=name)
    with self.name_scope:
      layer = sl.Residual(
          sl.Serial([
              sl.RMSNormalization(epsilon=1e-6, name='rms_normalization'),
              ProductOfDense(
                  units=hidden_dim,
                  activations=activations,
                  use_bias=False,
                  name='product_of_dense',
              ),
              sl.Dropout(rate=dropout_rate, noise_shape=[None, 1, None]),
              sl.Dense(units=output_dim, use_bias=False, name='dense'),
              sl.Dropout(rate=dropout_rate, noise_shape=[None, 1, None]),
          ])
      )
    self._set_layers([layer])


@dataclasses.dataclass
class DecoderBlockConfig:
  """Configuration for DecoderBlock."""

  num_heads: int = 8
  head_dim: int = 48
  hidden_dim: int = head_dim * num_heads
  alignment_rnn_units: int = 96
  max_past_horizon: int = 128

  self_attention_bias: InterpolatedRelativePositionBiasesConfig = (
      dataclasses.field(
          default_factory=lambda: InterpolatedRelativePositionBiasesConfig(
              num_buckets=32,
              max_distance=64,
              max_distance_penalty=1.0,
              init_scheme='truncated_normal_stddev',
              init_scheme_value=1.0,
          )
      )
  )

  cross_attention_bias: InterpolatedRelativePositionBiasesConfig = (
      dataclasses.field(
          default_factory=lambda: InterpolatedRelativePositionBiasesConfig(
              num_buckets=32,
              max_distance=64,
              max_distance_penalty=1.0,
              init_scheme='gaussian_window_stddev',
              init_scheme_value=5.0,
          )
      )
  )
  cross_attention_output_scale: float = 1.0
  feedforward_hidden_dim: int = hidden_dim * 4
  feedforward_activations: Sequence[str] = ['gelu']


class DecoderBlock(SequenceLayerBlock):
  """Self-attention, cross-attention, feed-forward block."""

  def __init__(
      self,
      source_name: str,
      config: DecoderBlockConfig,
      dropout_rate: float,
      name: str | None = None,
  ):
    super().__init__(name=name)
    with self.name_scope:
      layers = [
          SelfAttentionBlock(
              output_dim=config.hidden_dim,
              num_heads=config.num_heads,
              head_dim=config.head_dim,
              max_horizon=config.max_past_horizon,
              position_bias_config=config.self_attention_bias,
              dropout_rate=dropout_rate,
              name='self_attention_block',
          ),
          CrossAttentionBlock(
              source_name=source_name,
              output_dim=config.hidden_dim,
              num_heads=config.num_heads,
              head_dim=config.head_dim,
              position_bias_config=config.cross_attention_bias,
              dropout_rate=dropout_rate,
              output_scale=config.cross_attention_output_scale,
              name='cross_attention_block',
          ),
          FeedForwardBlock(
              hidden_dim=config.feedforward_hidden_dim,
              output_dim=config.hidden_dim,
              activations=config.feedforward_activations,
              dropout_rate=dropout_rate,
              name='feed_forward_block',
          ),
      ]
    self._set_layers(layers)


class DecoderBlockStack(SequenceLayerBlock):
  """Stack of DecoderBlocks with RMSNorm and Dropout."""

  def __init__(
      self,
      source_name: str,
      decoder_block_config: DecoderBlockConfig,
      num_decoder_blocks: int,
      dropout_rate: float,
      name: str | None = None,
  ):
    super().__init__(name=name)
    with self.name_scope:
      decoder_blocks = sl.Serial([
          DecoderBlock(
              source_name,
              decoder_block_config,
              dropout_rate,
              name=f'decoder_block_{i}',
          )
          for i in range(num_decoder_blocks)
      ])
      layers = [
          decoder_blocks,
          sl.RMSNormalization(epsilon=1e-6, name='rms_normalization'),
          sl.Dropout(rate=dropout_rate, noise_shape=[None, 1, None]),
      ]
    self._set_layers(layers)


@dataclasses.dataclass
class VeryAttentiveDecoderConfig:
  """Configuration for VeryAttentiveDecoder."""

  name: str | None
  # The name of the source sequence to use for cross-attention.
  source_name: str = 'text_encoder_top'
  # Configuration for the alignment layer.
  alignment_layer: AlignmentLayerConfig
  # Configuration for the decoder block stack.
  decoder_block: DecoderBlockConfig
  # Number of decoder blocks in the decoder block stack.
  num_decoder_blocks: int = 6
  # Dropout rate for the decoder.
  dropout_rate: float = 0.1


class VeryAttentiveDecoder(sl.Emitting, PreprocessConstants):
  """VeryAttentiveDecoder.

  This SequenceLayer composes the following operations:
  * Optional SequenceLayer prenet.
  * AlignmentBlock (which contains an AlignmentLayer that runs step-wise during
    training).
  * DecoderBlockStack (consisting of N DecoderBlocks).
  * Optional SequenceLayer postnet.
  """

  def __init__(self, config: VeryAttentiveDecoderConfig):
    """Construct VeryAttentiveDecoder.

    Args:
      config: VeryAttentiveDecoder config.
    """
    super().__init__(name=config.name or None)
    if not config.source_name:
      raise ValueError('source_name not defined.')
    self.config = config
    self._source_name = config.source_name

    with self.name_scope:
      self._build(config)

  def _build(self, config):
    with tf.name_scope('prenet'):
      self.prenet = sl.Conv1D(
          hidden_dim=config.decoder_block.hidden_dim, kernel_size=3
      )
    self.alignment_block = AlignmentBlock(
        source_name=config.source_name,
        config=config.alignment_layer,
        output_dim=config.decoder_block.hidden_dim,
        dropout_rate=config.dropout_rate,
        name='alignment_block',
    )
    self.decoder_block_stack = DecoderBlockStack(
        source_name=config.source_name,
        decoder_block_config=config.decoder_block,
        num_decoder_blocks=config.num_decoder_blocks,
        dropout_rate=config.dropout_rate,
        name='decoder_block_stack',
    )
    with tf.name_scope('postnet'):
      self.postnet = slp.build_sequence_layer(
          config.postnet, default_to_identity=True
      )

    # Wrap layers in an sl.Serial to use its initial state and output shape
    # helpers.
    self.serial_layer = sl.Serial([
        self.prenet,
        self.alignment_block,
        self.decoder_block_stack,
        self.postnet,
    ])

  @tf.Module.with_name_scope
  def preprocess_constants(self, constants: sl.Constants) -> None:
    self.alignment_block.preprocess_constants(constants)

  @tf.Module.with_name_scope
  def clear_preprocessed_constants(self) -> None:
    self.alignment_block.clear_preprocessed_constants()

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: sl.Sequence,
      training: bool,
      initial_state: sl.State = None,
      constants: sl.Constants = None,
  ) -> tuple[sl.Sequence, sl.Emits]:
    if initial_state is None:
      initial_state = self.get_initial_state(x, constants)

    (
        prenet_state,
        alignment_block_state,
        decoder_block_stack_state,
        postnet_state,
    ) = initial_state

    x = self.prenet.layer(x, training, prenet_state, constants)
    x, emits = self.alignment_block.layer_with_emits(
        x, training, alignment_block_state, constants
    )
    alignment_position = self.alignment_block.get_alignment_position(emits)
    # Feed alignment_position to decoder_block_stack.
    post_alignment_constants = constants | {
        ALIGNMENT_POSITION: alignment_position.values
    }
    x = self.decoder_block_stack.layer(
        x, training, decoder_block_stack_state, post_alignment_constants
    )
    outputs = self.postnet.layer(x, training, postnet_state, constants)
    emits = self._compute_emits(alignment_position, constants)

    return outputs, emits

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: sl.Sequence,
      state: sl.State,
      training: bool,
      constants: sl.Constants = None,
  ) -> tuple[sl.Sequence, sl.State, sl.Emits]:

    (
        prenet_state,
        alignment_block_state,
        decoder_block_stack_state,
        postnet_state,
    ) = state

    x, prenet_state = self.prenet.step(x, prenet_state, training, constants)
    x, alignment_block_state, emits = self.alignment_block.step_with_emits(
        x, alignment_block_state, training, constants
    )
    alignment_position = self.alignment_block.get_alignment_position(emits)
    # Feed alignment_position to decoder_block_stack.
    post_alignment_constants = constants | {
        ALIGNMENT_POSITION: alignment_position.values
    }
    x, decoder_block_stack_state = self.decoder_block_stack.step(
        x, decoder_block_stack_state, training, post_alignment_constants
    )
    outputs, postnet_state = self.postnet.step(
        x, postnet_state, training, constants
    )
    emits = self._compute_emits(alignment_position, constants)

    state = (
        prenet_state,
        alignment_block_state,
        decoder_block_stack_state,
        postnet_state,
    )

    return outputs, state, emits

  def _compute_emits(
      self, alignment_position: sl.Sequence, constants
  ) -> sl.Emits:
    """Compute soft alignment probabilities from alignment positions."""
    source = constants[self.config.source_name]
    source_time = sl.utils.smart_dimension_size(source.values, 1)

    # [batch, query_time]
    position_ind_low = tf.math.floor(alignment_position.values)
    position_ind_high = tf.math.ceil(alignment_position.values)
    high_weight = alignment_position.values - position_ind_low

    # [batch, query_time, source_time]
    compute_dtype = utils.compute_dtype()
    position_low = tf.one_hot(
        tf.cast(position_ind_low, tf.int32), source_time, dtype=compute_dtype
    )
    position_high = tf.one_hot(
        tf.cast(position_ind_high, tf.int32), source_time, dtype=compute_dtype
    )
    high_weight = high_weight[:, :, tf.newaxis]
    attention_probabilities = (
        1.0 - high_weight
    ) * position_low + high_weight * position_high

    # [batch, query_time, num_heads=1, source_time]
    attention_probabilities = attention_probabilities[:, :, tf.newaxis, :]
    attention_probabilities = sl.Sequence(
        attention_probabilities, alignment_position.mask
    ).mask_invalid()

    return sl.AttentionEmits(attention_probabilities)

  @tf.Module.with_name_scope
  def get_initial_state(
      self, x: sl.Sequence, constants: sl.Constants = None
  ) -> sl.State:
    return tuple(self.serial_layer.get_initial_state(x, constants))

  @tf.Module.with_name_scope
  def get_output_shape(
      self, input_shape: tf.TensorShape, constants: sl.Constants = None
  ) -> tf.TensorShape:
    return self.serial_layer.get_output_shape(input_shape, constants)

  @tf.Module.with_name_scope
  def get_emit_specs(
      self, input_spec: tf.TensorSpec, constants: sl.Constants = None
  ) -> sl.EmitSpecs:
    # We emit AttentionEmits with soft alignment position.
    source = constants[self.config.source_name]
    prenet_output_spec = self.prenet.get_output_spec(input_spec, constants)
    return sl.AttentionEmits(
        sl.Sequence(
            tf.TensorSpec(
                tf.TensorShape([1, source.values.shape[1]]),
                prenet_output_spec.dtype,
            ),
            tf.TensorSpec(tf.TensorShape([]), sl.MASK_DTYPE),
        )
    )
