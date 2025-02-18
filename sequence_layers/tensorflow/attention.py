# Copyright 2023 Google LLC
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
"""Attention layers."""

import abc
import fractions
import functools
import math
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
import scipy
from sequence_layers.tensorflow import dense
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


# A negative enough value such that it underflows to a hard zero in softmax.
_INVALID_LOGIT_VALUE = -1e9


def _ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.

  Copied from tensor2tensor.

  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.

  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.linalg.band_part(
        tf.ones([rows, cols]),
        tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64),
    )
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band


def _soft_cap_attention_logits(logits: tf.Tensor, cap: float) -> tf.Tensor:
  with tf.name_scope('soft_cap_attention_logits'):
    cap = tf.cast(cap, logits.dtype)
    return cap * tf.math.tanh(logits / cap)


def _mask_attention_logits(
    logits: tf.Tensor, invalid_mask: tf.Tensor
) -> tf.Tensor:
  with tf.name_scope('mask_attention_logits'):
    logits.shape.assert_has_rank(4)
    invalid_mask.shape.assert_has_rank(4)
    # Mask invalid timesteps, potentially broadcasting.
    # Adding can change the softmax output so replace values with tf.where.
    return tf.where(
        invalid_mask > 0.0, tf.cast(_INVALID_LOGIT_VALUE, logits.dtype), logits
    )


def _softmax_in_at_least_float32(logits: tf.Tensor) -> tf.Tensor:
  compute_dtype = logits.dtype
  assert compute_dtype.is_floating
  softmax_dtype = tf.float32 if compute_dtype.size < 4 else compute_dtype
  return tf.cast(tf.nn.softmax(tf.cast(logits, softmax_dtype)), compute_dtype)


def _validate_heads(num_heads: int, units_per_head: int, name: str):
  if num_heads <= 0:
    raise ValueError(f'Expected num_heads > 0 for {name}. Got {num_heads}')
  if units_per_head <= 0:
    raise ValueError(
        f'Expected units_per_head > 0 for {name}. Got {units_per_head}'
    )


def _validate_attention(
    source_name: str, num_heads: int, units_per_head: int, name: str
):
  if not source_name:
    raise ValueError(f'Expected non-empty source_name for {name}.')
  _validate_heads(num_heads, units_per_head, name)


def _get_source(
    layer: types.SequenceLayer, source_name: str, constants: types.Constants
) -> types.Sequence:
  """Gets the attention source from constants and does basic validation."""
  if constants is None:
    raise ValueError(
        f'{layer} requires the source to be provided via '
        f'constants, got: {constants}'
    )
  source = constants.get(source_name)
  if not source:
    raise ValueError(
        f'{layer} expected {source_name} to be present in '
        f'constants, got: {constants}'
    )
  if not isinstance(source, types.Sequence):
    raise ValueError(
        f'{layer} expected a Sequence for {source_name}, got: {source}'
    )
  if source.values.shape.rank != 3:
    raise ValueError(f'{layer} requires a rank 3 source, got: {source}')
  if source.values.shape.dims[2].value is None:
    raise ValueError(
        f'{layer} depends on source shape, but source has '
        f'unknown channels dimension: {source}'
    )
  return source


class AttentionEmits(NamedTuple):
  """A structure for emits produced by attention layers."""

  # The attention probabilities, generally shaped
  # [batch_size, query_time, num_heads, source_time].
  probabilities: types.Sequence


class AdditiveAttention(types.StatelessEmitting):
  """A multi-headed content-based attention layer."""

  def __init__(
      self,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      name: Optional[str] = None,
  ):
    super().__init__(name)
    _validate_attention(source_name, num_heads, units_per_head, name)
    self._source_name = source_name
    self._num_heads = num_heads
    self._units_per_head = units_per_head
    num_units = num_heads * units_per_head
    with self.name_scope as name_scope:
      self._attn_v = tf.Variable(
          lambda: tf.initializers.GlorotUniform()(
              [num_heads, units_per_head], dtype=utils.variable_dtype()
          ),
          name='v',
      )
      self._source_projection = tf.keras.layers.Dense(
          num_units, use_bias=True, name=name_scope + 'source_projection/'
      )
      self._query_projection = tf.keras.layers.Dense(
          num_units, use_bias=False, name=name_scope + 'query_projection/'
      )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'AdditiveAttention requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    source = _get_source(self, self._source_name, constants)
    return tf.TensorShape([self._num_heads, source.values.shape.dims[2].value])

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> AttentionEmits:
    source = _get_source(self, self._source_name, constants)
    return AttentionEmits(
        types.Sequence(
            tf.TensorSpec(
                tf.TensorShape(
                    [self._num_heads, source.values.shape.dims[1].value]
                ),
                input_spec.dtype,
            ),
            tf.TensorSpec(tf.TensorShape([]), tf.float32),
        )
    )

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, AttentionEmits]:
    source = _get_source(self, self._source_name, constants)
    num_heads = self._num_heads
    units_per_head = self._units_per_head
    query = x.apply_values(self._query_projection)
    batch_size, query_time = utils.smart_dimension_size(query.values, [0, 1])
    query = query.apply_values(
        tf.reshape, [batch_size, query_time, 1, num_heads, units_per_head]
    )

    keys = source.apply_values(self._source_projection).mask_invalid()
    keys_time = utils.smart_dimension_size(keys.values, 1)
    keys_values_for_broadcast = tf.reshape(
        keys.values, [batch_size, 1, keys_time, num_heads, units_per_head]
    )

    # Broadcast keys and queries to
    # [b, query_time, keys_time, num_heads, units_per_head]
    hidden = tf.tanh(keys_values_for_broadcast + query.values)

    logits = tf.einsum(
        'BijNH,NH->BiNj', hidden, tf.cast(self._attn_v, hidden.dtype)
    )
    # [b, query_time, num_heads, keys_time]
    mask = keys.mask[:, tf.newaxis, tf.newaxis, :]

    logits = tf.where(mask > 0.0, logits, _INVALID_LOGIT_VALUE)
    probabilities = _softmax_in_at_least_float32(logits)
    context_vector = tf.einsum('BiNj,BjS->BiNS', probabilities, source.values)
    emits = AttentionEmits(types.Sequence(probabilities, query.mask))
    return types.Sequence(context_vector, query.mask).mask_invalid(), emits


class _GmmCell(tf1.nn.rnn_cell.RNNCell):
  """Executes GMM attention step-by-step as an RNN."""

  def __init__(
      self, layer, output_shape, comb_weight_shape, state_size, training, source
  ):
    self._layer = layer
    self._output_shape = output_shape
    self._comb_weight_shape = comb_weight_shape
    self._state_size = state_size
    self._training = training
    self._source = source

  @property
  def output_size(self):
    return self._output_shape, self._comb_weight_shape

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, inputs, state):
    # Keras RNN state is quirky.
    return self.call(inputs, (state,), self._training)

  def call(self, query_2d, state, training):
    del training
    query = query_2d[:, tf.newaxis, :]
    (prev_position,) = state
    context_vector, comb_weights, new_position = self._layer.attention(
        self._source, query, prev_position
    )
    batch_size = utils.smart_dimension_size(context_vector, 0)
    context_vector = tf.reshape(
        context_vector,
        tf.concat([[batch_size], self._output_shape.as_list()], 0),
    )
    comb_weights = tf.reshape(
        comb_weights, tf.concat([[batch_size], self._comb_weight_shape], 0)
    )
    return (context_vector, comb_weights), new_position


class GmmAttention(types.Emitting):
  """A multi-headed Gaussian-mixture attention layer."""

  def __init__(
      self,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      num_components: int,
      monotonic: bool,
      init_offset_bias: float = 0.0,
      init_scale_bias: float = 0.0,
      max_offset: float = -1.0,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    _validate_attention(source_name, num_heads, units_per_head, name)
    if num_components <= 0:
      raise ValueError(
          f'Expected num_components > 0 for {name}. Got: {num_components}.'
      )

    self._source_name = source_name
    self._num_heads = num_heads
    self._units_per_head = units_per_head
    num_units = num_heads * units_per_head
    self._num_components = num_components
    self._monotonic = monotonic
    self._init_offset_bias = init_offset_bias
    self._init_scale_bias = init_scale_bias
    self._max_offset = max_offset
    self._built = False

    with self.name_scope as name_scope:
      self._mlp_hidden = tf.keras.layers.Dense(
          num_units, activation=tf.nn.relu, name=name_scope + 'gmm_mlp_hidden/'
      )
      # TODO(rryan): Pull this pattern into a "DenseShaped" layer?
      with tf.name_scope('gmm_mlp_output'):
        self._mlp_output_w = tf.Variable(
            lambda: tf.initializers.GlorotUniform()(
                [num_heads, units_per_head, num_components * 3],
                dtype=utils.variable_dtype(),
            ),
            name='kernel',
        )
        self._mlp_output_b = tf.Variable(
            lambda: tf.initializers.Zeros()(
                [1, 1, num_heads, num_components * 3],
                dtype=utils.variable_dtype(),
            ),
            name='bias',
        )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    if self._monotonic:
      batch_size = utils.smart_dimension_size(x.values, 0)
      # Start from zero positions.
      return tf.zeros(
          [batch_size, 1, self._num_heads, self._num_components],
          utils.compute_dtype(),
      )
    else:
      return ()

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'GmmAttention requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    source = _get_source(self, self._source_name, constants)
    return tf.TensorShape([self._num_heads, source.values.shape.dims[2].value])

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> AttentionEmits:
    source = _get_source(self, self._source_name, constants)
    return AttentionEmits(
        types.Sequence(
            tf.TensorSpec(
                tf.TensorShape(
                    [self._num_heads, source.values.shape.dims[1].value]
                ),
                input_spec.dtype,
            ),
            tf.TensorSpec(tf.TensorShape([]), tf.float32),
        )
    )

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
      unroll: bool = True,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    # Compute in parallel if stateless.
    if not self._monotonic:
      outputs, emits = self.layer_with_emits(x, training, state, constants)
      return outputs, state, emits

    source = _get_source(self, self._source_name, constants)

    state_size = tf.nest.map_structure(lambda t: t.shape[1:], state)
    output_shape: tf.TensorShape = self.get_output_shape_for_sequence(
        x, constants
    )
    source_time = utils.smart_dimension_size(source.values, 1)
    comb_weight_shape = tf.stack([self._num_heads, source_time])
    use_keras = False
    # TODO(rryan):  Reimplement using utils.step_by_step_{dynamic,static}
    # to harmonize with other parts of go/sequence-layers.
    cell = _GmmCell(
        self, output_shape, comb_weight_shape, state_size, training, source
    )
    if use_keras:
      rnn = tf.keras.layers.RNN(
          cell, return_sequences=True, return_state=True, unroll=unroll
      )
      result = rnn(
          x.values, mask=x.mask, initial_state=state, training=training
      )
      (values, comb_weights), state = result[0], result[1]
    else:
      if unroll:
        inputs = tf.unstack(x.values, axis=1)
        # To avoid creating complicated control flow for what is probably a
        # short sequence of frames, do not provide lengths to static_rnn. We are
        # masking the output values below, so this is safe.
        outputs, state = tf1.nn.static_rnn(cell, inputs, initial_state=state)
        values, comb_weights = zip(*outputs)
        values = tf.stack(values, axis=1)
        comb_weights = tf.stack(comb_weights, axis=1)
      else:
        (values, comb_weights), state = tf1.nn.dynamic_rnn(
            cell,
            x.values,
            initial_state=state,
            dtype=x.values.dtype,
            sequence_length=x.lengths(),
        )
    emits = AttentionEmits(types.Sequence(comb_weights, x.mask))
    return types.Sequence(values, x.mask).mask_invalid(), state, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    prev_position = initial_state
    if prev_position is None:
      prev_position = self.get_initial_state(x, constants)

    if self._monotonic:
      x, _, emits = self.step_with_emits(
          x, prev_position, training, constants, unroll=False
      )
      # step masks.
      return x, emits

    query = x
    source = _get_source(self, self._source_name, constants)
    context_vector, comb_weights, _ = self.attention(
        source, query.values, prev_position
    )
    emits = AttentionEmits(types.Sequence(comb_weights, query.mask))
    return types.Sequence(context_vector, query.mask).mask_invalid(), emits

  def attention(self, source, query_values, prev_position):
    """Computes GMM attention from query and previous position.

    Args:
      source: The source sequence to attend to.
      query_values: [batch_size, query_time, query_channels]
      prev_position: [batch_size, 1, num_heads, num_components]. Previous
        position of each mixture component per-head.

    Returns:
      context_vector: [batch_size, query_time, num_heads, source_dimension].
        The per-head context vector.
      comb_weights: [batch_size, query_time, num_heads, source_time]
        The per-head combination weights.
      position: [batch_size, 1, num_heads, num_components]. Current position
        of each mixture component per head.
    """
    num_heads = self._num_heads
    units_per_head = self._units_per_head

    batch_size, query_time = utils.smart_dimension_size(query_values, [0, 1])
    if self._monotonic and not isinstance(query_time, tf.Tensor):
      assert query_time == 1
    query_values = self._mlp_hidden(query_values)
    query_values = tf.reshape(
        query_values, [batch_size, query_time, num_heads, units_per_head]
    )

    # Per-head projection from units_per_head to num_components * 3.
    # Equivalent to tf.stack(
    # [tf.tensordot(query_values[:, :, i, :], mlp_output_w[i, :, :], 1)
    #  for i in range(num_heads)], axis=2)

    query_values = tf.einsum(
        'BiNH,NHM->BiNM',
        query_values,
        tf.cast(self._mlp_output_w, query_values.dtype),
    )
    # bias_add doesn't work for biases with rank > 1.
    query_values += tf.cast(self._mlp_output_b, query_values.dtype)

    # Each is [batch_size, query_time, num_heads, num_components].
    prior_logits, offset_logits, scale_logits = tf.split(
        query_values, 3, axis=3
    )

    offset_logits += self._init_offset_bias
    scale_logits += self._init_scale_bias
    assert prior_logits.shape.dims[-1].value == self._num_components

    # comb_weights: [b, query_time, num_heads, source_time]
    # new_position: [b, q, num_heads, num_components]
    comb_weights, new_position = self._v2_comb_weights(
        source, prior_logits, offset_logits, scale_logits, prev_position
    )

    # Expand source mask for broadcasting to
    # [b, query_time, num_heads, source_time].
    comb_weights *= tf.cast(
        source.mask[:, tf.newaxis, tf.newaxis, :], comb_weights.dtype
    )

    # [b, query_time, num_heads, source_time]
    # [b, source_time, source_dim]
    # -> [b, query_time, num_heads, source_dim]
    context_vector = tf.einsum('BiNj,BjS->BiNS', comb_weights, source.values)
    return context_vector, comb_weights, new_position

  def _v2_comb_weights(
      self, source, prior_logits, offset_logits, scale_logits, prev_position
  ):
    """Return comb weights computed using V2 method."""
    # V2 properties:
    # * Uses softmax for the mixture weights.
    # * Uses softplus for means and scales.
    # * Scale represents the standard deviation.
    priors = _softmax_in_at_least_float32(prior_logits)
    variances = tf.square(tf.nn.softplus(scale_logits))
    if self._max_offset > 0:
      # softplus(x) - softplus(x - M) gives a sigmoid that saturates outside
      # [0, M] and is approximately linear in between.
      position_offset = tf.nn.softplus(offset_logits) - tf.nn.softplus(
          offset_logits - self._max_offset
      )
    else:
      position_offset = tf.nn.softplus(offset_logits)

    comb_weights, new_position = self._eval_gmm_pdfs(
        source, priors, position_offset, variances, prev_position
    )
    return comb_weights, new_position

  def _eval_gmm_pdfs(
      self,
      source,
      priors,
      position_offset,
      variances,
      prev_position,
      normalize=True,
  ):
    """Evaluate the location GMMs on all encoder positions."""
    # priors, position_offset, variances are shaped
    # [b, query_time, num_heads, num_components].
    priors.shape.assert_is_compatible_with(
        [None, None, self._num_heads, self._num_components]
    )
    position_offset.shape.assert_is_compatible_with(
        [None, None, self._num_heads, self._num_components]
    )
    variances.shape.assert_is_compatible_with(
        [None, None, self._num_heads, self._num_components]
    )
    # prev_position is [b, 1, num_heads, num_components].
    if self._monotonic:
      prev_position.shape.assert_is_compatible_with(
          [None, 1, self._num_heads, self._num_components]
      )
      # If we're monotonic, then query time is always 1.
      position_offset.shape.assert_is_compatible_with(
          [None, 1, self._num_heads, self._num_components]
      )
      new_position = prev_position + position_offset
      new_position.shape.assert_is_compatible_with(
          [None, 1, self._num_heads, self._num_components]
      )
    else:
      new_position = position_offset

    # Expand all to [b, query_time, num_heads, source_time (1), num_components]
    priors = priors[:, :, :, tf.newaxis, :]
    means = new_position[:, :, :, tf.newaxis, :]
    variances = variances[:, :, :, tf.newaxis, :]

    # [1, 1, 1, source_timesteps, 1].
    source_length = utils.smart_dimension_size(source.values, 1)
    encoder_positions = tf.cast(tf.range(source_length), means.dtype)
    encoder_positions = encoder_positions[
        tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis
    ]

    if normalize:
      priors *= tf.math.rsqrt(2 * np.pi * variances + 1e-8)
    # Broadcast source time and query time.
    # encoder_positions is [1, 1, 1, source_time, 1]
    # means/priors/variances are
    # [batch, query_time, num_heads, 1, num_components]
    pdfs = priors * tf.exp(
        -((encoder_positions - means) ** 2) / (2 * variances + 1e-8)
    )
    # pdfs sized [batch, query_time, num_heads, source_time].
    pdfs = tf.reduce_sum(pdfs, 4)
    return pdfs, new_position


class RelativePositionEmbedding(tf.Module, metaclass=abc.ABCMeta):
  """Abstract base class for computing relative position biases."""

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
  def get_position_bias(self, queries: tf.Tensor) -> tf.Tensor:
    """Computes relative self-attention position biases for queries.

    Args:
      queries: [batch, queries_time, num_heads, units_per_head] queries.

    Returns:
      A tensor of relative position biases broadcastable to
      [batch, num_heads, queries_time, keys_time].
    """


class ShawRelativePositionEmbedding(RelativePositionEmbedding):
  """Computes query-dependent relative position embeddings.

  Based on:
  Self-Attention with Relative Position Representations
  https://arxiv.org/abs/1803.02155

  Warning: Implements the attention logits bias, but not the value biases
  described in the paper. Does not clip based on maximum distance (distances
  beyond max_backward/max_forward receive no relative position bias).

  Computes a [batch, num_heads, queries_time, keys_time] tensor of relative
  position biases, biasing the selection of keys for every query timestep.
  """

  def __init__(
      self,
      max_backward: int,
      max_forward: int,
      num_heads: int,
      units_per_head: int,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._max_forward = max_forward
    self._max_backward = max_backward
    self._num_heads = num_heads
    self._units_per_head = units_per_head

    if max_forward > 0:
      raise ValueError(f'{self} does not support forward relative biases yet.')

    if max_forward < 0 or max_backward < 0:
      raise ValueError(f'{max_forward=} and {max_backward=} must be >= 0')
    with self.name_scope:
      # Create an embedding vector for the max_backward previous timesteps and
      # the current timestep.
      # TODO(rryan): Support the ability to share embeddings across heads.
      self._relative_position_embedding = tf.Variable(
          lambda: tf.keras.initializers.VarianceScaling()(
              [max_backward + 1, num_heads, units_per_head],
              dtype=utils.variable_dtype(),
          ),
          name='embedding',
      )

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
    del queries_position
    assert self._max_forward == 0
    batch_size, queries_time = utils.smart_dimension_size(queries, [0, 1])
    queries_time_static = queries.shape.dims[1].value
    queries.shape.assert_is_compatible_with(
        [None, None, self._num_heads, self._units_per_head]
    )

    keys_time = utils.smart_dimension_size(keys, 1)
    expected_keys_time_static = (
        queries_time_static + self._max_backward
        if queries_time_static is not None
        else None
    )
    keys.shape.assert_is_compatible_with(
        [None, expected_keys_time_static, self._num_heads, self._units_per_head]
    )

    # First, we compute a
    # [batch_size, num_heads, queries_time, max_backward + 1]
    # tensor of relative position biases for each (query, relative) pair.
    relative_logits = tf.einsum(
        'jNH,BiNH->BNij',
        tf.cast(self._relative_position_embedding, queries.dtype),
        queries,
    )

    if queries_time_static == 1:
      # If queries_time_static is 1, there is no need to compute any coordinate
      # remappings, because keys_time is max_backward + 1 timesteps, and its
      # layout already matches the ordering of relative_logits.
      pass
    else:
      # Following the above example of the mask, if letters correspond to each
      # relative bias, and 'a' is the relative position bias for the current
      # queries_time timestep, then our goal for a given batch/head is the
      # following matrix:
      # f e d c b a 0 0
      # 0 f e d c b a 0
      # 0 0 f e d c b a
      #
      # In the above example, relative_logits is now this matrix:
      # f e d c b a
      # f e d c b a
      # f e d c b a
      #
      # To achieve the goal we use some reshaping trickery. We insert 3 zeros
      # at the end of each row:
      # f e d c b a 0 0 0
      # f e d c b a 0 0 0
      # f e d c b a 0 0 0
      pad = (queries_time - 1) + 1
      relative_logits = tf.pad(
          relative_logits, [[0, 0], [0, 0], [0, 0], [0, pad]]
      )

      # Then flatten, trim, and reshape into an 8x3 matrix, arriving at our
      # goal:
      # f e d c b a 0 0
      # 0 f e d c b a 0
      # 0 0 f e d c b a
      relative_logits = tf.reshape(
          relative_logits, [batch_size, self._num_heads, -1]
      )
      total = queries_time * keys_time
      relative_logits = relative_logits[:, :, :total]
      relative_logits = tf.reshape(
          relative_logits,
          [batch_size, self._num_heads, queries_time, keys_time],
      )

    # Shaw, et al. scale the relative logits by rsqrt(units_per_head).
    relative_logits *= tf.math.rsqrt(
        tf.cast(self._units_per_head, queries.dtype)
    )

    return relative_logits

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
    # This API cannot be implemented since we need key/query values.
    raise NotImplementedError(
        f'{type(self).__name__} does not support get_position_bias_raw.'
    )

  @tf.Module.with_name_scope
  def get_position_bias(self, queries: tf.Tensor) -> tf.Tensor:
    """Computes relative self-attention position biases for queries.

    Args:
      queries: [batch, queries_time, num_heads, units_per_head] queries.

    Returns:
      A tensor of relative position biases broadcastable to
      [batch, num_heads, queries_time, keys_time].
    """
    assert self._max_forward == 0
    batch_size, queries_time = utils.smart_dimension_size(queries, [0, 1])

    # This implementation closely follows tensor2tensor:
    # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_attention.py#L1934
    # If relative position embedding is enabled, then we need to compute a
    # [batch, num_heads, queries_time, keys_time] tensor of relative
    # position biases, biasing the selection of keys for every query
    # timestep.
    #
    # Following the above example of the mask, if letters correspond to each
    # relative bias, and 'a' is the relative position bias for the current
    # queries_time timestep, then our goal for a given batch/head is the
    # following matrix:
    # a 0 0 0 0
    # b a 0 0 0
    # c b a 0 0
    # 0 c b a 0
    # 0 0 c b a

    # First we pad from the left to queries_time so that the relative
    # embeddings are right aligned.
    # 0 0 c b a
    # In the corner case that queries_time is smaller than
    # max_previous + 1, make sure we use the right-most relative
    # embeddings.
    # TODO(rryan): Handle queries_time == 0! Bad slice.
    relative_position_embedding = tf.cast(
        self._relative_position_embedding[-queries_time:, :, :], queries.dtype
    )
    pad_amount = queries_time - tf.minimum(self._max_backward + 1, queries_time)
    relative_position_embedding = tf.pad(
        relative_position_embedding, [[pad_amount, 0], [0, 0], [0, 0]]
    )

    # Then we compute the relative position biases. Considering just one
    # batch/head slice, the matrix looks like this with queries_time = 5
    # and max_past_horizon = 2.
    # 0 0 c b a
    # 0 0 c b a
    # 0 0 c b a
    # 0 0 c b a
    # 0 0 c b a
    relative_logits = tf.einsum(
        'jNH,BiNH->BNij', relative_position_embedding, queries
    )

    # To achieve the goal we use some reshaping trickery. We insert 1 zero
    # at the beginning of each row:
    # 0 0 0 c b a
    # 0 0 0 c b a
    # 0 0 0 c b a
    # 0 0 0 c b a
    # 0 0 0 c b a
    relative_logits = tf.pad(relative_logits, [[0, 0], [0, 0], [0, 0], [1, 0]])

    # Then reshape into a 6x5 matrix, and trim off the first row to achieve
    # the goal:
    # 0 0 0 c b  <- Trim this row.
    # a 0 0 0 c
    # b a 0 0 0
    # c b a 0 0
    # 0 c b a 0
    # 0 0 c b a
    #
    # The biases that wrap around and inhabit the upper triangular portion of
    # the matrix have no effect because the invalid_mask computed above is a
    # banded matrix that only covers the diagonal and max_past_horizon lower
    # bandwidth.
    relative_logits = tf.reshape(
        relative_logits,
        [batch_size, self._num_heads, queries_time + 1, queries_time],
    )
    relative_logits = relative_logits[:, :, 1:, :]

    # Shaw, et al. scale the relative logits by rsqrt(units_per_head).
    relative_logits *= tf.math.rsqrt(
        tf.cast(self._units_per_head, queries.dtype)
    )

    return relative_logits


class T5RelativePositionEmbedding(RelativePositionEmbedding):
  """Relative position embeddings in the T5 style.

  Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer
  https://arxiv.org/abs/1910.10683

  Implementation copied from
  https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/t5.py
  """

  def __init__(
      self,
      num_buckets: int,
      num_heads: int,
      bidirectional: bool,
      max_distance: int = 128,
      embeddings_initializer: Optional[
          tf.keras.initializers.Initializer
      ] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_buckets = num_buckets
    self._bidirectional = bidirectional
    self._max_distance = max_distance
    with self.name_scope:
      self.relative_attention_bias = tf.keras.layers.Embedding(
          input_dim=self._num_buckets,
          output_dim=self._num_heads,
          embeddings_initializer=embeddings_initializer,
          name='rel_embedding',
      )

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
    queries_time = utils.smart_dimension_size(queries, 1)
    keys_time = utils.smart_dimension_size(keys, 1)

    context_position = queries_position + tf.range(queries_time)[:, tf.newaxis]
    # keys[:, 0, :]'s absolute position is queries_position - max_previous.
    max_previous = keys_time - queries_time
    memory_start_position = queries_position - max_previous
    memory_position = memory_start_position + tf.range(keys_time)[tf.newaxis, :]
    # Semantically, this tensor is [queries, keys].
    relative_position = memory_position - context_position
    rp_bucket = self._relative_position_bucket(relative_position)
    values = self.relative_attention_bias(rp_bucket)
    values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
    values.shape.assert_is_compatible_with(
        [1, self._num_heads, queries_time_static, keys_time_static]
    )
    return values

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
    queries_time_static = (
        queries_length if isinstance(queries_length, int) else None
    )
    keys_time_static = keys_length if isinstance(keys_length, int) else None

    queries_positions = (
        queries_position + tf.range(queries_length)[:, tf.newaxis]
    )
    keys_positions = keys_position + tf.range(keys_length)[tf.newaxis, :]

    relative_position = keys_positions - queries_positions
    rp_bucket = self._relative_position_bucket(relative_position)
    values = self.relative_attention_bias(rp_bucket)
    values = tf.expand_dims(tf.transpose(values, [0, 2, 1]), axis=0)
    values.shape.assert_is_compatible_with(
        [1, queries_time_static, self._num_heads, keys_time_static]
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
    queries_time = utils.smart_dimension_size(queries, 1)
    context_position = tf.range(queries_time)[:, tf.newaxis]
    memory_position = tf.range(queries_time)[tf.newaxis, :]
    # Semantically, this tensor is [queries, keys].
    relative_position = memory_position - context_position
    rp_bucket = self._relative_position_bucket(relative_position)
    values = self.relative_attention_bias(rp_bucket)
    values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
    values.shape.assert_is_compatible_with(
        [1, self._num_heads, queries_time_static, queries_time_static]
    )
    return values

  def _relative_position_bucket(
      self, relative_position: tf.Tensor
  ) -> tf.Tensor:
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.

    If bidirectional=False, then positive relative positions are invalid.

    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.

    All relative positions >=max_distance map to the same bucket.

    All relative positions <=-max_distance map to the same bucket.

    This should allow for more graceful generalization to longer sequences
    than the model has been trained on.

    Args:
      relative_position: A [query_time, keys_time] int32 Tensor of relative
        positions. r = relative_position[q, k] indicates key timestep k is r
        timesteps ahead (positive) or behind (negative) of query timestep q.

    Returns:
      a Tensor with the same shape as relative_position, containing int32
      values in the range [0, num_buckets)
    """
    with tf.name_scope('relative_position_bucket'):
      ret = 0
      n = -relative_position
      num_buckets = self._num_buckets
      max_distance = self._max_distance
      if self._bidirectional:
        num_buckets //= 2
        ret += tf.cast(tf.math.less(n, 0), tf.int32) * num_buckets
        n = tf.math.abs(n)
      else:
        n = tf.math.maximum(n, 0)
      # now n is in the range [0, inf)
      max_exact = num_buckets // 2
      is_small = tf.math.less(n, max_exact)
      compute_dtype = utils.compute_dtype()
      eps = np.finfo(tf.float32.as_numpy_dtype).eps
      # Note that `(num_buckets - 1 - max_exact)` below differs from the
      # reference implementation of T5 relative position biases which uses
      # `(num_buckets - max_exact)`, and therefore doesn't produce the
      # max_distance behavior described in the docstring above.
      val_if_large = max_exact + tf.dtypes.cast(
          tf.math.log(tf.cast(n, compute_dtype) / max_exact + eps)
          / math.log(max_distance / max_exact)
          * (num_buckets - 1 - max_exact),
          tf.int32,
      )
      val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
      ret += tf.where(is_small, n, val_if_large)
      return ret


def _dot_product_attention(
    queries: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    invalid_mask: tf.Tensor,
    logit_bias: Optional[tf.Tensor],
    training: bool,
    num_heads: int,
    units_per_head: int,
    attention_logits_soft_cap: float | None,
    attention_probabilities_dropout: tf.keras.layers.Dropout,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes standard dot product attention with queries, keys and values.

  Args:
    queries: A [batch, query_time, num_heads, units_per_head] tensor of queries.
    keys: A [batch, key_time, num_heads, units_per_head] tensor of keys.
    values: A [batch, key_time, num_heads, units_per_head] tensor of values.
    invalid_mask: A tensor broadcastable to [batch, num_heads, query_time,
      key_time] which is 1.0 for positions that are invalid (i.e. should not be
      attended to).
    logit_bias: A tensor broadcastable to [batch, num_heads, query_time,
      key_time] of attention logit biases to apply after computing the logits.
    training: Whether we are in training mode.
    num_heads: The number of attention heads.
    units_per_head: The number of units per head.
    attention_logits_soft_cap: If non-zero, a soft cap applied to attention
      logits to prevent outliers from dominating the softmax. Empirically, 50.0
      works well across a variety of tasks. Implemented as tanh(logits / cap) *
      cap.
    attention_probabilities_dropout: A dropout layer to apply to the attention
      probabilities.

  Returns:
    context_vectors: A [batch_size, query_time, num_heads, units_per_head]
      tensor of context vectors for the queries.
    probabilities: A [batch_size, query_time, num_heads, keys_time] tensor of
      attention probabilities (for debugging).
    logits: A [batch_size, query_time, num_heads, keys_time] tensor of attention
      logits (for debugging).
  """
  queries.shape.assert_is_compatible_with(
      [None, None, num_heads, units_per_head]
  )
  keys.shape.assert_is_compatible_with([None, None, num_heads, units_per_head])
  values.shape.assert_is_compatible_with(
      [None, None, num_heads, units_per_head]
  )

  # Attention weights are computed as
  # softmax((queries * keys^T) / sqrt(units_per_head)), but for efficiency we
  # scale the queries before the matmul with keys to get logits.
  queries *= tf.math.rsqrt(tf.cast(units_per_head, queries.dtype))

  # Babelfish uses this order because it's more efficient on TPU.
  logits = tf.einsum('BjNH,BiNH->BNij', keys, queries)

  if logit_bias is not None:
    # Layout matches logits.
    logit_bias.shape.assert_is_compatible_with([None, num_heads, None, None])
    # Check that relative logits is broadcastable to logits.
    tf.broadcast_static_shape(logit_bias.shape, logits.shape)
    logits += logit_bias

  # Cap attention logits before masking.
  if attention_logits_soft_cap is not None:
    logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

  logits = _mask_attention_logits(logits, invalid_mask)
  probabilities = _softmax_in_at_least_float32(logits)
  probabilities = attention_probabilities_dropout(
      probabilities, training=training
  )

  # Contract the keys_time dimension into per-head context vectors:
  # [batch, query_time, num_heads, units_per_head].
  context_vectors = tf.einsum('BNij,BjNH->BiNH', probabilities, values)

  # Transpose [batch, num_heads, query_time, source_time] to
  # [batch, query_time, num_heads, source_time].
  probabilities = tf.transpose(probabilities, [0, 2, 1, 3])
  logits = tf.transpose(logits, [0, 2, 1, 3])
  return context_vectors, probabilities, logits


class _StreamingSoftmaxState(NamedTuple):
  """State tuple for a streaming softmax.

  Tracks the numerator and the denominator as well as the maximum logit seen so
  far separately for each query position.
  """

  # [b, t, num_heads, units_per_head]
  numerator: tf.Tensor
  # [b, t, num_heads]
  denominator: tf.Tensor
  # [b, t, num_heads]
  max_so_far: tf.Tensor


def _streaming_logits_and_softmax_step(
    state: _StreamingSoftmaxState,
    queries: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    invalid_mask: tf.Tensor,
    logit_bias: tf.Tensor | None,
    num_heads: int,
    query_chunk_size: int,
    attention_logits_soft_cap: float | None,
    compute_dtype: tf.DType,
) -> _StreamingSoftmaxState:
  """Computes an update to the state given queries/keys/values."""
  state.max_so_far.shape.assert_is_compatible_with(
      [None, query_chunk_size, num_heads]
  )

  # Perform the query/key multiplication in compute_dtype to match the
  # _dot_product_attention implementation. If we need higher precision here in
  # _chunked_attention, we should also do it in _attention.
  logits = tf.einsum('BiNH,BjNH->BiNj', queries, keys)

  if logit_bias is not None:
    # Check that relative logits is broadcastable to logits.
    tf.broadcast_static_shape(logit_bias.shape, logits.shape)
    logits += logit_bias

  # Cap attention logits before masking.
  if attention_logits_soft_cap is not None:
    logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

  logits = _mask_attention_logits(logits, invalid_mask)

  if compute_dtype.size < 4:
    logits = tf.cast(logits, tf.float32)
    values = tf.cast(values, tf.float32)

  # Maximum key logit for each (batch/query/head) slice.
  chunk_max = tf.reduce_max(logits, axis=-1)

  chunk_max.shape.assert_is_compatible_with([None, None, num_heads])
  max_so_far = tf.math.maximum(state.max_so_far, chunk_max)
  max_so_far = tf.stop_gradient(max_so_far)
  max_so_far.shape.assert_is_compatible_with([None, None, num_heads])
  correction = tf.math.exp(state.max_so_far - max_so_far)

  corrected_weights = tf.math.exp(logits - max_so_far[:, :, :, tf.newaxis])

  numerator = state.numerator * correction[:, :, :, tf.newaxis]
  # Compute weighted values and add them to the numerator.
  numerator += tf.einsum('BiNj,BjNH->BiNH', corrected_weights, values)

  denominator = state.denominator * correction
  # Add sum of all weights for keys to the denominator.
  # Invalid locations are zero because e^(-inf - max) = 0.
  denominator += tf.reduce_sum(corrected_weights, axis=-1)

  return _StreamingSoftmaxState(numerator, denominator, max_so_far)


def _pad_dim_to(
    t: tf.Tensor,
    axis: int,
    cur_length: tf.Tensor,
    target_length: tf.Tensor,
    pad_value: float,
) -> tf.Tensor:
  paddings = [[0, 0]] * t.shape.rank
  paddings[axis] = [0, tf.maximum(target_length - cur_length, 0)]
  return tf.pad(t, paddings, constant_values=tf.constant(pad_value, t.dtype))


def _chunked_dot_product_attention(
    queries: tf.Tensor,
    keys: tf.Tensor,
    keys_mask: tf.Tensor,
    values: tf.Tensor,
    invalid_mask_fn: Callable[
        [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor
    ],
    logit_bias_fn: Callable[
        [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        tf.Tensor,
    ]
    | None,
    num_heads: int,
    units_per_head: int,
    attention_logits_soft_cap: float | None,
    query_chunk_size: int,
    key_chunk_size: int | None,
) -> tf.Tensor:
  """Computes chunked dot product attention with queries, keys and values.

  Chunked dot product attention avoids materializing the full [query_time,
  key_time] tensor of attention logits by computing the context vectors for the
  queries incrementally in chunks of [query_chunk_size, key_chunk_size].
  - This avoids using O(n^2) memory, allowing increased scale of dense
    attention.
  - Using the decomposed softmax trick, the the memory access patterns of
    computing the softmax are improved which can combine well with XLA kernel
    fusion. to reduce the total number of reads and writes from HBM and improve
    memory bandwidth consumption.

  References:
  - https://arxiv.org/abs/2112.05682
  - https://arxiv.org/abs/2205.14135

  Args:
    queries: A [batch, query_time, num_heads, units_per_head] tensor of queries.
    keys: A [batch, key_time, num_heads, units_per_head] tensor of keys.
    keys_mask: A [batch, key_time] tensor indicating valid positiosn in the keys
      sequence. Implementation detail required for gradient checkpointing.
    values: A [batch, key_time, num_heads, units_per_head] tensor of values.
    invalid_mask_fn: A callable that returns a tensor broadcastable to [batch,
      query_time, num_heads, key_time] which is 1.0 for positions that are
      invalid (i.e. should not be attended to).
    logit_bias_fn: A callable that returns a tensor broadcastable to [batch,
      query_time, num_heads, key_time] of attention logit biases to apply after
      computing the logits.
    num_heads: The number of attention heads.
    units_per_head: The number of units per head.
    attention_logits_soft_cap: If non-zero, a soft cap applied to attention
      logits to prevent outliers from dominating the softmax. Empirically, 50.0
      works well across a variety of tasks. Implemented as tanh(logits / cap) *
      cap.
    query_chunk_size: The query chunk size to use. Computes attention in
      parallel for blocks of queries of this size.
    key_chunk_size: The key chunk size to use. If None, computes chunked
      attention over the entire key sequence for each block of queries. If
      non-None, computes attention for blocks of [query_chunk_size,
      key_chunk_size] at a time, bounding the overall memory usage to
      O(query_chunk_size * key_chunk_size).

  Returns:
    context_vectors: A [batch_size, query_time, num_heads, units_per_head]
      tensor of context vectors for the queries.
  """
  if query_chunk_size <= 0:
    raise ValueError(f'{query_chunk_size=} must be positive.')
  compute_dtype = queries.dtype

  # This implementation supports the "extra logit" feature but
  # _dot_product_attention doesn't.
  # TODO(rryan): Enable this functionality.
  use_extra_logit = False

  # [b, q, h, d]
  queries.shape.assert_is_compatible_with(
      [None, None, num_heads, units_per_head]
  )
  # [b, k, h, d]
  keys.shape.assert_is_compatible_with([None, None, num_heads, units_per_head])
  # [b, v, h, d]
  values.shape.assert_is_compatible_with(
      [None, None, num_heads, units_per_head]
  )

  # Attention weights are computed as
  # softmax((queries * keys^T) / sqrt(units_per_head)), but for efficiency we
  # scale the queries before the matmul with keys to get logits.
  queries *= tf.math.rsqrt(tf.cast(units_per_head, queries.dtype))

  query_batch_size, query_time = utils.smart_dimension_size(queries, [0, 1])
  key_batch_size, key_time = utils.smart_dimension_size(keys, [0, 1])
  tf.debugging.assert_equal(query_batch_size, key_batch_size)

  num_query_chunks = (query_time + query_chunk_size - 1) // query_chunk_size

  # TODO(b/298748783): Work around slice failure by padding input to multiple of
  # slice length.
  query_time_padded = num_query_chunks * query_chunk_size
  queries = _pad_dim_to(
      queries,
      axis=1,
      cur_length=query_time,
      target_length=query_time_padded,
      pad_value=0.0,
  )

  num_key_chunks = 1
  if key_chunk_size is not None:
    num_key_chunks = (key_time + key_chunk_size - 1) // key_chunk_size
    key_time_padded = num_key_chunks * key_chunk_size
    keys = _pad_dim_to(
        keys,
        axis=1,
        cur_length=key_time,
        target_length=key_time_padded,
        pad_value=0.0,
    )
    keys_mask = _pad_dim_to(
        keys_mask,
        axis=1,
        cur_length=key_time,
        target_length=key_time_padded,
        pad_value=0.0,
    )
    keys_mask = tf.stop_gradient(keys_mask)
    values = _pad_dim_to(
        values,
        axis=1,
        cur_length=key_time,
        target_length=key_time_padded,
        pad_value=0.0,
    )
    del key_time

  state_dtype = tf.float32 if queries.dtype.size < 4 else queries.dtype
  zero_chunk = _StreamingSoftmaxState(
      tf.zeros(
          [
              query_batch_size,
              query_chunk_size,
              num_heads,
              units_per_head,
          ],
          dtype=state_dtype,
      ),
      tf.zeros(
          [
              query_batch_size,
              query_chunk_size,
              num_heads,
          ],
          dtype=state_dtype,
      ),
      tf.fill(
          [
              query_batch_size,
              query_chunk_size,
              num_heads,
          ],
          tf.constant(-np.inf, dtype=state_dtype),
      ),
  )

  def query_loop_cond(query_i, output_ta):
    del output_ta
    return query_i < num_query_chunks

  def query_loop_body(
      query_i,
      output_ta: tf.TensorArray,
  ):
    query_start = query_i * query_chunk_size
    query_chunk = None

    @tf.recompute_grad
    def no_key_chunking(
        queries: tf.Tensor,
        keys: tf.Tensor,
        keys_mask: tf.Tensor,
        values: tf.Tensor,
        *args,
    ) -> _StreamingSoftmaxState:
      # We can't pass None values through tf.recompute_grad so optionally unpack
      # it from *args.
      logit_bias_chunk = args[0] if args else None
      query_chunk = utils.slice_and_pad_tensor(
          queries,
          [(1, query_start, query_chunk_size)],
          pad_value=0.0,
          # TODO(b/298748783): Queries tensor is pre-padded.
          tensor_is_pre_padded=True,
          name='no_key_chunking_query_chunk',
      )
      key_chunk = keys
      value_chunk = values
      invalid_mask_chunk = invalid_mask_fn(
          query_start,
          query_chunk_size,
          0,
          utils.smart_dimension_size(keys, 1),
          # Don't know why keys_mask needs to be plumbed through to here despite
          # the top level stop gradient. It only seems to affect
          # DotProductAttention (error about dependence on source.mask).
          tf.stop_gradient(keys_mask),
      )

      # TODO(rryan): See what XLA does when we use tf.nn.softmax in this case?
      return _streaming_logits_and_softmax_step(
          state=zero_chunk,
          queries=query_chunk,
          keys=key_chunk,
          values=value_chunk,
          invalid_mask=invalid_mask_chunk,
          logit_bias=logit_bias_chunk,
          num_heads=num_heads,
          query_chunk_size=query_chunk_size,
          attention_logits_soft_cap=attention_logits_soft_cap,
          compute_dtype=compute_dtype,
      )

    def kv_loop_cond(kv_i: tf.Tensor, chunk_state: _StreamingSoftmaxState):
      del chunk_state
      return kv_i < num_key_chunks

    def kv_loop_body(kv_i: tf.Tensor, chunk_state: _StreamingSoftmaxState):
      kv_start = kv_i * key_chunk_size

      @tf.recompute_grad
      def kv_loop_body_inner(
          query_chunk: tf.Tensor,
          keys: tf.Tensor,
          values: tf.Tensor,
          chunk_state: _StreamingSoftmaxState,
          *args,
      ) -> _StreamingSoftmaxState:
        # We can't pass None values through tf.recompute_grad so optionally
        # unpack it from *args.
        logit_bias_chunk = args[0] if args else None
        key_chunk = utils.slice_and_pad_tensor(
            keys,
            [(1, kv_start, key_chunk_size)],
            pad_value=0.0,
            # TODO(b/298748783): Queries tensor is pre-padded.
            tensor_is_pre_padded=True,
            name='kv_loop_key_chunk',
        )
        value_chunk = utils.slice_and_pad_tensor(
            values,
            [(1, kv_start, key_chunk_size)],
            pad_value=0.0,
            # TODO(b/298748783): Queries tensor is pre-padded.
            tensor_is_pre_padded=True,
            name='kv_loop_value_chunk',
        )
        invalid_mask_chunk = invalid_mask_fn(
            query_start,
            query_chunk_size,
            kv_start,
            key_chunk_size,
            tf.stop_gradient(keys_mask),
        )

        chunk_state = _streaming_logits_and_softmax_step(
            state=chunk_state,
            queries=query_chunk,
            keys=key_chunk,
            values=value_chunk,
            invalid_mask=invalid_mask_chunk,
            logit_bias=logit_bias_chunk,
            num_heads=num_heads,
            query_chunk_size=query_chunk_size,
            attention_logits_soft_cap=attention_logits_soft_cap,
            compute_dtype=compute_dtype,
        )
        return chunk_state

      args = [query_chunk, keys, values, chunk_state]
      if logit_bias_fn is not None:
        logit_bias_chunk = logit_bias_fn(
            query_start,
            query_chunk_size,
            kv_start,
            key_chunk_size,
        )
        args.append(logit_bias_chunk)
      chunk_state = kv_loop_body_inner(*args)
      return kv_i + 1, chunk_state

    if key_chunk_size is None:
      # Query chunk happens inside no_key_chunking so it is re-sliced in the
      # backward pass.
      args = [queries, keys, keys_mask, values]
      if logit_bias_fn is not None:
        logit_bias_chunk = logit_bias_fn(
            query_start,
            query_chunk_size,
            0,
            utils.smart_dimension_size(keys, 1),
        )
        args.append(logit_bias_chunk)
      state = no_key_chunking(*args)
    else:
      # Read query_chunk outside of loop.
      query_chunk = utils.slice_and_pad_tensor(
          queries,
          [(1, query_start, query_chunk_size)],
          pad_value=0.0,
          # TODO(b/298748783): Queries tensor is pre-padded.
          tensor_is_pre_padded=True,
          name='query_chunk_for_key_chunk_loop',
      )
      _, state = tf.while_loop(
          kv_loop_cond,
          kv_loop_body,
          (tf.constant(0), zero_chunk),
          maximum_iterations=num_key_chunks,
          name='key_chunk_loop',
      )

    numerator, denominator, max_so_far = state
    if use_extra_logit:
      denominator += tf.math.exp(-max_so_far)
    output_i = numerator / denominator[:, :, :, tf.newaxis]
    output_i = tf.cast(output_i, compute_dtype)
    output_i.shape.assert_is_compatible_with(
        [None, None, num_heads, units_per_head]
    )

    output_ta = output_ta.write(query_i, output_i)
    return query_i + 1, output_ta

  output_ta = tf.TensorArray(
      dtype=compute_dtype, size=num_query_chunks, dynamic_size=False
  )

  _, output_ta = tf.while_loop(
      query_loop_cond,
      query_loop_body,
      (tf.constant(0), output_ta),
      maximum_iterations=num_query_chunks,
      name='query_chunk_loop',
  )
  output = output_ta.stack()
  output.shape.assert_is_compatible_with([
      num_query_chunks if isinstance(num_query_chunks, int) else None,
      None,  # batch_size
      query_chunk_size,
      num_heads,
      units_per_head,
  ])

  # Transpose to [b, num_query_chunks, query_chunk_size, num_heads,
  # units_per_head] and combine num_query_chunks and query_chunk_size.
  output = tf.transpose(output, [1, 0, 2, 3, 4])
  context_vectors = tf.reshape(
      output,
      [
          query_batch_size,
          num_query_chunks * query_chunk_size,
          num_heads,
          units_per_head,
      ],
  )
  # Slice off padded query timesteps from the last block.
  context_vectors = context_vectors[:, :query_time, :, :]

  return context_vectors


def _dot_product_attention_logit_bias_fn(
    query_time: tf.Tensor,
    query_length: tf.Tensor | int,
    key_time: tf.Tensor,
    key_length: tf.Tensor | int,
    *,
    relative_position_embedding: RelativePositionEmbedding,
) -> tf.Tensor:
  """Returns logit biases [b, q, h, k] for the provided positions."""
  return relative_position_embedding.get_position_bias_raw(
      query_time, query_length, key_time, key_length
  )


def _dot_product_attention_invalid_mask_fn(
    query_time: tf.Tensor,
    query_length: tf.Tensor,
    key_time: tf.Tensor,
    key_length: tf.Tensor,
    key_mask: tf.Tensor,
    *,
    max_past_horizon: int,
    max_future_horizon: int,
) -> tf.Tensor:
  """Returns an invalid mask [b, q, 1, k] for the provided parameters."""
  compute_dtype = utils.compute_dtype()
  if max_past_horizon == -1 and max_future_horizon == -1:
    visibility_mask = None
  else:
    query_ix = query_time + tf.range(query_length)
    key_ix = key_time + tf.range(key_length)

    # Positive offset means key position lies ahead of query position.
    offsets = key_ix[tf.newaxis, :] - query_ix[:, tf.newaxis]

    visibility_mask = None
    if max_past_horizon != -1:
      visibility_mask = offsets >= -max_past_horizon
    if max_future_horizon != -1:
      condition = offsets <= max_future_horizon
      visibility_mask = (
          condition
          if visibility_mask is None
          else tf.logical_and(visibility_mask, condition)
      )

    visibility_mask = tf.cast(visibility_mask, compute_dtype)[
        tf.newaxis, :, tf.newaxis, :
    ]
    # Check the mask is [1 (broadcast to batch), query_length, 1 (broadcast to
    # heads), key_length].
    visibility_mask.shape.assert_is_compatible_with([1, None, 1, None])

  valid_mask = tf.cast(key_mask, compute_dtype)
  valid_mask = tf.slice(valid_mask, [0, key_time], [-1, key_length])

  valid_mask = valid_mask[:, tf.newaxis, tf.newaxis, :]
  # Check the mask is [batch, 1 (broadcast to query), 1 (broadcast to heads),
  # key_length].
  valid_mask.shape.assert_is_compatible_with([None, 1, 1, None])

  invalid_mask = 1.0 - (
      visibility_mask * valid_mask
      if visibility_mask is not None
      else valid_mask
  )
  if visibility_mask is not None:
    invalid_mask.shape.assert_is_compatible_with([None, None, 1, None])
  else:
    invalid_mask.shape.assert_is_compatible_with([None, 1, 1, None])
  return invalid_mask


class DotProductSelfAttention(types.Emitting):
  """A multi-headed dot-product self attention layer.

  TODO(rryan): Support for stepping layers with max_horizon == -1.
  """

  def __init__(
      self,
      num_heads: int,
      units_per_head: int,
      max_horizon: int,
      use_relative_position_embedding: bool = False,
      relative_position_embedding: Optional[RelativePositionEmbedding] = None,
      max_future_horizon: int = 0,
      attention_probabilities_dropout_rate: float = 0.0,
      broadcast_dropout_across_queries: bool = False,
      use_bias: bool = False,
      kernel_initializer: tf.keras.initializers.Initializer = 'glorot_uniform',
      bias_initializer: tf.keras.initializers.Initializer = 'zeros',
      attention_logits_soft_cap: float | None = None,
      query_chunk_size: int | None = None,
      key_chunk_size: int | None = None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
      name: Optional[str] = None,
  ):
    """A dot-product self attention layer as in Transformers.

    Args:
      num_heads: The number of attention heads.
      units_per_head: The number of units per head.
      max_horizon: The number of past timesteps each timestep can see. -1:
        Disable masking of the past (all past timesteps are visible)
         0: No past timesteps are visible. The layer is only steppable when
           max_horizon >= 0.
      use_relative_position_embedding: Whether to use a relative position
        embedding.
      relative_position_embedding: An optional RelativePositionEmbedding to use
        to compute relative position biases. If not provided and
        use_relative_position_embedding is true, a ShawRelativePositionEmbedding
        is created.
      max_future_horizon: The number of future timesteps each timestep can see.
        -1: Disable masking of the future (all future timesteps are visible)
        0: No future timesteps are visible. The layer is only steppable when
          max_future_horizon == 0.
      attention_probabilities_dropout_rate: The dropout rate for the attention
        probabilities.
      broadcast_dropout_across_queries: Whether to broadcast the dropout across
        the query time dimension as is done in T5.
      use_bias: Whether to learn a bias in the query/key/value projection.
      kernel_initializer: Kernel initializer for the query/key/value projection.
      bias_initializer: Bias initialier for the query/key/value projection.
      attention_logits_soft_cap: If non-zero, a soft cap applied to attention
        logits to prevent outliers from dominating the softmax. Empirically,
        50.0 works well across a variety of tasks. Implemented as tanh(logits /
        cap) * cap.
      query_chunk_size: The query chunk size to use. Computes attention in
        parallel for blocks of queries of this size. If enabled, no emits are
        produced (e.g. attention probabilities for debugging).
      key_chunk_size: The key chunk size to use. If None, computes chunked
        attention over the entire key sequence for each block of queries. If
        non-None, computes attention for blocks of [query_chunk_size,
        key_chunk_size] at a time, bounding the overall memory usage to
        O(query_chunk_size * key_chunk_size). If enabled, no emits are produced
        (e.g. attention probabilities for debugging).
      query_network: An optional layer for processing queries before logit
        calculations are performed. Must have an output ratio and block size
        of 1.
      key_network: An optional layer for processing keys before logit
        calculations are performed. Must have an output ratio and block size
        of 1.
      value_network: An optional layer for processing values before attention
        weighted summation is performed. Must have an output ratio and block
        size of 1.
      name: An optional name for the layer.
    """
    super().__init__(name=name)
    _validate_heads(num_heads, units_per_head, name)
    if max_horizon < -1:
      raise ValueError(
          f'Expected max_horizon >= -1 for {name}, got {max_horizon}.'
      )
    if max_future_horizon < -1:
      raise ValueError(
          'Expected max_future_horizon >= -1 for '
          f'{name}, got {max_future_horizon}.'
      )
    if max_future_horizon == 0 and max_horizon == 0:
      raise ValueError(
          'Both max_horizon and max_future_horizon are 0, which '
          f'does not make sense for {self}.'
      )
    if (
        not use_relative_position_embedding
        and relative_position_embedding is not None
    ):
      raise ValueError(
          f'Inconsistent value for {use_relative_position_embedding=} and '
          f'{relative_position_embedding=}.'
      )

    chunking_enabled = query_chunk_size or key_chunk_size
    if attention_probabilities_dropout_rate > 0.0 and chunking_enabled:
      raise NotImplementedError(
          'Attention probability dropout is incompatible with chunking.'
      )

    self._num_heads = num_heads
    self._units_per_head = units_per_head
    self._max_past_horizon = max_horizon
    self._max_future_horizon = max_future_horizon
    self._use_relative_position_embedding = use_relative_position_embedding
    self._attention_logits_soft_cap = attention_logits_soft_cap
    self._query_chunk_size = query_chunk_size
    self._key_chunk_size = key_chunk_size
    if (
        attention_logits_soft_cap is not None
        and attention_logits_soft_cap <= 0.0
    ):
      raise ValueError(
          f'{attention_logits_soft_cap=} should be a positive number.'
      )
    with self.name_scope:
      # TODO(rryan): Benchmark whether we should split these up.
      self._qkv = dense.DenseShaped(
          [3, self._num_heads, self._units_per_head],
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          name='query_key_value_projection',
      )
      if broadcast_dropout_across_queries:
        # [batch, num_heads, query_time, source_time]
        noise_shape = [None, None, 1, None]
      else:
        noise_shape = None
      self._attention_probabilities_dropout = tf.keras.layers.Dropout(
          attention_probabilities_dropout_rate,
          noise_shape=noise_shape,
          name='attention_probabilities_dropout',
      )
      if use_relative_position_embedding:
        if relative_position_embedding:
          self._relative_position_embedding = relative_position_embedding
        else:
          if use_relative_position_embedding and max_future_horizon != 0:
            raise ValueError(
                'Future-dependent relative position embeddings are not '
                'supported yet.'
            )
          self._relative_position_embedding = ShawRelativePositionEmbedding(
              max_backward=self._max_past_horizon,
              max_forward=self._max_future_horizon,
              num_heads=num_heads,
              units_per_head=units_per_head,
          )
      else:
        self._relative_position_embedding = None

      if callable(query_network):
        query_network = query_network()
      if callable(key_network):
        key_network = key_network()
      if callable(value_network):
        value_network = value_network()

      if query_network and (
          query_network.output_ratio != 1 or query_network.block_size != 1
      ):
        raise ValueError(
            'Query network must have an output_ratio'
            f' ({query_network.output_ratio}) and block_size'
            f' ({query_network.block_size}) of 1.'
        )

      if key_network and (
          key_network.output_ratio != 1 or key_network.block_size != 1
      ):
        raise ValueError(
            'Key network must have an output_ratio'
            f' ({key_network.output_ratio}) and block_size'
            f' ({key_network.block_size}) of 1.'
        )

      if value_network and (
          value_network.output_ratio != 1 or value_network.block_size != 1
      ):
        raise ValueError(
            'Value network must have an output_ratio'
            f' ({value_network.output_ratio}) and block_size'
            f' ({value_network.block_size}) of 1.'
        )

      self._query_network = query_network
      self._key_network = key_network
      self._value_network = value_network

  @property
  def supports_step(self) -> bool:
    supports_step = (
        self._max_future_horizon == 0 and self._max_past_horizon >= 0
    )
    if self._query_network:
      supports_step = supports_step and self._query_network.supports_step

    if self._key_network:
      supports_step = supports_step and self._key_network.supports_step

    if self._value_network:
      supports_step = supports_step and self._value_network.supports_step

    return supports_step

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    # State to contain the max_horizon previous projected keys and values.
    batch_size = utils.smart_dimension_size(x.values, 0)
    # Note, the state is invalid since it is padding, so we don't want to attend
    # to it.
    max_past_horizon = max(0, self._max_past_horizon)
    zero_values = tf.zeros(
        (batch_size, max_past_horizon, self._num_heads, self._units_per_head),
        dtype=x.values.dtype,
    )
    zero_mask = tf.zeros([batch_size, max_past_horizon], dtype=types.MASK_DTYPE)
    state_keys = zero_values
    state_values = zero_values
    state_mask = zero_mask
    position = tf.constant(0, tf.int32)

    qkv_input = types.Sequence(
        tf.zeros(
            (batch_size, 0, self._num_heads, self._units_per_head),
            x.values.dtype,
        ),
        tf.zeros((batch_size, 0), types.MASK_DTYPE),
    )

    if self._query_network:
      query_state = self._query_network.get_initial_state(
          qkv_input,
          constants=constants,
      )
    else:
      query_state = ()
    if self._key_network:
      key_state = self._key_network.get_initial_state(
          qkv_input,
          constants=constants,
      )
    else:
      key_state = ()
    if self._value_network:
      value_state = self._value_network.get_initial_state(
          qkv_input,
          constants=constants,
      )
    else:
      value_state = ()

    return (
        state_keys,
        state_values,
        state_mask,
        position,
        query_state,
        key_state,
        value_state,
    )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'DotProductSelfAttention requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return tf.TensorShape([self._num_heads, self._units_per_head])

  def get_emit_specs(
      self,
      input_spec: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> types.Emits:
    # If query / key chunking is not enabled, emit attention probabilities.
    if self._query_chunk_size is None and self._key_chunk_size is None:
      # The input time dimension is unknown so we can't provide a known
      # shape for comb_weights.
      return AttentionEmits(
          types.Sequence(
              tf.TensorSpec(
                  tf.TensorShape([self._num_heads, None]), input_spec.dtype
              ),
              tf.TensorSpec(tf.TensorShape([]), tf.float32),
          )
      )
    return ()

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    if self._max_future_horizon != 0:
      raise ValueError(
          f'{self} is not steppable since max_future_horizon != 0.'
      )
    if self._max_past_horizon == -1:
      raise ValueError(f'{self} is not steppable since max_horizon is not set.')
    batch_size_static = x.values.shape.dims[0].value
    x_values_time = utils.smart_dimension_size(x.values, 1)
    x_values_time_static = x.values.shape.dims[1].value

    (
        state_keys,
        state_values,
        state_mask,
        position,
        query_state,
        key_state,
        value_state,
    ) = state

    # Micro-optimization: If use_bias is false, we can elide masking inside
    # _qkv. Maybe in general?
    x_qkv = self._qkv.layer(x, training=training)
    x_queries, x_keys, x_values = tf.unstack(x_qkv.values, axis=2)

    if self._query_network:
      x_queries, query_state = self._query_network.step(
          types.Sequence(x_queries, x_qkv.mask),
          query_state,
          training=training,
          constants=constants,
      )
      x_queries = x_queries.values
    if self._key_network:
      x_keys, key_state = self._key_network.step(
          types.Sequence(x_keys, x_qkv.mask),
          key_state,
          training=training,
          constants=constants,
      )
      x_keys = x_keys.values
    if self._value_network:
      x_values, value_state = self._value_network.step(
          types.Sequence(x_values, x_qkv.mask),
          value_state,
          training=training,
          constants=constants,
      )
      x_values = x_values.values

    # To process a step, concatenate our max_horizon state with the input
    # x_values_time_static timesteps. We need to make a banded mask tensor with
    # max_horizon + block_size timesteps per row.
    state_keys = tf.concat([state_keys, x_keys], axis=1)
    state_values = tf.concat([state_values, x_values], axis=1)
    state_mask = tf.concat([state_mask, x_qkv.mask], axis=1)

    state_values_time = x_values_time + self._max_past_horizon
    state_values_time_static = (
        x_values_time_static + self._max_past_horizon
        if x_values_time_static is not None
        else None
    )

    relative_logits = None
    if self._relative_position_embedding:
      relative_logits = (
          self._relative_position_embedding.get_position_bias_streaming(
              x_queries, state_keys, position
          )
      )

    if x_values_time_static is not None and x_values_time_static == 1:
      # If queries_time is 1, there is no need for causal masking because our
      # state covers the previous max_horizon timesteps (which we can look at)
      # and the 1 timestep in x that we concatenated. We can simply
      # use state's "invalid" mask, reshaped to [batch_size, 1, 1, keys_time] so
      # it broadcasts across heads and queries_time.
      combined_invalid_mask = 1.0 - state_mask[:, tf.newaxis, tf.newaxis, :]
    else:
      valid_mask = state_mask[:, tf.newaxis, tf.newaxis, :]
      # To obey causality, the output for each of the block_size input timesteps
      # may only depend on itself and max_horizon previous timesteps. Earlier
      # timesteps in block_size cannot depend on later timesteps.
      #
      # After the above concatenation, the state tensor is of length
      # max_horizon + block_size. For example, with max_horizon = 5 and
      # block_size = 3:
      #
      # 1 2 3 4 5 a b c
      #
      # Where numbers indicate previous state and letters indicate timesteps of
      # the current block.
      #
      # Timestep a can look at a and 1-5, but not b or c. Its mask looks like:
      # 1 1 1 1 1 1 0 0
      # Timestep b can look at b, a, and 2-5:
      # 0 1 1 1 1 1 1 0
      # Timestep c can look at c, b, a, and 3-5:
      # 0 0 1 1 1 1 1 1
      #
      # This matrix corresponds to a 3x8 banded matrix with zero lower-bandwidth
      # and max_horizon upper-bandwidth.
      causal_mask = _ones_matrix_band_part(
          x_values_time,
          state_values_time,
          num_lower=0,
          num_upper=self._max_past_horizon,
          out_shape=[1, 1, x_values_time, state_values_time],
      )

      # Broadcasting across batch_size, heads and query time.
      combined_invalid_mask = 1.0 - valid_mask * causal_mask

    combined_invalid_mask.shape.assert_is_compatible_with(
        [batch_size_static, 1, x_values_time_static, state_values_time_static]
    )

    # TODO(b/297462463): Step-wise support for chunked attention.
    context_vectors, probabilities, _ = _dot_product_attention(
        queries=x_queries,
        keys=state_keys,
        values=state_values,
        invalid_mask=combined_invalid_mask,
        logit_bias=relative_logits,
        training=training,
        num_heads=self._num_heads,
        units_per_head=self._units_per_head,
        attention_logits_soft_cap=self._attention_logits_soft_cap,
        attention_probabilities_dropout=self._attention_probabilities_dropout,
    )

    # Preserve last max_horizon as state for next step.
    state_keys = state_keys[:, -self._max_past_horizon :]
    state_values = state_values[:, -self._max_past_horizon :]
    state_mask = state_mask[:, -self._max_past_horizon :]

    state = (
        state_keys,
        state_values,
        state_mask,
        position + x_values_time,
        query_state,
        key_state,
        value_state,
    )

    if self._key_chunk_size is None and self._query_chunk_size is None:
      emits = AttentionEmits(types.Sequence(probabilities, x.mask))
    else:
      emits = ()

    # If this step is a mix of valid and invalid values, the mask_invalid here
    # will clear the invalid timesteps.
    context_vectors = types.Sequence(context_vectors, x.mask).mask_invalid()

    return context_vectors, state, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    values_time = utils.smart_dimension_size(x.values, 1)
    batch_size_static = x.values.shape.dims[0].value
    values_time_static = x.values.shape.dims[1].value
    compute_dtype = utils.compute_dtype()

    def default_invalid_mask() -> tf.Tensor:
      # Compute a connectivity mask indicating which of the [query, key]
      # timesteps can see each other. Since query_time == key_time, this is a
      # banded square matrix with max_past_horizon lower-bandwidth and
      # max_future_horizon upper-bandwidth.
      #
      # For example, for values_time = 5, max_past_horizon = 2,
      # max_future_horizon = 1:
      #
      # a b c d e
      #
      # a can only look at itself and b:
      # 1 1 0 0 0
      # b can look at a, itself and c:
      # 1 1 1 0 0
      # c can look at a, b, itself and d:
      # 1 1 1 1 0
      # d can look at b, c, itself and e.
      # 0 1 1 1 1
      # e can look at c, d and itself:
      # 0 0 1 1 1
      #
      # This corresponds to a 5x5 banded matrix with max_past_horizon
      # lower-bandwidth and max_future_horizon upper-bandwidth.
      if self._max_past_horizon != -1 or self._max_future_horizon != -1:
        num_lower = values_time - 1
        if self._max_past_horizon != -1:
          # tf.minimum handles the corner case where values_time - 1 is less
          # than max_past_horizon.
          num_lower = tf.minimum(num_lower, self._max_past_horizon)

        num_upper = values_time - 1
        if self._max_future_horizon != -1:
          # tf.minimum handles the corner case where values_time - 1 is less
          # than max_future_horizon.
          num_upper = tf.minimum(num_upper, self._max_future_horizon)
        visibility_mask = _ones_matrix_band_part(
            values_time,
            values_time,
            num_lower=num_lower,
            num_upper=num_upper,
            # [1, 1, queries_time, keys_time] so it broadcasts across batch size
            # and heads.
            out_shape=[1, 1, values_time, values_time],
        )
        visibility_mask = tf.cast(visibility_mask, compute_dtype)
        visibility_mask.shape.assert_is_compatible_with(
            [1, 1, values_time_static, values_time_static]
        )
      else:
        # If both max_past_horizon and max_future_horizon are -1 then we operate
        # unmasked.
        visibility_mask = None

      # Mask out invalid timesteps in the input sequence so that we do not
      # attend to invalid timesteps. By shaping it [b, 1, 1, key_time], we
      # ensure that each query timestep cannot see invalid timesteps. If the
      # query timestep itself is invalid, it will be masked below
      valid_mask = tf.cast(x.mask[:, tf.newaxis, tf.newaxis, :], compute_dtype)
      valid_mask.shape.assert_is_compatible_with(
          [batch_size_static, 1, 1, values_time_static]
      )

      # Since we need to specify which timesteps are invalid, we invert the
      # banded matrix.
      invalid_mask = 1.0 - (
          visibility_mask * valid_mask
          if visibility_mask is not None
          else valid_mask
      )
      if visibility_mask is not None:
        invalid_mask.shape.assert_is_compatible_with(
            [batch_size_static, 1, values_time_static, values_time_static]
        )
      else:
        invalid_mask.shape.assert_is_compatible_with(
            [batch_size_static, 1, 1, values_time_static]
        )
      return invalid_mask

    queries, keys, values = tf.unstack(
        self._qkv.layer(x, training=training).values, axis=2
    )

    if self._query_network:
      queries = self._query_network.layer(
          types.Sequence(queries, x.mask),
          training=training,
          constants=constants,
      ).values
    if self._key_network:
      keys = self._key_network.layer(
          types.Sequence(keys, x.mask),
          training=training,
          constants=constants,
      ).values
    if self._value_network:
      values = self._value_network.layer(
          types.Sequence(values, x.mask),
          training=training,
          constants=constants,
      ).values

    if self._query_chunk_size is not None or self._key_chunk_size is not None:
      invalid_mask_fn = functools.partial(
          _dot_product_attention_invalid_mask_fn,
          max_past_horizon=self._max_past_horizon,
          max_future_horizon=self._max_future_horizon,
      )
      if self._relative_position_embedding is None:
        logit_bias_fn = None
      else:
        logit_bias_fn = functools.partial(
            _dot_product_attention_logit_bias_fn,
            relative_position_embedding=self._relative_position_embedding,
        )
      context_vectors = _chunked_dot_product_attention(
          queries=queries,
          keys=keys,
          keys_mask=x.mask,
          values=values,
          invalid_mask_fn=invalid_mask_fn,
          logit_bias_fn=logit_bias_fn,
          num_heads=self._num_heads,
          units_per_head=self._units_per_head,
          attention_logits_soft_cap=self._attention_logits_soft_cap,
          query_chunk_size=self._query_chunk_size,
          key_chunk_size=self._key_chunk_size,
      )
      emits = ()
    else:
      relative_logits = None
      if self._relative_position_embedding:
        relative_logits = self._relative_position_embedding.get_position_bias(
            queries
        )
        # [1, h, q, k]
        relative_logits.shape.assert_is_compatible_with([
            None,
            self._num_heads,
            values_time_static,
            values_time_static,
        ])
        assert relative_logits.dtype == compute_dtype
      invalid_mask = default_invalid_mask()
      context_vectors, probabilities, _ = _dot_product_attention(
          queries=queries,
          keys=keys,
          values=values,
          invalid_mask=invalid_mask,
          logit_bias=relative_logits,
          training=training,
          num_heads=self._num_heads,
          units_per_head=self._units_per_head,
          attention_logits_soft_cap=self._attention_logits_soft_cap,
          attention_probabilities_dropout=self._attention_probabilities_dropout,
      )
      emits = AttentionEmits(types.Sequence(probabilities, x.mask))

    # Mask out invalid input timesteps. The invalid_mask we computed above
    # ensures that no timestep in context_vectors could have been computed from
    # invalid timesteps.
    context_vectors = types.Sequence(context_vectors, x.mask).mask_invalid()

    return context_vectors, emits


class DotProductAttention(types.Emitting):
  """Dot product attention."""

  def __init__(
      self,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      attention_probabilities_dropout_rate: float = 0.0,
      broadcast_dropout_across_queries: bool = False,
      use_bias: bool = False,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      attention_logits_soft_cap: float | None = None,
      query_chunk_size: int | None = None,
      key_chunk_size: int | None = None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
      name: Optional[str] = None,
  ):
    """A dot-product attention layer as in Transformers.

    Args:
      source_name: The key to lookup source sequence from constants dictionary.
      num_heads: The number of attention heads.
      units_per_head: The number of units per head.
      attention_probabilities_dropout_rate: The dropout rate for the attention
        probabilities.
      broadcast_dropout_across_queries: Whether to broadcast the dropout across
        the query time dimension as is done in T5.
      use_bias: Whether to learn a bias in the query/key/value projection.
      kernel_initializer: Kernel initializer for the query/key/value projection.
      bias_initializer: Bias initialier for the query/key/value projection.
      attention_logits_soft_cap: If non-zero, a soft cap applied to attention
        logits to prevent outliers from dominating the softmax. Empirically,
        50.0 works well across a variety of tasks. Implemented as tanh(logits /
        cap) * cap.
      query_chunk_size: The query chunk size to use. Computes attention in
        parallel for blocks of queries of this size. If enabled, no emits are
        produced (e.g. attention probabilities for debugging).
      key_chunk_size: The key chunk size to use. If None, computes chunked
        attention over the entire key sequence for each block of queries. If
        non-None, computes attention for blocks of [query_chunk_size,
        key_chunk_size] at a time, bounding the overall memory usage to
        O(query_chunk_size * key_chunk_size). If enabled, no emits are produced
        (e.g. attention probabilities for debugging).
      query_network: An optional layer for processing queries before logit
        calculations are performed.
      key_network: An optional layer for processing keys before logit
        calculations are performed. Must have same output ratio as
        values_network.
      value_network: An optional layer for processing values before attention
        weighted summation is performed. Must have same output ratio as
        keys_network.
      name: An optional name for the layer.
    """
    super().__init__(name=name)
    _validate_attention(source_name, num_heads, units_per_head, name)
    self._num_heads = num_heads
    self._units_per_head = units_per_head
    num_units = num_heads * units_per_head
    self._source_name = source_name
    self._attention_logits_soft_cap = attention_logits_soft_cap
    if (
        attention_logits_soft_cap is not None
        and attention_logits_soft_cap <= 0.0
    ):
      raise ValueError(
          f'{attention_logits_soft_cap=} should be a positive number.'
      )

    self._query_chunk_size = query_chunk_size
    self._key_chunk_size = key_chunk_size
    chunking_enabled = query_chunk_size or key_chunk_size
    if attention_probabilities_dropout_rate > 0.0 and chunking_enabled:
      raise NotImplementedError(
          'Attention probability dropout is incompatible with chunking.'
      )

    with self.name_scope as name_scope:
      if broadcast_dropout_across_queries:
        # [batch, num_heads, query_time, source_time]
        noise_shape = [None, None, 1, None]
      else:
        noise_shape = None
      self._attention_probabilities_dropout = tf.keras.layers.Dropout(
          attention_probabilities_dropout_rate,
          noise_shape=noise_shape,
          name='attention_probabilities_dropout',
      )
      self._q = tf.keras.layers.Dense(
          num_units,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          name=name_scope + 'query_projection/',
      )
      self._kv = tf.keras.layers.Dense(
          num_units * 2,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          name=name_scope + 'key_value_projection/',
      )

      if callable(query_network):
        query_network = query_network()
      if callable(key_network):
        key_network = key_network()
      if callable(value_network):
        value_network = value_network()

      self._query_network = query_network
      self._key_network = key_network
      self._value_network = value_network

      key_output_ratio = (
          key_network.output_ratio if key_network else fractions.Fraction(1)
      )
      value_output_ratio = (
          value_network.output_ratio if value_network else fractions.Fraction(1)
      )
      if key_output_ratio != value_output_ratio:
        raise ValueError(
            f'{key_output_ratio=} and {value_output_ratio=} must be equal.'
        )

  @property
  def supports_step(self) -> bool:
    return self._query_network.supports_step if self._query_network else True

  @property
  def block_size(self) -> int:
    return self._query_network.block_size if self._query_network else 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    return (
        self._query_network.output_ratio
        if self._query_network
        else fractions.Fraction(1)
    )

  def get_initial_state(
      self, x: types.Sequence, constants: types.Constants | None = None
  ) -> types.State:
    if self._query_network:
      batch_size = utils.smart_dimension_size(x.values, 0)
      q_input = types.Sequence(
          tf.zeros(
              (batch_size, 0, self._num_heads, self._units_per_head),
              x.values.dtype,
          ),
          tf.zeros((batch_size, 0), types.MASK_DTYPE),
      )
      return self._query_network.get_initial_state(q_input, constants)
    return ()

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'DotProductAttention requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return tf.TensorShape([self._num_heads, self._units_per_head])

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> types.Emits:
    if self._query_chunk_size is None and self._key_chunk_size is None:
      source = _get_source(self, self._source_name, constants)
      spec = types.Sequence(
          tf.TensorSpec(
              tf.TensorShape(
                  [self._num_heads, source.values.shape.dims[1].value]
              ),
              input_spec.dtype,
          ),
          tf.TensorSpec(tf.TensorShape([]), tf.float32),
      )
      return AttentionEmits(spec)
    return ()

  def _get_queries(self, x: types.Sequence) -> types.Sequence:
    batch_size, queries_time = utils.smart_dimension_size(x.values, [0, 1])
    return types.Sequence(
        tf.reshape(
            self._q(x.values),
            [batch_size, queries_time, self._num_heads, self._units_per_head],
        ),
        x.mask,
    )

  def _get_key_values(self, source: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size, keys_time = utils.smart_dimension_size(source.values, [0, 1])
    keys, values = tf.unstack(
        tf.reshape(
            self._kv(source.values),
            [batch_size, keys_time, 2, self._num_heads, self._units_per_head],
        ),
        axis=2,
    )
    keys = types.Sequence(keys, source.mask)
    values = types.Sequence(values, source.mask)
    return keys, values

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    source = _get_source(self, self._source_name, constants)

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
        logit_bias_fn=None,
        is_step=True,
        training=training,
    )
    return y, state, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    source = _get_source(self, self._source_name, constants)
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
        logit_bias_fn=None,
        is_step=False,
        training=training,
    )

  def _attention(
      self,
      queries: types.Sequence,
      keys: types.Sequence,
      values: types.Sequence,
      logit_bias_fn: (
          Callable[
              [tf.Tensor, tf.Tensor | int, tf.Tensor, tf.Tensor | int],
              tf.Tensor,
          ]
          | None
      ),
      is_step: bool,
      training: bool,
  ) -> tuple[types.Sequence, types.Emits]:
    # TODO(b/297462463): Step-wise support for chunked attention.
    if not is_step and (
        self._query_chunk_size is not None or self._key_chunk_size is not None
    ):
      invalid_mask_fn = functools.partial(
          _dot_product_attention_invalid_mask_fn,
          max_past_horizon=-1,
          max_future_horizon=-1,
      )
      context_vectors = _chunked_dot_product_attention(
          queries=queries.values,
          keys=keys.values,
          keys_mask=keys.mask,
          values=values.values,
          invalid_mask_fn=invalid_mask_fn,
          logit_bias_fn=logit_bias_fn,
          num_heads=self._num_heads,
          units_per_head=self._units_per_head,
          attention_logits_soft_cap=self._attention_logits_soft_cap,
          query_chunk_size=self._query_chunk_size,
          key_chunk_size=self._key_chunk_size,
      )
      emits = ()
    else:
      queries_time = utils.smart_dimension_size(queries.values, 1)
      keys_time = utils.smart_dimension_size(keys.values, 1)
      invalid_mask = 1.0 - keys.mask[:, tf.newaxis, tf.newaxis, :]
      logit_bias = None
      if logit_bias_fn is not None:
        logit_bias = logit_bias_fn(0, queries_time, 0, keys_time)
        # [b, q, h, k] -> [b, h, q, k]
        logit_bias = tf.transpose(logit_bias, [0, 2, 1, 3])
      context_vectors, probabilities, _ = _dot_product_attention(
          queries=queries.values,
          keys=keys.values,
          values=values.values,
          invalid_mask=invalid_mask,
          logit_bias=logit_bias,
          training=training,
          num_heads=self._num_heads,
          units_per_head=self._units_per_head,
          attention_logits_soft_cap=self._attention_logits_soft_cap,
          attention_probabilities_dropout=self._attention_probabilities_dropout,
      )
      # If we are stepping with chunking enabled, don't emit AttentionEmits.
      # TODO(rryan): Figure out a way to produce different emits in layer/step.
      if self._key_chunk_size is None and self._query_chunk_size is None:
        emits = AttentionEmits(types.Sequence(probabilities, queries.mask))
      else:
        emits = ()
    return types.Sequence(context_vectors, queries.mask).mask_invalid(), emits


class LocationSensitiveAttention(types.Emitting):
  """Location Sensitive attention mechanism.

  Similar to AdditiveAttention, but adds two additional features:
  1. prev attention alignment (local history)
  2. cum_sum of previous attention alignment (full history)
  """

  def __init__(
      self,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      location_num_filters: int,
      location_filter_size: int,
      location_filter_padding: str = 'same',
      name: Optional[str] = None,
  ):
    """Creates a location sensitive attention module."""
    super().__init__(name=name)
    _validate_attention(source_name, num_heads, units_per_head, name)
    if location_num_filters <= 0:
      raise ValueError(
          f'Expected location_num_filters > 0 for {name}. '
          f'Got {location_num_filters}'
      )
    if location_filter_size <= 0:
      raise ValueError(
          f'Expected location_filter_size > 0 for {name}. '
          f'Got {location_filter_size}'
      )
    if location_filter_padding not in ('same', 'valid'):
      raise ValueError(
          f'location_filter_padding must be "same" or "valid" for {name}. '
          f'Got "{location_filter_padding}".'
      )
    self._source_name = source_name
    self._num_heads = num_heads
    self._units_per_head = units_per_head
    num_units = num_heads * units_per_head
    with self.name_scope as name_scope:
      # GlorotNormal() scaled independently for each head.
      self._attn_v = tf.Variable(
          lambda: tf.initializers.VarianceScaling(
              mode='fan_out', distribution='truncated_normal'
          )([num_heads, units_per_head], dtype=utils.variable_dtype()),
          name='v',
      )
      self._source_projection = tf.keras.layers.Dense(
          num_units, use_bias=True, name=name_scope + 'source_projection/'
      )
      self._query_projection = tf.keras.layers.Dense(
          num_units, use_bias=False, name=name_scope + 'query_projection/'
      )
      # Implement independent location filters for each attention head using a
      # depthwise 2D convolution along a [batch, source_time, 2, num_heads]
      # shaped tensor. Since we really only need a 1D convlution along the
      # source_time axis, we use 'valid' padding and, if specified, manually
      # implement 'same' pre-padding along that axis.
      # If 'valid' padding is used, we post-pad the convolution output to have
      # the same length as the input.
      pad_amount = location_filter_size - 1
      if location_filter_padding == 'same':
        left = pad_amount // 2
        right = pad_amount - left
        pre_pad = [tf.keras.layers.ZeroPadding2D(((left, right), (0, 0)))]
        post_pad = []
      else:
        pre_pad = []
        post_pad = [tf.keras.layers.ZeroPadding2D(((0, pad_amount), (0, 0)))]
      self._location_filters = tf.keras.Sequential(
          pre_pad
          + [
              tf.keras.layers.DepthwiseConv2D(
                  (location_filter_size, 2),
                  depth_multiplier=location_num_filters,
                  padding='valid',
                  use_bias=False,
                  kernel_initializer=tf.random_uniform_initializer(),
                  name=name_scope + 'location_filters/',
              )
          ]
          + post_pad
          + [tf.keras.layers.Reshape((-1, num_heads, location_num_filters))]
      )
      self._location_projection_kernel = tf.Variable(
          lambda: tf.initializers.VarianceScaling(
              mode='fan_out', distribution='truncated_normal'
          )(
              [num_heads, location_num_filters, units_per_head],
              dtype=utils.variable_dtype(),
          ),
          name='location_projection',
      )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    source = _get_source(self, self._source_name, constants)
    batch_size, source_length = utils.smart_dimension_size(
        source.values, [0, 1]
    )

    # The location state tensor is shaped [batch_size, source_length, 2,
    # num_heads] where the 1st inner dimension is the current attention energy
    # for the  source timestep and the 2nd inner dimension is the cumulative
    # attention energy for the source timestep. We initialize the location state
    # with ones in the first timestep and zeros everywhere else.
    cumulative_prob_length = tf.maximum(0, source_length - 1)
    compute_dtype = utils.compute_dtype()
    location_initial_state = tf.concat(
        [
            tf.ones([batch_size, 1, 2, self._num_heads], dtype=compute_dtype),
            tf.zeros(
                [batch_size, cumulative_prob_length, 2, self._num_heads],
                dtype=compute_dtype,
            ),
        ],
        1,
    )
    return types.Sequence(location_initial_state, source.mask)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'LocationSensitiveAttention requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    source = _get_source(self, self._source_name, constants)
    return tf.TensorShape([self._num_heads, source.values.shape.dims[2].value])

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> AttentionEmits:
    source = _get_source(self, self._source_name, constants)
    return AttentionEmits(
        types.Sequence(
            tf.TensorSpec(
                tf.TensorShape(
                    [self._num_heads, source.values.shape.dims[1].value]
                ),
                input_spec.dtype,
            ),
            tf.TensorSpec(tf.TensorShape([]), tf.float32),
        )
    )

  def _single_step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, AttentionEmits]:
    """Implements a single step of location sensitive attention."""
    batch_size = utils.smart_dimension_size(x.values, 0)
    x.values.shape.assert_is_compatible_with([None, 1, None])
    source = _get_source(self, self._source_name, constants)
    num_heads = self._num_heads
    units_per_head = self._units_per_head
    query = x.apply_values(self._query_projection)
    query = query.apply_values(
        tf.reshape, [batch_size, 1, num_heads, units_per_head]
    )

    keys = source.apply_values(self._source_projection).mask_invalid()
    keys_time = utils.smart_dimension_size(keys.values, 1)
    keys = keys.apply_values(
        tf.reshape, [batch_size, keys_time, num_heads, units_per_head]
    )

    # Convolution over the [batch, keys_time, 2, num_heads] tensor containing
    # the last step's attention probabilities and the cumulative probabilities
    # across all previous frames produces the
    # [batch, keys_time, num_heads, units_per_head] "location features" used to
    # condition the attention on this frame.
    def compute_location_features(x):
      features = self._location_filters(x)
      return tf.einsum(
          'BiNL,NLH->BiNH',
          features,
          tf.cast(self._location_projection_kernel, features.dtype),
      )

    location = state.apply_values(compute_location_features).mask_invalid()

    # Broadcast to [b, keys_time, num_heads, units_per_head]
    hidden = tf.tanh(query.values + keys.values + location.values)

    logits = tf.einsum(
        'BjNH,NH->BNj', hidden, tf.cast(self._attn_v, hidden.dtype)
    )
    # [b, num_heads, keys_time]
    mask = keys.mask[:, tf.newaxis, :]
    logits = tf.where(mask > 0.0, logits, _INVALID_LOGIT_VALUE)
    probabilities = _softmax_in_at_least_float32(logits)

    # Expand to [b, 1, num_heads, keys_time].
    probabilities = probabilities[:, tf.newaxis, :, :]

    context_vector = types.Sequence(
        tf.einsum('BiNj,BjS->BiNS', probabilities, source.values), query.mask
    ).mask_invalid()
    # Compute the updated location state (the current step's probability
    # concatenated with the cumulative attention probabilities).
    prob_transposed = tf.transpose(probabilities[:, 0], [0, 2, 1])
    # [b, keys_time, num_heads]
    cumulative_prob = state.values[:, :, 1, :] + prob_transposed
    state = types.Sequence(
        tf.stack([prob_transposed, cumulative_prob], axis=2), state.mask
    )
    probabilities = types.Sequence(probabilities, query.mask).mask_invalid()
    emits = AttentionEmits(probabilities)
    return context_vector, state, emits

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
      unroll: bool = True,
  ) -> Tuple[types.Sequence, types.State, AttentionEmits]:
    step_fn = functools.partial(
        self._single_step_with_emits, training=training, constants=constants
    )
    x_length = utils.smart_dimension_size(x.values, 1)
    num_blocks = x_length // self.block_size
    if unroll:
      return utils.step_by_step_fn_static(
          step_fn, num_blocks, self.block_size, x, state
      )
    else:
      return utils.step_by_step_fn_dynamic(
          step_fn,
          x_length,
          x.channel_shape,
          self.get_output_spec_for_sequence(x, constants),
          self.get_emit_specs_for_sequence(x, constants),
          num_blocks,
          self.block_size,
          int(self.block_size * self.output_ratio),
          x,
          state,
      )

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    if initial_state is None:
      initial_state = self.get_initial_state(x, constants)
    outputs, _, emits = self.step_with_emits(
        x,
        training=training,
        state=initial_state,
        constants=constants,
        unroll=False,
    )
    return outputs, emits


class DynamicConvolutionAttention(types.Emitting):
  """A multi-headed dynamic convolution attention (DCA) layer.

  paper: https://arxiv.org/abs/1910.10288.
  """

  # To prevent attention from looking at regions of the encoder state that are
  # invalid, a bias mask is added before applying the attention. This scaling
  # factor indicates the absolute value of the bias mask for invalid regions.
  # Larger values make it less possible for the attention module to attend to
  # invalid regions.
  _LOGIT_BIAS_SCALE = 1e6

  def __init__(
      self,
      source_name: str,
      max_forward_step: int,
      prior_alpha: float,
      prior_beta: float,
      num_static_filters: int,
      num_dynamic_filters: int,
      dynamic_filter_hidden_dim: int,
      name: Optional[str] = None,
  ):
    """Initializes DynamicConvolutionAttention layer.

    Args:
      source_name: The key to lookup source sequence from constants dictionary.
      max_forward_step: Maximum forward step allowed by prior. This also
        determines size of all filters (filter_size = 2 * max_forward_step + 1).
      prior_alpha: The alpha parameters for the Beta-Binomial alignment step
        size prior. The prior filter taps for forward movement are computed as:
        BetaBinomialPMF(k, N, alpha, beta) for k in [0, 1, ..., N], where N is
        max_forward_step. Setting alpha = beta = 1.0 gives a uniform prior. For
        guidance on tweaking these parameters, see: -
        https://en.wikipedia.org/wiki/Beta-binomial_distribution -
      prior_beta: The beta parameter for the Beta-Binomial alignment step size
        prior.
      num_static_filters: Number of static filters.
      num_dynamic_filters: Number of dynamic filters computed from the query.
      dynamic_filter_hidden_dim: Size of the hidden layer that computes the
        dynamic filters from the query.
      name: An optional module name.
    """
    super().__init__(name)
    if not source_name:
      raise ValueError(f'Expected non-empty source_name for {source_name}.')
    if num_static_filters <= 0:
      raise ValueError(
          f'Expected num_static_filters > 0 for {name}. '
          f'Got {num_static_filters}'
      )
    if num_dynamic_filters <= 0:
      raise ValueError(
          f'Expected num_dynamic_filters > 0 for {name}. '
          f'Got {num_dynamic_filters}'
      )
    # TODO(soroosho): Support multiple heads. Currently, only single-head is
    # supported, but the output is accomodating extra dimension for heads, to
    # be consistent with other SL attention layers.
    self._num_heads = 1
    self._source_name = source_name
    self._max_forward_step = max_forward_step
    self._prior_alpha = prior_alpha
    self._prior_beta = prior_beta
    self._filter_size = filter_size = 2 * max_forward_step + 1

    with self.name_scope as name_scope:
      self._attn_v = tf.keras.layers.Dense(1, name=f'{name_scope}v/')

      # Input: [b, source_time, 1].
      self._static_location_conv = tf.keras.layers.Conv1D(
          num_static_filters,
          filter_size,
          padding='same',
          use_bias=True,
          kernel_initializer=tf.random_uniform_initializer(),
          name=f'{name_scope}static_location_filters/',
      )
      # Output: [b, source_time, num_static_filters].

      self._dynamic_location_filter_and_bias_mlp = tf.keras.Sequential(
          [
              # Input: [b, 1, query_dim].
              tf.keras.layers.experimental.EinsumDense(
                  'Biq,qH->BiH',
                  bias_axes='H',
                  output_shape=(1, dynamic_filter_hidden_dim),
                  activation=tf.nn.tanh,
                  name=f'{name_scope}dynamic_location_filters_einsum_dense0/',
              ),
              # [b, 1, dynamic_filter_hidden_dim].
              tf.keras.layers.experimental.EinsumDense(
                  'BiH,iHFjD->BFjD',
                  bias_axes='FjD',
                  output_shape=(filter_size + 1, 1, num_dynamic_filters),
                  name=f'{name_scope}dynamic_location_filters_einsum_dense1/',
              ),
              # Output: [b, filter_size + 1, 1, num_dynamic_filters].
          ]
      )

    # [b, source_time, 1]
    self._prior_filter = self._build_prior_filter()

  def _build_prior_filter(self) -> np.ndarray:
    """Build prior location filter."""

    # Construct prior filter.
    # [filter_size, chan_in=1, chan_out=1]
    prior_filter = np.zeros([self._filter_size, 1, 1], dtype='float32')
    # The filter tap at the center index provides no relative movement. Indices
    # after center_ind provide increasing amounts of forward movement; those
    # before center_ind provide increasing amounts of backward movement.
    center_ind = self._filter_size // 2
    # The prior filter acts as a monotonicity constraint and it limits the
    # maximum forward movement to `max_forward_step` steps.
    # The filter taps for forward movement are determined using a Beta-Binomial
    # distribution.
    prior_filter[center_ind:, 0, 0] = self._beta_binomial_pmf(
        np.arange(self._max_forward_step + 1),
        self._max_forward_step,
        self._prior_alpha,
        self._prior_beta,
    )

    # Because tf.nn.conv1d is actually correlation and not convolution, we have
    # to flip the filter.
    prior_filter = prior_filter[::-1]
    return prior_filter

  def _beta_binomial_pmf(self, k, n, a, b) -> np.ndarray:
    """Compute Beta-Binomial probability mass function (PMF).

    See: https://en.wikipedia.org/wiki/Beta-binomial_distribution

    Args:
      k: Tensor of locations at which to evaluate the PMF (integers in 0 <= k <=
        n).
      n: Number of Bernoulli trials.
      a: Alpha parameter of the distribution (float).
      b: Beta parameter of the distribution (float).

    Returns:
      A tensor containing the probability mass at each location in k.
    """
    return (
        scipy.special.binom(n, k)
        * scipy.special.beta(k + a, n - k + b)
        / scipy.special.beta(a, b)
    )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    source = _get_source(self, self._source_name, constants)
    batch_size, source_length = utils.smart_dimension_size(
        source.values, [0, 1]
    )

    # The state tensor is shaped [batch_size, source_length, 1] where the 1st
    # inner dimension is the current attention energy for the source timestep.
    # We initialize the location state with ones in the first timestep and zeros
    # everywhere else.
    location_initial_state = tf.sequence_mask(
        tf.fill([batch_size], 1),
        source_length,
        dtype=utils.compute_dtype(),
    )[:, :, tf.newaxis]

    # TODO(b/196245589): Fix SL gradient tests with zero-length sources.
    return types.Sequence(location_initial_state, source.mask)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'DynamicConvoluationAttention requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    source = _get_source(self, self._source_name, constants)
    return tf.TensorShape([self._num_heads, source.values.shape.dims[2].value])

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> AttentionEmits:
    source = _get_source(self, self._source_name, constants)
    return AttentionEmits(
        types.Sequence(
            tf.TensorSpec(
                tf.TensorShape(
                    [self._num_heads, source.values.shape.dims[1].value]
                ),
                input_spec.dtype,
            ),
            tf.TensorSpec(tf.TensorShape([]), types.MASK_DTYPE),
        )
    )

  def _single_step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, AttentionEmits]:
    """Implements a single step of dynamic convoluation attention."""
    # [b, 1, query_dim].
    query = x
    query.values.shape.assert_is_compatible_with([None, 1, None])
    # [b, source_time, key_dim].
    source = _get_source(self, self._source_name, constants)

    # Output: [b, source_time, num_static_filters].
    static_filter_output = state.apply_values(self._static_location_conv).values

    # [b, filter_size + 1, 1, num_dynamic_filters]
    dynamic_filters_and_biases = self._dynamic_location_filter_and_bias_mlp(
        query.values
    )

    # dynamic_filters: [b, filter_size, 1, num_dynamic_filters]
    # dynamic_biases: [b, 1, num_dynamic_filters]
    dynamic_biases, dynamic_filters = tf.split(
        dynamic_filters_and_biases, [1, self._filter_size], axis=1
    )

    # [b, 1, num_dynamic_filters]
    dynamic_biases = tf.squeeze(dynamic_biases, 1)

    # [b, source_time, num_dynamic_filters]
    dynamic_filter_output = utils.dynamic_filter_conv1d(
        state.values, dynamic_filters
    )
    # [b, source_time, num_dynamic_filters]
    dynamic_filter_output += dynamic_biases

    # Compute prior outputs and logits from location state using prior filter.
    # [b, source_time, 1]
    prior_outputs = tf.nn.conv1d(
        state.values,
        tf.constant(self._prior_filter, dtype=state.values.dtype),
        stride=1,
        padding='SAME',
    )
    # compute logits from values using log, and set values that would underflow
    # to -self._LOGIT_BIAS_SCALE to get hard zeros out of the softmax.
    # [b, source_time, 1]
    prior_logits = utils.log_without_underflow(
        prior_outputs, -self._LOGIT_BIAS_SCALE
    )

    # [b, source_time, loc_num_dynamic_filters + loc_num_static_filters]
    hidden = tf.concat([dynamic_filter_output, static_filter_output], axis=-1)
    hidden = tf.nn.tanh(hidden)

    # [b, source_time, 1]
    logits = self._attn_v(hidden, training=training) + prior_logits
    logits = tf.where(
        source.mask[:, :, tf.newaxis] > 0.0,
        logits,
        tf.constant(-self._LOGIT_BIAS_SCALE, dtype=logits.dtype),
    )
    # [b, source_time]
    logits = tf.squeeze(logits, axis=-1)

    # [b, source_time]
    probabilities = _softmax_in_at_least_float32(logits)
    # [b, 1, 1, source_time].
    probabilities = probabilities[:, tf.newaxis, tf.newaxis, :]

    context_vector = types.Sequence(
        tf.einsum('BiNj,BjS->BiNS', probabilities, source.values), query.mask
    ).mask_invalid()
    # Compute the updated location state (the current step's probability
    # concatenated with the cumulative attention probabilities).
    prob_transposed = tf.transpose(probabilities[:, 0, :, :], [0, 2, 1])
    state = types.Sequence(prob_transposed, state.mask)
    probabilities = types.Sequence(probabilities, query.mask).mask_invalid()
    emits = AttentionEmits(probabilities)
    return context_vector, state, emits

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
      unroll: bool = True,
  ) -> Tuple[types.Sequence, types.State, AttentionEmits]:
    step_fn = functools.partial(
        self._single_step_with_emits, training=training, constants=constants
    )
    x_length = utils.smart_dimension_size(x.values, 1)
    num_blocks = x_length // self.block_size
    if unroll:
      return utils.step_by_step_fn_static(
          step_fn, num_blocks, self.block_size, x, state
      )
    else:
      return utils.step_by_step_fn_dynamic(
          step_fn,
          x_length,
          x.channel_shape,
          self.get_output_spec_for_sequence(x, constants),
          self.get_emit_specs_for_sequence(x, constants),
          num_blocks,
          self.block_size,
          int(self.block_size * self.output_ratio),
          x,
          state,
      )

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    if initial_state is None:
      initial_state = self.get_initial_state(x, constants)
    outputs, _, emits = self.step_with_emits(
        x,
        training=training,
        state=initial_state,
        constants=constants,
        unroll=False,
    )
    return outputs, emits
