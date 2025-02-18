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
"""Time-varying layers."""

from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class SequenceEmbedding(types.SequenceLayer):
  """Computes sequence embeddings of integer input codes.

  Provides step-dependent embeddings with num_groups embeddings, where sequence
  step specifies the group index, and for steps above num_groups - 1, we use the
  last group (i.e., group_index = num_groups - 1). So, this layer is appropriate
  for fixed length sequences, where num_groups is set to the fixed length.
  """

  def __init__(
      self,
      dimension: int,
      num_embeddings: int,
      num_groups: int,
      embeddings_initializer='uniform',
      embeddings_regularizer=None,
      activity_regularizer=None,
      embeddings_constraint=None,
      trainable: bool = True,
      name: str | None = None,
  ):
    super().__init__(name)
    self._num_groups = num_groups
    # Reserve one embedding for masking token, in this case, -1 will be used for
    # padding outside this function
    self._num_embeddings = num_embeddings
    with self.name_scope:
      self._embedding = tf.keras.layers.Embedding(
          input_dim=self._num_embeddings * self._num_groups,
          output_dim=dimension,
          embeddings_initializer=embeddings_initializer,
          embeddings_regularizer=embeddings_regularizer,
          activity_regularizer=activity_regularizer,
          embeddings_constraint=embeddings_constraint,
          trainable=trainable,
          name='embedding',
      )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: types.Constants | None = None,
  ) -> tf.TensorShape:
    return input_shape.concatenate(self._embedding.output_dim)

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    assert input_dtype in [tf.uint8, tf.int32, tf.int64]
    return utils.compute_dtype()

  def get_initial_state(
      self, x: types.Sequence, constants: types.Constants | None = None
  ) -> types.State:
    return tf.zeros(shape=(), dtype=tf.int32)

  def _time_wise_embed(
      self, values: tf.Tensor, start_time: tf.Tensor | int, training: bool
  ) -> tf.Tensor:
    tf.debugging.assert_greater_equal(
        values, 0, message='Out of range lookup index (< 0).'
    )
    tf.debugging.assert_less(
        values,
        self._num_embeddings,
        message='Out of range lookup index (>= num_embeddings).',
    )
    values = tf.cast(values, tf.int32)

    block_size = utils.smart_dimension_size(values, 1)
    # [t].
    steps = start_time + tf.range(block_size, dtype=tf.int32)
    steps = tf.minimum(steps, self._num_groups - 1)

    # Reshape into [t, 1, ..., 1].
    steps = tf.reshape(
        steps, tf.stack([block_size, *([1] * (values.shape.rank - 2))], axis=0)
    )

    # The lookup indices are broadcast-added into shape of values ([b, t, ...])
    # and embeddings is shaped [b, t, ..., embedding_dim].
    embeddings = self._embedding(
        values + steps * self._num_embeddings, training=training
    )
    return embeddings

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    """Steps forward, input shape: [..., block_size, num_groups]."""
    x = x.apply_values(
        lambda v: self._time_wise_embed(v, state, training=training)
    ).mask_invalid()
    time = utils.smart_dimension_size(x.values, 1)
    return x, state + time

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State | None = None,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del initial_state
    del constants
    return x.apply_values(
        lambda v: self._time_wise_embed(v, 0, training=training)
    ).mask_invalid()


class SequenceDense(types.SequenceLayer):
  """Step-dependent Dense layer.

  Provides step-dependent Dense layers up to num_steps, where the sequence
  step determines which Dense layer is used. This layer can be used with fixed
  length sequences, where num_steps is set to the fixed length. For variable
  length sequences, set num_steps to the maximum sequence length.

  Each step-dependent Dense layer only operates on the inputs at the associated
  timestep, i.e., y[t] = Dense_t(x[t]), where Dense_t is the Dense layer
  used for timestep t.
  """

  def __init__(
      self,
      units: int,
      num_steps: int,
      activation=None,
      use_bias: bool = True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._units = units
    self._num_steps = num_steps
    self._activation_fn = tf.keras.activations.get(activation)
    self._use_bias = use_bias
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)

    self._kernel = None
    self._bias = None
    self._built = False

  def _build(self, x: types.Sequence):
    if self._built:
      return
    with self.name_scope:
      # TODO(ebattenberg): Support additional channel dims like standard Dense
      # layer.
      if x.values.shape.rank != 3:
        raise ValueError('SequenceDense requires rank 3 input.')
      input_dim = x.values.shape[2]
      kernel_shape = [self._num_steps, input_dim, self._units]
      self._kernel = tf.Variable(
          lambda: self._kernel_initializer(kernel_shape),
          name='kernel',
      )
      if self._use_bias:
        self._bias = tf.Variable(
            lambda: self._bias_initializer([self._num_steps, self._units]),
            name='bias',
        )
    self._built = True

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: types.Constants | None = None,
  ) -> tf.TensorShape:
    del constants
    input_shape.with_rank_at_least(1)
    return input_shape[:-1].concatenate(self._units)

  def get_initial_state(
      self, x: types.Sequence, constants: types.Constants | None = None
  ) -> types.State:
    del x, constants
    return tf.zeros(shape=(), dtype=tf.int32)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    """Steps forward, input shape: [..., block_size, num_groups]."""
    del training, constants
    self._build(x)

    start_time = state
    time_delta = utils.smart_dimension_size(x.values, 1)
    _check_step_bounds(start_time, start_time + time_delta, self._num_steps)

    # Slice [T, I, O] kernel to be used with current timesteps.
    step_kernel = tf.slice(
        # self._units is required to prevent loss of channel dim shape.
        self._kernel, [start_time, 0, 0], [time_delta, -1, self._units]
    )
    net = tf.einsum('BTI,TIO->BTO', x.values, step_kernel)
    if self._use_bias:
      # Slice [T, O] bias to be used with current timesteps.
      step_bias = tf.slice(
          self._bias, [start_time, 0], [time_delta, self._units]
      )
      net += step_bias
    net = self._activation_fn(net)
    y = types.Sequence(values=net, mask=x.mask).mask_invalid()

    return y, start_time + time_delta

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State | None = None,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if initial_state is None:
      initial_state = self.get_initial_state(x, constants)
    y, _ = self.step(x, initial_state, training, constants)
    return y


class MaskedDense(types.SequenceLayer):
  """Step-dependent causal masked Dense layers.

  Provides step-dependent masked Dense layers up to num_steps, where the
  sequence step determines which masked Dense layer is used. This layer can be
  used with fixed length sequences, where num_steps is set to the fixed length.
  For variable length sequences, set num_steps to the maximum sequence length.

  Masked Dense layers are causally masked so that y[t] (the output at time=t)
  is a Dense transformation of x[0:t+1] (the input from time=0 up to time=t,
  inclusive.); i.e., y[t] = Dense_t(x[0,...,t]), where Dense_t is the Dense
  layer used for timestep t.

  Note:
  This layer uses a masked kernel variable in order to maintain the
  autoregressive property, and therefore, half of the weights are unused and
  have zero gradient. This could affect the behavior of certain optimizers or
  normalization schemes (e.g., AdaFactor).
  """

  def __init__(
      self,
      units: int,
      num_steps: int,
      activation=None,
      use_bias: bool = True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._units = units
    self._num_steps = num_steps
    self._activation_fn = tf.keras.activations.get(activation)
    self._use_bias = use_bias
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)

    self._kernel = None
    self._bias = None
    self._input_dim = None
    self._built = False

  def _get_mask(self) -> tf.Tensor:
    # Construct a [T, T', 1, 1] upper triangular autoregressive mask matrix,
    # where T represents the input time dimension and T' represents the output
    # time dimension, and T = T' = num_steps.
    num_steps = self._num_steps
    mask = tf.linalg.band_part(tf.ones([num_steps, num_steps]), 0, -1)
    return tf.reshape(mask, [num_steps, num_steps, 1, 1], name='mask')

  def _build(self, x: types.Sequence):
    if self._built:
      return
    with self.name_scope:
      # TODO(ebattenberg): Support additional channel dims like standard Dense.
      if x.values.shape.rank != 3:
        raise ValueError('MaskedDense requires rank 3 input.')
      self._input_dim = x.values.shape[2]
      num_steps = self._num_steps

      # TODO(ebattenberg): Consider ways to eliminate unused (masked off)
      # weights (e.g., construct a masked kernel Tensor from smaller
      # sub-variables.)

      # Construct a [T, T', I, O] kernel tensor to project [B, T, I] input to
      # [B, T', O] output, where T represents the input time dimension and T'
      # represents the output time dimension, and T = T' = num_steps.
      kernel_shape = [num_steps, num_steps, self._input_dim, self._units]
      # Note that the mask is applied to the initializer and as a constraint to
      # keep the masked values from drifting due to things like weight decay.
      self._kernel = tf.Variable(
          lambda: self._get_mask() * self._kernel_initializer(kernel_shape),
          constraint=lambda x: self._get_mask() * x,
          name='kernel',
      )
      if self._use_bias:
        # Bias is applied to [B, T', O] kernel output.
        self._bias = tf.Variable(
            lambda: self._bias_initializer([num_steps, self._units]),
            name='bias',
        )
    self._built = True

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: types.Constants | None = None,
  ) -> tf.TensorShape:
    del constants
    input_shape.with_rank_at_least(1)
    return input_shape[:-1].concatenate(self._units)

  def get_initial_state(
      self, x: types.Sequence, constants: types.Constants | None = None
  ) -> types.State:
    del constants
    self._build(x)

    batch_size = utils.smart_dimension_size(x.values, 0)
    # When executing step-by-step, we need a buffer for tracking past inputs.
    t0 = tf.zeros(shape=(), dtype=tf.int32)
    input_buffer = types.Sequence(
        tf.zeros(
            (batch_size, self._num_steps, self._input_dim),
            dtype=x.values.dtype,
        ),
        tf.zeros((batch_size, self._num_steps), dtype=x.mask.dtype),
    )
    return t0, input_buffer

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    """Steps forward, input shape: [..., block_size, num_groups]."""
    del training, constants
    self._build(x)

    start_time, input_buffer = state
    time_delta = utils.smart_dimension_size(x.values, 1)
    end_time = start_time + time_delta
    _check_step_bounds(start_time, end_time, self._num_steps)

    # Append new input to end of buffer and truncate at the front:
    # e.g., [0, 0, x0, x1] -> [0, x0, x1, x2], where new input is [x2].
    input_buffer = input_buffer.concatenate(x)
    input_buffer = input_buffer[:, -self._num_steps:]
    # Roll buffer so padding is at the end:
    # e.g., [0, x0, x1, x2] -> [x0, x1, x2, 0]
    step_input = tf.roll(input_buffer.values, shift=end_time, axis=1)

    # Slice kernel and mask to produce current output timesteps.
    # Kernel is [T, T', I, O], mask is [T, T', 1, 1], where T represents the
    # input time dimension and T' represents the output time dimension,
    # and T = T' = num_steps.
    # step_kernel = self._kernel[:, start_time:end_time]
    step_kernel = tf.slice(
        # self._units is required to prevent loss of channel dim shape.
        self._kernel, [0, start_time, 0, 0], [-1, time_delta, -1, self._units]
    )
    # step_mask = self._get_mask()[:, start_time:end_time]
    step_mask = tf.slice(
        self._get_mask(), [0, start_time, 0, 0], [-1, time_delta, 1, 1]
    )
    # Even though the kernel tf.Variable has a mask constraint,
    # we still mask the op here to ensure the gradients to the masked off
    # regions are zero as well (so we don't pollute the gradient statistics).
    masked_step_kernel = step_mask * step_kernel

    net = tf.einsum('BTI,TtIO->BtO', step_input, masked_step_kernel)
    if self._use_bias:
      # Bias is [T', O].  Slice for current output timesteps.
      step_bias = tf.slice(
          self._bias, [start_time, 0], [time_delta, self._units]
      )
      net += step_bias
    net = self._activation_fn(net)
    y = types.Sequence(values=net, mask=x.mask).mask_invalid()

    state = (end_time, input_buffer)
    return y, state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State | None = None,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if initial_state is None:
      initial_state = self.get_initial_state(x, constants)
    y, _ = self.step(x, initial_state, training, constants)
    return y


def _check_step_bounds(start_time, end_time, num_steps):
  """Check that step indices are within range."""
  tf.debugging.assert_greater_equal(
      start_time, 0, message='Out of range step index (< 0).')
  tf.debugging.assert_less_equal(
      end_time, num_steps, message='Out of range step index (> num_steps).')
