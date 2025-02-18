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
"""Position layers."""

import math
from typing import Optional, Tuple

from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


# A negative enough value such that it underflows to a hard zero in softmax.
_INVALID_LOGIT_VALUE = -1e9


def _get_timing_signal_1d(
    length: tf.Tensor,
    channels: tf.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    start_index: int = 0,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
  """Gets a bunch of sinusoids of different frequencies.

  Copied from tensor2tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
      different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position
    dtype: Type of the returned timing signal.

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.cast(tf.range(length) + start_index, dtype)
  num_timescales = channels // 2
  log_timescale_increment = math.log(
      float(max_timescale) / float(min_timescale)
  ) / tf.maximum(tf.cast(num_timescales, dtype) - 1, 1)
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), dtype) * -log_timescale_increment
  )
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.math.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal


class AddTimingSignal(types.SequenceLayer):
  """Adds sinusoids at varying frequencies to the input channels dimension."""

  def __init__(
      self,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      trainable_scale: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale
    with self.name_scope:
      if trainable_scale:
        self._scale = tf.Variable(
            1.0, dtype=utils.variable_dtype(), name='scale'
        )
      else:
        self._scale = None

  def _check_inputs(self, x: types.Sequence):
    if not x.values.dtype.is_floating:
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    self._check_inputs(x)

    # State holds the current timestep (batched).
    batch_size = utils.smart_dimension_size(x.values, 0)
    return tf.zeros((batch_size, 1), dtype=tf.int32)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return input_shape

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    self._check_inputs(x)
    current_time = tf.reduce_max(state)
    time = utils.smart_dimension_size(x.values, 1)
    inner_shape = x.channel_shape
    num_elements = inner_shape.num_elements()
    timing_signal = _get_timing_signal_1d(
        time,
        num_elements,
        min_timescale=self._min_timescale,
        max_timescale=self._max_timescale,
        start_index=current_time,
        dtype=utils.compute_dtype(),
    )
    timing_signal = tf.reshape(timing_signal, [1, time] + inner_shape.as_list())
    if self._scale is not None:
      timing_signal *= tf.cast(self._scale, timing_signal.dtype)
    x = x.apply_values(lambda v: v + tf.cast(timing_signal, v.dtype))
    return x.mask_invalid(), state + time

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self._check_inputs(x)
    time = utils.smart_dimension_size(x.values, 1)
    inner_shape = x.channel_shape
    num_elements = inner_shape.num_elements()
    timing_signal = _get_timing_signal_1d(
        time,
        num_elements,
        min_timescale=self._min_timescale,
        max_timescale=self._max_timescale,
        start_index=0,
        dtype=utils.compute_dtype(),
    )
    timing_signal = tf.reshape(timing_signal, [1, time] + inner_shape.as_list())
    if self._scale is not None:
      timing_signal *= self._scale
    x = x.apply_values(lambda v: v + tf.cast(timing_signal, v.dtype))
    return x.mask_invalid()


class ConcatTimingSignal(types.SequenceLayer):
  """Concatenates sinusoid timing signals to the input inner channels dimension.

  If input channel rank > 1, then the timing signal is tiled over the outer
  channel dimensions, e.g.:
    - x: [b, t], t: [b, t, 4]) -> x_t: [b, t, 5]
    - x: [b, t, 3], t: [b, t, 4]) -> x_t: [b, t, 7]
    - x: [b, t, 3, 6], t: [b, t, 4]) -> x_t: [b, t, 3, 10]
  """

  def __init__(
      self,
      channels: int = 8,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._channels = channels
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale

  def _check_inputs(self, x: types.Sequence):
    if not x.values.dtype.is_floating:
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    self._check_inputs(x)
    # State holds the current timestep.
    return tf.constant(0, dtype=tf.int32)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    input_inner_dim = input_shape[-1] if input_shape.rank else 1
    return input_shape[:-1] + [input_inner_dim + self._channels]

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    self._check_inputs(x)
    current_time = state
    batch_size, time = utils.smart_dimension_size(x.values, [0, 1])
    timing_signal = _get_timing_signal_1d(
        time,
        self._channels,
        min_timescale=self._min_timescale,
        max_timescale=self._max_timescale,
        start_index=current_time,
        dtype=utils.compute_dtype(),
    )
    timing_signal = tf.tile(timing_signal, [batch_size, 1, 1])
    timing_signal_sequence = types.Sequence(timing_signal, x.mask)
    x = utils.sequence_broadcast_concat(x, timing_signal_sequence)
    return x.mask_invalid(), current_time + time

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self._check_inputs(x)
    batch_size, time = utils.smart_dimension_size(x.values, [0, 1])
    timing_signal = _get_timing_signal_1d(
        time,
        self._channels,
        min_timescale=self._min_timescale,
        max_timescale=self._max_timescale,
        start_index=0,
        dtype=utils.compute_dtype(),
    )
    timing_signal = tf.tile(timing_signal, [batch_size, 1, 1])
    timing_signal_sequence = types.Sequence(timing_signal, x.mask)
    x = utils.sequence_broadcast_concat(x, timing_signal_sequence)
    return x.mask_invalid()


class ConcatPositionEmbedding(types.SequenceLayer):
  """Concatenates a position embedding to the input inner channels dimension.

  Asserts that sequences processed by the layer are always shorter
  than timesteps to avoid a train-test mismatch.

  If input channel rank > 1, then the position embedding is tiled over the outer
  channel dimensions, e.g. (for channels = 4):
    - x: [b, t], t: [b, t, 4]) -> x_t: [b, t, 5]
    - x: [b, t, 3], t: [b, t, 4]) -> x_t: [b, t, 7]
    - x: [b, t, 3, 6], t: [b, t, 4]) -> x_t: [b, t, 3, 10]
  """

  def __init__(
      self,
      timesteps: int,
      channels: int,
      embeddings_initializer: tf.keras.initializers.Initializer | None = None,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._timesteps = timesteps
    self._channels = channels
    with self.name_scope:
      self._embedding = tf.keras.layers.Embedding(
          timesteps,
          channels,
          embeddings_initializer=embeddings_initializer,
          name='position_embedding',
      )
      self._embedding.build(tf.TensorShape([None, timesteps]))

  def _check_inputs(self, x: types.Sequence):
    if not x.values.dtype.is_floating:
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    self._check_inputs(x)
    # State holds the current timestep.
    return tf.constant(0, dtype=tf.int32)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    input_inner_dim = input_shape[-1] if input_shape.rank else 1
    return input_shape[:-1] + [input_inner_dim + self._channels]

  def _concat_position_embedding(
      self,
      x: types.Sequence,
      start_time: tf.Tensor,
  ) -> types.Sequence:
    batch_size, num_timesteps = utils.smart_dimension_size(x.values, [0, 1])
    position_embeddings = tf.cast(
        self._embedding.embeddings[start_time : start_time + num_timesteps, :],
        x.values.dtype,
    )
    num_embeddings = utils.smart_dimension_size(position_embeddings, 0)
    position_embeddings = tf.pad(
        position_embeddings, [[0, num_timesteps - num_embeddings], [0, 0]]
    )
    position_embeddings = tf.ensure_shape(
        position_embeddings,
        [
            num_timesteps if isinstance(num_timesteps, int) else None,
            self._channels,
        ],
    )
    position_embeddings = tf.tile(
        position_embeddings[tf.newaxis, :, :], [batch_size, 1, 1]
    )
    tf.debugging.assert_less_equal(
        x.lengths(),
        num_embeddings,
        message='Sequence is longer than valid position embeddings.',
    )
    position_embeddings = types.Sequence(position_embeddings, x.mask)
    return utils.sequence_broadcast_concat(
        x, position_embeddings
    ).mask_invalid()

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    self._check_inputs(x)
    current_time = state
    num_timesteps = utils.smart_dimension_size(x.values, 1)
    x = self._concat_position_embedding(x, start_time=current_time)
    return x, current_time + num_timesteps

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self._check_inputs(x)
    return self._concat_position_embedding(x, start_time=0)


class ApplyRotaryPositionalEncoding(types.PreservesShape, types.SequenceLayer):
  """Applies Rotary Positional Encodings (RoPE) to the sequence.

  See the blogpost https://blog.eleuther.ai/rotary-embeddings/ and the paper
  https://arxiv.org/abs/2104.09864.
  """

  def __init__(
      self,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      axis: int = -1,
      name: str | None = None,
  ):
    """Initializes the ApplyRotaryPositionalEncoding layer.

    Args:
      min_timescale: The minimum timescale for positional encoding.
      max_timescale: The maximum timescale for positional encoding.
      axis: The axis (dimension) to apply RoPE to (default -1).
      name: The name of the layer (optional).
    """
    super().__init__(name=name)
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.axis = axis

  def _check_inputs(self, input_spec: tf.TensorSpec):
    if not input_spec.dtype.is_floating:
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )
    input_shape = (None, None) + input_spec.shape
    axis = self.axis + len(input_shape) if self.axis < 0 else self.axis
    if axis <= 1:
      raise ValueError(
          f'{type(self).__name__} axis ({self.axis}) must refer to a'
          f' channels dimension ({input_spec=}).'
      )
    axis_size = input_shape[axis]
    if axis_size % 2 != 0:
      raise ValueError(
          f'{type(self).__name__} requires input_shape[{axis}]={axis_size} to'
          ' be even.'
      )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    del constants
    batch_size = utils.smart_dimension_size(x.values, 0)
    self._check_inputs(x.channel_spec)

    # State holds the current timestep (batched).
    return tf.zeros((batch_size, 1), dtype=tf.int32)

  def _apply_rope(self, x: tf.Tensor, positions: tf.Tensor) -> tf.Tensor:
    axis = self.axis + x.ndim if self.axis < 0 else self.axis
    assert axis > 1
    channel_ndim = x.ndim - 2
    broadcast_shape = [1] * x.ndim
    axis_dim = x.shape[axis]
    if axis_dim is None:
      raise ValueError(f'Axis {axis} is unknown: {x.shape=}')

    broadcast_shape[axis] = axis_dim // 2
    assert axis_dim % 2 == 0

    compute_dtype = utils.compute_dtype()
    freq_exponents = (2.0 / axis_dim) * tf.cast(
        tf.range(axis_dim // 2, dtype=tf.int32), compute_dtype
    )
    timescale = tf.reshape(
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** freq_exponents,
        broadcast_shape,
    )

    positions = tf.cast(
        tf.reshape(
            positions,
            utils.smart_dimension_size(positions, [0, 1]) + (1,) * channel_ndim,
        ),
        compute_dtype,
    )
    radians = positions / timescale
    sin, cos = tf.math.sin(radians), tf.math.cos(radians)

    x1, x2 = tf.split(x, 2, axis=axis)
    return tf.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=axis)

  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    del constants
    self._check_inputs(x.channel_spec)
    x_time = utils.smart_dimension_size(x.values, 1)

    # Get positions for the batch.
    positions = state + tf.range(x_time, dtype=tf.int32)
    positions.shape.assert_is_compatible_with(x.values.shape[:2])

    y = x.apply_values(self._apply_rope, positions).mask_invalid()
    state = state + x_time
    return y, state

  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State | None = None,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del constants
    self._check_inputs(x.channel_spec)
    x_time = utils.smart_dimension_size(x.values, 1)
    positions = tf.range(x_time, dtype=tf.int32)[tf.newaxis, :]
    x = x.apply_values(self._apply_rope, positions)
    return x.mask_invalid()
