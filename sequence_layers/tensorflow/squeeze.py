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
"""Squeze layers."""

import fractions
from typing import Optional, Tuple

from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class Squeeze(types.SequenceLayer):
  """Combines channels across `factor` timesteps."""

  def __init__(
      self,
      factor: int,
      padding: str = 'causal',
      debug=False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._factor = factor
    self._buffer_size = self._factor - 1
    padding = types.validate_padding(padding)
    if padding == 'same':
      raise ValueError('Squeeze does not yet support "same" padding mode.')
    self._causal = padding == 'causal'
    self._debug = debug

  @property
  def block_size(self) -> int:
    return self._factor

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._factor)

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    if self._factor == 1:
      return ()
    batch_size = utils.smart_dimension_size(x.values, 0)
    inner_shape = tf.shape(x.values)[2:]
    return types.Sequence(
        tf.zeros(
            tf.concat([[batch_size, self._buffer_size], inner_shape], 0),
            dtype=x.values.dtype,
        ),
        # The initial mask is all ones to prevent the first frame from being
        # masked due to being partially valid.
        tf.ones((batch_size, self._buffer_size), dtype=x.values.dtype),
    )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return tf.TensorShape([self._factor]).concatenate(input_shape)

  def _squeeze(self, x: types.Sequence) -> types.Sequence:
    with tf.name_scope('squeeze'):
      batch_size, time = utils.smart_dimension_size(x.values, [0, 1])
      inner_shape = tf.shape(x.values)[2:]
      time_squeezed = time // self._factor
      time_truncated = time_squeezed * self._factor

      result_shape = tf.concat(
          [[batch_size, time_squeezed, self._factor], inner_shape], 0
      )
      values = tf.reshape(x.values[:, :time_truncated], result_shape)
      mask = utils.squeeze_mask(x.mask[:, :time_truncated], self._factor)
      return types.Sequence(values, mask)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    if self._factor == 1:
      x = x.apply_values(lambda v: tf.expand_dims(v, 2))
      return x, state
    state = state.concatenate(x)
    x = self._squeeze(state)
    state = state[:, -self._buffer_size :]
    x = x.mask_invalid()
    if self._debug:
      x = x.print('Squeeze.step', summarize=1000)
    return x, state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    if self._factor == 1:
      return x.apply_values(lambda v: tf.expand_dims(v, 2))
    if self._causal:
      # Causal padding.
      x = x.pad_time(self._factor - 1, 0, valid=True)
    x = self._squeeze(x)
    x = x.mask_invalid()
    if self._debug:
      x = x.print('Squeeze.layer', summarize=1000)
    return x

  @property
  def supports_step(self) -> bool:
    return self._causal


class Unsqueeze(types.Stateless):
  """Splits input channels into `factor` timesteps."""

  def __init__(self, factor: int, name: Optional[str] = None):
    super().__init__(name=name)
    self._factor = factor
    self._buffer_size = self._factor - 1

  @property
  def block_size(self) -> int:
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self._factor)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    unsqueeze_dim = input_shape.dims[0].value
    if unsqueeze_dim is not None and unsqueeze_dim != self._factor:
      raise ValueError(
          'Input shape leading dimension must equal squeeze factor: %s, %d'
          % (input_shape, self._factor)
      )
    return input_shape[1:]

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    if self._factor == 1:
      return x.apply_values(lambda v: tf.squeeze(v, 2))
    batch_size, time = utils.smart_dimension_size(x.values, [0, 1])
    inner_shape = tf.shape(x.values)[3:]
    time_unsqueezed = time * self._factor

    result_shape = tf.concat([[batch_size, time_unsqueezed], inner_shape], 0)
    values = tf.reshape(x.values, result_shape)
    mask = utils.unsqueeze_mask(x.mask, self._factor)
    return types.Sequence(values, mask)
