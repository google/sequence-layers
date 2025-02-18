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
"""Pooling layers."""

import fractions
from typing import List, Optional, Tuple, Union

from sequence_layers.tensorflow import convolution
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class BasePooling1D(types.SequenceLayer):
  """A 1D pooling layer."""

  def __init__(
      self,
      pool_layer: tf.keras.layers.Layer,
      pool_size: int,
      strides: int,
      padding: str,
      name=None,
  ):
    super().__init__(name=name)
    if pool_size <= 0:
      raise ValueError('pool_size must be positive, got: %d' % pool_size)
    if strides <= 0:
      raise ValueError('strides must be positive, got: %d' % strides)
    self._pool_size = pool_size
    self._strides = strides
    self._padding = types.validate_padding(padding)
    self._layer = pool_layer
    self._buffer_width = self._pool_size - 1

  @property
  def supports_step(self) -> bool:
    return self._padding == types.PaddingMode.CAUSAL.value

  @property
  def block_size(self) -> int:
    return self._strides

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._strides)

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    # Special case pool_size 1 since it is stateless.
    if self._pool_size == 1:
      return ()
    # When executing a conv1d step-by-step, we need a buffer for tracking
    # the current causal convolution window.
    batch_size = utils.smart_dimension_size(x.values, 0)
    return types.Sequence(
        tf.zeros(
            (batch_size, self._buffer_width, x.values.shape.dims[2].value),
            dtype=x.values.dtype,
        ),
        tf.ones((batch_size, self._buffer_width), dtype=x.values.dtype),
    )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          '1D pooling requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return input_shape

  def _apply_pool(
      self,
      x: tf.Tensor,
      explicit_padding: tuple[int, int],
  ) -> tf.Tensor:
    if explicit_padding[0] != 0 or explicit_padding[1] != 0:
      x = tf.pad(x, ((0, 0), explicit_padding, (0, 0)))
    return self._layer(x)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    # Special case pool_size 1 since it is stateless.
    if self._pool_size == 1:
      values = self._apply_pool(x.values, (0, 0))
      mask = convolution.compute_conv_mask(
          x.mask,
          self._pool_size,
          self._strides,
          dilation_rate=1,
          padding=self._padding,
      )
      return types.Sequence(values, mask).mask_invalid(), state

    # Concatenate the new frames with the previous buffer_width frames.
    state = state.concatenate(x)

    # Compute the output for the current timestep.
    values = self._apply_pool(state.values, (0, 0))
    mask = convolution.compute_conv_mask(
        state.mask,
        self._pool_size,
        self._strides,
        dilation_rate=1,
        padding=self._padding,
    )

    # Keep the last (pool_size - stride) frames for the next step.
    state = state[:, -self._buffer_width :]

    return types.Sequence(values, mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    explicit_padding = utils.convolution_explicit_padding(
        self._padding,
        self._pool_size,
        dilation_rate=1,
    )

    values = self._apply_pool(x.values, explicit_padding)

    # In causal mode, x.mask is padded and compute_conv_mask uses a VALID
    # convolution. In SAME mode, x.mask is not padded, and compute_conv_mask
    # uses a strided slice mask[:, ::time_stride].
    if self._padding == types.PaddingMode.CAUSAL.value:
      # We can't use convolutional padding here since its value is 0.0, not 1.0.
      mask = tf.pad(
          x.mask,
          [[0, 0], [self._buffer_width, 0]],
          mode='constant',
          constant_values=1.0,
      )
    else:
      mask = x.mask

    mask = convolution.compute_conv_mask(
        mask,
        self._pool_size,
        self._strides,
        dilation_rate=1,
        padding=self._padding,
    )
    return types.Sequence(values, mask).mask_invalid()


class MaxPooling1D(BasePooling1D):
  """A 1D max pooling layer."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'max_pooling_1d'

  def __init__(
      self,
      pool_size: int,
      strides: int,
      padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(
        # We perform padding outside of Keras.
        tf.keras.layers.MaxPooling1D(pool_size, strides, padding='valid'),
        pool_size,
        strides,
        padding,
        name=name,
    )


class AveragePooling1D(BasePooling1D):
  """A 1D average pooling layer."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'average_pooling_1d'

  def __init__(
      self,
      pool_size: int,
      strides: int,
      padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(
        # We perform padding outside of Keras.
        tf.keras.layers.AveragePooling1D(pool_size, strides, padding='valid'),
        pool_size,
        strides,
        padding,
        name=name,
    )


class BasePooling2D(types.SequenceLayer):
  """A 2D pooling layer."""

  def __init__(
      self,
      pool_layer: tf.keras.layers.Layer,
      pool_size: Union[int, List[int], Tuple[int, int]],
      strides: Union[int, List[int], Tuple[int, int]],
      time_padding: str,
      spatial_padding: str,
      name=None,
  ):
    super().__init__(name=name)

    def normalize_2tuple(
        x: Union[int, List[int], Tuple[int, int]]
    ) -> Tuple[int, int]:
      if isinstance(x, int):
        return (x, x)
      result = tuple(x)
      if len(result) != 2:
        raise ValueError('Expected a 2-element iterable, got: %r' % x)
      return result

    self._pool_size = normalize_2tuple(pool_size)
    self._strides = normalize_2tuple(strides)
    if any(pool_size <= 0 for pool_size in self._pool_size):
      raise ValueError('pool_size must be positive, got: %s' % self._pool_size)
    if any(stride <= 0 for stride in self._strides):
      raise ValueError('strides must be positive, got: %s' % self._strides)
    self._time_padding = types.validate_padding(time_padding)
    self._spatial_padding = types.validate_padding(spatial_padding)
    self._layer = pool_layer
    self._buffer_width = self._pool_size[0] - 1

  @property
  def supports_step(self) -> bool:
    return self._time_padding == types.PaddingMode.CAUSAL.value

  @property
  def block_size(self) -> int:
    return self._strides[0]

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._strides[0])

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    # Special case pool_size 1 since it is stateless.
    if self._pool_size[0] == 1:
      return ()
    # When executing a conv1d step-by-step, we need a buffer for tracking
    # the current causal pooling window.
    batch_size = utils.smart_dimension_size(x.values, 0)
    return types.Sequence(
        tf.zeros(
            (
                batch_size,
                self._buffer_width,
                x.values.shape.dims[2].value,
                x.values.shape.dims[3].value,
            ),
            dtype=x.values.dtype,
        ),
        tf.ones((batch_size, self._buffer_width), dtype=x.mask.dtype),
    )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 2:
      raise ValueError(
          '2D pooling requires rank 4 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape.dims[0].value,
        self._spatial_padding,
        kernel_size=self._pool_size[1],
        stride=self._strides[1],
        dilation_rate=1,
    )
    return tf.TensorShape([spatial_output_size, input_shape.dims[1].value])

  def _apply_pool(
      self,
      x: tf.Tensor,
      explicit_padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> tf.Tensor:
    x = tf.pad(x, [(0, 0), explicit_padding[0], explicit_padding[1], (0, 0)])
    return self._layer(x)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    # In step mode, causal padding is handled by the state that is concatenated
    # below.
    time_padding = (0, 0)
    spatial_padding = utils.convolution_explicit_padding(
        self._spatial_padding, self._pool_size[1], dilation_rate=1
    )

    # Special case pool_size 1 since it is stateless.
    if self._pool_size[0] == 1:
      values = self._apply_pool(x.values, (time_padding, spatial_padding))
      mask = convolution.compute_conv_mask(
          x.mask,
          self._pool_size[0],
          self._strides[0],
          dilation_rate=1,
          padding=self._time_padding,
      )
      return types.Sequence(values, mask).mask_invalid(), state

    # Concatenate the new frames with the previous buffer_width frames.
    state = state.concatenate(x)

    # Compute the output for the current timestep.
    values = self._apply_pool(state.values, (time_padding, spatial_padding))
    mask = convolution.compute_conv_mask(
        state.mask,
        self._pool_size[0],
        self._strides[0],
        dilation_rate=1,
        padding=self._time_padding,
    )

    # Keep the last (pool_size - stride) frames for the next step.
    state = state[:, -self._buffer_width :]

    return types.Sequence(values, mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    time_padding = utils.convolution_explicit_padding(
        self._time_padding,
        self._pool_size[0],
        dilation_rate=1,
    )
    spatial_padding = utils.convolution_explicit_padding(
        self._spatial_padding, self._pool_size[1], dilation_rate=1
    )
    values = self._apply_pool(x.values, (time_padding, spatial_padding))

    # In causal mode, x.mask is padded and compute_conv_mask uses a VALID
    # convolution. In SAME mode, x.mask is not padded, and compute_conv_mask
    # uses a strided slice mask[:, ::time_stride].
    if self._time_padding == types.PaddingMode.CAUSAL.value:
      # We can't use convolutional padding here since its value is 0.0, not 1.0.
      mask = tf.pad(
          x.mask,
          [[0, 0], [self._buffer_width, 0]],
          mode='constant',
          constant_values=1.0,
      )
    else:
      mask = x.mask

    mask = convolution.compute_conv_mask(
        mask,
        self._pool_size[0],
        self._strides[0],
        dilation_rate=1,
        padding=self._time_padding,
    )

    return types.Sequence(values, mask).mask_invalid()


class MaxPooling2D(BasePooling2D):
  """A 2D max pooling layer."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'max_pooling_2d'

  def __init__(
      self,
      pool_size: Union[int, List[int], Tuple[int, int]],
      strides: Union[int, List[int], Tuple[int, int]],
      *,
      time_padding: str = 'causal',
      spatial_padding: str = 'valid',
      name: Optional[str] = None,
  ):
    super().__init__(
        # We perform padding outside of Keras.
        tf.keras.layers.MaxPooling2D(pool_size, strides, padding='valid'),
        pool_size,
        strides,
        time_padding,
        spatial_padding,
        name=name,
    )


class AveragePooling2D(BasePooling2D):
  """A 2D average pooling layer."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'average_pooling_2d'

  def __init__(
      self,
      pool_size: Union[int, List[int], Tuple[int, int]],
      strides: Union[int, List[int], Tuple[int, int]],
      *,
      time_padding: str = 'causal',
      spatial_padding: str = 'valid',
      name: Optional[str] = None,
  ):
    super().__init__(
        # We perform padding outside of Keras.
        tf.keras.layers.AveragePooling2D(pool_size, strides, padding='valid'),
        pool_size,
        strides,
        time_padding,
        spatial_padding,
        name=name,
    )
