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
"""Convolution layers."""

import fractions
from typing import List, Optional, Tuple, Union

import numpy as np
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow as tf
import tensorflow.compat.v2 as tf
import tf_keras


def compute_conv_mask(
    mask: tf.Tensor,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: str,
) -> tf.Tensor:
  """Given an input mask, computes the output mask for a convolution.

  The formula for the output time dimension for a [b, t] mask is:

    effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
    output_frames = ceil((t - effective_kernel_size + 1) / stride)

  If padding is 'causal', we assume that the mask has causal padding of
  `effective_kernel_size - 1` applied which ensures that the above
  formula is at least zero, because `t >= effective_kernel_size - 1`.

  Args:
    mask: The input [b, t] mask.
    kernel_size: The kernel size in the time dimension.
    stride: The stride in the time dimension.
    dilation_rate: The dilation rate in the time dimension.
    padding: The padding mode.

  Returns:
    The output mask. [b, ceil((t - effective_kernel_size + 1) / stride)]
  """
  assert kernel_size >= 1
  assert stride >= 1
  assert dilation_rate >= 1

  # Special-case kernel_size 1. Dilation has no effect at kernel_size 1.
  if kernel_size == 1:
    if stride == 1:
      return mask
    else:
      return mask[:, ::stride]

  padding = types.validate_padding(padding)
  if padding == types.PaddingMode.SAME.value:
    # If stride is 1, input mask equals the output mask.
    if stride == 1:
      return mask
    else:
      # We assume that we have corrected for TensorFlow's SAME padding quirks
      # which align the last input timestep with the last output timestep
      # instead of the first input timestep with the first output timestep.
      # Strided slice behaves like a kernel size 1, same convolution with this
      # adjustment applied -- the output length is (input + stride - 1) //
      # stride.
      return mask[:, ::stride]

  if padding == types.PaddingMode.REVERSE_CAUSAL.value:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        kernel_size, dilation_rate
    )
    padding = [(0, 0), (0, effective_kernel_size - 1), (0, 0), (0, 0)]
    # Use conv2d since conv1d doesn't support explicit padding.
    mask = tf.nn.conv2d(
        mask[:, :, tf.newaxis, tf.newaxis],
        tf.ones([kernel_size, 1, 1, 1]),
        strides=[stride, 1],
        dilations=[dilation_rate, 1],
        padding=padding,
    )
    mask = tf.squeeze(mask, [2, 3])
    # Each timestep contains the sum of the mask values for inputs in its
    # receptive field, so only timesteps equal to kernel_size are fully valid.
    return mask // kernel_size

  assert padding in (
      types.PaddingMode.VALID.value,
      types.PaddingMode.CAUSAL.value,
  ), padding
  if dilation_rate > 1:
    # This is the correct but expensive (conv1d) calculation of the mask.
    # TODO(rryan): Benchmark whether this is actually any slower than the
    # below "shortcut" methods.
    mask = tf.nn.conv1d(
        mask[:, :, tf.newaxis],
        tf.ones([kernel_size, 1, 1]),
        stride=stride,
        dilations=dilation_rate,
        padding='VALID',
    )
    mask = tf.squeeze(mask, 2)
    # Each timestep contains the sum of the mask values for inputs in its
    # receptive field, so only timesteps equal to kernel_size are fully valid.
    return mask // kernel_size
  else:
    assert dilation_rate == 1
    assert stride >= 1
    # To avoid running a second convolution for the mask, consider 3 cases
    # where stride >= 1:
    # - kernel_size > stride
    # - kernel_size == stride
    # - kernel_size < stride
    #
    # Example length 5 sequence:
    # x is causal padding, ABCDE is valid and 0 is padding.
    # values: x x A B C D E 0 0 0 0 0
    # mask:   1 1 1 1 1 1 1 0 0 0 0 0
    #
    # The formula for the output size of a VALID convolution is:
    # out_t = ceil((in_t - dilation_rate * (kernel_size - 1)) / stride)
    #
    # Causal padding means that we always output at least 0 samples, and will
    # never get an error that the convolution would produce a negative size
    # input.
    #
    # kernel_size 3 stride 2:
    # x x A B C D E 0 0 0 0 0
    # x x A
    #     A B C
    #         C D E
    #             E 0 0
    #                 0 0 0
    #                     0 0 <- not produced with VALID convolution.
    # expected output_size = ceil(10 / 2) = 5
    # expected mask: 1 1 1 0 0

    if kernel_size > stride:
      # Case 1: The convolution kernel overlaps from step to step. Use
      # tf.signal.frame to compute [batch, num_frames, kernel_size]
      # overlapping frames of the mask tensor, then reduce across channels so
      # that any frame containing partially valid samples is considered
      # invalid.
      mask = tf.signal.frame(
          mask, frame_length=kernel_size, frame_step=stride, pad_end=False
      )
      return tf.reduce_min(mask, axis=2)

    # kernel_size 3 stride 3:
    # x x A B C D E 0 0 0 0 0
    # x x A
    #       B C D
    #             E 0 0
    #                   0 0 0
    # expected output_size = ceil(10 / 3) = 4
    # expected mask: 1 1 0 0

    elif kernel_size == stride:
      # Case 2: The kernel size is equal to the stride. This means we can
      # just reshape the mask into [batch_size, num_frames, kernel_size],
      # then reduce across channels so that any frame containing partially
      # valid samples is considered invalid. We have to use the VALID formula
      # above to truncate the mask so the reshape produces the same number of
      # frames that the convolution will.
      batch_size, time = utils.smart_dimension_size(mask)
      num_frames = time // stride
      time_truncated = num_frames * stride
      mask = tf.reshape(
          mask[:, :time_truncated], [batch_size, num_frames, stride]
      )
      return tf.reduce_min(mask, axis=2)

    # kernel_size 3 stride 4:
    # x x A B C D E 0 0 0 0 0
    # x x A
    #         C D E
    #                 0 0 0
    #                         . . . <- past the end of the signal
    # expected output_size = ceil(10 / 4) = 3
    # expected mask: 1 1 0

    else:
      # In this case, the stride is greater than the kernel size. Like case 2
      # there is no overlap, and we just need to know the number of valid
      # frames the convolution will produce. We thus just reshape the mask to
      # [batch, ..., stride], and then trim that to [batch, ..., kernel_size].
      #
      # The general formula for the length of an overlap-added signal with
      # stride/kernel_size is:
      # (num_frames - 1) * stride + kernel_size == length
      #
      # Solving for num_frames, we can see the target to pad/truncate to:
      # num_frames = 1 + (length - kernel_size) // stride
      assert kernel_size < stride
      batch_size, time = utils.smart_dimension_size(mask)
      num_frames = 1 + (time - kernel_size) // stride
      # The length we need to pad or truncate mask to.
      padded_time = num_frames * stride
      pad_amount = tf.maximum(0, padded_time - time)
      # Pad and truncate so that mask is num_frames * stride long.
      mask = tf.pad(mask, [[0, 0], [0, pad_amount]])[:, :padded_time]
      # Reshape into windows of length stride.
      mask = tf.reshape(mask, [batch_size, num_frames, stride])
      # Trim the windows to cover only the kernel_size region.
      mask = mask[:, :, :kernel_size]
      # Mark any frame covering any invalid samples as invalid.
      return tf.reduce_min(mask, axis=2)


def _compute_conv_transpose_output_length(
    time: tf.Tensor,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: str,
):
  """Returns the output time for a transpose convolution, matching Keras."""
  # Based on tf-keras (tf_keras/utils/conv_utils.py).
  padding = types.validate_padding(padding)
  # Get the effective kernel size with dilation.
  kernel_size = utils.convolution_effective_kernel_size(
      kernel_size, dilation_rate
  )

  if padding == 'same':
    output_time = time * stride
  else:
    output_time = time * stride + max(kernel_size - stride, 0)

  return output_time


def _compute_conv_transpose_mask(
    mask: tf.Tensor, kernel_size: int, stride: int, padding: str
) -> tf.Tensor:
  """Given an input mask, computes the output mask for a transpose convolution.

  Args:
    mask: The input [b, t] mask.
    kernel_size: The kernel size in the time dimension.
    stride: The stride in the time dimension.
    padding: The padding mode. One of 'valid', 'same', or 'causal'.

  Returns:
    The output mask. [b, t * stride + max(kernel_size - stride, 0)].
  """
  padding = types.validate_padding(padding)
  with tf.name_scope('conv_transpose_mask'):
    batch_size, time = utils.smart_dimension_size(mask)

    if kernel_size <= stride or padding == 'same':
      return tf.repeat(mask, stride, axis=1)

    # If kernel_size > stride, use an actual transpose convolution to compute
    # the mask.
    invalid_mask = 1.0 - mask
    output_time = _compute_conv_transpose_output_length(
        time, kernel_size, stride, dilation_rate=1, padding=padding
    )
    mask = tf.nn.conv2d_transpose(
        invalid_mask[:, :, tf.newaxis, tf.newaxis],
        filters=tf.ones((kernel_size, 1, 1, 1), dtype=mask.dtype),
        output_shape=[batch_size, output_time, 1, 1],
        strides=stride,
        padding='SAME' if padding == 'same' else 'VALID',
        data_format='NHWC',
        dilations=None,
    )
    mask = tf.squeeze(mask, [2, 3])
    # Any non-zero values in mask have been corrupted by invalid timesteps.
    mask = tf.cast(tf.equal(mask, 0.0), mask.dtype)
    return mask


class BaseConv1D(types.SequenceLayer):
  """Shared base logic for Conv1D layers."""

  def __init__(
      self,
      kernel_size: int,
      strides: int,
      dilation_rate: int,
      padding: str,
      name=None,
  ):
    super().__init__(name=name)
    self._kernel_size = kernel_size
    self._strides = strides
    self._dilation_rate = dilation_rate
    self._padding = types.validate_padding(padding)

    if self._padding == 'causal':
      # (effective_kernel_size - 1) padding makes convolution fully causal.
      self._buffer_width = (
          utils.convolution_effective_kernel_size(kernel_size, dilation_rate)
          - 1
      )
    else:
      self._buffer_width = 0

  @property
  def supports_step(self) -> bool:
    return self._padding == 'causal'

  @property
  def block_size(self) -> int:
    return self._strides

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._strides)

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    # Special case kernel_size 1 since it is stateless.
    if not self._buffer_width:
      return ()
    # When executing a conv1d step-by-step, we need a buffer for tracking
    # the current causal convolution window.
    batch_size = utils.smart_dimension_size(x.values, 0)
    # This matches the causal padding added by `layer`.
    return types.Sequence(
        tf.zeros(
            (batch_size, self._buffer_width, x.values.shape.dims[2].value),
            dtype=x.values.dtype,
        ),
        tf.ones((batch_size, self._buffer_width), dtype=x.mask.dtype),
    )

  def _apply_conv(
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
    if not self.supports_step:
      raise ValueError(f'{self} does not support stepping.')

    # Special case kernel_size 1 since it is stateless.
    if not self._buffer_width:
      values = self._apply_conv(x.values, (0, 0))
      mask = compute_conv_mask(
          x.mask,
          self._kernel_size,
          self._strides,
          self._dilation_rate,
          self._padding,
      )
      return types.Sequence(values, mask).mask_invalid(), state

    # Concatenate the new frames with the previous buffer_width frames.
    state = state.concatenate(x)

    # Compute the output for the current timestep.
    values = self._apply_conv(state.values, (0, 0))
    mask = compute_conv_mask(
        state.mask,
        self._kernel_size,
        self._strides,
        self._dilation_rate,
        self._padding,
    )

    # Keep the trailing buffer_width samples for the next step.
    state = state[:, -self._buffer_width :]

    return types.Sequence(values, mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ):
    explicit_padding = utils.convolution_explicit_padding(
        self._padding,
        self._kernel_size,
        self._dilation_rate,
    )
    values = self._apply_conv(x.values, explicit_padding)

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

    mask = compute_conv_mask(
        mask,
        self._kernel_size,
        self._strides,
        self._dilation_rate,
        self._padding,
    )
    return types.Sequence(values, mask).mask_invalid()


class Conv1D(BaseConv1D):
  """A 1D convolution layer. Supports both strided and dilated convolution."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'conv1d'

  def __init__(
      self,
      filters: int,
      kernel_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      groups: int = 1,
      activation=None,
      use_bias: bool = True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable: bool = True,
      padding: str = 'causal',
      weight_norm: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(kernel_size, strides, dilation_rate, padding, name=name)
    self._filters = filters
    with self.name_scope as name_scope:
      self._layer = tf.keras.layers.Conv1D(
          filters,
          kernel_size,
          strides,
          # We apply padding outside of Keras.
          padding='valid',
          data_format='channels_last',
          dilation_rate=dilation_rate,
          groups=groups,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          kernel_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )
    if weight_norm:
      self._layer = utils.WeightNorm(self._layer, name='weight_norm')

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'Conv1D requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return tf.TensorShape([self._filters])


def _kaiser_sinc_filter1d(
    cutoff: float, kaiser_beta: float, window_length: int
) -> np.ndarray:
  """Creates a kaiser-windowed sinc filter."""
  if cutoff <= 0.0 or cutoff > 0.5:
    raise ValueError(f'{cutoff=} should be in (0, 0.5].')

  # The general form of the length M + 1 filter is:
  # h[i] = K * window * sin(2 * pi * cutoff * i) / (i * pi)
  #
  # With a shift of i = t - M/2 to center the length M + 1 filter.
  # Where K is a normalizing constant ensuring the filter sums to 1.
  #
  # https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
  #
  # Since np.sinc is sin(pi * x) / (pi * x):
  # h[i] = K * 2 * cutoff * window * np.sinc(2 * cutoff * i)
  window = np.kaiser(window_length, kaiser_beta)
  time = np.arange(window_length) - window_length / 2 + 0.5
  f = 2 * cutoff * window * np.sinc(2 * cutoff * time)

  # Normalize the filter to sum to 1. (K)
  f /= np.sum(f)

  return f.astype(np.float32)


class SincFilter1D(BaseConv1D):
  """Applies a windowed-sinc filter as a DepthwiseConv1D."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'sinc_filter1d'

  def __init__(
      self,
      kernel_size: int,
      strides: int = 1,
      cutoff: float = 0.5,
      kaiser_beta: float = 6.0,
      padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(
        kernel_size, strides, dilation_rate=1, padding=padding, name=name
    )
    self._filter = _kaiser_sinc_filter1d(cutoff, kaiser_beta, kernel_size)

  def _layer(self, x: tf.Tensor) -> tf.Tensor:
    if x.shape.rank != 3:
      raise ValueError(f'Expected rank 3 inputs. Got: {x.shape}')
    channels = x.shape.dims[2].value

    # Filter shape is [height, width=1, in_channels, channel_multiplier=1].
    sinc_filter = tf.tile(
        self._filter[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, channels, 1]
    )

    # Insert a placeholder width dimension.
    x = x[:, :, tf.newaxis, :]

    y = tf.nn.depthwise_conv2d(
        x,
        sinc_filter,
        # tf.nn.depthwise_conv2d requires equal strides in both dimensions.
        strides=[1, self._strides, self._strides, 1],
        # We apply padding outside of Keras.
        padding='VALID',
    )
    y.shape.assert_is_compatible_with([None, None, 1, channels])
    # Remove the placeholder width dimension.
    return tf.squeeze(y, 2)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'SincFilter1D requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return input_shape


class DepthwiseConv1D(BaseConv1D):
  """A 1D depthwise convolution. Supports strided and dilated convolution."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'depthwise_conv1d'

  def __init__(
      self,
      kernel_size: int,
      strides: int = 1,
      depth_multiplier: int = 1,
      dilation_rate: int = 1,
      activation=None,
      use_bias: bool = True,
      depthwise_initializer='glorot_uniform',
      bias_initializer='zeros',
      depthwise_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      depthwise_constraint=None,
      bias_constraint=None,
      trainable: bool = True,
      padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(kernel_size, strides, dilation_rate, padding, name=name)
    self._depth_multiplier = depth_multiplier
    with self.name_scope as name_scope:
      self._depthwise_2d_layer = tf.keras.layers.DepthwiseConv2D(
          (1, kernel_size),
          # tf.nn.depthwise_conv2d requires equal strides in both dimensions.
          strides=(strides, strides),
          # We apply padding outside of Keras.
          padding='valid',
          depth_multiplier=depth_multiplier,
          data_format='channels_last',
          dilation_rate=(1, dilation_rate),
          activation=activation,
          use_bias=use_bias,
          depthwise_initializer=depthwise_initializer,
          bias_initializer=bias_initializer,
          depthwise_regularizer=depthwise_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          depthwise_constraint=depthwise_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )

  def _layer(self, values):
    values = tf.expand_dims(values, 1)
    values = self._depthwise_2d_layer(values)
    return tf.squeeze(values, 1)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'DepthwiseConv1D requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return tf.TensorShape([input_shape.dims[0].value * self._depth_multiplier])


class _NormalizedDepthwiseConv1D(tf.keras.layers.Conv2D):
  """Normalized depthwise 1D convolution implemented as a Conv2D.

  The height dimension is assumed to be one and the width is the dimension to
  normalize.
  """

  def __init__(
      self,
      kernel_size,
      strides=(1, 1),
      padding='valid',
      depth_multiplier=1,
      data_format=None,
      dilation_rate=(1, 1),
      activation=None,
      use_bias=True,
      depthwise_initializer='glorot_uniform',
      bias_initializer='zeros',
      depthwise_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      depthwise_constraint=None,
      bias_constraint=None,
      num_heads: Optional[int] = None,
      depthwise_dropconnect_rate: float = 0.0,
      **kwargs,
  ):
    super().__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        **kwargs,
    )
    if self.kernel_size[0] != 1:
      raise ValueError(
          f'Expected kernel height to be one. Got: {self.kernel_size[0]}'
      )
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = tf.keras.initializers.get(
        depthwise_initializer
    )
    self.depthwise_regularizer = tf.keras.regularizers.get(
        depthwise_regularizer
    )
    self.depthwise_constraint = tf.keras.constraints.get(depthwise_constraint)
    self.depthwise_dropconnect = tf.keras.layers.Dropout(
        depthwise_dropconnect_rate
    )
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.num_heads = num_heads

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError(
          (
              'Inputs to `NormalizedDepthwiseConv2D` should have rank 4. '
              'Received input shape:'
          ),
          str(input_shape),
      )
    input_shape = tf.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError(
          'The channel dimension of the inputs to '
          '`DepthwiseConv2D` '
          'should be defined. Found `None`.'
      )
    input_dim = int(input_shape[channel_axis])

    if self.num_heads is not None and input_dim % self.num_heads != 0:
      raise ValueError(
          'Input channels (%d) must be divisible by num_heads (%d).'
          % (input_dim, self.num_heads)
      )
    depthwise_kernel_shape = (
        self.kernel_size[0],
        self.kernel_size[1],
        self.num_heads if self.num_heads else input_dim,
        self.depth_multiplier,
    )

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
    )

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(input_dim * self.depth_multiplier,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
      )
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=4, axes={channel_axis: input_dim}
    )
    self.built = True

  def call(self, inputs, training=None):
    depthwise_kernel = tf.nn.softmax(self.depthwise_kernel, axis=1)

    depthwise_kernel = self.depthwise_dropconnect(
        depthwise_kernel, training=training
    )

    # TODO(rryan): Benchmark reshaping inputs to
    # [b * channels / num_heads, t, num_heads, channels]
    if self.num_heads:
      input_dim = int(inputs.shape[self._get_channel_axis()])
      depthwise_kernel = tf.tile(
          depthwise_kernel, [1, 1, input_dim // self.num_heads, 1]
      )

    # We could apply DropConnect after tiling, but this is not what the fairseq
    # implementation does.
    outputs = tf.keras.backend.depthwise_conv2d(
        inputs,
        depthwise_kernel,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format,
    )

    if self.use_bias:
      outputs = tf.keras.backend.bias_add(
          outputs, self.bias, data_format=self.data_format
      )

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  @tf_keras.src.utils.tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
      out_filters = input_shape[1] * self.depth_multiplier
    elif self.data_format == 'channels_last':
      rows = input_shape[1]
      cols = input_shape[2]
      out_filters = input_shape[3] * self.depth_multiplier

    rows = tf_keras.src.utils.conv_utils.conv_output_length(
        rows,
        self.kernel_size[0],
        self.padding,
        self.strides[0],
        self.dilation_rate[0],
    )
    cols = tf_keras.src.utils.conv_utils.conv_output_length(
        cols,
        self.kernel_size[1],
        self.padding,
        self.strides[1],
        self.dilation_rate[1],
    )
    if self.data_format == 'channels_first':
      return (input_shape[0], out_filters, rows, cols)
    elif self.data_format == 'channels_last':
      return (input_shape[0], rows, cols, out_filters)

  def get_config(self):
    config = super().get_config()
    config.pop('filters')
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = tf.keras.initializers.serialize(
        self.depthwise_initializer
    )
    config['depthwise_regularizer'] = tf.keras.regularizers.serialize(
        self.depthwise_regularizer
    )
    config['depthwise_constraint'] = tf.keras.constraints.serialize(
        self.depthwise_constraint
    )
    config['num_heads'] = self.num_heads
    return config


class NormalizedDepthwiseConv1D(BaseConv1D):
  """A 1D normalized depthwise convolution.

  Based on "Pay Less Attention with Lightweight and Dynamic Convolutions"
  https://arxiv.org/abs/1901.10430

  Like a 1D depthwise convolution but:
  - The [kernel_size, input_channels, depthwise_multiplier] kernel weights are
    softmax normalized along the kernel_size dimension. This is akin to an
    attention weighted combination of the timesteps in the filter's receptive
    field.
  - Weight tying is supported via the num_heads parameter. If non-None, each
    group of (input_channels // num_heads) input channels share the same
    depthwise convolution weights. The total number of weights in the depthwise
    kernel is therefore `kernel_size * num_heads * depth_multiplier`.
  - Dropconnect regularization, dropping out the depthwise kernel after
    softmax normalization, is supported. This is akin to forgetting certain
    timesteps of the input for each depthwise filter.

  Supports strided and dilated convolution.
  """

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'normalized_depthwise_conv1d'

  def __init__(
      self,
      kernel_size: int,
      strides: int = 1,
      depth_multiplier: int = 1,
      dilation_rate: int = 1,
      activation=None,
      use_bias: bool = True,
      depthwise_initializer='glorot_uniform',
      bias_initializer='zeros',
      depthwise_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      depthwise_constraint=None,
      bias_constraint=None,
      trainable: bool = True,
      padding: str = 'causal',
      num_heads: Optional[int] = None,
      depthwise_dropconnect_rate: float = 0.0,
      name: Optional[str] = None,
  ):
    super().__init__(kernel_size, strides, dilation_rate, padding, name=name)
    self._depth_multiplier = depth_multiplier
    with self.name_scope as name_scope:
      self._depthwise_2d_layer = _NormalizedDepthwiseConv1D(
          (1, kernel_size),
          # tf.nn.depthwise_conv2d requires equal strides in both dimensions.
          strides=(strides, strides),
          # We apply padding outside of Keras.
          padding='valid',
          depth_multiplier=depth_multiplier,
          data_format='channels_last',
          dilation_rate=(1, dilation_rate),
          activation=activation,
          use_bias=use_bias,
          depthwise_initializer=depthwise_initializer,
          bias_initializer=bias_initializer,
          depthwise_regularizer=depthwise_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          depthwise_constraint=depthwise_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          num_heads=num_heads,
          depthwise_dropconnect_rate=depthwise_dropconnect_rate,
          name=name_scope,
      )

  def _layer(self, values):
    values = tf.expand_dims(values, 1)
    values = self._depthwise_2d_layer(values)
    return tf.squeeze(values, 1)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'NormalizedDepthwiseConv1D requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return tf.TensorShape([input_shape.dims[0].value * self._depth_multiplier])


class SeparableConv1D(BaseConv1D):
  """A 1D separable convolution. Supports strided and dilated convolution."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'separable_conv1d'

  def __init__(
      self,
      filters: int,
      kernel_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      depth_multiplier: int = 1,
      activation=None,
      use_bias: bool = True,
      depthwise_initializer='glorot_uniform',
      pointwise_initializer='glorot_uniform',
      bias_initializer='zeros',
      depthwise_regularizer=None,
      pointwise_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      depthwise_constraint=None,
      pointwise_constraint=None,
      bias_constraint=None,
      trainable: bool = True,
      padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(kernel_size, strides, dilation_rate, padding, name=name)
    self._filters = filters
    with self.name_scope as name_scope:
      self._layer = tf.keras.layers.SeparableConv1D(
          filters,
          kernel_size,
          strides=strides,
          # We apply padding outside of Keras.
          padding='valid',
          data_format='channels_last',
          dilation_rate=dilation_rate,
          depth_multiplier=depth_multiplier,
          activation=activation,
          use_bias=use_bias,
          depthwise_initializer=depthwise_initializer,
          pointwise_initializer=pointwise_initializer,
          bias_initializer=bias_initializer,
          depthwise_regularizer=depthwise_regularizer,
          pointwise_regularizer=pointwise_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          depthwise_constraint=depthwise_constraint,
          pointwise_constraint=pointwise_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'SeparableConv1D requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return tf.TensorShape([self._filters])


class Conv1DTranspose(types.SequenceLayer):
  """A 1D transpose convolution layer."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'conv1d_transpose'

  def __init__(
      self,
      filters: int,
      kernel_size: int,
      strides: int = 1,
      activation=None,
      use_bias: bool = True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable: bool = True,
      padding: str = 'causal',
      weight_norm: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = types.validate_padding(padding)
    if self._padding == 'causal':
      self._buffer_width = max(0, self._kernel_size - self._strides)
    else:
      self._buffer_width = 0
    # TODO(rryan): Add support for dilated transpose convolution (Keras
    # supports it).
    # We apply the activation ourselves outside of the layer.
    self._activation = activation
    with self.name_scope as name_scope:
      # To support streaming of transpose convolutions, we apply bias and
      # activation outside of the model, but use the bias in the
      # Conv2DTranspose layer. Since we might apply weight normalization, we
      # keep a reference to the original layer so we can disable bias when we
      # call it.
      self._inner_layer = self._layer = tf.keras.layers.Conv2DTranspose(
          filters=filters,
          kernel_size=(kernel_size, 1),
          strides=(strides, 1),
          padding='same' if self._padding == 'same' else 'valid',
          data_format='channels_last',
          activation=None,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          kernel_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )
      if weight_norm:
        self._layer = utils.WeightNorm(self._layer, name='weight_norm')

  @property
  def supports_step(self) -> bool:
    return self._padding == 'causal'

  def build(self, input_shape):
    if not self._layer.built:
      self._layer.build(
          input_shape[:2].concatenate(1).concatenate(input_shape[2:])
      )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    if self._buffer_width > 0:
      batch_size = utils.smart_dimension_size(x.values, 0)
      return types.Sequence(
          tf.zeros(
              (batch_size, self._buffer_width, self._filters),
              dtype=x.values.dtype,
          ),
          tf.ones((batch_size, self._buffer_width), dtype=x.mask.dtype),
      )
    else:
      return ()

  @property
  def block_size(self) -> int:
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self._strides)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'Conv1DTranspose requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return tf.TensorShape([self._filters])

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    if not self.supports_step:
      raise ValueError(f'{self} does not support stepping.')

    self.build(x.values.shape)
    # Our strategy for running Conv2DTranspose step-wise involves buffering the
    # output of the layer and overlap-adding that output with future timesteps.
    # Because of this, we have to delay bias and activation until we have
    # performed the overlap add.
    use_bias = self._inner_layer.use_bias
    self._inner_layer.use_bias = False
    values = tf.squeeze(self._layer(x.values[:, :, tf.newaxis, :]), 2)
    self._inner_layer.use_bias = use_bias
    mask = _compute_conv_transpose_mask(
        x.mask, self._kernel_size, self._strides, self._padding
    )

    # If kernel_size > stride, mix the state into the output and compute
    # state for the next timestep.
    if self._buffer_width > 0:
      time = utils.smart_dimension_size(x.values, 1)
      output_time = _compute_conv_transpose_output_length(
          time,
          self._kernel_size,
          self._strides,
          dilation_rate=1,
          padding=self._padding,
      )

      # Pad the state to extend it to the length of the layer output.
      # output_time is at least kernel_size and buffer_width is at most
      # kernel_size - 1, so output_time - buffer_width is positive.
      state = state.pad_time(0, output_time - self._buffer_width, valid=True)
      values += state.values
      mask *= state.mask

      # Stride samples are "ready" for output after one timestep, so the number
      # of output samples for the block is stride * time.
      output_samples = self._strides * time

      # We need to store output_time - output_samples samples for the next step,
      # since their value depends on future inputs.
      state_samples = output_time - output_samples
      values, state_values = tf.split(
          values, [output_samples, state_samples], axis=1
      )
      mask, state_mask = tf.split(mask, [output_samples, state_samples], axis=1)
      state = types.Sequence(state_values, state_mask)

    if use_bias:
      values = tf.nn.bias_add(values, self._inner_layer.bias, data_format='NWC')
    if self._activation:
      values = self._activation(values)
    return types.Sequence(values, mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self.build(x.values.shape)
    values = tf.squeeze(self._layer(x.values[:, :, tf.newaxis, :]), 2)
    mask = _compute_conv_transpose_mask(
        x.mask, self._kernel_size, self._strides, self._padding
    )

    if self._activation:
      values = self._activation(values)

    # Trim the last frame_length - stride samples since we don't produce these
    # in step mode (we can only produce stride samples at a time in step mode,
    # so we can't produce the final kernel_size - stride samples).
    x = types.Sequence(values, mask)
    if self._padding == 'causal':
      trim = max(self._kernel_size - self._strides, 0)
      if trim:
        x = x[:, :-trim]
    return x.mask_invalid()


class SincFilter1DTranspose(types.SequenceLayer):
  """Applies a windowed-sinc filter as a DepthwiseConv1DTranpsose."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'sinc_filter1d_transpose'

  def __init__(
      self,
      kernel_size: int,
      strides: int = 1,
      cutoff: float = 0.5,
      kaiser_beta: float = 6.0,
      padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = types.validate_padding(padding)
    if self._padding == 'causal':
      self._buffer_width = max(0, self._kernel_size - self._strides)
    else:
      self._buffer_width = 0
    self._filter = _kaiser_sinc_filter1d(cutoff, kaiser_beta, kernel_size)

  @property
  def supports_step(self) -> bool:
    return self._padding == 'causal'

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    if self._buffer_width > 0:
      batch_size, channels = utils.smart_dimension_size(x.values, [0, 2])
      return types.Sequence(
          tf.zeros(
              (batch_size, self._buffer_width, channels),
              dtype=x.values.dtype,
          ),
          tf.ones((batch_size, self._buffer_width), dtype=x.mask.dtype),
      )
    else:
      return ()

  @property
  def block_size(self) -> int:
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self._strides)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 1:
      raise ValueError(
          'SincFilter1DTranspose requires rank 3 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    return input_shape

  def _layer(self, x: tf.Tensor) -> tf.Tensor:
    if x.shape.rank != 3:
      raise ValueError(f'Expected rank 3 inputs. Got: {x.shape}')
    batch, time, channels = utils.smart_dimension_size(x)
    assert isinstance(channels, int)

    # Filter shape is [height, width=1, in_channels, channel_multiplier=1].
    sinc_filter = tf.tile(
        self._filter[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, channels, 1]
    )

    # Insert a placeholder width dimension.
    x = x[:, :, tf.newaxis, :]

    output_time = _compute_conv_transpose_output_length(
        time,
        kernel_size=self._kernel_size,
        stride=self._strides,
        dilation_rate=1,
        padding=self._padding,
    )
    input_sizes = [batch, output_time, 1, channels]

    # Why on earth doesn't TensorFlow have a depthwise_conv1d_transpose?
    y = tf.nn.depthwise_conv2d_backprop_input(
        input_sizes=input_sizes,
        filter=sinc_filter,
        out_backprop=x,
        strides=[1, self._strides, self._strides, 1],
        padding='SAME' if self._padding == 'same' else 'VALID',
    )

    y.shape.assert_is_compatible_with([None, None, 1, channels])
    return tf.squeeze(y, 2)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    if not self.supports_step:
      raise ValueError(f'{self} does not support stepping.')
    values = self._layer(x.values)
    mask = _compute_conv_transpose_mask(
        x.mask, self._kernel_size, self._strides, self._padding
    )

    # If kernel_size > stride, mix the state into the output and compute
    # state for the next timestep.
    if self._buffer_width > 0:
      time = utils.smart_dimension_size(x.values, 1)
      output_time = _compute_conv_transpose_output_length(
          time,
          self._kernel_size,
          self._strides,
          dilation_rate=1,
          padding=self._padding,
      )

      # Pad the state to extend it to the length of the layer output.
      # output_time is at least kernel_size and buffer_width is at most
      # kernel_size - 1, so output_time - buffer_width is positive.
      state = state.pad_time(0, output_time - self._buffer_width, valid=True)
      values += state.values
      mask *= state.mask

      # Stride samples are "ready" for output after one timestep, so the number
      # of output samples for the block is stride * time.
      output_samples = self._strides * time

      # We need to store output_time - output_samples samples for the next step,
      # since their value depends on future inputs.
      state_samples = output_time - output_samples
      values, state_values = tf.split(
          values, [output_samples, state_samples], axis=1
      )
      mask, state_mask = tf.split(mask, [output_samples, state_samples], axis=1)
      state = types.Sequence(state_values, state_mask)

    return types.Sequence(values, mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    values = self._layer(x.values)
    mask = _compute_conv_transpose_mask(
        x.mask, self._kernel_size, self._strides, self._padding
    )

    # Trim the last frame_length - stride samples since we don't produce these
    # in step mode (we can only produce stride samples at a time in step mode,
    # so we can't produce the final kernel_size - stride samples).
    x = types.Sequence(values, mask)
    if self._padding == 'causal':
      trim = max(self._kernel_size - self._strides, 0)
      if trim:
        x = x[:, :-trim]
    return x.mask_invalid()


class BaseConv2D(types.SequenceLayer):
  """Shared base logic for Conv2D layers."""

  def __init__(
      self,
      kernel_size: Union[int, List[int], Tuple[int, int]],
      strides: Union[int, List[int], Tuple[int, int]],
      time_padding: str,
      spatial_padding: str,
      dilation_rate: Union[int, List[int], Tuple[int, int]],
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

    self._kernel_size = normalize_2tuple(kernel_size)
    self._strides = normalize_2tuple(strides)
    self._time_padding = types.validate_padding(time_padding)
    self._spatial_padding = types.validate_padding(spatial_padding)
    self._dilation_rate = normalize_2tuple(dilation_rate)

    # (effective_kernel_size - 1) padding makes convolution fully causal.
    if self._time_padding == 'causal':
      effective_kernel_size = utils.convolution_effective_kernel_size(
          self._kernel_size[0], self._dilation_rate[0]
      )
      self._buffer_width = effective_kernel_size - 1
    else:
      self._buffer_width = 0

  @property
  def supports_step(self) -> bool:
    return self._time_padding == 'causal'

  @property
  def block_size(self) -> int:
    return self._strides[0]

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._strides[0])

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    # Special case kernel_size 1 in time since it is stateless.
    if not self._buffer_width:
      return ()
    if x.values.shape.rank != 4:
      raise ValueError('Conv2D requires rank 4 input, got: %s' % str(x))
    # When executing a conv1d step-by-step, we need a buffer for tracking
    # the current causal convolution window.
    batch_size = utils.smart_dimension_size(x.values, 0)
    # This matches the causal padding added by `layer`.
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

  def _apply_conv(
      self,
      x: tf.Tensor,
      explicit_padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> tf.Tensor:
    if (
        explicit_padding[0][0] != 0
        or explicit_padding[0][1] != 0
        or explicit_padding[1][0] != 0
        or explicit_padding[1][1] != 0
    ):
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
    if not self.supports_step:
      raise ValueError(f'{self} does not support stepping.')

    # In step mode, causal padding is handled by the state that is concatenated
    # below.
    time_padding = (0, 0)
    spatial_padding = utils.convolution_explicit_padding(
        self._spatial_padding,
        self._kernel_size[1],
        self._dilation_rate[1],
    )

    # Special case kernel_size 1 in time since it is stateless.
    if not self._buffer_width:
      values = self._apply_conv(x.values, (time_padding, spatial_padding))
      mask = compute_conv_mask(
          x.mask,
          self._kernel_size[0],
          self._strides[0],
          self._dilation_rate[0],
          self._time_padding,
      )
      return types.Sequence(values, mask).mask_invalid(), state

    # Concatenate the new frames with the previous buffer_width frames.
    state = state.concatenate(x)

    # Compute the output for the current timestep.
    values = self._apply_conv(state.values, (time_padding, spatial_padding))
    mask = compute_conv_mask(
        state.mask,
        self._kernel_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._time_padding,
    )

    # Keep the trailing buffer_width samples for the next step.
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
        self._kernel_size[0],
        self._dilation_rate[0],
    )
    spatial_padding = utils.convolution_explicit_padding(
        self._spatial_padding,
        self._kernel_size[1],
        self._dilation_rate[1],
    )
    values = self._apply_conv(x.values, (time_padding, spatial_padding))

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

    mask = compute_conv_mask(
        mask,
        self._kernel_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._time_padding,
    )
    return types.Sequence(values, mask).mask_invalid()


class Conv2D(BaseConv2D):
  """A 2D convolution layer.

  Supports both strided and dilated convolution.

  The first spatial dimension (time) is processed causally, but the second
  spatial dimension is non-causal (i.e. 'valid' or 'same' convolution).
  """

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'conv2d'

  def __init__(
      self,
      filters: int,
      kernel_size: Union[int, List[int], Tuple[int, int]],
      strides: Union[int, List[int], Tuple[int, int]],
      spatial_padding: str,
      dilation_rate: Union[int, List[int], Tuple[int, int]] = 1,
      activation=None,
      use_bias: bool = True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      time_padding: str = 'causal',
      weight_norm: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(
        kernel_size,
        strides,
        time_padding,
        spatial_padding,
        dilation_rate,
        name=name,
    )
    self._filters = filters
    # For the inner Conv2D layer we always use valid and handle padding within
    # this layer.
    with self.name_scope as name_scope:
      self._layer = tf.keras.layers.Conv2D(
          filters,
          self._kernel_size,
          self._strides,
          # We apply time and spatial padding outside of Keras.
          padding='valid',
          data_format='channels_last',
          dilation_rate=self._dilation_rate,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          kernel_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )
      if weight_norm:
        self._layer = utils.WeightNorm(self._layer, name='weight_norm')

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 2:
      raise ValueError(
          'Conv2D requires rank 4 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape.dims[0].value,
        self._spatial_padding,
        kernel_size=self._kernel_size[1],
        stride=self._strides[1],
        dilation_rate=self._dilation_rate[1],
    )
    return tf.TensorShape([spatial_output_size, self._filters])


class DepthwiseConv2D(BaseConv2D):
  """A 2D depthwise convolution layer.

  Supports strided and dilated convolution.

  The first spatial dimension (time) is processed causally, but the second
  spatial dimension is non-causal (i.e. 'valid' or 'same' convolution).
  """

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'depthwise_conv2d'

  def __init__(
      self,
      kernel_size: Union[int, List[int], Tuple[int, int]],
      strides: Union[int, List[int], Tuple[int, int]],
      spatial_padding: str,
      depth_multiplier: int,
      dilation_rate: Union[int, List[int], Tuple[int, int]] = 1,
      activation=None,
      use_bias: bool = True,
      depthwise_initializer='glorot_uniform',
      bias_initializer='zeros',
      depthwise_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      depthwise_constraint=None,
      bias_constraint=None,
      trainable=True,
      time_padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(
        kernel_size,
        strides,
        time_padding,
        spatial_padding,
        dilation_rate,
        name=name,
    )
    self._depth_multiplier = depth_multiplier

    # For the inner DepthwiseConv2D layer we always use valid and handle padding
    # within this layer.
    with self.name_scope as name_scope:
      self._layer = tf.keras.layers.DepthwiseConv2D(
          kernel_size=self._kernel_size,
          strides=self._strides,
          # We apply time and spatial padding outside of Keras.
          padding='valid',
          depth_multiplier=depth_multiplier,
          data_format='channels_last',
          dilation_rate=self._dilation_rate,
          activation=activation,
          use_bias=use_bias,
          depthwise_initializer=depthwise_initializer,
          bias_initializer=bias_initializer,
          depthwise_regularizer=depthwise_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          depthwise_constraint=depthwise_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 2:
      raise ValueError(
          'DepthwiseConv2D requires rank 4 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape.dims[0].value,
        self._spatial_padding,
        kernel_size=self._kernel_size[1],
        stride=self._strides[1],
        dilation_rate=self._dilation_rate[1],
    )
    output_filters = input_shape.dims[1].value * self._depth_multiplier
    return tf.TensorShape([spatial_output_size, output_filters])


class SeparableConv2D(BaseConv2D):
  """A 2D separable convolution layer.

  Supports strided and dilated convolution.

  The first spatial dimension (time) is processed causally, but the second
  spatial dimension is non-causal (i.e. 'valid' or 'same' convolution).
  """

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'separable_conv2d'

  def __init__(
      self,
      filters: int,
      kernel_size: Union[int, List[int], Tuple[int, int]],
      strides: Union[int, List[int], Tuple[int, int]],
      spatial_padding: str,
      depth_multiplier: int,
      dilation_rate: Union[int, List[int], Tuple[int, int]] = 1,
      activation=None,
      use_bias: bool = True,
      depthwise_initializer='glorot_uniform',
      pointwise_initializer='glorot_uniform',
      bias_initializer='zeros',
      depthwise_regularizer=None,
      pointwise_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      depthwise_constraint=None,
      pointwise_constraint=None,
      bias_constraint=None,
      trainable=True,
      time_padding: str = 'causal',
      name: Optional[str] = None,
  ):
    super().__init__(
        kernel_size,
        strides,
        time_padding,
        spatial_padding,
        dilation_rate,
        name=name,
    )
    self._filters = filters

    # For the inner SeparableConv2D layer we always use valid and handle padding
    # within this layer.
    with self.name_scope as name_scope:
      self._layer = tf.keras.layers.SeparableConv2D(
          filters=filters,
          kernel_size=self._kernel_size,
          strides=self._strides,
          # We apply time and spatial padding outside of Keras.
          padding='valid',
          depth_multiplier=depth_multiplier,
          data_format='channels_last',
          dilation_rate=self._dilation_rate,
          activation=activation,
          use_bias=use_bias,
          depthwise_initializer=depthwise_initializer,
          pointwise_initializer=pointwise_initializer,
          bias_initializer=bias_initializer,
          depthwise_regularizer=depthwise_regularizer,
          pointwise_regularizer=pointwise_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          depthwise_constraint=depthwise_constraint,
          pointwise_constraint=pointwise_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 2:
      raise ValueError(
          'SeparableConv2D requires rank 4 input got: %s'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape.dims[0].value,
        self._spatial_padding,
        kernel_size=self._kernel_size[1],
        stride=self._strides[1],
        dilation_rate=self._dilation_rate[1],
    )
    return tf.TensorShape([spatial_output_size, self._filters])
