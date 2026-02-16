# Copyright 2026 Google LLC
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
"""Pooling layers for MLX."""

import fractions

import mlx.core as mx

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import convolution as conv_utils
from sequence_layers.mlx import types

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence
PaddingMode = bt.PaddingMode

# Reuse convolution utilities.
_effective_kernel_size = conv_utils._effective_kernel_size
_explicit_padding = conv_utils._explicit_padding
_buffer_width = conv_utils._buffer_width
_compute_conv_mask = conv_utils._compute_conv_mask

# Pooling supports fewer step modes than convolution (no causal_valid).
_STEP_PADDINGS = frozenset({
    PaddingMode.REVERSE_CAUSAL_VALID.value,
    PaddingMode.CAUSAL.value,
    PaddingMode.REVERSE_CAUSAL.value,
    PaddingMode.SEMICAUSAL.value,
})


def _reduce_window_1d(values, pool_size, stride, dilation_rate, reduce_fn):
  """Gather pooling windows and reduce along the window axis.

  Args:
    values: [batch, time, *channels] input tensor (already padded).
    pool_size: Size of the pooling window.
    stride: Stride between windows.
    dilation_rate: Dilation of the pooling window.
    reduce_fn: Function(array, axis) -> array.

  Returns:
    [batch, num_outputs, *channels]
  """
  if pool_size == 1 and stride == 1:
    return values
  if pool_size == 1:
    return values[:, ::stride]

  t = values.shape[1]
  ek = _effective_kernel_size(pool_size, dilation_rate)
  num_outputs = max(0, (t - ek) // stride + 1)
  if num_outputs == 0:
    out_shape = (values.shape[0], 0) + values.shape[2:]
    return mx.zeros(out_shape, dtype=values.dtype)

  window_offsets = mx.arange(pool_size) * dilation_rate
  start_positions = mx.arange(num_outputs) * stride
  indices = start_positions[:, None] + window_offsets[None, :]
  gathered = values[:, indices]  # [b, n, pool_size, *channels]
  return reduce_fn(gathered, axis=2)


def _reduce_window_masked_avg_1d(
    values, mask, pool_size, stride, dilation_rate
):
  """Sum-then-divide pooling with mask-aware divisor.

  Args:
    values: [batch, time, *channels] already masked to zero.
    mask: [batch, time] boolean mask.
    pool_size: Size of the pooling window.
    stride: Stride between windows.
    dilation_rate: Dilation of the pooling window.

  Returns:
    [batch, num_outputs, *channels]
  """
  t = values.shape[1]
  ek = _effective_kernel_size(pool_size, dilation_rate)
  num_outputs = max(0, (t - ek) // stride + 1)
  if num_outputs == 0:
    out_shape = (values.shape[0], 0) + values.shape[2:]
    return mx.zeros(out_shape, dtype=values.dtype)

  window_offsets = mx.arange(pool_size) * dilation_rate
  start_positions = mx.arange(num_outputs) * stride
  indices = start_positions[:, None] + window_offsets[None, :]

  gathered = values[:, indices]
  v_sum = mx.sum(gathered, axis=2)

  gathered_mask = mask[:, indices].astype(mx.float32)
  count = mx.sum(gathered_mask, axis=2)  # [b, n]
  count = mx.maximum(count, 1.0)
  # Expand to broadcast over channel dims.
  for _ in range(values.ndim - 2):
    count = mx.expand_dims(count, axis=-1)

  return v_sum / count


def _compute_initial_state_pooling(
    batch_size, input_spec, buf_width, padding, pad_value=0.0
):
  """Create initial buffer state for pooling step mode."""
  if padding in (
      PaddingMode.CAUSAL_VALID.value,
      PaddingMode.REVERSE_CAUSAL_VALID.value,
      PaddingMode.SEMICAUSAL_FULL.value,
  ):
    mask = mx.ones((batch_size, buf_width), dtype=bt.MASK_DTYPE)
  elif padding in (
      PaddingMode.CAUSAL.value,
      PaddingMode.REVERSE_CAUSAL.value,
      PaddingMode.SEMICAUSAL.value,
  ):
    mask = mx.zeros((batch_size, buf_width), dtype=bt.MASK_DTYPE)
  else:
    raise ValueError(f'Step not supported with padding: {padding}')

  values = mx.full(
      (batch_size, buf_width) + input_spec.shape,
      pad_value,
      dtype=input_spec.dtype,
  )
  # Return Sequence (not MaskedSequence) — matches JAX's .unmask().
  return Sequence(values, mask)


class _Pooling1D(types.PreservesType, types.SequenceLayer):
  """Base class for 1D pooling layers."""

  def __init__(self, pool_size, strides=1, dilation_rate=1, padding='valid'):
    super().__init__()
    self._pool_size = pool_size
    self._strides = strides
    self._dilation_rate = dilation_rate
    self._padding = padding

  def _pad_value(self, dtype):
    raise NotImplementedError

  def _reduce(self, gathered, axis):
    raise NotImplementedError

  @property
  def supports_step(self):
    return self._padding in _STEP_PADDINGS

  @property
  def block_size(self):
    return self._strides

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self._strides)

  @property
  def input_latency(self):
    ek = _effective_kernel_size(self._pool_size, self._dilation_rate)
    if self._padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self._padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
    ):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    bw = _buffer_width(
        self._padding,
        self._pool_size,
        self._strides,
        self._dilation_rate,
    )
    if not bw:
      return ()
    return _compute_initial_state_pooling(
        batch_size,
        input_spec,
        bw,
        self._padding,
        pad_value=self._pad_value(input_spec.dtype),
    )

  @types.check_layer
  def layer(self, x, *, constants=None):
    pad_value = self._pad_value(x.dtype)
    if self._pool_size > 1:
      x = x.mask_invalid(pad_value)

    pad_left, pad_right = _explicit_padding(
        self._padding,
        self._pool_size,
        self._strides,
        self._dilation_rate,
    )
    values = x.values
    if pad_left > 0 or pad_right > 0:
      pad_widths = [(0, 0), (pad_left, pad_right)] + [(0, 0)] * (
          values.ndim - 2
      )
      values = mx.pad(values, pad_widths, constant_values=pad_value)

    values = _reduce_window_1d(
        values,
        self._pool_size,
        self._strides,
        self._dilation_rate,
        self._reduce,
    )
    mask = _compute_conv_mask(
        x.mask,
        self._pool_size,
        self._strides,
        self._dilation_rate,
        self._padding,
        is_step=False,
    )
    return Sequence(values, mask)

  @types.check_step
  def step(self, x, state, *, constants=None):
    pad_value = self._pad_value(x.dtype)
    ek = _effective_kernel_size(self._pool_size, self._dilation_rate)
    if ek > 1:
      x = x.mask_invalid(pad_value)

    bw = _buffer_width(
        self._padding,
        self._pool_size,
        self._strides,
        self._dilation_rate,
    )

    if bw:
      state = state.concatenate(x)
    else:
      state = x

    values = _reduce_window_1d(
        state.values,
        self._pool_size,
        self._strides,
        self._dilation_rate,
        self._reduce,
    )
    mask = _compute_conv_mask(
        state.mask,
        self._pool_size,
        self._strides,
        self._dilation_rate,
        self._padding,
        is_step=True,
    )

    if bw:
      state = state[:, -bw:]
    else:
      state = ()

    return Sequence(values, mask), state


class MaxPooling1D(_Pooling1D):
  """1D max pooling layer."""

  def _pad_value(self, dtype):
    return float('-inf')

  def _reduce(self, gathered, axis):
    return mx.max(gathered, axis=axis)

  @classmethod
  def from_config(cls, config):
    return cls(
        pool_size=config.pool_size,
        strides=config.strides,
        dilation_rate=config.dilation_rate,
        padding=config.padding,
    )


class MinPooling1D(_Pooling1D):
  """1D min pooling layer."""

  def _pad_value(self, dtype):
    return float('inf')

  def _reduce(self, gathered, axis):
    return mx.min(gathered, axis=axis)

  @classmethod
  def from_config(cls, config):
    return cls(
        pool_size=config.pool_size,
        strides=config.strides,
        dilation_rate=config.dilation_rate,
        padding=config.padding,
    )


class AveragePooling1D(_Pooling1D):
  """1D average pooling layer."""

  def __init__(
      self,
      pool_size,
      strides=1,
      dilation_rate=1,
      padding='valid',
      masked_average=False,
  ):
    super().__init__(pool_size, strides, dilation_rate, padding)
    self._masked_average = masked_average

  def _pad_value(self, dtype):
    return 0.0

  def _reduce(self, gathered, axis):
    return mx.mean(gathered, axis=axis)

  @types.check_layer
  def layer(self, x, *, constants=None):
    if not self._masked_average:
      return _Pooling1D.layer.__wrapped__(self, x, constants=constants)

    # Masked average: divide by count of valid elements.
    x = x.mask_invalid(0.0)
    pad_left, pad_right = _explicit_padding(
        self._padding,
        self._pool_size,
        self._strides,
        self._dilation_rate,
    )
    values = x.values
    input_mask = x.mask
    if pad_left > 0 or pad_right > 0:
      pad_widths = [(0, 0), (pad_left, pad_right)] + [(0, 0)] * (
          values.ndim - 2
      )
      values = mx.pad(values, pad_widths, constant_values=0.0)
      input_mask = mx.pad(
          input_mask,
          [(0, 0), (pad_left, pad_right)],
          constant_values=False,
      )

    values = _reduce_window_masked_avg_1d(
        values,
        input_mask,
        self._pool_size,
        self._strides,
        self._dilation_rate,
    )
    mask = _compute_conv_mask(
        x.mask,
        self._pool_size,
        self._strides,
        self._dilation_rate,
        self._padding,
        is_step=False,
    )
    return Sequence(values, mask)

  @types.check_step
  def step(self, x, state, *, constants=None):
    if not self._masked_average:
      return _Pooling1D.step.__wrapped__(self, x, state, constants=constants)

    # Masked average step.
    ek = _effective_kernel_size(self._pool_size, self._dilation_rate)
    if ek > 1:
      x = x.mask_invalid(0.0)

    bw = _buffer_width(
        self._padding,
        self._pool_size,
        self._strides,
        self._dilation_rate,
    )

    if bw:
      state = state.concatenate(x)
    else:
      state = x

    values = _reduce_window_masked_avg_1d(
        state.values,
        state.mask,
        self._pool_size,
        self._strides,
        self._dilation_rate,
    )
    mask = _compute_conv_mask(
        state.mask,
        self._pool_size,
        self._strides,
        self._dilation_rate,
        self._padding,
        is_step=True,
    )

    if bw:
      state = state[:, -bw:]
    else:
      state = ()

    return Sequence(values, mask), state

  @classmethod
  def from_config(cls, config):
    return cls(
        pool_size=config.pool_size,
        strides=config.strides,
        dilation_rate=config.dilation_rate,
        padding=config.padding,
        masked_average=config.masked_average,
    )
