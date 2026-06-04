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

import dataclasses
import fractions
from typing import Any, override

import mlx.core as mx
import numpy as np

from sequence_layers.mlx import convolution as conv_utils
from sequence_layers.mlx import types
from sequence_layers.specs import pooling as spec


def _is_floating(dtype):
  """Returns True if the given MLX/numpy dtype is a floating type."""
  return np.issubdtype(
      np.dtype(str(dtype).rsplit('.', maxsplit=1)[-1]), np.floating
  )


def _is_integer(dtype):
  """Returns True if the given MLX/numpy dtype is an integer type."""
  return np.issubdtype(
      np.dtype(str(dtype).rsplit('.', maxsplit=1)[-1]), np.integer
  )


def _max_pool_init_value(dtype) -> Any:
  """Returns the initial value for MaxPooling1D for a given dtype."""
  if _is_floating(dtype):
    return float('-inf')
  if _is_integer(dtype):
    np_dt = np.dtype(str(dtype).rsplit('.', maxsplit=1)[-1])
    return int(np.iinfo(np_dt).min)
  if str(dtype) == 'mlx.core.bool':
    return False
  raise ValueError(f'Unsupported dtype for max pool: {dtype}')


def _min_pool_init_value(dtype) -> Any:
  """Returns the initial value for MinPooling1D for a given dtype."""
  if _is_floating(dtype):
    return float('inf')
  if _is_integer(dtype):
    np_dt = np.dtype(str(dtype).rsplit('.', maxsplit=1)[-1])
    return int(np.iinfo(np_dt).max)
  if str(dtype) == 'mlx.core.bool':
    return True
  raise ValueError(f'Unsupported dtype for min pool: {dtype}')


bt = types

Sequence = bt.Sequence

MaskedSequence = bt.MaskedSequence
PaddingMode = bt.PaddingMode

# Reuse convolution utilities.
# pylint: disable=protected-access
_effective_kernel_size = conv_utils._effective_kernel_size
_explicit_padding = conv_utils._explicit_padding
_buffer_width = conv_utils._buffer_width
_compute_conv_mask = conv_utils._compute_conv_mask
# pylint: enable=protected-access

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

  if mx.issubdtype(values.dtype, mx.integer):
    gathered_mask = mask[:, indices].astype(mx.int32)
    count = mx.sum(gathered_mask, axis=2)  # [b, n]
    count = mx.maximum(count, 1)
    # Expand to broadcast over channel dims.
    for _ in range(values.ndim - 2):
      count = mx.expand_dims(count, axis=-1)
    return v_sum // count

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


class _Pooling1D(
    types.PreservesType,
    types.SequenceLayer,
    spec.BasePooling[types.Sequence, types.ShapeDType],
):
  """Base class for 1D pooling layers."""

  def __init__(self, pool_size, strides=1, dilation_rate=1, padding='valid'):
    super().__init__()
    self._pool_size = pool_size
    self._strides = strides
    self._dilation_rate = dilation_rate
    self._padding = padding

  def _pad_value(self, dtype):
    """Returns the pad value for the given dtype."""
    raise NotImplementedError

  def _reduce(self, gathered, axis):
    """Reduces gathered windows along the specified axis."""
    raise NotImplementedError

  @override
  @property
  def supports_step(self):
    return self._padding in _STEP_PADDINGS

  @override
  @property
  def block_size(self):
    return self._strides

  @override
  @property
  def output_ratio(self):
    return fractions.Fraction(1, self._strides)

  @override
  @property
  def input_latency(self):
    ek = _effective_kernel_size(self._pool_size, self._dilation_rate)
    if self._padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    if self._padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
    ):
      return ek - 1
    return 0

  @property
  @override
  def receptive_field(self) -> types.ReceptiveField:
    return super().receptive_field

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  @override
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants=None,
  ):
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

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool = False, constants=None
  ):
    pad_value = self._pad_value(x.dtype)
    # pylint: disable=protected-access
    output_length = conv_utils._compute_output_length(
        x.shape[1],
        self._pool_size,
        self._strides,
        self._dilation_rate,
        self._padding,
    )
    # pylint: enable=protected-access
    if output_length == 0:
      empty_values = mx.zeros(
          (x.shape[0], 0, x.shape[-1]), dtype=x.values.dtype
      )
      empty_mask = mx.zeros((x.shape[0], 0), dtype=mx.bool_)
      return Sequence(empty_values, empty_mask)

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

  @override
  @types.check_step
  def step(  # pyrefly: ignore[missing-override-decorator]
      self, x, state, *, training: bool = False, constants=None
  ):
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
      state = state.concatenate(x)  # pyrefly: ignore[missing-attribute]
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


class MaxPooling1D(
    _Pooling1D, spec.MaxPooling1D[types.Sequence, types.ShapeDType]
):
  """1D max pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.MaxPooling1D.Config):
    """Configuration for MaxPooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    name: str | None = None

    @override
    def make(self) -> 'MaxPooling1D':
      return MaxPooling1D(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      pool_size: int | None = None,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: types.PaddingModeString = 'valid',
  ):
    if config is not None:
      super().__init__(
          pool_size=config.pool_size,
          strides=config.strides,
          dilation_rate=config.dilation_rate,
          padding=config.padding,
      )
      self.config = config
    else:
      if pool_size is None:
        raise ValueError('Must provide either config or pool_size')
      super().__init__(
          pool_size=pool_size,
          strides=strides,
          dilation_rate=dilation_rate,
          padding=padding,
      )
      self.config = self.Config(
          pool_size=pool_size,
          strides=strides,
          dilation_rate=dilation_rate,
          padding=padding,
      )

  @override
  def _pad_value(self, dtype):
    return _max_pool_init_value(dtype)

  @override
  def _reduce(self, gathered, axis):
    return mx.max(gathered, axis=axis)

  @classmethod
  def from_config(cls, config):
    """Creates a MaxPooling1D instance from config."""
    return cls(config)


class MinPooling1D(
    _Pooling1D, spec.MinPooling1D[types.Sequence, types.ShapeDType]
):
  """1D min pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.MinPooling1D.Config):
    """Configuration for MinPooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    name: str | None = None

    @override
    def make(self) -> 'MinPooling1D':
      return MinPooling1D(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      pool_size: int | None = None,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: types.PaddingModeString = 'valid',
  ):
    if config is not None:
      super().__init__(
          pool_size=config.pool_size,
          strides=config.strides,
          dilation_rate=config.dilation_rate,
          padding=config.padding,
      )
      self.config = config
    else:
      if pool_size is None:
        raise ValueError('Must provide either config or pool_size')
      super().__init__(
          pool_size=pool_size,
          strides=strides,
          dilation_rate=dilation_rate,
          padding=padding,
      )
      self.config = self.Config(
          pool_size=pool_size,
          strides=strides,
          dilation_rate=dilation_rate,
          padding=padding,
      )

  @override
  def _pad_value(self, dtype):
    return _min_pool_init_value(dtype)

  @override
  def _reduce(self, gathered, axis):
    return mx.min(gathered, axis=axis)

  @classmethod
  def from_config(cls, config):
    """Creates a MinPooling1D instance from config."""
    return cls(config)


class AveragePooling1D(
    _Pooling1D,
    spec.AveragePooling1D[types.Sequence, types.ShapeDType],
):
  """1D average pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.AveragePooling1D.Config):
    """Configuration for AveragePooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    masked_average: bool = False
    name: str | None = None

    @override
    def make(self) -> 'AveragePooling1D':
      return AveragePooling1D(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      pool_size: int | None = None,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: types.PaddingModeString = 'valid',
      masked_average: bool = False,
  ):
    if config is not None:
      super().__init__(
          pool_size=config.pool_size,
          strides=config.strides,
          dilation_rate=config.dilation_rate,
          padding=config.padding,
      )
      self._masked_average = config.masked_average
      self.config = config
    else:
      if pool_size is None:
        raise ValueError('Must provide either config or pool_size')
      super().__init__(
          pool_size=pool_size,
          strides=strides,
          dilation_rate=dilation_rate,
          padding=padding,
      )
      self._masked_average = masked_average
      self.config = self.Config(
          pool_size=pool_size,
          strides=strides,
          dilation_rate=dilation_rate,
          padding=padding,
          masked_average=masked_average,
      )

  @override
  def _pad_value(self, dtype):
    """Returns the pad value for the given dtype."""
    if mx.issubdtype(dtype, mx.integer):
      return 0
    if dtype == mx.bool_:
      return False
    return 0.0

  @override
  def _reduce(self, gathered, axis):
    """Reduces gathered windows along the specified axis."""
    if mx.issubdtype(gathered.dtype, mx.integer):
      return mx.sum(gathered, axis=axis) // gathered.shape[axis]
    return mx.mean(gathered, axis=axis)

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool = False, constants=None
  ):
    if not self._masked_average:
      return _Pooling1D.layer.__wrapped__(  # pylint: disable=no-member  # pyrefly: ignore[missing-attribute]
          self, x, training=training, constants=constants
      )

    pad_value = self._pad_value(x.dtype)
    # Masked average: divide by count of valid elements.

    x = x.mask_invalid(pad_value)
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
      values = mx.pad(values, pad_widths, constant_values=pad_value)
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

  @override
  @types.check_step
  def step(  # pyrefly: ignore[missing-override-decorator]
      self, x, state, *, training: bool = False, constants=None
  ):
    if not self._masked_average:
      return _Pooling1D.step.__wrapped__(  # pylint: disable=no-member  # pyrefly: ignore[missing-attribute]
          self, x, state, training=training, constants=constants
      )

    # Masked average step.
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
    """Creates an AveragePooling1D instance from config."""
    return cls(config)
