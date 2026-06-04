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
"""Position embeddings and timing signals for MLX."""

import dataclasses
import math
from typing import Any, cast, override

import mlx.core as mx
import numpy as np

from sequence_layers.mlx import types
from sequence_layers.mlx.init_mapping import _to_mx_dtype
from sequence_layers.specs import position as position_spec

from sequence_layers.mlx import types as bt

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence


def _match_shape_along_axes(channel_shape, axes):
  """Return a shape matching channel_shape on axes and is equal to 1 elsewhere."""
  if axes is None:
    return tuple(channel_shape)

  target_shape = [1] * len(channel_shape)
  if isinstance(axes, int):
    axes = [axes]
  for axis in axes:
    if not -len(channel_shape) <= axis < len(channel_shape):
      raise ValueError(f'Invalid {axis=} found in {axes=}.')
    target_shape[axis] = channel_shape[axis]
  return tuple(target_shape)


def _get_timing_signal_1d_pos(
    position, channels, min_timescale=1.0, max_timescale=1.0e4, dtype=mx.float32
):
  """Compute 1D sinusoidal timing signal in MLX."""
  position = position.astype(mx.float32)
  num_timescales = channels // 2
  log_timescale_increment = math.log(
      float(max_timescale) / float(min_timescale)
  ) / max(num_timescales - 1, 1)

  inv_timescales = min_timescale * np.exp(
      np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment
  )
  inv_timescales_mx = mx.array(inv_timescales, dtype=mx.float32)

  scaled_time = (
      mx.expand_dims(position, axis=2) * inv_timescales_mx[None, None, :]
  )
  timing_signal = mx.concatenate(
      [mx.sin(scaled_time), mx.cos(scaled_time)], axis=2
  )
  if channels % 2 != 0:
    padding = mx.zeros(
        (timing_signal.shape[0], timing_signal.shape[1], 1),
        dtype=timing_signal.dtype,
    )
    timing_signal = mx.concatenate([timing_signal, padding], axis=2)
  return timing_signal.astype(dtype)


class AddTimingSignal(
    types.PreservesType,
    types.PreservesShape,
    types.SequenceLayer,
    position_spec.AddTimingSignal[types.Sequence, types.ChannelSpec],
):
  """Adds sinusoids at varying frequencies to the input channels dimension."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, position_spec.AddTimingSignal.Config):
    """Configuration for AddTimingSignal."""

    min_timescale: float = 1.0
    max_timescale: float = 1.0e4
    trainable_scale: bool = False
    axes: int | tuple[int, ...] | None = None
    sharding: Any = None
    param_dtype: types.DType = mx.float32
    only_advance_position_for_valid_timesteps: bool = True
    name: str | None = None

    @override
    def make(self) -> 'AddTimingSignal':
      return AddTimingSignal(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      trainable_scale: bool = False,
      axes: int | tuple[int, ...] | None = None,
      only_advance_position_for_valid_timesteps: bool = True,
      param_dtype: types.DType = mx.float32,
  ):
    super().__init__()
    if config is not None:
      self.config = config
    else:
      self.config = self.Config(
          min_timescale=min_timescale,
          max_timescale=max_timescale,
          trainable_scale=trainable_scale,
          axes=axes,
          only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
          param_dtype=param_dtype,
      )

    self.min_timescale = self.config.min_timescale
    self.max_timescale = self.config.max_timescale
    self.trainable_scale = self.config.trainable_scale
    self.axes = self.config.axes
    self.only_advance_position_for_valid_timesteps = (
        self.config.only_advance_position_for_valid_timesteps
    )
    self.param_dtype = self.config.param_dtype

    if self.trainable_scale:
      self.scale = mx.ones((), dtype=self.param_dtype)
    else:
      self.scale = cast(Any, None)

  def _check_inputs(self, input_spec):
    """Validates the input specification."""
    if input_spec.dtype not in (
        mx.float16,
        mx.bfloat16,
        mx.float32,
    ):
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    return input_dtype

  @override
  def get_initial_state(
      self, batch_size, input_spec, *, training: bool, constants=None
  ):
    self._check_inputs(input_spec)
    if self.only_advance_position_for_valid_timesteps:
      return mx.full((batch_size, 1), -1, dtype=mx.int32)
    return mx.zeros((batch_size, 1), dtype=mx.int32)

  @override
  @types.check_step
  def step(  # pyrefly: ignore[missing-override-decorator]
      self, x, state: Any, *, training: bool, constants=None
  ):
    self._check_inputs(x.channel_spec)
    time = x.shape[1]
    target_shape = _match_shape_along_axes(x.channel_shape, axes=self.axes)

    if self.only_advance_position_for_valid_timesteps:
      position = state + mx.cumsum(x.mask.astype(mx.int32), axis=1)
      state = position[:, -1:]
    else:
      position = state + mx.arange(time, dtype=mx.int32)
      state = state + time

    timing_signal = _get_timing_signal_1d_pos(
        position,
        np.prod(target_shape),
        min_timescale=self.min_timescale,
        max_timescale=self.max_timescale,
        dtype=self.param_dtype,
    )
    batch_size = x.shape[0]
    timing_signal = mx.reshape(
        timing_signal, [batch_size, time] + list(target_shape)
    )
    if self.scale is not None:
      timing_signal = timing_signal * self.scale
    x = x.apply_values(lambda v: v + timing_signal.astype(v.dtype))
    return x, state

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    self._check_inputs(x.channel_spec)
    target_shape = _match_shape_along_axes(x.channel_shape, axes=self.axes)

    if self.only_advance_position_for_valid_timesteps:
      position = mx.maximum(0, mx.cumsum(x.mask.astype(mx.int32), axis=1) - 1)
    else:
      position = mx.expand_dims(mx.arange(x.shape[1], dtype=mx.int32), axis=0)

    timing_signal = _get_timing_signal_1d_pos(
        position,
        np.prod(target_shape),
        min_timescale=self.min_timescale,
        max_timescale=self.max_timescale,
        dtype=self.param_dtype,
    )
    timing_signal = mx.reshape(
        timing_signal, list(position.shape[:2]) + list(target_shape)
    )
    if self.scale is not None:
      timing_signal = timing_signal * self.scale
    x = x.apply_values(lambda v: v + timing_signal.astype(v.dtype))
    return x

  @classmethod
  def from_config(cls, config):
    """Instantiates the layer from config."""
    mlx_config = cls.Config(
        min_timescale=config.min_timescale,
        max_timescale=config.max_timescale,
        trainable_scale=config.trainable_scale,
        axes=config.axes,
        only_advance_position_for_valid_timesteps=config.only_advance_position_for_valid_timesteps,
        param_dtype=_to_mx_dtype(config.param_dtype),
        name=config.name,
    )
    return cls(mlx_config)


class ApplyRotaryPositionalEncoding(
    types.PreservesType,
    types.PreservesShape,
    types.SequenceLayer,
    position_spec.ApplyRotaryPositionalEncoding[
        types.Sequence, types.ChannelSpec
    ],
):
  """Applies Rotary Positional Encodings (RoPE) to the sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(
      types.SequenceLayerConfig,
      position_spec.ApplyRotaryPositionalEncoding.Config,
  ):
    """Configuration for ApplyRotaryPositionalEncoding."""

    max_wavelength: float
    axis: int = -1
    only_advance_position_for_valid_timesteps: bool = True
    positions_in_at_least_fp32: bool = True
    positions_name: str | None = None
    name: str | None = None

    @override
    def make(self) -> 'ApplyRotaryPositionalEncoding':
      return ApplyRotaryPositionalEncoding(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      max_wavelength: float | None = None,
      axis: int = -1,
      only_advance_position_for_valid_timesteps: bool = True,
      positions_in_at_least_fp32: bool = True,
      positions_name: str | None = None,
  ):
    super().__init__()
    if config is not None:
      self.config = config
    else:
      if max_wavelength is None:
        raise ValueError('Must provide either config or max_wavelength')
      self.config = self.Config(
          max_wavelength=max_wavelength,
          axis=axis,
          only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
          positions_in_at_least_fp32=positions_in_at_least_fp32,
          positions_name=positions_name,
      )

    self.max_wavelength = self.config.max_wavelength
    self._axis = self.config.axis
    self.only_advance_position_for_valid_timesteps = (
        self.config.only_advance_position_for_valid_timesteps
    )
    self.positions_in_at_least_fp32 = self.config.positions_in_at_least_fp32
    self.positions_name = self.config.positions_name

  def _validate(self):
    """Validates the configuration properties."""
    if self.only_advance_position_for_valid_timesteps and self.positions_name:
      raise ValueError(
          'only_advance_position_for_valid_timesteps is incompatible with'
          f' {self.positions_name=}.'
      )

  def _check_inputs(self, input_spec):
    """Validates input specifications and shape constraints."""
    self._validate()
    if input_spec.dtype not in (
        mx.float16,
        mx.bfloat16,
        mx.float32,
    ):
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )
    input_shape = (None, None) + tuple(input_spec.shape)
    axis = self._axis + len(input_shape) if self._axis < 0 else self._axis
    if axis <= 1:
      raise ValueError(
          f'{type(self).__name__} axis ({self._axis}) must refer to a'
          f' channels dimension ({input_spec=}).'
      )
    axis_size = input_shape[axis]
    if axis_size is not None and axis_size % 2 != 0:
      raise ValueError(
          f'{type(self).__name__} requires input_shape[{axis}]={axis_size} to'
          ' be even.'
      )

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    return input_dtype

  def _apply_rope(self, x, offset_or_positions):
    """Applies rotary position encoding to x.

    If rotation axis is the last dimension and we are using a simple temporal offset
    (i.e. not custom positions from positions_name), we leverage the highly optimized
    `mx.fast.rope` C++ operation. Otherwise, we fall back to manual trig calculation.
    """
    axis = self._axis + x.ndim if self._axis < 0 else self._axis

    is_custom_positions = (
        hasattr(offset_or_positions, 'ndim') and offset_or_positions.ndim >= 2
    )

    if is_custom_positions or axis != x.ndim - 1:
      # Manual fallback
      channel_ndim = x.ndim - 2
      axis_dim = x.shape[axis]
      assert axis_dim % 2 == 0

      freq_exponents = (
          2.0 * mx.arange(axis_dim // 2).astype(mx.float32) / axis_dim
      )
      timescale = self.max_wavelength**freq_exponents

      broadcast_shape = [1] * x.ndim
      broadcast_shape[axis] = axis_dim // 2

      if is_custom_positions:
        positions = offset_or_positions
      else:
        offset = offset_or_positions
        positions = mx.expand_dims(
            mx.arange(x.shape[1]), axis=0
        ) + mx.expand_dims(offset, axis=1)

      positions_f = positions.astype(mx.float32)
      radians = positions_f.reshape(
          positions_f.shape + (1,) * channel_ndim
      ) / timescale.reshape(broadcast_shape)
      sin_r = mx.sin(radians)
      cos_r = mx.cos(radians)

      splits = mx.split(x, 2, axis=axis)
      x1, x2 = splits[0], splits[1]
      result = mx.concatenate(
          [x1 * cos_r - x2 * sin_r, x2 * cos_r + x1 * sin_r],
          axis=axis,
      )
      return result.astype(x.dtype)

    # Optimized mx.fast.rope path
    offset = offset_or_positions
    original_axes = list(range(x.ndim))
    if x.ndim >= 3:
      transpose_axes = original_axes.copy()
      transpose_axes.pop(1)
      transpose_axes.insert(-1, 1)
      x_t = mx.transpose(x, transpose_axes)
    else:
      x_t = x

    y_t = mx.fast.rope(
        x_t,
        dims=x.shape[-1],
        traditional=False,
        base=self.max_wavelength,
        scale=1.0,
        offset=offset,
    )

    if x.ndim >= 3:
      inv_axes = original_axes.copy()
      inv_axes.pop(-2)
      inv_axes.insert(1, x.ndim - 2)
      y = mx.transpose(y_t, inv_axes)
    else:
      y = y_t
    return y.astype(x.dtype)

  @override
  def get_initial_state(
      self, batch_size, input_spec, *, training: bool, constants=None
  ):
    self._validate()
    self._check_inputs(input_spec)
    if self.positions_name:
      return ()
    if self.only_advance_position_for_valid_timesteps:
      return mx.full((batch_size, 1), -1, dtype=mx.int32)
    return mx.zeros((batch_size, 1), dtype=mx.int32)

  @override
  @types.check_step
  def step(  # pyrefly: ignore[missing-override-decorator]
      self, x, state: Any, *, training: bool, constants=None
  ):
    self._check_inputs(x.channel_spec)
    x_time = x.shape[1]

    if self.positions_name:
      if constants is None or self.positions_name not in constants:
        raise ValueError(
            f'Expected constants dict containing {self.positions_name!r}'
        )
      positions_const = constants[self.positions_name]
      if isinstance(positions_const, (Sequence, MaskedSequence)):
        offset_or_positions = positions_const.values
      else:
        offset_or_positions = positions_const
    elif self.only_advance_position_for_valid_timesteps:
      offset = mx.maximum(0, state[:, 0] + 1)
      positions = state + mx.cumsum(x.mask.astype(mx.int32), axis=1)
      state = positions[:, -1:]
      offset_or_positions = offset
    else:
      offset = state[:, 0]
      state = state + x_time
      offset_or_positions = offset

    y = x.apply_values(self._apply_rope, offset_or_positions)
    return y, state

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    self._check_inputs(x.channel_spec)
    if self.positions_name:
      if constants is None or self.positions_name not in constants:
        raise ValueError(
            f'Expected constants dict containing {self.positions_name!r}'
        )
      positions_const = constants[self.positions_name]
      if isinstance(positions_const, (Sequence, MaskedSequence)):
        offset_or_positions = positions_const.values
      else:
        offset_or_positions = positions_const
    elif self.only_advance_position_for_valid_timesteps:
      offset_or_positions = mx.maximum(
          0, mx.cumsum(x.mask.astype(mx.int32), axis=1) - 1
      )
    else:
      offset_or_positions = mx.zeros((x.shape[0],), dtype=mx.int32)

    y = x.apply_values(self._apply_rope, offset_or_positions)
    return y

  @property
  @override
  def receptive_field(self) -> types.ReceptiveField:
    return (0, 0)

  @classmethod
  def from_config(cls, config):
    """Instantiates the layer from config."""
    mlx_config = cls.Config(
        max_wavelength=config.max_wavelength,
        axis=config.axis,
        only_advance_position_for_valid_timesteps=(
            config.only_advance_position_for_valid_timesteps
        ),
        positions_in_at_least_fp32=config.positions_in_at_least_fp32,
        positions_name=config.positions_name,
        name=config.name,
    )
    return cls(mlx_config)
