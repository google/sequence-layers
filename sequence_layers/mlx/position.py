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
from typing import override

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import types
from sequence_layers.specs import position as position_spec

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
  class Config(position_spec.AddTimingSignal.Config):
    param_dtype: types.DType = mx.float32

    @override
    def make(self) -> 'AddTimingSignal':
      return AddTimingSignal.from_config(self)

  def __init__(
      self,
      *,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      trainable_scale: bool = False,
      axes: int | tuple[int, ...] | None = None,
      only_advance_position_for_valid_timesteps: bool = True,
      param_dtype: types.DType = mx.float32,
  ):
    super().__init__()
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.trainable_scale = trainable_scale
    self.axes = axes
    self.only_advance_position_for_valid_timesteps = (
        only_advance_position_for_valid_timesteps
    )
    self.param_dtype = param_dtype

    if self.trainable_scale:
      self.scale = mx.ones((), dtype=self.param_dtype)
    else:
      self.scale = None

  def _check_inputs(self, input_spec):
    if input_spec.dtype not in (
        mx.float16,
        mx.bfloat16,
        mx.float32,
    ):
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )

  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return input_dtype

  def get_initial_state(
      self, batch_size, input_spec, *, training: bool, constants=None
  ):
    self._check_inputs(input_spec)
    if self.only_advance_position_for_valid_timesteps:
      return mx.full((batch_size, 1), -1, dtype=mx.int32)
    else:
      return mx.zeros((batch_size, 1), dtype=mx.int32)

  @types.check_step
  def step(self, x, state, *, training: bool, constants=None):
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

  @types.check_layer
  def layer(self, x, *, training: bool, constants=None):
    self._check_inputs(x.channel_spec)
    target_shape = _match_shape_along_axes(x.channel_shape, axes=self.axes)

    if self.only_advance_position_for_valid_timesteps:
      position = mx.maximum(0, mx.cumsum(x.mask.astype(mx.int32), axis=1) - 1)
    else:
      position = mx.arange(x.shape[1], dtype=mx.int32)[None, :]

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
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    layer = cls(
        min_timescale=config.min_timescale,
        max_timescale=config.max_timescale,
        trainable_scale=config.trainable_scale,
        axes=config.axes,
        only_advance_position_for_valid_timesteps=config.only_advance_position_for_valid_timesteps,
        param_dtype=_to_mx_dtype(config.param_dtype),
    )
    layer.config = config
    return layer


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
  class Config(position_spec.ApplyRotaryPositionalEncoding.Config):

    @override
    def make(self) -> 'ApplyRotaryPositionalEncoding':
      return ApplyRotaryPositionalEncoding.from_config(self)

  def __init__(
      self,
      *,
      max_wavelength: float,
      axis: int = -1,
      only_advance_position_for_valid_timesteps: bool = True,
      positions_in_at_least_fp32: bool = True,
      positions_name: str | None = None,
  ):
    super().__init__()
    self.max_wavelength = max_wavelength
    self._axis = axis
    self.only_advance_position_for_valid_timesteps = (
        only_advance_position_for_valid_timesteps
    )
    self.positions_in_at_least_fp32 = positions_in_at_least_fp32
    self.positions_name = positions_name

  def _validate(self):
    if self.only_advance_position_for_valid_timesteps and self.positions_name:
      raise ValueError(
          'only_advance_position_for_valid_timesteps is incompatible with'
          f' {self.positions_name=}.'
      )

  def _check_inputs(self, input_spec):
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

  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return input_dtype

  def _apply_rope(self, x, positions):
    """Applies rotary position encoding to x with given positions tensor."""
    axis = self._axis + x.ndim if self._axis < 0 else self._axis
    assert axis > 1

    channel_ndim = x.ndim - 2
    axis_dim = x.shape[axis]
    assert axis_dim % 2 == 0

    freq_exponents = (
        2.0 * mx.arange(axis_dim // 2).astype(mx.float32) / axis_dim
    )
    timescale = self.max_wavelength**freq_exponents

    broadcast_shape = [1] * x.ndim
    broadcast_shape[axis] = axis_dim // 2

    # Compute position angles
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

  def get_initial_state(
      self, batch_size, input_spec, *, training: bool, constants=None
  ):
    self._validate()
    self._check_inputs(input_spec)
    if self.positions_name:
      return ()
    elif self.only_advance_position_for_valid_timesteps:
      return mx.full((batch_size, 1), -1, dtype=mx.int32)
    else:
      return mx.zeros((batch_size, 1), dtype=mx.int32)

  @types.check_step
  def step(self, x, state, *, training: bool, constants=None):
    self._validate()
    self._check_inputs(x.channel_spec)
    x_time = x.shape[1]

    if self.positions_name:
      # Read from constants dictionary if specified
      if constants is None or self.positions_name not in constants:
        raise ValueError(
            f'Expected constants dict containing {self.positions_name!r}'
        )
      positions_const = constants[self.positions_name]
      if isinstance(positions_const, (Sequence, MaskedSequence)):
        positions = positions_const.values
      else:
        positions = positions_const
    elif self.only_advance_position_for_valid_timesteps:
      positions = state + mx.cumsum(x.mask.astype(mx.int32), axis=1)
      state = positions[:, -1:]
    else:
      positions = state + mx.arange(x_time, dtype=mx.int32)
      state = state + x_time

    y = x.apply_values(self._apply_rope, positions)
    return y, state

  @types.check_layer
  def layer(self, x, *, training: bool, constants=None):
    self._validate()
    self._check_inputs(x.channel_spec)
    if self.positions_name:
      if constants is None or self.positions_name not in constants:
        raise ValueError(
            f'Expected constants dict containing {self.positions_name!r}'
        )
      positions_const = constants[self.positions_name]
      if isinstance(positions_const, (Sequence, MaskedSequence)):
        positions = positions_const.values
      else:
        positions = positions_const
    elif self.only_advance_position_for_valid_timesteps:
      positions = mx.maximum(0, mx.cumsum(x.mask.astype(mx.int32), axis=1) - 1)
    else:
      positions = mx.arange(x.shape[1], dtype=mx.int32)[None, :]
    return x.apply_values(self._apply_rope, positions)

  @classmethod
  def from_config(cls, config):
    layer = cls(
        max_wavelength=config.max_wavelength,
        axis=config.axis,
        only_advance_position_for_valid_timesteps=(
            config.only_advance_position_for_valid_timesteps
        ),
        positions_in_at_least_fp32=config.positions_in_at_least_fp32,
        positions_name=config.positions_name,
    )
    layer.config = config
    return layer
