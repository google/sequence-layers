# Copyright 2024 Google LLC
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
"""Position embeddings and timing signals."""

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import types
from sequence_layers.jax import utils

__all__ = (
    # go/keep-sorted start
    'AddTimingSignal',
    'ApplyRotaryPositionalEncoding',
    # go/keep-sorted end
)


class AddTimingSignal(
    types.PreservesType, types.PreservesShape, types.SequenceLayer
):
  """Adds sinusoids at varying frequencies to the input channels dimension."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for AddTimingSignal."""

    min_timescale: float = 1.0
    max_timescale: float = 1.0e4
    trainable_scale: bool = False
    # Channel axes over which the timing signal's entries should vary.
    axes: int | tuple[int, ...] | None = None
    sharding: types.Sharding | None = None
    param_dtype: types.DType = jnp.float32
    # If true, only advances position counter for valid timesteps. If false, the
    # position is determined by the physical length of the inputs.
    only_advance_position_for_valid_timesteps: bool = True
    name: str | None = None

    def make(self) -> 'AddTimingSignal':
      return AddTimingSignal(self, name=self.name)

  config: Config

  def setup(self) -> None:
    if self.config.trainable_scale:
      self.scale = self.param(
          'scale',
          utils.shard_initializer(
              nn.initializers.ones_init(),
              self.config.sharding,
          ),
          [],
          self.config.param_dtype,
      )
    else:
      self.scale = None

  @nn.nowrap
  def _check_inputs(self, input_spec: types.ShapeDType):
    if input_spec.dtype not in (
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
        jnp.float64,
    ):
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._check_inputs(input_spec)

    # State holds the current timestep (batched).
    if self.config.only_advance_position_for_valid_timesteps:
      return jnp.full((batch_size, 1), -1, dtype=jnp.int32)
    else:
      return jnp.zeros((batch_size, 1), dtype=jnp.int32)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    self._check_inputs(x.channel_spec)
    time = x.shape[1]
    target_shape = utils.match_shape_along_axes(
        x.channel_shape, axes=self.config.axes
    )
    # Get positions for the batch.
    if self.config.only_advance_position_for_valid_timesteps:
      position = state + jnp.cumsum(x.mask.astype(jnp.int32), axis=1)
      state = position[:, -1:]
    else:
      position = state + jnp.arange(time, dtype=jnp.int32)
      state = state + time

    timing_signal = utils.get_timing_signal_1d_pos(
        position,
        np.prod(target_shape),
        min_timescale=self.config.min_timescale,
        max_timescale=self.config.max_timescale,
        dtype=self.config.param_dtype,
    )
    batch_size = x.shape[0]
    timing_signal = jnp.reshape(
        timing_signal, [batch_size, time] + list(target_shape)
    )
    if self.scale is not None:
      scale: jax.Array = self.scale
      timing_signal *= scale
    x = x.apply_values(lambda v: v + timing_signal.astype(v.dtype))
    return x, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._check_inputs(x.channel_spec)
    target_shape = utils.match_shape_along_axes(
        x.channel_shape, axes=self.config.axes
    )
    if self.config.only_advance_position_for_valid_timesteps:
      position = jnp.maximum(
          0, jnp.cumsum(x.mask.astype(jnp.int32), axis=1) - 1
      )
    else:
      position = jnp.arange(x.shape[1], dtype=jnp.int32)[jnp.newaxis, :]
    timing_signal = utils.get_timing_signal_1d_pos(
        position,
        np.prod(target_shape),
        min_timescale=self.config.min_timescale,
        max_timescale=self.config.max_timescale,
        dtype=self.config.param_dtype,
    )
    timing_signal = jnp.reshape(
        timing_signal, position.shape[:2] + target_shape
    )
    if self.scale is not None:
      scale: jax.Array = self.scale
      timing_signal *= scale
    x = x.apply_values(lambda v: v + timing_signal.astype(v.dtype))
    return x


class ApplyRotaryPositionalEncoding(
    types.PreservesType, types.PreservesShape, types.SequenceLayer
):
  """Applies Rotary Positional Encodings (RoPE) to the sequence.

  See the blogpost https://blog.eleuther.ai/rotary-embeddings/ and the paper
  https://arxiv.org/abs/2104.09864.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for ApplyRotaryPositionalEncoding."""

    max_wavelength: float
    axis: int = -1
    # If true, only advances position counter for valid timesteps. If false, the
    # position is determined by the physical length of the inputs.
    only_advance_position_for_valid_timesteps: bool = True
    # Whether RoPE should be applied with positions in at least float32. This
    # option is for backwards compatibility. True is the recommended value.
    positions_in_at_least_fp32: bool = True
    # If specified, the [batch_size, time] jnp.int32 position used for computing
    # RoPE will be read from the constants dictionary with this name. Otherwise,
    # the physical position in the array is used. If specified,
    # only_advance_position_for_valid_timesteps has no effect.
    positions_name: str | None = None
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'ApplyRotaryPositionalEncoding':
      return ApplyRotaryPositionalEncoding(self, name=self.name)

  config: Config

  def setup(self) -> None:
    if (
        self.config.only_advance_position_for_valid_timesteps
        and self.config.positions_name
    ):
      raise ValueError(
          'only_advance_position_for_valid_timesteps is incompatible with'
          f' {self.config.positions_name=}.'
      )

  @nn.nowrap
  def _check_inputs(self, input_spec: types.ShapeDType):
    if input_spec.dtype not in (
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
        jnp.float64,
    ):
      raise ValueError(
          f'{type(self).__name__} requires floating point argument.'
      )
    input_shape = (None, None) + input_spec.shape
    axis = (
        self.config.axis + len(input_shape)
        if self.config.axis < 0
        else self.config.axis
    )
    if axis <= 1:
      raise ValueError(
          f'{type(self).__name__} axis ({self.config.axis}) must refer to a'
          f' channels dimension ({input_spec=}).'
      )
    axis_size = input_shape[axis]
    if axis_size % 2 != 0:
      raise ValueError(
          f'{type(self).__name__} requires input_shape[{axis}]={axis_size} to'
          ' be even.'
      )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._check_inputs(input_spec)

    # State holds the current timestep (batched). If step_positions_name is
    # specified, the layer does not track position internally.
    if self.config.positions_name:
      return ()
    elif self.config.only_advance_position_for_valid_timesteps:
      return jnp.full((batch_size, 1), -1, dtype=jnp.int32)
    else:
      return jnp.zeros((batch_size, 1), dtype=jnp.int32)

  @nn.nowrap
  def _apply_rope(self, x: jax.Array, positions: jax.Array) -> jax.Array:
    axis = (
        self.config.axis + x.ndim if self.config.axis < 0 else self.config.axis
    )
    assert axis > 1
    channel_ndim = x.ndim - 2
    broadcast_shape = [1] * x.ndim
    axis_dim = x.shape[axis]
    broadcast_shape[axis] = axis_dim // 2
    assert axis_dim % 2 == 0

    freq_exponents = 2.0 * jnp.arange(axis_dim // 2) / axis_dim
    timescale = self.config.max_wavelength**freq_exponents

    @utils.maybe_in_at_least_fp32(
        self.config.positions_in_at_least_fp32,
        restore_dtypes=False,
    )
    def apply_position_embeddings(positions):
      radians = positions.reshape(
          positions.shape + (1,) * channel_ndim
      ) / timescale.reshape(broadcast_shape)
      sin, cos = jnp.sin(radians), jnp.cos(radians)
      x1, x2 = jnp.split(x, 2, axis=axis)
      return jnp.concatenate(
          [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=axis, dtype=x.dtype
      )

    return apply_position_embeddings(positions)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    self._check_inputs(x.channel_spec)
    x_time = x.shape[1]

    # Get positions for the batch.
    if self.config.positions_name:
      positions = utils.get_constant_array(
          self,
          constants,
          self.config.positions_name,
          expected_shape=(x.shape[0], x_time),
          unpack_sequence=True,
          allow_broadcastable=True,
      )
    elif self.config.only_advance_position_for_valid_timesteps:
      positions = state + jnp.cumsum(x.mask.astype(jnp.int32), axis=1)
      state = positions[:, -1:]
    else:
      positions = state + jnp.arange(x_time, dtype=jnp.int32)
      state = state + x_time
    utils.assert_is_compatible_with(positions.shape, [x.shape[0], x_time])

    y = x.apply_values(self._apply_rope, positions)
    return y, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._check_inputs(x.channel_spec)
    if self.config.positions_name:
      positions = utils.get_constant_array(
          self,
          constants,
          self.config.positions_name,
          expected_shape=x.shape[:2],
          unpack_sequence=True,
          allow_broadcastable=True,
      )
    elif self.config.only_advance_position_for_valid_timesteps:
      positions = jnp.maximum(
          0, jnp.cumsum(x.mask.astype(jnp.int32), axis=1) - 1
      )
    else:
      positions = jnp.arange(x.shape[1], dtype=jnp.int32)[jnp.newaxis, :]
    x = x.apply_values(self._apply_rope, positions)
    return x
