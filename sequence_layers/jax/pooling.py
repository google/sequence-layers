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
"""Pooling layers."""

import dataclasses
import fractions
from typing import Any, Sequence as TypingSequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils

__all__ = (
    # go/keep-sorted start
    'AveragePooling1D',
    'AveragePooling2D',
    'MaxPooling1D',
    'MaxPooling2D',
    'MinPooling1D',
    'MinPooling2D',
    # go/keep-sorted end
)


def _max_pool_init_value(dtype: types.DType) -> complex:
  """Returns a dtype-appropriate initial value for max reduction."""
  if issubclass(dtype.type, jnp.floating):
    return -jnp.inf
  elif issubclass(dtype.type, jnp.integer):
    return jnp.iinfo(dtype).min
  elif dtype == jnp.bool_:
    return False
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')


def _min_pool_init_value(dtype: types.DType) -> complex:
  """Returns a dtype-appropriate initial value for min reduction."""
  if issubclass(dtype.type, jnp.floating):
    return jnp.inf
  elif issubclass(dtype.type, jnp.integer):
    return jnp.iinfo(dtype).max
  elif dtype == jnp.bool_:
    return True
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')


def _reduce_window(
    x: jax.Array,
    init_value: complex,
    computation,
    window_dimensions: TypingSequence[int],
    window_strides: TypingSequence[int],
    window_dilation: TypingSequence[int],
    padding: TypingSequence[tuple[int, int]],
) -> jax.Array:
  """Executes a reduce_window with the provided parameters."""

  def pack(value: TypingSequence[Any], default: Any) -> tuple[Any, ...]:
    return (default,) + tuple(value) + (default,) * (x.ndim - len(value) - 1)

  return jax.lax.reduce_window(
      x,
      init_value=np.asarray(init_value, x.dtype),
      computation=computation,
      window_dimensions=pack(window_dimensions, 1),
      window_strides=pack(window_strides, 1),
      padding=pack(padding, (0, 0)),
      base_dilation=(1,) * x.ndim,
      window_dilation=pack(window_dilation, 1),
  )


class BasePooling1D(
    types.PreservesShape, types.PreservesType, types.SequenceLayer
):
  """Shared base logic for 1D pooling layers."""

  def setup(self):
    if self.config.padding == types.PaddingMode.CAUSAL_VALID.value:
      raise ValueError(
          'CAUSAL_VALID is not supported for 2D pooling. Use CAUSAL instead.'
      )

  @property
  def supports_step(self) -> bool:
    return self.config.padding in (
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.SEMICAUSAL.value,
    )

  @property
  def block_size(self) -> int:
    return self.config.strides

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self.config.strides)

  @property
  def input_latency(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self.config.pool_size, self.config.dilation_rate
    )

    match self.config.padding:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        # Causal padding eliminates latency.
        return 0
      case (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL.value
      ):
        # Reverse causal introduces no past padding in layer-wise mode, so we
        # need a full effective_pool_size window to compute the first output
        # layer-wise processing would produce. Since we do not count the current
        # input as part of the latency, the input latency is one smaller than
        # the effective pool size.
        return effective_pool_size - 1
      case _:
        # Unsupported.
        return 0

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return convolution.conv_receptive_field_per_step(
        self.config.pool_size,
        self.config.strides,
        self.config.dilation_rate,
        self.config.padding,
    )

  @property
  def _buffer_width(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self.config.pool_size, self.config.dilation_rate
    )
    match self.config.padding:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_pool_size - self.config.strides, 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (effective_pool_size - 1)
            // self.config.strides
            * self.config.strides
        )
      case _:
        return effective_pool_size - 1

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Special case pool_size 1 since it is stateless.
    if not self._buffer_width:
      return ()

    # When executing a pooling step-by-step, we need a buffer for tracking the
    # current pooling window. This matches the padding added by `layer`.
    return convolution.compute_conv_initial_state(
        batch_size,
        input_spec,
        self._buffer_width,
        self.config.padding,
        self._pad_value(input_spec.dtype),
    ).unmask()

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:

    # Replace masked timesteps with pad value.
    x = x.mask_invalid(self._pad_value(x.dtype))

    if buffer_width := self._buffer_width:
      # Concatenate the new frames with the previous buffer_width frames.
      state = state.concatenate(x)
    else:
      state = x

    # Compute the output for the current timestep.
    values = self._layer(state.values, state.mask, padding=(0, 0))
    mask = convolution.compute_conv_mask(
        state.mask,
        self.config.pool_size,
        self.config.strides,
        self.config.dilation_rate,
        self.config.padding,
        is_step=True,
    )

    # Keep the trailing buffer_width samples for the next step.
    if buffer_width:
      state = state[:, -buffer_width:]
    else:
      state = ()

    # Pooling can leave unmasked values with non-zero values.
    return types.Sequence(values, mask), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ):
    padding = utils.convolution_explicit_padding(
        self.config.padding,
        self.config.pool_size,
        self.config.strides,
        self.config.dilation_rate,
    )
    # Replace masked timesteps with pad value.
    x = x.mask_invalid(self._pad_value(x.dtype))
    values = self._layer(x.values, x.mask, padding)

    mask = convolution.compute_conv_mask(
        x.mask,
        self.config.pool_size,
        self.config.strides,
        self.config.dilation_rate,
        self.config.padding,
        is_step=False,
    )
    return types.Sequence(values, mask)

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    raise NotImplementedError()

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[int, int],
  ) -> jt.Float[jt.ArrayT, 'B T *D']:
    raise NotImplementedError()


class MinPooling1D(BasePooling1D):
  """A 1D min pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for MinPooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'padding', types.validate_padding(self.padding))

    def make(self) -> 'MinPooling1D':
      return MinPooling1D(self, name=self.name)

  config: Config

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _min_pool_init_value(input_dtype)

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[int, int],
  ) -> jt.Float[jt.ArrayT, 'B T *D']:
    return _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=jax.lax.min,
        window_dimensions=(self.config.pool_size,),
        window_dilation=(self.config.dilation_rate,),
        window_strides=(self.config.strides,),
        padding=(padding,),
    )


class MaxPooling1D(BasePooling1D):
  """A 1D max pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for MaxPooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'padding', types.validate_padding(self.padding))

    def make(self) -> 'MaxPooling1D':
      return MaxPooling1D(self, name=self.name)

  config: Config

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _max_pool_init_value(input_dtype)

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[int, int],
  ) -> jt.Float[jt.ArrayT, 'B T *D']:
    del mask
    return _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=jax.lax.max,
        window_dimensions=(self.config.pool_size,),
        window_dilation=(self.config.dilation_rate,),
        window_strides=(self.config.strides,),
        padding=(padding,),
    )


def div_no_nan_grad(x, y):
  """Divides x by y elment-wise, and return 0 value and grad where y is 0."""
  # Apply jnp.where before div to avoid NaN grad, due to inf x / y, when y == 0.
  is_zero = y == 0
  return jnp.where(is_zero, jnp.zeros_like(x), x) / jnp.where(is_zero, 1.0, y)


class AveragePooling1D(BasePooling1D):
  """A 1D average pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for AveragePooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # If true, divide by the number of valid items (i.e., sum of the mask)
    # instead of the pool size.
    masked_average: bool = False
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'padding', types.validate_padding(self.padding))

    def make(self) -> 'AveragePooling1D':
      return AveragePooling1D(self, name=self.name)

  config: Config

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return 0

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[int, int],
  ) -> jt.Float[jt.ArrayT, 'B T *D']:
    y_sum = _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=jax.lax.add,
        window_dimensions=(self.config.pool_size,),
        window_dilation=(self.config.dilation_rate,),
        window_strides=(self.config.strides,),
        padding=(padding,),
    )
    if self.config.masked_average:
      mask_sum = _reduce_window(
          mask.astype(x.dtype),
          init_value=self._pad_value(x.dtype),
          computation=jax.lax.add,
          window_dimensions=(self.config.pool_size,),
          window_dilation=(self.config.dilation_rate,),
          window_strides=(self.config.strides,),
          padding=(padding,),
      )
      mask_sum = jnp.expand_dims(mask_sum, range(2, y_sum.ndim))
      if issubclass(x.dtype.type, jnp.integer):
        y = jnp.floor_divide(y_sum, mask_sum)
      else:
        y = div_no_nan_grad(y_sum, mask_sum)
    elif issubclass(x.dtype.type, jnp.integer):
      # Divide the summed windows by the number of elements per window to get
      # the average. Ignore dilation since the holes in the pool operation do
      # not contribute to the sum.
      y = y_sum // self.config.pool_size
    else:
      # TODO(rryan): Should we divide first to avoid loss of precision due to
      # reduction, or alternatively increase precision of the reduction?
      y = y_sum / self.config.pool_size
    return y


class BasePooling2D(types.PreservesType, types.SequenceLayer):
  """Shared base logic for 2D pooling layers."""

  def setup(self):
    # Configs are normalized in __post_init__.
    assert (
        isinstance(self.config.pool_size, tuple)
        and len(self.config.pool_size) == 2
    )
    assert (
        isinstance(self.config.dilation_rate, tuple)
        and len(self.config.dilation_rate) == 2
    )
    if self.config.time_padding == types.PaddingMode.CAUSAL_VALID.value:
      raise ValueError(
          'CAUSAL_VALID is not supported for 2D pooling. Use CAUSAL instead.'
      )

  @property
  def supports_step(self) -> bool:
    return self.config.time_padding in (
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.SEMICAUSAL.value,
    )

  @property
  def block_size(self) -> int:
    return self.config.strides[0]

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self.config.strides[0])

  @property
  def input_latency(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self.config.pool_size[0], self.config.dilation_rate[0]
    )

    match self.config.time_padding:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        # Causal padding eliminates latency.
        return 0
      case (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL.value
      ):
        # Reverse causal introduces no past padding in layer-wise mode, so we
        # need a full effective_pool_size window to compute the first output
        # layer-wise processing would produce. Since we do not count the current
        # input as part of the latency, the input latency is one smaller than
        # the effective pool size.
        return effective_pool_size - 1
      case _:
        # Unsupported.
        return 0

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    past = -utils.convolution_explicit_padding(
        self.config.time_padding,
        self.config.pool_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
    )[0]
    effective_pool_size = utils.convolution_effective_kernel_size(
        self.config.pool_size[0], self.config.dilation_rate[0]
    )
    future = past + effective_pool_size - 1
    return {0: (past, future)}

  @property
  def _buffer_width(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self.config.pool_size[0], self.config.dilation_rate[0]
    )
    match self.config.time_padding:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_pool_size - self.config.strides[0], 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (effective_pool_size - 1)
            // self.config.strides[0]
            * self.config.strides[0]
        )
      case _:
        return effective_pool_size - 1

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Special case pool_size 1 since it is stateless.
    if not (buffer_width := self._buffer_width):
      return ()

    # When executing a pool step-by-step, we need a buffer for tracking
    # the current pooling window.
    # This matches the causal padding added by `layer`.
    return convolution.compute_conv_initial_state(
        batch_size,
        input_spec,
        buffer_width,
        self.config.time_padding,
        self._pad_value(input_spec.dtype),
    ).unmask()

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    output_shape = list(input_shape)
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape[0],
        self.config.spatial_padding,
        kernel_size=self.config.pool_size[1],
        stride=self.config.strides[1],
        dilation_rate=self.config.dilation_rate[1],
    )
    output_shape[0] = spatial_output_size
    return tuple(output_shape)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    # In step mode, padding is handled by the state that is concatenated below.
    time_padding = (0, 0)
    spatial_padding = utils.convolution_explicit_padding(
        self.config.spatial_padding,
        self.config.pool_size[1],
        self.config.strides[1],
        self.config.dilation_rate[1],
    )

    # Replace masked timesteps with pad value.
    x = x.mask_invalid(self._pad_value(x.dtype))

    if buffer_width := self._buffer_width:
      # Concatenate the new frames with the previous buffer_width frames.
      state = state.concatenate(x)
    else:
      state = x

    # Compute the output for the current timestep.
    values = self._layer(
        state.values, state.mask, padding=(time_padding, spatial_padding)
    )
    mask = convolution.compute_conv_mask(
        state.mask,
        self.config.pool_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
        self.config.time_padding,
        is_step=True,
    )

    # Keep the trailing buffer_width samples for the next step.
    if buffer_width:
      state = state[:, -buffer_width:]
    else:
      state = ()

    # Pooling can leave unmasked values with non-zero values.
    return types.Sequence(values, mask), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ):
    time_padding = utils.convolution_explicit_padding(
        self.config.time_padding,
        self.config.pool_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
    )
    spatial_padding = utils.convolution_explicit_padding(
        self.config.spatial_padding,
        self.config.pool_size[1],
        self.config.strides[1],
        self.config.dilation_rate[1],
    )
    x = x.mask_invalid(mask_value=self._pad_value(x.dtype))
    values = self._layer(
        x.values, x.mask, padding=(time_padding, spatial_padding)
    )

    mask = convolution.compute_conv_mask(
        x.mask,
        self.config.pool_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
        self.config.time_padding,
        is_step=False,
    )
    return types.Sequence(values, mask)

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T H *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> jt.Float[jt.ArrayT, 'B T H *D']:
    raise NotImplementedError()

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    raise NotImplementedError()


class MinPooling2D(BasePooling2D):
  """A 2D min pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for MinPooling2D."""

    pool_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # pool. MinPooling2D is only streamable if time_padding is 'causal_valid'
    # 'reverse_causal_valid', 'causal', or 'reverse_causal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial pooling dimension.
    spatial_padding: types.PaddingModeString | tuple[int, int] = (
        types.PaddingMode.SAME.value
    )
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if isinstance(self.spatial_padding, str):
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_padding(self.spatial_padding),
        )
      else:
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_explicit_padding(self.spatial_padding),
        )
      object.__setattr__(
          self, 'pool_size', utils.normalize_2tuple(self.pool_size)
      )
      object.__setattr__(self, 'strides', utils.normalize_2tuple(self.strides))
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_2tuple(self.dilation_rate)
      )

    def make(self) -> 'MinPooling2D':
      return MinPooling2D(self, name=self.name)

  config: Config

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _min_pool_init_value(input_dtype)

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T H *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> jt.Float[jt.ArrayT, 'B T H *D']:
    del mask
    return _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=jax.lax.min,
        window_dimensions=self.config.pool_size,
        window_dilation=self.config.dilation_rate,
        window_strides=self.config.strides,
        padding=padding,
    )


class MaxPooling2D(BasePooling2D):
  """A 2D max pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for MaxPooling2D."""

    pool_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # pool. MaxPooling2D is only streamable if time_padding is 'causal_valid'
    # 'reverse_causal_valid', 'causal', 'reverse_causal', or 'semicausal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial pooling dimension.
    spatial_padding: types.PaddingModeString | tuple[int, int] = (
        types.PaddingMode.SAME.value
    )
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if isinstance(self.spatial_padding, str):
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_padding(self.spatial_padding),
        )
      else:
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_explicit_padding(self.spatial_padding),
        )
      object.__setattr__(
          self, 'pool_size', utils.normalize_2tuple(self.pool_size)
      )
      object.__setattr__(self, 'strides', utils.normalize_2tuple(self.strides))
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_2tuple(self.dilation_rate)
      )

    def make(self) -> 'MaxPooling2D':
      return MaxPooling2D(self, name=self.name)

  config: Config

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _max_pool_init_value(input_dtype)

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T H *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> jt.Float[jt.ArrayT, 'B T H *D']:
    del mask
    return _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=jax.lax.max,
        window_dimensions=self.config.pool_size,
        window_dilation=self.config.dilation_rate,
        window_strides=self.config.strides,
        padding=padding,
    )


class AveragePooling2D(BasePooling2D):
  """A 2D average pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for AveragePooling2D."""

    pool_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # pool. AveragePooling2D is only streamable if time_padding is
    # 'causal_valid' 'reverse_causal_valid', 'causal', 'reverse_causal', or
    # 'semicausal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial pooling dimension.
    spatial_padding: types.PaddingModeString | tuple[int, int] = (
        types.PaddingMode.SAME.value
    )
    # If true, divide by the number of valid items (i.e., sum of the expanded
    # mask), instead of the overall pool size.
    masked_average: bool = False
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if isinstance(self.spatial_padding, str):
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_padding(self.spatial_padding),
        )
      else:
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_explicit_padding(self.spatial_padding),
        )
      object.__setattr__(
          self, 'pool_size', utils.normalize_2tuple(self.pool_size)
      )
      object.__setattr__(self, 'strides', utils.normalize_2tuple(self.strides))
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_2tuple(self.dilation_rate)
      )

    def make(self) -> 'AveragePooling2D':
      return AveragePooling2D(self, name=self.name)

  config: Config

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return 0

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T H *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> jt.Float[jt.ArrayT, 'B T H *D']:
    y_sum = _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=jax.lax.add,
        window_dimensions=self.config.pool_size,
        window_dilation=self.config.dilation_rate,
        window_strides=self.config.strides,
        padding=padding,
    )

    if self.config.masked_average:
      time_dim_val = lambda v: v if isinstance(v, int) else v[0]
      mask_sum = _reduce_window(
          mask.astype(x.dtype),
          init_value=self._pad_value(x.dtype),
          computation=jax.lax.add,
          window_dimensions=(time_dim_val(self.config.pool_size),),
          window_dilation=(time_dim_val(self.config.dilation_rate),),
          window_strides=(time_dim_val(self.config.strides),),
          padding=padding[:1],
      )
      mask_sum = jnp.expand_dims(mask_sum, range(2, y_sum.ndim))
      mask_sum *= np.prod(self.config.pool_size[1:])
      if issubclass(x.dtype.type, jnp.integer):
        y = jnp.floor_divide(y_sum, mask_sum)
      else:
        y = div_no_nan_grad(y_sum, mask_sum)
    elif issubclass(x.dtype.type, jnp.integer):
      # Divide the summed windows by the number of elements per window to get
      # the average. Ignore dilation since the holes in the pool operation do
      # not contribute to the sum.
      y = y_sum // np.prod(self.config.pool_size)
    else:
      # TODO(rryan): Should we divide first to avoid loss of precision due to
      # reduction, or alternatively increase precision of the reduction?
      y = y_sum / np.prod(self.config.pool_size)
    return y
