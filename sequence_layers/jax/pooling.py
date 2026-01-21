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

import abc
import dataclasses
import fractions
from typing import Any, Callable, Sequence as TypingSequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils
from typing_extensions import override

__all__ = (
    # go/keep-sorted start
    'AveragePooling1D',
    'AveragePooling2D',
    'AveragePooling3D',
    'MaxPooling1D',
    'MaxPooling2D',
    'MaxPooling3D',
    'MinPooling1D',
    'MinPooling2D',
    'MinPooling3D',
    # go/keep-sorted end
)


def div_no_nan_grad(x, y):
  """Divides x by y elment-wise, and return 0 value and grad where y is 0."""
  # Apply jnp.where before div to avoid NaN grad, due to inf x / y, when y == 0.
  is_zero = y == 0
  return jnp.where(is_zero, jnp.zeros_like(x), x) / jnp.where(is_zero, 1.0, y)


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


class BasePooling(
    types.PreservesType, types.SequenceLayer, metaclass=abc.ABCMeta
):
  """Shared base logic for pooling layers."""

  @property
  @abc.abstractmethod
  def _pool_size(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _strides(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _dilation_rate(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _paddings(self) -> tuple[Any, ...]:
    pass

  @property
  @abc.abstractmethod
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    pass

  def setup(self):
    pass

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    raise NotImplementedError()

  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[tuple[int, int], ...],
  ) -> jt.Float[jt.ArrayT, 'B T *D']:
    del mask
    y = _reduce_window(
        x,
        init_value=self._pad_value(x.dtype),
        computation=self._computation,
        window_dimensions=self._pool_size,
        window_strides=self._strides,
        window_dilation=self._dilation_rate,
        padding=padding,
    )
    return y

  @override
  @property
  def supports_step(self) -> bool:
    return self._paddings[0] in (
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.SEMICAUSAL.value,
    )

  @override
  @property
  def block_size(self) -> int:
    return self._strides[0]

  @override
  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._strides[0])

  @override
  @property
  def input_latency(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self._pool_size[0], self._dilation_rate[0]
    )

    match self._paddings[0]:
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

  @override
  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return convolution.conv_receptive_field_per_step(
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._paddings[0],
    )

  @property
  def _buffer_width(self) -> int:
    effective_pool_size = utils.convolution_effective_kernel_size(
        self._pool_size[0], self._dilation_rate[0]
    )
    match self._paddings[0]:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_pool_size - self._strides[0], 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (effective_pool_size - 1) // self._strides[0] * self._strides[0]
      case _:
        return effective_pool_size - 1

  @override
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Special case pool_size 1 in time since it is stateless.
    if not self._buffer_width:
      return ()

    # When executing a pooling step-by-step, we need a buffer for tracking the
    # current pooling window. This matches the padding added by `layer`.
    return convolution.compute_conv_initial_state(
        batch_size,
        input_spec,
        self._buffer_width,
        self._paddings[0],
        self._pad_value(input_spec.dtype),
    ).unmask()

  @override
  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    # In step mode, causal padding is handled by the state that is concatenated
    # below.
    explicit_paddings = ((0, 0),) + tuple(
        utils.convolution_explicit_padding(
            padding, kernel_size, stride, dilation_rate
        )
        for padding, kernel_size, stride, dilation_rate in zip(
            self._paddings[1:],
            self._pool_size[1:],
            self._strides[1:],
            self._dilation_rate[1:],
            strict=True,
        )
    )

    # Replace masked timesteps with pad value.
    x = x.mask_invalid(self._pad_value(x.dtype))

    if buffer_width := self._buffer_width:
      # Concatenate the new frames with the previous buffer_width frames.
      state = state.concatenate(x)
    else:
      state = x

    # Compute the output for the current timestep.
    values = self._layer(state.values, state.mask, padding=explicit_paddings)
    mask = convolution.compute_conv_mask(
        mask=state.mask,
        kernel_size=self._pool_size[0],
        stride=self._strides[0],
        dilation_rate=self._dilation_rate[0],
        padding=self._paddings[0],
        is_step=True,
    )

    # Keep the trailing buffer_width samples for the next step.
    if buffer_width:
      state = state[:, -buffer_width:]
    else:
      state = ()

    # Pooling can leave unmasked values with non-zero values.
    return types.Sequence(values, mask), state

  @override
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ):
    explicit_paddings = tuple(
        utils.convolution_explicit_padding(p, k, s, d)
        for p, k, s, d in zip(
            self._paddings,
            self._pool_size,
            self._strides,
            self._dilation_rate,
            strict=True,
        )
    )

    # Replace masked timesteps with pad value.
    x = x.mask_invalid(self._pad_value(x.dtype))
    values = self._layer(x.values, x.mask, explicit_paddings)
    mask = convolution.compute_conv_mask(
        x.mask,
        self._pool_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._paddings[0],
        is_step=False,
    )
    # Do not preserve the input type even if pool size is 1. Pooling can change
    # the value of masked regions to not be zero.
    return types.Sequence(values, mask)

  @override
  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    output_shape = list(input_shape)
    for i in range(1, len(self._pool_size)):
      spatial_output_size = utils.convolution_padding_output_size(
          input_shape[i - 1],
          self._paddings[i],
          kernel_size=self._pool_size[i],
          stride=self._strides[i],
          dilation_rate=self._dilation_rate[i],
      )
      output_shape[i - 1] = spatial_output_size
    return tuple(output_shape)


class BaseMinPooling(BasePooling):
  """Base class for min pooling layers."""

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _min_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.min


class BaseMaxPooling(BasePooling):
  """Base class for max pooling layers."""

  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return _max_pool_init_value(input_dtype)

  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.max


class BaseAveragePooling(BasePooling):
  """Base class for average pooling layers."""

  @override
  @nn.nowrap
  def _pad_value(self, input_dtype: types.DType) -> complex:
    return 0

  @override
  @property
  def _computation(self) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return jax.lax.add

  @override
  @nn.nowrap
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T *D'],
      mask: jt.Bool[jt.ArrayT, 'B T'],
      padding: tuple[tuple[int, int], ...],
  ) -> jt.Float[jt.ArrayT, 'B T *D']:
    pad_value = self._pad_value(x.dtype)
    y_sum = _reduce_window(
        x,
        init_value=pad_value,
        computation=jax.lax.add,
        window_dimensions=self._pool_size,
        window_dilation=self._dilation_rate,
        window_strides=self._strides,
        padding=padding,
    )
    if self.config.masked_average:
      mask_sum = _reduce_window(
          mask.astype(x.dtype),
          init_value=pad_value,
          computation=jax.lax.add,
          window_dimensions=self._pool_size[:1],
          window_dilation=self._dilation_rate[:1],
          window_strides=self._strides[:1],
          padding=padding[:1],
      )
      mask_sum = jnp.expand_dims(mask_sum, range(2, y_sum.ndim))
      mask_sum *= int(np.prod(self._pool_size[1:]))
      if issubclass(x.dtype.type, jnp.integer):
        y = jnp.floor_divide(y_sum, mask_sum)
      else:
        y = div_no_nan_grad(y_sum, mask_sum)
    elif issubclass(x.dtype.type, jnp.integer):
      # Divide the summed windows by the number of elements per window to get
      # the average. Ignore dilation since the holes in the pool operation do
      # not contribute to the sum.
      y = y_sum // np.prod(self._pool_size)
    else:
      y = y_sum / np.prod(self._pool_size)
    return y


class Pooling1DMixin:
  """Mixin for 1D pooling layers."""

  config: Any

  @override
  @property
  def _pool_size(self) -> tuple[int, ...]:
    return (self.config.pool_size,)

  @override
  @property
  def _strides(self) -> tuple[int, ...]:
    return (self.config.strides,)

  @override
  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return (self.config.dilation_rate,)

  @override
  @property
  def _paddings(self) -> tuple[Any, ...]:
    return (self.config.padding,)


class Pooling2DMixin:
  """Mixin for 2D pooling layers."""

  config: Any

  @override
  @property
  def _pool_size(self) -> tuple[int, ...]:
    return utils.normalize_2tuple(self.config.pool_size)

  @override
  @property
  def _strides(self) -> tuple[int, ...]:
    return utils.normalize_2tuple(self.config.strides)

  @override
  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return utils.normalize_2tuple(self.config.dilation_rate)

  @override
  @property
  def _paddings(self) -> tuple[Any, ...]:
    return (self.config.time_padding, self.config.spatial_padding)


class Pooling3DMixin:
  """Mixin for 3D pooling layers."""

  config: Any

  @override
  @property
  def _pool_size(self) -> tuple[int, ...]:
    return utils.normalize_ntuple(self.config.pool_size, 3)

  @override
  @property
  def _strides(self) -> tuple[int, ...]:
    return utils.normalize_ntuple(self.config.strides, 3)

  @override
  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return utils.normalize_ntuple(self.config.dilation_rate, 3)

  @override
  @property
  def _paddings(self) -> tuple[Any, ...]:
    return (self.config.time_padding, *self.config.spatial_padding)


class MinPooling1D(Pooling1DMixin, BaseMinPooling):
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


class MaxPooling1D(Pooling1DMixin, BaseMaxPooling):
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


class AveragePooling1D(Pooling1DMixin, BaseAveragePooling):
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


class MinPooling2D(Pooling2DMixin, BaseMinPooling):
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
          self, 'pool_size', utils.normalize_ntuple(self.pool_size, 2)
      )
      object.__setattr__(
          self, 'strides', utils.normalize_ntuple(self.strides, 2)
      )
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_ntuple(self.dilation_rate, 2)
      )

    def make(self) -> 'MinPooling2D':  # pytype: disable=invalid-annotation
      return MinPooling2D(self, name=self.name)

  config: Config


class MaxPooling2D(Pooling2DMixin, BaseMaxPooling):
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
          self, 'pool_size', utils.normalize_ntuple(self.pool_size, 2)
      )
      object.__setattr__(
          self, 'strides', utils.normalize_ntuple(self.strides, 2)
      )
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_ntuple(self.dilation_rate, 2)
      )

    def make(self) -> 'MaxPooling2D':  # pytype: disable=invalid-annotation
      return MaxPooling2D(self, name=self.name)

  config: Config


class AveragePooling2D(Pooling2DMixin, BaseAveragePooling):
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
          self, 'pool_size', utils.normalize_ntuple(self.pool_size, 2)
      )
      object.__setattr__(
          self, 'strides', utils.normalize_ntuple(self.strides, 2)
      )
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_ntuple(self.dilation_rate, 2)
      )

    def make(self) -> 'AveragePooling2D':  # pytype: disable=invalid-annotation
      return AveragePooling2D(self, name=self.name)

  config: Config


class MinPooling3D(Pooling3DMixin, BaseMinPooling):
  """A 3D min pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for MinPooling3D."""

    pool_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # pool. MinPooling3D is only streamable if time_padding is 'causal_valid'
    # 'reverse_causal_valid', 'causal', or 'reverse_causal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial pooling dimensions.
    spatial_padding: tuple[
        types.PaddingModeString | tuple[int, int],
        types.PaddingModeString | tuple[int, int],
    ] = (types.PaddingMode.SAME.value, types.PaddingMode.SAME.value)
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if len(self.spatial_padding) != 2:
        raise ValueError(
            f'Expected 2 spatial padding modes got: {self.spatial_padding}'
        )

      object.__setattr__(
          self,
          'spatial_padding',
          tuple(
              types.validate_padding(s) if isinstance(s, str) else s
              for s in self.spatial_padding
          ),
      )
      object.__setattr__(
          self, 'pool_size', utils.normalize_ntuple(self.pool_size, 3)
      )
      object.__setattr__(
          self, 'strides', utils.normalize_ntuple(self.strides, 3)
      )
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_ntuple(self.dilation_rate, 3)
      )

    def make(self) -> 'MinPooling3D':  # pytype: disable=invalid-annotation
      return MinPooling3D(self, name=self.name)

  config: Config


class MaxPooling3D(Pooling3DMixin, BaseMaxPooling):
  """A 3D max pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for MaxPooling3D."""

    pool_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # pool. MaxPooling3D is only streamable if time_padding is 'causal_valid'
    # 'reverse_causal_valid', 'causal', 'reverse_causal', or 'semicausal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial pooling dimensions.
    spatial_padding: tuple[
        types.PaddingModeString | tuple[int, int],
        types.PaddingModeString | tuple[int, int],
    ] = (types.PaddingMode.SAME.value, types.PaddingMode.SAME.value)
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if len(self.spatial_padding) != 2:
        raise ValueError(
            f'Expected 2 spatial padding modes got: {self.spatial_padding}'
        )

      object.__setattr__(
          self,
          'spatial_padding',
          tuple(
              types.validate_padding(s) if isinstance(s, str) else s
              for s in self.spatial_padding
          ),
      )
      object.__setattr__(
          self, 'pool_size', utils.normalize_ntuple(self.pool_size, 3)
      )
      object.__setattr__(
          self, 'strides', utils.normalize_ntuple(self.strides, 3)
      )
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_ntuple(self.dilation_rate, 3)
      )

    def make(self) -> 'MaxPooling3D':  # pytype: disable=invalid-annotation
      return MaxPooling3D(self, name=self.name)

  config: Config


class AveragePooling3D(Pooling3DMixin, BaseAveragePooling):
  """A 3D average pooling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for AveragePooling3D."""

    pool_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # pool. AveragePooling3D is only streamable if time_padding is
    # 'causal_valid' 'reverse_causal_valid', 'causal', 'reverse_causal', or
    # 'semicausal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial pooling dimensions.
    spatial_padding: tuple[
        types.PaddingModeString | tuple[int, int],
        types.PaddingModeString | tuple[int, int],
    ] = (types.PaddingMode.SAME.value, types.PaddingMode.SAME.value)
    # If true, divide by the number of valid items (i.e., sum of the expanded
    # mask), instead of the overall pool size.
    masked_average: bool = False
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if len(self.spatial_padding) != 2:
        raise ValueError(
            f'Expected 2 spatial padding modes got: {self.spatial_padding}'
        )

      object.__setattr__(
          self,
          'spatial_padding',
          tuple(
              types.validate_padding(s) if isinstance(s, str) else s
              for s in self.spatial_padding
          ),
      )
      object.__setattr__(
          self, 'pool_size', utils.normalize_ntuple(self.pool_size, 3)
      )
      object.__setattr__(
          self, 'strides', utils.normalize_ntuple(self.strides, 3)
      )
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_ntuple(self.dilation_rate, 3)
      )

    def make(self) -> 'AveragePooling3D':  # pytype: disable=invalid-annotation
      return AveragePooling3D(self, name=self.name)

  config: Config
