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
"""Simple (generally stateless) layers."""

import abc
from collections.abc import Mapping
import dataclasses
import fractions
import functools
import typing
from typing import Any, Callable, Sequence as TypingSequence

from absl import logging
import einops
import flax.linen as nn
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import types
from sequence_layers.jax import utils
from typing_extensions import override

from google3.learning.gemini.gemax.core.models import sharding as sharding_lib

# pylint: disable=logging-fstring-interpolation

__all__ = (
    # go/keep-sorted start
    'Abs',
    'Add',
    'Affine',
    'ApplySharding',
    'Argmax',
    'Cast',
    'CheckpointName',
    'Downsample1D',
    'Dropout',
    'EinopsRearrange',
    'Elu',
    'Embedding',
    'EmbeddingTranspose',
    'Emit',
    'Exp',
    'ExpandDims',
    'Flatten',
    'GatedLinearUnit',
    'GatedTanhUnit',
    'GatedUnit',
    'Gelu',
    'GlobalReshape',
    'GradientClipping',
    'Identity',
    'Lambda',
    'LeakyRelu',
    'Log',
    'Logging',
    'MaskInvalid',
    'Maximum',
    'Minimum',
    'Mod',
    'MoveAxis',
    'OneHot',
    'OptimizationBarrier',
    'PRelu',
    'Power',
    'Relu',
    'Reshape',
    'Scale',
    'Sigmoid',
    'Slice',
    'Softmax',
    'Softplus',
    'Squeeze',
    'SwapAxes',
    'Swish',
    'Tanh',
    'Transpose',
    'Upsample1D',
    'Upsample2D',
    # go/keep-sorted end
)


def _to_tuple(x: complex | list[Any]) -> complex | tuple[Any, ...]:
  """Replaces lists in a pytree of complex with tuples."""
  if isinstance(x, list):
    return tuple(_to_tuple(i) for i in x)
  else:
    return x


@dataclasses.dataclass(frozen=True)
class HashableArray:
  """Hashable multidimensional array of tuples."""

  data: complex | tuple[Any, ...]
  dtype: np.dtype

  @classmethod
  def from_array(cls, x: np.ndarray) -> 'HashableArray':
    x = np.asarray(x)
    return HashableArray(_to_tuple(x.tolist()), x.dtype)

  def to_array(self) -> np.ndarray:
    return np.asarray(self.data, dtype=self.dtype)


def _to_array(x: complex | np.ndarray | HashableArray) -> np.ndarray:
  if isinstance(x, HashableArray):
    return x.to_array()
  return np.asarray(x)


class StatelessPointwiseBroadcasting(
    types.PreservesType, types.Stateless, metaclass=abc.ABCMeta
):
  """Abstract base class for pointwise operations with broadcasting."""

  @property
  @abc.abstractmethod
  def _broadcast_parameter(self) -> np.ndarray:
    pass

  @nn.nowrap
  def _validate(
      self, parameter: np.ndarray, input_shape: types.ShapeLike
  ) -> None:
    if len(parameter.shape) > len(input_shape):
      raise ValueError(
          f'{self} parameter ({parameter}) has too many dimensions to broadcast'
          f' with the input  channel shape ({input_shape=}).'
      )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    parameter = self._broadcast_parameter
    self._validate(parameter, input_shape)
    return jnp.broadcast_shapes(input_shape, parameter.shape)


class Scale(StatelessPointwiseBroadcasting):
  """Scales the input by a provided constant or array."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Scale."""

    # The value to scale the input by. May be a numpy array, but must be
    # broadcastable to the channels dimension of the input. This value is cast
    # to the dtype of the input, so the type of the input is preserved.
    # PEP 484 says to use complex when you mean any numeric type.
    scale: complex | np.ndarray | HashableArray
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'scale', HashableArray.from_array(self.scale))

    def make(self) -> 'Scale':
      return Scale(self, name=self.name)

  config: Config

  @property
  def _broadcast_parameter(self) -> np.ndarray:
    return _to_array(self.config.scale)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    parameter = self._broadcast_parameter
    self._validate(parameter, x.channel_shape)
    assert isinstance(self.config.scale, HashableArray)
    return x.apply_values_masked(lambda v: v * parameter.astype(v.dtype))


class Affine(types.PreservesType, types.Stateless):
  """Learnable additive bias and multiplicative scale scalars."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config of the learnable scale and bias layer."""

    use_bias: bool = True
    use_scale: bool = True
    shape: types.ShapeLike | None = None
    param_dtype: types.DType = jnp.float32
    scale_init: nn.initializers.Initializer = nn.initializers.ones_init()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    scale_sharding: types.Sharding | None = None
    bias_sharding: types.Sharding | None = None

    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'shape', [] if self.shape is None else self.shape
      )

    def make(self) -> 'Affine':
      return Affine(self, name=self.name)

  def setup(self):
    cfg = self.config
    if cfg.use_scale:
      scale_init = utils.shard_initializer(
          self.config.scale_init, self.config.scale_sharding
      )
      self.scale = self.param(
          'scale', scale_init, self.config.shape, cfg.param_dtype
      )
    if cfg.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      self.bias = self.param(
          'bias', bias_init, self.config.shape, cfg.param_dtype
      )

  config: Config

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants

    # Check that the parameters do not have batch or time dimension.
    if len(input_shape) < len(self.config.shape):
      raise ValueError(
          f'The parameter has too many dimensions (input: {len(input_shape)},'
          f' parameter: {len(self.config.shape)}'
      )

    # This function throws a value error if the shapes are not broadcastable.
    return jnp.broadcast_shapes(input_shape, self.config.shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    # This will validate the shapes by the call to broadcast_shapes.
    # Throws a ValueError when the shapes do not match.
    _ = self.get_output_shape(x.shape[2:])

    if self.config.use_scale:
      x = x.apply_values(lambda v: v * self.scale.astype(v.dtype))
    if self.config.use_bias:
      x = x.apply_values(lambda v: v + self.bias.astype(v.dtype))
    return x


class Add(StatelessPointwiseBroadcasting):
  """Adds the provided constant or array to the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Add."""

    # The value to add to the input. May be a numpy array, but must be
    # broadcastable to the channels dimension of the input. This value is cast
    # to the dtype of the input, so the type of the input is preserved.
    # PEP 484 says to use complex when you mean any numeric type.
    shift: complex | np.ndarray | HashableArray
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'shift', HashableArray.from_array(self.shift))

    def make(self) -> 'Add':
      return Add(self, name=self.name)

  config: Config

  @property
  def _broadcast_parameter(self) -> np.ndarray:
    return _to_array(self.config.shift)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    parameter = self._broadcast_parameter
    self._validate(parameter, x.channel_shape)
    assert isinstance(self.config.shift, HashableArray)
    return x.apply_values(lambda v: v + parameter.astype(v.dtype))


class Maximum(StatelessPointwiseBroadcasting):
  """Clips the input with the provided maximum value."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Maximum."""

    # The value to clip the input with. May be a numpy array, but must be
    # broadcastable to the channels dimension of the input. This value is cast
    # to the dtype of the input, so the type of the input is preserved.
    # PEP 484 says to use complex when you mean any numeric type.
    maximum: complex | np.ndarray | HashableArray
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'maximum', HashableArray.from_array(self.maximum)
      )

    def make(self) -> 'Maximum':
      return Maximum(self, name=self.name)

  config: Config

  @property
  def _broadcast_parameter(self) -> np.ndarray:
    return _to_array(self.config.maximum)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    parameter = self._broadcast_parameter
    self._validate(parameter, x.channel_shape)
    assert isinstance(self.config.maximum, HashableArray)
    return x.apply_values(lambda v: jnp.maximum(v, parameter.astype(v.dtype)))


class Mod(StatelessPointwiseBroadcasting):
  """Returns the remainder of division of the input by the provided divisor."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Mod."""

    # The divisor to compute the remainder with. May be a numpy array, but must
    # be broadcastable to the channels dimension of the input. This value is
    # cast to the dtype of the input, so the type of the input is preserved.
    # PEP 484 says to use complex when you mean any numeric type.
    divisor: complex | np.ndarray | HashableArray
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'divisor', HashableArray.from_array(self.divisor)
      )

    def make(self) -> 'Mod':
      return Mod(self, name=self.name)

  config: Config

  @property
  def _broadcast_parameter(self) -> np.ndarray:
    return _to_array(self.config.divisor)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    parameter = self._broadcast_parameter
    self._validate(parameter, x.channel_shape)
    assert isinstance(self.config.divisor, HashableArray)
    return x.apply_values_masked(
        lambda v: jnp.mod(v, parameter.astype(v.dtype))
    )


class Minimum(StatelessPointwiseBroadcasting):
  """Clips the input with the provided minimum value."""

  @dataclasses.dataclass(frozen=True, unsafe_hash=True)
  class Config(types.SequenceLayerConfig):
    """Config for Minimum."""

    # The value to clip the input with. May be a numpy array, but must be
    # broadcastable to the channels dimension of the input. This value is cast
    # to the dtype of the input, so the type of the input is preserved.
    # PEP 484 says to use complex when you mean any numeric type.
    minimum: complex | np.ndarray | HashableArray
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'minimum', HashableArray.from_array(self.minimum)
      )

    def make(self) -> 'Minimum':
      return Minimum(self, name=self.name)

  config: Config

  @property
  def _broadcast_parameter(self) -> np.ndarray:
    return _to_array(self.config.minimum)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    parameter = self._broadcast_parameter
    self._validate(parameter, x.channel_shape)
    assert isinstance(self.config.minimum, HashableArray)
    return x.apply_values(lambda v: jnp.minimum(v, parameter.astype(v.dtype)))


class Abs(types.StatelessPointwiseFunctor):
  """Absolute value layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Abs':
      return Abs(self, name=self.name)

  config: Config

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.abs(values), mask

  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    # The absolute value of complex numbers is a real magnitude.
    match input_dtype:
      case jnp.complex64:
        return jnp.float32
      case jnp.complex128:
        return jnp.float64
      case _:
        return input_dtype


class Cast(types.StatelessPointwiseFunctor):
  """Cast input values to the specified type."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    dtype: types.DType
    name: str | None = None

    def make(self) -> 'Cast':
      return Cast(self, name=self.name)

  config: Config

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return values.astype(self.config.dtype), mask

  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return self.config.dtype


class GatedUnit(types.PreservesType, types.Stateless):
  """Computes a generalized Gated Unit, reducing the input channels by 2x."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    feature_activation: Callable[[types.ValuesT], types.ValuesT] | None
    gate_activation: Callable[[types.ValuesT], types.ValuesT] | None
    name: str | None = None

    def make(self) -> 'GatedUnit':
      return GatedUnit(self, name=self.name)

  config: Config

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    feature, gate = jnp.split(x.values, 2, axis=-1)
    if self.config.feature_activation:
      feature = self.config.feature_activation(feature)
    if self.config.gate_activation:
      gate = self.config.gate_activation(gate)
    values = feature * gate
    return types.Sequence(values, x.mask)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    channels = input_shape[-1]
    if channels % 2 != 0:
      raise ValueError(
          f'Final dimension of input ({input_shape=}) to {self} must have an'
          ' even number of channels.'
      )
    return tuple(input_shape[:-1]) + (channels // 2,)


class GatedLinearUnit(GatedUnit):
  """Computes a Gated Linear Unit, reducing the input channels by 2x."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'GatedLinearUnit':
      return GatedLinearUnit(
          GatedUnit.Config(None, jax.nn.sigmoid, name=self.name), name=self.name
      )


class GatedTanhUnit(GatedUnit):
  """Computes a Gated Tanh Unit, reducing the input channels by 2x."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'GatedTanhUnit':
      return GatedTanhUnit(
          GatedUnit.Config(jax.nn.tanh, jax.nn.sigmoid, name=self.name),
          name=self.name,
      )


class GradientClipping(types.PreservesType, types.StatelessPointwise):
  """Identity except the gradient is clipped."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    clip_value: float
    name: str | None = None

    def make(self) -> 'GradientClipping':
      assert self.clip_value > 0
      return GradientClipping(self, name=self.name)

  config: Config

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training

    @jax.custom_gradient
    def _clip_gradient(values):
      def _custom_gradient(input_gradients):
        return (
            jnp.clip(
                input_gradients,
                -self.config.clip_value,
                +self.config.clip_value,
            ),
        )

      return values, _custom_gradient

    return x.apply_values_masked(_clip_gradient)


class Identity(types.PreservesType, types.StatelessPointwise):
  """Identity pass-through of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Identity':
      return Identity(name=self.name)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    return x


class ApplySharding(types.PreservesType, types.StatelessPointwise):
  """Applies sharding annotations to the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    values_sharding: types.Sharding | None = None
    mask_sharding: types.Sharding | None = None
    name: str | None = None

    def make(self) -> 'ApplySharding':
      return ApplySharding(self, name=self.name)

  config: Config

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training

    # If only values sharding is provided, use the first two dimensions for the
    # mask sharding.
    values_sharding = self.config.values_sharding
    mask_sharding = self.config.mask_sharding
    if values_sharding and not mask_sharding:
      mask_sharding = values_sharding[:2]

    def shard_values_mask(values, mask):
      values = sharding_lib.shard(values, values_sharding)
      mask = sharding_lib.shard(mask, mask_sharding)
      return values, mask

    return x.apply_masked(shard_values_mask)


class OptimizationBarrier(types.PreservesType, types.StatelessPointwise):
  """Applies an optimization barrier to the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    apply_to_mask: bool = False
    name: str | None = None

    def make(self) -> 'OptimizationBarrier':
      return OptimizationBarrier(self, name=self.name)

  config: Config

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training

    def shard_values_mask(values, mask):
      values = jax.lax.optimization_barrier(values)
      if self.config.apply_to_mask:
        mask = jax.lax.optimization_barrier(mask)
      return values, mask

    return x.apply_masked(shard_values_mask)


class Lambda(types.Stateless):
  """A SequenceLayer that wraps a Python lambda function.

  The wrapped lambda is assumed to be stateless. The receptive field of the
  function in the time dimension should be 1 (i.e. no information crosses
  timesteps). The function may change the shape and dtype of the input.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for a Lambda layer."""

    # If sequence_input is True, a callable that takes an sl.Sequence and
    # returns an sl.Sequence. If sequence_input is False, a callable that takes
    # a jax.Array and returns a jax.Array. The function should be a pure,
    # stateless function of the inputs and its receptive field should be 1.
    fn: (
        Callable[[jax.Array], jax.Array]
        | Callable[[types.Sequence], types.Sequence]
    )
    # If true, the callable accepts and returns sequences.
    sequence_input: bool = False
    # If true, the output of fn is assumed to have potentially changed the
    # masked status of its inputs.
    mask_required: bool = True
    # If get_output_shape or get_output_dtype are called, the input_spec to use
    # for type or shape information (respectively). Prefer to use
    # get_output_spec to avoid having to specify this.
    expected_input_spec: types.ShapeDType | None = None
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'Lambda':
      return Lambda(self, name=self.name)

  config: Config

  @property
  def supports_step(self) -> bool:
    return True

  def _validate_input_spec(self, input_spec: types.ShapeDType) -> None:
    del input_spec
    # TODO(rryan): Re-enable when SoundStream works as expected with this
    # (including the test).
    # if (expected_input_spec := self.config.expected_input_spec) is not None:
    #   if (
    #       expected_input_spec.shape != input_spec.shape
    #       or expected_input_spec.dtype != input_spec.dtype
    #   ):
    #     raise ValueError(
    #         f'Input to Lambda layer ({input_spec=}) does not match expected'
    #         f' input spec {expected_input_spec=}'
    #     )

  def get_output_spec(
      self,
      input_spec: types.ChannelSpec,
      *,
      constants: types.Constants | None = None,
  ) -> types.ChannelSpec:
    self._validate_input_spec(input_spec)
    if self.config.sequence_input:
      input_spec = types.Sequence(
          types.ShapeDType(
              (1, 1) + tuple(input_spec.shape),
              input_spec.dtype,
          ),
          types.ShapeDType((1, 1), jnp.bool_),
      )
    else:
      input_spec = types.ShapeDType(
          (1, 1) + tuple(input_spec.shape),
          input_spec.dtype,
      )
    output_spec = jax.eval_shape(self.config.fn, input_spec)
    return jax.ShapeDtypeStruct(output_spec.shape[2:], output_spec.dtype)

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    # We don't know the input shape.
    if self.config.expected_input_spec is None:
      raise ValueError(
          f'get_output_dtype requires expected_input_spec. {self.config=}'
      )
    return self.get_output_spec(
        jax.ShapeDtypeStruct(
            tuple(self.config.expected_input_spec.shape), input_dtype
        )
    ).dtype

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    # We don't know the input dtype.
    if self.config.expected_input_spec is None:
      raise ValueError(
          f'get_output_shape requires expected_input_spec. {self.config=}'
      )
    return self.get_output_spec(
        jax.ShapeDtypeStruct(
            tuple(input_shape), self.config.expected_input_spec.dtype
        )
    ).shape

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._validate_input_spec(x.channel_spec)
    if self.config.sequence_input:
      fn = typing.cast(
          Callable[[types.Sequence], types.Sequence], self.config.fn
      )
      y = fn(x)
      if y.shape[:2] != x.shape[:2]:
        raise ValueError(
            f'sl.Lambda function ({self.config.fn=}) should not change the'
            f' batch or time shape of the input. fn({x.shape=}) -> {y.shape}'
        )
    else:
      # TODO(b/325346885): Use apply_values/apply_values_masked when the pytype
      # bug is resolved.
      values = self.config.fn(x.values)
      if values.shape[:2] != x.shape[:2]:
        raise ValueError(
            f'sl.Lambda function ({self.config.fn=}) should not change the'
            f' batch or time shape of the input. fn({x.shape=}) ->'
            f' {values.shape=}'
        )
      if self.config.mask_required:
        y = types.Sequence(values, x.mask)
      else:
        y = type(x)(values, x.mask)

    return y


class CheckpointName(types.PreservesType, types.StatelessPointwiseFunctor):
  """Applies a checkpoint name to the sequence values."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    checkpoint_name: str
    name: str | None = None

    def make(self) -> 'CheckpointName':
      return CheckpointName(self, name=self.name)

  config: Config

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    values = jax.ad_checkpoint.checkpoint_name(
        values, self.config.checkpoint_name
    )
    return values, mask


class Tanh(types.PreservesType, types.StatelessPointwiseFunctor):
  """A tanh layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Tanh':
      return Tanh(self, name=self.name)

  config: Config

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.tanh(values), mask


class Relu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Relu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Relu':
      return Relu(name=self.name)

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.relu(values), mask


class LeakyRelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Leaky Relu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    negative_slope: complex = 0.01
    name: str | None = None

    def make(self) -> 'LeakyRelu':
      return LeakyRelu(self, name=self.name)

  config: Config

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.leaky_relu(values, self.config.negative_slope), mask


class PRelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """Parametric Relu, i.e., a Leaky Relu where the negative slope is learnable."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    negative_slope_init: float = 0.01
    param_dtype: types.DType = jnp.float32
    name: str | None = None

    def make(self) -> 'PRelu':
      return PRelu(self, name=self.name)

  config: Config

  def setup(self):
    self.negative_slope = self.param(
        'negative_slope',
        lambda k: jnp.array(
            self.config.negative_slope_init, self.config.param_dtype
        ),
    )

  @property
  def mask_required(self) -> bool:
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:

    return (
        jnp.where(
            values >= 0,
            values,
            self.negative_slope.astype(values.dtype) * values,
        ),
        mask,
    )


class Elu(types.PreservesType, types.StatelessPointwiseFunctor):
  """An elu activation layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    alpha: complex = 1.0
    name: str | None = None

    def make(self) -> 'Elu':
      return Elu(self, name=self.name)

  config: Config

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.elu(values, self.config.alpha), mask


class Exp(types.PreservesType, types.StatelessPointwiseFunctor):
  """An exp layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Exp':
      return Exp(self, name=self.name)

  config: Config

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.exp(values), mask


class Log(types.PreservesType, types.StatelessPointwiseFunctor):
  """A log layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Log':
      return Log(self, name=self.name)

  config: Config

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.log(values), mask


class Power(types.PreservesType, types.StatelessPointwiseFunctor):
  """Raises the input to the specified power."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    power: float = 1.0
    name: str | None = None

    def make(self) -> 'Power':
      return Power(self, name=self.name)

  config: Config

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jnp.power(values, self.config.power), mask


class Sigmoid(types.PreservesType, types.StatelessPointwiseFunctor):
  """A sigmoid layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Sigmoid':
      return Sigmoid(self, name=self.name)

  config: Config

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.sigmoid(values), mask


class Softplus(types.PreservesType, types.StatelessPointwiseFunctor):
  """A softplus layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Softplus':
      return Softplus(self, name=self.name)

  config: Config

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.softplus(values), mask


class Softmax(types.PreservesType, types.StatelessPointwiseFunctor):
  """A softmax layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    axis: int = -1
    name: str | None = None

    def make(self) -> 'Softmax':
      return Softmax(self, name=self.name)

  config: Config

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    axis = self.config.axis
    if (axis if axis >= 0 else values.ndim + axis) < 2:
      raise ValueError(
          'The softmax cannot be applied on the batch or time dimension (got'
          f' {axis=} for shape={values.shape})'
      )
    return jax.nn.softmax(values, axis=axis), mask


class Swish(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Swish layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Swish':
      return Swish(name=self.name)

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.swish(values), mask


class Gelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Gaussian Error Linear Unit (GELU) layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    approximate: bool = True
    name: str | None = None

    def make(self) -> 'Gelu':
      return Gelu(self, name=self.name)

  config: Config

  @property
  def mask_required(self):
    return False

  @nn.nowrap
  def fn(
      self,
      values: types.ValuesT,
      mask: types.MaskT,
  ) -> tuple[types.ValuesT, types.MaskT]:
    return jax.nn.gelu(values, approximate=self.config.approximate), mask


class Slice(types.PreservesType, types.Stateless):
  """Slices the channels dimensions of input tensors."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Slice."""

    # A slice to apply to each channel dimension of the input sequence. Follows
    # numpy strided-slice semantics:
    # - tuple[int | None, int | None, int | None]: An explicit
    #   slice(start, stop, step) for the dimension.
    # - int: An index to slice.
    # - None: Insert a dimension at this position.
    # The number of non-None entries in slices must match the number of channel
    # dimensions.
    slices: TypingSequence[
        tuple[int | None, int | None, int | None] | int | None
    ]
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'slices', tuple(self.slices))

    def as_slices(self) -> tuple[slice | int | None]:
      return tuple(
          slice(*s) if isinstance(s, tuple) else s for s in self.slices
      )

    def make(self) -> 'Slice':
      return Slice(self, name=self.name)

  config: Config

  def _validate_slice_for_input_shape(self, input_shape: types.ShapeLike):
    # None (jnp.newaxis) dimensions are expansions.
    non_none_dimensions = sum(1 for s in self.config.slices if s is not None)
    if non_none_dimensions != len(input_shape):
      raise ValueError(
          'Slice has wrong size for input: input_shape=%s slices=%s'
          % (input_shape, self.config.slices)
      )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    self._validate_slice_for_input_shape(input_shape)
    output_dims = []
    input_index = 0

    # Compute the output shape:
    # - int: Remove the current input dimension.
    # - slice: Compute the output dimension size using slice.indices.
    # - None (tf.newaxis): Add a dimension.
    for slice_i in self.config.slices:
      if isinstance(slice_i, tuple):
        slice_i = slice(*slice_i)
        # Use slice.indices(dim_size) to figure out the new length of the
        # dimension.
        output_dims.append(
            len(range(*slice_i.indices(input_shape[input_index])))
        )
        input_index += 1
      elif isinstance(slice_i, int):
        # Skip this input dimension.
        input_index += 1
      elif slice_i is None:
        # Add a dimension.
        output_dims.append(1)
      else:
        raise NotImplementedError(
            f'Unsupported slice type: {type(slice_i)}, {slice_i}'
        )
    return tuple(output_dims)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Slice the batch and time dimensions with [:, :].
    full_slice = (
        slice(None, None, None),
        slice(None, None, None),
    ) + self.config.as_slices()
    self._validate_slice_for_input_shape(x.channel_shape)
    # Masking status is unchanged by the slice.
    return x.apply_values_masked(lambda v: v.__getitem__(full_slice))


class Flatten(types.PreservesType, types.Stateless):
  """Flattens the channel dimensions of the input sequence.

  An input sequence with shape [batch_size, time, ...] is reshaped to
  [batch_size, time, prod(...)]. The mask is unchanged.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Flatten':
      return Flatten(name=self.name)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    return (np.prod(input_shape),)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    batch_size, time = x.values.shape[:2]
    num_elements = np.prod(x.channel_shape)
    return x.apply_values_masked(jnp.reshape, [batch_size, time, num_elements])


class OneHot(types.Stateless):
  """Computes one-hot vector of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    depth: int
    compute_dtype: types.DType = jnp.float32
    name: str | None = None

    def make(self) -> 'OneHot':
      return OneHot(self, name=self.name)

  config: Config

  @nn.nowrap
  def _validate(self, dtype: types.DType):
    if not jnp.issubdtype(dtype, jnp.integer):
      raise ValueError(
          'Input to OneHot must be an integer or unsigned integer. Got:'
          f' {dtype}'
      )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape) + (self.config.depth,)

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    self._validate(input_dtype)
    return self.config.compute_dtype

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    self._validate(x.dtype)
    # "0" is mapped to "[1] + [0] * (depth - 1)", so mask status is not
    # preserved.
    return x.apply_values(
        lambda v: jax.nn.one_hot(
            v, self.config.depth, dtype=self.config.compute_dtype
        )
    )


class Embedding(types.Stateless):
  """Computes embeddings of integer input codes."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Embedding."""

    # Dimensionality of the embedded values.
    dimension: int
    # The number of embeddings in the embedding table. Inputs are expected to be
    # in the range [0, num_embeddings).
    num_embeddings: int
    # The dtype of the embeddings output by the layer.
    compute_dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # By default, initialize embeddings to have a norm of 1.
    embedding_init: nn.initializers.Initializer = nn.linear.default_embed_init
    # Optional sharding for the embedding table.
    embedding_sharding: types.Sharding | None = None
    # Optional name for the layer.
    name: str | None = None

    def make(self) -> 'Embedding':
      return Embedding(self, name=self.name)

  config: Config

  def setup(self):
    self.embedding = self.param(
        'embedding',
        utils.shard_initializer(
            self.config.embedding_init, self.config.embedding_sharding
        ),
        (self.config.num_embeddings, self.config.dimension),
        self.config.param_dtype,
    )

  @nn.nowrap
  def _validate(self, dtype: types.DType):
    if not jnp.issubdtype(dtype, jnp.integer):
      raise ValueError(
          'Input to Embedding must be an integer or unsigned integer, got:'
          f' {dtype}'
      )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape) + (self.config.dimension,)

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    self._validate(input_dtype)
    if self.config.compute_dtype is None:
      return self.config.param_dtype
    return self.config.compute_dtype

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._validate(x.dtype)
    embedding = self.embedding
    (embedding,) = nn.dtypes.promote_dtype(
        embedding, dtype=self.config.compute_dtype, inexact=False
    )
    return x.apply_values(lambda v: jnp.take(embedding, v, axis=0))

  def attend(
      self,
      x: types.Sequence,
      embedding_dtype: types.DType | None = None,
      compute_dtype: types.DType | None = None,
  ) -> types.Sequence:
    """Attend over the embedding using a query array.

    Args:
      x: Sequence whose last dimension's size is equal to ``dimension`` of the
        embedding.
      embedding_dtype: Optionally override the Embedding parameters' dtype. Used
        by containers like EmbeddingTranspose that call attend(), to achieve
        behavior consistent with their dtype settings.
      compute_dtype: Optionally overrides Embedding.dtype for computation.
        Useful for e.g. computing logits with a upcast dtype.

    Returns:
      A Sequence with final dim ``num_embeddings`` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.

    Raises:
      ValueError: Input query does not have a channel dimension (i.e., 0-dim).
    """
    if not x.channel_shape:
      raise ValueError('Input query must have a channel dimension.')
    values = x.values
    embedding = (
        self.embedding.astype(embedding_dtype)
        if embedding_dtype
        else self.embedding
    )
    values, embedding = nn.dtypes.promote_dtype(
        values, embedding, dtype=(compute_dtype or self.config.compute_dtype)
    )
    return type(x)(jnp.dot(values, embedding.T), x.mask)


class EmbeddingTranspose(types.Stateless):
  """Wraps a bound Embedding layer to be attended upon (e.g. pre-softmax)."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for EmbeddingTranspose."""

    # Embedding layer to allow weight sharing. If not weight sharing, prefer to
    # use Embedding directly instead of wrapping it with EmbeddingTranspose.
    embedding: Embedding

    # Whether to use a bias, with shape imputed from `output_shape`.
    use_bias: bool = True
    # Initializer for the bias.
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # Optional sharding for the bias.
    bias_sharding: types.Sharding = None

    # The dtype for layer compute; if set, overrides embedding's dtype.
    compute_dtype: types.DType | None = None
    # The dtype for layer parameters; if set, overrides embedding's param_dtype.
    param_dtype: types.DType | None = None
    # Optional name for the layer.
    name: str | None = None

    @override
    def make(self) -> 'EmbeddingTranspose':
      return EmbeddingTranspose(self, name=self.name)

  config: Config

  @override
  def setup(self):
    # Bind the Embedding as a submodule. If it is not already parented, this
    # also ensures that Embedding is initialized alongside this layer.
    self.embedding = self.config.embedding

  @override
  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype,
        self.config.param_dtype or self.embedding.config.param_dtype,
        dtype=self.config.compute_dtype or self.embedding.config.compute_dtype,
    )

  @override
  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if (
        not input_shape
        or input_shape[-1] != self.config.embedding.config.dimension
    ):
      raise ValueError(
          "Input query's final channel dimension must be equal to the embedding"
          ' dimension.'
      )
    return (*input_shape[:-1], self.config.embedding.config.num_embeddings)

  @override
  @types.check_layer
  @nn.compact
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training, constants

    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (self.embedding.config.num_embeddings,),
          self.config.param_dtype or self.embedding.config.param_dtype,
      )
    else:
      bias = None

    ret = self.embedding.attend(
        x,
        compute_dtype=self.config.compute_dtype,
        embedding_dtype=self.config.param_dtype,
    )
    if bias is not None:
      ret = ret.apply_values(lambda x: utils.bias_add(x, bias.astype(x.dtype)))

    return ret


class ExpandDims(types.PreservesType, types.Stateless):
  """Applies jnp.expand_dims to the channels dimension of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for ExpandDims."""

    # The axis or axes in the channel shape to expand dims on.
    # For example, axis (0, -2) maps [b, t, c, d] to [b, t, 1, c, 1, d].
    axis: int | TypingSequence[int]
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    def make(self) -> 'ExpandDims':
      return ExpandDims(self, name=self.name)

  config: Config

  @nn.nowrap
  def _normalize_and_validate_axes(
      self, input_shape: types.ShapeLike
  ) -> list[int]:
    rank = len(input_shape)
    axis = self.config.axis
    if isinstance(axis, int):
      axis = [axis]

    # Negative axes refer to one past their normal index.
    dims = [a + rank + 1 if a < 0 else a for a in axis]
    dims = sorted(set(dims))
    for dim in dims:
      if dim < 0 or dim > rank:
        raise ValueError(
            'ExpandDims axes must all refer to channels dimensions. Got:'
            f' {self.config.axis}.'
        )
    return dims

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    dims = self._normalize_and_validate_axes(input_shape)
    output_shape = list(input_shape)
    for a in dims:
      output_shape.insert(a, 1)
    return tuple(output_shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    dims = [2 + d for d in self._normalize_and_validate_axes(x.channel_shape)]
    return x.apply_values_masked(jnp.expand_dims, dims)


class Reshape(types.PreservesType, types.Stateless):
  """Reshapes the channels dimension of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Reshape."""

    # The new shape of the channels dimension. Can't contain -1, and must have
    # the same number of elements as the input channels shape.
    output_shape: TypingSequence[int]
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    def make(self) -> 'Reshape':
      return Reshape(self, name=self.name)

  config: Config

  def _validate_output_shape(self, input_shape: types.ShapeLike) -> None:
    input_elements = np.prod(input_shape)
    output_elements = np.prod(self.config.output_shape)
    if input_elements != output_elements:
      raise ValueError(
          f'Reshape output_shape={self.config.output_shape} must have the same'
          f' number of elements as {input_shape=}.'
      )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    self._validate_output_shape(input_shape)
    return tuple(self.config.output_shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._validate_output_shape(x.channel_shape)
    return x.apply_values_masked(
        lambda v: jnp.reshape(v, v.shape[:2] + self.config.output_shape)
    )


class GlobalReshape(types.PreservesType, types.Stateless):
  """Reshapes the time and channel dimensions of the input sequence globally.

  This layer reshapes inputs of shape [batch_size, time_in, *channel_shape_in]
  to outputs of shape `[batch_size, *output_shape]`.  The total number of
  elements across the time and channel dimensions must remain constant.

  Examples:
  * shape = [4, 3, 6], tensor shape [8, 6, 2, 6], output shape is [8, 4, 3, 6]
  * shape = [12, 6], tensor shape [8, 6, 2, 6], output shape is [8, 12, 6]
  * shape = [4, 18], tensor shape [8, 6, 2, 6], output shape is [8, 4, 18]

  Important notes:
  * A timestep in the output is valid (mask=True) only if *all* values that were
    reshaped into that timestep came from a valid timestep in the input.
  * Because this operation depends on the full time dimension of the input,
    it is neither padding invariant nor streamable (`supports_step` is False).
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for GlobalReshape."""

    # The desired [new_time, *new_channels] shape *after* the batch dimension.
    # Can't contain -1, and the product of elements in output_shape must equal
    # the product of the input time and channel dimensions.
    output_shape: TypingSequence[int]
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    def make(self) -> 'GlobalReshape':
      return GlobalReshape(self, name=self.name)

  config: Config

  def _validate_reshape(self, input_shape: types.ShapeLike) -> None:
    input_elements = np.prod(input_shape)
    output_elements = np.prod(self.config.output_shape)
    if input_elements != output_elements:
      raise ValueError(
          f'{self.config.output_shape=} must have the same number of '
          f'elements as {input_shape=}.'
      )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    return tuple(self.config.output_shape[1:])

  @property
  def supports_step(self) -> bool:
    return False

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:

    self._validate_reshape(x.shape[1:])

    # Perform reshape across time and channels.
    out_shape = (x.shape[0],) + tuple(self.config.output_shape)
    out = jnp.reshape(x.values, out_shape)
    # Since output_time can be {=, >, <} input_time, we may be splitting or
    # joining input timesteps into the reshaped output. This means we need to
    # carefully calculate the output mask:
    # 1. Broadcast the mask to the full input shape, and then reshape it to the
    # output shape.
    mask = jnp.reshape(x.mask, x.shape[:2] + (1,) * len(x.channel_shape))
    mask = jnp.broadcast_to(mask, x.shape)
    # 2. Reshape the mask to the output shape, and then call jnp.all to ensure
    # that any timestep with *any* invalid input is invalid in the output).
    mask = jnp.reshape(mask, out_shape)
    mask = jnp.all(mask, axis=range(2, mask.ndim))
    return types.Sequence(out, mask=mask)


class Transpose(types.PreservesType, types.Stateless):
  """Transposes (i.e., permutes) the channels dimension of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Transpose.

    The usage is the same as that of jax.numpy.transpose.

    Attributes:
      axes: This parameters allows to specify the desired order of the axis. It
        must be a permutation of range(2, ndim), where ndim is the number of
        dimensions of the input array. The first axis that can be permuted is 2
        because the batch and time dimensions are excluded. When axes is not
        specified (i.e., is None), then the order of the channel axes are
        reversed. E.g., an input with shape (batch, time, ch1, ch2, ch3) will
        results in an output of shape (batch, time, ch3, ch2, ch1).
    """

    # The axis can only be between 2 and ndim - 1.
    # We do not allow to permute batch and time.
    axes: TypingSequence[int] | None = None
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      if self.axes is not None:
        object.__setattr__(self, 'axes', tuple(self.axes))

    def make(self) -> 'Transpose':

      if self.axes is not None and (0 in self.axes or 1 in self.axes):
        raise ValueError("Can't transpose batch or time dimension.")

      return Transpose(self, name=self.name)

  config: Config

  def _validate_axes(self, input_shape: types.ShapeLike) -> tuple[int, ...]:
    axes = self.config.axes
    input_axes = tuple(range(2, 2 + len(input_shape)))

    if axes is None:
      return input_axes[::-1]

    sorted_axes = tuple(sorted(axes))
    if sorted_axes != input_axes:
      raise ValueError(
          f'The provided axes {sorted_axes} does not match those'
          f' of the input {input_axes}.'
      )

    return tuple(axes)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    axes = self._validate_axes(input_shape)
    return tuple(input_shape[a - 2] for a in axes)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    axes = self._validate_axes(x.channel_shape)
    return x.apply_values_masked(lambda v: jnp.transpose(v, (0, 1) + axes))


class SwapAxes(Transpose):
  """Swap two channel axes."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    axis1: int
    axis2: int
    name: str | None = None

    def make(self) -> 'SwapAxes':

      axes = [self.axis1, self.axis2]
      if 0 in axes or 1 in axes:
        raise ValueError("Can't swap batch or time dimension.")

      return SwapAxes(
          Transpose.Config(axes=axes, name=self.name), name=self.name
      )

  @override
  def _validate_axes(self, input_shape: types.ShapeLike) -> tuple[int, ...]:
    ndim = 2 + len(input_shape)  # ndim including batch and time.
    axes = [a if a >= 0 else ndim + a for a in self.config.axes]
    if 0 in axes or 1 in axes:
      raise ValueError("Can't move batch or time dimension.")
    axis1, axis2 = axes
    if axis1 < 1 or axis1 >= ndim or axis2 < 1 or axis2 >= ndim:
      raise ValueError(
          f'Out of bound axis (got {axis1=} {axis2=} for {ndim=}).'
      )
    axes = list(range(2 + len(input_shape)))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return tuple(axes[2:])


class MoveAxis(Transpose):
  """Moves one or several channel axes to new locations."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config of MoveAxis layer."""

    source: int | TypingSequence[int]
    destination: int | TypingSequence[int]
    name: str | None = None

    def __post_init__(self):
      to_tuple = lambda x: (x,) if isinstance(x, int) else x
      object.__setattr__(self, 'source', to_tuple(self.source))
      object.__setattr__(self, 'destination', to_tuple(self.destination))

    def make(self) -> 'MoveAxis':

      if (
          0 in self.source
          or 1 in self.source
          or 0 in self.destination
          or 1 in self.destination
      ):
        raise ValueError("Can't move batch or time dimension.")

      if len(self.source) != len(self.destination):
        raise ValueError(
            f'Inconsistent number of elements: {len(self.source)} vs'
            f' {len(self.destination)}'
        )

      return MoveAxis(self, name=self.name)

  config: Config

  @override
  def _validate_axes(self, input_shape: types.ShapeLike) -> tuple[int, ...]:
    ndim = 2 + len(input_shape)  # ndim including batch and time.

    canonicalize_axes = lambda axes: [a if a >= 0 else ndim + a for a in axes]
    source = canonicalize_axes(self.config.source)
    destination = canonicalize_axes(self.config.destination)

    if 0 in source or 1 in source or 0 in destination or 1 in destination:
      raise ValueError("Can't move batch or time dimension.")

    perm = [i for i in range(ndim) if i not in source]
    for dest, src in sorted(zip(destination, source, strict=True)):
      perm.insert(dest, src)

    return tuple(perm[2:])


class Emit(types.PreservesType, types.PreservesShape, types.StatelessEmitting):
  """An identity layer that emits its input."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Emit':
      return Emit(self, name=self.name)

  config: Config

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Sequence]:
    # To match the emit spec above, we need to unmask x if it is masked.
    # TODO(rryan): Remove the type distinction between masked and unmasked
    # sequences.
    return x, x.unmask()


class Dropout(types.PreservesType, types.StatelessPointwise):
  """Computes dropout using Flax RNGs."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    rate: float = 0.0
    broadcast_dims: TypingSequence[int] = ()
    rng_collection: str = 'dropout'
    name: str | None = None

    def make(self) -> 'Dropout':
      return Dropout(self, name=self.name)

  config: Config

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if training:
      # TODO(rryan): Layer/step-wise equivalence is not yet possible. Requires
      # deterministic dropout.
      raise ValueError('Step-wise training is not supported for Dropout yet.')
    return self.layer(x, training=training, constants=constants), state

  def apply_dropout(self, x: jax.Array, training: bool) -> jax.Array:
    """Applies dropout to array x."""
    if (self.config.rate == 0.0) or not training:
      return x

    # Prevent gradient NaNs in 1.0 edge-case.
    if self.config.rate == 1.0:
      return jnp.zeros_like(x)

    keep_prob = 1.0 - self.config.rate
    rng = self.make_rng(self.config.rng_collection)
    broadcast_shape = list(x.shape)
    for dim in self.config.broadcast_dims:
      broadcast_shape[dim] = 1

    input_is_floating = utils.is_floating(x.dtype)

    # If the input is not floating point, we use jnp.where to apply the dropout
    # mask. Otherwise, we use multiplication by 1 or 0 to apply dropout. In
    # practice, multiplication leads to faster compiled code because it is
    # easier to fuse with other operations.
    use_select = not input_is_floating

    if input_is_floating:
      mask_dtype = x.dtype
    else:
      mask_dtype = jnp.float32

    dropout_mask = jax.random.uniform(
        rng, shape=broadcast_shape, dtype=mask_dtype
    )

    if use_select:
      dropout_mask = dropout_mask < keep_prob
      # Only scale the input if floating. This is a hack to support integer /
      # boolean dropout.
      if input_is_floating:
        x /= keep_prob
    else:
      # If not using select, scale the dropout mask.
      dropout_mask = jnp.floor(keep_prob + dropout_mask) / keep_prob

    dropout_mask = jnp.broadcast_to(dropout_mask, x.shape)

    if use_select:
      x = jax.lax.select(dropout_mask, x, jnp.zeros_like(x))
    else:
      x *= dropout_mask
    return x

  @nn.compact
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # No mask status change since dropout only zeros values.
    return x.apply_values_masked(self.apply_dropout, training=training)


class Downsample1D(types.PreservesType, types.PreservesShape, types.Stateless):
  """A 1D downsampling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Downsample1D."""

    rate: int
    name: str | None = None

    def make(self) -> 'Downsample1D':
      return Downsample1D(self, name=self.name)

  config: Config

  @property
  def block_size(self) -> int:
    return self.config.rate

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self.config.rate)

  @property
  def input_latency(self) -> int:
    return self.config.rate - 1

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Downsampling does not change the masked state, so use the type of x to
    # repack the downsampled values and mask.
    return type(x)(
        x.values[:, :: self.config.rate], x.mask[:, :: self.config.rate]
    )


class Upsample1D(types.PreservesType, types.PreservesShape, types.Stateless):
  """A 1D upsampling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Upsample1D."""

    rate: int
    name: str | None = None

    def make(self) -> 'Upsample1D':
      return Upsample1D(self, name=self.name)

  config: Config

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self.config.rate)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Upsampling does not change the masked state, so use the type of x to
    # repack the upsampled values and mask.
    return type(x)(
        jnp.repeat(x.values, self.config.rate, axis=1),
        jnp.repeat(x.mask, self.config.rate, axis=1),
    )


class Upsample2D(types.PreservesType, types.Stateless):
  """A 2D upsampling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Upsample2D."""

    rate: int | TypingSequence[int]
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'rate', utils.normalize_2tuple(self.rate))

    def make(self) -> 'Upsample2D':
      return Upsample2D(self, name=self.name)

  config: Config

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self.config.rate[0])

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 2:
      raise ValueError(
          'Upsample2D requires rank 4 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )

    return (
        input_shape[0] * self.config.rate[1],
        input_shape[1],
    )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    values = jnp.repeat(x.values, self.config.rate[0], axis=1)
    values = jnp.repeat(values, self.config.rate[1], axis=2)
    mask = jnp.repeat(x.mask, self.config.rate[0], axis=1)

    # Upsampling does not change the masked state, so use the type of x to
    # repack the upsampled values and mask.
    return type(x)(values, mask)


class MaskInvalid(types.PreservesType, types.StatelessPointwise):
  """Masks the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'MaskInvalid':
      return MaskInvalid(self, name=self.name)

  config: Config

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    return x.mask_invalid()


class Logging(types.PreservesType, types.StatelessPointwise):
  """Layer that logs input arguments to get_initial_state, step, and layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for the Logging layer."""

    prefix: str = ''
    dump_tensors: bool = False
    _: dataclasses.KW_ONLY
    layer_format_str: str = (
        '{prefix} layer():\n'
        '\tx={x}\n\ttraining={training}\n\tconstants={constants}'
    )
    get_initial_state_format_str: str = (
        '{prefix} get_initial_state():\n'
        '\tbatch_size={batch_size}\n\tinput_spec={input_spec}\n\t'
        'training={training}\n\tconstants={constants}'
    )
    step_format_str: str = (
        '{prefix} step():\n'
        '\tx={x}\n\tstate={state}\n'
        '\ttraining={training}\n\tconstants={constants}'
    )

    def make(self) -> 'Logging':
      return Logging(self)

  config: Config

  @nn.nowrap
  def _register_callback(self, format_str: str, **kwargs) -> None:
    if self.config.dump_tensors:
      # Non-JAX types are not accepted by jax.debug.callback, so we separate
      # them out.
      nonjax_kwargs = {'prefix': self.config.prefix}
      for k, v in list(kwargs.items()):
        try:
          jax.interpreters.xla.canonicalize_dtype(v)
        except:  # pylint: disable=bare-except
          nonjax_kwargs[k] = v
          del kwargs[k]
      # We then set up a callback for the remaining tensor values:
      jax.debug.callback(
          lambda *cb_args, **cb_kwargs: logging.info(
              format_str.format(*cb_args, **nonjax_kwargs, **cb_kwargs)
          ),
          **kwargs,
      )
    else:

      def arrays_to_specs(leaf: Any) -> types.ShapeDType | str:
        if isinstance(leaf, jax.Array) or isinstance(leaf, np.ndarray):
          return types.ShapeDType(leaf.shape, leaf.dtype)
        # `types.Sequence`s repr() as their specs; we can return them directly.
        return leaf

      kwargs = jax.tree.map(arrays_to_specs, kwargs)
      logging.info(format_str.format(prefix=self.config.prefix, **kwargs))

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._register_callback(
        self.config.layer_format_str,
        x=x,
        training=training,
        constants=constants,
    )
    return x

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._register_callback(
        self.config.get_initial_state_format_str,
        batch_size=batch_size,
        input_spec=input_spec,
        training=training,
        constants=constants,
    )
    return super().get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    self._register_callback(
        self.config.step_format_str,
        x=x,
        state=state,
        training=training,
        constants=constants,
    )
    return x, state


class Argmax(types.Stateless):
  """An Argmax layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Argmax':
      return Argmax(self, name=self.name)

  config: Config

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return x.apply_values(jnp.argmax, axis=-1)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape[:-1])

  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return jnp.int32


class EinopsRearrange(types.PreservesType, types.Stateless):
  """A wrapper for einops.rearrange applied to the channel dimensions."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config of EinopsRearrange."""

    # Rearrangement pattern exluding the batch and time dimensions.
    pattern: str
    # A dictionary of additional specifications for dimensions.
    axes_lengths: Mapping[str, int] | None = None
    # Optional name for the module.
    name: str | None = None

    def __post_init__(self):
      if '->' not in self.pattern:
        raise ValueError(
            f'The input pattern is not valid (got {self.pattern}).'
        )

      # Find all unique labels.
      labels = set(self.pattern.replace('(', ' ').replace(')', ' ').split(' '))
      if 'batch' in labels or 'time' in labels:
        raise ValueError(
            f'`batch` and `time` are reserved axes labels (got {self.pattern}).'
        )

    def make(self) -> 'EinopsRearrange':
      return EinopsRearrange(self, name=self.name)

  config: Config

  def _get_rearrange_fn(self) -> Callable[[jax.Array], jax.Array]:
    before, after = self.config.pattern.split('->')
    pattern = f'batch time {before} -> batch time {after}'
    axes_lengths = self.config.axes_lengths if self.config.axes_lengths else {}
    return functools.partial(einops.rearrange, pattern=pattern, **axes_lengths)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training, constants
    rearrange_fn = self._get_rearrange_fn()
    return x.apply_values(rearrange_fn)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    rearrange_fn = self._get_rearrange_fn()
    output = jax.eval_shape(rearrange_fn, jnp.zeros((1, 1) + input_shape))
    return tuple(output.shape[2:])


class Squeeze(types.PreservesType, types.Stateless):
  """This layer squeezes all the depth dimensions of the input.

  I.e. [batch_size, time, *depth_dims -> [batch_size, time] (where all the
  `depth_dims` need to be 1).
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config of Squeeze."""

    axis: int | TypingSequence[int] | None = None

    name: str | None = None

    def make(self) -> 'Squeeze':

      axis = self.axis
      if isinstance(axis, int):
        if axis in (0, 1):
          raise ValueError('Batch and time (axis=0 or 1) cannot be squeezed.')
      elif axis is not None and (0 in axis or 1 in axis):
        raise ValueError('Batch and time (axis=0 or 1) cannot be squeezed.')

      return Squeeze(self, name=self.name)

  config: Config

  def _validate_axis(self, input_shape: types.ShapeLike) -> tuple[int, ...]:
    axis = self.config.axis

    if isinstance(axis, int):
      return (axis,)

    if axis is None:
      return tuple(a + 2 for a, n in enumerate(input_shape) if n == 1)

    return tuple(axis)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    axis = self._validate_axis(input_shape)
    return jax.eval_shape(
        lambda v: jnp.squeeze(v, axis=axis),
        types.ShapeDType((0, 1) + tuple(input_shape), jnp.float32),
    ).shape[2:]

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    axis = self._validate_axis(x.channel_shape)
    return x.apply_values_masked(jnp.squeeze, axis)
