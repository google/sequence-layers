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
"""Dense layers."""

import dataclasses
import typing
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from sequence_layers.jax import types
from sequence_layers.jax import utils

from google3.learning.gemini.gemax.core.models import meta


__all__ = (
    # go/keep-sorted start
    'Dense',
    'DenseShaped',
    'EinsumDense',
    # go/keep-sorted end
)


class Dense(types.Stateless, utils.EinsumCommon):
  """A basic dense layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Dense config."""
    # The number of output features for the dense layer.
    features: int
    # Whether to use a bias.
    use_bias: bool = True
    # An optional activation to apply after the dense layer.
    activation: Callable[[jax.Array], jax.Array] | None = None
    # The dtype to use for layer compute.
    dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # An optional precision to use for the einsum.
    precision: nn.linear.PrecisionLike = None
    # Initializer for the kernel.
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    # Optional sharding for the kernel. Any axes that are present in the input
    # spec are marked as FANIN.
    kernel_sharding: types.Sharding | None = None
    # Initializer for the bias, if used and not gated by another config option.
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # Optional sharding for the bias.
    bias_sharding: types.Sharding | None = None
    # Optional callable that returns a jnp.einsum-compatible function to use
    # instead of jnp.einsum. For example, to enable quantization aware training.
    einsum_factory: types.EinsumFactoryT | None = None
    # Optional name for the layer.
    name: str | None = None

    def make(self) -> 'Dense':
      return Dense(self, name=self.name)

  config: Config

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.dtype
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    if not input_shape:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.config.features,)

  @nn.compact
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del constants

    if x.ndim < 3:
      raise ValueError(f'Dense requires at least rank 3 input. Got: {x.shape=}')

    # Preserve masked state if no bias or activation are in use.
    apply_fn = (
        x.apply_values
        if self.config.use_bias or self.config.activation is not None
        else x.apply_values_masked
    )

    return apply_fn(
        self.einsum,
        '...a,ab->...b',
        (x.shape[-1], self.config.features),
        bias_shape=(self.config.features,) if self.config.use_bias else None,
        activation=self.config.activation,
        dtype=self.config.dtype,
        param_dtype=self.config.param_dtype,
        precision=self.config.precision,
        kernel_init=self.config.kernel_init,
        kernel_sharding=self.config.kernel_sharding,
        bias_init=self.config.bias_init,
        bias_sharding=self.config.bias_sharding,
        projectable=True,
        axes_types=(meta.AxisType.FANIN, None),
        einsum_factory=self.config.einsum_factory,
    )


class DenseShaped(types.Stateless, utils.EinsumCommon):
  """A dense layer that transforms the channel shape."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """DenseShaped config."""

    # The expected shape of the output tensor (excluding the batch and time
    # dimension).
    output_shape: types.ShapeLike
    # Whether to use a bias, with shape imputed from `output_shape`.
    use_bias: bool = True
    # An optional activation to apply after the dense layer.
    activation: Callable[[jax.Array], jax.Array] | None = None
    # The dtype to use for layer compute.
    dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # An optional precision to use for the einsum.
    precision: nn.linear.PrecisionLike = None
    # Initializer for the kernel.
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    # Optional sharding for the kernel. Any axes that are present in the input
    # spec are marked as FANIN.
    kernel_sharding: types.Sharding | None = None
    # Initializer for the bias, if used and not gated by another config option.
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # Optional sharding for the bias.
    bias_sharding: types.Sharding | None = None
    # Optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    def make(self) -> 'DenseShaped':
      return DenseShaped(self, name=self.name)

  config: Config

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.dtype
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    return tuple(self.config.output_shape)

  @nn.compact
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del constants

    input_channel_shape = x.channel_shape
    output_channel_shape = tuple(self.config.output_shape)

    input_einsum_dims = ''.join(
        [chr(ord('a') + i) for i in range(len(input_channel_shape))]
    )
    output_einsum_dims = ''.join([
        chr(ord('a') + i + len(input_channel_shape))
        for i in range(len(output_channel_shape))
    ])
    input_kernel_shape = input_channel_shape
    input_weight_dims = input_einsum_dims
    if not input_weight_dims:
      input_kernel_shape = (1,)
      input_weight_dims = 'I'
    output_kernel_shape = output_channel_shape
    output_weight_dims = output_einsum_dims
    if not output_weight_dims:
      output_kernel_shape = (1,)
      output_weight_dims = 'O'

    equation = f'BT{input_einsum_dims},{input_weight_dims}{output_weight_dims}->BT{output_einsum_dims}'
    kernel_shape = input_kernel_shape + output_kernel_shape
    axes_types = (meta.AxisType.FANIN,) * len(input_kernel_shape) + (
        None,
    ) * len(output_kernel_shape)

    # Preserve masked state if no bias or activation are in use.
    apply_fn = (
        x.apply_values
        if self.config.use_bias or self.config.activation is not None
        else x.apply_values_masked
    )

    return apply_fn(
        self.einsum,
        equation,
        kernel_shape,
        bias_shape=output_kernel_shape if self.config.use_bias else None,
        activation=self.config.activation,
        dtype=self.config.dtype,
        param_dtype=self.config.param_dtype,
        precision=self.config.precision,
        kernel_init=self.config.kernel_init,
        kernel_sharding=self.config.kernel_sharding,
        bias_init=self.config.bias_init,
        bias_sharding=self.config.bias_sharding,
        projectable=True,
        axes_types=axes_types,
    )


class EinsumDense(types.Stateless, utils.EinsumCommon):
  """A dense layer that transforms the channel shape with an einsum equation.

  Equation input and output specs must have leading ellipses to broadcast over
  the batch and time dimension.

  Example:

  Input sequence: [b, t, c1, c2, c3]
  - equation = '...abc,bd->...bd'
  - output_shape = [None, c4]
  - bias_axes = 'd'
  Output sequence: [b, t, c2, c4]

  Kernel shape: [c2, c4]
  Bias shape: [c4]

  Interpretation: Every [c1, c2, c3] tensor per timestep is transformed with the
  einsum formula abc,bd->bd.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """EinsumDense config."""

    # An equation describing the einsum to perform. This equation must be a
    # valid einsum string of the form `...ab,bc->...ac`, or where 'ab', 'bc',
    # and 'ac' can be any valid einsum axis expression sequence.
    equation: str
    # The expected shape of the output tensor (excluding the batch and time
    # dimension). You can specify None for any dimension that should be inferred
    # from the input shape.
    output_shape: list[int | None] | tuple[int | None, ...]
    # A string containing the output dimension(s) to apply a bias to. Each
    # character in the `bias_axes` string should correspond to a character in
    # the output portion of the `equation` string.
    bias_axes: str = ''

    # An optional activation to apply after the dense layer.
    activation: Callable[[jax.Array], jax.Array] | None = None
    # The dtype to use for layer compute.
    dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # An optional precision to use for the einsum.
    precision: nn.linear.PrecisionLike = None
    # Initializer for the kernel.
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    # Optional sharding for the kernel. Any axes that are present in the input
    # spec are marked as FANIN.
    kernel_sharding: types.Sharding | None = None
    # Initializer for the bias, if used and not gated by another config option.
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # Optional sharding for the bias.
    bias_sharding: types.Sharding | None = None
    # Optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    def make(self) -> 'EinsumDense':
      return EinsumDense(self, name=self.name)

  config: Config

  @nn.nowrap
  def _parse_and_validate_equation(self, equation) -> tuple[str, str, str]:
    if '->' not in equation:
      raise ValueError(f'equation is not valid for EinsumDense: {equation}')
    left, output_spec = equation.split('->')
    input_spec, kernel_spec = left.split(',')
    if not input_spec.startswith('...') or not output_spec.startswith('...'):
      raise ValueError('Equation must be of the form "...X,Y->...Z".')
    if 3 + len(set(input_spec[3:])) != len(input_spec):
      raise ValueError(
          f'Equation {input_spec=} must not contain duplicate variables.'
      )

    if 3 + len(set(output_spec[3:])) != len(output_spec):
      raise ValueError(
          f'Equation {output_spec=} must not contain duplicate variables.'
      )
    return input_spec, kernel_spec, output_spec

  @nn.nowrap
  def _get_and_validate_output_shape(
      self,
      input_shape: types.ShapeLike,
  ) -> types.Shape:
    input_spec, _, output_spec = self._parse_and_validate_equation(
        self.config.equation
    )
    assert input_spec.startswith('...')
    assert output_spec.startswith('...')
    # Trim '...' off.
    input_spec, output_spec = input_spec[3:], output_spec[3:]

    if len(input_spec) != len(input_shape):
      raise ValueError(
          f'Equation {input_spec=} does not match {input_shape=} rank.'
      )

    input_dims = {d: input_shape[i] for i, d in enumerate(input_spec)}
    assert len(input_dims) == len(input_spec)

    output_shape = list(self.config.output_shape)
    if len(output_spec) != len(output_shape):
      raise ValueError(
          f'Equation {output_spec=} does not match {output_shape=}.'
      )

    for i, d in enumerate(output_spec):
      if output_shape[i] is None:
        output_shape[i] = input_dims[d]
      elif d in input_dims and output_shape[i] != input_dims[d]:
        raise ValueError(
            'Input shape and output shape inconsistent for dimension '
            f'{d=}. {output_shape=} {input_shape=}'
        )
    return typing.cast(types.Shape, tuple(output_shape))

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.dtype
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    return self._get_and_validate_output_shape(input_shape)

  @nn.compact
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del constants
    input_spec, kernel_spec, output_spec = self._parse_and_validate_equation(
        self.config.equation
    )
    assert input_spec.startswith('...')
    assert output_spec.startswith('...')
    # Trim '...' off.
    input_spec, output_spec = input_spec[3:], output_spec[3:]

    input_shape = x.shape
    output_shape = self._get_and_validate_output_shape(x.channel_shape)

    kernel_shape, bias_shape, _ = utils.einsum_analyze_split_string(
        (input_spec, kernel_spec, output_spec),
        self.config.bias_axes,
        input_shape,
        output_shape,
        left_elided=True,
    )

    axes_types = []
    input_only_dims = {dim for dim in input_spec if dim not in set(output_spec)}
    for dim in kernel_spec:
      if dim in input_only_dims:
        axes_types.append(meta.AxisType.FANIN)
      else:
        axes_types.append(None)

    # Preserve masked state if no bias or activation are in use.
    apply_fn = (
        x.apply_values
        if self.config.bias_axes or self.config.activation is not None
        else x.apply_values_masked
    )

    return apply_fn(
        self.einsum,
        self.config.equation,
        kernel_shape,
        bias_shape,
        activation=self.config.activation,
        dtype=self.config.dtype,
        param_dtype=self.config.param_dtype,
        precision=self.config.precision,
        kernel_init=self.config.kernel_init,
        kernel_sharding=self.config.kernel_sharding,
        bias_init=self.config.bias_init,
        bias_sharding=self.config.bias_sharding,
        projectable=True,
        axes_types=axes_types,
    )
