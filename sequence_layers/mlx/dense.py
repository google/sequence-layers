"""Dense sequence layer for MLX."""

import dataclasses
from typing import Callable, override

from mlx import nn
import mlx.core as mx

from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import types as mlx_types
from sequence_layers.mlx.init_mapping import _to_mx_dtype
from sequence_layers.specs import dense as spec


class _DenseEager(mlx_types.Stateless):
  """A basic dense layer backed by mlx.nn.Linear.

  Requires in_features at initialization.
  """

  def __init__(
      self,
      *,
      in_features: int,
      features: int,
      use_bias: bool = True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
      name: str | None = None,
  ):
    """Initialize _DenseEager."""
    super().__init__()
    self.features = features
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.name = name
    self._linear = nn.Linear(in_features, features, bias=use_bias)

  @property
  def use_bias(self):
    """Return whether bias is used."""
    return 'bias' in self._linear

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    """Get output shape."""
    if not input_shape:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.features,)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @override
  @mlx_types.check_layer
  def layer(self, x, *, training: bool, constants=None):
    compute_dtype = self.get_output_dtype(x.dtype)

    def dense_fn(v):
      y = self._linear(v.astype(compute_dtype))
      if self.activation is not None:
        y = self.activation(y)
      return y

    if self.use_bias or self.activation is not None:
      return x.apply_values(dense_fn)
    return x.apply_values_masked(dense_fn)


class Dense(mlx_types.Stateless, spec.Dense):
  """A basic dense layer with deferred initialization.

  Matches JAX interface where in_features is inferred.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(mlx_types.SequenceLayerConfig):
    """Dense config."""

    features: int
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: mlx_types.DType | None = None
    param_dtype: mlx_types.DType = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'Dense':
      return Dense.from_config(self)

  def __init__(
      self,
      *,
      features: int,
      use_bias: bool = True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
      name: str | None = None,
  ):
    """Initialize Dense."""
    super().__init__()
    self.features = features
    self._use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.name = name
    self.inner = None

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _ensure_initialized(self, in_features: int):
    """Ensure inner _DenseEager is initialized."""
    if self.inner is not None:
      return
    self.inner = _DenseEager(
        in_features=in_features,
        features=self.features,
        use_bias=self._use_bias,
        activation=self.activation,
        compute_dtype=self.compute_dtype,
        param_dtype=self._param_dtype,
        name=self.name,
    )

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    """Get output shape."""
    if not input_shape:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.features,)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @override
  @mlx_types.check_layer
  def layer(self, x, *, training: bool, constants=None):
    self._ensure_initialized(x.shape[-1])
    assert self.inner is not None
    return self.inner.layer(x, training=training, constants=constants)

  @classmethod
  def from_config(cls, config):
    """Create Dense from config."""
    compute_dtype = getattr(config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = _to_mx_dtype(compute_dtype)
    return cls(
        features=config.features,
        use_bias=config.use_bias,
        activation=init_mapping.map_activation(config.activation),
        compute_dtype=compute_dtype,
        param_dtype=_to_mx_dtype(config.param_dtype),
        name=config.name,
    )


# pylint: disable=too-many-instance-attributes
class EinsumDense(mlx_types.Stateless, spec.EinsumDense):
  """Dense layer using Einstein summation notation."""

  @dataclasses.dataclass(frozen=True)
  class Config(mlx_types.SequenceLayerConfig):
    """MLX-native configuration for EinsumDense."""

    equation: str = ''
    output_shape: tuple[int | None, ...] = ()
    bias_axes: str = ''
    activation: Callable | None = None
    compute_dtype: mlx_types.DType | None = None
    param_dtype: mlx_types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    @override
    def make(self) -> 'EinsumDense':
      return EinsumDense.from_config(self)

  def __init__(
      self,
      *,
      equation,
      output_shape,
      bias_axes='',
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
      name: str | None = None,
  ):
    """Initialize EinsumDense."""
    super().__init__()
    self._equation = equation
    self._output_shape_spec = tuple(output_shape)
    self._bias_axes = bias_axes
    self._activation = activation
    self._compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.name = name
    self.kernel = None
    self.bias = None
    self._initialized = False
    self._resolved_output_shape = None

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _ensure_initialized(self, input_shape):
    """Ensure parameters are initialized."""
    if self._initialized:
      return
    output_shape, kernel_shape, bias_shape = _compute_shapes(
        self._equation, input_shape, self._output_shape_spec, self._bias_axes
    )
    self._resolved_output_shape = output_shape
    self.kernel = mx.zeros(kernel_shape, dtype=self._param_dtype)
    if bias_shape is not None:
      self.bias = mx.zeros(bias_shape, dtype=self._param_dtype)
    self._initialized = True

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    """Get output shape."""
    output_shape, _, _ = _compute_shapes(
        self._equation, input_shape, self._output_shape_spec, self._bias_axes
    )
    return output_shape

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    if self._compute_dtype is not None:
      return self._compute_dtype
    return self._param_dtype

  @override
  @mlx_types.check_layer
  def layer(self, x, *, training: bool, constants=None):
    self._ensure_initialized(x.channel_shape)
    compute_dtype = self.get_output_dtype(x.dtype)

    def einsum_fn(v):
      y = mx.einsum(self._equation, v.astype(compute_dtype), self.kernel)
      if self.bias is not None:
        y = y + self.bias
      if self._activation is not None:
        y = self._activation(y)
      return y

    if self.bias is not None or self._activation is not None:
      return x.apply_values(einsum_fn)
    return x.apply_values_masked(einsum_fn)

  @classmethod
  def from_config(cls, config):
    """Create EinsumDense from config."""
    compute_dtype = getattr(config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = _to_mx_dtype(compute_dtype)
    return cls(
        equation=config.equation,
        output_shape=config.output_shape,
        bias_axes=config.bias_axes,
        activation=init_mapping.map_activation(config.activation),
        compute_dtype=compute_dtype,
        param_dtype=_to_mx_dtype(config.param_dtype),
        name=config.name,
    )


def _parse_equation(equation):
  """Parse einsum equation of form '...ab,bc->...ac'."""
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


def _compute_shapes(equation, input_shape, output_shape_spec, bias_axes):
  """Compute kernel_shape and bias_shape from equation and shapes."""
  input_spec, kernel_spec, output_spec = _parse_equation(equation)
  in_spec = input_spec[3:]
  out_spec = output_spec[3:]

  if len(in_spec) != len(input_shape):
    raise ValueError(f'Equation {in_spec=} does not match {input_shape=} rank.')

  input_dims = {d: input_shape[i] for i, d in enumerate(in_spec)}
  output_shape = list(output_shape_spec)
  if len(out_spec) != len(output_shape):
    raise ValueError(f'Equation {out_spec=} does not match {output_shape=}.')

  for i, d in enumerate(out_spec):
    if output_shape[i] is None:
      output_shape[i] = input_dims[d]
    elif d in input_dims and output_shape[i] != input_dims[d]:
      raise ValueError(
          f'Inconsistent dimension {d=}. {output_shape=} vs {input_shape=}'
      )

  output_dim_map = {d: output_shape[i] for i, d in enumerate(out_spec)}

  kernel_shape = []
  for d in kernel_spec:
    if d in input_dims:
      kernel_shape.append(input_dims[d])
    elif d in output_dim_map:
      kernel_shape.append(output_dim_map[d])
    else:
      raise ValueError(f"Weight dimension '{d}' not in input or output spec.")

  if bias_axes:
    first_bias_loc = min(out_spec.find(c) for c in bias_axes)
    bias_out_spec = out_spec[first_bias_loc:]
    bias_shape = [
        output_dim_map[c] if c in bias_axes else 1 for c in bias_out_spec
    ]
  else:
    bias_shape = None

  return tuple(output_shape), tuple(kernel_shape), bias_shape
