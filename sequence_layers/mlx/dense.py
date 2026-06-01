"""Dense sequence layer for MLX."""

import dataclasses
from typing import Callable, override

from mlx import nn
import mlx.core as mx

from sequence_layers.mlx import types
from sequence_layers.mlx.simple import _to_mx_dtype
from sequence_layers.specs import dense as spec


class Dense(types.Stateless, spec.Dense):
  """A basic dense layer with deferred initialization.

  Matches JAX interface where in_features is inferred on first call.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.Dense.Config):
    """Dense config."""

    features: int
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'Dense':
      return Dense(self)

  def __init__(self, config: Config):
    """Initialize Dense."""
    super().__init__()
    self.config = config
    self._compute_dtype = _to_mx_dtype(config.compute_dtype)
    self._param_dtype = _to_mx_dtype(config.param_dtype)
    self._linear = None

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _ensure_initialized(self, in_features: int):
    """Ensure nn.Linear is initialized on first call."""
    if self._linear is not None:
      return
    self._linear = nn.Linear(
        in_features, self.config.features, bias=self.config.use_bias
    )

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    """Get output shape."""
    if not input_shape:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.config.features,)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    if self._compute_dtype is not None:
      return self._compute_dtype
    assert self._param_dtype is not None
    return self._param_dtype

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    if x.ndim < 3:
      raise ValueError(f'Dense requires at least rank 3 input. Got: {x.shape=}')
    self._ensure_initialized(x.shape[-1])
    assert self._linear is not None
    activation = self.config.activation
    compute_dtype = self.get_output_dtype(x.dtype)

    def dense_fn(v):
      y = self._linear(v.astype(compute_dtype))
      if activation is not None:
        y = activation(y)
      return y

    if self.config.use_bias or activation is not None:
      return x.apply_values(dense_fn)
    return x.apply_values_masked(dense_fn)


class EinsumDense(types.Stateless, spec.EinsumDense):
  """Dense layer using Einstein summation notation."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.EinsumDense.Config):
    """MLX-native configuration for EinsumDense."""

    equation: str = ''
    output_shape: tuple[int | None, ...] = ()
    bias_axes: str = ''
    activation: Callable | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    @override
    def make(self) -> 'EinsumDense':
      return EinsumDense(self)

  def __init__(self, config: Config):
    """Initialize EinsumDense."""
    super().__init__()
    self.config = config
    self._compute_dtype = _to_mx_dtype(config.compute_dtype)
    self._param_dtype = _to_mx_dtype(config.param_dtype)
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
        self.config.equation,
        input_shape,
        self.config.output_shape,
        self.config.bias_axes,
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
        self.config.equation,
        input_shape,
        self.config.output_shape,
        self.config.bias_axes,
    )
    return output_shape

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    if self._compute_dtype is not None:
      return self._compute_dtype
    assert self._param_dtype is not None
    return self._param_dtype

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    self._ensure_initialized(x.channel_shape)
    compute_dtype = self.get_output_dtype(x.dtype)
    activation = self.config.activation

    def einsum_fn(v):
      y = mx.einsum(self.config.equation, v.astype(compute_dtype), self.kernel)
      if self.bias is not None:
        y = y + self.bias
      if activation is not None:
        y = activation(y)
      return y

    if self.bias is not None or activation is not None:
      return x.apply_values(einsum_fn)
    return x.apply_values_masked(einsum_fn)


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
