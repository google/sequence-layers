"""Dense sequence layer for MLX."""

import dataclasses
import math

from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import init_mapping
from sequence_layers.mlx.init_mapping import _to_mx_dtype
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence


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
  """Compute kernel_shape and bias_shape from equation and shapes.

  Args:
    equation: einsum equation string.
    input_shape: channel shape of input (excluding batch/time).
    output_shape_spec: user-specified output shape with possible Nones.
    bias_axes: string of output axes that get bias.

  Returns:
    (output_shape, kernel_shape, bias_shape) where bias_shape may be None.
  """
  input_spec, kernel_spec, output_spec = _parse_equation(equation)
  in_spec = input_spec[3:]  # Strip '...'
  out_spec = output_spec[3:]

  if len(in_spec) != len(input_shape):
    raise ValueError(
        f'Equation {in_spec=} does not match {input_shape=} rank.'
    )

  input_dims = {d: input_shape[i] for i, d in enumerate(in_spec)}
  output_shape = list(output_shape_spec)
  if len(out_spec) != len(output_shape):
    raise ValueError(
        f'Equation {out_spec=} does not match {output_shape=}.'
    )

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
      raise ValueError(
          f"Weight dimension '{d}' not in input or output spec."
      )

  if bias_axes:
    first_bias_loc = min(out_spec.find(c) for c in bias_axes)
    bias_out_spec = out_spec[first_bias_loc:]
    bias_shape = [
        output_dim_map[c] if c in bias_axes else 1 for c in bias_out_spec
    ]
  else:
    bias_shape = None

  return tuple(output_shape), tuple(kernel_shape), bias_shape


class Dense(types.Stateless):
  """A basic dense layer backed by mlx.nn.Linear.

  Unlike mlx.nn.Linear (which stores weight as [out, in]), this layer
  presents a SequenceLayer interface. Weight conversion from Linen
  [in, out] requires a single transpose at load time.
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
  ):
    super().__init__()
    self.features = features
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self._linear = nn.Linear(in_features, features, bias=use_bias)

  @property
  def use_bias(self):
    return 'bias' in self._linear

  def get_output_shape(self, input_shape, *, constants=None):
    if not input_shape:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.features,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @types.check_layer
  def layer(self, x, *, constants=None):
    compute_dtype = self.get_output_dtype(x.dtype)

    def dense_fn(v):
      y = self._linear(v.astype(compute_dtype))
      if self.activation is not None:
        y = self.activation(y)
      return y

    if self.use_bias or self.activation is not None:
      return x.apply_values(dense_fn)
    else:
      return x.apply_values_masked(dense_fn)



  def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = 'affine'):
    if self.kernel is None or self._equation != '...nh,dnh->...d' or (self.kernel.shape[-1] * self.kernel.shape[-2]) % group_size != 0:
      return self

    _d, _n, _h = self.kernel.shape
    kernel_2d = self.kernel.reshape(_d, _n * _h)
    self.q_weight, self.q_scales, self.q_biases = mx.quantize(
        kernel_2d, group_size=group_size, bits=bits
    )
    self._group_size = group_size
    self._bits = bits
    self.kernel = None

    def layer(self, x, *, constants=None):
        compute_dtype = self.get_output_dtype(x.dtype)
        def quantized_einsum_fn(v):
            original_shape = v.shape
            v_2d = v.reshape(*original_shape[:-2], _n * _h)
            v_2d = v_2d.astype(compute_dtype)
            y = mx.quantized_matmul(
                v_2d,
                self.q_weight,
                scales=self.q_scales,
                biases=self.q_biases,
                transpose=True,
                group_size=self._group_size,
                bits=self._bits,
            )
            if self.bias is not None:
                y = y + self.bias
            if self._activation is not None:
                y = self._activation(y)
            return y

        if self.bias is not None or self._activation is not None:
            return x.apply_values(quantized_einsum_fn)
        return x.apply_values_masked(quantized_einsum_fn)
    
    import types
    self.layer = types.MethodType(layer, self)
    
    return self


  @classmethod
  def from_config(cls, config):
    """Create a Dense layer from a Linen Dense.Config."""
    return cls(
        in_features=None,  # Deferred; set by DenseDeferred.
        features=config.features,
        use_bias=config.use_bias,
        activation=init_mapping.map_activation(config.activation),
        compute_dtype=getattr(config, 'compute_dtype', None),
        param_dtype=config.param_dtype,
    )


class DenseDeferred(types.Stateless):
  """Dense layer that defers weight creation until first use."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    """MLX-native configuration for Dense."""
    features: int = 1
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def make(self) -> 'DenseDeferred':
      return DenseDeferred.from_config(self)

  def __init__(
      self,
      *,
      features: int,
      use_bias: bool = True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self.features = features
    self._use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self.inner = None

  def _ensure_initialized(self, in_features: int):
    if self.inner is not None:
      return
    self.inner = Dense(
        in_features=in_features,
        features=self.features,
        use_bias=self._use_bias,
        activation=self.activation,
        compute_dtype=self.compute_dtype,
        param_dtype=self._param_dtype,
    )

  def get_output_shape(self, input_shape, *, constants=None):
    if not input_shape:
      raise ValueError(
          f'Dense requires at least rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.features,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self.inner.layer(x, constants=constants)



  def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = 'affine'):
    if self.kernel is None or self._equation != '...nh,dnh->...d' or (self.kernel.shape[-1] * self.kernel.shape[-2]) % group_size != 0:
      return self

    _d, _n, _h = self.kernel.shape
    kernel_2d = self.kernel.reshape(_d, _n * _h)
    self.q_weight, self.q_scales, self.q_biases = mx.quantize(
        kernel_2d, group_size=group_size, bits=bits
    )
    self._group_size = group_size
    self._bits = bits
    self.kernel = None

    def layer(self, x, *, constants=None):
        compute_dtype = self.get_output_dtype(x.dtype)
        def quantized_einsum_fn(v):
            original_shape = v.shape
            v_2d = v.reshape(*original_shape[:-2], _n * _h)
            v_2d = v_2d.astype(compute_dtype)
            y = mx.quantized_matmul(
                v_2d,
                self.q_weight,
                scales=self.q_scales,
                biases=self.q_biases,
                transpose=True,
                group_size=self._group_size,
                bits=self._bits,
            )
            if self.bias is not None:
                y = y + self.bias
            if self._activation is not None:
                y = self._activation(y)
            return y

        if self.bias is not None or self._activation is not None:
            return x.apply_values(quantized_einsum_fn)
        return x.apply_values_masked(quantized_einsum_fn)
    
    import types
    self.layer = types.MethodType(layer, self)
    
    return self


  @classmethod
  def from_config(cls, config):
    """Create from a Linen Dense.Config."""
    compute_dtype = getattr(config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = _to_mx_dtype(compute_dtype)
    return cls(
        features=config.features,
        use_bias=config.use_bias,
        activation=init_mapping.map_activation(config.activation),
        compute_dtype=compute_dtype,
        param_dtype=_to_mx_dtype(config.param_dtype),
    )


class EinsumDense(types.Stateless):
  """Dense layer using Einstein summation notation."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
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
  ):
    super().__init__()
    self._equation = equation
    self._output_shape_spec = tuple(output_shape)
    self._bias_axes = bias_axes
    self._activation = activation
    self._compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    # Deferred: created on first call.
    self.kernel = None
    self.bias = None
    self._initialized = False

  def _ensure_initialized(self, input_shape):
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

  def get_output_shape(self, input_shape, *, constants=None):
    output_shape, _, _ = _compute_shapes(
        self._equation, input_shape, self._output_shape_spec, self._bias_axes
    )
    return output_shape

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self._compute_dtype is not None:
      return self._compute_dtype
    return self._param_dtype

  @types.check_layer
  def layer(self, x, *, constants=None):
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



  def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = 'affine'):
    if self.kernel is None or self._equation != '...nh,dnh->...d' or (self.kernel.shape[-1] * self.kernel.shape[-2]) % group_size != 0:
      return self

    _d, _n, _h = self.kernel.shape
    kernel_2d = self.kernel.reshape(_d, _n * _h)
    self.q_weight, self.q_scales, self.q_biases = mx.quantize(
        kernel_2d, group_size=group_size, bits=bits
    )
    self._group_size = group_size
    self._bits = bits
    self.kernel = None

    def layer(self, x, *, constants=None):
        compute_dtype = self.get_output_dtype(x.dtype)
        def quantized_einsum_fn(v):
            original_shape = v.shape
            v_2d = v.reshape(*original_shape[:-2], _n * _h)
            v_2d = v_2d.astype(compute_dtype)
            y = mx.quantized_matmul(
                v_2d,
                self.q_weight,
                scales=self.q_scales,
                biases=self.q_biases,
                transpose=True,
                group_size=self._group_size,
                bits=self._bits,
            )
            if self.bias is not None:
                y = y + self.bias
            if self._activation is not None:
                y = self._activation(y)
            return y

        if self.bias is not None or self._activation is not None:
            return x.apply_values(quantized_einsum_fn)
        return x.apply_values_masked(quantized_einsum_fn)
    
    import types
    self.layer = types.MethodType(layer, self)
    
    return self


  @classmethod
  def from_config(cls, config):
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
    )


# Alias so that sl.Dense.Config(...) works like sl_jax.Dense.Config(...).
Dense.Config = DenseDeferred.Config

