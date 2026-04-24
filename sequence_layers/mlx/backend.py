"""Backend-specific helpers (MLX)"""

from typing import override

import mlx.core as mx
import mlx.nn as nn_mlx

from sequence_layers.specs import backend as spec
from sequence_layers.specs import types as types_spec


class BackendWrapper(spec.xp):
  """Thin wrapper around MLX to match NumPy interface for tests."""

  bool_ = mx.bool_
  int32 = mx.int32
  float32 = mx.float32

  @override
  def array(self, a, dtype=None) -> types_spec.Array:
    return mx.array(a, dtype=dtype)

  @override
  def zeros(self, shape, dtype=None) -> types_spec.Array:
    return mx.zeros(shape, dtype=dtype)

  @override
  def concatenate(self, arrays, axis=0) -> types_spec.Array:
    return mx.concatenate(arrays, axis=axis)

  @override
  def abs(self, x) -> types_spec.Array:
    return mx.abs(x)

  @override
  def exp(self, x) -> types_spec.Array:
    return mx.exp(x)

  @override
  def log(self, x) -> types_spec.Array:
    return mx.log(x)


xp: spec.xp = BackendWrapper()


class NNWrapper(spec.nn):
  """Wrapper around MLX activations to match backend protocol."""

  @override
  def relu(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.relu(x)

  @override
  def sigmoid(self, x: types_spec.Array) -> types_spec.Array:
    return mx.sigmoid(x)

  @override
  def tanh(self, x: types_spec.Array) -> types_spec.Array:
    return mx.tanh(x)

  @override
  def swish(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.silu(x)

  @override
  def gelu(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.gelu(x)

  @override
  def elu(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.elu(x)

  @override
  def softplus(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.softplus(x)

  @override
  def softmax(self, x: types_spec.Array, axis: int = -1) -> types_spec.Array:
    return mx.softmax(x, axis=axis)


nn: spec.nn = NNWrapper()
