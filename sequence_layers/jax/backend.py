"""Backend-specific helpers (JAX)"""

from typing import override

import jax.nn as jnn
import jax.numpy as jnp

from sequence_layers.specs import backend as spec
from sequence_layers.specs import types as types_spec


class BackendWrapper(spec.xp):
  """Thin wrapper around JAX to match NumPy interface for tests."""

  bool_ = jnp.bool_
  int32 = jnp.int32
  float32 = jnp.float32

  @override
  def array(self, a, dtype=None) -> types_spec.Array:
    return jnp.array(a, dtype=dtype)

  @override
  def zeros(self, shape, dtype=None) -> types_spec.Array:
    return jnp.zeros(shape, dtype=dtype)

  @override
  def concatenate(self, arrays, axis=0) -> types_spec.Array:
    return jnp.concatenate(arrays, axis=axis)

  @override
  def abs(self, x) -> types_spec.Array:
    return jnp.abs(x)

  @override
  def exp(self, x) -> types_spec.Array:
    return jnp.exp(x)

  @override
  def log(self, x) -> types_spec.Array:
    return jnp.log(x)


xp: spec.xp = BackendWrapper()


class NNWrapper(spec.nn):
  """Wrapper around JAX activations to match backend protocol."""

  @override
  def relu(self, x: types_spec.Array) -> types_spec.Array:
    return jnn.relu(x)

  @override
  def sigmoid(self, x: types_spec.Array) -> types_spec.Array:
    return jnn.sigmoid(x)

  @override
  def tanh(self, x: types_spec.Array) -> types_spec.Array:
    return jnn.tanh(x)

  @override
  def swish(self, x: types_spec.Array) -> types_spec.Array:
    return jnn.swish(x)

  @override
  def gelu(self, x: types_spec.Array) -> types_spec.Array:
    return jnn.gelu(x)

  @override
  def elu(self, x: types_spec.Array) -> types_spec.Array:
    return jnn.elu(x)

  @override
  def softplus(self, x: types_spec.Array) -> types_spec.Array:
    return jnn.softplus(x)

  @override
  def softmax(self, x: types_spec.Array, axis: int = -1) -> types_spec.Array:
    return jnn.softmax(x, axis=axis)


nn: spec.nn = NNWrapper()
