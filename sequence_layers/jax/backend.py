"""Backend-specific helpers (JAX)"""

from typing import override

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

  def concatenate(self, arrays, axis=0) -> types_spec.Array:
    return jnp.concatenate(arrays, axis=axis)


xp: spec.xp = BackendWrapper()
