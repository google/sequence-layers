"""Backend-specific helpers (JAX)"""

import jax.numpy as jnp

from sequence_layers.specs import backend
from sequence_layers.specs import types as types_spec


class BackendWrapper:
  """Thin wrapper around JAX to match NumPy interface for tests."""

  bool_ = jnp.bool_
  int32 = jnp.int32

  def array(self, a, dtype=None) -> types_spec.Array:
    return jnp.array(a, dtype=dtype)

  def zeros(self, shape, dtype=None) -> types_spec.Array:
    return jnp.zeros(shape, dtype=dtype)


xp: backend.xp = BackendWrapper()
