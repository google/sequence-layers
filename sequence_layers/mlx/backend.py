"""Backend-specific helpers (MLX)"""

from typing import override

import mlx.core as mx

from sequence_layers.specs import backend as spec
from sequence_layers.specs import types as types_spec


class BackendWrapper(spec.xp):
  """Thin wrapper around MLX to match NumPy interface for tests."""

  bool_ = mx.bool_
  int32 = mx.int32

  @override
  def array(self, a, dtype=None) -> types_spec.Array:
    return mx.array(a, dtype=dtype)

  @override
  def zeros(self, shape, dtype=None) -> types_spec.Array:
    return mx.zeros(shape, dtype=dtype)


xp: spec.xp = BackendWrapper()
