"""Lightweight typing utilities for MLX sequence layers.

Provides type annotation helpers compatible with the jaxtyping-style API
used in the JAX version, but without JAX dependencies. Since runtime type
checking is disabled, these are purely for documentation and IDE support.
"""

from typing import Any, Callable, TypeVar

import mlx.core as mx
import numpy as np

try:
  from jaxtyping import Float, Int, Shaped, PyTree
except ImportError:
  # Fallback: define no-op type aliases if jaxtyping is not available.
  Float = Any
  Int = Any
  Shaped = Any
  PyTree = Any


class _MetaArrayT(type):
  types = ()

  def __instancecheck__(cls, obj):
    return isinstance(obj, cls.types)


class ArrayT(metaclass=_MetaArrayT):
  types = (mx.array, np.ndarray)


ScalarInt = Any
ScalarFloat = Any
AnyPyTree = Any

_F = TypeVar('_F', bound=Callable)


def typed(function: _F) -> _F:
  """No-op decorator for type-checked functions (runtime checking disabled)."""
  return function
