"""Specification for backend-specific helpers."""

from typing import Any, Protocol, runtime_checkable

from sequence_layers.specs import types as types_spec


Array = types_spec.Array


class xp(Protocol):
  """NumPy-compatible interface to enable generic behavior tests.

  https://numpy.org/doc/stable/reference/routines.html#routines
  https://docs.jax.dev/en/latest/jax.numpy.html
  """

  bool_: Any
  int32: Any

  def array(self, a: Any, dtype: Any = None) -> Array:
    ...

  def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Array:
    ...


@runtime_checkable
class ModuleSpec(Protocol):
  """Specification for sequence_layers.<backend>.backend"""

  @property
  def xp(self) -> xp:
    ...


__all__ = [
    name
    for name, attr in ModuleSpec.__dict__.items()
    if isinstance(attr, property)
]
