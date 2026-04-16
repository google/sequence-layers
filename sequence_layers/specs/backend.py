"""Specification for backend-specific helpers."""

from typing import Any, Protocol, runtime_checkable

from sequence_layers.specs import types as types_spec

Array = types_spec.Array


# pylint: disable=invalid-name
class xp(Protocol):
  """NumPy-compatible interface to enable generic behavior tests.

  https://numpy.org/doc/stable/reference/routines.html#routines
  https://docs.jax.dev/en/latest/jax.numpy.html
  """

  bool_: Any
  int32: Any
  float32: Any

  def array(self, a: Any, dtype: Any = None) -> Array:
    """Creates an array."""

  def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Array:
    """Creates an array of zeros."""

  def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
    """Concatenates arrays."""

  def abs(self, x: Array) -> Array:
    """Computes absolute value."""

  def exp(self, x: Array) -> Array:
    """Computes exponential."""

  def log(self, x: Array) -> Array:
    """Computes natural logarithm."""


class nn(Protocol):
  """Protocol for neural network operations (activations)."""

  def relu(self, x: Array) -> Array:
    """Computes ReLU activation."""

  def sigmoid(self, x: Array) -> Array:
    """Computes sigmoid activation."""

  def tanh(self, x: Array) -> Array:
    """Computes tanh activation."""

  def swish(self, x: Array) -> Array:
    """Computes swish activation."""

  def gelu(self, x: Array) -> Array:
    """Computes GeLU activation."""

  def elu(self, x: Array) -> Array:
    """Computes ELU activation."""

  def softplus(self, x: Array) -> Array:
    """Computes softplus activation."""

  def softmax(self, x: Array, axis: int = -1) -> Array:
    """Computes softmax activation."""


# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
@runtime_checkable
class ModuleSpec(Protocol):
  """Specification for sequence_layers.<backend>.backend"""

  @property
  def xp(self) -> xp:
    ...

  @property
  def nn(self) -> nn:
    ...


__all__ = [
    name
    for name, attr in ModuleSpec.__dict__.items()
    if isinstance(attr, property)
]
