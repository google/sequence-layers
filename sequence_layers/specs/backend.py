# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Specification for backend-specific helpers."""

from typing import Any, Protocol, runtime_checkable
from typing import Sequence as TypingSequence

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

  def concatenate(self, arrays: TypingSequence[Array], axis: int = 0) -> Array:
    """Concatenates a list of arrays."""

  def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
    """Broadcasts an array to a new shape."""

  def abs(self, x: Array) -> Array:
    """Computes absolute value."""

  def exp(self, x: Array) -> Array:
    """Computes exponential."""

  def log(self, x: Array) -> Array:
    """Computes natural logarithm."""

  def mean(
      self,
      x: Array,
      axis: int | tuple[int, ...] | None = None,
      dtype: Any = None,
      keepdims: bool = False,
      where: Array | None = None,
  ) -> Array:
    """Computes the arithmetic mean along the specified axes."""

  def var(
      self,
      x: Array,
      axis: int | tuple[int, ...] | None = None,
      dtype: Any = None,
      keepdims: bool = False,
      where: Array | None = None,
  ) -> Array:
    """Computes the variance along the specified axes."""


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
  """Specification for sequence_layers.<backend>.backend."""

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
