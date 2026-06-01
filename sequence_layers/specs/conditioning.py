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
"""Specifications for conditioning layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
import enum
from typing import Any, override, Protocol, runtime_checkable

from sequence_layers.specs import types as types_spec


@enum.unique
class Projection(enum.Enum):
  """The type of projection to perform."""

  # No projection.
  IDENTITY = 1
  # Dense projection from every element of c at a given time step, to a tensor
  # of the same shape as x at given time step.
  LINEAR = 2
  # Dense projection from every element of c at a given time step, to a tensor
  # of shape [2, x.shape...] at given time step.
  LINEAR_AFFINE = 3


@enum.unique
class Combination(enum.Enum):
  """The type of combination to perform."""

  # Broadcast-add conditioning.
  ADD = 1
  # Broadcast-concat conditioning.
  CONCAT = 2
  # Affine conditioning. Requires LINEAR_AFFINE projection.
  AFFINE = 3
  # Affine shift conditioning. Requires LINEAR projection.
  AFFINE_SHIFT = 4
  # Affine scale conditioning. Requires LINEAR projection.
  AFFINE_SCALE = 5
  # Broadcast-multiply conditioning. Requires LINEAR or IDENTITY projection.
  MUL = 6
  # Broadcast-concat conditioning via prepending.
  CONCAT_BEFORE = 7


class BaseConditioning[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Base specification for conditioning layers."""

  # For backward compatibility with nested enum references
  Projection = Projection
  Combination = Combination


class Conditioning[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BaseConditioning[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Conditioning layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Conditioning."""

    conditioning_name: str
    projection: Projection
    combination: Combination
    projection_channel_shape: types_spec.Shape | None = None
    streaming: bool = False
    affine_scale_offset: complex = 1.0
    compute_dtype: Any = None
    param_dtype: Any = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for conditioning module."""

  @property
  def Conditioning(self) -> type[Conditioning]:
    ...
