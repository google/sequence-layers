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
"""Specifications for combinator layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
import enum
from typing import (Any, Callable, override, Protocol, runtime_checkable,
                    Sequence)

from sequence_layers.specs import types as types_spec


@enum.unique
class CombinationMode(enum.Enum):
  """The type of combination to perform."""

  STACK = 1
  CONCAT = 2
  ADD = 3
  MEAN = 4
  PRODUCT = 5


class Serial[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Serial layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Serial."""

    layers: Sequence[types_spec.SequenceLayerConfig] = ()
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class SerialModules[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for SerialModules layer."""


class Residual[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Residual layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Residual."""

    layers: Sequence[types_spec.SequenceLayerConfig] = ()
    shortcut_layers: Sequence[types_spec.SequenceLayerConfig] | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Repeat[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Repeat layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Repeat."""

    layer: types_spec.SequenceLayerConfig
    num_repeats: int
    remat: bool = False
    prevent_cse: bool = False
    policy: Callable[..., bool] | None = None
    unroll_layer: bool = False
    unroll_step: bool = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Parallel[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Parallel layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Parallel."""

    layers: Sequence[types_spec.SequenceLayerConfig]
    combination: CombinationMode = CombinationMode.STACK
    share_scope: bool | Sequence[bool] = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for combinators module."""

  @property
  def CombinationMode(self) -> type[CombinationMode]:
    ...

  @property
  def Serial(self) -> type[Serial]:
    ...

  @property
  def SerialModules(self) -> type[SerialModules]:
    ...

  @property
  def Residual(self) -> type[Residual]:
    ...

  @property
  def Repeat(self) -> type[Repeat]:
    ...

  @property
  def Parallel(self) -> type[Parallel]:
    ...
