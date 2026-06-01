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
"""Specifications for pooling layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
from typing import Any, override, Sequence

from sequence_layers.specs import types as types_spec


class BasePooling[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Base specification for pooling layers."""


class MinPooling1D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for MinPooling1D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for MinPooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types_spec.PaddingModeString = types_spec.PaddingMode.VALID.value
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class MaxPooling1D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for MaxPooling1D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for MaxPooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types_spec.PaddingModeString = types_spec.PaddingMode.VALID.value
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class AveragePooling1D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for AveragePooling1D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for AveragePooling1D."""

    pool_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types_spec.PaddingModeString = types_spec.PaddingMode.VALID.value
    masked_average: bool = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class MinPooling2D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for MinPooling2D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for MinPooling2D."""

    pool_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    dilation_rate: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: types_spec.PaddingModeString | tuple[int, int] = (
        types_spec.PaddingMode.SAME.value
    )
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class MaxPooling2D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for MaxPooling2D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for MaxPooling2D."""

    pool_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    dilation_rate: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: types_spec.PaddingModeString | tuple[int, int] = (
        types_spec.PaddingMode.SAME.value
    )
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class AveragePooling2D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for AveragePooling2D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for AveragePooling2D."""

    pool_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    dilation_rate: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: types_spec.PaddingModeString | tuple[int, int] = (
        types_spec.PaddingMode.SAME.value
    )
    masked_average: bool = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class MinPooling3D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for MinPooling3D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for MinPooling3D."""

    pool_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    dilation_rate: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: Sequence[
        types_spec.PaddingModeString | tuple[int, int]
    ] = (
        types_spec.PaddingMode.SAME.value,
        types_spec.PaddingMode.SAME.value,
    )
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class MaxPooling3D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for MaxPooling3D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for MaxPooling3D."""

    pool_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    dilation_rate: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: Sequence[
        types_spec.PaddingModeString | tuple[int, int]
    ] = (
        types_spec.PaddingMode.SAME.value,
        types_spec.PaddingMode.SAME.value,
    )
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class AveragePooling3D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BasePooling[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for AveragePooling3D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for AveragePooling3D."""

    pool_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    dilation_rate: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: Sequence[
        types_spec.PaddingModeString | tuple[int, int]
    ] = (
        types_spec.PaddingMode.SAME.value,
        types_spec.PaddingMode.SAME.value,
    )
    masked_average: bool = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""
