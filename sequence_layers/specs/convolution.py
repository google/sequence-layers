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
"""Specifications for convolution layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
from typing import (Any, Callable, override, Protocol, runtime_checkable,
                    Sequence)

from sequence_layers.specs import types as types_spec


class BaseConv[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Base specification for convolution layers."""


class Conv1D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BaseConv[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Conv1D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Conv1D."""

    filters: int
    kernel_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types_spec.PaddingModeString = types_spec.PaddingMode.VALID.value
    groups: int = 1
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: Any = None
    param_dtype: Any = None  # Can be numpy, jax, or mlx dtype
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class DepthwiseConv1D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BaseConv[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for DepthwiseConv1D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for DepthwiseConv1D."""

    kernel_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types_spec.PaddingModeString = types_spec.PaddingMode.VALID.value
    channel_multiplier: int = 1
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: Any = None
    param_dtype: Any = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Conv1DTranspose[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Conv1DTranspose layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Conv1DTranspose."""

    filters: int
    kernel_size: int
    strides: int = 1
    padding: types_spec.PaddingModeString = types_spec.PaddingMode.VALID.value
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: Any = None
    param_dtype: Any = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Conv2D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    BaseConv[SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Conv2D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Conv2D."""

    filters: int
    kernel_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    dilation_rate: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: types_spec.PaddingModeString | tuple[int, int] = (
        types_spec.PaddingMode.SAME.value
    )
    groups: int = 1
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: Any = None
    param_dtype: Any = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Conv2DTranspose[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Conv2DTranspose layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Conv2DTranspose."""

    filters: int
    kernel_size: int | Sequence[int]
    strides: int | Sequence[int] = 1
    time_padding: types_spec.PaddingModeString = (
        types_spec.PaddingMode.VALID.value
    )
    spatial_padding: types_spec.PaddingModeString | tuple[int, int] = (
        types_spec.PaddingMode.SAME.value
    )
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: Any = None
    param_dtype: Any = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for convolution module."""

  # pylint: disable=invalid-name
  # pylint: disable=missing-function-docstring

  @property
  def Conv1D(self) -> type[Conv1D]:
    ...

  @property
  def DepthwiseConv1D(self) -> type[DepthwiseConv1D]:
    ...

  @property
  def Conv1DTranspose(self) -> type[Conv1DTranspose]:
    ...

  @property
  def Conv2D(self) -> type[Conv2D]:
    ...

  @property
  def Conv2DTranspose(self) -> type[Conv2DTranspose]:
    ...
