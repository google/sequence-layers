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
"""Specifications for position and timing layers."""

import abc
import dataclasses
from typing import Any, override, Protocol, runtime_checkable

from sequence_layers.specs import types as types_spec


class AddTimingSignal[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Adds sinusoids at varying frequencies to the input channels dimension."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Config for AddTimingSignal."""

    min_timescale: float = 1.0
    max_timescale: float = 1.0e4
    trainable_scale: bool = False
    axes: int | tuple[int, ...] | None = None
    sharding: types_spec.Sharding | None = None
    param_dtype: Any = None
    only_advance_position_for_valid_timesteps: bool = True
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class ApplyRotaryPositionalEncoding[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Applies Rotary Positional Encodings (RoPE) to the sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Config for ApplyRotaryPositionalEncoding."""

    max_wavelength: float
    axis: int = -1
    only_advance_position_for_valid_timesteps: bool = True
    positions_in_at_least_fp32: bool = True
    positions_name: str | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for position module."""

  @property
  def AddTimingSignal(self) -> type[AddTimingSignal]:
    ...

  @property
  def ApplyRotaryPositionalEncoding(
      self,
  ) -> type[ApplyRotaryPositionalEncoding]:
    ...
