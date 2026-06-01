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
"""Specifications for digital signal processing (DSP) layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
from typing import Any, Callable, override, Protocol, runtime_checkable

from sequence_layers.specs import types as types_spec


class Delay[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesShape,
    types_spec.PreservesType,
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Delay layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Delay layer."""

    length: int = 0
    delay_layer_output: bool = True
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Lookahead[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesShape,
    types_spec.PreservesType,
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Lookahead layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Lookahead layer."""

    length: int = 0
    preserve_length_in_layer: bool = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Window[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesShape,
    types_spec.PreservesType,
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Window layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Window layer."""

    axis: int
    window_fn: Callable[..., Any] | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class Frame[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType,
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Frame layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Frame layer."""

    frame_length: int
    frame_step: int
    padding: tuple[int, int] | types_spec.PaddingModeString = (
        'reverse_causal_valid'
    )
    explicit_padding_is_same_like: bool = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class OverlapAdd[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType,
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for OverlapAdd layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for OverlapAdd layer."""

    frame_length: int
    frame_step: int
    padding: types_spec.PaddingModeString = 'valid'
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class FFT[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType,
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for FFT layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for FFT layer."""

    fft_length: int | None = None
    axis: int = -1
    padding: str = 'right'
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class IFFT[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType,
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for IFFT layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for IFFT layer."""

    fft_length: int | None = None
    frame_length: int | None = None
    axis: int = -1
    padding: str = 'right'
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class RFFT[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for RFFT layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for RFFT layer."""

    fft_length: int | None = None
    axis: int = -1
    padding: str = 'right'
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class IRFFT[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for IRFFT layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for IRFFT layer."""

    fft_length: int | None = None
    frame_length: int | None = None
    axis: int = -1
    padding: str = 'right'
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class STFT[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for STFT layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for STFT layer."""

    frame_length: int
    frame_step: int
    fft_length: int
    window_fn: Callable[..., Any] | None = None
    time_padding: types_spec.PaddingModeString = 'reverse_causal_valid'
    fft_padding: str = 'right'
    output_magnitude: bool = False
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class InverseSTFT[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.SequenceLayer[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for InverseSTFT layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for InverseSTFT layer."""

    frame_length: int
    frame_step: int
    fft_length: int
    window_fn: Callable[..., Any] | None = None
    time_padding: types_spec.PaddingModeString = 'causal'
    fft_padding: str = 'right'
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class LinearToMelSpectrogram[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType,
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for LinearToMelSpectrogram layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for LinearToMelSpectrogram layer."""

    num_mel_bins: int
    sample_rate: float
    lower_edge_hertz: float
    upper_edge_hertz: float
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for DSP module."""

  # pylint: disable=invalid-name
  # pylint: disable=missing-function-docstring

  @property
  def Delay(self) -> type[Delay]:
    ...

  @property
  def Lookahead(self) -> type[Lookahead]:
    ...

  @property
  def Window(self) -> type[Window]:
    ...

  @property
  def Frame(self) -> type[Frame]:
    ...

  @property
  def OverlapAdd(self) -> type[OverlapAdd]:
    ...

  @property
  def FFT(self) -> type[FFT]:
    ...

  @property
  def IFFT(self) -> type[IFFT]:
    ...

  @property
  def RFFT(self) -> type[RFFT]:
    ...

  @property
  def IRFFT(self) -> type[IRFFT]:
    ...

  @property
  def STFT(self) -> type[STFT]:
    ...

  @property
  def InverseSTFT(self) -> type[InverseSTFT]:
    ...

  @property
  def LinearToMelSpectrogram(self) -> type[LinearToMelSpectrogram]:
    ...
