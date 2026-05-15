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
"""Abstract base classes and types for SequenceLayers."""

import abc
import enum
import fractions
from typing import Any, Callable, Generic, Iterable, Literal, TypeVar

# Type aliases for generic usage
T = TypeVar('T')
ValuesT = TypeVar('ValuesT')
MaskT = TypeVar('MaskT')
SequenceSelf = TypeVar('SequenceSelf', bound='Sequence')
Shape = tuple[int, ...]
ShapeLike = list[int] | tuple[int, ...]
DType = Any  # Can be numpy, jax, or mlx dtype
ChannelSpec = Any  # Typically ShapeDType or compatible


class PaddingMode(enum.Enum):
  """Supported padding modes."""

  # In VALID padding mode, no padding is applied.
  #
  # Key properties:
  # * The physical length of an input array to a VALID padded function shrinks,
  #   dropping any timesteps whose inputs are computed from implicit edge
  #   padding.
  # * An output timestep is valid when all of its input timesteps are also
  #   valid.
  VALID = 'valid'

  # In SAME padding mode, the input sequence is padded such that the output
  # length is equal to the input length before applying striding.
  #
  # Key properties:
  # * The input length is equal to the output length, before applying striding.
  # * Padding of `effective_kernel_size - 1` is applied. Half is applied to the
  #   front and half to the back. If `effective_kernel_size` is even, the extra
  #   padding is added to the end.
  # * An output timestep is valid when its corresponding input timestep is
  #   valid.
  SAME = 'same'

  # In CAUSAL_VALID padding mode, the input sequence is padded such that the
  # output length is equal to the input length before applying striding. Padding
  # is applied such that the output timestep `to` can only depend on input
  # timesteps at or before `ti` where `ti * output_ratio = to`.
  #
  # Key properties:
  # * As in SAME padding, the input length is equal to the output length, before
  #   applying striding.
  # * Padding of `effective_kernel_size - 1` is applied to the front of the
  #   sequence.
  # * As in VALID padding, an output timestep is valid iff all of its input
  #   timesteps are also valid.
  CAUSAL_VALID = 'causal_valid'

  # In REVERSE_CAUSAL_VALID padding mode, the input sequence is padded such that
  # the output length is equal to the input length before applying striding.
  # Padding is applied such that the output timestep `to` can only depend on
  # input timesteps at or after `ti` where `ti * output_ratio = to`.
  #
  # Key properties:
  # * As in SAME padding, the input length is equal to the output length, before
  #   applying striding.
  # * Padding of `effective_kernel_size - 1` is applied to the back of the
  #   sequence.
  REVERSE_CAUSAL_VALID = 'reverse_causal_valid'

  # In CAUSAL padding mode, the input sequence is padded such that the output
  # length is equal to the input length before applying striding. Padding is
  # applied such that the output timestep `to` can only depend on input
  # timesteps at or before `ti` where `ti * output_ratio = to`.
  #
  # Key properties:
  # * As in SAME padding, the input length is equal to the output length, before
  #   applying striding.
  # * Padding of `effective_kernel_size - 1` is applied to the front of the
  #   sequence.
  # * As in SAME padding, an output timestep is valid when its corresponding
  #   input timestep is valid.
  CAUSAL = 'causal'

  # In REVERSE_CAUSAL padding mode, the input sequence is padded such that the
  # output length is equal to the input length before applying striding. Padding
  # is applied such that the output timestep `to` can only depend on input
  # timesteps at or after `ti` where `ti * output_ratio = to`.
  #
  # Key properties:
  # * As in SAME padding, the input length is equal to the output length, before
  #   applying striding.
  # * Padding of `effective_kernel_size - 1` is applied to the back of the
  #   sequence.
  # * As in SAME padding, an output timestep is valid when its corresponding
  #   input timestep is valid.
  REVERSE_CAUSAL = 'reverse_causal'

  # In SEMICAUSAL padding mode, the input sequence is padded such that the
  # output length is equal to the input length before applying striding. Padding
  # is applied such that the output timestep `to` can only depend on input
  # timesteps at or before `ti` where `ti * output_ratio = to`.
  #
  # Key properties:
  # * As in SAME padding, the input length is equal to the output length, before
  #   applying striding.
  # * Padding of `effective_kernel_size - stride` is applied to the front of the
  #   sequence, and padding of `stride - 1` timesteps is applied to the back of
  #   the sequence for a total of `effective_kernel_size - 1` timesteps of
  #   padding. If `effective_kernel_size` < `stride`, then padding of
  #   `effective_kernel_size - 1` is applied to the back of the sequence.
  # * As in SAME padding, an output timestep is valid when its corresponding
  #   input timestep is valid.
  SEMICAUSAL = 'semicausal'

  # In SEMICAUSAL_FULL padding mode, the input sequence is padded such that the
  # output of the corresponding overlap-add or transpose convolution is of the
  # same size as the input sequence and perfect reconstruction can be achieved.
  # The reconstructed signal is of the same length or of length rounded up to
  # cover the full input sequence.
  SEMICAUSAL_FULL = 'semicausal_full'


PaddingModeString = Literal[
    'valid',
    'same',
    'causal_valid',
    'reverse_causal_valid',
    'causal',
    'reverse_causal',
    'semicausal',
    'semicausal_full',
]


class Sequence(Generic[ValuesT, MaskT], metaclass=abc.ABCMeta):
  """Abstract base class for Sequence."""

  values: ValuesT
  mask: MaskT

  def __init__(self, values: ValuesT, mask: MaskT):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def shape(self) -> Shape:
    pass

  @property
  @abc.abstractmethod
  def ndim(self) -> int:
    pass

  @property
  @abc.abstractmethod
  def channel_shape(self) -> Shape:
    pass

  @property
  @abc.abstractmethod
  def dtype(self) -> DType:
    pass

  @classmethod
  @abc.abstractmethod
  def from_values(cls, values: ValuesT) -> 'Sequence':
    pass

  @classmethod
  @abc.abstractmethod
  def concatenate_sequences(cls, sequences: Iterable['Sequence']) -> 'Sequence':
    pass

  @abc.abstractmethod
  def expanded_mask(self) -> Any:
    pass

  @abc.abstractmethod
  def apply_values(
      self,
      values_fn: Callable[..., ValuesT],
      *args,
      **kwargs,
  ) -> 'Sequence':
    pass

  @abc.abstractmethod
  def apply_values_masked(
      self: SequenceSelf,
      values_fn: Callable[..., ValuesT],
      *args,
      **kwargs,
  ) -> SequenceSelf:
    pass

  @abc.abstractmethod
  def apply(
      self,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args,
      **kwargs,
  ) -> 'Sequence':
    pass

  @abc.abstractmethod
  def apply_masked(
      self: SequenceSelf,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args,
      **kwargs,
  ) -> SequenceSelf:
    pass

  @abc.abstractmethod
  def astype(self: SequenceSelf, dtype: DType | None) -> SequenceSelf:
    pass

  @abc.abstractmethod
  def lengths(self) -> Any:
    pass

  @abc.abstractmethod
  def __getitem__(self: SequenceSelf, the_slice: Any) -> SequenceSelf:
    pass

  @abc.abstractmethod
  def pad_time(
      self: SequenceSelf,
      pad_left: int,
      pad_right: int,
      valid: bool,
      pad_value: Any | None = None,
  ) -> SequenceSelf:
    pass

  @abc.abstractmethod
  def concatenate(self, other: 'Sequence') -> 'Sequence':
    pass

  @abc.abstractmethod
  def mask_invalid(self, mask_value: Any | None = None) -> 'Sequence':
    pass

  @abc.abstractmethod
  def unmask(self) -> 'Sequence':
    pass


class SequenceLayerConfig(metaclass=abc.ABCMeta):
  """Configuration for a SequenceLayer."""

  @abc.abstractmethod
  def make(self) -> Any:
    """Creates the sequence layer."""

  @abc.abstractmethod
  def copy(self, **kwargs) -> 'SequenceLayerConfig':
    """Returns a copy of the config with updated fields."""


class Steppable(metaclass=abc.ABCMeta):
  """A sequence processing layer that can be executed layerwise or stepwise."""

  @property
  @abc.abstractmethod
  def block_size(self) -> int:
    pass

  @property
  @abc.abstractmethod
  def output_ratio(self) -> fractions.Fraction:
    pass

  @property
  @abc.abstractmethod
  def supports_step(self) -> bool:
    pass

  @property
  @abc.abstractmethod
  def input_latency(self) -> int:
    pass

  @property
  @abc.abstractmethod
  def output_latency(self) -> int:
    pass

  @abc.abstractmethod
  def get_accumulated_input_latency(self, input_latency: int) -> int:
    pass

  @abc.abstractmethod
  def get_accumulated_output_latency(self, output_latency: int) -> int:
    pass

  @property
  @abc.abstractmethod
  def receptive_field(self) -> Any:
    pass
