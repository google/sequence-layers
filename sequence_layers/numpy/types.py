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
"""Basic sequence types using NumPy."""

import dataclasses
from typing import Any, Callable, Generic, Iterable, ParamSpec, TypeVar

import jax
import numpy as np

__all__ = (
    # go/keep-sorted start
    'ArrayLike',
    'ChannelSpec',
    'DType',
    'ExpandedMaskT',
    'MASK_DTYPE',
    'MaskT',
    'MaskedSequence',
    'Sequence',
    'Shape',
    'ShapeDType',
    'ShapeLike',
    'ValuesT',
    # go/keep-sorted end
)

# Sequence type aliases:
MASK_DTYPE = np.bool_

# A rank 2+ tensor of any type.
ValuesT = TypeVar('ValuesT', bound=np.ndarray)

# A boolean batched mask tensor. True indicates a given timepoint is valid, and
# False indicates it is invalid.
MaskT = TypeVar('MaskT', bound=np.ndarray)

# An integer batched lengths tensor.
LengthsT = TypeVar('LengthsT', bound=np.ndarray)

# A rank 2 boolean tensor with unit dimensions inserted to match their
# corresponding values (e.g. for broadcasting).
ExpandedMaskT = TypeVar('ExpandedMaskT', bound=np.ndarray)

# Args and keyword args for Sequence.apply_values.
ApplyValuesParams = ParamSpec('ApplyValuesParams')
ApplyValuesMaskedParams = ParamSpec('ApplyValuesMaskedParams')

# Args and keyword args for Sequence.apply.
ApplyParams = ParamSpec('ApplyParams')
ApplyMaskedParams = ParamSpec('ApplyMaskedParams')

# SequenceLayer type aliases:
State = Any
ShapeLike = list[int] | tuple[int, ...]
Shape = tuple[int, ...]
DType = np.dtype
ShapeDType = jax.ShapeDtypeStruct
ChannelSpec = ShapeDType
ArrayLike = np.typing.ArrayLike


def sequence_mask(lengths: np.ndarray, maxlen: int) -> np.ndarray:
  return np.arange(maxlen)[np.newaxis, :] < np.asarray(lengths)[:, np.newaxis]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Sequence(Generic[ValuesT, MaskT]):
  """A generic sequence container that preserves masking information."""

  values: ValuesT
  mask: MaskT

  def __post_init__(self):
    if not isinstance(self.values, np.ndarray) or not isinstance(
        self.mask, np.ndarray
    ):
      raise ValueError('values and mask must be numpy arrays')

    if self.mask.dtype != np.bool_:
      raise ValueError(f'Mask must be boolean, got {self.mask.dtype}')

    if self.values.ndim < 2 or self.mask.ndim < 2:
      raise ValueError('values and mask must have rank at least 2 (B, T, ...)')

    if self.values.shape[: self.mask.ndim] != self.mask.shape:
      raise ValueError(
          f'Values shape {self.values.shape} does not match mask shape'
          f' {self.mask.shape}'
      )

  @property
  def shape(self) -> Shape:
    """Returns the shape of the sequence values."""
    return self.values.shape

  @property
  def ndim(self) -> int:
    """Returns the rank of the sequence values."""
    return self.values.ndim

  @property
  def channel_shape(self) -> Shape:
    """Returns the channel shape (the shape without batch and time)."""
    return self.values.shape[2:]

  @property
  def channel_spec(self) -> ChannelSpec:
    """Returns a "spec" for this sequence (the channel shape and dtype)."""
    return ChannelSpec(self.channel_shape, self.dtype)

  @property
  def dtype(self) -> np.dtype:
    """Returns the dtype of the sequence values."""
    return self.values.dtype

  @classmethod
  def from_lengths(
      cls, values: ValuesT, lengths: np.ndarray, is_masked: bool = False
  ) -> 'Sequence':
    """Returns a Sequence for values, with masking information from lengths."""
    values = np.asarray(values)
    mask = sequence_mask(lengths, maxlen=values.shape[1])
    return MaskedSequence(values, mask) if is_masked else Sequence(values, mask)

  @classmethod
  def from_values(cls, values: ValuesT) -> 'MaskedSequence':
    """Returns a MaskedSequence for values, assuming every timestep is valid."""
    if values.ndim < 2:
      raise ValueError(f'Expected values.ndim={values.ndim} to be at least 2.')
    return MaskedSequence(values, np.ones(values.shape[:2], dtype=np.bool_))

  @classmethod
  def concatenate_sequences(cls, sequences: Iterable['Sequence']) -> 'Sequence':
    """Concatenates sequences and their masks on the time axis."""
    values = []
    masks = []
    all_masked = True
    for sequence in sequences:
      if not isinstance(sequence, MaskedSequence):
        all_masked = False
      values.append(sequence.values)
      masks.append(sequence.mask)

    sequence_type = MaskedSequence if all_masked else Sequence
    return sequence_type(
        np.concatenate(values, axis=1), np.concatenate(masks, axis=1)
    )

  def expanded_mask(self) -> np.ndarray:
    """Returns the Sequence mask with dimensions expanded to match values."""
    return self.mask.reshape(self.mask.shape + (1,) * (self.values.ndim - 2))

  def concatenate(self, other: 'Sequence') -> 'Sequence':
    """Concatenates this sequence with other on the time dimension."""
    values = np.concatenate([self.values, other.values], axis=1)
    mask = np.concatenate([self.mask, other.mask], axis=1)
    return_type = type(self) if type(self) is type(other) else Sequence
    return return_type(values, mask)

  def apply_values(
      self, values_fn: Callable[..., ValuesT], *args, **kwargs
  ) -> 'Sequence':
    """Transforms values with values_fn, assuming result is unmasked."""
    return Sequence(values_fn(self.values, *args, **kwargs), self.mask)

  def apply_values_masked(
      self, values_fn: Callable[..., ValuesT], *args, **kwargs
  ) -> 'Sequence':
    """Transforms values with values_fn, preserving masked state."""
    return type(self)(values_fn(self.values, *args, **kwargs), self.mask)

  def apply(
      self, apply_fn: Callable[..., tuple[ValuesT, MaskT]], *args, **kwargs
  ) -> 'Sequence':
    """Transforms values/mask with apply_fn, assuming result is unmasked."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return Sequence(values, mask)

  def apply_masked(
      self, apply_fn: Callable[..., tuple[ValuesT, MaskT]], *args, **kwargs
  ) -> 'Sequence':
    """Transforms values/mask with apply_fn, preserving masked state."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return type(self)(values, mask)

  def astype(self, dtype: Any) -> 'Sequence':
    """Returns a copy of this sequence with its values cast to dtype."""
    return type(self)(self.values.astype(dtype), self.mask)

  def lengths(self) -> np.ndarray:
    """Returns the number of valid timesteps per batch item."""
    return np.sum(self.mask.astype(np.int32), axis=1)

  def __getitem__(self, the_slice) -> 'Sequence':
    """Slices the Sequence values and mask with the provided slice."""
    if isinstance(the_slice, slice):
      the_slice = (the_slice,)
    if not all(isinstance(dim, slice) for dim in the_slice[:2]):
      raise ValueError(
          'Sequence[...] must only be used to slice batch and time dimensions.'
      )
    return type(self)(self.values[the_slice], self.mask[the_slice[:2]])

  def pad_time(
      self, pad_left: int, pad_right: int, valid: bool, pad_value: Any = None
  ) -> 'Sequence':
    """Pads this sequence with timesteps on the left and right.

    Args:
      pad_left: Amount to pad on the left.
      pad_right: Amount to pad on the right.
      valid: Whether padded values are considered valid (i.e. unmasked).
      pad_value: An optional pad value to use for padding.

    Returns:
      A sequence of the same time with the padded values appended.
    """
    pad_value = 0.0 if pad_value is None else pad_value
    if not pad_left and not pad_right:
      return self

    values_rank = self.values.ndim
    values_pad = [(0, 0), (pad_left, pad_right)] + [(0, 0)] * (values_rank - 2)
    values = np.pad(
        self.values,
        values_pad,
        constant_values=pad_value,
    )

    mask_pad = [(0, 0), (pad_left, pad_right)]
    mask = np.pad(
        self.mask,
        mask_pad,
        constant_values=valid,
    )

    return_type = type(self)
    # If padding invalid timesteps with a non-zero value, the result is not
    # a MaskedSequence, even if the input was.
    if isinstance(self, MaskedSequence) and not valid and pad_value != 0.0:
      return_type = Sequence

    return return_type(values, mask)

  def reverse_time(self) -> 'Sequence':
    """Reverses the sequence along the time dimension.

    Note that this only reverses the physical array with no assumptions about
    whether the sequence is contiguous / ragged.

    Returns:
      The sequence with values and mask reversed along the time dimension of the
      physical array.
    """
    return type(self)(np.flip(self.values, axis=1), np.flip(self.mask, axis=1))

  def pad_to_multiple(self, block_size: int) -> 'Sequence':
    pad_length = (
        self.shape[1] + block_size - 1
    ) // block_size * block_size - self.shape[1]
    return self.pad_time(pad_left=0, pad_right=pad_length, valid=False)

  def mask_invalid(self, mask_value: Any = None) -> 'Sequence':
    """Returns a sequence with invalid timesteps replaced with mask_value."""
    raise NotImplementedError('Replaced below.')

  def unmask(self) -> 'Sequence':
    """Returns an unmasked version of this sequence with unchanged values."""
    # We are already an unmasked sequence.
    return self


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MaskedSequence(Sequence[ValuesT, MaskT]):
  """Sequence whose invalid timesteps are masked to zero."""

  def mask_invalid(self, mask_value: Any = None) -> 'Sequence':
    """Returns a sequence with invalid timesteps replaced with mask_value."""
    if mask_value is None:
      return self
    else:
      return mask_invalid(self, mask_value)

  def unmask(self) -> Sequence:
    """Returns an unmasked version of this sequence with unchanged values."""
    return Sequence(self.values, self.mask)


def mask_invalid(sequence: Sequence, mask_value: Any = None) -> 'Sequence':
  """Returns a sequence whose invalid timesteps are replaced with mask_value."""
  expanded_mask = sequence.expanded_mask()
  if mask_value is None:
    masked_values = np.zeros_like(sequence.values)
    result_type = MaskedSequence
  else:
    masked_values = np.full(
        sequence.values.shape, mask_value, sequence.values.dtype
    )
    result_type = Sequence
  masked_values = np.where(expanded_mask, sequence.values, masked_values)
  return result_type(masked_values, sequence.mask)


# Defined outside to avoid circular dependency.
Sequence.mask_invalid = mask_invalid
