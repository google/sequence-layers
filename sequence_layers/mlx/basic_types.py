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
"""Basic sequence types for MLX."""

import dataclasses
from typing import Generic, TypeVar

import mlx.core as mx
import numpy as np

# A rank 2+ tensor of any type.
# Note: MLX does not support jaxtyping-style shape annotations out of the box,
# so we simply bind to mx.array.
ValuesT = TypeVar('ValuesT', bound=mx.array)

# You can also add the others if you need them:
MaskT = TypeVar('MaskT', bound=mx.array)
LengthsT = TypeVar('LengthsT', bound=mx.array)
ExpandedMaskT = TypeVar('ExpandedMaskT', bound=mx.array)
# A "self" type alias to allow Sequence and subclasses to return their own
# Sequence subtype.
SequenceSelf = TypeVar('SequenceSelf', bound='Sequence')
Shape = tuple[int, ...]
DType = np.dtype


def sequence_mask(lengths: LengthsT, maxlen: int) -> MaskT:
  return mx.arange(maxlen)[None, :] < mx.array(lengths)[:, None]


@dataclasses.dataclass(frozen=True)
class ChannelSpec:
  """A specification for the channel shape and dtype of a sequence."""

  shape: Shape
  dtype: DType


class Sequence(Generic[ValuesT, MaskT]):
  """A generic sequence container that preserves masking information."""

  values: ValuesT
  mask: MaskT

  def __init__(self, values: ValuesT, mask: MaskT):
    self.values = values
    self.mask = mask

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
  def dtype(self) -> DType:
    """Returns the dtype of the sequence values."""
    return self.values.dtype

  def expanded_mask(self) -> ExpandedMaskT:
    """Returns the Sequence mask with dimensions expanded to match values."""
    return self.mask.reshape(self.mask.shape + (1,) * (self.values.ndim - 2))

  def mask_invalid(self, mask_value: complex | None = None) -> 'Sequence':
    """Returns a sequence with invalid timesteps replaced with mask_value."""
    raise NotImplementedError('Replaced below.')

  def unmask(self) -> 'Sequence':
    """Returns an unmasked version of this sequence with unchanged values."""
    # We are already an unmasked sequence.
    return self


class MaskedSequence(Sequence[ValuesT, MaskT]):
  """Sequence whose invalid timesteps are masked to zero."""

  def mask_invalid(self, mask_value: complex | None = None) -> 'Sequence':
    """Returns a sequence with invalid timesteps replaced with mask_value."""
    if mask_value is None:
      return self
    else:
      return mask_invalid(self, mask_value)

  def unmask(self) -> Sequence:
    """Returns an unmasked version of this sequence with unchanged values."""
    return Sequence(self.values, self.mask)


def mask_invalid(
    sequence: Sequence,
    mask_value: complex | None = None,
) -> 'Sequence':
  """Returns a sequence whose invalid timesteps are replaced with mask_value."""
  expanded_mask = sequence.expanded_mask()
  if mask_value is None:
    masked_values = mx.zeros_like(sequence.values)
    result_type = MaskedSequence
  else:
    masked_values = mx.full(
        sequence.values.shape, mask_value, sequence.values.dtype
    )
    result_type = Sequence
  masked_values = mx.where(expanded_mask, sequence.values, masked_values)
  return result_type(masked_values, sequence.mask)


# Defined outside of Sequence so that mask_invalid can return a MaskedSequence.
Sequence.mask_invalid = mask_invalid
