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
"""Basic sequence types and hierarchy for MLX."""

import abc
import dataclasses
import fractions
import functools
import math
from typing import Callable, Generic, Iterable, TypeVar, override

from mlx import nn
import mlx.core as mx
import numpy as np
from sequence_layers.abstract import types

# Type aliases.
MASK_DTYPE = mx.bool_

ValuesT = TypeVar('ValuesT', bound=mx.array)
MaskT = TypeVar('MaskT', bound=mx.array)
LengthsT = TypeVar('LengthsT', bound=mx.array)
ExpandedMaskT = TypeVar('ExpandedMaskT', bound=mx.array)
SequenceSelf = TypeVar('SequenceSelf', bound='Sequence')

Shape = tuple[int, ...]
ShapeLike = list[int] | tuple[int, ...]
DType = np.dtype
State = object  # Any pytree.
Constants = dict[str, object]
Emits = object

# Receptive field.
ReceptiveField = tuple[float | int, float | int] | None


__all__ = (
    # go/keep-sorted start
    'ChannelSpec',
    'Constants',
    'DType',
    'Emits',
    'Emitting',
    'ExpandedMaskT',
    'LengthsT',
    'MASK_DTYPE',
    'MaskT',
    'MaskedSequence',
    'PaddingMode',
    'PreservesShape',
    'PreservesType',
    'ReceptiveField',
    'Sequence',
    'SequenceLayer',
    'SequenceLayerConfig',
    'Shape',
    'ShapeDType',
    'ShapeLike',
    'State',
    'Stateless',
    'StatelessEmitting',
    'StatelessPointwise',
    'StatelessPointwiseFunctor',
    'Steppable',
    'ValuesT',
    'check_layer',
    'check_step',
    # go/keep-sorted end
)


class ShapeDType:
  """Lightweight replacement for jax.ShapeDtypeStruct."""

  def __init__(self, shape: Shape, dtype: DType):
    self.shape = shape
    self.dtype = dtype

  def __repr__(self) -> str:
    return f'ShapeDType(shape={self.shape}, dtype={self.dtype})'

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, ShapeDType):
      return NotImplemented
    return self.shape == other.shape and self.dtype == other.dtype

  def __hash__(self) -> int:
    return hash((self.shape, self.dtype))


ChannelSpec = ShapeDType

PaddingMode = types.PaddingMode


def sequence_mask(lengths: LengthsT, maxlen: int) -> MaskT:
  return mx.arange(maxlen)[None, :] < mx.array(lengths)[:, None]


class Sequence(types.Sequence[ValuesT, MaskT], Generic[ValuesT, MaskT]):
  """A generic sequence container that preserves masking information."""

  values: ValuesT
  mask: MaskT

  def __init__(self, values: ValuesT, mask: MaskT):
    self.values = values
    self.mask = mask

  @property
  @override
  def shape(self) -> Shape:
    """Returns the shape of the sequence values."""
    return self.values.shape

  @property
  @override
  def ndim(self) -> int:
    """Returns the rank of the sequence values."""
    return self.values.ndim

  @property
  @override
  def channel_shape(self) -> Shape:
    """Returns the channel shape (the shape without batch and time)."""
    return self.values.shape[2:]

  @property
  def channel_spec(self) -> ChannelSpec:
    """Returns a "spec" for this sequence (the channel shape and dtype)."""
    return ChannelSpec(self.channel_shape, self.dtype)

  @property
  @override
  def dtype(self) -> DType:
    """Returns the dtype of the sequence values."""
    return self.values.dtype

  @classmethod
  @override
  def from_values(cls, values: ValuesT) -> 'MaskedSequence':
    """Returns a MaskedSequence assuming every timestep is valid."""
    if values.ndim < 2:
      raise ValueError(f'Expected {values.ndim=} to be at least 2.')
    return MaskedSequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))

  @classmethod
  @override
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
    seq_type = MaskedSequence if all_masked else Sequence
    return seq_type(
        mx.concatenate(values, axis=1),
        mx.concatenate(masks, axis=1),
    )

  @override
  def expanded_mask(self) -> ExpandedMaskT:
    """Returns the Sequence mask expanded to match values rank."""
    return self.mask.reshape(self.mask.shape + (1,) * (self.values.ndim - 2))

  @override
  def apply_values(
      self,
      values_fn: Callable[..., ValuesT],
      *args,
      **kwargs,
  ) -> 'Sequence':
    """Transforms values, assuming result is unmasked."""
    return Sequence(values_fn(self.values, *args, **kwargs), self.mask)

  @override
  def apply_values_masked(
      self: SequenceSelf,
      values_fn: Callable[..., ValuesT],
      *args,
      **kwargs,
  ) -> SequenceSelf:
    """Transforms values, preserving masked state."""
    return type(self)(values_fn(self.values, *args, **kwargs), self.mask)

  @override
  def apply(
      self,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args,
      **kwargs,
  ) -> 'Sequence':
    """Transforms values/mask, assuming result is unmasked."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return Sequence(values, mask)

  @override
  def apply_masked(
      self: SequenceSelf,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args,
      **kwargs,
  ) -> SequenceSelf:
    """Transforms values/mask, preserving masked state."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return type(self)(values, mask)

  @override
  def astype(self: SequenceSelf, dtype: DType | None) -> SequenceSelf:
    """Returns a copy with values cast to dtype."""
    if dtype is None:
      return self
    return type(self)(self.values.astype(dtype), self.mask)

  @override
  def lengths(self) -> mx.array:
    """Returns the number of valid timesteps per batch item."""
    return mx.sum(self.mask.astype(mx.int32), axis=1)

  @override
  def __getitem__(
      self: SequenceSelf,
      the_slice,
  ) -> SequenceSelf:
    """Slices the Sequence values and mask."""
    if isinstance(the_slice, slice):
      the_slice = (the_slice,)
    return type(self)(
        self.values.__getitem__(the_slice),
        self.mask.__getitem__(the_slice[:2]),
    )

  @override
  def pad_time(
      self: SequenceSelf,
      pad_left: int,
      pad_right: int,
      valid: bool,
      pad_value: float | None = None,
  ) -> SequenceSelf:
    """Pads this sequence with timesteps on the left and right."""
    if not pad_left and not pad_right:
      return self
    pad_val = 0.0 if pad_value is None else pad_value
    values_rank = self.values.ndim
    values = mx.pad(
        self.values,
        [(0, 0), (pad_left, pad_right)] + [(0, 0)] * (values_rank - 2),
        constant_values=pad_val,
    )
    mask = mx.pad(
        self.mask,
        [(0, 0), (pad_left, pad_right)],
        constant_values=valid,
    )
    return type(self)(values, mask)

  @override
  def concatenate(self, other: 'Sequence') -> 'Sequence':
    """Concatenates with other on the time dimension."""
    values = mx.concatenate([self.values, other.values], axis=1)
    mask = mx.concatenate([self.mask, other.mask], axis=1)
    return_type = type(self) if type(self) is type(other) else Sequence
    return return_type(values, mask)

  @override
  def mask_invalid(self, mask_value: complex | None = None) -> 'Sequence':
    """Returns a sequence with invalid timesteps replaced."""
    raise NotImplementedError('Replaced below.')

  @override
  def unmask(self) -> 'Sequence':
    """Returns an unmasked version with unchanged values."""
    return self


class MaskedSequence(Sequence[ValuesT, MaskT], Generic[ValuesT, MaskT]):
  """Sequence whose invalid timesteps are masked to zero."""

  @override
  def mask_invalid(self, mask_value: complex | None = None) -> 'Sequence':
    if mask_value is None:
      return self
    return mask_invalid(self, mask_value)

  @override
  def unmask(self) -> Sequence:
    return Sequence(self.values, self.mask)


def mask_invalid(
    sequence: Sequence,
    mask_value: complex | None = None,
) -> 'Sequence':
  """Returns a sequence with invalid timesteps replaced."""
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


# Defined outside of Sequence so mask_invalid can return MaskedSequence.
Sequence.mask_invalid = mask_invalid

# ---------------------------------------------------------------------------
# Check decorators
# ---------------------------------------------------------------------------


def _check_output_spec(layer, x, y, constants):
  expected = layer.get_output_spec(x.channel_spec, constants=constants)
  if y.channel_shape != expected.shape:
    raise ValueError(
        f'{layer.__class__.__name__} produced output'
        f' ({y.channel_spec}) for input ({x.channel_spec}),'
        ' whose shape does not match get_output_spec'
        f' ({expected}).'
    )


def _check_output_ratio(layer, x, y):
  expected_length = x.shape[1] * layer.output_ratio
  if y.shape[1] != expected_length:
    raise ValueError(
        f'{layer.__class__.__name__} produced output ({y.shape})'
        f' for input ({x.shape}), whose length does not equal'
        f' {expected_length} (output_ratio={layer.output_ratio}).'
    )


def check_layer(layer_fn):
  """Validates layer inputs and outputs."""

  @functools.wraps(layer_fn)
  def wrapper(self, x, *, constants=None):
    y = layer_fn(self, x, constants=constants)
    _check_output_spec(self, x, y, constants)
    return y

  return wrapper


def check_step(step_fn):
  """Validates step inputs and outputs."""

  @functools.wraps(step_fn)
  def wrapper(self, x, state, *, constants=None):
    if not self.supports_step:
      raise ValueError(f'{self.__class__.__name__} does not support step().')
    block_size = self.block_size
    if x.shape[1] % block_size != 0:
      raise ValueError(
          f'{self.__class__.__name__} received input with'
          f' {x.shape=} not a multiple of {block_size=}.'
      )
    y, state = step_fn(self, x, state, constants=constants)
    _check_output_spec(self, x, y, constants)
    _check_output_ratio(self, x, y)
    return y, state

  return wrapper


# ---------------------------------------------------------------------------
# Steppable ABC
# ---------------------------------------------------------------------------


class Steppable(types.Steppable):
  """A sequence processing layer that supports layer and step modes."""

  @property
  @override
  def block_size(self) -> int:
    return 1

  @property
  @override
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1)

  @property
  @override
  def supports_step(self) -> bool:
    return True

  @property
  @override
  def input_latency(self) -> int:
    return 0

  @property
  @override
  def output_latency(self) -> int:
    return int(self.input_latency * self.output_ratio)

  @override
  def get_accumulated_input_latency(self, input_latency: int) -> int:
    return math.ceil(input_latency / self.output_ratio) + self.input_latency

  @override
  def get_accumulated_output_latency(self, output_latency: int) -> int:
    output_ratio = self.output_ratio
    if required_delay := -output_latency % (1 / output_ratio):
      path = (
          '/'.join(self.path)
          if hasattr(self, 'path')
          else self.__class__.__name__
      )
      raise ValueError(
          f'Input to {self.__class__.__name__}(path={path!r}) has a step-wise'
          f' incoming {output_latency=} which is not divisible'
          f" by the layer's {output_ratio=}. Insert a delay of"
          f' -output_latency % (1/output_ratio)={required_delay} before the'
          ' layer to compensate.'
      )
    return int(output_latency * output_ratio) + self.output_latency

  @property
  @override
  def receptive_field(self) -> ReceptiveField:
    raise NotImplementedError(
        'receptive_field is not implemented by MLX Steppable.'
    )

  @abc.abstractmethod
  @override
  def layer(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> Sequence:
    """Process this layer layer-wise."""

  def layer_with_emits(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> tuple[Sequence, Emits]:
    return self.layer(x, constants=constants), ()

  @abc.abstractmethod
  @override
  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    """Process this layer step-wise."""

  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    y, state = self.step(x, state, constants=constants)
    return y, state, ()

  @abc.abstractmethod
  @override
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> State:
    """Returns the initial state for step-wise processing."""

  @abc.abstractmethod
  @override
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    """Returns the output channel shape for an input channel shape."""

  @abc.abstractmethod
  @override
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    """Returns the output dtype for an input dtype."""

  def get_output_spec(
      self,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> ChannelSpec:
    shape = self.get_output_shape(input_spec.shape, constants=constants)
    dtype = self.get_output_dtype(input_spec.dtype, constants=constants)
    return ChannelSpec(shape, dtype)


# ---------------------------------------------------------------------------
# SequenceLayer — MLX base
# ---------------------------------------------------------------------------


class SequenceLayer(nn.Module, Steppable):
  """Base MLX Module for Sequence Layers."""


class SequenceLayerConfig(types.SequenceLayerConfig):
  """Base class for SequenceLayer configuration objects."""

  @abc.abstractmethod
  def make(self) -> SequenceLayer:
    """Builds a SequenceLayer from this config."""

  def copy(self, **kwargs) -> 'SequenceLayerConfig':
    """Returns a copy of the config with updated fields."""
    return dataclasses.replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class PreservesType:
  """Mix-in: layer does not change the input dtype."""

  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    del constants
    return input_dtype


class PreservesShape:
  """Mix-in: layer does not change the input channel shape."""

  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    del constants
    return tuple(input_shape)


# ---------------------------------------------------------------------------
# Stateless variants
# ---------------------------------------------------------------------------


class Stateless(SequenceLayer):
  """A SequenceLayer with no step state."""

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> State:
    return ()

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    return self.layer(x, constants=constants), state


class StatelessPointwise(PreservesShape, Stateless):
  """Stateless layer that operates pointwise (preserves shape)."""


class StatelessPointwiseFunctor(StatelessPointwise, metaclass=abc.ABCMeta):
  """Stateless pointwise layer defined by a fn(values, mask)."""

  @abc.abstractmethod
  def fn(self, values: ValuesT, mask: MaskT) -> tuple[ValuesT, MaskT]:
    """Transforms each scalar in values independently."""

  @property
  def mask_required(self):
    return True

  @check_layer
  def layer(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> Sequence:
    if self.mask_required:
      y = x.apply(self.fn)
    else:
      y = x.apply_masked(self.fn)
    # Ensure MaskedSequence -> Sequence conversion for apply.
    if isinstance(y, MaskedSequence) and self.mask_required:
      y = Sequence(y.values, y.mask)
    return y


# ---------------------------------------------------------------------------
# Emitting variants
# ---------------------------------------------------------------------------


class Emitting(SequenceLayer, metaclass=abc.ABCMeta):
  """A SequenceLayer that emits auxiliary tensors."""

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    y, state, _ = self.step_with_emits(x, state, constants=constants)
    return y, state

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    pass

  def layer(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> Sequence:
    y, _ = self.layer_with_emits(x, constants=constants)
    return y

  @abc.abstractmethod
  def layer_with_emits(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> tuple[Sequence, Emits]:
    pass


class StatelessEmitting(Emitting):
  """Stateless layer that emits auxiliary tensors."""

  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    y, emits = self.layer_with_emits(x, constants=constants)
    return y, state, emits

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> State:
    return ()
