# Copyright 2024 Google LLC
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
"""Basic sequence types."""

import abc
import dataclasses
import enum
import fractions
import functools
import typing
from typing import Any, Callable, Generic, Iterable, Literal, MutableMapping, ParamSpec, Self, Sequence as TypingSequence, TypeVar

from absl import logging
from flax import linen as nn
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
import typeguard

from google3.learning.deepmind.jax.typing import typing as jt


__all__ = (
    # go/keep-sorted start
    'ArrayLike',
    'ChannelSpec',
    'Constants',
    'DType',
    'EmitSpecs',
    'Emits',
    'Emitting',
    'ExpandedMaskT',
    'MASK_DTYPE',
    'MaskT',
    'MaskedSequence',
    'PaddingMode',
    'PaddingModeString',
    'PreservesShape',
    'PreservesType',
    'Sequence',
    'SequenceLayer',
    'SequenceLayerConfig',
    'SequenceT',
    'Shape',
    'ShapeDType',
    'ShapeLike',
    'Sharding',
    'State',
    'Stateless',
    'StatelessEmitting',
    'StatelessPointwise',
    'StatelessPointwiseFunctor',
    'Steppable',
    'ValuesT',
    'check_layer',
    'check_layer_with_emits',
    'check_step',
    'check_step_with_emits',
    'validate_padding',
    # go/keep-sorted end
)

UNCONSTRAINED = jax.sharding.PartitionSpec.UNCONSTRAINED
DimSharding = str | TypingSequence[str] | None | type(UNCONSTRAINED)

# Sharding is the *args to jax.sharding.PartitionSpec.
Sharding = TypingSequence[DimSharding] | None

# Sequence type aliases:
MASK_DTYPE = np.bool_

# A rank 2+ tensor of any type.
ValuesT = TypeVar('ValuesT', bound=jt.Shaped[jt.ArrayT, 'B T *C'])

# A boolean batched mask tensor. True indicates a given timepoint is valid, and
# False indicates it is invalid.
MaskT = TypeVar('MaskT', bound=jt.Bool[jt.ArrayT, 'B T'])

# An integer batched lengths tensor.
LengthsT = TypeVar('LengthsT', bound=jt.Int[jt.ArrayT, 'B'])

# A rank 2 boolean tensor with unit dimensions inserted to match their
# corresponding values (e.g. for broadcasting).
ExpandedMaskT = TypeVar('ExpandedMaskT', bound=jt.Bool[jt.ArrayT, 'B T *C'])

# A "self" type alias to allow Sequence and subclasses to return their own
# Sequence subtype.
# TODO(rryan): Remove when PEP-0673 lands.
SequenceSelf = TypeVar('SequenceSelf', bound='Sequence')

# Args and keyword args for Sequence.apply_values.
ApplyValuesParams = ParamSpec('ApplyValuesParams')
ApplyValuesMaskedParams = ParamSpec('ApplyValuesMaskedParams')

# Args and keyword args for Sequence.apply.
ApplyParams = ParamSpec('ApplyParams')
ApplyMaskedParams = ParamSpec('ApplyMaskedParams')


# SequenceLayer type aliases:
State = jt.AnyPyTree
ShapeLike = list[int] | tuple[int, ...]
Shape = tuple[int, ...]
DType = np.dtype
ShapeDType = jax.ShapeDtypeStruct
Constants = MutableMapping[str, jt.AnyPyTree]
Emits = jt.AnyPyTree
EmitSpecs = jt.AnyPyTree
ChannelSpec = ShapeDType
ArrayLike = jax.Array | jax.core.Tracer | jax.core.ShapedArray | np.ndarray

ARRAY_LIKE_TYPES = (
    jax.Array,
    jax.core.Tracer,
    np.ndarray,
    jax.core.ShapedArray,
)


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


PaddingModeString = Literal[
    'valid',
    'same',
    'causal_valid',
    'reverse_causal_valid',
    'causal',
    'reverse_causal',
    'semicausal',
]


def validate_padding(padding: str) -> PaddingModeString:
  """Checks that the provided padding string matches the PaddingMode enum."""
  if padding not in [mode.value for mode in PaddingMode]:
    raise ValueError(
        'Expected padding of "valid", "same", "causal_valid",'
        ' "reverse_causal_valid", "causal", "reverse_causal", or "semicausal".'
        f' Got {padding}'
    )
  return typing.cast(PaddingModeString, padding)


def validate_explicit_padding(padding: TypingSequence[int]) -> tuple[int, int]:
  """Checks that padding is a 2-element sequence of non-negative numbers."""
  padding = tuple(padding)

  if len(padding) != 2:
    raise ValueError(
        f'Expected explicit padding to be a 2 element sequence. Got: {padding}.'
    )

  if padding[0] < 0 or padding[1] < 0:
    raise ValueError(
        f'Expected explicit padding to be non-negative. Got: {padding}.'
    )

  return padding


def sequence_mask(lengths: LengthsT, maxlen: int) -> MaskT:
  return (
      jnp.arange(maxlen)[jnp.newaxis, :] < jnp.asarray(lengths)[:, jnp.newaxis]
  )


class Sequence(Generic[ValuesT, MaskT], struct.PyTreeNode):
  """A generic sequence container that preserves masking information."""

  values: ValuesT
  mask: MaskT

  def __post_init__(self):
    """Applies type and shape checks for values and mask."""
    values_arraylike = isinstance(self.values, ARRAY_LIKE_TYPES)
    mask_arraylike = isinstance(self.mask, ARRAY_LIKE_TYPES)

    # NOTE: We can't have `tuple` in the bad types, as e.g. PartitionSpec
    #   inherits from it.
    bad_types = (int, str, list, set, Sequence, MaskedSequence)
    if isinstance(self.values, bad_types) or isinstance(self.mask, bad_types):
      raise jt.JaxTypeCheckError(
          'Sequence values and mask must be array-like. Got values:'
          f' {type(self.values)} and mask: {type(self.mask)}'
      )

    if isinstance(self.values, ShapeDType) != isinstance(self.mask, ShapeDType):
      raise jt.JaxTypeCheckError(
          'Either Sequence values and mask must be either both ShapeDType or'
          f' neither of them. Got values: {type(self.values)} and mask:'
          f' {type(self.mask)}'
      )

    if not values_arraylike or not mask_arraylike:
      # Avoid checking types if the values and mask are not array-like. This
      # happens e.g. during various passes that JAX does during
      # compilation/optimization.
      return

    # Check mask dtype is correct.
    #
    # Allow JAX float0 types, which are used in jvp/linearization of booleans.
    # go/jax-integer-autodiff
    if self.mask.dtype not in (MASK_DTYPE, jax.dtypes.float0):
      raise jt.JaxTypeCheckError(
          f'Sequence mask dtype ({self.mask.dtype}) is not {MASK_DTYPE}.'
      )

    if self.mask.ndim < 2 or self.values.ndim < 2:
      raise jt.JaxTypeCheckError(
          f'Expected values rank ({self.values.ndim}) and mask rank'
          f' ({self.mask.ndim}) to be at least 2.'
      )

    # Since Sequence is a struct.PyTree, and we pass Sequence through
    # transformations like scan, values and mask might get stacked to have a
    # higher rank batch dimension. For this reason, all we do here is check that
    # the values and mask shapes are coherent (mask shape is a prefix of values
    # shape).
    # TODO(rryan): We may want to allow broadcastable shapes at some point.
    if self.values.shape[: self.mask.ndim] != self.mask.shape:
      raise jt.JaxTypeCheckError(
          f'Sequence values shape ({self.values.shape}) does not match mask'
          f' shape ({self.mask.shape}).'
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
  def dtype(self) -> DType:
    """Returns the dtype of the sequence values."""
    return self.values.dtype

  @classmethod
  def from_lengths(
      cls, values: ValuesT, lengths: LengthsT, is_masked: bool = False
  ) -> 'Sequence':
    values = jnp.asarray(values)
    mask = sequence_mask(lengths, maxlen=values.shape[1])
    return MaskedSequence(values, mask) if is_masked else Sequence(values, mask)

  @classmethod
  def from_values(cls, values: ValuesT) -> 'MaskedSequence':
    """Returns a MaskedSequence for values, assuming every timestep is valid."""
    if values.ndim < 2:
      raise ValueError(f'Expected {values.ndim=} to be at least 2.')
    return MaskedSequence(values, jnp.ones(values.shape[:2], jnp.bool_))

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
    if all_masked:
      sequence_type = MaskedSequence
    else:
      sequence_type = Sequence
    return sequence_type(
        jnp.concatenate(values, axis=1), jnp.concatenate(masks, axis=1)
    )

  def expanded_mask(self) -> ExpandedMaskT:
    """Returns the Sequence mask with dimensions expanded to match values."""
    return self.mask.reshape(self.mask.shape + (1,) * (self.values.ndim - 2))

  def concatenate(self, other: 'Sequence') -> 'Sequence':
    """Concatenates this sequence with other on the time dimension."""
    values = jnp.concatenate([self.values, other.values], axis=1)
    mask = jnp.concatenate([self.mask, other.mask], axis=1)

    # Preserve mask state if both sequences are masked.
    return_type = type(self) if type(self) is type(other) else Sequence
    return return_type(values, mask)

  def apply_values(
      self,
      values_fn: Callable[..., ValuesT],
      *args: ApplyValuesParams.args,
      **kwargs: ApplyValuesParams.kwargs,
  ) -> 'Sequence':
    """Transforms values with values_fn, assuming result is unmasked."""
    return Sequence(values_fn(self.values, *args, **kwargs), self.mask)

  def apply_values_masked(
      self: SequenceSelf,
      values_fn: Callable[..., ValuesT],
      *args: ApplyValuesMaskedParams.args,
      **kwargs: ApplyValuesMaskedParams.kwargs,
  ) -> SequenceSelf:
    """Transforms values with values_fn, preserving masked state."""
    return type(self)(values_fn(self.values, *args, **kwargs), self.mask)

  def apply(
      self,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args: ApplyParams.args,
      **kwargs: ApplyParams.kwargs,
  ) -> 'Sequence':
    """Transforms values/mask with apply_fn, assuming result is unmasked."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return Sequence(values, mask)

  def apply_masked(
      self: SequenceSelf,
      apply_fn: Callable[..., tuple[ValuesT, MaskT]],
      *args: ApplyMaskedParams.args,
      **kwargs: ApplyMaskedParams.kwargs,
  ) -> SequenceSelf:
    """Transforms values/mask with apply_fn, preserving masked state."""
    # TODO(rryan): Dig into bug preventing the use of
    # Callable[Concatenate[ValuesT, MaskT, ApplyMaskedParams], tuple[ValuesT,
    # MaskT]] for apply_fn.
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return type(self)(values, mask)

  def astype(
      self: SequenceSelf,
      dtype: DType | None,
  ) -> SequenceSelf:
    """Returns a copy of this sequence with its values cast to dtype."""
    return type(self)(self.values.astype(dtype), self.mask)

  def lengths(self) -> jt.Int[jt.ArrayT, 'B']:
    """Returns the number of valid timesteps per batch item."""
    return jnp.sum(self.mask.astype(jnp.int32), axis=1)

  def __getitem__(
      self: SequenceSelf,
      the_slice: slice | tuple[int | slice | None | type(Ellipsis), ...],
  ) -> SequenceSelf:
    """Slices the Sequence values and mask with the provided slice."""
    if isinstance(the_slice, slice):
      the_slice = (the_slice,)
    if not all(isinstance(dim, slice) for dim in the_slice[:2]):
      raise ValueError(
          'Sequence[...] must only be used to slice, not index, batch and time '
          'dimensions. Got: %s'
          % repr(the_slice)
      )
    return type(self)(
        self.values.__getitem__(the_slice), self.mask.__getitem__(the_slice[:2])
    )

  def pad_time(
      self: SequenceSelf,
      pad_left: jt.ScalarInt,
      pad_right: jt.ScalarInt,
      valid: bool,
      pad_value: jt.Scalar | None = None,
  ) -> SequenceSelf:
    """Pads this sequence with timesteps on the left and right.

    Args:
      pad_left: Amount to pad on the left.
      pad_right: Amount to pad on the right.
      valid: Whether padded values are considered valid (i.e. unmasked).
      pad_value: An optional pad value to use for padding.

    Returns:
      A sequence of the same time with the padded values appended.
    """
    values_rank = self.values.ndim
    pad_value = 0.0 if pad_value is None else pad_value
    if not pad_left and not pad_right:
      return self
    values = jnp.pad(
        self.values,
        [[0, 0], [pad_left, pad_right]] + [[0, 0]] * (values_rank - 2),
        constant_values=jnp.asarray(pad_value).astype(self.values.dtype),
    )
    mask = jnp.pad(
        self.mask,
        [[0, 0], [pad_left, pad_right]],
        constant_values=valid,
    )
    return type(self)(values, mask)

  def reverse_time(self: SequenceSelf) -> SequenceSelf:
    """Reverses the sequence along the time dimension.

    Note that this only reverses the physical array with no assumptions about
    whether the sequence is contiguous / ragged.

    Returns:
      The sequence with values and mask reversed along the time dimension of the
      physical array.
    """
    return type(self)(
        jnp.flip(self.values, axis=1), jnp.flip(self.mask, axis=1)
    )

  def pad_to_multiple(self, block_size: jt.ScalarInt) -> SequenceSelf:
    pad_length = (
        self.shape[1] + block_size - 1
    ) // block_size * block_size - self.shape[1]
    return self.pad_time(pad_left=0, pad_right=pad_length, valid=False)

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
    masked_values = jnp.zeros_like(sequence.values)
    result_type = MaskedSequence
  else:
    masked_values = jnp.full(
        sequence.values.shape, mask_value, sequence.values.dtype
    )
    result_type = Sequence
  masked_values = jnp.where(expanded_mask, sequence.values, masked_values)
  return result_type(masked_values, sequence.mask)


# Defined outside of Sequence so that mask_invalid can return a MaskedSequence.
Sequence.mask_invalid = mask_invalid


class MetaSequenceT(type):
  """A metaclass for SequenceT to support jaxtyping-style type parameters."""

  def __getitem__(cls, item):
    dtype, shape = item
    axes = shape.split(' ')
    if len(axes) < 2:
      raise ValueError(
          'Shape spec must contain at least two components for the batch and'
          f' time dimensions, got: {shape}.'
      )

    values_dtype = dtype[jt.ArrayT, shape]
    batch_time = ' '.join(axes[:2])
    mask_dtype = jt.Bool[jt.ArrayT, batch_time]
    return (
        Sequence[values_dtype, mask_dtype]
        | MaskedSequence[values_dtype, mask_dtype]
    )


class SequenceT(metaclass=MetaSequenceT):
  pass


def _sequence_checker_fn(value, origin_type, args, memo):
  """Checks that Sequence's values/mask adheres to the type spec."""
  del origin_type

  if not isinstance(value, Sequence):
    raise TypeError(f'{value} is not a Sequence.')

  values_dtype, mask_dtype = args

  try:
    typeguard.check_type_internal(
        value.values,
        values_dtype,
        memo=memo,
    )
  except typeguard.TypeCheckError as exc:
    exc.append_path_element('values')
    raise

  try:
    typeguard.check_type_internal(
        value.mask,
        mask_dtype,
        memo=memo,
    )
  except typeguard.TypeCheckError as exc:
    exc.append_path_element('mask')
    raise


def _sequence_checker_lookup_fn(
    origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]
) -> typeguard.TypeCheckerCallable | None:
  """Lookup function to register Sequence type checker in typeguard."""
  del args
  del extras
  return (
      _sequence_checker_fn
      if origin_type in (Sequence, MaskedSequence)
      else None
  )


def _add_custom_checker_lookup_fn(lookup_fn):
  """Add custom Sequence checker lookup function to typeguard."""
  if hasattr(typeguard, 'checker_lookup_functions'):
    # Recent `typeguard` has different API.
    checker_lookup_fns = typeguard.checker_lookup_functions
  else:
    # TODO(rryan): Remove once typeguard is updated
    checker_lookup_fns = typeguard.config.checker_lookup_functions
  for i, f in enumerate(checker_lookup_fns):
    # Check qualname instead of equality, to avoid many copies when reloading
    # modules from colab.
    if f.__qualname__ == lookup_fn.__qualname__:
      # replace
      checker_lookup_fns[i : i + 1] = [lookup_fn]
      break
  else:  # prepend
    checker_lookup_fns[:0] = [lookup_fn]


_add_custom_checker_lookup_fn(_sequence_checker_lookup_fn)


class Steppable(metaclass=abc.ABCMeta):
  """A sequence processing layer that can be executed layerwise or stepwise.

  # Step-wise execution:

  A SequenceLayer supports step-wise execution if its `supports_step` property
  is true. Most built-in SequenceLayers support step-wise processing by default,
  but may support processing features that are not causal and therefore cannot
  be executed step-by-step (e.g. non-causal convolutions, bidirectional RNNs,
  etc.).

  When executing step-wise, use the `step` or `step_with_emits` method to
  process a block of inputs (a `Sequence` shaped `[b, block_size * n, ...]`) and
  a `state` input whose structure matches `get_initial_state`.

  This produces:
  - An output `Sequence` shaped  `[b, block_size * n * output_ratio, ...]`
    whose `...` shape matches `get_output_shape`.
  - A `state` output whose structure matches `get_initial_state`.
  - (Optionally) an `emits` output whose structure/specs match `get_emit_specs`.

  The output `Sequence` is the primary output of the step, while the `emits`
  represent "auxiliary" outputs that are produced by the layer (for example,
  debug output).

  # Layer-wise execution:

  When executing layer-wise, use the `layer` or `layer_with_emits` method to
  process inputs (a `Sequence` shaped `[b, t, ...]`).

  This produces:
  - An output `Sequence` shaped  `[b,  t * output_ratio, ...]`
    whose `...` shape matches `get_output_shape`.
  - (Optionally) an `emits` output whose structure/specs match `get_emit_specs`.

  The output `Sequence` is the primary output of the layer, while the `emits`
  represent "auxiliary" outputs that are produced by the layer (for example,
  debug output).

  # Latency

  SequenceLayers have an input and output "latency" to describe their latency
  characteristics. Latency is the number of input or output timesteps from
  step-wise excecution that are input or output before the step-wise output of
  the layer matches the layer-wise output of the layer.

  An invariant that all layers must maintain is that for the layer-wise output
  and step-wise output:

  ```
  y_layer = l.layer(x, ...)

  # Pad x with input_latency timesteps to process the entire sequence:
  x = x.pad_time(0, l.input_latency, valid=False)
  y_step = utils.step_by_step_dynamic(l, x, ...)
  ```

  The step-wise output is equivalent to the layer-wise output after dropping the
  initial latency timesteps of the step-wise output:

  ```
  y_layer == y_step[:, l.output_latency:]
  ```
  """

  @property
  def block_size(self) -> int:
    """The block size multiple required by the layer.

    Sequences (`[b, t, ...]`) passed to `step` will come in multiples of
    `block_size` timesteps. In other words, `t % block_size == 0`.

    Returns:
      The layer's block size.
    """
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    """The number of output frames for one input frame."""
    return fractions.Fraction(1)

  @property
  def supports_step(self) -> bool:
    """Returns whether this layer supports the SequenceLayer.step method."""
    return True

  @property
  def input_latency(self) -> int:
    """Returns the input latency of this layer.

    Input latency is defined as the number of input timesteps before the
    step-wise output of the layer matches its layer-wise output.
    """
    return 0

  @property
  def output_latency(self) -> fractions.Fraction:
    """Returns the output latency of this layer.

    Output latency is defined as the number of output timesteps before the
    step-wise output of the layer matches its layer-wise output.
    """
    return self.input_latency * self.output_ratio

  @abc.abstractmethod
  def layer(
      self, x: Sequence, *, training: bool, constants: Constants | None = None
  ) -> Sequence:
    """Process this layer layer-wise.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      y: The outputs corresponding to this layer with values shaped
        [b, t_o, ...] where `t_o == t_i * output_ratio`. t_o may have been
        truncated to only represent valid frames.
    """

  def layer_with_emits(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    """Process this layer layer-wise, producing emitted tensors.

    This is like `layer`, except it has an additional return value which is the
    "emitted" tensors for the laeyr. The emitted tensors are a structure of
    tensors whose spec matches the return value of `get_emit_specs` and whose
    values are `ArrayLike`s or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this layer with values shaped
        [b, t_o, ...] where `t_o == t_i * output_ratio`. t_o may have been
        truncated to only represent valid frames.
      emits: A nest of emitted tensors or Sequences. The nest structure and
        tensor specs match `get_emit_specs`.
    """
    outputs = self.layer(x, training=training, constants=constants)
    return outputs, ()

  def __call__(
      self, x: Sequence, training: bool, constants: Constants | None = None
  ) -> Sequence:
    """For Flax-compatibility, define __call__ as an alias for layer."""
    return self.layer(x, training=training, constants=constants)

  @abc.abstractmethod
  def step(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    """Process this layer step-wise, producing emitted tensors.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state tensors matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this step with values shaped [b, t_o, ...]
        where `t_o == t_i * output_ratio`.
      state: A structure of state tensors matching get_initial_state. The
        new state for this layer.
    """

  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    """Process this layer step-wise, producing emitted tensors.

    This is like `step`, except it has an additional return value which is the
    "emitted" tensors for the step. The emitted tensors are a structure of
    tensors whose spec matches the return value of `get_emit_specs` and whose
    values are `ArrayLike`s or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state tensors matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this step with values shaped [b, t_o, ...]
        where `t_o == t_i * output_ratio`.
      state: A structure of state tensors matching get_initial_state. The
        new state for this layer.
      emits: A nest of emitted tensors or Sequences. The nest structure and
        tensor specs match `get_emit_specs`.
    """
    outputs, state = self.step(x, state, training=training, constants=constants)
    return outputs, state, ()

  @abc.abstractmethod
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> State:
    """Returns the initial state for this SequenceLayer.

    Args:
      batch_size: The batch size to create state for.
      input_spec: An input ChannelSpec representing the channel shape and dtype
        of the input that will be stepped.
      training: Whether we are in training mode.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      An integer, TensorShape or structure of integer/TensorShapes.
    """

  @abc.abstractmethod
  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    """Returns the output shape this layer produces for an input shape.

    Args:
      input_shape: A shape representing the channels dimension of the input
        sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      A shape representing the output channels dimensions (i.e. not including
      the batch or time dimension).
    """

  @nn.nowrap
  def get_output_shape_for_sequence(
      self,
      x: Sequence,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    """Returns the output shape this layer produces for the provided Sequence.

    A convenience wrapper around get_output_shape.

    Args:
      x: Sequence. An input sequence.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      A shape representing the output channels dimensions (i.e. not including
      the batch or time dimension).
    """
    return self.get_output_shape(x.channel_shape, constants=constants)

  @nn.nowrap
  def get_output_spec_for_sequence(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> ChannelSpec:
    """Returns the output spec this layer produces for the provided Sequence.

    A convenience wrapper around get_output_spec.

    Args:
      x: Sequence. An input sequence.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      A ChannelSpec which represents the output channels dimensions (i.e. not
      including the batch or time dimension) and dtype.
    """
    return self.get_output_spec(x.channel_spec, constants=constants)

  @nn.nowrap
  def get_emit_specs_for_sequence(
      self, x: Sequence, *, constants: Constants | None = None
  ) -> EmitSpecs:
    """Returns the emit specs this layer produces for the provided Sequence.

    A convenience wrapper around get_emit_specs.

    Args:
      x: Sequence. An input sequence.
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      A nest of ShapeDType whose structure matches that of the emit structure
      returned from layer_with_emits or step_with_emits. Shapes represent the
      channels dimensions (i.e. not including the batch or time dimension).
    """
    return self.get_emit_specs(x.channel_spec, constants=constants)

  @abc.abstractmethod
  def get_output_dtype(self, input_dtype: DType) -> DType:
    """Returns the layer's output dtype for the specified input dtype."""

  @nn.nowrap
  def get_output_spec(
      self,
      input_spec: ChannelSpec,
      *,
      constants: Constants | None = None,
  ) -> ChannelSpec:
    """Returns the output spec this layer produces for the provided input spec.

    Args:
      input_spec: A ChannelSpec which represents the channels shape and dtype of
        the input sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      A ChannelSpec which represents the output channels dimensions (i.e. not
      including the batch or time dimension) and dtype.
    """
    shape = self.get_output_shape(input_spec.shape, constants=constants)
    dtype = self.get_output_dtype(input_spec.dtype)
    return ChannelSpec(shape, dtype)

  @nn.nowrap
  def get_emit_specs(
      self, input_spec: ChannelSpec, *, constants: Constants | None = None
  ) -> EmitSpecs:
    """Returns the emit specs this layer produces for an input spec.

    Args:
      input_spec: A ChannelSpec which represents the channels shape and dtype of
        the input sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to ArrayLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      A nest of ShapeDtype whose structure matches the emit structure
      returned from `layer_with_emits` or `step_with_emits`. Shapes represent
      the channels dimensions (i.e. not including the batch or time dimension).
    """
    del input_spec
    del constants
    return ()


def _check_step_common(layer: Steppable, x: Sequence) -> None:
  """Checks layer is steppable and step input is a multiple of block size."""
  if not layer.supports_step:
    raise ValueError(f'{layer.__class__.__name__} does not support step().')

  block_size = layer.block_size

  if x.shape[1] % block_size != 0:
    raise ValueError(
        f'{layer.__class__.__name__} received input with shape'
        f' {x.shape=} which is not a multiple of {block_size=}.'
    )


def _check_output_spec(
    layer: Steppable, x: Sequence, y: Sequence, constants: Constants | None
):
  """Checks the layer output matches get_output_spec."""
  expected_output_spec = layer.get_output_spec(
      x.channel_spec, constants=constants
  )
  if y.dtype != expected_output_spec.dtype:
    logging.warning(
        '%s produced output (%s) for input (%s) whose dtype does not match'
        ' get_output_spec (%s)',
        layer.__class__.__name__,
        y.channel_spec,
        x.channel_spec,
        expected_output_spec,
    )

  if y.channel_shape != expected_output_spec.shape:
    raise ValueError(
        f'{layer.__class__.__name__} produced output ({y.channel_spec}) for'
        f' input ({x.channel_spec}), whose shape does not match get_output_spec'
        f' ({expected_output_spec}).'
    )


def _check_output_ratio(layer: Steppable, x: Sequence, y: Sequence):
  output_ratio = layer.output_ratio
  expected_output_length = x.shape[1] * output_ratio
  if y.shape[1] != expected_output_length:
    raise ValueError(
        f'{layer.__class__.__name__} produced output ({y.shape}) for input'
        f' ({x.shape}), whose length does not equal'
        f' {expected_output_length} ({output_ratio=}).'
    )


def check_layer(layer_fn):
  """A decorator that validates layer inputs and outputs."""

  @functools.wraps(layer_fn)
  def check_layer_fn(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    y = layer_fn(self, x, training=training, constants=constants)
    _check_output_spec(self, x, y, constants)

    return y

  return check_layer_fn


def check_layer_with_emits(layer_with_emits_fn):
  """A decorator that validates layer_with_emits inputs and outputs."""

  @functools.wraps(layer_with_emits_fn)
  def check_layer_with_emits_fn(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    # TODO(rryan): Validate emits against get_emit_specs.
    y, emits = layer_with_emits_fn(
        self, x, training=training, constants=constants
    )
    _check_output_spec(self, x, y, constants)

    return y, emits

  return check_layer_with_emits_fn


def check_step(step_fn):
  """A decorator that validates step inputs and outputs."""

  @functools.wraps(step_fn)
  def check_step_fn(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:

    _check_step_common(self, x)

    # TODO(rryan): Validate state pytree does not change.
    y, state = step_fn(self, x, state, training=training, constants=constants)
    _check_output_spec(self, x, y, constants)
    _check_output_ratio(self, x, y)

    return y, state

  return check_step_fn


def check_step_with_emits(step_with_emits_fn):
  """A decorator that validates step_with_emits inputs and outputs."""

  @functools.wraps(step_with_emits_fn)
  def check_step_with_emits_fn(
      self: Steppable,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    _check_step_common(self, x)

    # TODO(rryan): Validate emits against get_emit_specs.
    # TODO(rryan): Validate state pytree does not change.
    y, state, emits = step_with_emits_fn(
        self, x, state, training=training, constants=constants
    )
    _check_output_spec(self, x, y, constants)
    _check_output_ratio(self, x, y)

    return y, state, emits

  return check_step_with_emits_fn


class SequenceLayer(nn.Module, Steppable):
  """Base Module for Sequence Layers."""


class PreservesType:
  """A mix-in for layers that do not change the input dtype."""

  @nn.nowrap
  def get_output_dtype(self, input_dtype: DType) -> DType:
    return input_dtype


class PreservesShape:
  """A mix-in for layers that do not change the input shape."""

  @nn.nowrap
  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    del constants
    return tuple(input_shape)


class Emitting(SequenceLayer, metaclass=abc.ABCMeta):
  """A SequenceLayer that emits auxiliary tensors.

  This is a convenience subclass that implements step and layer in terms of
  step_with_emits and layer_with_emits, so that implementors need only implement
  two of the four methods. For emits that are substantially expensive to compute
  subclasses can choose to implement all four and save computation in those that
  do not produce emits.
  """

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    output, state, _ = self.step_with_emits(
        x, state, training=training, constants=constants
    )
    return output, state

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    pass

  def layer(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    outputs, _ = self.layer_with_emits(
        x, training=training, constants=constants
    )
    return outputs

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    pass

  @abc.abstractmethod
  def get_emit_specs(
      self, input_spec: ChannelSpec, *, constants: Constants | None = None
  ) -> EmitSpecs:
    pass


class Stateless(SequenceLayer):
  """A SequenceLayer with no state over time required for step-wise processing.

  Sub-classes must only implement:
  - layer
  - get_output_shape
  - get_output_dtype
  """

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> State:
    del batch_size
    del input_spec
    del constants
    return ()

  def step(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    return self.layer(x, training=training, constants=constants), state


class StatelessEmitting(Emitting):
  """A SequenceLayer with no state over time that emits auxiliary tensors.

  Sub-classes must only implement:
  - layer_with_emits
  - get_output_shape
  - get_output_dtype
  - get_emit_specs
  """

  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    outputs, emits = self.layer_with_emits(
        x, training=training, constants=constants
    )
    return outputs, state, emits

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> State:
    return ()


class StatelessPointwise(PreservesShape, Stateless):
  """A SequenceLayer that has no state and operates pointwise on its input."""


class StatelessPointwiseFunctor(StatelessPointwise, metaclass=abc.ABCMeta):
  """A stateless SequenceLayer for simple pointwise processing fns."""

  @abc.abstractmethod
  def fn(self, values: ValuesT, mask: MaskT) -> tuple[ValuesT, MaskT]:
    """Transforms each scalar in values independently."""

  @property
  def mask_required(self):
    """Returns true if fn can change the sequence's masked state.

    If fn(0) -> 0, then mask_required() is False.
    """
    return True

  @check_layer
  def layer(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    del training
    # If mask is not required, use Sequence.apply_masked, which preserves x's
    # type (i.e. a MaskedSequence stays a MaskedSequence).
    if self.mask_required:
      y = x.apply(self.fn)
    else:
      y = x.apply_masked(self.fn)
    return y


class SequenceLayerConfig(metaclass=abc.ABCMeta):
  """Base class for SequenceLayer configuration objects.

  Requires a no-argument make() method which returns a SequenceLayer.

  This generic type allows users of SequenceLayers to abstract away the details
  of the specific SequenceLayer in use. This allows easy swapping of
  implementations based on behaviors (steppability, output ratio, latency,
  etc.).
  """

  @abc.abstractmethod
  def make(self) -> SequenceLayer:
    """Builds a SequenceLayer from this config."""

  def copy(self, **kwargs) -> Self:
    """Create a copy of this config.

    Args:
      **kwargs: Keywords which correspond to attributes in the config copy, that
        should be overridden with their respective provided values.

    Returns:
      A new instance of this config, possibly with overridden attributes.

    Raises:
      AttributeError: The provided kwargs are not valid fields of this config.
      TypeError: The implementation does not support this config type. In this
        base implementation, only dataclasses with hashable fields are supported
        (a proxy for immutability, to avoid accidental mutation via e.g. shared
        child configs).
    """
    if not dataclasses.is_dataclass(self):
      raise TypeError(f'SequenceLayerConfig {self=} is not a dataclass.')

    unhashable = dict()
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      if not value.__hash__:
        unhashable[field.name] = value
    if unhashable:
      raise TypeError(
          f'SequenceLayerConfig {self=} has unhashable fields: {unhashable}.'
      )

    try:
      return typing.cast(Self, dataclasses.replace(self, **kwargs))
    except TypeError as type_error:
      raise AttributeError(
          f'Failed to create a copy of the SequenceLayerConfig {self=}; are all'
          f' keys in {kwargs=} fields of this dataclass?'
      ) from type_error
