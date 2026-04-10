"""Basic sequence types and hierarchy for MLX."""

import abc
import dataclasses
import fractions
import functools
import math
import types
from typing import (Any, Callable, cast, Iterable, MutableMapping, override,
                    Self, TypeVar)

import jaxtyping as jt
from mlx import nn
import mlx.core as mx

from sequence_layers.specs import types as spec

# Type aliases.
MASK_DTYPE = mx.bool_

ValuesT = TypeVar('ValuesT', bound=mx.array)
MaskT = TypeVar('MaskT', bound=mx.array)
LengthsT = TypeVar('LengthsT', bound=mx.array)
ExpandedMaskT = TypeVar('ExpandedMaskT', bound=mx.array)
NewValuesT = TypeVar('NewValuesT', bound=mx.array)
NewMaskT = TypeVar('NewMaskT', bound=mx.array)
SequenceSelf = TypeVar('SequenceSelf', bound='Sequence')

Shape = tuple[int, ...]
ShapeLike = list[int] | tuple[int, ...]
DType = mx.Dtype
State = object  # Any pytree.
Constants = MutableMapping[str, jt.PyTree[mx.array]]
Emits = jt.PyTree[mx.array]

# Receptive field.
ReceptiveField = tuple[float | int, float | int] | None

InputT = TypeVar('InputT', bound='Sequence')
OutputT = TypeVar('OutputT', bound='Sequence')

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

  @override
  def __repr__(self) -> str:
    return f'ShapeDType(shape={self.shape}, dtype={self.dtype})'

  @override
  def __eq__(self, other: object) -> bool:
    if not isinstance(other, ShapeDType):
      return NotImplemented
    return self.shape == other.shape and self.dtype == other.dtype

  def __hash__(self) -> int:
    return hash((self.shape, self.dtype))


ChannelSpec = ShapeDType

PaddingMode = spec.PaddingMode


def sequence_mask(lengths: LengthsT, maxlen: int) -> mx.array:
  """Generates a boolean mask for sequences based on lengths."""
  return mx.arange(maxlen)[None, :] < mx.array(lengths)[:, None]  # pylint: disable=unsubscriptable-object


class Sequence[ValuesT: mx.array, MaskT: mx.array](
    spec.Sequence[ValuesT, MaskT]
):
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
  def from_lengths(
      cls,
      values: ValuesT,
      lengths: LengthsT,
      is_masked: bool = False,
  ) -> 'Sequence':
    """Constructs a sequence from values and per-batch element lengths."""
    values_arr = mx.array(values)
    mask = sequence_mask(lengths, maxlen=values_arr.shape[1])
    return (
        MaskedSequence(values_arr, mask)
        if is_masked
        else Sequence(values_arr, mask)
    )

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
  def expanded_mask(self) -> mx.array:
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
      self,
      values_fn: Callable[..., NewValuesT],
      *args,
      **kwargs,
  ) -> 'Sequence[NewValuesT, MaskT]':
    """Transforms values, preserving masked state."""
    return cast(
        'Sequence[NewValuesT, MaskT]',
        type(self)(values_fn(self.values, *args, **kwargs), self.mask),
    )

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
      self,
      apply_fn: Callable[..., tuple[NewValuesT, NewMaskT]],
      *args,
      **kwargs,
  ) -> 'Sequence[NewValuesT, NewMaskT]':
    """Transforms values/mask, preserving masked state."""
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return cast('Sequence[NewValuesT, NewMaskT]', type(self)(values, mask))

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
      the_slice: slice | tuple[int | slice | None | types.EllipsisType, ...],
  ) -> SequenceSelf:
    """Slices the Sequence values and mask."""
    if isinstance(the_slice, slice):
      the_slice = (the_slice,)
    return type(self)(
        self.values[the_slice],
        self.mask[the_slice[:2]],
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
    if type(self) is type(other):
      return type(self)(values, mask)
    return Sequence(values, mask)

  @override
  def mask_invalid(self, mask_value: complex | None = None) -> 'Sequence':
    """Returns a sequence with invalid timesteps replaced."""
    raise NotImplementedError('Replaced below.')

  @override
  def unmask(self) -> 'Sequence':
    """Returns an unmasked version with unchanged values."""
    return self


class MaskedSequence[ValuesT: mx.array, MaskT: mx.array](
    Sequence[ValuesT, MaskT], spec.MaskedSequence[ValuesT, MaskT]
):
  """Sequence whose invalid timesteps are masked to zero."""

  @override
  def apply_values_masked(
      self,
      values_fn: Callable[..., NewValuesT],
      *args,
      **kwargs,
  ) -> 'MaskedSequence[NewValuesT, MaskT]':
    return cast(
        'MaskedSequence[NewValuesT, MaskT]',
        type(self)(values_fn(self.values, *args, **kwargs), self.mask),
    )

  @override
  def apply_masked(
      self,
      apply_fn: Callable[..., tuple[NewValuesT, NewMaskT]],
      *args,
      **kwargs,
  ) -> 'MaskedSequence[NewValuesT, NewMaskT]':
    values, mask = apply_fn(self.values, self.mask, *args, **kwargs)
    return cast(
        'MaskedSequence[NewValuesT, NewMaskT]', type(self)(values, mask)
    )

  @override
  def mask_invalid(self, mask_value: complex | None = None) -> Sequence:
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
    result_type: type[Sequence] = MaskedSequence
  else:
    masked_values = mx.full(
        sequence.values.shape, mask_value, sequence.values.dtype  # type: ignore[arg-type]
    )
    result_type = Sequence
  masked_values = mx.where(expanded_mask, sequence.values, masked_values)
  return result_type(masked_values, sequence.mask)


# Defined outside of Sequence so mask_invalid can return MaskedSequence.
Sequence.mask_invalid = mask_invalid  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Check decorators
# ---------------------------------------------------------------------------


def _check_output_spec(layer, x, y, constants):
  """Checks that the output spec of a layer matches the expected spec."""
  expected = layer.get_output_spec(x.channel_spec, constants=constants)
  if y.channel_shape != expected.shape:
    raise ValueError(
        f'{layer.__class__.__name__} produced output'
        f' ({y.channel_spec}) for input ({x.channel_spec}),'
        ' whose shape does not match get_output_spec'
        f' ({expected}).'
    )


def _check_output_ratio(layer, x, y):
  """Checks that the output length of a layer matches the expected length."""
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
  def wrapper(self, x, *, training: bool, constants=None):
    y = layer_fn(self, x, training=training, constants=constants)
    _check_output_spec(self, x, y, constants)
    return y

  return wrapper


def check_step(step_fn):
  """Validates step inputs and outputs."""

  @functools.wraps(step_fn)
  def wrapper(self, x, state, *, training: bool, constants=None):
    if not self.supports_step:
      raise ValueError(f'{self.__class__.__name__} does not support step().')
    block_size = self.block_size
    if x.shape[1] % block_size != 0:
      raise ValueError(
          f'{self.__class__.__name__} received input with'
          f' {x.shape=} not a multiple of {block_size=}.'
      )
    y, state = step_fn(self, x, state, training=training, constants=constants)
    _check_output_spec(self, x, y, constants)
    _check_output_ratio(self, x, y)
    return y, state

  return wrapper


# ---------------------------------------------------------------------------
# Steppable ABC
# ---------------------------------------------------------------------------


class Steppable(spec.Steppable[Sequence, Sequence, ChannelSpec]):
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
  - (Optionally) an `emits` output.

  The output `Sequence` is the primary output of the step, while the `emits`
  represent "auxiliary" outputs that are produced by the layer (for example,
  debug output).

  # Layer-wise execution:

  When executing layer-wise, use the `layer` or `layer_with_emits` method to
  process inputs (a `Sequence` shaped `[b, t, ...]`).

  This produces:
  - An output `Sequence` shaped  `[b,  t * output_ratio, ...]`
    whose `...` shape matches `get_output_shape`.
  - (Optionally) an `emits` output.

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
  y_layer = l.layer(x, training=training)

  # Pad x with input_latency timesteps to process the entire sequence:
  x = x.pad_time(0, l.input_latency, valid=False)

  y_step, _, _ = utils.step_by_step_dynamic(l, x, training=training)
  ```

  The step-wise output is equivalent to the layer-wise output after dropping the
  initial latency timesteps of the step-wise output:

  ```
  y_layer == y_step[:, l.output_latency:]
  ```
  """

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
      self, x: Sequence, *, training: bool, constants: Constants | None = None
  ) -> Sequence:
    """Process this layer layer-wise.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      y: The outputs corresponding to this layer with values shaped
        [b, t_o, ...] where `t_o == t_i * output_ratio`. t_o may have been
        truncated to only represent valid frames.
    """

  @override
  def layer_with_emits(
      self, x: Sequence, *, training: bool, constants: Constants | None = None
  ) -> tuple[Sequence, Emits]:
    """Process this layer layer-wise, producing emitted arrays.

    This is like `layer`, except it has an additional return value which is the
    "emitted" arrays for the layer. The emitted arrays are a structure of
    arrays whose values are arrays or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this layer with values shaped
        [b, t_o, ...] where `t_o == t_i * output_ratio`. t_o may have been
        truncated to only represent valid frames.
      emits: A nest of emitted arrays or Sequences.
    """
    return self.layer(x, training=training, constants=constants), ()

  @abc.abstractmethod
  @override
  def step(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    """Process this layer step-wise.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state arrays matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this step with values shaped [b, t_o, ...]
        where `t_o == t_i * output_ratio`.
      state: A structure of state arrays matching get_initial_state. The
        new state for this layer.
    """

  @override
  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    """Process this layer step-wise, producing emitted arrays.

    This is like `step`, except it has an additional return value which is the
    "emitted" arrays for the step. The emitted arrays are a structure of
    arrays whose values are arrays or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state arrays matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this step with values shaped [b, t_o, ...]
        where `t_o == t_i * output_ratio`.
      state: A structure of state arrays matching get_initial_state. The
        new state for this layer.
      emits: A nest of emitted arrays or Sequences.
    """
    y, state = self.step(x, state, training=training, constants=constants)
    return y, state, ()

  @abc.abstractmethod
  @override
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
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      An integer, shape, or structure of integer/shapes.
    """

  @abc.abstractmethod
  @override
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    """Returns the output channel shape this layer produces for an input channel shape.

    Args:
      input_shape: A shape representing the channels dimension of the input
        sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      A shape representing the output channels dimensions (i.e. not including
      the batch or time dimension).
    """

  @abc.abstractmethod
  @override
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    """Returns the layer's output dtype for the specified input dtype.

    Args:
      input_dtype: The dtype of the input features.
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing.

    Returns:
      The dtype of the output features.
    """

  @override
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
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing.

    Returns:
      The ChannelSpec of the output features.
    """
    shape = self.get_output_shape(input_spec.shape, constants=constants)
    dtype = self.get_output_dtype(input_spec.dtype, constants=constants)
    return ChannelSpec(shape, dtype)


# ---------------------------------------------------------------------------
# SequenceLayer — MLX base
# ---------------------------------------------------------------------------


class SequenceLayer(
    nn.Module,
    Steppable,
    spec.SequenceLayer[Sequence, Sequence, ChannelSpec],
    metaclass=abc.ABCMeta,
):
  """Base Module for Sequence Layers."""


class SequenceLayerConfig(spec.SequenceLayerConfig):
  """Base class for SequenceLayer configuration objects."""

  @abc.abstractmethod
  @override
  def make(self) -> SequenceLayer:
    """Builds a SequenceLayer from this config."""

  @override
  def copy(self, **kwargs) -> Self:
    """Returns a copy of the config with updated fields."""
    return cast(Self, dataclasses.replace(cast(Any, self), **kwargs))


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class PreservesType(
    SequenceLayer,
    spec.PreservesType[Sequence, Sequence, ChannelSpec],
    metaclass=abc.ABCMeta,
):
  """A mix-in for layers that do not change the input dtype."""

  @override
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    del constants
    return input_dtype


class PreservesShape(
    SequenceLayer,
    spec.PreservesShape[Sequence, Sequence, ChannelSpec],
    metaclass=abc.ABCMeta,
):
  """A mix-in for layers that do not change the input shape."""

  @override
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


class Stateless(SequenceLayer, spec.Stateless[Sequence, Sequence, ChannelSpec]):
  """A SequenceLayer with no state over time required for step-wise processing.

  Sub-classes must also implement:
  - layer
  - get_output_shape
  - get_output_dtype
  """

  @override
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
    del training
    del constants
    return ()

  @abc.abstractmethod
  @override
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    ...

  @abc.abstractmethod
  @override
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    ...

  @abc.abstractmethod
  @override
  def layer(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    ...

  @override
  def step(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    return self.layer(x, training=training, constants=constants), state


class StatelessPointwise(
    PreservesShape,
    Stateless,
    spec.StatelessPointwise[Sequence, Sequence, ChannelSpec],
    metaclass=abc.ABCMeta,
):
  """A SequenceLayer that has no state and operates pointwise on its input."""


class StatelessPointwiseFunctor(
    StatelessPointwise,
    spec.StatelessPointwiseFunctor[Sequence, Sequence, ChannelSpec],
):
  """A stateless SequenceLayer for simple pointwise processing fns."""

  @abc.abstractmethod
  @override
  def fn(self, values: ValuesT, mask: MaskT) -> tuple[ValuesT, MaskT]:
    """Transforms each scalar in values independently."""

  @property
  @override
  def mask_required(self):
    """Returns true if fn can change the sequence's masked state.

    If fn(0) -> 0, then mask_required() is False.
    """
    return True

  @check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    del training
    if self.mask_required:
      y = x.apply(self.fn)
    else:
      y = x.apply_masked(self.fn)
    # Ensure MaskedSequence -> Sequence conversion for apply.
    if isinstance(y, MaskedSequence) and self.mask_required:
      y = Sequence(y.values, y.mask)
    return cast(Sequence, y)


# ---------------------------------------------------------------------------
# Emitting variants
# ---------------------------------------------------------------------------


class Emitting(
    SequenceLayer,
    spec.Emitting[Sequence, Sequence, ChannelSpec],
):
  """A SequenceLayer that emits auxiliary arrays.

  This is a convenience subclass that implements step and layer in terms of
  step_with_emits and layer_with_emits, so that implementors need only implement
  two of the four methods. For emits that are substantially expensive to compute
  subclasses can choose to implement all four and save computation in those that
  do not produce emits.
  """

  @abc.abstractmethod
  @override
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> State:
    ...

  @abc.abstractmethod
  @override
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    ...

  @abc.abstractmethod
  @override
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    ...

  @abc.abstractmethod
  @override
  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    ...

  @abc.abstractmethod
  @override
  def layer_with_emits(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    ...

  @override
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

  @override
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


class StatelessEmitting(
    Emitting,
    spec.StatelessEmitting[Sequence, Sequence, ChannelSpec],
):
  """A SequenceLayer with no state over time that emits auxiliary arrays.

  Sub-classes must implement:
  - layer_with_emits
  - get_output_shape
  - get_output_dtype
  """

  @abc.abstractmethod
  @override
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    pass

  @abc.abstractmethod
  @override
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    ...

  @abc.abstractmethod
  @override
  def layer_with_emits(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    ...

  @override
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
    del training
    del constants
    return ()

  @override
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
