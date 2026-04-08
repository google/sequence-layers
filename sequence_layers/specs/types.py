"""Signatures for the types module.

See the corresponding _behaviors module for behaviors.

If you are adding a new class or method to be implemented per backend, make
sure to add it to the ModuleSpec protocol.
"""

import abc
import enum
import fractions
from types import EllipsisType
from typing import Any, Callable, Concatenate, Generic, Iterable, Literal, Protocol, Self, TypeVar, override, runtime_checkable

import numpy as np
import numpy.typing as npt
import jaxtyping as jt


# NEW
ArrayLike = npt.ArrayLike

Array = jt.Shaped[Any, '...']

# Type aliases for generic usage
T = TypeVar('T')
Shape = tuple[int, ...]
ShapeLike = list[int] | tuple[int, ...]
DType = Any  # Can be numpy, jax, or mlx dtype
ChannelSpec = Any  # Typically ShapeDType or compatible
State = Any
Constants = Any
Emits = Any

# TODO: Do these defaults do anything? apparently not
ValuesT = TypeVar('ValuesT', bound=Array)
MaskT = TypeVar('MaskT', bound=Array)

LengthsT = TypeVar('LengthsT', bound=Array)
# SequenceT = TypeVar('SequenceT', bound='Sequence[Array, Array]', default='Sequence[Array, Array]')
InputT = TypeVar('InputT', bound='Sequence')
OutputT = TypeVar('OutputT', bound='Sequence')


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


class Sequence[ValuesT = Array, MaskT = Array](metaclass=abc.ABCMeta):
  """Abstract base class for Sequence."""

  values: ValuesT
  mask: MaskT

  @abc.abstractmethod
  def __init__(self, values: ValuesT, mask: MaskT):
    ...

  @property
  @abc.abstractmethod
  def shape(self) -> Shape:
    ...

  @property
  @abc.abstractmethod
  def ndim(self) -> int:
    ...

  @property
  @abc.abstractmethod
  def channel_shape(self) -> Shape:
    ...

  @property
  @abc.abstractmethod
  def dtype(self) -> DType:
    ...

  @classmethod
  @abc.abstractmethod
  def from_values(cls, values: ValuesT) -> Self:
    ...

  @classmethod
  @abc.abstractmethod
  def from_lengths(
      cls,
      values: ValuesT,
      lengths: LengthsT,
      is_masked: bool = False,
  ) -> Self:
    ...

  @classmethod
  @abc.abstractmethod
  def concatenate_sequences(cls, sequences: Iterable[Self]) -> Self:
    ...

  @abc.abstractmethod
  def expanded_mask(self) -> Any:
    ...

  @abc.abstractmethod
  def apply_values[NewValuesT: Array, **P](
      self,
      values_fn: Callable[Concatenate[ValuesT, P], NewValuesT],
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> 'Sequence[NewValuesT, MaskT]':
    ...

  @abc.abstractmethod
  def apply_values_masked[NewValuesT: Array, **P](
      self,
      values_fn: Callable[Concatenate[ValuesT, P], NewValuesT],
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> 'Sequence[NewValuesT, MaskT]':
    ...

  @abc.abstractmethod
  def apply[NewValuesT: Array, NewMaskT: Array, **P](
      self,
      apply_fn: Callable[Concatenate[ValuesT, P], tuple[NewValuesT, NewMaskT]],
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> 'Sequence[NewValuesT, NewMaskT]':
    ...

  @abc.abstractmethod
  def apply_masked[NewValuesT: Array, NewMaskT: Array, **P](
      self,
      apply_fn: Callable[Concatenate[ValuesT, P], tuple[NewValuesT, NewMaskT]],
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> 'Sequence[NewValuesT, NewMaskT]':
    ...

  @abc.abstractmethod
  def astype(self, dtype: DType | None) -> Self:
    ...

  @abc.abstractmethod
  def lengths(self) -> Any:
    ...

  @abc.abstractmethod
  def __getitem__(
      self,
      the_slice: slice | tuple[int | slice | None | EllipsisType, ...],
  ) -> Self:
    ...

  @abc.abstractmethod
  def pad_time(
      self,
      pad_left: int,
      pad_right: int,
      valid: bool,
      pad_value: Any | None = None,
  ) -> Self:
    ...

  @abc.abstractmethod
  def concatenate(self, other: Self) -> Self:
    ...

  @abc.abstractmethod
  def mask_invalid(
      self, mask_value: Any | None = None
  ) -> 'Sequence[ValuesT, MaskT]':
    ...

  @abc.abstractmethod
  def unmask(self) -> 'Sequence[ValuesT, MaskT]':
    ...


class MaskedSequence(Sequence[ValuesT, MaskT]):
  """A sequence whose invalid timesteps are masked to zero."""

  @abc.abstractmethod
  def apply_values_masked[NewValuesT: Array, **P](
      self,
      values_fn: Callable[Concatenate[ValuesT, P], NewValuesT],
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> 'MaskedSequence[NewValuesT, MaskT]':
    ...

  @abc.abstractmethod
  def apply_masked[NewValuesT: Array, NewMaskT: Array, **P](
      self,
      apply_fn: Callable[Concatenate[ValuesT, P], tuple[NewValuesT, NewMaskT]],
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> 'MaskedSequence[NewValuesT, NewMaskT]':
    ...


class SequenceLayerConfig(metaclass=abc.ABCMeta):
  """Configuration for a SequenceLayer."""

  @abc.abstractmethod
  def make(self) -> Any:
    """Creates the sequence layer."""

  @abc.abstractmethod
  def copy(self, **kwargs: Any) -> Self:
    """Returns a copy of the config with updated fields."""


class Steppable[InputT = Sequence, OutputT = Sequence](metaclass=abc.ABCMeta):
  """A sequence processing layer that can be executed layerwise or stepwise.

  The backend must implement:
  - layer_with_emits
  - step_with_emits
  """

  @property
  @abc.abstractmethod
  def block_size(self) -> int:
    ...

  @property
  @abc.abstractmethod
  def output_ratio(self) -> fractions.Fraction:
    ...

  @property
  @abc.abstractmethod
  def supports_step(self) -> bool:
    ...

  @property
  @abc.abstractmethod
  def input_latency(self) -> int:
    ...

  @property
  @abc.abstractmethod
  def output_latency(self) -> int:
    ...

  @abc.abstractmethod
  def get_accumulated_input_latency(self, input_latency: int) -> int:
    ...

  @abc.abstractmethod
  def get_accumulated_output_latency(self, output_latency: int) -> int:
    ...

  @abc.abstractmethod
  def layer(
      self, x: InputT, *, training: bool, constants: Constants | None = None
  ) -> OutputT:
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

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: InputT,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, Emits]:
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

  @abc.abstractmethod
  def step(
      self,
      x: InputT,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, State]:
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

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: InputT,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, State, Emits]:
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
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to array or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        attention layer this may contain the source sequence to attend to.

    Returns:
      An integer, shape, or structure of integer/shapes.
    """

  @abc.abstractmethod
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

  @property
  @abc.abstractmethod
  def receptive_field(self) -> Any:
    ...


class SequenceLayer[InputT = Sequence, OutputT = Sequence](
    Steppable[InputT, OutputT]
):
  """Base class for Sequence Layers."""

  ...


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class PreservesType(SequenceLayer):
  """A mix-in for layers that do not change the input dtype."""

  @abc.abstractmethod
  @override
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    ...


class PreservesShape[InputT = Sequence, OutputT = Sequence](
    SequenceLayer[InputT, OutputT]
):
  """A mix-in for layers that do not change the input channel shape."""

  @abc.abstractmethod
  @override
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    ...


# ---------------------------------------------------------------------------
# Stateless variants
# ---------------------------------------------------------------------------


class Stateless[InputT = Sequence, OutputT = Sequence](
    SequenceLayer[InputT, OutputT]
):
  """A layer with no state over time required for step-wise processing.

  The backend must implement:
  - get_initial_state
  - step
  Further sub-classes must only implement:
  - layer
  - get_output_shape
  - get_output_dtype
  """

  @abc.abstractmethod
  @override
  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    ...

  @abc.abstractmethod
  @override
  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    ...

  @abc.abstractmethod
  @override
  def layer(
      self,
      x: InputT,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> OutputT:
    ...

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
  def step(
      self,
      x: InputT,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, State]:
    ...


class StatelessPointwise[InputT = Sequence, OutputT = Sequence](
    PreservesShape[InputT, OutputT], Stateless[InputT, OutputT]
):
  """Stateless layer that operates pointwise (preserves shape)."""


class StatelessPointwiseFunctor[InputT = Sequence, OutputT = Sequence](
    StatelessPointwise[InputT, OutputT]
):
  """Stateless pointwise layer defined by a fn(values, mask).

  The backend must implement:
  - layer
  Further sub-classes must only implement:
  - fn
  - mask_required
  """

  @abc.abstractmethod
  def fn(self, values: Any, mask: Any) -> tuple[Any, Any]:
    """Transforms each scalar in values independently."""

  @property
  @abc.abstractmethod
  def mask_required(self) -> bool:
    """Returns true if fn can change the sequence's masked state.

    If fn(0) -> 0, then mask_required() is False.
    """

  @abc.abstractmethod
  @override
  def layer(
      self,
      x: InputT,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> OutputT:
    ...


# ---------------------------------------------------------------------------
# Emitting variants
# ---------------------------------------------------------------------------


class Emitting[InputT = Sequence, OutputT = Sequence](
    SequenceLayer[InputT, OutputT]
):
  """A Steppable layer that emits auxiliary arrays.

  This is a convenience subclass that implements step and layer in terms of
  step_with_emits and layer_with_emits.

  The backend must implement:
  - step
  - layer
  Further sub-classes must only implement:
  - step_with_emits
  - layer_with_emits
  """

  @abc.abstractmethod
  def step(
      self,
      x: InputT,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, State]:
    ...

  @abc.abstractmethod
  def layer(
      self,
      x: InputT,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> OutputT:
    ...

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: InputT,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, State, Emits]:
    ...

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: InputT,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, Emits]:
    ...


class StatelessEmitting[InputT = Sequence, OutputT = Sequence](
    Emitting[InputT, OutputT]
):
  """A Steppable layer with no state over time that emits auxiliary arrays.

  The backend must implement:
  - get_initial_state
  - step_with_emits
  Further sub-classes must only implement:
  - layer_with_emits
  - get_output_shape
  - get_output_dtype
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
  def step_with_emits(
      self,
      x: InputT,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, State, Emits]:
    ...

  @abc.abstractmethod
  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    ...

  @abc.abstractmethod
  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    ...

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: InputT,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[OutputT, Emits]:
    ...


@runtime_checkable
class ModuleSpec(Protocol):
  """Specification for sequence_layers.<backend>.types"""

  @property
  def Sequence(self) -> type[Sequence]:
    ...

  @property
  def MaskedSequence(self) -> type[MaskedSequence]:
    ...

  @property
  def SequenceLayer(self) -> type[SequenceLayer]:
    ...

  @property
  def SequenceLayerConfig(self) -> type[SequenceLayerConfig]:
    ...

  @property
  def Steppable(self) -> type[Steppable]:
    ...

  @property
  def PreservesShape(self) -> type[PreservesShape]:
    ...

  @property
  def Stateless(self) -> type[Stateless]:
    ...

  @property
  def StatelessPointwise(self) -> type[StatelessPointwise]:
    ...

  @property
  def StatelessPointwiseFunctor(self) -> type[StatelessPointwiseFunctor]:
    ...

  @property
  def PreservesType(self) -> type[PreservesType]:
    ...

  @property
  def Emitting(self) -> type[Emitting]:
    ...

  @property
  def StatelessEmitting(self) -> type[StatelessEmitting]:
    ...


__all__ = [
    name
    for name, attr in ModuleSpec.__dict__.items()
    if isinstance(attr, property)
]
