"""Abstract base classes and types for SequenceLayers."""

import abc
import enum
import fractions
from typing import Any, Callable, Generic, Iterable, Literal, TypeVar

import numpy as np

# Type aliases for generic usage
T = TypeVar('T')
ValuesT = TypeVar('ValuesT')
MaskT = TypeVar('MaskT')
SequenceSelf = TypeVar('SequenceSelf', bound='Sequence')
Shape = tuple[int, ...]
ShapeLike = list[int] | tuple[int, ...]
DType = Any  # Can be numpy, jax, or mlx dtype
ChannelSpec = Any  # Typically ShapeDType or compatible
State = Any
Constants = Any
Emits = Any

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
  def from_lengths(
      cls,
      values: ValuesT,
      lengths: Any,
      is_masked: bool = False,
  ) -> 'Sequence':
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
  def __getitem__(
    self: SequenceSelf,
    the_slice: slice | tuple[int | slice | None | type(Ellipsis), ...],
  ) -> SequenceSelf:
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


class MaskedSequence(Sequence[ValuesT, MaskT], metaclass=abc.ABCMeta):
  """A sequence whose invalid timesteps are masked to zero."""
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

  @abc.abstractmethod
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

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
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

  @abc.abstractmethod
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

  @abc.abstractmethod
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
    pass


class SequenceLayer(Steppable):
  """Base class for Sequence Layers."""
  pass


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class PreservesType:
  """A mix-in for layers that do not change the input dtype."""

  @abc.abstractmethod
  def get_output_dtype(
      self,
      input_dtype: DType,
      *,
      constants: Constants | None = None,
  ) -> DType:
    pass


class PreservesShape:
  """A mix-in for layers that do not change the input channel shape."""

  @abc.abstractmethod
  def get_output_shape(
      self,
      input_shape: ShapeLike,
      *,
      constants: Constants | None = None,
  ) -> Shape:
    pass


# ---------------------------------------------------------------------------
# Stateless variants
# ---------------------------------------------------------------------------


class Stateless(SequenceLayer):
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
  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    pass

  @abc.abstractmethod
  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    pass

  @abc.abstractmethod
  def layer(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    pass

  @abc.abstractmethod
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> State:
    pass

  @abc.abstractmethod
  def step(
      self,
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    pass


class StatelessPointwise(PreservesShape, Stateless):
  """Stateless layer that operates pointwise (preserves shape)."""


class StatelessPointwiseFunctor(StatelessPointwise, metaclass=abc.ABCMeta):
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
  def layer(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    pass


# ---------------------------------------------------------------------------
# Emitting variants
# ---------------------------------------------------------------------------


class Emitting(SequenceLayer, metaclass=abc.ABCMeta):
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
      x: Sequence,
      state: State,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    pass

  @abc.abstractmethod
  def layer(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> Sequence:
    pass

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

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    pass


class StatelessEmitting(Emitting):
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
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: ChannelSpec,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> State:
    pass

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

  @abc.abstractmethod
  def get_output_shape(
      self, input_shape: ShapeLike, *, constants: Constants | None = None
  ) -> Shape:
    pass

  @abc.abstractmethod
  def get_output_dtype(
      self, input_dtype: DType, *, constants: Constants | None = None
  ) -> DType:
    pass

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: Sequence,
      *,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    pass
