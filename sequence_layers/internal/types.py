# Copyright 2023 Google LLC
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
"""Type ABC definitions for streaming sequence layers."""

import abc
import enum
import fractions
import re
from typing import Any, Callable, Generator, Iterable, Optional, Union


# Type hint aliases for `Any`, but are used here to represent semantic meaning.
TensorLike = Any
DType = Any
Shape = Any
OpLike = Any
SequenceLike = Union['Sequence', Any]
State = SequenceLike
Constants = Optional[dict[str, SequenceLike]]
# Any nest of Tensors or Sequences.
Emits = SequenceLike


_CAMEL_TO_SNAKE_R = re.compile(r'((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')


def _camel_to_snake(value):
  return _CAMEL_TO_SNAKE_R.sub(r'_\1', value).lower()


class PaddingMode(enum.Enum):
  CAUSAL = 'causal'
  VALID = 'valid'
  SAME = 'same'
  REVERSE_CAUSAL = 'reverse_causal'


def validate_padding(padding: str) -> str:
  if padding not in [mode.value for mode in PaddingMode]:
    raise ValueError(
        'Expected padding of "causal", "valid", "same" or "reverse_causal".'
        f' Got {padding}'
    )
  return padding


class Sequence(metaclass=abc.ABCMeta):
  """A structure representing a sequence and its valid timesteps (a mask).

  Sequences are immutable.
  """

  @classmethod
  @abc.abstractmethod
  def concatenate_sequences(
      cls, sequences: Iterable[SequenceLike]
  ) -> SequenceLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def apply(
      self,
      map_fn: Callable[..., tuple[TensorLike, TensorLike]],
      *args,
      **kwargs,
  ) -> SequenceLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def apply_values(
      self, map_fn: Callable[..., TensorLike], *args, **kwargs
  ) -> SequenceLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def lengths(self) -> TensorLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def mask_invalid(self) -> SequenceLike:
    """Returns a new Sequence with invalid timesteps replaced with zeros."""
    raise NotImplementedError()

  @abc.abstractmethod
  def expanded_mask(self) -> TensorLike:
    """Returns mask reshaped to the same rank as values."""
    raise NotImplementedError()

  @abc.abstractmethod
  def concatenate(self, other: SequenceLike) -> SequenceLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def __getitem__(
      self, the_slice: Union[slice, tuple[slice], tuple[slice, slice]]
  ) -> SequenceLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def pad_time(
      self,
      pad_left: Union[int, TensorLike],
      pad_right: Union[int, TensorLike],
      valid: bool,
      pad_value=0,
  ) -> SequenceLike:
    """Pads this sequence with timesteps on the left and right."""
    raise NotImplementedError()

  @abc.abstractmethod
  def reverse_time(self) -> SequenceLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def print(self, message, summarize):
    """Prints this Sequence with an optional message."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def channel_shape(self) -> Shape:
    """Returns the sequence's channel shape (i.e. excluding batch and time)."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def dtype(self) -> DType:
    """Returns the values dtype."""
    raise NotImplementedError()


class SequenceArray(metaclass=abc.ABCMeta):
  """A SequenceArray is a Array container type for sl.Sequences.

  Typically used to dynamically concatenate Sequences of fixed length into a
  single Sequence in a loop. For example (in Tensorflow):

  sa = sl.SequenceArray.new(tf.float32, size=num_steps)
  for i in range(num_steps):
    input_block = inputs[:, i * layer.block_size:(i + 1) * layer.block_size]
    output, state = layer.step(input_block, state, training=False)
    sa = sa.write(i, output)
  result: Sequence = sa.concat()

  The size of a SequenceArray is the number of Sequence blocks that can be
  written to the SequenceArray, not the total length of the resulting Sequence
  returned from concat.
  """

  @classmethod
  @abc.abstractmethod
  def new(
      cls, dtype: DType, size: TensorLike | None, dynamic_size: bool | None
  ) -> 'SequenceArray':
    """Constructs a new SequenceArray.

    Args:
      dtype: The dtype of the Sequence's values.
      size: The number of Sequence blocks that will be written to this
        SequenceArray.
      dynamic_size: Whether the array should dynamically grow if write indices
        exceed size.

    Returns:
      A SequenceArray containing the appropriate Array typess for the
      Sequence values and mask.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def write(self, index: TensorLike, sequence: SequenceLike) -> 'SequenceArray':
    """Writes the provided Sequence to the specified index.

    Args:
      index: An integer tensor indicating the index to write to.
      sequence: A Sequence to write.

    Returns:
      A new SequenceArray object that ensures the write occurs. Use this object
      for all subsequent operations.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def concat(self) -> SequenceLike:
    """Concatenates the Sequences written to this SequenceArray together."""
    raise NotImplementedError()


class SequenceLayer(metaclass=abc.ABCMeta):
  """A sequence processing layer that can be executed layerwise or stepwise.

  Step-wise execution:

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

  Layer-wise execution:

  When executing layer-wise, use the `layer` or `layer_with_emits` method to
  process inputs (a `Sequence` shaped `[b, t, ...]`).

  This produces:
  - An output `Sequence` shaped  `[b,  t * output_ratio, ...]`
    whose `...` shape matches `get_output_shape`.
  - (Optionally) an `emits` output whose structure/specs match `get_emit_specs`.

  The output `Sequence` is the primary output of the layer, while the `emits`
  represent "auxiliary" outputs that are produced by the layer (for example,
  debug output).
  """

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return _camel_to_snake(cls.__name__)

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

  def supports_step(self) -> bool:
    """Returns whether this layer supports the SequenceLayer.step method."""
    return True

  @abc.abstractmethod
  def step(
      self,
      x: SequenceLike,
      state: State,
      training: bool,
      constants: Constants | None,
  ) -> tuple[SequenceLike, State]:
    """Process this layer step-wise.

    If `supports_step` is False, it is an error to call this function and
    incorrect behavior will result.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state tensors matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to TensorLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this step with values shaped [b, t_o, ...]
        where `t_o == t_i * output_ratio`.
      state: A structure of state tensors matching get_initial_state. The
        new state for this layer.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: SequenceLike,
      state: State,
      training: bool,
      constants: Constants | None,
  ) -> tuple[SequenceLike, State, Emits]:
    """Process this layer step-wise, producing emitted tensors.

    This is like `step`, except it has an additional return value which is the
    "emitted" tensors for the step. The emitted tensors are a structure of
    tensors whose spec matches the return value of `get_emit_specs` and whose
    values are `TensorLikes` or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state tensors matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to TensorLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this step with values shaped [b, t_o, ...]
        where `t_o == t_i * output_ratio`.
      state: A structure of state tensors matching get_initial_state. The
        new state for this layer.
      emits: A nest of emitted tensors or Sequences. The nest structure and
        tensor specs match `get_emit_specs`.
    """
    outputs, state = self.step(x, state, training, constants)
    return outputs, state, ()

  @abc.abstractmethod
  def layer(
      self,
      x: SequenceLike,
      training: bool,
      initial_state: State | None,
      constants: Constants | None,
  ) -> SequenceLike:
    """Process this layer layer-wise.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      initial_state: A structure of state tensors matching get_initial_state.
        The initial state to use for this layer.
      constants: A dictionary of constant name to TensorLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this layer with values shaped
        [b, t_o, ...] where `t_o == t_i * output_ratio`. t_o may have been
        truncated to only represent valid frames.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: SequenceLike,
      training: bool,
      initial_state: State | None,
      constants: Constants | None,
  ) -> tuple[Sequence, Emits]:
    """Process this layer layer-wise, producing emitted tensors.

    This is like `layer`, except it has an additional return value which is the
    "emitted" tensors for the laeyr. The emitted tensors are a structure of
    tensors whose spec matches the return value of `get_emit_specs` and whose
    values are `TensorLikes` or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      initial_state: A structure of state tensors matching get_initial_state.
        The initial state to use for this layer.
      constants: A dictionary of constant name to TensorLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this layer with values shaped
        [b, t_o, ...] where `t_o == t_i * output_ratio`. t_o may have been
        truncated to only represent valid frames.
      emits: A nest of emitted tensors or Sequences. The nest structure and
        tensor specs match `get_emit_specs`.
    """
    outputs = self.layer(x, training, initial_state, constants)
    return outputs, ()

  @abc.abstractmethod
  def get_initial_state(
      self, x: SequenceLike, constants: Constants | None
  ) -> State:
    """Returns the initial state for this SequenceLayer.

    Args:
      x: Sequence. A representative input sequence. Do not rely on its values,
        just shape.
      constants: A dictionary of constant name to TensorLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      An integer, TensorShape or structure of integer/TensorShapes.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_output_shape_for_sequence(
      self, x: SequenceLike, constants: Constants | None
  ) -> Shape:
    """Returns the output shape this layer produces for the provided Sequence.

    A convenience wrapper around get_output_shape.

    Args:
      x: Sequence. An input sequence.
      constants: A dictionary of constant name to TensorLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A TensorShape representing the channels dimensions (i.e. not including the
      batch or time dimension).
    """
    return self.get_output_shape(x.channel_shape, constants=constants)

  @abc.abstractmethod
  def get_output_shape(
      self, input_shape: Shape, constants: Constants | None
  ) -> Shape:
    """Returns the output shape this layer produces for an input shape.

    Args:
      input_shape: A TensorShape representing the channels dimension of the
        input sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to TensorLike or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A TensorShape representing the channels dimensions (i.e. not including the
      batch or time dimension).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_output_dtype(self, input_dtype: DType) -> DType:
    """Returns the layer's output dtype."""
    # Default to no change of dtype in the layer. The dtype altering layers
    # should override this.
    return input_dtype

  @abc.abstractmethod
  def _yield_emits(
      self, emits: Emits
  ) -> Generator[tuple['SequenceLayer', Emits], None, None]:
    """Yields (layer, emits) tuples to allow associating emits with layers."""
    yield self, emits


class Emitting(SequenceLayer, metaclass=abc.ABCMeta):
  """A SequenceLayer that emits auxiliary tensors.

  This is a convenience subclass that implements step and layer in terms of
  step_with_emits and layer_with_emits, so that implementors need only implement
  two of the four methods. For emits that are substantially expensive to compute
  subclasses can choose to implement all four and save computation in those that
  do not produce emits.
  """

  @abc.abstractmethod
  def step(
      self,
      x: SequenceLike,
      state: State,
      training: bool,
      constants: Constants | None,
  ) -> tuple[SequenceLike, State]:
    raise NotImplementedError()

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: SequenceLike,
      state: State,
      training: bool,
      constants: Constants | None,
  ) -> tuple[SequenceLike, State, Emits]:
    raise NotImplementedError()

  @abc.abstractmethod
  def layer(
      self,
      x: SequenceLike,
      training: bool,
      initial_state: State | None,
      constants: Constants | None,
  ) -> SequenceLike:
    raise NotImplementedError()

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: SequenceLike,
      training: bool,
      initial_state: State | None,
      constants: Constants | None,
  ) -> tuple[SequenceLike, Emits]:
    raise NotImplementedError()
