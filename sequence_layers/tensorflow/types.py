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
"""Type definitions for streaming sequence layers."""

import abc
import sys
from typing import Any, Callable, Generator, Iterable, NamedTuple, Optional, Type, Union

import attr
import numpy as np
from sequence_layers.internal import types as internal
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import type_spec


TensorLike = Union[tf.Tensor, np.ndarray]
# Type hint that is basically Any but represents the semantic meaning.
State = Union[tf.Tensor, 'Sequence', Any]
Constants = Optional[dict[str, Union[tf.Tensor, 'Sequence']]]
# Any nest of Tensors or Sequences.
Emits = Union[tf.Tensor, 'Sequence', Any]
# Any nest of TensorSpecs.
EmitSpecs = Union[tf.TensorSpec, Any]

MASK_DTYPE = tf.float32


def _check_shape_constraints(values, mask):
  """Raises ValueError if values/lengths do not satisfy shape constraints."""
  # Special-case types that are used as part of tf.nest
  # and tf.data.Dataset machinery.
  for tp in [
      tf.TensorArray,
      tf.TensorArraySpec,
      tf.TensorSpec,
      tf.TensorShape,
      tf.DType,
      tf.TypeSpec,
      type,
      bool,
  ]:
    if isinstance(values, tp) and isinstance(mask, tp):
      return
  # Allow variant tensors (TensorArrayV2).
  if (
      isinstance(values, tf.Tensor)
      and values.dtype == tf.variant
      and isinstance(mask, tf.Tensor)
      and mask.dtype == tf.variant
  ):
    return
  if (
      type(values).__name__ == 'PerReplica'
      and type(mask).__name__ == 'PerReplica'
  ):
    return
  # _DotString is used to pretty print some tf.nest error messages.
  if (
      type(values).__name__ == '_DotString'
      and type(mask).__name__ == '_DotString'
  ):
    return
  # Allow empty tuple, which also shows up in nest/tpu.rewrite machinery.
  if values is None and mask is None:
    return
  # Checking the rank in this way allows these checks to work for Tensors,
  # ndarrays, and anything else with a shape attribute.
  try:
    values_rank = len(values.shape)
    mask_rank = len(mask.shape)
  except ValueError as e:
    raise ValueError(
        ' '.join([
            str(e),
            f'str(values)={values} str(mask)={mask}',
            'Sequence has unknown rank. This can happen when e.g. a Tensor is the '  # pylint: disable=implicit-str-concat
            'output of a tf.cond() whose two branches output different'
            ' shapes.',
        ])
    ).with_traceback(sys.exc_info()[2]) from e

  # tf.data.Dataset batching creates a pad_value Sequence with rank 0
  # values and mask, so we have to special-case this.
  if values_rank < 2 and mask_rank != 0:
    raise ValueError(
        '`values` must have rank of at least 2. Found: %s' % str(values)
    )
  if mask_rank != 2:
    raise ValueError('`mask` must have rank of 2. Found: %s' % str(mask))
  elif values_rank and mask_rank == 2:
    # values and mask are Tensor-like (have a shape property) but we can't
    # convert_to_tensor so we can assume shape is a TensorShape.
    def get_dim(tensor, index):
      """Returns tensor's dimension size at position index. None if unknown."""
      dim = tensor.shape[index]
      return dim.value if isinstance(dim, tf1.Dimension) else dim

    values_batch = get_dim(values, 0)
    mask_batch = get_dim(mask, 0)
    # If values_batch or lengths_batch are None, then we can't check whether
    # there is a mismatch. Additionally, to support using SequenceTuples with
    # tf.data, we allow mismatches if either dimension is zero (tf.data's
    # padded_batch machinery builds SequenceTuples with mismatched batch
    # dimension as an intermediate step).
    if values_batch and mask_batch and values_batch != mask_batch:
      raise ValueError(
          'Sequence `values` and `mask` must have equal batch dimension: '
          'values=%s mask=%s' % (values, mask)
      )

    values_time = get_dim(values, 1)
    mask_time = get_dim(mask, 1)
    # If values_batch or lengths_batch are None, then we can't check whether
    # there is a mismatch. Additionally, to support using SequenceTuples with
    # tf.data, we allow mismatches if either dimension is zero (tf.data's
    # padded_batch machinery builds SequenceTuples with mismatched batch
    # dimension as an intermediate step).
    if values_time and mask_time and values_time != mask_time:
      raise ValueError(
          'Sequence `values` and `mask` must have equal time dimension: '
          'values=%s mask=%s' % (values, mask)
      )


PaddingMode = internal.PaddingMode
validate_padding = internal.validate_padding


class SequenceSpec(tf.TypeSpec):
  """A tf.TypeSpec for Sequence objects."""

  __slots__ = ('_spec',)

  def __init__(self, values_spec: tf.TensorSpec, mask_spec: tf.TensorSpec):
    self._spec = (values_spec, mask_spec)

  @property
  def value_type(self) -> Type['Sequence']:
    return Sequence

  def _to_components(self, obj: 'Sequence') -> tuple[tf.Tensor, tf.Tensor]:
    return (obj.values, obj.mask)

  def _from_components(
      self, components: tuple[tf.Tensor, tf.Tensor]
  ) -> 'Sequence':
    return Sequence(*components)

  @property
  def _component_specs(self) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    return self._spec

  def _serialize(self) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    return self._spec

  @classmethod
  def _deserialize(
      cls, encoded: tuple[tf.TensorSpec, tf.TensorSpec]
  ) -> 'SequenceSpec':
    return cls(*encoded)

  @classmethod
  def from_sequence(cls, sequence: 'Sequence') -> 'SequenceSpec':
    return SequenceSpec(
        tf.TensorSpec.from_tensor(sequence.values),
        tf.TensorSpec.from_tensor(sequence.mask),
    )


# TODO(rryan): Switch to dataclass once we turn off Python 2.
@attr.s(frozen=True)
class Sequence(internal.Sequence):
  """A structure representing a sequence and its valid timesteps (a mask).

  Sequences are immutable.
  """

  # TODO(rryan): Switch the types to TensorLike after updating to V2
  # TensorShape (so that values.shape works equivalently to numpy).
  # The sequence values. A [batch_size, time, ...] tensor of any dtype.
  values: tf.Tensor = attr.ib()
  # The sequence mask. A [batch_size, time] tensor of type tf.float32.
  # batch_size and time must match values.
  mask: tf.Tensor = attr.ib()

  def __attrs_post_init__(self):
    _check_shape_constraints(self.values, self.mask)

  @classmethod
  def concatenate_sequences(cls, sequences: Iterable['Sequence']) -> 'Sequence':
    with tf.name_scope('Sequence.concatenate_sequences'):
      values = []
      masks = []
      for sequence in sequences:
        values.append(sequence.values)
        masks.append(sequence.mask)
      return Sequence(tf.concat(values, axis=1), tf.concat(masks, axis=1))

  def apply(
      self, map_fn: Callable[..., tuple[tf.Tensor, tf.Tensor]], *args, **kwargs
  ) -> 'Sequence':
    # Don't create a name scope here to avoid dirtying variable names.
    return Sequence(*map_fn(self.values, self.mask, *args, **kwargs))

  def apply_values(
      self, map_fn: Callable[..., tf.Tensor], *args, **kwargs
  ) -> 'Sequence':
    # Don't create a name scope here to avoid dirtying variable names.
    return Sequence(map_fn(self.values, *args, **kwargs), self.mask)

  def lengths(self) -> tf.Tensor:
    with tf.name_scope('Sequence.lengths'):
      return tf.reduce_sum(tf.cast(self.mask, tf.int32), axis=1)

  def mask_invalid(self) -> 'Sequence':
    """Returns a new Sequence with invalid timesteps replaced with zeros."""
    with tf.name_scope('Sequence.mask_invalid'):
      mask = self.expanded_mask() > 0.0
      return Sequence(
          tf.where(mask, self.values, tf.zeros([], dtype=self.values.dtype)),
          self.mask,
      )

  def expanded_mask(self) -> tf.Tensor:
    """Returns mask reshaped to the same rank as values."""
    with tf.name_scope('Sequence.expanded_mask'):
      return expand_to_rank(self.mask, len(self.values.shape))

  def concatenate(self, other: 'Sequence') -> 'Sequence':
    with tf.name_scope('Sequence.concatenate'):
      values = tf.concat([self.values, other.values], axis=1)
      mask = tf.concat([self.mask, other.mask], axis=1)
      return Sequence(values, mask)

  def __getitem__(
      self, the_slice: Union[slice, tuple[slice], tuple[slice, slice]]
  ) -> 'Sequence':
    """Slices the Sequence values and mask with the provided slice."""
    if isinstance(the_slice, slice):
      the_slice = (the_slice,)
    if len(the_slice) > 2:
      raise ValueError(
          'sl.Sequence[...] can only be used for slicing the sequence batch '
          'and time dimension. Use apply_values to slice channels dimensions. '
          'Got: %s'
          % repr(the_slice)
      )
    if not all(isinstance(dim, slice) for dim in the_slice):
      raise ValueError(
          'sl.Sequence[...] must only be used to slice, not index. Got: %s'
          % repr(the_slice)
      )
    return Sequence(
        self.values.__getitem__(the_slice), self.mask.__getitem__(the_slice)
    )

  def pad_time(
      self,
      pad_left: Union[int, tf.Tensor],
      pad_right: Union[int, tf.Tensor],
      valid: bool,
      pad_value=0,
  ) -> 'Sequence':
    """Pads this sequence with timesteps on the left and right."""
    with tf.name_scope('Sequence.pad_time'):
      values_rank = len(self.values.shape)
      values = tf.pad(
          self.values,
          [[0, 0], [pad_left, pad_right]] + [[0, 0]] * (values_rank - 2),
          constant_values=tf.cast(pad_value, self.values.dtype),
      )
      mask = tf.pad(
          self.mask,
          [[0, 0], [pad_left, pad_right]],
          constant_values=tf.cast(1.0 if valid else 0.0, self.mask.dtype),
      )
      return Sequence(values, mask)

  def reverse_time(self) -> 'Sequence':
    with tf.name_scope('Sequence.reverse_time'):
      return Sequence(
          tf.reverse(self.values, axis=[1], name='values'),
          tf.reverse(self.mask, axis=[1], name='mask'),
      )

  def print(self, message=None, summarize=None):
    """Prints this Sequence with an optional message."""
    with tf.name_scope('Sequence.print'):
      values = [self.values, self.mask]
      if message:
        values.append(message)
      # If in Eager mode or within a tf.function this print happens
      # automatically. In TF1 Graph mode, we need a control dependency to make
      # sure it runs.
      print_op = tf.print(*values, summarize=summarize)
      # Ensure that touching either values or mask causes the print.
      with tf.control_dependencies([print_op]):
        return Sequence(tf.identity(self.values), tf.identity(self.mask))

  def to_spec(self):
    """Returns a Sequence with values/mask replaced with TensorSpecs."""
    return Sequence(
        tf.TensorSpec.from_tensor(self.values),
        tf.TensorSpec.from_tensor(self.mask),
    )

  @property
  def channel_shape(self) -> tf.TensorShape:
    """Returns the sequence's channel shape (i.e. excluding batch and time)."""
    return tf.TensorShape(self.values.shape[2:])

  @property
  def dtype(self) -> tf.DType:
    """Returns the values dtype."""
    return self.values.dtype

  @property
  def channel_spec(self) -> tf.TensorSpec:
    """Returns the sequence's channel spec (i.e. excluding batch and time)."""
    return tf.TensorSpec(self.channel_shape, self.values.dtype)

  def __len__(self):
    """Workaround to enable nest.assert_shallow_structure compatibility."""
    # nest.assert_shallow_structure assumes __len__ is implemented, even though
    # it supports attr.s which does not define a __len__.
    return 2


type_spec.register_type_spec_from_value_converter(
    Sequence, SequenceSpec.from_sequence
)


# TODO(dthkao): This should inherit from internal.SequenceArray, but
# NamedTuples and ABCs do not play nicely.
class SequenceArray(NamedTuple):
  """A SequenceArray is a TensorArray for sl.Sequences.

  Typically used to dynamically concatenate Sequences of fixed length into a
  single Sequence in a loop. For example:

  sa = sl.SequenceArray.new(tf.float32, size=num_steps)
  for i in range(num_steps):
    input_block = inputs[:, i * layer.block_size:(i + 1) * layer.block_size]
    output, state = layer.step(input_block, state, training=False)
    sa = sa.write(i, output)
  result = sa.concat()

  The size of a SequenceArray is the number of Sequence blocks that can be
  written to the SequenceArray, not the total length of the resulting Sequence
  returned from concat.
  """

  values: tf.TensorArray
  mask: tf.TensorArray

  @classmethod
  def new(
      cls,
      dtype: tf.DType,
      size: tf.Tensor | None = None,
      dynamic_size: bool | None = None,
  ) -> 'SequenceArray':
    """Constructs a new SequenceArray.

    Args:
      dtype: The dtype of the Sequence's values.
      size: The number of Sequence blocks that will be written to this
        SequenceArray.
      dynamic_size: Whether the array should dynamically grow if write indices
        exceed size.

    Returns:
      A SequenceArray containing TensorArrays for the Sequence values and mask.
    """
    return SequenceArray(
        tf.TensorArray(dtype, size=size, dynamic_size=dynamic_size),
        tf.TensorArray(MASK_DTYPE, size=size, dynamic_size=dynamic_size),
    )

  def write(self, index: tf.Tensor, sequence: Sequence) -> 'SequenceArray':
    """Writes the provided Sequence to the specified index.

    Args:
      index: An integer Tensor indicating the index to write to.
      sequence: A Sequence to write.

    Returns:
      A new SequenceArray object that ensures the write occurs. Use this object
      for all subsequent operations.
    """
    return SequenceArray(
        self.values.write(index, sequence.values),
        self.mask.write(index, sequence.mask),
    )

  def concat(self) -> Sequence:
    """Concatenates the Sequences written to this SequenceArray together."""
    values = self.values.stack()  # [steps, batch, time, ...]
    mask = self.mask.stack()  # [steps, batch, time]

    def to_batch_major_and_reshape(t):
      shape = t.shape.as_list()
      if not t.shape.is_fully_defined():
        dynamic_shape = tf.shape(t)
        shape = [
            dim if dim else dynamic_shape[i] for i, dim in enumerate(shape)
        ]
      batch_major = tf.transpose(t, [1, 0] + list(range(2, t.shape.rank)))
      return tf.reshape(batch_major, [shape[1], -1] + shape[3:])

    # Transpose to [batch, steps, time, ...] and collapse steps/time.
    return Sequence(
        to_batch_major_and_reshape(values), to_batch_major_and_reshape(mask)
    )


class SequenceLayer(tf.Module, internal.SequenceLayer, metaclass=abc.ABCMeta):
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

  def __init__(self, name=None):
    # Imitate how Keras layers work by creating a unique name from the class
    # name when no layer name is provided.
    if not name:
      name = tf._keras_internal.backend.unique_object_name(
          self._default_name(),
          zero_based=True,
          namespace='audio_hearing_tensorflow_python_sequence_layers',
      )
    super().__init__(name=name)

  def supports_step(self) -> bool:
    """Returns whether this layer supports the SequenceLayer.step method."""
    return True

  @abc.abstractmethod
  def step(
      self,
      x: Sequence,
      state: State,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    """Process this layer step-wise.

    If `supports_step` is False, it is an error to call this function and
    incorrect behavior will result.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state tensors matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this step with values shaped [b, t_o, ...]
        where `t_o == t_i * output_ratio`.
      state: A structure of state tensors matching get_initial_state. The
        new state for this layer.
    """
    pass

  # Despite calling the parent directly, the type annotations in this method
  # definition are more specific
  def step_with_emits(  # pylint: disable=useless-parent-delegation
      self,
      x: Sequence,
      state: State,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    """Process this layer step-wise, producing emitted tensors.

    This is like `step`, except it has an additional return value which is the
    "emitted" tensors for the step. The emitted tensors are a structure of
    tensors whose spec matches the return value of `get_emit_specs` and whose
    values are `tf.Tensors` or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...], where t_i is a
        multiple of block_size.
      state: A structure of state tensors matching get_initial_state. The
        previous state for this layer.
      training: Python bool. Whether we are in training mode.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
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
    return super().step_with_emits(x, state, training, constants)

  @abc.abstractmethod
  def layer(
      self,
      x: Sequence,
      training: bool,
      initial_state: State | None = None,
      constants: Constants | None = None,
  ) -> Sequence:
    """Process this layer layer-wise.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      initial_state: A structure of state tensors matching get_initial_state.
        The initial state to use for this layer.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      y: The outputs corresponding to this layer with values shaped
        [b, t_o, ...] where `t_o == t_i * output_ratio`. t_o may have been
        truncated to only represent valid frames.
    """
    pass

  # Despite calling the parent directly, the type annotations in this method
  # definition are more specific
  def layer_with_emits(  # pylint: disable=useless-parent-delegation
      self,
      x: Sequence,
      training: bool,
      initial_state: State | None = None,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    """Process this layer layer-wise, producing emitted tensors.

    This is like `layer`, except it has an additional return value which is the
    "emitted" tensors for the laeyr. The emitted tensors are a structure of
    tensors whose spec matches the return value of `get_emit_specs` and whose
    values are `tf.Tensors` or `Sequence`s.

    Args:
      x: Input sequence with values shaped [b, t_i, ...].
      training: Python bool. Whether we are in training mode.
      initial_state: A structure of state tensors matching get_initial_state.
        The initial state to use for this layer.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
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
    return super().layer_with_emits(x, training, initial_state, constants)

  @abc.abstractmethod
  def get_initial_state(
      self, x: Sequence, constants: Constants | None = None
  ) -> State:
    """Returns the initial state for this SequenceLayer.

    Args:
      x: Sequence. A representative input sequence. Do not rely on its values,
        just shape.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      An integer, TensorShape or structure of integer/TensorShapes.
    """
    pass

  # Despite calling the parent directly, the type annotations in this method
  # definition are more specific
  def get_output_shape_for_sequence(  # pylint: disable=useless-parent-delegation
      self, x: Sequence, constants: Constants | None = None
  ) -> tf.TensorShape:
    """Returns the output shape this layer produces for the provided Sequence.

    A convenience wrapper around get_output_shape.

    Args:
      x: Sequence. An input sequence.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A TensorShape representing the channels dimensions (i.e. not including the
      batch or time dimension).
    """
    return super().get_output_shape_for_sequence(x, constants)

  def get_output_spec_for_sequence(
      self, x: Sequence, constants: Constants | None = None
  ) -> tf.TensorSpec:
    """Returns the output spec this layer produces for the provided Sequence.

    A convenience wrapper around get_output_spec.

    Args:
      x: Sequence. An input sequence.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A TensorSpec whose shape represents the channels dimensions (i.e. not
      including the batch or time dimension).
    """
    return self.get_output_spec(
        tf.TensorSpec(x.channel_shape, x.dtype), constants=constants
    )

  @abc.abstractmethod
  def get_output_shape(
      self, input_shape: tf.TensorShape, constants: Constants | None = None
  ) -> tf.TensorShape:
    """Returns the output shape this layer produces for an input shape.

    Args:
      input_shape: A TensorShape representing the channels dimension of the
        input sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A TensorShape representing the channels dimensions (i.e. not including the
      batch or time dimension).
    """
    pass

  def get_output_spec(
      self, input_spec: tf.TensorSpec, constants: Constants | None = None
  ) -> tf.TensorSpec:
    """Returns the output spec this layer produces for an input spec.

    Args:
      input_spec: A TensorSpec whose shape represents the channels dimension of
        the input sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A TensorSpec representing the channels dimensions (i.e. not including the
      batch or time dimension).
    """
    return tf.TensorSpec(
        self.get_output_shape(input_spec.shape, constants),
        self.get_output_dtype(input_spec.dtype),
    )

  def get_emit_specs(
      self, input_spec: tf.TensorSpec, constants: Constants | None = None
  ) -> EmitSpecs:
    """Returns the emit specs this layer produces for an input spec.

    Args:
      input_spec: A TensorSpec whose shape represents the channels dimension of
        the input sequence (i.e. not including the batch or time dimension).
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A nest of TensorSpec whose structure matches the emit structure
      returned from `layer_with_emits` or `step_with_emits`. Shapes represent
      the channels dimensions (i.e. not including the batch or time dimension).
    """
    del input_spec
    del constants
    return ()

  def get_emit_specs_for_sequence(
      self, x: Sequence, constants: Constants | None = None
  ) -> EmitSpecs:
    """Returns the emit specs this layer produces for the provided Sequence.

    A convenience wrapper around get_emit_specs.

    Args:
      x: Sequence. An input sequence.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.

    Returns:
      A nest of TensorSpec whose structure matches that of the emit structure
      returned from layer_with_emits or step_with_emits. Shapes represent the
      channels dimensions (i.e. not including the batch or time dimension).
    """
    return self.get_emit_specs(
        tf.TensorSpec(x.channel_shape, x.values.dtype), constants=constants
    )

  # Despite calling the parent directly, the type annotations in this method
  # definition are more specific
  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:  # pylint: disable=useless-parent-delegation
    """Returns the layer's output dtype."""
    return super().get_output_dtype(input_dtype)

  # Despite calling the parent directly, the type annotations in this method
  # definition are more specific
  def _yield_emits(  # pylint: disable=useless-parent-delegation
      self, emits: Emits
  ) -> Generator[tuple['SequenceLayer', Emits], None, None]:
    """Yields (layer, emits) tuples to allow associating emits with layers."""
    yield from super()._yield_emits(emits)


class PreservesShape:
  """A mix-in for layers that do not change the input shape."""

  def get_output_shape(
      self, input_shape: tf.TensorShape, constants: Constants | None = None
  ) -> tf.TensorShape:
    del constants
    if not input_shape.is_fully_defined():
      raise ValueError(
          '%s depends on input shape, but input has unknown '
          'channels dimension: %s' % (self, input_shape)
      )
    return input_shape


class Emitting(SequenceLayer, internal.Emitting, metaclass=abc.ABCMeta):
  """A SequenceLayer that emits auxiliary tensors.

  This is a convenience subclass that implements step and layer in terms of
  step_with_emits and layer_with_emits, so that implementors need only implement
  two of the four methods. For emits that are substantially expensive to compute
  subclasses can choose to implement all four and save computation in those that
  do not produce emits.
  """

  @tf.Module.with_name_scope
  def step(
      self, x, state, training, constants: Constants | None = None
  ) -> tuple[Sequence, State]:
    output, state, _ = self.step_with_emits(x, state, training, constants)
    return output, state

  @abc.abstractmethod
  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    pass

  @tf.Module.with_name_scope
  def layer(
      self,
      x: Sequence,
      training: bool,
      initial_state: State | None = None,
      constants: Constants | None = None,
  ) -> Sequence:
    outputs, _ = self.layer_with_emits(x, training, initial_state, constants)
    return outputs

  @abc.abstractmethod
  def layer_with_emits(
      self,
      x: Sequence,
      training: bool,
      initial_state: State | None = None,
      constants: Constants | None = None,
  ) -> tuple[Sequence, Emits]:
    pass

  @abc.abstractmethod
  def get_emit_specs(
      self, input_spec: tf.TensorShape, constants: Constants | None = None
  ) -> EmitSpecs:
    pass


class Stateless(SequenceLayer):
  """A SequenceLayer with no state.

  Sub-classes must only implement:
  - layer
  - get_output_shape
  """

  @tf.Module.with_name_scope
  def step(
      self,
      x: Sequence,
      state: State,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State]:
    return (
        self.layer(x, training, initial_state=state, constants=constants),
        state,
    )

  def get_initial_state(
      self, x: Sequence, constants: Constants | None = None
  ) -> State:
    return ()


class StatelessEmitting(Emitting):
  """A SequenceLayer with no state that emits auxiliary tensors.

  Sub-classes must only implement:
  - layer_with_emits
  - get_output_shape
  - get_emit_specs
  """

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: Sequence,
      state: State,
      training: bool,
      constants: Constants | None = None,
  ) -> tuple[Sequence, State, Emits]:
    outputs, emits = self.layer_with_emits(
        x, training, initial_state=state, constants=constants
    )
    return outputs, state, emits

  def get_initial_state(
      self, x: Sequence, constants: Constants | None = None
  ) -> State:
    return ()


class StatelessPointwise(PreservesShape, Stateless):
  """A SequenceLayer that operates pointwise (per-scalar) on its input."""


class StatelessPointwiseFunctor(StatelessPointwise, metaclass=abc.ABCMeta):
  """A stateless SequenceLayer for simple pointwise processing fns."""

  def __init__(self, name=None):
    super().__init__(name=name)

  @abc.abstractmethod
  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor]:
    pass

  @tf.Module.with_name_scope
  def layer(
      self,
      x: Sequence,
      training: bool,
      initial_state: State | None = None,
      constants: Constants | None = None,
  ) -> Sequence:
    del training
    del initial_state
    return x.apply(self.fn).mask_invalid()


def expand_to_rank(t: tf.Tensor, rank: int) -> tf.Tensor:
  # TODO(rryan): This iterative approach is tf.lite compatible and
  # preserves shape information.. Using Tensor.__getitem__((Ellipsis, ...))
  # would do it in one strided slice, but is not supported by tf.lite.
  while len(t.shape) < rank:
    t = tf.expand_dims(t, -1)
  return t


def experimental_iterate_emits(
    layer: SequenceLayer, emits: Emits
) -> Generator[tuple[SequenceLayer, Emits], None, None]:
  """Yields (layer, emits) tuples to allow associating emits with layers.

  This enables associating emits in the structure of emits with specific layers
  that produced the emits.

  Args:
    layer: SequenceLayer to iterate the emits for.
    emits: The emits or emit specs associated with layer.

  Yields:
    (layer, emits) tuples for layer and all of its sublayers.
  """
  yield from layer._yield_emits(emits)  # pylint: disable=protected-access
