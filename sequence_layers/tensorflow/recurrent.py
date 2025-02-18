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
"""Recurrent layers."""

import collections
import copy
import fractions
from typing import Any, Generator, Optional, Tuple

from sequence_layers.tensorflow import combinators
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class RNN(types.SequenceLayer):
  """A (Keras) recurrent layer."""

  def __init__(self, cell, name: Optional[str] = None):
    super().__init__(name=name)
    self._output_size = cell.output_size
    self._built = False
    self._cell = cell
    with self.name_scope as name_scope:
      self._dynamic_rnn = tf.keras.layers.RNN(
          cell, return_sequences=True, return_state=True, name=name_scope
      )
      self._static_rnn = tf.keras.layers.RNN(
          cell,
          return_sequences=True,
          return_state=True,
          unroll=True,
          name=name_scope,
      )

  def _build(self, x: types.Sequence):
    """Builds the RNN cell."""
    if self._built:
      return
    # The cell scope differs based on whether _dynamic_rnn or _static_rnn is the
    # layer to build the RNN cell, so we explicitly build the cell first to give
    # it a stable name regardless of which method is called first.
    # TODO(b/152153746): The cell loses its name when built manually.
    with tf.name_scope(self._cell.name):
      self._cell.build(x.values.shape)
    self._built = True

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    return self._dynamic_rnn.get_initial_state(x.values)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    del input_shape

    def to_tensor_shape(x):
      if isinstance(x, int):
        return tf.TensorShape([x])
      return x

    # An RNN cell's output_size can be a structures of int/TensorShape. Convert
    # to TensorShape.
    return tf.nest.map_structure(to_tensor_shape, self._output_size)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    self._build(x)
    result = self._static_rnn(
        x.values, initial_state=state, mask=x.mask, training=training
    )
    values, state = result[0], result[1:]
    # Keras's RNN copies outputs for masked states instead of zeroing them.
    return types.Sequence(values, x.mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self._build(x)
    result = self._dynamic_rnn(
        x.values, mask=x.mask, initial_state=initial_state, training=training
    )
    values = result[0]

    # Keras's RNN copies outputs for masked states instead of zeroing them.
    return types.Sequence(values, x.mask).mask_invalid()


class LegacyRNN(types.SequenceLayer):
  """An RNN layer that uses tf.nn.dynamic_rnn and tf.nn.static_rnn."""

  def __init__(self, cell, name: Optional[str] = None):
    super().__init__(name=name)
    self._output_size = cell.output_size
    self._cell = cell

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    batch_size = x.values.shape.dims[0].value
    dtype = x.values.dtype
    # TODO(rryan): This doesn't belong here, but creates the variables for the
    # cell outside of layer/step.
    if not self._cell.built:
      self._cell.build(x.values.shape)
    if hasattr(self._cell, 'get_initial_state'):
      return self._cell.get_initial_state(x.values, batch_size, dtype)
    return self._cell.zero_state(batch_size, dtype)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    del input_shape

    def to_tensor_shape(x):
      if isinstance(x, int):
        return tf.TensorShape([x])
      return x

    # An RNN cell's output_size can be a structures of int/TensorShape. Convert
    # to TensorShape.
    return tf.nest.map_structure(to_tensor_shape, self._output_size)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    values = tf.unstack(x.values, axis=1)
    # To avoid creating complicated control flow for what is probably a
    # short sequence of frames, do not provide lengths to static_rnn. We are
    # masking the output values below, so this is safe.
    values, state = tf1.nn.static_rnn(self._cell, values, initial_state=state)
    values = tf.stack(values, axis=1)
    return types.Sequence(values, x.mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    if initial_state is None:
      initial_state = self.get_initial_state(x)
    lengths = x.lengths()
    values, _ = tf1.nn.dynamic_rnn(
        self._cell,
        x.values,
        initial_state=initial_state,
        dtype=x.values.dtype,
        sequence_length=lengths,
    )
    return types.Sequence(values, x.mask)


class Autoregressive(types.Emitting):
  """Wraps any SequenceLayer, concatenating input with its previous output.

  If the wrapped layer downsamples the output will be "upsampled" to the input
  rate via tiling. Upsampling is currently unsupported.

  TODO(rryan): User-specified downsampling or upsampling.
  TODO(rryan): User-specified feedback combination function (concat, add, etc.)
  """

  def __init__(
      self,
      layer: combinators.SequenceLayerListOrCallable,
      feedback_layer: Optional[combinators.SequenceLayerListOrCallable] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    # TODO(rryan): Support upsample rate?
    self._layer = combinators._maybe_wrap_layers(self, layer)
    if feedback_layer is not None:
      feedback_layer = combinators._maybe_wrap_layers(self, feedback_layer)
    self._feedback_layer = feedback_layer

    if self._layer.output_ratio > 1:
      raise ValueError(
          'Autoregressive requires output_ratio of <= 1: %s %s'
          % (self._layer, self._layer.output_ratio)
      )
    if not self._layer.supports_step:
      raise ValueError(
          f'Autoregressive layer ({self._layer}) does not support stepping.'
      )

  @property
  def block_size(self) -> int:
    return self._layer.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._layer.output_ratio

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    input_shape = x.channel_shape
    output_shape = self._layer.get_output_shape(input_shape, constants)
    output_dtype = self._layer.get_output_dtype(x.dtype)
    feedback_shape = output_shape
    feedback_dtype = output_dtype
    feedback_state = ()
    if self._feedback_layer:
      tensor_output_shape = tf.concat(
          [tf.shape(x.values)[:2], output_shape.as_list()], 0
      )
      # Create a fake output sequence to get initial state for the feedback
      # layer.
      x_out = x.apply_values(
          lambda v: tf.zeros(tensor_output_shape, dtype=output_dtype)
      )
      feedback_state = self._feedback_layer.get_initial_state(x_out, constants)
      feedback_shape = self._feedback_layer.get_output_shape(
          output_shape, constants
      )
      feedback_dtype = self._feedback_layer.get_output_dtype(output_dtype)
    # We concatenate the result of the feedback layer, so the actual input
    # is augmented with the output.
    if input_shape[1:] != feedback_shape[1:]:
      raise ValueError(
          'Input and feedback shape must match on every dimension '
          'but the first: input_shape=%s feedback_shape=%s'
          % (input_shape, feedback_shape)
      )
    if x.dtype != feedback_dtype:
      raise ValueError(
          f'Autoregressive feedback dtype ({feedback_dtype}) '
          f'does not match the input dtype ({x.dtype}).'
      )

    extra_dims = input_shape.ndims - 1
    feedback_size = feedback_shape.dims[0].value

    autoregressive_input_shape = tf.TensorShape(
        [input_shape.dims[0].value + feedback_size]
    ).concatenate(input_shape[1:])
    x = x.apply_values(
        tf.pad, [[0, 0], [0, 0], [0, feedback_size]] + [[0, 0]] * extra_dims
    )
    autoregressive_output_shape = self._layer.get_output_shape(
        autoregressive_input_shape, constants
    )
    if autoregressive_output_shape != output_shape:
      raise ValueError(
          'Autoregressive cannot be used with a layer whose '
          'output shape depends on its input shape: %s != %s'
          % (autoregressive_output_shape, output_shape)
      )
    batch_size = utils.smart_dimension_size(x.values, 0)
    output_block_size = self._layer.block_size * self._layer.output_ratio
    assert output_block_size.denominator == 1
    # TODO(rryan): Should we track the mask through the feedback connection?
    # If we're processing contiguous sequences causally, then we can skip
    # tracking the mask in the feedback connection, because fed-back frames
    # can only be valid where the current block is invalid.
    zero_feedback = tf.zeros(
        [batch_size, int(output_block_size)] + feedback_shape.as_list(),
        dtype=feedback_dtype,
    )
    # x is a correct input here because zero padding is equivalent to
    # concatenating zero_feedback.
    layer_state = self._layer.get_initial_state(x, constants)
    return zero_feedback, feedback_state, layer_state

  def _get_autoregressive_input_shape(
      self,
      input_shape: tf.TensorShape,
      output_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    feedback_shape = output_shape
    if self._feedback_layer:
      feedback_shape = self._feedback_layer.get_output_shape(
          output_shape, constants
      )
    feedback_size = feedback_shape.dims[0].value
    return tf.TensorShape(
        [input_shape.dims[0].value + feedback_size]
    ).concatenate(input_shape[1:])

  def _get_autoregressive_input_spec(
      self,
      input_spec: tf.TensorSpec,
      output_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorSpec:
    return tf.TensorSpec(
        self._get_autoregressive_input_shape(
            input_spec.shape, output_shape, constants
        ),
        input_spec.dtype,
    )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    output_shape = self._layer.get_output_shape(input_shape, constants)
    autoregressive_input_shape = self._get_autoregressive_input_shape(
        input_shape, output_shape, constants
    )
    autoregressive_output_shape = self._layer.get_output_shape(
        autoregressive_input_shape, constants
    )
    if autoregressive_output_shape != output_shape:
      raise ValueError(
          'Autoregressive cannot be used with a layer whose '
          'output shape depends on its input shape: %s != %s'
          % (autoregressive_output_shape, output_shape)
      )
    return output_shape

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> types.EmitSpecs:
    # TODO(rryan): Feedback layer emits?
    output_shape = self._layer.get_output_shape(input_spec.shape, constants)
    autoregressive_input_spec = self._get_autoregressive_input_spec(
        input_spec, output_shape, constants
    )
    return collections.OrderedDict([[
        self._layer.name,
        self._layer.get_emit_specs(autoregressive_input_spec, constants),
    ]])

  def _autoregressive_step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants,
      unroll: bool,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    input_block_size = self._layer.block_size
    output_block_size = int(input_block_size * self._layer.output_ratio)
    # Guaranteed by the constructor.
    assert output_block_size <= input_block_size
    assert input_block_size % output_block_size == 0
    block_ratio = input_block_size // output_block_size

    input_spec = x.channel_spec
    output_spec = self._layer.get_output_spec(input_spec, constants)
    feedback_shape = output_spec.shape
    if self._feedback_layer:
      feedback_shape = self._feedback_layer.get_output_shape(
          output_spec.shape, constants
      )
    autoregressive_input_spec = self._get_autoregressive_input_spec(
        x.channel_spec, output_spec.shape, constants
    )
    emit_specs = self._layer.get_emit_specs(
        autoregressive_input_spec, constants
    )

    input_time = utils.smart_dimension_size(x.values, 1)
    if unroll and not isinstance(input_time, int):
      raise ValueError(
          'To statically unroll, the Sequence time dimension must be known: %s'
          % x
      )

    # The number of blocks we will process (i.e. iterations of while loop).
    num_blocks = (input_time + input_block_size - 1) // input_block_size
    # The timesteps for the number of whole blocks we can process.
    padded_time = num_blocks * input_block_size

    pad_amount = padded_time - input_time
    x = x.pad_time(0, pad_amount, valid=False)
    input_time = padded_time

    def transition_fn(
        x_block: types.Sequence, state: types.State
    ) -> Tuple[types.Sequence, types.State, types.Emits]:
      """Concatenates previous output to input and runs the wrapped layer."""
      previous_output, feedback_state, layer_state = state
      previous_output.shape.assert_is_compatible_with(
          tf.TensorShape([None, output_block_size]).concatenate(feedback_shape)
      )

      if block_ratio > 1:
        previous_output = tf.tile(
            previous_output,
            [1, block_ratio] + [1] * (previous_output.shape.rank - 2),
        )
      x_block = x_block.apply_values(
          lambda v: tf.concat([v, previous_output], axis=2)
      )  # pylint: disable=cell-var-from-loop

      y_block, layer_state, y_emits = self._layer.step_with_emits(
          x_block, layer_state, training=training, constants=constants
      )

      if self._feedback_layer is not None:
        feedback_block, feedback_state = self._feedback_layer.step(
            y_block, feedback_state, training=training, constants=constants
        )
        feedback_block = feedback_block.values
      else:
        feedback_block = y_block.values

      return y_block, (feedback_block, feedback_state, layer_state), y_emits

    if unroll:
      return utils.step_by_step_fn_static(
          transition_fn, num_blocks, input_block_size, x, state
      )
    else:
      return utils.step_by_step_fn_dynamic(
          transition_fn,
          input_time,
          input_spec.shape,
          output_spec,
          emit_specs,
          num_blocks,
          input_block_size,
          output_block_size,
          x,
          state,
      )

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    outputs, state, emits = self._autoregressive_step(
        x, state, training, constants, unroll=True
    )
    emits = collections.OrderedDict([[self._layer.name, emits]])
    return outputs, state, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    if initial_state is None:
      initial_state = self.get_initial_state(x, constants)
    x, _, emits = self._autoregressive_step(
        x, initial_state, training=training, constants=constants, unroll=False
    )
    emits = collections.OrderedDict([[self._layer.name, emits]])
    return x, emits

  def _yield_emits(
      self, emits: types.Emits
  ) -> Generator[tuple[types.SequenceLayer, types.Emits], None, None]:
    yield from super()._yield_emits(emits)
    yield from self._layer._yield_emits(emits[self._layer.name])  # pylint: disable=protected-access


def _compute_zoneout(output, state, new_state, zoneout_probability, is_train):
  """Implements zoneout. See KerasZoneoutWrapper for details."""
  zoneout_probability = tf.convert_to_tensor(zoneout_probability, output.dtype)
  keep_probability = 1.0 - zoneout_probability

  # Special-case zoneout being off to save computation.
  if tf.get_static_value(zoneout_probability) == 0.0:
    return output, new_state

  def _zoneout_train(new_state, state):
    # A copy of tf.nn.dropout without scaling by 1/keep_probability.
    random_tensor = keep_probability
    random_tensor += tf.random.uniform(
        tf.shape(new_state), dtype=new_state.dtype
    )

    # keep_selector is 0.0 if random_tensor lies in [keep_probability, 1.0)
    # and 1.0 if random_tensor lies in [1.0, 1.0 + keep_probability).
    keep_selector = tf.floor(random_tensor)
    return (new_state - state) * keep_selector + state

  def _zoneout_eval(new_state, state):
    return keep_probability * new_state + zoneout_probability * state

  with tf.name_scope('zoneout'):
    zoneout_fn = _zoneout_train if is_train else _zoneout_eval
    # Applies zoneout_fn to each pair of members of the
    # potentially-hierarchical structures of new_state and state.
    # Work around Keras's quirky RNN behavior by flattening first then
    # re-packing.
    flat_new_state = tf.nest.map_structure(
        zoneout_fn, tf.nest.flatten(new_state), tf.nest.flatten(state)
    )
    new_state = tf.nest.pack_sequence_as(new_state, flat_new_state)
    return output, new_state


class KerasZoneoutWrapper(tf.keras.layers.AbstractRNNCell):
  """Implements ZoneOut regularization for RNN states.

  From "Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations".
  https://arxiv.org/abs/1606.01305

  Uses the same approach as the canonical implementation from the authors:
  https://github.com/teganmaharaj/zoneout

  Computed as

  `(1 - zoneout_probability) * tf.nn.dropout(new_state - old_state, 1 -
  zoneout_probability) + old_state`

  in training, and

  `zoneout_probability * old_state + (1 - zoneout_probability) * new_state`

  in inference.
  """

  def __init__(self, cell, zoneout_probability, name=None):
    """Constructs a ZoneoutWrapper to apply ZoneOut to the provided RNNCell.

    Args:
      cell: An RNNCell to apply to ZoneOut regularization to.
      zoneout_probability: The probability or rate at which state updates for
        `cell`'s units are forgotten. Must be a Python float.
      name: An optional name for the wrapper.

    Raises:
      ValueError: If zoneout_probability is not a Python float in [0.0, 1.0].
    """
    super().__init__(name=name)
    if not isinstance(zoneout_probability, float):
      raise ValueError('zoneout_probability must be a float.')
    if zoneout_probability < 0.0 or zoneout_probability > 1.0:
      raise ValueError('Invalid zoneout_probability: %s' % zoneout_probability)

    self._cell = cell
    self._zoneout_probability = zoneout_probability

  def build(self, input_shape):
    super().build(input_shape)
    # TODO(b/152153746): The cell loses its name when built manually.
    with tf.name_scope(self._cell.name):
      self._cell.build(input_shape)
    self.built = True

  @property
  def state_size(self):
    """Returns the `state_size` of the wrapped RNNCell."""
    return self._cell.state_size

  @property
  def output_size(self):
    """Returns the `output_size` of the wrapped RNNCell."""
    return self._cell.output_size

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Returns initial state for the wrapped RNNCell.

    Args:
      inputs: The input tensor to the RNN layer, which should contain the batch
        size as its shape[0], and also dtype.
      batch_size: a scalar tensor that represents the batch size of the inputs.
      dtype: a tf.DType that represents the type of the inputs.

    Returns:
      A Tensor or hierarchical structure of initial state Tensors of type
      `dtype` and `batch_size` outer dimension.
    """
    return self._cell.get_initial_state(inputs, batch_size, dtype)

  def call(self, inputs, state, training=None):
    """Runs the RNN cell with `inputs` and `state` and applies ZoneOut.

    Args:
      inputs: A rank 2 Tensor with shape `[batch_size, input_size]`.
      state: A Tensor or hierarchical structure of Tensors representing the
        current RNN cell state.
      training: Whether training is enabled.

    Returns:
      output: A rank 2 Tensor with shape `[batch_size, output_size]`.
      new_state: A Tensor or hierarchical structure of Tensors representing the
        new RNN cell state, with ZoneOut applied.
    """
    if training is None:
      raise ValueError('You must provide training kwarg to ZoneoutWrapper.')
    output, new_state = self._cell(inputs, state, training=training)
    return _compute_zoneout(
        output, state, new_state, self._zoneout_probability, training
    )

  def get_config(self) -> dict[str, Any]:
    return {
        'cell': tf.keras.utils.legacy.serialize_keras_object(self._cell),
        'zoneout_probability': self._zoneout_probability,
    }

  @classmethod
  def from_config(
      cls, config: dict[str, Any], custom_objects: ... = None
  ) -> 'KerasZoneoutWrapper':
    config = copy.deepcopy(config)
    cell = tf.keras.layers.deserialize(
        config.pop('cell'),
        custom_objects=custom_objects,
        use_legacy_format=True,
    )
    return cls(cell, **config)
