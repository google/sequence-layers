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
"""Tests for sequence_layers.tensorflow.recurrent."""

from absl.testing import parameterized
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class RecurrentTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (
          tf.keras.layers.LSTMCell,
          [
              'test/rnn/lstm_cell/kernel:0',
              'test/rnn/lstm_cell/recurrent_kernel:0',
              'test/rnn/lstm_cell/bias:0',
          ],
      ),
      (
          tf.keras.layers.GRUCell,
          [
              'test/rnn/gru_cell/kernel:0',
              'test/rnn/gru_cell/recurrent_kernel:0',
              'test/rnn/gru_cell/bias:0',
          ],
      ),
      # Legacy cells use variable scopes to name things.
      (tf1.nn.rnn_cell.LSTMCell, ['lstm_cell/bias:0', 'lstm_cell/kernel:0']),
      (
          tf1.nn.rnn_cell.GRUCell,
          [
              'gru_cell/candidate/bias:0',
              'gru_cell/candidate/kernel:0',
              'gru_cell/gates/bias:0',
              'gru_cell/gates/kernel:0',
          ],
      ),
  )
  def test_rnn(self, cell, expected_variable_names):
    num_units = 2
    with tf.name_scope('test'):
      l = sl.RNN(cell(num_units))
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    batch_size, channels = 2, 3
    time = 5
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([num_units])
    )
    self.verify_contract(l, x, training=False)
    self.assertLen(l.variables, len(expected_variable_names))
    self.assertLen(l.trainable_variables, len(expected_variable_names))
    self.assertCountEqual(
        [v.name for v in l.variables], expected_variable_names
    )
    self.verify_tflite_step(l, x)

  @parameterized.parameters(True, False)
  def test_rnn_var_names(self, use_layer):
    """Test variable naming is independent of step/layer order."""
    x = self.random_sequence(2, 3, 5)

    with tf.name_scope('test'):
      l = sl.RNN(tf.keras.layers.LSTMCell(3))
    if use_layer:
      l.layer(x, training=True)
    else:
      state = l.get_initial_state(x)
      l.step(x, state, training=True)

    self.assertCountEqual(
        [v.name for v in l.variables],
        [
            'test/rnn/lstm_cell/kernel:0',
            'test/rnn/lstm_cell/recurrent_kernel:0',
            'test/rnn/lstm_cell/bias:0',
        ],
    )

  @parameterized.parameters(
      (
          tf1.nn.rnn_cell.GRUCell,
          [
              'gru_cell/candidate/bias:0',
              'gru_cell/candidate/kernel:0',
              'gru_cell/gates/bias:0',
              'gru_cell/gates/kernel:0',
          ],
      ),
      (tf1.nn.rnn_cell.LSTMCell, ['lstm_cell/bias:0', 'lstm_cell/kernel:0']),
  )
  def test_legacy_rnn(self, cell, expected_variable_names):
    num_units = 2
    with tf.name_scope('test'):
      l = sl.LegacyRNN(cell(num_units))
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    batch_size, channels = 2, 3
    time = 3 * l.block_size
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([num_units])
    )
    self.verify_contract(l, x, training=False)
    self.assertLen(l.variables, len(expected_variable_names))
    self.assertLen(l.trainable_variables, len(expected_variable_names))
    self.assertCountEqual(
        [v.name for v in l.variables], expected_variable_names
    )
    self.verify_tflite_step(l, x)

  def test_autoregressive(self):
    filters = 2
    l = sl.Autoregressive(sl.Conv1D(filters=filters, kernel_size=3, strides=2))
    self.assertEqual(l.block_size, 2)
    self.assertEqual(1 / l.output_ratio, 2)

    batch_size, channels = 2, 3
    for time in range(3 * l.block_size - 3, 3 * l.block_size + 1):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), tf.TensorShape([filters])
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 2)
      self.assertLen(l.trainable_variables, 2)
    self.verify_tflite_step(l, x)

  def test_autoregressive_layer_list(self):
    num_layers = 2
    filters = 2
    l = sl.Autoregressive(
        [sl.Conv1D(filters=filters, kernel_size=3) for _ in range(num_layers)]
    )

    batch_size, time, channels = 2, 4, 3
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([filters])
    )
    self.verify_contract(l, x, training=False)
    self.assertLen(l.variables, 2 * num_layers)
    self.assertLen(l.trainable_variables, 2 * num_layers)
    self.verify_tflite_step(l, x)

  def test_autoregressive_rank4(self):
    l = sl.Autoregressive(sl.Slice()[:1, :])
    self.assertEqual(l.block_size, 1)
    self.assertEqual(1 / l.output_ratio, 1)

    batch_size, time, space, channels = 2, 3, 5, 9
    x = self.random_sequence(batch_size, time, space, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([1, channels])
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  def test_autoregressive_feedback(self):
    filters = 2
    # Feedback shape is different from output shape.
    l = sl.Autoregressive(
        sl.Conv1D(filters=filters, kernel_size=3, strides=2),
        feedback_layer=sl.Dense(3),
    )
    self.assertEqual(l.block_size, 2)
    self.assertEqual(1 / l.output_ratio, 2)

    batch_size, channels = 2, 3
    for time in range(3 * l.block_size - 3, 3 * l.block_size + 1):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), tf.TensorShape([filters])
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 4)
      self.assertLen(l.trainable_variables, 4)
    self.verify_tflite_step(l, x)

  def test_autoregressive_feedback_rank4(self):
    l = sl.Autoregressive(sl.Slice()[:1, :], feedback_layer=sl.Slice()[:1, :])
    self.assertEqual(l.block_size, 1)
    self.assertEqual(1 / l.output_ratio, 1)

    batch_size, time, space, channels = 2, 3, 5, 9
    x = self.random_sequence(batch_size, time, space, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([1, channels])
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  def test_autoregressive_constants(self):
    filters = 2
    # Feedback shape is different from output shape.
    l = sl.Autoregressive(
        sl.Serial([
            sl.Conv1D(filters=filters, kernel_size=3, strides=2),
            test_util.AssertConstantsLayer(),
        ]),
        feedback_layer=sl.Serial(
            [sl.Dense(3), test_util.AssertConstantsLayer()]
        ),
    )
    x = self.random_sequence(2, 3, 5)
    constants = {'test': tf.convert_to_tensor('!')}
    with self.assertRaises(ValueError):
      state = l.get_initial_state(x)
    state = l.get_initial_state(x, constants)
    with self.assertRaises(ValueError):
      l.layer(x, training=True)
    with self.assertRaises(ValueError):
      l.step(x, state, training=True)
    l.layer(x, training=True, constants=constants)
    l.step(x, state, training=True, constants=constants)

  def test_autoregressive_emits(self):
    l = sl.Autoregressive(sl.Serial([sl.Emit(name='emit'), sl.Dense(2)]))
    x = sl.Sequence(tf.zeros((2, 16, 1)), tf.ones((2, 16)))
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    tf.nest.assert_same_structure(emit_specs, emits)
    # One emit for each layer.
    self.assertLen(list(sl.experimental_iterate_emits(l, emits)), 4)
    tf.nest.map_structure(
        lambda s, e: s.shape.assert_is_compatible_with(e.shape[2:]),
        emit_specs,
        emits,
    )
    # The input to Emit is input (1 unit) concatenated with output (2 units).
    self.assertEqual(emits['serial']['emit'].values.shape, [2, 16, 3])
    self.assertEqual(y.values.shape, [2, 16, 2])

  def test_autoregressive_invalid_feedback_dtype(self):
    batch_size, time, channels = 2, 3, 4
    l = sl.Autoregressive(sl.Dense(channels), feedback_layer=sl.Cast(tf.int32))
    x = self.random_sequence(batch_size, time, channels)
    with self.assertRaises(ValueError):
      l.get_initial_state(x)


class ZoneoutWrapperTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for ZoneoutWrapper."""

  @parameterized.named_parameters(
      ('GRU train', tf.keras.layers.GRUCell, True),
      ('GRU inference', tf.keras.layers.GRUCell, False),
      ('LSTM train', tf.keras.layers.LSTMCell, True),
      ('LSTM inference', tf.keras.layers.LSTMCell, False),
  )
  def test_keras_zoneout_wrapper(self, cell_type, training):
    """Check that ZoneoutWrapper works with different cell types."""

    batch_size, units, input_size = (2, 3, 5)
    wrapper = sl.KerasZoneoutWrapper(cell_type(units), zoneout_probability=0.5)

    old_state = wrapper.get_initial_state(
        batch_size=batch_size, dtype=tf.float32
    )
    cell_input = tf.ones([batch_size, input_size])
    wrapper(cell_input, old_state, training=training)

  def test_keras_zoneout_wrapper_train_vs_inference(self):
    """Tests the output of ZoneoutWrapper with known mocked inputs."""

    class FixedStateRNNCell(tf.keras.layers.Layer):
      """A fake RNN cell that allows us to mock out the returned state."""

      def __init__(self, fixed_state):
        super(FixedStateRNNCell, self).__init__()
        self._fixed_state = fixed_state

      # The cell does not accept a training kwarg.
      def call(self, inputs, unused_state):
        # Pass through the inputs and return the fixed state.
        return inputs, self._fixed_state

    batch_size, units, input_size = (2, 1000, 5)
    fixed_state = tf.ones([batch_size, units])
    zoneout_probability = 0.65

    wrapper = sl.KerasZoneoutWrapper(
        FixedStateRNNCell(fixed_state), zoneout_probability=zoneout_probability
    )

    old_state = tf.zeros([batch_size, units]) + 0.25
    cell_input = tf.ones([batch_size, input_size])

    _, new_state_train = wrapper(cell_input, old_state, training=True)
    _, new_state_inference = wrapper(cell_input, old_state, training=False)
    with self.cached_session() as sess:
      new_state_train, new_state_inference = sess.run(
          [new_state_train, new_state_inference]
      )
      # In expectation, the result should be new_state (1.0) * keep_probability
      # (0.35) + old_state (0.25) * zoneout_probability (0.65): 0.5125. The
      # sampling involved in the training path requires a much looser bound.
      self.assertNear(0.5125, new_state_train.mean(), 1e-2)
      self.assertNear(0.5125, new_state_inference.mean(), 1e-6)

  def test_keras_zoneout_wrapper_invalid_arguments(self):
    """Tests that invalid arguments to ZoneoutWrapper trigger assertions."""

    # Zoneout probability < 0.0.
    with self.assertRaises(ValueError):
      sl.KerasZoneoutWrapper(tf.keras.layers.GRUCell(3), -0.1)

    # Zoneout probability > 1.0.
    with self.assertRaises(ValueError):
      sl.KerasZoneoutWrapper(tf.keras.layers.GRUCell(3), 1.1)

    # Zoneout probability as Tensor.
    with self.assertRaises(ValueError):
      sl.KerasZoneoutWrapper(tf.keras.layers.GRUCell(3), tf.constant(1.0))

    # Training not provided:
    layer = sl.KerasZoneoutWrapper(tf.keras.layers.GRUCell(3), 0.5)
    inputs = tf.zeros((1, 1))
    state = layer.get_initial_state(batch_size=1, dtype=tf.float32)
    with self.assertRaises(ValueError):
      layer(inputs, state)


if __name__ == '__main__':
  tf.test.main()
