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
"""Tests for sequence_layers.tensorflow.utils."""

from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class SmartDimensionSizeTest(tf.test.TestCase):

  def test_smart_dimension_size(self):
    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32)])
    def fn(t):
      first = utils.smart_dimension_size(t, 0)
      second = utils.smart_dimension_size(t, 1)

      self.assertIsInstance(first, tf.Tensor)
      self.assertIsInstance(second, int)
      self.assertEqual(1, second)
      return first

    self.assertAllEqual(self.evaluate(fn(tf.zeros([5, 1]))), 5)

  def test_smart_dimension_size_unknown_rank(self):
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def fn(t):
      first = utils.smart_dimension_size(t, 0)
      second = utils.smart_dimension_size(t, 1)
      third = utils.smart_dimension_size(t, -1)

      self.assertIsInstance(first, tf.Tensor)
      self.assertIsInstance(second, tf.Tensor)
      self.assertIsInstance(third, tf.Tensor)
      return first, second, third

    self.assertAllEqual(self.evaluate(fn(tf.zeros([1, 2, 3]))), [1, 2, 3])

  def test_smart_dimension_size_negative_indexing(self):
    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32)])
    def fn(t):
      first = utils.smart_dimension_size(t, -2)
      second = utils.smart_dimension_size(t, -1)

      self.assertIsInstance(first, tf.Tensor)
      self.assertIsInstance(second, int)
      self.assertEqual(1, second)
      return first

    self.assertEqual(self.evaluate(fn(tf.zeros([5, 1]))), 5)

  def test_smart_dimension_size_invalid(self):
    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32)])
    def fn(t):
      with self.assertRaises(IndexError):
        utils.smart_dimension_size(t, 3)

    fn(tf.zeros([5, 1]))

  def test_smart_dimension_all_dims(self):
    """Returns all dims when dim isn't provided."""

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32)])
    def fn(t):
      first, second = utils.smart_dimension_size(t)
      self.assertIsInstance(first, tf.Tensor)
      self.assertIsInstance(second, int)
      self.assertEqual(1, second)
      return first

    self.assertEqual(self.evaluate(fn(tf.zeros([5, 1]))), 5)

  def test_smart_dimension_size_list_of_ints(self):
    """Works with dim as a list of ints."""

    @tf.function(input_signature=[tf.TensorSpec([None, 1, 2, 3], tf.float32)])
    def fn(t):
      first, second = utils.smart_dimension_size(t, dim=[0, 2])
      self.assertIsInstance(first, tf.Tensor)
      self.assertIsInstance(second, int)
      self.assertEqual(2, second)
      return first

    self.assertEqual(self.evaluate(fn(tf.zeros([5, 1, 2, 3]))), 5)


class StepByStepTestLayer(sl.SequenceLayer):

  def get_output_shape(self, input_shape, constants=None):
    return input_shape

  def get_initial_state(self, x, constants=None):
    return tf.convert_to_tensor(0, tf.int32)

  def step(self, x, state, training, constants=None):
    y = x.apply_values(lambda v: v + state + constants['test'])
    return y.mask_invalid(), state + 10

  def layer(self, x, training, initial_state=None, constants=None):
    raise NotImplementedError()


class StepByStepTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      utils.step_by_step_static, utils.step_by_step_dynamic
  )
  def test_step_by_step(self, step_by_step_fn):
    l = StepByStepTestLayer()
    x = sl.Sequence(tf.range(5)[tf.newaxis, :, tf.newaxis], tf.ones((1, 5)))
    constants = {'test': tf.convert_to_tensor(100, tf.int32)}

    # Check that constants are used.
    y, _, _ = step_by_step_fn(
        l,
        x,
        training=False,
        initial_state=None,
        blocks_per_step=1,
        constants=constants,
    )
    self.assertAllEqual(y.values, [[[100], [111], [122], [133], [144]]])
    self.assertAllEqual(y.mask, [[1.0, 1.0, 1.0, 1.0, 1.0]])

    # Check that initial state is used.
    initial_state = l.get_initial_state(x, constants) + 1000
    y, _, _ = step_by_step_fn(
        l,
        x,
        training=False,
        initial_state=initial_state,
        blocks_per_step=1,
        constants=constants,
    )
    self.assertAllEqual(y.values, [[[1100], [1111], [1122], [1133], [1144]]])
    self.assertAllEqual(y.mask, [[1.0, 1.0, 1.0, 1.0, 1.0]])

    # Check that blocks_per_step processes blocks_per_step at a time.
    # StepByStepTestLayer does not obey causality and adds its state to
    # all timesteps so that we can tell which timesteps were processed
    # concurrently.
    y, _, _ = step_by_step_fn(
        l,
        x,
        training=False,
        initial_state=None,
        blocks_per_step=2,
        constants=constants,
    )

    # One padding timestep is added.
    self.assertAllEqual(y.values, [[[100], [101], [112], [113], [124], [0]]])
    self.assertAllEqual(y.mask, [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])


class SequenceBroadcastTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (tuple(), tuple(), tuple()),
      (tuple(), (5,), (5,)),
      (tuple(), (2, 5), (2, 5)),
      ((2,), tuple(), (2,)),
      ((2, 5), tuple(), (2, 5)),
      ((5,), (5,), (5,)),
      ((5,), (2, 5), (2, 5)),
      ((2, 5), (5,), (2, 5)),
      ((3, 1, 5), (2, 5), (3, 2, 5)),
      ((2, 5), (3, 1, 5), (3, 2, 5)),
  )
  def test_sequence_broadcast_add(
      self, x_channel_shape, y_channel_shape, expected_channel_shape
  ):
    batch_size, time = 2, 4
    x = self.random_sequence(batch_size, time, *x_channel_shape)
    y = self.random_sequence(batch_size, time, *y_channel_shape)
    output = utils.sequence_broadcast_add(x, y)

    self.assertEqual(
        output.channel_shape, tf.TensorShape(expected_channel_shape)
    )

  @parameterized.parameters(
      (tuple(), tuple(), (2,)),
      (tuple(), (5,), (6,)),
      (tuple(), (2, 5), (2, 6)),
      ((2,), tuple(), (3,)),
      ((2, 5), tuple(), (2, 6)),
      ((5,), (7,), (12,)),
      ((5,), (2, 7), (2, 12)),
      ((2, 5), (7,), (2, 12)),
      ((3, 1, 5), (2, 7), (3, 2, 12)),
      ((2, 5), (3, 1, 7), (3, 2, 12)),
  )
  def test_sequence_broadcast_concat(
      self, x_channel_shape, y_channel_shape, expected_channel_shape
  ):
    batch_size, time = 2, 4
    x = self.random_sequence(batch_size, time, *x_channel_shape)
    y = self.random_sequence(batch_size, time, *y_channel_shape)
    output = utils.sequence_broadcast_concat(x, y)

    self.assertEqual(
        output.channel_shape, tf.TensorShape(expected_channel_shape)
    )


class ConvOpsTest(tf.test.TestCase):

  def test_dynamic_filter_conv1d(self):
    batch = 4
    in_width = 9
    in_chans = 5
    filt_width = 3
    out_chans = 6

    inputs = tf.random.normal([batch, in_width, in_chans])
    filters = tf.random.normal([batch, filt_width, in_chans, out_chans])

    outputs = utils.dynamic_filter_conv1d(inputs, filters)

    self.assertEqual(outputs.shape, (batch, in_width, out_chans))

  def test_dynamic_filter_conv1d_input_change(self):
    """Changing a batch member's input only changes the corresponding output."""
    batch = 4
    in_width = 9
    in_chans = 5
    filt_width = 3
    out_chans = 6

    np_rng = np.random.RandomState(213)

    filters = tf.constant(
        np_rng.randn(batch, filt_width, in_chans, out_chans), dtype=tf.float32
    )

    # Change first batch member of inputs2 relative to inputs1.
    inputs1 = np_rng.randn(batch, in_width, in_chans)
    inputs2 = inputs1.copy()
    inputs2[0] = np_rng.randn(in_width, in_chans)
    self.assertAllEqual(inputs1[1:], inputs2[1:])
    self.assertTrue(np.any(inputs1[0] != inputs2[0]))

    inputs1 = tf.constant(inputs1, dtype=tf.float32)
    inputs2 = tf.constant(inputs2, dtype=tf.float32)

    outputs1 = utils.dynamic_filter_conv1d(inputs1, filters)
    outputs2 = utils.dynamic_filter_conv1d(inputs2, filters)
    self.assertAllEqual(outputs1[1:], outputs2[1:])
    self.assertTrue(np.any(outputs1[0] != outputs2[0]))

  def test_dynamic_filter_conv1d_filter_change(self):
    """Changing a batch member's filters only changes the corresponding output."""
    batch = 4
    in_width = 9
    in_chans = 5
    filt_width = 3
    out_chans = 6

    np_rng = np.random.RandomState(213)
    inputs = tf.constant(
        np_rng.randn(batch, in_width, in_chans), dtype=tf.float32
    )

    # Change first batch member of filters2 relative to filters1.
    filters1 = np_rng.randn(batch, filt_width, in_chans, out_chans)
    filters2 = filters1.copy()
    filters2[0] = np_rng.randn(filt_width, in_chans, out_chans)
    self.assertAllEqual(filters1[1:], filters2[1:])
    self.assertTrue(np.any(filters1[0] != filters2[0]))
    filters1 = tf.constant(filters1, dtype=tf.float32)
    filters2 = tf.constant(filters2, dtype=tf.float32)

    outputs1 = utils.dynamic_filter_conv1d(inputs, filters1)
    outputs2 = utils.dynamic_filter_conv1d(inputs, filters2)
    self.assertAllEqual(outputs1[1:], outputs2[1:])
    self.assertTrue(np.any(outputs1[0] != outputs2[0]))

  def test_dynamic_filter_conv1d_output_values(self):
    """Confirm expected output values with known inputs."""

    # Note that conv1d actually does correlation instead of convolution.

    # [batch, in_width, in_chans]
    inputs = np.zeros((2, 4, 2), dtype='float32')
    inputs[0, :, 0] = [1, 0, 0, 0]
    inputs[0, :, 1] = [0, 0, 0, 1]
    inputs[1, :, 0] = [0, 1, 0, 0]
    inputs[1, :, 1] = [0, 0, 1, 0]
    inputs = tf.constant(inputs)

    # [batch, filt_width, in_chans, out_chans]
    filters = np.zeros((2, 4, 2, 2), dtype='float32')
    filters[0, :, 0, 0] = [1, 0, 0, 0]
    filters[0, :, 1, 0] = [0, 0, 0, 0]
    filters[0, :, 0, 1] = [0, 1, 0, 0]
    filters[0, :, 1, 1] = [0, 1, 0, 0]
    filters[1, :, 0, 0] = [0, 1, 0, 0]
    filters[1, :, 1, 0] = [0, 0, 0, 0]
    filters[1, :, 0, 1] = [1, 0, 0, 0]
    filters[1, :, 1, 1] = [0, 1, 0, 0]
    filters = tf.constant(filters)

    # [batch, out_width, out_chans]
    expected_outputs = np.zeros((2, 4, 2), dtype='float32')
    expected_outputs[0, :, 0] = [0, 1, 0, 0]
    expected_outputs[0, :, 1] = [1, 0, 0, 1]
    expected_outputs[1, :, 0] = [0, 1, 0, 0]
    expected_outputs[1, :, 1] = [0, 0, 2, 0]

    outputs = utils.dynamic_filter_conv1d(inputs, filters)

    self.assertAllEqual(outputs, expected_outputs)


class LogWithoutUnderFlowTest(tf.test.TestCase):

  def test_log_without_underflow(self):
    """Correct outputs are produced by log_without_underflow."""
    inputs = np.array(
        [2.0, 1.0, 1e-3, 1e-5, 1e-7, 1e-100, 0.0], dtype=np.float32
    )
    min_output = -1e3
    min_input = 1e-6
    outputs = utils.log_without_underflow(inputs, min_output, min_input)

    self.assertAllClose(
        np.concatenate([np.log(inputs[:4]), np.array([-1e3, -1e3, -1e3])]),
        outputs,
    )


if __name__ == '__main__':
  tf.test.main()
