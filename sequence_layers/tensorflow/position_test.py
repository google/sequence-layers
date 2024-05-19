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
"""Tests for sequence_layers.tensorflow.position."""

import itertools

from absl.testing import parameterized
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class AddTimingSignalTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (1.0, 1.0e4, True), (1.0, 1.0e4, False), (10.0, 1.0e5, False)
  )
  def test_add_timing_signal(
      self, min_timescale, max_timescale, trainable_scale
  ):
    batch_size, channels = 2, 3
    with tf.name_scope('test'):
      l = sl.AddTimingSignal(
          min_timescale=min_timescale,
          max_timescale=max_timescale,
          trainable_scale=trainable_scale,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    for time in range(5 * l.block_size, 7 * l.block_size):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), tf.TensorShape([channels])
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 1 if trainable_scale else 0)
      self.assertLen(l.trainable_variables, 1 if trainable_scale else 0)
      self.assertCountEqual(
          [v.name for v in l.variables],
          ['test/add_timing_signal/scale:0'] if trainable_scale else [],
      )
    self.verify_tflite_step(l, x)

  def test_add_timing_signal_rank4(self):
    batch_size, time, space, channels = 2, 3, 5, 9
    with tf.name_scope('test'):
      l = sl.AddTimingSignal(trainable_scale=True)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    x = self.random_sequence(batch_size, time, space, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([space, channels])
    )
    self.verify_contract(l, x, training=False)
    self.assertLen(l.variables, 1)
    self.assertLen(l.trainable_variables, 1)
    self.assertCountEqual(
        [v.name for v in l.variables], ['test/add_timing_signal/scale:0']
    )
    self.verify_tflite_step(l, x)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_add_timing_signal_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    with tf.name_scope('test'):
      l = sl.AddTimingSignal()

    x = self.random_sequence(2, 5, 3, dtype=utils.compute_dtype())
    _, y_np = self.verify_contract(l, x, training=True)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


class ConcatTimingSignalTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (tuple(), 4, (5,)), ((2,), 4, (6,)), ((2, 6), 4, (2, 10))
  )
  def test_concat_timing_signal(
      self, x_channel_shape, timing_signal_channels, expected_output_shape
  ):
    min_timescale = 1.0
    max_timescale = 1.0e4
    batch_size = 8
    l = sl.ConcatTimingSignal(
        channels=timing_signal_channels,
        min_timescale=min_timescale,
        max_timescale=max_timescale,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    for time in range(5 * l.block_size, 7 * l.block_size):
      x = self.random_sequence(batch_size, time, *x_channel_shape)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), expected_output_shape
      )
      self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_concat_timing_signal_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    with tf.name_scope('test'):
      l = sl.ConcatTimingSignal()

    x = self.random_sequence(2, 5, 3, dtype=utils.compute_dtype())
    _, y_np = self.verify_contract(l, x, training=True)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


class ConcatPositionEmbeddingTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (tuple(), 4, (5,)), ((2,), 4, (6,)), ((2, 6), 4, (2, 10))
  )
  def test_concat_position_embeddings(
      self, x_channel_shape, position_embedding_channels, expected_output_shape
  ):
    batch_size = 8
    l = sl.ConcatPositionEmbedding(
        timesteps=6,
        channels=position_embedding_channels,
        name='test',
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    for time in range(5 * l.block_size, 7 * l.block_size):
      x = self.random_sequence(batch_size, time, *x_channel_shape)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), expected_output_shape
      )
      self.verify_contract(l, x, training=False)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/embeddings:0',
          ],
      )
      self.assertLen(l.variables, 1)
      self.assertLen(l.trainable_variables, 1)
    self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_concat_position_embedding_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    with tf.name_scope('test'):
      l = sl.ConcatPositionEmbedding(timesteps=12, channels=2)

    x = self.random_sequence(2, 5, 3, dtype=utils.compute_dtype())
    _, y_np = self.verify_contract(l, x, training=True)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


class ApplyRotaryPositionalEncodingTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(itertools.product((1.0e4, 1.0e5), ((4,), (3, 6))))
  def test_basic(self, max_timescale, channel_shape):
    l = sl.ApplyRotaryPositionalEncoding(
        max_timescale=max_timescale,
        name='rope',
    )

    batch_size = 2
    x = self.random_sequence(batch_size, 1, *channel_shape)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'rope')
    self.assertEqual(l.get_output_shape_for_sequence(x), x.values.shape[2:])
    self.assertEmpty(l.variables)

    for time in range(13 * l.block_size, 15 * l.block_size):
      x = self.random_sequence(batch_size, time, *channel_shape)
      self.verify_contract(l, x, training=False)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    with tf.name_scope('test'):
      l = sl.ApplyRotaryPositionalEncoding(1.0e4)

    x = self.random_sequence(2, 5, 4, dtype=utils.compute_dtype())
    _, y_np = self.verify_contract(l, x, training=True)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    self.assertEmpty(l.variables)
    tf.keras.mixed_precision.set_global_policy(default_policy)


if __name__ == '__main__':
  tf.test.main()
