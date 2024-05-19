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
"""Tests for sequence_layers.tensorflow.squeeze."""

import fractions
import itertools

from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class SqueezeTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product((1, 2, 3), ((2, 15, 5), (2, 15, 5, 9)))
  )
  def test_squeeze(self, factor, shape):
    l = sl.Squeeze(factor)
    self.assertEqual(l.block_size, factor)
    self.assertEqual(l.output_ratio, fractions.Fraction(1, factor))

    for time in range(shape[1] - factor, shape[1] + factor + 1):
      x = self.random_sequence(shape[0], time, *shape[2:])
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape((factor,) + shape[2:]),
      )
      self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(1, 2, 3)
  def test_causal(self, factor):
    l = sl.Squeeze(factor)
    self.assertTrue(l.supports_step)

    values = tf.convert_to_tensor([[[1.0], [np.nan]]], tf.float32)
    mask = tf.convert_to_tensor([[1.0, 1.0]])

    x = sl.Sequence(values, mask)
    y_layer = l.layer(x, training=False)
    y_step, _, _ = utils.step_by_step_static(l, x, training=False)

    y_layer, y_step = self.evaluate([y_layer, y_step])

    self.assertAllFinite(y_layer.values[:, 0, :])
    self.assertAllFinite(y_step.values[:, 0, :])
    self.verify_tflite_step(l, x)

  @parameterized.parameters(1, 2, 3)
  def test_noncausal(self, factor):
    l = sl.Squeeze(factor, padding='valid')
    self.assertFalse(l.supports_step)

    values = tf.range(6, dtype=tf.float32)[tf.newaxis] + 1.0
    mask = tf.ones_like(values)

    x = sl.Sequence(values, mask)
    y_layer = l.layer(x, training=False)

    self.assertAllEqual(tf.ones([1, 6 // factor]), y_layer.mask)
    self.assertAllEqual(
        tf.reshape(values, [1, 6 // factor, factor]), y_layer.values
    )

  @parameterized.parameters(
      itertools.product((1, 2, 3), ((2, 16, 5), (2, 16, 5, 9)))
  )
  def test_unsqueeze(self, factor, shape):
    l = sl.Unsqueeze(factor)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, factor)

    for time in range(shape[1] - factor, shape[1] + factor + 1):
      x = self.random_sequence(shape[0], time, factor, *shape[2:])
      self.assertEqual(
          l.get_output_shape_for_sequence(x), tf.TensorShape(shape[2:])
      )
      self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)


if __name__ == '__main__':
  tf.test.main()
