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
"""Tests for sequence_layers.tensorflow.dense."""

from absl.testing import parameterized
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class DenseTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),))  # pyformat: disable
  def test_dense(self, shape):
    num_units = 4
    x = self.random_sequence(*shape)
    with tf.name_scope('test'):
      l = sl.Dense(num_units)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + [num_units]
    )
    self.verify_contract(l, x, training=False)
    # A kernel and bias.
    self.assertLen(l.variables, 2)
    self.assertLen(l.trainable_variables, 2)
    self.assertEqual(l.name_scope.name, 'test/dense/')
    self.assertCountEqual(
        [v.name for v in l.variables],
        ['test/dense/kernel:0', 'test/dense/bias:0'],
    )
    self.verify_tflite_step(l, x)

  def test_dense_invalid_2d_shape(self):
    num_units = 4
    x = self.random_sequence(2, 3)
    l = sl.Dense(num_units)
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class DenseShapedTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3)), tf.TensorShape([])),
      (tf.TensorShape((2, 3, 1)), tf.TensorShape([])),
      (tf.TensorShape((2, 3, 5)), tf.TensorShape([])),
      (tf.TensorShape((2, 3, 5, 7)), tf.TensorShape([])),
      (tf.TensorShape((2, 3)), tf.TensorShape([1])),
      (tf.TensorShape((2, 3, 1)), tf.TensorShape([1])),
      (tf.TensorShape((2, 3, 5)), tf.TensorShape([1])),
      (tf.TensorShape((2, 3, 5, 7)), tf.TensorShape([1])),
      (tf.TensorShape((2, 3)), tf.TensorShape([6])),
      (tf.TensorShape((2, 3, 1)), tf.TensorShape([6])),
      (tf.TensorShape((2, 3, 5)), tf.TensorShape([6])),
      (tf.TensorShape((2, 3, 5, 7)), tf.TensorShape([6])),
      (tf.TensorShape((2, 3)), tf.TensorShape([6, 8])),
      (tf.TensorShape((2, 3, 1)), tf.TensorShape([6, 8])),
      (tf.TensorShape((2, 3, 5)), tf.TensorShape([6, 8])),
      (tf.TensorShape((2, 3, 5, 7)), tf.TensorShape([6, 8])),
      (tf.TensorShape((2, 3, 5, 7)), (6, 8)),
      (tf.TensorShape((2, 3, 5, 7)), [6, 8]),
  )
  def test_dense_shaped(self, shape, output_shape):
    x = self.random_sequence(*shape)
    with tf.name_scope('test'):
      l = sl.DenseShaped(output_shape)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    self.verify_contract(l, x, training=False)
    # A kernel and bias.
    self.assertLen(l.variables, 2)
    self.assertLen(l.trainable_variables, 2)
    self.assertEqual(l.name_scope.name, 'test/dense_shaped/')
    self.assertCountEqual(
        [v.name for v in l.variables],
        ['test/dense_shaped/kernel:0', 'test/dense_shaped/bias:0'],
    )
    self.verify_tflite_step(l, x)


class EinsumDenseTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (
          tf.TensorShape((2, 3, 5)),
          '...a,ab->...b',
          tf.TensorShape([7]),
          tf.TensorShape([7]),
      ),
      (
          tf.TensorShape((2, 3, 5, 7)),
          '...ab,ac->...cb',
          tf.TensorShape([11, 7]),
          tf.TensorShape([11, 7]),
      ),
      (
          tf.TensorShape((2, 3, 5, 7)),
          '...ab,b->...a',
          tf.TensorShape([None]),
          tf.TensorShape([5]),
      ),
      (
          tf.TensorShape((2, 3, 5, 7)),
          '...ab,ab->...ba',
          tf.TensorShape([None, None]),
          tf.TensorShape([7, 5]),
      ),
      (
          tf.TensorShape((2, 3, 5, 7)),
          '...ab,abc->...bac',
          tf.TensorShape([None, None, 2]),
          tf.TensorShape([7, 5, 2]),
      ),
  )
  def test_einsum_dense(
      self, shape, equation, output_shape, expected_output_shape
  ):
    x = self.random_sequence(*shape)
    with tf.name_scope('test'):
      l = sl.EinsumDense(equation, output_shape)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), expected_output_shape)
    self.verify_contract(l, x, training=False)
    # A kernel.
    self.assertLen(l.variables, 1)
    self.assertLen(l.trainable_variables, 1)
    self.assertEqual(l.name_scope.name, 'test/einsum_dense/')
    self.assertCountEqual(
        [v.name for v in l.variables],
        ['test/einsum_dense/einsum_dense/kernel:0'],
    )
    self.verify_tflite_step(l, x)

  def test_einsum_dense_nonbroadcasting_equation(self):
    with self.assertRaises(ValueError):
      sl.EinsumDense('btabc,bc->btad', output_shape=[None, 2])

  def test_einsum_dense_inconsistent_input_shape(self):
    x = self.random_sequence(2, 3, 5)
    with tf.name_scope('test'):
      l = sl.EinsumDense('...abc,bc->...ad', output_shape=[None, 2])
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)
    # Show it works with the right input shape.
    x = self.random_sequence(2, 3, 5, 7, 11)
    self.assertEqual(l.get_output_shape_for_sequence(x), [5, 2])

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_einsum_dense_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    x = self.random_sequence(2, 3, 5, dtype=utils.compute_dtype())
    with tf.name_scope('test'):
      l = sl.EinsumDense('...a,ab->...b', output_shape=[2])
    x_np, y_np = self.verify_contract(l, x, training=True)
    self.assertEqual(x_np.dtype, utils.compute_dtype())
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


if __name__ == '__main__':
  tf.test.main()
