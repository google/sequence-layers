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
"""Tests for sequence_layers.tensorflow.types."""

import itertools
from typing import Tuple

from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class SequenceTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_rank_and_shape_errors(self):
    sl.Sequence(tf.zeros((2, 3, 1)), tf.zeros((2, 3)))
    sl.Sequence(tf.zeros((2, 3, 1, 1)), tf.zeros((2, 3)))

    # Must have known rank.
    if not tf.executing_eagerly():
      with self.assertRaises(ValueError):
        sl.Sequence(
            tf1.placeholder_with_default(
                tf.zeros((2, 3, 1)), shape=tf.TensorShape(None)
            ),
            tf.zeros((2, 3)),
        )
      with self.assertRaises(ValueError):
        sl.Sequence(
            tf.zeros((2, 3, 1)),
            tf1.placeholder_with_default(
                tf.zeros((2, 3)), shape=tf.TensorShape(None)
            ),
        )

    # Values must be at least rank 2.
    with self.assertRaises(ValueError):
      sl.Sequence(tf.zeros((2)), tf.zeros((2, 3)))

    # Mask must be rank 2.
    with self.assertRaises(ValueError):
      sl.Sequence(tf.zeros((2, 3, 1)), tf.zeros((2)))

    # Mismatched batch size.
    with self.assertRaises(ValueError):
      sl.Sequence(tf.zeros((2, 3, 1)), tf.zeros((3, 3)))

    # Mismatched time.
    with self.assertRaises(ValueError):
      sl.Sequence(tf.zeros((2, 4, 1)), tf.zeros((2, 3)))

  def test_concatenate(self):
    x = self.random_sequence(2, 3, 5)
    y = self.random_sequence(2, 3, 5)

    z = x.concatenate(y)
    x, y, z = self.evaluate([x, y, z])
    self.assertAllEqual(z.values, np.concatenate([x.values, y.values], 1))
    self.assertAllEqual(z.mask, np.concatenate([x.mask, y.mask], 1))

  def test_concatenate_sequences(self):
    x = self.random_sequence(2, 3, 5)
    y = self.random_sequence(2, 3, 5)
    z = self.random_sequence(2, 3, 5)

    xyz = sl.Sequence.concatenate_sequences([x, y, z])
    x, y, z, xyz = self.evaluate([x, y, z, xyz])
    self.assertAllEqual(
        xyz.values, np.concatenate([x.values, y.values, z.values], 1)
    )
    self.assertAllEqual(xyz.mask, np.concatenate([x.mask, y.mask, z.mask], 1))

  def test_slice(self):
    x = self.random_sequence(2, 3, 5)

    # Slice batch alone.
    y = x[:1]
    self.assertAllEqual(y.values, x.values[:1, :, :])
    self.assertAllEqual(y.mask, x.mask[:1, :])

    # Slice time alone.
    y = x[:, :1]
    self.assertAllEqual(y.values, x.values[:, :1, :])
    self.assertAllEqual(y.mask, x.mask[:, :1])

    # Slice batch and time.
    y = x[1:, :1]
    self.assertAllEqual(y.values, x.values[1:, :1, :])
    self.assertAllEqual(y.mask, x.mask[1:, :1])

    # Slicing time with an integer or ellipsis is not allowed.
    with self.assertRaises(ValueError):
      y = x[..., :]  # pytype: disable=unsupported-operands
    with self.assertRaises(ValueError):
      y = x[:, ...]  # pytype: disable=unsupported-operands
    with self.assertRaises(ValueError):
      y = x[0, :]  # pytype: disable=unsupported-operands
    with self.assertRaises(ValueError):
      y = x[:, 0]  # pytype: disable=unsupported-operands
    # Slicing more than batch/time is not allowed.
    with self.assertRaises(ValueError):
      y = x[:, :, 1:]

  @parameterized.parameters(True, False)
  def test_pad_time(self, valid):
    x = self.random_sequence(2, 3, 5)
    y = x.pad_time(1, 2, valid=valid, pad_value=-1.0)
    x, y = self.evaluate([x, y])

    self.assertAllEqual(
        y.values,
        np.pad(
            x.values,
            [[0, 0], [1, 2], [0, 0]],
            mode='constant',
            constant_values=-1.0,
        ),
    )
    self.assertAllEqual(
        y.mask,
        np.pad(
            x.mask,
            [[0, 0], [1, 2]],
            mode='constant',
            constant_values=1.0 if valid else 0.0,
        ),
    )

  @parameterized.parameters(tf.bool, tf.int32, tf.float32, tf.complex128)
  def test_mask_invalid(self, dtype):
    if dtype.is_complex:
      values = tf.complex(
          tf.ones((2, 3, 5, 6), dtype=dtype.real_dtype),
          tf.ones((2, 3, 5, 6), dtype=dtype.real_dtype),
      )
    else:
      values = tf.ones((2, 3, 5, 6), dtype=dtype)
    x = sl.Sequence(
        values, tf.cast(tf.random.uniform((2, 3)) > 0.5, tf.float32)
    )
    x_masked = x.mask_invalid()
    x, x_masked = self.evaluate([x, x_masked])
    self.assertAllEqual(
        x_masked.values, x.values * x.mask[:, :, np.newaxis, np.newaxis]
    )

  @parameterized.parameters(
      ([None, 3, 1], [None, 3], [None, 3, 1], [None, 3]),
      ([2, None, 1], [2, None], [2, None, 1], [2, None]),
      ([None, None, 1], [None, None], [None, None, 1], [None, None]),
      ([2, 3, 1], [2, 3], [2, 3, 1], [2, 3]),
  )
  def test_mask_invalid_shape_preserved(
      self, x_values_shape, x_mask_shape, y_values_shape, y_mask_shape
  ):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=x_values_shape),
            tf.TensorSpec(x_mask_shape),
        ]
    )
    def fn(values, mask):
      y = sl.Sequence(values, mask).mask_invalid()
      self.assertAllEqual(y.values.shape.as_list(), y_values_shape)
      self.assertAllEqual(y.mask.shape.as_list(), y_mask_shape)
      return y

    fn(tf.ones((2, 3, 1)), tf.ones((2, 3)))

  def test_print(self):
    self.evaluate(self.random_sequence(2, 3, 5).print())

  def test_to_spec(self):
    x = self.random_sequence(2, 3, 5, dtype=tf.float64)
    spec = x.to_spec()
    self.assertEqual(spec.values.shape, tf.TensorShape([2, 3, 5]))
    self.assertEqual(spec.values.dtype, tf.float64)
    self.assertEqual(spec.mask.shape, tf.TensorShape([2, 3]))
    self.assertEqual(spec.mask.dtype, sl.MASK_DTYPE)

  def test_nest(self):
    x = self.random_sequence(2, 3, 5)
    values, mask = tf.nest.flatten(x)
    self.assertIs(values, x.values)
    self.assertIs(mask, x.mask)
    y = tf.nest.pack_sequence_as(x, (values, mask))
    self.assertIsInstance(y, sl.Sequence)
    self.assertIs(y.values, x.values)
    self.assertIs(y.mask, x.mask)

  def test_nest_error_message(self):
    x = sl.Sequence(tf.zeros((2, 3, 5)), tf.zeros((2, 3)))

    tf.nest.map_structure(lambda s, _: s, (x, x), (x, x))

    with self.assertRaisesRegex(ValueError, r'same nested structure'):
      tf.nest.map_structure(lambda s, _: s, (x, x), (x, x, x))

  def test_function(self):
    x = self.random_sequence(2, 3, 5)

    @tf.function
    def fn(x: sl.Sequence) -> sl.Sequence:
      return x

    self.assertIsInstance(fn(x), sl.Sequence)

  def test_function_signature(self):
    x = self.random_sequence(2, 3, 5)

    @tf.function(input_signature=[x.to_spec()])
    def fn(x: sl.Sequence) -> sl.Sequence:
      return x

    self.assertIsInstance(fn(x), sl.Sequence)

  @parameterized.parameters(((2, 3, 4),), ((2, 3, 5),), ((2, 3, 4, 5),))
  def test_channel_shape(self, dims):
    x = self.random_sequence(*dims)
    self.assertEqual(x.channel_shape, dims[2:])

  def test_per_replica(self):
    x = self.random_sequence(2, 3, 5)
    strategy = tf.distribute.MirroredStrategy(['/cpu:0', '/cpu:1'])

    @tf.function
    def fn(x: sl.Sequence) -> sl.Sequence:
      return x.apply_values(lambda v: v + 1)

    per_replica = strategy.run(fn, args=(x,))
    self.assertEqual(type(per_replica).__name__, 'PerReplica')
    for sequence in strategy.experimental_local_results(per_replica):
      self.assertEqual(sequence.values.shape.as_list(), [2, 3, 5])
      self.assertEqual(sequence.mask.shape.as_list(), [2, 3])


class SequenceArrayTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      *itertools.product(
          ([], [5], [5, 7]), (tf.float32, tf.int32), (False, True)
      )
  )
  def test_sequence_array(self, channels_shape, dtype, dynamic_size):
    sa = sl.SequenceArray.new(
        dtype, size=0 if dynamic_size else 5, dynamic_size=dynamic_size
    )

    x = self.random_sequence(2, 10, *channels_shape, dtype=dtype)
    for i in range(5):
      sa = sa.write(i, x[:, i * 2 : i * 2 + 2])
    y = sa.concat()

    self.assertAllEqual(y.values, x.values)
    self.assertAllEqual(y.mask, x.mask)


class TestLayer(sl.StatelessPointwiseFunctor):

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'custom_name'

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return values, mask


class SequenceLayerTest(test_util.SequenceLayerTest):

  def test_unique_name(self):
    with tf.name_scope('foo'):
      layer1 = TestLayer()
      layer2 = TestLayer()
    with tf.name_scope('bar'):
      layer3 = TestLayer()
      layer4 = TestLayer('baz')
      layer5 = TestLayer('baz')

    self.assertEqual(layer1.name, 'custom_name')
    self.assertEqual(layer1.name_scope.name, 'foo/custom_name/')
    self.assertEqual(layer2.name, 'custom_name_1')
    self.assertEqual(layer2.name_scope.name, 'foo/custom_name_1/')
    self.assertEqual(layer3.name, 'custom_name_2')
    self.assertEqual(layer3.name_scope.name, 'bar/custom_name_2/')
    self.assertEqual(layer4.name, 'baz')
    self.assertEqual(layer4.name_scope.name, 'bar/baz/')
    self.assertEqual(layer5.name, 'baz')
    if tf.executing_eagerly():
      self.assertEqual(layer5.name_scope.name, 'bar/baz/')
    else:
      self.assertEqual(layer5.name_scope.name, 'bar/baz_1/')


if __name__ == '__main__':
  tf.test.main()
