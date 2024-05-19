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
"""Tests for sequence_layers.tensorflow.combinators."""

import fractions

from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v2 as tf


class SerialTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_noop(self):
    l = sl.Serial([])
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, channels = 2, 5, 1
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([1]))
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  def test_serial(self):
    l = sl.Serial([
        sl.Conv1D(filters=1, kernel_size=2, strides=2),
        sl.Conv1DTranspose(filters=1, kernel_size=2, strides=2),
        sl.Conv1D(filters=1, kernel_size=2, strides=2),
        sl.Conv1DTranspose(filters=1, kernel_size=2, strides=2),
        sl.Conv1D(filters=1, kernel_size=2, strides=2),
        sl.Conv1DTranspose(filters=1, kernel_size=2, strides=2),
    ])

    self.assertEqual(l.block_size, 2)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, channels = 2, 1

    for i in range(2 * l.block_size):
      time = i + 1
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([1]))
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 12)
      self.assertLen(l.trainable_variables, 12)
    self.verify_tflite_step(l, x)

  def test_non_steppable(self):
    l = sl.Serial([
        sl.Conv1D(filters=1, kernel_size=2, strides=2),
        test_util.NonSteppableLayer(),
    ])
    self.assertFalse(l.supports_step)

  @parameterized.parameters(
      (sl.Serial([sl.Squeeze(2)]), 2, fractions.Fraction(1, 2)),
      (sl.Serial([sl.Unsqueeze(2)]), 1, fractions.Fraction(2, 1)),
      (sl.Serial([sl.Squeeze(2), sl.Squeeze(2)]), 4, fractions.Fraction(1, 4)),
      (
          sl.Serial([sl.Squeeze(2), sl.Unsqueeze(2)]),
          2,
          fractions.Fraction(1, 1),
      ),
      (
          sl.Serial([sl.Squeeze(3), sl.Unsqueeze(2)]),
          3,
          fractions.Fraction(2, 3),
      ),
      (
          sl.Serial([sl.Squeeze(3), sl.Unsqueeze(3), sl.Squeeze(3)]),
          3,
          fractions.Fraction(1, 3),
      ),
      (
          sl.Serial([sl.Squeeze(3), sl.Unsqueeze(3), sl.Squeeze(2)]),
          6,
          fractions.Fraction(1, 2),
      ),
      (
          sl.Serial([sl.Squeeze(2), sl.Unsqueeze(2), sl.Squeeze(8)]),
          8,
          fractions.Fraction(1, 8),
      ),
  )
  def test_output_ratio(self, l, expected_block_size, expected_output_ratio):
    self.assertEqual(l.block_size, expected_block_size)
    self.assertEqual(l.output_ratio, expected_output_ratio)

  def test_serial_rank4(self):
    l = sl.Serial([
        sl.Squeeze(2),  # -> [b, t, 2, s, c]
        sl.Dense(2),  # -> [b, t, 2, s, 2]
        sl.Unsqueeze(2),  # -> [b, t, s, 2]
    ])
    self.assertEqual(l.block_size, 2)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, space, channels = 2, 10, 5, 1
    x = self.random_sequence(batch_size, time, space, channels)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([5, 2]))
    self.verify_contract(l, x, training=False)
    self.assertLen(l.trainable_variables, 2)
    self.assertLen(l.variables, 2)
    self.verify_tflite_step(l, x)

  def test_serial_constants(self):
    """Serial passes constants to its sublayers."""
    l = sl.Serial([test_util.AssertConstantsLayer()])
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

  def test_emits(self):
    l = sl.Serial([
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit1'),
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit2'),
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit3'),
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit4'),
    ])

    x = sl.Sequence(tf.zeros((2, 16, 1)), tf.ones((2, 16)))
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    tf.nest.assert_same_structure(emit_specs, emits)
    # One emit for each layer.
    self.assertLen(list(sl.experimental_iterate_emits(l, emits)), 9)

    tf.nest.map_structure(
        lambda s, e: s.shape.assert_is_compatible_with(e.shape[2:]),
        emit_specs,
        emits,
    )

    self.assertEqual(
        list(emits.keys()),
        [
            'conv1d',
            'emit1',
            'conv1d_1',
            'emit2',
            'conv1d_2',
            'emit3',
            'conv1d_3',
            'emit4',
        ],
    )
    self.assertEqual(emits['emit1'].values.shape, [2, 8, 1])
    self.assertEqual(emits['emit2'].values.shape, [2, 4, 1])
    self.assertEqual(emits['emit3'].values.shape, [2, 2, 1])
    self.assertEqual(emits['emit4'].values.shape, [2, 1, 1])
    self.assertEqual(y.values.shape, [2, 1, 1])

  def test_serial_emits_with_dtype_change(self):
    l = sl.Serial([
        sl.OneHot(4),
        sl.GmmAttention('source', 1, 4, 4, True),
    ])

    x = sl.Sequence(tf.zeros((2, 16), dtype=tf.int32), tf.ones((2, 16)))
    constants = {'source': self.random_sequence(2, 10, 6, dtype=tf.float32)}
    y, emits = l.layer_with_emits(x, training=False, constants=constants)
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    tf.nest.assert_same_structure(emit_specs, emits)

    def emit_spec_assert(s, e):
      self.assertTrue(s.is_compatible_with(tf.TensorSpec(e.shape[2:], e.dtype)))

    tf.nest.map_structure(emit_spec_assert, emit_specs, emits)

    self.assertEqual(list(emits.keys()), ['one_hot', 'gmm_attention'])
    self.assertTrue(
        tf.TensorSpec([2, 16, 1, 10], dtype=tf.float32).is_compatible_with(
            tf.TensorSpec.from_tensor(
                emits['gmm_attention'].probabilities.values
            )
        )
    )
    self.assertEqual(y.values.shape, [2, 16, 1, 6])

  def test_serial_emits_duplicate_names(self):
    l = sl.Serial([
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit'),
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit'),
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit'),
        sl.Conv1D(1, 1, 2),
        sl.Emit(name='emit'),
    ])

    x = sl.Sequence(tf.zeros((2, 16, 1)), tf.ones((2, 16)))
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    tf.nest.assert_same_structure(emit_specs, emits)
    tf.nest.map_structure(
        lambda s, e: s.shape.assert_is_compatible_with(e.shape[2:]),
        emit_specs,
        emits,
    )

    self.assertEqual(
        list(emits.keys()),
        [
            'conv1d',
            'emit',
            'conv1d_1',
            'emit_1',
            'conv1d_2',
            'emit_2',
            'conv1d_3',
            'emit_3',
        ],
    )
    self.assertEqual(emits['emit'].values.shape, [2, 8, 1])
    self.assertEqual(emits['emit_1'].values.shape, [2, 4, 1])
    self.assertEqual(emits['emit_2'].values.shape, [2, 2, 1])
    self.assertEqual(emits['emit_3'].values.shape, [2, 1, 1])
    self.assertEqual(y.values.shape, [2, 1, 1])

  def test_serial_dtype_altering_layers(self):
    l = sl.Serial([sl.OneHot(5), sl.Cast(tf.int32), sl.OneHot(2)])

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    batch_size, time = 2, 3
    x = self.random_sequence(batch_size, time, dtype=tf.int32, low=0, high=4)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([5, 2]))
    self.verify_contract(
        l, x, training=False, pad_nan=False, test_gradients=False
    )
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)


class ParallelTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      sl.Parallel.Combination.STACK,
      sl.Parallel.Combination.ADD,
      sl.Parallel.Combination.MEAN,
  )
  def test_noop(self, combination):
    l = sl.Parallel([], combination=combination)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, channels = 2, 5, 1
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([1]))
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(
      sl.Parallel.Combination.STACK,
      sl.Parallel.Combination.ADD,
      sl.Parallel.Combination.MEAN,
  )
  def test_steppable(self, combination):
    is_stack = combination == sl.Parallel.Combination.STACK
    l = sl.Parallel(
        [
            sl.Conv1D(filters=1, kernel_size=2, strides=2),
            sl.Conv1D(filters=1, kernel_size=2, strides=2),
            sl.Conv1D(filters=1, kernel_size=2, strides=2),
        ],
        combination=combination,
    )

    self.assertEqual(l.block_size, 2)
    self.assertEqual(l.output_ratio, fractions.Fraction(1, 2))
    self.assertTrue(l.supports_step)

    batch_size, channels = 2, 1
    for time in range(5, 5 + 2 * l.block_size):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape([3, 1]) if is_stack else tf.TensorShape([1]),
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 6)
      self.assertLen(l.trainable_variables, 6)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(
      sl.Parallel.Combination.STACK,
      sl.Parallel.Combination.ADD,
      sl.Parallel.Combination.MEAN,
  )
  def test_not_steppable(self, combination):
    is_stack = combination == sl.Parallel.Combination.STACK
    l = sl.Parallel(
        [
            sl.Dense(1),
            sl.Conv1D(filters=1, kernel_size=2, strides=1, padding='same'),
        ],
        combination=combination,
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertFalse(l.supports_step)

    batch_size, channels = 2, 1
    for time in range(5, 5 + 2 * l.block_size):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape([2, 1]) if is_stack else tf.TensorShape([1]),
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 4)
      self.assertLen(l.trainable_variables, 4)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(
      sl.Parallel.Combination.STACK,
      sl.Parallel.Combination.ADD,
      sl.Parallel.Combination.MEAN,
  )
  def test_broadcast(self, combination):
    is_stack = combination == sl.Parallel.Combination.STACK
    l = sl.Parallel(
        [
            sl.DenseShaped([]),
            sl.Conv1D(filters=1, kernel_size=2, strides=1),
            sl.Conv1D(filters=8, kernel_size=2, strides=1),
        ],
        combination=combination,
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, channels = 2, 5, 1
    x = self.random_sequence(batch_size, time, channels)
    # [], [1], and [8] broadcast together to form [3, 8].

    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        tf.TensorShape([3, 8]) if is_stack else tf.TensorShape([8]),
    )
    self.verify_contract(l, x, training=False)
    self.assertLen(l.variables, 6)
    self.assertLen(l.trainable_variables, 6)
    self.verify_tflite_step(l, x)

  def test_invalid(self):
    # Different output ratios.
    with self.assertRaises(ValueError):
      sl.Parallel([
          sl.Conv1D(filters=1, kernel_size=2, strides=2),
          sl.Conv1D(filters=1, kernel_size=2, strides=1),
      ])

  @parameterized.parameters(
      sl.Parallel.Combination.STACK,
      sl.Parallel.Combination.ADD,
      sl.Parallel.Combination.MEAN,
  )
  def test_emits(self, combination):
    is_stack = combination == sl.Parallel.Combination.STACK
    l = sl.Parallel(
        [
            sl.Serial([sl.Conv1D(1, 1, 2), sl.Emit(name='emit1')]),
            sl.Serial([sl.Conv1D(1, 1, 2), sl.Emit(name='emit2')]),
            sl.Serial([sl.Conv1D(1, 1, 2), sl.Emit(name='emit3')]),
            sl.Serial([sl.Conv1D(1, 1, 2), sl.Emit(name='emit4')]),
        ],
        combination=combination,
    )

    x = sl.Sequence(tf.zeros((2, 16, 1)), tf.ones((2, 16)))
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    tf.nest.assert_same_structure(emit_specs, emits)
    # One emit for each layer.
    self.assertLen(list(sl.experimental_iterate_emits(l, emits)), 13)

    tf.nest.map_structure(
        lambda s, e: s.shape.assert_is_compatible_with(e.shape[2:]),
        emit_specs,
        emits,
    )

    self.assertEqual(
        list(emits.keys()), ['serial', 'serial_1', 'serial_2', 'serial_3']
    )
    self.assertEqual(emits['serial']['emit1'].values.shape, [2, 8, 1])
    self.assertEqual(emits['serial_1']['emit2'].values.shape, [2, 8, 1])
    self.assertEqual(emits['serial_2']['emit3'].values.shape, [2, 8, 1])
    self.assertEqual(emits['serial_3']['emit4'].values.shape, [2, 8, 1])
    self.assertEqual(y.values.shape, [2, 8, 4, 1] if is_stack else [2, 8, 1])


class ResidualTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_residual(self):
    l = sl.Residual(sl.Conv1D(filters=2, kernel_size=3, strides=1))

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, channels = 2, 1
    for i in range(2 * l.block_size):
      time = i + 1
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([2]))
      self.verify_contract(l, x, training=False)
      # 2 for the conv1d, 2 for the residual projection.
      self.assertLen(l.trainable_variables, 4)
      self.assertLen(l.variables, 4)
    self.verify_tflite_step(l, x)

  def test_residual_with_shortcut(self):
    l = sl.Residual(
        layer=sl.Conv1D(filters=2, kernel_size=3, strides=2),
        shortcut_layer=sl.Conv1D(filters=2, kernel_size=3, strides=2),
    )

    self.assertEqual(l.block_size, 2)
    self.assertEqual(1 / l.output_ratio, 2)
    self.assertTrue(l.supports_step)

    batch_size, channels = 2, 1
    for i in range(2 * l.block_size):
      time = i + 1
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([2]))
      self.verify_contract(l, x, training=False)
      # 2 for each conv1d.
      self.assertLen(l.trainable_variables, 4)
      self.assertLen(l.variables, 4)
    self.verify_tflite_step(l, x)

  def test_non_steppable(self):
    l = sl.Residual(test_util.NonSteppableLayer())
    self.assertFalse(l.supports_step)

  def test_residual_layer_list(self):
    l = sl.Residual([sl.Translate(-20), sl.Scale(2)])

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, channels = 2, 3, 5
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([5]))
    x_np, y_np = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.trainable_variables)
    self.assertEmpty(l.variables)
    expected = (
        np.concatenate([x_np.values + 2 * (x_np.values - 20)], axis=2)
        * y_np.mask[:, :, np.newaxis]
    )
    self.assertAllClose(y_np.values, expected)
    self.verify_tflite_step(l, x)

  def test_residual_no_projection(self):
    l = sl.Residual(sl.Conv1D(filters=2, kernel_size=3, strides=1))
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, channels = 2, l.block_size, 2
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([2]))
    self.verify_contract(l, x, training=False)
    # 2 for the conv1d, no residual projection.
    self.assertLen(l.trainable_variables, 2)
    self.assertLen(l.variables, 2)
    self.verify_tflite_step(l, x)

  def test_residual_equal_num_elements_different_shapes(self):
    l = sl.Residual(sl.DenseShaped([4, 2]))
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, width, height = 2, 3, 2, 4
    x = self.random_sequence(batch_size, time, width, height)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([4, 2]))
    self.verify_contract(l, x, training=False)
    # 2 for the conv1d, 2 for the residual projection.
    self.assertLen(l.trainable_variables, 4)
    self.assertLen(l.variables, 4)
    self.verify_tflite_step(l, x)

  def test_residual_rank4(self):
    l = sl.Residual(sl.Dense(4))
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, space, channels = 2, 3, 5, 1
    x = self.random_sequence(batch_size, time, space, channels)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([5, 4]))
    self.verify_contract(l, x, training=False)
    # 2 for the dense, 2 for the residual projection.
    self.assertLen(l.trainable_variables, 4)
    self.assertLen(l.variables, 4)
    self.verify_tflite_step(l, x)

  def test_residual_constants(self):
    """Residual passes constants to its sublayers."""
    l = sl.Residual(test_util.AssertConstantsLayer())
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

  def test_residual_emits(self):
    l = sl.Residual(sl.Emit(name='emit'))
    x = sl.Sequence(tf.zeros((2, 16, 1)), tf.ones((2, 16)))
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    tf.nest.assert_same_structure(emit_specs, emits)
    # One emit for each layer.
    self.assertLen(list(sl.experimental_iterate_emits(l, emits)), 2)
    tf.nest.map_structure(
        lambda s, e: s.shape.assert_is_compatible_with(e.shape[2:]),
        emit_specs,
        emits,
    )
    self.assertEqual(emits['emit'].values.shape, [2, 16, 1])
    self.assertAllEqual(y.values, 2.0 * emits['emit'].values)

  def test_residual_invalid_dtype_altering(self):
    l = sl.Residual(sl.Cast(tf.int32))

    batch_size, time, channels = 2, 3, 4
    x = self.random_sequence(batch_size, time, channels)
    with self.assertRaises(ValueError):
      l.get_initial_state(x)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class SkipTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_skip(self):
    filters = 2
    l = sl.Skip(sl.Conv1D(filters=filters, kernel_size=3, strides=1))

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, channels = 2, 1
    for i in range(2 * l.block_size):
      time = i + 1
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape([filters + channels]),
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.trainable_variables, 2)
      self.assertLen(l.variables, 2)
    self.verify_tflite_step(l, x)

  def test_non_steppable(self):
    l = sl.Skip(test_util.NonSteppableLayer())
    self.assertFalse(l.supports_step)

  def test_skip_layer_list(self):
    l = sl.Skip([sl.Translate(-20), sl.Scale(2)])

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, channels = 2, 3, 5
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([10]))
    x_np, y_np = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.trainable_variables)
    self.assertEmpty(l.variables)
    expected = (
        np.concatenate([x_np.values, 2 * (x_np.values - 20)], axis=2)
        * y_np.mask[:, :, np.newaxis]
    )
    self.assertAllClose(y_np.values, expected)
    self.verify_tflite_step(l, x)

  def test_skip_rank4(self):
    l = sl.Skip(sl.Exp())
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    batch_size, time, space, channels = 2, 3, 5, 1
    x = self.random_sequence(batch_size, time, space, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([10, 1])
    )
    x_np, y_np = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.trainable_variables)
    self.assertEmpty(l.variables)
    expected = (
        np.concatenate([x_np.values, np.exp(x_np.values)], axis=2)
        * y_np.mask[:, :, np.newaxis, np.newaxis]
    )
    self.assertAllClose(y_np.values, expected)
    self.verify_tflite_step(l, x)

  def test_skip_constants(self):
    """Skip passes constants to its sublayers."""
    l = sl.Skip(test_util.AssertConstantsLayer())
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

  def test_skip_emits(self):
    l = sl.Skip(sl.Emit(name='emit'))
    x = sl.Sequence(tf.zeros((2, 16, 1)), tf.ones((2, 16)))
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    tf.nest.assert_same_structure(emit_specs, emits)
    # One emit for each layer.
    self.assertLen(list(sl.experimental_iterate_emits(l, emits)), 3)
    tf.nest.map_structure(
        lambda s, e: s.shape.assert_is_compatible_with(e.shape[2:]),
        emit_specs,
        emits,
    )
    self.assertEqual(emits['emit'].values.shape, [2, 16, 1])
    self.assertAllEqual(y.values.shape, [2, 16, 2])

  def test_skip_invalid_dtype_altering(self):
    l = sl.Skip(sl.Cast(tf.int32))

    batch_size, time, channels = 2, 3, 4
    x = self.random_sequence(batch_size, time, channels)
    state = l.get_initial_state(x)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)
    with self.assertRaises(ValueError):
      l.step(x, state, training=False)


class BlockwiseTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_blockwise(self):
    base_layer = sl.Conv1D(filters=2, kernel_size=3, strides=2)
    blockwise = sl.Blockwise(base_layer, 4)

    self.assertEqual(blockwise.block_size, 4)
    self.assertEqual(1 / blockwise.output_ratio, 2)
    self.assertTrue(blockwise.supports_step)

    batch_size, time, channels = 2, 10, 1
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(
        blockwise.get_output_shape_for_sequence(x), tf.TensorShape([2])
    )
    self.verify_contract(blockwise, x, training=False)
    self.assertLen(blockwise.trainable_variables, 2)
    self.assertLen(blockwise.variables, 2)
    self.assertCountEqual(
        [v.ref() for v in blockwise.variables],
        [v.ref() for v in base_layer.variables],
    )
    self.verify_tflite_step(blockwise, x)

    y_base = base_layer.layer(x, training=False)
    y_blockwise = blockwise.layer(x, training=False)
    self.assertSequencesClose(y_blockwise, y_base)

  def test_non_steppable(self):
    l = sl.Blockwise(test_util.NonSteppableLayer(), 4)
    self.assertFalse(l.supports_step)


class BidirectionalTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_reverse(self):
    forward = sl.Frame(frame_length=2, frame_step=1)
    backward = sl.Frame(frame_length=2, frame_step=1)
    l = sl.Bidirectional(forward, backward)

    x = sl.Sequence(
        tf.tile(tf.range(10)[tf.newaxis, :, tf.newaxis], [2, 1, 1]),
        tf.sequence_mask([10, 4], maxlen=10, dtype=sl.MASK_DTYPE),
    ).mask_invalid()

    y = l.layer(x, training=False)

    # Forward layer produces:
    # - [0, 0], [0, 1], [1, 2], ...
    # - [0, 0], [0, 1], [1, 2], [2, 3]
    # Backward layer produces:
    # - [0, 9], [9, 8], [8, 7], ...
    # - [0, 3], [3, 2], [2, 1], [1, 0]
    # but is reversed to:
    # - [1, 0], [2, 1], [3, 2], ...
    # - [1, 0], [2, 1], [3, 2], [0, 3]
    # Forward and backward are concated to produce:
    expected = sl.Sequence(
        tf.convert_to_tensor([
            # Sequence 0. Length is 10.
            [[[0, 1], [0, 0]],
             [[0, 2], [1, 1]],
             [[1, 3], [2, 2]],
             [[2, 4], [3, 3]],
             [[3, 5], [4, 4]],
             [[4, 6], [5, 5]],
             [[5, 7], [6, 6]],
             [[6, 8], [7, 7]],
             [[7, 9], [8, 8]],
             [[8, 0], [9, 9]]],

            # Sequence 1. Length is 4.
            [[[0, 1], [0, 0]],
             [[0, 2], [1, 1]],
             [[1, 3], [2, 2]],
             # This timestep is invalidated if we do not do a sequence reversal,
             # because frame appends leading, valid zero padding, but with a
             # tensor reverse, the padding connects with invalid timesteps at
             # the end of this sequence instead of making this last frame valid.
             [[0, 0], [0, 0]],  # [[2, 0], [3, 3]],
             [[0, 0], [0, 0]],
             [[0, 0], [0, 0]],
             [[0, 0], [0, 0]],
             [[0, 0], [0, 0]],
             [[0, 0], [0, 0]],
             [[0, 0], [0, 0]]]
        ]),
        tf.convert_to_tensor(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             # The last timestep is invalidated if we do not do a sequence
             # reversal, because frame appends leading, valid zero padding, but
             # with a tensor reverse, the padding connects with invalid
             # timesteps at the end of this sequence instead of making the last
             # frame of the sequence valid.
             [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]]))  # pyformat: disable
    self.assertSequencesEqual(y, expected)

  @parameterized.parameters(False, True)
  def test_keras_equivalence(self, random_mask):
    forward_cell = tf.keras.layers.GRUCell(1)
    backward_cell = tf.keras.layers.GRUCell(1)
    forward = sl.RNN(forward_cell)
    backward = sl.RNN(backward_cell)
    l = sl.Bidirectional(forward, backward)

    batch_size, time, channels = 2, 10, 1
    x = self.random_sequence(
        batch_size, time, channels, low_length=5, random_mask=random_mask
    )
    y = l.layer(x, training=False)

    keras_forward = tf.keras.layers.RNN(
        forward_cell, return_sequences=True, zero_output_for_mask=True
    )
    keras_backward = tf.keras.layers.RNN(
        backward_cell,
        return_sequences=True,
        go_backwards=True,
        zero_output_for_mask=True,
    )
    keras_bidirectional = tf.keras.layers.Bidirectional(
        keras_forward, backward_layer=keras_backward
    )
    # Keras Bidirectional recreates the forward layer from config, but not the
    # backward layer. We actually want it to re-use the forward layer we hand
    # it.
    keras_bidirectional.forward_layer = keras_forward

    y_keras = sl.Sequence(
        keras_bidirectional(x.values, mask=x.mask > 0.0, training=False), x.mask
    )
    y, y_keras = self.evaluate([y, y_keras])
    self.assertSequencesEqual(y, y_keras)

  def test_bidirectional(self):
    forward = sl.Serial(
        [
            sl.Conv1D(filters=2, kernel_size=3, strides=2),
        ]
    )
    backward = sl.Serial([
        sl.Conv1D(filters=4, kernel_size=3, strides=2),
        sl.Dense(3),
    ])

    l = sl.Bidirectional(forward, backward)

    # Not applicable for Bidirectional.
    self.assertEqual(l.block_size, 0)
    self.assertEqual(l.output_ratio, forward.output_ratio)
    self.assertEqual(l.output_ratio, backward.output_ratio)
    self.assertEqual(
        l.get_output_dtype(tf.float16), forward.get_output_dtype(tf.float16)
    )
    self.assertEqual(
        l.get_output_dtype(tf.float16), backward.get_output_dtype(tf.float16)
    )
    self.assertFalse(l.supports_step)

    batch_size, channels = 2, 1
    for i in range(5, 8):
      time = i + 1
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([5]))
      self.verify_contract(l, x, training=False)
      # kernel and bias for each conv and dense layer.
      self.assertLen(l.trainable_variables, 6)
      self.assertLen(l.variables, 6)
    self.verify_tflite_step(l, x)

  def test_emit(self):
    forward = sl.Emit(name='forward_emit')
    backward = sl.Emit(name='backward_emit')
    l = sl.Bidirectional(forward, backward)

    input_spec = tf.TensorSpec(tf.TensorShape([None]), dtype=tf.float16)
    emit_spec = l.get_emit_specs(input_spec)
    forward_emit_spec, backward_emit_spec = emit_spec
    tf.nest.assert_same_structure(
        forward_emit_spec, forward.get_emit_specs(input_spec)
    )
    tf.nest.assert_same_structure(
        backward_emit_spec, backward.get_emit_specs(input_spec)
    )

    self.assertEqual(forward_emit_spec.values, input_spec)
    self.assertEqual(backward_emit_spec.values, input_spec)

    x = self.random_sequence(2, 3, 5)
    _, emits = l.layer_with_emits(x, training=False)
    tf.nest.assert_same_structure(emits, emit_spec)
    # One emit for each layer.
    self.assertLen(list(sl.experimental_iterate_emits(l, emits)), 3)

  def test_incompatible_output_ratio(self):
    # Output ratios do not match.
    with self.assertRaises(ValueError):
      sl.Bidirectional(
          sl.Conv1D(filters=2, kernel_size=3, strides=2),
          sl.Conv1D(filters=2, kernel_size=3, strides=3),
      )


if __name__ == '__main__':
  tf.test.main()
