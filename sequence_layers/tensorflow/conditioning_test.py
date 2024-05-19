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
"""Tests for sequence_layers.tensorflow.conditioning."""

from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import conditioning
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v2 as tf

IDENTITY = conditioning.Conditioning.Projection.IDENTITY
LINEAR = conditioning.Conditioning.Projection.LINEAR
LINEAR_AFFINE = conditioning.Conditioning.Projection.LINEAR_AFFINE
ADD = conditioning.Conditioning.Combination.ADD
CONCAT = conditioning.Conditioning.Combination.CONCAT
AFFINE = conditioning.Conditioning.Combination.AFFINE


def _float_tensor(values):
  return tf.constant(values, dtype=tf.float32)


class ConditioningTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (IDENTITY, ADD, tuple(), tuple(), tuple()),
      (IDENTITY, ADD, tuple(), (5,), (5,)),
      (IDENTITY, ADD, tuple(), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2,), tuple(), (2,)),
      (IDENTITY, ADD, (2, 5), tuple(), (2, 5)),
      (IDENTITY, ADD, (5,), (5,), (5,)),
      (IDENTITY, ADD, (5,), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2, 5), (5,), (2, 5)),
      (IDENTITY, ADD, (3, 1, 5), (2, 5), (3, 2, 5)),
      (IDENTITY, ADD, (2, 5), (3, 1, 5), (3, 2, 5)),
      (IDENTITY, CONCAT, tuple(), tuple(), (2,)),
      (IDENTITY, CONCAT, tuple(), (5,), (6,)),
      (IDENTITY, CONCAT, tuple(), (2, 5), (2, 6)),
      (IDENTITY, CONCAT, (2,), tuple(), (3,)),
      (IDENTITY, CONCAT, (2, 5), tuple(), (2, 6)),
      (IDENTITY, CONCAT, (5,), (7,), (12,)),
      (IDENTITY, CONCAT, (5,), (2, 7), (2, 12)),
      (IDENTITY, CONCAT, (2, 5), (7,), (2, 12)),
      (IDENTITY, CONCAT, (3, 1, 5), (2, 7), (3, 2, 12)),
      (IDENTITY, CONCAT, (2, 5), (3, 1, 7), (3, 2, 12)),
      (LINEAR, ADD, tuple(), tuple(), tuple()),
      (LINEAR, ADD, tuple(), (5,), tuple()),
      (LINEAR, ADD, tuple(), (2, 5), tuple()),
      (LINEAR, ADD, (2,), tuple(), (2,)),
      (LINEAR, ADD, (2, 5), tuple(), (2, 5)),
      (LINEAR, ADD, (5,), (7,), (5,)),
      (LINEAR, ADD, (7,), (2, 5), (7,)),
      (LINEAR, ADD, (2, 5), (7,), (2, 5)),
      (LINEAR, ADD, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR, ADD, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR_AFFINE, AFFINE, tuple(), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (5,), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (2, 5), tuple()),
      (LINEAR_AFFINE, AFFINE, (2,), tuple(), (2,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), tuple(), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (5,)),
      (LINEAR_AFFINE, AFFINE, (7,), (2, 5), (7,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), (7,), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR_AFFINE, AFFINE, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR, CONCAT, tuple(), tuple(), (2,)),
      (LINEAR, CONCAT, tuple(), (5,), (2,)),
      (LINEAR, CONCAT, tuple(), (2, 5), (2,)),
      (LINEAR, CONCAT, (2,), tuple(), (4,)),
      (LINEAR, CONCAT, (2, 5), tuple(), (2, 10)),
      (LINEAR, CONCAT, (5,), (7,), (10,)),
      (LINEAR, CONCAT, (7,), (2, 5), (14,)),
      (LINEAR, CONCAT, (2, 5), (7,), (2, 10)),
      (LINEAR, CONCAT, (3, 1, 5), (2, 7), (3, 1, 10)),
      (LINEAR, CONCAT, (2, 7), (3, 1, 5), (2, 14)),
  )
  def test_conditioning(
      self,
      projection,
      combination,
      x_channel_shape,
      c_channel_shape,
      expected_channel_shape,
  ):
    batch_size, time = 2, 4
    x = self.random_sequence(batch_size, time, *x_channel_shape)
    c = self.random_sequence(batch_size, time, *c_channel_shape)
    with tf.name_scope('test'):
      l = sl.Conditioning('test', projection, combination, 'conditioning_layer')
    constants = {'test': c}
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants),
        tf.TensorShape(expected_channel_shape),
    )
    self.verify_contract(
        l, x, training=False, pad_constants=True, constants=constants
    )
    expected_variable_names = {
        IDENTITY: [],
        LINEAR: [
            'test/conditioning_layer/dense/kernel:0',
            'test/conditioning_layer/dense/bias:0',
        ],
        LINEAR_AFFINE: [
            'test/conditioning_layer/dense/kernel:0',
            'test/conditioning_layer/dense/bias:0',
        ],
    }[projection]
    self.assertLen(l.variables, len(expected_variable_names))
    self.assertLen(l.trainable_variables, len(expected_variable_names))
    self.assertCountEqual(
        [v.name for v in l.variables], expected_variable_names
    )
    # TODO(soroosho): The tflite test fails on a subset of cases with
    # 2D x sequences, likely due to an issue in tflite/toco creating empty
    # tensors that I need to investigate.
    if x_channel_shape:
      self.verify_tflite_step(l, x, constants=constants)

  @parameterized.parameters(
      (IDENTITY, ADD, tuple(), tuple(), tuple()),
      (IDENTITY, ADD, tuple(), (5,), (5,)),
      (IDENTITY, ADD, tuple(), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2,), tuple(), (2,)),
      (IDENTITY, ADD, (2, 5), tuple(), (2, 5)),
      (IDENTITY, ADD, (5,), (5,), (5,)),
      (IDENTITY, ADD, (5,), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2, 5), (5,), (2, 5)),
      (IDENTITY, ADD, (3, 1, 5), (2, 5), (3, 2, 5)),
      (IDENTITY, ADD, (2, 5), (3, 1, 5), (3, 2, 5)),
      (IDENTITY, CONCAT, tuple(), tuple(), (2,)),
      (IDENTITY, CONCAT, tuple(), (5,), (6,)),
      (IDENTITY, CONCAT, tuple(), (2, 5), (2, 6)),
      (IDENTITY, CONCAT, (2,), tuple(), (3,)),
      (IDENTITY, CONCAT, (2, 5), tuple(), (2, 6)),
      (IDENTITY, CONCAT, (5,), (7,), (12,)),
      (IDENTITY, CONCAT, (5,), (2, 7), (2, 12)),
      (IDENTITY, CONCAT, (2, 5), (7,), (2, 12)),
      (IDENTITY, CONCAT, (3, 1, 5), (2, 7), (3, 2, 12)),
      (IDENTITY, CONCAT, (2, 5), (3, 1, 7), (3, 2, 12)),
      (LINEAR, ADD, tuple(), tuple(), tuple()),
      (LINEAR, ADD, tuple(), (5,), tuple()),
      (LINEAR, ADD, tuple(), (2, 5), tuple()),
      (LINEAR, ADD, (2,), tuple(), (2,)),
      (LINEAR, ADD, (2, 5), tuple(), (2, 5)),
      (LINEAR, ADD, (5,), (7,), (5,)),
      (LINEAR, ADD, (7,), (2, 5), (7,)),
      (LINEAR, ADD, (2, 5), (7,), (2, 5)),
      (LINEAR, ADD, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR, ADD, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR_AFFINE, AFFINE, tuple(), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (5,), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (2, 5), tuple()),
      (LINEAR_AFFINE, AFFINE, (2,), tuple(), (2,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), tuple(), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (5,)),
      (LINEAR_AFFINE, AFFINE, (7,), (2, 5), (7,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), (7,), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR_AFFINE, AFFINE, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR, CONCAT, tuple(), tuple(), (2,)),
      (LINEAR, CONCAT, tuple(), (5,), (2,)),
      (LINEAR, CONCAT, tuple(), (2, 5), (2,)),
      (LINEAR, CONCAT, (2,), tuple(), (4,)),
      (LINEAR, CONCAT, (2, 5), tuple(), (2, 10)),
      (LINEAR, CONCAT, (5,), (7,), (10,)),
      (LINEAR, CONCAT, (7,), (2, 5), (14,)),
      (LINEAR, CONCAT, (2, 5), (7,), (2, 10)),
      (LINEAR, CONCAT, (3, 1, 5), (2, 7), (3, 1, 10)),
      (LINEAR, CONCAT, (2, 7), (3, 1, 5), (2, 14)),
  )
  def test_conditioning_tensor(
      self,
      projection,
      combination,
      x_channel_shape,
      c_channel_shape,
      expected_channel_shape,
  ):
    batch_size, time = 2, 4
    x = self.random_sequence(batch_size, time, *x_channel_shape)
    c = tf.constant(
        np.random.normal(size=(batch_size,) + c_channel_shape).astype(
            np.float32
        )
    )
    with tf.name_scope('test'):
      l = sl.Conditioning('test', projection, combination, 'conditioning_layer')
    constants = {'test': c}
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants),
        tf.TensorShape(expected_channel_shape),
    )
    self.verify_contract(
        l, x, training=False, pad_constants=True, constants=constants
    )
    expected_variable_names = {
        IDENTITY: [],
        LINEAR: [
            'test/conditioning_layer/dense/kernel:0',
            'test/conditioning_layer/dense/bias:0',
        ],
        LINEAR_AFFINE: [
            'test/conditioning_layer/dense/kernel:0',
            'test/conditioning_layer/dense/bias:0',
        ],
    }[projection]
    self.assertLen(l.variables, len(expected_variable_names))
    self.assertLen(l.trainable_variables, len(expected_variable_names))
    self.assertCountEqual(
        [v.name for v in l.variables], expected_variable_names
    )
    # TODO(soroosho): The tflite test fails on a subset of cases with
    # 2D x sequences, likely due to an issue in tflite/toco creating empty
    # tensors that I need to investigate.
    if x_channel_shape:
      self.verify_tflite_step(l, x, constants=constants)

  @parameterized.parameters(
      (IDENTITY, ADD, (5,), (6,)),
      (IDENTITY, ADD, (3, 4, 5), (2, 5)),
      (IDENTITY, CONCAT, (2, 5), (3, 7)),
  )
  def test_conditioning_invalid_shapes(
      self, projection, combination, x_channel_shape, c_channel_shape
  ):
    batch_size, time = 2, 4
    x = self.random_sequence(batch_size, time, *x_channel_shape)
    c = self.random_sequence(batch_size, time, *c_channel_shape)
    l = sl.Conditioning('test', projection, combination, 'conditioning_layer')
    constants = {'test': c}
    s0 = l.get_initial_state(x, constants)

    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x, constants)
    with self.assertRaises(Exception):
      l.layer(x, training=False, initial_state=s0, constants=constants)
    with self.assertRaises(Exception):
      l.step(x, state=s0, training=False, constants=constants)

  def test_condition_sequence_add_combination(self):
    # [2, 3, 3]
    x = sl.Sequence(
        _float_tensor([
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
            [[0, 0, 0], [2, 4, 6], [3, 5, 7]],
        ]),
        _float_tensor([[1, 1, 0], [0, 1, 1]]),
    )

    # [2, 3, 2, 3]
    c = sl.Sequence(
        _float_tensor([
            [
                [[0, 0, 0], [0, 0, 0]],
                [[-1, -2, -3], [-5, -6, -7]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            [
                [[-4, -3, -2], [-8, -7, -6]],
                [[0, 0, 0], [0, 0, 0]],
                [[-2, -4, -6], [-1, -3, -5]],
            ],
        ]),
        _float_tensor([[0, 1, 0], [1, 0, 1]]),
    )

    # [2, 3, 2, 3]
    expected_conditioned_x = sl.Sequence(
        _float_tensor([
            [
                [[0, 0, 0], [0, 0, 0]],
                [[4 - 1, 5 - 2, 6 - 3], [4 - 5, 5 - 6, 6 - 7]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
                [[3 - 2, 5 - 4, 7 - 6], [3 - 1, 5 - 3, 7 - 5]],
            ],
        ]),
        _float_tensor([[0, 1, 0], [0, 0, 1]]),
    )

    l = sl.Conditioning('test', IDENTITY, ADD, 'conditioning_layer')
    constants = {'test': c}

    s0 = l.get_initial_state(x, constants)
    conditioned_x = l.layer(
        x, training=False, initial_state=s0, constants=constants
    )

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_condition_tensor_add_combination(self):
    # [2, 3, 3]
    x = sl.Sequence(
        _float_tensor([
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
            [[0, 0, 0], [2, 4, 6], [3, 5, 7]],
        ]),
        _float_tensor([[1, 1, 0], [0, 1, 1]]),
    )

    # [2, 2, 3]
    c = _float_tensor(
        [[[-1, -2, -3], [-5, -6, -7]], [[-4, -3, -2], [-8, -7, -6]]]
    )

    # [2, 3, 2, 3]
    expected_conditioned_x = sl.Sequence(
        _float_tensor([
            [
                [[1 - 1, 2 - 2, 3 - 3], [1 - 5, 2 - 6, 3 - 7]],
                [[4 - 1, 5 - 2, 6 - 3], [4 - 5, 5 - 6, 6 - 7]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, 0]],
                [[2 - 4, 4 - 3, 6 - 2], [2 - 8, 4 - 7, 6 - 6]],
                [[3 - 4, 5 - 3, 7 - 2], [3 - 8, 5 - 7, 7 - 6]],
            ],
        ]),
        _float_tensor([[1, 1, 0], [0, 1, 1]]),
    )

    l = sl.Conditioning('test', IDENTITY, ADD, 'conditioning_layer')
    constants = {'test': c}

    s0 = l.get_initial_state(x, constants)
    conditioned_x = l.layer(
        x, training=False, initial_state=s0, constants=constants
    )

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_condition_sequence_concat_combination(self):
    # [2, 3, 2, 4]
    x = sl.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[4, 3, 2, 1], [8, 7, 6, 5]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[2, 4, 6, 8], [1, 3, 5, 7]],
            ],
        ]),
        _float_tensor([[0, 1, 0], [1, 0, 1]]),
    )

    # [2, 3, 2]
    c = sl.Sequence(
        _float_tensor([
            [[-1, -2], [-3, -4], [-5, -6]],
            [[-1, -3], [-3, -5], [-7, -9]],
        ]),
        _float_tensor([[1, 1, 0], [0, 1, 1]]),
    )

    # [2, 3, 2, 6]
    expected_conditioned_x = sl.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[1, 2, 3, 4, -3, -4], [5, 6, 7, 8, -3, -4]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[2, 4, 6, 8, -7, -9], [1, 3, 5, 7, -7, -9]],
            ],
        ]),
        _float_tensor([[0, 1, 0], [0, 0, 1]]),
    )

    l = sl.Conditioning('test', IDENTITY, CONCAT, 'conditioning_layer')
    constants = {'test': c}

    s0 = l.get_initial_state(x, constants)
    conditioned_x = l.layer(
        x, training=False, initial_state=s0, constants=constants
    )

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_condition_tensor_concat_combination(self):
    # [2, 3, 2, 4]
    x = sl.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[4, 3, 2, 1], [8, 7, 6, 5]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[2, 4, 6, 8], [1, 3, 5, 7]],
            ],
        ]),
        _float_tensor([[0, 1, 0], [1, 0, 1]]),
    )

    # [2, 2]
    c = _float_tensor([[-1, -2], [-3, -4]])

    # [2, 3, 2, 6]
    expected_conditioned_x = sl.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[1, 2, 3, 4, -1, -2], [5, 6, 7, 8, -1, -2]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            ],
            [
                [[4, 3, 2, 1, -3, -4], [8, 7, 6, 5, -3, -4]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[2, 4, 6, 8, -3, -4], [1, 3, 5, 7, -3, -4]],
            ],
        ]),
        _float_tensor([[0, 1, 0], [1, 0, 1]]),
    )

    l = sl.Conditioning('test', IDENTITY, CONCAT, 'conditioning_layer')
    constants = {'test': c}

    s0 = l.get_initial_state(x, constants)
    conditioned_x = l.layer(
        x, training=False, initial_state=s0, constants=constants
    )

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_conditioned_values_add_combination_2d_sequences(self):
    # [2, 3]
    x = sl.Sequence(
        _float_tensor([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        _float_tensor([[0, 1, 0], [1, 0, 1]]),
    )

    # [2, 3]
    c = sl.Sequence(
        _float_tensor([
            [2, 4, 6],
            [3, 5, 7],
        ]),
        _float_tensor([[1, 1, 0], [0, 1, 1]]),
    )

    # [2, 3]
    expected_conditioned_x = sl.Sequence(
        _float_tensor([
            [0, 2 + 4, 0],
            [0, 0, 6 + 7],
        ]),
        _float_tensor([[0, 1, 0], [0, 0, 1]]),
    )

    l = sl.Conditioning('test', IDENTITY, ADD, 'conditioning_layer')
    constants = {'test': c}

    s0 = l.get_initial_state(x, constants)
    conditioned_x = l.layer(
        x, training=False, initial_state=s0, constants=constants
    )

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_conditioned_values_concat_combination_2d_sequences(self):
    # [2, 3]
    x = sl.Sequence(
        _float_tensor([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        _float_tensor([[0, 1, 0], [1, 0, 1]]),
    )

    # [2, 3]
    c = sl.Sequence(
        _float_tensor([
            [2, 4, 6],
            [3, 5, 7],
        ]),
        _float_tensor([[1, 1, 0], [0, 1, 1]]),
    )

    # [2, 3, 2]
    expected_conditioned_x = sl.Sequence(
        _float_tensor([
            [[0, 0], [2, 4], [0, 0]],
            [[0, 0], [0, 0], [6, 7]],
        ]),
        _float_tensor([[0, 1, 0], [0, 0, 1]]),
    )

    l = sl.Conditioning('test', IDENTITY, CONCAT, 'conditioning_layer')
    constants = {'test': c}

    s0 = l.get_initial_state(x, constants)
    conditioned_x = l.layer(
        x, training=False, initial_state=s0, constants=constants
    )

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_conditioning_block_sizes(self):
    time, length = 9, 5

    x = sl.Sequence(
        tf.range(time)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask([length], time, dtype=sl.MASK_DTYPE),
    )

    c = sl.Sequence(
        10 * tf.range(1, time + 1)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask([length], time, dtype=sl.MASK_DTYPE),
    )

    cond = sl.Conditioning('test', IDENTITY, ADD)
    y_layer = cond.layer(x, training=True, constants={'test': c})

    for block_size in range(2, time + 2):
      l = sl.Blockwise(cond, block_size=block_size)
      y_block = l.layer(x, training=True, constants={'test': c})
      self.assertSequencesEqual(y_block, y_layer)

  @parameterized.parameters(
      (IDENTITY, ADD, (5,), (5,), (5,)),
      (IDENTITY, ADD, (5,), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2, 5), (5,), (2, 5)),
      (IDENTITY, ADD, (3, 1, 5), (2, 5), (3, 2, 5)),
      (IDENTITY, ADD, (2, 5), (3, 1, 5), (3, 2, 5)),
      (IDENTITY, CONCAT, (5,), (7,), (12,)),
      (IDENTITY, CONCAT, (5,), (2, 7), (2, 12)),
      (IDENTITY, CONCAT, (2, 5), (7,), (2, 12)),
      (IDENTITY, CONCAT, (3, 1, 5), (2, 7), (3, 2, 12)),
      (IDENTITY, CONCAT, (2, 5), (3, 1, 7), (3, 2, 12)),
      (LINEAR, ADD, (5,), (7,), (5,)),
      (LINEAR, ADD, (7,), (2, 5), (7)),
      (LINEAR, ADD, (2, 5), (7,), (2, 5)),
      (LINEAR, ADD, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR, ADD, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (5,)),
      (LINEAR_AFFINE, AFFINE, (7,), (2, 5), (7)),
      (LINEAR_AFFINE, AFFINE, (2, 5), (7,), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR_AFFINE, AFFINE, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR, CONCAT, (5,), (7,), (10,)),
      (LINEAR, CONCAT, (7,), (2, 5), (14)),
      (LINEAR, CONCAT, (2, 5), (7,), (2, 10)),
      (LINEAR, CONCAT, (3, 1, 5), (2, 7), (3, 1, 10)),
      (LINEAR, CONCAT, (2, 7), (3, 1, 5), (2, 14)),
  )
  def test_upsample_conditioning(
      self,
      projection,
      combination,
      x_channel_shape,
      c_channel_shape,
      expected_channel_shape,
  ):
    batch_size, time, frames, upsample_ratio = 2, 10, 5, 2
    x = self.random_sequence(batch_size, time, *x_channel_shape)
    c = self.random_sequence(batch_size, frames, *c_channel_shape)
    with tf.name_scope('test'):
      l = sl.UpsampleConditioning(
          'test',
          projection=projection,
          combination=combination,
          upsample_ratio=upsample_ratio,
      )
    constants = {'test': c}
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants),
        tf.TensorShape(expected_channel_shape),
    )
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        pad_constants=True,
        pad_constants_ratio=upsample_ratio,
    )
    expected_variable_names = {
        IDENTITY: [],
        LINEAR: [
            'test/upsample_conditioning/dense/kernel:0',
            'test/upsample_conditioning/dense/bias:0',
        ],
        LINEAR_AFFINE: [
            'test/upsample_conditioning/dense/kernel:0',
            'test/upsample_conditioning/dense/bias:0',
        ],
    }[projection]
    self.assertLen(l.variables, len(expected_variable_names))
    self.assertLen(l.trainable_variables, len(expected_variable_names))
    self.assertCountEqual(
        [v.name for v in l.variables], expected_variable_names
    )
    self.verify_tflite_step(
        l, x, constants=constants, use_flex=True, rtol=1e-6, atol=1e-6
    )

  def test_upsample_conditioning_identity_concat(self):
    frames, upsample_ratio = 3, 3
    time = frames * upsample_ratio

    x = sl.Sequence(
        tf.range(time)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask([7], time, dtype=tf.float32),
    )

    c = sl.Sequence(
        10 * tf.range(1, frames + 1)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask([2], frames, dtype=tf.float32),
    )

    l = sl.UpsampleConditioning(
        'test',
        projection=IDENTITY,
        combination=CONCAT,
        upsample_ratio=upsample_ratio,
    )
    y = l.layer(x, training=True, constants={'test': c})
    self.assertSequencesClose(
        y,
        sl.Sequence(
            np.array([[[0, 10], [1, 10], [2, 10],
                       [3, 20], [4, 20], [5, 20],
                       [0, 0], [0, 0], [0, 0]]]),
            np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0]])))  # pyformat: disable

  def test_upsample_conditioning_identity_add(self):
    frames, upsample_ratio = 3, 3
    time = frames * upsample_ratio

    x = sl.Sequence(
        tf.range(time)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask([7], time, dtype=tf.float32),
    )

    c = sl.Sequence(
        10 * tf.range(1, frames + 1)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask([2], frames, dtype=tf.float32),
    )

    l = sl.UpsampleConditioning(
        'test',
        projection=IDENTITY,
        combination=ADD,
        upsample_ratio=upsample_ratio,
    )
    y = l.layer(x, training=True, constants={'test': c})
    self.assertSequencesClose(
        y,
        sl.Sequence(
            np.array([[[10], [11], [12],
                       [23], [24], [25],
                       [0], [0], [0]]]),
            np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0]])))  # pyformat: disable

  def test_upsample_conditioning_block_sizes(self):
    frames, frame_length, upsample_ratio = 3, 2, 3
    time = frames * upsample_ratio

    x = sl.Sequence(
        tf.range(time)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask(
            [frame_length * upsample_ratio], time, dtype=sl.MASK_DTYPE
        ),
    )

    c = sl.Sequence(
        10 * tf.range(1, frames + 1)[tf.newaxis, :, tf.newaxis],
        tf.sequence_mask([frame_length], frames, dtype=sl.MASK_DTYPE),
    )

    upsample = sl.UpsampleConditioning(
        'test',
        projection=IDENTITY,
        combination=ADD,
        upsample_ratio=upsample_ratio,
    )

    y_layer = upsample.layer(x, training=True, constants={'test': c})

    for block_size in range(2, time + 2):
      l = sl.Blockwise(upsample, block_size=block_size)
      y_block = l.layer(x, training=True, constants={'test': c})
      self.assertSequencesEqual(y_block, y_layer)

  @parameterized.parameters(
      (IDENTITY, ADD, tuple(), tuple(), tuple()),
      (IDENTITY, ADD, tuple(), (5,), (5,)),
      (IDENTITY, ADD, tuple(), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2,), tuple(), (2,)),
      (IDENTITY, ADD, (2, 5), tuple(), (2, 5)),
      (IDENTITY, ADD, (5,), (5,), (5,)),
      (IDENTITY, ADD, (5,), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2, 5), (5,), (2, 5)),
      (IDENTITY, ADD, (3, 1, 5), (2, 5), (3, 2, 5)),
      (IDENTITY, ADD, (2, 5), (3, 1, 5), (3, 2, 5)),
      (IDENTITY, CONCAT, tuple(), tuple(), (2,)),
      (IDENTITY, CONCAT, tuple(), (5,), (6,)),
      (IDENTITY, CONCAT, tuple(), (2, 5), (2, 6)),
      (IDENTITY, CONCAT, (2,), tuple(), (3,)),
      (IDENTITY, CONCAT, (2, 5), tuple(), (2, 6)),
      (IDENTITY, CONCAT, (5,), (7,), (12,)),
      (IDENTITY, CONCAT, (5,), (2, 7), (2, 12)),
      (IDENTITY, CONCAT, (2, 5), (7,), (2, 12)),
      (IDENTITY, CONCAT, (3, 1, 5), (2, 7), (3, 2, 12)),
      (IDENTITY, CONCAT, (2, 5), (3, 1, 7), (3, 2, 12)),
      (LINEAR, ADD, tuple(), tuple(), tuple()),
      (LINEAR, ADD, tuple(), (5,), tuple()),
      (LINEAR, ADD, tuple(), (2, 5), tuple()),
      (LINEAR, ADD, (2,), tuple(), (2,)),
      (LINEAR, ADD, (2, 5), tuple(), (2, 5)),
      (LINEAR, ADD, (5,), (7,), (5,)),
      (LINEAR, ADD, (7,), (2, 5), (7)),
      (LINEAR, ADD, (2, 5), (7,), (2, 5)),
      (LINEAR, ADD, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR, ADD, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR_AFFINE, AFFINE, tuple(), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (5,), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (2, 5), tuple()),
      (LINEAR_AFFINE, AFFINE, (2,), tuple(), (2,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), tuple(), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (5,)),
      (LINEAR_AFFINE, AFFINE, (7,), (2, 5), (7)),
      (LINEAR_AFFINE, AFFINE, (2, 5), (7,), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR_AFFINE, AFFINE, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR, CONCAT, tuple(), tuple(), (2,)),
      (LINEAR, CONCAT, tuple(), (5,), (2,)),
      (LINEAR, CONCAT, tuple(), (2, 5), (2,)),
      (LINEAR, CONCAT, (2,), tuple(), (4,)),
      (LINEAR, CONCAT, (2, 5), tuple(), (2, 10)),
      (LINEAR, CONCAT, (5,), (7,), (10,)),
      (LINEAR, CONCAT, (7,), (2, 5), (14)),
      (LINEAR, CONCAT, (2, 5), (7,), (2, 10)),
      (LINEAR, CONCAT, (3, 1, 5), (2, 7), (3, 1, 10)),
      (LINEAR, CONCAT, (2, 7), (3, 1, 5), (2, 14)),
  )
  def test_noise_conditioning(
      self,
      projection,
      combination,
      x_channel_shape,
      noise_channel_shape,
      expected_channel_shape,
  ):
    batch_size, time = 2, 4
    x = self.random_sequence(batch_size, time, *x_channel_shape)
    with tf.name_scope('test'):
      # Noise is all zeros so output is deterministic.
      noise_sampler = sl.NormalSampler(0.0, 0.0)
      l = sl.NoiseConditioning(
          noise_channel_shape,
          noise_sampler,
          projection,
          combination,
          'noise_conditioning_layer',
      )
    constants = {}
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants),
        tf.TensorShape(expected_channel_shape),
    )
    self.verify_contract(
        l, x, training=False, pad_constants=True, constants=constants
    )
    expected_variable_names = {
        IDENTITY: [],
        LINEAR: [
            'test/noise_conditioning_layer/dense/kernel:0',
            'test/noise_conditioning_layer/dense/bias:0',
        ],
        LINEAR_AFFINE: [
            'test/noise_conditioning_layer/dense/kernel:0',
            'test/noise_conditioning_layer/dense/bias:0',
        ],
    }[projection]
    self.assertLen(l.variables, len(expected_variable_names))
    self.assertLen(l.trainable_variables, len(expected_variable_names))
    self.assertCountEqual(
        [v.name for v in l.variables], expected_variable_names
    )
    # TODO(soroosho): The tflite test fails on a subset of cases with
    # 2D x sequences, likely due to an issue in tflite/toco creating empty
    # tensors that I need to investigate.
    if x_channel_shape:
      self.verify_tflite_step(l, x, constants=constants)


if __name__ == '__main__':
  tf.test.main()
