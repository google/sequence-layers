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
"""Tests for sequence_layers.tensorflow.simple."""

import itertools

from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class ScaleTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),))  # pyformat: disable
  def test_scale(self, shape):
    x = self.random_sequence(*shape)
    l = sl.Scale(2.0)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_scale_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    with test_util.keras_precision_policy_scope(precision_policy):
      x = self.random_sequence(2, 3, 5, dtype=utils.compute_dtype())
      with tf.name_scope('test'):
        l = sl.Scale(2.0)
      rtol, atol = test_util.rtol_atol_for_dtype(x.values.dtype)
      x_np, y_np = self.verify_contract(
          l, x, training=True, rtol=rtol, atol=atol
      )
      self.assertEqual(x_np.dtype, utils.compute_dtype())
      self.assertEqual(y_np.dtype, utils.compute_dtype())
      for variable in l.variables:
        self.assertEqual(variable.dtype, utils.variable_dtype())


class TranslateTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),))  # pyformat: disable
  def test_translate(self, shape):
    x = self.random_sequence(*shape)
    l = sl.Translate(-2.0)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_translate_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    with test_util.keras_precision_policy_scope(precision_policy):
      x = self.random_sequence(2, 3, 5, dtype=utils.compute_dtype())
      with tf.name_scope('test'):
        l = sl.Translate(-2.0)
      x_np, y_np = self.verify_contract(l, x, training=True)
      self.assertEqual(x_np.dtype, utils.compute_dtype())
      self.assertEqual(y_np.dtype, utils.compute_dtype())
      for variable in l.variables:
        self.assertEqual(variable.dtype, utils.variable_dtype())


class DropoutTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),))  # pyformat: disable
  def test_dropout(self, shape):
    x = self.random_sequence(*shape)
    l = sl.Dropout(0.5)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x)

  def test_dropout_noise_shape(self):
    x = sl.Sequence(tf.ones((2, 3, 5, 9)), tf.ones((2, 3)))
    l = sl.Dropout(0.5, noise_shape=[1, None, None, None])
    y = l.layer(x, training=True)
    self.assertAllEqual(y.values[0], y.values[1])

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),))  # pyformat: disable
  def test_dropout_disabled(self, shape):
    x = self.random_sequence(*shape)
    l = sl.Dropout(0.0)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=True)
    self.assertEmpty(l.variables)


class DeterministicDropoutTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),),
      # Internally 128 switches implementations to a tf.map_fn.
      (tf.TensorShape((2, 128, 5)),),
      (tf.TensorShape((2, 128, 5, 9)),))  # pyformat: disable
  def test_deterministic_dropout(self, shape):
    x = self.random_sequence(*shape)
    l = sl.DeterministicDropout(0.5, initial_seed_name='step_seed')
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    constants = {'step_seed': tf.constant(1234, tf.int32)}
    self.verify_contract(l, x, training=False, constants=constants)
    self.verify_contract(l, x, training=True, constants=constants)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x, constants=constants)

    y = l.layer(x, training=True, constants=constants)
    y_same = l.layer(x, training=True, constants=constants)
    constants = {'step_seed': tf.constant(4567, tf.int32)}
    y_different = l.layer(x, training=True, constants=constants)
    self.assertSequencesClose(y, y_same)
    self.assertSequencesNotClose(y, y_different)

  # Internally 128 switches implementations to a tf.map_fn.
  @parameterized.parameters(3, 128)
  def test_deterministic_dropout_noise_shape(self, time):
    x = sl.Sequence(tf.ones((2, time, 5, 9)), tf.ones((2, time)))
    l = sl.DeterministicDropout(
        0.5, noise_shape=[1, None, None, None], initial_seed_name='step_seed'
    )

    constants = {'step_seed': tf.constant(1234, tf.int32)}

    y = l.layer(x, training=True, constants=constants)
    self.assertAllEqual(y.values[0], y.values[1])

    constants = {'step_seed': tf.constant(4567, tf.int32)}
    y_different = l.layer(x, training=True, constants=constants)
    self.assertAllEqual(y_different.values[0], y_different.values[1])
    self.assertNotAllEqual(y.values[0], y_different.values[0])

  def test_deterministic_dropout_always_dropout(self):
    x = sl.Sequence(tf.ones((2, 3, 5, 9)), tf.ones((2, 3)))
    l = sl.DeterministicDropout(
        0.5, always_dropout=True, initial_seed_name='step_seed'
    )

    constants = {'step_seed': tf.constant(1234, tf.int32)}
    y = l.layer(x, training=False, constants=constants)
    y_same = l.layer(x, training=False, constants=constants)
    constants = {'step_seed': tf.constant(4567, tf.int32)}
    y_different = l.layer(x, training=False, constants=constants)
    self.assertSequencesClose(y, y_same)
    self.assertSequencesNotClose(y, y_different)


class NoiseTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (tf.TensorShape((2, 3)),
           tf.TensorShape((2, 3, 5)),
           tf.TensorShape((2, 3, 5, 9))),
          (sl.UniformSampler(0.0, 1.0),
           sl.NormalSampler(0.0, 1.0),
           sl.TruncatedNormalSampler(0.0, 1.0))))  # pyformat: disable
  def test_noise(self, shape, sampler):
    x = self.random_sequence(*shape)
    l = sl.Noise(sampler, training_only=True)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x)

  def test_noise_training_only(self):
    x = self.random_sequence(2, 3, 5)
    l = sl.Noise(sl.NormalSampler(1.0, 0.0), training_only=True)

    @tf.function
    def apply_noise(x):
      return l.layer(x, training=True)

    # Test tracing with variable batch size.
    _ = apply_noise.get_concrete_function(
        sl.Sequence(
            tf.TensorSpec([None, 3, 5], tf.float32),
            tf.TensorSpec([None, 3], sl.MASK_DTYPE),
        )
    )

    y = apply_noise(x)
    y_expected = sl.Sequence(
        x.values + tf.ones_like(x.values), x.mask
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)

  def test_noise_variable_batch(self):
    x = self.random_sequence(2, 3, 5)
    l = sl.Noise(sl.NormalSampler(1.0, 0.0), training_only=True)
    y = l.layer(x, training=True)
    y_expected = sl.Sequence(
        x.values + tf.ones_like(x.values), x.mask
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)
    y = l.layer(x, training=False)
    self.assertSequencesClose(y, x)


class GatedActivationTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (lambda: sl.GatedUnit(None, None),  # Bilinear
           lambda: sl.GatedUnit(None, tf.nn.swish),  # SwiGLU
           lambda: sl.GatedUnit(None, tf.nn.gelu),  # GeGLU
           lambda: sl.GatedUnit(lambda x: x, None),  # Bilinear
           lambda: sl.GatedUnit(tf.nn.swish, tf.nn.tanh),
           sl.GatedTanhUnit,
           sl.GatedLinearUnit),
          (tf.TensorShape((2, 3, 6)), tf.TensorShape((2, 3, 5, 10))))
      )  # pyformat: disable
  def test_gated_activation(self, layer_class, shape):
    x = self.random_sequence(*shape)
    l = layer_class()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + [shape[-1] // 2]
    )
    self.verify_contract(l, x, training=True)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(
      (sl.Abs, tf.math.abs, True, (tf.float32, tf.complex64)),
      (sl.Exp, tf.math.exp, True, (tf.float32,)),
      # TODO(rryan): Handle NaNs in masked regions better.
      # (sl.Log, tf.math.log, True, (tf.float32,)),
      (sl.Relu, tf.nn.relu, True, (tf.float32,)),
      (sl.Elu, tf.nn.elu, True, (tf.float32,)),
      (sl.Sigmoid, tf.math.sigmoid, True, (tf.float32,)),
      (sl.Softmax, tf.math.softmax, True, (tf.float32,)),
      # TODO(b/143444502): Not supported on tf.lite.
      (sl.Softplus, tf.math.softplus, False, (tf.float32,)),
      (sl.Tanh, tf.math.tanh, True, (tf.float32,)),
      (sl.Swish, tf.nn.swish, True, (tf.float32,)),
  )
  def test_pointwise_math(self, layer_class, tf_op, test_tflite, dtypes):
    batch_size, time, channels = 2, 10, 4
    for dtype in dtypes:
      x = self.random_sequence(batch_size, time, channels, dtype=dtype)
      l = layer_class()
      self.assertEqual(l.block_size, 1)
      self.assertEqual(l.output_ratio, 1)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), tf.TensorShape([channels])
      )
      self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      if test_tflite:
        self.verify_tflite_step(l, x)

      y = l.layer(x, training=False)
      self.assertAllEqual(y.values, x.apply_values(tf_op).mask_invalid().values)


class CastTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)), tf.float64),
      (tf.TensorShape((2, 3, 5, 9)), tf.int32),
  )
  def test_cast(self, shape, target_dtype):
    x = self.random_sequence(*shape, dtype=tf.float32)
    l = sl.Cast(target_dtype)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    # NOTE(b/155014555): Don't cast NaNs to int since it triggers ubsan.
    self.verify_contract(
        l,
        x,
        training=False,
        test_gradients=target_dtype.is_floating,
        pad_nan=target_dtype != tf.int32,
    )
    self.assertEmpty(l.variables)
    y = l.layer(x, training=False)
    self.assertEqual(y.values.dtype, target_dtype)
    if target_dtype in (tf.float32, tf.int32):
      self.verify_tflite_step(l, x)


class SliceTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters((tf.TensorShape((2, 3, 5)),),
                            (tf.TensorShape((2, 3, 5, 9)),)
                            )  # pyformat: disable
  def test_slice(self, shape):
    x = self.random_sequence(*shape)
    l = sl.Slice((slice(None, -1, None),) * (len(shape) - 2))
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        tf.TensorShape([dim - 1 for dim in shape[2:]]),
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x)

  def test_slice_newxis(self):
    x = self.random_sequence(1, 2, 3, 4)
    l = sl.Slice()[:, tf.newaxis, :]
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([3, 1, 4])
    )
    y = l.layer(x, training=True)
    self.assertEqual(y.values.shape, tf.TensorShape([1, 2, 3, 1, 4]))
    self.assertAllEqual(y.values, x.values[:, :, :, tf.newaxis, :])

    l = sl.Slice()[tf.newaxis, 0, tf.newaxis, 1:3]
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape([1, 1, 2])
    )
    y = l.layer(x, training=True)
    self.assertEqual(y.values.shape, tf.TensorShape([1, 2, 1, 1, 2]))
    self.assertAllEqual(
        y.values, x.values[:, :, tf.newaxis, 0, tf.newaxis, 1:3]
    )

    l = sl.Slice()[tf.newaxis, 0, 0, tf.newaxis]
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([1, 1]))
    y = l.layer(x, training=True)
    self.assertEqual(y.values.shape, tf.TensorShape([1, 2, 1, 1]))
    self.assertAllEqual(y.values, x.values[:, :, tf.newaxis, 0, 0, tf.newaxis])

  def test_slice_wrongsize(self):
    batch_size, time, channels = 2, 10, 3
    x = self.random_sequence(batch_size, time, channels)
    l = sl.Slice((slice(None, None, None), slice(None, None, None)))
    with self.assertRaises(ValueError):
      l.layer(x, training=False)

    l = sl.Slice((slice(None, None, None), None, slice(None, None, None)))
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class FlattenTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),),
      (tf.TensorShape((2, 3, 5, 9, 2)),))  # pyformat: disable
  def test_flatten(self, shape):
    x = self.random_sequence(*shape)
    l = sl.Flatten()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:].num_elements()
    )
    self.verify_contract(l, x, training=False)
    self.verify_tflite_step(l, x)

    self.assertEmpty(l.variables)
    y = l.layer(x, training=False)
    x, y = self.evaluate([x, y])
    self.assertAllEqual(
        y.values, np.reshape(x.values, shape[:2] + [shape[2:].num_elements()])
    )


class TransposeTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      ((2, 3, 4, 8), (0, 1), (0, 1, 2, 3), (4, 8)),
      ((2, 3, 5, 9), (1, 0), (0, 1, 3, 2), (9, 5)),
      ((7, 6, 5, 9, 3), (1, 0, 2), (0, 1, 3, 2, 4), (9, 5, 3)),
      ((7, 6, 5, 6, 3), (2, 1, 0), (0, 1, 4, 3, 2), (3, 6, 5)),
      ((8, 3, 4, 2, 5, 7), (1, 2, 0, 3), (0, 1, 3, 4, 2, 5), (2, 5, 4, 7)),
  )
  def test_transpose(self, shape, perm, overall_perm, expected_output_shape):
    x = self.random_sequence(*shape)
    l = sl.Transpose(perm=perm)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        expected_output_shape,
    )
    self.verify_contract(l, x, training=False)
    self.verify_tflite_step(l, x)

    self.assertEmpty(l.variables)
    y = l.layer(x, training=False)
    x, y = self.evaluate([x, y])
    self.assertAllEqual(y.values, np.transpose(x.values, axes=overall_perm))

  def test_transpose_negative_perm(self):
    perm = [-1, 0]
    with self.assertRaises(ValueError):
      sl.Transpose(perm=perm)

  @parameterized.parameters(
      ((2, 3, 4, 8), (2, 1)),
      ((2, 3, 4, 8), (0, 1, 2)),
      ((2, 3, 4, 8), (0, 1, 0)),
  )
  def test_transpose_incompatible_perm(self, shape, perm):
    x = self.random_sequence(*shape)
    l = sl.Transpose(perm=perm)
    # TF1 throws ValueError and TF2 throws tf.errors.InvalidArgumentError.
    with self.assertRaises(Exception):
      l.layer(x, training=False)


class ReshapeTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      ((2, 3, 4, 8), [2, 16], (2, 16)),
      ((2, 3, 4, 8), [8, -1], (8, 4)),
      ((3, 4, 5, 9), [5, 3, 3], (5, 3, 3)),
      ((3, 4, 5, 9), [3, 5, -1, 1], (3, 5, 3, 1)),
  )
  def test_reshape(self, shape, target_channel_shape, expected_output_shape):
    x = self.random_sequence(*shape)
    l = sl.Reshape(shape=target_channel_shape)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        expected_output_shape,
    )
    self.verify_contract(l, x, training=False)
    self.verify_tflite_step(l, x)

    self.assertEmpty(l.variables)
    y = l.layer(x, training=False)
    x, y = self.evaluate([x, y])
    self.assertAllEqual(
        y.values,
        np.reshape(x.values, newshape=shape[:2] + expected_output_shape),
    )

  @parameterized.parameters(
      ([4, -2],),
      ([-1, -1],),
      ([4, -1, 0],),
  )
  def test_reshape_invalid_shape(self, target_channel_shape):
    with self.assertRaises(ValueError):
      sl.Reshape(shape=target_channel_shape)

  @parameterized.parameters(
      ((2, 3, 4), [2, 1]),
      ((2, 3, 4, 8), [4, 5, -1]),
      ((2, 3, 6, 7), [5]),
  )
  def test_reshape_incompatible_shape(self, shape, target_channel_shape):
    x = self.random_sequence(*shape)
    l = sl.Reshape(shape=target_channel_shape)
    # TF1 throws ValueError and TF2 throws tf.errors.InvalidArgumentError.
    with self.assertRaises(Exception):
      l.layer(x, training=False)


class EmitTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_emit(self):
    l = sl.Serial([sl.Translate(5), sl.Emit(name='emit')])

    x = self.random_sequence(2, 3, 5)
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertAllEqual(y.values, emits['emit'].values)
    self.assertAllEqual(y.mask, emits['emit'].mask)


class OneHotTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((1, 2, 3)),),
      (tf.TensorShape((2, 3, 5, 9)),),
      (tf.TensorShape((2, 3, 5, 9, 2)),))  # pyformat: disable
  def test_one_hot(self, shape):
    depth = 4
    x = self.random_sequence(*shape, dtype=tf.int32, low=0, high=depth - 1)
    l = sl.OneHot(depth)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:].concatenate(depth)
    )
    self.verify_contract(
        l,
        x,
        training=False,
        pad_nan=False,
        # Integer tensors have no gradient to test.
        test_gradients=False,
    )
    self.verify_tflite_step(l, x, use_flex=True)
    self.assertEmpty(l.variables)
    y = l.layer(x, training=False)
    x, y = self.evaluate([x, y])
    self.assertAllEqual(y.values, (np.eye(depth)[x.values].T * x.mask.T).T)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_onehot_precision_policy(self, precision_policy):
    depth = 10
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    with test_util.keras_precision_policy_scope(precision_policy):
      x = self.random_sequence(2, 3, 5, low=0, high=depth - 1, dtype=tf.int32)
      with tf.name_scope('test'):
        l = sl.OneHot(depth)
      _, y_np = self.verify_contract(
          l,
          x,
          training=True,
          pad_nan=False,
          # Integer tensors have no gradient to test.
          test_gradients=False,
      )
      self.assertEqual(y_np.dtype, utils.compute_dtype())
      for variable in l.variables:
        self.assertEqual(variable.dtype, utils.variable_dtype())


class EmbeddingTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((1, 2, 3)),),
      (tf.TensorShape((2, 3, 5, 9)),),
      (tf.TensorShape((2, 3, 5, 9, 2)),))  # pyformat: disable
  def test_embedding(self, shape):
    dimension, num_embeddings = 8, 5
    x = self.random_sequence(
        *shape, dtype=tf.int32, low=0, high=num_embeddings - 1
    )
    with tf.name_scope('test'):
      l = sl.Embedding(dimension=dimension, num_embeddings=num_embeddings)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:].concatenate(dimension)
    )
    self.verify_contract(
        l,
        x,
        training=False,
        test_causality=False,
        # Integer tensors have no gradient to test.
        test_gradients=False,
    )
    self.assertLen(l.variables, 1)
    self.assertLen(l.trainable_variables, 1)
    self.assertCountEqual(
        [v.name for v in l.variables],
        [
            'test/embedding/embedding/embeddings:0',
        ],
    )
    self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_embedding_precision_policy(self, precision_policy):
    num_embeddings = 10
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    with test_util.keras_precision_policy_scope(precision_policy):
      x = self.random_sequence(
          2, 3, 5, low=0, high=num_embeddings - 1, dtype=tf.int32
      )
      with tf.name_scope('test'):
        l = sl.Embedding(dimension=5, num_embeddings=num_embeddings)
      _, y_np = self.verify_contract(
          l,
          x,
          training=True,
          test_causality=False,
          # Integer tensors have no gradient to test.
          test_gradients=False,
      )
      self.assertEqual(y_np.dtype, utils.compute_dtype())
      for variable in l.variables:
        self.assertEqual(variable.dtype, utils.variable_dtype())


class StyleTokenTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),))  # pyformat: disable
  def test_style_token(self, shape):
    num_style_tokens, num_heads, units_per_head = 3, 5, 7
    x = self.random_sequence(*shape)
    with tf.name_scope('test'):
      l = sl.StyleToken(num_style_tokens, num_heads, units_per_head)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        tf.TensorShape([num_heads, units_per_head]),
    )
    self.verify_contract(l, x, training=False)
    # 6 variables:
    # - The style tokens.
    # - Style token keys.
    # - A projection matrix and bias for the query.
    # - A logits projection matrix and bias.
    self.assertLen(l.variables, 6)
    self.assertLen(l.trainable_variables, 6)
    self.assertCountEqual(
        [v.name for v in l.variables],
        [
            'test/style_token/style_token_keys:0',
            'test/style_token/style_tokens:0',
            'test/style_token/query_projection/kernel:0',
            'test/style_token/query_projection/bias:0',
            'test/style_token/to_logits/kernel:0',
            'test/style_token/to_logits/bias:0',
        ],
    )
    self.verify_tflite_step(l, x, use_flex=True)


class IdentityTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)),),
      (tf.TensorShape((2, 3, 5, 9)),))  # pyformat: disable
  def test_identity(self, shape):
    x = self.random_sequence(*shape)
    l = sl.Identity()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.verify_tflite_step(l, x)


class ReverseSequenceTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_reverse_sequence(self):
    x = sl.Sequence(
        tf.convert_to_tensor([[1, 2, 3, 0, 0], [5, 2, 8, 9, 0]]),
        tf.convert_to_tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]),
    )
    reverse_x = sl.Sequence(
        tf.convert_to_tensor([[3, 2, 1, 0, 0], [9, 8, 2, 5, 0]]),
        tf.convert_to_tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]),
    )
    l = sl.ReverseSequence()
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), [])
    self.assertEmpty(l.variables)
    self.assertSequencesClose(l.layer(x, training=True), reverse_x)
    # Skip verifying contract since this layer doesn't support step().
    with self.assertRaises(NotImplementedError):
      l.step(x, (), True)
    with self.assertRaises(NotImplementedError):
      _ = l.block_size


class Upsample1DTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)), 2),
      (tf.TensorShape((2, 3, 5, 9)), 3))  # pyformat: disable
  def test_upsample1d(self, shape, rate):
    l = sl.Upsample1D(rate)
    x = self.random_sequence(*shape)
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y = l.layer(x, training=False)
    for i in range(rate):
      self.assertAllEqual(x.values, y.values[:, i::rate])


class SnakeTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)), False),
      (tf.TensorShape((2, 3, 5)), True),
      (tf.TensorShape((2, 3, 5, 9)), False),
      (tf.TensorShape((2, 3, 5, 9)), True))  # pyformat: disable
  def test_snake(self, shape: tf.TensorShape, separate_beta: bool):
    l = sl.Snake(separate_beta=separate_beta)
    x = self.random_sequence(*shape)
    self.verify_contract(l, x, training=False)
    self.assertLen(l.trainable_variables, 2 if separate_beta else 1)

  def test_assert_channel_spec(self):
    x = self.random_sequence(2, 3, 1, dtype=tf.int32)

    # Does not raise.
    sl.AssertChannelSpec(tf.TensorShape([1]), tf.int32).layer(x, training=False)

    with self.assertRaisesRegex(
        ValueError, 'does not have self._expected_channel_shape='
    ):
      sl.AssertChannelSpec(tf.TensorShape([2]), tf.int32).layer(
          x, training=False
      )
    with self.assertRaisesRegex(
        ValueError, 'does not have self._expected_dtype'
    ):
      sl.AssertChannelSpec(tf.TensorShape([1]), tf.float32).layer(
          x, training=False
      )


if __name__ == '__main__':
  tf.test.main()
