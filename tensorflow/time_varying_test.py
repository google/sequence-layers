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
from absl.testing import parameterized
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class SequenceEmbeddingTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (tf.TensorShape((3, 4)),),
      (tf.TensorShape((1, 2, 3)),),
      (tf.TensorShape((2, 3, 5, 9)),),
  )
  def test_sequence_embedding(self, shape):
    dimension, num_embeddings = 8, 5
    x = self.random_sequence(
        *shape, dtype=tf.int32, low=0, high=num_embeddings - 1
    )
    with tf.name_scope('test'):
      l = sl.SequenceEmbedding(
          dimension=dimension,
          num_embeddings=num_embeddings,
          num_groups=4,
      )
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
            'test/sequence_embedding/embedding/embeddings:0',
        ],
    )
    self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_sequence_embedding_precision_policy(self, precision_policy):
    num_embeddings = 10
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    x = self.random_sequence(
        2, 3, 5, low=0, high=num_embeddings - 1, dtype=tf.int32
    )
    with tf.name_scope('test'):
      l = sl.SequenceEmbedding(
          dimension=5, num_embeddings=num_embeddings, num_groups=4
      )
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
    tf.keras.mixed_precision.set_global_policy(default_policy)


class SequenceDenseTest(test_util.SequenceLayerTest):

  def test_sequence_dense(self):
    batch_size = 2
    num_steps = 3
    input_dim, output_dim = 4, 5
    x = self.random_sequence(batch_size, num_steps, input_dim)
    l = sl.SequenceDense(units=output_dim, num_steps=num_steps)
    y_layer = l.layer(x, training=False)
    state = l.get_initial_state(x)
    for t in range(num_steps):
      y_step, state = l.step(x[:, t:t + 1], state, training=False)
      self.assertSequencesClose(y_step, y_layer[:, t:t + 1])

  def test_sequence_dense_contract(self):
    batch_size = 2
    # We need to support more steps than the input time dim because
    # verify_contract() pads the input in various ways.
    input_time, num_steps = 4, 7
    input_dim, output_dim = 5, 6
    x = self.random_sequence(batch_size, input_time, input_dim)
    with tf.name_scope('test'):
      l = sl.SequenceDense(units=output_dim, num_steps=num_steps)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape(output_dim)
    )
    self.verify_contract(l, x, training=False)

    self.assertLen(l.variables, 2)
    self.assertLen(l.trainable_variables, 2)
    self.assertEqual(l.name_scope.name, 'test/sequence_dense/')
    self.assertCountEqual(
        [v.name for v in l.variables],
        ['test/sequence_dense/kernel:0', 'test/sequence_dense/bias:0'],
    )
    self.verify_tflite_step(l, x)


class MaskedDenseTest(test_util.SequenceLayerTest):

  def test_masked_dense(self):
    batch_size = 2
    num_steps = 8
    input_dim, output_dim = 4, 5
    input_shape = [batch_size, num_steps, input_dim]
    x = self.random_sequence(*input_shape, random_lengths=False)
    l = sl.MaskedDense(units=output_dim, num_steps=num_steps)
    y_layer = l.layer(x, training=False)
    state = l.get_initial_state(x)
    # Verify that step is equivalent to layer for fixed length sequences.
    for t in range(num_steps):
      print(t)
      y_step, state = l.step(x[:, t:t + 1], state, training=False)
      self.assertSequencesClose(y_step, y_layer[:, t:t + 1])

  def test_masked_dense_causality(self):
    batch_size = 2
    num_steps = 3
    input_dim, output_dim = 4, 5

    input_shape = [batch_size, num_steps, input_dim]
    output_shape = [batch_size, num_steps, output_dim]

    x = self.random_sequence(*input_shape, random_lengths=False)
    l = sl.MaskedDense(units=output_dim, num_steps=num_steps)

    with tf.GradientTape() as tape:
      tape.watch(x)
      y = l.layer(x, training=False)

    self.assertEqual(y.values.shape, output_shape)

    # Confirm that autoregressive causality is obeyed.
    # Take Jacobian of y=[By, Ty, O] wrt x=[Bx, Tx, I].
    jac = tape.jacobian(y.values, x.values)  # [By, Ty, O, Bx, Tx, I]
    self.assertEqual(jac.shape, output_shape + input_shape)

    # Reduce Jacobian across channel dim to get boolean connectedness.
    # [By, Ty, O, Bx, Tx, I] -> [By, Ty, Bx, Tx]
    connected = tf.reduce_any(tf.not_equal(jac, 0.0), axis=[2, 5])
    connected = self.evaluate(connected)
    for by in range(batch_size):
      for bx in range(batch_size):
        for ty in range(num_steps):  # y[by, ty]
          if bx == by:
            # For matching batch index, y[n] is only connected to x[<=n].
            for tx in range(ty + 1):  # x[bx=by, tx<=ty]
              self.assertTrue(connected[by, ty, bx, tx])
            for tx in range(ty + 1, num_steps):  # x[bx=by, tx>ty]
              self.assertFalse(connected[by, ty, bx, tx])
          else:
            # For non-matching batch index, nothing is connected.
            for tx in range(num_steps):  # x[bx!=by, tx]
              self.assertFalse(connected[by, ty, bx, tx])

  def test_masked_dense_contract(self):
    batch_size = 2
    # We need to support more steps than the input time dim because
    # verify_contract() pads the input in various ways.
    input_time, num_steps = 4, 7
    input_dim, output_dim = 5, 6
    x = self.random_sequence(batch_size, input_time, input_dim)
    with tf.name_scope('test'):
      l = sl.MaskedDense(units=output_dim, num_steps=num_steps)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape(output_dim)
    )
    # pad_nan=False because the nans can pass through the causality mask.
    self.verify_contract(l, x, training=False, pad_nan=False)

    self.assertLen(l.variables, 2)
    self.assertLen(l.trainable_variables, 2)
    self.assertEqual(l.name_scope.name, 'test/masked_dense/')
    self.assertCountEqual(
        [v.name for v in l.variables],
        ['test/masked_dense/kernel:0', 'test/masked_dense/bias:0'],
    )
    # The tf.roll() used in the step method isn't currently supported by TFLite.
    # self.verify_tflite_step(l, x)


if __name__ == '__main__':
  tf.test.main()
