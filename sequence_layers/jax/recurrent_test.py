# Copyright 2024 Google LLC
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
"""Tests for recurrent layers."""

import chex
import flax
import jax
import jax.numpy as jnp
from sequence_layers.jax import recurrent
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
import tensorflow.compat.v2 as tf

from google3.testing.pybase import parameterized


class LSTMTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      param_dtype=(jnp.float32, jnp.float64), random_mask=(False, True)
  )
  def test_lstm(self, param_dtype, random_mask):
    key = jax.random.PRNGKey(1234)
    l = recurrent.LSTM.Config(
        units=8, param_dtype=param_dtype, name='lstm'
    ).make()

    batch_size, channels = 2, 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'lstm')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': {'kernel': jnp.zeros((channels, 32), param_dtype)},
                'recurrent_kernel': {'kernel': jnp.zeros((8, 32), param_dtype)},
                'bias': jnp.zeros((32,), param_dtype),
            }
        },
    )

    for time in range(8, 11):
      x = test_utils.random_sequence(
          batch_size, time, channels, random_mask=random_mask
      )
      self.verify_contract(l, x, training=False)

  @parameterized.parameters(
      *test_utils.standard_dtype_configs(input=True, compute=True)
  )
  def test_dtypes(self, input_dtype, compute_dtype):
    """Check that input and compute dtypes interact correctly."""
    # jnp.bfloat16 isn't currently supported by orthogonal initializer.
    param_dtype = jnp.float32
    key = jax.random.PRNGKey(1234)
    l = recurrent.LSTM.Config(
        units=8, param_dtype=param_dtype, dtype=compute_dtype, name='lstm'
    ).make()

    batch_size, channels, time = 2, 3, 8
    x = test_utils.random_sequence(batch_size, 1, channels, dtype=input_dtype)
    l = self.init_and_bind_layer(key, l, x)

    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'lstm')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': {'kernel': jnp.zeros((channels, 32), param_dtype)},
                'recurrent_kernel': {'kernel': jnp.zeros((8, 32), param_dtype)},
                'bias': jnp.zeros((32,), param_dtype),
            }
        },
    )

    x = test_utils.random_sequence(batch_size, time, channels)
    # Output dtypes are checked in verify_contract.
    # TODO(rryan): Figure out why jit=True changes the tolerances we can
    # achieve with mixed precision.
    self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype),
        jit=False,
    )

  @parameterized.parameters(True, False)
  def test_keras_equivalence(self, random_mask):
    x = test_utils.random_sequence(8, 31, 13, random_mask=random_mask)
    l_keras = tf.keras.layers.LSTM(units=8, return_sequences=True)
    y_keras = jnp.asarray(
        l_keras(
            tf.convert_to_tensor(x.values),
            mask=tf.convert_to_tensor(x.mask),
        ).numpy()
    )
    kernel, recurrent_kernel, bias = l_keras.variables
    kernel = jnp.asarray(kernel.numpy())
    recurrent_kernel = jnp.asarray(recurrent_kernel.numpy())
    bias = jnp.asarray(bias.numpy())

    l = (
        recurrent.LSTM.Config(units=8)
        .make()
        .bind({
            'params': {
                'kernel': {'kernel': kernel},
                'recurrent_kernel': {'kernel': recurrent_kernel},
                'bias': bias,
            }
        })
    )

    y = l.layer(x, training=False).mask_invalid()
    y_keras = types.Sequence(y_keras, y.mask).mask_invalid()
    self.assertSequencesClose(y, y_keras)


if __name__ == '__main__':
  test_utils.main()
