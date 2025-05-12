# Copyright 2025 Google LLC
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

from absl.testing import parameterized
import chex
import flax
import jax
import jax.numpy as jnp
from recurrentgemma._src.jax import layers as recurrentgemma_layers
from sequence_layers.jax import recurrent
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
import tensorflow.compat.v2 as tf


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
    self.assertEqual(l.output_latency, 0)
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
        units=8,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        name='lstm',
    ).make()

    batch_size, channels, time = 2, 3, 8
    x = test_utils.random_sequence(batch_size, 1, channels, dtype=input_dtype)
    l = self.init_and_bind_layer(key, l, x)

    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.output_latency, 0)
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
    # TODO(b/398200724): Figure out why jit=True changes the tolerances we can
    # achieve with mixed precision.
    self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
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


class RGLRUTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      param_dtype=(jnp.float32, jnp.bfloat16),
      random_mask=(False,),
      scan_type=(
          'auto',
          'linear_native',
          'associative_native',
          # 'linear_pallas',  # Doesn't work on CPU.
      ),
      only_real=(True, False),
  )
  def test_rglru(self, param_dtype, random_mask, scan_type, only_real):
    key = jax.random.PRNGKey(1234)
    batch_size, time, units, num_heads = 2, 32, 8, 2
    l = recurrent.RGLRU.Config(
        units=units,
        num_heads=num_heads,
        scan_type=scan_type,
        only_real=only_real,
        param_dtype=param_dtype,
        name='rglru',
    ).make()

    x = test_utils.random_sequence(batch_size, 1, units)
    l = self.init_and_bind_layer(key, l, x)

    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.output_latency, 0)
    self.assertEqual(l.name, 'rglru')

    variables = flax.core.meta.unbox(l.variables)
    complex_units = units if only_real else units // 2
    expected_variables = {
        'params': {
            'input_gate': {
                'kernel': jnp.zeros(
                    (num_heads, units // num_heads, complex_units // num_heads),
                    param_dtype,
                ),
                'bias': jnp.zeros(
                    (num_heads, complex_units // num_heads), param_dtype
                ),
            },
            'a_gate': {
                'kernel': jnp.zeros(
                    (num_heads, units // num_heads, complex_units // num_heads),
                    param_dtype,
                ),
                'bias': jnp.zeros(
                    (num_heads, complex_units // num_heads), param_dtype
                ),
            },
            'a_param': jnp.zeros([complex_units], param_dtype),
        }
    }
    if not only_real:
      expected_variables['params']['a_imag_param'] = jnp.zeros(
          complex_units, param_dtype
      )

    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        expected_variables,
    )

    reference_layer = recurrentgemma_layers.RGLRU(
        width=units,
        num_heads=num_heads,
        scan_type=l._scan_type,
        param_dtype=param_dtype,
        only_real=only_real,
    )

    reference_params = {
        'params': {
            'a_param': variables['params']['a_param'],
            'input_gate': {
                'w': variables['params']['input_gate']['kernel'],
                'b': variables['params']['input_gate']['bias'],
            },
            'a_gate': {
                'w': variables['params']['a_gate']['kernel'],
                'b': variables['params']['a_gate']['bias'],
            },
        }
    }
    if not only_real:
      reference_params['params']['a_imag_param'] = variables['params'][
          'a_imag_param'
      ]

    reference_layer = reference_layer.bind(reference_params)
    reference_state = reference_layer.init_cache(batch_size, units)

    for time in range(time, time + 1):
      x = test_utils.random_sequence(
          batch_size, time, units, random_mask=random_mask
      )

      y = self.verify_contract(
          l,
          x,
          training=False,
          **test_utils.get_grad_tols(l, x, param_dtype, None),
      )

      # Compare to recurrentgemma reference implementation.
      # NOTE: The recurrentgemma layer implementation uses x[: 0] by default as
      #   the state, while the step implementation uses all-zeros. This is a
      #   layer/step mismatch! To avoid this, use all zeros as the initial state
      #   when running the reference layer.
      segment_pos = jnp.tile(
          jnp.arange(x.shape[1], dtype=jnp.int32)[jnp.newaxis, :],
          (batch_size, 1),
      )
      expected_values, _ = reference_layer(
          x.values, segment_pos, reference_state
      )
      y_expected = types.Sequence(expected_values, x.mask).mask_invalid()
      self.assertAllClose(y, y_expected)


if __name__ == '__main__':
  test_utils.main()
