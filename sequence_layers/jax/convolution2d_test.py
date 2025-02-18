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
"""Tests for 2D convolution layers."""

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution
from sequence_layers.jax import test_utils
from sequence_layers.jax import utils

from google3.testing.pybase import parameterized


class Conv2DTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      kernel_size_strides_dilation_rate=(
          # 1x1 conv.
          (1, 1, 1),
          # even kernel_size with smaller, equal and larger strides.
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          (3, 2, 1),
          # odd kernel_size with smaller, equal and larger strides.
          (3, 3, 1),
          (3, 4, 1),
          # kernel_size smaller, equal and larger than even dilation_rate.
          (1, 1, 2),
          (2, 1, 2),
          (3, 1, 2),
          # kernel_size smaller, equal and larger than odd dilation_rate.
          (1, 1, 3),
          (2, 1, 3),
          (3, 1, 3),
      ),
      time_padding=(
          'same',
          'valid',
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ),
      spatial_padding=(
          'same',
          'valid',
          'causal_valid',
          'reverse_causal_valid',
          'semicausal',
          (2, 3),
      ),
  )
  def test_conv2d(
      self,
      kernel_size_strides_dilation_rate,
      time_padding,
      spatial_padding,
  ):
    key = jax.random.PRNGKey(1234)
    kernel_size, strides, dilation_rate = kernel_size_strides_dilation_rate
    filters = 2
    l = convolution.Conv2D.Config(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        bias_init=nn.initializers.normal(),
        name='conv2d',
    ).make()
    self.assertEqual(l.block_size, strides)
    self.assertEqual(1 / l.output_ratio, strides)
    self.assertEqual(l.name, 'conv2d')
    self.assertEqual(
        l.supports_step,
        time_padding
        in (
            'causal_valid',
            'reverse_causal_valid',
            'causal',
            'reverse_causal',
            'semicausal',
        ),
    )

    effective_kernel_size = utils.convolution_effective_kernel_size(
        kernel_size, dilation_rate
    )
    expected_input_latency = (
        effective_kernel_size - 1
        if time_padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(
        l.input_latency,
        expected_input_latency,
    )
    self.assertEqual(int(l.output_latency), expected_input_latency // strides)

    batch_size, spatial, channels = 2, 7, 3
    x = test_utils.random_sequence(batch_size, 1, spatial, channels)
    l = self.init_and_bind_layer(key, l, x)
    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros(
                    (kernel_size, kernel_size, channels, filters)
                ),
                'bias': jnp.zeros((filters)),
            }
        },
    )

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(batch_size, time, spatial, channels)
        spatial_output = utils.convolution_padding_output_size(
            spatial, spatial_padding, kernel_size, strides, dilation_rate
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x),
            (spatial_output, filters),
        )
        y = self.verify_contract(
            l,
            x,
            training=False,
            grad_rtol=1e-5,
            grad_atol=1e-5,
        )

        # Correctness test: Compare the layer-wise output to manually executing
        # a 2D convolution with the same weights. We manually pad the
        # input to be causal in time and have same/valid behavior over space.
        kernel = variables['params']['kernel']
        bias = variables['params']['bias']

        explicit_time_padding = utils.convolution_explicit_padding(
            time_padding, kernel_size, strides, dilation_rate
        )
        explicit_spatial_padding = utils.convolution_explicit_padding(
            spatial_padding, kernel_size, strides, dilation_rate
        )

        values_golden = jax.lax.conv_general_dilated(
            x.values,
            kernel,
            window_strides=[strides, strides],
            rhs_dilation=[dilation_rate, dilation_rate],
            padding=[explicit_time_padding, explicit_spatial_padding],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        values_golden += jnp.reshape(
            bias, (1,) * (values_golden.ndim - 1) + (-1,)
        )
        mask_golden = convolution.compute_conv_mask(
            x.mask,
            kernel_size,
            strides,
            dilation_rate,
            time_padding,
            is_step=False,
        )

        # Apply masking.
        values_golden = (
            values_golden * mask_golden[:, :, np.newaxis, np.newaxis]
        )

        chex.assert_trees_all_close(y.values, values_golden)
        chex.assert_trees_all_equal(y.mask, mask_golden)

  @parameterized.product(
      input_dtype=[jnp.float32, jnp.bfloat16],
      param_dtype=[jnp.float32],
      compute_dtype=[None, jnp.float32, jnp.bfloat16],
      use_weight_norm=[False, True],
  )
  def test_conv2d_dtypes(
      self, input_dtype, param_dtype, compute_dtype, use_weight_norm
  ):
    key = jax.random.PRNGKey(1234)
    kernel_size, strides, dilation_rate = 3, 2, 1
    time_padding, spatial_padding = 'same', 'same'
    filters = 2
    l = convolution.Conv2D.Config(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        use_weight_norm=use_weight_norm,
        dtype=compute_dtype,
        param_dtype=param_dtype,
        name='conv2d',
    ).make()

    batch_size, time, spatial, channels = 2, 20, 7, 3
    x = test_utils.random_sequence(
        batch_size, time, spatial, channels, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)
    variables = flax.core.meta.unbox(l.variables)
    expected_variables = {
        'params': {
            'kernel': jnp.zeros(
                (kernel_size, kernel_size, channels, filters), param_dtype
            ),
            'bias': jnp.zeros((filters), param_dtype),
        }
    }
    if use_weight_norm:
      expected_variables['params']['scale'] = jnp.zeros((2), param_dtype)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        expected_variables,
    )
    self.verify_contract(l, x, training=False, grad_rtol=1e-5, grad_atol=1e-5)


class Conv2DTransposeTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      params=[
          # 1x1 conv.
          (1, 1, 1),
          (1, 2, 1),
          # even kernel_size with smaller, equal and larger strides.
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          # odd kernel_size with smaller, equal and larger strides.
          (3, 2, 1),
          (3, 3, 1),
          (3, 4, 1),
          # kernel_size smaller, equal and larger than even dilation_rate.
          # TODO(rryan): Support dilated transpose convolution.
          # (1, 1, 2),
          # (2, 1, 2),
          # (3, 1, 2),
          # kernel_size smaller, equal and larger than odd dilation_rate.
          # TODO(rryan): Support dilated transpose convolution.
          # (1, 1, 3),
          # (2, 1, 3),
          # (3, 1, 3),
      ],
      time_padding=['causal', 'valid', 'same'],
      spatial_padding=['causal', 'valid', 'same'],
  )
  def test_conv2d_transpose(self, params, time_padding, spatial_padding):
    key = jax.random.PRNGKey(1234)
    kernel_size, strides, dilation_rate = params
    l = convolution.Conv2DTranspose.Config(
        filters=2,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        strides=strides,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        name='transpose_conv2d',
    ).make()

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, strides)
    self.assertEqual(l.name, 'transpose_conv2d')
    self.assertEqual(l.supports_step, time_padding == 'causal')

    batch_size, spatial, channels = 2, 7, 3
    x = test_utils.random_sequence(batch_size, 1, spatial, channels)
    l = self.init_and_bind_layer(key, l, x)

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros(
                    (kernel_size, kernel_size, channels, 2),
                ),
                'bias': jnp.zeros((2,)),
            }
        },
    )

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(batch_size, time, spatial, channels)
        spatial_output = convolution._compute_conv_transpose_output_length(
            spatial, kernel_size, strides, dilation_rate, spatial_padding
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x),
            (
                spatial_output,
                2,
            ),
        )
        y = self.verify_contract(
            l,
            x,
            training=False,
            grad_rtol=1e-5,
            grad_atol=1e-5,
        )

        # Correctness test: Compare the layer-wise output to manually executing
        # a transpose 1D convolution with the same weights and explicit padding
        # of the input.
        kernel = variables['params']['kernel']
        bias = variables['params']['bias']

        explicit_time_padding = convolution._transpose_conv_explicit_padding(
            kernel_size,
            strides,
            dilation_rate,
            time_padding,
        )
        explicit_spatial_padding = convolution._transpose_conv_explicit_padding(
            kernel_size,
            strides,
            dilation_rate,
            spatial_padding,
        )

        values_golden = jax.lax.conv_transpose(
            x.values,
            kernel,
            strides=(strides, strides),
            padding=(explicit_time_padding, explicit_spatial_padding),
            rhs_dilation=(dilation_rate, dilation_rate),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        values_golden += jnp.reshape(
            bias, (1,) * (values_golden.ndim - 1) + (-1,)
        )
        mask_golden = convolution.compute_conv_transpose_mask(
            x.mask,
            kernel_size,
            strides,
            dilation_rate,
            time_padding,
        )

        # Apply masking.
        values_golden = (
            values_golden * mask_golden[:, :, np.newaxis, np.newaxis]
        )

        chex.assert_trees_all_close(
            y.values, values_golden, rtol=1e-6, atol=1e-6
        )
        chex.assert_trees_all_equal(y.mask, mask_golden)

  @parameterized.product(
      input_dtype=[jnp.float32, jnp.bfloat16],
      param_dtype=[jnp.float32],
      compute_dtype=[None, jnp.float32, jnp.bfloat16],
      use_weight_norm=[False, True],
  )
  def test_conv2d_transpose_dtypes(
      self, input_dtype, param_dtype, compute_dtype, use_weight_norm
  ):
    key = jax.random.PRNGKey(1234)
    kernel_size, strides, dilation_rate = 3, 2, 1
    time_padding, spatial_padding = 'same', 'same'
    l = convolution.Conv2DTranspose.Config(
        filters=2,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        strides=strides,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        use_weight_norm=use_weight_norm,
        dtype=compute_dtype,
        name='transpose_conv2d',
    ).make()

    batch_size, time, spatial, channels = 2, 20, 7, 3
    x = test_utils.random_sequence(
        batch_size, time, spatial, channels, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)

    variables = flax.core.meta.unbox(l.variables)
    expected_variables = {
        'params': {
            'kernel': jnp.zeros(
                (kernel_size, kernel_size, channels, 2), param_dtype
            ),
            'bias': jnp.zeros((2,), param_dtype),
        }
    }
    if use_weight_norm:
      expected_variables['params']['scale'] = jnp.zeros((2), param_dtype)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        expected_variables,
    )
    self.verify_contract(l, x, training=False, grad_rtol=1e-5, grad_atol=1e-5)


if __name__ == '__main__':
  test_utils.main()
