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
"""Tests for 1D convolution layers."""

import fractions

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import combinators
from sequence_layers.jax import convolution
from sequence_layers.jax import dsp
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils
import tensorflow as tf

from google3.testing.pybase import parameterized


class LatencyTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      kernel_size=[1, 2, 3, 4, 7],
      stride=[1, 2, 3, 4],
      padding=['causal', 'semicausal', 'reverse_causal'],
  )
  def test_serial_downsample_delay(
      self, kernel_size: int, stride: int, padding
  ):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config([
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            name='conv1',
        ),
        dsp.Delay.Config(1, delay_layer_output=False),
    ]).make()

    x = test_utils.random_sequence(1, 64, 1, random_lengths=False)

    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(
        l,
        x,
        training=False,
        rtol=1e-6,
        atol=1e-6,
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )

  @parameterized.product(
      kernel_size=[1, 2, 3, 4, 7],
      stride=[1, 2, 3, 4],
      padding=['causal', 'semicausal'],
  )
  def test_serial_latency_causal(self, kernel_size: int, stride: int, padding):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config([
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            name='conv1',
        ),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            name='conv2',
        ),
    ]).make()

    x = test_utils.random_sequence(1, 64, 1, random_lengths=False)

    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.output_latency, 0)

    self.verify_contract(
        l,
        x,
        training=False,
        rtol=1e-6,
        atol=1e-6,
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )

  @parameterized.product(
      kernel_size=[1, 2, 3, 4, 7],
      stride=[1, 2, 3, 4],
  )
  def test_serial_latency_reverse_causal(
      self,
      kernel_size: int,
      stride: int,
  ):
    key = jax.random.PRNGKey(1234)

    input_latency = kernel_size - 1
    output_latency = input_latency // stride
    delay_amount = -output_latency % stride

    l = combinators.Serial.Config([
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv1',
        ),
        dsp.Delay.Config(delay_amount, delay_layer_output=False),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv2',
        ),
    ]).make()

    x = test_utils.random_sequence(1, 32, 1, random_lengths=False)
    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(
        l,
        x,
        training=False,
        rtol=5e-8,
        atol=5e-7,
        grad_rtol=1e-5,
        grad_atol=1e-5,
    )

  @parameterized.product(
      kernel_size=[1, 2, 3, 4, 7],
      stride=[1, 2, 3, 4],
      delays=[
          (0, 0),
          (1, 0),
          (0, 1),
          (2, 3),
          (3, 2),
      ],
  )
  def test_serial_latency_upsample_downsample(
      self,
      kernel_size: int,
      stride: int,
      delays: tuple[int, int],
  ):
    key = jax.random.PRNGKey(1234)

    pre_delay, post_delay = delays
    upsample_input_latency = 0
    upsample_output_ratio = fractions.Fraction(stride, 1)
    upsample_output_latency = upsample_input_latency * upsample_output_ratio
    conv_output_ratio = fractions.Fraction(1, stride)

    middle_delay = int(
        -int((pre_delay * upsample_output_ratio + upsample_output_latency))
        % (1 / conv_output_ratio)
    )
    l = combinators.Serial.Config([
        # Input/output latency is pre_delay.
        dsp.Delay.Config(pre_delay, delay_layer_output=False),
        # Input/output latency is 0.
        convolution.Conv1DTranspose.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='causal',
            use_bias=False,
            precision='highest',
            name='conv1',
        ),
        # Input/output latency is middle_delay.
        dsp.Delay.Config(middle_delay, delay_layer_output=False),
        # Input latency is effective_kernel_size - 1.
        # Output latency is input_latency // stride.
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv2',
        ),
        # Input/output latency is post_delay.
        dsp.Delay.Config(post_delay, delay_layer_output=False),
    ]).make()

    x = test_utils.random_sequence(1, 32, 1, random_lengths=False)
    l = self.init_and_bind_layer(key, l, x)

    self.verify_contract(
        l,
        x,
        training=False,
        rtol=5e-8,
        atol=5e-7,
        grad_rtol=1e-5,
        grad_atol=1e-5,
    )

  @parameterized.product(
      kernel_size=[1, 2, 3, 4, 7],
      stride=[1, 2, 3, 4],
      delays=[
          (0, 0),
          (1, 0),
          (0, 1),
          (2, 3),
          (3, 2),
      ],
  )
  def test_serial_latency_downsample_upsample(
      self, kernel_size: int, stride: int, delays: tuple[int, int]
  ):
    key = jax.random.PRNGKey(1234)

    pre_delay, post_delay = delays
    conv_output_ratio = fractions.Fraction(1, stride)

    # The additional delay on top of pre_delay required to achieve layer/step
    # equivalence.
    extra_pre_delay = int(-int(pre_delay) % (1 / conv_output_ratio))
    middle_delay = 5

    l = combinators.Serial.Config([
        # Input/output latency is pre_delay.
        dsp.Delay.Config(pre_delay, delay_layer_output=False),
        dsp.Delay.Config(extra_pre_delay, delay_layer_output=False),
        # Input latency is effective_kernel_size - 1.
        # Output latency is input_latency // stride.
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv1',
        ),
        # Input/output latency is middle_delay.
        dsp.Delay.Config(middle_delay, delay_layer_output=False),
        # Input/output latency is 0.
        convolution.Conv1DTranspose.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='causal',
            use_bias=False,
            precision='highest',
            name='conv2',
        ),
        # Input/output latency is post_delay.
        dsp.Delay.Config(post_delay, delay_layer_output=False),
    ]).make()

    x = test_utils.random_sequence(1, 32, 1, random_lengths=False)
    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(
        l,
        x,
        training=False,
    )

  @parameterized.product(
      kernel_size=[1, 2, 3, 4, 5],
      stride=[1, 2, 3, 4],
  )
  def test_serial_latency_mixed(
      self,
      kernel_size: int,
      stride: int,
  ):
    key = jax.random.PRNGKey(1234)

    input_latency = kernel_size - 1
    output_latency = input_latency // stride
    delay_amount = -output_latency % stride

    l = combinators.Serial.Config([
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv1',
        ),
        dsp.Delay.Config(delay_amount, delay_layer_output=False),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv2',
        ),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding='causal',
            use_bias=False,
            precision='highest',
            name='conv3',
        ),
    ]).make()

    x = test_utils.random_sequence(1, 32, 1, random_lengths=False)
    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(
        l,
        x,
        training=False,
        rtol=5e-8,
        atol=5e-7,
        grad_rtol=1e-5,
        grad_atol=1e-5,
    )

  @parameterized.product(take_every_n=[2, 3, 4])
  def test_serial_usm(self, take_every_n: int):
    key = jax.random.PRNGKey(1234)

    frame_kernel_size = 513
    frame_stride = 160
    frame_input_latency = frame_kernel_size - 1
    frame_output_latency = frame_input_latency // frame_stride

    conv_kernel_size = 3
    conv_stride = 2

    conv_dilation_rate = 1
    conv1_input_latency = (
        utils.convolution_effective_kernel_size(
            conv_kernel_size, conv_dilation_rate
        )
        - 1
    )
    conv2_input_latency = conv1_input_latency
    conv1_output_latency = conv1_input_latency // conv_stride
    conv2_output_latency = conv2_input_latency // conv_stride
    conv3_output_latency = 0

    pre_conv1_accumulated_latency = frame_output_latency
    pre_conv1_delay = -pre_conv1_accumulated_latency % conv_stride

    assert (pre_conv1_accumulated_latency + pre_conv1_delay) % conv_stride == 0
    pre_conv2_accumulated_latency = (
        pre_conv1_accumulated_latency + pre_conv1_delay
    ) // conv_stride + conv1_output_latency
    pre_conv2_delay = -pre_conv2_accumulated_latency % conv_stride

    assert (pre_conv2_accumulated_latency + pre_conv2_delay) % conv_stride == 0
    pre_conv3_accumulated_latency = (
        pre_conv2_accumulated_latency + pre_conv2_delay
    ) // conv_stride + conv2_output_latency
    pre_conv3_delay = -pre_conv3_accumulated_latency % 1

    assert (pre_conv3_accumulated_latency + pre_conv3_delay) % 1 == 0
    pre_conv4_accumulated_latency = (
        pre_conv3_accumulated_latency + pre_conv3_delay
    ) // 1 + conv3_output_latency
    pre_conv4_delay = -pre_conv4_accumulated_latency % take_every_n

    l = combinators.Serial.Config([
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=frame_kernel_size,
            strides=frame_stride,
            dilation_rate=conv_dilation_rate,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='frame',
        ),
        dsp.Delay.Config(pre_conv1_delay, delay_layer_output=False),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=conv_kernel_size,
            strides=conv_stride,
            dilation_rate=conv_dilation_rate,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv1',
        ),
        dsp.Delay.Config(pre_conv2_delay, delay_layer_output=False),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=conv_kernel_size,
            strides=conv_stride,
            dilation_rate=conv_dilation_rate,
            padding='reverse_causal',
            use_bias=False,
            precision='highest',
            name='conv2',
        ),
        dsp.Delay.Config(pre_conv3_delay, delay_layer_output=False),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=conv_kernel_size,
            strides=1,
            dilation_rate=conv_dilation_rate,
            padding='causal',
            use_bias=False,
            precision='highest',
            name='conv3',
        ),
        # A reducer-like convolution.
        dsp.Delay.Config(pre_conv4_delay, delay_layer_output=False),
        convolution.Conv1D.Config(
            filters=1,
            kernel_size=1,
            strides=take_every_n,
            dilation_rate=1,
            use_bias=False,
            precision='highest',
            padding='causal',
            name='conv4',
        ),
    ]).make()

    x = test_utils.random_sequence(1, 16000, 1, random_lengths=False)

    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(
        l,
        x,
        training=False,
        rtol=5e-8,
        atol=5e-7,
        grad_rtol=1e-5,
        grad_atol=1e-5,
    )


class Conv1DTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      params=[
          # 1x1 conv.
          (1, 1, 1),
          # even kernel_size with smaller, equal and larger strides.
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          # odd kernel_size with smaller, equal and larger strides.
          (3, 2, 1),
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
      ],
      padding=[
          'same',
          'valid',
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
  )
  def test_conv1d(self, params, padding):
    key = jax.random.PRNGKey(1234)
    kernel_size, stride, dilation_rate = params
    # Initialize the bias randomly so that we can tell it is being applied.
    l = convolution.Conv1D.Config(
        filters=2,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        bias_init=nn.initializers.normal(),
        name='conv1d',
    ).make()
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)
    self.assertEqual(l.name, 'conv1d')
    self.assertEqual(
        l.supports_step,
        padding
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
        if padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(
        l.input_latency,
        expected_input_latency,
    )
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size, channels = 2, 3

    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)
    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros((kernel_size, channels, l.config.filters)),
                'bias': jnp.zeros((l.config.filters)),
            }
        },
    )

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.dtype, jnp.float32)
    self.assertEqual(output_spec.shape, (l.config.filters,))

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = test_utils.random_sequence(batch_size, time, channels)
      y = self.verify_contract(l, x, training=False)

      # Correctness test: Compare the layer-wise output to manually executing a
      # 1D convolution with the same weights and explicit padding of the input.
      kernel = variables['params']['kernel']
      bias = variables['params']['bias']

      explicit_padding = utils.convolution_explicit_padding(
          padding, kernel_size, stride, dilation_rate
      )

      values_golden = jax.lax.conv_general_dilated(
          x.values,
          kernel,
          window_strides=[stride],
          rhs_dilation=[dilation_rate],
          padding=[explicit_padding],
          dimension_numbers=('NHC', 'HIO', 'NHC'),
      )
      values_golden += jnp.reshape(
          bias, (1,) * (values_golden.ndim - 1) + (-1,)
      )
      mask_golden = convolution.compute_conv_mask(
          x.mask,
          kernel_size,
          stride,
          dilation_rate,
          padding,
          is_step=False,
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, np.newaxis]

      chex.assert_trees_all_close(y.values, values_golden, atol=1e-5)
      chex.assert_trees_all_equal(y.mask, mask_golden)

  @parameterized.product(
      input_dtype=[jnp.float32, jnp.bfloat16],
      param_dtype=[jnp.float32],
      compute_dtype=[None, jnp.float32, jnp.bfloat16],
      use_weight_norm=[False, True],
  )
  def test_conv1d_dtypes(
      self, input_dtype, param_dtype, compute_dtype, use_weight_norm
  ):
    key = jax.random.PRNGKey(1234)
    kernel_size, stride, dilation_rate = 3, 2, 1
    padding = 'same'
    l = convolution.Conv1D.Config(
        filters=2,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        use_weight_norm=use_weight_norm,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        name='conv1d',
    ).make()

    batch_size, time, channels = 2, 20, 3
    x = test_utils.random_sequence(
        batch_size, time, channels, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)
    variables = flax.core.meta.unbox(l.variables)
    expected_variables = {
        'params': {
            'kernel': jnp.zeros(
                (kernel_size, channels, l.config.filters), param_dtype
            ),
            'bias': jnp.zeros((l.config.filters), param_dtype),
        }
    }
    if use_weight_norm:
      expected_variables['params']['scale'] = jnp.zeros((2), param_dtype)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        expected_variables,
    )
    self.verify_contract(l, x, training=False)

  @parameterized.product(
      params=[
          # 1x1 conv.
          (1, 1, 1),
          # even kernel_size with smaller, equal and larger strides.
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          # odd kernel_size with smaller, equal and larger strides.
          (3, 2, 1),
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
      ],
      padding=[
          'same',
          'valid',
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
  )
  def test_depthwise_conv1d(self, params, padding):
    key = jax.random.PRNGKey(1234)
    kernel_size, stride, dilation_rate = params
    # Initialize the bias randomly so that we can tell it is being applied.
    l = convolution.DepthwiseConv1D.Config(
        kernel_size=kernel_size,
        strides=stride,
        depth_multiplier=2,
        dilation_rate=dilation_rate,
        bias_init=nn.initializers.normal(),
        padding=padding,
        name='depthwise_conv1d',
    ).make()
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)
    self.assertEqual(l.name, 'depthwise_conv1d')
    self.assertEqual(
        l.supports_step,
        padding
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
        if padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(
        l.input_latency,
        expected_input_latency,
    )
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size, channels = 2, 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)
    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros(
                    (kernel_size, 1, channels * l.config.depth_multiplier)
                ),
                'bias': jnp.zeros((channels * l.config.depth_multiplier)),
            }
        },
    )

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), (6,))
      y = self.verify_contract(
          l,
          x,
          training=False,
          grad_rtol=1e-5,
          grad_atol=1e-5,
      )

      # Correctness test: Compare the layer-wise output to manually executing a
      # depthwise 1D convolution with the same weights and explicit padding of
      # the input.
      kernel = variables['params']['kernel']
      bias = variables['params']['bias']

      explicit_padding = utils.convolution_explicit_padding(
          padding, kernel_size, stride, dilation_rate
      )

      values_golden = jax.lax.conv_general_dilated(
          x.values,
          kernel,
          window_strides=[stride],
          rhs_dilation=[dilation_rate],
          padding=[explicit_padding],
          dimension_numbers=('NHC', 'HIO', 'NHC'),
          feature_group_count=channels,
      )
      values_golden += jnp.reshape(
          bias, (1,) * (values_golden.ndim - 1) + (-1,)
      )
      mask_golden = convolution.compute_conv_mask(
          x.mask, kernel_size, stride, dilation_rate, padding, is_step=False
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, np.newaxis]

      chex.assert_trees_all_close(y.values, values_golden)
      chex.assert_trees_all_equal(y.mask, mask_golden)

  @parameterized.product(
      input_dtype=[jnp.float32, jnp.bfloat16],
      param_dtype=[jnp.float32],
      compute_dtype=[None, jnp.float32, jnp.bfloat16],
      use_weight_norm=[False, True],
  )
  def test_depthwise_conv1d_dtypes(
      self, input_dtype, param_dtype, compute_dtype, use_weight_norm
  ):
    key = jax.random.PRNGKey(1234)
    kernel_size, stride, dilation_rate = 3, 2, 1
    padding = 'same'
    l = convolution.DepthwiseConv1D.Config(
        kernel_size=kernel_size,
        strides=stride,
        depth_multiplier=2,
        dilation_rate=dilation_rate,
        padding=padding,
        use_weight_norm=use_weight_norm,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        name='depthwise_conv1d',
    ).make()

    batch_size, time, channels = 2, 20, 3
    x = test_utils.random_sequence(
        batch_size, time, channels, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)
    variables = flax.core.meta.unbox(l.variables)
    expected_variables = {
        'params': {
            'kernel': jnp.zeros(
                (kernel_size, 1, channels * l.config.depth_multiplier),
                param_dtype,
            ),
            'bias': jnp.zeros(
                (channels * l.config.depth_multiplier), param_dtype
            ),
        }
    }
    if use_weight_norm:
      expected_variables['params']['scale'] = jnp.zeros((6), param_dtype)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        expected_variables,
    )
    self.verify_contract(l, x, training=False, grad_rtol=1e-5, grad_atol=1e-5)

  def test_tf_equivalence(self):
    key = jax.random.PRNGKey(1234)
    param_dtype = jnp.float32
    filters, kernel_size, stride, dilation_rate, padding = 2, 3, 1, 1, 'same'
    l = convolution.Conv1D.Config(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        use_bias=False,
        strides=stride,
        padding=padding,
        use_weight_norm=False,
    ).make()
    batch_size, time, channels = 2, 20, 5
    x = test_utils.random_sequence(
        batch_size,
        time,
        channels,
    )
    l = self.init_and_bind_layer(key, l, x)

    variables = flax.core.meta.unbox(l.variables)
    expected_variables = {
        'params': {
            'kernel': jnp.zeros((kernel_size, channels, filters), param_dtype),
        }
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        expected_variables,
    )

    y = l.layer(x, training=False).mask_invalid()

    l_tf = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=False,
    )

    x_tf = tf.convert_to_tensor(x.values)
    l_tf.build(x_tf.shape)
    l_tf.kernel.assign(variables['params']['kernel'])
    y_tf = types.Sequence(l_tf(x_tf).numpy(), y.mask).mask_invalid()

    self.assertSequencesClose(y, y_tf)

    for i in range(5):
      x = test_utils.random_sequence(
          batch_size,
          time + i,
          channels,
      )
      y = l.layer(x, training=False).mask_invalid()
      y_tf = types.Sequence(
          l_tf(tf.convert_to_tensor(x.values)).numpy(), y.mask
      ).mask_invalid()
      self.assertSequencesClose(y, y_tf)


class Conv1DTransposeTest(test_utils.SequenceLayerTest):

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
      padding=['causal', 'valid', 'same'],
  )
  def test_conv1d_transpose(self, params, padding):
    key = jax.random.PRNGKey(1234)
    kernel_size, stride, dilation_rate = params
    l = convolution.Conv1DTranspose.Config(
        filters=2,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        strides=stride,
        padding=padding,
        name='transpose_conv1d',
    ).make()

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, stride)
    self.assertEqual(l.name, 'transpose_conv1d')
    self.assertEqual(l.supports_step, padding == 'causal')

    batch_size, channels = 2, 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros(
                    (kernel_size, channels, 2),
                ),
                'bias': jnp.zeros((2,)),
            }
        },
    )

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(batch_size, time, channels)
        self.assertEqual(l.get_output_shape_for_sequence(x), (2,))
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

        explicit_padding = convolution._transpose_conv_explicit_padding(
            kernel_size, stride, dilation_rate, padding
        )

        values_golden = jax.lax.conv_transpose(
            x.values,
            kernel,
            strides=(stride,),
            padding=(explicit_padding,),
            rhs_dilation=(dilation_rate,),
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        )
        values_golden += jnp.reshape(
            bias, (1,) * (values_golden.ndim - 1) + (-1,)
        )
        mask_golden = convolution.compute_conv_transpose_mask(
            x.mask, kernel_size, stride, dilation_rate, padding
        )

        # Apply masking.
        values_golden = values_golden * mask_golden[:, :, np.newaxis]

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
  def test_conv1d_transpose_dtypes(
      self, input_dtype, param_dtype, compute_dtype, use_weight_norm
  ):
    key = jax.random.PRNGKey(1234)
    kernel_size, stride, dilation_rate = 3, 2, 1
    padding = 'same'
    l = convolution.Conv1DTranspose.Config(
        filters=2,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        strides=stride,
        padding=padding,
        use_weight_norm=use_weight_norm,
        compute_dtype=compute_dtype,
        name='transpose_conv1d',
    ).make()

    batch_size, time, channels = 2, 20, 3
    x = test_utils.random_sequence(
        batch_size, time, channels, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)

    variables = flax.core.meta.unbox(l.variables)
    expected_variables = {
        'params': {
            'kernel': jnp.zeros((kernel_size, channels, 2), param_dtype),
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

  def test_conv1d_transpose_groups_invalid(self):
    key = jax.random.PRNGKey(1234)
    l = convolution.Conv1DTranspose.Config(
        filters=2,
        kernel_size=3,
        groups=2,
        name='transpose_conv1d',
    ).make()

    # 3 channels is not divisible into 2 groups.
    batch_size, time, channels = 2, 20, 3
    x = test_utils.random_sequence(
        batch_size, time, channels, dtype=jnp.float32
    )
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  def test_tf_equivalence(self):
    key = jax.random.PRNGKey(1234)
    param_dtype = jnp.float32
    filters, kernel_size, stride, dilation_rate, padding = 2, 3, 2, 1, 'same'
    l = convolution.Conv1DTranspose.Config(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        use_bias=False,
        strides=stride,
        padding=padding,
        use_weight_norm=False,
    ).make()
    batch_size, time, channels = 2, 20, 5
    x = test_utils.random_sequence(
        batch_size,
        time,
        channels,
    )
    l = self.init_and_bind_layer(key, l, x)

    variables = flax.core.meta.unbox(l.variables)
    expected_variables = {
        'params': {
            'kernel': jnp.zeros((kernel_size, channels, filters), param_dtype),
        }
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        expected_variables,
    )

    y = l.layer(x, training=False).mask_invalid()

    l_tf = tf.keras.layers.Conv1DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=False,
    )

    x_tf = tf.convert_to_tensor(x.values)
    l_tf.build(x_tf.shape)
    l_tf.kernel.assign(
        jnp.flip(
            jnp.transpose(variables['params']['kernel'], [0, 2, 1]), axis=0
        )
    )
    y_tf = l_tf(x_tf)

    y_tf = types.Sequence(y_tf.numpy(), y.mask).mask_invalid()

    self.assertSequencesClose(y, y_tf)


if __name__ == '__main__':
  test_utils.main()
