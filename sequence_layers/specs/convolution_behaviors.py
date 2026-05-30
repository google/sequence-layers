# Copyright 2026 Google LLC
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
"""Behavior tests for convolution layers.

Backend-specific test files should inherit from these tests.
"""

# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation

import fractions
from typing import Any

from absl.testing import parameterized

from sequence_layers.specs import test_utils
from sequence_layers.specs import types as types_spec


class Conv1DTest(test_utils.SequenceLayerTest):
  """Test behavior of Conv1D layer."""

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
    kernel_size, stride, dilation_rate = params
    config = self.sl.Conv1D.Config(
        filters=2,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        name='conv1d',
    )
    l = self.make_layer(config)
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)
    self.assertEqual(l.name, 'conv1d')

    supports_step = padding in (
        'causal_valid',
        'reverse_causal_valid',
        'causal',
        'reverse_causal',
        'semicausal',
    )
    self.assertEqual(l.supports_step, supports_step)

    effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
    expected_input_latency = (
        effective_kernel_size - 1
        if padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size, channels = 2, 3
    x = self.random_sequence(batch_size, 1, channels)
    l = self.init_layer(l, x)

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.shape, (2,))  # config.filters = 2

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.verify_contract(l, x, training=False)


class DepthwiseConv1DTest(test_utils.SequenceLayerTest):
  """Test behavior of DepthwiseConv1D layer."""

  @parameterized.product(
      params=[
          # kernel_size with smaller, equal and larger strides.
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          (3, 2, 1),
          (3, 3, 1),
          (3, 4, 1),
          # dilation_rate.
          (3, 1, 2),
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
      channel_multiplier=[1, 2],
  )
  def test_depthwise_conv1d(self, params, padding, channel_multiplier):
    kernel_size, stride, dilation_rate = params
    config = self.sl.DepthwiseConv1D.Config(
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        channel_multiplier=channel_multiplier,
        name='depthwise_conv1d',
    )
    l = self.make_layer(config)
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)
    self.assertEqual(l.name, 'depthwise_conv1d')

    supports_step = padding in (
        'causal_valid',
        'reverse_causal_valid',
        'causal',
        'reverse_causal',
        'semicausal',
    )
    self.assertEqual(l.supports_step, supports_step)

    effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
    expected_input_latency = (
        effective_kernel_size - 1
        if padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size, channels = 2, 3
    x = self.random_sequence(batch_size, 1, channels)
    l = self.init_layer(l, x)

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.shape, (channels * channel_multiplier,))

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.verify_contract(l, x, training=False)


class Conv1DTransposeTest(test_utils.SequenceLayerTest):
  """Test behavior of Conv1DTranspose layer."""

  @parameterized.product(
      params=[
          (1, 1),
          (2, 1),
          (2, 2),
          (2, 3),
          (3, 2),
          (3, 3),
          (3, 4),
      ],
      padding=[
          'same',
          'valid',
          'causal',
      ],
  )
  def test_conv1d_transpose(self, params, padding):
    kernel_size, stride = params
    config = self.sl.Conv1DTranspose.Config(
        filters=2,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        name='conv1d_transpose',
    )
    l = self.make_layer(config)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, fractions.Fraction(stride))
    self.assertEqual(l.name, 'conv1d_transpose')

    # Transpose convolution layers in Step mode are only supported for causal padding.
    self.assertEqual(l.supports_step, padding == 'causal')

    batch_size, channels = 2, 3
    x = self.random_sequence(batch_size, 1, channels)
    l = self.init_layer(l, x)

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.shape, (2,))

    for time in range(5, 10):
      x = self.random_sequence(batch_size, time, channels)
      # Just verify it runs and produces expected shape
      y = l.layer(x, training=False)
      self.assertEqual(y.channel_shape, (2,))
      # Test basic verify_contract (without step check since not supported)
      self.verify_contract(l, x, training=False)


class Conv2DTest(test_utils.SequenceLayerTest):
  """Test behavior of Conv2D layer."""

  @parameterized.product(
      params=[
          # kernel_size, strides, dilation_rate
          ((3, 3), (1, 1), (1, 1)),
          ((3, 3), (2, 2), (1, 1)),
          ((3, 3), (1, 1), (2, 2)),
      ],
      time_padding=[
          'same',
          'valid',
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
      spatial_padding=[
          'same',
          'valid',
      ],
  )
  def test_conv2d(self, params, time_padding, spatial_padding):
    kernel_size, stride, dilation_rate = params
    config = self.sl.Conv2D.Config(
        filters=2,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        name='conv2d',
    )
    l = self.make_layer(config)
    self.assertEqual(l.block_size, stride[0])
    self.assertEqual(1 / l.output_ratio, stride[0])
    self.assertEqual(l.name, 'conv2d')

    supports_step = time_padding in (
        'causal_valid',
        'reverse_causal_valid',
        'causal',
        'reverse_causal',
        'semicausal',
    )
    self.assertEqual(l.supports_step, supports_step)

    effective_kernel_size_t = (kernel_size[0] - 1) * dilation_rate[0] + 1
    expected_input_latency = (
        effective_kernel_size_t - 1
        if time_padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride[0])

    batch_size, spatial_dim, channels = 2, 8, 3
    x = self.random_sequence(batch_size, 1, spatial_dim, channels)
    l = self.init_layer(l, x)

    # Channel shape of Conv2D sequence contains the spatial dimension + filters
    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.shape[-1], 2)

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, spatial_dim, channels)
      self.verify_contract(l, x, training=False)


class Conv2DTransposeTest(test_utils.SequenceLayerTest):
  """Test behavior of Conv2DTranspose layer."""

  @parameterized.product(
      params=[
          ((3, 3), (2, 2)),
          ((3, 3), (1, 1)),
      ],
      time_padding=[
          'same',
          'valid',
          'causal',
      ],
      spatial_padding=[
          'same',
          'valid',
      ],
  )
  def test_conv2d_transpose(self, params, time_padding, spatial_padding):
    kernel_size, stride = params
    config = self.sl.Conv2DTranspose.Config(
        filters=2,
        kernel_size=kernel_size,
        strides=stride,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        name='conv2d_transpose',
    )
    l = self.make_layer(config)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, fractions.Fraction(stride[0]))
    self.assertEqual(l.name, 'conv2d_transpose')
    # Transpose convolution layers in Step mode are only supported for causal padding.
    self.assertEqual(l.supports_step, time_padding == 'causal')

    batch_size, spatial_dim, channels = 2, 8, 3
    x = self.random_sequence(batch_size, 1, spatial_dim, channels)
    l = self.init_layer(l, x)

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.shape[-1], 2)

    for time in range(5, 10):
      x = self.random_sequence(batch_size, time, spatial_dim, channels)
      self.verify_contract(l, x, training=False)
