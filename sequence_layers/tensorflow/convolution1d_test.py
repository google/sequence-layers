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
"""Tests for sequence_layers.tensorflow.convolution."""

from absl.testing import parameterized
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class ConvolutionTest(test_util.SequenceLayerTest, parameterized.TestCase):

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
      padding=['same', 'valid', 'causal', 'reverse_causal'],
      weight_norm=[True, False],
  )
  def test_conv1d(self, params, padding, weight_norm):
    kernel_size, stride, dilation_rate = params
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      l = sl.Conv1D(
          filters=2,
          kernel_size=kernel_size,
          strides=stride,
          dilation_rate=dilation_rate,
          padding=padding,
          bias_initializer=tf.random_uniform_initializer,
          weight_norm=weight_norm,
      )
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)

    batch_size, channels = 2, 3
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), 2)
      x_np, y_np = self.verify_contract(l, x, training=False)
      expected_variables = 3 if weight_norm else 2
      self.assertLen(l.variables, expected_variables)
      self.assertLen(l.trainable_variables, expected_variables)
      if weight_norm:
        self.assertIsInstance(l._layer, utils.WeightNorm)
        self.assertCountEqual(
            [v.name for v in l.variables],
            [
                'test/conv1d/weight_norm/kernel:0',
                'test/conv1d/weight_norm/bias:0',
                'test/conv1d/weight_norm/g:0',
            ],
        )
      else:
        self.assertIsInstance(l._layer, tf.keras.layers.Conv1D)
        self.assertCountEqual(
            [v.name for v in l.variables],
            ['test/conv1d/kernel:0', 'test/conv1d/bias:0'],
        )

      # Correctness test: Compare the layer-wise output to manually executing
      # a 1D convolution with the same weights.
      if weight_norm:
        v, bias, g = l.trainable_variables
        assert 'kernel' in v.name
        assert 'bias' in bias.name
        kernel = tf.nn.l2_normalize(v, axis=[0, 1]) * g
      else:
        kernel, bias = l.trainable_variables
        assert 'kernel' in kernel.name
        assert 'bias' in bias.name

      explicit_padding = test_util.convolution_explicit_padding(
          padding, kernel_size, dilation_rate
      )

      x_np_values = tf.pad(x_np.values, [(0, 0), explicit_padding, (0, 0)])
      if padding == 'causal':
        x_np_mask = tf.pad(
            x_np.mask, [(0, 0), explicit_padding], constant_values=1.0
        )
      else:
        x_np_mask = x_np.mask
      values_golden = tf.nn.conv1d(
          x_np_values,
          kernel,
          stride=stride,
          dilations=dilation_rate,
          padding='VALID',
      )
      values_golden = tf.nn.bias_add(values_golden, bias)
      mask_golden = test_util.conv1d_mask(
          x_np_mask, kernel_size, stride, dilation_rate, padding
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)

    # TODO(b/143543055): tf.lite conversion of dilation_rate > 1 doesn't work
    # yet. It appears to assume 4D input to SpaceToBatchNd.
    if dilation_rate == 1:
      self.verify_tflite_step(l, x)

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
      padding=['same', 'valid', 'causal', 'reverse_causal'],
  )
  def test_depthwise_conv1d(self, params, padding):
    kernel_size, stride, dilation_rate = params
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      l = sl.DepthwiseConv1D(
          kernel_size=kernel_size,
          strides=stride,
          depth_multiplier=2,
          dilation_rate=dilation_rate,
          bias_initializer=tf.random_uniform_initializer,
          padding=padding,
      )
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)

    batch_size, channels = 2, 3
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), [6])
      x_np, y_np = self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 2)
      self.assertLen(l.trainable_variables, 2)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/depthwise_conv1d/depthwise_kernel:0',
              'test/depthwise_conv1d/bias:0',
          ],
      )

      # Correctness test: Compare the layer-wise output to manually executing
      # a depthwise 1D convolution with the same weights.
      kernel, bias = l.trainable_variables
      assert 'kernel' in kernel.name
      assert 'bias' in bias.name

      explicit_padding = test_util.convolution_explicit_padding(
          padding, kernel_size, dilation_rate
      )

      x_np_values = tf.pad(x_np.values, [(0, 0), explicit_padding, (0, 0)])
      if padding == 'causal':
        x_np_mask = tf.pad(
            x_np.mask, [(0, 0), explicit_padding], constant_values=1.0
        )
      else:
        x_np_mask = x_np.mask

      values_golden = tf.nn.depthwise_conv2d(
          x_np_values[:, tf.newaxis, :, :],
          kernel,
          strides=(1, stride, stride, 1),
          dilations=(1, dilation_rate),
          padding='VALID',
      )
      values_golden = tf.nn.bias_add(values_golden, bias)
      values_golden = tf.squeeze(values_golden, 1)
      mask_golden = test_util.conv1d_mask(
          x_np_mask, kernel_size, stride, dilation_rate, padding
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)

    self.verify_tflite_step(l, x)

  @parameterized.product(
      params=[
          # 1x1 conv.
          (1, 1),
          # even kernel_size with smaller, equal and larger strides.
          (2, 1),
          (2, 2),
          (2, 3),
          # odd kernel_size with smaller, equal and larger strides.
          (3, 2),
          (3, 3),
          (3, 4),
      ],
      padding=['same', 'valid', 'causal', 'reverse_causal'],
  )
  def test_sinc_filter1d(self, params, padding):
    kernel_size, stride = params
    with tf.name_scope('test'):
      l = sl.SincFilter1D(
          kernel_size=kernel_size,
          strides=stride,
          padding=padding,
      )
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)

    batch_size, channels = 2, 3
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), [channels])
      self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      self.assertEmpty(l.trainable_variables)

    self.verify_tflite_step(l, x, use_flex=True)

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
      padding=['same', 'valid', 'causal', 'reverse_causal'],
      num_heads=[None, 1, 2, 4],
  )
  def test_normalized_depthwise_conv1d(self, params, padding, num_heads):
    kernel_size, stride, dilation_rate = params
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      depth_multiplier = 2
      l = sl.NormalizedDepthwiseConv1D(
          kernel_size=kernel_size,
          strides=stride,
          depth_multiplier=depth_multiplier,
          dilation_rate=dilation_rate,
          bias_initializer=tf.random_uniform_initializer,
          padding=padding,
          num_heads=num_heads,
      )
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)

    batch_size, channels = 2, 4
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), [8])
      x_np, y_np = self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 2)
      self.assertLen(l.trainable_variables, 2)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/normalized_depthwise_conv1d/depthwise_kernel:0',
              'test/normalized_depthwise_conv1d/bias:0',
          ],
      )
      num_parameters = sum(v.shape.num_elements() for v in l.variables)
      # Normal depthwise conv is kernel_size * channels * depth_multiplier for
      # depthwise_kernel and channels * depthwise_multiplier for bias. The
      # number of heads simply groups the channels into num_heads groups of tied
      # weights.
      self.assertEqual(
          num_parameters,
          kernel_size * (num_heads or channels) * depth_multiplier
          + channels * depth_multiplier,
      )

      # Correctness test: Compare the layer-wise output to manually executing
      # a depthwise 1D convolution with the same weights.
      kernel, bias = l.trainable_variables
      assert 'kernel' in kernel.name
      assert 'bias' in bias.name

      explicit_padding = test_util.convolution_explicit_padding(
          padding, kernel_size, dilation_rate
      )

      x_np_values = tf.pad(x_np.values, [(0, 0), explicit_padding, (0, 0)])
      if padding == 'causal':
        x_np_mask = tf.pad(
            x_np.mask, [(0, 0), explicit_padding], constant_values=1.0
        )
      else:
        x_np_mask = x_np.mask

      normalized_kernel = tf.nn.softmax(kernel, axis=1)
      if num_heads:
        normalized_kernel = tf.tile(
            normalized_kernel, [1, 1, channels // num_heads, 1]
        )
      values_golden = tf.nn.depthwise_conv2d(
          x_np_values[:, tf.newaxis, :, :],
          normalized_kernel,
          strides=(1, stride, stride, 1),
          dilations=(1, dilation_rate),
          padding='VALID',
      )
      values_golden = tf.nn.bias_add(values_golden, bias)
      values_golden = tf.squeeze(values_golden, 1)
      mask_golden = test_util.conv1d_mask(
          x_np_mask, kernel_size, stride, dilation_rate, padding
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)

    self.verify_tflite_step(l, x)

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
      padding=['same', 'valid', 'causal', 'reverse_causal'],
  )
  def test_separable_conv1d(self, params, padding):
    kernel_size, stride, dilation_rate = params
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      l = sl.SeparableConv1D(
          filters=3,
          kernel_size=kernel_size,
          strides=stride,
          depth_multiplier=2,
          dilation_rate=dilation_rate,
          bias_initializer=tf.random_uniform_initializer,
          padding=padding,
      )
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)

    batch_size, channels = 2, 3
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), [3])
      x_np, y_np = self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 3)
      self.assertLen(l.trainable_variables, 3)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/separable_conv1d/depthwise_kernel:0',
              'test/separable_conv1d/pointwise_kernel:0',
              'test/separable_conv1d/bias:0',
          ],
      )

      # Correctness test: Compare the layer-wise output to manually executing
      # a separable 1D convolution with the same weights.
      depthwise_kernel, pointwise_kernel, bias = l.trainable_variables
      assert 'depthwise_kernel' in depthwise_kernel.name
      assert 'pointwise_kernel' in pointwise_kernel.name
      assert 'bias' in bias.name

      explicit_padding = test_util.convolution_explicit_padding(
          padding, kernel_size, dilation_rate
      )

      x_np_values = tf.pad(x_np.values, [(0, 0), explicit_padding, (0, 0)])
      if padding == 'causal':
        x_np_mask = tf.pad(
            x_np.mask, [(0, 0), explicit_padding], constant_values=1.0
        )
      else:
        x_np_mask = x_np.mask

      values_golden = tf.nn.separable_conv2d(
          x_np_values[:, tf.newaxis, :, :],
          depthwise_kernel[tf.newaxis, :, :, :],
          pointwise_kernel[tf.newaxis, :, :, :],
          strides=(1, stride, stride, 1),
          dilations=(1, dilation_rate),
          padding='VALID',
      )
      values_golden = tf.nn.bias_add(values_golden, bias)
      values_golden = tf.squeeze(values_golden, 1)
      mask_golden = test_util.conv1d_mask(
          x_np_mask, kernel_size, stride, dilation_rate, padding
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)

    self.verify_tflite_step(l, x)

  @parameterized.product(
      params=[(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4)],
      padding=['causal', 'valid', 'same'],
      weight_norm=[True, False],
  )
  def test_conv1d_transpose(self, params, padding, weight_norm):
    kernel_size, stride = params
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      l = sl.Conv1DTranspose(
          filters=2,
          kernel_size=kernel_size,
          strides=stride,
          bias_initializer=tf.random_uniform_initializer,
          padding=padding,
          weight_norm=weight_norm,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, stride)

    batch_size, channels = 10, 1
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([2]))
      x_np, y_np = self.verify_contract(l, x, training=False)
      expected_variables = 3 if weight_norm else 2
      self.assertLen(l.variables, expected_variables)
      self.assertLen(l.trainable_variables, expected_variables)
      if weight_norm:
        self.assertIsInstance(l._layer, utils.WeightNorm)
        self.assertCountEqual(
            [v.name for v in l.variables],
            [
                'test/conv1d_transpose/kernel:0',
                'test/conv1d_transpose/bias:0',
                'test/conv1d_transpose/g:0',
            ],
        )
      else:
        self.assertIsInstance(l._layer, tf.keras.layers.Conv2DTranspose)
        self.assertCountEqual(
            [v.name for v in l.variables],
            ['test/conv1d_transpose/kernel:0', 'test/conv1d_transpose/bias:0'],
        )

      # Correctness test: Compare the layer-wise output to manually executing
      # a 1D transpose convolution with the same weights.
      if weight_norm:
        v, bias, g = l.trainable_variables
        assert 'kernel' in v.name
        assert 'bias' in bias.name
        # It's actually a Conv2DTranpsose so kernel has 4 dims.
        kernel = tf.nn.l2_normalize(v, axis=[0, 1, 2]) * g
      else:
        kernel, bias = l.trainable_variables
        assert 'kernel' in kernel.name
        assert 'bias' in bias.name

      if padding == 'same':
        output_time = time * stride
      else:
        output_time = time * stride + max(kernel_size - stride, 0)
      values_golden = tf.nn.conv2d_transpose(
          x_np.values[:, :, tf.newaxis, :],
          kernel,
          output_shape=[batch_size, output_time, 1, 2],
          strides=stride,
          padding='SAME' if padding == 'same' else 'VALID',
          data_format='NHWC',
      )[:, :, 0, :]
      values_golden = tf.nn.bias_add(values_golden, bias)
      mask_golden = test_util.conv1d_transpose_mask(
          x_np.mask, kernel_size, stride, padding
      )
      # In causal mode we trim off the final kernel_size - strides samples
      # because we cannot produce them in step mode.
      if padding == 'causal':
        trim = max(0, kernel_size - stride)
        if trim:
          values_golden = values_golden[:, :-trim]
          mask_golden = mask_golden[:, :-trim]
      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])
      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)

    self.verify_tflite_step(l, x)

  @parameterized.product(
      params=[(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4)],
      padding=['causal', 'valid', 'same'],
  )
  def test_sinc_filter1d_transpose(self, params, padding):
    kernel_size, stride = params
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      l = sl.SincFilter1DTranspose(
          kernel_size=kernel_size,
          strides=stride,
          padding=padding,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, stride)

    batch_size, channels = 10, 2
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), tf.TensorShape([channels])
      )
      self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      self.assertEmpty(l.trainable_variables)

    self.verify_tflite_step(l, x, use_flex=True)


if __name__ == '__main__':
  tf.test.main()
