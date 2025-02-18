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


def _get_conv_output_size(
    padding, input_size, kernel_size, stride, dilation_rate
):
  if padding in ('same', 'causal', 'reverse_causal'):
    # Ceiling division.
    return (input_size + stride - 1) // stride
  else:
    output_size = input_size - (kernel_size - 1) * dilation_rate
    # Ceiling division.
    return (output_size + stride - 1) // stride


class ConvolutionTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.product(
      kernel_size_strides_dilation_rate=(
          (1, 1, 1),
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          (3, 2, 1),
          (3, 3, 1),
          (3, 4, 1),
          # TODO(b/143912700): Enable dilated tests. Dilation causes the
          # spatial dimension to become unknown when the time dimension is
          # unknown, and the SequenceLayer contract requires input_shape to
          # remain statically known.
          # (1, 1, 2),
          # (2, 1, 2),
          # (3, 1, 2),
          # (1, 1, 3),
          # (2, 1, 3),
          # (3, 1, 3)
      ),
      time_padding=('same', 'valid', 'causal', 'reverse_causal'),
      spatial_padding=('same', 'valid', 'causal', 'reverse_causal'),
      weight_norm=(False, True),
  )
  def test_conv2d(
      self,
      kernel_size_strides_dilation_rate,
      time_padding,
      spatial_padding,
      weight_norm,
  ):
    kernel_size, strides, dilation_rate = kernel_size_strides_dilation_rate
    with tf.name_scope('test'):
      l = sl.Conv2D(
          filters=2,
          kernel_size=kernel_size,
          strides=strides,
          spatial_padding=spatial_padding,
          dilation_rate=dilation_rate,
          bias_initializer=tf.random_uniform_initializer,
          time_padding=time_padding,
          weight_norm=weight_norm,
      )
    self.assertEqual(l.block_size, strides)
    self.assertEqual(1 / l.output_ratio, strides)

    batch_size, spatial, channels = 2, 7, 3
    for time in range(5 * l.block_size - 1, 5 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, spatial, channels)
      spatial_output = _get_conv_output_size(
          spatial_padding, spatial, kernel_size, strides, dilation_rate
      )
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape([spatial_output, 2]),
      )
      x_np, y_np = self.verify_contract(l, x, training=False)
      expected_variables = 3 if weight_norm else 2
      self.assertLen(l.variables, expected_variables)
      self.assertLen(l.trainable_variables, expected_variables)
      if weight_norm:
        self.assertIsInstance(l._layer, utils.WeightNorm)
        self.assertCountEqual(
            [v.name for v in l.variables],
            [
                'test/conv2d/weight_norm/kernel:0',
                'test/conv2d/weight_norm/bias:0',
                'test/conv2d/weight_norm/g:0',
            ],
        )
      else:
        self.assertIsInstance(l._layer, tf.keras.layers.Conv2D)
        self.assertCountEqual(
            [v.name for v in l.variables],
            ['test/conv2d/kernel:0', 'test/conv2d/bias:0'],
        )

      # Correctness test: Compare the layer-wise output to manually executing
      # a 2D convolution with the same weights. We manually pad the
      # input to be causal in time and have same/valid behavior over space.
      if weight_norm:
        v, bias, g = l.trainable_variables
        assert 'kernel' in v.name
        assert 'bias' in bias.name
        kernel = tf.nn.l2_normalize(v, axis=[0, 1, 2]) * g
      else:
        kernel, bias = l.trainable_variables
        assert 'kernel' in kernel.name
        assert 'bias' in bias.name

      explicit_time_padding = test_util.convolution_explicit_padding(
          time_padding, kernel_size, dilation_rate
      )
      explicit_spatial_padding = test_util.convolution_explicit_padding(
          spatial_padding, kernel_size, dilation_rate
      )

      x_np_values = tf.pad(
          x_np.values,
          [(0, 0), explicit_time_padding, explicit_spatial_padding, (0, 0)],
      )
      if time_padding == 'causal':
        x_np_mask = tf.pad(
            x_np.mask, [(0, 0), explicit_time_padding], constant_values=1.0
        )
      else:
        x_np_mask = x_np.mask

      values_golden = tf.nn.conv2d(
          x_np_values,
          kernel,
          strides=strides,
          dilations=dilation_rate,
          padding='VALID',
      )
      values_golden = tf.nn.bias_add(values_golden, bias)
      mask_golden = test_util.conv1d_mask(
          x_np_mask, kernel_size, strides, dilation_rate, time_padding
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)
    self.verify_tflite_step(l, x, rtol=1e-6, atol=1e-6)

  @parameterized.product(
      kernel_size_strides_dilation_rate=(
          (1, 1, 1),
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          (3, 2, 1),
          (3, 4, 1),
          # TODO(b/143912700): Enable dilated tests. Dilation causes the
          # spatial dimension to become unknown when the time dimension is
          # unknown, and the SequenceLayer contract requires input_shape to
          # remain statically known.
          # (1, 1, 2),
          # (2, 1, 2),
          # (3, 1, 2),
          # (1, 1, 3),
          # (2, 1, 3),
          # (3, 1, 3)
      ),
      time_padding=('causal', 'valid', 'same', 'reverse_causal'),
      spatial_padding=('causal', 'valid', 'same', 'reverse_causal'),
  )
  def test_depthwise_conv2d(
      self, kernel_size_strides_dilation_rate, time_padding, spatial_padding
  ):
    kernel_size, strides, dilation_rate = kernel_size_strides_dilation_rate
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      l = sl.DepthwiseConv2D(
          kernel_size=kernel_size,
          strides=strides,
          spatial_padding=spatial_padding,
          depth_multiplier=2,
          dilation_rate=dilation_rate,
          bias_initializer=tf.random_uniform_initializer,
          time_padding=time_padding,
      )
    self.assertEqual(l.block_size, strides)
    self.assertEqual(1 / l.output_ratio, strides)

    batch_size, spatial, channels = 2, 7, 3
    for time in range(5 * l.block_size - 1, 5 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, spatial, channels)
      spatial_output = _get_conv_output_size(
          spatial_padding, spatial, kernel_size, strides, dilation_rate
      )
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape([spatial_output, 6]),
      )
      x_np, y_np = self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 2)
      self.assertLen(l.trainable_variables, 2)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/depthwise_conv2d/depthwise_kernel:0',
              'test/depthwise_conv2d/bias:0',
          ],
      )

      # Correctness test: Compare the layer-wise output to manually executing
      # a depthwise 2D convolution with the same weights. We manually pad the
      # input to be causal in time and have same/valid behavior over space.
      kernel, bias = l.trainable_variables
      assert 'kernel' in kernel.name
      assert 'bias' in bias.name

      explicit_time_padding = test_util.convolution_explicit_padding(
          time_padding, kernel_size, dilation_rate
      )
      explicit_spatial_padding = test_util.convolution_explicit_padding(
          spatial_padding, kernel_size, dilation_rate
      )

      x_np_values = tf.pad(
          x_np.values,
          [(0, 0), explicit_time_padding, explicit_spatial_padding, (0, 0)],
      )
      if time_padding == 'causal':
        x_np_mask = tf.pad(
            x_np.mask, [(0, 0), explicit_time_padding], constant_values=1.0
        )
      else:
        x_np_mask = x_np.mask

      values_golden = tf.nn.depthwise_conv2d(
          x_np_values,
          kernel,
          strides=(1, strides, strides, 1),
          dilations=(dilation_rate, dilation_rate),
          padding='VALID',
      )
      values_golden = tf.nn.bias_add(values_golden, bias)
      mask_golden = test_util.conv1d_mask(
          x_np_mask, kernel_size, strides, dilation_rate, time_padding
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)
    self.verify_tflite_step(l, x)

  @parameterized.product(
      kernel_size_strides_dilation_rate=(
          (1, 1, 1),
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          (3, 2, 1),
          (3, 4, 1),
          # TODO(b/143912700): Enable dilated tests. Dilation causes the
          # spatial dimension to become unknown when the time dimension is
          # unknown, and the SequenceLayer contract requires input_shape to
          # remain statically known.
          # (1, 1, 2),
          # (2, 1, 2),
          # (3, 1, 2),
          # (1, 1, 3),
          # (2, 1, 3),
          # (3, 1, 3)
      ),
      time_padding=('causal', 'valid', 'same', 'reverse_causal'),
      spatial_padding=('causal', 'valid', 'same', 'reverse_causal'),
  )
  def test_separable_conv2d(
      self, kernel_size_strides_dilation_rate, time_padding, spatial_padding
  ):
    kernel_size, strides, dilation_rate = kernel_size_strides_dilation_rate
    # Initialize the bias randomly so that we can tell it is being applied.
    with tf.name_scope('test'):
      l = sl.SeparableConv2D(
          filters=3,
          kernel_size=kernel_size,
          strides=strides,
          spatial_padding=spatial_padding,
          depth_multiplier=2,
          dilation_rate=dilation_rate,
          bias_initializer=tf.random_uniform_initializer,
          time_padding=time_padding,
      )
    self.assertEqual(l.block_size, strides)
    self.assertEqual(1 / l.output_ratio, strides)

    batch_size, spatial, channels = 2, 7, 3
    for time in range(5 * l.block_size - 1, 5 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, spatial, channels)
      spatial_output = _get_conv_output_size(
          spatial_padding, spatial, kernel_size, strides, dilation_rate
      )
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape([spatial_output, 3]),
      )
      x_np, y_np = self.verify_contract(l, x, training=False)
      self.assertLen(l.variables, 3)
      self.assertLen(l.trainable_variables, 3)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/separable_conv2d/depthwise_kernel:0',
              'test/separable_conv2d/pointwise_kernel:0',
              'test/separable_conv2d/bias:0',
          ],
      )

      # Correctness test: Compare the layer-wise output to manually executing
      # a separable 2D convolution with the same weights. We manually pad the
      # input to be causal in time and have same/valid behavior over space.
      depthwise_kernel, pointwise_kernel, bias = l.trainable_variables
      assert 'depthwise_kernel' in depthwise_kernel.name
      assert 'pointwise_kernel' in pointwise_kernel.name
      assert 'bias' in bias.name

      explicit_time_padding = test_util.convolution_explicit_padding(
          time_padding, kernel_size, dilation_rate
      )
      explicit_spatial_padding = test_util.convolution_explicit_padding(
          spatial_padding, kernel_size, dilation_rate
      )

      x_np_values = tf.pad(
          x_np.values,
          [(0, 0), explicit_time_padding, explicit_spatial_padding, (0, 0)],
      )
      if time_padding == 'causal':
        x_np_mask = tf.pad(
            x_np.mask, [(0, 0), explicit_time_padding], constant_values=1.0
        )
      else:
        x_np_mask = x_np.mask

      values_golden = tf.nn.separable_conv2d(
          x_np_values,
          depthwise_kernel,
          pointwise_kernel,
          strides=(1, strides, strides, 1),
          dilations=(dilation_rate, dilation_rate),
          padding='VALID',
      )
      values_golden = tf.nn.bias_add(values_golden, bias)
      mask_golden = test_util.conv1d_mask(
          x_np_mask, kernel_size, strides, dilation_rate, time_padding
      )

      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis, tf.newaxis]
      values_golden, mask_golden = self.evaluate([values_golden, mask_golden])

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)
    self.verify_tflite_step(l, x)


if __name__ == '__main__':
  tf.test.main()
