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
"""Tests for sequence_layers.tensorflow.pooling."""

import itertools

from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v2 as tf


class PoolingTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          ((1, 1), (2, 1), (2, 2), (2, 3), (3, 2), (3, 4)),
          (
              (sl.MaxPooling1D, tf.keras.layers.MaxPooling1D),
              (sl.AveragePooling1D, tf.keras.layers.AveragePooling1D),
          ),
          ('causal', 'same', 'valid', 'reverse_causal'),
      )
  )
  def test_pooling_1d(self, pool_size_strides, sl_class_keras_class, padding):
    pool_size, strides = pool_size_strides
    sl_class, keras_class = sl_class_keras_class
    l = sl_class(pool_size, strides, padding)
    self.assertEqual(l.block_size, strides)
    self.assertEqual(1 / l.output_ratio, strides)
    self.assertEqual(l.supports_step, padding == 'causal')

    batch_size, channels = 2, 3
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), tf.TensorShape(channels)
      )
      x_np, y_np = self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      self.assertEmpty(l.trainable_variables)

      # Correctness test: Compare the layer-wise output to manually executing
      # a causal pooling over x and its mask.

      explicit_padding = test_util.convolution_explicit_padding(
          padding, pool_size, dilation_rate=1
      )

      x_np_values = np.pad(x_np.values, [(0, 0), explicit_padding, (0, 0)])
      # Only pad the mask in causal/reverse_causal mode.
      if padding == 'causal':
        x_mask = np.pad(
            x_np.mask, [(0, 0), explicit_padding], constant_values=1.0
        )
      else:
        x_mask = x.mask

      pooling_layer = keras_class(pool_size, strides, padding='valid')
      values_golden = pooling_layer(x_np_values)
      mask_golden = test_util.conv1d_mask(
          x_mask, pool_size, strides, dilation_rate=1, padding=padding
      )
      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis]

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(
      itertools.product(
          (
              (1, 1),
              (2, 1),
              (2, 2),
              (2, 3),
              (3, 2),
              (3, 4),
              ((2, 3), (1, 2)),
              ((3, 2), (1, 2)),
              ((2, 3), (2, 1)),
              ((3, 2), (2, 1)),
          ),
          (
              (sl.MaxPooling2D, tf.keras.layers.MaxPooling2D),
              (sl.AveragePooling2D, tf.keras.layers.AveragePooling2D),
          ),
          ('causal', 'same', 'valid', 'reverse_causal'),
          ('causal', 'same', 'valid', 'reverse_causal'),
      )
  )
  def test_pooling_2d(
      self,
      pool_size_strides,
      sl_class_keras_class,
      time_padding,
      spatial_padding,
  ):
    pool_size, strides = pool_size_strides
    time_pool_size, spatial_pool_size = (
        (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
    )
    time_strides, spatial_strides = (
        (strides, strides) if isinstance(strides, int) else strides
    )
    sl_class, keras_class = sl_class_keras_class
    l = sl_class(
        pool_size,
        strides,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
    )
    self.assertEqual(l.block_size, time_strides)
    self.assertEqual(1 / l.output_ratio, time_strides)
    self.assertEqual(l.supports_step, time_padding == 'causal')

    batch_size, spatial, channels = 2, 11, 3
    for time in range(10 * l.block_size - 1, 10 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, spatial, channels)
      if spatial_padding in ('same', 'causal', 'reverse_causal'):
        # Ceiling division.
        spatial_output = (spatial + spatial_strides - 1) // spatial_strides
      else:
        assert spatial_padding == 'valid'
        dilation_rate = 1
        spatial_output = spatial - (spatial_pool_size - 1) * dilation_rate
        # Ceiling division.
        spatial_output = (
            spatial_output + spatial_strides - 1
        ) // spatial_strides
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          tf.TensorShape([spatial_output, channels]),
      )
      x_np, y_np = self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)
      self.assertEmpty(l.trainable_variables)

      # Correctness test: Compare the layer-wise output to manually executing
      # a causal pooling over x and its mask.

      explicit_spatial_padding = test_util.convolution_explicit_padding(
          spatial_padding, spatial_pool_size, dilation_rate=1
      )
      explicit_time_padding = test_util.convolution_explicit_padding(
          time_padding, time_pool_size, dilation_rate=1
      )

      x_np_values = np.pad(
          x_np.values,
          [(0, 0), explicit_time_padding, explicit_spatial_padding, (0, 0)],
      )
      # Only pad the mask in causal mode.
      if time_padding == 'causal':
        x_np_mask = np.pad(
            x_np.mask, [(0, 0), explicit_time_padding], constant_values=1.0
        )
      else:
        x_np_mask = x.mask

      pooling_layer = keras_class(pool_size, strides, padding='valid')
      values_golden = pooling_layer(x_np_values)
      mask_golden = test_util.conv1d_mask(
          x_np_mask,
          time_pool_size,
          time_strides,
          dilation_rate=1,
          padding=time_padding,
      )
      # Apply masking.
      values_golden = values_golden * mask_golden[:, :, tf.newaxis, tf.newaxis]

      self.assertAllClose(y_np.values, values_golden)
      self.assertAllEqual(y_np.mask, mask_golden)
    self.verify_tflite_step(l, x)


if __name__ == '__main__':
  tf.test.main()
