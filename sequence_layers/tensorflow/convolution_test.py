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
import tensorflow.compat.v2 as tf


class ComputeConvMaskTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.product(
      params=[
          # kernel_size 1
          (1, 1, 1),
          (1, 2, 1),
          (1, 1, 2),
          (1, 3, 1),
          (1, 1, 3),
          # kernel_size > stride or dilation:
          (5, 1, 1),
          (5, 2, 1),
          (5, 1, 2),
          (5, 3, 1),
          (5, 1, 3),
          # kernel_size = stride or dilation:
          (2, 2, 1),
          (2, 1, 2),
          (3, 3, 1),
          (3, 1, 3),
          # kernel_size < stride or dilation:
          (3, 4, 1),
          (3, 1, 4),
          (3, 5, 1),
          (3, 1, 5),
      ],
      padding=['causal', 'valid', 'same', 'reverse_causal'],
  )
  def test_dense_left_aligned_mask(self, params, padding):
    """Test that compute_conv_mask matches TF's behavior for dense masks."""
    kernel_size, stride, dilation_rate = params
    # Try even and odd tensor lengths.
    for maxlen in [16, 17]:
      # Sweep all possible lengths.
      lengths = tf.range(maxlen + 1, dtype=tf.int32)
      mask = tf.sequence_mask(lengths, maxlen=maxlen, dtype=sl.MASK_DTYPE)
      actual = sl.compute_conv_mask(
          mask,
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
      )
      expected = test_util.conv1d_mask(
          mask,
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
      )
      self.assertAllEqual(actual, expected)

  @parameterized.product(
      params=[
          # kernel_size 1
          (1, 1, 1),
          (1, 2, 1),
          (1, 1, 2),
          (1, 3, 1),
          (1, 1, 3),
          # kernel_size > stride or dilation:
          (5, 1, 1),
          (5, 2, 1),
          (5, 1, 2),
          (5, 3, 1),
          (5, 1, 3),
          # kernel_size = stride or dilation:
          (2, 2, 1),
          (2, 1, 2),
          (3, 3, 1),
          (3, 1, 3),
          # kernel_size < stride or dilation:
          (3, 4, 1),
          (3, 1, 4),
          (3, 5, 1),
          (3, 1, 5),
      ],
      padding=['causal', 'valid', 'same', 'reverse_causal'],
  )
  def test_sparse_mask(self, params, padding):
    """Test that compute_conv_mask matches TF's behavior for dense masks."""
    kernel_size, stride, dilation_rate = params
    # Try even and odd tensor lengths.
    for maxlen in [16, 17]:
      # Sweep all possible lengths.
      lengths = tf.range(maxlen + 1, dtype=tf.int32)
      mask = tf.sequence_mask(lengths, maxlen=maxlen, dtype=sl.MASK_DTYPE)
      # Randomly flip some of the internal values.
      mask = mask * tf.cast(
          tf.random.uniform(
              tf.shape(mask), minval=0.0, maxval=1.0, dtype=tf.float32
          )
          > 0.5,
          sl.MASK_DTYPE,
      )
      actual = sl.compute_conv_mask(
          mask,
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
      )
      expected = test_util.conv1d_mask(
          mask,
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
      )
      self.assertAllEqual(actual, expected)


if __name__ == '__main__':
  tf.test.main()
