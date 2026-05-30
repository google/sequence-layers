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
"""Tests for convolution MLX sequence layers."""

from absl.testing import absltest
from absl.testing import parameterized

from sequence_layers.mlx import convolution
from sequence_layers.mlx import test_utils
from sequence_layers.specs import convolution_behaviors as spec


class Conv1DTest(
    spec.Conv1DTest, test_utils.SequenceLayerTest, parameterized.TestCase
):

  def test_from_config(self):
    config = convolution.Conv1D.Config(
        filters=8,
        kernel_size=3,
        padding='causal',
    )
    mlx_layer = config.make()
    self.assertIsInstance(
        mlx_layer,
        convolution.DeferredConv1D,
    )
    x = self.random_sequence(1, 8, 4)
    y = mlx_layer.layer(x, training=False)
    self.assertEqual(y.channel_shape, (8,))


class DepthwiseConv1DTest(
    spec.DepthwiseConv1DTest,
    test_utils.SequenceLayerTest,
    parameterized.TestCase,
):

  def test_from_config(self):
    config = convolution.DepthwiseConv1D.Config(
        kernel_size=3,
        padding='causal',
    )
    mlx_layer = config.make()
    self.assertIsInstance(
        mlx_layer,
        convolution.DeferredDepthwiseConv1D,
    )
    x = self.random_sequence(1, 8, 4)
    y = mlx_layer.layer(x, training=False)
    self.assertEqual(y.channel_shape, (4,))


class Conv1DTransposeTest(
    spec.Conv1DTransposeTest,
    test_utils.SequenceLayerTest,
    parameterized.TestCase,
):

  def test_from_config(self):
    config = convolution.Conv1DTranspose.Config(
        filters=8,
        kernel_size=3,
        strides=2,
        padding='causal',
    )
    mlx_layer = config.make()
    self.assertIsInstance(
        mlx_layer,
        convolution.DeferredConv1DTranspose,
    )
    x = self.random_sequence(1, 4, 4)
    y = mlx_layer.layer(x, training=False)
    self.assertEqual(y.channel_shape, (8,))


class Conv2DTest(
    spec.Conv2DTest, test_utils.SequenceLayerTest, parameterized.TestCase
):
  pass


class Conv2DTransposeTest(
    spec.Conv2DTransposeTest,
    test_utils.SequenceLayerTest,
    parameterized.TestCase,
):
  pass


if __name__ == '__main__':
  absltest.main()
