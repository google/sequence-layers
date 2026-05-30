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
"""Tests for pooling MLX sequence layers."""

from absl.testing import absltest
from absl.testing import parameterized
import mlx.core as mx
import numpy as np

from sequence_layers.jax import pooling as jax_pooling
from sequence_layers.mlx import pooling
from sequence_layers.mlx import test_utils
from sequence_layers.specs import pooling_behaviors as spec


class Pooling1DTest(test_utils.SequenceLayerTest, spec.Pooling1DTest):
  """Shared behavior tests for 1D pooling layers in MLX."""


class MaxPooling1DTest(test_utils.SequenceLayerTest):
  """MLX-specific tests for MaxPooling1D."""

  def test_max_values(self):
    values = mx.array([[[1.0], [3.0], [2.0], [5.0], [4.0]]])
    mask = mx.ones((1, 5), dtype=mx.bool_)
    x = type(self.random_sequence(1, 5, 1))(values, mask)
    layer = pooling.MaxPooling1D(pool_size=3, padding='valid')
    y = layer.layer(x)
    expected = np.array([[[3.0], [5.0], [5.0]]])
    np.testing.assert_allclose(y.values, expected)

  def test_from_config(self):
    config = jax_pooling.MaxPooling1D.Config(
        pool_size=3,
        padding='causal',
    )
    mlx_layer = pooling.MaxPooling1D.from_config(config)
    self.assertIsInstance(mlx_layer, pooling.MaxPooling1D)
    self.verify_contract(mlx_layer, self.random_sequence(2, 10, 4))


class MinPooling1DTest(test_utils.SequenceLayerTest):
  """MLX-specific tests for MinPooling1D."""

  def test_min_values(self):
    values = mx.array([[[5.0], [3.0], [4.0], [1.0], [2.0]]])
    mask = mx.ones((1, 5), dtype=mx.bool_)
    x = type(self.random_sequence(1, 5, 1))(values, mask)
    layer = pooling.MinPooling1D(pool_size=3, padding='valid')
    y = layer.layer(x)
    expected = np.array([[[3.0], [1.0], [1.0]]])
    np.testing.assert_allclose(y.values, expected)

  def test_from_config(self):
    config = jax_pooling.MinPooling1D.Config(
        pool_size=3,
        padding='causal',
    )
    mlx_layer = pooling.MinPooling1D.from_config(config)
    self.assertIsInstance(mlx_layer, pooling.MinPooling1D)
    self.verify_contract(mlx_layer, self.random_sequence(2, 10, 4))


class AveragePooling1DTest(test_utils.SequenceLayerTest):
  """MLX-specific tests for AveragePooling1D."""

  def test_average_values(self):
    values = mx.array([[[3.0], [6.0], [9.0], [12.0], [15.0]]])
    mask = mx.ones((1, 5), dtype=mx.bool_)
    x = type(self.random_sequence(1, 5, 1))(values, mask)
    layer = pooling.AveragePooling1D(pool_size=3, padding='valid')
    y = layer.layer(x)
    expected = np.array([[[6.0], [9.0], [12.0]]])
    np.testing.assert_allclose(y.values, expected)

  def test_masked_average(self):
    values = mx.array([[[3.0], [6.0], [0.0]]])
    mask = mx.array([[True, True, False]])
    x = type(self.random_sequence(1, 3, 1))(values, mask)
    layer = pooling.AveragePooling1D(
        pool_size=3,
        padding='valid',
        masked_average=True,
    )
    y = layer.layer(x)
    expected = np.array([[[4.5]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-5)

  def test_from_config(self):
    config = jax_pooling.AveragePooling1D.Config(
        pool_size=3,
        padding='causal',
    )
    mlx_layer = pooling.AveragePooling1D.from_config(config)
    self.assertIsInstance(mlx_layer, pooling.AveragePooling1D)
    self.verify_contract(mlx_layer, self.random_sequence(2, 10, 4))


if __name__ == '__main__':
  absltest.main()
