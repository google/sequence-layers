"""Tests for normalization MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import normalization
from sequence_layers.mlx import test_utils


class L2NormalizeTest(parameterized.TestCase):

  def test_layer(self):
    layer = normalization.L2Normalize()
    test_utils.verify_contract(self, layer, (8,))

  def test_normalizes(self):
    layer = normalization.L2Normalize()
    values = mx.array([[[3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = test_utils.random_sequence(1, 1, 2).unmask()
    x = type(x)(values, mask)
    y = layer.layer(x)
    # L2 norm of [3, 4] is 5, so output should be [0.6, 0.8].
    np.testing.assert_allclose(np.array(y.values), [[[0.6, 0.8]]], atol=1e-6)

  def test_multi_axis(self):
    layer = normalization.L2Normalize(axis=(-2, -1))
    test_utils.verify_contract(self, layer, (4, 3))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.L2Normalize.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, normalization.L2Normalize)
    test_utils.verify_contract(self, mlx_layer, (8,))


class RMSNormalizationTest(parameterized.TestCase):

  def test_layer(self):
    layer = normalization.RMSNormalization()
    test_utils.verify_contract(self, layer, (8,))

  def test_no_scale(self):
    layer = normalization.RMSNormalization(use_scale=False)
    test_utils.verify_contract(self, layer, (8,))

  def test_normalizes(self):
    layer = normalization.RMSNormalization(use_scale=False)
    values = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = test_utils.random_sequence(1, 1, 4).unmask()
    x = type(x)(values, mask)
    y = layer.layer(x)
    # After RMS norm, the RMS of the output should be ~1.
    rms = float(mx.sqrt(mx.mean(mx.square(y.values))))
    np.testing.assert_allclose(rms, 1.0, atol=0.1)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.RMSNormalization.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, normalization.RMSNormalization)
    test_utils.verify_contract(self, mlx_layer, (8,))


class LayerNormalizationTest(parameterized.TestCase):

  def test_layer(self):
    layer = normalization.LayerNormalization()
    test_utils.verify_contract(self, layer, (8,))

  def test_no_affine(self):
    layer = normalization.LayerNormalization(use_scale=False, use_bias=False)
    test_utils.verify_contract(self, layer, (8,))

  def test_normalizes(self):
    layer = normalization.LayerNormalization(use_scale=False, use_bias=False)
    values = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = test_utils.random_sequence(1, 1, 4).unmask()
    x = type(x)(values, mask)
    y = layer.layer(x)
    # After layer norm, mean should be ~0, std should be ~1.
    mean = float(mx.mean(y.values))
    std = float(mx.sqrt(mx.mean(mx.square(y.values - mean))))
    np.testing.assert_allclose(mean, 0.0, atol=1e-5)
    np.testing.assert_allclose(std, 1.0, atol=0.15)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.LayerNormalization.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, normalization.LayerNormalization)
    test_utils.verify_contract(self, mlx_layer, (8,))


class BatchNormalizationTest(parameterized.TestCase):

  def test_layer(self):
    layer = normalization.BatchNormalization()
    test_utils.verify_contract(self, layer, (8,))

  def test_no_affine(self):
    layer = normalization.BatchNormalization(use_scale=False, use_bias=False)
    test_utils.verify_contract(self, layer, (8,))

  def test_normalizes(self):
    layer = normalization.BatchNormalization(use_scale=False, use_bias=False)
    # Set known running stats.
    layer._ensure_initialized((1, 1, 4))
    layer._running_mean = mx.array([1.0, 2.0, 3.0, 4.0])
    layer._running_var = mx.array([1.0, 1.0, 1.0, 1.0])
    values = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = type(test_utils.random_sequence(1, 1, 4))(values, mask)
    y = layer.layer(x)
    # (x - mean) / sqrt(var + eps) should be ~0
    np.testing.assert_allclose(y.values, np.zeros((1, 1, 4)), atol=1e-3)

  def test_scale_and_bias(self):
    layer = normalization.BatchNormalization()
    layer._ensure_initialized((1, 1, 4))
    layer._running_mean = mx.zeros((4,))
    layer._running_var = mx.ones((4,))
    layer._scale = mx.array([2.0, 2.0, 2.0, 2.0])
    layer._bias = mx.array([1.0, 1.0, 1.0, 1.0])
    values = mx.array([[[1.0, 0.0, -1.0, 2.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = type(test_utils.random_sequence(1, 1, 4))(values, mask)
    y = layer.layer(x)
    # (x - 0) / sqrt(1 + 0.001) * 2 + 1
    scale = 2.0 / float(mx.sqrt(mx.array(1.001)))
    expected = np.array([[[
        1.0 * scale + 1.0,
        0.0 * scale + 1.0,
        -1.0 * scale + 1.0,
        2.0 * scale + 1.0,
    ]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-5)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.BatchNormalization.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, normalization.BatchNormalization)
    test_utils.verify_contract(self, mlx_layer, (8,))


class GroupNormalizationTest(parameterized.TestCase):

  def test_layer(self):
    layer = normalization.GroupNormalization(num_groups=2)
    test_utils.verify_contract(self, layer, (8,))

  def test_no_affine(self):
    layer = normalization.GroupNormalization(
        num_groups=4, use_scale=False, use_bias=False
    )
    test_utils.verify_contract(self, layer, (8,))

  def test_num_groups_must_divide(self):
    layer = normalization.GroupNormalization(num_groups=3)
    with self.assertRaises(ValueError):
      layer.layer(test_utils.random_sequence(1, 2, 8))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.GroupNormalization.Config(num_groups=2)
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, normalization.GroupNormalization)
    test_utils.verify_contract(self, mlx_layer, (8,))


if __name__ == '__main__':
  absltest.main()
