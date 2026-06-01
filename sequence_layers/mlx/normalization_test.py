"""Tests for normalization MLX sequence layers."""

# pylint: disable=import-outside-toplevel,protected-access

from absl.testing import absltest
import mlx.core as mx
import numpy as np

from sequence_layers.mlx import normalization
from sequence_layers.mlx import test_utils
from sequence_layers.specs import normalization_behaviors


class L2NormalizeTest(
    test_utils.SequenceLayerTest,
    normalization_behaviors.L2NormalizeTest,
):

  def test_layer(self):
    layer = normalization.L2Normalize.Config().make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_normalizes(self):
    layer = normalization.L2Normalize.Config().make()
    values = mx.array([[[3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = self.random_sequence(1, 1, 2).unmask()
    x = type(x)(values, mask)
    y = layer.layer(x, training=False)
    # L2 norm of [3, 4] is 5, so output should be [0.6, 0.8].
    np.testing.assert_allclose(np.array(y.values), [[[0.6, 0.8]]], atol=1e-6)

  def test_multi_axis(self):
    layer = normalization.L2Normalize.Config(axis=(-2, -1)).make()
    x = self.random_sequence(2, 3, 4, 3)
    self.verify_contract(layer, x)

  def test_from_config(self):
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.L2Normalize.Config()
    mlx_config = normalization.L2Normalize.Config(
        axis=config.axis,
        epsilon=config.epsilon,
        name=config.name,
    )
    mlx_layer = mlx_config.make()
    self.assertIsInstance(mlx_layer, normalization.L2Normalize)
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(mlx_layer, x)


class RMSNormalizationTest(
    test_utils.SequenceLayerTest,
    normalization_behaviors.RMSNormalizationTest,
):

  def test_layer(self):
    layer = normalization.RMSNormalization.Config().make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_no_scale(self):
    layer = normalization.RMSNormalization.Config(use_scale=False).make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_normalizes(self):
    layer = normalization.RMSNormalization.Config(use_scale=False).make()
    values = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = self.random_sequence(1, 1, 4).unmask()
    x = type(x)(values, mask)
    y = layer.layer(x, training=False)
    # After RMS norm, the RMS of the output should be ~1.
    rms = float(mx.sqrt(mx.mean(mx.square(y.values))))
    np.testing.assert_allclose(rms, 1.0, atol=0.1)

  def test_from_config(self):
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.RMSNormalization.Config()
    mlx_config = normalization.RMSNormalization.Config(
        axis=config.axis,
        epsilon=config.epsilon,
        use_scale=config.use_scale,
        name=config.name,
    )
    mlx_layer = mlx_config.make()
    self.assertIsInstance(mlx_layer, normalization.RMSNormalization)
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(mlx_layer, x)


class LayerNormalizationTest(
    test_utils.SequenceLayerTest,
    normalization_behaviors.LayerNormalizationTest,
):

  def test_layer(self):
    layer = normalization.LayerNormalization.Config().make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_no_affine(self):
    layer = normalization.LayerNormalization.Config(
        use_scale=False,
        use_bias=False,
    ).make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_normalizes(self):
    layer = normalization.LayerNormalization.Config(
        use_scale=False,
        use_bias=False,
    ).make()
    values = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = self.random_sequence(1, 1, 4).unmask()
    x = type(x)(values, mask)
    y = layer.layer(x, training=False)
    # After layer norm, mean should be ~0, std should be ~1.
    mean = float(mx.mean(y.values))
    std = float(mx.sqrt(mx.mean(mx.square(y.values - mean))))
    np.testing.assert_allclose(mean, 0.0, atol=1e-5)
    np.testing.assert_allclose(std, 1.0, atol=0.15)

  def test_from_config(self):
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.LayerNormalization.Config()
    mlx_config = normalization.LayerNormalization.Config(
        axis=config.axis,
        epsilon=config.epsilon,
        use_scale=config.use_scale,
        use_bias=config.use_bias,
        name=config.name,
    )
    mlx_layer = mlx_config.make()
    self.assertIsInstance(mlx_layer, normalization.LayerNormalization)
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(mlx_layer, x)


class BatchNormalizationTest(
    test_utils.SequenceLayerTest,
    normalization_behaviors.BatchNormalizationTest,
):

  def test_layer(self):
    layer = normalization.BatchNormalization.Config().make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_no_affine(self):
    layer = normalization.BatchNormalization.Config(
        use_scale=False,
        use_bias=False,
    ).make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_normalizes(self):
    layer = normalization.BatchNormalization.Config(
        use_scale=False,
        use_bias=False,
    ).make()
    # Set known running stats.
    layer._ensure_initialized((1, 1, 4))
    layer._running_mean = mx.array([1.0, 2.0, 3.0, 4.0])
    layer._running_var = mx.array([1.0, 1.0, 1.0, 1.0])
    values = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = type(self.random_sequence(1, 1, 4))(values, mask)
    y = layer.layer(x, training=False)
    # (x - mean) / sqrt(var + eps) should be ~0
    np.testing.assert_allclose(y.values, np.zeros((1, 1, 4)), atol=1e-3)

  def test_scale_and_bias(self):
    layer = normalization.BatchNormalization.Config(epsilon=1e-3).make()
    layer._ensure_initialized((1, 1, 4))
    layer._running_mean = mx.zeros((4,))
    layer._running_var = mx.ones((4,))
    layer._scale = mx.array([2.0, 2.0, 2.0, 2.0])
    layer._bias = mx.array([1.0, 1.0, 1.0, 1.0])
    values = mx.array([[[1.0, 0.0, -1.0, 2.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = type(self.random_sequence(1, 1, 4))(values, mask)
    y = layer.layer(x, training=False)
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
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.BatchNormalization.Config()
    mlx_config = normalization.BatchNormalization.Config(
        axis=config.axis,
        epsilon=config.epsilon,
        use_scale=config.use_scale,
        use_bias=config.use_bias,
        name=config.name,
    )
    mlx_layer = mlx_config.make()
    self.assertIsInstance(mlx_layer, normalization.BatchNormalization)
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(mlx_layer, x)


class GroupNormalizationTest(
    test_utils.SequenceLayerTest,
    normalization_behaviors.GroupNormalizationTest,
):

  def test_layer(self):
    layer = normalization.GroupNormalization.Config(num_groups=2).make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_no_affine(self):
    layer = normalization.GroupNormalization.Config(
        num_groups=4,
        use_scale=False,
        use_bias=False,
    ).make()
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(layer, x)

  def test_num_groups_must_divide(self):
    layer = normalization.GroupNormalization.Config(num_groups=3).make()
    with self.assertRaises(ValueError):
      layer.layer(self.random_sequence(1, 2, 8), training=False)

  def test_from_config(self):
    from sequence_layers.jax import normalization as jax_norm

    config = jax_norm.GroupNormalization.Config(num_groups=2)
    mlx_config = normalization.GroupNormalization.Config(
        num_groups=config.num_groups,
        axis=config.axis,
        epsilon=config.epsilon,
        use_scale=config.use_scale,
        use_bias=config.use_bias,
        name=config.name,
    )
    mlx_layer = mlx_config.make()
    self.assertIsInstance(mlx_layer, normalization.GroupNormalization)
    x = self.random_sequence(2, 3, 8)
    self.verify_contract(mlx_layer, x)


if __name__ == '__main__':
  absltest.main()
