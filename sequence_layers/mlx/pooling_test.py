"""Tests for pooling MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import pooling
from sequence_layers.mlx import test_utils


class MaxPooling1DTest(parameterized.TestCase):

  @parameterized.parameters(
      ('causal',),
      ('semicausal',),
  )
  def test_causal_paddings(self, padding):
    layer = pooling.MaxPooling1D(pool_size=3, padding=padding)
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_valid(self):
    layer = pooling.MaxPooling1D(pool_size=3, padding='valid')
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (4,))
    self.assertEqual(y.shape[1], 6)  # 8 - 3 + 1

  def test_same(self):
    layer = pooling.MaxPooling1D(pool_size=3, padding='same')
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 8)

  def test_stride(self):
    layer = pooling.MaxPooling1D(
        pool_size=3,
        strides=2,
        padding='causal',
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        time=8,
        atol=1e-5,
        rtol=1e-5,
    )

  def test_dilation(self):
    layer = pooling.MaxPooling1D(
        pool_size=3,
        dilation_rate=2,
        padding='causal',
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_max_values(self):
    values = mx.array([[[1.0], [3.0], [2.0], [5.0], [4.0]]])
    mask = mx.ones((1, 5), dtype=mx.bool_)
    x = test_utils.random_sequence(1, 5, 1)
    x = type(x)(values, mask)
    layer = pooling.MaxPooling1D(pool_size=3, padding='valid')
    y = layer.layer(x)
    expected = np.array([[[3.0], [5.0], [5.0]]])
    np.testing.assert_allclose(y.values, expected)

  def test_pool_size_1(self):
    layer = pooling.MaxPooling1D(pool_size=1)
    test_utils.verify_contract(self, layer, (4,))

  def test_output_shape(self):
    layer = pooling.MaxPooling1D(pool_size=3, padding='causal')
    self.assertEqual(layer.get_output_shape((4,)), (4,))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import pooling as jax_pooling

    config = jax_pooling.MaxPooling1D.Config(
        pool_size=3,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, pooling.MaxPooling1D)
    test_utils.verify_contract(
        self,
        mlx_layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )


class MinPooling1DTest(parameterized.TestCase):

  @parameterized.parameters(
      ('causal',),
      ('semicausal',),
  )
  def test_causal_paddings(self, padding):
    layer = pooling.MinPooling1D(pool_size=3, padding=padding)
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_reverse_causal_layer(self):
    layer = pooling.MinPooling1D(
        pool_size=3,
        padding='reverse_causal',
    )
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 8)
    self.assertEqual(y.channel_shape, (4,))

  def test_valid(self):
    layer = pooling.MinPooling1D(pool_size=3, padding='valid')
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 6)

  def test_min_values(self):
    values = mx.array([[[5.0], [3.0], [4.0], [1.0], [2.0]]])
    mask = mx.ones((1, 5), dtype=mx.bool_)
    x = type(test_utils.random_sequence(1, 5, 1))(values, mask)
    layer = pooling.MinPooling1D(pool_size=3, padding='valid')
    y = layer.layer(x)
    expected = np.array([[[3.0], [1.0], [1.0]]])
    np.testing.assert_allclose(y.values, expected)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import pooling as jax_pooling

    config = jax_pooling.MinPooling1D.Config(
        pool_size=3,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, pooling.MinPooling1D)
    test_utils.verify_contract(
        self,
        mlx_layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )


class AveragePooling1DTest(parameterized.TestCase):

  @parameterized.parameters(
      ('causal',),
      ('semicausal',),
  )
  def test_causal_paddings(self, padding):
    layer = pooling.AveragePooling1D(pool_size=3, padding=padding)
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_valid(self):
    layer = pooling.AveragePooling1D(pool_size=3, padding='valid')
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 6)

  def test_average_values(self):
    values = mx.array([[[3.0], [6.0], [9.0], [12.0], [15.0]]])
    mask = mx.ones((1, 5), dtype=mx.bool_)
    x = type(test_utils.random_sequence(1, 5, 1))(values, mask)
    layer = pooling.AveragePooling1D(pool_size=3, padding='valid')
    y = layer.layer(x)
    expected = np.array([[[6.0], [9.0], [12.0]]])
    np.testing.assert_allclose(y.values, expected)

  def test_stride(self):
    layer = pooling.AveragePooling1D(
        pool_size=3,
        strides=2,
        padding='causal',
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        time=8,
        atol=1e-5,
        rtol=1e-5,
    )

  def test_masked_average(self):
    values = mx.array([[[3.0], [6.0], [0.0]]])
    mask = mx.array([[True, True, False]])
    x = type(test_utils.random_sequence(1, 3, 1))(values, mask)
    layer = pooling.AveragePooling1D(
        pool_size=3,
        padding='valid',
        masked_average=True,
    )
    y = layer.layer(x)
    # Only 2 valid elements: mean should be (3+6)/2 = 4.5
    expected = np.array([[[4.5]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-5)

  def test_masked_average_causal(self):
    layer = pooling.AveragePooling1D(
        pool_size=3,
        padding='causal',
        masked_average=True,
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import pooling as jax_pooling

    config = jax_pooling.AveragePooling1D.Config(
        pool_size=3,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, pooling.AveragePooling1D)
    test_utils.verify_contract(
        self,
        mlx_layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
