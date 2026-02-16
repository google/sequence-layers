"""Tests for convolution MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import convolution
from sequence_layers.mlx import test_utils


class Conv1DTest(parameterized.TestCase):

  @parameterized.parameters(
      ('causal',),
      ('causal_valid',),
  )
  def test_causal_paddings(self, padding):
    layer = convolution.Conv1D(
        in_features=4,
        filters=8,
        kernel_size=3,
        padding=padding,
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_valid(self):
    layer = convolution.Conv1D(
        in_features=4,
        filters=8,
        kernel_size=3,
        padding='valid',
    )
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (8,))
    # Valid: output time = input_time - kernel_size + 1 = 6
    self.assertEqual(y.shape[1], 6)

  def test_same(self):
    layer = convolution.Conv1D(
        in_features=4,
        filters=8,
        kernel_size=3,
        padding='same',
    )
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 8)

  def test_stride(self):
    layer = convolution.Conv1D(
        in_features=4,
        filters=8,
        kernel_size=3,
        strides=2,
        padding='causal',
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        time=8,
        atol=1e-4,
        rtol=1e-4,
    )

  def test_dilation(self):
    layer = convolution.Conv1D(
        in_features=4,
        filters=8,
        kernel_size=3,
        dilation_rate=2,
        padding='causal',
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_output_shape(self):
    layer = convolution.Conv1D(
        in_features=4,
        filters=16,
        kernel_size=3,
        padding='causal',
    )
    self.assertEqual(layer.get_output_shape((4,)), (16,))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import convolution as jax_conv

    config = jax_conv.Conv1D.Config(
        filters=8,
        kernel_size=3,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        convolution.DeferredConv1D,
    )
    x = test_utils.random_sequence(1, 8, 4)
    y = mlx_layer.layer(x)
    self.assertEqual(y.channel_shape, (8,))


class DepthwiseConv1DTest(parameterized.TestCase):

  @parameterized.parameters(
      ('causal',),
      ('causal_valid',),
  )
  def test_causal_paddings(self, padding):
    layer = convolution.DepthwiseConv1D(
        in_features=4,
        kernel_size=3,
        padding=padding,
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_depth_multiplier(self):
    layer = convolution.DepthwiseConv1D(
        in_features=4,
        kernel_size=3,
        depth_multiplier=2,
        padding='causal',
    )
    self.assertEqual(layer.get_output_shape((4,)), (8,))

  def test_valid(self):
    layer = convolution.DepthwiseConv1D(
        in_features=4,
        kernel_size=3,
        padding='valid',
    )
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 6)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import convolution as jax_conv

    config = jax_conv.DepthwiseConv1D.Config(
        kernel_size=3,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        convolution.DeferredDepthwiseConv1D,
    )
    x = test_utils.random_sequence(1, 8, 4)
    y = mlx_layer.layer(x)
    self.assertEqual(y.channel_shape, (4,))


class Conv1DTransposeTest(parameterized.TestCase):

  def test_causal(self):
    layer = convolution.Conv1DTranspose(
        in_features=4,
        filters=8,
        kernel_size=3,
        strides=2,
        padding='causal',
    )
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_valid(self):
    layer = convolution.Conv1DTranspose(
        in_features=4,
        filters=8,
        kernel_size=3,
        strides=2,
        padding='valid',
    )
    x = test_utils.random_sequence(1, 4, 4)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (8,))
    # Valid: output = input * stride + max(ek - stride, 0)
    expected_time = 4 * 2 + max(3 - 2, 0)
    self.assertEqual(y.shape[1], expected_time)

  def test_same(self):
    layer = convolution.Conv1DTranspose(
        in_features=4,
        filters=8,
        kernel_size=3,
        strides=2,
        padding='same',
    )
    x = test_utils.random_sequence(1, 4, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 8)

  def test_output_ratio(self):
    layer = convolution.Conv1DTranspose(
        in_features=4,
        filters=8,
        kernel_size=3,
        strides=3,
        padding='causal',
    )
    import fractions

    self.assertEqual(layer.output_ratio, fractions.Fraction(3))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import convolution as jax_conv

    config = jax_conv.Conv1DTranspose.Config(
        filters=8,
        kernel_size=3,
        strides=2,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        convolution.DeferredConv1DTranspose,
    )
    x = test_utils.random_sequence(1, 4, 4)
    y = mlx_layer.layer(x)
    self.assertEqual(y.channel_shape, (8,))


if __name__ == '__main__':
  absltest.main()
