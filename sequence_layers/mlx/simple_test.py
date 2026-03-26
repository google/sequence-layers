"""Tests for simple MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import simple
from sequence_layers.mlx import test_utils


class IdentityTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Identity()
    test_utils.verify_contract(self, layer, (4,))

  def test_preserves_values(self):
    layer = simple.Identity()
    x = test_utils.random_sequence(2, 3, 4)
    y = layer.layer(x)
    np.testing.assert_array_equal(y.values, x.values)
    np.testing.assert_array_equal(y.mask, x.mask)


class ReluTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Relu()
    test_utils.verify_contract(self, layer, (4,))

  def test_negative_zeroed(self):
    layer = simple.Relu()
    values = mx.array([[-1.0, 0.5, -0.3, 2.0]]).reshape(1, 1, 4)
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    expected = mx.array([[[0.0, 0.5, 0.0, 2.0]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-6)


class GeluTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Gelu()
    test_utils.verify_contract(self, layer, (4,))


class SwishTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Swish()
    test_utils.verify_contract(self, layer, (4,))


class TanhTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Tanh()
    test_utils.verify_contract(self, layer, (4,))

  def test_values(self):
    layer = simple.Tanh()
    values = mx.array([[[0.0, 1.0, -1.0, 100.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    np.testing.assert_allclose(
        y.values, np.tanh([[[0.0, 1.0, -1.0, 100.0]]]), atol=1e-5
    )


class SigmoidTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Sigmoid()
    test_utils.verify_contract(self, layer, (4,))


class LeakyReluTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.LeakyRelu(negative_slope=0.2)
    test_utils.verify_contract(self, layer, (4,))

  def test_negative_slope(self):
    layer = simple.LeakyRelu(negative_slope=0.1)
    values = mx.array([[[-2.0, 0.5, -1.0, 3.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    expected = mx.array([[[-0.2, 0.5, -0.1, 3.0]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-6)


class EluTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Elu()
    test_utils.verify_contract(self, layer, (4,))


class SoftmaxTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Softmax()
    test_utils.verify_contract(self, layer, (4,))

  def test_sums_to_one(self):
    layer = simple.Softmax(axis=-1)
    values = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    np.testing.assert_allclose(float(mx.sum(y.values)), 1.0, atol=1e-5)


class SoftplusTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Softplus()
    test_utils.verify_contract(self, layer, (4,))


class CastTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Cast(dtype=mx.float16)
    test_utils.verify_contract(self, layer, (4,), atol=1e-3, rtol=1e-3)

  def test_cast(self):
    layer = simple.Cast(dtype=mx.float16)
    x = test_utils.random_sequence(1, 3, 4)
    y = layer.layer(x)
    self.assertEqual(y.dtype, mx.float16)


class ScaleTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Scale(scale=2.0)
    test_utils.verify_contract(self, layer, (4,))

  def test_scalar(self):
    layer = simple.Scale(scale=2.0)
    values = mx.array([[[1.0, 2.0, 3.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    expected = mx.array([[[2.0, 4.0, 6.0]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-6)


class AddTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Add(shift=1.0)
    test_utils.verify_contract(self, layer, (4,))

  def test_scalar(self):
    layer = simple.Add(shift=10.0)
    values = mx.array([[[1.0, 2.0, 3.0]]])
    mask = mx.ones((1, 1), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    expected = mx.array([[[11.0, 12.0, 13.0]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-6)


class MaskInvalidTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.MaskInvalid()
    test_utils.verify_contract(self, layer, (4,))

  def test_masks_to_zero(self):
    layer = simple.MaskInvalid()
    values = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = mx.array([[True, False, True]])
    x = bt.Sequence(values, mask)
    y = layer.layer(x)
    expected = mx.array([[[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]])
    np.testing.assert_allclose(y.values, expected, atol=1e-6)


class GatedUnitTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.GatedUnit()
    test_utils.verify_contract(self, layer, (8,))

  def test_with_activations(self):
    import mlx.nn as nn

    layer = simple.GatedUnit(
        feature_activation=nn.relu, gate_activation=nn.sigmoid
    )
    test_utils.verify_contract(self, layer, (8,))


class GatedLinearUnitTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.GatedLinearUnit()
    test_utils.verify_contract(self, layer, (8,))

  def test_halves_channels(self):
    layer = simple.GatedLinearUnit()
    self.assertEqual(layer.get_output_shape((8,)), (4,))


class GatedTanhUnitTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.GatedTanhUnit()
    test_utils.verify_contract(self, layer, (8,))


class FlattenTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Flatten()
    test_utils.verify_contract(self, layer, (2, 3, 4))

  def test_flatten(self):
    layer = simple.Flatten()
    self.assertEqual(layer.get_output_shape((2, 3, 4)), (24,))


class ReshapeTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Reshape(output_shape=(2, 6))
    test_utils.verify_contract(self, layer, (12,))

  def test_reshape(self):
    layer = simple.Reshape(output_shape=(2, 6))
    x = test_utils.random_sequence(1, 3, 12)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (2, 6))

  def test_mismatch_raises(self):
    layer = simple.Reshape(output_shape=(5,))
    with self.assertRaises(ValueError):
      layer.get_output_shape((12,))


class ExpandDimsTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.ExpandDims(axis=-1)
    test_utils.verify_contract(self, layer, (4,))

  def test_expand(self):
    layer = simple.ExpandDims(axis=0)
    self.assertEqual(layer.get_output_shape((4, 8)), (1, 4, 8))

  def test_layer_values(self):
    layer = simple.ExpandDims(axis=-1)
    x = test_utils.random_sequence(1, 3, 4)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (4, 1))


class SqueezeTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Squeeze()
    test_utils.verify_contract(self, layer, (4, 1))

  def test_squeeze(self):
    layer = simple.Squeeze()
    x = bt.MaskedSequence(
        mx.ones((1, 3, 1, 4, 1)),
        mx.ones((1, 3), dtype=mx.bool_),
    )
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (4,))


class TransposeTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Transpose()
    test_utils.verify_contract(self, layer, (2, 3, 4))

  def test_reverse(self):
    layer = simple.Transpose()
    self.assertEqual(layer.get_output_shape((2, 3, 4)), (4, 3, 2))

  def test_explicit(self):
    layer = simple.Transpose(axes=(3, 2, 4))
    self.assertEqual(layer.get_output_shape((5, 6, 7)), (6, 5, 7))


class OneHotTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.OneHot(depth=5)
    x = bt.MaskedSequence(
        mx.array([[0, 2, 4]]),
        mx.ones((1, 3), dtype=mx.bool_),
    )
    y = layer.layer(x)
    self.assertEqual(y.shape, (1, 3, 5))
    # Check that index 0 -> [1,0,0,0,0]
    np.testing.assert_allclose(np.array(y.values[0, 0]), [1, 0, 0, 0, 0])


class EmbeddingTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Embedding(num_embeddings=10, dimension=8)
    x = bt.MaskedSequence(
        mx.array([[1, 3, 5]]),
        mx.ones((1, 3), dtype=mx.bool_),
    )
    y = layer.layer(x)
    self.assertEqual(y.shape, (1, 3, 8))

  def test_output_shape(self):
    layer = simple.Embedding(num_embeddings=10, dimension=8)
    self.assertEqual(layer.get_output_shape(()), (8,))
    self.assertEqual(layer.get_output_shape((3,)), (3, 8))


class DropoutTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Dropout(rate=0.5)
    test_utils.verify_contract(self, layer, (4,))

  def test_passthrough(self):
    layer = simple.Dropout(rate=0.5)
    x = test_utils.random_sequence(1, 3, 4)
    y = layer.layer(x)
    # Inference-only: should be identity.
    np.testing.assert_array_equal(y.values, x.values)


class Downsample1DTest(parameterized.TestCase):

  def test_verify_contract(self):
    layer = simple.Downsample1D(rate=2)
    test_utils.verify_contract(self, layer, (4,))

  def test_layer(self):
    layer = simple.Downsample1D(rate=2)
    x = test_utils.random_sequence(1, 6, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape, (1, 3, 4))

  def test_values(self):
    layer = simple.Downsample1D(rate=3)
    values = mx.arange(12).reshape(1, 6, 2).astype(mx.float32)
    mask = mx.ones((1, 6), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    # Should keep timesteps 0, 3.
    np.testing.assert_array_equal(y.values, values[:, ::3])


class Upsample1DTest(parameterized.TestCase):

  def test_verify_contract(self):
    layer = simple.Upsample1D(rate=3)
    test_utils.verify_contract(self, layer, (4,))

  def test_layer(self):
    layer = simple.Upsample1D(rate=3)
    x = test_utils.random_sequence(1, 4, 2)
    y = layer.layer(x)
    self.assertEqual(y.shape, (1, 12, 2))

  def test_values(self):
    layer = simple.Upsample1D(rate=2)
    values = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = mx.ones((1, 2), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    expected = mx.array([[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]])
    np.testing.assert_allclose(y.values, expected)
    self.assertEqual(y.mask.shape, (1, 4))


class BackendDispatchTest(parameterized.TestCase):
  """Test config.make(backend='mlx') for simple layers."""

  def test_identity(self):
    import sequence_layers.mlx  # Register backends.
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Identity.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.Identity)

  def test_relu(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Relu.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.Relu)

  def test_tanh(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Tanh.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.Tanh)

  def test_gated_linear_unit(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.GatedLinearUnit.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.GatedLinearUnit)

  def test_reshape(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Reshape.Config(output_shape=(2, 3))
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.Reshape)

  def test_downsample(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Downsample1D.Config(rate=2)
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.Downsample1D)


class CheckpointNameTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.CheckpointName(checkpoint_name='test')
    test_utils.verify_contract(self, layer, (4,))

  def test_passthrough(self):
    layer = simple.CheckpointName(checkpoint_name='test')
    x = test_utils.random_sequence(1, 3, 4)
    y = layer.layer(x)
    np.testing.assert_array_equal(y.values, x.values)
    np.testing.assert_array_equal(y.mask, x.mask)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.CheckpointName.Config(checkpoint_name='test')
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.CheckpointName)


class LambdaTest(parameterized.TestCase):

  def test_values_fn(self):
    layer = simple.Lambda(fn=lambda v: v * 2.0)
    x = test_utils.random_sequence(1, 3, 4)
    y = layer.layer(x)
    np.testing.assert_allclose(y.values, x.values * 2.0, atol=1e-6)

  def test_sequence_fn(self):
    def double_seq(s):
      return bt.Sequence(s.values * 2.0, s.mask)

    layer = simple.Lambda(fn=double_seq, sequence_input=True)
    x = test_utils.random_sequence(1, 3, 4)
    y = layer.layer(x)
    np.testing.assert_allclose(y.values, x.values * 2.0, atol=1e-6)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Lambda.Config(fn=lambda v: v)
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.Lambda)


class LoggingTest(parameterized.TestCase):

  def test_layer(self):
    layer = simple.Logging(prefix='test')
    test_utils.verify_contract(self, layer, (4,))

  def test_passthrough(self):
    layer = simple.Logging()
    x = test_utils.random_sequence(1, 3, 4)
    y = layer.layer(x)
    np.testing.assert_array_equal(y.values, x.values)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Logging.Config(prefix='test')
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, simple.Logging)


if __name__ == '__main__':
  absltest.main()
