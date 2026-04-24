"""Tests for simple MLX sequence layers."""

from typing import override

from absl.testing import absltest
import numpy as np

from sequence_layers.mlx import simple
from sequence_layers.mlx import test_utils
from sequence_layers.specs import simple_behaviors as spec


class ModuleSpecTest(test_utils.SequenceLayerTest, spec.ModuleSpecTest):
  pass


class IdentityTest(test_utils.SequenceLayerTest, spec.IdentityTest):

  def test_preserves_values(self):
    layer = simple.Identity.Config().make()
    x = self.random_sequence(2, 3, 4)
    y = layer.layer(x, training=False)
    np.testing.assert_array_equal(y.values, x.values)
    np.testing.assert_array_equal(y.mask, x.mask)


class PointwiseMathTest(test_utils.SequenceLayerTest, spec.PointwiseMathTest):

  @override
  def make_layer(self, layer_name):
    layer_cls = getattr(self.sl, layer_name)
    return layer_cls(layer_cls.Config())


class CastTest(test_utils.SequenceLayerTest, spec.CastTest):
  pass


class ScaleTest(test_utils.SequenceLayerTest, spec.ScaleTest):
  pass


class AddTest(test_utils.SequenceLayerTest, spec.AddTest):
  pass


class MaskInvalidTest(test_utils.SequenceLayerTest, spec.MaskInvalidTest):
  pass


class GatedUnitTest(test_utils.SequenceLayerTest, spec.GatedUnitTest):
  pass


class FlattenTest(test_utils.SequenceLayerTest, spec.FlattenTest):
  pass


class ReshapeTest(test_utils.SequenceLayerTest, spec.ReshapeTest):

  def test_mismatch_raises(self):
    layer = simple.Reshape.Config(output_shape=(5,)).make()

    with self.assertRaises(ValueError):
      layer.get_output_shape((12,))


class ExpandDimsTest(test_utils.SequenceLayerTest, spec.ExpandDimsTest):
  pass


class SqueezeTest(test_utils.SequenceLayerTest, spec.SqueezeTest):
  pass


class TransposeTest(test_utils.SequenceLayerTest, spec.TransposeTest):

  def test_reverse(self):
    layer = simple.Transpose.Config().make()
    self.assertEqual(layer.get_output_shape((2, 3, 4)), (4, 3, 2))

  def test_explicit(self):
    layer = simple.Transpose.Config(axes=(3, 2, 4)).make()

    self.assertEqual(layer.get_output_shape((5, 6, 7)), (6, 5, 7))


class OneHotTest(test_utils.SequenceLayerTest, spec.OneHotTest):
  pass


class EmbeddingTest(test_utils.SequenceLayerTest, spec.EmbeddingTest):
  pass


class DropoutTest(test_utils.SequenceLayerTest, spec.DropoutTest):
  pass


class Downsample1DTest(test_utils.SequenceLayerTest, spec.Downsample1DTest):
  pass


class Upsample1DTest(test_utils.SequenceLayerTest, spec.Upsample1DTest):
  pass


# class BackendDispatchTest(parameterized.TestCase):
#   """Test config.make(backend='mlx') for simple layers."""
#
#   def test_identity(self):
#     import sequence_layers.mlx  # Register backends.
#     from sequence_layers.jax import simple as jax_simple
#
#     config = jax_simple.Identity.Config()
#     mlx_layer = config.make(backend='mlx')
#     self.assertIsInstance(mlx_layer, simple.Identity)
#
#   def test_relu(self):
#     import sequence_layers.mlx
#     from sequence_layers.jax import simple as jax_simple
#
#     config = jax_simple.Relu.Config()
#     mlx_layer = config.make(backend='mlx')
#     self.assertIsInstance(mlx_layer, simple.Relu)
#
#   def test_tanh(self):
#     import sequence_layers.mlx
#     from sequence_layers.jax import simple as jax_simple
#
#     config = jax_simple.Tanh.Config()
#     mlx_layer = config.make(backend='mlx')
#     self.assertIsInstance(mlx_layer, simple.Tanh)
#
#   def test_gated_linear_unit(self):
#     import sequence_layers.mlx
#     from sequence_layers.jax import simple as jax_simple
#
#     config = jax_simple.GatedLinearUnit.Config()
#     mlx_layer = config.make(backend='mlx')
#     self.assertIsInstance(mlx_layer, simple.GatedLinearUnit)
#
#   def test_reshape(self):
#     import sequence_layers.mlx
#     from sequence_layers.jax import simple as jax_simple
#
#     config = jax_simple.Reshape.Config(output_shape=(2, 3))
#     mlx_layer = config.make(backend='mlx')
#     self.assertIsInstance(mlx_layer, simple.Reshape)
#
#   def test_downsample(self):
#     import sequence_layers.mlx
#     from sequence_layers.jax import simple as jax_simple
#
#     config = jax_simple.Downsample1D.Config(rate=2)
#     mlx_layer = config.make(backend='mlx')
#     self.assertIsInstance(mlx_layer, simple.Downsample1D)


class CheckpointNameTest(test_utils.SequenceLayerTest, spec.CheckpointNameTest):

  def test_layer(self):
    layer = simple.CheckpointName.Config(checkpoint_name='test').make()

    x = self.random_sequence(2, 3, 4)
    self.verify_contract(layer, x)

  def test_passthrough(self):
    layer = simple.CheckpointName.Config(checkpoint_name='test').make()

    x = self.random_sequence(1, 3, 4)
    y = layer.layer(x, training=False)
    np.testing.assert_array_equal(y.values, x.values)
    np.testing.assert_array_equal(y.mask, x.mask)

  #   def test_from_config(self):


#     import sequence_layers.mlx
#     from sequence_layers.jax import simple as jax_simple
#
#     config = jax_simple.CheckpointName.Config(checkpoint_name='test')
#     mlx_layer = config.make(backend='mlx')
#     self.assertIsInstance(mlx_layer, simple.CheckpointName)


class LambdaTest(test_utils.SequenceLayerTest, spec.LambdaTest):
  """Test behavior of Lambda layer."""


class LoggingTest(test_utils.SequenceLayerTest, spec.LoggingTest):
  """Test behavior of Logging layer."""


if __name__ == '__main__':
  absltest.main()
