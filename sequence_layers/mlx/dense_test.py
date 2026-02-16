"""Tests for Dense MLX sequence layer."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import dense
from sequence_layers.mlx import test_utils


class DenseTest(parameterized.TestCase):

  def test_layer(self):
    layer = dense.Dense(in_features=4, features=8)
    test_utils.verify_contract(self, layer, (4,))

  def test_output_shape(self):
    layer = dense.Dense(in_features=4, features=8)
    self.assertEqual(layer.get_output_shape((4,)), (8,))

  def test_no_bias(self):
    layer = dense.Dense(in_features=4, features=8, use_bias=False)
    test_utils.verify_contract(self, layer, (4,))

  @parameterized.named_parameters(
      ('relu', mx.array.__class__),
      ('none', None),
  )
  def test_activation(self, activation):
    import mlx.nn as nn

    act = nn.relu if activation is not None else None
    layer = dense.Dense(in_features=4, features=8, activation=act)
    test_utils.verify_contract(self, layer, (4,))


class DenseDeferredTest(parameterized.TestCase):

  def test_layer(self):
    layer = dense.DenseDeferred(features=8)
    test_utils.verify_contract(self, layer, (4,))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dense as jax_dense

    config = jax_dense.Dense.Config(features=16)
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dense.DenseDeferred)

    x = test_utils.random_sequence(2, 5, 8)
    y = mlx_layer.layer(x)
    self.assertEqual(y.channel_shape, (16,))


class EinsumDenseTest(parameterized.TestCase):

  def test_basic(self):
    layer = dense.EinsumDense(
        equation='...a,ab->...b',
        output_shape=(8,),
    )
    test_utils.verify_contract(self, layer, (4,))

  def test_output_shape(self):
    layer = dense.EinsumDense(
        equation='...a,ab->...b',
        output_shape=(8,),
    )
    self.assertEqual(layer.get_output_shape((4,)), (8,))

  def test_inferred_output(self):
    layer = dense.EinsumDense(
        equation='...ab,bc->...ac',
        output_shape=(None, 7),
    )
    self.assertEqual(layer.get_output_shape((3, 5)), (3, 7))

  def test_with_bias(self):
    layer = dense.EinsumDense(
        equation='...a,ab->...b',
        output_shape=(8,),
        bias_axes='b',
    )
    test_utils.verify_contract(self, layer, (4,))

  def test_multi_dim(self):
    layer = dense.EinsumDense(
        equation='...abc,bd->...bd',
        output_shape=(None, 6),
    )
    self.assertEqual(layer.get_output_shape((2, 3, 5)), (3, 6))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dense as jax_dense

    config = jax_dense.EinsumDense.Config(
        equation='...a,ab->...b',
        output_shape=(16,),
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dense.EinsumDense)

    x = test_utils.random_sequence(2, 5, 8)
    y = mlx_layer.layer(x)
    self.assertEqual(y.channel_shape, (16,))


if __name__ == '__main__':
  absltest.main()
