"""Tests for Dense MLX sequence layers."""

from absl.testing import absltest
from mlx import nn

from sequence_layers.mlx import dense
from sequence_layers.mlx import test_utils
from sequence_layers.specs import dense_behaviors as spec


class DenseTest(test_utils.SequenceLayerTest, spec.DenseTest):
  """Test behavior of Dense layer."""

  def test_activation(self):
    """Test activation in Dense."""
    layer = dense.Dense.Config(features=8, activation=nn.relu).make()
    x = self.random_sequence(2, 3, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x)


class EinsumDenseTest(test_utils.SequenceLayerTest, spec.EinsumDenseTest):
  """Test behavior of EinsumDense layer."""


if __name__ == '__main__':
  absltest.main()
