"""Tests for Dense MLX sequence layers."""

from absl.testing import absltest
from mlx import nn

from sequence_layers.mlx import dense
from sequence_layers.mlx import test_utils
from sequence_layers.specs import dense_behaviors as spec


class DenseTest(test_utils.SequenceLayerTest, spec.DenseTest):
  """Test behavior of Dense layer."""

  def test_eager_layer(self):
    """Test DenseEager which is not in the spec."""
    layer = dense._DenseEager(in_features=4, features=8)
    x = self.random_sequence(2, 3, 4)
    self.verify_contract(layer, x)

  def test_activation(self):
    """Test activation in Dense."""
    # Test with deferred Dense
    layer = dense.Dense(features=8, activation=nn.relu)
    x = self.random_sequence(2, 3, 4)
    self.verify_contract(layer, x)


class EinsumDenseTest(test_utils.SequenceLayerTest, spec.EinsumDenseTest):
  """Test behavior of EinsumDense layer."""


if __name__ == '__main__':
  absltest.main()
