"""Tests for MLX backend utilities."""

from absl.testing import absltest

from sequence_layers.mlx import test_utils
from sequence_layers.specs import backend_behaviors as spec


class ModuleSpecTest(test_utils.SequenceLayerTest, spec.ModuleSpecTest):
  pass


class BackendNNTest(test_utils.SequenceLayerTest, spec.BackendNNTest):
  """Tests for MLX backend.nn operations."""


if __name__ == '__main__':
  absltest.main()
