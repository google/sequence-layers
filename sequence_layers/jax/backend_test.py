"""Tests for JAX backend utilities."""

from absl.testing import absltest

from sequence_layers.jax import test_utils
from sequence_layers.specs import backend_behaviors as spec


class ModuleSpecTest(test_utils.SequenceLayerTest, spec.ModuleSpecTest):
  pass


class BackendNNTest(test_utils.SequenceLayerTest, spec.BackendNNTest):
  """Tests for JAX backend.nn operations."""


class BackendXPTest(test_utils.SequenceLayerTest, spec.BackendXPTest):
  """Tests for JAX backend.xp operations."""


if __name__ == '__main__':
  absltest.main()
