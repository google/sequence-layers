"""Tests for JAX backend utilities."""

from absl.testing import absltest

from sequence_layers.jax import test_utils
from sequence_layers.specs import backend_behaviors as spec


class ModuleSpecTest(test_utils.SequenceLayerTest, spec.ModuleSpecTest):
  pass


if __name__ == '__main__':
  absltest.main()
