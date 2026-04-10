"""Tests for the test utilities."""

from sequence_layers.mlx import test_utils
from sequence_layers.specs import test_utils_behaviors as spec


class ModuleSpecTest(test_utils.SequenceLayerTest, spec.ModuleSpecTest):
  pass


class NamedProductTest(test_utils.SequenceLayerTest, spec.NamedProductTest):
  pass


class ZipLongestTest(test_utils.SequenceLayerTest, spec.ZipLongestTest):
  pass


class VerifyContractTest(test_utils.SequenceLayerTest, spec.VerifyContractTest):
  pass


if __name__ == '__main__':
  test_utils.main()
