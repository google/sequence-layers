from sequence_layers.mlx import test_utils
from sequence_layers.specs import types_behaviors as spec
from absl.testing import absltest


class ModuleInterfaceTest(
    test_utils.SequenceLayerTest, spec.ModuleInterfaceTest
):
  pass


class SequenceTest(test_utils.SequenceLayerTest, spec.SequenceTest):
  pass


class SequenceLayerConfigTest(
    test_utils.SequenceLayerTest, spec.SequenceLayerConfigTest
):
  pass


class SteppableTest(test_utils.SequenceLayerTest, spec.SteppableTest):
  pass


class PreservesTypeTest(test_utils.SequenceLayerTest, spec.PreservesTypeTest):
  pass


class PreservesShapeTest(test_utils.SequenceLayerTest, spec.PreservesShapeTest):
  pass


class StatelessTest(test_utils.SequenceLayerTest, spec.StatelessTest):
  pass


class EmittingTest(test_utils.SequenceLayerTest, spec.EmittingTest):
  pass


class StatelessEmittingTest(
    test_utils.SequenceLayerTest, spec.StatelessEmittingTest
):
  pass


class StatelessPointwiseFunctorTest(
    test_utils.SequenceLayerTest, spec.StatelessPointwiseFunctorTest
):
  pass


if __name__ == '__main__':
  absltest.main()
