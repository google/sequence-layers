import mlx.core as mx
import numpy as np

import sequence_layers.mlx as sl
from sequence_layers.specs import types_behaviors as spec
from absl.testing import parameterized
from absl.testing import absltest


class ModuleInterfaceTest(spec.ModuleInterfaceTest):
  sl = sl


class SequenceTest(spec.SequenceTest):
  sl = sl


class SequenceLayerConfigTest(spec.SequenceLayerConfigTest):
  sl = sl


class SteppableTest(spec.SteppableTest):
  sl = sl


class PreservesTypeTest(spec.PreservesTypeTest):
  sl = sl


class PreservesShapeTest(spec.PreservesShapeTest):
  sl = sl


class StatelessTest(spec.StatelessTest):
  sl = sl


class EmittingTest(spec.EmittingTest):
  sl = sl


class StatelessEmittingTest(spec.StatelessEmittingTest):
  sl = sl


class StatelessPointwiseFunctorTest(spec.StatelessPointwiseFunctorTest):
  sl = sl


if __name__ == '__main__':
  absltest.main()
