# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for MLX sequence types."""

from absl.testing import absltest
from sequence_layers.mlx import test_utils
from sequence_layers.specs import types_behaviors as spec


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
