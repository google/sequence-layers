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
"""Utilities for testing sequence layers."""

import abc
from typing import Any

from absl.testing import parameterized
from sequence_layers import specs
from sequence_layers.specs import backend as backend_spec
from sequence_layers.specs import test_utils_spec
from sequence_layers.specs import types as types_spec


class _AbcParameterizedTestCaseMeta(abc.ABCMeta, type(parameterized.TestCase)):
  """Metaclass for abstract parameterized test cases."""


class SequenceLayerTest(
    parameterized.TestCase,
    metaclass=_AbcParameterizedTestCaseMeta,
):
  """Base test class providing common sequence testing assertions."""

  sl: specs.ModuleSpec

  @property
  def xp(self) -> backend_spec.xp:
    """Returns the backend wrapper."""
    return self.sl.backend.xp

  # pylint: disable=invalid-name

  @abc.abstractmethod
  def assertSequencesEqual(
      self, x: types_spec.Sequence, y: types_spec.Sequence
  ) -> None:
    """Asserts that two sequences are equal."""

  @abc.abstractmethod
  def assertAllEqual(self, x: Any, y: Any) -> None:
    """Asserts that all elements are equal."""

  # pylint: enable=invalid-name


class ModuleSpecTest(SequenceLayerTest):
  """Test that a backend-specific module implements the ModuleSpec protocol."""

  @abc.abstractmethod
  def module_spec_pairs(self, backend_sl: specs.ModuleSpec) -> dict[Any, Any]:
    """Returns a mapping of module to protocol to be verified."""

  def test_backend_specific_module_has_interface(self) -> None:
    pairs = self.module_spec_pairs(self.sl)
    for mod, protocol in pairs.items():
      self.assertIsInstance(mod, protocol)


ModuleSpec = test_utils_spec.ModuleSpec
__all__ = test_utils_spec.__all__
