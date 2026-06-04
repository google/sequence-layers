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
"""Specification for sequence_layers backend implementations.

https://typing.python.org/en/latest/spec/protocol.html#modules-as-implementations-of-protocols
"""

from typing import Any, Protocol, runtime_checkable

from sequence_layers.specs import test_utils_spec as _test_utils_spec
from sequence_layers.specs import types as _types


@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for a backend-specific SequenceLayers module (sequence_layers.<backend> as sl)."""

  # pylint: disable=missing-function-docstring

  @property
  def types(self) -> _types.ModuleSpec:
    ...

  # pylint: disable=invalid-name

  # Identifiers that backend-specific implementations should expose at top
  # level. Demonstrating read-only allows for covariance (subclasses of
  # types_module.Sequence to satisfy the protocol).

  @property
  def Sequence(self) -> type[_types.Sequence]:
    ...

  @property
  def MaskedSequence(self) -> type[_types.MaskedSequence]:
    ...

  @property
  def SequenceLayer(self) -> type[_types.SequenceLayer]:
    ...

  @property
  def SequenceLayerConfig(self) -> type[_types.SequenceLayerConfig]:
    ...

