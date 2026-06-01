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
"""Protocol specification for test_utils modules.

This is in a separate file to break import cycle between test_utils.py and
specs/__init__.py.
"""

from typing import Any, Callable, Iterable, Protocol, runtime_checkable


@runtime_checkable
class ModuleSpec(Protocol):
  """Specification for sequence_layers.<backend>.test_utils."""

  def zip_longest(
      self,
      targets: Iterable[Iterable[Any]],
      sources: Iterable[Any],
  ) -> list[Any]:
    """Zips targets and sources."""

  def named_product(
      self,
      first: Iterable[Any],
      second: Iterable[Any],
  ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Creates a named product."""

  @property
  def SequenceLayerTest(self) -> type[Any]:  # pylint: disable=invalid-name,missing-function-docstring
    ...


__all__ = [
    name
    for name, attr in ModuleSpec.__dict__.items()
    if isinstance(attr, property)
    or (callable(attr) and not name.startswith('__'))
]
