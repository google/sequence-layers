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
import inspect
import itertools
from typing import Any, Callable, Iterable, Mapping
from typing import Sequence as TypingSequence
from typing import TypeVar

from absl.testing import parameterized
import numpy as np

from sequence_layers import specs
from sequence_layers.specs import backend as backend_spec
from sequence_layers.specs import test_utils_spec
from sequence_layers.specs import types as types_spec
import typeguard

_T = TypeVar('_T')


class _AbcParameterizedTestCaseMeta(abc.ABCMeta, type(parameterized.TestCase)):
  """Metaclass for abstract parameterized test cases."""


def zip_longest(
    targets: Iterable[Iterable[Any]],
    sources: Iterable[_T],
) -> list[_T]:
  """Applies zip_longest, specialized to @parameterized's argument format.

  Args:
    targets: Iterable of parameterized test arguments.
    sources: Iterable of parameterized test arguments. If `targets` is a mapping
      `sources` must be a mapping as well.

  Returns:
    A list of the zipped arguments, of the type of `targets` and with each
    zipped argument internally sorted (target, source). If either input sequence
    was longer, the last element of the shorter input sequence is repeated.
  """
  results: list[Any] = []
  prev_source, prev_target = None, None
  for source, target in itertools.zip_longest(sources, targets):
    if source is None:
      source = prev_source
    if target is None:
      target = prev_target

    if isinstance(target, Mapping):
      assert isinstance(source, Mapping)
      results.append({**target, **source})
    elif isinstance(target, Iterable) and not isinstance(target, (str, bytes)):
      if isinstance(source, Mapping):
        raise ValueError('Cannot zip mapping source with non-mapping target')
      assert isinstance(source, Iterable)
      results.append(tuple(target) + tuple(source))
    else:
      results.append((target, source))

    prev_source, prev_target = source, target

  return results


_TestFnT = Callable[..., None]


def named_product(
    first: Iterable[TypingSequence[Any] | Mapping[str, Any]],
    second: Iterable[TypingSequence[Any] | Mapping[str, Any]],
) -> Callable[[_TestFnT], _TestFnT]:
  """Builds named parameters from the product of iterators of named parameters.

  As in parameterized.named_parameters, if an iterator's items are sequences,
  the first element is interpreted as the name. If an iterator's items are
  mappings, the `testcase_name` key is used.

  Args:
    first: Iterable of named parameters, whose names will be the first part of
      the named product's test names.
    second: Iterable of named parameters, whose names will be the second part of
      the named product's test names.

  Returns:
    A decorator that calls the test function with the cartesian product of the
    given iterators, whose items are named parameters with names of the form
    `{first_item_name}_{second_item_name}`. If both iterators' items are
    mappings, the product's items are mappings; otherwise they are ordered
    tuples.
  """
  results: list[Any] = []

  for p1, p2 in itertools.product(first, second):
    for source, parameters in enumerate([p1, p2]):
      if isinstance(parameters, Mapping):
        if 'testcase_name' not in parameters:
          raise ValueError(
              f'Mapping {parameters} from iterable #{source+1} does not have'
              ' key `testcase_name`.'
          )
      elif not parameters:
        raise ValueError(
            f'An sequence from iterable #{source+1} is empty; the first entry'
            ' is expected to be a testcase name.'
        )

    if isinstance(p1, Mapping) and isinstance(p2, Mapping):
      testcase_name = f'{p1["testcase_name"]}_{p2["testcase_name"]}'
      p1 = {k: v for k, v in p1.items() if k != 'testcase_name'}
      p2 = {k: v for k, v in p2.items() if k != 'testcase_name'}
      results.append({**p1, **p2, 'testcase_name': testcase_name})
    else:
      if isinstance(p1, Mapping):
        p1_name = p1['testcase_name']
        p1 = tuple(v for k, v in p1.items() if k != 'testcase_name')
      else:
        p1_name = p1[0]
        p1 = p1[1:]

      if isinstance(p2, Mapping):
        p2_name = p2['testcase_name']
        p2 = tuple(v for k, v in p2.items() if k != 'testcase_name')
      else:
        p2_name = p2[0]
        p2 = p2[1:]

      testcase_name = f'{p1_name}_{p2_name}'
      results.append((testcase_name, *p1, *p2))

  return parameterized.named_parameters(*results)


class SequenceLayerTest(
    parameterized.TestCase,
    metaclass=_AbcParameterizedTestCaseMeta,
):
  """Base test class providing common sequence testing assertions.

  Binds a backend implementation to tests.
  """

  # sequence_layers.<backend> module
  sl: specs.ModuleSpec

  @property
  def xp(self) -> backend_spec.xp:
    """Returns the backend wrapper."""
    return self.sl.backend.xp

  @property
  def nn(self) -> backend_spec.nn:
    """Returns the backend nn wrapper."""
    return self.sl.backend.nn

  def make_layer(self, config: types_spec.SequenceLayerConfig) -> Any:
    """Instantiates a layer from its config, delegating to the backend."""
    return config.make()

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

  def assertNotAllEqual(self, x: Any, y: Any) -> None:  # pylint: disable=invalid-name
    """Asserts that not all elements are equal."""
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    self.assertFalse(np.all(x_np == y_np))

  @abc.abstractmethod
  def get_variables(self, layer: types_spec.SequenceLayer) -> dict[str, Any]:
    """Returns the variables or parameters of the layer."""

  @abc.abstractmethod
  def random_sequence(
      self,
      *dims: int,
      dtype=None,
      random_mask: bool = False,
      random_lengths: bool | None = None,
      low: int | None = 0,
      high: int | None = 10,
      low_length: int = 0,
      high_length: int | None = None,
  ) -> types_spec.Sequence:
    """Generates a random sequence."""

  @abc.abstractmethod
  def init_layer(
      self,
      layer: types_spec.SequenceLayer,
      x: types_spec.Sequence,
      bind_only: bool = False,
      constants: types_spec.Constants | None = None,
  ) -> types_spec.SequenceLayer:
    """Initializes and binds a SequenceLayer for testing.

    Args:
      layer: Layer to initialize and bind.
      x: Example input sequence to use for initialization.
      bind_only: If True, skip initialization and only bind the layer (if
        applicable to the backend).
      constants: Optional constants for initialization.
    """

  @abc.abstractmethod
  def _step_by_step(
      self,
      layer: types_spec.SequenceLayer,
      x: types_spec.Sequence,
      *,
      block_size: int = 1,
      constants=None,
      stream_constants=None,
  ) -> tuple[types_spec.Sequence, Any]:
    """Runs a layer step by step."""

  @abc.abstractmethod
  def verify_contract(
      self,
      l: types_spec.SequenceLayer,
      x: types_spec.Sequence,
      *,
      training: bool = False,
      constants=None,
      stream_constants: bool = False,
      stream_constants_list: list[Any] | None = None,
      atol: float = 1e-5,
      rtol: float = 1e-5,
      **kwargs,
  ) -> types_spec.Sequence:
    """Verifies that a layer satisfies the contract."""

  @abc.abstractmethod
  def assertSequencesClose(self, x: Any, y: Any, **kwargs) -> None:  # pylint: disable=invalid-name
    """Asserts that two sequences are close."""

  def assertConfigDefaults(  # pylint: disable=invalid-name
      self, config_cls: type[Any], expected_defaults: dict[str, Any], **kwargs
  ) -> None:
    """Helper to verify that a config class has the expected defaults."""
    config = config_cls(**kwargs)
    for field_name, expected_val in expected_defaults.items():
      self.assertEqual(
          getattr(config, field_name),
          expected_val,
          f'Default for {field_name} in {config_cls.__name__} does not match!',
      )


class ModuleSpecTest(SequenceLayerTest):
  """Test that a backend-specific module implements the ModuleSpec protocol."""

  @abc.abstractmethod
  def module_spec_pairs(self, backend_sl: specs.ModuleSpec) -> dict[Any, Any]:
    """Returns a mapping of module to protocol to be verified."""

  def test_backend_specific_module_has_interface(self) -> None:
    pairs = self.module_spec_pairs(self.sl)
    for mod, protocol in pairs.items():
      self.assertIsInstance(mod, protocol)

  def test_module_spec_with_typeguard(self) -> None:
    pairs = self.module_spec_pairs(self.sl)
    sig = inspect.signature(typeguard.check_type)
    check_fn = getattr(typeguard, 'check_type')
    for mod, protocol in pairs.items():
      if 'argname' in sig.parameters:
        check_fn('mod', mod, protocol)
      else:
        check_fn(mod, protocol)  # pylint: disable=no-value-for-parameter


# Re-export the protocol and __all__ from the leaf module so that existing
# imports (e.g. ``from sequence_layers.specs import test_utils``) continue
# to resolve ``test_utils.ModuleSpec``.
ModuleSpec = test_utils_spec.ModuleSpec
__all__ = test_utils_spec.__all__
