"""Test utilities for sequence layers."""

import abc
from typing import Any

from absl.testing import parameterized

from sequence_layers import specs
from sequence_layers.specs import backend as backend_spec
from sequence_layers.specs import types as spec


class _AbcParameterizedTestCaseMeta(abc.ABCMeta, type(parameterized.TestCase)):
  pass


class SequenceLayerTest[SequenceT: spec.Sequence = spec.Sequence](
    parameterized.TestCase, metaclass=_AbcParameterizedTestCaseMeta
):
  """Base test class providing common sequence testing assertions.

  Binds a backend implementation to tests.
  """

  # sequence_layers.<backend> module
  sl: specs.ModuleSpec

  @property
  def xp(self) -> backend_spec.xp:
    """Returns the backend module."""
    return self.sl.backend.xp

  @abc.abstractmethod
  def assertSequencesEqual(self, x: SequenceT, y: SequenceT) -> None:  # pylint: disable=invalid-name
    """After padding, checks sequence values are equal and masks are equal."""

  @abc.abstractmethod
  def assertAllEqual(self, x: Any, y: Any) -> None:  # pylint: disable=invalid-name
    """Asserts that two arrays are equal."""
