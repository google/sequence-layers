"""Test utilities for MLX sequence layers."""

from typing import override
import numpy as np
import mlx.core as mx
import sequence_layers.mlx as sl
from sequence_layers.mlx import types
from sequence_layers.specs import test_utils as spec


def _mask_and_pad_to_max_length(
    a: types.Sequence, b: types.Sequence
) -> tuple[types.Sequence, types.Sequence]:
  """Masks and pads two sequences to the same max length."""
  # Only compare values in non-masked regions.
  a = a.mask_invalid()
  b = b.mask_invalid()
  a_time = a.values.shape[1]
  b_time = b.values.shape[1]
  max_time = max(a_time, b_time)
  a = a.pad_time(0, max_time - a_time, valid=False)
  b = b.pad_time(0, max_time - b_time, valid=False)
  return a, b


class SequenceLayerTest(spec.SequenceLayerTest):
  """Base class for MLX SequenceLayer tests."""

  sl = sl

  @override
  def assertAllEqual(self, x, y):
    """Asserts that two arrays are equal."""
    x_np = np.array(x) if isinstance(x, mx.array) else x
    y_np = np.array(y) if isinstance(y, mx.array) else y
    np.testing.assert_array_equal(x_np, y_np)

  @override
  def assertSequencesEqual(self, x: types.Sequence, y: types.Sequence):
    """After padding, checks sequence values are equal and masks are equal."""
    x, y = _mask_and_pad_to_max_length(x, y)
    self.assertAllEqual(x.values, y.values)
    self.assertAllEqual(x.mask, y.mask)
