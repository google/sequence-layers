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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sequence_layers.numpy import types


def random_sequence(
    *dims: int,
    dtype=np.float32,
    random_mask: bool = False,
    random_lengths: bool | None = None,
    low: int | None = 0,
    high: int | None = 10,
    low_length: int = 0,
    high_length: int | None = None,
) -> types.Sequence:
  """Generates a random Sequence with dims dimension."""
  if random_lengths is None:
    random_lengths = not random_mask
  if random_mask and random_lengths:
    raise ValueError('Must not specify random_mask and random_lengths.')
  if len(dims) < 2:
    raise ValueError(
        'random_sequence expects at least 2 dimensions, got: %s' % (dims,)
    )

  is_complex = dtype in (np.complex64, np.complex128)
  is_integer = np.issubdtype(dtype, np.integer)

  if is_complex:
    np_values = np.random.normal(size=dims) + 1j * np.random.normal(size=dims)
  elif is_integer:
    np_values = np.random.randint(low, high, size=dims)
  else:
    np_values = np.random.normal(size=dims)

  batch_size, time = dims[0], dims[1]
  values = np_values.astype(dtype)
  if random_mask:
    mask = np.random.uniform(size=(batch_size, time)) > 0.5
  else:
    if time > 0:
      if random_lengths:
        if high_length is None:
          high_length = time + 1
        lengths = np.random.randint(
            low_length, high_length, size=[batch_size]
        ).astype(np.int32)
      else:
        lengths = np.full([batch_size], time).astype(np.int32)
    else:
      lengths = np.full([batch_size], 0).astype(np.int32)

    mask = np.arange(time)[np.newaxis, :] < lengths[:, np.newaxis]

  return types.Sequence(values, mask).mask_invalid()


def _mask_and_pad_to_max_length(
    a: types.Sequence, b: types.Sequence
) -> tuple[types.Sequence, types.Sequence]:
  a = a.mask_invalid()
  b = b.mask_invalid()
  a_time = a.values.shape[1]
  b_time = b.values.shape[1]
  max_time = max(a_time, b_time)
  a = a.pad_time(0, max_time - a_time, valid=False)
  b = b.pad_time(0, max_time - b_time, valid=False)
  return a, b


class SequenceTest(parameterized.TestCase):
  """Tests for the Sequence class."""

  def assertSequencesEqual(self, a: types.Sequence, b: types.Sequence):
    a, b = _mask_and_pad_to_max_length(a, b)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.mask, b.mask)

  def assertAllEqual(self, a, b):
    np.testing.assert_array_equal(a, b)

  def test_type_checks(self):
    """Test type checks in Sequence.__post_init__."""

    # Allowed: Both array-like.
    types.Sequence(np.zeros((2, 3, 5)), np.zeros((2, 3), dtype=np.bool_))

    # Disallowed: Values less than rank 2.
    with self.assertRaises(ValueError):
      types.Sequence(np.zeros((2,)), np.zeros((2, 3), dtype=np.bool_))

    # Disallowed: Mask less than rank 2.
    with self.assertRaises(ValueError):
      types.Sequence(np.zeros((3, 5)), np.zeros((3,), dtype=np.bool_))

    # Disallowed: Mask shape is not a prefix of values shape.
    with self.assertRaises(ValueError):
      types.Sequence(np.zeros((2, 4, 5)), np.zeros((2, 3), dtype=np.bool_))

  @parameterized.parameters(None, 0.0, -1.0)
  def test_mask_invalid(self, mask_value: float | None):
    values = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = np.array([[True, True, False, False], [False, False, False, True]])

    output = types.Sequence(values, mask).mask_invalid(mask_value)
    mask_value = 0.0 if mask_value is None else mask_value

    np.testing.assert_array_equal(
        output.values,
        np.array([
            [1.0, 2.0, mask_value, mask_value],
            [mask_value, mask_value, mask_value, 40.0],
        ]),
    )
    np.testing.assert_array_equal(output.mask, mask)

  def test_slice(self):
    x = random_sequence(3, 5, 9)

    self.assertSequencesEqual(
        x[:, 1:], types.MaskedSequence(x.values[:, 1:], x.mask[:, 1:])
    )

    self.assertSequencesEqual(
        x[:, ::2], types.MaskedSequence(x.values[:, ::2], x.mask[:, ::2])
    )

    self.assertSequencesEqual(
        x[::2, ::3], types.MaskedSequence(x.values[::2, ::3], x.mask[::2, ::3])
    )

    self.assertSequencesEqual(
        x[1::2, 2::3],
        types.MaskedSequence(x.values[1::2, 2::3], x.mask[1::2, 2::3]),
    )

    self.assertSequencesEqual(
        x[1::2],
        types.MaskedSequence(x.values[1::2, :], x.mask[1::2, :]),
    )

    with self.assertRaises(ValueError):
      _ = x[0, :]

    with self.assertRaises(ValueError):
      _ = x[:, 0]

  def test_slice_can_slice_channel_dimensions(self):
    x = random_sequence(3, 5, 9, 4)

    self.assertSequencesEqual(
        x[:, 1:, :], types.MaskedSequence(x.values[:, 1:], x.mask[:, 1:])
    )

    self.assertSequencesEqual(
        x[:, ::2, :3],
        types.MaskedSequence(x.values[:, ::2, :3], x.mask[:, ::2]),
    )

    self.assertSequencesEqual(
        x[:, :, ::2], types.MaskedSequence(x.values[:, :, ::2], x.mask)
    )

    self.assertSequencesEqual(
        x[:, :, ::2, :1], types.MaskedSequence(x.values[:, :, ::2, :1], x.mask)
    )

    self.assertSequencesEqual(
        x[:, :, 0, ::2], types.MaskedSequence(x.values[:, :, 0, ::2], x.mask)
    )

    self.assertSequencesEqual(
        x[:, :, np.newaxis],
        types.MaskedSequence(np.expand_dims(x.values, 2), x.mask),
    )

    self.assertSequencesEqual(
        x[:, :, ..., 0],
        types.MaskedSequence(x.values[:, :, :, 0], x.mask),
    )

  def test_mask_invalid_idempotent(self):
    values = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = np.array([[True, True, False, False], [False, False, False, True]])

    x = types.Sequence(values, mask)
    masked = x.mask_invalid()
    self.assertIsNot(masked, x)
    self.assertIsInstance(masked, types.MaskedSequence)

    masked_again = masked.mask_invalid()
    self.assertIs(masked_again, masked)
    self.assertIsInstance(masked_again, types.MaskedSequence)

    masked2 = x.mask_invalid()
    self.assertIsNot(masked2, masked)
    self.assertIsInstance(masked2, types.MaskedSequence)

  def test_apply_values(self):
    values = np.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = np.array([[True, True, False, False], [False, True, False, True]])

    x = types.Sequence(values, mask)
    masked = x.mask_invalid()

    relu = lambda x: np.maximum(0, x)

    # x stays unmasked if we use apply_values.
    y = x.apply_values(relu)
    self.assertNotIsInstance(y, types.MaskedSequence)
    np.testing.assert_array_equal(y.values, relu(x.values))
    np.testing.assert_array_equal(y.mask, x.mask)

    # x does not become masked if we use apply_values_masked.
    y = x.apply_values_masked(relu)
    self.assertNotIsInstance(y, types.MaskedSequence)
    np.testing.assert_array_equal(y.values, relu(x.values))
    np.testing.assert_array_equal(y.mask, x.mask)

    # masked loses its masked state if we use apply_values.
    y = masked.apply_values(relu)
    self.assertNotIsInstance(y, types.MaskedSequence)
    np.testing.assert_array_equal(y.values, relu(masked.values))
    np.testing.assert_array_equal(y.mask, x.mask)

    # masked keeps its masked state if we use apply_values_masked.
    y = masked.apply_values_masked(relu)
    self.assertIsInstance(y, types.MaskedSequence)
    np.testing.assert_array_equal(y.values, relu(masked.values))
    np.testing.assert_array_equal(y.mask, x.mask)

  def test_apply_values_args(self):
    values = np.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = np.array([[True, True, False, False], [False, True, False, True]])

    x = types.Sequence(values, mask)
    y = x.apply_values(np.reshape, (2, 4, 1))
    self.assertEqual(y.values.shape, (2, 4, 1))
    self.assertEqual(y.mask.shape, (2, 4))
    y = x.apply_values_masked(np.reshape, (2, 4, 1))
    self.assertEqual(y.values.shape, (2, 4, 1))
    self.assertEqual(y.mask.shape, (2, 4))

  def test_pad_time(self):
    x = types.Sequence(
        np.array([
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]),
        np.array([[True, True, False, False], [False, False, False, True]]),
    ).mask_invalid()

    y = x.pad_time(0, 0, valid=False)
    np.testing.assert_array_equal(y.values, x.values)
    np.testing.assert_array_equal(y.mask, x.mask)

    y = x.pad_time(1, 0, valid=False)

    x_left1 = types.Sequence(
        np.array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
        ]),
        np.array([
            [False, True, True, False, False],
            [False, False, False, False, True],
        ]),
    ).mask_invalid()
    np.testing.assert_array_equal(y.values, x_left1.values)
    np.testing.assert_array_equal(y.mask, x_left1.mask)

  def test_from_lengths(self):
    x_expected = random_sequence(5, 17, 2)
    x = types.Sequence.from_lengths(x_expected.values, x_expected.lengths())
    self.assertSequencesEqual(x, x_expected)

    # Out of range lengths are clipped to 0 or max.
    x = types.Sequence.from_lengths(
        x_expected.values, np.array([-1, 0, 5, 17, 18])
    )
    self.assertAllEqual(x.lengths(), np.array([0, 0, 5, 17, 17]))
    self.assertNotIsInstance(x, types.MaskedSequence)

    # Return type is MaskedSequence if is_masked=True.
    x = types.Sequence.from_lengths(
        x_expected.values, np.array([-1, 0, 5, 17, 18]), is_masked=True
    )
    self.assertAllEqual(x.lengths(), np.array([0, 0, 5, 17, 17]))
    self.assertIsInstance(x, types.MaskedSequence)

  def test_from_values(self):
    x_expected = random_sequence(5, 17, 2).values
    x = types.Sequence.from_values(x_expected)

    # Mask is all ones.
    self.assertAllEqual(x.mask, np.ones_like(x.mask))

    # Result is a MaskedSequence.
    self.assertIsInstance(x, types.MaskedSequence)

    # The values are unchanged by from_values.
    self.assertAllEqual(x.values, x_expected)

  def test_astype(self):
    x_float = np.array([[1.0, 2.9, 3.14]])
    x_float_mask = np.array([[True, True, True]], dtype=np.bool_)
    x_expected = np.array([[1, 2, 3]], dtype=np.int8)
    x = types.Sequence(x_float, x_float_mask).astype(np.int8)
    with self.subTest('values_are_casted'):
      self.assertAllEqual(x.values, x_expected)
    with self.subTest('values_dtype_is_set'):
      self.assertEqual(x.values.dtype, x_expected.dtype)
    with self.subTest('mask_unchanged'):
      self.assertAllEqual(x.mask, x_float_mask)
      self.assertEqual(x.mask.dtype, x_float_mask.dtype)


if __name__ == '__main__':
  absltest.main()
