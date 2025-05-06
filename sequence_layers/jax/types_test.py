# Copyright 2024 Google LLC
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
"""Types test."""

import dataclasses
import typing
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from sequence_layers.jax import simple
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt

from google3.testing.pybase import parameterized


class Foo(nn.Module):

  @nn.compact
  def __call__(self, x: types.Sequence) -> types.Sequence:
    return x


class SequenceTest(test_utils.SequenceLayerTest):
  """Tests for the Sequence class."""

  def test_type_checks(self):
    """Test type checks in Sequence.__post_init__."""

    # Allowed: Both array-like.
    types.Sequence(jnp.zeros((2, 3, 5)), jnp.zeros((2, 3), dtype=jnp.bool_))
    types.Sequence(np.zeros((2, 3, 5)), np.zeros((2, 3), dtype=jnp.bool_))

    # Allowed: Both ShapeDType.
    types.Sequence(
        types.ShapeDType([], jnp.float32), types.ShapeDType([], jnp.bool_)
    )

    # Allowed: nn.summary._ArrayRepresentation passthrough.
    Foo().module_paths(
        jax.random.key(0),
        types.Sequence(jnp.zeros((2, 3, 5)), jnp.zeros((2, 3), dtype=bool)),
    )
    nn.tabulate(Foo(), jax.random.key(0))(
        types.Sequence(jnp.zeros((2, 3, 5)), jnp.zeros((2, 3), dtype=bool))
    )

    # Disallowed: Only one ShapeDType.
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(jnp.zeros((2, 3, 5)), types.ShapeDType([], jnp.bool_))
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(
          types.ShapeDType([], jnp.float32), jnp.zeros((2, 3), dtype=jnp.bool_)
      )

    # Disallowed: Values less than rank 2.
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(jnp.zeros((2, 3, 5)), jnp.zeros((3,), dtype=jnp.bool_))
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(
          jnp.zeros((2,)),
          jnp.zeros(
              (
                  2,
                  3,
              ),
              dtype=jnp.bool_,
          ),
      )

    # Disallowed: Mask less than rank 2.
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(jnp.zeros((3, 5)), jnp.zeros((3,), dtype=jnp.bool_))

    # Disallowed: Mask shape is not a prefix of values shape.
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(jnp.zeros((2, 4, 5)), jnp.zeros((2, 3), dtype=jnp.bool_))
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(np.zeros((2, 3, 5)), np.zeros((2, 4), dtype=jnp.bool_))
    with self.assertRaises(jaxtyping.TypeCheckError):
      types.Sequence(np.zeros((2, 3, 5)), np.zeros((1, 3), dtype=jnp.bool_))

  @parameterized.parameters(None, 0.0, -1.0)
  def test_mask_invalid(self, mask_value: float | None):
    values = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = jnp.array([[True, True, False, False], [False, False, False, True]])

    output = types.Sequence(values, mask).mask_invalid(mask_value)
    mask_value = 0.0 if mask_value is None else mask_value

    chex.assert_trees_all_equal(
        output.values,
        jnp.array([
            [1.0, 2.0, mask_value, mask_value],
            [mask_value, mask_value, mask_value, 40.0],
        ]),
    )
    chex.assert_trees_all_equal(output.mask, mask)

  def test_slice(self):
    x = test_utils.random_sequence(3, 5, 9)

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
      _ = typing.cast(Any, x)[0, :]

    with self.assertRaises(ValueError):
      _ = typing.cast(Any, x)[:, 0]

  def test_slice_can_slice_channel_dimensions(self):
    x = test_utils.random_sequence(3, 5, 9, 4)

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
        x[:, :, jnp.newaxis],
        types.MaskedSequence(jnp.expand_dims(x.values, 2), x.mask),
    )

    self.assertSequencesEqual(
        x[:, :, ..., 0],
        types.MaskedSequence(x.values[:, :, :, 0], x.mask),
    )

  def test_mask_invalid_idempotent(self):
    values = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = jnp.array([[True, True, False, False], [False, False, False, True]])

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
    values = jnp.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = jnp.array([[True, True, False, False], [False, True, False, True]])

    x = types.Sequence(values, mask)
    masked = x.mask_invalid()

    # x stays unmasked if we use apply_values.
    y = x.apply_values(jax.nn.relu)
    self.assertNotIsInstance(y, types.MaskedSequence)
    chex.assert_trees_all_equal(y.values, jax.nn.relu(x.values))
    chex.assert_trees_all_equal(y.mask, x.mask)

    # x does not become masked if we use apply_values_masked.
    y = x.apply_values_masked(jax.nn.relu)
    self.assertNotIsInstance(y, types.MaskedSequence)
    chex.assert_trees_all_equal(y.values, jax.nn.relu(x.values))
    chex.assert_trees_all_equal(y.mask, x.mask)

    # masked loses its masked state if we use apply_values.
    y = masked.apply_values(jax.nn.relu)
    self.assertNotIsInstance(y, types.MaskedSequence)
    chex.assert_trees_all_equal(y.values, jax.nn.relu(masked.values))
    chex.assert_trees_all_equal(y.mask, x.mask)

    # masked keeps its masked state if we use apply_values_masked.
    y = masked.apply_values_masked(jax.nn.relu)
    self.assertIsInstance(y, types.MaskedSequence)
    chex.assert_trees_all_equal(y.values, jax.nn.relu(masked.values))
    chex.assert_trees_all_equal(y.mask, x.mask)

  def test_apply_values_args(self):
    values = jnp.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = jnp.array([[True, True, False, False], [False, True, False, True]])

    x = types.Sequence(values, mask)
    y = x.apply_values(jnp.reshape, [2, 4, 1])
    self.assertEqual(y.values.shape, (2, 4, 1))
    self.assertEqual(y.mask.shape, (2, 4))
    y = x.apply_values_masked(jnp.reshape, [2, 4, 1])
    self.assertEqual(y.values.shape, (2, 4, 1))
    self.assertEqual(y.mask.shape, (2, 4))

  def test_pad_time(self):
    x = types.Sequence(
        jnp.array([
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]),
        jnp.array([[True, True, False, False], [False, False, False, True]]),
    ).mask_invalid()

    y = x.pad_time(0, 0, valid=False)
    chex.assert_trees_all_equal(y.values, x.values)
    chex.assert_trees_all_equal(y.mask, x.mask)

    y = x.pad_time(1, 0, valid=False)

    x_left1 = types.Sequence(
        jnp.array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
        ]),
        jnp.array([
            [False, True, True, False, False],
            [False, False, False, False, True],
        ]),
    ).mask_invalid()
    chex.assert_trees_all_equal(y.values, x_left1.values)
    chex.assert_trees_all_equal(y.mask, x_left1.mask)

  def test_type_annotation(self):
    @jt.typed
    def f(
        x: types.SequenceT[jt.Float, 'B T C'],
    ) -> types.SequenceT[jt.Float, 'B T C 1']:
      return types.Sequence(x.values[..., jnp.newaxis], x.mask)

    values = jnp.zeros((2, 3, 5))
    mask = jnp.zeros((2, 3), dtype=jnp.bool_)
    x = types.Sequence(values, mask)

    # Works with Sequence.
    f(x)

    with self.assertRaises(jaxtyping.TypeCheckError):
      x = types.Sequence(
          jnp.zeros((2, 3, 5)), jnp.zeros((3, 3), dtype=jnp.bool_)
      )
      f(x)

    with self.assertRaises(jaxtyping.TypeCheckError):
      x = types.Sequence(
          jnp.zeros((2, 3, 5)).astype(jnp.int32),
          jnp.zeros((2, 3), dtype=jnp.bool_),
      )
      f(x)

    with self.assertRaises(jaxtyping.TypeCheckError):
      x = types.Sequence(jnp.zeros((2, 3)), jnp.zeros((2, 3), dtype=jnp.bool_))
      f(x)

  def test_type_annotation_masked_sequence(self):
    @jt.typed
    def f(
        x: types.SequenceT[jt.Float, 'B T C'],
    ) -> types.SequenceT[jt.Float, 'B T C 1']:
      return types.MaskedSequence(x.values[..., jnp.newaxis], x.mask)

    values = jnp.zeros((2, 3, 5))
    mask = jnp.zeros((2, 3), dtype=jnp.bool_)
    x = types.MaskedSequence(values, mask)

    f(x)

    with self.assertRaises(jaxtyping.TypeCheckError):
      x = types.MaskedSequence(
          jnp.zeros((2, 3, 5)), jnp.zeros((3, 3), dtype=jnp.bool_)
      )
      f(x)

    with self.assertRaises(jaxtyping.TypeCheckError):
      x = types.MaskedSequence(
          jnp.zeros((2, 3, 5)).astype(jnp.int32),
          jnp.zeros((2, 3), dtype=jnp.bool_),
      )
      f(x)

    with self.assertRaises(jaxtyping.TypeCheckError):
      x = types.MaskedSequence(
          jnp.zeros((2, 3)), jnp.zeros((2, 3), dtype=jnp.bool_)
      )
      f(x)

  def test_pytree(self):
    x = types.MaskedSequence(
        jnp.zeros((2, 3, 5)), jnp.zeros((2, 3), dtype=jnp.bool_)
    )
    y = types.MaskedSequence(
        jnp.ones((2, 3, 5)), jnp.ones((2, 3), dtype=jnp.bool_)
    )

    tree = {'foo': {'bar': x}, 'baz': x}
    expected = {'foo': {'bar': y}, 'baz': y}
    tree = jax.tree_util.tree_map(
        lambda x: (x.astype(jnp.float32) + 1.0).astype(x.dtype), tree
    )
    chex.assert_trees_all_equal(tree, expected)

  def test_jax_jit_compatible(self):
    @jax.jit
    def fn(x: types.Sequence) -> types.Sequence:
      return x

    x = types.Sequence(jnp.zeros((2, 3, 5)), jnp.zeros((2, 3), dtype=jnp.bool_))
    y = fn(x)
    self.assertSequencesEqual(y, x)

  def test_from_lengths(self):
    x_expected = test_utils.random_sequence(5, 17, 2)
    x = types.Sequence.from_lengths(x_expected.values, x_expected.lengths())
    self.assertSequencesEqual(x, x_expected)

    # Out of range lengths are clipped to 0 or max.
    x = types.Sequence.from_lengths(x_expected.values, [-1, 0, 5, 17, 18])
    self.assertAllEqual(x.lengths(), jnp.asarray([0, 0, 5, 17, 17]))
    self.assertNotIsInstance(x, types.MaskedSequence)

    # Return type is MaskedSequence if is_masked=True.
    x = types.Sequence.from_lengths(
        x_expected.values, [-1, 0, 5, 17, 18], is_masked=True
    )
    self.assertAllEqual(x.lengths(), jnp.asarray([0, 0, 5, 17, 17]))
    self.assertIsInstance(x, types.MaskedSequence)

  def test_from_values(self):
    x_expected = test_utils.random_sequence(5, 17, 2).values
    x = types.Sequence.from_values(x_expected)

    # Mask is all ones.
    self.assertAllEqual(x.mask, jnp.ones_like(x.mask))

    # Result is a MaskedSequence.
    self.assertIsInstance(x, types.MaskedSequence)

    # The values are unchanged by from_values.
    self.assertAllEqual(x.values, x_expected)

  def test_astype(self):
    x_float = jnp.asarray([[1.0, 2.9, 3.14]])
    x_float_mask = jnp.asarray([[True, True, True]], dtype=types.MASK_DTYPE)
    x_expected = jnp.asarray([[1, 2, 3]], dtype=jnp.int8)
    x = types.Sequence(x_float, x_float_mask).astype(jnp.int8)
    with self.subTest('values_are_casted'):
      self.assertAllEqual(x.values, x_expected)
    with self.subTest('values_dtype_is_set'):
      self.assertEqual(x.values.dtype, x_expected.dtype)
    with self.subTest('mask_unchanged'):
      self.assertAllEqual(x.mask, x_float_mask)
      self.assertEqual(x.mask.dtype, x_float_mask.dtype)


class SequenceLayerConfigTest(test_utils.SequenceLayerTest):

  def test_copy(self):

    @dataclasses.dataclass(frozen=True)
    class Config(types.SequenceLayerConfig):
      a: int = 1234
      b: str = 'default string'

      def make(self) -> simple.Identity:
        return simple.Identity.Config().make()

    config = Config()
    new_config = config.copy(b='new string')
    self.assertEqual(new_config.a, config.a)
    self.assertEqual(new_config.b, 'new string')

  def test_copy_raises_on_non_dataclass(self):

    class NonDataclassConfig(types.SequenceLayerConfig):

      def make(self) -> simple.Identity:
        return simple.Identity.Config().make()

    config = NonDataclassConfig()
    with self.assertRaises(TypeError):
      new_config = config.copy()
      del new_config

  def test_copy_raises_on_mutable_attribute(self):

    @dataclasses.dataclass(slots=True)
    class ConfigWithSequence(types.SequenceLayerConfig):
      seq: typing.Sequence[int]

      def make(self) -> simple.Identity:
        return simple.Identity.Config().make()

    config = ConfigWithSequence(seq=[1, 2, 3])
    with self.assertRaises(TypeError):
      new_config = config.copy()
      del new_config
    config = ConfigWithSequence(seq=(1, 2, 3))
    new_config = config.copy()
    del new_config

  def test_copy_raises_on_unfrozen_child_config(self):

    @dataclasses.dataclass(slots=True)
    class UnfrozenConfig(types.SequenceLayerConfig):
      a: int = 1234

      def make(self) -> simple.Identity:
        return simple.Identity.Config().make()

    @dataclasses.dataclass(frozen=True)
    class ConfigWithChild(types.SequenceLayerConfig):
      child: UnfrozenConfig

      def make(self) -> simple.Identity:
        return simple.Identity.Config().make()

    config = ConfigWithChild(child=UnfrozenConfig())
    with self.assertRaises(TypeError):
      new_config = config.copy()
      del new_config

  def test_copy_disallows_new_fields(self):

    @dataclasses.dataclass(frozen=True)
    class Config(types.SequenceLayerConfig):

      def make(self) -> simple.Identity:
        return simple.Identity.Config().make()

    config = Config()
    with self.assertRaises(AttributeError):
      new_config = config.copy(field_does_not_exist=1234)
      del new_config


if __name__ == '__main__':
  test_utils.main()
