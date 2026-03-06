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

from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np

from sequence_layers.abstract import types_test_base
from sequence_layers.jax import simple
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt


class Foo(nn.Module):

  @nn.compact
  def __call__(self, x: types.Sequence) -> types.Sequence:
    return x


class SequenceTest(types_test_base.SequenceTest, test_utils.SequenceLayerTest):
  """Tests for the Sequence class."""

  def get_backend(self):
    return jnp

  @property
  def Sequence(self):
    return types.Sequence

  @property
  def MaskedSequence(self):
    return types.MaskedSequence

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

  def test_type_annotation(self):
    if not jt.runtime_type_checking_enabled:
      self.skipTest('Type checking is disabled.')

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
    if not jt.runtime_type_checking_enabled:
      self.skipTest('Type checking is disabled.')

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


class SequenceLayerConfigTest(types_test_base.SequenceLayerConfigTest):

  def get_config_base_cls(self):
    return types.SequenceLayerConfig

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


class SteppableTest(types_test_base.SteppableTest):

  def create_steppable(self):

    class DefaultSteppable(types.Steppable):

      def layer(self, x, *, constants=None):
        return x

      def step(self, x, state, *, constants=None):
        return x, state

      def get_initial_state(self, batch_size, input_spec, *, constants=None):
        return 0

      def get_output_shape(self, input_shape, *, constants=None):
        return input_shape

      def get_output_dtype(self, input_dtype, *, constants=None):
        return input_dtype

    return DefaultSteppable()


if __name__ == '__main__':
  test_utils.main()
