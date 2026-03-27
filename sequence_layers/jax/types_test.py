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


class SequenceTest(test_utils.SequenceLayerTest, types_test_base.SequenceTest):
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

      def layer(self, x, *, training: bool, constants=None):
        return x

      def step(self, x, state, *, training: bool, constants=None):
        return x, state

      def get_initial_state(self, batch_size, input_spec, *, constants=None):
        return 0

      def get_output_shape(self, input_shape, *, constants=None):
        return input_shape

      def get_output_dtype(self, input_dtype, *, constants=None):
        return input_dtype

    return DefaultSteppable()


class PreservesTypeTest(types_test_base.PreservesTypeTest):
  def create_layer(self):
    class DummyLayer(types.PreservesType, types.SequenceLayer):
      def layer(self, x, *, training: bool, constants=None): return x
      def step(self, x, state, *, training: bool, constants=None): return x, state
      def get_initial_state(self, batch_size, input_spec, *, training: bool, constants=None): return ()
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
    return DummyLayer()


class PreservesShapeTest(types_test_base.PreservesShapeTest):
  def create_layer(self):
    class DummyLayer(types.PreservesShape, types.SequenceLayer):
      def layer(self, x, *, training: bool, constants=None): return x
      def step(self, x, state, *, training: bool, constants=None): return x, state
      def get_initial_state(self, batch_size, input_spec, *, training: bool, constants=None): return ()
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()


class StatelessTest(types_test_base.StatelessTest):
  def create_layer(self):
    class DummyLayer(types.Stateless, types.SequenceLayer):
      def layer(self, x, *, training: bool, constants=None): return x
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()


class EmittingTest(types_test_base.EmittingTest):
  def create_layer(self):
    class DummyLayer(types.Emitting, types.SequenceLayer):
      def get_initial_state(self, batch_size, input_spec, *, training: bool, constants=None): return ()
      def layer_with_emits(self, x, *, training: bool, constants=None): return x, ()
      def step_with_emits(self, x, state, *, training: bool, constants=None): return x, state, ()
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
      @property
      def receptive_field_per_step(self): return {0: (0, 0)}
    return DummyLayer()


class StatelessEmittingTest(types_test_base.StatelessEmittingTest):
  def create_layer(self):
    class DummyLayer(types.StatelessEmitting, types.SequenceLayer):
      def layer_with_emits(self, x, *, training: bool, constants=None): return x, ()
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()


class StatelessPointwiseFunctorTest(types_test_base.StatelessPointwiseFunctorTest):
  def create_layer(self, is_mask_required: bool):
    class DummyLayer(types.StatelessPointwiseFunctor, types.SequenceLayer):
      @property
      def mask_required(self): return is_mask_required
      def fn(self, values, mask): return values, mask
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()

  def create_sequence(self):
    return types.Sequence(jnp.zeros((2, 3, 5)), jnp.zeros((2, 3), dtype=jnp.bool_))

if __name__ == '__main__':
  test_utils.main()
