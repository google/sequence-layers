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
"""Simple tests."""

import dataclasses
import functools
import itertools
from unittest import mock

from absl import logging
from absl.testing import parameterized
import chex
import einops
import flax
import flax.linen as nn
import jax
import jax._src.ad_checkpoint
import jax.experimental.mesh_utils  # Required for OSS.
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import sharding as sharding_lib
from sequence_layers.jax import simple
from sequence_layers.jax import test_utils
from sequence_layers.jax import types

from google3.testing.pymocks import matchers


class ScaleTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 5, 9),))
  def test_basic(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Scale.Config(scale=2.0, name='scale').make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'scale')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(lambda v: v * 2.0)
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 9, 5),))
  def test_ndarray(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Scale.Config(
        scale=np.arange(5, dtype=np.float32), name='scale'
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'scale')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(lambda v: v * np.arange(5, dtype=np.float32))
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Scale.Config(scale=np.ones((5, 9))).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))

  def test_too_many_dims(self):
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Scale.Config(scale=np.ones((5, 5, 5))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = test_utils.random_sequence(2, 3, 5, 9)
    l = simple.Scale.Config(scale=np.ones((5,))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class AddTest(test_utils.SequenceLayerTest):

  @parameterized.parameters((((2, 13, 5)),), (((2, 13, 5, 9)),))
  def test_add(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Add.Config(-2.0, name='add').make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'add')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(lambda v: v - 2.0).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 9, 5),))
  def test_ndarray(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Add.Config(
        shift=np.arange(5, dtype=np.float32), name='add'
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'add')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(
        lambda v: v + np.arange(5, dtype=np.float32)
    ).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Add.Config(shift=np.ones((5, 9))).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))

  def test_too_many_dims(self):
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Add.Config(shift=np.ones((5, 5, 5))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = test_utils.random_sequence(2, 3, 5, 9)
    l = simple.Add.Config(shift=np.ones((5,))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class MinimumTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 5, 9),))
  def test_basic(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Minimum.Config(minimum=0.0, name='minimum').make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'minimum')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(lambda v: jnp.minimum(v, 0.0)).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 9, 5),))
  def test_ndarray(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Minimum.Config(
        minimum=np.arange(5, dtype=np.float32), name='minimum'
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'minimum')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(
        lambda v: jnp.minimum(v, np.arange(5, dtype=np.float32))
    ).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Minimum.Config(minimum=np.ones((5, 9))).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))

  def test_too_many_dims(self):
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Minimum.Config(minimum=np.ones((5, 5, 5))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = test_utils.random_sequence(2, 3, 5, 9)
    l = simple.Minimum.Config(minimum=np.ones((5,))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class MaximumTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 5, 9),))
  def test_basic(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Maximum.Config(maximum=0.0, name='maximum').make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'maximum')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(lambda v: jnp.maximum(v, 0.0)).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 9, 5),))
  def test_ndarray(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Maximum.Config(
        maximum=np.arange(5, dtype=np.float32), name='maximum'
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'maximum')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(
        lambda v: jnp.maximum(v, np.arange(5, dtype=np.float32))
    ).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Maximum.Config(maximum=np.ones((5, 9))).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))

  def test_too_many_dims(self):
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Maximum.Config(maximum=np.ones((5, 5, 5))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = test_utils.random_sequence(2, 3, 5, 9)
    l = simple.Maximum.Config(maximum=np.ones((5,))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class ModTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 5, 9),))
  def test_basic(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Mod.Config(divisor=5, name='mod').make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'mod')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(lambda v: jnp.mod(v, 5)).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 9, 5),))
  def test_ndarray(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Mod.Config(
        divisor=np.arange(1, 6, dtype=np.float32), name='mod'
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'mod')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    y_expected = x.apply_values(
        lambda v: jnp.mod(v, np.arange(1, 6, dtype=np.float32))
    ).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Mod.Config(divisor=np.ones((5, 9))).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))

  def test_too_many_dims(self):
    x = test_utils.random_sequence(2, 3, 5, 1)
    l = simple.Mod.Config(divisor=np.ones((5, 5, 5))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = test_utils.random_sequence(2, 3, 5, 9)
    l = simple.Mod.Config(divisor=np.ones((5,))).make().bind({})
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class GatedUnitTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      itertools.product(
          (simple.GatedUnit.Config(None, None),  # Bilinear
           simple.GatedUnit.Config(None, jax.nn.swish),  # SwiGLU
           simple.GatedUnit.Config(None, jax.nn.gelu),  # GeGLU
           simple.GatedUnit.Config(lambda x: x, None),  # Bilinear
           simple.GatedUnit.Config(jax.nn.swish, jax.nn.tanh),
           simple.GatedTanhUnit.Config(),
           simple.GatedLinearUnit.Config()),
          ((2, 13, 6), (2, 13, 5, 10)))
      )  # pyformat: disable
  def test_gated_activation(self, layer_config, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = layer_config.make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + (shape[-1] // 2,)
    )
    self.verify_contract(l, x, training=True)
    self.assertEmpty(l.variables)


class DropoutTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      jnp.float32, jnp.bfloat16, jnp.int32, jnp.int8, jnp.bool
  )
  def test_basic(self, dtype):
    key = jax.random.PRNGKey(1234)
    l = simple.Dropout.Config(
        rate=0.1, rng_collection='foo', name='dropout'
    ).make()
    x = test_utils.random_sequence(2, 13, 5, dtype=dtype, random_mask=True)
    l = self.init_and_bind_layer(key, l, x)
    y = self.verify_contract(l, x, training=False)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dropout')
    self.assertEqual(l.get_output_dtype(x.dtype), dtype)
    self.assertEqual(y.dtype, dtype)
    self.assertSequencesEqual(y, x)
    y = l.apply({}, x, training=True, rngs={'foo': jax.random.key(1)})
    self.assertEqual(y.dtype, dtype)
    self.assertSequencesNotEqual(y, x)

  def test_rate_zero(self):
    l = simple.Dropout.Config(
        rate=0.0, rng_collection='foo', name='dropout'
    ).make()
    x = test_utils.random_sequence(2, 13, 5, random_mask=True)
    y = l.apply({}, x, training=True, rngs={'foo': jax.random.key(1)})
    self.assertSequencesEqual(y, x)

  def test_rate_one(self):
    l = simple.Dropout.Config(
        rate=1.0, rng_collection='foo', name='dropout'
    ).make()
    x = test_utils.random_sequence(2, 13, 5, random_mask=True)
    y = l.apply({}, x, training=True, rngs={'foo': jax.random.key(1)})
    self.assertAllEqual(y.values, jnp.zeros_like(x.values))

  def test_broadcast_dims(self):
    l = simple.Dropout.Config(
        rate=0.5, broadcast_dims=(0,), rng_collection='foo', name='dropout'
    ).make()
    x = types.Sequence(jnp.ones((2, 13, 5)), jnp.ones((2, 13), dtype=jnp.bool_))
    y = l.apply({}, x, training=True, rngs={'foo': jax.random.key(1)})

    # Variance should be zero if same dropout mask is applied to all batch
    # items.
    chex.assert_trees_all_equal(
        jnp.var(y.values, axis=0), jnp.zeros_like(y.values[0])
    )

  def test_integer_input(self):
    l = simple.Dropout.Config(
        rate=0.5, broadcast_dims=(0,), rng_collection='foo', name='dropout'
    ).make()
    x = types.Sequence.from_values(jnp.ones((2, 13, 5), jnp.int32))
    y = l.apply({}, x, training=True, rngs={'foo': jax.random.key(1)})
    self.assertTrue(jnp.all(jnp.logical_or(y.values == 1, y.values == 0)))


class SliceTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 3, 5),), ((2, 3, 5, 9),))
  def test_slice(self, shape):
    x = test_utils.random_sequence(*shape)
    l = (
        simple.Slice.Config(
            ((None, -1, None),) * (len(shape) - 2), name='slice'
        )
        .make()
        .bind({})
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'slice')
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        tuple(dim - 1 for dim in shape[2:]),
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

  def test_slice_newxis(self):
    x = test_utils.random_sequence(1, 2, 3, 4)
    l = (
        simple.Slice.Config(
            ((None, None, None), jnp.newaxis, (None, None, None))
        )
        .make()
        .bind({})
    )
    self.assertEqual(l.get_output_shape_for_sequence(x), (3, 1, 4))
    y = l.layer(x, training=True)
    self.assertEqual(y.values.shape, (1, 2, 3, 1, 4))
    self.assertAllEqual(y.values, x.values[:, :, :, jnp.newaxis, :])

    l = (
        simple.Slice.Config((jnp.newaxis, 0, jnp.newaxis, (1, 3, 1)))
        .make()
        .bind({})
    )
    self.assertEqual(l.get_output_shape_for_sequence(x), (1, 1, 2))
    y = l.layer(x, training=True)
    self.assertEqual(y.values.shape, (1, 2, 1, 1, 2))
    self.assertAllEqual(
        y.values, x.values[:, :, jnp.newaxis, 0, jnp.newaxis, 1:3]
    )

    l = simple.Slice.Config((jnp.newaxis, 0, 0, jnp.newaxis)).make().bind({})
    self.assertEqual(l.get_output_shape_for_sequence(x), (1, 1))
    y = l.layer(x, training=True)
    self.assertEqual(y.values.shape, (1, 2, 1, 1))
    self.assertAllEqual(
        y.values, x.values[:, :, jnp.newaxis, 0, 0, jnp.newaxis]
    )

  def test_slice_wrongsize(self):
    batch_size, time, channels = 2, 10, 3
    x = test_utils.random_sequence(batch_size, time, channels)
    l = (
        simple.Slice.Config(((None, None, None), (None, None, None)))
        .make()
        .bind({})
    )
    with self.assertRaises(ValueError):
      l.layer(x, training=False)

    l = (
        simple.Slice.Config(((None, None, None), None, (None, None, None)))
        .make()
        .bind({})
    )
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class FlattenTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (((2, 3, 5)),), (((2, 3, 5, 9)),), (((2, 3, 5, 9, 2)),)
  )
  def test_flatten(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Flatten.Config(name='flatten').make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    num_elements = np.prod(shape[2:])
    self.assertEqual(l.get_output_shape_for_sequence(x), (num_elements,))
    self.assertEqual(l.name, 'flatten')

    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    y_expected = x.apply_values(jnp.reshape, shape[:2] + (num_elements,))
    self.assertSequencesEqual(y, y_expected)


class GlobalReshapeTest(test_utils.SequenceLayerTest):

  def run_global_reshape(self, shape, output_shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.GlobalReshape.Config(output_shape, name='global_reshape').make()
    l = self.init_and_bind_layer(key, l, x)

    with self.subTest('properties'):
      self.assertEqual(l.block_size, 1)
      self.assertEqual(l.output_ratio, 1)
      self.assertEqual(l.get_output_shape_for_sequence(x), output_shape[1:])
      self.assertEqual(l.name, 'global_reshape')
      self.assertFalse(l.supports_step)

    with self.subTest('verify_contract'):
      self.verify_contract(
          l, x, training=False, test_padding_invariance=False
      )
      self.assertEmpty(l.variables)

  @parameterized.parameters(
      # same in/out channel_shape, channel_ndims=0
      ((2, 3), (3,)),
      # same in/out channel_shape, channel_ndims=1, channel_dim=1
      ((2, 3, 1), (3, 1)),
      # same in/out channel_shape, channel_ndims=1
      ((2, 3, 5), (3, 5)),
      # same in/out channel_shape, channel_ndims=2
      ((2, 3, 5, 9), (3, 5, 9)),
  )
  def test_input_time_equals_output_time(self, shape, output_shape):
    """Cases where shape[1] is equal to output_shape[0]."""
    self.run_global_reshape(shape, output_shape)

  @parameterized.parameters(
      # input_time < output_time, input channel_ndims = output channel_ndims
      ((2, 3, 5, 7), (7, 5, 3)),
      # input_time < output_time, input channel_ndims < output channel_ndims
      ((2, 3, 35), (7, 5, 3)),
      # input_time < output_time, input channel_ndims > output channel_ndims
      ((2, 3, 5, 7), (7, 15)),
  )
  def test_input_time_is_less_than_output_time(self, shape, output_shape):
    """Cases where shape[1] is less than output_shape[0]."""
    self.run_global_reshape(shape, output_shape)

  @parameterized.parameters(
      # input_time > output_time, input channel_ndims = output channel_ndims
      ((2, 7, 5, 3), (3, 5, 7)),
      # input_time > output_time, input channel_ndims < output channel_ndims
      ((2, 7, 15), (3, 5, 7)),
      # input_time > output_time, input channel_ndims > output channel_ndims
      ((2, 7, 5, 3), (3, 35)),
  )
  def test_input_time_is_greater_than_output_time(self, shape, output_shape):
    """Cases where shape[1] is greater than output_shape[0]."""
    self.run_global_reshape(shape, output_shape)

  def test_wrong_shape(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = simple.GlobalReshape.Config([4], name='global_reshape').make().bind({})

    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)


class ReshapeTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 5), (1, 5, 1)),
      ((2, 3, 5, 9), (3, 3, 5)),
      ((2, 3, 1), ()),
      ((2, 3), (1,)),
  )
  def test_reshape(self, shape, output_shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Reshape.Config(output_shape, name='reshape').make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    self.assertEqual(l.name, 'reshape')

    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    y_expected = x.apply_values(jnp.reshape, shape[:2] + output_shape)
    self.assertSequencesEqual(y, y_expected)

  def test_wrong_shape(self):
    l = simple.Reshape.Config([4], name='reshape').make().bind({})
    x = test_utils.random_sequence(2, 3, 5)

    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class TransposeTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 4, 5), (2, 3), (4, 5)),
      ((2, 3, 4, 5, 6), (4, 2, 3), (6, 4, 5)),
      ((2, 3, 1, 2, 3), None, (3, 2, 1)),
      ((2, 3), tuple(), tuple()),
      ((2, 3), None, tuple()),
  )
  def test_transpose(self, input_shape, axes, output_shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*input_shape)
    l = simple.Transpose.Config(axes=axes, name='transpose').make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    self.assertEqual(l.name, 'transpose')

    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    if axes is not None:
      y_expected = x.apply_values(jnp.transpose, (0, 1) + axes)
    else:
      axes = (0, 1) + tuple(range(2, x.ndim))[::-1]
      y_expected = x.apply_values(jnp.transpose, axes)

    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(
      ((2, 3), (2,)),
      ((2, 3, 4, 5, 6), (2, 2, 3)),
      ((2, 3, 4, 5), (5, 4)),
  )
  def test_wrong_axes(self, input_shape, axes):
    l = simple.Transpose.Config(axes=axes, name='transpose').make().bind({})
    x = test_utils.random_sequence(*input_shape)

    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class SwapAxesTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 4, 5), 2, 3, (5, 4)),
      ((2, 3, 4, 5, 6), 4, 2, (6, 5, 4)),
      ((2, 3, 4), 2, 2, (4,)),
      ((2, 3, 4, 5, 6), -1, -3, (6, 5, 4)),
  )
  def test_swapaxes(self, input_shape, axis1, axis2, output_shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*input_shape)
    l = simple.SwapAxes.Config(axis1, axis2, name='swapaxes').make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    self.assertEqual(l.name, 'swapaxes')

    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    y_expected = x.apply_values(jnp.swapaxes, axis1=axis1, axis2=axis2)
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(
      ((2, 3), 0, 3),
      ((2, 3), 2, 1),
      ((2, 3, 4, 5, 6), 0, 2),
      ((2, 3, 4, 5, 6), 3, 1),
  )
  def test_swapaxes_error_batch_time_axis(self, input_shape, axis1, axis2):
    del input_shape
    with self.assertRaises(ValueError):
      simple.SwapAxes.Config(axis1, axis2, name='swapaxes').make()

  @parameterized.parameters(
      ((2, 3), 2, 3),
      ((2, 3, 4, 5, 6), -4, 2),
      ((2, 3, 4, 5, 6), 3, -5),
  )
  def test_swapaxes_wrong_axes(self, input_shape, axis1, axis2):
    l = simple.SwapAxes.Config(axis1, axis2, name='swapaxes').make().bind({})
    x = test_utils.random_sequence(*input_shape)

    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class MoveAxisTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 4, 5), 2, -1, (5, 4)),
      ((2, 3, 4, 5, 6), -1, 2, (6, 4, 5)),
      ((2, 3, 4), 2, 2, (4,)),
      ((2, 3, 4, 5, 6), -2, 2, (5, 4, 6)),
      ((2, 3, 4, 5, 6), (2, 3), (3, 4), (6, 4, 5)),
      ((2, 3, 4, 5, 6), (3, 2), (3, 4), (6, 5, 4)),
      ((2, 3, 4, 5, 6), (2, 4), (3, 2), (6, 4, 5)),
  )
  def test_moveaxis(self, input_shape, source, destination, output_shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*input_shape)
    l = simple.MoveAxis.Config(source, destination, name='moveaxis').make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    self.assertEqual(l.name, 'moveaxis')

    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    y_expected = x.apply_values(
        jnp.moveaxis, source=source, destination=destination
    )
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(
      ((2, 3), 0, 3),
      ((2, 3, 4), 2, 1),
      ((2, 3, 4, 5, 6), (2,), (3, 4)),
      ((2, 3, 4, 5, 6), 2, (3, 4)),
  )
  def test_moveaxis_error_make(self, input_shape, source, destination):
    del input_shape
    with self.assertRaises(ValueError):
      simple.MoveAxis.Config(source, destination, name='moveaxis').make()

  @parameterized.parameters(
      ((2, 3, 4), -3, 2),
      ((2, 3, 4), 2, -2),
      ((2, 3, 4, 5, 6), (-4, 2), (2, 3)),
      ((2, 3, 4, 5, 6), (3, 2), (2, -5)),
  )
  def test_moveaxis_wrong_axes(self, input_shape, source, destination):
    l = (
        simple.MoveAxis.Config(source, destination, name='moveaxis')
        .make()
        .bind({})
    )
    x = test_utils.random_sequence(*input_shape)

    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class EinopsRearrangeTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 4, 5), 'c d -> (c d)', None, (20,)),
      ((2, 3, 20), '(c d) -> c d', {'c': 4}, (4, 5)),
      ((2, 3, 4, 5), '(c d) e ->  c d e', {'c': 2}, (2, 2, 5)),
      ((2, 3, 6, 5), '(c d) e ->  d e c', {'c': 2}, (3, 5, 2)),
  )
  def test_einops_rearrange(
      self, input_shape, pattern, axes_lengths, output_shape
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*input_shape)
    l = simple.EinopsRearrange.Config(
        pattern, axes_lengths, name='rearrange'
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    self.assertEqual(l.name, 'rearrange')

    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    before, after = pattern.split('->')
    pattern = f'batch time {before} -> batch time {after}'
    if not axes_lengths:
      axes_lengths = {}
    y_expected = x.apply_values(
        lambda v: einops.rearrange(v, pattern, **axes_lengths)
    )
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(
      ('(c d) time ->  c d time', {'c': 2}),
      ('(c d) batch ->  c d batch', {'c': 2}),
      ('(c d) batch - c d batch', {'c': 2}),
  )
  def test_einops_rearrange_error_make(self, pattern, axes_lengths):
    with self.assertRaises(ValueError):
      simple.EinopsRearrange.Config(
          pattern, axes_lengths, name='rearrange'
      ).make()


class GradientClippingTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 5, 9),))
  def test_forward_pass_is_identity(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.GradientClipping.Config(
        name='clip_gradient', clip_value=1e12
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertSequencesEqual(y, x)

  @parameterized.named_parameters(
      dict(
          testcase_name='too_high',
          input_gradient_value=1e13,
          expected_output_gradient_value=1e12,
      ),
      dict(
          testcase_name='too_low',
          input_gradient_value=-1e13,
          expected_output_gradient_value=-1e12,
      ),
      dict(
          testcase_name='just_right',
          input_gradient_value=2016.0913,
          expected_output_gradient_value=2016.0913,
      ),
  )
  def test_backward_pass_clips_gradient(
      self, input_gradient_value, expected_output_gradient_value
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 13, 5)

    def get_dy(y: types.Sequence) -> types.Sequence:
      # Do not flow gradient through masked regions by masking dy.
      dy = (
          types.Sequence(jnp.full_like(y.values, input_gradient_value), y.mask)
          .mask_invalid()
          .values
      )
      dmask = jnp.zeros_like(y.mask)
      return type(y)(dy, dmask)

    def layer_fn(
        l: types.SequenceLayer, x: types.Sequence, constants: types.Constants
    ) -> types.Sequence:
      return l.layer(x, training=True, constants=constants).mask_invalid()

    def layer_vjp_fn(
        l: types.SequenceLayer, x: types.Sequence, constants: types.Constants
    ):
      y, layer_vjp_fn = nn.vjp(layer_fn, l, x, constants)
      params_grad, x_grad, unused_constants_grad = layer_vjp_fn(get_dy(y))
      x_grad = types.Sequence(x_grad.values, x.mask).mask_invalid()
      return y, x_grad, params_grad

    l = simple.GradientClipping.Config(
        name='clip_gradient', clip_value=1e12
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    _, y_layer_x_grad, _ = layer_vjp_fn(l, x, {})
    expected_gradients = types.Sequence(
        jnp.full_like(y_layer_x_grad.values, expected_output_gradient_value),
        y_layer_x_grad.mask,
    ).mask_invalid()
    self.assertSequencesEqual(expected_gradients, y_layer_x_grad)


class IdentityTest(test_utils.SequenceLayerTest):

  @parameterized.parameters((((2, 3, 5)),), (((2, 3, 5, 9)),))
  def test_identity(self, shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = simple.Identity(name='identity')
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'identity')
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)


class EmitTest(test_utils.SequenceLayerTest):

  def test_emit(self):
    key = jax.random.PRNGKey(1234)
    l = simple.Emit.Config(name='emit').make()
    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5,))
    self.assertEqual(l.name, 'emit')
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    y, emits = l.layer_with_emits(x, training=False)
    self.assertSequencesEqual(y, emits)


class OneHotTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((1, 2, 3),), ((2, 3, 5, 9),), ((2, 3, 5, 9, 2),))
  def test_one_hot(self, shape):
    key = jax.random.PRNGKey(1234)
    depth = 4
    l = simple.OneHot.Config(depth, name='one_hot').make()
    x = test_utils.random_sequence(
        *shape, dtype=jnp.int32, low=0, high=depth - 1
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:] + (depth,))
    self.assertEqual(l.name, 'one_hot')

    l = self.init_and_bind_layer(key, l, x)

    y = self.verify_contract(
        l,
        x,
        training=False,
        padding_invariance_pad_value=0,
        # Integer tensors have no gradient to test.
        test_gradients=False,
    )
    self.assertEmpty(l.variables)
    self.assertAllEqual(
        y.values, (np.eye(depth)[x.values].T * x.mask.astype(jnp.float32).T).T
    )


def embedding_layer_from_weights(
    weights: jax.Array,
    query: jax.Array,
    compute_dtype: types.DType | None = None,
    param_dtype: types.DType = jnp.float32,
) -> simple.Embedding:
  num_embeddings, dimension = weights.shape
  key = jax.random.PRNGKey(1234)
  layer = simple.Embedding.Config(
      compute_dtype=compute_dtype,
      param_dtype=param_dtype,
      dimension=dimension,
      num_embeddings=num_embeddings,
      name='embedding',
  ).make()
  inputs = test_utils.random_sequence(
      query.shape[0], query.shape[1], dtype=jnp.int32, low=0, high=dimension - 1
  )
  layer.init(key, inputs, training=False)
  layer = layer.bind({'params': {'embedding': weights}})
  return layer


class EmbeddingTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((1, 2, 3),), ((2, 3, 5, 9),), ((2, 3, 5, 9, 2),))
  def test_embedding(self, shape):
    key = jax.random.PRNGKey(1234)
    dimension, num_embeddings = 8, 5

    l = simple.Embedding.Config(
        dimension=dimension, num_embeddings=num_embeddings, name='embedding'
    ).make()
    x = test_utils.random_sequence(
        *shape, dtype=jnp.int32, low=0, high=num_embeddings - 1
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:] + (dimension,)
    )
    self.assertEqual(l.name, 'embedding')
    l = self.init_and_bind_layer(key, l, x)

    y = self.verify_contract(
        l,
        x,
        training=False,
        # Integer tensors have no gradient to test.
        test_gradients=False,
    )

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'embedding': jnp.zeros((num_embeddings, dimension)),
            }
        },
    )

    embedding = variables['params']['embedding']
    expected = types.Sequence(
        jnp.take(embedding, x.values, axis=0), x.mask
    ).mask_invalid()
    self.assertSequencesClose(y, expected)

  @parameterized.parameters(((1, 2),), ((2, 3, 5),))
  def test_embedding_attend(self, shape):
    key = jax.random.PRNGKey(1234)

    embeddings = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    embedding_squared_norms = (embeddings**2).sum(axis=-1)
    num_embeddings = embeddings.shape[0]
    dimension = embeddings.shape[1]

    l = simple.Embedding.Config(
        dimension=dimension, num_embeddings=num_embeddings, name='embedding'
    ).make()
    x = test_utils.random_sequence(
        *shape, 1, dtype=jnp.int32, low=0, high=num_embeddings - 1
    )
    l.init(key, x, training=False)
    l = l.bind({'params': {'embedding': embeddings}})

    with self.subTest(name='discrete'):
      query = jnp.tile(jnp.array([0, 0, 1, 0, 0]), (*shape, 1))
      query_results = jnp.tile(
          jnp.asarray([1.0, -1.0, 2.0], dtype=jnp.float32), (*shape, 1)
      )
      y = l.attend(types.Sequence.from_values(query))
      self.assertSequencesClose(y, types.Sequence.from_values(query_results))

    with self.subTest(name='weighted'):
      y = l.attend(types.Sequence.from_values(jnp.expand_dims(embeddings, 0)))
      self.assertSequencesClose(
          types.Sequence.from_values(
              jnp.diagonal(y.values, axis1=-1, axis2=-2)
          ),
          types.Sequence.from_values(
              jnp.expand_dims(embedding_squared_norms, 0)
          ),
      )

  @parameterized.parameters(
      dict(param_dtype=None, compute_dtype=None, expected_dtype=jnp.float16),
      dict(
          param_dtype=jnp.float32,
          compute_dtype=None,
          expected_dtype=jnp.float32,
      ),
      dict(
          param_dtype=jnp.float32,
          compute_dtype=jnp.bfloat16,
          expected_dtype=jnp.bfloat16,
      ),
      dict(
          param_dtype=jnp.bfloat16,
          compute_dtype=jnp.bfloat16,
          expected_dtype=jnp.bfloat16,
      ),
      # This upcasts since bfloat16 w/ float16 promotes to float32:
      dict(
          param_dtype=jnp.bfloat16,
          compute_dtype=None,
          expected_dtype=jnp.float32,
      ),
  )
  def test_embedding_attend_dtypes(
      self, param_dtype, compute_dtype, expected_dtype
  ):
    """Tests if the dtype override options are respected."""
    batch_size, seq_len = 2, 3
    default_dtype = jnp.float16

    embeddings = jnp.array(
        [[0.5, 1.5, 0], [-1.0, 0, 1.0]],
        dtype=default_dtype,
    )
    query = jnp.tile(
        jnp.array([2.0, 0, 0], dtype=default_dtype),
        (batch_size, seq_len, 1),
    )
    query_results = jnp.tile(
        jnp.array([1.0, -2.0], dtype=expected_dtype), (batch_size, seq_len, 1)
    )

    layer = embedding_layer_from_weights(
        embeddings, query, compute_dtype=None, param_dtype=default_dtype
    )

    outputs = layer.attend(
        types.Sequence.from_values(query),
        embedding_dtype=param_dtype,
        compute_dtype=compute_dtype,
    )

    with self.subTest('embedding_dtype_does_not_change'):
      self.assertEqual(embeddings.dtype, default_dtype)
    with self.subTest('results_values_and_dtypes'):
      self.assertSequencesClose(
          outputs, types.Sequence.from_values(query_results)
      )
      self.assertEqual(outputs.values.dtype, expected_dtype)


class EmbeddingTransposeTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      (
          dict(param_dtype=None, input_dtype=jnp.float32, compute_dtype=None),
          *test_utils.standard_dtype_configs(),
      ),
      use_bias=(False, True),
      other_channels_shape=((), (2, 3)),
  )
  def test_embedding_transpose(
      self,
      param_dtype,
      input_dtype,
      compute_dtype,
      use_bias,
      other_channels_shape,
  ):
    batch_size, seq_len = 2, 3
    default_dtype = jnp.float16

    embeddings = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ],
        # This should be overridden for the computation.
        dtype=default_dtype,
    )
    query = jnp.tile(
        jnp.array([0, 0, 0, 1, 1], dtype=input_dtype),
        (batch_size, seq_len, *other_channels_shape, 1),
    )
    expected = jnp.tile(
        jnp.array([2.0, -2.0, 7.0], dtype=compute_dtype),
        (batch_size, seq_len, *other_channels_shape, 1),
    )

    embedding = embedding_layer_from_weights(
        embeddings,
        jnp.zeros((batch_size, 1), dtype=jnp.int32),
        compute_dtype=default_dtype,
        param_dtype=default_dtype,
    )

    layer = simple.EmbeddingTranspose.Config(
        embedding=embedding,
        use_bias=use_bias,
        bias_init=nn.initializers.ones_init(),
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        name='embedding_transpose',
    ).make()
    inputs = types.Sequence.from_values(query)
    layer = self.init_and_bind_layer(jax.random.PRNGKey(1234), layer, inputs)
    outputs = self.verify_contract(
        layer,
        inputs,
        training=False,
        **test_utils.get_grad_tols(layer, inputs, param_dtype, compute_dtype),
    )

    with self.subTest('config'):
      self.assertEqual(layer.block_size, 1)
      self.assertEqual(layer.output_ratio, 1)
      self.assertEqual(layer.name, 'embedding_transpose')

    with self.subTest('params'):
      variables = flax.core.meta.unbox(layer.variables)
      if use_bias:
        expected_params = {
            'params': {
                'bias': jnp.ones(
                    (embeddings.shape[0],), dtype=param_dtype or default_dtype
                ),
            }
        }
      else:
        expected_params = {}
      chex.assert_trees_all_equal_shapes_and_dtypes(variables, expected_params)

    with self.subTest('outputs'):
      self.assertEqual(outputs.values.dtype, compute_dtype or default_dtype)
      if use_bias:
        expected += 1.0
      self.assertSequencesClose(outputs, types.Sequence.from_values(expected))

  def test_embedding_transpose_binds_embedding_if_not_bound(self):
    key = jax.random.PRNGKey(1234)

    embedding = simple.Embedding.Config(
        dimension=5, num_embeddings=2, name='embedding'
    ).make()
    embedding_transpose = simple.EmbeddingTranspose.Config(
        embedding=embedding, name='embedding_transpose'
    ).make()
    embedding_transpose = self.init_and_bind_layer(
        key,
        embedding_transpose,
        types.Sequence.from_values(
            jnp.zeros((1, 1, embedding.config.dimension), dtype=jnp.float32)
        ),
    )

    result = embedding_transpose.layer(
        types.Sequence.from_values(
            jnp.zeros((1, 1, embedding.config.dimension), dtype=jnp.float32)
        ),
        training=False,
    )

    self.assertEqual(result.shape, (1, 1, embedding.config.num_embeddings))

  @parameterized.parameters(
      dict(input_shape=(1, 3), embed_dim=5),
      dict(input_shape=(), embed_dim=1),
  )
  def test_embedding_transpose_raises_on_invalid_input_shape(
      self, input_shape, embed_dim
  ):
    embedding = simple.Embedding.Config(
        dimension=embed_dim, num_embeddings=2, name='embedding'
    ).make()
    embedding_transpose = simple.EmbeddingTranspose.Config(
        embedding=embedding, name='embedding_transpose'
    ).make()
    with self.assertRaises(ValueError):
      embedding_transpose.get_output_shape(input_shape)


class AffineTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (
          simple.Affine.Config(
              use_scale=True,
              use_bias=True,
              scale_init=nn.initializers.constant(0.1),
              bias_init=nn.initializers.constant(0.5),
              param_dtype=jnp.float32,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=True,
              use_bias=True,
              scale_init=nn.initializers.constant(0.1 + 0.2j),
              bias_init=nn.initializers.constant(0.5 - 0.1j),
              param_dtype=jnp.complex64,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=True,
              use_bias=False,
              scale_init=nn.initializers.constant(0.1 + 0.2j),
              param_dtype=jnp.complex64,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=True,
              use_bias=False,
              shape=(4, 3),
              scale_init=nn.initializers.constant(0.1),
              param_dtype=jnp.float32,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=True,
              use_bias=True,
              shape=(4, 3),
              scale_init=nn.initializers.constant(0.5),
              bias_init=nn.initializers.constant(-0.1),
              param_dtype=jnp.float32,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=True,
              use_bias=False,
              shape=(3,),
              scale_init=nn.initializers.constant(0.1),
              param_dtype=jnp.float32,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=True,
              use_bias=True,
              shape=(3,),
              scale_init=nn.initializers.constant(0.5),
              bias_init=nn.initializers.constant(-0.1),
              param_dtype=jnp.float32,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=False,
              use_bias=True,
              shape=(3,),
              bias_init=nn.initializers.constant(-0.1),
              param_dtype=jnp.float32,
          )
      ),
      (
          simple.Affine.Config(
              use_scale=False,
              use_bias=True,
              shape=(4, 3),
              bias_init=nn.initializers.constant(-0.1),
              param_dtype=jnp.float32,
          )
      ),
  )
  def test_scale_bias_scalar(self, config):
    key = jax.random.PRNGKey(1234)
    batch_size, time, channels, channels2 = 2, 10, 4, 3

    dtype = config.param_dtype

    x = test_utils.random_sequence(
        batch_size, time, channels, channels2, dtype=dtype
    )
    l = dataclasses.replace(config, name='test').make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), (channels, channels2))
    self.assertEqual(l.name, 'test')
    y = self.verify_contract(l, x, training=False)

    # Split state and params (which are updated by optimizer).
    state, params = flax.core.pop(l.variables, 'params')
    self.assertEmpty(state)

    # Check the parameters shapes and dtypes.
    params = flax.core.meta.unbox(params)
    params_expected = {}
    if config.use_bias:
      params_expected['bias'] = jnp.zeros(
          shape=config.shape, dtype=config.param_dtype
      )
    if config.use_scale:
      params_expected['scale'] = jnp.zeros(
          shape=config.shape, dtype=config.param_dtype
      )
    chex.assert_trees_all_equal_shapes_and_dtypes(params, params_expected)

    y_expected = x.apply_values(
        lambda v: v
        * (1.0 if not config.use_scale else params['scale'].astype(dtype))
        + (0.0 if not config.use_bias else params['bias'].astype(dtype))
    ).mask_invalid()

    self.assertSequencesClose(y, y_expected)

  def test_too_many_dims(self):
    x = test_utils.random_sequence(2, 3, 5, 1)
    variables = {
        'params': {
            'scale': jnp.ones((3, 5, 1), dtype=jnp.float32),
            'bias': jnp.zeros((3, 5, 1), dtype=jnp.float32),
        }
    }
    l = (
        simple.Affine.Config(
            use_bias=True,
            shape=(3, 5, 1),
            scale_init=nn.initializers.constant(0.5),
            bias_init=nn.initializers.constant(-0.1),
            param_dtype=jnp.float32,
        )
        .make()
        .bind(variables)
    )
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = test_utils.random_sequence(2, 3, 5, 9)
    variables = {
        'params': {
            'scale': jnp.ones((5, 5), dtype=jnp.float32),
            'bias': jnp.zeros((5, 5), dtype=jnp.float32),
        }
    }
    l = (
        simple.Affine.Config(
            use_bias=True,
            shape=(5, 5),
            scale_init=nn.initializers.constant(0.5),
            bias_init=nn.initializers.constant(-0.1),
            param_dtype=jnp.float32,
        )
        .make()
        .bind(variables)
    )
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class PointwiseMathTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (simple.Abs.Config(), jnp.abs, (jnp.float32, jnp.complex64), None),
      (simple.Elu.Config(), jax.nn.elu, (jnp.float32,), None),
      (simple.Exp.Config(), jnp.exp, (jnp.float32,), None),
      (simple.Gelu.Config(), jax.nn.gelu, (jnp.float32,), None),
      (simple.LeakyRelu.Config(), jax.nn.leaky_relu, (jnp.float32,), None),
      (
          simple.PRelu.Config(param_dtype=jnp.float32),
          jax.nn.leaky_relu,
          (jnp.float32,),
          {'negative_slope': {'shape': [], 'dtype': jnp.float32}},
      ),
      (simple.Log.Config(), jnp.log, (jnp.float32,), None),
      (simple.Power.Config(2), jnp.square, (jnp.float32,), None),
      (simple.Power.Config(0.5), jnp.sqrt, (jnp.float32,), None),
      (simple.Relu.Config(), jax.nn.relu, (jnp.float32,), None),
      (simple.Sigmoid.Config(), jax.nn.sigmoid, (jnp.float32,), None),
      (simple.Softmax.Config(), jax.nn.softmax, (jnp.float32,), None),
      (simple.Softplus.Config(), jax.nn.softplus, (jnp.float32,), None),
      (simple.Swish.Config(), jax.nn.swish, (jnp.float32,), None),
      (simple.Tanh.Config(), jnp.tanh, (jnp.float32,), None),
  )
  def test_pointwise_math(self, config, op, dtypes, expected_params):
    key = jax.random.PRNGKey(1234)
    batch_size, time, channels = 2, 10, 4
    for dtype in dtypes:
      x = test_utils.random_sequence(batch_size, time, channels, dtype=dtype)
      l = dataclasses.replace(config, name='test').make()
      l = self.init_and_bind_layer(key, l, x)

      self.assertEqual(l.block_size, 1)
      self.assertEqual(l.output_ratio, 1)
      self.assertEqual(l.get_output_shape_for_sequence(x), (channels,))
      self.assertEqual(l.name, 'test')
      y = self.verify_contract(l, x, training=False)

      # Split state and params (which are updated by optimizer).
      try:
        state, params = flax.core.pop(l.variables, 'params')

        if expected_params is not None:
          expected_params = jax.tree.map(
              lambda kwargs: jnp.zeros(**kwargs),
              expected_params,
              is_leaf=lambda d: ('shape' in d and 'dtype' in d),
          )
          chex.assert_trees_all_equal_shapes_and_dtypes(params, expected_params)
      except KeyError:
        state = l.variables
      self.assertEmpty(state)

      y_expected = x.apply_values(op).mask_invalid()
      self.assertSequencesClose(y, y_expected)

  @parameterized.parameters(
      (simple.Softmax.Config(), jax.nn.softmax, (jnp.float32,), -1),
      (simple.Softmax.Config(), jax.nn.softmax, (jnp.float32,), -2),
      (simple.Softmax.Config(), jax.nn.softmax, (jnp.float32,), 2),
      (simple.Softmax.Config(), jax.nn.softmax, (jnp.float32,), 3),
  )
  def test_pointwise_math_axis(self, config, op, dtypes, axis):
    key = jax.random.PRNGKey(1234)
    batch_size, time, channels, channels2 = 2, 10, 4, 3
    for dtype in dtypes:
      x = test_utils.random_sequence(
          batch_size, time, channels, channels2, dtype=dtype
      )
      l = dataclasses.replace(config, name='test', axis=axis).make()
      l = self.init_and_bind_layer(key, l, x)

      self.assertEqual(l.block_size, 1)
      self.assertEqual(l.output_ratio, 1)
      self.assertEqual(
          l.get_output_shape_for_sequence(x), (channels, channels2)
      )
      self.assertEqual(l.name, 'test')
      y = self.verify_contract(l, x, training=False)
      self.assertEmpty(l.variables)

      y_expected = x.apply_values(
          functools.partial(op, axis=axis)
      ).mask_invalid()
      self.assertSequencesClose(y, y_expected)

  @parameterized.parameters(
      (simple.Softmax.Config(), (2, 10, 4), -2),
      (simple.Softmax.Config(), (2, 10, 4), -3),
      (simple.Softmax.Config(), (2, 10, 4), 0),
      (simple.Softmax.Config(), (2, 10, 4), 1),
      (simple.Softmax.Config(), (2, 10), -1),
  )
  def test_pointwise_math_axis_invalid(self, config, shape, axis):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = dataclasses.replace(config, name='test', axis=axis).make()

    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)


class CastTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (((2, 3, 5)), jnp.float16),
      (((2, 3, 5, 9)), jnp.int32),
  )
  def test_cast(self, shape, target_dtype):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape, dtype=jnp.float32)
    l = simple.Cast.Config(target_dtype, name='cast').make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.assertEqual(l.name, 'cast')

    y = self.verify_contract(
        l,
        x,
        training=False,
        padding_invariance_pad_value=jnp.nan
        if target_dtype == jnp.float16
        else 32768,
    )
    self.assertEmpty(l.variables)
    self.assertEqual(y.values.dtype, target_dtype)


class ApplyShardingTest(test_utils.SequenceLayerTest):

  def test_basic(self):
    key = jax.random.PRNGKey(1234)
    x_original = test_utils.random_sequence(2, 3, 5)

    # Run with a mesh and under jit to verify that sharding annotations are
    # correctly applied.
    mesh = jax.sharding.Mesh(
        jax.experimental.mesh_utils.create_device_mesh(
            (1, 1, 1, 1),
            devices=jax.local_devices(),
        ),
        ('replica', 'data', 'seq', 'model'),
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # Replicate x_original on all shards of the mesh.
    x_values_sharded = jax.device_put(x_original.values, replicated_sharding)
    x_mask_sharded = jax.device_put(x_original.mask, replicated_sharding)
    x_sharded = type(x_original)(values=x_values_sharded, mask=x_mask_sharded)

    l = simple.ApplySharding.Config(
        (None, 'data', 'model'), name='apply_sharding'
    ).make()
    with sharding_lib.use_mesh(mesh):
      l = self.init_and_bind_layer(key, l, x_sharded)

      self.assertEqual(l.block_size, 1)
      self.assertEqual(l.output_ratio, 1)
      self.assertEqual(l.get_output_shape_for_sequence(x_sharded), (5,))
      self.assertEqual(l.name, 'apply_sharding')

      self.verify_contract(l, x_sharded, training=False)
      self.assertEmpty(l.variables)
      # TODO(rryan): Test sharding was applied.


class LambdaTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(True, False)
  def test_array_fn(self, mask_required: bool):
    def fn(v: jax.Array) -> jax.Array:
      if mask_required:
        # Change the masked status by adding 1.
        v = v + 1.0
      return v.reshape(v.shape + (1,)) > 0.5

    l = (
        simple.Lambda.Config(
            fn,
            mask_required=mask_required,
            expected_input_spec=types.ShapeDType((5,), jnp.float32),
            name='lambda',
        )
        .make()
        .bind({})
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    # Output spec reflects the changed shape and dtype.
    x = test_utils.random_sequence(2, 3, 5)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 1))
    self.assertEqual(l.get_output_dtype(x.dtype), jnp.bool_)
    self.assertEqual(l.name, 'lambda')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertSequencesClose(y, x.apply_values(fn).mask_invalid())

  @parameterized.parameters(True, False)
  def test_sequence_fn(self, mask_required: bool):
    def fn(x: types.Sequence) -> types.Sequence:
      if mask_required:
        # Change the masked status by adding 1.
        x = x.apply_values(lambda v: v + 1.0)
      return x.apply_values_masked(lambda v: v.reshape(v.shape + (1,)) > 0.5)

    l = (
        simple.Lambda.Config(
            fn,
            sequence_input=True,
            expected_input_spec=types.ShapeDType((5,), jnp.float32),
            name='lambda',
        )
        .make()
        .bind({})
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    # Output spec reflects the changed shape and dtype.
    x = test_utils.random_sequence(2, 3, 5)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 1))
    self.assertEqual(l.get_output_dtype(x.dtype), jnp.bool_)
    self.assertEqual(l.name, 'lambda')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertSequencesClose(y, fn(x).mask_invalid())

  def test_invalid_input(self):
    """Input that does not match expected_input_spec raises ValueError."""
    self.skipTest(
        'TODO(rryan): Re-enable when SoundStream works as expected with this.'
    )

    l = (
        simple.Lambda.Config(
            lambda v: v,
            expected_input_spec=types.ShapeDType((5,), jnp.float32),
            name='lambda',
        )
        .make()
        .bind({})
    )

    x = test_utils.random_sequence(2, 3, 6)
    with self.assertRaises(ValueError):
      l.get_output_spec(x.channel_spec)

    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_invalid_fn(self):
    """Functions that change the batch or time shape are invalid."""
    l = (
        simple.Lambda.Config(
            lambda v: v[:, :-1],
            expected_input_spec=types.ShapeDType((5,), jnp.float32),
            name='lambda',
        )
        .make()
        .bind({})
    )

    x = test_utils.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class CheckpointNameTest(test_utils.SequenceLayerTest):

  def test_basic(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = simple.CheckpointName.Config(
        checkpoint_name='test', name='checkpoint_name'
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5,))
    self.assertEqual(l.name, 'checkpoint_name')
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    policy = jax.checkpoint_policies.save_only_these_names('test')

    @functools.partial(jax.checkpoint, policy=policy)
    def f(x: types.Sequence) -> types.Sequence:
      # Modify x so there's something to checkpoint.
      x = x.apply_values(jax.nn.sigmoid)
      return l.layer(x, training=False)

    # Check that the checkpoint name was applied:
    # TODO(rryan): Don't use private JAX APIs.
    self.assertLen(
        jax._src.ad_checkpoint.saved_residuals(f, x),
        1,
    )


class Downsample1DTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 3, 5), 2), ((2, 3, 5, 9), 3))
  def test_downsample1d(self, shape, rate):
    l = simple.Downsample1D.Config(rate, name='downsample_1d').make().bind({})

    self.assertEqual(l.block_size, rate)
    self.assertEqual(1/l.output_ratio, rate)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.name, 'downsample_1d')
    self.assertEmpty(l.variables)

    x = test_utils.random_sequence(*shape)
    self.assertEqual(l.get_output_shape_for_sequence(x), x.channel_shape)
    y = self.verify_contract(l, x, training=False)
    self.assertAllEqual(x.values[:, ::rate], y.values)
    self.assertAllEqual(x.mask[:, ::rate], y.mask)


class Upsample1DTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(((2, 3, 5), 2), ((2, 3, 5, 9), 3))
  def test_upsample1d(self, shape, rate):
    l = simple.Upsample1D.Config(rate, name='upsample_1d').make().bind({})

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, rate)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.name, 'upsample_1d')
    self.assertEmpty(l.variables)

    x = test_utils.random_sequence(*shape)
    self.assertEqual(l.get_output_shape_for_sequence(x), x.channel_shape)
    y = self.verify_contract(l, x, training=False)
    for i in range(rate):
      self.assertAllEqual(x.values, y.values[:, i::rate])


class Upsample2DTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 5, 9), (2, 1)), ((2, 3, 5, 9), (2, 3)), ((2, 3, 5, 9), (1, 3))
  )
  def test_upsample2d(self, shape, rate):
    l = simple.Upsample2D.Config(rate, name='upsample_2d').make().bind({})

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, rate[0])
    self.assertTrue(l.supports_step)
    self.assertEqual(l.name, 'upsample_2d')
    self.assertEmpty(l.variables)

    x = test_utils.random_sequence(*shape)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (x.shape[2] * rate[1], x.shape[3])
    )
    y = self.verify_contract(l, x, training=False)
    for i, j in itertools.product(range(rate[0]), range(rate[1])):
      self.assertAllEqual(x.values, y.values[:, i :: rate[0], j :: rate[1], :])


class MaskInvalidTest(test_utils.SequenceLayerTest):

  def test_basic(self):
    x = test_utils.random_sequence(2, 15, 5)
    l = simple.MaskInvalid.Config(name='mask_invalid').make().bind({})

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5,))
    self.assertEqual(l.name, 'mask_invalid')
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    x = x.mask_invalid(np.nan)
    self.assertIsInstance(x, types.Sequence)
    y = l.layer(x, training=False)
    self.assertIsInstance(y, types.MaskedSequence)
    self.assertSequencesEqual(x.mask_invalid(), y)


class LoggingTest(test_utils.SequenceLayerTest):

  @mock.patch.object(logging, 'info', wraps=logging.info)
  def test_logs_tensors(self, mock_logger):
    x = types.Sequence.from_values(jnp.asarray([[1.414, 2, 3, 4]]))
    state = types.Sequence.from_values(jnp.asarray([[1, 2.718, 3, 4]]))
    training = False
    constants = {
        'foo': jnp.asarray([[1, 2, 3.14, 4]]),
        'bar': np.asarray([[1, 2, 3, 4.2]]),
    }

    with self.subTest('prefix'):
      l = simple.Logging.Config(prefix='test string').make().bind({})
      l.layer(x, training=training, constants=constants)
      mock_logger.assert_called_with(matchers.HAS('test string'))

    with self.subTest('specs_only'):
      l = simple.Logging.Config(dump_tensors=False).make().bind({})
      with self.subTest('layer'):
        l.layer(x, training=training, constants=constants)
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('1.414')))
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('3.14')))
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('4.2')))
        mock_logger.assert_called_with(matchers.HAS('(1, 4)'))
        mock_logger.assert_called_with(matchers.HAS('float32'))
      with self.subTest('get_initial_state'):
        l.get_initial_state(
            batch_size=x.shape[0],
            input_spec=x.channel_spec,
            training=training,
            constants=constants,
        )
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('3.14')))
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('4.2')))
        mock_logger.assert_called_with(matchers.HAS('(1, 4)'))
        mock_logger.assert_called_with(matchers.HAS('float32'))
      with self.subTest('step'):
        l.step(x, state, training=training, constants=constants)
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('1.414')))
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('2.718')))
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('3.14')))
        mock_logger.assert_called_with(matchers.NOT(matchers.HAS('4.2')))
        mock_logger.assert_called_with(matchers.HAS('(1, 4)'))
        mock_logger.assert_called_with(matchers.HAS('float32'))

    with self.subTest('dumps_tensors'):
      l = simple.Logging.Config(dump_tensors=True).make().bind({})
      with self.subTest('layer'):
        l.layer(x, training=training, constants=constants)
        mock_logger.assert_called_with(matchers.HAS('1.414'))
        mock_logger.assert_called_with(matchers.HAS('3.14'))
        mock_logger.assert_called_with(matchers.HAS('4.2'))
      with self.subTest('get_initial_state'):
        l.get_initial_state(
            batch_size=x.shape[0],
            input_spec=x.channel_spec,
            training=training,
            constants=constants,
        )
        mock_logger.assert_called_with(matchers.HAS('3.14'))
        mock_logger.assert_called_with(matchers.HAS('4.2'))
      with self.subTest('step'):
        l.step(x, state, training=training, constants=constants)
        mock_logger.assert_called_with(matchers.HAS('1.414'))
        mock_logger.assert_called_with(matchers.HAS('2.718'))
        mock_logger.assert_called_with(matchers.HAS('3.14'))
        mock_logger.assert_called_with(matchers.HAS('4.2'))


class ArgmaxTest(test_utils.SequenceLayerTest):

  @parameterized.named_parameters(
      dict(
          testcase_name='float_input',
          input_array=np.array(
              [[[0.1, 0.2, 0.3]], [[0.3, 0.2, 0.1]]], dtype=np.float32
          ),
      ),
      dict(
          testcase_name='integer_input',
          input_array=np.array([[[1, 2, 3]], [[3, 2, 1]]], dtype=np.int32),
      ),
  )
  def test_argmax(self, input_array: jnp.ndarray):
    key = jax.random.PRNGKey(1234)
    x = types.Sequence.from_values(jnp.asarray(input_array))
    l = simple.Argmax.Config(name='argmax').make()
    l = self.init_and_bind_layer(key, l, x)

    y = l.layer(x, training=False)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), ())
    self.assertEqual(l.name, 'argmax')
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertAllEqual(y.values, jnp.array([[2], [0]]))


class SqueezeTest(test_utils.SequenceLayerTest):

  @parameterized.named_parameters(
      dict(
          testcase_name='float_input',
          input_array=np.array(
              [[[3]]],
              dtype=np.float32,
          ),
          expected_output=np.array([[3]]),
      ),
      dict(
          testcase_name='int_input',
          input_array=np.array(
              [[[3]]],
              dtype=np.int32,
          ),
          expected_output=np.array([[3]], dtype=np.int32),
      ),
      dict(
          testcase_name='no_op_input',
          input_array=np.array(
              [[3]],
              dtype=np.float32,
          ),
          expected_output=np.array([[3]]),
      ),
      dict(
          testcase_name='input_with_extra_dims',
          input_array=np.array(
              [[[[[3], [4]]]]],
              dtype=np.float32,
          ),
          expected_output=np.array([[[3, 4]]]),
      ),
  )
  def test_squeeze(
      self, input_array: jnp.ndarray, expected_output: jnp.ndarray
  ):
    key = jax.random.PRNGKey(1234)
    x = types.Sequence.from_values(input_array)
    l = simple.Squeeze.Config(name='squeeze').make()
    l = self.init_and_bind_layer(key, l, x)

    _ = l.layer(x, training=False)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), expected_output.shape[2:]
    )
    self.assertEqual(l.name, 'squeeze')
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

  @parameterized.parameters(
      ((2, 3, 1, 1, 1), 2, (1, 1)),
      ((2, 3, 1, 2, 1), -1, (1, 2)),
      ((2, 3, 1, 1, 1), (2,), (1, 1)),
      ((2, 3, 1, 2, 1), (-1,), (1, 2)),
      ((2, 3, 1, 2, 1), (2, -1), (2,)),
      ((2, 3, 1, 2, 1), None, (2,)),
  )
  def test_squeeze_axis(self, input_shape, axis, output_shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*input_shape)
    l = simple.Squeeze.Config(axis=axis, name='squeeze').make()
    l = self.init_and_bind_layer(key, l, x)

    y = l.layer(x, training=False)

    expected_output = x.apply_values_masked(lambda v: jnp.squeeze(v, axis=axis))

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    self.assertEqual(l.name, 'squeeze')
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertAllEqual(y.values, expected_output.values)


if __name__ == '__main__':
  test_utils.main()
