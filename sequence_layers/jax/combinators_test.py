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
"""Combinator tests."""

import dataclasses

import chex
import flax
import flax.linen as nn
import jax
import jax._src.ad_checkpoint
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import attention
from sequence_layers.jax import combinators
from sequence_layers.jax import convolution
from sequence_layers.jax import dense
from sequence_layers.jax import normalization
from sequence_layers.jax import recurrent
from sequence_layers.jax import simple
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils

from google3.learning.gemini.gemax.core.models import sharding as sharding_lib
from google3.testing.pybase import parameterized


class SerialTest(test_utils.SequenceLayerTest):

  def test_noop(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config([]).make()

    batch_size, time, channels = 2, 5, 1
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.get_output_shape(x.channel_shape), (1,))
    self.verify_contract(l, x, training=False)
    self.assertEmpty(jax.tree_util.tree_leaves(l.variables))

  def test_serial(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='causal_valid',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='reverse_causal_valid',
                name='conv2',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='causal_valid',
                name='conv3',
            ),
        ],
        name='serial',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 8)
    self.assertEqual(1 / l.output_ratio, 8)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 1)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'serial')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'conv1': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv2': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv3': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
            }
        },
    )

    for i in range(2 * l.block_size):
      time = i + 1
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape(x.channel_shape), (1,))
      self.verify_contract(l, x, training=False)

  def test_serial_with_modules(self):
    key = jax.random.PRNGKey(1234)

    class SerialModule(combinators.SerialCombinatorMixin, types.Emitting):

      def setup(self):
        self.conv1 = convolution.Conv1D.Config(
            filters=1,
            kernel_size=2,
            strides=2,
            padding='causal_valid',
        ).make()
        self.conv2 = convolution.Conv1D.Config(
            filters=1,
            kernel_size=2,
            strides=2,
            padding='reverse_causal_valid',
        ).make()
        self.conv3 = convolution.Conv1D.Config(
            filters=1,
            kernel_size=2,
            strides=2,
            padding='causal_valid',
        ).make()
        self.layers = [self.conv1, self.conv2, self.conv3]

    l = SerialModule(name='serial')
    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 8)
    self.assertEqual(1 / l.output_ratio, 8)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 1)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'serial')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'conv1': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv2': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv3': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
            }
        },
    )

    for i in range(2 * l.block_size):
      time = i + 1
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape(x.channel_shape), (1,))
      self.verify_contract(l, x, training=False)

  def test_non_steppable(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config([
        convolution.Conv1D.Config(
            filters=1, kernel_size=2, strides=2, padding='causal_valid'
        ),
        test_utils.NonSteppableLayer.Config(),
    ]).make()
    x = test_utils.random_sequence(1, 1, 1)
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

  def test_constants(self):
    """Serial passes constants to its sublayers."""
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = combinators.Serial.Config(
        [test_utils.AssertConstantsLayer.Config()]
    ).make()
    constants = {'test': jnp.zeros(())}
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    with self.assertRaises(ValueError):
      l.get_initial_state(1, x.channel_spec, training=False)
    state = l.get_initial_state(
        1, x.channel_spec, training=False, constants=constants
    )
    with self.assertRaises(ValueError):
      l.layer(x, training=True)
    with self.assertRaises(ValueError):
      l.step(x, state, training=True)
    l.layer(x, training=True, constants=constants)
    l.step(x, state, training=True, constants=constants)

  def test_emits(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config([
        convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
        simple.Emit.Config(name='emit1'),
        convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
        simple.Emit.Config(name='emit2'),
        convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
        simple.Emit.Config(name='emit3'),
        convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
        simple.Emit.Config(name='emit4'),
    ]).make()
    x = test_utils.random_sequence(2, 16, 1)
    l = self.init_and_bind_layer(key, l, x)

    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    self.assertEmitsCompatible(emit_specs, emits)

    self.assertEqual(
        list(emits.keys()),
        [
            'layers_0',
            'emit1',
            'layers_2',
            'emit2',
            'layers_4',
            'emit3',
            'layers_6',
            'emit4',
        ],
    )
    self.assertEqual(emits['emit1'].values.shape, (2, 8, 1))
    self.assertEqual(emits['emit2'].values.shape, (2, 4, 1))
    self.assertEqual(emits['emit3'].values.shape, (2, 2, 1))
    self.assertEqual(emits['emit4'].values.shape, (2, 1, 1))
    self.assertEqual(y.values.shape, (2, 1, 1))

  def test_changes_shape_and_type(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config([
        dense.DenseShaped.Config([7, 11]),
        simple.Cast.Config(jnp.float16),
    ]).make()

    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(l, x, training=False)

    self.assertEqual(l.get_output_shape_for_sequence(x), (7, 11))
    self.assertEqual(l.get_output_dtype(x.dtype), jnp.float16)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

  def test_share_scope(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='causal_valid',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='reverse_causal_valid',
                name='conv2',
            ),
            combinators.Serial.Config(
                [
                    convolution.Conv1D.Config(
                        filters=1,
                        kernel_size=2,
                        strides=2,
                        padding='causal_valid',
                        name='conv3',
                    ),
                    dense.Dense.Config(1, name='dense'),
                ],
            ),
        ],
        share_scope=(True, False, True),
        name='serial',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 8)
    self.assertEqual(1 / l.output_ratio, 8)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 1)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'serial')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'bias': jnp.zeros((1,)),
                'kernel': jnp.zeros((2, 1, 1)),
                'conv2': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv3': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'dense': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((1, 1)),
                },
            }
        },
    )

  def test_share_scope_wrong_share_scope_length(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='causal',
                name='conv1',
            ),
            simple.Tanh.Config(),
        ],
        share_scope=(True, True, False),
        name='serial',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)

    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  def test_share_scope_param_overlap(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Serial.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='causal',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=2,
                padding='causal',
                name='conv2',
            ),
        ],
        share_scope=True,
        name='serial',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)

    with self.assertRaises(flax.errors.NameInUseError):
      self.init_and_bind_layer(key, l, x)


class ParallelTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      utils.CombinationMode.STACK,
      utils.CombinationMode.ADD,
      utils.CombinationMode.MEAN,
  )
  def test_noop(self, combination):
    key = jax.random.PRNGKey(1234)
    l = combinators.Parallel.Config([], combination=combination).make()

    batch_size, time, channels = 2, 5, 1
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.get_output_shape_for_sequence(x), (1,))
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

  @parameterized.parameters(
      utils.CombinationMode.STACK,
      utils.CombinationMode.ADD,
      utils.CombinationMode.MEAN,
  )
  def test_steppable(self, combination):
    key = jax.random.PRNGKey(1234)
    is_stack = combination == utils.CombinationMode.STACK
    l = combinators.Parallel.Config(
        [
            convolution.Conv1D.Config(
                filters=1, kernel_size=2, strides=2, padding='causal_valid'
            ),
            convolution.Conv1D.Config(
                filters=1, kernel_size=2, strides=2, padding='causal_valid'
            ),
            convolution.Conv1D.Config(
                filters=1, kernel_size=2, strides=2, padding='causal_valid'
            ),
        ],
        combination=combination,
    ).make()

    batch_size, time, channels = 2, 5, 1
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 2)
    self.assertEqual(1 / l.output_ratio, 2)
    self.assertTrue(l.supports_step)

    for time in range(5, 5 + 2 * l.block_size):
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          (3, 1) if is_stack else (1,),
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables['params'], 3)

  @parameterized.parameters(
      utils.CombinationMode.STACK,
      utils.CombinationMode.ADD,
      utils.CombinationMode.MEAN,
  )
  def test_not_steppable(self, combination):
    key = jax.random.PRNGKey(1234)
    is_stack = combination == utils.CombinationMode.STACK
    l = combinators.Parallel.Config(
        [
            dense.Dense.Config(1),
            convolution.Conv1D.Config(
                filters=1, kernel_size=2, strides=1, padding='same'
            ),
        ],
        combination=combination,
    ).make()

    batch_size, time, channels = 2, 5, 1
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertFalse(l.supports_step)

    batch_size, channels = 2, 1
    for time in range(5, 5 + 2 * l.block_size):
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x),
          (2, 1) if is_stack else (1,),
      )
      self.verify_contract(l, x, training=False)
      self.assertLen(l.variables['params'], 2)

  @parameterized.parameters(
      utils.CombinationMode.STACK,
      utils.CombinationMode.ADD,
      utils.CombinationMode.MEAN,
  )
  def test_broadcast(self, combination):
    key = jax.random.PRNGKey(1234)
    is_stack = combination == utils.CombinationMode.STACK
    l = combinators.Parallel.Config(
        [
            dense.DenseShaped.Config([1]),
            convolution.Conv1D.Config(
                filters=1, kernel_size=2, strides=1, padding='causal_valid'
            ),
            convolution.Conv1D.Config(
                filters=8, kernel_size=2, strides=1, padding='causal_valid'
            ),
        ],
        combination=combination,
    ).make()

    batch_size, time, channels = 2, 5, 1
    x = test_utils.random_sequence(batch_size, time, channels)
    # [1], [1], and [8] broadcast together to form [3, 8].

    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        (3, 8) if is_stack else (8,),
    )
    self.verify_contract(l, x, training=False)
    self.assertLen(l.variables['params'], 3)

  def test_dimension_change(self):
    # Apply a different layer on every index of the last channel.
    # Then, stack the results to obtain the same shape as the input.
    l = combinators.Parallel.Config(
        [
            simple.Slice.Config(slices=(0,)),
            simple.Slice.Config(slices=(1,)),
        ],
        combination=utils.CombinationMode.STACK,
    ).make()

    batch_size, time, channels = 3, 5, 2
    x = test_utils.random_sequence(batch_size, time, channels)

    key = jax.random.PRNGKey(1234)
    l = self.init_and_bind_layer(key, l, x)
    y = self.verify_contract(l, x, training=False)

    self.assertSequencesEqual(x, y)

  def test_invalid(self):
    # Different output ratios.
    l = combinators.Parallel.Config([
        convolution.Conv1D.Config(filters=1, kernel_size=2, strides=2),
        convolution.Conv1D.Config(filters=1, kernel_size=2, strides=1),
    ]).make()
    key = jax.random.PRNGKey(1234)
    batch_size, time, channels = 2, 10, 1
    x = test_utils.random_sequence(batch_size, time, channels)
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  @parameterized.parameters(
      utils.CombinationMode.STACK,
      utils.CombinationMode.ADD,
      utils.CombinationMode.MEAN,
  )
  def test_emits(self, combination):
    key = jax.random.PRNGKey(1234)
    is_stack = combination == utils.CombinationMode.STACK
    l = combinators.Parallel.Config(
        [
            combinators.Serial.Config([
                convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
                simple.Emit.Config(name='emit1'),
            ]),
            combinators.Serial.Config([
                convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
                simple.Emit.Config(name='emit2'),
            ]),
            combinators.Serial.Config([
                convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
                simple.Emit.Config(name='emit3'),
            ]),
            combinators.Serial.Config([
                convolution.Conv1D.Config(1, 1, 2, padding='causal_valid'),
                simple.Emit.Config(name='emit4'),
            ]),
        ],
        combination=combination,
    ).make()

    x = types.Sequence(jnp.zeros((2, 16, 1)), jnp.ones((2, 16), jnp.bool_))
    l = self.init_and_bind_layer(key, l, x)

    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    self.assertEmitsCompatible(emit_specs, emits)

    self.assertEqual(
        list(emits.keys()), ['layers_0', 'layers_1', 'layers_2', 'layers_3']
    )
    self.assertEqual(emits['layers_0']['emit1'].values.shape, (2, 8, 1))
    self.assertEqual(emits['layers_1']['emit2'].values.shape, (2, 8, 1))
    self.assertEqual(emits['layers_2']['emit3'].values.shape, (2, 8, 1))
    self.assertEqual(emits['layers_3']['emit4'].values.shape, (2, 8, 1))
    self.assertEqual(y.values.shape, (2, 8, 4, 1) if is_stack else (2, 8, 1))

  def test_share_scope(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Parallel.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv2',
            ),
            combinators.Serial.Config(
                [
                    convolution.Conv1D.Config(
                        filters=1,
                        kernel_size=2,
                        strides=1,
                        padding='causal',
                        name='conv3',
                    ),
                    dense.Dense.Config(1, name='dense'),
                ],
            ),
        ],
        share_scope=(True, False, True),
        name='parallel',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'parallel')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'bias': jnp.zeros((1,)),
                'kernel': jnp.zeros((2, 1, 1)),
                'conv2': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv3': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'dense': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((1, 1)),
                },
            }
        },
    )

  def test_share_scope_wrong_share_scope_length(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Parallel.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv1',
            ),
            simple.Tanh.Config(),
        ],
        share_scope=(True, True, False),
        name='parallel',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)

    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  def test_share_scope_param_overlap(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Parallel.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv2',
            ),
        ],
        share_scope=True,
        name='residual',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)

    with self.assertRaises(flax.errors.NameInUseError):
      self.init_and_bind_layer(key, l, x)


class ResidualTest(test_utils.SequenceLayerTest):

  def test_noop(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Residual.Config([]).make()

    batch_size, time, channels = 2, 5, 1
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.get_output_shape(x.channel_shape), (1,))
    self.verify_contract(l, x, training=False)
    self.assertEmpty(jax.tree_util.tree_leaves(l.variables))

  def test_residual(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Residual.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal_valid',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal_valid',
                name='conv2',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal_valid',
                name='conv3',
            ),
        ],
        shortcut_layers=[
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal_valid',
                name='shortcut_conv1',
            ),
        ],
        name='residual',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(1 / l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'residual')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'conv1': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv2': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv3': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'shortcut_layer': {
                    'shortcut_conv1': {
                        'bias': jnp.zeros((1,)),
                        'kernel': jnp.zeros((2, 1, 1)),
                    },
                },
            },
        },
    )

    for i in range(2 * l.block_size):
      time = i + 1
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(l.get_output_shape(x.channel_shape), (1,))
      self.verify_contract(l, x, training=False)

  def test_residual_function(self):
    key = jax.random.PRNGKey(1234)

    class ResidualFunction(combinators.Residual):

      @dataclasses.dataclass(frozen=True)
      class Config(combinators.Residual.Config):

        def make(self):
          return ResidualFunction(self)

      def residual_function(
          self, y_body: types.Sequence, y_shortcut: types.Sequence
      ) -> types.Sequence:
        return types.Sequence(
            y_body.values - y_shortcut.values,
            utils.combine_mask(y_body.mask, y_shortcut.mask),
        )

    l = ResidualFunction.Config(
        [simple.Scale.Config(5)],
        shortcut_layers=[simple.Scale.Config(3)],
        name='residual',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)
    y = self.verify_contract(l, x, training=False)

    self.assertSequencesClose(y, x.apply_values(lambda v: 2 * v).mask_invalid())

  def test_non_steppable(self):
    l = combinators.Residual.Config([
        convolution.Conv1D.Config(
            filters=1, kernel_size=2, strides=1, padding='causal_valid'
        ),
        test_utils.NonSteppableLayer.Config(),
    ]).make()
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(1, 1, 1)
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

  def test_output_ratio_mismatch(self):

    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(1, 1, 1)

    l = combinators.Residual.Config(
        [
            convolution.Conv1D.Config(
                filters=1, kernel_size=2, strides=2, padding='causal_valid'
            ),
        ],
        shortcut_layers=[
            convolution.Conv1D.Config(
                filters=1, kernel_size=2, strides=2, padding='causal_valid'
            ),
        ],
    ).make()
    self.init_and_bind_layer(key, l, x)

    l = combinators.Residual.Config([
        convolution.Conv1D.Config(
            filters=1, kernel_size=2, strides=2, padding='causal_valid'
        ),
    ]).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  def test_shortcut_latency(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(1, 1, 1)
    l = combinators.Residual.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=5,
                strides=3,
                padding='reverse_causal_valid',
            ),
        ],
        shortcut_layers=[
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=5,
                strides=3,
                padding='reverse_causal_valid',
            ),
        ],
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.input_latency, 4)
    self.assertEqual(int(l.output_latency), 1)

    l = combinators.Residual.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=5,
                strides=3,
                padding='reverse_causal_valid',
            ),
        ],
        shortcut_layers=[
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=3,
                strides=3,
                padding='reverse_causal_valid',
            )
        ],
    ).make()

    # Input latency of shortcut and body do not match.
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  def test_constants(self):
    """Residual passes constants to its sublayers."""
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = combinators.Residual.Config(
        [test_utils.AssertConstantsLayer.Config()],
        shortcut_layers=[test_utils.AssertConstantsLayer.Config()],
    ).make()
    constants = {'test': jnp.zeros(())}
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    with self.assertRaises(ValueError):
      l.get_initial_state(1, x.channel_spec, training=False)
    state = l.get_initial_state(
        1, x.channel_spec, training=False, constants=constants
    )
    with self.assertRaises(ValueError):
      l.layer(x, training=True)
    with self.assertRaises(ValueError):
      l.step(x, state, training=True)
    l.layer(x, training=True, constants=constants)
    l.step(x, state, training=True, constants=constants)

  def test_emits(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Residual.Config(
        [simple.Emit.Config(name='emit')],
        shortcut_layers=[simple.Emit.Config(name='shortcut_emit')],
    ).make()
    x = test_utils.random_sequence(2, 16, 1)
    l = self.init_and_bind_layer(key, l, x)
    y, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    self.assertEmitsCompatible(emit_specs, emits)

    body_emits, shortcut_emits = emits
    self.assertEqual(body_emits['emit'].values.shape, (2, 16, 1))
    self.assertEqual(shortcut_emits['shortcut_emit'].values.shape, (2, 16, 1))
    chex.assert_trees_all_equal(y.values, 2.0 * body_emits['emit'].values)

  def test_changes_shape_and_type(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Residual.Config(
        [
            dense.DenseShaped.Config([7, 11]),
            simple.Cast.Config(jnp.float16),
        ],
        shortcut_layers=[
            dense.DenseShaped.Config([7, 11]),
            simple.Cast.Config(jnp.float16),
        ],
    ).make()

    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(l, x, training=False)

    self.assertEqual(l.get_output_shape_for_sequence(x), (7, 11))
    self.assertEqual(l.get_output_dtype(x.dtype), jnp.float16)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

  @parameterized.parameters(
      (jnp.float32, jnp.float32, jnp.float32, jnp.float32),
      (jnp.float32, jnp.bfloat16, jnp.float32, jnp.float32),
      (jnp.float32, jnp.bfloat16, jnp.bfloat16, jnp.bfloat16),
      (jnp.int8, jnp.int32, jnp.int8, jnp.int32),
      (jnp.int8, jnp.float32, jnp.int8, jnp.float32),
  )
  def test_dtype_promotion(
      self, input_dtype, residual_dtype, shortcut_dtype, expected_output_dtype
  ):
    l = (
        combinators.Residual.Config(
            [
                simple.Cast.Config(residual_dtype),
            ],
            shortcut_layers=[
                simple.Cast.Config(shortcut_dtype),
            ],
        )
        .make()
        .bind({})
    )

    x = test_utils.random_sequence(2, 3, 5, dtype=input_dtype)
    y = l.layer(x, training=False)

    self.assertEqual(l.get_output_dtype(x.dtype), expected_output_dtype)
    self.assertEqual(y.dtype, expected_output_dtype)

  def test_share_scope(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Residual.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv2',
            ),
            combinators.Serial.Config(
                [
                    convolution.Conv1D.Config(
                        filters=1,
                        kernel_size=2,
                        strides=1,
                        padding='causal',
                        name='conv3',
                    ),
                    dense.Dense.Config(1, name='dense'),
                ],
            ),
        ],
        share_scope=(True, False, True),
        name='residual',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'residual')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'bias': jnp.zeros((1,)),
                'kernel': jnp.zeros((2, 1, 1)),
                'conv2': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'conv3': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((2, 1, 1)),
                },
                'dense': {
                    'bias': jnp.zeros((1,)),
                    'kernel': jnp.zeros((1, 1)),
                },
            }
        },
    )

  def test_share_scope_wrong_share_scope_length(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Residual.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv1',
            ),
            simple.Tanh.Config(),
        ],
        share_scope=(True, True, False),
        name='residual',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)

    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  def test_share_scope_param_overlap(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Residual.Config(
        [
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv1',
            ),
            convolution.Conv1D.Config(
                filters=1,
                kernel_size=2,
                strides=1,
                padding='causal',
                name='conv2',
            ),
        ],
        share_scope=True,
        name='residual',
    ).make()

    batch_size, channels = 2, 1
    x = test_utils.random_sequence(batch_size, 1, channels)

    with self.assertRaises(flax.errors.NameInUseError):
      self.init_and_bind_layer(key, l, x)


class BidirectionalTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      utils.CombinationMode.CONCAT,
      utils.CombinationMode.STACK,
      utils.CombinationMode.ADD,
      utils.CombinationMode.MEAN,
  )
  def test_bidirectional(self, combination):
    key = jax.random.PRNGKey(1234)
    l = combinators.Bidirectional.Config(
        recurrent.LSTM.Config(5, use_bias=False, name='forward'),
        recurrent.LSTM.Config(5, use_bias=False, name='backward'),
        combination=combination,
        name='bidirectional',
    ).make()

    batch_size, channels = 2, 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(1 / l.output_ratio, 1)
    self.assertFalse(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'bidirectional')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'forward': {
                    'kernel': {'kernel': jnp.zeros((3, 20))},
                    'recurrent_kernel': {'kernel': jnp.zeros((5, 20))},
                },
                'backward': {
                    'kernel': {'kernel': jnp.zeros((3, 20))},
                    'recurrent_kernel': {'kernel': jnp.zeros((5, 20))},
                },
            },
        },
    )

    match combination:
      case utils.CombinationMode.CONCAT:
        expected_output_shape = (10,)
      case utils.CombinationMode.STACK:
        expected_output_shape = (2, 5)
      case utils.CombinationMode.ADD:
        expected_output_shape = (5,)
      case utils.CombinationMode.MEAN:
        expected_output_shape = (5,)
      case _:
        raise NotImplementedError(f'Unsupported combination: {combination}')

    for i in range(2 * l.block_size):
      time = i + 1
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape(x.channel_shape), expected_output_shape
      )
      self.verify_contract(l, x, training=False)

  def test_output_ratio_mismatch(self):

    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(1, 1, 1)

    l = combinators.Bidirectional.Config(
        forward=convolution.Conv1D.Config(
            filters=1, kernel_size=2, strides=2, padding='causal_valid'
        ),
        backward=convolution.Conv1D.Config(
            filters=1, kernel_size=2, strides=2, padding='causal_valid'
        ),
    ).make()
    self.init_and_bind_layer(key, l, x)

    l = combinators.Bidirectional.Config(
        forward=convolution.Conv1D.Config(
            filters=1, kernel_size=2, strides=1, padding='causal_valid'
        ),
        backward=convolution.Conv1D.Config(
            filters=1, kernel_size=2, strides=2, padding='causal_valid'
        ),
    ).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  def test_constants(self):
    """Bidirectional passes constants to its sublayers."""
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = combinators.Bidirectional.Config(
        forward=test_utils.AssertConstantsLayer.Config(),
        backward=test_utils.AssertConstantsLayer.Config(),
    ).make()
    constants = {'test': jnp.zeros(())}
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    with self.assertRaises(ValueError):
      l.get_initial_state(1, x.channel_spec, training=False)
    state = l.get_initial_state(
        1, x.channel_spec, training=False, constants=constants
    )
    with self.assertRaises(ValueError):
      l.layer(x, training=True)
    with self.assertRaises(ValueError):
      l.step(x, state, training=True)
    l.layer(x, training=True, constants=constants)

  def test_emits(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Bidirectional.Config(
        forward=simple.Emit.Config(name='forward_emit'),
        backward=simple.Emit.Config(name='backward_emit'),
    ).make()
    x = test_utils.random_sequence(2, 16, 1)
    l = self.init_and_bind_layer(key, l, x)
    _, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    self.assertEmitsCompatible(emit_specs, emits)

    forward_emits, backward_emits = emits
    self.assertEqual(forward_emits.values.shape, (2, 16, 1))
    self.assertEqual(backward_emits.values.shape, (2, 16, 1))

  def test_changes_shape_and_type(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Bidirectional.Config(
        forward=combinators.Serial.Config([
            dense.DenseShaped.Config([7, 11]),
            simple.Cast.Config(jnp.float16),
        ]),
        backward=combinators.Serial.Config([
            dense.DenseShaped.Config([7, 11]),
            simple.Cast.Config(jnp.float16),
        ]),
    ).make()

    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)
    self.verify_contract(l, x, training=False)

    self.assertEqual(l.get_output_shape_for_sequence(x), (2, 7, 11))
    self.assertEqual(l.get_output_dtype(x.dtype), jnp.float16)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertFalse(l.supports_step)

  @parameterized.parameters(
      (jnp.float32, jnp.float32, jnp.float32, jnp.float32),
      (jnp.float32, jnp.bfloat16, jnp.float32, jnp.float32),
      (jnp.float32, jnp.bfloat16, jnp.bfloat16, jnp.bfloat16),
      (jnp.int8, jnp.int32, jnp.int8, jnp.int32),
      (jnp.int8, jnp.float32, jnp.int8, jnp.float32),
  )
  def test_dtype_promotion(
      self, input_dtype, forward_dtype, backward_dtype, expected_output_dtype
  ):
    l = (
        combinators.Bidirectional.Config(
            forward=simple.Cast.Config(forward_dtype),
            backward=simple.Cast.Config(backward_dtype),
        )
        .make()
        .bind({})
    )

    x = test_utils.random_sequence(2, 3, 5, dtype=input_dtype)
    y = l.layer(x, training=False)

    self.assertEqual(l.get_output_dtype(x.dtype), expected_output_dtype)
    self.assertEqual(y.dtype, expected_output_dtype)


class RepeatTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_repeat_dense(self, remat):
    key = jax.random.PRNGKey(1234)

    # Repeat a (stateless) dense layer.
    l = combinators.Repeat.Config(
        dense.Dense.Config(5, bias_init=nn.initializers.normal(), name='dense'),
        num_repeats=2,
        remat=remat,
        name='repeat',
    ).make()

    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'repeat')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'dense': {
                    'bias': jnp.zeros((2, 5)),
                    'kernel': jnp.zeros((2, 5, 5)),
                }
            },
        },
    )

    # Check the variables are not close (we use different RNGs for each
    # repeat).
    kernel = variables['params']['dense']['kernel']
    bias = variables['params']['dense']['bias']
    self.assertNotAllClose(kernel[0], kernel[1])
    self.assertNotAllClose(bias[0], bias[1])

    y = self.verify_contract(l, x, training=False)

    # Construct layer manually and verify that execution is the same as running
    # it serially with sliced variables.
    child = l.config.layer.make()
    y_manual = x
    child_params = {'params': l.variables['params']['dense']}
    for i in range(l.config.num_repeats):
      vars_i = jax.tree_util.tree_map(lambda v: v[i], child_params)  # pylint: disable=cell-var-from-loop
      y_manual = child.apply(
          vars_i, y_manual, training=False, method=child.layer
      )
    self.assertSequencesClose(y, y_manual.mask_invalid())

  def test_conv1d(self):
    key = jax.random.PRNGKey(1234)

    # Repeat a (stateful) strided conv layer.
    l = combinators.Repeat.Config(
        convolution.Conv1D.Config(
            5,
            kernel_size=3,
            strides=1,
            padding='reverse_causal_valid',
            bias_init=nn.initializers.normal(),
            name='conv',
        ),
        num_repeats=2,
        name='repeat',
    ).make()

    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 4)
    self.assertEqual(int(l.output_latency), 4)
    self.assertEqual(l.name, 'repeat')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'conv': {
                    'bias': jnp.zeros((2, 5)),
                    'kernel': jnp.zeros((2, 3, 5, 5)),
                }
            },
        },
    )

    # Check the variables are not close (we use different RNGs for each
    # repeat).
    kernel = variables['params']['conv']['kernel']
    bias = variables['params']['conv']['bias']
    self.assertNotAllClose(kernel[0], kernel[1])
    self.assertNotAllClose(bias[0], bias[1])

    y = self.verify_contract(l, x, training=False)

    # Construct layer manually and verify that execution is the same as running
    # it serially with sliced variables.
    child = l.config.layer.make()
    y_manual = x
    child_params = {'params': variables['params']['conv']}
    for i in range(l.config.num_repeats):
      vars_i = jax.tree_util.tree_map(lambda v: v[i], child_params)  # pylint: disable=cell-var-from-loop
      y_manual = child.apply(
          vars_i, y_manual, training=False, method=child.layer
      )
    self.assertSequencesClose(y, y_manual.mask_invalid())

  def test_dot_product_self_attention(self):
    key = jax.random.PRNGKey(1234)

    l = combinators.Repeat.Config(
        combinators.Serial.Config(
            [
                attention.DotProductSelfAttention.Config(
                    num_heads=2,
                    units_per_head=3,
                    max_past_horizon=10,
                    max_future_horizon=0,
                    per_dim_scale=True,
                    name='attention',
                ),
                simple.Flatten.Config(),
            ],
            name='serial',
        ),
        num_repeats=2,
        name='repeat',
    ).make()

    x = test_utils.random_sequence(2, 3, 6)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'repeat')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'serial': {
                    'attention': {
                        'query_key_value_projection': {
                            'kernel': jnp.zeros((2, 6, 3, 2, 3))
                        },
                        'per_dim_scale': jnp.zeros((2, 3)),
                    }
                },
            },
        },
    )

    # Check the variables are not close (we use different RNGs for each
    # repeat).
    qkv_kernel = variables['params']['serial']['attention'][
        'query_key_value_projection'
    ]['kernel']
    per_dim_scale = variables['params']['serial']['attention']['per_dim_scale']
    self.assertNotAllClose(qkv_kernel[0], qkv_kernel[1])
    # per_dim_scale is initialized to zero, so we do not test it.
    self.assertAllEqual(per_dim_scale[0], per_dim_scale[1])

    y = self.verify_contract(l, x, training=False)

    # Construct layer manually and verify that execution is the same as running
    # it serially with sliced variables.
    child = l.config.layer.make()
    y_manual = x
    child_params = {'params': variables['params']['serial']}
    for i in range(l.config.num_repeats):
      vars_i = jax.tree_util.tree_map(lambda v: v[i], child_params)  # pylint: disable=cell-var-from-loop
      y_manual = child.apply(
          vars_i, y_manual, training=False, method=child.layer
      )
    self.assertSequencesClose(y, y_manual.mask_invalid())

  def test_batch_norm(self):
    """Tests that variable collections other than "params" are scanned."""
    key = jax.random.PRNGKey(1234)

    # Repeat a (stateful) strided conv layer.
    l = combinators.Repeat.Config(
        normalization.BatchNormalization.Config(
            momentum=0.5,
            scale_init=nn.initializers.normal(),
            bias_init=nn.initializers.normal(),
            name='bn',
        ),
        num_repeats=2,
        name='repeat',
    ).make()

    x = test_utils.random_sequence(2, 16, 5)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'repeat')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'bn': {
                    'bn': {
                        'bias': jnp.zeros((2, 5)),
                        'scale': jnp.zeros((2, 5)),
                    }
                }
            },
            'batch_stats': {
                'bn': {
                    'bn': {
                        'mean': jnp.zeros((2, 5)),
                        'var': jnp.zeros((2, 5)),
                    }
                }
            },
        },
    )

    # Check the variables are not close (we use different RNGs for each
    # repeat).
    bias = variables['params']['bn']['bn']['bias']
    scale = variables['params']['bn']['bn']['scale']
    mean = variables['batch_stats']['bn']['bn']['mean']
    var = variables['batch_stats']['bn']['bn']['var']
    self.assertNotAllClose(bias[0], bias[1])
    self.assertNotAllClose(scale[0], scale[1])
    self.assertAllEqual(mean, jnp.zeros_like(mean))
    self.assertAllEqual(var, jnp.ones_like(var))

    self.verify_contract(l, x, training=False)

    y, new_vars = l.apply(
        l.variables, x, training=True, mutable='batch_stats', method=l.layer
    )

    new_mean = new_vars['batch_stats']['bn']['bn']['mean']
    new_var = new_vars['batch_stats']['bn']['bn']['var']

    # Moving moments have updated.
    self.assertNotAllEqual(new_mean, jnp.zeros_like(mean))
    self.assertNotAllEqual(new_var, jnp.ones_like(var))

    # Construct layer manually and verify that execution is the same as running
    # it serially with sliced variables.
    child = l.config.layer.make()
    child_params = {
        'params': {
            'bn': {
                'scale': scale,
                'bias': bias,
            }
        },
        'batch_stats': {'bn': {'mean': mean, 'var': var}},
    }
    y_manual = x
    for i in range(l.config.num_repeats):
      vars_i = jax.tree_util.tree_map(lambda v: v[i], child_params)  # pylint: disable=cell-var-from-loop
      y_manual, new_vars_i = child.apply(
          vars_i,
          y_manual,
          training=True,
          method=child.layer,
          mutable='batch_stats',
      )
      # Updates when repeated are the same as when manually executed.
      self.assertAllClose(new_vars_i['batch_stats']['bn']['mean'], new_mean[i])
      self.assertAllClose(new_vars_i['batch_stats']['bn']['var'], new_var[i])

    self.assertSequencesClose(y.mask_invalid(), y_manual.mask_invalid())

    # Bind new variables and re-calculate in test mode.
    l, variables = l.unbind()
    variables = {
        'params': variables['params'],
        'batch_stats': new_vars['batch_stats'],
    }
    l = l.bind(variables)

    y_test = l.layer(x, training=False)

    y_manual_test = x
    child_params = {
        'params': {
            'bn': {
                'scale': scale,
                'bias': bias,
            }
        },
        'batch_stats': {'bn': {'mean': new_mean, 'var': new_var}},
    }
    for i in range(l.config.num_repeats):
      vars_i = jax.tree_util.tree_map(lambda v: v[i], child_params)  # pylint: disable=cell-var-from-loop
      y_manual_test = child.apply(
          vars_i,
          y_manual_test,
          training=False,
          method=child.layer,
      )

    self.assertSequencesClose(
        y_test.mask_invalid(), y_manual_test.mask_invalid()
    )

  def test_rng_splitting(self):
    """Test that RNGs are different for each repeat of a stochastic layer."""
    key = jax.random.PRNGKey(1234)

    # Repeat a stochastic dropout layer followed by Emit to test that the RNG is
    # different for each repeat.
    l = combinators.Repeat.Config(
        combinators.Serial.Config([
            simple.Dropout.Config(0.5, rng_collection='test'),
            simple.Emit.Config(),
        ]),
        num_repeats=2,
        name='repeat',
    ).make()

    # Use a full sequence so we have plenty of valid timesteps for the below
    # test.
    x = types.Sequence(jnp.ones((2, 16, 5)), jnp.ones((2, 16), dtype=jnp.bool_))
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'repeat')
    self.assertEmpty(l.variables)

    _, emits = l.apply(
        l.variables,
        x,
        training=True,
        rngs={'test': jax.random.PRNGKey(0)},
        method=l.layer_with_emits,
    )

    dropout1, dropout2 = jax.tree_util.tree_leaves(
        emits, is_leaf=lambda x: isinstance(x, types.Sequence)
    )
    # We want to test the dropout masks were different same, so un-do the
    # scaling by the dropout rate to compensate for masked values.
    dropout1 = dropout1.apply_values(lambda v: v // 2)
    dropout2 = dropout2.apply_values(lambda v: v // 4)
    np.testing.assert_allclose(np.mean(dropout1.values), 0.5, atol=0.1)
    np.testing.assert_allclose(np.mean(dropout2.values), 0.25, atol=0.1)

    # dropout1 is the result of applying one dropout layer, and dropout2 is the
    # result of both dropout layers.
    self.assertSequencesNotClose(dropout1, dropout2)

  def test_nested_repeat(self):
    key = jax.random.PRNGKey(1234)

    # Repeat a (stateful) strided conv layer.
    conv_config = convolution.Conv1D.Config(
        5,
        kernel_size=3,
        strides=1,
        padding='causal_valid',
        bias_init=nn.initializers.normal(),
        name='conv',
    )
    l = combinators.Repeat.Config(
        combinators.Serial.Config(
            [
                combinators.Repeat.Config(
                    conv_config,
                    num_repeats=3,
                    name='repeat_inner',
                ),
                # One conv outside of a repeat.
                conv_config,
            ],
            name='serial',
        ),
        num_repeats=2,
        name='repeat_outer',
    ).make()

    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'repeat_outer')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'serial': {
                    'repeat_inner': {
                        'conv': {
                            'bias': jnp.zeros((2, 3, 5)),
                            'kernel': jnp.zeros((2, 3, 3, 5, 5)),
                        }
                    },
                    'conv': {
                        'bias': jnp.zeros((2, 5)),
                        'kernel': jnp.zeros((2, 3, 5, 5)),
                    },
                }
            },
        },
    )

    # Check the variables are not close (we use different RNGs for each
    # repeat).
    inner_conv_kernel = variables['params']['serial']['repeat_inner']['conv'][
        'kernel'
    ]
    inner_conv_bias = variables['params']['serial']['repeat_inner']['conv'][
        'bias'
    ]
    self.assertNotAllClose(inner_conv_kernel[0], inner_conv_kernel[1])
    self.assertNotAllClose(inner_conv_kernel[0, 0], inner_conv_kernel[0, 1])
    self.assertNotAllClose(inner_conv_bias[0], inner_conv_bias[1])
    self.assertNotAllClose(inner_conv_bias[0], inner_conv_bias[1])
    self.assertNotAllClose(inner_conv_bias[0, 0], inner_conv_bias[0, 1])
    outer_conv_kernel = variables['params']['serial']['conv']['kernel']
    outer_conv_bias = variables['params']['serial']['conv']['bias']
    self.assertNotAllClose(outer_conv_kernel[0], outer_conv_kernel[1])
    self.assertNotAllClose(outer_conv_bias[0], outer_conv_bias[1])

    y = self.verify_contract(l, x, training=False)

    # Execute the conv manually with sliced variables to verify the output is
    # equivalent.
    child = conv_config.make()
    y_manual = x
    for i in range(2):
      for j in range(3):
        vars_i = {
            'params': {
                'kernel': inner_conv_kernel[i][j],
                'bias': inner_conv_bias[i][j],
            }
        }
        y_manual = child.apply(
            vars_i, y_manual, training=False, method=child.layer
        )
      vars_i = {
          'params': {
              'kernel': outer_conv_kernel[i],
              'bias': outer_conv_bias[i],
          }
      }
      y_manual = child.apply(
          vars_i, y_manual, training=False, method=child.layer
      )

    self.assertSequencesClose(y, y_manual.mask_invalid())

  @parameterized.product(
      unroll_layer=(True, False),
      unroll_step=(True, False),
  )
  def test_unroll(self, unroll_layer: bool, unroll_step: bool):
    key = jax.random.PRNGKey(1234)

    l = combinators.Repeat.Config(
        combinators.Serial.Config(
            [
                attention.DotProductAttention.Config(
                    source_name='source',
                    num_heads=2,
                    units_per_head=3,
                    name='attention',
                ),
                simple.Flatten.Config(),
            ],
            name='serial',
        ),
        num_repeats=2,
        unroll_layer=unroll_layer,
        unroll_step=unroll_step,
        name='repeat',
    ).make()

    x = test_utils.random_sequence(2, 3, 6, low_length=2)
    source = test_utils.random_sequence(2, 5, 1)
    constants = {'source': source}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'repeat')

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'serial': {
                    'attention': {
                        'query_projection': {'kernel': jnp.zeros((2, 6, 2, 3))},
                        'key_value_projection': {
                            'kernel': jnp.zeros((2, 1, 2, 2, 3))
                        },
                    }
                },
            },
        },
    )

    # Run with a mesh and under jit to verify that sharding annotations are
    # correctly modified when unrolling.
    mesh = jax.sharding.Mesh(
        jax.experimental.mesh_utils.create_device_mesh(
            (1, 1, 1, 1),
            devices=jax.local_devices(),
        ),
        ('replica', 'data', 'seq', 'model'),
    )
    with sharding_lib.set_global_mesh(mesh):
      y = self.verify_contract(
          l, x, training=False, constants=constants, jit=True
      )

    # Construct layer manually and verify that execution is the same as running
    # it serially with sliced variables.
    child = l.config.layer.make()
    y_manual = x
    child_params = {'params': variables['params']['serial']}
    for i in range(l.config.num_repeats):
      vars_i = jax.tree_util.tree_map(lambda v: v[i], child_params)  # pylint: disable=cell-var-from-loop
      y_manual = child.apply(
          vars_i,
          y_manual,
          training=False,
          constants=constants,
          method=child.layer,
      )
    self.assertSequencesClose(y, y_manual.mask_invalid())


class CheckpointGradientTest(test_utils.SequenceLayerTest):
  """Tests for CheckpointGradient.

  TODO(rryan): Test memory usage with statix.
  """

  def test_checkpoint_dense(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.CheckpointGradient.Config(
        convolution.Conv1D.Config(
            7,
            kernel_size=5,
            strides=2,
            padding='reverse_causal_valid',
            name='conv',
        )
    ).make()
    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 4)
    self.assertEqual(int(l.output_latency), 2)
    self.assertEqual(l.block_size, 2)
    self.assertEqual(l.output_ratio, 1 / 2)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'conv': {
                    'bias': jnp.zeros((7,)),
                    'kernel': jnp.zeros((5, 5, 7)),
                },
            }
        },
    )

    self.verify_contract(l, x, training=False)

  def test_checkpoint_repeat(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.CheckpointGradient.Config(
        combinators.Repeat.Config(
            dense.Dense.Config(5, name='dense'),
            num_repeats=2,
            name='repeat',
        )
    ).make()
    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'repeat': {
                    'dense': {
                        'bias': jnp.zeros((
                            2,
                            5,
                        )),
                        'kernel': jnp.zeros((2, 5, 5)),
                    },
                },
            }
        },
    )

    self.verify_contract(l, x, training=False)

  def test_repeat_checkpoint(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.Repeat.Config(
        combinators.CheckpointGradient.Config(
            dense.Dense.Config(5, name='dense'),
            prevent_cse=False,
            name='checkpoint_gradient',
        ),
        num_repeats=2,
    ).make()
    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'checkpoint_gradient': {
                    'dense': {
                        'bias': jnp.zeros((
                            2,
                            5,
                        )),
                        'kernel': jnp.zeros((2, 5, 5)),
                    },
                },
            }
        },
    )

    self.verify_contract(l, x, training=False)

  def test_checkpoint_policy(self):
    """Test that checkpoint policy is applied to the checkpointed model."""
    key = jax.random.PRNGKey(1234)
    l = combinators.CheckpointGradient.Config(
        combinators.Serial.Config(
            [
                dense.Dense.Config(7, name='dense1'),
                simple.CheckpointName.Config('test'),
                dense.Dense.Config(9, name='dense2'),
            ],
            name='serial',
        ),
        policy=jax.checkpoint_policies.save_only_these_names('test'),
    ).make()
    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'serial': {
                    'dense1': {
                        'bias': jnp.zeros((7,)),
                        'kernel': jnp.zeros((5, 7)),
                    },
                    'dense2': {
                        'bias': jnp.zeros((9,)),
                        'kernel': jnp.zeros((7, 9)),
                    },
                }
            }
        },
    )
    # TODO(rryan): Test that the checkpoint policy was actually used.


class ParallelChannelsTest(
    test_utils.SequenceLayerTest, parameterized.TestCase
):
  """Tests for ParallelChannels."""

  @parameterized.product(
      combination=(
          utils.CombinationMode.STACK,
          utils.CombinationMode.CONCAT,
          utils.CombinationMode.ADD,
          utils.CombinationMode.MEAN,
      ),
      padding=('reverse_causal', 'same'),
  )
  def test_combination(self, combination, padding):
    key = jax.random.PRNGKey(1234)
    l = combinators.ParallelChannels.Config(
        convolution.Conv1D.Config(
            8, kernel_size=3, strides=2, padding=padding, name='conv'
        ),
        num_groups=3,
        combination=combination,
        name='parallel_channels',
    ).make()
    x = test_utils.random_sequence(2, 12, 9, low_length=6)
    l = self.init_and_bind_layer(key, l, x)

    match combination:
      case utils.CombinationMode.STACK:
        expected_shape = (3, 8)
      case utils.CombinationMode.CONCAT:
        expected_shape = (24,)
      case utils.CombinationMode.ADD:
        expected_shape = (8,)
      case utils.CombinationMode.MEAN:
        expected_shape = (8,)
      case _:
        raise ValueError(f'Unsupported combination: {combination}')
    self.assertEqual(l.get_output_shape_for_sequence(x), expected_shape)
    self.assertEqual(l.get_output_dtype(x.dtype), jnp.float32)

    self.assertEqual(l.block_size, 2)
    self.assertEqual(1 / l.output_ratio, 2)
    self.assertEqual(l.supports_step, padding == 'reverse_causal')
    if padding == 'reverse_causal':
      self.assertEqual(l.input_latency, 2)
      self.assertEqual(int(l.output_latency), 1)
    self.assertEqual(l.name, 'parallel_channels')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'bias': jnp.zeros((8,)),
                'kernel': jnp.zeros((3, 3, 8)),
            }
        },
    )

    self.verify_contract(l, x, training=False)

  @parameterized.parameters(
      utils.CombinationMode.STACK,
      utils.CombinationMode.CONCAT,
      utils.CombinationMode.ADD,
      utils.CombinationMode.MEAN,
  )
  def test_2d_output(self, combination):
    key = jax.random.PRNGKey(1234)
    l = combinators.ParallelChannels.Config(
        dense.DenseShaped.Config([], name='dense'),
        num_groups=3,
        combination=combination,
        name='parallel_channels',
    ).make()
    x = test_utils.random_sequence(2, 12, 9, low_length=6)
    l = self.init_and_bind_layer(key, l, x)

    match combination:
      case utils.CombinationMode.STACK:
        expected_shape = (3,)
      case utils.CombinationMode.CONCAT:
        expected_shape = (3,)
      case utils.CombinationMode.ADD:
        expected_shape = ()
      case utils.CombinationMode.MEAN:
        expected_shape = ()
      case _:
        raise ValueError(f'Unsupported combination: {combination}')
    self.assertEqual(l.get_output_shape_for_sequence(x), expected_shape)
    self.assertEqual(l.get_output_dtype(x.dtype), jnp.float32)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(int(l.output_latency), 0)
    self.assertEqual(l.name, 'parallel_channels')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'bias': jnp.zeros((1,)),
                'kernel': jnp.zeros((3, 1)),
            }
        },
    )

    self.verify_contract(l, x, training=False)

  def test_wrong_channel_shape(self):
    key = jax.random.PRNGKey(1234)
    l = combinators.ParallelChannels.Config(
        convolution.Conv1D.Config(
            8, kernel_size=3, strides=2, padding='reverse_causal', name='conv'
        ),
        num_groups=3,
        name='parallel_channels',
    ).make()

    with self.assertRaises(ValueError):
      x = test_utils.random_sequence(2, 12)
      self.init_and_bind_layer(key, l, x)

    with self.assertRaises(ValueError):
      x = test_utils.random_sequence(2, 12, 8)
      self.init_and_bind_layer(key, l, x)


if __name__ == '__main__':
  test_utils.main()
