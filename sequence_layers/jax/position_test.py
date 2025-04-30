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
"""Tests for position layers."""

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import position as position_lib
from sequence_layers.jax import test_utils
from sequence_layers.jax import types

from google3.testing.pybase import parameterized


class AddTimingSignalTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(3,),
          axes=None,
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=False,
          channel_shape=(3,),
          axes=None,
      ),
      dict(
          min_timescale=10.0,
          max_timescale=1.0e5,
          trainable_scale=False,
          channel_shape=(3,),
          axes=0,
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(5, 9),
          axes=(1,),
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(5, 9, 3),
          axes=[1, 2],
      ),
      dict(
          min_timescale=1.0,
          max_timescale=1.0e4,
          trainable_scale=True,
          channel_shape=(5, 9),
          axes=(1,),
          only_advance_position_for_valid_timesteps=False,
      ),
  )
  def test_basic(
      self,
      min_timescale,
      max_timescale,
      trainable_scale,
      channel_shape,
      axes,
      only_advance_position_for_valid_timesteps=True,
  ):
    key = jax.random.PRNGKey(1234)
    l = position_lib.AddTimingSignal.Config(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        trainable_scale=trainable_scale,
        axes=axes,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
        name='add_timing_signal',
    ).make()

    batch_size = 8
    x = test_utils.random_sequence(batch_size, 1, *channel_shape)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'add_timing_signal')
    self.assertEqual(l.get_output_shape_for_sequence(x), x.shape[2:])

    unboxed_variables = flax.core.meta.unbox(l.variables)
    if trainable_scale:
      chex.assert_trees_all_equal_shapes_and_dtypes(
          unboxed_variables,
          {
              'params': {
                  'scale': jnp.zeros([]),
              }
          },
      )
    else:
      self.assertEmpty(jax.tree_util.tree_leaves(unboxed_variables))

    for time in range(13 * l.block_size, 15 * l.block_size):
      # Test non-contiguous masks to demonstrate that
      # only_advance_position_for_valid_timesteps works.
      x = test_utils.random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=True,
      )
      self.verify_contract(l, x, training=False, grad_atol=1e-5, grad_rtol=1e-5)

  @parameterized.parameters(
      dict(channel_shape=(2, 3), axes=-1, normalized_axes=(1,)),
      dict(channel_shape=(2, 3, 5), axes=[0, 2], normalized_axes=(0, 2)),
  )
  def test_timing_signal_along_axes(self, channel_shape, axes, normalized_axes):
    key = jax.random.PRNGKey(1234)
    layer = position_lib.AddTimingSignal.Config(
        axes=axes,
        name='add_timing_signal',
    ).make()

    batch_size = 2
    seq_len = 3
    inputs = types.Sequence.from_values(
        jnp.zeros((batch_size, seq_len, *channel_shape))
    )
    layer = self.init_and_bind_layer(key, layer, inputs)
    outputs = layer(inputs, training=False)
    outputs = np.asarray(outputs.values[0, -1])

    channel_dims = len(channel_shape)

    with self.subTest('equal_along_broadcasted_axes'):
      broadcast_slice_0 = tuple(
          slice(None) if axis in normalized_axes else 0
          for axis in range(channel_dims)
      )
      broadcast_slice_1 = tuple(
          slice(None) if axis in normalized_axes else 1
          for axis in range(channel_dims)
      )
      self.assertAllEqual(
          outputs[broadcast_slice_0], outputs[broadcast_slice_1]
      )

    with self.subTest('not_equal_over_all_axes'):
      complementary_slice_0 = tuple(
          0 if axis in normalized_axes else slice(None)
          for axis in range(channel_dims)
      )
      complementary_slice_1 = tuple(
          1 if axis in normalized_axes else slice(None)
          for axis in range(channel_dims)
      )
      self.assertNotAllEqual(
          outputs[complementary_slice_0], outputs[complementary_slice_1]
      )

  @parameterized.product(
      test_utils.standard_dtype_configs(param=True, input=True),
      trainable_scale=(False, True),
  )
  def test_dtypes(
      self,
      param_dtype,
      input_dtype,
      trainable_scale,
  ):
    channel_shape = (2, 3)
    min_timescale = 1.0
    max_timescale = 1.0e4
    key = jax.random.PRNGKey(1234)
    l = position_lib.AddTimingSignal.Config(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        trainable_scale=trainable_scale,
        param_dtype=param_dtype,
        name='add_timing_signal',
    ).make()

    batch_size = 2
    x = test_utils.random_sequence(
        batch_size, 1, *channel_shape, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)
    unboxed_variables = flax.core.meta.unbox(l.variables)
    if trainable_scale:
      chex.assert_trees_all_equal_shapes_and_dtypes(
          unboxed_variables,
          {
              'params': {
                  'scale': jnp.zeros([], dtype=param_dtype),
              }
          },
      )
    else:
      self.assertEmpty(jax.tree_util.tree_leaves(unboxed_variables))

    for time in range(13 * l.block_size, 15 * l.block_size):
      x = test_utils.random_sequence(
          batch_size, time, *channel_shape, dtype=input_dtype
      )
      self.verify_contract(
          l,
          x,
          training=False,
          **test_utils.get_grad_tols(l, x, param_dtype, input_dtype),
      )


class ApplyRotaryPositionalEncodingTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      max_wavelength=(1.0e4, 1.0e5),
      channel_shape=((4,), (3, 6)),
      only_advance_position_for_valid_timesteps=(False, True),
  )
  def test_basic(
      self,
      max_wavelength,
      channel_shape,
      only_advance_position_for_valid_timesteps,
  ):
    key = jax.random.PRNGKey(1234)
    l = position_lib.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=max_wavelength,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
        name='rope',
    ).make()

    batch_size = 2
    x = test_utils.random_sequence(batch_size, 1, *channel_shape)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'rope')
    self.assertEqual(l.get_output_shape_for_sequence(x), x.shape[2:])
    self.assertEmpty(jax.tree_util.tree_leaves(l.variables))

    for time in range(13 * l.block_size, 15 * l.block_size):
      x = test_utils.random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=only_advance_position_for_valid_timesteps,
      )
      self.verify_contract(l, x, training=False)

  @parameterized.product(
      test_utils.standard_dtype_configs(input=True),
      only_advance_position_for_valid_timesteps=(False, True),
      positions_in_at_least_fp32=(False, True),
  )
  def test_dtypes(
      self,
      input_dtype,
      only_advance_position_for_valid_timesteps,
      positions_in_at_least_fp32,
  ):
    max_wavelength = 1.0e4
    channel_shape = (2,)
    key = jax.random.PRNGKey(1234)
    l = position_lib.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=max_wavelength,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
        positions_in_at_least_fp32=positions_in_at_least_fp32,
        name='rope',
    ).make()

    batch_size = 2
    x = test_utils.random_sequence(
        batch_size, 1, *channel_shape, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)
    for time in range(13 * l.block_size, 15 * l.block_size):
      x = test_utils.random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=only_advance_position_for_valid_timesteps,
          dtype=input_dtype,
      )
      self.verify_contract(l, x, training=False)

  def test_only_advance_position_for_valid_timesteps(self):
    l = (
        position_lib.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=1.0e5,
            only_advance_position_for_valid_timesteps=True,
            name='rope',
        )
        .make()
        .bind({})
    )

    x = types.Sequence(
        jax.random.normal(jax.random.PRNGKey(1234), (3, 3, 6)),
        jnp.asarray(
            [[False, True, True], [True, False, True], [True, True, False]]
        ),
    ).mask_invalid()

    y = l.layer(x, training=False)

    # Verify the layer ignores invalid timesteps by showing the output is equal
    # to processing a sequence without the invalid timesteps.
    self.assertSequencesEqual(
        y[0:1, 1:],
        l.layer(x[0:1, 1:], training=False),
    )
    self.assertSequencesEqual(
        types.Sequence.concatenate_sequences([y[1:2, :1], y[1:2, 2:]]),
        l.layer(
            types.Sequence.concatenate_sequences([x[1:2, :1], x[1:2, 2:]]),
            training=False,
        ),
    )
    self.assertSequencesEqual(
        y[2:3, :-1],
        l.layer(x[2:3, :-1], training=False),
    )

  def test_external_positions(self):
    key = jax.random.PRNGKey(1234)
    l = position_lib.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=1.0e4,
        only_advance_position_for_valid_timesteps=False,
        positions_name='positions',
        name='rope',
    ).make()

    x = test_utils.random_sequence(1, 5, 8, random_lengths=False)
    x = types.Sequence.concatenate_sequences([x, x])
    constants = {
        'positions': types.Sequence.from_values(jnp.arange(10)[jnp.newaxis] % 5)
    }
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'rope')
    self.assertEqual(l.get_output_shape_for_sequence(x), x.shape[2:])
    self.assertEmpty(jax.tree_util.tree_leaves(l.variables))
    y = self.verify_contract(
        l,
        x,
        constants=constants,
        training=False,
        stream_constants=True,
        pad_constants=True,
    )
    # Since the positions repeat, the first half should equal the second half.
    self.assertSequencesEqual(y[:, :5], y[:, 5:])

  def test_error_only_advance_position_for_valid_timesteps_and_external_positions(
      self,
  ):
    with self.assertRaises(ValueError):
      key = jax.random.PRNGKey(1234)
      l = position_lib.ApplyRotaryPositionalEncoding.Config(
          max_wavelength=1.0e4,
          positions_name='positions',
          only_advance_position_for_valid_timesteps=True,
          name='rope',
      ).make()
      x = test_utils.random_sequence(1, 5, 8, random_lengths=False)
      x = types.Sequence.concatenate_sequences([x, x])
      constants = {
          'positions': types.Sequence.from_values(
              jnp.arange(10)[jnp.newaxis] % 5
          )
      }
      self.init_and_bind_layer(key, l, x, constants=constants)


if __name__ == '__main__':
  test_utils.main()
