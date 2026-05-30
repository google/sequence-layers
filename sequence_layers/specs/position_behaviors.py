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
"""Shared behavior tests for position and timing layers."""

# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation

from absl.testing import parameterized
import numpy as np

from sequence_layers.specs import position as position_spec
from sequence_layers.specs import test_utils


class AddTimingSignalTest(test_utils.SequenceLayerTest):
  """Test behavior of AddTimingSignal layer."""

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
    config = self.sl.AddTimingSignal.Config(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        trainable_scale=trainable_scale,
        axes=axes,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
        name='add_timing_signal',
    )
    layer = self.make_layer(config)
    batch_size = 8
    x = self.random_sequence(batch_size, 1, *channel_shape)
    layer = self.init_layer(layer, x)

    self.assertEqual(layer.block_size, 1)
    self.assertEqual(layer.output_ratio, 1)
    self.assertEqual(layer.name, 'add_timing_signal')
    self.assertEqual(layer.get_output_shape(x.channel_shape), x.channel_shape)

    # Verify trainable scale presence in variables
    variables = self.get_variables(layer)
    if isinstance(variables, dict) and 'params' in variables:
      params = variables['params']
    else:
      params = variables
    if trainable_scale:
      self.assertIn('scale', params)
    else:
      self.assertNotIn('scale', params)

    for time in range(13 * layer.block_size, 15 * layer.block_size):
      x = self.random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=True,
      )
      self.verify_contract(layer, x, training=False)

  @parameterized.parameters(
      dict(channel_shape=(2, 3), axes=-1, normalized_axes=(1,)),
      dict(channel_shape=(2, 3, 5), axes=[0, 2], normalized_axes=(0, 2)),
  )
  def test_timing_signal_along_axes(self, channel_shape, axes, normalized_axes):
    config = self.sl.AddTimingSignal.Config(
        axes=axes,
        name='add_timing_signal',
    )
    layer = self.make_layer(config)
    batch_size = 2
    seq_len = 3
    inputs = self.sl.Sequence.from_values(
        self.xp.zeros((batch_size, seq_len, *channel_shape))
    )
    layer = self.init_layer(layer, inputs)
    outputs = layer.layer(inputs, training=False)
    outputs_np = np.asarray(outputs.values[0, -1])

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
          outputs_np[broadcast_slice_0], outputs_np[broadcast_slice_1]
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
          outputs_np[complementary_slice_0], outputs_np[complementary_slice_1]
      )


class ApplyRotaryPositionalEncodingTest(test_utils.SequenceLayerTest):
  """Test behavior of ApplyRotaryPositionalEncoding layer."""

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
    config = self.sl.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=max_wavelength,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
        name='rope',
    )
    layer = self.make_layer(config)
    batch_size = 2
    x = self.random_sequence(batch_size, 1, *channel_shape)
    layer = self.init_layer(layer, x)

    self.assertEqual(layer.block_size, 1)
    self.assertEqual(layer.output_ratio, 1)
    self.assertEqual(layer.name, 'rope')
    self.assertEqual(layer.get_output_shape(x.channel_shape), x.channel_shape)

    for time in range(13 * layer.block_size, 15 * layer.block_size):
      x = self.random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=only_advance_position_for_valid_timesteps,
      )
      self.verify_contract(layer, x, training=False)

  def test_only_advance_position_for_valid_timesteps(self):
    config = self.sl.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=1.0e5,
        only_advance_position_for_valid_timesteps=True,
        name='rope',
    )
    layer = self.make_layer(config)

    x = self.sl.Sequence(
        self.xp.array(np.random.normal(size=(3, 3, 6)).astype(np.float32)),
        self.xp.array(
            [[False, True, True], [True, False, True], [True, True, False]]
        ),
    ).mask_invalid()

    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    # Verify the layer ignores invalid timesteps by showing the output is equal
    # to processing a sequence without the invalid timesteps.
    self.assertSequencesClose(
        y[0:1, 1:],
        layer.layer(x[0:1, 1:], training=False),
    )
    self.assertSequencesClose(
        self.sl.Sequence.concatenate_sequences([y[1:2, :1], y[1:2, 2:]]),
        layer.layer(
            self.sl.Sequence.concatenate_sequences([x[1:2, :1], x[1:2, 2:]]),
            training=False,
        ),
    )
    self.assertSequencesClose(
        y[2:3, :-1],
        layer.layer(x[2:3, :-1], training=False),
    )

  def test_external_positions(self):
    config = self.sl.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=1.0e4,
        only_advance_position_for_valid_timesteps=False,
        positions_name='positions',
        name='rope',
    )
    layer = self.make_layer(config)

    x = self.random_sequence(1, 5, 8, random_lengths=False)
    x = self.sl.Sequence.concatenate_sequences([x, x])

    # Ensure position indices list is constructed via xp wrapper
    positions_arr = self.xp.array(np.arange(10)[np.newaxis] % 5)
    constants = {'positions': self.sl.Sequence.from_values(positions_arr)}
    layer = self.init_layer(layer, x, constants=constants)
    self.assertEqual(layer.block_size, 1)
    self.assertEqual(layer.output_ratio, 1)
    self.assertEqual(layer.name, 'rope')
    self.assertEqual(layer.get_output_shape(x.channel_shape), x.channel_shape)

    y = self.verify_contract(
        layer,
        x,
        constants=constants,
        training=False,
        stream_constants=True,
        pad_constants=True,
    )
    # Since the positions repeat, the first half should equal the second half.
    self.assertSequencesClose(y[:, :5], y[:, 5:])

  def test_error_only_advance_position_for_valid_timesteps_and_external_positions(
      self,
  ):
    config = self.sl.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=1.0e4,
        positions_name='positions',
        only_advance_position_for_valid_timesteps=True,
        name='rope',
    )
    layer = self.make_layer(config)
    x = self.random_sequence(1, 5, 8, random_lengths=False)
    x = self.sl.Sequence.concatenate_sequences([x, x])
    positions_arr = self.xp.array(np.arange(10)[np.newaxis] % 5)
    constants = {'positions': self.sl.Sequence.from_values(positions_arr)}
    with self.assertRaises(ValueError):
      self.init_layer(layer, x, constants=constants)
