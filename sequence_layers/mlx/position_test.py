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
"""Tests for position encoding MLX sequence layers."""

from absl.testing import parameterized
import mlx.core as mx

from sequence_layers.mlx import test_utils
from sequence_layers.specs import position_behaviors


class AddTimingSignalTest(
    position_behaviors.AddTimingSignalTest,
    test_utils.SequenceLayerTest,
    parameterized.TestCase,
):

  @parameterized.product(
      param_dtype=(mx.float32, mx.float16),
      input_dtype=(mx.float32, mx.float16, mx.bfloat16),
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
    config = self.sl.AddTimingSignal.Config(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        trainable_scale=trainable_scale,
        param_dtype=param_dtype,
        name='add_timing_signal',
    )
    layer = self.make_layer(config)
    batch_size = 2
    x = self.random_sequence(batch_size, 1, *channel_shape, dtype=input_dtype)
    layer = self.init_layer(layer, x)

    # Check params dtype if trainable
    variables = self.get_variables(layer)
    params = variables.get('params', {}) if isinstance(variables, dict) else {}
    if trainable_scale:
      # In MLX, scale is a direct parameter attribute on the module if defined
      scale_param = getattr(layer, 'scale', None)
      self.assertIsNotNone(scale_param)
      self.assertEqual(scale_param.dtype, param_dtype)

    for time in range(13 * layer.block_size, 15 * layer.block_size):
      x = self.random_sequence(
          batch_size, time, *channel_shape, dtype=input_dtype
      )
      self.verify_contract(
          layer,
          x,
          training=False,
      )


class ApplyRotaryPositionalEncodingTest(
    position_behaviors.ApplyRotaryPositionalEncodingTest,
    test_utils.SequenceLayerTest,
    parameterized.TestCase,
):

  @parameterized.product(
      input_dtype=(mx.float32, mx.float16, mx.bfloat16),
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
    config = self.sl.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=max_wavelength,
        only_advance_position_for_valid_timesteps=only_advance_position_for_valid_timesteps,
        positions_in_at_least_fp32=positions_in_at_least_fp32,
        name='rope',
    )
    layer = self.make_layer(config)
    batch_size = 2
    x = self.random_sequence(batch_size, 1, *channel_shape, dtype=input_dtype)
    layer = self.init_layer(layer, x)
    for time in range(13 * layer.block_size, 15 * layer.block_size):
      x = self.random_sequence(
          batch_size,
          time,
          *channel_shape,
          random_mask=only_advance_position_for_valid_timesteps,
          dtype=input_dtype,
      )
      self.verify_contract(layer, x, training=False)


if __name__ == '__main__':
  parameterized.absltest.main()
