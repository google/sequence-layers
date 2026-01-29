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

from typing import Literal
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from sequence_layers.jax import position
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common
from sequence_layers.jax.attention import streaming_dot_product_attention
from sequence_layers.jax.attention import test_utils as attention_test_utils


class StreamingDotProductAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, 0, False),
      (3, 5, 3, 0, 0, False),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, 0, False),
      (3, 5, 3, 5, 0, False),
      # max_past_horizon > 0, max_future_horizon > 0, with attention sinks.
      (3, 5, 3, 5, 1, False),
      (3, 5, 3, 5, 0, True),
  )
  def test_streaming_local_dot_product_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      num_sink_embeddings,
      use_sink_scalars,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_channels = 2, 2
    source_name = 'source'

    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        per_dim_scale=True,
        precision=jax.lax.Precision.HIGHEST,
        name='streaming_dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
    ).make()

    source = test_utils.random_sequence(batch_size, 1, source_channels)
    constants = {source_name: source}
    channels = 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'streaming_dot_product_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
        input_projection=l.config.input_projection,
    )

    channels = 3
    for time in [11, 12]:
      # Source and x must be the same length.
      x = test_utils.random_sequence(
          batch_size, time, channels, low_length=time // 2
      )
      # Re-use x's random mask.
      source = types.Sequence(
          test_utils.random_sequence(
              batch_size, time, source_channels, random_lengths=False
          ).values,
          x.mask,
      ).mask_invalid()
      constants = {source_name: source}
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants=constants),
          (num_heads, units_per_head),
      )
      self.verify_contract(
          l,
          x,
          training=False,
          constants=constants,
          stream_constants=True,
          pad_constants=True,
      )

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      config=(
          dict(per_dim_scale=True),
          dict(attention_logits_soft_cap=50.0),
          dict(
              query_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
              key_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
          ),
      ),
  )
  def test_dtypes(self, param_dtype, input_dtype, compute_dtype, config):
    key = jax.random.PRNGKey(1234)
    batch_size, source_channels = 2, 2
    random_mask = True
    source_name = 'source'
    num_heads, units_per_head = 5, 2
    config = (
        dict(
            source_name='source',
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=1,
            max_future_horizon=3,
            precision=jax.lax.Precision.HIGHEST,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        | config
    )
    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        **config
    ).make()

    source = test_utils.random_sequence(
        batch_size,
        1,
        source_channels,
        random_mask=random_mask,
        dtype=input_dtype,
    )
    constants = {source_name: source}
    channels = 4
    x = test_utils.random_sequence(
        batch_size, 1, channels, random_mask=random_mask, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, constants=constants, input_projection=l.config.input_projection
    )

    time = 3
    # Source and x must be the same length.
    x = test_utils.random_sequence(
        batch_size,
        time,
        channels,
        random_mask=random_mask,
        low_length=time // 2,
        dtype=input_dtype,
    )
    # Re-use x's random mask.
    source = types.Sequence(
        test_utils.random_sequence(
            batch_size,
            time,
            source_channels,
            random_lengths=False,
            dtype=input_dtype,
        ).values,
        x.mask,
    ).mask_invalid()
    constants = {source_name: source}
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        (num_heads, units_per_head),
    )
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        stream_constants=True,
        pad_constants=True,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )

  @parameterized.product(
      config=(
          dict(per_dim_scale=True),
          dict(attention_logits_soft_cap=50.0),
          dict(
              query_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
              key_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
          ),
      ),
  )
  def test_bf16_mode(self, config):
    """Tests that when inputs / params are bfloat16, the output is bfloat16."""
    key = jax.random.PRNGKey(1234)
    batch_size, source_channels = 3, 4
    random_mask = True
    source_name = 'source'
    num_heads, units_per_head = 2, 6
    config = (
        dict(
            source_name='source',
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=1,
            max_future_horizon=3,
            precision=jax.lax.Precision.HIGHEST,
        )
        | config
    )
    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        **config
    ).make()

    time, channels = 5, 2
    source = test_utils.random_sequence(
        batch_size,
        time,
        source_channels,
        random_mask=random_mask,
        dtype=jnp.bfloat16,
    )
    constants = {source_name: source}
    x = test_utils.random_sequence(
        batch_size, time, channels, random_mask=random_mask, dtype=jnp.bfloat16
    )
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    bf16_params = test_utils.cast_from_to(
        l.variables, jnp.float32, jnp.bfloat16
    )
    l = l.bind(bf16_params)
    y = l.layer(x, training=False, constants=constants)
    self.assertEqual(y.dtype, jnp.bfloat16)

  @parameterized.parameters(True, False)
  def test_no_query_delay_buffer(self, use_rope: bool):
    key = jax.random.PRNGKey(1234)
    max_past_horizon, max_future_horizon = 2, 3
    batch_size, source_channels = 2, 2
    source_name = 'source'

    if use_rope:
      query_network = position.ApplyRotaryPositionalEncoding.Config(
          max_wavelength=10000,
          # Critical: RoPE positions must not advance for invalid timesteps.
          only_advance_position_for_valid_timesteps=True,
          positions_in_at_least_fp32=False,
      )
      key_network = position.ApplyRotaryPositionalEncoding.Config(
          max_wavelength=10000,
          # Critical: RoPE positions must not advance for invalid timesteps.
          only_advance_position_for_valid_timesteps=True,
          positions_in_at_least_fp32=False,
      )
    else:
      query_network = None
      key_network = None

    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        source_name,
        num_heads=3,
        units_per_head=6,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        use_query_delay_buffer=False,
        query_network=query_network,
        key_network=key_network,
        name='streaming_dot_product_attention',
    ).make()

    source = test_utils.random_sequence(batch_size, 1, source_channels)
    constants = {source_name: source}
    channels = 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    # When not using a query delay buffer, the layer has no input or output
    # latency.
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.output_latency, 0)

    time, channels = 30, 3
    # Source and x must be the same length.
    x = test_utils.random_sequence(
        batch_size, time, channels, low_length=time // 2
    )
    # Re-use x's random mask.
    source = types.Sequence(
        test_utils.random_sequence(
            batch_size, time, source_channels, random_lengths=False
        ).values,
        x.mask,
    ).mask_invalid()
    constants = {source_name: source}

    y_layer = l.layer(x, training=False, constants=constants)

    # Delay the input by max_future_horizon, and pad the constants at the end so
    # we can process the entire sequence.
    x = x.pad_time(max_future_horizon, 0, valid=False)
    constants = {
        source_name: source.pad_time(0, max_future_horizon, valid=False)
    }

    y_step, _, _ = utils.step_by_step_static(
        l,
        x,
        training=False,
        constants=constants,
        with_emits=False,
        stream_constants=True,
    )

    self.assertSequencesClose(
        y_layer.mask_invalid(), y_step[:, max_future_horizon:].mask_invalid()
    )

  def test_query_key_value_network_supports_step(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 1, 3)
    source = test_utils.random_sequence(2, 1, 5)
    constants = {'source': source}
    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertTrue(l.supports_step)

    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertFalse(l.supports_step)

    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    # The key/value network must be steppable for streaming.
    self.assertFalse(l.supports_step)

    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=test_utils.NonSteppableLayer.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    # The key/value network must be steppable for streaming.
    self.assertFalse(l.supports_step)

  @parameterized.product(
      (
          {
              'input_projection': common.SeparateQueryKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': common.QueryAndKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': common.QueryAndSharedKeyValueProjection(),
              'num_heads': 3,
          },
      ),
      attention_sinks=('none', 'embeddings', 'scalars'),
  )
  def test_projection_config(
      self,
      input_projection: (
          common.SeparateQueryKeyValueProjection
          | common.QueryAndKeyValueProjection
          | common.QueryAndSharedKeyValueProjection
      ),
      num_heads: int,
      attention_sinks: Literal['none', 'embeddings', 'scalars'],
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time, channels, source_channels = 2, 11, 3, 2
    source_name = 'source'
    units_per_head = 5
    max_past_horizon = 3
    max_future_horizon = 3
    num_sink_embeddings = 2 if attention_sinks == 'embeddings' else 0
    use_sink_scalars = attention_sinks == 'scalars'
    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        source_name,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        num_heads=num_heads,
        input_projection=input_projection,
        per_dim_scale=True,
        name='streaming_dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
    ).make()

    x = test_utils.random_sequence(batch_size, time, channels)
    # Re-use x's random mask.
    source = types.Sequence(
        test_utils.random_sequence(
            batch_size, time, source_channels, random_lengths=False
        ).values,
        x.mask,
    ).mask_invalid()
    constants = {source_name: source}

    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'streaming_dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
        input_projection=l.config.input_projection,
    )

    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        (num_heads, units_per_head),
    )
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        stream_constants=True,
        pad_constants=True,
    )

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0),
      (3, 5, 3, 0),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2),
      (3, 5, 3, 5),
  )
  def test_use_kv_cache_ringbuffer(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_channels = 2, 2
    source_name = 'source'
    l = streaming_dot_product_attention.StreamingDotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        per_dim_scale=True,
        precision=jax.lax.Precision.HIGHEST,
        name='streaming_dot_product_attention',
        use_kv_cache_ringbuffer=True,
    ).make()

    source = test_utils.random_sequence(batch_size, 1, source_channels)
    constants = {source_name: source}
    channels = 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'streaming_dot_product_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=0,
        input_projection=l.config.input_projection,
    )

    channels = 3
    for time in [11, 12]:
      # Source and x must be the same length.
      x = test_utils.random_sequence(
          batch_size, time, channels, low_length=time // 2
      )
      # Re-use x's random mask.
      source = types.Sequence(
          test_utils.random_sequence(
              batch_size, time, source_channels, random_lengths=False
          ).values,
          x.mask,
      ).mask_invalid()
      constants = {source_name: source}
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants=constants),
          (num_heads, units_per_head),
      )
      self.verify_contract(
          l,
          x,
          training=False,
          constants=constants,
          stream_constants=True,
          pad_constants=True,
          test_2x_step=False,
          grad_atol=1e-5,
          grad_rtol=1e-5,
      )


if __name__ == '__main__':
  test_utils.main()
