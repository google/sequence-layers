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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from sequence_layers.jax import position
from sequence_layers.jax import test_utils
from sequence_layers.jax.attention import local_dot_product_self_attention
from sequence_layers.jax.attention import test_utils as attention_test_utils
from sequence_layers.jax.attention import transformer_xl_relative_position_embedding


class LocalDotProductSelfAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, False, 0, False),
      (1, 2, 3, 0, True, 0, False),
      (3, 5, 3, 0, False, 0, False),
      (3, 5, 3, 0, True, 0, False),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, False, 0, False),
      (3, 5, 3, 2, True, 0, False),
      (3, 5, 3, 5, False, 0, False),
      (3, 5, 3, 5, True, 0, False),
      # max_past_horizon > 0, max_future_horizon > 0, with attention sink.
      (3, 5, 3, 2, False, 1, False),
      (3, 5, 3, 2, True, 1, False),
      (3, 5, 3, 2, False, 0, True),
      (3, 5, 3, 2, True, 0, True),
  )
  def test_local_dot_product_self_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      random_mask,
      num_sink_embeddings,
      use_sink_scalars,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    block_size = max(1, max_future_horizon, max_past_horizon - 1)

    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=None,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        name='local_dot_product_self_attention',
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'local_dot_product_self_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
        input_projection=l.config.input_projection,
    )

    # Sweep time dimension shorter and longer than max_horizon.
    for time in [1, 2, 3, 11, 12]:
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(
            batch_size, time, channels, random_mask=random_mask
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
        )
        self.verify_contract(
            l, x, training=False, grad_atol=1e-5, grad_rtol=1e-5
        )

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, False),
      (3, 5, 3, 0, False),
      (1, 2, 3, 0, True),
      (3, 5, 3, 0, True),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, False),
      (3, 5, 3, 2, True),
      (3, 5, 3, 5, False),
      (3, 5, 3, 5, True),
  )
  def test_transformer_xl_relative_position_embeddings(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      random_mask,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    block_size = max(1, max_future_horizon, max_past_horizon - 1)
    transformer_xl_dim = 8
    relative_embedding = transformer_xl_relative_position_embedding.TransformerXLRelativePositionEmbedding.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_backward=max_past_horizon,
        max_forward=max_future_horizon,
        position_bias_dim=transformer_xl_dim,
    )

    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=relative_embedding,
        precision=jax.lax.Precision.HIGHEST,
        name='local_dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'local_dot_product_self_attention')
    # Only streamable with a relative_position_embedding when max_future_horizon
    # is zero.
    self.assertEqual(l.supports_step, not max_future_horizon)
    self.assertEqual(l.input_latency, max_future_horizon)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    # Sweep time dimension shorter and longer than max_horizon.
    for time in [1, 2, 3, 11, 12]:
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(
            batch_size, time, channels, random_mask=random_mask
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
        )
        self.verify_contract(
            l, x, training=False, grad_atol=1e-5, grad_rtol=1e-5
        )

  @parameterized.product(
      test_utils.standard_dtype_configs(param=True),
      test_utils.standard_dtype_configs(compute=True),
  )
  def test_transformer_xl_relative_position_embeddings_dtypes(
      self,
      param_dtype,
      compute_dtype,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    num_heads, units_per_head = 2, 1
    max_past_horizon, max_future_horizon = 3, 2
    block_size = max(1, max_future_horizon, max_past_horizon - 1)
    transformer_xl_dim = 8
    relative_embedding = transformer_xl_relative_position_embedding.TransformerXLRelativePositionEmbedding.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_backward=max_past_horizon,
        max_forward=max_future_horizon,
        position_bias_dim=transformer_xl_dim,
        param_dtype=param_dtype,
    )
    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=relative_embedding,
        precision=jax.lax.Precision.HIGHEST,
        compute_dtype=compute_dtype,
        name='local_dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels, random_mask=True)
    l = self.init_and_bind_layer(key, l, x)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    x = test_utils.random_sequence(
        batch_size,
        3,
        channels,
        random_mask=True,
    )
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )
    self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, False),
      (1, 2, 3, 0, True),
      (3, 6, 3, 0, False),
      (3, 6, 3, 0, True),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 6, 3, 2, False),
      (3, 6, 3, 2, True),
      (3, 6, 3, 5, False),
      (3, 6, 3, 5, True),
  )
  def test_rotary_positional_encoding(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      random_mask,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    block_size = max(1, max_future_horizon, max_past_horizon - 1)

    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=None,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        query_network=position.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10000
        ),
        key_network=position.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10000
        ),
        # Doesn't make sense but just to test the plumbing.
        value_network=position.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10000
        ),
        name='local_dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'local_dot_product_self_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    # Sweep time dimension shorter and longer than max_horizon.
    for time in [1, 2, 3, 11, 12]:
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(
            batch_size, time, channels, random_mask=random_mask
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
        )
        self.verify_contract(
            l,
            x,
            training=False,
            grad_atol=1e-5,
            grad_rtol=1e-5,
        )

  def test_query_key_value_network_supports_step(
      self,
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 1, 3)
    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        block_size=1,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)

    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        block_size=1,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        block_size=1,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        max_future_horizon=0,
        block_size=1,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=test_utils.NonSteppableLayer.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      config=(
          dict(per_dim_scale=True),
          dict(attention_logits_soft_cap=50.0),
          dict(max_past_horizon=2, max_future_horizon=0),
          dict(block_size=3),
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
    batch_size = 2
    random_mask = True
    defaults = dict(
        num_heads=3,
        units_per_head=4,
        max_past_horizon=3,
        max_future_horizon=3,
        block_size=1,
        relative_position_embedding=None,
        precision=jax.lax.Precision.HIGHEST,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )
    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        **(defaults | config)
    ).make()

    channels = 1
    x = test_utils.random_sequence(
        batch_size, 1, channels, random_mask=random_mask, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    x = test_utils.random_sequence(
        batch_size,
        7,
        channels,
        random_mask=random_mask,
        dtype=input_dtype,
    )
    self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )

  @parameterized.product(
      config=(
          dict(per_dim_scale=True),
          dict(attention_logits_soft_cap=50.0),
          dict(max_past_horizon=2, max_future_horizon=0),
          dict(block_size=3),
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
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    random_mask = True
    defaults = dict(
        num_heads=3,
        units_per_head=4,
        max_past_horizon=3,
        max_future_horizon=3,
        block_size=1,
        relative_position_embedding=None,
        precision=jax.lax.Precision.HIGHEST,
    )
    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        **(defaults | config)
    ).make()

    channels = 2
    x = test_utils.random_sequence(
        batch_size, 1, channels, random_mask=random_mask, dtype=jnp.bfloat16
    )
    l = self.init_and_bind_layer(key, l, x)

    bf16_params = test_utils.cast_from_to(
        l.variables, jnp.float32, jnp.bfloat16
    )
    l = l.bind(bf16_params)

    y = l.layer(x, training=False)
    self.assertEqual(y.dtype, jnp.bfloat16)

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, False),
      (1, 2, 3, 0, True),
      (3, 5, 3, 0, False),
      (3, 5, 3, 0, True),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, False),
      (3, 5, 3, 2, True),
      (3, 5, 3, 5, False),
      (3, 5, 3, 5, True),
  )
  def test_use_kv_cache_ringbuffer(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      random_mask,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    block_size = max(1, max_future_horizon, max_past_horizon - 1)

    l = local_dot_product_self_attention.LocalDotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=None,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        use_kv_cache_ringbuffer=True,
        name='local_dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'local_dot_product_self_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        num_sink_embeddings=0,
        input_projection=l.config.input_projection,
    )

    # Sweep time dimension shorter and longer than max_horizon.
    for time in [1, 2, 3, 11, 12]:
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(
            batch_size, time, channels, random_mask=random_mask
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
        )
        self.verify_contract(
            l,
            x,
            training=False,
            # Ring buffer does not support step size > 1.
            test_2x_step=False,
            grad_atol=1e-5,
            grad_rtol=1e-5,
        )


if __name__ == '__main__':
  test_utils.main()
