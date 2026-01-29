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
import dataclasses
from typing import Literal
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from sequence_layers.jax import position
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax.attention import common
from sequence_layers.jax.attention import dot_product_self_attention
from sequence_layers.jax.attention import shaw_relative_position_embedding
from sequence_layers.jax.attention import t5_relative_position_embedding
from sequence_layers.jax.attention import test_utils as attention_test_utils


class DotProductSelfAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0. Steppable.
      (1, 2, 3, 0, False, 0, False),
      (1, 2, 3, 0, True, 0, False),
      (3, 5, 3, 0, False, 0, False),
      (3, 5, 3, 0, True, 0, False),
      # max_past_horizon > 0, max_future_horizon > 0. Steppable.
      (3, 5, 3, 2, False, 0, False),
      (3, 5, 3, 2, True, 0, False),
      (3, 5, 3, 5, False, 0, False),
      (3, 5, 3, 5, True, 0, False),
      # max_past_horizon == -1, max_future_horizon > 0. Not steppable.
      (3, 5, -1, 2, False, 0, False),
      (3, 5, -1, 2, True, 0, False),
      # max_past_horizon > 0, max_future_horizon == -1. Not steppable.
      (3, 5, 3, -1, False, 0, False),
      (3, 5, 3, -1, True, 0, False),
      # max_past_horizon == -1, max_future_horizon == -1. Not steppable.
      (3, 5, -1, -1, False, 0, False),
      (3, 5, -1, -1, True, 0, False),
      # max_past_horizon > 0, max_future_horizon > 0. Steppable with sink
      # attention.
      (3, 5, 3, 2, False, 1, False),
      (3, 5, 3, 2, True, 1, False),
      (3, 5, 3, 2, False, 0, True),
      (3, 5, 3, 2, True, 0, True),
  )
  def test_dot_product_self_attention(
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
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        name='dot_product_self_attention',
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )
    self.assertEqual(
        l.supports_step, max_past_horizon >= 0 and max_future_horizon >= 0
    )
    self.assertEqual(l.input_latency, max(0, max_future_horizon))

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
        self.verify_contract(
            l,
            x,
            training=False,
            grad_atol=1e-5,
            grad_rtol=1e-5,
        )

  @parameterized.parameters(
      # Fully unmasked.
      (
          -1,
          -1,
          5,
          5,
      ),
      # Causal, fully unmasked past.
      (-1, 0, 8, 0),
      (-1, 0, 8, 8),
      # No past, fully unmasked future.
      (0, -1, 4, 4),
      # Limited past and future.
      (4, 4, 4, 4),
      (4, 4, 3, 2),
      # Causal, limited past context (streaming capable).
      (8, 0, 8, 0),  # Causal / streaming.
  )
  def test_shaw_relative_position_bias(
      self,
      max_past_horizon: int,
      max_future_horizon: int,
      max_backward: int,
      max_forward: int,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, num_heads, units_per_head = 2, 3, 5
    relative_embedding = (
        shaw_relative_position_embedding.ShawRelativePositionEmbedding.Config(
            max_forward=max_backward,
            max_backward=max_forward,
            num_heads=num_heads,
            units_per_head=units_per_head,
        )
    )
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=relative_embedding,
        name='dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    for time in [21, 22]:
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(batch_size, time, channels)
        self.verify_contract(
            l, x, training=False, grad_rtol=1e-5, grad_atol=1e-5
        )

  @parameterized.product(
      test_utils.standard_dtype_configs(
          param=True,
          compute=True,
          compute_dtype='inner_compute_dtype',
      ),
      test_utils.standard_dtype_configs(compute=True),
  )
  def test_shaw_relative_position_bias_dtypes(
      self,
      param_dtype,
      inner_compute_dtype,
      compute_dtype,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, num_heads, units_per_head = 2, 3, 5
    max_past_horizon, max_future_horizon = 2, 2
    max_backward = 2
    max_forward = 1
    relative_embedding = (
        shaw_relative_position_embedding.ShawRelativePositionEmbedding.Config(
            max_forward=max_forward,
            max_backward=max_backward,
            num_heads=num_heads,
            units_per_head=units_per_head,
            compute_dtype=inner_compute_dtype,
            param_dtype=param_dtype,
        )
    )
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=relative_embedding,
        compute_dtype=compute_dtype,
        name='dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    time = 8
    x = test_utils.random_sequence(batch_size, time, channels)
    self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )

  @parameterized.parameters(
      # Fully unmasked.
      (-1, -1, 8, True, 16),
      (-1, -1, 8, False, 16),
      # Causal, fully unmasked past.
      (-1, 0, 8, True, 16),
      (-1, 0, 8, False, 16),
      # No past, fully unmasked future.
      (0, -1, 8, True, 16),
      # Limited past and future.
      (4, 4, 8, True, 16),
      (4, 4, 8, False, 16),
      # Causal, limited past context (streaming capable).
      (8, 0, 8, False, 16),  # Causal / streaming.
  )
  def test_t5_relative_position_bias(
      self,
      max_past_horizon,
      max_future_horizon,
      num_buckets,
      bidirectional,
      max_distance,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, num_heads, units_per_head = 2, 3, 5
    relative_embedding = (
        t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
            num_buckets=num_buckets,
            bidirectional=bidirectional,
            max_distance=max_distance,
            num_heads=num_heads,
        )
    )
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=relative_embedding,
        name='dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    for time in [21, 22]:
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(batch_size, time, channels)
        self.verify_contract(
            l, x, training=False, grad_rtol=1e-5, grad_atol=1e-5
        )

  @parameterized.product(
      test_utils.standard_dtype_configs(
          param=True,
          compute=True,
          compute_dtype='inner_compute_dtype',
      ),
      test_utils.standard_dtype_configs(compute=True),
      bidirectional=(True, False),
  )
  def test_t5_relative_position_bias_dtypes(
      self,
      param_dtype,
      inner_compute_dtype,
      compute_dtype,
      bidirectional,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, num_heads, units_per_head = 2, 3, 5
    max_past_horizon, max_future_horizon, num_buckets = 2, 2, 4
    max_distance = 7
    relative_embedding = (
        t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
            num_buckets=num_buckets,
            bidirectional=bidirectional,
            max_distance=max_distance,
            num_heads=num_heads,
            compute_dtype=inner_compute_dtype,
            param_dtype=param_dtype,
        )
    )
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        relative_position_embedding=relative_embedding,
        compute_dtype=compute_dtype,
        name='dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    time = 8
    x = test_utils.random_sequence(batch_size, time, channels)
    self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )

  @parameterized.parameters(True, False)
  def test_dropout(self, broadcast_dropout_across_queries):
    key = jax.random.PRNGKey(1234)
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=8,
        units_per_head=3,
        max_past_horizon=-1,
        max_future_horizon=-1,
        attention_probabilities_dropout_rate=0.99999,
        broadcast_dropout_across_queries=broadcast_dropout_across_queries,
    ).make()
    x = test_utils.random_sequence(2, 12, 3, random_mask=True)
    l = self.init_and_bind_layer(key, l, x)
    rngs = {'dropout': jax.random.key(1)}

    y_dropout = l.apply(l.variables, x, training=True, rngs=rngs)
    y_no_dropout = l.apply(l.variables, x, training=False, rngs=rngs)
    y_no_dropout2 = l.apply(l.variables, x, training=False, rngs=rngs)

    chex.assert_trees_all_equal(
        y_dropout.values, jnp.zeros_like(y_dropout.values)
    )
    self.assertSequencesClose(y_no_dropout, y_no_dropout2)

  def test_emit_outputs(self):
    key = jax.random.PRNGKey(1234)
    num_heads, units_per_head, max_past_horizon = 3, 5, 10
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
    ).make()

    batch_size, time, channels = 2, 15, 11
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    _, emits = l.layer_with_emits(x, training=False)
    self.assertEqual(
        emits.probabilities.values.shape,
        (batch_size, time, num_heads, time),
    )

    # Only run three timesteps of the sequence. The attention pdfs
    # are 3 + max_horizon.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(
        x,
        l.get_initial_state(batch_size, x.channel_spec, training=False),
        training=False,
    )
    self.assertEqual(
        emits.probabilities.values.shape,
        (batch_size, 3, num_heads, 3 + max_past_horizon),
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
      # max_past_horizon == -1, max_future_horizon > 0
      (3, 6, -1, 2, False),
      (3, 6, -1, 2, True),
      # max_past_horizon > 0, max_future_horizon == -1
      (3, 6, 3, -1, False),
      (3, 6, 3, -1, True),
      # max_past_horizon == -1, max_future_horizon == -1 (unmasked)
      (3, 6, -1, -1, False),
      (3, 6, -1, -1, True),
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
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
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
        name='dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    # Sweep time dimension shorter and longer than max_horizon.
    for time in [1, 2, 3, 11, 12]:
      with self.subTest(f'time{time}'):
        x = test_utils.random_sequence(
            batch_size, time, channels, random_mask=random_mask
        )
        self.verify_contract(
            l,
            x,
            training=False,
            grad_atol=1e-5,
            grad_rtol=1e-5,
        )

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      config=(
          dict(per_dim_scale=True),
          dict(attention_logits_soft_cap=50.0),
          dict(max_future_horizon=0),
          dict(max_past_horizon=3, max_future_horizon=3),
          dict(max_past_horizon=0, max_future_horizon=-1),
          dict(
              query_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
              key_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
              # RoPE requires even embeddings:
              units_per_head=4,
          ),
      ),
  )
  def test_dtypes(self, param_dtype, input_dtype, compute_dtype, config):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    random_mask = True
    defaults = dict(
        num_heads=4,
        units_per_head=3,
        max_past_horizon=-1,
        precision=jax.lax.Precision.HIGHEST,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        **(defaults | config)
    ).make()

    channels = 2
    x = test_utils.random_sequence(
        batch_size, 1, channels, random_mask=random_mask, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection
    )

    x = test_utils.random_sequence(
        batch_size,
        3,
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
          dict(max_future_horizon=0),
          dict(max_past_horizon=3, max_future_horizon=3),
          dict(max_past_horizon=0, max_future_horizon=-1),
          dict(
              query_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
              key_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=10000
              ),
              # RoPE requires even embeddings:
              units_per_head=4,
          ),
      )
  )
  def test_bf16_mode(self, config):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    random_mask = True
    defaults = dict(
        num_heads=4,
        units_per_head=3,
        max_past_horizon=-1,
        precision=jax.lax.Precision.HIGHEST,
    )
    l = dot_product_self_attention.DotProductSelfAttention.Config(
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

  def test_query_key_value_network_supports_step(
      self,
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 1, 3)
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)

    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=test_utils.NonSteppableLayer.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

  @parameterized.product(
      (
          # CombinedQueryKeyValueProjection. GQA is not supported.
          {  # MHA, no sharing.
              'input_projection': common.CombinedQueryKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # MHA, sharing supported.
              'input_projection': common.CombinedQueryKeyValueProjection(
                  share_kv_projection=True
              ),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          # SeparateQueryKeyValueProjection. MHA and GQA supported.
          {  # MHA, sharing not supported.
              'input_projection': common.SeparateQueryKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # GQA, sharing not supported.
              'input_projection': common.SeparateQueryKeyValueProjection(),
              'num_heads': 6,
              'num_kv_heads': 3,
          },
          # QueryAndKeyValueProjection. MHA and GQA supported.
          {  # MHA, sharing not supported.
              'input_projection': common.QueryAndKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # GQA, sharing not supported.
              'input_projection': common.QueryAndKeyValueProjection(),
              'num_heads': 6,
              'num_kv_heads': 3,
          },
          # QueryAndSharedKeyValueProjection. MHA and GQA supported.
          {  # MHA, sharing required.
              'input_projection': common.QueryAndSharedKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # GQA, sharing required.
              'input_projection': common.QueryAndSharedKeyValueProjection(),
              'num_heads': 6,
              'num_kv_heads': 3,
          },
      ),
      attention_sinks=('none', 'embeddings', 'scalars'),
  )
  def test_projection_config(
      self,
      input_projection: common.QueryKeyValueProjectionConfig,
      num_heads: int,
      num_kv_heads: int | None,
      attention_sinks: Literal['none', 'embeddings', 'scalars'],
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, units_per_head = 2, 5
    max_past_horizon = 7
    max_future_horizon = 11
    num_sink_embeddings = 2 if attention_sinks == 'embeddings' else 0
    use_sink_scalars = attention_sinks == 'scalars'
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=units_per_head,
        input_projection=input_projection,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
        name='dot_product_self_attention',
    ).make()

    # Input length is unlikely to affect this test so no need to sweep it.
    x = test_utils.random_sequence(batch_size, 16, 2)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max(0, max_future_horizon))

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
        input_projection=input_projection,
    )

    self.verify_contract(
        l,
        x,
        training=False,
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )

  @parameterized.parameters(
      (common.CombinedQueryKeyValueProjection(),),
      (common.SeparateQueryKeyValueProjection(),),
      (common.QueryAndKeyValueProjection(),),
      (common.QueryAndSharedKeyValueProjection(),),
  )
  def test_einsum_factory(
      self,
      input_projection: common.QueryKeyValueProjectionConfig,
  ):
    def einsum_factory() -> types.JnpEinsumT:
      def custom_einsum(equation, *args, **kwargs):
        return jnp.multiply(jnp.einsum(equation, *args, **kwargs), 3)

      return custom_einsum

    key = jax.random.PRNGKey(1234)
    batch_size, units_per_head = 2, 5
    max_past_horizon = 7
    max_future_horizon = 11
    num_sink_embeddings = 0
    use_sink_scalars = False
    num_heads = 3
    num_kv_heads = None
    l_default = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=units_per_head,
        input_projection=input_projection,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
        name='dot_product_self_attention',
    ).make()
    l_einsum = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=units_per_head,
        input_projection=dataclasses.replace(
            input_projection, einsum_factory=einsum_factory
        ),
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
        name='dot_product_self_attention',
    ).make()

    # Input length is unlikely to affect this test so no need to sweep it.
    x = test_utils.random_sequence(batch_size, 16, 2)
    l_einsum = self.init_and_bind_layer(key, l_einsum, x)
    l_default = self.init_and_bind_layer(key, l_default, x)

    y_einsum = l_einsum.layer(x, training=False).mask_invalid()
    y_default = l_default.layer(x, training=False).mask_invalid()
    self.assertSequencesNotClose(y_einsum, y_default)

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0. Steppable.
      (1, 2, 3, 0, False),
      (1, 2, 3, 0, True),
      (3, 5, 3, 0, False),
      (3, 5, 3, 0, True),
      # max_past_horizon > 0, max_future_horizon > 0. Steppable.
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
    l = dot_product_self_attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        use_kv_cache_ringbuffer=True,
        name='dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )
    self.assertEqual(
        l.supports_step, max_past_horizon >= 0 and max_future_horizon >= 0
    )
    self.assertEqual(l.input_latency, max(0, max_future_horizon))

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
