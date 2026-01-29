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
from sequence_layers.jax import position
from sequence_layers.jax import test_utils
from sequence_layers.jax.attention import blockwise_dot_product_self_attention
from sequence_layers.jax.attention import common
from sequence_layers.jax.attention import test_utils as attention_test_utils


class BlockwiseDotProductSelfAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0. Steppable.
      (3, 5, 3, 0),
      # max_past_horizon > 0, max_future_horizon > 0. Steppable.
      (3, 5, 2, 1),
      (3, 5, 2, 3),
      # max_past_horizon == -1, max_future_horizon > 0. Not steppable.
      (3, 5, -1, 2),
      # max_past_horizon > 0, max_future_horizon == -1. Not steppable.
      (3, 5, 2, -1),
      # max_past_horizon == -1, max_future_horizon == -1. Not steppable.
      (3, 5, -1, -1),
  )
  def test_blockwise_dot_product_self_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon_blocks,
      max_future_horizon_blocks,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    block_size = 4
    l = blockwise_dot_product_self_attention.BlockwiseDotProductSelfAttention.Config(
        block_size=block_size,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon_blocks=max_past_horizon_blocks,
        max_future_horizon_blocks=max_future_horizon_blocks,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        flash_attention_query_block_size=31,
        flash_attention_key_block_size=17,
        name='blockwise_dot_product_self_attention',
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 64, channels, random_mask=True)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 4)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'blockwise_dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )
    self.assertEqual(
        l.supports_step,
        max_past_horizon_blocks >= 0 and max_future_horizon_blocks >= 0,
    )
    self.assertEqual(
        l.input_latency, max(0, max_future_horizon_blocks * block_size)
    )

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        input_projection=l.config.input_projection,
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
          dict(max_future_horizon_blocks=0),
          dict(max_past_horizon_blocks=3, max_future_horizon_blocks=3),
          dict(max_past_horizon_blocks=0, max_future_horizon_blocks=-1),
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
        block_size=3,
        num_heads=4,
        units_per_head=3,
        max_past_horizon_blocks=-1,
        precision=jax.lax.Precision.HIGHEST,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )
    l = blockwise_dot_product_self_attention.BlockwiseDotProductSelfAttention.Config(
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

  def test_query_key_value_network_supports_step(
      self,
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 1, 3)
    l = blockwise_dot_product_self_attention.BlockwiseDotProductSelfAttention.Config(
        block_size=3,
        num_heads=3,
        units_per_head=5,
        max_past_horizon_blocks=3,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)

    l = blockwise_dot_product_self_attention.BlockwiseDotProductSelfAttention.Config(
        block_size=3,
        num_heads=3,
        units_per_head=5,
        max_past_horizon_blocks=3,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = blockwise_dot_product_self_attention.BlockwiseDotProductSelfAttention.Config(
        block_size=3,
        num_heads=3,
        units_per_head=5,
        max_past_horizon_blocks=3,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = blockwise_dot_product_self_attention.BlockwiseDotProductSelfAttention.Config(
        block_size=3,
        num_heads=3,
        units_per_head=5,
        max_past_horizon_blocks=3,
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
  )
  def test_projection_config(
      self,
      input_projection: common.QueryKeyValueProjectionConfig,
      num_heads: int,
      num_kv_heads: int | None,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, units_per_head = 2, 5
    block_size = 4
    max_past_horizon_blocks = 1
    max_future_horizon_blocks = 2
    l = blockwise_dot_product_self_attention.BlockwiseDotProductSelfAttention.Config(
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=units_per_head,
        input_projection=input_projection,
        max_past_horizon_blocks=max_past_horizon_blocks,
        max_future_horizon_blocks=max_future_horizon_blocks,
        precision=jax.lax.Precision.HIGHEST,
        name='blockwise_dot_product_self_attention',
    ).make()

    # Input length is unlikely to affect this test so no need to sweep it.
    x = test_utils.random_sequence(batch_size, 16, 2)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, block_size)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'blockwise_dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), (num_heads, units_per_head)
    )
    self.assertTrue(l.supports_step)
    self.assertEqual(
        l.input_latency, max(0, max_future_horizon_blocks * block_size)
    )

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        input_projection=input_projection,
    )

    self.verify_contract(
        l,
        x,
        training=False,
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )


if __name__ == '__main__':
  test_utils.main()
