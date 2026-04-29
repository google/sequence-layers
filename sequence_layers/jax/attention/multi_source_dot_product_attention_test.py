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
from sequence_layers.jax.attention import common
from sequence_layers.jax.attention import multi_source_dot_product_attention
from sequence_layers.jax.attention import test_utils as attention_test_utils


class MultiSourceDotProductAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      {
          'num_heads': 1,
          'units_per_head': 2,
          'use_separate_kv_projections': False,
      },
      {
          'num_heads': 3,
          'units_per_head': 5,
          'use_separate_kv_projections': False,
      },
      {
          'num_heads': 3,
          'units_per_head': 5,
          'use_separate_kv_projections': True,
      },
  )
  def test_multi_source_dot_product_attention(
      self,
      num_heads: int,
      units_per_head: int,
      use_separate_kv_projections: bool,
  ):
    key = jax.random.PRNGKey(1234)
    time, channels = 11, 3
    batch_size, source_time, source_channels = 2, 11, 2
    source1_name, source2_name = 'source1', 'source2'
    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        source_names=(source1_name, source2_name),
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=num_heads,
        units_per_head=units_per_head,
        per_dim_scale=True,
        use_separate_kv_projections=use_separate_kv_projections,
        name='multi_source_dot_product_attention',
    ).make()

    x = test_utils.random_sequence(batch_size, time, channels)

    source1 = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    source2 = test_utils.random_sequence(
        batch_size, source_time * 2, source_channels
    )
    constants = {source1_name: source1, source2_name: source2}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'multi_source_dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        input_projection=l.config.input_projection,
        kv_projection_source_names=(source1_name, source2_name)
        if use_separate_kv_projections
        else (),
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
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      config=(
          dict(per_dim_scale=True),
          dict(attention_logits_soft_cap=20.0),
          dict(
              query_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=1000
              ),
              key_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=1000
              ),
          ),
      ),
  )
  def test_dtypes(self, param_dtype, input_dtype, compute_dtype, config):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 3, 7, 4
    random_mask = True
    source_name = 'source'
    num_heads, units_per_head = 2, 6
    defaults = dict(
        source_names=source_name,
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=num_heads,
        units_per_head=units_per_head,
        precision=jax.lax.Precision.HIGHEST,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )
    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        **(defaults | config)
    ).make()

    source = test_utils.random_sequence(
        batch_size,
        source_time,
        source_channels,
        random_mask=random_mask,
        dtype=input_dtype,
    )
    constants = {source_name: source}
    time, channels = 5, 2
    x = test_utils.random_sequence(
        batch_size, time, channels, random_mask=random_mask, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, constants=constants, input_projection=l.config.input_projection
    )

    x = test_utils.random_sequence(
        batch_size, time, channels, random_mask=random_mask, dtype=input_dtype
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
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )

  @parameterized.product(
      config=(
          dict(per_dim_scale=True),
          dict(attention_logits_soft_cap=20.0),
          dict(
              query_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=1000
              ),
              key_network=position.ApplyRotaryPositionalEncoding.Config(
                  max_wavelength=1000
              ),
          ),
      )
  )
  def test_bf16_mode(self, config):
    """Tests that when inputs / params are bfloat16, the output is bfloat16."""
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 3, 7, 4
    random_mask = True
    source_name = 'source'
    num_heads, units_per_head = 2, 6
    defaults = dict(
        source_names=source_name,
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=num_heads,
        units_per_head=units_per_head,
        precision=jax.lax.Precision.HIGHEST,
    )
    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        **(defaults | config)
    ).make()

    source = test_utils.random_sequence(
        batch_size,
        source_time,
        source_channels,
        random_mask=random_mask,
        dtype=jnp.bfloat16,
    )
    constants = {source_name: source}
    time, channels = 5, 2
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

  def test_query_key_value_network_supports_step(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 1, 3)
    source = test_utils.random_sequence(2, 1, 5)
    constants = {'source': source}
    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        'source',
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=3,
        units_per_head=5,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertTrue(l.supports_step)

    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        'source',
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=3,
        units_per_head=5,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertFalse(l.supports_step)

    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        'source',
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=3,
        units_per_head=5,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    # Even if key / value network not steppable, we can still step.
    self.assertTrue(l.supports_step)

    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        'source',
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=3,
        units_per_head=5,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=test_utils.NonSteppableLayer.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    # Even if key / value network not steppable, we can still step.
    self.assertTrue(l.supports_step)

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
  )
  def test_projection_config(
      self,
      input_projection: (
          common.SeparateQueryKeyValueProjection
          | common.QueryAndKeyValueProjection
          | common.QueryAndSharedKeyValueProjection
      ),
      num_heads: int,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    units_per_head = 5
    l = multi_source_dot_product_attention.MultiSourceDotProductAttention.Config(
        source_name,
        flash_attention_source_block_sizes=3,
        flash_attention_query_block_size=5,
        num_heads=num_heads,
        input_projection=input_projection,
        units_per_head=units_per_head,
        per_dim_scale=True,
        name='multi_source_dot_product_attention',
    ).make()

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}
    channels = 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'multi_source_dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        input_projection=l.config.input_projection,
    )

    x = test_utils.random_sequence(batch_size, 10, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        (num_heads, units_per_head),
    )
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )


if __name__ == '__main__':
  test_utils.main()
