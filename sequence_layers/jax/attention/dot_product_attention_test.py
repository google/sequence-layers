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
import functools
import itertools
from typing import Literal
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from sequence_layers.jax import position
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax.attention import common
from sequence_layers.jax.attention import dot_product_attention
from sequence_layers.jax.attention import shaw_relative_position_embedding
from sequence_layers.jax.attention import t5_relative_position_embedding
from sequence_layers.jax.attention import test_utils as attention_test_utils


# Custom init function so that position bias decreases as absolute
# relative distance increases.
def _t5_position_bias_mat_init(
    key: jax.Array,
    shape: tuple[int, int],
    dtype: jnp.dtype,
    bidirectional: bool,
) -> jax.Array:
  del key
  num_buckets, num_heads = shape
  if bidirectional:
    mid = num_buckets // 2
    biases = jnp.concatenate(
        [-jnp.arange(mid), -jnp.arange(mid, num_buckets)], dtype=dtype
    )
  else:
    biases = -jnp.arange(num_buckets, dtype=dtype)
  bias_matrix = jnp.tile(biases[:, None], [1, num_heads])
  assert bias_matrix.shape == (num_buckets, num_heads)
  return bias_matrix


class DotProductAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (1, 2, 0, False),
      (3, 5, 0, False),
      (3, 5, 1, False),
      (3, 5, 0, True),
  )
  def test_dot_product_attention(
      self, num_heads, units_per_head, num_sink_embeddings, use_sink_scalars
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        per_dim_scale=True,
        name='dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
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
    self.assertEqual(l.name, 'dot_product_attention')

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
      x = test_utils.random_sequence(batch_size, time, channels)
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

  def test_emit_outputs(self):
    key = jax.random.PRNGKey(1234)
    num_heads, units_per_head = 3, 5
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        precision=jax.lax.Precision.HIGHEST,
        name='dot_product_attention',
    ).make()

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}
    time, channels = 7, 3
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_attention')

    _, emits = l.layer_with_emits(x, training=False, constants=constants)
    self.assertEqual(
        emits.probabilities_by_source[source_name].shape,
        (batch_size, time, num_heads, source_time),
    )

    # Only run three timesteps of the sequence.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(
        x,
        l.get_initial_state(
            batch_size=batch_size,
            input_spec=x.channel_spec,
            training=False,
            constants=constants,
        ),
        training=False,
        constants=constants,
    )
    self.assertEqual(
        emits.probabilities_by_source[source_name].shape,
        (batch_size, 3, num_heads, source_time),
    )

  @parameterized.parameters(True, False)
  def test_dropout(self, broadcast_dropout_across_queries):
    key = jax.random.PRNGKey(1234)
    source_name = 'source'
    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=3,
        units_per_head=5,
        attention_probabilities_dropout_rate=0.99999,
        broadcast_dropout_across_queries=broadcast_dropout_across_queries,
        name='dot_product_attention',
    ).make()

    source = test_utils.random_sequence(2, 4, 5)
    constants = {source_name: source}
    x = test_utils.random_sequence(2, 21, 3, random_mask=True)
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    rngs = {'dropout': jax.random.key(1)}

    y_dropout = l.apply(
        l.variables,
        x,
        training=True,
        constants=constants,
        rngs=rngs,
    )
    y_no_dropout = l.apply(
        l.variables, x, training=False, constants=constants, rngs=rngs
    )
    y_no_dropout2 = l.apply(
        l.variables, x, training=False, constants=constants, rngs=rngs
    )

    chex.assert_trees_all_equal(
        y_dropout.values, jnp.zeros_like(y_dropout.values)
    )
    self.assertSequencesClose(y_no_dropout, y_no_dropout2)

  def test_logits_soft_cap(self):
    key = jax.random.PRNGKey(1234)
    num_heads, units_per_head = 3, 5
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        precision=jax.lax.Precision.HIGHEST,
        attention_logits_soft_cap=50.0,
        name='dot_product_attention',
    ).make()

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}
    time, channels = 21, 3
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_attention')

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

  @parameterized.parameters(
      (1, 2),
      (3, 6),
  )
  def test_rotary_positional_encoding(self, num_heads, units_per_head):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
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
        name='dot_product_attention',
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
    self.assertEqual(l.name, 'dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, constants=constants, input_projection=l.config.input_projection
    )

    channels = 3
    for time in [11, 12]:
      x = test_utils.random_sequence(batch_size, time, channels)
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

  @parameterized.parameters(
      (1, 2, 3, 0),
      (3, 6, 2, 3),
  )
  def test_shaw_relative_position_bias(
      self,
      num_heads: int,
      units_per_head: int,
      max_backward: int,
      max_forward: int,
  ):
    key = jax.random.PRNGKey(3457)
    batch_size, source_seq_length, source_channels = 2, 7, 2
    input_seq_length, input_channels = 9, 3
    source_name = 'source'

    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        relative_position_embedding=shaw_relative_position_embedding.ShawRelativePositionEmbedding.Config(
            max_backward=max_backward,
            max_forward=max_forward,
            num_heads=num_heads,
            units_per_head=units_per_head,
        ),
        name='dot_product_attention',
    ).make()

    source = test_utils.random_sequence(
        batch_size, source_seq_length, source_channels
    )
    constants = {source_name: source}
    x = test_utils.random_sequence(batch_size, 1, input_channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, constants=constants, input_projection=l.config.input_projection
    )

    x = test_utils.random_sequence(batch_size, input_seq_length, input_channels)
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

  @parameterized.parameters(
      (1, 2),
      (3, 6),
  )
  def test_t5_relative_position_bias(
      self,
      num_heads: int,
      units_per_head: int,
  ):
    key = jax.random.PRNGKey(3457)
    batch_size, source_seq_length, source_channels = 2, 7, 2
    input_seq_length, input_channels = 9, 3
    source_name = 'source'
    bidirectional = True

    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        relative_position_embedding=t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
            num_buckets=8,
            bidirectional=bidirectional,
            num_heads=num_heads,
            max_distance=128,
            bias_matrix_init=functools.partial(
                _t5_position_bias_mat_init,
                bidirectional=bidirectional,
            ),
        ),
        name='dot_product_attention',
    ).make()

    source = test_utils.random_sequence(
        batch_size, source_seq_length, source_channels
    )
    constants = {source_name: source}
    x = test_utils.random_sequence(batch_size, 1, input_channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, constants=constants, input_projection=l.config.input_projection
    )

    x = test_utils.random_sequence(batch_size, input_seq_length, input_channels)
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

    # Check that off-diagonals have lower attention probabilities than
    # on-diagonals.
    x = types.Sequence(
        jnp.ones((batch_size, 5, input_channels), dtype=jnp.float32),
        jnp.ones((batch_size, 5), dtype=jnp.bool_),
    )
    cond = types.Sequence(
        jnp.ones((batch_size, 5, source_channels), dtype=jnp.float32),
        jnp.ones((batch_size, 5), dtype=jnp.bool_),
    )
    constants = {source_name: cond}
    _, emits = l.layer_with_emits(x, training=False, constants=constants)
    attention_probs = emits.probabilities_by_source['source'].values

    for i, j in itertools.product(range(5), range(5)):
      if i != j:
        self.assertTrue(
            jnp.all(attention_probs[:, i, :, j] < attention_probs[:, i, :, i])
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
        source_name=source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        precision=jax.lax.Precision.HIGHEST,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )
    l = dot_product_attention.DotProductAttention.Config(
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

  @parameterized.parameters(
      (1, 3),
      (2, 5),
  )
  def test_relative_position_bias_with_position_source(
      self,
      num_heads: int,
      units_per_head: int,
  ):
    key = jax.random.PRNGKey(3457)
    input_seq_length, input_channels, source_channels = 9, 3, 2
    bidirectional = True
    source_name = 'source'

    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        relative_position_embedding=t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
            num_buckets=8,
            bidirectional=bidirectional,
            num_heads=num_heads,
            max_distance=128,
            bias_matrix_init=functools.partial(
                _t5_position_bias_mat_init,
                bidirectional=bidirectional,
            ),
        ),
        query_positions_name='positions',
        name='dot_product_attention',
    ).make()

    x = types.Sequence(
        jnp.ones((2, input_seq_length, input_channels), dtype=jnp.float32),
        jnp.ones((2, input_seq_length), dtype=jnp.bool_),
    )
    source = types.Sequence(
        jnp.ones((2, input_seq_length, source_channels), dtype=jnp.float32),
        jnp.ones((2, input_seq_length), dtype=jnp.bool_),
    )
    # First example has usual positions going up from 0 to 7.
    # Second example has positions going down from 7 to 0.
    positions = jnp.stack(
        [
            jnp.arange(input_seq_length),  # pos = 0, 1, 2, 3, 4, ...
            jnp.arange(
                input_seq_length - 1, -1, -1
            ),  # pos = 7, 6, 5, 4, 3, ...
        ],
        axis=0,
    )
    positions = types.Sequence(
        positions, jnp.ones(positions.shape, dtype=jnp.bool_)
    )
    constants = {
        source_name: source,
        'positions': positions,
    }
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l, x, input_projection=l.config.input_projection, constants=constants
    )
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        (num_heads, units_per_head),
    )

    # Check the attention matrices.
    out, emits = l.layer_with_emits(x, training=False, constants=constants)
    attention_probs = emits.probabilities_by_source['source'].values

    # First example should have higest attention on diagonal from top left to
    # bottom right.
    for i, j in itertools.product(range(5), range(5)):
      if i != j:
        self.assertTrue(
            jnp.all(attention_probs[0, i, :, j] < attention_probs[0, i, :, i])
        )
    # Attention map of the second example should be the same but flipped.
    chex.assert_trees_all_close(
        attention_probs[0, ...], attention_probs[1, ::-1, ...]
    )

    # Check that repeated step matched the layer function.
    state = l.get_initial_state(
        2,
        types.ShapeDType((2, 10, 32), jnp.float32),
        training=False,
        constants=constants,
    )
    ys, step_emits = [], []
    for i in range(input_seq_length):
      y, state, step_emit = l.step_with_emits(
          x=types.Sequence(x.values[:, i : i + 1], x.mask[:, i : i + 1]),
          state=state,
          training=False,
          constants={
              source_name: source,
              'positions': types.Sequence(
                  positions.values[:, i : i + 1], positions.mask[:, i : i + 1]
              ),
          },
      )
      ys.append(y)
      step_emits.append(step_emit)
    attention_probs_step = jnp.concatenate(
        [e.probabilities_by_source['source'].values for e in step_emits],
        axis=1,
    )
    ys_step = jnp.concatenate([y.values for y in ys], axis=1)
    chex.assert_trees_all_close(
        attention_probs_step,
        attention_probs,
    )
    chex.assert_trees_all_close(ys_step, out.values)

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
        source_name=source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        precision=jax.lax.Precision.HIGHEST,
    )
    l = dot_product_attention.DotProductAttention.Config(
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
    l = dot_product_attention.DotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertTrue(l.supports_step)

    l = dot_product_attention.DotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertFalse(l.supports_step)

    l = dot_product_attention.DotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    # Even if key / value network not steppable, we can still step.
    self.assertTrue(l.supports_step)

    l = dot_product_attention.DotProductAttention.Config(
        'source',
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
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    units_per_head = 5
    num_sink_embeddings = 2 if attention_sinks == 'embeddings' else 0
    use_sink_scalars = attention_sinks == 'scalars'
    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        input_projection=input_projection,
        units_per_head=units_per_head,
        per_dim_scale=True,
        name='dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
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
    self.assertEqual(l.name, 'dot_product_attention')

    attention_test_utils.assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
        use_sink_scalars=use_sink_scalars,
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

  def test_recompute_kv_per_step(self):
    """Tests that recompute_kv_per_step re-reads source from constants."""
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 7, 4
    num_heads, units_per_head = 3, 5
    source_name = 'source'
    channels = 3

    # Create two layers: one with recompute, one without.
    config_kwargs = dict(
        source_name=source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        precision=jax.lax.Precision.HIGHEST,
        name='dot_product_attention',
    )
    l_recompute = dot_product_attention.DotProductAttention.Config(
        **config_kwargs, recompute_kv_per_step=True
    ).make()
    l_cached = dot_product_attention.DotProductAttention.Config(
        **config_kwargs, recompute_kv_per_step=False
    ).make()

    source1 = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    source2 = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants1 = {source_name: source1}
    constants2 = {source_name: source2}

    x = test_utils.random_sequence(batch_size, 1, channels)

    # Init both layers with the same params using source1.
    l_recompute = self.init_and_bind_layer(
        key, l_recompute, x, constants=constants1
    )
    l_cached = self.init_and_bind_layer(key, l_cached, x, constants=constants1)

    # Get initial states (both use source1 for init).
    state_recompute = l_recompute.get_initial_state(
        batch_size, x.channel_spec, training=False, constants=constants1
    )
    state_cached = l_cached.get_initial_state(
        batch_size, x.channel_spec, training=False, constants=constants1
    )

    # Step with source1 — both should produce the same output.
    y_recompute_s1, state_recompute, _ = l_recompute.step_with_emits(
        x, state_recompute, training=False, constants=constants1
    )
    y_cached_s1, state_cached, _ = l_cached.step_with_emits(
        x, state_cached, training=False, constants=constants1
    )
    self.assertSequencesClose(y_recompute_s1, y_cached_s1, atol=1e-5)

    # Step with source2 — only recompute layer should change.
    y_recompute_s2, _, _ = l_recompute.step_with_emits(
        x, state_recompute, training=False, constants=constants2
    )
    y_cached_s2, _, _ = l_cached.step_with_emits(
        x, state_cached, training=False, constants=constants2
    )

    # Recompute layer: output should differ when source changes.
    with self.assertRaises(AssertionError):
      self.assertSequencesClose(y_recompute_s1, y_recompute_s2, atol=1e-5)

    # Cached layer: output should be the same regardless of source change
    # (K/V were cached at init from source1).
    self.assertSequencesClose(y_cached_s1, y_cached_s2, atol=1e-5)

  @parameterized.parameters(5, 11)
  def test_recompute_kv_per_step_verify_contract(self, time):
    """Tests that recompute_kv_per_step passes the standard contract."""
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 7, 4
    num_heads, units_per_head = 3, 5
    source_name = 'source'

    l = dot_product_attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        recompute_kv_per_step=True,
        name='dot_product_attention',
    ).make()

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}
    channels = 3

    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)
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
