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
"""Tests for attention layers."""
import dataclasses
import functools
import itertools

from absl.testing import parameterized
import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import attention
from sequence_layers.jax import position
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils


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


def assert_param_dtypes_inits_shapes(
    layer: types.SequenceLayer,
    inputs: types.Sequence,
    input_projection: attention.QueryKeyValueProjectionConfig,
    constants: types.Constants | None = None,
    num_sink_embeddings: int = 0,
) -> None:
  config = layer.config
  params = {}

  num_query_heads = config.num_heads
  if hasattr(config, 'num_kv_heads'):
    num_kv_heads = config.num_kv_heads or num_query_heads
  else:
    num_kv_heads = num_query_heads

  # A per-dimension scale factor shared across heads:
  if config.per_dim_scale:
    params['per_dim_scale'] = jnp.zeros(
        (config.units_per_head,), dtype=config.param_dtype
    )

  if num_sink_embeddings > 0:
    params['sink_key_embeddings'] = jnp.zeros(
        (
            num_sink_embeddings,
            num_query_heads,
            config.units_per_head,
        ),
        dtype=config.param_dtype,
    )

    params['sink_value_embeddings'] = jnp.zeros(
        (
            num_sink_embeddings,
            num_kv_heads,
            config.units_per_head,
        ),
        dtype=config.param_dtype,
    )

  q_in_channels = inputs.shape[2]
  # Cross- versus self-attention:
  if constants is not None:
    kv_in_channels = constants[config.source_name].shape[2]
  else:
    kv_in_channels = q_in_channels

  match input_projection:
    case attention.CombinedQueryKeyValueProjection():
      # A bias-less dense layer for qkv projection
      params['query_key_value_projection'] = {
          'kernel': jnp.zeros(
              (
                  q_in_channels,
                  2 if input_projection.share_kv_projection else 3,
                  num_query_heads,
                  config.units_per_head,
              ),
              dtype=config.param_dtype,
          ),
      }
    case attention.SeparateQueryKeyValueProjection():
      params['query_projection'] = {
          'kernel': jnp.zeros(
              (q_in_channels, num_query_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['key_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['value_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
    case attention.QueryAndKeyValueProjection():
      params['query_projection'] = {
          'kernel': jnp.zeros(
              (q_in_channels, num_query_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['key_value_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, 2, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
    case attention.QueryAndSharedKeyValueProjection():
      params['query_projection'] = {
          'kernel': jnp.zeros(
              (q_in_channels, num_query_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['shared_key_value_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }

  # Position embeddings:
  if pos_config := getattr(config, 'relative_position_embedding', None):
    if isinstance(pos_config, attention.ShawRelativePositionEmbedding.Config):
      params['relative_position_embedding'] = {
          'embedding': jnp.zeros(
              (
                  pos_config.max_backward + pos_config.max_forward + 1,
                  num_query_heads,
                  config.units_per_head,
              ),
              dtype=pos_config.param_dtype,
          ),
      }
    if isinstance(pos_config, attention.T5RelativePositionEmbedding.Config):
      params['relative_position_embedding'] = {
          'embedding': jnp.zeros(
              (
                  pos_config.num_buckets,
                  num_query_heads,
              ),
              dtype=pos_config.param_dtype,
          ),
      }
    elif isinstance(
        pos_config, attention.TransformerXLRelativePositionEmbedding.Config
    ):
      params['relative_position_embedding'] = {
          'u': jnp.zeros(
              (num_query_heads, config.units_per_head),
              dtype=pos_config.param_dtype,
          ),
          'v': jnp.zeros(
              (num_query_heads, config.units_per_head),
              dtype=pos_config.param_dtype,
          ),
          'pos_proj': {
              'kernel': jnp.zeros(
                  (
                      pos_config.position_bias_dim,
                      num_query_heads,
                      config.units_per_head,
                  ),
                  dtype=pos_config.param_dtype,
              )
          },
      }

  chex.assert_trees_all_equal_shapes_and_dtypes(
      flax.core.meta.unbox(layer.variables),
      {'params': params},
  )


class ShawRelativePositionEmbeddingTest(test_utils.SequenceLayerTest):

  @property
  def prng_key(self):
    return jax.random.PRNGKey(6784)

  def _fake_initializer(self, key, shape, dtype):
    """Initializer that outputs jnp.arange() values for testing."""
    del key
    return jnp.tile(
        jnp.arange(shape[0], dtype=dtype)[:, jnp.newaxis], [1, shape[1]]
    )

  def test_shape(self):
    bs, q_len, k_len, num_heads, units_per_head = 2, 3, 4, 5, 8
    l = attention.ShawRelativePositionEmbedding.Config(
        max_forward=4,
        max_backward=3,
        num_heads=num_heads,
        units_per_head=units_per_head,
    ).make()

    query_positions = jnp.array([[0, 1, 2], [0, 1, 2]])
    self.assertEqual(query_positions.shape, (bs, q_len))
    key_positions = jnp.array([[-1, 1, 2.5, 3], [5, -5, 200, 10]])
    self.assertEqual(key_positions.shape, (bs, k_len))
    queries = np.random.normal(size=(bs, q_len, num_heads, units_per_head))

    biases, _ = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=query_positions,
        key_positions=key_positions,
        queries=queries,
    )
    self.assertEqual(biases.shape, (bs, num_heads, q_len, k_len))

  @parameterized.parameters(*test_utils.standard_dtype_configs(expected=True))
  def test_dtype(self, param_dtype, input_dtype, compute_dtype, expected_dtype):
    bs, q_len, k_len, num_heads, units_per_head = 2, 3, 4, 2, 8
    l = attention.ShawRelativePositionEmbedding.Config(
        max_forward=4,
        max_backward=3,
        num_heads=num_heads,
        units_per_head=units_per_head,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    ).make()

    query_positions = jnp.array([[0, 1, 2], [0, 1, 2]], dtype=input_dtype)
    self.assertEqual(query_positions.shape, (bs, q_len))
    key_positions = jnp.array(
        [[-1, 1, 2.5, 3], [5, -5, 200, 10]], dtype=input_dtype
    )
    self.assertEqual(key_positions.shape, (bs, k_len))
    queries = jax.random.normal(
        key=self.prng_key,
        shape=(bs, q_len, num_heads, units_per_head),
        dtype=input_dtype,
    )

    biases, _ = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=query_positions,
        key_positions=key_positions,
        queries=queries,
    )
    self.assertEqual(biases.dtype, expected_dtype)

  @parameterized.parameters(
      (0, 4, jnp.float32), (5, 0, jnp.bfloat16), (3, 3, jnp.bfloat16)
  )
  def test_slicing_consistency(
      self,
      max_forward: int,
      max_backward: int,
      compute_dtype: jnp.dtype,
  ):
    # Check that slicing along the query and key dimension of the input
    # positions is the same as slicing the full position bias.
    batch_size, q_len, k_len, num_heads, units_per_head = 2, 5, 4, 3, 8
    rng_query_pos, rng_key_pos, rng_queries = jax.random.split(self.prng_key, 3)
    l = attention.ShawRelativePositionEmbedding.Config(
        max_forward=max_forward,
        max_backward=max_backward,
        num_heads=num_heads,
        units_per_head=units_per_head,
        compute_dtype=compute_dtype,
    ).make()
    _, params = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=jnp.array([[0.0]]),
        key_positions=jnp.array([[0.0]]),
        queries=jnp.ones((batch_size, 1, num_heads, units_per_head)),
    )
    l = l.bind(params)
    query_positions = jax.random.randint(
        rng_query_pos, (batch_size, q_len), 0, 10
    )
    key_positions = jax.random.randint(rng_key_pos, (batch_size, k_len), 0, 10)
    queries = jax.random.normal(
        rng_queries,
        shape=(batch_size, q_len, num_heads, units_per_head),
    )
    pos_bias_full = l.get_position_bias(query_positions, key_positions, queries)
    pos_bias_part1 = l.get_position_bias(
        query_positions[:, -3:],
        key_positions[:, :2],
        queries[:, -3:, :, :],
    )
    self.assertAllClose(pos_bias_full[:, :, -3:, :2], pos_bias_part1)


class T5RelativePositionEmbeddingTest(test_utils.SequenceLayerTest):

  @property
  def prng_key(self):
    return jax.random.PRNGKey(123)

  def _random_normal(self, shape):
    return jnp.asarray(np.random.normal(size=shape))

  def _fake_initializer(self, key, shape, dtype):
    """Initializer that outputs jnp.arange() values for testing."""
    del key
    return jnp.tile(
        jnp.arange(shape[0], dtype=dtype)[:, jnp.newaxis], [1, shape[1]]
    )

  def test_shape(self):
    bs, q_len, k_len, num_heads = 2, 3, 4, 5
    l = attention.T5RelativePositionEmbedding.Config(
        num_heads=num_heads,
        num_buckets=16,
        bidirectional=True,
        max_distance=100,
    ).make()

    query_positions = jnp.array([[0, 1, 2], [0, 1, 2]])
    self.assertEqual(query_positions.shape, (bs, q_len))
    key_positions = jnp.array([[-1, 1, 2.5, 3], [5, -5, 200, 10]])
    self.assertEqual(key_positions.shape, (bs, k_len))
    queries = jnp.zeros((bs, q_len, num_heads, 16), dtype=jnp.float32)
    _, params = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=query_positions,
        key_positions=key_positions,
        queries=queries,
    )
    l = l.bind(params)
    # Test full batch
    biases = l.get_position_bias(query_positions, key_positions, queries)
    self.assertEqual(biases.shape, (bs, num_heads, q_len, k_len))
    # Test batch broadcasting
    biases = l.get_position_bias(query_positions[:1], key_positions, queries)
    self.assertEqual(biases.shape, (bs, num_heads, q_len, k_len))
    biases = l.get_position_bias(
        query_positions[:1], key_positions[:1], queries
    )
    self.assertEqual(biases.shape, (1, num_heads, q_len, k_len))

  @parameterized.parameters(*test_utils.standard_dtype_configs(expected=True))
  def test_dtype(self, param_dtype, input_dtype, compute_dtype, expected_dtype):
    l = attention.T5RelativePositionEmbedding.Config(
        num_heads=5,
        num_buckets=16,
        bidirectional=True,
        max_distance=100,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    ).make()

    query_positions = jnp.array([[0, 1, 2], [0, 1, 2]])
    key_positions = jnp.array([[-1, 1, 2.5, 3], [5, -5, 200, 10]])
    queries = jnp.zeros((2, 3, 5, 4), dtype=input_dtype)
    biases, _ = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=query_positions,
        key_positions=key_positions,
        queries=queries,
    )
    self.assertEqual(biases.dtype, expected_dtype)

  def test_bucket_retrieval_bidirectional(self):
    """Test biases returned by exact buckets for close relative distances."""

    num_buckets, max_distance = 8, 10
    l = attention.T5RelativePositionEmbedding.Config(
        num_heads=1,
        num_buckets=num_buckets,
        bidirectional=True,
        max_distance=max_distance,
        bias_matrix_init=self._fake_initializer,
    ).make()

    # max_exact = 8 / 2 / 2 = 2
    query_positions = jnp.array([[0.0]])
    key_positions = jnp.array([[-11, -2, -1, 0, 1, 2, 11]])
    queries = jnp.zeros((1, 1, 1, 4))

    biases, _ = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=query_positions,
        key_positions=key_positions,
        queries=queries,
    )
    # [bs, heads, qlen, klen] -> [klen]
    biases = biases[0, 0, 0, :]

    # Check that all positions get a unique bias.
    self.assertLen(set((int(b) for b in biases)), key_positions.shape[-1])

  def test_bucket_retrieval_unidirectional(self):
    """Test biases returned by exact buckets, undirectional."""
    num_buckets, max_distance = 8, 10
    l = attention.T5RelativePositionEmbedding.Config(
        num_heads=1,
        num_buckets=num_buckets,
        bidirectional=False,
        max_distance=max_distance,
        bias_matrix_init=self._fake_initializer,
    ).make()

    # max_exact = 8 / 2 = 4  (bidirectional=False)
    query_positions = jnp.array([[0.0]])
    key_positions = jnp.array([[-4, -3, -2, -1, 0]])
    queries = jnp.zeros((1, 1, 1, 4))

    biases, _ = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=query_positions,
        key_positions=key_positions,
        queries=queries,
    )
    # [bs, heads, qlen, klen] -> [klen]
    biases = biases[0, 0, 0, :]

    self.assertAllClose(biases, -key_positions[0])

  @parameterized.parameters(True, False)
  def test_slicing_consistency(self, bidirectional: bool):
    # Check that slicing along the query and key dimension of the input
    # positions is the same as slicing the full position bias.
    batch_size, q_len, k_len = 2, 5, 4
    num_buckets, max_distance = 8, 10
    rng_query, rng_key = jax.random.split(jax.random.PRNGKey(45672))
    l = attention.T5RelativePositionEmbedding.Config(
        num_heads=2,
        num_buckets=num_buckets,
        bidirectional=bidirectional,
        max_distance=max_distance,
        bias_matrix_init=self._fake_initializer,
    ).make()
    _, params = l.init_with_output(
        self.prng_key,
        method=l.get_position_bias,
        query_positions=jnp.array([[0.0]]),
        key_positions=jnp.array([[0.0]]),
        queries=jnp.zeros((1, 1, 2, 5)),
    )
    l = l.bind(params)
    query_positions = jax.random.randint(rng_query, (batch_size, q_len), 0, 10)
    key_positions = jax.random.randint(rng_key, (batch_size, k_len), 0, 10)
    queries = jnp.zeros((batch_size, q_len, 2, 5))
    pos_bias_full = l.get_position_bias(query_positions, key_positions, queries)
    pos_bias_part1 = l.get_position_bias(
        query_positions[:, -2:], key_positions[:, :3], queries[:, -2:, ...]
    )
    self.assertAllClose(pos_bias_full[:, :, -2:, :3], pos_bias_part1)


# TODO(b/332395698): verify precision outputs by patching jnp.einsum().
class GmmAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (1, 2, True),
      (1, 2, False),
      (3, 5, True),
      (3, 5, False),
  )
  def test_gmm_attention(self, num_heads, units_per_head, monotonic):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 11, 5
    channels = 3
    num_components = 7
    source_name = 'source'
    l = attention.GmmAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        num_components=num_components,
        monotonic=monotonic,
        init_offset_bias=1.0,
        init_scale_bias=1.0,
        name='gmm_attention',
    ).make()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}

    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'gmm_attention')

    # Shapes here should be the FlaxEinsumDense equation's second argument.
    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'hidden': {
                    'kernel': jnp.zeros((channels, num_heads, units_per_head)),
                },
                'output': {
                    'kernel': jnp.zeros(
                        (num_heads, units_per_head, num_components, 3)
                    ),
                    'bias': jnp.zeros((num_heads, num_components, 3)),
                },
            },
        },
    )

    for time in [11, 12]:
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants=constants),
          (num_heads, source_channels),
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
    """Run a separate test since verify_contract() can't return emits."""
    key = jax.random.PRNGKey(1234)
    num_heads, units_per_head, monotonic = 3, 5, True
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = attention.GmmAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        num_components=5,
        monotonic=monotonic,
        init_offset_bias=1.0,
        init_scale_bias=1.0,
        name='gmm_attention',
    ).make()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}
    time, channels = 7, 3
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    _, emits = l.layer_with_emits(x, training=False, constants=constants)
    self.assertEqual(
        emits.probabilities_by_source[source_name].values.shape,
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
        emits.probabilities_by_source[source_name].values.shape,
        (batch_size, 3, num_heads, source_time),
    )


class DotProductAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (1, 2, 0),
      (3, 5, 0),
      (3, 5, 1),
  )
  def test_dot_product_attention(
      self, num_heads, units_per_head, num_sink_embeddings
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        per_dim_scale=True,
        name='dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
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

    assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
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
    l = attention.DotProductAttention.Config(
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
    l = attention.DotProductAttention.Config(
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
    l = attention.DotProductAttention.Config(
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
    l = attention.DotProductAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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

    l = attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        relative_position_embedding=attention.ShawRelativePositionEmbedding.Config(
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

    assert_param_dtypes_inits_shapes(
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

    l = attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        relative_position_embedding=attention.T5RelativePositionEmbedding.Config(
            num_buckets=8,
            bidirectional=bidirectional,
            num_heads=num_heads,
            max_distance=128,
            bias_matrix_init=functools.partial(
                _t5_position_bias_mat_init, bidirectional=bidirectional
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

    assert_param_dtypes_inits_shapes(
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
    l = attention.DotProductAttention.Config(**(defaults | config)).make()

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

    assert_param_dtypes_inits_shapes(
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

    l = attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        relative_position_embedding=attention.T5RelativePositionEmbedding.Config(
            num_buckets=8,
            bidirectional=bidirectional,
            num_heads=num_heads,
            max_distance=128,
            bias_matrix_init=functools.partial(
                _t5_position_bias_mat_init, bidirectional=bidirectional
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

    assert_param_dtypes_inits_shapes(
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
    l = attention.DotProductAttention.Config(**(defaults | config)).make()

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
    l = attention.DotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertTrue(l.supports_step)

    l = attention.DotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertFalse(l.supports_step)

    l = attention.DotProductAttention.Config(
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

    l = attention.DotProductAttention.Config(
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
              'input_projection': attention.SeparateQueryKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': attention.QueryAndKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': attention.QueryAndSharedKeyValueProjection(),
              'num_heads': 3,
          },
      ),
      use_attention_sink=(False, True),
  )
  def test_projection_config(
      self,
      input_projection: (
          attention.SeparateQueryKeyValueProjection
          | attention.QueryAndKeyValueProjection
          | attention.QueryAndSharedKeyValueProjection
      ),
      num_heads: int,
      use_attention_sink: bool,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    units_per_head = 5
    num_sink_embeddings = 2 if use_attention_sink else 0
    l = attention.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        input_projection=input_projection,
        units_per_head=units_per_head,
        per_dim_scale=True,
        name='dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
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

    assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
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


class DotProductSelfAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0. Steppable.
      (1, 2, 3, 0, False, 0),
      (1, 2, 3, 0, True, 0),
      (3, 5, 3, 0, False, 0),
      (3, 5, 3, 0, True, 0),
      # max_past_horizon > 0, max_future_horizon > 0. Steppable.
      (3, 5, 3, 2, False, 0),
      (3, 5, 3, 2, True, 0),
      (3, 5, 3, 5, False, 0),
      (3, 5, 3, 5, True, 0),
      # max_past_horizon == -1, max_future_horizon > 0. Not steppable.
      (3, 5, -1, 2, False, 0),
      (3, 5, -1, 2, True, 0),
      # max_past_horizon > 0, max_future_horizon == -1. Not steppable.
      (3, 5, 3, -1, False, 0),
      (3, 5, 3, -1, True, 0),
      # max_past_horizon == -1, max_future_horizon == -1. Not steppable.
      (3, 5, -1, -1, False, 0),
      (3, 5, -1, -1, True, 0),
      # max_past_horizon > 0, max_future_horizon > 0. Steppable with sink
      # attention.
      (3, 5, 3, 2, False, 1),
      (3, 5, 3, 2, True, 1),
  )
  def test_dot_product_self_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      random_mask,
      num_sink_embeddings,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    l = attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        per_dim_scale=True,
        name='dot_product_self_attention',
        num_sink_embeddings=num_sink_embeddings,
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

    assert_param_dtypes_inits_shapes(
        l,
        x,
        num_sink_embeddings=num_sink_embeddings,
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
    relative_embedding = attention.ShawRelativePositionEmbedding.Config(
        max_forward=max_backward,
        max_backward=max_forward,
        num_heads=num_heads,
        units_per_head=units_per_head,
    )
    l = attention.DotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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
    relative_embedding = attention.ShawRelativePositionEmbedding.Config(
        max_forward=max_forward,
        max_backward=max_backward,
        num_heads=num_heads,
        units_per_head=units_per_head,
        compute_dtype=inner_compute_dtype,
        param_dtype=param_dtype,
    )
    l = attention.DotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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
    relative_embedding = attention.T5RelativePositionEmbedding.Config(
        num_buckets=num_buckets,
        bidirectional=bidirectional,
        max_distance=max_distance,
        num_heads=num_heads,
    )
    l = attention.DotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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
    relative_embedding = attention.T5RelativePositionEmbedding.Config(
        num_buckets=num_buckets,
        bidirectional=bidirectional,
        max_distance=max_distance,
        num_heads=num_heads,
        compute_dtype=inner_compute_dtype,
        param_dtype=param_dtype,
    )
    l = attention.DotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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
    l = attention.DotProductSelfAttention.Config(
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
    l = attention.DotProductSelfAttention.Config(
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
    l = attention.DotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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
    l = attention.DotProductSelfAttention.Config(**(defaults | config)).make()

    channels = 2
    x = test_utils.random_sequence(
        batch_size, 1, channels, random_mask=random_mask, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)

    assert_param_dtypes_inits_shapes(
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
    l = attention.DotProductSelfAttention.Config(**(defaults | config)).make()

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
    l = attention.DotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)

    l = attention.DotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = attention.DotProductSelfAttention.Config(
        num_heads=3,
        units_per_head=5,
        max_past_horizon=3,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertFalse(l.supports_step)

    l = attention.DotProductSelfAttention.Config(
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
              'input_projection': attention.CombinedQueryKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # MHA, sharing supported.
              'input_projection': attention.CombinedQueryKeyValueProjection(
                  share_kv_projection=True
              ),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          # SeparateQueryKeyValueProjection. MHA and GQA supported.
          {  # MHA, sharing not supported.
              'input_projection': attention.SeparateQueryKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # GQA, sharing not supported.
              'input_projection': attention.SeparateQueryKeyValueProjection(),
              'num_heads': 6,
              'num_kv_heads': 3,
          },
          # QueryAndKeyValueProjection. MHA and GQA supported.
          {  # MHA, sharing not supported.
              'input_projection': attention.QueryAndKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # GQA, sharing not supported.
              'input_projection': attention.QueryAndKeyValueProjection(),
              'num_heads': 6,
              'num_kv_heads': 3,
          },
          # QueryAndSharedKeyValueProjection. MHA and GQA supported.
          {  # MHA, sharing required.
              'input_projection': attention.QueryAndSharedKeyValueProjection(),
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {  # GQA, sharing required.
              'input_projection': attention.QueryAndSharedKeyValueProjection(),
              'num_heads': 6,
              'num_kv_heads': 3,
          },
      ),
      use_attention_sink=(False, True),
  )
  def test_projection_config(
      self,
      input_projection: attention.QueryKeyValueProjectionConfig,
      num_heads: int,
      num_kv_heads: int | None,
      use_attention_sink: bool,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, units_per_head = 2, 5
    max_past_horizon = 7
    max_future_horizon = 11
    num_sink_embeddings = 2 if use_attention_sink else 0
    l = attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=units_per_head,
        input_projection=input_projection,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        num_sink_embeddings=num_sink_embeddings,
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

    assert_param_dtypes_inits_shapes(
        l,
        x,
        num_sink_embeddings=num_sink_embeddings,
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
      (attention.CombinedQueryKeyValueProjection(),),
      (attention.SeparateQueryKeyValueProjection(),),
      (attention.QueryAndKeyValueProjection(),),
      (attention.QueryAndSharedKeyValueProjection(),),
  )
  def test_einsum_factory(
      self,
      input_projection: attention.QueryKeyValueProjectionConfig,
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
    num_heads = 3
    num_kv_heads = None
    l_default = attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=units_per_head,
        input_projection=input_projection,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        precision=jax.lax.Precision.HIGHEST,
        num_sink_embeddings=num_sink_embeddings,
        name='dot_product_self_attention',
    ).make()
    l_einsum = attention.DotProductSelfAttention.Config(
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
    l = attention.DotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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


class LocalDotProductSelfAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, False, 0),
      (1, 2, 3, 0, True, 0),
      (3, 5, 3, 0, False, 0),
      (3, 5, 3, 0, True, 0),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, False, 0),
      (3, 5, 3, 2, True, 0),
      (3, 5, 3, 5, False, 0),
      (3, 5, 3, 5, True, 0),
      # max_past_horizon > 0, max_future_horizon > 0, with attention sink.
      (3, 5, 3, 2, False, 1),
      (3, 5, 3, 2, True, 1),
  )
  def test_local_dot_product_self_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      random_mask,
      num_sink_embeddings,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    block_size = max(1, max_future_horizon, max_past_horizon - 1)

    l = attention.LocalDotProductSelfAttention.Config(
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
    ).make()

    channels = 1
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'local_dot_product_self_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    assert_param_dtypes_inits_shapes(
        l,
        x,
        num_sink_embeddings=num_sink_embeddings,
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
    relative_embedding = (
        attention.TransformerXLRelativePositionEmbedding.Config(
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_backward=max_past_horizon,
            max_forward=max_future_horizon,
            position_bias_dim=transformer_xl_dim,
        )
    )

    l = attention.LocalDotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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
    relative_embedding = (
        attention.TransformerXLRelativePositionEmbedding.Config(
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_backward=max_past_horizon,
            max_forward=max_future_horizon,
            position_bias_dim=transformer_xl_dim,
            param_dtype=param_dtype,
        )
    )
    l = attention.LocalDotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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

    l = attention.LocalDotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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
    l = attention.LocalDotProductSelfAttention.Config(
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

    l = attention.LocalDotProductSelfAttention.Config(
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

    l = attention.LocalDotProductSelfAttention.Config(
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

    l = attention.LocalDotProductSelfAttention.Config(
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
    l = attention.LocalDotProductSelfAttention.Config(
        **(defaults | config)
    ).make()

    channels = 1
    x = test_utils.random_sequence(
        batch_size, 1, channels, random_mask=random_mask, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)

    assert_param_dtypes_inits_shapes(
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
    l = attention.LocalDotProductSelfAttention.Config(
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

    l = attention.LocalDotProductSelfAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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


class DotProductAttentionHelperTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      query_scale=(None, 2.0), use_per_dim_scale=(False, True)
  )
  def test_query_scale(
      self, query_scale: float | None, use_per_dim_scale: bool
  ):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 3, 5, 7, 11
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    queries = jax.random.normal(
        k1, (batch, query_time, num_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_heads, units_per_head))
    values = jax.random.normal(k3, (batch, kv_time, num_heads, units_per_head))
    valid_mask = jnp.ones((batch, num_heads, query_time, kv_time), jnp.bool_)

    if query_scale is None:
      query_scale = 1 / jnp.sqrt(units_per_head)

    if use_per_dim_scale:
      per_dim_scale = jax.random.normal(k4, [units_per_head])
      queries_scaled = (
          queries * query_scale * 1.442695041 * jax.nn.softplus(per_dim_scale)
      )
    else:
      per_dim_scale = None
      queries_scaled = queries * query_scale

    _, probabilities = attention._dot_product_attention(
        queries,
        keys,
        values,
        valid_mask,
        logit_bias=None,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=per_dim_scale,
        query_scale=query_scale,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_embeddings=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    expected_probabilities = jax.nn.softmax(
        jnp.einsum('bqnh,bknh->bqnk', queries_scaled, keys)
    )
    self.assertAllClose(probabilities, expected_probabilities)

  @parameterized.product(
      (
          {'num_query_heads': 1, 'num_kv_heads': 1},
          {'num_query_heads': 3, 'num_kv_heads': 1},
          {'num_query_heads': 4, 'num_kv_heads': 2},
      ),
      broadcast_valid_mask=(True, False),
  )
  def test_grouped_query_attention(
      self,
      num_query_heads: int,
      num_kv_heads: int,
      broadcast_valid_mask: bool,
  ):
    batch, query_time, kv_time, units_per_head = 1, 3, 5, 11
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    queries = jax.random.normal(
        k1, (batch, query_time, num_query_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_kv_heads, units_per_head))
    values = jax.random.normal(
        k3, (batch, kv_time, num_kv_heads, units_per_head)
    )
    valid_mask = jnp.ones(
        (
            batch,
            1 if broadcast_valid_mask else num_query_heads,
            query_time,
            kv_time,
        ),
        jnp.bool_,
    )

    _, probabilities = attention._dot_product_attention(
        queries,
        keys,
        values,
        valid_mask,
        logit_bias=None,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=None,
        query_scale=None,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_embeddings=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    query_scale = 1 / jnp.sqrt(units_per_head)

    num_query_heads_per_kv_head = num_query_heads // num_kv_heads
    queries = utils.split_dimension(
        queries, axis=2, shape=(num_kv_heads, num_query_heads_per_kv_head)
    )

    expected_probabilities = jax.nn.softmax(
        jnp.einsum('bikqh,bjkh->bikqj', queries * query_scale, keys)
    ).reshape((batch, query_time, num_query_heads, kv_time))
    self.assertAllClose(probabilities, expected_probabilities)

  def test_zero_fully_masked(self):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 3, 5, 1, 1

    queries = jnp.ones((batch, query_time, num_heads, units_per_head))
    keys = jnp.ones((batch, kv_time, num_heads, units_per_head))
    values = jnp.ones((batch, kv_time, num_heads, units_per_head))

    # [batch, num_heads, query_time, key_time]
    valid_mask = jnp.asarray([[[
        [True, True, True, True, True],
        [True, True, True, True, True],
        [False, False, False, False, False],
    ]]])

    context_vectors, _ = attention._dot_product_attention(
        queries,
        keys,
        values,
        valid_mask,
        logit_bias=None,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=None,
        query_scale=None,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_embeddings=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    # Timestep 2 is fully masked so context vector is zero.
    expected_context_vectors = jnp.asarray([[[[1.0]], [[1.0]], [[0.0]]]])

    self.assertAllEqual(context_vectors, expected_context_vectors)

  @parameterized.product(
      use_per_dim_scale=(False, True),
      zero_fully_masked=(True, False),
  )
  def test_multi_key_value_dot_product_attention(
      self,
      use_per_dim_scale: bool,
      zero_fully_masked: bool,
  ):
    batch, query_time, kv_time, num_heads, units_per_head = 3, 10, 12, 7, 11
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(42), 5)
    queries = jax.random.normal(
        k1, (batch, query_time, num_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_heads, units_per_head))
    values = jax.random.normal(k3, (batch, kv_time, num_heads, units_per_head))
    valid_mask = (
        jax.random.uniform(k5, (batch, num_heads, query_time, kv_time)) > 0.5
    )

    if use_per_dim_scale:
      per_dim_scale = jax.random.normal(k4, [units_per_head])
    else:
      per_dim_scale = None

    expected_context_vectors, expected_probabilities = (
        attention._dot_product_attention(
            queries,
            keys,
            values,
            valid_mask,
            logit_bias=None,
            training=False,
            attention_logits_soft_cap=None,
            attention_probabilities_dropout=None,
            per_dim_scale=per_dim_scale,
            query_scale=None,
            precision=None,
            get_logits_fn=None,
            zero_fully_masked=zero_fully_masked,
            compute_dtype=None,
            num_sink_embeddings=0,
            sink_key_logits=None,
            sink_value_embeddings=None,
        )
    )

    keys1, keys2 = jnp.split(keys, 2, axis=1)
    values1, values2 = jnp.split(values, 2, axis=1)
    valid_mask1, valid_mask2 = jnp.split(valid_mask, 2, axis=3)

    context_vectors, (probabilities1, probabilities2) = (
        attention._multi_key_value_dot_product_attention(
            queries,
            (
                (keys1, values1, valid_mask1),
                (keys2, values2, valid_mask2),
            ),
            logit_bias=None,
            training=False,
            attention_logits_soft_cap=None,
            attention_probabilities_dropout=None,
            per_dim_scale=per_dim_scale,
            query_scale=None,
            precision=None,
            get_logits_fn=None,
            zero_fully_masked=zero_fully_masked,
            compute_dtype=None,
        )
    )
    probabilities = jnp.concatenate([probabilities1, probabilities2], axis=-1)

    self.assertAllClose(context_vectors, expected_context_vectors)
    self.assertAllClose(probabilities, expected_probabilities)


class LocalDotProductAttentionHelperTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      query_scale=(None, 2.0), use_per_dim_scale=(False, True)
  )
  def test_query_scale(
      self, query_scale: float | None, use_per_dim_scale: bool
  ):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 16, 16, 1, 1
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    queries = jax.random.normal(
        k1, (batch, query_time, num_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_heads, units_per_head))
    values = jax.random.normal(k3, (batch, kv_time, num_heads, units_per_head))
    keys_mask = jnp.ones((batch, kv_time), jnp.bool_)

    if query_scale is None:
      query_scale = 1 / jnp.sqrt(units_per_head)

    if use_per_dim_scale:
      per_dim_scale = jax.random.normal(k4, [units_per_head])
      queries_scaled = (
          queries * query_scale * 1.442695041 * jax.nn.softplus(per_dim_scale)
      )
    else:
      per_dim_scale = None
      queries_scaled = queries * query_scale

    max_past_horizon = 3
    max_future_horizon = 0
    block_size = 2
    context_size = block_size + max_past_horizon + max_future_horizon

    _, probabilities = attention._local_dot_product_attention(
        queries,
        keys,
        keys_mask,
        values,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=per_dim_scale,
        query_scale=query_scale,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=False,
        compute_dtype=None,
        num_sink_embeddings=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    keys_blocks = attention._extract_block_context(
        keys,
        block_size=block_size,
        left_context=max_past_horizon,
        right_context=max_future_horizon,
    )

    # [B, T, N, H] -> [B, U, W, N, H]; (U = T/W).
    queries_blocks = attention._convert_to_block(
        queries_scaled, block_size=block_size
    )

    valid_mask_blocked = attention._extract_block_context(
        keys_mask,
        block_size=block_size,
        left_context=max_past_horizon,
        right_context=max_future_horizon,
        # Mask is False for invalid timesteps.
        padding_val=False,
    )
    # Reshape to [b, h=1, num_blocks, block_size=1, context_size].
    valid_mask_blocked = valid_mask_blocked[:, jnp.newaxis, :, jnp.newaxis, :]

    local_causal_valid_mask = utils.ones_matrix_band_part(
        block_size,
        context_size,
        num_upper=max_past_horizon + max_future_horizon,
        num_lower=0,
        out_dtype=jnp.bool_,
        out_shape=[1, 1, 1, block_size, context_size],
    )

    valid_mask_blocked = jnp.logical_and(
        valid_mask_blocked,
        local_causal_valid_mask,
    )

    logits = jnp.einsum(
        'BuwNH,BucNH->BNuwc',
        queries_blocks,
        keys_blocks,
    )

    logits = jnp.where(
        valid_mask_blocked,
        logits,
        attention._INVALID_LOGIT_VALUE,
    )
    logits = logits.transpose((0, 2, 3, 1, 4)).reshape(
        (batch, -1, num_heads, context_size)
    )
    expected_probabilities = jax.nn.softmax(logits)
    self.assertAllClose(probabilities, expected_probabilities)

  def test_zero_fully_masked(self):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 5, 5, 1, 1

    queries = jnp.ones((batch, query_time, num_heads, units_per_head))
    keys = jnp.ones((batch, kv_time, num_heads, units_per_head))
    values = jnp.ones((batch, kv_time, num_heads, units_per_head))

    keys_mask = jnp.asarray([[False, True, True, True, True]])

    context_vectors, _ = attention._local_dot_product_attention(
        queries,
        keys,
        keys_mask,
        values,
        block_size=2,
        max_past_horizon=2,
        max_future_horizon=0,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=None,
        query_scale=None,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_embeddings=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    # Timestep 0 is fully masked since the first key timestep is masked and the
    # attention is causal.
    expected_context_vectors = jnp.asarray(
        [[[[0.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]]
    )

    self.assertAllEqual(context_vectors, expected_context_vectors)


class StreamingDotProductAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, 0),
      (3, 5, 3, 0, 0),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, 0),
      (3, 5, 3, 5, 0),
      # max_past_horizon > 0, max_future_horizon > 0, with attention sinks.
      (3, 5, 3, 5, 1),
  )
  def test_streaming_local_dot_product_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      num_sink_embeddings,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_channels = 2, 2
    source_name = 'source'

    l = attention.StreamingDotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        per_dim_scale=True,
        precision=jax.lax.Precision.HIGHEST,
        name='streaming_dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
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

    assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
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
    l = attention.StreamingDotProductAttention.Config(**config).make()

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

    assert_param_dtypes_inits_shapes(
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
    l = attention.StreamingDotProductAttention.Config(**config).make()

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

  def test_no_query_delay_buffer(self):
    key = jax.random.PRNGKey(1234)
    max_past_horizon, max_future_horizon = 2, 3
    batch_size, source_channels = 2, 2
    source_name = 'source'

    l = attention.StreamingDotProductAttention.Config(
        source_name,
        num_heads=3,
        units_per_head=5,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        use_query_delay_buffer=False,
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
    self.assertEqual(int(l.output_latency), 0)

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
    l = attention.StreamingDotProductAttention.Config(
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

    l = attention.StreamingDotProductAttention.Config(
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

    l = attention.StreamingDotProductAttention.Config(
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

    l = attention.StreamingDotProductAttention.Config(
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
              'input_projection': attention.SeparateQueryKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': attention.QueryAndKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': attention.QueryAndSharedKeyValueProjection(),
              'num_heads': 3,
          },
      ),
      use_attention_sink=(False, True),
  )
  def test_projection_config(
      self,
      input_projection: (
          attention.SeparateQueryKeyValueProjection
          | attention.QueryAndKeyValueProjection
          | attention.QueryAndSharedKeyValueProjection
      ),
      num_heads: int,
      use_attention_sink: bool,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time, channels, source_channels = 2, 11, 3, 2
    source_name = 'source'
    units_per_head = 5
    max_past_horizon = 3
    max_future_horizon = 3
    num_sink_embeddings = 2 if use_attention_sink else 0
    l = attention.StreamingDotProductAttention.Config(
        source_name,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        num_heads=num_heads,
        input_projection=input_projection,
        per_dim_scale=True,
        name='streaming_dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
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

    assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
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
    l = attention.StreamingDotProductAttention.Config(
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

    assert_param_dtypes_inits_shapes(
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


class StreamingLocalDotProductAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, 0),
      (3, 5, 3, 0, 0),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, 0),
      (3, 5, 3, 5, 0),
      # max_past_horizon > 0, max_future_horizon > 0, with attention sinks.
      (3, 5, 3, 5, 1),
  )
  def test_streaming_local_dot_product_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      num_sink_embeddings,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, source_channels = 2, 2
    source_name = 'source'
    block_size = max(1, max_future_horizon, max_past_horizon - 1)

    l = attention.StreamingLocalDotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        per_dim_scale=True,
        precision=jax.lax.Precision.HIGHEST,
        name='streaming_local_dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
    ).make()

    source = test_utils.random_sequence(batch_size, 1, source_channels)
    constants = {source_name: source}
    channels = 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'streaming_local_dot_product_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
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
          grad_atol=1e-5,
          grad_rtol=1e-5,
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
    config['block_size'] = max(
        1, config['max_future_horizon'], config['max_past_horizon'] - 1
    )
    l = attention.StreamingLocalDotProductAttention.Config(**config).make()

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

    assert_param_dtypes_inits_shapes(
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
    config['block_size'] = max(
        1, config['max_future_horizon'], config['max_past_horizon'] - 1
    )
    l = attention.StreamingLocalDotProductAttention.Config(**config).make()

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

  def test_no_query_delay_buffer(self):
    key = jax.random.PRNGKey(1234)
    max_past_horizon, max_future_horizon = 2, 3
    batch_size, source_channels = 2, 2
    source_name = 'source'
    block_size = max(1, max_future_horizon, max_past_horizon - 1)

    l = attention.StreamingLocalDotProductAttention.Config(
        source_name,
        num_heads=3,
        units_per_head=5,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        use_query_delay_buffer=False,
        name='streaming_local_dot_product_attention',
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
    l = attention.StreamingLocalDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        block_size=3,
        max_past_horizon=3,
        max_future_horizon=0,
        query_network=position.AddTimingSignal.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertTrue(l.supports_step)

    l = attention.StreamingLocalDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        block_size=3,
        max_past_horizon=3,
        max_future_horizon=0,
        query_network=test_utils.NonSteppableLayer.Config(),
        key_network=position.AddTimingSignal.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertFalse(l.supports_step)

    l = attention.StreamingLocalDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        block_size=3,
        max_past_horizon=3,
        max_future_horizon=0,
        query_network=position.AddTimingSignal.Config(),
        key_network=test_utils.NonSteppableLayer.Config(),
        value_network=position.AddTimingSignal.Config(),
    ).make()
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    # The key/value network must be steppable for streaming.
    self.assertFalse(l.supports_step)

    l = attention.StreamingLocalDotProductAttention.Config(
        'source',
        num_heads=3,
        units_per_head=5,
        block_size=3,
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
              'input_projection': attention.SeparateQueryKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': attention.QueryAndKeyValueProjection(),
              'num_heads': 3,
          },
          {
              'input_projection': attention.QueryAndSharedKeyValueProjection(),
              'num_heads': 3,
          },
      ),
      use_attention_sink=(False, True),
  )
  def test_projection_config(
      self,
      input_projection: (
          attention.SeparateQueryKeyValueProjection
          | attention.QueryAndKeyValueProjection
          | attention.QueryAndSharedKeyValueProjection
      ),
      num_heads: int,
      use_attention_sink: bool,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time, channels, source_channels = 2, 11, 3, 2
    source_name = 'source'
    units_per_head = 5
    max_past_horizon = 3
    max_future_horizon = 3
    block_size = max(1, max_future_horizon, max_past_horizon - 1)
    num_sink_embeddings = 2 if use_attention_sink else 0
    l = attention.StreamingLocalDotProductAttention.Config(
        source_name,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        num_heads=num_heads,
        input_projection=input_projection,
        per_dim_scale=True,
        name='streaming_local_dot_product_attention',
        num_sink_embeddings=num_sink_embeddings,
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
    self.assertEqual(l.name, 'streaming_local_dot_product_attention')

    assert_param_dtypes_inits_shapes(
        l,
        x,
        constants=constants,
        num_sink_embeddings=num_sink_embeddings,
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
        grad_atol=1e-5,
        grad_rtol=1e-5,
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
    block_size = max(1, max_future_horizon, max_past_horizon - 1)

    l = attention.StreamingLocalDotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        per_dim_scale=True,
        precision=jax.lax.Precision.HIGHEST,
        name='streaming_local_dot_product_attention',
        use_kv_cache_ringbuffer=True,
    ).make()

    source = test_utils.random_sequence(batch_size, 1, source_channels)
    constants = {source_name: source}
    channels = 3
    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'streaming_local_dot_product_attention')
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max_future_horizon)

    assert_param_dtypes_inits_shapes(
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
