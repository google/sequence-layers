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
import numpy as np
from sequence_layers.jax import test_utils
from sequence_layers.jax.attention import t5_relative_position_embedding


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
    l = t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
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
    l = t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
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
    l = t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
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
    l = t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
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
    l = t5_relative_position_embedding.T5RelativePositionEmbedding.Config(
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


if __name__ == '__main__':
  test_utils.main()
