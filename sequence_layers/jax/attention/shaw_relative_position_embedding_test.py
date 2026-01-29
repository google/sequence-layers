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
from sequence_layers.jax.attention import shaw_relative_position_embedding


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
    l = shaw_relative_position_embedding.ShawRelativePositionEmbedding.Config(
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
    l = shaw_relative_position_embedding.ShawRelativePositionEmbedding.Config(
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
    l = shaw_relative_position_embedding.ShawRelativePositionEmbedding.Config(
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


if __name__ == '__main__':
  test_utils.main()
