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
"""Shaw-style relative position embeddings."""

import dataclasses
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping
from sequence_layers.jax import meta
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class ShawRelativePositionEmbedding(common.RelativePositionEmbedding):
  """Computes query-dependent relative position embeddings.

  Based on:
  Self-Attention with Relative Position Representations
  https://arxiv.org/abs/1803.02155

  Computes a [batch, num_heads, queries_time, keys_time] tensor of relative
  position biases, biasing the selection of keys for every query timestep.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(common.RelativePositionEmbedding.Config):
    max_backward: int
    max_forward: int
    num_heads: int
    units_per_head: int
    embedding_init: nn.initializers.Initializer = nn.linear.default_embed_init
    embedding_sharding: types.Sharding | None = None
    param_dtype: types.DType = jnp.float32
    compute_dtype: types.DType = jnp.float32

    def make(self) -> 'ShawRelativePositionEmbedding':
      return ShawRelativePositionEmbedding(self)

  config: Config

  @property
  def supports_position_bias(self):
    """Returns whether relative position bias is supported."""
    return True

  @property
  def num_buckets(self) -> int:
    return self.config.max_backward + self.config.max_forward + 1

  def setup(self) -> None:
    init_fn = utils.shard_initializer(
        self.config.embedding_init,
        self.config.embedding_sharding,
        labels=[meta.IS_EMBEDDING],
    )
    self.embedding = self.param(
        'embedding',
        init_fn,
        (self.num_buckets, self.config.num_heads, self.config.units_per_head),
        self.config.param_dtype,
    )

  def get_position_bias(
      self,
      query_positions: jaxtyping.Num[jt.ArrayT, 'b q'],
      key_positions: jaxtyping.Num[jt.ArrayT, 'b k'],
      queries: jt.Float[jt.ArrayT, 'b q nq d'],
      keys: jt.Float[jt.ArrayT, 'b k nk d'] | None = None,
  ) -> jt.Float[jt.ArrayT, 'b h q k']:
    """Computes the relative attention logit biases from position indices.

    Args:
      query_positions: [batch, query_length] int/float Tensor containing query
        position indices.
      key_positions: [batch, key_length] int/float Tensor containing position
        indices.
      queries: Queries of shape [batch, query_length, num_heads,
        units_per_head].
      keys: Has no effect on the output.

    Returns:
      position_biases: [batch, num_heads, query_length, key_length] Tensor
        containing head-wise biases for each query-key pair.
    """
    # Compute clipped relative positions.
    # -> all values in [-max_backward, max_forward]
    relative_position = jnp.expand_dims(key_positions, -2) - jnp.expand_dims(
        query_positions, -1
    )
    relative_position = jnp.clip(
        relative_position, -self.config.max_backward, self.config.max_forward
    )
    # Translate into bucket ids
    # -> all values in [0, num_buckets)
    bucket_id = relative_position + self.config.max_backward

    # Promote inputs and params to compute_dtype.
    # embedding.shape=(num_buckets, num_heads, units_per_head)=(r, h, d).
    (embedding,) = nn.dtypes.promote_dtype(
        self.embedding,
        dtype=self.config.compute_dtype,
        inexact=False,
    )
    # queries.shape=(batch, query_length, num_heads, units_per_head).
    (queries,) = nn.dtypes.promote_dtype(
        queries, dtype=self.config.compute_dtype, inexact=False
    )

    # Compute the logit biases for possible (clipped) relative positions.
    relative_logits = jnp.einsum(
        'rhd,bqhd->bhqr',
        embedding,
        queries,
    )

    # Pick the right logit bias for each key-query position pair.
    one_hots = jax.nn.one_hot(
        bucket_id,
        self.num_buckets,
        axis=-1,
        dtype=self.config.compute_dtype,
    )
    position_biases = jnp.einsum('bhqr,bqkr->bhqk', relative_logits, one_hots)

    num_queries, num_keys = query_positions.shape[1], key_positions.shape[1]
    utils.assert_is_compatible_with(
        position_biases.shape,
        [1, self.config.num_heads, num_queries, num_keys],
    )
    return position_biases
