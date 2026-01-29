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
"""T5-style relative position embeddings."""

import dataclasses
import math
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping
from sequence_layers.jax import meta
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class T5RelativePositionEmbedding(common.RelativePositionEmbedding):
  """Relative position embeddings in the T5 style.

  Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer
  https://arxiv.org/abs/1910.10683
  """

  @dataclasses.dataclass(frozen=True)
  class Config(common.RelativePositionEmbedding.Config):
    """Config for T5RelativePositionEmbedding."""

    num_buckets: int
    num_heads: int
    bidirectional: bool
    max_distance: int = 128
    bias_matrix_init: nn.initializers.Initializer = nn.linear.default_embed_init
    bias_matrix_sharding: types.Sharding | None = None
    param_dtype: types.DType = jnp.float32
    compute_dtype: types.DType | None = None

    def make(self) -> 'T5RelativePositionEmbedding':
      return T5RelativePositionEmbedding(self)

  config: Config

  def setup(self) -> None:
    init_fn = utils.shard_initializer(
        self.config.bias_matrix_init,
        self.config.bias_matrix_sharding,
        labels=[meta.IS_EMBEDDING],
    )
    self.bias_matrix = self.param(
        'embedding',
        init_fn,
        (self.config.num_buckets, self.config.num_heads),
        self.config.param_dtype,
    )

  @property
  def supports_position_bias(self):
    """Returns whether relative position bias is supported."""
    return True

  @jt.typed
  def get_position_bias(
      self,
      query_positions: jaxtyping.Num[jt.ArrayT, '#b q'],
      key_positions: jaxtyping.Num[jt.ArrayT, '#b k'],
      queries: jt.Float[jt.ArrayT, 'b q nq d'],
      keys: jt.Float[jt.ArrayT, 'b k nk d'] | None = None,
  ) -> jt.Float[jt.ArrayT, '#b h q k']:
    """Computes the relative attention logit biases from position indices.

    Args:
      query_positions: [batch, query_length] int/float Tensor containing query
        position indices.
      key_positions: [batch, key_length] int/float Tensor containing position
        indices.
      queries: Is only used to determine the input dtype.
      keys: Has no effect on the output.

    Returns:
      position_biases: [batch, num_heads, query_length, key_length] Tensor
        containing head-wise biases for each query-key pair.
    """
    relative_position = (
        key_positions[:, jnp.newaxis, :] - query_positions[:, :, jnp.newaxis]
    )
    # [batch, query_length, key_length].
    rp_bucket_ids = self._relative_position_bucket(relative_position)

    compute_dtype = utils.get_promoted_dtype(
        queries.dtype, relative_position.dtype, dtype=self.config.compute_dtype
    )
    # [num_head, num_buckets].
    (rp_bucket_vals,) = nn.dtypes.promote_dtype(
        self.bias_matrix,
        dtype=compute_dtype,
        inexact=False,
    )

    # Perform lookup of the bucket values via multiplication with the one-hot
    # array.

    # [batch, query_length, key_length, num_buckets].
    relative_bucket_one_hot = jax.nn.one_hot(
        rp_bucket_ids,
        self.config.num_buckets,
        dtype=compute_dtype,
    )

    # [batch, query_length, key_length, num_heads]
    position_biases = jnp.einsum(
        'bqki,ih->bhqk', relative_bucket_one_hot, rp_bucket_vals
    )
    return position_biases

  @nn.nowrap
  def _relative_position_bucket(
      self, relative_position: jax.Array
  ) -> jax.Array:
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.

    If bidirectional=False, then positive relative positions are invalid.

    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.

    All relative positions >=max_distance map to the same bucket.

    All relative positions <=-max_distance map to the same bucket.

    This should allow for more graceful generalization to longer sequences
    than the model has been trained on.

    Args:
      relative_position: A [query_time, keys_time] int32 Tensor of relative
        positions. r = relative_position[q, k] indicates key timestep k is r
        timesteps ahead (positive) or behind (negative) of query timestep q.

    Returns:
      a Tensor with the same shape as relative_position, containing int32
      values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    num_buckets = self.config.num_buckets
    max_distance = self.config.max_distance
    if self.config.bidirectional:
      num_buckets //= 2
      ret += jnp.array(n < 0, jnp.int32) * num_buckets
      n = jnp.abs(n)
    else:
      n = jnp.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    compute_dtype = jnp.float32
    eps = jnp.finfo(compute_dtype).eps
    # Note that `(num_buckets - 1 - max_exact)` below differs from the
    # reference implementation of T5 relative position biases which uses
    # `(num_buckets - max_exact)`, and therefore doesn't produce the
    # max_distance behavior described in the docstring above.
    val_if_large = max_exact + (
        jnp.log(n.astype(compute_dtype) / max_exact + eps)
        / math.log(max_distance / max_exact)
        * (num_buckets - 1 - max_exact)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    ret += jnp.where(is_small, n, val_if_large)
    return ret
