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
"""TransformerXL-style relative position embeddings."""

import dataclasses
from flax import linen as nn
import jax
import jax.numpy as jnp
from sequence_layers.jax import meta
from sequence_layers.jax import types
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class TransformerXLRelativePositionEmbedding(common.RelativePositionEmbedding):
  """TransformerXL-style relative position embeddings.

  TODO(rryan): Currently only works with LocalDotProductSelfAttention due to
  batched queries in get_logits_streaming.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(common.RelativePositionEmbedding.Config):
    """Config for TransformerXLRelativePositionEmbedding."""

    num_heads: int
    units_per_head: int
    max_backward: int
    max_forward: int
    position_bias_dim: int
    use_bias: bool = True
    u_sharding: types.Sharding | None = None
    v_sharding: types.Sharding | None = None
    pos_proj_sharding: types.Sharding | None = None
    param_dtype: types.DType = jnp.float32
    einsum_factory: types.EinsumFactoryT | None = None

    def make(self) -> 'TransformerXLRelativePositionEmbedding':
      return TransformerXLRelativePositionEmbedding(self)

  config: Config

  def setup(self) -> None:
    self.pos_proj = utils.FlaxEinsumDense(
        '...d,dnh->...nh',
        output_shape=(self.config.num_heads, self.config.units_per_head),
        kernel_init=utils.shard_initializer(
            nn.linear.default_kernel_init,
            self.config.pos_proj_sharding,
            projectable=True,
            axes_types=(meta.AxisType.FANIN, None, None),
        ),
        param_dtype=self.config.param_dtype,
        einsum_factory=self.config.einsum_factory,
        name='pos_proj',
    )
    if self.config.use_bias:
      self.u = self.param(
          'u',
          utils.shard_initializer(
              nn.initializers.zeros_init(), self.config.u_sharding
          ),
          [self.config.num_heads, self.config.units_per_head],
          self.config.param_dtype,
      )
      self.v = self.param(
          'v',
          utils.shard_initializer(
              nn.initializers.zeros_init(), self.config.v_sharding
          ),
          [self.config.num_heads, self.config.units_per_head],
          self.config.param_dtype,
      )

  @property
  def supports_get_logits(self):
    """Returns whether get_logits is supported."""
    return True

  def get_logits_streaming(
      self,
      queries: jax.Array,
      keys: jax.Array,
      precision: nn.linear.PrecisionLike,
  ) -> jax.Array:
    # Compute term_ac.
    term_ac = jnp.einsum(
        'BiNH,BjNH->BNij',
        queries + self.u.astype(queries.dtype)
        if self.config.use_bias
        else queries,
        keys,
        precision=precision,
    )

    b = queries.shape[0]
    w = queries.shape[1]
    n = self.config.num_heads
    l = self.config.max_backward
    r = self.config.max_forward
    c = w + l + r

    # For now we can't step with future context. We'll support this eventually
    # so keep it in the formulae.
    assert r == 0
    assert keys.shape[1] == c, keys.shape

    pos = jnp.arange(l, -r - 1, -1)[jnp.newaxis, :]
    assert pos.shape[1] == l + r + 1

    # [1, F, position_bias_dim]
    sin_emb = utils.get_timing_signal_1d_pos(
        pos,
        self.config.position_bias_dim,
        min_timescale=1,
        max_timescale=10000,
        dtype=queries.dtype,
    )
    # [1, F, N, H]
    sin_emb = self.pos_proj(sin_emb)
    # [F, N, H]
    sin_emb = jnp.squeeze(sin_emb, 0)

    # [B, N, U, W, F]
    term_bd = jnp.einsum(
        'BiNH,FNH->BNiF',
        queries + self.v.astype(queries.dtype)
        if self.config.use_bias
        else queries,
        sin_emb,
        precision=precision,
    )

    # Perform relative shift in order to get [B, N, i, C]
    # Pads the input to [B, N, i, C + 1]
    term_bd = jnp.pad(
        term_bd, ((0, 0), (0, 0), (0, 0), (0, (c + 1) - (l + r + 1)))
    )
    term_bd = jnp.reshape(term_bd, [b, n, w * (c + 1)])
    term_bd = term_bd[:, :, : w * c]
    # Reshapes to [B, N, W, C]. Note the output last dim is 1-smaller
    # than the input, which "pushes" one element off to the next row for each
    # row. The accumulated effect is row_i is right-shifted i steps (i>=0).
    term_bd = jnp.reshape(term_bd, [b, n, w, c])

    return term_ac + term_bd

  def get_logits(
      self,
      queries: jax.Array,
      keys: jax.Array,
      precision: nn.linear.PrecisionLike,
  ) -> jax.Array:
    # Compute term_ac.
    term_ac = jnp.einsum(
        'BuwNH,BucNH->BNuwc',
        queries + self.u.astype(queries.dtype)
        if self.config.use_bias
        else queries,
        keys,
        precision=precision,
    )

    b = queries.shape[0]
    u = queries.shape[1]
    w = queries.shape[2]
    c = keys.shape[2]
    n = self.config.num_heads
    l = self.config.max_backward
    r = self.config.max_forward
    assert c == w + l + r

    pos = jnp.arange(l, -r - 1, -1)[jnp.newaxis, :]
    assert pos.shape[1] == l + r + 1

    # [1, F, position_bias_dim]
    sin_emb = utils.get_timing_signal_1d_pos(
        pos,
        self.config.position_bias_dim,
        min_timescale=1,
        max_timescale=10000,
        dtype=queries.dtype,
    )
    # [1, F, N, H]
    sin_emb = self.pos_proj(sin_emb)
    # [F, N, H]
    sin_emb = jnp.squeeze(sin_emb, 0)

    # [B, N, U, W, F]
    term_bd = jnp.einsum(
        'BuwNH,FNH->BNuwF',
        queries + self.v.astype(queries.dtype)
        if self.config.use_bias
        else queries,
        sin_emb,
        precision=precision,
    )

    # Perform relative shift in order to get [B, N, U, W, C]
    # Pads the input to [B, N, U, W, C + 1]
    term_bd = jnp.pad(
        term_bd, ((0, 0), (0, 0), (0, 0), (0, 0), (0, (c + 1) - (l + r + 1)))
    )
    term_bd = jnp.reshape(term_bd, [b, n, u, w * (c + 1)])
    term_bd = term_bd[:, :, :, : w * c]
    # Reshapes to [B, N, U, W, C]. Note the output last dim is 1-smaller
    # than the input, which "pushses" one element off to the next row for each
    # row. The accumulated effect is row_i is right-shifted i steps (i>=0).
    term_bd = jnp.reshape(term_bd, [b, n, u, w, c])

    return term_ac + term_bd
