# Copyright 2025 Google LLC
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
"""A global style token layer."""

import dataclasses
import math

from flax import linen as nn
import jax
import jax.numpy as jnp
import sequence_layers.jax as sl
from sequence_layers.jax import utils


class StyleToken(sl.Stateless):
  """Projects input onto a "style token" bottleneck.

  The inputs are used to predict a soft-weighting of a set of N style token
  vectors. This is equivalent to projecting the input onto an N-dimensional
  simplex formed by the style token vectors.

  Based on: https://arxiv.org/abs/1803.09017
  """

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    """Config for StyleToken."""

    # The number of style tokens to learn per head.
    num_style_tokens: int
    # The number of parallel style token heads to learn.
    num_heads: int
    # The number of units per head.
    units_per_head: int
    # Initializer for the style token embeddings.
    # The original style tokens implementation used glorot_uniform
    # initialization, which initializes with U[-limit, limit],
    # limit = sqrt(6 / (num_style_tokens + units_per_head)).
    # The scale shouldn't depend on the number of tokens so we use fan_out
    # here to initialize with limit = sqrt(6 / units_per_head).
    style_tokens_init: nn.initializers.Initializer = (
        nn.initializers.variance_scaling(
            scale=1.0, mode='fan_out', distribution='uniform'
        )
    )
    # Initializer for the style token keys.
    style_token_keys_init: nn.initializers.Initializer | None = None
    # The dtype to use for layer compute.
    compute_dtype: sl.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: sl.DType = jnp.float32
    # An optional precision to use for the einsum.
    precision: nn.linear.PrecisionLike = None
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'StyleToken':
      return StyleToken(self, name=self.name)

  config: Config

  def setup(self):
    self.style_tokens = self.param(
        'style_tokens',
        self.config.style_tokens_init,
        (self.config.num_style_tokens, self.config.units_per_head),
        self.config.param_dtype,
    )
    style_token_keys_init = self.config.style_token_keys_init
    if style_token_keys_init is None:
      # Traditionally the key vectors are a projection from the style tokens
      # themselves, but making the keys a disconnected variable is
      # equally (potentially more?) flexible and more efficient.
      # Since this variable represents a batch of projection matrices,
      # compute the limit as glorot_uniform for a
      # [num_style_tokens, units_per_head] matrix.
      limit = math.sqrt(
          6.0 / (self.config.num_style_tokens + self.config.units_per_head)
      )
      style_token_keys_init = nn.initializers.uniform(scale=limit)
    self.style_token_keys = self.param(
        'style_token_keys',
        style_token_keys_init,
        (
            1,
            1,
            self.config.num_heads,
            self.config.num_style_tokens,
            self.config.units_per_head,
        ),
        self.config.param_dtype,
    )
    self.query_projection = sl.DenseShaped.Config(
        output_shape=(self.config.num_heads, 1, self.config.units_per_head),
        compute_dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
        name='query_projection',
    ).make()
    self.to_logits = sl.Dense.Config(
        features=1,
        compute_dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
        name='to_logits',
    ).make()

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: sl.ShapeLike,
      *,
      constants: sl.Constants | None = None,
  ) -> sl.Shape:
    return (self.config.num_heads, self.config.units_per_head)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: sl.DType,
      *,
      constants: sl.Constants | None = None,
  ) -> sl.DType:
    return utils.get_promoted_dtype(input_dtype, jnp.float32)

  @sl.check_layer
  def layer(
      self,
      x: sl.Sequence,
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> sl.Sequence:
    del constants

    # Terminology for shapes below:
    # b: Batch size.
    # t: Query time.
    # n: Number of heads.
    # s: Number of style tokens.
    # h: Units per head.

    # Project x to per-head queries: [b, t, n, s=1, h]
    q = self.query_projection.layer(x, training=training)

    # Broadcast-add the key vector across batch, query time, and style tokens
    # and apply tanh nonlinearity. [b, t, n, s, h]
    q = q.apply_values(lambda v: jax.nn.tanh(v + self.style_token_keys))

    # Project the inner h dimension to a single dimension, then squeeze.
    # [b, t, n, s]
    logits = self.to_logits.layer(q, training=training).apply_values(
        jnp.squeeze, axis=-1
    )

    # Normalize the logits to probabilities with softmax.
    # [b, t, n, s]
    probabilities = logits.apply_values(
        utils.run_in_at_least_fp32(jax.nn.softmax)
    )

    # Compute weighted sums of the style tokens.
    context_vector = probabilities.apply_values(
        lambda v: jnp.einsum(
            '...s,sh->...h',
            v,
            self.style_tokens,
            precision=self.config.precision,
        )
    )
    return context_vector
