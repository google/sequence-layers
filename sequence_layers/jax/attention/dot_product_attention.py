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
"""Dot product attention."""

import dataclasses
from flax import linen as nn
import jax.numpy as jnp
import jaxtyping
from sequence_layers.jax import simple
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class DotProductAttention(
    types.Emitting, common.AttentionInputProjectionHelper
):
  """Dot product attention."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for DotProductAttention."""

    # The key to lookup source sequence from constants dictionary.
    source_name: str
    # The number of attention heads.
    num_heads: int
    # The number of units per head.
    units_per_head: int
    # The dropout rate for the attention probabilities.
    attention_probabilities_dropout_rate: float = 0.0
    # Whether to broadcast the dropout across the query time dimension as is
    # done in T5.
    broadcast_dropout_across_queries: bool = False
    # Whether to learn a bias in the query/key/value projection.
    use_bias: bool = False
    # Configuration for the query, key and value input projections.
    input_projection: (
        common.QueryAndKeyValueProjection
        | common.SeparateQueryKeyValueProjection
        | common.QueryAndSharedKeyValueProjection
    ) = dataclasses.field(default_factory=common.QueryAndKeyValueProjection)
    # Optional query processing network. Useful to apply stateful processing to
    # the queries, e.g. enabling RoPE.
    query_network: types.SequenceLayerConfig | None = None
    # Optional key processing network. Useful to apply stateful processing to
    # the keys, e.g. enabling RoPE.
    key_network: types.SequenceLayerConfig | None = None
    # Optional value processing network. Useful to apply stateful processing to
    # the values.
    value_network: types.SequenceLayerConfig | None = None
    # Relative position biases/embeddings for cross-attention.
    # NOTE: Relative position embeddings should be used with care since in
    # cross-attention, the default positions between a query and key may not
    # be properly aligned.
    relative_position_embedding: (
        common.RelativePositionEmbedding.Config | None
    ) = None
    # If set, the query positions for the relative position bias will be taken
    # from this key in the constants dictionary. Else, the position along the
    # time dimension will be used.
    query_positions_name: str | None = None
    # Whether to learn a [units_per_head] query scale factor across all query
    # heads. At initialization (or if per_dim_scale is false), queries are
    # scaled by 1/sqrt(units_per_head) or query_scale.
    per_dim_scale: bool = False
    # Sharding configuration for the per_dim_scale factor.
    per_dim_scale_sharding: types.Sharding | None = None
    # A manual query scale to apply. If unset, queries are scaled by
    # 1/sqrt(units_per_head).
    query_scale: float | None = None
    # If non-zero, a soft cap applied to attention logits to prevent outliers
    # from dominating the softmax. Empirically, 50.0 works well across a variety
    # of tasks. Implemented as tanh(logits / cap) * cap.
    attention_logits_soft_cap: float | None = None
    # Precision config to use for einsums.
    precision: nn.linear.PrecisionLike = None
    # Outputs all-zeros context vectors for queries which have nothing to attend
    # to (i.e. all possible keys are masked).
    zero_fully_masked: bool = False
    # The dtype of the layer's computations.
    compute_dtype: types.DType | None = None
    # The dtype of the layer's parameters.
    param_dtype: types.DType = jnp.float32
    # The number of sink embeddings to include in the key and value.
    # Paper: https://arxiv.org/pdf/2309.17453.pdf.
    num_sink_embeddings: int = 0
    # By default initialize the sink token embeddings to have a norm of 1.
    sink_embeddings_init: nn.initializers.Initializer = (
        nn.linear.default_embed_init
    )
    sink_embeddings_sharding: types.Sharding | None = None
    # If True, use a learned scalar per attention head as an extra "sink" logit
    # for softmax.
    use_sink_scalars: bool = False
    sink_scalars_init: nn.initializers.Initializer = (
        nn.initializers.zeros_init()
    )
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'DotProductAttention':
      return DotProductAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    common.validate_attention(
        self.config.source_name,
        self.config.num_heads,
        self.config.units_per_head,
        self.name,
    )
    if (
        self.config.attention_logits_soft_cap
        and self.config.attention_logits_soft_cap < 0.0
    ):
      raise ValueError(
          f'{self.config.attention_logits_soft_cap=} should be None or non-neg.'
      )

    self._per_dim_scale = None
    if self.config.per_dim_scale:
      self._per_dim_scale = self.param(
          'per_dim_scale',
          utils.shard_initializer(
              nn.initializers.zeros_init(), self.config.per_dim_scale_sharding
          ),
          [self.config.units_per_head],
          self.config.param_dtype,
      )

    if self.config.broadcast_dropout_across_queries:
      # [batch, num_heads, query_time, source_time]
      broadcast_dims = (2,)
    else:
      broadcast_dims = ()
    self._attention_probabilities_dropout = simple.Dropout.Config(
        self.config.attention_probabilities_dropout_rate,
        broadcast_dims=broadcast_dims,
        name='attention_probabilities_dropout',
    ).make()

    # TODO(b/394829779): Support GQA for DotProductAttention.
    num_kv_heads = self.config.num_heads

    self._setup_projection_layers(
        self.config.input_projection,
        num_query_heads=self.config.num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=self.config.units_per_head,
        use_bias=self.config.use_bias,
        precision=self.config.precision,
        compute_dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
        # Not possible for cross attention.
        allow_combined_qkv=False,
    )

    if (
        hasattr(self.config, 'num_sink_embeddings')
        and self.config.num_sink_embeddings > 0
        and hasattr(self.config, 'use_sink_scalars')
        and self.config.use_sink_scalars
    ):
      raise ValueError('Cannot use both sink embeddings and sink scalars.')

    if hasattr(self.config, 'num_sink_embeddings') and (
        self.config.num_sink_embeddings > 0
    ):
      self._sink_key_embeddings = self.param(
          'sink_key_embeddings',
          utils.shard_initializer(
              self.config.sink_embeddings_init,
              self.config.sink_embeddings_sharding,
          ),
          (
              self.config.num_sink_embeddings,
              self.config.num_heads,
              self.config.units_per_head,
          ),
          self.config.param_dtype,
      )
      self._sink_value_embeddings = self.param(
          'sink_value_embeddings',
          utils.shard_initializer(
              self.config.sink_embeddings_init,
              self.config.sink_embeddings_sharding,
          ),
          (
              self.config.num_sink_embeddings,
              self.config.num_heads,
              self.config.units_per_head,
          ),
          self.config.param_dtype,
      )

    elif hasattr(self.config, 'use_sink_scalars') and (
        self.config.use_sink_scalars
    ):
      self._sink_scalars = self.param(
          'sink_scalars',
          self.config.sink_scalars_init,
          (self.config.num_heads,),
          self.config.param_dtype,
      )
      self._sink_value_embeddings = jnp.zeros(
          (
              1,
              self.config.num_heads,
              self.config.units_per_head,
          ),
          dtype=self.config.param_dtype,
      )

    else:
      self._sink_value_embeddings = None

    self.query_network = (
        self.config.query_network.make() if self.config.query_network else None
    )
    if self.query_network and (
        self.query_network.output_ratio != 1
        or self.query_network.block_size != 1
    ):
      raise ValueError(
          'Query network must have an output_ratio'
          f' ({self.query_network.output_ratio}) of 1 and block_size'
          f' ({self.query_network.block_size}) of 1.'
      )

    self.key_network = (
        self.config.key_network.make() if self.config.key_network else None
    )
    if self.key_network and (
        self.key_network.output_ratio != 1 or self.key_network.block_size != 1
    ):
      raise ValueError(
          'Key network must have an output_ratio'
          f' ({self.key_network.output_ratio}) of 1 and block_size'
          f' ({self.key_network.block_size}) of 1.'
      )

    self.value_network = (
        self.config.value_network.make() if self.config.value_network else None
    )
    if self.value_network and (
        self.value_network.output_ratio != 1
        or self.value_network.block_size != 1
    ):
      raise ValueError(
          'Value network must have an output_ratio'
          f' ({self.value_network.output_ratio}) of 1 and block_size'
          f' ({self.value_network.block_size}) of 1.'
      )
    self.relative_position_embedding = None
    if self.config.relative_position_embedding:
      self.relative_position_embedding = (
          self.config.relative_position_embedding.make()
      )
      if not self.relative_position_embedding.supports_position_bias:
        raise ValueError(
            'Relative position embedding for cross-attention must support'
            f' position bias. Got {self.config.relative_position_embedding=}.'
        )

  @property
  def supports_step(self) -> bool:
    supports_step = True

    if self.query_network:
      supports_step = supports_step and self.query_network.supports_step

    return supports_step

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    if self.query_network:
      return self.query_network.receptive_field_per_step
    else:
      return {0: (0, 0)}

  def _get_source(self, constants: types.Constants | None) -> types.Sequence:
    return common.get_source(
        self, self.config.source_name, constants, required_rank=3
    )

  def _get_query_positions_from_constants(
      self, constants: types.Constants | None
  ) -> types.Sequence:
    return common.get_source(
        self, self.config.query_positions_name, constants, required_rank=2
    )

  @jt.typed
  @nn.nowrap
  def _get_query_positions(
      self,
      batch_size: int,
      queries_seq_length: int,
      constants: types.Constants | None = None,
      time_step: jt.Int[jt.ArrayT, '#b'] | int = 0,
  ) -> jaxtyping.Num[jt.ArrayT, '#b q']:
    """Returns query positions for relative position biases."""
    if self.config.query_positions_name:
      # Use the position source as query position.
      query_positions = self._get_query_positions_from_constants(
          constants
      ).values
      if query_positions.shape != (batch_size, queries_seq_length):
        raise ValueError(
            f'Query positions shape {query_positions.shape} does not match'
            f' expected shape ({batch_size}, {queries_seq_length}).'
        )
    else:
      if isinstance(time_step, int):
        time_step = jnp.full((1,), time_step, dtype=jnp.int32)
      query_positions = (
          time_step[:, jnp.newaxis]
          + jnp.arange(queries_seq_length)[jnp.newaxis, :]
      )
    return query_positions

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Pre-process the source with key/value projections and networks.
    source = self._get_source(constants)

    keys, values = self.get_kv(self.config.input_projection, source)

    if self.key_network:
      keys = self.key_network.layer(
          keys, training=training, constants=constants
      )

    if self.value_network:
      values = self.value_network.layer(
          values, training=training, constants=constants
      )

    if self.query_network:
      query_state = self.query_network.get_initial_state(
          batch_size,
          types.ShapeDType(
              (self.config.num_heads, self.config.units_per_head),
              input_spec.dtype,
          ),
          training=training,
          constants=constants,
      )
    else:
      query_state = ()

    time_step = jnp.zeros((batch_size,), dtype=jnp.int32)

    # Mask before storing in state:
    keys = keys.mask_invalid()
    values = values.mask_invalid()
    mask = utils.combine_mask(keys.mask, values.mask)

    return (keys.values, values.values, mask, query_state, time_step)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
      *,
      constants: types.Constants | None = None,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'DotProductAttention requires rank 3 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    return (self.config.num_heads, self.config.units_per_head)

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    keys, values, mask, query_state, time_step = state

    x_num_timesteps = x.shape[1]

    # No mask required, since query timesteps are independent.
    queries = self.get_q(self.config.input_projection, x)

    if self.query_network:
      queries, query_state = self.query_network.step(
          queries, query_state, training=training, constants=constants
      )

    y, emits = self._attention(
        queries,
        # get_initial_state masks before storing in state.
        types.MaskedSequence(keys, mask),
        types.MaskedSequence(values, mask),
        training=training,
        constants=constants,
        time_step=time_step,
    )
    time_step += x_num_timesteps
    state = (keys, values, mask, query_state, time_step)
    return y, state, emits

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    source = self._get_source(constants)

    keys, values = self.get_kv(self.config.input_projection, source)

    if self.key_network:
      keys = self.key_network.layer(
          keys, training=training, constants=constants
      )

    if self.value_network:
      values = self.value_network.layer(
          values, training=training, constants=constants
      )

    # No mask required, since query timesteps are independent.
    queries = self.get_q(self.config.input_projection, x)

    if self.query_network:
      queries = self.query_network.layer(
          queries,
          training=training,
          constants=constants,
      )

    return self._attention(
        queries,
        keys,
        values,
        constants=constants,
        training=training,
    )

  @nn.nowrap
  def _attention(
      self,
      query: types.Sequence,
      keys: types.Sequence,
      values: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
      time_step: jaxtyping.Num[jt.ArrayT, 'b k'] | int = 0,
  ) -> tuple[types.Sequence, types.Emits]:
    batch_size, queries_seq_len = query.shape[:2]
    keys_seq_len = keys.shape[1]
    compute_dtype = utils.get_promoted_dtype(
        query.values.dtype,
        keys.values.dtype,
        dtype=self.config.compute_dtype,
    )

    # Guard against NaN/Inf in values, since values are contracted when
    # computing context vectors.
    values = values.mask_invalid()
    valid_mask = keys.mask[:, jnp.newaxis, jnp.newaxis, :]
    logit_bias = None
    if self.relative_position_embedding:
      query_positions = self._get_query_positions(
          batch_size, queries_seq_len, constants, time_step
      )
      key_positions = jnp.arange(keys_seq_len)[jnp.newaxis, :]
      # logit_bias is of shape [b, h, q, k].
      logit_bias = self.relative_position_embedding.get_position_bias(
          query_positions=query_positions,
          key_positions=key_positions,
          queries=query.values,
          keys=keys.values,
      )

    if hasattr(self.config, 'num_sink_embeddings') and (
        self.config.num_sink_embeddings > 0
    ):
      # TODO(b/414834251): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', query.values, self._sink_key_embeddings
      )
      sink_value_embeddings = self._sink_value_embeddings
    elif (
        hasattr(self.config, 'use_sink_scalars')
        and self.config.use_sink_scalars
    ):
      sink_key_logits = jnp.tile(
          self._sink_scalars[None, :, None, None],
          (query.shape[0], 1, query.shape[1], 1),
      )
      sink_value_embeddings = self._sink_value_embeddings
    else:
      sink_key_logits = None
      sink_value_embeddings = None

    context_vectors, probabilities = common.dot_product_attention(
        queries=query.values,
        keys=keys.values,
        values=values.values,
        logit_visibility_mask=valid_mask,
        logit_bias=logit_bias,
        training=training,
        attention_logits_soft_cap=self.config.attention_logits_soft_cap,
        attention_probabilities_dropout=self._attention_probabilities_dropout,
        precision=self.config.precision,
        per_dim_scale=self._per_dim_scale,
        query_scale=self.config.query_scale,
        get_logits_fn=None,
        zero_fully_masked=self.config.zero_fully_masked,
        compute_dtype=compute_dtype,
        num_sink_positions=self.config.num_sink_embeddings
        + self.config.use_sink_scalars,
        sink_key_logits=sink_key_logits,
        sink_value_embeddings=sink_value_embeddings,
    )
    emits = common.CrossAttentionEmits(
        {self.config.source_name: types.Sequence(probabilities, query.mask)}
    )
    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, query.mask)
    return context_vectors, emits
