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
"""Streaming dot-product attention layer."""

import dataclasses
from flax import linen as nn
import jax
import jax.numpy as jnp
from sequence_layers.jax import simple
from sequence_layers.jax import types
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class StreamingDotProductAttention(
    types.Emitting, common.AttentionInputProjectionHelper
):
  """A multi-headed streaming dot-product attention layer.

  Unlike most SequenceLayers, this cross-attention layer assumes that when using
  the step-wise APIs, the source sequence provided in the constants dictionary
  is provided in a streaming fashion with the same block size as the input to
  the step API. The layer-wise API functions like DotProductAttention, and
  expects the entire source sequence to be provided at once.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for StreamingDotProductAttention."""

    # The key to lookup source sequence from constants dictionary.
    source_name: str

    # The number of attention heads.
    num_heads: int
    # The number of units per head.
    units_per_head: int

    # The number of past timesteps each timestep can see. Must be non-negative.
    # 0: No past timesteps are visible.
    max_past_horizon: int
    # The number of future timesteps each timestep can see. Must be
    # non-negative.
    # 0: No future timesteps are visible.
    max_future_horizon: int = 0
    # An optional RelativePositionEmbedding to use to compute relative position
    # biases or logits.
    relative_position_embedding: (
        common.RelativePositionEmbedding.Config | None
    ) = None
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
    # If non-zero, a soft cap applied to attention logits to prevent outliers
    # from dominating the softmax. Empirically, 50.0 works well across a variety
    # of tasks. Implemented as tanh(logits / cap) * cap.
    attention_logits_soft_cap: float | None = None
    # Whether to learn a [units_per_head] query scale factor across all query
    # heads. At initialization (or if per_dim_scale is false), queries are
    # scaled by 1/sqrt(units_per_head) or query_scale.
    per_dim_scale: bool = False
    # Sharding configuration for the per_dim_scale factor.
    per_dim_scale_sharding: types.Sharding | None = None
    # A manual query scale to apply. If unset, queries are scaled by
    # 1/sqrt(units_per_head).
    query_scale: float | None = None
    # Precision config to use for all operations.
    precision: nn.linear.PrecisionLike = None
    # Outputs all-zeros context vectors for queries which have nothing to attend
    # to (i.e. all possible keys are masked).
    zero_fully_masked: bool = False
    # If true, uses a query delay buffer for supporting max_future_horizon > 0.
    # If false, assumes the user has handled delaying the queries with respect
    # to the source outside of this layer.
    use_query_delay_buffer: bool = True
    # The dtype of the layer's computations.
    compute_dtype: types.DType | None = None
    # DType of parameters.
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
    # Whether to use an experimental ring buffer implementation for the KV cache
    # updates. This implementation is more compute and memory efficient than the
    # default implementation on TPU.
    #
    # Limitations:
    # * Incompatible with attention sinks.
    # * Incompatible with relative_position_embedding.
    # * Requires streaming step sizes of 1.
    use_kv_cache_ringbuffer: bool = False
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'StreamingDotProductAttention':
      return StreamingDotProductAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    common.validate_attention(
        self.config.source_name,
        self.config.num_heads,
        self.config.units_per_head,
        self.name,
    )
    if self.config.max_past_horizon < 1:
      raise ValueError(
          f'Expected {self.config.max_past_horizon=} >= 1 for {self.name}.'
      )
    if self.config.max_future_horizon < 0:
      raise ValueError(
          f'Expected {self.config.max_future_horizon=} >= 0 for {self.name}.'
      )
    if (
        self.config.max_future_horizon == 0
        and self.config.max_past_horizon == 0
    ):
      raise ValueError(
          'Both max_horizon and max_future_horizon are 0, which '
          f'does not make sense for {self}.'
      )

    if (
        self.config.attention_logits_soft_cap
        and self.config.attention_logits_soft_cap < 0.0
    ):
      raise ValueError(
          f'{self.config.attention_logits_soft_cap=} should be None or non-neg.'
      )
    # TODO(b/394829779): Support GQA for StreamingDotProductAttention.
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

    if self.config.num_sink_embeddings > 0 and self.config.use_sink_scalars:
      raise NotImplementedError(
          'Cannot use both sink embeddings and sink scalars.'
      )

    if self.config.num_sink_embeddings > 0:
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

    elif self.config.use_sink_scalars:
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

    if self.config.use_kv_cache_ringbuffer:
      if self.config.num_sink_embeddings > 0 or self.config.use_sink_scalars:
        raise NotImplementedError(
            'Sinks are not supported with use_kv_cache_ringbuffer.'
        )
      if self.config.relative_position_embedding:
        raise NotImplementedError(
            'Relative position embeddings are not supported with'
            ' use_kv_cache_ringbuffer.'
        )

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

    self.relative_position_embedding = (
        (self.config.relative_position_embedding.make())
        if self.config.relative_position_embedding
        else None
    )
    if (
        self.relative_position_embedding
        and self.relative_position_embedding.supports_position_bias
    ):
      raise ValueError(
          f'{self} does not support relative position embeddings with position'
          ' bias.'
      )

  @property
  def supports_step(self) -> bool:
    supports_step = (
        self.config.max_future_horizon >= 0
        and self.config.max_past_horizon >= 0
        # Relative position embeddings and future horizon > 0 are not currently
        # supported.
        and (
            not self.config.max_future_horizon
            or not self.config.relative_position_embedding
        )
    )
    if self.query_network:
      supports_step = supports_step and self.query_network.supports_step

    if self.key_network:
      supports_step = supports_step and self.key_network.supports_step

    if self.value_network:
      supports_step = supports_step and self.value_network.supports_step

    return supports_step

  @property
  def input_latency(self) -> int:
    return (
        self.config.max_future_horizon
        if self.config.max_future_horizon >= 0
        and self.config.use_query_delay_buffer
        else 0
    )

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    if self.query_network:
      return self.query_network.receptive_field_per_step
    return {0: (0, 0)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    compute_dtype = self.get_input_projection_output_dtype(
        self.config.input_projection, input_spec.dtype, constants=constants
    )
    # State to contain the max_past_horizon + max_future_horizon projected keys
    # and values. Note, the initial state is invalid since we don't want to
    # attend to it.
    max_past_horizon = max(0, self.config.max_past_horizon)
    max_future_horizon = max(0, self.config.max_future_horizon)
    kv_buffer_size = max_past_horizon + max_future_horizon
    kv_zero_values = jnp.zeros(
        (
            batch_size,
            kv_buffer_size,
            self.config.num_heads,
            self.config.units_per_head,
        ),
        dtype=compute_dtype,
    )
    kv_zero_mask = jnp.zeros(
        [batch_size, kv_buffer_size], dtype=types.MASK_DTYPE
    )

    # Store these two sequences "unpacked" since they share a mask.
    kv_buffer_keys = kv_zero_values
    kv_buffer_values = kv_zero_values
    kv_buffer_mask = kv_zero_mask

    # If we have a finite future horizon, we cannot produce outputs for timestep
    # t until the max_future_horizon KV timesteps have arrived. Store incoming
    # queries in a delay buffer so we do not compute context vectors for them
    # until max_future_horizon KV timesteps have arrived.
    if max_future_horizon and self.config.use_query_delay_buffer:
      # Queries are buffered unmasked since we delay masking until we have
      # computed context vectors.
      query_delay_buffer = types.Sequence(
          jnp.zeros(
              (
                  batch_size,
                  max_future_horizon,
                  self.config.num_heads,
                  self.config.units_per_head,
              ),
              dtype=compute_dtype,
          ),
          jnp.zeros([batch_size, max_future_horizon], dtype=types.MASK_DTYPE),
      )
    else:
      query_delay_buffer = ()

    query_key_input_spec = types.ShapeDType(
        (self.config.num_heads, self.config.units_per_head), compute_dtype
    )

    if self.query_network:
      query_network_state = self.query_network.get_initial_state(
          batch_size,
          query_key_input_spec,
          training=training,
          constants=constants,
      )
    else:
      query_network_state = ()
    if self.key_network:
      key_network_state = self.key_network.get_initial_state(
          batch_size,
          query_key_input_spec,
          training=training,
          constants=constants,
      )
    else:
      key_network_state = ()
    if self.value_network:
      value_network_state = self.value_network.get_initial_state(
          batch_size,
          query_key_input_spec,
          training=training,
          constants=constants,
      )
    else:
      value_network_state = ()

    if self.config.use_kv_cache_ringbuffer:
      # We only need timestep tracking for KV cache insertions.
      time_step = jnp.zeros([batch_size], dtype=jnp.int32)
    else:
      time_step = ()

    return (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
        time_step,
    )

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
          'DotProductSelfAttention requires rank 3 input got:'
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
    if not self.supports_step:
      raise ValueError(f'{self} is not steppable.')
    batch_size = x.shape[0]
    x_values_time = x.shape[1]

    (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
        time_step,
    ) = state
    kv_buffer_size = kv_buffer_keys.shape[1]

    source = common.get_source(self, self.config.source_name, constants)

    if x.shape[1] != source.shape[1]:
      raise ValueError(f'Expected {x.shape[1]=} to match {source.shape[1]=}')

    # No mask required, since query timesteps are independent.
    queries = self.get_q(self.config.input_projection, x)

    # Our params might not match param_dtype, so delegate the compute_dtype to
    # the output of the QKV layer.
    compute_dtype = queries.dtype

    if self.query_network:
      queries, query_network_state = self.query_network.step(
          queries,
          query_network_state,
          training=training,
          constants=constants,
      )

    keys, values = self.get_kv(self.config.input_projection, source)

    if self.key_network:
      keys, key_network_state = self.key_network.step(
          keys,
          key_network_state,
          training=training,
          constants=constants,
      )
    if self.value_network:
      values, value_network_state = self.value_network.step(
          values,
          value_network_state,
          training=training,
          constants=constants,
      )

    # Guard against NaN/Inf in values, since values are contracted when
    # computing context vectors.
    values = values.mask_invalid()

    # The key and value network could have changed the mask, so we combine the
    # keys and values mask. This is inexpensive but would be nice to skip.
    combined_mask = utils.combine_mask(keys.mask, values.mask)

    if self.config.use_kv_cache_ringbuffer:
      if queries.shape[1] != 1:
        raise ValueError(
            'use_kv_cache_ringbuffer requires a step size of 1. Got:'
            f' {queries.shape=}'
        )
      # Leave the input key/values and KV cache separate for multi key/value dot
      # product attention.
      kv_buffer_time = kv_buffer_size
    else:
      # To process a step, concatenate our kv_buffer_size KV buffer with the
      # input x_values_time timesteps.
      kv_buffer_keys = jnp.concatenate([kv_buffer_keys, keys.values], axis=1)
      kv_buffer_values = jnp.concatenate(
          [kv_buffer_values, values.values], axis=1
      )
      # The mask could differ based on key_network / value_network.
      kv_buffer_mask = jnp.concatenate([kv_buffer_mask, combined_mask], axis=1)
      kv_buffer_time = x_values_time + kv_buffer_size

    # If we have a query delay buffer, we need to insert the current block of
    # queries into it, and pop the oldest x_values_time queries off. If no query
    # cache, the below logic is a no-op so we save the compute of running it.
    if query_delay_buffer:
      assert self.config.max_future_horizon > 0
      query_delay_buffer = types.Sequence.concatenate_sequences(
          [query_delay_buffer, queries]
      )
      assert (
          query_delay_buffer.shape[1]
          == x_values_time + self.config.max_future_horizon
      )

      # Use the oldest x_values_time queries as the current step's queries. They
      # each have max_future_horizon context available in
      # kv_buffer_keys/kv_buffer_values so we can produce valid output for them.
      queries = query_delay_buffer[:, :x_values_time]

      # Preserve the last max_future_horizon queries for the next step.
      query_delay_buffer = query_delay_buffer[
          :, -self.config.max_future_horizon :
      ]

    if self.config.num_sink_embeddings > 0:
      # TODO(b/414834251): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', queries.values, self._sink_key_embeddings
      )
    elif self.config.use_sink_scalars:
      sink_key_logits = jnp.tile(
          self._sink_scalars[None, :, None, None],
          [queries.shape[0], 1, queries.shape[1], 1],
      )
    else:
      sink_key_logits = None

    valid_mask = kv_buffer_mask[:, jnp.newaxis, jnp.newaxis, :]
    if (
        visibility_mask := common.self_attention_step_visibility_mask(
            self.config.max_past_horizon,
            self.config.max_future_horizon,
            x_values_time,
            kv_buffer_time,
        )
    ) is not None:
      # Broadcasting across batch_size, heads and query time.
      valid_mask = jnp.logical_and(valid_mask, visibility_mask)

    utils.assert_is_compatible_with(
        valid_mask.shape,
        [batch_size, 1, x_values_time, kv_buffer_time],
    )

    get_logits_fn = None
    if (
        self.relative_position_embedding
        and self.relative_position_embedding.supports_get_logits
    ):
      get_logits_fn = self.relative_position_embedding.get_logits_streaming

    # Compute context vectors:
    if self.config.use_kv_cache_ringbuffer:
      assert not self.relative_position_embedding
      assert not self.config.num_sink_embeddings
      assert not self.config.use_sink_scalars
      assert not get_logits_fn

      context_vectors, probabilities = (
          common.multi_key_value_dot_product_attention(
              queries=queries.values,
              kv_buffers=(
                  (kv_buffer_keys, kv_buffer_values, valid_mask),
                  (
                      keys.values,
                      values.values,
                      combined_mask[:, jnp.newaxis, jnp.newaxis, :],
                  ),
              ),
              logit_bias=None,
              training=training,
              attention_logits_soft_cap=self.config.attention_logits_soft_cap,
              attention_probabilities_dropout=self._attention_probabilities_dropout,
              per_dim_scale=self._per_dim_scale,
              query_scale=self.config.query_scale,
              precision=self.config.precision,
              get_logits_fn=None,
              zero_fully_masked=self.config.zero_fully_masked,
              compute_dtype=compute_dtype,
          )
      )
      probabilities = jnp.concatenate(probabilities, axis=-1)
    else:
      context_vectors, probabilities = common.dot_product_attention(
          queries=queries.values,
          keys=kv_buffer_keys,
          values=kv_buffer_values,
          logit_visibility_mask=valid_mask,
          logit_bias=None,
          training=training,
          attention_logits_soft_cap=self.config.attention_logits_soft_cap,
          attention_probabilities_dropout=self._attention_probabilities_dropout,
          per_dim_scale=self._per_dim_scale,
          query_scale=self.config.query_scale,
          precision=self.config.precision,
          get_logits_fn=get_logits_fn,
          zero_fully_masked=self.config.zero_fully_masked,
          compute_dtype=compute_dtype,
          num_sink_positions=self.config.num_sink_embeddings
          + self.config.use_sink_scalars,
          sink_key_logits=sink_key_logits,
          sink_value_embeddings=self._sink_value_embeddings,
      )

    # Update KV caches and state.
    if self.config.use_kv_cache_ringbuffer:
      # Write latest keys, values and masks to the appropriate position in the
      # KV cache ring buffers.
      i = time_step % kv_buffer_size
      assert combined_mask.shape[1] == keys.shape[1]
      assert values.shape[1] == keys.shape[1]

      time_step += x_values_time

      update_fn = jax.vmap(lambda x, u, i: x.at[i].set(u.squeeze(0)))
      kv_buffer_keys = update_fn(kv_buffer_keys, keys.values, i)
      kv_buffer_values = update_fn(kv_buffer_values, values.values, i)
      kv_buffer_mask = update_fn(kv_buffer_mask, combined_mask, i)
    else:
      # Preserve last max_past_horizon as state for next step.
      kv_buffer_keys = kv_buffer_keys[:, -kv_buffer_size:]
      kv_buffer_values = kv_buffer_values[:, -kv_buffer_size:]
      kv_buffer_mask = kv_buffer_mask[:, -kv_buffer_size:]

    state = (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
        time_step,
    )

    emits = common.CrossAttentionEmits(
        {self.config.source_name: types.Sequence(probabilities, queries.mask)}
    )

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, queries.mask)

    return context_vectors, state, emits

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    source = common.get_source(self, self.config.source_name, constants)

    # No mask required, since query timesteps are independent.
    queries = self.get_q(self.config.input_projection, x)
    queries_time = queries.shape[1]

    # Our params might not match param_dtype, so delegate the compute_dtype to
    # the output of the QKV layer.
    compute_dtype = queries.dtype

    if self.query_network:
      queries = self.query_network.layer(
          queries,
          training=training,
          constants=constants,
      )

    keys, values = self.get_kv(self.config.input_projection, source)
    keys_time = keys.shape[1]

    if self.key_network:
      keys = self.key_network.layer(
          keys,
          training=training,
          constants=constants,
      )
    if self.value_network:
      values = self.value_network.layer(
          values,
          training=training,
          constants=constants,
      )

    # Guard against NaN/Inf in values, since values are contracted when
    # computing context vectors.
    values = values.mask_invalid()

    if self.config.num_sink_embeddings > 0:
      # TODO(b/414834251): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', queries.values, self._sink_key_embeddings
      )
    elif self.config.use_sink_scalars:
      sink_key_logits = jnp.tile(
          self._sink_scalars[None, :, None, None],
          [queries.shape[0], 1, queries.shape[1], 1],
      )
    else:
      sink_key_logits = None

    get_logits_fn = None
    if (
        self.relative_position_embedding
        and self.relative_position_embedding.supports_get_logits
    ):
      get_logits_fn = self.relative_position_embedding.get_logits

    # Mask out invalid timesteps in the input sequence so that we do not
    # attend to invalid timesteps. By shaping it [b, 1, 1, key_time], we
    # ensure that each query timestep cannot see invalid timesteps. If the
    # query timestep itself is invalid, it will be masked below
    valid_mask = keys.mask[:, jnp.newaxis, jnp.newaxis, :]

    visibility_mask = utils.ones_matrix_band_part(
        queries_time,
        keys_time,
        num_lower=self.config.max_past_horizon,
        num_upper=self.config.max_future_horizon,
    )
    valid_mask = jnp.logical_and(visibility_mask, valid_mask)

    context_vectors, probabilities = common.dot_product_attention(
        queries=queries.values,
        keys=keys.values,
        values=values.values,
        logit_visibility_mask=valid_mask,
        logit_bias=None,
        training=training,
        attention_logits_soft_cap=self.config.attention_logits_soft_cap,
        attention_probabilities_dropout=self._attention_probabilities_dropout,
        precision=self.config.precision,
        per_dim_scale=self._per_dim_scale,
        query_scale=self.config.query_scale,
        get_logits_fn=get_logits_fn,
        zero_fully_masked=self.config.zero_fully_masked,
        compute_dtype=compute_dtype,
        num_sink_positions=self.config.num_sink_embeddings
        + self.config.use_sink_scalars,
        sink_key_logits=sink_key_logits,
        sink_value_embeddings=self._sink_value_embeddings,
    )
    emits = common.CrossAttentionEmits(
        {self.config.source_name: types.Sequence(probabilities, x.mask)}
    )

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, x.mask)

    return context_vectors, emits
