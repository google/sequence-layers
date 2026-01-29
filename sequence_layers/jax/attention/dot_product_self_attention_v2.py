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
"""Dot product self attention layer v2."""

import dataclasses
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import types
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class DotProductSelfAttentionV2(
    types.Emitting, common.AttentionInputProjectionHelper
):
  """A multi-headed dot-product self attention layer.

  This layer is a rewrite of the original DotProductSelfAttention layer with
  flash attention support.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for DotProductSelfAttentionV2."""

    # The number of attention heads. If num_kv_heads is set, num_heads must be
    # divisible by num_kv_heads.
    num_heads: int
    # The number of units per head.
    units_per_head: int
    # The number of past timesteps each timestep can see.
    # -1: Disable masking of the past (all past timesteps are visible)
    # 0: No past timesteps are visible.
    # The layer is only steppable when max_past_horizon >= 0.
    max_past_horizon: int
    # The number of future timesteps each timestep can see.
    # -1: Disable masking of the future (all future timesteps are visible)
    # 0: No future timesteps are visible.
    # The layer is only steppable when max_future_horizon >= 0.
    max_future_horizon: int = 0

    # If set, the number of heads to use for key/value projections. If
    # num_kv_heads is set, num_heads must be divisible by num_kv_heads.
    num_kv_heads: int | None = None

    # Whether to learn a bias in the query/key/value projection.
    use_bias: bool = False
    # Configuration for the query, key and value input projection parameters.
    # If num_kv_heads is set, must not be CombinedQueryKeyValueProjection.
    # If shared_kv_projection is set, must be QueryAndSharedKeyValueProjection.
    input_projection: common.QueryKeyValueProjectionConfig = (
        dataclasses.field(
            default_factory=common.CombinedQueryKeyValueProjection
        )
    )
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
    # Precision config to use for einsums.
    precision: nn.linear.PrecisionLike = None
    # The dtype of the layer's computations.
    compute_dtype: types.DType | None = None
    # The dtype of the layer's parameters.
    param_dtype: types.DType = jnp.float32

    # Whether to use an experimental ring buffer implementation for the KV cache
    # updates. This implementation is more compute and memory efficient than the
    # default implementation on TPU.
    #
    # Limitations:
    # * Incompatible with attention sinks.
    # * Incompatible with GQA.
    # * Incompatible with relative_position_embedding.
    # * Requires streaming step sizes of 1.
    use_kv_cache_ringbuffer: bool = False

    # If enabled, the name of the segment IDs in the constants. Only timesteps
    # with the same segment ID will be allowed to attend to each other.
    segment_ids_name: str | None = None
    # If enabled, the name of a [b, t] position in the constants. If set, the
    # position will be provided externally to the layer. If unset, position will
    # be tracked internally.
    position_name: str | None = None

    # Flash attention block sizes. If non-None, uses flash attention to split
    # the attention computation into smaller blocks. This reduces peak memory
    # usage and may lead to speedups. On TPU, block sizes less than 128 are
    # typically not worth it.
    flash_attention_query_block_size: int | None = None
    flash_attention_key_block_size: int | None = None
    # Experimental: Whether to remat the entire attention computation.
    experimental_remat_outer: bool = False
    # Experimental: Whether ot remat the inner query block calculations.
    experimental_remat_inner: bool = False

    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'DotProductSelfAttentionV2':
      return DotProductSelfAttentionV2(self, name=self.name)

  config: Config

  def setup(self) -> None:
    common.validate_heads(
        self.config.num_heads, self.config.units_per_head, self.name
    )
    num_kv_heads = self.config.num_kv_heads or self.config.num_heads
    self._setup_projection_layers(
        self.config.input_projection,
        num_query_heads=self.config.num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=self.config.units_per_head,
        use_bias=self.config.use_bias,
        precision=self.config.precision,
        compute_dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
    )
    if self.config.max_past_horizon < -1:
      raise ValueError(
          f'Expected max_horizon >= -1 for {self}, got'
          f' {self.config.max_past_horizon}.'
      )
    if self.config.max_future_horizon < -1:
      raise ValueError(
          'Expected max_future_horizon >= -1 for '
          f'{self}, got {self.config.max_future_horizon}.'
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

    if self.config.use_kv_cache_ringbuffer:
      num_kv_heads = self.config.num_kv_heads or self.config.num_heads
      if num_kv_heads != self.config.num_heads:
        raise NotImplementedError(
            'num_kv_heads is not supported with use_kv_cache_ringbuffer.'
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

  @property
  def supports_step(self) -> bool:
    supports_step = (
        self.config.max_future_horizon >= 0
        and self.config.max_past_horizon >= 0
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
    # Default to 0 even if the layer is not steppable.
    return (
        self.config.max_future_horizon
        if self.config.max_future_horizon >= 0
        else 0
    )

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    max_past_horizon = self.config.max_past_horizon
    max_future_horizon = self.config.max_future_horizon
    start = -np.inf if max_past_horizon == -1 else -max_past_horizon
    end = np.inf if max_future_horizon == -1 else max_future_horizon
    return {0: (start, end)}

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

    num_kv_heads = self.config.num_kv_heads or self.config.num_heads
    kv_zero_values = jnp.zeros(
        (
            batch_size,
            kv_buffer_size,
            num_kv_heads,
            self.config.units_per_head,
        ),
        dtype=compute_dtype,
    )
    kv_zero_mask = jnp.zeros(
        [batch_size, kv_buffer_size], dtype=types.MASK_DTYPE
    )
    kv_zero_ids = jnp.zeros(
        [batch_size, kv_buffer_size],
        dtype=jnp.int32,
    )

    kv_bundle = common.KVBundle(
        keys=kv_zero_values,
        values=kv_zero_values,
        segment_ids=kv_zero_ids if self.config.segment_ids_name else None,
        position=kv_zero_ids,
        mask=kv_zero_mask,
    )

    # If we have a finite future horizon, we cannot produce outputs for timestep
    # t until the max_future_horizon KV timesteps have arrived. Store incoming
    # queries in a delay buffer so we do not compute context vectors for them
    # until max_future_horizon KV timesteps have arrived.
    if max_future_horizon:
      query_delay_buffer = common.QBundle(
          queries=jnp.zeros(
              (
                  batch_size,
                  max_future_horizon,
                  self.config.num_heads,
                  self.config.units_per_head,
              ),
              dtype=compute_dtype,
          ),
          segment_ids=jnp.zeros(
              [batch_size, max_future_horizon], dtype=jnp.int32
          )
          if self.config.segment_ids_name
          else None,
          position=jnp.zeros([batch_size, max_future_horizon], dtype=jnp.int32),
          mask=jnp.zeros(
              [batch_size, max_future_horizon], dtype=types.MASK_DTYPE
          ),
      )
    else:
      query_delay_buffer = ()

    time_step = jnp.zeros((batch_size,), dtype=jnp.int32)

    query_input_spec = types.ShapeDType(
        (self.config.num_heads, self.config.units_per_head), compute_dtype
    )
    num_kv_heads = self.config.num_kv_heads or self.config.num_heads
    key_value_input_spec = types.ShapeDType(
        (num_kv_heads, self.config.units_per_head), compute_dtype
    )

    if self.query_network:
      query_network_state = self.query_network.get_initial_state(
          batch_size,
          query_input_spec,
          training=training,
          constants=constants,
      )
    else:
      query_network_state = ()
    if self.key_network:
      key_network_state = self.key_network.get_initial_state(
          batch_size,
          key_value_input_spec,
          training=training,
          constants=constants,
      )
    else:
      key_network_state = ()
    if self.value_network:
      value_network_state = self.value_network.get_initial_state(
          batch_size,
          key_value_input_spec,
          training=training,
          constants=constants,
      )
    else:
      value_network_state = ()

    return (
        kv_bundle,
        time_step,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
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

    x_values_time = x.shape[1]

    x_queries, x_keys, x_values = self.get_qkv(self.config.input_projection, x)

    # Our params might not match param_dtype, so delegate the compute_dtype to
    # the output of the QKV layer.
    compute_dtype = x_queries.dtype

    (
        kv_bundle,
        time_step,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    ) = state

    constants = constants or {}

    if self.config.position_name and self.config.position_name in constants:
      x_position = utils.get_constant_array(
          self, constants, self.config.position_name
      )
    else:
      x_position = (
          jnp.arange(x_values_time)[jnp.newaxis, :] + time_step[:, jnp.newaxis]
      )

    if self.config.segment_ids_name:
      x_segment_ids = utils.get_constant_array(
          self, constants, self.config.segment_ids_name
      )
    else:
      x_segment_ids = None

    kv_buffer_size = kv_bundle.keys.shape[1]

    if self.query_network:
      x_queries, query_network_state = self.query_network.step(
          x_queries,
          query_network_state,
          training=training,
          constants=constants,
      )
    if self.key_network:
      x_keys, key_network_state = self.key_network.step(
          x_keys,
          key_network_state,
          training=training,
          constants=constants,
      )
    if self.value_network:
      x_values, value_network_state = self.value_network.step(
          x_values,
          value_network_state,
          training=training,
      )

    # Guard against NaN/Inf in values, since values are contracted when
    # computing context vectors.
    x_values = x_values.mask_invalid()

    # The key and value network could have changed the mask, so we combine the
    # keys and values mask. This is inexpensive but would be nice to skip.
    combined_mask = utils.combine_mask(x_keys.mask, x_values.mask)

    new_query_bundle = common.QBundle(
        queries=x_queries.values,
        segment_ids=x_segment_ids,
        position=x_position,
        mask=x_queries.mask,
    )
    new_kv_bundle = common.KVBundle(
        keys=x_keys.values,
        values=x_values.values,
        segment_ids=x_segment_ids,
        position=x_position,
        # The mask could differ based on key_network / value_network.
        mask=combined_mask,
    )

    if self.config.use_kv_cache_ringbuffer:
      if x_queries.shape[1] != 1:
        raise ValueError(
            'use_kv_cache_ringbuffer requires a step size of 1. Got:'
            f' {x_queries.shape=}'
        )
    else:
      # To process a step, concatenate our kv_buffer_size KV buffer with the
      # input source.shape[1] timesteps.
      kv_bundle = jax.tree.map(
          lambda a, b: jnp.concatenate([a, b], axis=1), kv_bundle, new_kv_bundle
      )

    # If we have a query delay buffer, we need to insert the current block of
    # queries into it, and pop the oldest x_values_time queries off. If no query
    # cache, the below logic is a no-op so we save the compute of running it.
    if query_delay_buffer:
      assert self.config.max_future_horizon > 0

      query_delay_buffer = jax.tree.map(
          lambda a, b: jnp.concatenate([a, b], axis=1),
          query_delay_buffer,
          new_query_bundle,
      )
      assert (
          query_delay_buffer.queries.shape[1]
          == x_values_time + self.config.max_future_horizon
      )

      # Use the oldest x_values_time queries as the current step's queries. They
      # each have max_future_horizon context available in
      # kv_buffer_keys/kv_buffer_values so we can produce valid output for them.
      new_query_bundle = jax.tree.map(
          lambda a: a[:, :x_values_time], query_delay_buffer
      )

      # Preserve the last max_future_horizon queries for the next step.
      query_delay_buffer = jax.tree.map(
          lambda a: a[:, -self.config.max_future_horizon :], query_delay_buffer
      )

    # Compute context vectors:
    if self.config.use_kv_cache_ringbuffer:
      kv_buffers = (kv_bundle, new_kv_bundle)
      key_block_sizes = (
          self.config.flash_attention_key_block_size,
          x_keys.shape[1],
      )
    else:
      kv_buffers = (kv_bundle,)
      key_block_sizes = self.config.flash_attention_key_block_size

    attention_mask_fns = []
    if (
        self.config.max_past_horizon != -1
        or self.config.max_future_horizon != -1
    ):
      attention_mask_fns.append(
          common.LocalCausalMask(
              max_past_horizon=None
              if self.config.max_past_horizon == -1
              else self.config.max_past_horizon,
              max_future_horizon=None
              if self.config.max_future_horizon == -1
              else self.config.max_future_horizon,
          )
      )
    if self.config.segment_ids_name:
      attention_mask_fns.append(common.SegmentMask())

    def fn(query_bundle, kv_bundles):
      return common.multi_key_value_dot_product_flash_attention(
          queries=query_bundle,
          query_block_size=self.config.flash_attention_query_block_size,
          kv_bundles=kv_bundles,
          kv_block_sizes=key_block_sizes,
          attention_mask_fns=attention_mask_fns,
          attention_logits_soft_cap=self.config.attention_logits_soft_cap,
          per_dim_scale=self._per_dim_scale,
          query_scale=self.config.query_scale,
          precision=self.config.precision,
          compute_dtype=compute_dtype,
          remat=self.config.experimental_remat_inner,
      )

    if self.config.experimental_remat_outer:
      fn = jax.checkpoint(fn, prevent_cse=False)

    context_vectors = fn(
        dataclasses.replace(new_query_bundle, mask=None), kv_buffers
    )

    # Update KV caches and state.
    if self.config.use_kv_cache_ringbuffer:
      # Write latest keys, values and masks to the appropriate position in the
      # KV cache ring buffers.
      i = time_step % kv_buffer_size
      assert combined_mask.shape[1] == x_keys.shape[1]
      assert x_values.shape[1] == x_keys.shape[1]

      update_fn = jax.vmap(lambda x, u, i: x.at[i].set(u.squeeze(0)))

      kv_bundle = jax.tree.map(
          lambda a, b: update_fn(a, b, i), kv_bundle, new_kv_bundle
      )
    else:
      # Preserve last kv_buffer_size timesteps as state for next step.
      kv_bundle = jax.tree.map(lambda a: a[:, -kv_buffer_size:], kv_bundle)

    state = (
        kv_bundle,
        time_step + x_values_time,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    )

    emits = ()

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, new_query_bundle.mask)

    return context_vectors, state, emits

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    values_time = x.shape[1]

    queries, keys, values = self.get_qkv(self.config.input_projection, x)

    if self.config.position_name:
      x_position = utils.get_constant_array(
          self, constants, self.config.position_name
      )
    else:
      x_position = jnp.arange(values_time)[jnp.newaxis, :]

    if self.config.segment_ids_name:
      x_segment_ids = utils.get_constant_array(
          self,
          constants,
          self.config.segment_ids_name,
      )
    else:
      x_segment_ids = None

    # Our params might not match param_dtype, so delegate the compute_dtype to
    # the output of the QKV layer.
    compute_dtype = queries.dtype

    if self.query_network:
      queries = self.query_network.layer(
          queries,
          training=training,
          constants=constants,
      )
    if self.key_network:
      keys = self.key_network.layer(
          keys,
          training=training,
          constants=constants,
      )
    if self.value_network:
      values = self.value_network.layer(
          values, training=training, constants=constants
      )

    # The key and value network could have changed the mask, so we combine the
    # keys and values mask. This is inexpensive but would be nice to skip.
    combined_mask = utils.combine_mask(keys.mask, values.mask)

    attention_mask_fns = []
    if (
        self.config.max_past_horizon != -1
        or self.config.max_future_horizon != -1
    ):
      attention_mask_fns.append(
          common.LocalCausalMask(
              max_past_horizon=None
              if self.config.max_past_horizon == -1
              else self.config.max_past_horizon,
              max_future_horizon=None
              if self.config.max_future_horizon == -1
              else self.config.max_future_horizon,
          )
      )
    if self.config.segment_ids_name:
      attention_mask_fns.append(common.SegmentMask())

    def fn(query_bundle, kv_bundles):
      return common.multi_key_value_dot_product_flash_attention(
          queries=query_bundle,
          query_block_size=self.config.flash_attention_query_block_size,
          kv_bundles=kv_bundles,
          kv_block_sizes=self.config.flash_attention_key_block_size,
          attention_mask_fns=attention_mask_fns,
          attention_logits_soft_cap=self.config.attention_logits_soft_cap,
          per_dim_scale=self._per_dim_scale,
          query_scale=self.config.query_scale,
          precision=self.config.precision,
          compute_dtype=compute_dtype,
          remat=self.config.experimental_remat_inner,
      )

    if self.config.experimental_remat_outer:
      fn = jax.checkpoint(fn)

    context_vectors = fn(
        common.QBundle(
            queries.values,
            segment_ids=x_segment_ids,
            position=x_position,
            mask=None,
        ),
        (
            common.KVBundle(
                keys=keys.values,
                values=values.values,
                segment_ids=x_segment_ids,
                position=x_position,
                mask=combined_mask,
            ),
        ),
    )

    emits = ()

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, x.mask)

    return context_vectors, emits
