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
"""Multi-source dot product attention layer."""

from collections.abc import Sequence as TypingSequence
import dataclasses
from flax import linen as nn
import jax.numpy as jnp
import jaxtyping
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class MultiSourceDotProductAttention(
    types.Emitting, common.AttentionInputProjectionHelper
):
  """Multi-source dot product attention."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for MultiSourceDotProductAttention."""

    # The name(s) of the constant(s) containing the source sequences to attend
    # to.
    source_names: str | TypingSequence[str]

    # The number of attention heads.
    num_heads: int

    # The number of units per head.
    units_per_head: int

    # The number of key/value heads. If None, defaults to num_heads.
    num_kv_heads: int | None = None

    # Flash attention block sizes. If non-None, uses flash attention to split
    # the attention computation into smaller blocks. This reduces peak memory
    # usage and may lead to speedups. On TPU, block sizes less than 128 are
    # typically not worth it.
    flash_attention_query_block_size: int | None = None
    flash_attention_source_block_sizes: int | TypingSequence[int] | None = None

    # The name of the constant containing the query positions. If unset,
    # positions are created from the time step.
    query_positions_name: str | None = None
    # The name of the constant containing the query segment ids. If unset,
    # no segment IDs are used for attention masking.
    query_segment_ids_name: str | None = None
    # The name(s) of the constant(s) containing the source positions. If unset,
    # positions are created from the time step.
    source_position_names: str | TypingSequence[str] | None = None
    # The name(s) of the constant(s) containing the source segment ids. If
    # unset, no segment IDs are used for attention masking.
    source_segment_ids_names: str | TypingSequence[str] | None = None

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
    # The dtype of the layer's computations.
    compute_dtype: types.DType | None = None
    # The dtype of the layer's parameters.
    param_dtype: types.DType = jnp.float32
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'MultiSourceDotProductAttention':
      return MultiSourceDotProductAttention(self, name=self.name)

  config: Config

  @property
  def _source_names(self) -> TypingSequence[str]:
    if isinstance(self.config.source_names, str):
      return (self.config.source_names,)
    else:
      return self.config.source_names

  def setup(self) -> None:
    for source_name in self._source_names:
      if not source_name:
        raise ValueError('Source name cannot be empty.')
    common.validate_attention(
        'unused',
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
        # Not possible for cross attention.
        allow_combined_qkv=False,
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
      # Use the position constant as query position.
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

  def _get_kv_bundles(
      self, training: bool, constants: types.Constants | None
  ) -> list[common.KVBundle]:
    """Returns key/value bundles for all sources."""
    kv_bundles = []

    source_position_names = self.config.source_position_names
    if source_position_names is None:
      source_position_names = [None] * len(self._source_names)
    elif isinstance(source_position_names, str):
      source_position_names = [source_position_names] * len(self._source_names)
    source_segment_ids_names = self.config.source_segment_ids_names
    if source_segment_ids_names is None:
      source_segment_ids_names = [None] * len(self._source_names)
    elif isinstance(source_segment_ids_names, str):
      source_segment_ids_names = [source_segment_ids_names] * len(
          self._source_names
      )
    if len(source_position_names) != len(self._source_names):
      raise ValueError(
          'source_position_names must be the same length as source_names.'
      )
    if len(source_segment_ids_names) != len(self._source_names):
      raise ValueError(
          'source_segment_ids_names must be the same length as source_names.'
      )

    source_channel_shapes = {
        common.get_source(self, source_name, constants).channel_shape
        for source_name in self._source_names
    }

    if len(source_channel_shapes) != 1:
      raise ValueError(
          'All sources must have the same channel shape, got:'
          f' {source_channel_shapes=} for {self._source_names=}'
      )

    for source_name, position_name, segment_ids_name in zip(
        self._source_names,
        source_position_names,
        source_segment_ids_names,
        strict=True,
    ):
      # Pre-process sources with key/value projections and networks.
      source = common.get_source(self, source_name, constants)
      keys, values = self.get_kv(self.config.input_projection, source)
      if self.key_network:
        keys = self.key_network.layer(
            keys, training=training, constants=constants
        )

      if self.value_network:
        values = self.value_network.layer(
            values, training=training, constants=constants
        )

      if position_name:
        positions = common.get_source(
            self, position_name, constants, required_rank=2
        )
      else:
        positions = jnp.arange(keys.shape[1])[jnp.newaxis, :]

      if segment_ids_name:
        segment_ids = common.get_source(
            self, segment_ids_name, constants, required_rank=2
        )
      else:
        segment_ids = None

      # Mask before storing in state:
      keys = keys.mask_invalid()
      values = values.mask_invalid()
      combined_mask = utils.combine_mask(keys.mask, values.mask)

      kv_bundles.append(
          common.KVBundle(
              keys.values, values.values, segment_ids, positions, combined_mask
          )
      )
    return kv_bundles

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    kv_bundles = self._get_kv_bundles(training, constants)

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
    return (kv_bundles, query_state, time_step)

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
    kv_bundles, query_state, time_step = state

    x_num_timesteps = x.shape[1]

    # No mask required, since query timesteps are independent.
    queries = self.get_q(self.config.input_projection, x)

    if self.query_network:
      queries, query_state = self.query_network.step(
          queries, query_state, training=training, constants=constants
      )

    y, emits = self._attention(
        queries,
        kv_bundles,
        constants=constants,
        time_step=time_step,
    )
    time_step += x_num_timesteps
    state = (kv_bundles, query_state, time_step)
    return y, state, emits

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    kv_bundles = self._get_kv_bundles(training, constants)

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
        kv_bundles,
        constants=constants,
    )

  @nn.nowrap
  def _attention(
      self,
      query: types.Sequence,
      kv_bundles: list[common.KVBundle],
      constants: types.Constants | None = None,
      time_step: jaxtyping.Num[jt.ArrayT, 'b k'] | int = 0,
  ) -> tuple[types.Sequence, types.Emits]:
    compute_dtype = utils.get_promoted_dtype(
        query.values.dtype,
        *[k.keys.dtype for k in kv_bundles],
        dtype=self.config.compute_dtype,
    )

    query_positions = self._get_query_positions(
        query.shape[0], query.shape[1], constants, time_step
    )

    if self.config.query_segment_ids_name:
      query_segment_ids = common.get_source(
          self,
          self.config.query_segment_ids_name,
          constants,
          required_rank=2,
      )
    else:
      query_segment_ids = None

    query_bundle = common.QBundle(
        queries=query.values,
        segment_ids=query_segment_ids,
        position=query_positions,
        mask=None,
    )

    attention_mask_fns = []
    if self.config.query_segment_ids_name:
      attention_mask_fns.append(common.SegmentMask())

    context_vectors = common.multi_key_value_dot_product_flash_attention(
        queries=query_bundle,
        query_block_size=self.config.flash_attention_query_block_size,
        kv_bundles=kv_bundles,
        kv_block_sizes=self.config.flash_attention_source_block_sizes,
        attention_mask_fns=attention_mask_fns,
        attention_logits_soft_cap=self.config.attention_logits_soft_cap,
        precision=self.config.precision,
        per_dim_scale=self._per_dim_scale,
        query_scale=self.config.query_scale,
        compute_dtype=compute_dtype,
    )
    emits = ()
    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, query.mask)
    return context_vectors, emits
