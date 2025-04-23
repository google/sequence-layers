# Copyright 2024 Google LLC
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
"""Attention layers."""

import dataclasses
import functools
import math
from typing import Any, Callable, Mapping, Protocol, Sequence as TypingSequence

from flax import linen as nn
from flax import struct
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import jaxtyping
import numpy as np
from sequence_layers.jax import signal
from sequence_layers.jax import simple
from sequence_layers.jax import types
from sequence_layers.jax import utils

from google3.learning.deepmind.jax.typing import typing as jt
from google3.learning.gemini.cms.core.models import labels
from google3.learning.gemini.gemax.core.models import meta


__all__ = (
    # go/keep-sorted start
    'CombinedQueryKeyValueProjection',
    'CrossAttentionEmits',
    'DotProductAttention',
    'DotProductSelfAttention',
    'GmmAttention',
    'LocalDotProductSelfAttention',
    'QueryAndKeyValueProjection',
    'QueryAndSharedKeyValueProjection',
    'RelativePositionEmbedding',
    'SelfAttentionEmits',
    'SeparateQueryKeyValueProjection',
    'ShawRelativePositionEmbedding',
    'StreamingLocalDotProductAttention',
    'T5RelativePositionEmbedding',
    'TransformerXLRelativePositionEmbedding',
    # go/keep-sorted end
)


# A negative enough value such that it underflows to a hard zero in softmax.
_INVALID_LOGIT_VALUE = -1e9

# Input projection from [dimension] to [num_heads, units_per_head].
# Kernel matrix is [dimension, num_heads, units_per_head].
input_projection_default_kernel_init = nn.initializers.lecun_normal(
    in_axis=-3, out_axis=[-2, -1]
)

# Input projection from [dimension] to [num_stacked, num_heads, units_per_head].
# Kernel matrix is [dimension, num_stacked, num_heads, units_per_head].
stacked_input_projection_default_kernel_init = nn.initializers.lecun_normal(
    in_axis=-4, out_axis=[-2, -1], batch_axis=[-3]
)


def _soft_cap_attention_logits(logits: jax.Array, cap: float) -> jax.Array:
  cap = jnp.asarray(cap, logits.dtype)
  return cap * jax.nn.tanh(logits / cap)


def _mask_attention_logits(
    logits: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
  utils.assert_has_rank(logits.shape, valid_mask.ndim)
  # Check broadcastable.
  jnp.broadcast_shapes(logits.shape, valid_mask.shape)
  assert valid_mask.dtype == jnp.bool_
  # Mask invalid timesteps, potentially broadcasting.
  # Adding can change the softmax output so replace values with jnp.where.
  return jnp.where(
      valid_mask,
      logits,
      jnp.asarray(_INVALID_LOGIT_VALUE, dtype=logits.dtype),
  )


@jt.typed
def _zero_fully_masked_attention_probabilities(
    probabilities: jt.Float[jt.ArrayT, '*B K'],
    logit_visibility_mask: jt.Bool[jt.ArrayT, '#*B K'],
) -> jt.Float[jt.ArrayT, '*B K']:
  not_fully_masked = jnp.any(logit_visibility_mask, keepdims=True, axis=-1)
  return jnp.where(
      not_fully_masked,
      probabilities,
      jnp.zeros_like(probabilities),
  )


def _sharded_softmax(
    scores: TypingSequence[jax.Array], axis: int, output_dtype: types.DType
) -> TypingSequence[jax.Array]:
  """Compute a softmax across the axis of all arrays in scores."""
  if not scores:
    return scores
  elif len(scores) == 1:
    return [jax.nn.softmax(scores[0], axis=axis).astype(output_dtype)]

  global_maximum = functools.reduce(
      jnp.maximum, [jnp.max(s, axis=axis, keepdims=True) for s in scores]
  )

  # Subtract global maximum for numerical stability.
  numerators = [jnp.exp(s - global_maximum) for s in scores]
  denominators = [
      jnp.sum(numerator, axis=axis, keepdims=True) for numerator in numerators
  ]
  global_denominator = functools.reduce(jnp.add, denominators)

  return [
      (numerator / global_denominator).astype(output_dtype)
      for numerator in numerators
  ]


def _validate_heads(num_heads: int, units_per_head: int, name: str):
  if num_heads <= 0:
    raise ValueError(f'Expected num_heads > 0 for {name}. Got {num_heads}')
  if units_per_head <= 0:
    raise ValueError(
        f'Expected units_per_head > 0 for {name}. Got {units_per_head}'
    )


def _validate_attention(
    source_name: str, num_heads: int, units_per_head: int, name: str
):
  if not source_name:
    raise ValueError(f'Expected non-empty source_name for {name}.')
  _validate_heads(num_heads, units_per_head, name)


def _get_source(
    layer: types.SequenceLayer,
    source_name: str,
    constants: types.Constants,
    required_rank: int = 3,
) -> types.Sequence:
  """Gets the attention source from constants and does basic validation."""
  if constants is None:
    raise ValueError(
        f'{layer} requires the source to be provided via '
        f'constants, got: {constants}'
    )
  source = constants.get(source_name)
  if source is None:
    raise ValueError(
        f'{layer} expected {source_name} to be present in '
        f'constants, got: {constants}'
    )
  if not isinstance(source, types.Sequence):
    raise ValueError(
        f'{layer} expected a Sequence for {source_name}, got: {source}'
    )
  if len(source.values.shape) != required_rank:
    raise ValueError(
        f'{layer} requires a rank {required_rank} source, got: {source}'
    )
  return source


@dataclasses.dataclass(frozen=True)
class QueryKeyValueProjectionConfig:
  # Optional callable that returns a jnp.einsum-compatible function to use
  # instead of jnp.einsum for the query, key and value projections.
  # For example, to enable quantization aware training.
  einsum_factory: types.EinsumFactoryT | None = None


@dataclasses.dataclass(frozen=True)
class CombinedQueryKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use a single projection matrix for query/key/value projection.

  * Incompatible with Grouped Query Attention (num_query_heads != num_kv_heads).
  * Supports shared key and value projection.
  """

  # Kernel initializer and sharding for the combined query/key/value projection.
  # The variable shape is [input_dimension, 3, num_heads, units_per_head].
  # If share_kv_projection is True, the variable shape is [input_dimension, 2,
  # num_heads, units_per_head].
  qkv_kernel_init: nn.initializers.Initializer = (
      stacked_input_projection_default_kernel_init
  )
  qkv_kernel_sharding: types.Sharding | None = None

  # Bias initializer and sharding for the combined query/key/value projection.
  # The variable shape is [3, num_heads, units_per_head].
  bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  bias_sharding: types.Sharding | None = None

  # If true, share the key and value projection matrices.
  share_kv_projection: bool = False


@dataclasses.dataclass(frozen=True)
class SeparateQueryKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate projection matrices for query/key/value projection.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Does not support shared key and value projection. Use
    QueryAndSharedKeyValueProjection.
  """

  # Kernel initializers and sharding for the separate query/key/value
  # projections.
  # The variable shape is [input_dimension, num_heads or num_kv_heads, u
  # nits_per_head].
  q_kernel_init: nn.initializers.Initializer = (
      input_projection_default_kernel_init
  )
  k_kernel_init: nn.initializers.Initializer = (
      input_projection_default_kernel_init
  )
  v_kernel_init: nn.initializers.Initializer = (
      input_projection_default_kernel_init
  )
  q_kernel_sharding: types.Sharding | None = None
  k_kernel_sharding: types.Sharding | None = None
  v_kernel_sharding: types.Sharding | None = None

  # Bias initializer and sharding for the separate query/key/value projections.
  # The variable shape is [num_heads or num_kv_heads, units_per_head].
  bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  bias_sharding: types.Sharding | None = None


@dataclasses.dataclass(frozen=True)
class QueryAndKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate query and key/value projection matrices.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Does not support shared key and value projection. Use
    QueryAndSharedKeyValueProjection.
  """

  # Kernel initializer and sharding for the query projection.
  # The variable shape is [input_dimension, num_heads, units_per_head].
  q_kernel_init: nn.initializers.Initializer = (
      input_projection_default_kernel_init
  )
  q_kernel_sharding: types.Sharding | None = None

  # Bias initializer and sharding for the query projection.
  # The variable shape is [num_heads, units_per_head].
  q_bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  q_bias_sharding: types.Sharding | None = None

  # Kernel initializer and sharding for the key/value projection.
  # The variable shape is [input_dimension, 2, num_kv_heads, units_per_head].
  kv_kernel_init: nn.initializers.Initializer = (
      input_projection_default_kernel_init
  )
  kv_kernel_sharding: types.Sharding | None = None

  # Bias initializer and sharding for the key/value projection.
  # The variable shape is [2, num_kv_heads, units_per_head].
  kv_bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  kv_bias_sharding: types.Sharding | None = None


@dataclasses.dataclass(frozen=True)
class QueryAndSharedKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate query and shared key/value projection matrices.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Requires shared key and value projection.
  """

  # Kernel initializer and sharding for the query projection.
  # The variable shape is [input_dimension, num_heads, units_per_head].
  q_kernel_init: nn.initializers.Initializer = (
      input_projection_default_kernel_init
  )
  q_kernel_sharding: types.Sharding | None = None
  # Bias initializer and sharding for the query projection.
  # The variable shape is [num_heads, units_per_head].
  q_bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  q_bias_sharding: types.Sharding | None = None

  # Kernel initializer and sharding for the shared key/value projection.
  # The variable shape is [input_dimension, num_kv_heads, units_per_head].
  kv_kernel_init: nn.initializers.Initializer = (
      input_projection_default_kernel_init
  )
  kv_kernel_sharding: types.Sharding | None = None

  # Bias initializer and sharding for the shared key/value projection.
  # The variable shape is [num_kv_heads, units_per_head].
  kv_bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  kv_bias_sharding: types.Sharding | None = None


class AttentionInputProjectionHelper:
  """Helper class for shared attention input projection logic."""

  def _setup_projection_layers(
      self,
      config: QueryKeyValueProjectionConfig,
      num_query_heads: int,
      num_kv_heads: int,
      units_per_head: int,
      use_bias: bool,
      precision: jax.lax.PrecisionLike,
      compute_dtype: types.DType,
      param_dtype: types.DType,
      allow_combined_qkv: bool = True,
  ) -> None:
    """Creates submodules, must be called from nn.Module.setup in subclasses."""
    match config:
      case CombinedQueryKeyValueProjection():
        if not allow_combined_qkv:
          raise ValueError(
              'CombinedQueryKeyValueProjection is not supported. Use'
              ' SeparateQueryKeyValueProjection or'
              ' QueryAndSharedKeyValueProjection.'
          )
        if num_query_heads != num_kv_heads:
          raise ValueError(
              f'num_query_heads={num_query_heads} !='
              f' num_kv_heads={num_kv_heads}'
          )
        num_stacked = 2 if config.share_kv_projection else 3
        self._qkv = utils.FlaxEinsumDense(
            equation='...a,abcd->...bcd',
            output_shape=(num_stacked, num_query_heads, units_per_head),
            bias_axes='bcd' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.qkv_kernel_init,
                config.qkv_kernel_sharding,
                projectable=True,
                axes_types=(
                    meta.AxisType.FANIN,
                    meta.AxisType.STACKED,
                    None,
                    None,
                ),
            ),
            bias_init=utils.shard_initializer(
                config.bias_init, config.bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='query_key_value_projection',
        )
      case SeparateQueryKeyValueProjection():
        self._q = utils.FlaxEinsumDense(
            equation='...a,abc->...bc',
            output_shape=(num_query_heads, units_per_head),
            bias_axes='bc' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.q_kernel_init,
                config.q_kernel_sharding,
                projectable=True,
                axes_types=(meta.AxisType.FANIN, None, None),
            ),
            bias_init=utils.shard_initializer(
                config.bias_init, config.bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='query_projection',
        )
        self._k = utils.FlaxEinsumDense(
            equation='...a,abc->...bc',
            output_shape=(num_kv_heads, units_per_head),
            bias_axes='bc' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.k_kernel_init,
                config.k_kernel_sharding,
                projectable=True,
                axes_types=(meta.AxisType.FANIN, None, None),
            ),
            bias_init=utils.shard_initializer(
                config.bias_init, config.bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='key_projection',
        )
        self._v = utils.FlaxEinsumDense(
            equation='...a,abc->...bc',
            output_shape=(num_kv_heads, units_per_head),
            bias_axes='bc' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.v_kernel_init,
                config.v_kernel_sharding,
                projectable=True,
                axes_types=(meta.AxisType.FANIN, None, None),
            ),
            bias_init=utils.shard_initializer(
                config.bias_init, config.bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='value_projection',
        )
      case QueryAndKeyValueProjection():
        self._q = utils.FlaxEinsumDense(
            equation='...a,abc->...bc',
            output_shape=(num_query_heads, units_per_head),
            bias_axes='bc' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.q_kernel_init,
                config.q_kernel_sharding,
                projectable=True,
                axes_types=(meta.AxisType.FANIN, None, None),
            ),
            bias_init=utils.shard_initializer(
                config.q_bias_init, config.q_bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='query_projection',
        )
        self._kv = utils.FlaxEinsumDense(
            equation='...a,abcd->...bcd',
            output_shape=(2, num_kv_heads, units_per_head),
            bias_axes='bcd' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.kv_kernel_init,
                config.kv_kernel_sharding,
                projectable=True,
                axes_types=(
                    meta.AxisType.FANIN,
                    meta.AxisType.STACKED,
                    None,
                    None,
                ),
            ),
            bias_init=utils.shard_initializer(
                config.kv_bias_init, config.kv_bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='key_value_projection',
        )
      case QueryAndSharedKeyValueProjection():
        self._q = utils.FlaxEinsumDense(
            equation='...a,abc->...bc',
            output_shape=(num_query_heads, units_per_head),
            bias_axes='bc' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.q_kernel_init,
                config.q_kernel_sharding,
                projectable=True,
                axes_types=(meta.AxisType.FANIN, None, None),
            ),
            bias_init=utils.shard_initializer(
                config.q_bias_init, config.q_bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='query_projection',
        )
        self._shared_kv = utils.FlaxEinsumDense(
            equation='...a,abc->...bc',
            output_shape=(num_kv_heads, units_per_head),
            bias_axes='bc' if use_bias else None,
            kernel_init=utils.shard_initializer(
                config.kv_kernel_init,
                config.kv_kernel_sharding,
                projectable=True,
                axes_types=(
                    meta.AxisType.FANIN,
                    None,
                    None,
                ),
            ),
            bias_init=utils.shard_initializer(
                config.kv_bias_init, config.kv_bias_sharding
            ),
            precision=precision,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            einsum_factory=config.einsum_factory,
            name='shared_key_value_projection',
        )

  def get_input_projection_output_dtype(
      self, config: QueryKeyValueProjectionConfig, input_dtype: types.DType
  ) -> types.DType:
    """Returns the output dtype of the QKV projection."""
    match config:
      case CombinedQueryKeyValueProjection():
        return self._qkv.get_output_dtype(input_dtype)
      case (
          SeparateQueryKeyValueProjection()
          | QueryAndKeyValueProjection()
          | QueryAndSharedKeyValueProjection()
      ):
        return self._q.get_output_dtype(input_dtype)
      case _:
        raise NotImplementedError(config)

  def get_qkv(
      self, config: QueryKeyValueProjectionConfig, x: types.Sequence
  ) -> tuple[types.Sequence, types.Sequence, types.Sequence]:
    """Project input to query/key/value sequences."""
    match config:
      case CombinedQueryKeyValueProjection():
        projection = utils.sequence_unstack(
            self._qkv.project_sequence(x), axis=2
        )

        if len(projection) == 2:
          # Shared K and V.
          queries, keys = projection
          values = keys
        else:
          queries, keys, values = projection
      case SeparateQueryKeyValueProjection():
        queries = self._q.project_sequence(x)
        keys = self._k.project_sequence(x)
        values = self._v.project_sequence(x)
      case QueryAndKeyValueProjection():
        queries = self._q.project_sequence(x)
        keys, values = utils.sequence_unstack(
            self._kv.project_sequence(x), axis=2
        )
      case QueryAndSharedKeyValueProjection():
        queries = self._q.project_sequence(x)
        keys = values = self._shared_kv.project_sequence(x)
      case _:
        raise NotImplementedError(config)
    return queries, keys, values

  def get_q(
      self, config: QueryKeyValueProjectionConfig, x: types.Sequence
  ) -> types.Sequence:
    """Project input to query sequence."""
    match config:
      case SeparateQueryKeyValueProjection():
        queries = self._q.project_sequence(x)
      case QueryAndKeyValueProjection():
        queries = self._q.project_sequence(x)
      case QueryAndSharedKeyValueProjection():
        queries = self._q.project_sequence(x)
      case _:
        raise NotImplementedError(config)
    return queries

  def get_kv(
      self, config: QueryKeyValueProjectionConfig, x: types.Sequence
  ) -> tuple[types.Sequence, types.Sequence]:
    """Project input to key/value sequences."""
    match config:
      case SeparateQueryKeyValueProjection():
        keys = self._k.project_sequence(x)
        values = self._v.project_sequence(x)
      case QueryAndKeyValueProjection():
        keys, values = utils.sequence_unstack(
            self._kv.project_sequence(x), axis=2
        )
      case QueryAndSharedKeyValueProjection():
        keys = values = self._shared_kv.project_sequence(x)
      case _:
        raise NotImplementedError(config)
    return keys, values


class SelfAttentionEmits(struct.PyTreeNode):
  """A structure for emits produced by self attention layers."""

  # The attention probabilities, generally shaped
  # [batch_size, query_time, num_heads, source_time].
  probabilities: types.Sequence | types.ShapeDType


class CrossAttentionEmits(struct.PyTreeNode):
  """A structure for emits produced by attention layers."""

  # The attention probabilities, generally shaped
  # [batch_size, query_time, num_heads, source_time].
  probabilities_by_source: Mapping[str, types.Sequence | types.ShapeDType]


class RelativePositionEmbedding(nn.Module):
  """Abstract base class for computing relative position biases."""

  @dataclasses.dataclass(frozen=True)
  class Config(Protocol):

    def make(self) -> 'RelativePositionEmbedding':
      ...

  @property
  def supports_position_bias(self):
    """Returns whether relative position bias is supported."""
    return False

  @property
  def supports_get_logits(self):
    """Returns whether get_logits is supported."""
    return False

  def get_position_bias(
      self,
      query_positions: jaxtyping.Num[jt.ArrayT, '#b q'],
      key_positions: jaxtyping.Num[jt.ArrayT, '#b k'],
      queries: jt.Float[jt.ArrayT, 'b q nq d'],
      keys: jt.Float[jt.ArrayT, 'b k nk d'] | None,
  ) -> jt.Float[jt.ArrayT, '#b h q k']:
    """Get attention logit biases from position indices.

    Args:
      query_positions: [batch, query_length] int/float Tensor containing query
        position indices.
      key_positions: [batch, key_length] int/float Tensor containing position
        indices.
      queries: Queries of shape [batch, query_length, num_heads,
        units_per_head]. In some cases, is only used to determine the dtype of
        the output.
      keys: Keys of shape [batch, key_length, num_heads, units_per_head], can be
        None for specific relative position embeddings.

    Returns:
      position_biases: [batch, num_heads, query_length, key_length] Tensor
        containing head-wise biases for each query-key pair.
    """
    raise NotImplementedError('get_position_bias is not implemented.')

  def get_logits_streaming(
      self,
      queries: jax.Array,
      keys: jax.Array,
      precision: nn.linear.PrecisionLike,
  ) -> jax.Array:
    """Computes attention logits for streaming queries and keys.

    Args:
      queries: The queries to compute logits for.
      keys: The keys to compute logits for.
      precision: Precision config for einsums.

    Returns:
      logits: Logits for the queries and keys.
    """
    raise NotImplementedError('get_logits_streaming is not implemented.')

  def get_logits(
      self,
      queries: jax.Array,
      keys: jax.Array,
      precision: nn.linear.PrecisionLike,
  ) -> jax.Array:
    """Computes attention logits for queries and keys.

    Args:
      queries: The queries to compute logits for.
      keys: The keys to compute logits for.
      precision: Precision config for einsums.

    Returns:
      logits: Logits for the queries and keys.
    """
    raise NotImplementedError('get_logits is not implemented.')


class ShawRelativePositionEmbedding(RelativePositionEmbedding):
  """Computes query-dependent relative position embeddings.

  Based on:
  Self-Attention with Relative Position Representations
  https://arxiv.org/abs/1803.02155

  Computes a [batch, num_heads, queries_time, keys_time] tensor of relative
  position biases, biasing the selection of keys for every query timestep.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(RelativePositionEmbedding.Config):
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
        labels=[labels.IS_EMBEDDING],
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


class T5RelativePositionEmbedding(RelativePositionEmbedding):
  """Relative position embeddings in the T5 style.

  Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer
  https://arxiv.org/abs/1910.10683
  """

  @dataclasses.dataclass(frozen=True)
  class Config(RelativePositionEmbedding.Config):
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
        labels=[labels.IS_EMBEDDING],
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


class TransformerXLRelativePositionEmbedding(RelativePositionEmbedding):
  """TransformerXL-style relative position embeddings.

  TODO(rryan): Currently only works with LocalDotProductSelfAttention due to
  batched queries in get_logits_streaming.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(RelativePositionEmbedding.Config):
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


@jt.typed
def _dot_product_attention(
    queries: jt.Float[jt.ArrayT, 'b q nq h'],
    keys: jt.Float[jt.ArrayT, 'b k nk h'],
    values: jt.Float[jt.ArrayT, 'b k nk h'],
    logit_visibility_mask: jt.Bool[jt.ArrayT, 'b #nq #q k'],
    logit_bias: jt.Float[jt.ArrayT, '#b #nq #q #k'] | None,
    training: bool,
    attention_logits_soft_cap: float | None,
    attention_probabilities_dropout: simple.Dropout | None,
    per_dim_scale: jt.Float[jt.ArrayT, 'h'] | None,
    query_scale: float | jt.ScalarFloat | None,
    precision: nn.linear.PrecisionLike,
    get_logits_fn: (
        Callable[[jax.Array, jax.Array, nn.linear.PrecisionLike], jax.Array]
        | None
    ),
    zero_fully_masked: bool,
    compute_dtype: Any | types.DType | None,
    num_sink_embeddings: int,
    sink_key_logits: jt.Float[jt.ArrayT, 'b nq q s'] | None,
    sink_value_embeddings: jt.Float[jt.ArrayT, 's nk h'] | None,
) -> tuple[
    jt.Float[jt.ArrayT, 'b q nq h'],
    jt.Float[jt.ArrayT, 'b q nq k+{num_sink_embeddings}'],
]:
  """Computes standard dot product attention with queries, keys and values.

  Args:
    queries: A [batch, query_time, num_query_heads, units_per_head] tensor of
      queries.
    keys: A [batch, key_time, num_key_value_heads, units_per_head] tensor of
      keys.
    values: A [batch, key_time, num_key_value_heads, units_per_head] tensor of
      values.
    logit_visibility_mask: A tensor broadcastable to [batch, num_heads,
      query_time, key_time] which is True for positions that are valid (i.e.
      should be attended to).
    logit_bias: A tensor broadcastable to [batch, num_heads, query_time,
      key_time] of attention logit biases to apply after computing the logits.
    training: Whether we are in training mode.
    attention_logits_soft_cap: If non-zero, a soft cap applied to attention
      logits to prevent outliers from dominating the softmax. Empirically, 50.0
      works well across a variety of tasks. Implemented as tanh(logits / cap) *
      cap.
    attention_probabilities_dropout: A dropout layer to apply to the attention
      probabilities.
    per_dim_scale: A [units_per_head] query scale factor for all query heads.
      Assuming per_dim_scale is zeros at initialization (or if not specified),
      the scaling applied to each head is query_scale.
    query_scale: A float query scale factor for all query heads. If None,
      queries are scaled by 1/sqrt(units_per_head).
    precision: Precision config for einsums.
    get_logits_fn: If provided, a function that computes logits given queries,
      keys, and precision. Used to augment logit calculation with relative
      position biases.
    zero_fully_masked: Outputs all-zeros context vectors for queries which have
      nothing to attend to (i.e. all possible keys are masked).
    compute_dtype: The dtype to use for computations.
    num_sink_embeddings: The number of sink embeddings (only needed for
      jaxtyping).
    sink_key_logits: Logits used to sink the attention [B, N, T, K].
    sink_value_embeddings: Value embeddings corresponding to the sink keys [K,
      N, H].

  Returns:
    context_vectors: A [batch_size, query_time, num_query_heads, units_per_head]
      tensor of context vectors for the queries.
    probabilities: A [batch_size, query_time, num_query_heads, keys_time] tensor
    of
      attention probabilities (for debugging).
  """
  num_heads = queries.shape[2]
  num_kv_heads = keys.shape[2]

  if num_heads != num_kv_heads:
    if get_logits_fn is not None:
      raise NotImplementedError(
          f'get_logits_fn is incompatible with {num_heads=} != {num_kv_heads=}.'
      )
    return _dot_product_attention_gqa(
        queries,
        keys,
        values,
        logit_visibility_mask,
        logit_bias,
        training,
        attention_logits_soft_cap,
        attention_probabilities_dropout,
        per_dim_scale,
        query_scale,
        precision,
        zero_fully_masked,
        compute_dtype,
        num_sink_embeddings,
        sink_key_logits,
        sink_value_embeddings,
    )

  q_dtype = utils.get_promoted_dtype(queries.dtype, dtype=compute_dtype)
  qk_dtype = utils.get_promoted_dtype(q_dtype, keys.dtype, dtype=compute_dtype)
  qkv_dtype = utils.get_promoted_dtype(
      qk_dtype, values.dtype, dtype=compute_dtype
  )
  queries = _scale_query(queries.astype(q_dtype), per_dim_scale, query_scale)
  queries = queries.astype(qk_dtype)
  keys = keys.astype(qk_dtype)
  if get_logits_fn is None:
    logits = jnp.einsum('BiNH,BjNH->BNij', queries, keys, precision=precision)
  else:
    logits = get_logits_fn(queries, keys, precision).astype(qk_dtype)

  if logit_bias is not None:
    # Layout matches logits.
    utils.assert_is_compatible_with(
        logit_bias.shape, [None, num_heads, None, None]
    )
    # Check that logit_bias is broadcastable to logits.
    jnp.broadcast_shapes(logit_bias.shape, logits.shape)
    logits += logit_bias.astype(logits.dtype)

  logits = jax.ad_checkpoint.checkpoint_name(logits, 'logits')

  # Cap attention logits before masking.
  if attention_logits_soft_cap:
    logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

  if sink_key_logits is not None:
    if attention_logits_soft_cap:
      sink_key_logits = _soft_cap_attention_logits(
          sink_key_logits, attention_logits_soft_cap
      )
    # [B, K+S, N, H] shape.
    values = jnp.concatenate(
        [
            jnp.tile(
                sink_value_embeddings[jnp.newaxis], [values.shape[0], 1, 1, 1]
            ),
            values,
        ],
        axis=1,
    )

  def mask_and_softmax(
      logits: jax.Array, sink_key_logits: jax.Array | None
  ) -> jax.Array:
    logits = _mask_attention_logits(logits, logit_visibility_mask)

    if sink_key_logits is not None:
      # [B, N, T, K+S] shape.
      logits = jnp.concatenate([sink_key_logits, logits], axis=-1)

    return jax.nn.softmax(logits, axis=-1).astype(qkv_dtype)

  probabilities = utils.run_in_at_least_fp32(
      mask_and_softmax, restore_dtypes=False
  )(logits, sink_key_logits)

  if attention_probabilities_dropout:
    probabilities = attention_probabilities_dropout.apply_dropout(
        probabilities, training=training
    )

  if zero_fully_masked:
    probabilities = _zero_fully_masked_attention_probabilities(
        probabilities, logit_visibility_mask
    )

  # Contract the keys_time dimension into per-head context vectors:
  # [batch, query_time, num_heads, units_per_head].
  context_vectors = jnp.einsum(
      'BNts,BsNH->BtNH',
      probabilities,
      values.astype(qkv_dtype),
      precision=precision,
  )
  context_vectors = jax.ad_checkpoint.checkpoint_name(
      context_vectors, 'context'
  )

  # Transpose [batch, num_heads, query_time, source_time] to
  # [batch, query_time, num_heads, source_time].
  probabilities = jnp.transpose(probabilities, [0, 2, 1, 3])
  return context_vectors, probabilities


@jt.typed
def _multi_key_value_dot_product_attention(
    queries: jt.Float[jt.ArrayT, 'b q nq h'],
    kv_buffers: TypingSequence[tuple[jax.Array, jax.Array, jax.Array]],
    logit_bias: jt.Float[jt.ArrayT, '#b #nq #q #k'] | None,
    training: bool,
    attention_logits_soft_cap: float | None,
    attention_probabilities_dropout: simple.Dropout | None,
    per_dim_scale: jt.Float[jt.ArrayT, 'h'] | None,
    query_scale: float | jt.ScalarFloat | None,
    precision: nn.linear.PrecisionLike,
    get_logits_fn: (
        Callable[[jax.Array, jax.Array, nn.linear.PrecisionLike], jax.Array]
        | None
    ),
    zero_fully_masked: bool,
    compute_dtype: Any | types.DType | None,
) -> tuple[
    jt.Float[jt.ArrayT, 'b q nq h'],
    list[jt.Float[jt.ArrayT, 'b q nq _']],
]:
  """Computes "multi key-value" dot product attention with queries.

  This is equivalent to _dot_product_attention but allows providing the keys and
  values to attend over as a list of multiple arrays, each key/value group
  having its own visibility mask.

  Args:
    queries: A [batch, query_time, num_query_heads, units_per_head] tensor of
      queries.
    kv_buffers: A sequence of (key, values, valid_mask) tuples, where: keys are
      [batch, key_time, num_key_value_heads, units_per_head] values are [batch,
      key_time, num_key_value_heads, units_per_head] valid_mask is broadcastable
      to [batch, num_heads, query_time, key_time] which is True for positions
      that are valid (i.e. should be attended to).
    logit_bias: A tensor broadcastable to [batch, num_heads, query_time,
      key_time] of attention logit biases to apply after computing the logits.
    training: Whether we are in training mode.
    attention_logits_soft_cap: If non-zero, a soft cap applied to attention
      logits to prevent outliers from dominating the softmax. Empirically, 50.0
      works well across a variety of tasks. Implemented as tanh(logits / cap) *
      cap.
    attention_probabilities_dropout: A dropout layer to apply to the attention
      probabilities.
    per_dim_scale: A [units_per_head] query scale factor for all query heads.
      Assuming per_dim_scale is zeros at initialization (or if not specified),
      the scaling applied to each head is query_scale.
    query_scale: A float query scale factor for all query heads. If None,
      queries are scaled by 1/sqrt(units_per_head).
    precision: Precision config for einsums.
    get_logits_fn: If provided, a function that computes logits given queries,
      keys, and precision. Used to augment logit calculation with relative
      position biases.
    zero_fully_masked: Outputs all-zeros context vectors for queries which have
      nothing to attend to (i.e. all possible keys are masked).
    compute_dtype: The dtype to use for computations.

  Returns:
    context_vectors: A [batch_size, query_time, num_query_heads, units_per_head]
      tensor of context vectors for the queries.
    probabilities: A list of [batch_size, query_time, num_query_heads,
      keys_time] tensor of attention probabilities (for debugging).
  """
  num_heads = queries.shape[2]

  if not kv_buffers:
    raise ValueError('kv_buffers must be non-empty.')

  if get_logits_fn is not None:
    raise NotImplementedError(
        'get_logits_fn is incompatible with'
        ' multi_key_value_dot_product_attention.'
    )
  if logit_bias is not None:
    raise NotImplementedError(
        'logit_bias is incompatible with multi_key_value_dot_product_attention.'
    )

  keys, values, valid_masks = zip(*kv_buffers, strict=True)
  del kv_buffers

  for keys_i in keys:
    num_kv_heads = keys_i.shape[2]
    if num_heads != num_kv_heads:
      raise ValueError(f'{num_heads=} must be equal to {num_kv_heads=}')

  q_dtype = utils.get_promoted_dtype(queries.dtype, dtype=compute_dtype)
  qk_dtype = utils.get_promoted_dtype(
      q_dtype, keys[0].dtype, dtype=compute_dtype
  )
  qkv_dtype = utils.get_promoted_dtype(
      qk_dtype, values[0].dtype, dtype=compute_dtype
  )
  queries = _scale_query(queries.astype(q_dtype), per_dim_scale, query_scale)

  queries = queries.astype(qk_dtype)

  logit_buffers = []
  for keys_i in keys:
    keys_i = keys_i.astype(qk_dtype)
    logits_i = jnp.einsum(
        'BiNH,BjNH->BNij', queries, keys_i, precision=precision
    )

    # Cap attention logits before masking.
    if attention_logits_soft_cap:
      logits_i = _soft_cap_attention_logits(logits_i, attention_logits_soft_cap)

    logit_buffers.append(logits_i)

  def mask_and_softmax(*scores: jax.Array) -> TypingSequence[jax.Array]:
    common_shape = scores[0].shape[:-1]
    for buffer in scores:
      assert buffer.ndim == 4, scores
      assert buffer.shape[:-1] == common_shape, scores

    scores = [
        _mask_attention_logits(scores_i, mask_i)
        for scores_i, mask_i in zip(scores, valid_masks, strict=True)
    ]
    return _sharded_softmax(scores, axis=-1, output_dtype=qkv_dtype)

  probabilities = utils.run_in_at_least_fp32(
      mask_and_softmax, restore_dtypes=False
  )(*logit_buffers)

  if attention_probabilities_dropout:
    probabilities = [
        attention_probabilities_dropout.apply_dropout(p, training=training)
        for p in probabilities
    ]

  if zero_fully_masked:
    probabilities = [
        _zero_fully_masked_attention_probabilities(p, v)
        for p, v in zip(probabilities, valid_masks, strict=True)
    ]

  # Contract the keys_time dimension into per-head context vectors:
  # [batch, query_time, num_heads, units_per_head].
  context_vectors = []
  for probabilities_i, values_i in zip(probabilities, values, strict=True):
    context_vectors_i = jnp.einsum(
        'BNts,BsNH->BtNH',
        probabilities_i,
        values_i.astype(qkv_dtype),
        precision=precision,
    )
    context_vectors.append(context_vectors_i)

  context_vectors = functools.reduce(jnp.add, context_vectors)
  context_vectors = jax.ad_checkpoint.checkpoint_name(
      context_vectors, 'context'
  )

  # Transpose [batch, num_heads, query_time, source_time] to
  # [batch, query_time, num_heads, source_time].
  probabilities = [jnp.transpose(p, [0, 2, 1, 3]) for p in probabilities]
  return context_vectors, probabilities


@jt.typed
def _dot_product_attention_gqa(
    queries: jt.Float[jt.ArrayT, 'b q nq h'],
    keys: jt.Float[jt.ArrayT, 'b k nk h'],
    values: jt.Float[jt.ArrayT, 'b k nk h'],
    logit_visibility_mask: jt.Bool[jt.ArrayT, 'b #nq #q k'],
    logit_bias: jt.Float[jt.ArrayT, '#b #nq #q #k'] | None,
    training: bool,
    attention_logits_soft_cap: float | None,
    attention_probabilities_dropout: simple.Dropout | None,
    per_dim_scale: jt.Float[jt.ArrayT, 'h'] | None,
    query_scale: float | jt.ScalarFloat | None,
    precision: nn.linear.PrecisionLike,
    zero_fully_masked: bool,
    compute_dtype: Any | types.DType | None,
    num_sink_embeddings: int,
    sink_key_logits: jt.Float[jt.ArrayT, 'b nq q s'] | None,
    sink_value_embeddings: jt.Float[jt.ArrayT, 's nk h'] | None,
) -> tuple[
    jt.Float[jt.ArrayT, 'b q nq h'],
    jt.Float[jt.ArrayT, 'b q nq k+{num_sink_embeddings}'],
]:
  """Computes standard dot product attention with queries, keys and values.

  Args:
    queries: A [batch, query_time, num_query_heads, units_per_head] tensor of
      queries.
    keys: A [batch, key_time, num_key_value_heads, units_per_head] tensor of
      keys.
    values: A [batch, key_time, num_key_value_heads, units_per_head] tensor of
      values.
    logit_visibility_mask: A tensor broadcastable to [batch, num_heads,
      query_time, key_time] which is True for positions that are valid (i.e.
      should be attended to).
    logit_bias: A tensor broadcastable to [batch, num_heads, query_time,
      key_time] of attention logit biases to apply after computing the logits.
    training: Whether we are in training mode.
    attention_logits_soft_cap: If non-zero, a soft cap applied to attention
      logits to prevent outliers from dominating the softmax. Empirically, 50.0
      works well across a variety of tasks. Implemented as tanh(logits / cap) *
      cap.
    attention_probabilities_dropout: A dropout layer to apply to the attention
      probabilities.
    per_dim_scale: A [units_per_head] query scale factor for all query heads.
      Assuming per_dim_scale is zeros at initialization (or if not specified),
      the scaling applied to each head is query_scale.
    query_scale: A float query scale factor for all query heads. If None,
      queries are scaled by 1/sqrt(units_per_head).
    precision: Precision config for einsums.
    zero_fully_masked: Outputs all-zeros context vectors for queries which have
      nothing to attend to (i.e. all possible keys are masked).
    compute_dtype: The dtype to use for computations.
    num_sink_embeddings: The number of sink embeddings (only needed for
      jaxtyping).
    sink_key_logits: Logits used to sink the attention [batch, num_query_heads,
      query_time, num_sink_embeddings].
    sink_value_embeddings: Value embeddings corresponding to the sink keys
      [num_sink_embeddings, num_key_value_heads, units_per_head].

  Returns:
    context_vectors: A [batch_size, query_time, num_query_heads, units_per_head]
      tensor of context vectors for the queries.
    probabilities: A [batch_size, query_time, num_query_heads, keys_time +
      num_sink_embeddings] tensor of attention probabilities (for debugging).
  """
  del num_sink_embeddings
  num_heads, units_per_head = queries.shape[-2:]
  num_kv_heads = keys.shape[2]

  if num_heads % num_kv_heads != 0:
    raise ValueError(f'{num_heads=} must be divisible by {num_kv_heads=}')

  num_query_heads_per_kv_head = num_heads // num_kv_heads

  q_dtype = utils.get_promoted_dtype(queries.dtype, dtype=compute_dtype)
  qk_dtype = utils.get_promoted_dtype(q_dtype, keys.dtype, dtype=compute_dtype)
  qkv_dtype = utils.get_promoted_dtype(
      qk_dtype, values.dtype, dtype=compute_dtype
  )
  queries = _scale_query(queries.astype(q_dtype), per_dim_scale, query_scale)
  queries = queries.astype(qk_dtype)
  keys = keys.astype(qk_dtype)

  queries = utils.split_dimension(
      queries, axis=2, shape=(num_kv_heads, num_query_heads_per_kv_head)
  )

  logits = jnp.einsum('BjKH,BiKQH->BKQij', keys, queries, precision=precision)

  if logit_bias is not None:
    if logit_bias.shape[1] == 1:
      logit_bias = logit_bias[:, :, jnp.newaxis, :, :]
    else:
      logit_bias = utils.split_dimension(
          logit_bias, axis=1, shape=(num_kv_heads, num_query_heads_per_kv_head)
      )

    # Check that logit_bias is broadcastable to logits.
    jnp.broadcast_shapes(logit_bias.shape, logits.shape)
    logits += logit_bias.astype(logits.dtype)

  # Cap attention logits before masking.
  if attention_logits_soft_cap:
    logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

  if logit_visibility_mask.shape[1] == 1:
    logit_visibility_mask = logit_visibility_mask[:, jnp.newaxis, :, :, :]
  else:
    logit_visibility_mask = utils.split_dimension(
        logit_visibility_mask,
        axis=1,
        shape=(num_kv_heads, num_query_heads_per_kv_head),
    )

  if sink_key_logits is not None:
    sink_key_logits = utils.split_dimension(
        sink_key_logits,
        axis=1,
        shape=(num_kv_heads, num_query_heads_per_kv_head),
    )
    if attention_logits_soft_cap:
      sink_key_logits = _soft_cap_attention_logits(
          sink_key_logits, attention_logits_soft_cap
      )
    # [B, K+S, N, H] shape.
    values = jnp.concatenate(
        [
            jnp.tile(
                sink_value_embeddings[jnp.newaxis], [values.shape[0], 1, 1, 1]
            ),
            values,
        ],
        axis=1,
    )

  def mask_and_softmax(
      logits: jax.Array, sink_key_logits: jax.Array | None
  ) -> jax.Array:
    logits = _mask_attention_logits(logits, logit_visibility_mask)

    if sink_key_logits is not None:
      # [B, NK, QPK, T, K+S] shape.
      logits = jnp.concatenate([sink_key_logits, logits], axis=-1)

    return jax.nn.softmax(logits, axis=-1).astype(qkv_dtype)

  probabilities = utils.run_in_at_least_fp32(
      mask_and_softmax, restore_dtypes=False
  )(logits, sink_key_logits)

  if attention_probabilities_dropout:
    probabilities = attention_probabilities_dropout.apply_dropout(
        probabilities, training=training
    )

  if zero_fully_masked:
    probabilities = _zero_fully_masked_attention_probabilities(
        probabilities, logit_visibility_mask
    )

  # Contract the keys_time dimension into per-head context vectors:
  # [batch, query_time, num_heads, units_per_head].
  context_vectors = jnp.einsum(
      'BKQts,BsKH->BtKQH',
      probabilities,
      values.astype(qkv_dtype),
      precision=precision,
  )

  context_vectors = context_vectors.reshape(
      context_vectors.shape[:2] + (num_heads, units_per_head)
  )

  # Transpose and reshape [batch, num_kv_heads, num_query_heads_per_kv_head,
  # query_time, source_time] to [batch, query_time, num_query_heads,
  # source_time].
  probabilities = jnp.transpose(probabilities, [0, 3, 1, 2, 4]).reshape(
      probabilities.shape[0],
      probabilities.shape[3],
      num_heads,
      probabilities.shape[4],
  )
  return context_vectors, probabilities


@jt.typed
def _local_dot_product_attention(
    queries: jt.Float[jt.ArrayT, 'b q nq h'],
    keys: jt.Float[jt.ArrayT, 'b k nk h'],
    keys_mask: jt.Bool[jt.ArrayT, 'b k'],
    values: jt.Float[jt.ArrayT, 'b k nk h'],
    block_size: int,
    max_past_horizon: int,
    max_future_horizon: int,
    training: bool,
    attention_logits_soft_cap: float | None,
    attention_probabilities_dropout: simple.Dropout | None,
    per_dim_scale: jt.Float[jt.ArrayT, 'h'] | None,
    query_scale: float | jt.ScalarFloat | None,
    precision: nn.linear.PrecisionLike | None,
    get_logits_fn: (
        Callable[[jax.Array, jax.Array, nn.linear.PrecisionLike], jax.Array]
        | None
    ),
    zero_fully_masked: bool,
    compute_dtype: Any | types.DType | None,
    num_sink_embeddings: int,
    sink_key_logits: jt.Float[jt.ArrayT, 'b nq q s'] | None,
    sink_value_embeddings: jt.Float[jt.ArrayT, 's nq h'] | None,
) -> tuple[
    jt.Float[jt.ArrayT, 'b q nq h'],
    jt.Float[
        jt.ArrayT,
        'b q nq'  # pylint: disable=implicit-str-concat
        ' {max_past_horizon+block_size+max_future_horizon+num_sink_embeddings}',
    ],
]:
  """Computes "local" dot product attention with queries, keys and values.

  Assumes "sliding window attention" is in use, where queries have a limited
  backward and forward receptive field of max_past_horizon and
  max_future_horizon. For efficiency, extracts query blocks of length block_size
  and key blocks of length (max_past_horizon + block_size + max_future_horizon)
  and computes query/key dot products within these pairs of blocks instead of
  materializing the full [query, key] logit matrix (which is highly sparse in
  sliding window attention).

  _local_dot_product_attention is functionally equivalent to
  _dot_product_attention with a valid_mask constructed to respect
  max_past_horizon and max_future_horizon. It is purely a performance
  optimization that can save memory and compute.

  Implementation follows Praxis:
  google3/third_party/py/praxis/layers/attentions.py

  Note that query_time and key_time are equal (i.e. S == T).

  WARNING: Praxis includes the current timestep in the "past_context", while
  this implementation defines max_past_horizon as only past timesteps.

  Args:
    queries: A [batch, query_time, num_heads, units_per_head] tensor of queries.
    keys: A [batch, key_time, num_heads, units_per_head] tensor of keys.
    keys_mask: A [batch, key_time] mask matrix for the keys.
    values: A [batch, key_time, num_heads, units_per_head] tensor of values.
    block_size: The block size of queries to process at once. Peak memory usage
      scales by block_size * (past_context + block_size + future_context).
    max_past_horizon: The amount of past context that each query can attend to.
    max_future_horizon: The amount of future context that each query can attend
      to.
    training: Whether we are in training mode.
    attention_logits_soft_cap: If non-zero, a soft cap applied to attention
      logits to prevent outliers from dominating the softmax. Empirically, 50.0
      works well across a variety of tasks. Implemented as tanh(logits / cap) *
      cap.
    attention_probabilities_dropout: A dropout layer to apply to the attention
      probabilities.
    per_dim_scale: A [units_per_head] query scale factor for all query heads.
      Assuming per_dim_scale is zeros at initialization (or if not specified),
      the scaling applied to each head is query_scale.
    query_scale: A float query scale factor for all query heads. If None,
      queries are scaled by 1/sqrt(units_per_head).
    precision: Precision config for einsums.
    get_logits_fn: If provided, a function that computes logits given queries,
      keys, and precision. Used to augment logit calculation with relative
      position biases.
    zero_fully_masked: Outputs all-zeros context vectors for queries which have
      nothing to attend to (i.e. all possible keys are masked).
    compute_dtype: The dtype to use for computations.
    num_sink_embeddings: The number of sink embeddings (only needed for
      jaxtyping).
    sink_key_logits: Logits used to sink the attention [B, N, T, K].
    sink_value_embeddings: Value embeddings corresponding to the sink keys [K,
      N, H].

  Returns:
    context_vectors: A [batch_size, query_time, num_heads, units_per_head]
      tensor of context vectors for the queries.
    probabilities: A [batch_size, query_time, num_heads, keys_time] tensor of
      attention probabilities (for debugging).
  """
  del num_sink_embeddings
  num_heads, units_per_head = queries.shape[-2:]
  num_kv_heads = keys.shape[2]

  if num_heads != num_kv_heads:
    raise NotImplementedError(
        f'_local_dot_product_attention requires {num_heads=} =='
        f' {num_kv_heads=}.'
    )

  q_dtype = utils.get_promoted_dtype(queries.dtype, dtype=compute_dtype)
  qk_dtype = utils.get_promoted_dtype(q_dtype, keys.dtype, dtype=compute_dtype)
  qkv_dtype = utils.get_promoted_dtype(
      qk_dtype, values.dtype, dtype=compute_dtype
  )

  assert block_size > 0
  assert max_past_horizon >= 0
  assert max_future_horizon >= 0

  queries = _scale_query(queries.astype(q_dtype), per_dim_scale, query_scale)

  batch_size, original_query_time = queries.shape[:2]
  num_sink_embeddings = (
      0 if sink_key_logits is None else sink_key_logits.shape[-1]
  )
  keys_time = keys.shape[1]

  # We assume self-attention so keys and queries time match.
  assert (
      keys_time == original_query_time
  ), f'{keys_time} != {original_query_time}'
  context_size = block_size + max_past_horizon + max_future_horizon

  # [B, S, N, H] -> [B, U, C, N, H]; (U = S/W).
  keys_blocks = _extract_block_context(
      keys,
      block_size=block_size,
      left_context=max_past_horizon,
      right_context=max_future_horizon,
  )

  # [B, T, N, H] -> [B, U, W, N, H]; (U = T/W).
  queries_blocks = _convert_to_block(queries, block_size=block_size)
  num_query_blocks = queries_blocks.shape[1]

  # Squeeze the heads and query time out to get [b, keys_time].
  # valid_mask_blocked is [b, num_blocks, context_size]
  valid_mask_blocked = _extract_block_context(
      keys_mask,
      block_size=block_size,
      left_context=max_past_horizon,
      right_context=max_future_horizon,
      # Valid mask is False for invalid timesteps.
      padding_val=False,
  )

  # Reshape to [b, h=1, num_blocks, block_size=1, context_size].
  valid_mask_blocked = valid_mask_blocked[:, jnp.newaxis, :, jnp.newaxis, :]

  # Make local causal mask [block_size, context_size].
  # For block_size queries, which timesteps in context they can attend to.
  # max_past_horizon = 4, max_future_horizon = 2, block_size = 3
  # Input:
  # 0 1 2 3 a b c 4 5
  # Causal mask:
  # 1 1 1 1 a 1 1 0 0
  # 0 1 1 1 1 b 1 1 0
  # 0 0 1 1 1 1 c 1 1
  local_causal_valid_mask = utils.ones_matrix_band_part(
      block_size,
      context_size,
      num_upper=max_past_horizon + max_future_horizon,
      num_lower=0,
      out_dtype=jnp.bool_,
  )

  # Combine valid timesteps with and.
  valid_mask_blocked = jnp.logical_and(
      valid_mask_blocked,
      local_causal_valid_mask[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :],
  )

  queries_blocks = queries_blocks.astype(qk_dtype)
  keys_blocks = keys_blocks.astype(qk_dtype)
  if get_logits_fn is not None:
    logits = get_logits_fn(queries_blocks, keys_blocks, precision).astype(
        qk_dtype
    )
  else:
    logits = jnp.einsum(
        'BuwNH,BucNH->BNuwc',
        queries_blocks,
        keys_blocks,
        precision=precision,
    )

  # Cap attention logits before masking.
  if attention_logits_soft_cap:
    logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

  if sink_key_logits is not None:
    if attention_logits_soft_cap:
      sink_key_logits = _soft_cap_attention_logits(
          sink_key_logits, attention_logits_soft_cap
      )
    # BNTK -> BTNK
    sink_key_logits = jnp.transpose(sink_key_logits, [0, 2, 1, 3])
    # BTNK -> BUWNK
    sink_key_logits = _convert_to_block(sink_key_logits, block_size=block_size)
    # BUWNK -> BNUWK
    sink_key_logits = jnp.transpose(sink_key_logits, [0, 3, 1, 2, 4])

  def mask_and_softmax(
      logits: jax.Array, sink_key_logits: jax.Array | None
  ) -> jax.Array:
    # [B, N, U, W, C]
    logits = _mask_attention_logits(logits, valid_mask_blocked)

    if sink_key_logits is not None:
      # [B, N, U, W, K+C] shape.
      logits = jnp.concatenate([sink_key_logits, logits], axis=-1)

    return jax.nn.softmax(logits, axis=-1).astype(qkv_dtype)

  probabilities = utils.run_in_at_least_fp32(
      mask_and_softmax, restore_dtypes=False
  )(logits, sink_key_logits)

  if attention_probabilities_dropout:
    probabilities = attention_probabilities_dropout.apply_dropout(
        probabilities, training=training
    )

  if zero_fully_masked:
    probabilities = _zero_fully_masked_attention_probabilities(
        probabilities, valid_mask_blocked
    )

  # [B, U, C, N, H]
  values_blocks = _extract_block_context(
      values,
      block_size=block_size,
      left_context=max_past_horizon,
      right_context=max_future_horizon,
  )
  if sink_key_logits is not None:
    # KNH -> BUKNH
    sink_value_embeddings = jnp.tile(
        sink_value_embeddings[jnp.newaxis, jnp.newaxis],
        [values_blocks.shape[0], values_blocks.shape[1], 1, 1, 1],
    )

    # [B, U, K+C, N, H] shape.
    values_blocks = jnp.concatenate(
        [sink_value_embeddings, values_blocks], axis=2
    )

  # Contract the context windows dimension (c) into per-head context vectors
  # across each local block: [batch, num_query_blocks, block_size, num_heads,
  # units_per_head].
  context_vectors = jnp.einsum(
      'BNuwc,BucNH->BuwNH',
      probabilities,
      values_blocks.astype(qkv_dtype),
      precision=precision,
  )

  context_vectors = jnp.reshape(
      context_vectors,
      [batch_size, num_query_blocks * block_size, num_heads, units_per_head],
  )

  # Trim off query padding.
  context_vectors = context_vectors[:, :original_query_time]

  # Transpose [batch, num_heads, num_query_blocks, block_size, context_size]
  # to [batch, query_time, num_heads, num_sink_embeddings+context_size].
  probabilities = jnp.transpose(probabilities, [0, 2, 3, 1, 4])
  probabilities = jnp.reshape(
      probabilities,
      [
          batch_size,
          num_query_blocks * block_size,
          num_heads,
          num_sink_embeddings + context_size,
      ],
  )
  probabilities = probabilities[:, :original_query_time]

  return context_vectors, probabilities


def _self_attention_step_visibility_mask(
    max_past_horizon: int,
    max_future_horizon: int,
    query_time: int,
    key_time: int,
) -> jax.Array | None:
  """Compute a step-wise visibility mask for self-attention."""
  if query_time == 1:
    # If queries_time is 1, there is no need for causal masking because our
    # state covers the previous max_past_horizon timesteps, the current query
    # timestep, and the future max_future_horizon timesteps that we
    # concatenated (all of which we can look at). We can simply use the KV
    # cache mask mask, reshaped to [batch_size, 1, 1, keys_time] so it
    # broadcasts across heads and queries_time.
    return None
  else:
    # To obey causality, the output for each of the block_size input timesteps
    # may only depend on itself, the max_past_horizon previous timesteps and
    # the max_future_horizon future timesteps. Earlier timesteps in block_size
    # cannot depend on later timesteps.
    #
    # After the above concatenation, the state tensor is of length
    # max_past_horizon + max_future_horizon + block_size. For example, with
    # max_past_horizon = 5, max_future_horizon=2, and block_size = 3:
    #
    # 1 2 3 4 5 a b c 6 7
    #
    # Note that when max_future_horizon > 0, the above query_cache handling
    # delays the incoming queries so the block of context vectors we produce
    # for each step corresponds to the interior [a, b, c] timesteps, not the
    # final block_size timesteps in the state buffer.
    #
    # Timestep a can look at 1-5 and a-c:
    # 1 1 1 1 1 1 1 1 0 0
    # Timestep b can look at 2-5, a-c, and 6:
    # 0 1 1 1 1 1 1 1 1 0
    # Timestep c can look at 3-5, a-c, and 6-7:
    # 0 0 1 1 1 1 1 1 1 1
    #
    # This matrix corresponds to a 3x10 banded matrix with zero lower-bandwidth
    # and max_past_horizon + max_future_horizon upper-bandwidth.
    return utils.ones_matrix_band_part(
        query_time,
        key_time,
        num_lower=0,
        num_upper=max_past_horizon + max_future_horizon,
        out_shape=[1, 1, query_time, key_time],
        out_dtype=jnp.bool_,
    )


def _self_attention_layer_visibility_mask(
    max_past_horizon: int,
    max_future_horizon: int,
    time: int,
) -> jax.Array | None:
  """Compute a layer-wise visibility mask for self-attention."""
  # Compute a connectivity mask indicating which of the [query, key]
  # timesteps can see each other. Since query_time == key_time, this is a
  # banded square matrix with max_past_horizon lower-bandwidth and
  # max_future_horizon upper-bandwidth.
  #
  # For example, for values_time = 5, max_past_horizon = 2,
  # max_future_horizon = 1:
  #
  # a b c d e
  #
  # a can only look at itself and b:
  # 1 1 0 0 0
  # b can look at a, itself and c:
  # 1 1 1 0 0
  # c can look at a, b, itself and d:
  # 1 1 1 1 0
  # d can look at b, c, itself and e.
  # 0 1 1 1 1
  # e can look at c, d and itself:
  # 0 0 1 1 1
  #
  # This corresponds to a 5x5 banded matrix with max_past_horizon
  # lower-bandwidth and max_future_horizon upper-bandwidth.
  if max_past_horizon != -1 or max_future_horizon != -1:
    num_lower = max(0, time - 1)
    if max_past_horizon != -1:
      # Handles the corner case where values_time - 1 is less than
      # max_past_horizon.
      num_lower = min(num_lower, max_past_horizon)

    num_upper = max(0, time - 1)
    if max_future_horizon != -1:
      # Handle the corner case where values_time - 1 is less than
      # max_future_horizon.
      num_upper = min(num_upper, max_future_horizon)
    visibility_mask = utils.ones_matrix_band_part(
        time,
        time,
        num_lower=num_lower,
        num_upper=num_upper,
        # [1, 1, queries_time, keys_time] so it broadcasts across batch size
        # and heads.
        out_shape=[1, 1, time, time],
        out_dtype=jnp.bool_,
    )
    utils.assert_is_compatible_with(visibility_mask.shape, [1, 1, time, time])
  else:
    # If both max_past_horizon and max_future_horizon are -1 then we operate
    # unmasked.
    visibility_mask = None

  return visibility_mask


class DotProductSelfAttention(types.Emitting, AttentionInputProjectionHelper):
  """A multi-headed dot-product self attention layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for DotProductSelfAttention."""

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

    # An optional RelativePositionEmbedding to use to compute relative position
    # biases.
    # The layer is not steppable if a relative_position_embedding is provided
    # and max_future_horizon > 0.
    relative_position_embedding: RelativePositionEmbedding.Config | None = None
    # The dropout rate for the attention probabilities.
    attention_probabilities_dropout_rate: float = 0.0
    # Whether to broadcast the dropout across the query time dimension as is
    # done in T5.
    broadcast_dropout_across_queries: bool = False
    # Whether to learn a bias in the query/key/value projection.
    use_bias: bool = False
    # Configuration for the query, key and value input projection parameters.
    # If num_kv_heads is set, must not be CombinedQueryKeyValueProjection.
    # If shared_kv_projection is set, must be QueryAndSharedKeyValueProjection.
    input_projection: QueryKeyValueProjectionConfig = dataclasses.field(
        default_factory=CombinedQueryKeyValueProjection
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
    # Outputs all-zeros context vectors for queries which have nothing to attend
    # to (i.e. all possible keys are masked).
    zero_fully_masked: bool = False
    # The dtype of the layer's computations.
    compute_dtype: types.DType | None = None
    # The dtype of the layer's parameters.
    param_dtype: types.DType = jnp.float32
    # The number of sink embeddings to include in the key and value.
    # Pax implementation:
    # google3/learning/multipod/pax/audio/lm/layers/streaming/attentions.py.
    # Paper: https://arxiv.org/pdf/2309.17453.pdf.
    num_sink_embeddings: int = 0
    # By default initialize the sink token embeddings to have a norm of 1.
    sink_embeddings_init: nn.initializers.Initializer = (
        nn.linear.default_embed_init
    )
    sink_embeddings_sharding: types.Sharding | None = None
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
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'DotProductSelfAttention':
      return DotProductSelfAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    _validate_heads(
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
              num_kv_heads,
              self.config.units_per_head,
          ),
          self.config.param_dtype,
      )

    if self.config.use_kv_cache_ringbuffer:
      num_kv_heads = self.config.num_kv_heads or self.config.num_heads
      if num_kv_heads != self.config.num_heads:
        raise NotImplementedError(
            'num_kv_heads is not supported with use_kv_cache_ringbuffer.'
        )
      if self.config.num_sink_embeddings > 0:
        raise NotImplementedError(
            'Sink embeddings are not supported with use_kv_cache_ringbuffer.'
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
          f' ({self.query_network.output_ratio}) and block_size'
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
          f' ({self.key_network.output_ratio}) and block_size'
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
          f' ({self.value_network.output_ratio}) and block_size'
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
        self.config.relative_position_embedding.make()
        if self.config.relative_position_embedding is not None
        else None
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
    # Default to 0 even if the layer is not steppable.
    return (
        self.config.max_future_horizon
        if self.config.max_future_horizon >= 0
        else 0
    )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    compute_dtype = self.get_input_projection_output_dtype(
        self.config.input_projection, input_spec.dtype
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

    # Store these two sequences "unpacked" since they share a mask.
    kv_buffer_keys = kv_zero_values
    kv_buffer_values = kv_zero_values
    kv_buffer_mask = kv_zero_mask

    # If we have a finite future horizon, we cannot produce outputs for timestep
    # t until the max_future_horizon KV timesteps have arrived. Store incoming
    # queries in a delay buffer so we do not compute context vectors for them
    # until max_future_horizon KV timesteps have arrived.
    if max_future_horizon:
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
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    )

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
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

    x_queries, x_keys, x_values = self.get_qkv(self.config.input_projection, x)

    # Our params might not match param_dtype, so delegate the compute_dtype to
    # the output of the QKV layer.
    compute_dtype = x_queries.dtype

    (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    ) = state

    kv_buffer_size = kv_buffer_keys.shape[1]

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

    if self.config.use_kv_cache_ringbuffer:
      # Leave the input key/values and KV cache separate for multi key/value dot
      # product attention.
      kv_buffer_time = kv_buffer_size
    else:
      # To process a step, concatenate our kv_buffer_size KV buffer with the
      # input source.shape[1] timesteps.
      kv_buffer_keys = jnp.concatenate([kv_buffer_keys, x_keys.values], axis=1)
      kv_buffer_values = jnp.concatenate(
          [kv_buffer_values, x_values.values], axis=1
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
          [query_delay_buffer, x_queries]
      )
      assert (
          query_delay_buffer.shape[1]
          == x_values_time + self.config.max_future_horizon
      )

      # Use the oldest x_values_time queries as the current step's queries. They
      # each have max_future_horizon context available in
      # kv_buffer_keys/kv_buffer_values so we can produce valid output for them.
      x_queries = query_delay_buffer[:, :x_values_time]

      # Preserve the last max_future_horizon queries for the next step.
      query_delay_buffer = query_delay_buffer[
          :, -self.config.max_future_horizon :
      ]

    if self.config.num_sink_embeddings > 0:
      # TODO(mvelimirovic): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', x_queries.values, self._sink_key_embeddings
      )
    else:
      sink_key_logits = None

    logit_position_bias = None
    if self.relative_position_embedding:
      # Position tracks the position of the output queries, not the number of
      # input queries. This is appropriate, since we are delaying the outputs of
      # the layer by max_future_horizon timesteps.
      query_positions = (
          time_step[:, jnp.newaxis] + jnp.arange(x_values_time)[jnp.newaxis, :]
      )
      max_previous = kv_buffer_time - x_values_time
      key_start_position = time_step - max_previous
      key_positions = (
          key_start_position[:, jnp.newaxis]
          + jnp.arange(kv_buffer_time)[jnp.newaxis, :]
      )
      logit_position_bias = self.relative_position_embedding.get_position_bias(
          query_positions=query_positions,
          key_positions=key_positions,
          queries=x_queries.values,
          keys=kv_buffer_keys,
      )

    valid_mask = kv_buffer_mask[:, jnp.newaxis, jnp.newaxis, :]
    if (
        visibility_mask := _self_attention_step_visibility_mask(
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

    # Compute context vectors:
    if self.config.use_kv_cache_ringbuffer:
      assert not self.relative_position_embedding
      assert not self.config.num_sink_embeddings

      context_vectors, probabilities = _multi_key_value_dot_product_attention(
          queries=x_queries.values,
          kv_buffers=(
              (kv_buffer_keys, kv_buffer_values, valid_mask),
              (
                  x_keys.values,
                  x_values.values,
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
      probabilities = jnp.concatenate(probabilities, axis=-1)
    else:
      context_vectors, probabilities = _dot_product_attention(
          queries=x_queries.values,
          keys=kv_buffer_keys,
          values=kv_buffer_values,
          logit_visibility_mask=valid_mask,
          logit_bias=logit_position_bias,
          training=training,
          attention_logits_soft_cap=self.config.attention_logits_soft_cap,
          attention_probabilities_dropout=self._attention_probabilities_dropout,
          per_dim_scale=self._per_dim_scale,
          query_scale=self.config.query_scale,
          precision=self.config.precision,
          get_logits_fn=None,
          zero_fully_masked=self.config.zero_fully_masked,
          compute_dtype=compute_dtype,
          num_sink_embeddings=self.config.num_sink_embeddings,
          sink_key_logits=sink_key_logits,
          sink_value_embeddings=self._sink_value_embeddings
          if self.config.num_sink_embeddings > 0
          else None,
      )

    # Update KV caches and state.
    if self.config.use_kv_cache_ringbuffer:
      # Write latest keys, values and masks to the appropriate position in the
      # KV cache ring buffers.
      i = time_step % kv_buffer_size
      assert combined_mask.shape[1] == x_keys.shape[1]
      assert x_values.shape[1] == x_keys.shape[1]

      update_fn = jax.vmap(lambda x, u, i: x.at[i].set(u.squeeze(0)))
      kv_buffer_keys = update_fn(kv_buffer_keys, x_keys.values, i)
      kv_buffer_values = update_fn(kv_buffer_values, x_values.values, i)
      kv_buffer_mask = update_fn(kv_buffer_mask, combined_mask, i)
    else:
      # Preserve last kv_buffer_size timesteps as state for next step.
      kv_buffer_keys = kv_buffer_keys[:, -kv_buffer_size:]
      kv_buffer_values = kv_buffer_values[:, -kv_buffer_size:]
      kv_buffer_mask = kv_buffer_mask[:, -kv_buffer_size:]

    state = (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step + x_values_time,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    )

    emits = SelfAttentionEmits(types.Sequence(probabilities, x_queries.mask))

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, x_queries.mask)

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
    batch_size = x.shape[0]

    queries, keys, values = self.get_qkv(self.config.input_projection, x)

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

    # Guard against NaN/Inf in values, since values are contracted when
    # computing context vectors.
    values = values.mask_invalid()

    if self.config.num_sink_embeddings > 0:
      # TODO(mvelimirovic): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', queries.values, self._sink_key_embeddings
      )
    else:
      sink_key_logits = None

    logit_position_bias = None
    if (
        self.relative_position_embedding
        and self.relative_position_embedding.supports_position_bias
    ):
      positions = jnp.arange(values_time)[jnp.newaxis]
      logit_position_bias = self.relative_position_embedding.get_position_bias(
          query_positions=positions,
          key_positions=positions,
          queries=queries.values,
          keys=keys.values,
      )

    # Mask out invalid timesteps in the input sequence so that we do not
    # attend to invalid timesteps. By shaping it [b, 1, 1, key_time], we
    # ensure that each query timestep cannot see invalid timesteps. If the
    # query timestep itself is invalid, it will be masked below
    valid_mask = x.mask[:, jnp.newaxis, jnp.newaxis, :]

    if (
        visibility_mask := _self_attention_layer_visibility_mask(
            self.config.max_past_horizon,
            self.config.max_future_horizon,
            values_time,
        )
    ) is not None:
      valid_mask = jnp.logical_and(visibility_mask, valid_mask)

    utils.assert_is_compatible_with(
        valid_mask.shape,
        [batch_size, 1, 1, values_time],
    )

    context_vectors, probabilities = _dot_product_attention(
        queries=queries.values,
        keys=keys.values,
        values=values.values,
        logit_visibility_mask=valid_mask,
        logit_bias=logit_position_bias,
        training=training,
        attention_logits_soft_cap=self.config.attention_logits_soft_cap,
        attention_probabilities_dropout=self._attention_probabilities_dropout,
        per_dim_scale=self._per_dim_scale,
        query_scale=self.config.query_scale,
        precision=self.config.precision,
        get_logits_fn=None,
        zero_fully_masked=self.config.zero_fully_masked,
        compute_dtype=compute_dtype,
        num_sink_embeddings=self.config.num_sink_embeddings,
        sink_key_logits=sink_key_logits,
        sink_value_embeddings=self._sink_value_embeddings
        if self.config.num_sink_embeddings > 0
        else None,
    )
    emits = SelfAttentionEmits(types.Sequence(probabilities, x.mask))

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, x.mask)

    return context_vectors, emits


class DotProductAttention(types.Emitting, AttentionInputProjectionHelper):
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
        QueryAndKeyValueProjection
        | SeparateQueryKeyValueProjection
        | QueryAndSharedKeyValueProjection
    ) = dataclasses.field(default_factory=QueryAndKeyValueProjection)
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
    relative_position_embedding: RelativePositionEmbedding.Config | None = None
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
    # Pax implementation:
    # google3/learning/multipod/pax/audio/lm/layers/streaming/attentions.py.
    # Paper: https://arxiv.org/pdf/2309.17453.pdf.
    num_sink_embeddings: int = 0
    # By default initialize the sink token embeddings to have a norm of 1.
    sink_embeddings_init: nn.initializers.Initializer = (
        nn.linear.default_embed_init
    )
    sink_embeddings_sharding: types.Sharding | None = None
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'DotProductAttention':
      return DotProductAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    _validate_attention(
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

    self.query_network = (
        self.config.query_network.make() if self.config.query_network else None
    )
    if self.query_network and (
        self.query_network.output_ratio != 1
        or self.query_network.block_size != 1
    ):
      raise ValueError(
          'Query network must have an output_ratio'
          f' ({self.query_network.output_ratio}) and block_size'
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
          f' ({self.key_network.output_ratio}) and block_size'
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
          f' ({self.value_network.output_ratio}) and block_size'
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

  def _get_source(self, constants: types.Constants | None) -> types.Sequence:
    return _get_source(
        self, self.config.source_name, constants, required_rank=3
    )

  def _get_query_positions_from_constants(
      self, constants: types.Constants | None
  ) -> types.Sequence:
    return _get_source(
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
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
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
    queries = self._q.project_sequence(x)

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
      # TODO(mvelimirovic): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', query.values, self._sink_key_embeddings
      )
      sink_value_embeddings = self._sink_value_embeddings
    else:
      sink_key_logits = None
      sink_value_embeddings = None

    context_vectors, probabilities = _dot_product_attention(
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
        num_sink_embeddings=self.config.num_sink_embeddings,
        sink_key_logits=sink_key_logits,
        sink_value_embeddings=sink_value_embeddings,
    )
    emits = CrossAttentionEmits(
        {self.config.source_name: types.Sequence(probabilities, query.mask)}
    )
    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, query.mask)
    return context_vectors, emits


class GmmAttention(types.PreservesType, types.Emitting):
  """A multi-headed Gaussian-mixture attention layer.

  Uses GMMs to model the probability distribution of where to focus attention
  in the source sequence.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for GmmAttention."""

    # The name of the source to attend to.
    source_name: str
    # The number of attention heads.
    num_heads: int
    # The number of units per head.
    units_per_head: int
    # The number of Gaussian mixture components.
    num_components: int
    # Whether to restrict the mean of each mixture component to
    # monotonically increase over time.
    monotonic: bool
    # Precision config to use for einsums.
    precision: nn.linear.PrecisionLike = None
    # Kernel initializer for the hidden layer.
    hidden_kernel_init: nn.initializers.Initializer | None = (
        nn.linear.default_kernel_init
    )
    # Sharding configuration for the hidden layer.
    hidden_kernel_sharding: types.Sharding | None = None
    # Kernel initializer for the output layer.
    output_kernel_init: nn.initializers.Initializer | None = (
        nn.linear.default_kernel_init
    )
    # Whether to learn a bias in the output layer.
    output_use_bias: bool = True
    # Bias initializer for the output layer.
    output_bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # If use_bias=True, sharding configuration for output layer bias.
    output_bias_sharding: types.Sharding | None = None
    # Sharding configuration for the output layer kernel.
    output_kernel_sharding: types.Sharding | None = None
    # Initial bias for the offset.
    init_offset_bias: float = 0.0
    # Initial bias for the scale.
    init_scale_bias: float = 0.0
    # Maximum offset.
    max_offset: float = -1.0
    # Name of the layer.
    name: str | None = None

    # TODO(b/330395665): compute and param dtype.

    def make(self) -> 'GmmAttention':
      return GmmAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    _validate_attention(
        self.config.source_name,
        self.config.num_heads,
        self.config.units_per_head,
        self.name,
    )
    if self.config.num_components <= 0:
      raise ValueError(
          f'Expected num_components > 0 for {self.name}. Got:'
          f' {self.config.num_components}.'
      )

    # c: channels, n: num_heads, u: units_per_head.
    self._mlp_hidden = utils.FlaxEinsumDense(
        '...c,cnu->...nu',
        output_shape=(self.config.num_heads, self.config.units_per_head),
        precision=self.config.precision,
        kernel_init=utils.shard_initializer(
            self.config.hidden_kernel_init,
            self.config.hidden_kernel_sharding,
            projectable=True,
            axes_types=(meta.AxisType.FANIN, None, None),
        ),
        activation=nn.relu,
        name='hidden',
    )
    # n: num_heads, u: units_per_head, c: num_components, l: num_logits (3).
    self._mlp_output = utils.FlaxEinsumDense(
        equation='...nu,nucl->...ncl',
        # Final axis holds 3 logits: prior_logits, offset_logits, scale_logits.
        output_shape=(self.config.num_heads, self.config.num_components, 3),
        precision=self.config.precision,
        bias_axes='ncl' if self.config.output_use_bias else None,
        kernel_init=utils.shard_initializer(
            self.config.output_kernel_init,
            self.config.output_kernel_sharding,
            projectable=True,
            axes_types=(None, meta.AxisType.FANIN, None, meta.AxisType.STACKED),
        ),
        bias_init=(
            utils.shard_initializer(
                self.config.output_bias_init,
                self.config.output_bias_sharding,
                projectable=False,
                axes_types=(None, None, None),
            )
            if self.config.output_use_bias
            else None
        ),
        bias_sharding=(
            self.config.output_bias_sharding
            if self.config.output_use_bias
            else None
        ),
        activation=None,
        name='output',
    )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    if self.config.monotonic:
      # Start from zero positions.
      return jnp.zeros(
          [batch_size, 1, self.config.num_heads, self.config.num_components],
          input_spec.dtype,
      )
    else:
      return ()

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'GmmAttention requires rank 3 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    source = _get_source(self, self.config.source_name, constants)
    # Unlike other multi-headed attention classes in this file,
    # get_output_shape()[1] is the number of source channels (rather than
    # units_per_head) because GmmAttention does not perform a values projection.
    # (We may try adding this).
    return (self.config.num_heads, source.shape[2])

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    position = self.get_initial_state(
        x.shape[0], x.channel_spec, training=training, constants=constants
    )

    if self.config.monotonic and x.shape[1] != 1:
      context_vector, _, attention_weights = utils.step_by_step_dynamic(
          l=self,
          x=x,
          training=training,
          initial_state=position,
          blocks_per_step=1,
          constants=constants,
          with_emits=True,
      )
      return context_vector, attention_weights

    source = _get_source(self, self.config.source_name, constants)
    # Compute in parallel, since either not monotonic, or num_timesteps=1.
    context_vector, _, attention_weights = self.attention(
        source=source,
        query=x,
        prev_position=position,
    )
    return context_vector, attention_weights

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    source = _get_source(self, self.config.source_name, constants)
    if self.config.monotonic and x.shape[1] != 1:
      # step_by_step_static return types already match those of this function:
      # Sequence(context_vector), State(new_position), Emits(attention_weights).
      return utils.step_by_step_static(
          l=self,
          x=x,
          training=training,
          initial_state=state,
          blocks_per_step=1,
          constants=constants,
          with_emits=True,
      )
    # Compute in parallel, since either not monotonic, or num_timesteps=1.
    context_vector, new_position, attention_weights = self.attention(
        source=source,
        query=x,
        prev_position=state,
    )

    return (
        context_vector,
        new_position,
        attention_weights,
    )

  def attention(
      self,
      source: types.Sequence,
      query: types.Sequence,
      prev_position: types.State,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    """Computes GMM attention from query and previous position.

    Args:
      source: The source sequence to attend to. A [batch, time, dim]
      query: [batch_size, query_time, query_channels]
      prev_position (state): [batch_size, 1, num_heads, num_components].
        Previous position of each mixture component per-head.

    Returns:
      context_vector: [batch_size, query_time, num_heads,
        source_dimension]. The per-head context vector.
      position: [batch_size, 1, num_heads, num_components]. Current
        position of each mixture component per head.
      attention_weights: [batch_size, query_time, num_heads, source_time].
        Per-head attention values representing the probability density at each
        source sequence position.
    """
    if query.values.ndim != 3:
      raise ValueError(f'Expected [b, 1, d] inputs, got: {query.values.shape}.')

    query_time = query.values.shape[1]
    if self.config.monotonic:
      assert (
          query_time == 1
      ), f'Expected [b, 1, d] inputs, got: {query.values.shape}.'
    query = self._mlp_hidden.project_sequence(query)
    assert query.values.ndim == 4, (
        f'After self._mlp_hidden(), {query.values.shape=}, but expected [b, 1,'
        ' num_heads, units_per_head]'
    )

    # Per-head projection from [b, 1, num_heads, units_per_head] to
    # [b, 1, num_heads, num_components, 3] (the 3 logits are unstacked below).
    query = self._mlp_output.project_sequence(query)
    assert query.values.ndim == 5, (
        f'After self._mlp_output(), {query.values.shape=}, but expected [b, 1,'
        ' num_heads, num_components, 3]'
    )

    # Each is [batch_size, query_time, num_heads, num_components].
    prior_logits, offset_logits, scale_logits = utils.unstack(
        query.values, axis=4
    )

    offset_logits += self.config.init_offset_bias
    scale_logits += self.config.init_scale_bias
    assert prior_logits.shape[-1] == self.config.num_components

    # attention_weights: [b, query_time, num_heads, source_time]
    # new_position: [b, q, num_heads, num_components]
    attention_weights, new_position = self._eval_gmm_pdfs(
        source, prior_logits, offset_logits, scale_logits, prev_position
    )

    # Expand source mask for broadcasting to
    # [b, query_time, num_heads, source_time].
    attention_weights = jnp.where(
        source.mask[:, jnp.newaxis, jnp.newaxis, :],
        attention_weights,
        jnp.zeros_like(attention_weights),
    )

    # [b, query_time, num_heads, source_time]
    # [b, source_time, source_dim]
    # -> [b, query_time, num_heads, source_dim]
    context_vector = jnp.einsum(
        'BiNj,BjS->BiNS',
        attention_weights,
        source.values,
        precision=self.config.precision,
    )

    # Make sure we don't update state if not monotonic (we should override the
    # return value of self.attention() to make state be ()).
    state = new_position if self.config.monotonic else ()
    # Convert types for return value.
    assert isinstance(attention_weights, jax.Array)
    attention_weights = CrossAttentionEmits(
        {self.config.source_name: types.Sequence(attention_weights, query.mask)}
    )
    # Returns (output, state, emits)
    return (
        types.Sequence(context_vector, query.mask),
        state,
        attention_weights,
    )

  def _eval_gmm_pdfs(
      self,
      source: types.Sequence,
      prior_logits: jax.Array,
      offset_logits: jax.Array,
      scale_logits: jax.Array,
      prev_position: types.State,
      normalize=True,
  ) -> tuple[jax.Array, jax.Array]:
    """Evaluate the location GMMs on all encoder positions.

    * Uses softmax for the mixture weights.
    * Uses softplus for means and scales.
    * Scale represents the standard deviation.

    Args:
      source: The source sequence to attend to.
      prior_logits: [b, query_time, num_heads, num_components].
      offset_logits: [b, query_time, num_heads, num_components].
      scale_logits: [b, query_time, num_heads, num_components].
      prev_position: [b, 1, num_heads, num_components].
      normalize: Whether to normalize the attention probabilities.

    Returns:
      attention_weights: Shaped [b, query_time, num_heads, source_time],
        probability densities for each attention head over the source sequence.
      new_position: [b, query_time, num_heads, num_components].
    """
    priors = utils.run_in_at_least_fp32(jax.nn.softmax)(prior_logits)
    variances = jnp.square(jax.nn.softplus(scale_logits))
    if self.config.max_offset > 0:
      # softplus(x) - softplus(x - M) gives a sigmoid that saturates outside
      # [0, M] and is approximately linear in between.
      position_offset = jax.nn.softplus(offset_logits) - jax.nn.softplus(
          offset_logits - self.config.max_offset
      )
    else:
      position_offset = jax.nn.softplus(offset_logits)

    # priors, position_offset, variances are shaped
    # [b, query_time, num_heads, num_components].
    utils.assert_is_compatible_with(
        priors.shape,
        [None, None, self.config.num_heads, self.config.num_components],
    )
    utils.assert_is_compatible_with(
        position_offset.shape,
        [None, None, self.config.num_heads, self.config.num_components],
    )
    utils.assert_is_compatible_with(
        variances.shape,
        [None, None, self.config.num_heads, self.config.num_components],
    )
    # prev_position is [b, 1, num_heads, num_components].
    if self.config.monotonic:
      utils.assert_is_compatible_with(
          prev_position.shape,
          [None, 1, self.config.num_heads, self.config.num_components],
      )
      # If we're monotonic, then query time is always 1.
      utils.assert_is_compatible_with(
          position_offset.shape,
          [None, 1, self.config.num_heads, self.config.num_components],
      )
      new_position = prev_position + position_offset
      utils.assert_is_compatible_with(
          new_position.shape,
          [None, 1, self.config.num_heads, self.config.num_components],
      )
    else:
      new_position = position_offset

    # Expand all to [b, query_time, num_heads, source_time (1), num_components]
    priors = priors[:, :, :, jnp.newaxis, :]
    means = new_position[:, :, :, jnp.newaxis, :]
    variances = variances[:, :, :, jnp.newaxis, :]

    # [1, 1, 1, source_timesteps, 1].
    source_length = source.values.shape[1]
    encoder_positions = jnp.asarray(
        jnp.arange(source_length), dtype=means.dtype
    )
    encoder_positions = encoder_positions[
        jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis
    ]

    if normalize:
      priors *= jnp.sqrt(2 * np.pi * variances + 1e-8)
    # Broadcast source time and query time.
    # encoder_positions is [1, 1, 1, source_time, 1]
    # means/priors/variances are
    # [batch, query_time, num_heads, 1, num_components]
    probabilities = priors * jnp.exp(
        -((encoder_positions - means) ** 2) / (2 * variances + 1e-9)
    )
    # probabilities shaped [batch, query_time, num_heads, source_time].
    probabilities = jnp.sum(probabilities, 4)
    return probabilities, new_position


def _convert_to_block(
    x: jax.Array, block_size: int, padding_val: float = 0.0
) -> jax.Array:
  """Turns a sequence to non overlapping blocks.

  Args:
    x: a tensor of [batch, time, ...].
    block_size: int. Number of time frames in a block.
    padding_val: float. value on the padded frames.

  Returns:
    A tensor of [batch, num_blocks, block_size, ...], with necessary paddings,
    where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
  """
  shape = x.shape
  b, t = shape[0], shape[1]
  if block_size < 1:
    raise ValueError(f'{block_size=} must be at least 1.')
  # Pad it to be a multiple of w.
  num_blocks = (t + block_size - 1) // block_size
  pad_length = num_blocks * block_size - t

  if pad_length > 0:
    paddings = [[0, 0]] * len(shape)
    paddings[1] = [0, pad_length]
    x = jnp.pad(x, paddings, constant_values=jnp.array(padding_val, x.dtype))
  reshaped = jnp.reshape(x, (b, num_blocks, block_size) + shape[2:])
  return reshaped


def _extract_block_context(
    x: jax.Array,
    block_size: int,
    left_context: int,
    right_context: int,
    padding_val: float | jnp.bool_ = 0.0,
) -> jax.Array:
  """Extracts temporal context for every block.

  Args:
    x: a tensor of [batch, time, ...].
    block_size: int. Number of time frames in a block.
    left_context: int. Left context size.
    right_context: int. Right context size.
    padding_val: float. value on the padded frames.

  Returns:
    A tensor of [batch, num_blocks, context_size, ...], with necessary paddings,
    where context_size = block_size + left_context + right_context,
    and output[:, i, ...] are x[:, start-left_context:end+right_context, ...],
    start = i * block_size, end = (i + 1) * block_size.
  """
  if block_size < 1:
    raise ValueError(f'{block_size=} must be at least 1.')
  if left_context < 0:
    raise ValueError(f'{left_context=} must be >= 0.')
  if right_context < 0:
    raise ValueError(f'{right_context=} must be >= 0.')

  # Pad outside of signal.frame so that we get the desired left/right context
  # and padding behavior.
  paddings = [(0, 0)] * len(x.shape)
  paddings[1] = (left_context, right_context + block_size - 1)
  x = jnp.pad(x, paddings, constant_values=jnp.array(padding_val, x.dtype))

  return signal.frame(
      x,
      frame_length=block_size + left_context + right_context,
      frame_step=block_size,
      axis=1,
      pad_mode=types.PaddingMode.VALID.value,
      pad_value=padding_val,
  )


@jt.typed
def _scale_query(
    queries: jt.Float[jt.ArrayT, 'b q n h'],
    per_dim_scale: jt.Float[jt.ArrayT, 'h'] | None,
    query_scale: float | jt.ScalarFloat | None,
) -> jt.Float[jt.ArrayT, 'b q n h']:
  """Scales queries with per_dim_scale or rsqrt(units_per_head)."""
  if query_scale is None:
    units_per_head = queries.shape[3]
    query_scale = 1 / np.sqrt(units_per_head)

  # Attention weights are computed as
  # softmax((queries * keys^T) / sqrt(units_per_head)), but for efficiency we
  # scale the queries before the matmul with keys to get logits.
  if per_dim_scale is not None:
    # Compute initial scale factor so that when per_dim_scale is zero (at
    # initialization) the scale is query_scale.
    # 1.0/jax.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
    # can avoid unnecessary XLA op fusion mess on TPU.
    r_softplus_0 = 1.442695041
    scale = jnp.array(r_softplus_0 * query_scale, dtype=queries.dtype)
    queries *= scale * jax.nn.softplus(per_dim_scale.astype(queries.dtype))
  else:
    queries *= jnp.array(query_scale, queries.dtype)
  return queries


class LocalDotProductSelfAttention(
    types.Emitting, AttentionInputProjectionHelper
):
  """A multi-headed dot-product self attention layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for LocalDotProductSelfAttention."""

    # The number of attention heads.
    num_heads: int
    # The number of units per head.
    units_per_head: int

    block_size: int
    # The number of past timesteps each timestep can see. Must be non-negative.
    # 0: No past timesteps are visible.
    max_past_horizon: int
    # The number of future timesteps each timestep can see. Must be
    # non-negative.
    # 0: No future timesteps are visible.
    max_future_horizon: int = 0

    # An optional RelativePositionEmbedding to use to compute relative position
    # biases or logits.
    relative_position_embedding: RelativePositionEmbedding.Config | None = None
    # The dropout rate for the attention probabilities.
    attention_probabilities_dropout_rate: float = 0.0
    # Whether to broadcast the dropout across the query time dimension as is
    # done in T5.
    broadcast_dropout_across_queries: bool = False
    # Whether to learn a bias in the query/key/value projection.
    use_bias: bool = False
    # Configuration for the query, key and value input projection parameters.
    # If num_kv_heads is set, must not be CombinedQueryKeyValueProjection.
    # If shared_kv_projection is set, must be QueryAndSharedKeyValueProjection.
    input_projection: QueryKeyValueProjectionConfig = dataclasses.field(
        default_factory=CombinedQueryKeyValueProjection
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
    # Precision config to use for all operations.
    precision: nn.linear.PrecisionLike = None
    # Outputs all-zeros context vectors for queries which have nothing to attend
    # to (i.e. all possible keys are masked).
    zero_fully_masked: bool = False
    # The dtype of the layer's computations.
    compute_dtype: types.DType | None = None
    # The dtype of the layer's parameters.
    param_dtype: types.DType = jnp.float32
    # The number of sink embeddings to include in the key and value.
    # Pax implementation:
    # google3/learning/multipod/pax/audio/lm/layers/streaming/attentions.py.
    # Paper: https://arxiv.org/pdf/2309.17453.pdf.
    num_sink_embeddings: int = 0
    # By default initialize the sink token embeddings to have a norm of 1.
    sink_embeddings_init: nn.initializers.Initializer = (
        nn.linear.default_embed_init
    )
    sink_embeddings_sharding: types.Sharding | None = None
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

    def make(self) -> 'LocalDotProductSelfAttention':
      return LocalDotProductSelfAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    _validate_heads(
        self.config.num_heads, self.config.units_per_head, self.name
    )

    # TODO(b/394829779): Support GQA for LocalDotProductSelfAttention.
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
              num_kv_heads,
              self.config.units_per_head,
          ),
          self.config.param_dtype,
      )

    if self.config.use_kv_cache_ringbuffer:
      if self.config.num_sink_embeddings > 0:
        raise NotImplementedError(
            'Sink embeddings are not supported with use_kv_cache_ringbuffer.'
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
          f' ({self.query_network.output_ratio}) and block_size'
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
          f' ({self.key_network.output_ratio}) and block_size'
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
          f' ({self.value_network.output_ratio}) and block_size'
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
        self.config.relative_position_embedding.make()
        if self.config.relative_position_embedding is not None
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
    # Default to 0 even if the layer is not steppable.
    return (
        self.config.max_future_horizon
        if self.config.max_future_horizon >= 0
        else 0
    )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    compute_dtype = self.get_input_projection_output_dtype(
        self.config.input_projection, input_spec.dtype
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
    if max_future_horizon:
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

    time_step = jnp.zeros([batch_size], dtype=jnp.int32)

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

    return (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    )

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
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

    x_queries, x_keys, x_values = self.get_qkv(self.config.input_projection, x)

    # Our params might not match param_dtype, so delegate the compute_dtype to
    # the output of the QKV layer.
    compute_dtype = x_queries.dtype

    (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    ) = state
    kv_buffer_size = kv_buffer_keys.shape[1]

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
          constants=constants,
      )

    # Guard against NaN/Inf in values, since values are contracted when
    # computing context vectors.
    x_values = x_values.mask_invalid()

    # The key and value network could have changed the mask, so we combine the
    # keys and values mask. This is inexpensive but would be nice to skip.
    combined_mask = utils.combine_mask(x_keys.mask, x_values.mask)

    if self.config.use_kv_cache_ringbuffer:
      # Leave the input key/values and KV cache separate for multi key/value dot
      # product attention.
      kv_buffer_time = kv_buffer_size
    else:
      # To process a step, concatenate our kv_buffer_size KV buffer with the
      # input x_values_time timesteps.
      kv_buffer_keys = jnp.concatenate([kv_buffer_keys, x_keys.values], axis=1)
      kv_buffer_values = jnp.concatenate(
          [kv_buffer_values, x_values.values], axis=1
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
          [query_delay_buffer, x_queries]
      )
      assert (
          query_delay_buffer.shape[1]
          == x_values_time + self.config.max_future_horizon
      )

      # Use the oldest x_values_time queries as the current step's queries. They
      # each have max_future_horizon context available in
      # kv_buffer_keys/kv_buffer_values so we can produce valid output for them.
      x_queries = query_delay_buffer[:, :x_values_time]

      # Preserve the last max_future_horizon queries for the next step.
      query_delay_buffer = query_delay_buffer[
          :, -self.config.max_future_horizon :
      ]

    if self.config.num_sink_embeddings > 0:
      # TODO(mvelimirovic): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', x_queries.values, self._sink_key_embeddings
      )
    else:
      sink_key_logits = None

    valid_mask = kv_buffer_mask[:, jnp.newaxis, jnp.newaxis, :]
    if (
        visibility_mask := _self_attention_step_visibility_mask(
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

      context_vectors, probabilities = _multi_key_value_dot_product_attention(
          queries=x_queries.values,
          kv_buffers=(
              (kv_buffer_keys, kv_buffer_values, valid_mask),
              (
                  x_keys.values,
                  x_values.values,
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
      probabilities = jnp.concatenate(probabilities, axis=-1)
    else:
      context_vectors, probabilities = _dot_product_attention(
          queries=x_queries.values,
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
          num_sink_embeddings=self.config.num_sink_embeddings,
          sink_key_logits=sink_key_logits,
          sink_value_embeddings=self._sink_value_embeddings
          if self.config.num_sink_embeddings > 0
          else None,
      )

    # Update KV caches and state.
    if self.config.use_kv_cache_ringbuffer:
      # Write latest keys, values and masks to the appropriate position in the
      # KV cache ring buffers.
      i = time_step % kv_buffer_size
      assert combined_mask.shape[1] == x_keys.shape[1]
      assert x_values.shape[1] == x_keys.shape[1]

      update_fn = jax.vmap(lambda x, u, i: x.at[i].set(u.squeeze(0)))
      kv_buffer_keys = update_fn(kv_buffer_keys, x_keys.values, i)
      kv_buffer_values = update_fn(kv_buffer_values, x_values.values, i)
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
        time_step + x_values_time,
        query_network_state,
        key_network_state,
        value_network_state,
        query_delay_buffer,
    )

    emits = SelfAttentionEmits(types.Sequence(probabilities, x_queries.mask))

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, x_queries.mask)

    return context_vectors, state, emits

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    queries, keys, values = self.get_qkv(self.config.input_projection, x)

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
          values,
          training=training,
          constants=constants,
      )

    if self.config.num_sink_embeddings > 0:
      # TODO(mvelimirovic): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', queries.values, self._sink_key_embeddings
      )
    else:
      sink_key_logits = None

    # Guard against NaN/Inf in values, since values are contracted when
    # computing context vectors.
    values = values.mask_invalid()

    get_logits_fn = None
    if (
        self.relative_position_embedding
        and self.relative_position_embedding.supports_get_logits
    ):
      get_logits_fn = self.relative_position_embedding.get_logits

    context_vectors, probabilities = _local_dot_product_attention(
        queries=queries.values,
        keys=keys.values,
        keys_mask=keys.mask,
        values=values.values,
        block_size=self.config.block_size,
        max_past_horizon=self.config.max_past_horizon,
        max_future_horizon=self.config.max_future_horizon,
        training=training,
        attention_logits_soft_cap=self.config.attention_logits_soft_cap,
        attention_probabilities_dropout=self._attention_probabilities_dropout,
        precision=self.config.precision,
        per_dim_scale=self._per_dim_scale,
        query_scale=self.config.query_scale,
        get_logits_fn=get_logits_fn,
        zero_fully_masked=self.config.zero_fully_masked,
        compute_dtype=compute_dtype,
        num_sink_embeddings=self.config.num_sink_embeddings,
        sink_key_logits=sink_key_logits,
        sink_value_embeddings=self._sink_value_embeddings
        if self.config.num_sink_embeddings > 0
        else None,
    )
    emits = SelfAttentionEmits(types.Sequence(probabilities, x.mask))

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, x.mask)

    return context_vectors, emits


class StreamingLocalDotProductAttention(
    types.Emitting, AttentionInputProjectionHelper
):
  """A multi-headed streaming local dot-product attention layer.

  Unlike most SequenceLayers, this cross-attention layer assumes that when using
  the step-wise APIs, the source sequence provided in the constants dictionary
  is provided in a streaming fashion with the same block size as the input to
  the step API. The layer-wise API functions like DotProductAttention, and
  expects the entire source sequence to be provided at once.

  TODO(rryan): Support different source/input ratios to enable strided
  attention to the source.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for StreamingLocalDotProductAttention."""

    # The key to lookup source sequence from constants dictionary.
    source_name: str

    # The number of attention heads.
    num_heads: int
    # The number of units per head.
    units_per_head: int

    block_size: int
    # The number of past timesteps each timestep can see. Must be non-negative.
    # 0: No past timesteps are visible.
    max_past_horizon: int
    # The number of future timesteps each timestep can see. Must be
    # non-negative.
    # 0: No future timesteps are visible.
    max_future_horizon: int = 0
    # An optional RelativePositionEmbedding to use to compute relative position
    # biases or logits.
    relative_position_embedding: RelativePositionEmbedding.Config | None = None
    # The dropout rate for the attention probabilities.
    attention_probabilities_dropout_rate: float = 0.0
    # Whether to broadcast the dropout across the query time dimension as is
    # done in T5.
    broadcast_dropout_across_queries: bool = False
    # Whether to learn a bias in the query/key/value projection.
    use_bias: bool = False
    # Configuration for the query, key and value input projections.
    input_projection: (
        QueryAndKeyValueProjection
        | SeparateQueryKeyValueProjection
        | QueryAndSharedKeyValueProjection
    ) = dataclasses.field(default_factory=QueryAndKeyValueProjection)
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
    # Pax implementation:
    # google3/learning/multipod/pax/audio/lm/layers/streaming/attentions.py.
    # Paper: https://arxiv.org/pdf/2309.17453.pdf.
    num_sink_embeddings: int = 0
    # By default initialize the sink token embeddings to have a norm of 1.
    sink_embeddings_init: nn.initializers.Initializer = (
        nn.linear.default_embed_init
    )
    sink_embeddings_sharding: types.Sharding | None = None
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

    def make(self) -> 'StreamingLocalDotProductAttention':
      return StreamingLocalDotProductAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    _validate_heads(
        self.config.num_heads, self.config.units_per_head, self.name
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
    # TODO(b/394829779): Support GQA for StreamingLocalDotProductAttention.
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

    if self.config.use_kv_cache_ringbuffer:
      if self.config.num_sink_embeddings > 0:
        raise NotImplementedError(
            'Sink embeddings are not supported with use_kv_cache_ringbuffer.'
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
          f' ({self.query_network.output_ratio}) and block_size'
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
          f' ({self.key_network.output_ratio}) and block_size'
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
          f' ({self.value_network.output_ratio}) and block_size'
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

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    compute_dtype = self._q.get_output_dtype(input_spec.dtype)
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
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
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

    source = _get_source(self, self.config.source_name, constants)

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
      # TODO(mvelimirovic): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', queries.values, self._sink_key_embeddings
      )
    else:
      sink_key_logits = None

    valid_mask = kv_buffer_mask[:, jnp.newaxis, jnp.newaxis, :]
    if (
        visibility_mask := _self_attention_step_visibility_mask(
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
      assert not get_logits_fn

      context_vectors, probabilities = _multi_key_value_dot_product_attention(
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
      probabilities = jnp.concatenate(probabilities, axis=-1)
    else:
      context_vectors, probabilities = _dot_product_attention(
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
          num_sink_embeddings=self.config.num_sink_embeddings,
          sink_key_logits=sink_key_logits,
          sink_value_embeddings=self._sink_value_embeddings
          if self.config.num_sink_embeddings > 0
          else None,
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

    emits = CrossAttentionEmits(
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
    source = _get_source(self, self.config.source_name, constants)

    # No mask required, since query timesteps are independent.
    queries = self.get_q(self.config.input_projection, x)

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
      # TODO(mvelimirovic): Maybe compute this before query network.
      sink_key_logits = jnp.einsum(
          'BTNH,KNH->BNTK', queries.values, self._sink_key_embeddings
      )
    else:
      sink_key_logits = None

    get_logits_fn = None
    if (
        self.relative_position_embedding
        and self.relative_position_embedding.supports_get_logits
    ):
      get_logits_fn = self.relative_position_embedding.get_logits

    context_vectors, probabilities = _local_dot_product_attention(
        queries=queries.values,
        keys=keys.values,
        # The mask could differ based on key_network / value_network.
        keys_mask=utils.combine_mask(keys.mask, values.mask),
        values=values.values,
        block_size=self.config.block_size,
        max_past_horizon=self.config.max_past_horizon,
        max_future_horizon=self.config.max_future_horizon,
        training=training,
        attention_logits_soft_cap=self.config.attention_logits_soft_cap,
        attention_probabilities_dropout=self._attention_probabilities_dropout,
        precision=self.config.precision,
        per_dim_scale=self._per_dim_scale,
        query_scale=self.config.query_scale,
        get_logits_fn=get_logits_fn,
        zero_fully_masked=self.config.zero_fully_masked,
        compute_dtype=compute_dtype,
        num_sink_embeddings=self.config.num_sink_embeddings,
        sink_key_logits=sink_key_logits,
        sink_value_embeddings=self._sink_value_embeddings
        if self.config.num_sink_embeddings > 0
        else None,
    )
    emits = CrossAttentionEmits(
        {self.config.source_name: types.Sequence(probabilities, x.mask)}
    )

    # Context vectors contain invalid data in padding regions.
    context_vectors = types.Sequence(context_vectors, x.mask)

    return context_vectors, emits
