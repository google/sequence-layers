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
"""Common attention logic."""

from collections.abc import Sequence as TypingSequence
import dataclasses
import functools
from typing import Any, Callable, Mapping, Protocol

from flax import linen as nn
from flax import struct
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import jaxtyping
import numpy as np
from sequence_layers.jax import meta
from sequence_layers.jax import signal
from sequence_layers.jax import simple
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils


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


def validate_heads(num_heads: int, units_per_head: int, name: str):
  if num_heads <= 0:
    raise ValueError(f'Expected num_heads > 0 for {name}. Got {num_heads}')
  if units_per_head <= 0:
    raise ValueError(
        f'Expected units_per_head > 0 for {name}. Got {units_per_head}'
    )


def validate_attention(
    source_name: str, num_heads: int, units_per_head: int, name: str
):
  if not source_name:
    raise ValueError(f'Expected non-empty source_name for {name}.')
  validate_heads(num_heads, units_per_head, name)


def get_source(
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
  quantization_provider: types.QuantizationProviderT | None = None


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
            quantization_provider=config.quantization_provider,
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
            quantization_provider=config.quantization_provider,
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
            quantization_provider=config.quantization_provider,
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
            quantization_provider=config.quantization_provider,
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
            quantization_provider=config.quantization_provider,
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
            quantization_provider=config.quantization_provider,
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
            quantization_provider=config.quantization_provider,
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
            quantization_provider=config.quantization_provider,
            name='shared_key_value_projection',
        )

  def get_input_projection_output_dtype(
      self,
      config: QueryKeyValueProjectionConfig,
      input_dtype: types.DType,
      constants: types.Constants | None = None,
  ) -> types.DType:
    """Returns the output dtype of the QKV projection."""
    match config:
      case CombinedQueryKeyValueProjection():
        return self._qkv.get_output_dtype(input_dtype, constants=constants)
      case (
          SeparateQueryKeyValueProjection()
          | QueryAndKeyValueProjection()
          | QueryAndSharedKeyValueProjection()
      ):
        return self._q.get_output_dtype(input_dtype, constants=constants)
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


@jt.typed
def dot_product_attention(
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
    num_sink_positions: int,
    sink_key_logits: jt.Float[jt.ArrayT, 'b nq q s'] | None,
    sink_value_embeddings: jt.Float[jt.ArrayT, 's nk h'] | None,
) -> tuple[
    jt.Float[jt.ArrayT, 'b q nq h'],
    jt.Float[jt.ArrayT, 'b q nq k+{num_sink_positions}'],
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
    num_sink_positions: The number of sink positions (only needed for
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

  if zero_fully_masked and sink_key_logits is not None:
    raise ValueError('zero_fully_masked and sink_key_logits are incompatible.')

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
        num_sink_positions,
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
def multi_key_value_dot_product_attention(
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
    num_sink_positions: int,
    sink_key_logits: jt.Float[jt.ArrayT, 'b nq q s'] | None,
    sink_value_embeddings: jt.Float[jt.ArrayT, 's nk h'] | None,
) -> tuple[
    jt.Float[jt.ArrayT, 'b q nq h'],
    jt.Float[jt.ArrayT, 'b q nq k+{num_sink_positions}'],
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
    num_sink_positions: The number of sink positions (only needed for
      jaxtyping).
    sink_key_logits: Logits used to sink the attention [batch, num_query_heads,
      query_time, num_sink_positions].
    sink_value_embeddings: Value embeddings corresponding to the sink keys
      [num_sink_positions, num_key_value_heads, units_per_head].

  Returns:
    context_vectors: A [batch_size, query_time, num_query_heads, units_per_head]
      tensor of context vectors for the queries.
    probabilities: A [batch_size, query_time, num_query_heads, keys_time +
      num_sink_positions] tensor of attention probabilities (for debugging).
  """
  del num_sink_positions
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


@struct.dataclass
class QBundle:
  """A bundle of query arrays used for dot product attention."""

  # The queries.
  queries: jt.Float[jt.ArrayT, 'b t n h']
  # The segment ids associated with the queries.
  segment_ids: jt.Float[jt.ArrayT, '#b t'] | None
  # The position of the queries.
  position: jt.Float[jt.ArrayT, '#b t'] | None
  # Optional. The validity mask of the query.
  mask: jt.Float[jt.ArrayT, '#b t'] | None


@struct.dataclass
class KVBundle:
  """A bundle of key and value arrays used for dot product attention."""

  # The keys data array.
  keys: jt.Float[jt.ArrayT, 'b t n h']
  # The values data array.
  values: jt.Float[jt.ArrayT, 'b t n h']
  # The segment ids associated with the keys / values.
  segment_ids: jt.Float[jt.ArrayT, '#b t'] | None
  # The position of the keys / values.
  position: jt.Float[jt.ArrayT, '#b t'] | None
  # The validity mask of the keys / values. Can be None for key/values that
  # don't need masking.
  mask: jt.Float[jt.ArrayT, '#b t'] | None


@struct.dataclass
class _OnlineSoftmaxWeightedSumState:
  """State for an online softmax weighted sum."""

  # TODO(b/394101048): Generalize this to a pytree of parallel weighted sums.
  # [b, q_t, num_query_heads_per_kv_head, num_kv_heads, units_per_head]
  numerator: jax.Array
  # [b, q_t, num_query_heads_per_kv_head, num_kv_heads, 1]
  denominator: jax.Array
  # [b, q_t, num_query_heads_per_kv_head, num_kv_heads, 1]
  max_so_far: jax.Array

  def to_context(self) -> jax.Array:
    return utils.divide_no_nan(self.numerator, self.denominator)


def _online_softmax_weighted_sum_initial_state(
    batch_shape: tuple[int, ...],
    h: int,
    dtype: types.DType,
) -> _OnlineSoftmaxWeightedSumState:
  return _OnlineSoftmaxWeightedSumState(
      numerator=jnp.zeros((*batch_shape, h), dtype),
      denominator=jnp.zeros((*batch_shape, 1), dtype),
      max_so_far=jnp.full(
          (*batch_shape, 1), jnp.array(_INVALID_LOGIT_VALUE, dtype)
      ),
  )


def _online_softmax_weighted_sum_step(
    logits_fn: Callable[[QBundle, KVBundle], tuple[jax.Array, jax.Array]],
    state: _OnlineSoftmaxWeightedSumState,
    query_chunk: QBundle,
    kv_chunk: KVBundle,
    precision: nn.linear.PrecisionLike,
) -> _OnlineSoftmaxWeightedSumState:
  """Computes one step of an online softmax weighted sum."""
  b, q, nq, h = query_chunk.queries.shape
  _, k, nk, _ = kv_chunk.keys.shape

  assert nq % nk == 0
  nqpk = nq // nk

  assert kv_chunk.keys.shape == (b, k, nk, h)
  assert kv_chunk.values.shape == (b, k, nk, h)
  assert state.numerator.shape == (b, q, nqpk, nk, h)
  assert state.denominator.shape == (b, q, nqpk, nk, 1)
  assert state.max_so_far.shape == (b, q, nqpk, nk, 1)

  logits_chunk, any_valid = logits_fn(query_chunk, kv_chunk)
  if logits_chunk.shape != (b, q, nqpk, nk, k):
    raise ValueError(
        f'Expected {logits_chunk.shape=} to be {(b, q, nqpk, nk, k)=} but got'
        f' {logits_chunk.shape=}.'
    )
  if any_valid.ndim != 4:
    raise ValueError(f'Expected {any_valid.shape=} to be {(b, q, nqpk, nk)=}.')

  # Maximum logit for each (batch/head/query) slice.
  chunk_max = logits_chunk.max(axis=-1, keepdims=True)
  assert chunk_max.shape == (b, q, nqpk, nk, 1)

  max_so_far = jnp.maximum(state.max_so_far, chunk_max)
  assert max_so_far.shape == (b, q, nqpk, nk, 1)
  max_so_far = jax.lax.stop_gradient(max_so_far)

  correction = jnp.exp(state.max_so_far - max_so_far)
  assert correction.shape == (b, q, nqpk, nk, 1)

  corrected_weights = jnp.exp(logits_chunk - max_so_far)
  assert corrected_weights.shape == (b, q, nqpk, nk, k)

  # Apply correction to numerator. Broadcast across head dim.
  numerator = state.numerator * correction
  assert numerator.shape == (b, q, nqpk, nk, h)

  # Clear invalid locations in values to make sure corrected_weights does not
  # leak invalid values into the numerator.
  if kv_chunk.mask is not None:
    values = jnp.where(
        kv_chunk.mask[:, :, jnp.newaxis, jnp.newaxis],
        kv_chunk.values,
        0,
    )
  else:
    values = kv_chunk.values

  # Compute weighted values and add them to the numerator.
  numerator += jnp.einsum(
      'BiQNj,BjNH->BiQNH', corrected_weights, values, precision=precision
  )
  assert numerator.shape == (b, q, nqpk, nk, h)

  # Apply correction to denominator.
  denominator = state.denominator * correction
  # Add sum of all weights for keys to the denominator.
  # Invalid locations are zero because e^(-inf - max) = 0.
  denominator += corrected_weights.sum(axis=-1, keepdims=True)
  assert denominator.shape == (b, q, nqpk, nk, 1)

  new_state = _OnlineSoftmaxWeightedSumState(
      numerator=numerator,
      denominator=denominator,
      max_so_far=max_so_far,
  )

  # Don't update state for (b, q, nqpk, nk) slices of state that have no valid
  # keys.
  any_valid = any_valid[:, :, :, :, jnp.newaxis]
  new_state = jax.tree.map(
      lambda a, b: jnp.where(any_valid, a, b), new_state, state
  )

  return new_state


class AttentionMaskFn(Protocol):

  def __call__(
      self, query: QBundle, key: KVBundle
  ) -> jt.Bool[jt.ArrayT, '#b #q #k'] | None:
    ...


@dataclasses.dataclass(frozen=True)
class SegmentMask:
  """Only allow attention within the same segment ID."""

  def __call__(
      self, query: QBundle, key: KVBundle
  ) -> jt.Bool[jt.ArrayT, '#b #q #k'] | None:
    if query.segment_ids is None or key.segment_ids is None:
      raise ValueError(
          'SegmentMask requires segment IDs:'
          f' {query.segment_ids=} {key.segment_ids=}'
      )
    return (
        query.segment_ids[:, :, jnp.newaxis]
        == key.segment_ids[:, jnp.newaxis, :]
    )


@dataclasses.dataclass(frozen=True)
class LocalCausalMask:
  """Only allow attention within a local causal window."""

  # The maximum number of previous timesteps each query can attend to.
  # If zero, no past timesteps are visible.
  # If None, the infinite past is visible.
  max_past_horizon: int | None
  # The maximum number of future timesteps each query can attend to.
  # If zero, no future timesteps are visible.
  # If None, the infinite future is visible.
  max_future_horizon: int | None

  def __post_init__(self):
    if self.max_past_horizon is not None and self.max_past_horizon < 0:
      raise ValueError(f'{self.max_past_horizon=} must be non-negative.')
    if self.max_future_horizon is not None and self.max_future_horizon < 0:
      raise ValueError(f'{self.max_future_horizon=} must be non-negative.')

  def __call__(
      self, query: QBundle, key: KVBundle
  ) -> jt.Bool[jt.ArrayT, '#b #q #k'] | None:

    if self.max_past_horizon is None and self.max_future_horizon is None:
      return None

    distance = (
        query.position[:, :, jnp.newaxis] - key.position[:, jnp.newaxis, :]
    )

    masks = []

    # Positive distance: query is ahead of key by distance timesteps.
    if self.max_past_horizon is not None:
      masks.append(distance <= self.max_past_horizon)

    # Negative distance: query is behind key by distance timesteps.
    if self.max_future_horizon is not None:
      masks.append(distance >= -self.max_future_horizon)

    return functools.reduce(jnp.logical_and, masks)


@dataclasses.dataclass(frozen=True)
class BlockwiseLocalCausalMask:
  """Allow bidirectional attention within blocks of a specified size."""

  # The size of each block.
  block_size: int
  # The number of past blocks the current block can attend to.
  # If None, all past blocks can be attended to.
  max_past_horizon_blocks: int | None
  # The number of future blocks the current block can attend to.
  # If None, all future blocks can be attended to.
  max_future_horizon_blocks: int | None

  def __post_init__(self):
    if self.block_size <= 0:
      raise ValueError(f'{self.block_size=} must be positive.')
    if (
        self.max_past_horizon_blocks is not None
        and self.max_past_horizon_blocks < 0
    ):
      raise ValueError(f'{self.max_past_horizon_blocks=} must be non-negative.')
    if (
        self.max_future_horizon_blocks is not None
        and self.max_future_horizon_blocks < 0
    ):
      raise ValueError(
          f'{self.max_future_horizon_blocks=} must be non-negative.'
      )

  def __call__(
      self, query: QBundle, key: KVBundle
  ) -> jt.Bool[jt.ArrayT, '#b #q #k'] | None:

    if (
        self.max_past_horizon_blocks is None
        and self.max_future_horizon_blocks is None
    ):
      return None

    query_blocks_ids = query.position // self.block_size
    key_blocks_ids = key.position // self.block_size

    distance = (
        query_blocks_ids[:, :, jnp.newaxis] - key_blocks_ids[:, jnp.newaxis, :]
    )

    masks = []

    # Positive distance: query is ahead of key by distance timesteps.
    if self.max_past_horizon_blocks is not None:
      masks.append(distance <= self.max_past_horizon_blocks)

    # Negative distance: query is behind key by distance timesteps.
    if self.max_future_horizon_blocks is not None:
      masks.append(distance >= -self.max_future_horizon_blocks)

    return functools.reduce(jnp.logical_and, masks)


@jt.typed
def multi_key_value_dot_product_flash_attention(
    *,
    queries: QBundle,
    query_block_size: int | None,
    kv_bundles: TypingSequence[KVBundle],
    kv_block_sizes: int | None | TypingSequence[int | None],
    attention_mask_fns: TypingSequence[AttentionMaskFn],
    attention_logits_soft_cap: float | None,
    per_dim_scale: jt.Float[jt.ArrayT, 'h'] | None,
    query_scale: float | jt.ScalarFloat | None,
    precision: nn.linear.PrecisionLike,
    compute_dtype: Any | types.DType | None,
    remat: bool = False,
) -> jt.Float[jt.ArrayT, 'b q nq h']:
  """Computes "multi key-value" dot product flash attention with queries.

  This is equivalent to _dot_product_attention but allows providing the keys and
  values to attend over as a list of multiple arrays, each key/value group
  having its own mask, positions, and segment IDs.

  Unlike _dot_product_attention, this routine uses an online softmax weighted
  sum to compute attention context vectors, which reduces peak memory usage and
  may result in faster execution times at the expense of reduced numerical
  stability due to computing the softmax online (logit outliers in each block
  can cause underflow or overflow in the accumulators).

  Args:
    queries: A [batch, query_time, num_query_heads, units_per_head] tensor of
      queries.
    query_block_size: The block size to use when splitting up the query time
      axis. If None, no query block splitting is performed.
    kv_bundles: A sequence of KVBundle objects containing keys and values to
      attend to, as well as their associated arrays (segment IDs, positions,
      mask). Key/values are shapes [batch, kv_time, num_kv_heads,
      units_per_head]. If num_kv_heads != num_query_heads, the grouped query
      attention (GQA) attention algorithm is used.
    kv_block_sizes: The block sizes to use when splitting up the key/value time
      axis. If a single int is provided, it is used for all key/value groups. If
      None is provided, no key/value block splitting is performed.
    attention_mask_fns: A sequence of functions that take a query and key bundle
      and return a boolean mask array of shape [batch, query_time, kv_time].
    attention_logits_soft_cap: If non-zero, a soft cap applied to attention
      logits to prevent outliers from dominating the softmax. Empirically, 50.0
      works well across a variety of tasks. Implemented as tanh(logits / cap) *
      cap.
    per_dim_scale: A [units_per_head] query scale factor for all query heads.
      Assuming per_dim_scale is zeros at initialization (or if not specified),
      the scaling applied to each head is query_scale.
    query_scale: A float query scale factor for all query heads. If None,
      queries are scaled by 1/sqrt(units_per_head).
    precision: Precision config for einsums.
    compute_dtype: The dtype to use for logit calculations and outputs. All
      online softmax weighted sum state operations are performed in float32
      regardless of this value for numerical stability.
    remat: Whether to rematerialize the logits function.

  Returns:
    context_vectors: A [batch_size, query_time, num_query_heads, units_per_head]
      array of context vectors for the queries.
  """
  b, q_time, nq, h = queries.queries.shape

  if not kv_bundles:
    raise ValueError('kv_buffers must be non-empty.')

  nks = {kv_buffer.keys.shape[2] for kv_buffer in kv_bundles}
  if len(nks) != 1:
    raise ValueError(
        f'Expected all key bundles to have the same number of heads, got: {nks}'
    )
  nk = nks.pop()

  if nq % nk != 0:
    raise ValueError(f'Expected {nq=} % {nk=} to be zero.')
  nqpk = nq // nk

  q_dtype = utils.get_promoted_dtype(queries.queries.dtype, dtype=compute_dtype)
  qk_dtype = utils.get_promoted_dtype(
      q_dtype,
      *(k.keys.dtype for k in kv_bundles),
      dtype=compute_dtype,
  )
  qkv_dtype = utils.get_promoted_dtype(
      qk_dtype, *(k.values.dtype for k in kv_bundles), dtype=compute_dtype
  )
  queries = dataclasses.replace(
      queries,
      queries=_scale_query(
          queries.queries.astype(q_dtype), per_dim_scale, query_scale
      ).astype(qk_dtype),
  )

  def pad_and_split_blocks(v, pad_amount, num_blocks, block_size):
    if pad_amount:
      paddings = [(0, 0)] * v.ndim
      paddings[1] = (0, pad_amount)
      v = jnp.pad(v, paddings, mode='constant')

    return utils.split_dimension(v, axis=1, shape=(num_blocks, block_size))

  def transpose_for_scan(v):
    return jnp.moveaxis(v, 1, 0)

  if query_block_size is None:
    num_q_blocks = 1
    q_pad = 0
  else:
    num_q_blocks = (q_time + query_block_size - 1) // query_block_size
    q_pad = num_q_blocks * query_block_size - q_time

  use_query_scan = num_q_blocks > 1
  if use_query_scan:
    queries = jax.tree.map(
        functools.partial(
            pad_and_split_blocks,
            pad_amount=q_pad,
            num_blocks=num_q_blocks,
            block_size=query_block_size,
        ),
        queries,
    )
    queries = jax.tree.map(transpose_for_scan, queries)

  def logits_fn(query: QBundle, key: KVBundle) -> tuple[jax.Array, jax.Array]:
    num_query_heads = query.queries.shape[2]
    num_key_heads = key.keys.shape[2]
    assert num_query_heads % num_key_heads == 0
    query_heads_per_kv_head = num_query_heads // num_key_heads

    query_data = utils.split_dimension(
        query.queries, axis=2, shape=(query_heads_per_kv_head, num_key_heads)
    )
    logits = jnp.einsum(
        'biqnh,bjnh->biqnj', query_data, key.keys, precision=precision
    )
    logits = logits.astype(jnp.float32)

    masks = []

    if key.mask is not None:
      masks.append(key.mask[:, jnp.newaxis, :])

    for mask_fn in attention_mask_fns:
      mask = mask_fn(query, key)
      if mask is not None:
        masks.append(mask)

    # Cap attention logits before masking.
    if attention_logits_soft_cap:
      logits = _soft_cap_attention_logits(logits, attention_logits_soft_cap)

    if masks:
      mask = functools.reduce(jnp.logical_and, masks)
      mask = mask[:, :, jnp.newaxis, jnp.newaxis, :]
      logits = jnp.where(mask, logits, _INVALID_LOGIT_VALUE)
      any_valid = jnp.any(mask, axis=-1)
    else:
      any_valid = jnp.ones((1, 1, 1, 1), jnp.bool_)

    return logits, any_valid

  if isinstance(kv_block_sizes, int) or kv_block_sizes is None:
    kv_block_sizes = [kv_block_sizes] * len(kv_bundles)
  elif len(kv_block_sizes) != len(kv_bundles):
    raise ValueError(
        f'Expected {len(kv_bundles)=}, key_blocks, got: {kv_block_sizes}'
    )

  # Always use float32 for online softmax state.
  if use_query_scan:
    state_batch_shape = (num_q_blocks, b, query_block_size, nqpk, nk)
  else:
    state_batch_shape = (b, q_time, nqpk, nk)
  state = _online_softmax_weighted_sum_initial_state(
      state_batch_shape, h, jnp.float32
  )

  for kv_buffer, key_block_size in zip(kv_bundles, kv_block_sizes, strict=True):
    # Cast to qk_dtype and qkv_dtype.
    kv_buffer = dataclasses.replace(
        kv_buffer,
        keys=kv_buffer.keys.astype(qk_dtype),
        values=kv_buffer.values.astype(qkv_dtype),
    )

    k_time = kv_buffer.keys.shape[1]

    if key_block_size is None:
      num_k_blocks = 1
    else:
      num_k_blocks = (k_time + key_block_size - 1) // key_block_size

    use_key_scan = num_k_blocks > 1
    if use_key_scan:
      k_pad = num_k_blocks * key_block_size - k_time
      kv_buffer = jax.tree.map(
          functools.partial(
              pad_and_split_blocks,
              pad_amount=k_pad,
              num_blocks=num_k_blocks,
              block_size=key_block_size,
          ),
          kv_buffer,
      )
      kv_buffer = jax.tree.map(transpose_for_scan, kv_buffer)

    def q_body(
        carry, slice_bundle: tuple[_OnlineSoftmaxWeightedSumState, QBundle]
    ):
      del carry
      state_slice, q_slice = slice_bundle

      @functools.partial(jax.checkpoint, prevent_cse=False)
      def kv_scan_fn(state: _OnlineSoftmaxWeightedSumState, kv_slice: KVBundle):
        state = _online_softmax_weighted_sum_step(
            logits_fn, state, q_slice, kv_slice, precision=precision
        )
        return state, ()

      if use_key_scan:  # pylint: disable=cell-var-from-loop
        state_slice, _ = jax.lax.scan(
            kv_scan_fn, state_slice, kv_buffer, length=num_k_blocks  # pylint: disable=cell-var-from-loop
        )
      else:
        state_slice, _ = kv_scan_fn(state_slice, kv_buffer)  # pylint: disable=cell-var-from-loop

      return (), state_slice

    if remat:
      q_body = jax.checkpoint(q_body, prevent_cse=False)

    if use_query_scan:
      _, state = jax.lax.scan(q_body, (), (state, queries), length=num_q_blocks)
    else:
      _, state = q_body((), (state, queries))

  c = state.to_context().astype(qkv_dtype)

  if use_query_scan:
    assert c.shape == (num_q_blocks, b, query_block_size, nqpk, nk, h), c.shape
    c = jax.tree.map(transpose_for_scan, c)
    c = c.reshape((b, num_q_blocks * query_block_size, nq, h))
    # Strip off padded queries (if any).
    if q_pad:
      c = c[:, :-q_pad]
  else:
    assert c.shape == (b, q_time, nqpk, nk, h), c.shape
    c = c.reshape((b, q_time, nq, h))
  return c


@jt.typed
def local_dot_product_attention(
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
    num_sink_positions: int,
    sink_key_logits: jt.Float[jt.ArrayT, 'b nq q s'] | None,
    sink_value_embeddings: jt.Float[jt.ArrayT, 's nq h'] | None,
) -> tuple[
    jt.Float[jt.ArrayT, 'b q nq h'],
    jt.Float[
        jt.ArrayT,
        'b q nq'  # pylint: disable=implicit-str-concat
        ' {max_past_horizon+block_size+max_future_horizon+num_sink_positions}',
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
    num_sink_positions: The number of sink positions (only needed for
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
  del num_sink_positions
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


def self_attention_step_visibility_mask(
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


def self_attention_layer_visibility_mask(
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
