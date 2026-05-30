"""Specifications for attention layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
from typing import Any, Protocol, runtime_checkable

from sequence_layers.specs import types as types_spec

# =============================================================================
# Projection Config Specifications
# =============================================================================


@dataclasses.dataclass(frozen=True)
class QueryKeyValueProjectionConfig:
  """Base class for QKV projection configuration."""


@dataclasses.dataclass(frozen=True)
class CombinedQueryKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use a single projection matrix for query/key/value projection.

  * Incompatible with Grouped Query Attention (num_query_heads != num_kv_heads).
  * Supports shared key and value projection.
  """

  # If true, share the key and value projection matrices.
  share_kv_projection: bool = False


@dataclasses.dataclass(frozen=True)
class SeparateQueryKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate projection matrices for query/key/value projection.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Does not support shared key and value projection. Use
    QueryAndSharedKeyValueProjection.
  """


@dataclasses.dataclass(frozen=True)
class QueryAndKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate query and key/value projection matrices.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Does not support shared key and value projection. Use
    QueryAndSharedKeyValueProjection.
  """


@dataclasses.dataclass(frozen=True)
class QueryAndSharedKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate query and shared key/value projection matrices.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Requires shared key and value projection.
  """


# =============================================================================
# Attention Layer Specifications
# =============================================================================


class DotProductSelfAttention[
    SequenceT: types_spec.Sequence,
    ShapeDTypeT: types_spec.ChannelSpec,
](types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT], metaclass=abc.ABCMeta):
  """Specification for DotProductSelfAttention layer.

  Multi-headed dot-product self-attention with causal masking and KV caching.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for DotProductSelfAttention layer."""

    num_heads: int
    units_per_head: int
    max_past_horizon: int
    max_future_horizon: int = 0
    num_kv_heads: int | None = None
    attention_probabilities_dropout_rate: float = 0.0
    broadcast_dropout_across_queries: bool = False
    use_bias: bool = False
    input_projection: QueryKeyValueProjectionConfig = dataclasses.field(
        default_factory=CombinedQueryKeyValueProjection
    )
    query_network: types_spec.SequenceLayerConfig | None = None
    key_network: types_spec.SequenceLayerConfig | None = None
    value_network: types_spec.SequenceLayerConfig | None = None
    attention_logits_soft_cap: float | None = None
    per_dim_scale: bool = False
    query_scale: float | None = None
    zero_fully_masked: bool = False
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    num_sink_embeddings: int = 0
    use_sink_scalars: bool = False
    use_kv_cache_ringbuffer: bool = False
    name: str | None = None

    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class LocalDotProductSelfAttention[
    SequenceT: types_spec.Sequence,
    ShapeDTypeT: types_spec.ChannelSpec,
](DotProductSelfAttention[SequenceT, ShapeDTypeT], metaclass=abc.ABCMeta):
  """Specification for LocalDotProductSelfAttention layer.

  Local/block-based self-attention.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(DotProductSelfAttention.Config):
    """Configuration for LocalDotProductSelfAttention layer."""

    block_size: int = 1

    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class DotProductAttention[
    SequenceT: types_spec.Sequence,
    ShapeDTypeT: types_spec.ChannelSpec,
](types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT], metaclass=abc.ABCMeta):
  """Specification for DotProductAttention layer.

  Multi-headed cross-attention attending to an external source.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for DotProductAttention layer."""

    source_name: str
    num_heads: int
    units_per_head: int
    attention_probabilities_dropout_rate: float = 0.0
    broadcast_dropout_across_queries: bool = False
    use_bias: bool = False
    input_projection: QueryKeyValueProjectionConfig = dataclasses.field(
        default_factory=QueryAndKeyValueProjection
    )
    query_network: types_spec.SequenceLayerConfig | None = None
    key_network: types_spec.SequenceLayerConfig | None = None
    value_network: types_spec.SequenceLayerConfig | None = None
    attention_logits_soft_cap: float | None = None
    per_dim_scale: bool = False
    query_scale: float | None = None
    zero_fully_masked: bool = False
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    name: str | None = None

    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class StreamingDotProductAttention[
    SequenceT: types_spec.Sequence,
    ShapeDTypeT: types_spec.ChannelSpec,
](types_spec.Emitting[SequenceT, SequenceT, ShapeDTypeT], metaclass=abc.ABCMeta):
  """Specification for StreamingDotProductAttention layer.

  Streaming cross-attention with rolling KV buffer. Also covers
  StreamingLocalDotProductAttention (which differs only in layer-mode
  efficiency via block_size, not in step-mode behavior or output).
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for StreamingDotProductAttention layer."""

    source_name: str
    num_heads: int
    units_per_head: int
    max_past_horizon: int
    max_future_horizon: int = 0
    block_size: int = 1
    attention_probabilities_dropout_rate: float = 0.0
    broadcast_dropout_across_queries: bool = False
    use_bias: bool = False
    use_query_delay_buffer: bool = True
    input_projection: QueryKeyValueProjectionConfig = dataclasses.field(
        default_factory=QueryAndKeyValueProjection
    )
    query_network: types_spec.SequenceLayerConfig | None = None
    key_network: types_spec.SequenceLayerConfig | None = None
    value_network: types_spec.SequenceLayerConfig | None = None
    attention_logits_soft_cap: float | None = None
    per_dim_scale: bool = False
    query_scale: float | None = None
    zero_fully_masked: bool = False
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    num_sink_embeddings: int = 0
    use_sink_scalars: bool = False
    use_kv_cache_ringbuffer: bool = False
    name: str | None = None

    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


# =============================================================================
# ModuleSpec Protocol
# =============================================================================

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for the attention submodule of a backend."""

  @property
  def DotProductSelfAttention(self) -> type[DotProductSelfAttention]:
    ...

  @property
  def DotProductAttention(self) -> type[DotProductAttention]:
    ...

  @property
  def StreamingDotProductAttention(self) -> type[StreamingDotProductAttention]:
    ...

  @property
  def StreamingLocalDotProductAttention(
      self,
  ) -> type[StreamingDotProductAttention]:
    ...

  @property
  def LocalDotProductSelfAttention(
      self,
  ) -> type[LocalDotProductSelfAttention]:
    ...

  @property
  def CombinedQueryKeyValueProjection(
      self,
  ) -> type[CombinedQueryKeyValueProjection]:
    ...

  @property
  def SeparateQueryKeyValueProjection(
      self,
  ) -> type[SeparateQueryKeyValueProjection]:
    ...

  @property
  def QueryAndKeyValueProjection(self) -> type[QueryAndKeyValueProjection]:
    ...

  @property
  def QueryAndSharedKeyValueProjection(
      self,
  ) -> type[QueryAndSharedKeyValueProjection]:
    ...
