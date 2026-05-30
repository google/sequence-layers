"""Specifications for normalization layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
from typing import Any, Sequence, override

from sequence_layers.specs import types as types_spec


class L2Normalize[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for L2Normalize layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for L2Normalize."""

    axis: int | Sequence[int] = -1
    epsilon: float = 1e-12
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class RMSNormalization[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for RMSNormalization layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for RMSNormalization."""

    axis: int | Sequence[int] = -1
    epsilon: float = 1e-6
    use_scale: bool = True
    scale_init: Any | None = None
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class LayerNormalization[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for LayerNormalization layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for LayerNormalization."""

    axis: int | Sequence[int] = -1
    epsilon: float = 1e-6
    use_scale: bool = True
    use_bias: bool = True
    reductions_in_at_least_fp32: bool = True
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class BatchNormalization[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for BatchNormalization layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for BatchNormalization."""

    axis: int | Sequence[int] = -1
    epsilon: float = 1e-5
    momentum: float = 0.99
    use_scale: bool = True
    use_bias: bool = True
    use_fast_variance: bool = True
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class GroupNormalization[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for GroupNormalization layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for GroupNormalization."""

    num_groups: int
    axis: int | Sequence[int] = -1
    epsilon: float = 1e-6
    cumulative: bool = False
    use_scale: bool = True
    use_bias: bool = True
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""
