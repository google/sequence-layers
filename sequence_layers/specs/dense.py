"""Specifications for dense layers.

See the corresponding _behaviors module for behaviors.
"""

import abc
import dataclasses
from typing import Any, Callable, override, Sequence

from sequence_layers.specs import types as types_spec


class Dense[
    SequenceT: types_spec.Sequence = types_spec.Sequence,
    ShapeDTypeT: types_spec.ChannelSpec = types_spec.ChannelSpec,
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Dense layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Dense layer."""

    features: int
    use_bias: bool = True
    activation: Callable | None = None
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""


class EinsumDense[
    SequenceT: types_spec.Sequence = types_spec.Sequence,
    ShapeDTypeT: types_spec.ChannelSpec = types_spec.ChannelSpec,
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for EinsumDense layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for EinsumDense layer."""

    equation: str
    output_shape: Sequence[int | None]
    bias_axes: str = ''
    activation: Callable | None = None
    compute_dtype: types_spec.DType | None = None
    param_dtype: types_spec.DType | None = None
    name: str | None = None

    @override
    def make(self) -> Any:
      """Dummy make to satisfy Pyrefly."""
