"""Specifications for simple layers.

See the corresponding _behaviors module for behaviors.
"""

# pylint: disable=abstract-method

import abc
import dataclasses
from typing import (Any, Callable, Generic, Protocol, runtime_checkable,
                    Sequence, TypeVar)

from sequence_layers.specs import types as types_spec
from sequence_layers.specs.types import HashableArray

# ---------------------------------------------------------------------------
# Activation Functions (StatelessPointwiseFunctor)
# ---------------------------------------------------------------------------


class Identity[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwise[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Identity layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Identity layer."""


class Relu[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Relu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Relu layer."""


class Gelu[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Gelu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Gelu layer."""


class Abs[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Abs layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Abs layer."""


class Exp[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Exp layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Exp layer."""


class Log[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Log layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Log layer."""


class Swish[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Swish layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Swish layer."""


class Tanh[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Tanh layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Tanh layer."""


class Sigmoid[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Sigmoid layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Sigmoid layer."""


class LeakyRelu[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for LeakyRelu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for LeakyRelu layer."""


class Elu[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Elu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Elu layer."""


class Softmax[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Softmax layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Softmax layer."""


class Softplus[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Softplus layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Softplus layer."""


# ---------------------------------------------------------------------------
# Simple Math and Pointwise (StatelessPointwise)
# ---------------------------------------------------------------------------


class Cast[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Cast layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Cast layer."""


class Scale[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwise[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Scale layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Scale layer."""


class Add[SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwise[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Add layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Add layer."""


class MaskInvalid[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwise[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for MaskInvalid layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for MaskInvalid layer."""


# ---------------------------------------------------------------------------
# Gating (Stateless)
# ---------------------------------------------------------------------------


T = TypeVar('T')


class GatedUnit[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for GatedUnit layer."""

  @dataclasses.dataclass(frozen=True)
  class Config[T](types_spec.SequenceLayerConfig):
    """Configuration for GatedUnit layer."""

    feature_activation: Callable[[T], T] | None
    gate_activation: Callable[[T], T] | None


class GatedLinearUnit[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](GatedUnit[SequenceT, ShapeDTypeT], metaclass=abc.ABCMeta):
  """Specification for GatedLinearUnit layer."""

  @dataclasses.dataclass(frozen=True)
  class Config[T](GatedUnit.Config[T]):
    """Configuration for GatedLinearUnit layer."""


class GatedTanhUnit[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](GatedUnit[SequenceT, ShapeDTypeT], metaclass=abc.ABCMeta):
  """Specification for GatedTanhUnit layer."""

  @dataclasses.dataclass(frozen=True)
  class Config[T](GatedUnit.Config[T]):
    """Configuration for GatedTanhUnit layer."""


# ---------------------------------------------------------------------------
# Shape Operations (Stateless)
# ---------------------------------------------------------------------------


class Flatten[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Flatten layer."""


class Reshape[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Reshape layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Reshape layer."""

    output_shape: Sequence[int]


class ExpandDims[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for ExpandDims layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for ExpandDims layer."""

    axis: int | Sequence[int]


class Squeeze[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Squeeze layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Squeeze layer."""

    axis: int | Sequence[int] | None


class Transpose[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Transpose layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Transpose layer."""

    axes: Sequence[int] | None


# ---------------------------------------------------------------------------
# Other Simple Layers
# ---------------------------------------------------------------------------


class OneHot[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for OneHot layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for OneHot layer."""

    depth: int


class Embedding[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Embedding layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Embedding layer."""

    dimension: int
    num_embeddings: int


class Dropout[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwise[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Dropout layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Dropout layer."""

    rate: float


class Downsample1D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Downsample1D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Downsample1D layer."""

    rate: int


class Upsample1D[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Upsample1D layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Upsample1D layer."""

    rate: int


class CheckpointName[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwiseFunctor[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for CheckpointName layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for CheckpointName layer."""

    checkpoint_name: str


class Lambda[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.Stateless[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Lambda layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Lambda layer."""

    fn: Callable[..., Any]


class Logging[
    SequenceT: types_spec.Sequence, ShapeDTypeT: types_spec.ChannelSpec
](
    types_spec.PreservesType[SequenceT, SequenceT, ShapeDTypeT],
    types_spec.StatelessPointwise[SequenceT, SequenceT, ShapeDTypeT],
    metaclass=abc.ABCMeta,
):
  """Specification for Logging layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types_spec.SequenceLayerConfig):
    """Configuration for Logging layer."""

    prefix: str


# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for simple layers module."""

  @property
  def Identity(self) -> type[Identity]:
    ...

  @property
  def Relu(self) -> type[Relu]:
    ...

  @property
  def Gelu(self) -> type[Gelu]:
    ...

  @property
  def Swish(self) -> type[Swish]:
    ...

  @property
  def Tanh(self) -> type[Tanh]:
    ...

  @property
  def Sigmoid(self) -> type[Sigmoid]:
    ...

  @property
  def LeakyRelu(self) -> type[LeakyRelu]:
    ...

  @property
  def Elu(self) -> type[Elu]:
    ...

  @property
  def Softmax(self) -> type[Softmax]:
    ...

  @property
  def Softplus(self) -> type[Softplus]:
    ...

  @property
  def Cast(self) -> type[Cast]:
    ...

  @property
  def Scale(self) -> type[Scale]:
    ...

  @property
  def Add(self) -> type[Add]:
    ...

  @property
  def MaskInvalid(self) -> type[MaskInvalid]:
    ...

  @property
  def GatedUnit(self) -> type[GatedUnit]:
    ...

  @property
  def GatedLinearUnit(self) -> type[GatedLinearUnit]:
    ...

  @property
  def GatedTanhUnit(self) -> type[GatedTanhUnit]:
    ...

  @property
  def Flatten(self) -> type[Flatten]:
    ...

  @property
  def Reshape(self) -> type[Reshape]:
    ...

  @property
  def ExpandDims(self) -> type[ExpandDims]:
    ...

  @property
  def Squeeze(self) -> type[Squeeze]:
    ...

  @property
  def Transpose(self) -> type[Transpose]:
    ...

  @property
  def OneHot(self) -> type[OneHot]:
    ...

  @property
  def Embedding(self) -> type[Embedding]:
    ...

  @property
  def Dropout(self) -> type[Dropout]:
    ...

  @property
  def Downsample1D(self) -> type[Downsample1D]:
    ...

  @property
  def Upsample1D(self) -> type[Upsample1D]:
    ...

  @property
  def CheckpointName(self) -> type[CheckpointName]:
    ...

  @property
  def Lambda(self) -> type[Lambda]:
    ...

  @property
  def Logging(self) -> type[Logging]:
    ...


__all__ = [
    name
    for name, attr in globals().items()
    if isinstance(attr, type) and not name.startswith('_')
]
