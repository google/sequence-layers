"""Abstract base classes for simple sequence layers."""

import abc
import dataclasses
from typing import Any, Callable
from sequence_layers.abstract import types

class Identity(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Relu(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Gelu(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Swish(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Tanh(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Sigmoid(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Softplus(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class LeakyRelu(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    negative_slope: float = 0.01
    name: str | None = None

class Elu(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    alpha: float = 1.0
    name: str | None = None

class Softmax(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    axis: int = -1
    name: str | None = None

class Cast(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    dtype: types.DType
    name: str | None = None

class Scale(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    scale: Any = 1.0
    name: str | None = None

class Add(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    shift: Any = 0.0
    name: str | None = None

class MaskInvalid(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    mask_value: Any | None = 0.0
    name: str | None = None

class GatedUnit(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    feature_activation: Callable | None = None
    gate_activation: Callable | None = None
    name: str | None = None

class GatedLinearUnit(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class GatedTanhUnit(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Flatten(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

class Reshape(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    output_shape: tuple[int, ...]
    name: str | None = None

class ExpandDims(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    axis: int | tuple[int, ...]
    name: str | None = None

class Squeeze(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    axis: int | tuple[int, ...] | None = None
    name: str | None = None

class Transpose(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    axes: tuple[int, ...] | None = None
    name: str | None = None

class OneHot(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    depth: int
    compute_dtype: types.DType | None = types.FLOAT32
    name: str | None = None

class Embedding(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    num_embeddings: int
    dimension: int
    param_dtype: types.DType = types.FLOAT32
    compute_dtype: types.DType | None = None
    name: str | None = None

class Dropout(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    rate: float = 0.0
    broadcast_dims: tuple[int, ...] = ()
    name: str | None = None

class Downsample1D(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    rate: int
    name: str | None = None

class Upsample1D(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    rate: int
    name: str | None = None

class CheckpointName(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    checkpoint_name: str = ''
    name: str | None = None

class Lambda(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    fn: Callable = None
    sequence_input: bool = False
    mask_required: bool = True
    expected_input_spec: Any = None
    expected_output_spec: Any = None
    name: str | None = None

class Logging(abc.ABC):
  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    prefix: str = ''
    dump_tensors: bool = False
    name: str | None = None
