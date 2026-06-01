"""Specifications for sequence layers."""

# https://typing.python.org/en/latest/spec/protocol.html#modules-as-implementations-of-protocols

from typing import Protocol, runtime_checkable, TYPE_CHECKING

from . import backend as _backend
from . import convolution as _convolution
from . import dense as _dense
from . import normalization as _normalization
from . import pooling as _pooling
from . import simple as _simple
from . import types as _types

# Import test_utils only for type checking to avoid circular imports,
# as test_utils.py imports specs.ModuleSpec defined below.
if TYPE_CHECKING:
  from . import test_utils as _test_utils


# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for a backend-specific SequenceLayers module (sequence_layers.<backend> as sl)."""

  @property
  def backend(self) -> _backend.ModuleSpec:
    ...

  @property
  def types(self) -> _types.ModuleSpec:
    ...

  @property
  def simple(self) -> _simple.ModuleSpec:
    ...

  @property
  def convolution(self) -> _convolution.ModuleSpec:
    ...

  @property
  def test_utils(self) -> '_test_utils.ModuleSpec':
    ...

  # Identifiers that backend-specific implementations should expose at top level.
  # Demonstrating read-only allows for covariance (subclasses of
  # types_module.Sequence to satisfy the protocol).

  @property
  def Sequence(self) -> type[_types.Sequence]:
    ...

  @property
  def MaskedSequence(self) -> type[_types.MaskedSequence]:
    ...

  @property
  def SequenceLayer(self) -> type[_types.SequenceLayer]:
    ...

  @property
  def SequenceLayerConfig(self) -> type[_types.SequenceLayerConfig]:
    ...

  @property
  def SequenceLayerTest(self) -> type:
    ...

  # Privileged layers appearing top-level
  @property
  def Flatten(self) -> type[_simple.Flatten]:
    ...

  @property
  def Reshape(self) -> type[_simple.Reshape]:
    ...

  @property
  def ExpandDims(self) -> type[_simple.ExpandDims]:
    ...

  @property
  def Squeeze(self) -> type[_simple.Squeeze]:
    ...

  @property
  def Scale(self) -> type[_simple.Scale]:
    ...

  @property
  def Add(self) -> type[_simple.Add]:
    ...

  @property
  def Cast(self) -> type[_simple.Cast]:
    ...

  @property
  def MaskInvalid(self) -> type[_simple.MaskInvalid]:
    ...

  @property
  def GatedUnit(self) -> type[_simple.GatedUnit]:
    ...

  @property
  def GatedLinearUnit(self) -> type[_simple.GatedLinearUnit]:
    ...

  @property
  def GatedTanhUnit(self) -> type[_simple.GatedTanhUnit]:
    ...

  @property
  def OneHot(self) -> type[_simple.OneHot]:
    ...

  @property
  def Embedding(self) -> type[_simple.Embedding]:
    ...

  @property
  def Softmax(self) -> type[_simple.Softmax]:
    ...

  @property
  def Dense(self) -> type[_dense.Dense]:
    ...

  @property
  def EinsumDense(self) -> type[_dense.EinsumDense]:
    ...

  @property
  def Conv1D(self) -> type[_convolution.Conv1D]:
    ...

  @property
  def DepthwiseConv1D(self) -> type[_convolution.DepthwiseConv1D]:
    ...

  @property
  def Conv1DTranspose(self) -> type[_convolution.Conv1DTranspose]:
    ...

  @property
  def Conv2D(self) -> type[_convolution.Conv2D]:
    ...

  @property
  def Conv2DTranspose(self) -> type[_convolution.Conv2DTranspose]:
    ...

  @property
  def MinPooling1D(self) -> type[_pooling.MinPooling1D]:
    ...

  @property
  def MaxPooling1D(self) -> type[_pooling.MaxPooling1D]:
    ...

  @property
  def AveragePooling1D(self) -> type[_pooling.AveragePooling1D]:
    ...

  @property
  def MinPooling2D(self) -> type[_pooling.MinPooling2D]:
    ...

  @property
  def MaxPooling2D(self) -> type[_pooling.MaxPooling2D]:
    ...

  @property
  def AveragePooling2D(self) -> type[_pooling.AveragePooling2D]:
    ...

  @property
  def MinPooling3D(self) -> type[_pooling.MinPooling3D]:
    ...

  @property
  def MaxPooling3D(self) -> type[_pooling.MaxPooling3D]:
    ...

  @property
  def AveragePooling3D(self) -> type[_pooling.AveragePooling3D]:
    ...

  @property
  def L2Normalize(self) -> type[_normalization.L2Normalize]:
    ...

  @property
  def RMSNormalization(self) -> type[_normalization.RMSNormalization]:
    ...

  @property
  def LayerNormalization(self) -> type[_normalization.LayerNormalization]:
    ...

  @property
  def BatchNormalization(self) -> type[_normalization.BatchNormalization]:
    ...

  @property
  def GroupNormalization(self) -> type[_normalization.GroupNormalization]:
    ...
