# https://typing.python.org/en/latest/spec/protocol.html#modules-as-implementations-of-protocols

from typing import Protocol, runtime_checkable

from . import backend as _backend
from . import types as _types


@runtime_checkable
class ModuleSpec(Protocol):
  """Protocol for a backend-specific SequenceLayers module (sequence_layers.<backend> as sl)."""

  @property
  def backend(self) -> _backend.ModuleSpec:
    ...

  @property
  def types(self) -> _types.ModuleSpec:
    ...

  # Identifiers that backend-specific implementations should expose at top level.
  # Demonstrating read-only allows for covariance (subclasses of types_module.Sequence to satisfy the protocol).

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
