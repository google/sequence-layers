"""Simple sequence layers for MLX."""

import dataclasses
from fractions import Fraction
import math
from typing import Any, Callable, override

from absl import logging
from mlx import nn
import mlx.core as mx
import numpy as np

from sequence_layers.mlx import types
from sequence_layers.specs import simple as spec

Sequence = types.Sequence
MaskedSequence = types.MaskedSequence
ShapeDType = types.ShapeDType


def _to_tuple(x: complex | list[Any]) -> complex | tuple[Any, ...]:
  """Converts a nested list to a nested tuple."""
  if isinstance(x, list):
    return tuple(_to_tuple(item) for item in x)
  return x


@dataclasses.dataclass(frozen=True)
class HashableArray(spec.HashableArray):
  """Hashable multidimensional array of tuples."""

  data: complex | tuple[Any, ...]
  dtype: np.dtype

  @classmethod
  def from_array(cls, x: np.ndarray) -> 'HashableArray':
    """Creates a HashableArray from a numpy array."""
    x = np.asarray(x)
    return HashableArray(_to_tuple(x.tolist()), x.dtype)

  @override
  def to_array(self) -> np.ndarray:
    return np.asarray(self.data, dtype=self.dtype)


def _to_mx_dtype(dtype: Any) -> mx.Dtype | None:
  """Converts various dtype representations to MLX DType."""
  if dtype is None:
    return None
  if isinstance(dtype, str):
    if dtype == 'float32':
      return mx.float32
    if dtype == 'float16':
      return mx.float16
    if dtype == 'int32':
      return mx.int32
    if dtype == 'bool':
      return mx.bool_
  if dtype == np.float32:
    return mx.float32
  if dtype == np.float16:
    return mx.float16
  if dtype == np.int32:
    return mx.int32
  if dtype in (np.bool_, bool):
    return mx.bool_
  return dtype


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Identity(
    types.PreservesType,
    types.StatelessPointwise,
    spec.Identity[types.Sequence, types.ShapeDType],
):
  """Identity pass-through of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Identity.Config):
    """Configuration for Identity layer."""

    name: str | None = None

    @override
    def make(self) -> 'Identity':
      """Creates the Identity layer."""
      return Identity(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Returns the input sequence unchanged."""
    return x


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------


class Relu(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Relu[types.Sequence, types.ShapeDType],
):
  """A Relu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Relu.Config):
    """Configuration for Relu layer."""

    name: str | None = None

    @override
    def make(self) -> 'Relu':
      """Creates the Relu layer."""
      return Relu(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using ReLU."""
    return nn.relu(values), mask


class Gelu(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Gelu[types.Sequence, types.ShapeDType],
):
  """A Gelu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Gelu.Config):
    """Configuration for Gelu layer."""

    name: str | None = None

    @override
    def make(self) -> 'Gelu':
      """Creates the Gelu layer."""
      return Gelu(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using GELU."""
    return nn.gelu(values), mask


class Abs(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Abs[types.Sequence, types.ShapeDType],
):
  """Absolute value layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Abs.Config):
    """Configuration for Abs layer."""

    name: str | None = None

    @override
    def make(self) -> 'Abs':
      """Creates the Abs layer."""
      return Abs(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using absolute value."""
    return mx.abs(values), mask


class Exp(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Exp[types.Sequence, types.ShapeDType],
):
  """Exponential layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Exp.Config):
    """Configuration for Exp layer."""

    name: str | None = None

    @override
    def make(self) -> 'Exp':
      """Creates the Exp layer."""
      return Exp(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using exponential."""
    return mx.exp(values), mask


class Log(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Log[types.Sequence, types.ShapeDType],
):
  """Logarithm layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Log.Config):
    """Configuration for Log layer."""

    name: str | None = None

    @override
    def make(self) -> 'Log':
      """Creates the Log layer."""
      return Log(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using natural logarithm."""
    return mx.log(values), mask


class Swish(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Swish[types.Sequence, types.ShapeDType],
):
  """A Swish/SiLU layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Swish.Config):
    """Configuration for Swish layer."""

    name: str | None = None

    @override
    def make(self) -> 'Swish':
      """Creates the Swish layer."""
      return Swish(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using Swish (SiLU)."""
    return nn.silu(values), mask


class Tanh(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Tanh[types.Sequence, types.ShapeDType],
):
  """A tanh layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Tanh.Config):
    """Configuration for Tanh layer."""

    name: str | None = None

    @override
    def make(self) -> 'Tanh':
      """Creates the Tanh layer."""
      return Tanh(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using hyperbolic tangent."""
    return mx.tanh(values), mask


class Sigmoid(
    types.PreservesType, types.StatelessPointwiseFunctor, spec.Sigmoid
):
  """A sigmoid layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Sigmoid.Config):
    """Configuration for Sigmoid layer."""

    name: str | None = None

    @override
    def make(self) -> 'Sigmoid':
      """Creates the Sigmoid layer."""
      return Sigmoid(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using Sigmoid."""
    return mx.sigmoid(values), mask


class LeakyRelu(
    types.PreservesType, types.StatelessPointwiseFunctor, spec.LeakyRelu
):
  """A Leaky Relu layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.LeakyRelu.Config):
    """Configuration for LeakyRelu layer."""

    negative_slope: float = 0.01
    name: str | None = None

    @override
    def make(self) -> 'LeakyRelu':
      """Creates the LeakyRelu layer."""
      return LeakyRelu(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using Leaky ReLU."""
    return nn.leaky_relu(values, self.config.negative_slope), mask


class Elu(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.Elu[types.Sequence, types.ShapeDType],
):
  """An ELU activation layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Elu.Config):
    """Configuration for Elu layer."""

    alpha: complex = 1.0
    name: str | None = None

    @override
    def make(self) -> 'Elu':
      """Creates the Elu layer."""
      return Elu(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using ELU."""
    return nn.elu(values, self.config.alpha), mask


class Softmax(
    types.PreservesType, types.StatelessPointwiseFunctor, spec.Softmax
):
  """A softmax layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Softmax.Config):
    """Configuration for Softmax layer."""

    axis: int = -1
    name: str | None = None

    @override
    def make(self) -> 'Softmax':
      """Creates the Softmax layer."""
      return Softmax(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using Softmax."""
    axis = self.config.axis
    if (axis if axis >= 0 else values.ndim + axis) < 2:
      raise ValueError(
          'The softmax cannot be applied on the batch or time'
          f' dimension (got {axis=} for shape={values.shape})'
      )
    return mx.softmax(values, axis=axis), mask


class Softplus(
    types.PreservesType, types.StatelessPointwiseFunctor, spec.Softplus
):
  """A softplus layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Softplus.Config):
    """Configuration for Softplus layer."""

    name: str | None = None

    @override
    def make(self) -> 'Softplus':
      """Creates the Softplus layer."""
      return Softplus(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Transforms each scalar in values independently using Softplus."""
    return nn.softplus(values), mask


# ---------------------------------------------------------------------------
# Value manipulation
# ---------------------------------------------------------------------------


class Cast(
    types.StatelessPointwiseFunctor, spec.Cast[types.Sequence, types.ShapeDType]
):
  """Cast input values to the specified type."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.Cast.Config):
    """Configuration for Cast layer."""

    dtype: object = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'Cast':
      return Cast(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self._dtype = _to_mx_dtype(config.dtype)

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Casts input values to the specified type."""
    return values.astype(self._dtype), mask  # type: ignore

  @override
  def get_output_dtype(self, input_dtype, *, constants=None) -> mx.Dtype:
    assert self._dtype is not None
    return self._dtype


class Scale(
    types.PreservesType,
    types.StatelessPointwise,
    spec.Scale[types.Sequence, types.ShapeDType],
):
  """Scales the input by a provided constant or array."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Scale.Config):
    """Configuration for Scale layer."""

    scale: complex | np.ndarray | types.HashableArray = 1.0
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'scale', types.HashableArray.from_array(self.scale)
      )

    @override
    def make(self) -> 'Scale':
      """Creates the Scale layer."""
      return Scale(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    assert isinstance(config.scale, types.HashableArray)
    self._scale = config.scale.to_array()

  @override
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    s_shape = (
        ()
        if isinstance(self._scale, (int, float, complex))
        else self._scale.shape
    )
    if len(s_shape) > len(input_shape):
      raise ValueError(
          f'Scale parameter has too many dimensions ({len(s_shape)}) to'
          f' broadcast with input shape ({len(input_shape)}).'
      )
    try:
      return np.broadcast_shapes(tuple(input_shape), s_shape)
    except ValueError as e:
      raise ValueError(
          f'Cannot broadcast shape {input_shape} with scale shape {s_shape}'
      ) from e

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Scales the input sequence by a learned or fixed scale."""
    return x.apply_values_masked(lambda v: v * self._scale)


class Add(
    types.PreservesType,
    types.StatelessPointwise,
    spec.Add[types.Sequence, types.ShapeDType],
):
  """Adds a provided constant or array to the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Add.Config):
    """Configuration for Add layer."""

    shift: Any
    name: str | None = None

    @override
    def make(self) -> 'Add':
      """Creates the Add layer."""
      return Add(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    shift = config.shift
    if hasattr(shift, 'data') and hasattr(shift, 'dtype'):
      self._shift = np.array(shift.data, dtype=shift.dtype)
    elif hasattr(shift, 'array'):
      self._shift = np.asarray(shift.array)
    else:
      self._shift = shift

  @override
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    s_shape = (
        ()
        if isinstance(self._shift, (int, float, complex))
        else self._shift.shape
    )
    if len(s_shape) > len(input_shape):
      raise ValueError(
          f'Shift parameter has too many dimensions ({len(s_shape)}) to'
          f' broadcast with input shape ({len(input_shape)}).'
      )
    try:
      return np.broadcast_shapes(tuple(input_shape), s_shape)
    except ValueError as e:
      raise ValueError(
          f'Cannot broadcast shape {input_shape} with shift shape {s_shape}'
      ) from e

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Adds a learned or fixed shift to the input sequence."""
    return x.apply_values(lambda v: v + self._shift)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class MaskInvalid(
    types.PreservesType, types.StatelessPointwise, spec.MaskInvalid
):
  """Masks invalid timesteps to zero (or a specified value)."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.MaskInvalid.Config):
    """Configuration for MaskInvalid layer."""

    mask_value: Any = None
    name: str | None = None

    @override
    def make(self) -> 'MaskInvalid':
      """Creates the MaskInvalid layer."""
      return MaskInvalid(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Masks invalid values (NaN, Inf) in the input sequence."""
    return x.mask_invalid(self.config.mask_value)


# ---------------------------------------------------------------------------
# Gated units
# ---------------------------------------------------------------------------


class GatedUnit(
    types.PreservesType,
    types.Stateless,
    spec.GatedUnit[types.Sequence, types.ShapeDType],
):
  """Computes a generalized Gated Unit, reducing input channels by 2x."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.GatedUnit.Config):
    """Configuration for GatedUnit layer."""

    feature_activation: Callable | None = None
    gate_activation: Callable | None = None
    name: str | None = None

    @override
    def make(self) -> 'GatedUnit':
      return GatedUnit(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self._feature_activation = config.feature_activation
    self._gate_activation = config.gate_activation

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    channels = input_shape[-1]
    if channels % 2 != 0:
      raise ValueError(
          f'Final dimension of input ({input_shape=}) must have'
          ' an even number of channels.'
      )
    return tuple(input_shape[:-1]) + (channels // 2,)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Applies a gated unit to the input sequence."""
    feature, gate = mx.split(x.values, 2, axis=-1)
    if self._feature_activation:
      feature = self._feature_activation(feature)
    if self._gate_activation:
      gate = self._gate_activation(gate)
    return Sequence(feature * gate, x.mask)


class GatedLinearUnit(
    GatedUnit, spec.GatedLinearUnit[types.Sequence, types.ShapeDType]
):
  """Computes a Gated Linear Unit, reducing input channels by 2x."""

  @dataclasses.dataclass(frozen=True)
  class Config(GatedUnit.Config, spec.GatedLinearUnit.Config):
    """Configuration for GatedLinearUnit layer."""

    name: str | None = None

    @override
    def make(self) -> 'GatedLinearUnit':
      """Create GatedLinearUnit layer."""
      return GatedLinearUnit(
          GatedUnit.Config(
              feature_activation=None,
              gate_activation=mx.sigmoid,
              name=self.name,
          )
      )


class GatedTanhUnit(
    GatedUnit, spec.GatedTanhUnit[types.Sequence, types.ShapeDType]
):
  """Computes a Gated Tanh Unit, reducing input channels by 2x."""

  @dataclasses.dataclass(frozen=True)
  class Config(GatedUnit.Config, spec.GatedTanhUnit.Config):
    """Configuration for GatedTanhUnit layer."""

    name: str | None = None

    @override
    def make(self) -> 'GatedTanhUnit':
      return GatedTanhUnit(
          GatedUnit.Config(
              feature_activation=mx.tanh,
              gate_activation=mx.sigmoid,
              name=self.name,
          )
      )


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


class Flatten(
    types.PreservesType,
    types.StatelessPointwise,
    spec.Flatten[types.Sequence, types.ShapeDType],
):
  """Flattens the channel dimensions of the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Flatten layer."""

    name: str | None = None

    @override
    def make(self) -> 'Flatten':
      return Flatten(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return (math.prod(input_shape),)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Flattens the channel dimensions of the input sequence."""
    batch_size, time = x.values.shape[:2]
    num_elements = math.prod(x.channel_shape)
    new_values = mx.reshape(x.values, (batch_size, time, num_elements))
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)


class Reshape(
    types.PreservesType,
    types.Stateless,
    spec.Reshape[types.Sequence, types.ShapeDType],
):
  """Reshapes the channels dimension of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Reshape.Config):
    """Configuration for Reshape layer."""

    output_shape: tuple[int, ...] = ()
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    @override
    def make(self) -> 'Reshape':
      return Reshape(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _validate(self, input_shape):
    """Validates that input and output shapes have the same number of elements."""
    in_elems = math.prod(input_shape)

    out_elems = math.prod(self.config.output_shape)
    if in_elems != out_elems:
      raise ValueError(
          f'Reshape output_shape={self.config.output_shape} must have'
          f' the same number of elements as {input_shape=}.'
      )

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    self._validate(input_shape)
    return self.config.output_shape

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Reshapes the channel dimensions of the input sequence."""
    self._validate(x.channel_shape)
    b, t = x.values.shape[:2]
    new_values = mx.reshape(x.values, (b, t) + self.config.output_shape)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)


class ExpandDims(
    types.PreservesType,
    types.Stateless,
    spec.ExpandDims[types.Sequence, types.ShapeDType],
):
  """Expands channel dimensions of the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.ExpandDims.Config):
    """Configuration for ExpandDims layer."""

    axis: int | tuple[int, ...] = 0
    name: str | None = None

    def __post_init__(self):
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    @override
    def make(self) -> 'ExpandDims':
      return ExpandDims(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self._axis: tuple[int, ...] = (
        (config.axis,) if isinstance(config.axis, int) else tuple(config.axis)
    )

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _normalize_axes(self, input_shape):
    """Normalizes axes to positive indices."""
    rank = len(input_shape)

    dims = sorted(set(a + rank + 1 if a < 0 else a for a in self._axis))
    for d in dims:
      if d < 0 or d > rank:
        raise ValueError(
            f'ExpandDims axes must refer to channel dims. Got: {self._axis}.'
        )
    return dims

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    dims = self._normalize_axes(input_shape)
    out = list(input_shape)
    for a in dims:
      out.insert(a, 1)
    return tuple(out)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Expands the dimensions of the input sequence by inserting new axes."""
    dims = [2 + d for d in self._normalize_axes(x.channel_shape)]
    new_values = mx.expand_dims(x.values, axis=dims)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)


class Squeeze(
    types.PreservesType,
    types.Stateless,
    spec.Squeeze[types.Sequence, types.ShapeDType],
):
  """Squeezes singleton channel dimensions of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Squeeze.Config):
    """Configuration for Squeeze layer."""

    axis: int | tuple[int, ...] | None = None
    name: str | None = None

    @override
    def make(self) -> 'Squeeze':
      return Squeeze(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _channel_squeeze_axes(self, input_shape):
    """Return channel-relative axes to squeeze."""
    if self.config.axis is None:
      # Squeeze all singleton channel dims.
      return tuple(i for i, n in enumerate(input_shape) if n == 1)
    # If axis is given, it's in full-tensor coords. Convert to channel.
    if isinstance(self.config.axis, int):
      axes = (self.config.axis,)
    else:
      axes = tuple(self.config.axis)
    return axes

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    squeeze_axes = self._channel_squeeze_axes(input_shape)
    out = []
    for i, s in enumerate(input_shape):
      if i not in squeeze_axes:
        out.append(s)
    return tuple(out)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Squeezes the dimensions of the input sequence by removing axes of size 1."""
    ch_axes = self._channel_squeeze_axes(x.channel_shape)
    # Convert to full-tensor axes (offset by 2 for batch, time).
    full_axes = tuple(a + 2 for a in ch_axes)
    new_values = mx.squeeze(x.values, axis=full_axes)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)


class Transpose(
    types.PreservesType,
    types.Stateless,
    spec.Transpose[types.Sequence, types.ShapeDType],
):
  """Permutes the channel axes of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Transpose.Config):
    """Configuration for Transpose layer."""

    axes: tuple[int, ...] | None = None
    name: str | None = None

    @override
    def make(self) -> 'Transpose':
      return Transpose(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self._axes: tuple[int, ...] | None = (
        tuple(config.axes) if config.axes is not None else None
    )

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _resolve_axes(self, input_shape):
    """Resolves axes for transpose."""
    input_axes = tuple(range(2, 2 + len(input_shape)))

    if self._axes is None:
      return input_axes[::-1]
    sorted_axes = tuple(sorted(self._axes))
    if sorted_axes != input_axes:
      raise ValueError(
          f'Provided axes {sorted_axes} do not match input axes {input_axes}.'
      )
    return tuple(self._axes)

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    axes = self._resolve_axes(input_shape)
    return tuple(input_shape[a - 2] for a in axes)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Transposes the channel dimensions of the input sequence."""
    axes = self._resolve_axes(x.channel_shape)
    new_values = mx.transpose(x.values, (0, 1) + axes)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


class OneHot(types.Stateless, spec.OneHot[types.Sequence, types.ShapeDType]):
  """Computes one-hot vector of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.OneHot.Config):
    """Configuration for OneHot layer."""

    depth: int
    compute_dtype: Any = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'OneHot':
      return OneHot(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self._compute_dtype = _to_mx_dtype(config.compute_dtype)

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape) + (self.config.depth,)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None) -> mx.Dtype:
    assert self._compute_dtype is not None
    return self._compute_dtype

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Converts integer values to one-hot representations."""

    def one_hot_fn(v):
      indices = v.astype(mx.int32)
      return mx.eye(self.config.depth, dtype=self._compute_dtype)[indices]

    return x.apply_values(one_hot_fn)


class Embedding(
    types.Stateless, spec.Embedding[types.Sequence, types.ShapeDType]
):
  """Computes embeddings of integer input codes.

  Backed by mlx.nn.Embedding.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Embedding.Config):
    """Configuration for Embedding layer."""

    num_embeddings: int = 1
    dimension: int = 1
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'Embedding':
      return Embedding(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self._param_dtype = _to_mx_dtype(config.param_dtype)
    self._compute_dtype = (
        _to_mx_dtype(config.compute_dtype)
        if config.compute_dtype is not None
        else None
    )
    self._embedding = nn.Embedding(config.num_embeddings, config.dimension)

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape) + (self.config.dimension,)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None) -> mx.Dtype:
    if self._compute_dtype is not None:
      return self._compute_dtype
    assert self._param_dtype is not None
    return self._param_dtype

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Embeds integer values using a learned embedding matrix."""

    def embed_fn(v):
      result = self._embedding(v.astype(mx.int32))
      compute_dtype = self._compute_dtype
      if compute_dtype is not None:
        result = result.astype(compute_dtype)  # type: ignore
      return result

    return x.apply_values(embed_fn)


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


class Dropout(
    types.PreservesType,
    types.StatelessPointwise,
    spec.Dropout[types.Sequence, types.ShapeDType],
):
  """Dropout layer (pass-through during inference)."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Dropout.Config):
    """Configuration for Dropout layer."""

    rate: float = 0.0
    broadcast_dims: tuple[int, ...] = ()
    name: str | None = None

    @override
    def make(self) -> 'Dropout':
      """Creates the Dropout layer."""
      return Dropout(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Applies dropout to the input sequence."""
    if training:
      raise NotImplementedError('Dropout training is not implemented in MLX.')
    # Inference-only: dropout is a no-op.
    return x


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class Downsample1D(
    types.PreservesType,
    types.Stateless,
    spec.Downsample1D[types.Sequence, types.ShapeDType],
):
  """A 1D downsampling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Downsample1D.Config):
    """Configuration for Downsample1D layer."""

    rate: int
    name: str | None = None

    @override
    def make(self) -> 'Downsample1D':
      return Downsample1D(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def block_size(self):
    return self.config.rate

  @property
  @override
  def output_ratio(self):
    return Fraction(1, self.config.rate)

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Downsamples the input sequence along the time axis."""
    new_values = x.values[:, :: self.config.rate]
    new_mask = x.mask[:, :: self.config.rate]
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, new_mask)
    return Sequence(new_values, new_mask)


class Upsample1D(
    types.PreservesType,
    types.Stateless,
    spec.Upsample1D[types.Sequence, types.ShapeDType],
):
  """A 1D upsampling layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Upsample1D.Config):
    """Configuration for Upsample1D layer."""

    rate: int
    name: str | None = None

    @override
    def make(self) -> 'Upsample1D':
      return Upsample1D(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @property
  @override
  def output_ratio(self):
    return Fraction(self.config.rate, 1)

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Upsamples the input sequence along the time axis."""
    # Repeat each timestep `rate` times along the time axis.
    b, t = x.values.shape[:2]
    channel_shape = x.values.shape[2:]
    # [b, t, 1, ...] -> [b, t, rate, ...] -> [b, t*rate, ...]
    expanded = mx.expand_dims(x.values, axis=2)
    tiled = mx.repeat(expanded, self.config.rate, axis=2)
    new_values = mx.reshape(tiled, (b, t * self.config.rate) + channel_shape)
    # Same for mask: [b, t] -> [b, t*rate]
    new_mask = mx.repeat(
        mx.expand_dims(x.mask, axis=2), self.config.rate, axis=2
    )
    new_mask = mx.reshape(new_mask, (b, t * self.config.rate))
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, new_mask)
    return Sequence(new_values, new_mask)


# ---------------------------------------------------------------------------
# CheckpointName (identity for inference)
# ---------------------------------------------------------------------------


class CheckpointName(
    types.PreservesType,
    types.StatelessPointwiseFunctor,
    spec.CheckpointName[types.Sequence, types.ShapeDType],
):
  """Identity pass-through (checkpoint naming is JAX-only)."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.CheckpointName.Config):
    """Configuration for CheckpointName layer."""

    checkpoint_name: str = ''
    name: str | None = None

    @override
    def make(self) -> 'CheckpointName':
      """Creates the CheckpointName layer."""
      return CheckpointName(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @override
  def get_accumulated_input_latency(self, input_latency: int) -> int:
    return super().get_accumulated_input_latency(input_latency)

  @property
  @override
  def mask_required(self):
    return False

  @override
  def fn(self, values: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Identity function for CheckpointName."""
    return values, mask


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------


class Lambda(types.Stateless, spec.Lambda[types.Sequence, types.ShapeDType]):
  """A SequenceLayer that wraps a Python callable."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Lambda.Config):
    """Configuration for Lambda layer."""

    fn: Callable
    sequence_input: bool = False
    mask_required: bool = True
    # Accepted for JAX compatibility but ignored by MLX Lambda.
    expected_input_spec: object = None
    expected_output_spec: object = None
    name: str | None = None

    @override
    def make(self) -> 'Lambda':
      return Lambda(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self._cached_output_spec = None

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  def _probe_output(self, input_shape, input_dtype):
    """Probe the function with a dummy to infer output shape/dtype."""
    if self.config.expected_output_spec is not None:
      return self.config.expected_output_spec
    if self._cached_output_spec is not None:
      return self._cached_output_spec
    try:
      dummy_values = mx.zeros((1, 1) + tuple(input_shape), dtype=input_dtype)
      dummy_mask = mx.ones((1, 1), dtype=mx.bool_)
      assert self.config.fn is not None
      if self.config.sequence_input:
        result = self.config.fn(Sequence(dummy_values, dummy_mask))
        out_shape = result.values.shape[2:]
        out_dtype = result.values.dtype
      else:
        out_values = self.config.fn(dummy_values)
        out_shape = out_values.shape[2:]
        out_dtype = out_values.dtype
      self._cached_output_spec = types.ShapeDType(out_shape, out_dtype)
      return self._cached_output_spec
    except Exception:  # pylint: disable=broad-exception-caught
      return None

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    out_spec = self._probe_output(input_shape, mx.float32)
    if out_spec is not None:
      return tuple(out_spec.shape)
    return tuple(input_shape)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    out_spec = self._probe_output((1,), input_dtype)
    if out_spec is not None:
      return out_spec.dtype
    return input_dtype

  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Applies a custom Python callable to the input sequence."""
    assert self.config.fn is not None
    if self.config.sequence_input:
      result = self.config.fn(x)
      if not isinstance(result, (Sequence, MaskedSequence)):
        raise ValueError(
            'Lambda with sequence_input=True must return a Sequence, '
            f'got {type(result)}'
        )
      return result

    new_values = self.config.fn(x.values)
    if self.config.mask_required or not isinstance(x, MaskedSequence):
      return Sequence(new_values, x.mask)
    return MaskedSequence(new_values, x.mask)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class Logging(
    types.PreservesType,
    types.StatelessPointwise,
    spec.Logging[types.Sequence, types.ShapeDType],
):
  """Logs input info and returns the input unchanged."""

  @dataclasses.dataclass(frozen=True)
  class Config(spec.Logging.Config):
    """Configuration for Logging layer."""

    prefix: str = ''
    dump_tensors: bool = False
    name: str | None = None

    @override
    def make(self) -> 'Logging':
      """Creates the Logging layer."""
      return Logging(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @override
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    if self.config.dump_tensors:
      logging.info(
          f'{self.config.prefix} get_initial_state(): batch_size={batch_size}, '
          f'input_spec={input_spec}, training={training}, constants={constants}'
      )
    else:
      logging.info(
          f'{self.config.prefix} get_initial_state(): batch_size={batch_size}, '
          f'input_spec={input_spec}, training={training}'
      )
    return super().get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )

  @override
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if self.config.dump_tensors:
      logging.info(
          f'{self.config.prefix} step(): x={x.values}, state={state}, '
          f'training={training}, constants={constants}'
      )
    else:
      logging.info(
          f'{self.config.prefix} step(): x.shape={x.shape}, x.dtype={x.dtype}, '
          f'state={state}, training={training}'
      )
    return super().step(x, state, training=training, constants=constants)

  @types.check_layer
  @override
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool, constants=None
  ):
    """Logs the input sequence values for debugging."""
    if self.config.dump_tensors:
      logging.info(
          f'{self.config.prefix} layer(): x={x.values}, training={training},'
          f' constants={constants}'
      )
    else:
      logging.info(
          f'{self.config.prefix} layer(): x.shape={x.shape}, x.dtype={x.dtype},'
          f' training={training}'
      )
    return x
