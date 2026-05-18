"""Simple sequence layers for MLX."""

import dataclasses
import math

from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Identity(types.PreservesType, types.StatelessPointwise):
  """Identity pass-through of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Identity':
      return Identity.from_config(self)

  @types.check_layer
  def layer(self, x, *, constants=None):
    return x

  @classmethod
  def from_config(cls, config):
    return cls()


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------


class Relu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Relu layer."""

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return nn.relu(values), mask

  @classmethod
  def from_config(cls, config):
    return cls()


class Gelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Gelu layer."""

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return nn.gelu(values), mask

  @classmethod
  def from_config(cls, config):
    return cls()


class Swish(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Swish/SiLU layer."""

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return nn.silu(values), mask

  @classmethod
  def from_config(cls, config):
    return cls()


class Tanh(types.PreservesType, types.StatelessPointwiseFunctor):
  """A tanh layer."""

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return mx.tanh(values), mask

  @classmethod
  def from_config(cls, config):
    return cls()


class Sigmoid(types.PreservesType, types.StatelessPointwiseFunctor):
  """A sigmoid layer."""

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return mx.sigmoid(values), mask

  @classmethod
  def from_config(cls, config):
    return cls()


class LeakyRelu(types.PreservesType, types.StatelessPointwiseFunctor):
  """A Leaky Relu layer."""

  def __init__(self, negative_slope=0.01):
    super().__init__()
    self._negative_slope = negative_slope

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return nn.leaky_relu(values, self._negative_slope), mask

  @classmethod
  def from_config(cls, config):
    return cls(negative_slope=config.negative_slope)


class Elu(types.PreservesType, types.StatelessPointwiseFunctor):
  """An ELU activation layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    alpha: complex = 1.0
    name: str | None = None

    def make(self) -> 'Elu':
      return Elu.from_config(self)

  def __init__(self, alpha=1.0):
    super().__init__()
    self._alpha = alpha

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return nn.elu(values, self._alpha), mask

  @classmethod
  def from_config(cls, config):
    return cls(alpha=config.alpha)


class Softmax(types.PreservesType, types.StatelessPointwiseFunctor):
  """A softmax layer."""

  def __init__(self, axis=-1):
    super().__init__()
    self._axis = axis

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    axis = self._axis
    if (axis if axis >= 0 else values.ndim + axis) < 2:
      raise ValueError(
          'The softmax cannot be applied on the batch or time'
          f' dimension (got {axis=} for shape={values.shape})'
      )
    return mx.softmax(values, axis=axis), mask

  @classmethod
  def from_config(cls, config):
    return cls(axis=config.axis)


class Softplus(types.PreservesType, types.StatelessPointwiseFunctor):
  """A softplus layer."""

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return nn.softplus(values), mask

  @classmethod
  def from_config(cls, config):
    return cls()


# ---------------------------------------------------------------------------
# Value manipulation
# ---------------------------------------------------------------------------


class Cast(types.StatelessPointwiseFunctor):
  """Cast input values to the specified type."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    dtype: object = mx.float32
    name: str | None = None

    def make(self) -> 'Cast':
      return Cast.from_config(self)

  def __init__(self, dtype):
    super().__init__()
    self._dtype = dtype

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return values.astype(self._dtype), mask

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self._dtype

  @classmethod
  def from_config(cls, config):
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    return cls(dtype=_to_mx_dtype(config.dtype))


class Scale(types.PreservesType, types.StatelessPointwise):
  """Scales the input by a provided constant or array."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    scale: object = 1.0
    name: str | None = None

    def make(self) -> 'Scale':
      return Scale.from_config(self)

  def __init__(self, scale):
    super().__init__()
    if isinstance(scale, (int, float, complex)):
      self._scale = scale
    else:
      self._scale = mx.array(np.asarray(scale))

  @types.check_layer
  def layer(self, x, *, constants=None):
    s = self._scale
    if isinstance(s, mx.array):
      s = s.astype(x.dtype)
    return x.apply_values_masked(lambda v: v * s)

  @classmethod
  def from_config(cls, config):
    scale = config.scale
    if hasattr(scale, 'data') and hasattr(scale, 'dtype'):
      scale = np.array(scale.data, dtype=scale.dtype)
    elif hasattr(scale, 'array'):
      scale = np.asarray(scale.array)
    return cls(scale=scale)


class Add(types.PreservesType, types.StatelessPointwise):
  """Adds a provided constant or array to the input."""

  def __init__(self, shift):
    super().__init__()
    if isinstance(shift, (int, float, complex)):
      self._shift = shift
    else:
      self._shift = mx.array(np.asarray(shift))

  @types.check_layer
  def layer(self, x, *, constants=None):
    s = self._shift
    if isinstance(s, mx.array):
      s = s.astype(x.dtype)
    return x.apply_values(lambda v: v + s)

  @classmethod
  def from_config(cls, config):
    shift = config.shift
    if hasattr(shift, 'data') and hasattr(shift, 'dtype'):
      shift = np.array(shift.data, dtype=shift.dtype)
    elif hasattr(shift, 'array'):
      shift = np.asarray(shift.array)
    return cls(shift=shift)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class MaskInvalid(types.PreservesType, types.StatelessPointwise):
  """Masks invalid timesteps to zero (or a specified value)."""

  def __init__(self, mask_value=None):
    super().__init__()
    self._mask_value = mask_value

  @types.check_layer
  def layer(self, x, *, constants=None):
    return x.mask_invalid(self._mask_value)

  @classmethod
  def from_config(cls, config):
    mask_value = getattr(config, 'mask_value', None)
    return cls(mask_value=mask_value)


# ---------------------------------------------------------------------------
# Gated units
# ---------------------------------------------------------------------------


class GatedUnit(types.PreservesType, types.Stateless):
  """Computes a generalized Gated Unit, reducing input channels by 2x."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    feature_activation: Callable | None = None
    gate_activation: Callable | None = None
    name: str | None = None

    def make(self) -> 'GatedUnit':
      return GatedUnit.from_config(self)

  def __init__(self, feature_activation=None, gate_activation=None):
    super().__init__()
    self._feature_activation = feature_activation
    self._gate_activation = gate_activation

  def get_output_shape(self, input_shape, *, constants=None):
    channels = input_shape[-1]
    if channels % 2 != 0:
      raise ValueError(
          f'Final dimension of input ({input_shape=}) must have'
          ' an even number of channels.'
      )
    return tuple(input_shape[:-1]) + (channels // 2,)

  @types.check_layer
  def layer(self, x, *, constants=None):
    feature, gate = mx.split(x.values, 2, axis=-1)
    if self._feature_activation:
      feature = self._feature_activation(feature)
    if self._gate_activation:
      gate = self._gate_activation(gate)
    return Sequence(feature * gate, x.mask)

  @classmethod
  def from_config(cls, config):
    fa = init_mapping.map_activation(config.feature_activation)
    ga = init_mapping.map_activation(config.gate_activation)
    return cls(feature_activation=fa, gate_activation=ga)


class GatedLinearUnit(GatedUnit):
  """Computes a Gated Linear Unit, reducing input channels by 2x."""

  def __init__(self):
    super().__init__(
        feature_activation=None,
        gate_activation=mx.sigmoid,
    )

  @classmethod
  def from_config(cls, config):
    return cls()


class GatedTanhUnit(GatedUnit):
  """Computes a Gated Tanh Unit, reducing input channels by 2x."""

  def __init__(self):
    super().__init__(
        feature_activation=mx.tanh,
        gate_activation=mx.sigmoid,
    )

  @classmethod
  def from_config(cls, config):
    return cls()


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


class Flatten(types.PreservesType, types.StatelessPointwise):
  """Flattens the channel dimensions of the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'Flatten':
      return Flatten.from_config(self)

  def get_output_shape(self, input_shape, *, constants=None):
    return (math.prod(input_shape),)

  @types.check_layer
  def layer(self, x, *, constants=None):
    batch_size, time = x.values.shape[:2]
    num_elements = math.prod(x.channel_shape)
    new_values = mx.reshape(x.values, (batch_size, time, num_elements))
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)

  @classmethod
  def from_config(cls, config):
    return cls()


class Reshape(types.PreservesType, types.Stateless):
  """Reshapes the channels dimension of the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    output_shape: tuple[int, ...] = ()
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'output_shape', tuple(self.output_shape))

    def make(self) -> 'Reshape':
      return Reshape.from_config(self)

  def __init__(self, output_shape):
    super().__init__()
    self._output_shape = tuple(output_shape)

  def _validate(self, input_shape):
    in_elems = math.prod(input_shape)
    out_elems = math.prod(self._output_shape)
    if in_elems != out_elems:
      raise ValueError(
          f'Reshape output_shape={self._output_shape} must have'
          f' the same number of elements as {input_shape=}.'
      )

  def get_output_shape(self, input_shape, *, constants=None):
    self._validate(input_shape)
    return self._output_shape

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._validate(x.channel_shape)
    b, t = x.values.shape[:2]
    new_values = mx.reshape(x.values, (b, t) + self._output_shape)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)

  @classmethod
  def from_config(cls, config):
    return cls(output_shape=config.output_shape)


class ExpandDims(types.PreservesType, types.Stateless):
  """Expands channel dimensions of the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    axis: int | tuple[int, ...] = 0
    name: str | None = None

    def __post_init__(self):
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    def make(self) -> 'ExpandDims':
      return ExpandDims.from_config(self)

  def __init__(self, axis):
    super().__init__()
    if isinstance(axis, int):
      self._axis = (axis,)
    else:
      self._axis = tuple(axis)

  def _normalize_axes(self, input_shape):
    rank = len(input_shape)
    dims = sorted(set(a + rank + 1 if a < 0 else a for a in self._axis))
    for d in dims:
      if d < 0 or d > rank:
        raise ValueError(
            f'ExpandDims axes must refer to channel dims. Got: {self._axis}.'
        )
    return dims

  def get_output_shape(self, input_shape, *, constants=None):
    dims = self._normalize_axes(input_shape)
    out = list(input_shape)
    for a in dims:
      out.insert(a, 1)
    return tuple(out)

  @types.check_layer
  def layer(self, x, *, constants=None):
    dims = [2 + d for d in self._normalize_axes(x.channel_shape)]
    new_values = mx.expand_dims(x.values, axis=dims)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)

  @classmethod
  def from_config(cls, config):
    return cls(axis=config.axis)


class Squeeze(types.PreservesType, types.Stateless):
  """Squeezes singleton channel dimensions of the input."""

  def __init__(self, axis=None):
    super().__init__()
    self._axis = axis

  def _channel_squeeze_axes(self, input_shape):
    """Return channel-relative axes to squeeze."""
    if self._axis is None:
      # Squeeze all singleton channel dims.
      return tuple(i for i, n in enumerate(input_shape) if n == 1)
    # If axis is given, it's in full-tensor coords. Convert to channel.
    if isinstance(self._axis, int):
      axes = (self._axis,)
    else:
      axes = tuple(self._axis)
    return axes

  def get_output_shape(self, input_shape, *, constants=None):
    squeeze_axes = self._channel_squeeze_axes(input_shape)
    out = []
    for i, s in enumerate(input_shape):
      if i not in squeeze_axes:
        out.append(s)
    return tuple(out) if out else (1,)

  @types.check_layer
  def layer(self, x, *, constants=None):
    ch_axes = self._channel_squeeze_axes(x.channel_shape)
    # Convert to full-tensor axes (offset by 2 for batch, time).
    full_axes = tuple(a + 2 for a in ch_axes)
    new_values = mx.squeeze(x.values, axis=full_axes)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)

  @classmethod
  def from_config(cls, config):
    return cls(axis=config.axis)


class Transpose(types.PreservesType, types.Stateless):
  """Permutes the channel axes of the input."""

  def __init__(self, axes=None):
    super().__init__()
    if axes is not None:
      axes = tuple(axes)
    self._axes = axes

  def _resolve_axes(self, input_shape):
    input_axes = tuple(range(2, 2 + len(input_shape)))
    if self._axes is None:
      return input_axes[::-1]
    sorted_axes = tuple(sorted(self._axes))
    if sorted_axes != input_axes:
      raise ValueError(
          f'Provided axes {sorted_axes} do not match input axes {input_axes}.'
      )
    return tuple(self._axes)

  def get_output_shape(self, input_shape, *, constants=None):
    axes = self._resolve_axes(input_shape)
    return tuple(input_shape[a - 2] for a in axes)

  @types.check_layer
  def layer(self, x, *, constants=None):
    axes = self._resolve_axes(x.channel_shape)
    new_values = mx.transpose(x.values, (0, 1) + axes)
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, x.mask)
    return Sequence(new_values, x.mask)

  @classmethod
  def from_config(cls, config):
    return cls(axes=config.axes)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


class OneHot(types.Stateless):
  """Computes one-hot vector of the input."""

  def __init__(self, depth, compute_dtype=mx.float32):
    super().__init__()
    self._depth = depth
    self._compute_dtype = compute_dtype

  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape) + (self._depth,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self._compute_dtype

  @types.check_layer
  def layer(self, x, *, constants=None):
    def one_hot_fn(v):
      indices = v.astype(mx.int32)
      return mx.eye(self._depth, dtype=self._compute_dtype)[indices]

    return x.apply_values(one_hot_fn)

  @classmethod
  def from_config(cls, config):
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    return cls(
        depth=config.depth,
        compute_dtype=_to_mx_dtype(config.compute_dtype),
    )


class Embedding(types.Stateless):
  """Computes embeddings of integer input codes.

  Backed by mlx.nn.Embedding.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    num_embeddings: int = 1
    dimension: int = 1
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def make(self) -> 'Embedding':
      return Embedding.from_config(self)

  def __init__(
      self,
      *,
      num_embeddings: int,
      dimension: int,
      param_dtype=mx.float32,
      compute_dtype=None,
  ):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.dimension = dimension
    self._param_dtype = param_dtype
    self.compute_dtype = compute_dtype
    self._embedding = nn.Embedding(num_embeddings, dimension)

  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape) + (self.dimension,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @types.check_layer
  def layer(self, x, *, constants=None):
    def embed_fn(v):
      result = self._embedding(v.astype(mx.int32))
      if self.compute_dtype is not None:
        result = result.astype(self.compute_dtype)
      return result

    return x.apply_values(embed_fn)

  @classmethod
  def from_config(cls, config):
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    compute_dtype = getattr(config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = _to_mx_dtype(compute_dtype)
    return cls(
        num_embeddings=config.num_embeddings,
        dimension=config.dimension,
        param_dtype=_to_mx_dtype(config.param_dtype),
        compute_dtype=compute_dtype,
    )


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


class Dropout(types.PreservesType, types.StatelessPointwise):
  """Dropout layer (pass-through during inference)."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    rate: float = 0.0
    broadcast_dims: tuple[int, ...] = ()
    name: str | None = None

    def make(self) -> 'Dropout':
      return Dropout.from_config(self)

  def __init__(self, rate=0.0):
    super().__init__()
    self._rate = rate

  @types.check_layer
  def layer(self, x, *, constants=None):
    # Inference-only: dropout is a no-op.
    return x

  @classmethod
  def from_config(cls, config):
    return cls(rate=config.rate)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class Downsample1D(types.PreservesType, types.Stateless):
  """A 1D downsampling layer."""

  def __init__(self, rate):
    super().__init__()
    self._rate = rate

  @property
  def block_size(self):
    return self._rate

  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  @types.check_layer
  def layer(self, x, *, constants=None):
    new_values = x.values[:, :: self._rate]
    new_mask = x.mask[:, :: self._rate]
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, new_mask)
    return Sequence(new_values, new_mask)

  @classmethod
  def from_config(cls, config):
    return cls(rate=config.rate)


class Upsample1D(types.PreservesType, types.Stateless):
  """A 1D upsampling layer."""

  def __init__(self, rate):
    super().__init__()
    self._rate = rate

  def get_output_shape(self, input_shape, *, constants=None):
    return tuple(input_shape)

  @types.check_layer
  def layer(self, x, *, constants=None):
    # Repeat each timestep `rate` times along the time axis.
    b, t = x.values.shape[:2]
    channel_shape = x.values.shape[2:]
    # [b, t, 1, ...] -> [b, t, rate, ...] -> [b, t*rate, ...]
    expanded = mx.expand_dims(x.values, axis=2)
    tiled = mx.repeat(expanded, self._rate, axis=2)
    new_values = mx.reshape(tiled, (b, t * self._rate) + channel_shape)
    # Same for mask: [b, t] -> [b, t*rate]
    new_mask = mx.repeat(mx.expand_dims(x.mask, axis=2), self._rate, axis=2)
    new_mask = mx.reshape(new_mask, (b, t * self._rate))
    if isinstance(x, MaskedSequence):
      return MaskedSequence(new_values, new_mask)
    return Sequence(new_values, new_mask)

  @classmethod
  def from_config(cls, config):
    return cls(rate=config.rate)


# ---------------------------------------------------------------------------
# CheckpointName (identity for inference)
# ---------------------------------------------------------------------------


class CheckpointName(types.PreservesType, types.StatelessPointwiseFunctor):
  """Identity pass-through (checkpoint naming is JAX-only)."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    checkpoint_name: str = ''
    name: str | None = None

    def make(self) -> 'CheckpointName':
      return CheckpointName.from_config(self)

  def __init__(self, checkpoint_name=''):
    super().__init__()
    self._checkpoint_name = checkpoint_name

  @property
  def mask_required(self):
    return False

  def fn(self, values, mask):
    return values, mask

  @classmethod
  def from_config(cls, config):
    return cls(checkpoint_name=config.checkpoint_name)


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------


class Lambda(types.Stateless):
  """A SequenceLayer that wraps a Python callable."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    fn: Callable = None
    sequence_input: bool = False
    mask_required: bool = True
    # Accepted for JAX compatibility but ignored by MLX Lambda.
    expected_input_spec: object = None
    expected_output_spec: object = None
    name: str | None = None

    def make(self) -> 'Lambda':
      return Lambda.from_config(self)

  def __init__(self, fn, *, sequence_input=False, mask_required=True,
               expected_output_spec=None):
    super().__init__()
    self._fn = fn
    self._sequence_input = sequence_input
    self._mask_required = mask_required
    self._expected_output_spec = expected_output_spec
    self._cached_output_spec = None

  def _probe_output(self, input_shape, input_dtype):
    """Probe the function with a dummy to infer output shape/dtype."""
    if self._expected_output_spec is not None:
      return self._expected_output_spec
    if self._cached_output_spec is not None:
      return self._cached_output_spec
    try:
      dummy_values = mx.zeros((1, 1) + tuple(input_shape), dtype=input_dtype)
      dummy_mask = mx.ones((1, 1), dtype=mx.bool_)
      if self._sequence_input:
        result = self._fn(Sequence(dummy_values, dummy_mask))
        out_shape = result.values.shape[2:]
        out_dtype = result.values.dtype
      else:
        out_values = self._fn(dummy_values)
        out_shape = out_values.shape[2:]
        out_dtype = out_values.dtype
      # Don't cache here; shape and dtype probes use different dummy dtypes,
      # so a single cache would return stale dtype info.
      return bt.ShapeDType(out_shape, out_dtype)
    except Exception:
      return None

  def get_output_shape(self, input_shape, *, constants=None):
    spec = self._probe_output(input_shape, mx.float32)
    if spec is not None:
      return tuple(spec.shape)
    return tuple(input_shape)

  def get_output_dtype(self, input_dtype, *, constants=None):
    spec = self._probe_output((1,), input_dtype)
    if spec is not None:
      return spec.dtype
    return input_dtype

  def layer(self, x, *, constants=None):
    if self._sequence_input:
      result = self._fn(x)
      if not isinstance(result, (Sequence, MaskedSequence)):
        raise ValueError(
            'Lambda with sequence_input=True must return a Sequence, '
            f'got {type(result)}'
        )
      return result
    else:
      new_values = self._fn(x.values)
      if self._mask_required or not isinstance(x, MaskedSequence):
        return Sequence(new_values, x.mask)
      return MaskedSequence(new_values, x.mask)

  @classmethod
  def from_config(cls, config):
    return cls(
        fn=config.fn,
        sequence_input=config.sequence_input,
        mask_required=config.mask_required,
        expected_output_spec=getattr(config, 'expected_output_spec', None),
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class Logging(types.PreservesType, types.StatelessPointwise):
  """Logs input info and returns the input unchanged."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    prefix: str = ''
    dump_tensors: bool = False
    name: str | None = None

    def make(self) -> 'Logging':
      return Logging.from_config(self)

  def __init__(self, prefix='', dump_tensors=False):
    super().__init__()
    self._prefix = prefix
    self._dump_tensors = dump_tensors

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self._dump_tensors:
      print(f'{self._prefix} layer(): x={x.values}')
    else:
      print(
          f'{self._prefix} layer(): x.shape={x.shape}, '
          f'x.dtype={x.dtype}'
      )
    return x

  @classmethod
  def from_config(cls, config):
    return cls(
        prefix=config.prefix,
        dump_tensors=config.dump_tensors,
    )
