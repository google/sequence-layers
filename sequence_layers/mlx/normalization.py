"""Normalization layers for MLX."""

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence


def _normalize_axes(axis, input_shape):
  """Normalize axes and check batch/time are not specified."""
  if isinstance(axis, int):
    axis = (axis,)
  normalized = set()
  for a in axis:
    if a < 0:
      a += len(input_shape)
    normalized.add(a)
  axes = tuple(sorted(normalized))
  for a in axes:
    if a in (0, 1):
      raise ValueError(
          f'Normalizing over batch or time is not allowed. Got: {axes}'
      )
  return axes


class L2Normalize(types.PreservesType, types.StatelessPointwise):
  """L2 normalization over the specified channel axes."""

  def __init__(self, *, axis=-1, epsilon: float = 1e-12):
    super().__init__()
    self._axis = axis
    self.epsilon = epsilon

  @types.check_layer
  def layer(self, x, *, constants=None):
    values = x.values
    axes = _normalize_axes(self._axis, values.shape)

    v = values.astype(mx.float32)
    squared_sum = mx.sum(mx.square(v), axis=axes, keepdims=True)
    normed = v * mx.rsqrt(squared_sum + self.epsilon)
    return Sequence(normed.astype(values.dtype), x.mask)

  @classmethod
  def from_config(cls, config):
    axis = config.axis
    if not isinstance(axis, int):
      axis = tuple(axis)
    return cls(axis=axis, epsilon=config.epsilon)


class RMSNormalization(types.PreservesType, types.StatelessPointwise):
  """RMS Normalization backed by mlx.nn.RMSNorm."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    axis: int | tuple[int, ...] = -1
    epsilon: float = 1e-6
    use_scale: bool = True
    scale_init: object = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    def make(self) -> 'RMSNormalization':
      return RMSNormalization.from_config(self)

  def __init__(
      self,
      *,
      axis=-1,
      epsilon: float = 1e-6,
      use_scale: bool = True,
      param_dtype=mx.float32,
      scale_init=None,
  ):
    super().__init__()
    self._axis = axis
    self.epsilon = epsilon
    self.use_scale = use_scale
    self._param_dtype = param_dtype
    self._scale_init = scale_init
    # mlx.nn.RMSNorm created lazily since we need input shape.
    self._rms_norm = None
    self._use_builtin = False

  def _ensure_initialized(self, input_shape):
    """Create internal RMSNorm on first call."""
    if self._rms_norm is not None or not self.use_scale:
      return
    axes = _normalize_axes(self._axis, input_shape)
    # mlx.nn.RMSNorm only supports normalizing over the last dim.
    if axes == (len(input_shape) - 1,) and self._scale_init is None:
      dims = input_shape[-1]
      self._rms_norm = nn.RMSNorm(dims, eps=self.epsilon)
      self._use_builtin = True
    else:
      # Multi-axis or custom init: manual scale parameter.
      scale_shape = tuple(input_shape[a] for a in axes)
      if self._scale_init is not None:
        key = mx.random.key(0)
        self._scale = self._scale_init(key, scale_shape, self._param_dtype)
      else:
        self._scale = mx.ones(scale_shape, dtype=self._param_dtype)

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.values.shape)

    if self._use_builtin and self._rms_norm is not None:
      # Cast back to input dtype to preserve bfloat16 compute.
      result = self._rms_norm(x.values).astype(x.values.dtype)
      return Sequence(result, x.mask)

    values = x.values
    axes = _normalize_axes(self._axis, values.shape)

    # Manual RMS norm in float32.
    v = values.astype(mx.float32)
    mean_sq = mx.mean(mx.square(v), axis=axes, keepdims=True)
    normed = v * mx.rsqrt(mean_sq + self.epsilon)
    normed = normed.astype(values.dtype)

    # Apply learned scale.
    if self.use_scale:
      scale = self._scale.astype(normed.dtype)
      shape = [1] * len(values.shape)
      for i, a in enumerate(axes):
        shape[a] = self._scale.shape[i]
      scale = scale.reshape(shape)
      normed = normed * scale

    return Sequence(normed, x.mask)

  @classmethod
  def from_config(cls, config):
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    axis = config.axis
    if not isinstance(axis, int):
      axis = tuple(axis)
    return cls(
        axis=axis,
        epsilon=config.epsilon,
        use_scale=config.use_scale,
        param_dtype=_to_mx_dtype(config.param_dtype),
        scale_init=init_mapping.map_initializer(config.scale_init),
    )


class LayerNormalization(types.PreservesType, types.StatelessPointwise):
  """Layer Normalization backed by mlx.nn.LayerNorm.

  For simple axis=-1 normalization, delegates to mlx.nn.LayerNorm.
  Falls back to manual computation for multi-axis cases.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    axis: int | tuple[int, ...] = -1
    epsilon: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    # Accepted for JAX compatibility but ignored: MLX always reduces in fp32.
    reductions_in_at_least_fp32: bool = True
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    def make(self) -> 'LayerNormalization':
      return LayerNormalization.from_config(self)


  def __init__(
      self,
      *,
      axis=-1,
      epsilon: float = 1e-6,
      use_bias: bool = True,
      use_scale: bool = True,
      param_dtype=mx.float32,
      reductions_in_at_least_fp32: bool = True,
  ):
    super().__init__()
    self._axis = axis
    self.epsilon = epsilon
    self.use_bias = use_bias
    self.use_scale = use_scale
    self._param_dtype = param_dtype
    self.reductions_in_at_least_fp32 = reductions_in_at_least_fp32
    self._layer_norm = None
    self._use_builtin = False
    self._manual_scale = None
    self._manual_bias = None

  def _ensure_initialized(self, input_shape):
    if self._layer_norm is not None or self._manual_scale is not None:
      return
    if not self.use_scale and not self.use_bias:
      return
    axes = _normalize_axes(self._axis, input_shape)
    # mlx.nn.LayerNorm supports a single last-dim normalization.
    if axes == (len(input_shape) - 1,):
      dims = input_shape[-1]
      self._layer_norm = nn.LayerNorm(
          dims,
          eps=self.epsilon,
          affine=self.use_scale or self.use_bias,
          bias=self.use_bias,
      )
      self._use_builtin = True
    else:
      # Multi-axis: manual parameters.
      scale_shape = tuple(input_shape[a] for a in axes)
      if self.use_scale:
        self._manual_scale = mx.ones(scale_shape, dtype=self._param_dtype)
      if self.use_bias:
        self._manual_bias = mx.zeros(scale_shape, dtype=self._param_dtype)

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.values.shape)

    if self._use_builtin and self._layer_norm is not None:
      x_values = x.values
      original_dtype = x_values.dtype
      if self.reductions_in_at_least_fp32:
        x_values = x_values.astype(mx.float32)
      # Cast back to input dtype to preserve bfloat16 compute.
      result = self._layer_norm(x_values).astype(original_dtype)
      return Sequence(result, x.mask)

    values = x.values
    axes = _normalize_axes(self._axis, values.shape)

    # Manual layer norm in float32.
    v = values.astype(mx.float32)
    mean = mx.mean(v, axis=axes, keepdims=True)
    variance = mx.mean(mx.square(v - mean), axis=axes, keepdims=True)
    normed = (v - mean) * mx.rsqrt(variance + self.epsilon)
    normed = normed.astype(values.dtype)

    # Apply learned scale and bias.
    if self.use_scale and self._manual_scale is not None:
      scale = self._manual_scale.astype(normed.dtype)
      shape = [1] * len(values.shape)
      for i, a in enumerate(axes):
        shape[a] = self._manual_scale.shape[i]
      normed = normed * scale.reshape(shape)

    if self.use_bias and self._manual_bias is not None:
      bias = self._manual_bias.astype(normed.dtype)
      shape = [1] * len(values.shape)
      for i, a in enumerate(axes):
        shape[a] = self._manual_bias.shape[i]
      normed = normed + bias.reshape(shape)

    return Sequence(normed, x.mask)

  @classmethod
  def from_config(cls, config):
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    axis = config.axis
    if not isinstance(axis, int):
      axis = tuple(axis)
    return cls(
        axis=axis,
        epsilon=config.epsilon,
        use_bias=config.use_bias,
        use_scale=config.use_scale,
        param_dtype=_to_mx_dtype(config.param_dtype),
        reductions_in_at_least_fp32=config.reductions_in_at_least_fp32
    )


class BatchNormalization(types.PreservesType, types.StatelessPointwise):
  """Batch Normalization (inference-only).

  Uses stored running mean/variance for normalization. Training-mode
  batch stat computation is not supported (MLX port is inference-only).
  Running stats are loaded via weight_converter.load_linen_params().
  """

  def __init__(
      self,
      *,
      axis=-1,
      epsilon: float = 0.001,
      use_bias: bool = True,
      use_scale: bool = True,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self._axis = axis
    self.epsilon = epsilon
    self.use_bias = use_bias
    self.use_scale = use_scale
    self._param_dtype = param_dtype
    self._running_mean = None
    self._running_var = None
    self._scale = None
    self._bias = None

  def _ensure_initialized(self, input_shape):
    if self._running_mean is not None:
      return
    axes = _normalize_axes(self._axis, input_shape)
    axis_size = input_shape[axes[0]]
    self._running_mean = mx.zeros((axis_size,), dtype=self._param_dtype)
    self._running_var = mx.ones((axis_size,), dtype=self._param_dtype)
    if self.use_scale:
      self._scale = mx.ones((axis_size,), dtype=self._param_dtype)
    if self.use_bias:
      self._bias = mx.zeros((axis_size,), dtype=self._param_dtype)

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.values.shape)

    values = x.values
    axes = _normalize_axes(self._axis, values.shape)

    # Broadcast running stats over batch and time.
    shape = [1] * len(values.shape)
    shape[axes[0]] = self._running_mean.shape[0]

    mean = self._running_mean.reshape(shape)
    var = self._running_var.reshape(shape)

    normed = (values.astype(mx.float32) - mean) * mx.rsqrt(var + self.epsilon)
    normed = normed.astype(values.dtype)

    if self.use_scale and self._scale is not None:
      normed = normed * self._scale.reshape(shape)
    if self.use_bias and self._bias is not None:
      normed = normed + self._bias.reshape(shape)

    return Sequence(normed, x.mask)

  @classmethod
  def from_config(cls, config):
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    return cls(
        axis=config.axis,
        epsilon=config.epsilon,
        use_bias=config.use_bias,
        use_scale=config.use_scale,
        param_dtype=_to_mx_dtype(config.param_dtype),
    )


class GroupNormalization(types.PreservesType, types.StatelessPointwise):
  """Group Normalization.

  Normalizes per-timestep within each group (not across time), so
  that step() and layer() produce identical results.

  Note: mlx.nn.GroupNorm normalizes across all spatial dims including
  time, which is incompatible with the SequenceLayer step/layer contract.
  """

  def __init__(
      self,
      *,
      num_groups: int,
      axis: int = -1,
      epsilon: float = 1e-6,
      use_bias: bool = True,
      use_scale: bool = True,
      param_dtype=mx.float32,
  ):
    super().__init__()
    if num_groups <= 0:
      raise ValueError(f'{num_groups=} must be positive.')
    self._num_groups = num_groups
    self._axis = axis
    self.epsilon = epsilon
    self.use_bias = use_bias
    self.use_scale = use_scale
    self._param_dtype = param_dtype
    self._scale = None
    self._bias = None

  def _ensure_initialized(self, input_shape):
    if self._scale is not None or self._bias is not None:
      return
    axes = _normalize_axes(self._axis, input_shape)
    axis_size = input_shape[axes[0]]
    if self.use_scale:
      self._scale = mx.ones((axis_size,), dtype=self._param_dtype)
    if self.use_bias:
      self._bias = mx.zeros((axis_size,), dtype=self._param_dtype)

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.values.shape)

    values = x.values
    axes = _normalize_axes(self._axis, values.shape)
    axis = axes[0]
    axis_size = values.shape[axis]

    if axis_size % self._num_groups != 0:
      raise ValueError(
          f'Input axis {axis} size {axis_size} must be'
          f' divisible by {self._num_groups}.'
      )
    group_size = axis_size // self._num_groups

    # Reshape to [... num_groups, group_size ...]
    shape = list(values.shape)
    grouped_shape = (
        shape[:axis] + [self._num_groups, group_size] + shape[axis + 1 :]
    )
    grouped = mx.reshape(values, grouped_shape)

    # Normalize over group_size only (per-timestep).
    g = grouped.astype(mx.float32)
    reduce_axis = axis + 1
    mean = mx.mean(g, axis=reduce_axis, keepdims=True)
    variance = mx.mean(mx.square(g - mean), axis=reduce_axis, keepdims=True)
    normed = (g - mean) * mx.rsqrt(variance + self.epsilon)
    normed = mx.reshape(normed.astype(values.dtype), values.shape)

    # Apply learned scale and bias.
    if self.use_scale and self._scale is not None:
      scale_shape = [1] * len(values.shape)
      scale_shape[axis] = axis_size
      normed = normed * self._scale.reshape(scale_shape)
    if self.use_bias and self._bias is not None:
      bias_shape = [1] * len(values.shape)
      bias_shape[axis] = axis_size
      normed = normed + self._bias.reshape(bias_shape)

    return Sequence(normed, x.mask)

  @classmethod
  def from_config(cls, config):
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    return cls(
        num_groups=config.num_groups,
        axis=config.axis,
        epsilon=config.epsilon,
        use_bias=config.use_bias,
        use_scale=config.use_scale,
        param_dtype=_to_mx_dtype(config.param_dtype),
    )
