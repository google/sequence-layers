"""Normalization layers for MLX."""

import dataclasses
from typing import Any, override
from typing import Sequence as _Sequence

import mlx.core as mx
import mlx.nn as nn

from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import types
from sequence_layers.specs import normalization as spec

Sequence = types.Sequence


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


class L2Normalize(
    types.PreservesType,
    types.StatelessPointwise,
    spec.L2Normalize[types.Sequence, types.ShapeDType],
):
  """L2 normalization over the specified channel axes."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.L2Normalize.Config):
    """Configuration for L2Normalize."""

    axis: int | _Sequence[int] = -1
    epsilon: float = 1e-12
    name: str | None = None

    @override
    def make(self) -> 'L2Normalize':
      return L2Normalize(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool = False, constants=None
  ):
    values = x.values
    axes = _normalize_axes(self.config.axis, values.shape)

    v = values.astype(mx.float32)
    squared_sum = mx.sum(mx.square(v), axis=axes, keepdims=True)
    normed = v * mx.rsqrt(squared_sum + self.config.epsilon)
    return Sequence(normed.astype(values.dtype), x.mask)


class RMSNormalization(
    types.PreservesType,
    types.StatelessPointwise,
    spec.RMSNormalization[types.Sequence, types.ShapeDType],
):
  """RMS Normalization backed by mlx.nn.RMSNorm."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.RMSNormalization.Config):
    """Configuration for RMSNormalization."""

    axis: int | _Sequence[int] = -1
    epsilon: float = 1e-6
    use_scale: bool = True
    scale_init: Any | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    @override
    def make(self) -> 'RMSNormalization':
      return RMSNormalization(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      axis: int | _Sequence[int] = -1,
      epsilon: float = 1e-6,
      use_scale: bool = True,
      scale_init: Any | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = mx.float32,
  ):
    super().__init__()
    if config is not None:
      self.config = config
    else:
      self.config = self.Config(
          axis=axis,
          epsilon=epsilon,
          use_scale=use_scale,
          scale_init=scale_init,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
      )
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    self._param_dtype = _to_mx_dtype(self.config.param_dtype)
    self._scale_init = init_mapping.map_initializer(self.config.scale_init)
    # mlx.nn.RMSNorm created lazily since we need input shape.
    self._rms_norm = None
    self._use_builtin = False

  def _ensure_initialized(self, input_shape):
    """Create internal RMSNorm on first call."""
    if self._rms_norm is not None or not self.config.use_scale:
      return
    axes = _normalize_axes(self.config.axis, input_shape)
    # mlx.nn.RMSNorm only supports normalizing over the last dim.
    if axes == (len(input_shape) - 1,) and self._scale_init is None:
      dims = input_shape[-1]
      self._rms_norm = nn.RMSNorm(dims, eps=self.config.epsilon)
      self._use_builtin = True
    else:
      # Multi-axis or custom init: manual scale parameter.
      scale_shape = tuple(input_shape[a] for a in axes)
      if self._scale_init is not None:
        key = mx.random.key(0)
        self._scale = self._scale_init(key, scale_shape, self._param_dtype)
      else:
        self._scale = mx.ones(scale_shape, dtype=self._param_dtype)

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool = False, constants=None
  ):
    self._ensure_initialized(x.values.shape)

    if self._use_builtin and self._rms_norm is not None:
      # Cast back to input dtype to preserve bfloat16 compute.
      result = self._rms_norm(x.values).astype(x.values.dtype)
      return Sequence(result, x.mask)

    values = x.values
    axes = _normalize_axes(self.config.axis, values.shape)

    # Manual RMS norm in float32.
    v = values.astype(mx.float32)
    mean_sq = mx.mean(mx.square(v), axis=axes, keepdims=True)
    normed = v * mx.rsqrt(mean_sq + self.config.epsilon)
    normed = normed.astype(values.dtype)

    # Apply learned scale.
    if self.config.use_scale:
      scale = self._scale.astype(normed.dtype)
      shape = [1] * len(values.shape)
      for i, a in enumerate(axes):
        shape[a] = self._scale.shape[i]
      scale = scale.reshape(shape)
      normed = normed * scale

    return Sequence(normed, x.mask)


class LayerNormalization(
    types.PreservesType,
    types.StatelessPointwise,
    spec.LayerNormalization[types.Sequence, types.ShapeDType],
):
  """Layer Normalization backed by mlx.nn.LayerNorm."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.LayerNormalization.Config):
    """Configuration for LayerNormalization."""

    axis: int | _Sequence[int] = -1
    epsilon: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    reductions_in_at_least_fp32: bool = True
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    @override
    def make(self) -> 'LayerNormalization':
      return LayerNormalization(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    self._param_dtype = _to_mx_dtype(config.param_dtype)
    self._layer_norm = None
    self._use_builtin = False
    self._manual_scale = None
    self._manual_bias = None

  def _ensure_initialized(self, input_shape):
    if self._layer_norm is not None or self._manual_scale is not None:
      return
    if not self.config.use_scale and not self.config.use_bias:
      return
    axes = _normalize_axes(self.config.axis, input_shape)
    # mlx.nn.LayerNorm supports a single last-dim normalization.
    if axes == (len(input_shape) - 1,):
      dims = input_shape[-1]
      self._layer_norm = nn.LayerNorm(
          dims,
          eps=self.config.epsilon,
          affine=self.config.use_scale or self.config.use_bias,
          bias=self.config.use_bias,
      )
      self._use_builtin = True
    else:
      # Multi-axis: manual parameters.
      scale_shape = tuple(input_shape[a] for a in axes)
      if self.config.use_scale:
        self._manual_scale = mx.ones(scale_shape, dtype=self._param_dtype)
      if self.config.use_bias:
        self._manual_bias = mx.zeros(scale_shape, dtype=self._param_dtype)

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool = False, constants=None
  ):
    self._ensure_initialized(x.values.shape)

    if self._use_builtin and self._layer_norm is not None:
      x_values = x.values
      original_dtype = x_values.dtype
      if self.config.reductions_in_at_least_fp32:
        x_values = x_values.astype(mx.float32)
      # Cast back to input dtype to preserve bfloat16 compute.
      result = self._layer_norm(x_values).astype(original_dtype)
      return Sequence(result, x.mask)

    values = x.values
    axes = _normalize_axes(self.config.axis, values.shape)

    # Manual layer norm in float32.
    v = values.astype(mx.float32)
    mean = mx.mean(v, axis=axes, keepdims=True)
    variance = mx.mean(mx.square(v - mean), axis=axes, keepdims=True)
    normed = (v - mean) * mx.rsqrt(variance + self.config.epsilon)
    normed = normed.astype(values.dtype)

    # Apply learned scale and bias.
    if self.config.use_scale and self._manual_scale is not None:
      scale = self._manual_scale.astype(normed.dtype)
      shape = [1] * len(values.shape)
      for i, a in enumerate(axes):
        shape[a] = self._manual_scale.shape[i]
      normed = normed * scale.reshape(shape)

    if self.config.use_bias and self._manual_bias is not None:
      bias = self._manual_bias.astype(normed.dtype)
      shape = [1] * len(values.shape)
      for i, a in enumerate(axes):
        shape[a] = self._manual_bias.shape[i]
      normed = normed + bias.reshape(shape)

    return Sequence(normed, x.mask)


class BatchNormalization(
    types.PreservesType,
    types.StatelessPointwise,
    spec.BatchNormalization[types.Sequence, types.ShapeDType],
):
  """Batch Normalization (inference-only)."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.BatchNormalization.Config):
    """Configuration for BatchNormalization."""

    axis: int | _Sequence[int] = -1
    epsilon: float = 1e-5
    momentum: float = 0.99
    use_scale: bool = True
    use_bias: bool = True
    use_fast_variance: bool = True
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'BatchNormalization':
      return BatchNormalization(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    self._param_dtype = _to_mx_dtype(config.param_dtype)
    self._running_mean = None
    self._running_var = None
    self._scale = None
    self._bias = None

  def _ensure_initialized(self, input_shape):
    if self._running_mean is not None:
      return
    axes = _normalize_axes(self.config.axis, input_shape)
    axis_size = input_shape[axes[0]]
    self._running_mean = mx.zeros((axis_size,), dtype=self._param_dtype)
    self._running_var = mx.ones((axis_size,), dtype=self._param_dtype)
    if self.config.use_scale:
      self._scale = mx.ones((axis_size,), dtype=self._param_dtype)
    if self.config.use_bias:
      self._bias = mx.zeros((axis_size,), dtype=self._param_dtype)

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool = False, constants=None
  ):
    self._ensure_initialized(x.values.shape)
    assert self._running_mean is not None
    assert self._running_var is not None

    values = x.values
    axes = _normalize_axes(self.config.axis, values.shape)

    # Broadcast running stats over batch and time.
    shape = [1] * len(values.shape)
    shape[axes[0]] = self._running_mean.shape[0]

    mean = self._running_mean.reshape(shape)
    var = self._running_var.reshape(shape)

    normed = (values.astype(mx.float32) - mean) * mx.rsqrt(
        var + self.config.epsilon
    )
    normed = normed.astype(values.dtype)

    if self.config.use_scale and self._scale is not None:
      normed = normed * self._scale.reshape(shape)
    if self.config.use_bias and self._bias is not None:
      normed = normed + self._bias.reshape(shape)

    return Sequence(normed, x.mask)


class GroupNormalization(
    types.PreservesType,
    types.StatelessPointwise,
    spec.GroupNormalization[types.Sequence, types.ShapeDType],
):
  """Group Normalization.

  Normalizes per-timestep within each group (not across time), so
  that step() and layer() produce identical results.

  Note: mlx.nn.GroupNorm normalizes across all spatial dims including
  time, which is incompatible with the SequenceLayer step/layer contract.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.GroupNormalization.Config):
    """Configuration for GroupNormalization."""

    num_groups: int
    axis: int | _Sequence[int] = -1
    epsilon: float = 1e-6
    cumulative: bool = False
    use_scale: bool = True
    use_bias: bool = True
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'GroupNormalization':
      if self.num_groups <= 0:
        raise ValueError(f'{self.num_groups=} must be positive.')
      return GroupNormalization(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    from sequence_layers.mlx.init_mapping import _to_mx_dtype

    self._param_dtype = _to_mx_dtype(config.param_dtype)
    self._scale = None
    self._bias = None

  def _ensure_initialized(self, input_shape):
    if self._scale is not None or self._bias is not None:
      return
    axes = _normalize_axes(self.config.axis, input_shape)
    axis_size = input_shape[axes[0]]
    if self.config.use_scale:
      self._scale = mx.ones((axis_size,), dtype=self._param_dtype)
    if self.config.use_bias:
      self._bias = mx.zeros((axis_size,), dtype=self._param_dtype)

  @override
  @types.check_layer
  def layer(  # pyrefly: ignore[missing-override-decorator]
      self, x, *, training: bool = False, constants=None
  ):
    self._ensure_initialized(x.values.shape)

    values = x.values
    axes = _normalize_axes(self.config.axis, values.shape)
    axis = axes[0]
    axis_size = values.shape[axis]

    if axis_size % self.config.num_groups != 0:
      raise ValueError(
          f'Input axis {axis} size {axis_size} must be'
          f' divisible by {self.config.num_groups}.'
      )
    group_size = axis_size // self.config.num_groups

    # Reshape to [... num_groups, group_size ...]
    shape = list(values.shape)
    grouped_shape = (
        shape[:axis] + [self.config.num_groups, group_size] + shape[axis + 1 :]
    )
    grouped = mx.reshape(values, grouped_shape)

    # Normalize over group_size only (per-timestep).
    g = grouped.astype(mx.float32)
    reduce_axis = axis + 1
    mean = mx.mean(g, axis=reduce_axis, keepdims=True)
    variance = mx.mean(mx.square(g - mean), axis=reduce_axis, keepdims=True)
    normed = (g - mean) * mx.rsqrt(variance + self.config.epsilon)
    normed = mx.reshape(normed.astype(values.dtype), values.shape)

    # Apply learned scale and bias.
    if self.config.use_scale and self._scale is not None:
      scale_shape = [1] * len(values.shape)
      scale_shape[axis] = axis_size
      normed = normed * self._scale.reshape(scale_shape)
    if self.config.use_bias and self._bias is not None:
      bias_shape = [1] * len(values.shape)
      bias_shape[axis] = axis_size
      normed = normed + self._bias.reshape(bias_shape)

    return Sequence(normed, x.mask)
