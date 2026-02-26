"""2D Convolution, transpose convolution, pooling, and upsampling layers for MLX."""

import dataclasses
import fractions
import math

import mlx.core as mx
import mlx.nn as nn

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import convolution as conv_utils
from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence
PaddingMode = bt.PaddingMode


def _normalize_2tuple(x):
  """Normalizes an int or sequence to a 2-tuple."""
  if isinstance(x, int):
    return (x, x)
  return tuple(x)


def _explicit_padding_2d(padding, kernel_size, stride, dilation_rate):
  """Returns ((pad_time_left, pad_time_right), (pad_spatial_left, pad_spatial_right))."""
  time_pad = conv_utils._explicit_padding(
      padding[0] if isinstance(padding, (list, tuple)) else padding,
      kernel_size[0], stride[0], dilation_rate[0],
  )
  spatial_padding = padding[1] if isinstance(padding, (list, tuple)) else padding
  spatial_pad = conv_utils._explicit_padding(
      spatial_padding, kernel_size[1], stride[1], dilation_rate[1],
  )
  return time_pad, spatial_pad


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------


class Conv2D(types.SequenceLayer):
  """2D convolution layer with separate time and spatial padding."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    filters: int = 1
    kernel_size: tuple[int, int] = (1, 1)
    strides: tuple[int, int] = (1, 1)
    dilation_rate: tuple[int, int] = (1, 1)
    time_padding: str = 'valid'
    spatial_padding: str | tuple[int, int] = 'same'
    groups: int = 1
    use_bias: bool = True
    activation: object = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'kernel_size', _normalize_2tuple(self.kernel_size))
      object.__setattr__(self, 'strides', _normalize_2tuple(self.strides))
      object.__setattr__(self, 'dilation_rate', _normalize_2tuple(self.dilation_rate))
      if isinstance(self.spatial_padding, str):
        pass  # Keep as string.
      else:
        object.__setattr__(self, 'spatial_padding', tuple(self.spatial_padding))

    def make(self) -> 'Conv2D':
      return Conv2D.from_config(self)

  def __init__(
      self,
      *,
      in_features,
      filters,
      kernel_size,
      strides=(1, 1),
      dilation_rate=(1, 1),
      time_padding='valid',
      spatial_padding='same',
      groups=1,
      use_bias=True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self.in_features = in_features
    self.filters = filters
    self.kernel_size = _normalize_2tuple(kernel_size)
    self.strides = _normalize_2tuple(strides)
    self.dilation_rate = _normalize_2tuple(dilation_rate)
    self.time_padding = time_padding
    self.spatial_padding = spatial_padding
    self.groups = groups
    self.use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    # Create kernel: [out_channels, kH, kW, in_channels // groups]
    key = mx.random.key(0)
    init_fn = init_mapping._make_variance_scaling_init('fan_in', 'truncated_normal')
    self.kernel = init_fn(
        key,
        (filters, self.kernel_size[0], self.kernel_size[1], in_features // groups),
        param_dtype,
    )
    if use_bias:
      self.bias = mx.zeros((filters,), dtype=param_dtype)

  @property
  def supports_step(self):
    return conv_utils._supports_step(self.time_padding)

  @property
  def block_size(self):
    return self.strides[0]

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self.strides[0])

  @property
  def input_latency(self):
    ek = conv_utils._effective_kernel_size(self.kernel_size[0], self.dilation_rate[0])
    if self.time_padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self.time_padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
    ):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 2:
      raise ValueError(
          f'Conv2D requires rank 4 input. Got channel_shape={input_shape}'
      )
    freq_dim = input_shape[0]
    # Compute spatial output size.
    if isinstance(self.spatial_padding, str):
      sp_pad = conv_utils._explicit_padding(
          self.spatial_padding, self.kernel_size[1],
          self.strides[1], self.dilation_rate[1],
      )
    else:
      sp_pad = self.spatial_padding
    ek_sp = conv_utils._effective_kernel_size(self.kernel_size[1], self.dilation_rate[1])
    out_freq = (freq_dim + sp_pad[0] + sp_pad[1] - ek_sp) // self.strides[1] + 1
    return (out_freq, self.filters)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.compute_dtype or self._param_dtype

  def _forward(self, values, time_pad, spatial_pad):
    """Apply 2D conv with explicit padding."""
    if time_pad[0] > 0 or time_pad[1] > 0 or spatial_pad[0] > 0 or spatial_pad[1] > 0:
      values = mx.pad(
          values,
          [(0, 0), (time_pad[0], time_pad[1]), (spatial_pad[0], spatial_pad[1]), (0, 0)],
      )
    compute_dtype = self.compute_dtype or self._param_dtype
    values = values.astype(compute_dtype)
    # mlx.core.conv2d: input [B, H, W, C_in], weight [C_out, kH, kW, C_in/groups]
    y = mx.conv2d(
        values,
        self.kernel.astype(compute_dtype),
        stride=self.strides,
        padding=0,
        dilation=self.dilation_rate,
        groups=self.groups,
    )
    if self.use_bias:
      y = y + self.bias.astype(compute_dtype)
    if self.activation is not None:
      y = self.activation(y)
    return y

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    bw = conv_utils._buffer_width(
        self.time_padding,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    if not bw:
      return ()
    # State is a MaskedSequence of shape [B, bw, freq, channels].
    freq_dim = input_spec.shape[0]
    channels = input_spec.shape[1] if len(input_spec.shape) > 1 else 1
    if self.time_padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL_VALID.value,
    ):
      mask = mx.ones((batch_size, bw), dtype=bt.MASK_DTYPE)
    else:
      mask = mx.zeros((batch_size, bw), dtype=bt.MASK_DTYPE)
    values = mx.zeros(
        (batch_size, bw) + input_spec.shape,
        dtype=input_spec.dtype,
    )
    return MaskedSequence(values, mask)

  @types.check_step
  def step(self, x, state, *, constants=None):
    ek_time = conv_utils._effective_kernel_size(self.kernel_size[0], self.dilation_rate[0])
    if ek_time > 1:
      x = x.mask_invalid()

    bw = conv_utils._buffer_width(
        self.time_padding,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )

    if bw:
      state = state.concatenate(x)
    else:
      state = x

    # Spatial padding always applied; time padding from buffer.
    if isinstance(self.spatial_padding, str):
      sp_pad = conv_utils._explicit_padding(
          self.spatial_padding, self.kernel_size[1],
          self.strides[1], self.dilation_rate[1],
      )
    else:
      sp_pad = self.spatial_padding

    values = self._forward(state.values, (0, 0), sp_pad)
    mask = conv_utils._compute_conv_mask(
        state.mask,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
        self.time_padding,
        is_step=True,
    )

    if bw:
      state = state[:, -bw:]
    else:
      state = ()

    return Sequence(values, mask), state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self.kernel_size[0] > 1:
      x = x.mask_invalid()

    time_pad = conv_utils._explicit_padding(
        self.time_padding,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    if isinstance(self.spatial_padding, str):
      sp_pad = conv_utils._explicit_padding(
          self.spatial_padding, self.kernel_size[1],
          self.strides[1], self.dilation_rate[1],
      )
    else:
      sp_pad = self.spatial_padding

    values = self._forward(x.values, time_pad, sp_pad)
    mask = conv_utils._compute_conv_mask(
        x.mask,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
        self.time_padding,
        is_step=False,
    )
    return Sequence(values, mask)

  @classmethod
  def from_config(cls, config):
    compute_dtype = getattr(config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    activation = init_mapping.map_activation(getattr(config, 'activation', None))
    spatial_padding = config.spatial_padding
    if isinstance(spatial_padding, str):
      pass
    else:
      spatial_padding = tuple(spatial_padding)
    return DeferredConv2D(config)


class DeferredConv2D(types.SequenceLayer):
  """Deferred Conv2D: delays kernel creation until first use."""

  def __init__(self, config):
    super().__init__()
    self._config = config
    self.inner = None

  def _ensure_built(self, input_shape):
    if self.inner is not None:
      return
    in_features = input_shape[-1]
    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    activation = init_mapping.map_activation(getattr(self._config, 'activation', None))
    spatial_padding = self._config.spatial_padding
    if isinstance(spatial_padding, str):
      pass
    else:
      spatial_padding = tuple(spatial_padding)

    self.inner = Conv2D(
        in_features=in_features,
        filters=self._config.filters,
        kernel_size=_normalize_2tuple(self._config.kernel_size),
        strides=_normalize_2tuple(self._config.strides),
        dilation_rate=_normalize_2tuple(getattr(self._config, 'dilation_rate', (1, 1))),
        time_padding=getattr(self._config, 'time_padding', 'valid'),
        spatial_padding=spatial_padding,
        groups=getattr(self._config, 'groups', 1),
        use_bias=getattr(self._config, 'use_bias', True),
        activation=activation,
        compute_dtype=compute_dtype,
        param_dtype=init_mapping._to_mx_dtype(self._config.param_dtype),
    )

  @property
  def supports_step(self):
    return conv_utils._supports_step(
        getattr(self._config, 'time_padding', 'valid')
    )

  @property
  def block_size(self):
    return _normalize_2tuple(self._config.strides)[0]

  @property
  def output_ratio(self):
    return fractions.Fraction(1, _normalize_2tuple(self._config.strides)[0])

  @property
  def input_latency(self):
    ks = _normalize_2tuple(self._config.kernel_size)
    dr = _normalize_2tuple(getattr(self._config, 'dilation_rate', (1, 1)))
    tp = getattr(self._config, 'time_padding', 'valid')
    ek = conv_utils._effective_kernel_size(ks[0], dr[0])
    if tp in (PaddingMode.CAUSAL_VALID.value, PaddingMode.CAUSAL.value,
              PaddingMode.SEMICAUSAL.value):
      return 0
    elif tp in (PaddingMode.REVERSE_CAUSAL_VALID.value,
                PaddingMode.REVERSE_CAUSAL.value):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    self._ensure_built(input_shape)
    return self.inner.get_output_shape(input_shape, constants=constants)

  def get_output_dtype(self, input_dtype, *, constants=None):
    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      return init_mapping._to_mx_dtype(compute_dtype)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    self._ensure_built(input_spec.shape)
    return self.inner.get_initial_state(batch_size, input_spec, constants=constants)

  @types.check_step
  def step(self, x, state, *, constants=None):
    self._ensure_built(x.channel_shape)
    return self.inner.step(x, state, constants=constants)

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._ensure_built(x.channel_shape)
    return self.inner.layer(x, constants=constants)


# ---------------------------------------------------------------------------
# Conv2DTranspose
# ---------------------------------------------------------------------------


class Conv2DTranspose(types.SequenceLayer):
  """2D transposed convolution layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    filters: int = 1
    kernel_size: tuple[int, int] = (1, 1)
    strides: tuple[int, int] = (1, 1)
    dilation_rate: tuple[int, int] = (1, 1)
    time_padding: str = 'valid'
    spatial_padding: str | tuple[int, int] = 'same'
    groups: int = 1
    use_bias: bool = True
    activation: object = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'kernel_size', _normalize_2tuple(self.kernel_size))
      object.__setattr__(self, 'strides', _normalize_2tuple(self.strides))
      object.__setattr__(self, 'dilation_rate', _normalize_2tuple(self.dilation_rate))

    def make(self) -> 'Conv2DTranspose':
      return Conv2DTranspose.from_config(self)

  def __init__(
      self,
      *,
      in_features,
      filters,
      kernel_size,
      strides=(1, 1),
      dilation_rate=(1, 1),
      time_padding='valid',
      spatial_padding='same',
      groups=1,
      use_bias=True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self.in_features = in_features
    self.filters = filters
    self.kernel_size = _normalize_2tuple(kernel_size)
    self.strides = _normalize_2tuple(strides)
    self.dilation_rate = _normalize_2tuple(dilation_rate)
    self.time_padding = time_padding
    self.spatial_padding = spatial_padding
    self.groups = groups
    self.use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    # Kernel: [out_channels, kH, kW, in_channels // groups]
    key = mx.random.key(0)
    init_fn = init_mapping._make_variance_scaling_init('fan_in', 'truncated_normal')
    self.kernel = init_fn(
        key,
        (filters, self.kernel_size[0], self.kernel_size[1], in_features // groups),
        param_dtype,
    )
    if use_bias:
      self.bias = mx.zeros((filters,), dtype=param_dtype)

  @property
  def supports_step(self):
    return self.time_padding == PaddingMode.CAUSAL.value

  @property
  def block_size(self):
    return 1

  @property
  def output_ratio(self):
    return fractions.Fraction(self.strides[0])

  @property
  def input_latency(self):
    return 0

  def _time_trim(self):
    """Returns (trim_left, trim_right) for time dimension."""
    return conv_utils._transpose_conv_output_trim(
        self.kernel_size[0], self.strides[0],
        self.dilation_rate[0], self.time_padding,
    )

  def _spatial_trim(self):
    """Returns (trim_left, trim_right) for spatial dimension."""
    if isinstance(self.spatial_padding, str):
      return conv_utils._transpose_conv_output_trim(
          self.kernel_size[1], self.strides[1],
          self.dilation_rate[1], self.spatial_padding,
      )
    else:
      return self.spatial_padding

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 2:
      raise ValueError(
          f'Conv2DTranspose requires rank 4 input. Got channel_shape={input_shape}'
      )
    freq_dim = input_shape[0]
    ek_sp = conv_utils._effective_kernel_size(self.kernel_size[1], self.dilation_rate[1])
    raw_sp = (freq_dim - 1) * self.strides[1] + ek_sp
    sp_trim = self._spatial_trim()
    out_freq = raw_sp - sp_trim[0] - sp_trim[1]
    return (out_freq, self.filters)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.compute_dtype or self._param_dtype

  def _conv_raw(self, values, trim_time=True):
    """Compute raw conv_transpose2d, optionally trimming time.

    Args:
      values: Input values.
      trim_time: If True, trim time dimension (for layer mode).
                 If False, skip time trim (for step mode overlap-add).
    Returns:
      Raw convolution output WITHOUT bias or activation.
    """
    compute_dtype = self.compute_dtype or self._param_dtype
    values = values.astype(compute_dtype)
    # mx.conv_transpose2d: input [B, H, W, C_in], weight [C_out, kH, kW, C_in/groups]
    y = mx.conv_transpose2d(
        values,
        self.kernel.astype(compute_dtype),
        stride=self.strides,
        padding=0,
        dilation=self.dilation_rate,
        groups=self.groups,
    )
    # Time trim (only in layer mode; step mode handles it via overlap-add).
    if trim_time:
      tl, tr = self._time_trim()
      if tl > 0:
        y = y[:, tl:]
      if tr > 0:
        y = y[:, :-tr]
    # Spatial trim (always applied).
    sl_val, sr = self._spatial_trim()
    if sl_val > 0:
      y = y[:, :, sl_val:]
    if sr > 0:
      y = y[:, :, :-sr]
    return y

  def _apply_bias_and_activation(self, y):
    """Apply bias and activation to conv output."""
    compute_dtype = self.compute_dtype or self._param_dtype
    if self.use_bias:
      y = y + self.bias.astype(compute_dtype)
    if self.activation is not None:
      y = self.activation(y)
    return y

  def _forward(self, values):
    """Full forward: conv + trim + bias + activation (for layer mode)."""
    y = self._conv_raw(values, trim_time=True)
    return self._apply_bias_and_activation(y)

  @types.check_layer
  def layer(self, x, *, constants=None):
    values = self._forward(x.values)
    mask = conv_utils._compute_conv_transpose_mask(
        x.mask,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
        self.time_padding,
    )
    return Sequence(values, mask)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if not self.supports_step:
      raise ValueError('Conv2DTranspose step only supported with causal padding.')
    ola_buf = max(
        0,
        conv_utils._effective_kernel_size(self.kernel_size[0], self.dilation_rate[0])
        - self.strides[0],
    )
    if not ola_buf:
      return ()
    out_shape = self.get_output_shape(input_spec.shape, constants=constants)
    values = mx.zeros((batch_size, ola_buf) + out_shape, dtype=self.get_output_dtype(input_spec.dtype))
    mask = mx.zeros((batch_size, ola_buf), dtype=bt.MASK_DTYPE)
    return MaskedSequence(values, mask)

  @types.check_step
  def step(self, x, state, *, constants=None):
    x = x.mask_invalid()
    # Conv WITHOUT time trimming — keep full temporal output for overlap-add.
    # Bias is also deferred until after overlap-add (matching JAX behavior).
    raw = self._conv_raw(x.values, trim_time=False)
    input_time = x.shape[1]
    out_time = input_time * self.strides[0]
    mask = mx.repeat(x.mask, self.strides[0], axis=1)

    ola_buf = max(
        0,
        conv_utils._effective_kernel_size(self.kernel_size[0], self.dilation_rate[0])
        - self.strides[0],
    )
    if ola_buf:
      # Pad the state buffer to match the raw output length, then overlap-add.
      # raw has shape (B, raw_time, ...) where raw_time >= out_time + ola_buf
      buf_values = state.values  # (B, ola_buf, ...)
      pad_len = raw.shape[1] - ola_buf
      if pad_len > 0:
        buf_values = mx.concatenate(
            [buf_values, mx.zeros_like(raw[:, :pad_len])], axis=1
        )
      # Overlap-add: add state to raw output.
      out_values = buf_values + raw
      # Split: first out_time samples are output, rest is new buffer.
      out = out_values[:, :out_time]
      new_buf = out_values[:, out_time:]
      if new_buf.shape[1] < ola_buf:
        pad_width = ola_buf - new_buf.shape[1]
        new_buf = mx.pad(new_buf, [(0, 0), (0, pad_width)] + [(0, 0)] * (new_buf.ndim - 2))
      elif new_buf.shape[1] > ola_buf:
        new_buf = new_buf[:, :ola_buf]
      new_mask = mx.zeros((x.values.shape[0], ola_buf), dtype=bt.MASK_DTYPE)
      state = MaskedSequence(new_buf, new_mask)
    else:
      out = raw[:, :out_time]
      state = ()

    # Apply bias and activation AFTER overlap-add (only once per sample).
    out = self._apply_bias_and_activation(out)

    out_mask = mask[:, :out.shape[1]]
    return Sequence(out, out_mask), state

  @classmethod
  def from_config(cls, config):
    return DeferredConv2DTranspose(config)


class DeferredConv2DTranspose(types.SequenceLayer):
  """Deferred Conv2DTranspose."""

  def __init__(self, config):
    super().__init__()
    self._config = config
    self.inner = None

  def _ensure_built(self, input_shape):
    if self.inner is not None:
      return
    in_features = input_shape[-1]
    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    activation = init_mapping.map_activation(getattr(self._config, 'activation', None))
    spatial_padding = getattr(self._config, 'spatial_padding', 'same')
    self.inner = Conv2DTranspose(
        in_features=in_features,
        filters=self._config.filters,
        kernel_size=_normalize_2tuple(self._config.kernel_size),
        strides=_normalize_2tuple(self._config.strides),
        dilation_rate=_normalize_2tuple(getattr(self._config, 'dilation_rate', (1, 1))),
        time_padding=getattr(self._config, 'time_padding', 'valid'),
        spatial_padding=spatial_padding,
        groups=getattr(self._config, 'groups', 1),
        use_bias=getattr(self._config, 'use_bias', True),
        activation=activation,
        compute_dtype=compute_dtype,
        param_dtype=init_mapping._to_mx_dtype(self._config.param_dtype),
    )

  @property
  def supports_step(self):
    return getattr(self._config, 'time_padding', 'valid') == PaddingMode.CAUSAL.value

  @property
  def block_size(self):
    return 1

  @property
  def output_ratio(self):
    return fractions.Fraction(_normalize_2tuple(self._config.strides)[0])

  @property
  def input_latency(self):
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    self._ensure_built(input_shape)
    return self.inner.get_output_shape(input_shape, constants=constants)

  def get_output_dtype(self, input_dtype, *, constants=None):
    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      return init_mapping._to_mx_dtype(compute_dtype)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    self._ensure_built(input_spec.shape)
    return self.inner.get_initial_state(batch_size, input_spec, constants=constants)

  @types.check_step
  def step(self, x, state, *, constants=None):
    self._ensure_built(x.channel_shape)
    return self.inner.step(x, state, constants=constants)

  @types.check_layer
  def layer(self, x, *, constants=None):
    self._ensure_built(x.channel_shape)
    return self.inner.layer(x, constants=constants)


# ---------------------------------------------------------------------------
# AveragePooling2D
# ---------------------------------------------------------------------------


class AveragePooling2D(types.SequenceLayer):
  """2D average pooling with separate time and spatial padding."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    pool_size: tuple[int, int] = (1, 1)
    strides: tuple[int, int] = (1, 1)
    dilation_rate: tuple[int, int] = (1, 1)
    time_padding: str = 'valid'
    spatial_padding: str | tuple[int, int] = 'same'
    masked_average: bool = False
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'pool_size', _normalize_2tuple(self.pool_size))
      object.__setattr__(self, 'strides', _normalize_2tuple(self.strides))
      object.__setattr__(self, 'dilation_rate', _normalize_2tuple(self.dilation_rate))

    def make(self) -> 'AveragePooling2D':
      return AveragePooling2D.from_config(self)

  def __init__(
      self,
      *,
      pool_size,
      strides=(1, 1),
      dilation_rate=(1, 1),
      time_padding='valid',
      spatial_padding='same',
      masked_average=False,
  ):
    super().__init__()
    self.pool_size = _normalize_2tuple(pool_size)
    self.strides = _normalize_2tuple(strides)
    self.dilation_rate = _normalize_2tuple(dilation_rate)
    self.time_padding = time_padding
    self.spatial_padding = spatial_padding
    self.masked_average = masked_average

  @property
  def supports_step(self):
    return conv_utils._supports_step(self.time_padding)

  @property
  def block_size(self):
    return self.strides[0]

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self.strides[0])

  @property
  def input_latency(self):
    ek = conv_utils._effective_kernel_size(self.pool_size[0], self.dilation_rate[0])
    if self.time_padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self.time_padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
    ):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 2:
      raise ValueError(
          f'AveragePooling2D requires rank 4 input. Got channel_shape={input_shape}'
      )
    freq_dim = input_shape[0]
    if isinstance(self.spatial_padding, str):
      sp_pad = conv_utils._explicit_padding(
          self.spatial_padding, self.pool_size[1],
          self.strides[1], self.dilation_rate[1],
      )
    else:
      sp_pad = self.spatial_padding
    ek_sp = conv_utils._effective_kernel_size(self.pool_size[1], self.dilation_rate[1])
    out_freq = (freq_dim + sp_pad[0] + sp_pad[1] - ek_sp) // self.strides[1] + 1
    return (out_freq, input_shape[1])

  def get_output_dtype(self, input_dtype, *, constants=None):
    return input_dtype

  def _pool(self, values, time_pad, spatial_pad):
    """Apply 2D average pooling with explicit padding."""
    if time_pad[0] > 0 or time_pad[1] > 0 or spatial_pad[0] > 0 or spatial_pad[1] > 0:
      values = mx.pad(
          values,
          [(0, 0), (time_pad[0], time_pad[1]), (spatial_pad[0], spatial_pad[1]), (0, 0)],
      )
    # Implement average pooling via im2col-style approach.
    # For simplicity, use a strided mean.
    b, t, h, c = values.shape
    pt, ps = self.pool_size
    st, ss = self.strides
    out_t = (t - pt) // st + 1
    out_h = (h - ps) // ss + 1
    # Extract patches and average.
    result = mx.zeros((b, out_t, out_h, c), dtype=values.dtype)
    patches = []
    for dt in range(pt):
      for ds in range(ps):
        patch = values[:, dt:dt + out_t * st:st, ds:ds + out_h * ss:ss, :]
        patches.append(patch)
    result = sum(patches) / len(patches)
    return result

  @types.check_layer
  def layer(self, x, *, constants=None):
    time_pad = conv_utils._explicit_padding(
        self.time_padding,
        self.pool_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    if isinstance(self.spatial_padding, str):
      sp_pad = conv_utils._explicit_padding(
          self.spatial_padding, self.pool_size[1],
          self.strides[1], self.dilation_rate[1],
      )
    else:
      sp_pad = self.spatial_padding

    values = self._pool(x.values, time_pad, sp_pad)
    mask = conv_utils._compute_conv_mask(
        x.mask,
        self.pool_size[0],
        self.strides[0],
        self.dilation_rate[0],
        self.time_padding,
        is_step=False,
    )
    return Sequence(values, mask)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    bw = conv_utils._buffer_width(
        self.time_padding,
        self.pool_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    if not bw:
      return ()
    if self.time_padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL_VALID.value,
    ):
      mask = mx.ones((batch_size, bw), dtype=bt.MASK_DTYPE)
    else:
      mask = mx.zeros((batch_size, bw), dtype=bt.MASK_DTYPE)
    values = mx.zeros(
        (batch_size, bw) + input_spec.shape,
        dtype=input_spec.dtype,
    )
    return MaskedSequence(values, mask)

  @types.check_step
  def step(self, x, state, *, constants=None):
    bw = conv_utils._buffer_width(
        self.time_padding,
        self.pool_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    if bw:
      state = state.concatenate(x)
    else:
      state = x

    if isinstance(self.spatial_padding, str):
      sp_pad = conv_utils._explicit_padding(
          self.spatial_padding, self.pool_size[1],
          self.strides[1], self.dilation_rate[1],
      )
    else:
      sp_pad = self.spatial_padding

    values = self._pool(state.values, (0, 0), sp_pad)
    mask = conv_utils._compute_conv_mask(
        state.mask,
        self.pool_size[0],
        self.strides[0],
        self.dilation_rate[0],
        self.time_padding,
        is_step=True,
    )

    if bw:
      state = state[:, -bw:]
    else:
      state = ()

    return Sequence(values, mask), state

  @classmethod
  def from_config(cls, config):
    pool_size = _normalize_2tuple(config.pool_size)
    strides = _normalize_2tuple(config.strides)
    dilation_rate = _normalize_2tuple(getattr(config, 'dilation_rate', (1, 1)))
    return cls(
        pool_size=pool_size,
        strides=strides,
        dilation_rate=dilation_rate,
        time_padding=getattr(config, 'time_padding', 'valid'),
        spatial_padding=getattr(config, 'spatial_padding', 'same'),
        masked_average=getattr(config, 'masked_average', False),
    )


# ---------------------------------------------------------------------------
# Upsample2D
# ---------------------------------------------------------------------------


class Upsample2D(types.PreservesType, types.Stateless):
  """2D upsampling layer using nearest-neighbor repetition."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    rate: tuple[int, int] = (1, 1)
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'rate', _normalize_2tuple(self.rate))

    def make(self) -> 'Upsample2D':
      return Upsample2D.from_config(self)

  def __init__(self, *, rate):
    super().__init__()
    self._rate = _normalize_2tuple(rate)

  @property
  def output_ratio(self):
    return fractions.Fraction(self._rate[0])

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 2:
      raise ValueError(
          f'Upsample2D requires rank 4 input, got channel_shape={input_shape}'
      )
    return (input_shape[0] * self._rate[1], input_shape[1])

  @types.check_layer
  def layer(self, x, *, constants=None):
    values = mx.repeat(x.values, self._rate[0], axis=1)
    values = mx.repeat(values, self._rate[1], axis=2)
    mask = mx.repeat(x.mask, self._rate[0], axis=1)
    return type(x)(values, mask)

  @classmethod
  def from_config(cls, config):
    return cls(rate=_normalize_2tuple(config.rate))


# ---------------------------------------------------------------------------
# ParallelChannels
# ---------------------------------------------------------------------------


class ParallelChannels(types.Emitting):
  """Applies a layer with shared parameters to groups of input channels.

  The input sequence is split on its final channels dimension into num_groups
  separate sequences and processed with the child layer. Parameters for the
  child layer are shared across all parallel invocations.
  """

  # CombinationMode values matching the JAX utils version.
  STACK = 1
  CONCAT = 2

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    child_layer: _SequenceLayerConfig = None
    num_groups: int = 1
    combination: object = None  # CombinationMode enum value
    name: str | None = None

    def make(self, backend='mlx') -> 'ParallelChannels':
      return ParallelChannels.from_config(self, backend=backend)

  def __init__(self, *, child_layer, num_groups, combination=CONCAT):
    super().__init__()
    self.child = child_layer
    self._num_groups = num_groups
    # Default to CONCAT (2) which is what soundstream uses.
    if combination is None:
      self._combination = self.CONCAT
    elif hasattr(combination, 'value'):
      self._combination = combination.value
    else:
      self._combination = int(combination)

  @property
  def supports_step(self):
    return self.child.supports_step

  @property
  def block_size(self):
    return self.child.block_size

  @property
  def output_ratio(self):
    return self.child.output_ratio

  @property
  def input_latency(self):
    return self.child.input_latency

  def _split(self, x):
    """Split sequence along last channel dim into num_groups."""
    vals = x.values
    c = vals.shape[-1]
    if c % self._num_groups != 0:
      raise ValueError(
          f'Input channels ({c}) must be divisible by num_groups ({self._num_groups}).'
      )
    group_size = c // self._num_groups
    groups = []
    for i in range(self._num_groups):
      g_vals = vals[..., i * group_size:(i + 1) * group_size]
      groups.append(type(x)(g_vals, x.mask))
    return groups

  def _combine(self, outputs):
    """Combine group outputs."""
    if self._combination == self.CONCAT:
      # Concatenate along last axis.
      combined_vals = mx.concatenate([o.values for o in outputs], axis=-1)
      return Sequence(combined_vals, outputs[0].mask)
    elif self._combination == self.STACK:
      # Stack along a new axis before the last.
      stacked = mx.stack([o.values for o in outputs], axis=-2)
      return Sequence(stacked, outputs[0].mask)
    else:
      raise ValueError(f'Unsupported combination mode: {self._combination}')

  def get_output_shape(self, input_shape, *, constants=None):
    if not input_shape:
      raise ValueError(f'Input must be at least 3D, got: {input_shape=}.')
    if input_shape[-1] % self._num_groups != 0:
      raise ValueError(
          f'Input channels ({input_shape[-1]}) must be divisible by'
          f' num_groups ({self._num_groups}).'
      )
    group_shape = list(input_shape)
    group_shape[-1] //= self._num_groups
    child_shape = self.child.get_output_shape(tuple(group_shape), constants=constants)
    if self._combination == self.CONCAT:
      return child_shape[:-1] + (child_shape[-1] * self._num_groups,)
    elif self._combination == self.STACK:
      return child_shape[:-1] + (self._num_groups,) + (child_shape[-1],)
    else:
      raise ValueError(f'Unsupported combination mode: {self._combination}')

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.child.get_output_dtype(input_dtype, constants=constants)

  @types.check_layer
  def layer(self, x, *, constants=None):
    groups = self._split(x)
    outputs = [self.child.layer(g, constants=constants) for g in groups]
    return self._combine(outputs)

  def layer_with_emits(self, x, *, constants=None):
    groups = self._split(x)
    outputs, emits = [], []
    for g in groups:
      y, e = self.child.layer_with_emits(g, constants=constants)
      outputs.append(y)
      emits.append(e)
    return self._combine(outputs), tuple(emits)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if not input_spec.shape:
      raise ValueError(f'Input must be at least 3D, got: {input_spec.shape=}.')
    if input_spec.shape[-1] % self._num_groups != 0:
      raise ValueError(
          f'Input channels ({input_spec.shape[-1]}) must be divisible by'
          f' num_groups ({self._num_groups}).'
      )
    group_shape = list(input_spec.shape)
    group_shape[-1] //= self._num_groups
    from sequence_layers.mlx import types as sl_types
    group_spec = sl_types.ChannelSpec(
        shape=tuple(group_shape),
        dtype=input_spec.dtype,
    )
    state = self.child.get_initial_state(batch_size, group_spec, constants=constants)
    return (state,) * self._num_groups

  @types.check_step
  def step(self, x, state, *, constants=None):
    groups = self._split(x)
    outputs = []
    new_states = []
    for g, s in zip(groups, state):
      y, ns = self.child.step(g, s, constants=constants)
      outputs.append(y)
      new_states.append(ns)
    return self._combine(outputs), tuple(new_states)

  def step_with_emits(self, x, state, *, constants=None):
    groups = self._split(x)
    outputs, new_states, emits = [], [], []
    for g, s in zip(groups, state):
      y, ns, e = self.child.step_with_emits(g, s, constants=constants)
      outputs.append(y)
      new_states.append(ns)
      emits.append(e)
    return self._combine(outputs), tuple(new_states), tuple(emits)

  @classmethod
  def from_config(cls, config, backend='mlx'):
    child = config.child_layer.make(backend=backend)
    return cls(
        child_layer=child,
        num_groups=config.num_groups,
        combination=config.combination,
    )
