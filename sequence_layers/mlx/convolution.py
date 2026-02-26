"""Convolution layers for MLX."""

import fractions
import math

import mlx.core as mx
import mlx.nn as nn

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import types

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence
PaddingMode = bt.PaddingMode


# ---------------------------------------------------------------------------
# Padding utilities (ported from jax/utils.py and jax/convolution.py)
# ---------------------------------------------------------------------------


def _effective_kernel_size(kernel_size, dilation_rate):
  return (kernel_size - 1) * dilation_rate + 1


def _explicit_padding(padding, kernel_size, stride, dilation_rate):
  """Returns (pad_left, pad_right) for the given padding mode."""
  if not isinstance(padding, str):
    return tuple(padding)

  ek = _effective_kernel_size(kernel_size, dilation_rate)

  if padding in (PaddingMode.CAUSAL_VALID.value, PaddingMode.CAUSAL.value):
    return (ek - 1, 0)
  elif padding == PaddingMode.SEMICAUSAL.value:
    pad_left = max(ek - stride, 0)
    return (pad_left, ek - 1 - pad_left)
  elif padding in (
      PaddingMode.REVERSE_CAUSAL_VALID.value,
      PaddingMode.REVERSE_CAUSAL.value,
  ):
    return (0, ek - 1)
  elif padding == PaddingMode.SAME.value:
    pad_amount = ek - 1
    pad_left = pad_amount // 2
    return (pad_left, pad_amount - pad_left)
  elif padding == PaddingMode.VALID.value:
    return (0, 0)
  elif padding == PaddingMode.SEMICAUSAL_FULL.value:
    return (ek - stride, ek - 1)
  else:
    raise ValueError(f'Unsupported padding mode: {padding}')


def _buffer_width(padding, kernel_size, stride, dilation_rate):
  """Returns the buffer width for step mode."""
  ek = _effective_kernel_size(kernel_size, dilation_rate)

  if padding == PaddingMode.SEMICAUSAL.value:
    return max(ek - stride, 0)
  elif padding in (
      PaddingMode.REVERSE_CAUSAL.value,
      PaddingMode.REVERSE_CAUSAL_VALID.value,
  ):
    return (ek - 1) // stride * stride
  elif padding in (
      PaddingMode.CAUSAL.value,
      PaddingMode.CAUSAL_VALID.value,
  ):
    return ek - 1
  else:
    raise ValueError(f'Unsupported step padding: {padding}')


def _supports_step(padding):
  """Returns True if the padding mode supports step-by-step processing."""
  return padding in (
      PaddingMode.CAUSAL_VALID.value,
      PaddingMode.REVERSE_CAUSAL_VALID.value,
      PaddingMode.CAUSAL.value,
      PaddingMode.REVERSE_CAUSAL.value,
      PaddingMode.SEMICAUSAL.value,
  )


def _compute_conv_mask(
    mask, kernel_size, stride, dilation_rate, padding, is_step
):
  """Compute the output mask for a convolution-like operation."""
  ek = _effective_kernel_size(kernel_size, dilation_rate)

  if is_step:
    if isinstance(padding, str) and padding in (
        PaddingMode.SAME.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.REVERSE_CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      pad_left, pad_right = _explicit_padding(
          padding, kernel_size, stride, dilation_rate
      )
      # Use a simple convolution-like mask computation with float kernel.
      kernel = [0.0] * pad_left + [1.0] + [0.0] * pad_right
      kernel = mx.array(kernel, dtype=mx.float32).reshape(1, -1, 1)
      mask_f = mask[:, :, None].astype(mx.float32)
      mask_conv = mx.conv1d(mask_f, kernel, stride=stride)
      return mx.squeeze(mask_conv, axis=-1).astype(mx.bool_)
    elif not isinstance(padding, str) or padding in (
        PaddingMode.VALID.value,
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL_VALID.value,
    ):
      return _compute_conv_mask_logical(
          mask, kernel_size, stride, dilation_rate
      )
    else:
      return _compute_conv_mask_logical(
          mask, kernel_size, stride, dilation_rate
      )

  # Layer mode.
  if isinstance(padding, str) and padding in (
      PaddingMode.SAME.value,
      PaddingMode.CAUSAL.value,
      PaddingMode.REVERSE_CAUSAL.value,
      PaddingMode.SEMICAUSAL.value,
  ):
    if stride > 1:
      mask = mask[:, ::stride]
    return mask

  # VALID-like modes: need to compute mask through reduce_window equiv.
  pad_left, pad_right = _explicit_padding(
      padding, kernel_size, stride, dilation_rate
  )
  is_causal_valid = (
      isinstance(padding, str) and padding == PaddingMode.CAUSAL_VALID.value
  )
  mask = mx.pad(
      mask,
      [(0, 0), (pad_left, pad_right)],
      constant_values=is_causal_valid,
  )
  is_semicausal_full = (
      isinstance(padding, str) and padding == PaddingMode.SEMICAUSAL_FULL.value
  )
  return _compute_conv_mask_logical(
      mask,
      kernel_size,
      stride,
      dilation_rate,
      use_logical_or=is_semicausal_full,
  )


def _compute_conv_mask_logical(
    mask, kernel_size, stride, dilation_rate, use_logical_or=False
):
  """Windowed AND/OR mask computation."""
  # Optimized path for dilation=1 and kernel_size divisible by stride.
  if dilation_rate == 1 and kernel_size % stride == 0:
    num_frames = mask.shape[1] // stride
    mask = mask[:, : num_frames * stride]
    mask = mask.reshape(mask.shape[0], num_frames, stride)
    if use_logical_or:
      mask = mx.max(mask, axis=-1)
    else:
      mask = mx.min(mask, axis=-1)
    kernel_size = kernel_size // stride
    stride = 1

  if kernel_size == 1 and stride == 1:
    return mask

  # Use float conv to simulate reduce_window.
  mask_f = mask[:, :, None].astype(mx.float32)
  # Build a kernel with ones at dilated positions.
  if dilation_rate == 1:
    kernel = mx.ones((1, kernel_size, 1), dtype=mx.float32)
  else:
    ek = _effective_kernel_size(kernel_size, dilation_rate)
    k = [0.0] * ek
    for i in range(kernel_size):
      k[i * dilation_rate] = 1.0
    kernel = mx.array(k, dtype=mx.float32).reshape(1, -1, 1)

  result = mx.conv1d(mask_f, kernel, stride=stride)
  result = mx.squeeze(result, axis=-1)

  if use_logical_or:
    return result > 0.0
  else:
    return result >= float(kernel_size)


def _compute_initial_state(batch_size, input_spec, buf_width, padding):
  """Create initial buffer state for step mode."""
  if padding in (
      PaddingMode.CAUSAL_VALID.value,
      PaddingMode.REVERSE_CAUSAL_VALID.value,
      PaddingMode.SEMICAUSAL_FULL.value,
  ):
    mask = mx.ones((batch_size, buf_width), dtype=bt.MASK_DTYPE)
  elif padding in (
      PaddingMode.CAUSAL.value,
      PaddingMode.REVERSE_CAUSAL.value,
      PaddingMode.SEMICAUSAL.value,
  ):
    mask = mx.zeros((batch_size, buf_width), dtype=bt.MASK_DTYPE)
  else:
    raise ValueError(f'Step not supported with padding: {padding}')

  values = mx.zeros(
      (batch_size, buf_width) + input_spec.shape,
      dtype=input_spec.dtype,
  )
  return MaskedSequence(values, mask)


# ---------------------------------------------------------------------------
# Conv1D
# ---------------------------------------------------------------------------


class Conv1D(types.SequenceLayer):
  """1D strided or dilated convolution layer.

  Supports causal, reverse_causal, same, and valid padding modes.
  Step-by-step processing is supported for causal padding modes.
  """

  def __init__(
      self,
      *,
      in_features: int,
      filters: int,
      kernel_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
      groups: int = 1,
      use_bias: bool = True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self.in_features = in_features
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.dilation_rate = dilation_rate
    self.padding = padding
    self.groups = groups
    self.use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    if in_features % groups != 0:
      raise ValueError(f'{in_features=} must be divisible by {groups=}.')

    # Create kernel: [out_channels, kernel_size, in_channels // groups]
    # This is the MLX Conv1d convention.
    self._conv = nn.Conv1d(
        in_channels=in_features,
        out_channels=filters,
        kernel_size=kernel_size,
        stride=strides,
        # Padding handled manually.
        padding=0,
        dilation=dilation_rate,
        bias=use_bias,
    )

  @property
  def supports_step(self):
    return _supports_step(self.padding)

  @property
  def block_size(self):
    return self.strides

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self.strides)

  @property
  def input_latency(self):
    ek = _effective_kernel_size(self.kernel_size, self.dilation_rate)
    if self.padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self.padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
        PaddingMode.SEMICAUSAL_FULL.value,
    ):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 1:
      raise ValueError(
          f'Conv1D requires rank 3 input, got channel_shape={input_shape}.'
      )
    return (self.filters,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.compute_dtype or self._param_dtype

  def _forward(self, values, pad_left, pad_right):
    """Apply convolution with explicit padding."""
    if pad_left > 0 or pad_right > 0:
      values = mx.pad(
          values,
          [(0, 0), (pad_left, pad_right), (0, 0)],
      )
    compute_dtype = self.compute_dtype or self._param_dtype
    values = values.astype(compute_dtype)
    y = self._conv(values)
    if self.activation is not None:
      y = self.activation(y)
    return y

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    bw = _buffer_width(
        self.padding,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
    )
    if not bw:
      return ()
    return _compute_initial_state(
        batch_size,
        input_spec,
        bw,
        self.padding,
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    ek = _effective_kernel_size(self.kernel_size, self.dilation_rate)
    if ek > 1:
      x = x.mask_invalid()

    bw = _buffer_width(
        self.padding,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
    )

    if bw:
      state = state.concatenate(x)
    else:
      state = x

    # In step mode, padding is provided by the buffer — use valid conv.
    values = self._forward(state.values, 0, 0)
    mask = _compute_conv_mask(
        state.mask,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
        self.padding,
        is_step=True,
    )

    if bw:
      state = state[:, -bw:]
    else:
      state = ()

    return Sequence(values, mask), state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self.kernel_size > 1:
      x = x.mask_invalid()

    pad_left, pad_right = _explicit_padding(
        self.padding,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
    )
    values = self._forward(x.values, pad_left, pad_right)
    mask = _compute_conv_mask(
        x.mask,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
        self.padding,
        is_step=False,
    )
    return Sequence(values, mask)

  @classmethod
  def from_config(cls, config):
    """Create from a Linen Conv1D.Config (deferred)."""
    return DeferredConv1D(config)


# ---------------------------------------------------------------------------
# DepthwiseConv1D
# ---------------------------------------------------------------------------


class DepthwiseConv1D(types.SequenceLayer):
  """1D depthwise convolution layer.

  Each input channel is convolved independently. The output has
  in_features * depth_multiplier channels.
  """

  def __init__(
      self,
      *,
      in_features: int,
      kernel_size: int,
      depth_multiplier: int = 1,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
      use_bias: bool = True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self.in_features = in_features
    self.kernel_size = kernel_size
    self.depth_multiplier = depth_multiplier
    self.strides = strides
    self.dilation_rate = dilation_rate
    self.padding = padding
    self.use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    out_features = in_features * depth_multiplier
    # Depthwise: groups = in_features, each group has depth_multiplier
    # output channels.
    self._conv = nn.Conv1d(
        in_channels=in_features,
        out_channels=out_features,
        kernel_size=kernel_size,
        stride=strides,
        padding=0,
        dilation=dilation_rate,
        groups=in_features,
        bias=use_bias,
    )

  @property
  def supports_step(self):
    return _supports_step(self.padding)

  @property
  def block_size(self):
    return self.strides

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self.strides)

  @property
  def input_latency(self):
    ek = _effective_kernel_size(self.kernel_size, self.dilation_rate)
    if self.padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self.padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
        PaddingMode.SEMICAUSAL_FULL.value,
    ):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 1:
      raise ValueError(
          'DepthwiseConv1D requires rank 3 input, got '
          f'channel_shape={input_shape}.'
      )
    return (input_shape[0] * self.depth_multiplier,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.compute_dtype or self._param_dtype

  def _forward(self, values, pad_left, pad_right):
    if pad_left > 0 or pad_right > 0:
      values = mx.pad(
          values,
          [(0, 0), (pad_left, pad_right), (0, 0)],
      )
    compute_dtype = self.compute_dtype or self._param_dtype
    values = values.astype(compute_dtype)
    y = self._conv(values)
    if self.activation is not None:
      y = self.activation(y)
    return y

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    bw = _buffer_width(
        self.padding,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
    )
    if not bw:
      return ()
    return _compute_initial_state(
        batch_size,
        input_spec,
        bw,
        self.padding,
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    ek = _effective_kernel_size(self.kernel_size, self.dilation_rate)
    if ek > 1:
      x = x.mask_invalid()

    bw = _buffer_width(
        self.padding,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
    )

    if bw:
      state = state.concatenate(x)
    else:
      state = x

    values = self._forward(state.values, 0, 0)
    mask = _compute_conv_mask(
        state.mask,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
        self.padding,
        is_step=True,
    )

    if bw:
      state = state[:, -bw:]
    else:
      state = ()

    return Sequence(values, mask), state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self.kernel_size > 1:
      x = x.mask_invalid()

    pad_left, pad_right = _explicit_padding(
        self.padding,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
    )
    values = self._forward(x.values, pad_left, pad_right)
    mask = _compute_conv_mask(
        x.mask,
        self.kernel_size,
        self.strides,
        self.dilation_rate,
        self.padding,
        is_step=False,
    )
    return Sequence(values, mask)

  @classmethod
  def from_config(cls, config):
    return DeferredDepthwiseConv1D(config)


# ---------------------------------------------------------------------------
# Conv1DTranspose
# ---------------------------------------------------------------------------


def _transpose_conv_output_trim(kernel_size, stride, dilation_rate, padding):
  """Output-side trimming for transpose convolutions in MLX.

  MLX conv_transpose1d with padding=0 produces output of size:
    raw = (t - 1) * stride + ek
  This function returns (trim_left, trim_right) to cut raw output
  to the desired size.
  """
  ek = _effective_kernel_size(kernel_size, dilation_rate)
  total_trim = max(0, ek - stride)

  if padding == PaddingMode.CAUSAL.value:
    return (0, total_trim)
  elif padding == PaddingMode.SAME.value:
    trim_left = total_trim // 2
    return (trim_left, total_trim - trim_left)
  elif padding == PaddingMode.VALID.value:
    return (0, 0)
  elif padding == PaddingMode.SEMICAUSAL_FULL.value:
    return (0, 0)
  else:
    raise ValueError(f'Unsupported padding: {padding}')


def _compute_conv_transpose_output_length(
    time, kernel_size, stride, dilation_rate, padding
):
  ek = _effective_kernel_size(kernel_size, dilation_rate)
  if padding in (
      PaddingMode.SAME.value,
      PaddingMode.CAUSAL.value,
      PaddingMode.SEMICAUSAL_FULL.value,
  ):
    return time * stride
  elif padding == PaddingMode.VALID.value:
    return time * stride + max(ek - stride, 0)
  else:
    raise ValueError(f'Unsupported padding: {padding}')


def _compute_conv_transpose_mask(
    mask, kernel_size, stride, dilation_rate, padding
):
  """Compute output mask for a transpose convolution."""
  ek = _effective_kernel_size(kernel_size, dilation_rate)

  if ek <= stride or padding in (
      PaddingMode.SAME.value,
      PaddingMode.CAUSAL.value,
  ):
    return mx.repeat(mask, stride, axis=1)

  # Use transpose convolution to compute the mask.
  tl, tr = _transpose_conv_output_trim(
      kernel_size,
      stride,
      dilation_rate,
      padding,
  )

  if padding == PaddingMode.SEMICAUSAL_FULL.value:
    test_signal = mask
    test_fn = lambda m: m > 0.0
  else:
    test_signal = mx.logical_not(mask)
    test_fn = lambda m: m == 0.0

  kernel = mx.ones((1, kernel_size, 1), dtype=mx.float32)
  signal = test_signal.astype(mx.float32)[:, :, None]

  result = mx.conv_transpose1d(
      signal,
      kernel,
      stride=stride,
      padding=0,
      dilation=dilation_rate,
  )
  # Trim to match desired output.
  if tl > 0:
    result = result[:, tl:]
  if tr > 0:
    result = result[:, :-tr]
  result = mx.squeeze(result, axis=-1)
  return test_fn(result)


class Conv1DTranspose(types.SequenceLayer):
  """1D transpose (deconvolution) layer for upsampling.

  Supports 'valid', 'causal', and 'same' padding modes.
  """

  def __init__(
      self,
      *,
      in_features: int,
      filters: int,
      kernel_size: int,
      strides: int = 1,
      dilation_rate: int = 1,
      padding: str = 'valid',
      groups: int = 1,
      use_bias: bool = True,
      activation=None,
      compute_dtype=None,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self.in_features = in_features
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.dilation_rate = dilation_rate
    self.padding = padding
    self.groups = groups
    self.use_bias = use_bias
    self.activation = activation
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    # Create kernel and bias manually — nn.ConvTranspose1d layout differs.
    key = mx.random.key(0)
    init = init_mapping._make_variance_scaling_init(
        'fan_in', 'truncated_normal'
    )
    # Kernel: [out_channels, kernel_size, in_channels // groups]
    self.kernel = init(
        key,
        (filters, kernel_size, in_features // groups),
        param_dtype,
    )
    if use_bias:
      self.bias = mx.zeros((filters,), dtype=param_dtype)

  @property
  def supports_step(self):
    return self.padding == PaddingMode.CAUSAL.value

  @property
  def block_size(self):
    return 1

  @property
  def output_ratio(self):
    return fractions.Fraction(self.strides)

  @property
  def input_latency(self):
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 1:
      raise ValueError(
          'Conv1DTranspose requires rank 3 input, got '
          f'channel_shape={input_shape}.'
      )
    return (self.filters,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.compute_dtype or self._param_dtype

  def _raw_conv_transpose(self, values):
    """Apply raw transpose convolution (no padding trim)."""
    compute_dtype = self.compute_dtype or self._param_dtype
    values = values.astype(compute_dtype)
    y = mx.conv_transpose1d(
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

  def _forward(self, values):
    """Apply transpose convolution with output trimming."""
    y = self._raw_conv_transpose(values)
    tl, tr = _transpose_conv_output_trim(
        self.kernel_size,
        self.strides,
        self.dilation_rate,
        self.padding,
    )
    if tl > 0:
      y = y[:, tl:]
    if tr > 0:
      y = y[:, :-tr]
    return y

  @property
  def _ola_buffer_width(self):
    return max(
        0,
        _effective_kernel_size(self.kernel_size, self.dilation_rate)
        - self.strides,
    )

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if not self.supports_step:
      return ()
    bw = self._ola_buffer_width
    if not bw:
      return ()
    compute_dtype = self.compute_dtype or self._param_dtype
    return mx.zeros(
        (batch_size, bw, self.filters),
        dtype=compute_dtype,
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    # Use raw conv (no trimming) for overlap-add.
    values = self._raw_conv_transpose(x.values)
    mask = mx.repeat(x.mask, self.strides, axis=1)

    bw = self._ola_buffer_width
    if bw:
      # Overlap-add: the first bw samples overlap with buffer.
      overlap = values[:, :bw] + state
      rest = values[:, bw:]
      values = mx.concatenate([overlap, rest], axis=1)

      output_samples = self.strides * x.shape[1]
      output = values[:, :output_samples]
      state = values[:, output_samples : output_samples + bw]
      if state.shape[1] < bw:
        pad_right = bw - state.shape[1]
        state = mx.pad(state, [(0, 0), (0, pad_right), (0, 0)])
      values = output

    return Sequence(values, mask), state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self.padding == PaddingMode.CAUSAL.value:
      # For causal, use raw conv and trim trailing overlap.
      values = self._raw_conv_transpose(x.values)
      expected_time = x.shape[1] * self.strides
      values = values[:, :expected_time]
      mask = mx.repeat(x.mask, self.strides, axis=1)
    else:
      values = self._forward(x.values)
      mask = _compute_conv_transpose_mask(
          x.mask,
          self.kernel_size,
          self.strides,
          self.dilation_rate,
          self.padding,
      )
      expected_time = _compute_conv_transpose_output_length(
          x.shape[1],
          self.kernel_size,
          self.strides,
          self.dilation_rate,
          self.padding,
      )
      values = values[:, :expected_time]
      mask = mask[:, :expected_time]

    return Sequence(values, mask)

  @classmethod
  def from_config(cls, config):
    return DeferredConv1DTranspose(config)


# ---------------------------------------------------------------------------
# Deferred wrappers (Linen configs lack in_features)
# ---------------------------------------------------------------------------


class DeferredConv1D(types.SequenceLayer):
  """Conv1D that defers weight creation until first input."""

  def __init__(self, config):
    super().__init__()
    self._config = config
    self.inner = None

  def _ensure_initialized(self, in_features):
    if self.inner is not None:
      return
    c = self._config
    compute_dtype = getattr(c, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    param_dtype = init_mapping._to_mx_dtype(c.param_dtype)
    activation = init_mapping.map_activation(getattr(c, 'activation', None))
    self.inner = Conv1D(
        in_features=in_features,
        filters=c.filters,
        kernel_size=c.kernel_size,
        strides=c.strides,
        dilation_rate=c.dilation_rate,
        padding=c.padding,
        groups=c.groups,
        use_bias=c.use_bias,
        activation=activation,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )

  @property
  def supports_step(self):
    return _supports_step(self._config.padding)

  @property
  def block_size(self):
    return self._config.strides

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self._config.strides)

  @property
  def input_latency(self):
    ek = _effective_kernel_size(
        self._config.kernel_size, self._config.dilation_rate
    )
    if self._config.padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self._config.padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
        PaddingMode.SEMICAUSAL_FULL.value,
    ):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    return (self._config.filters,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    cd = getattr(self._config, 'compute_dtype', None)
    if cd is not None:
      return init_mapping._to_mx_dtype(cd)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    self._ensure_initialized(input_spec.shape[-1])
    return self.inner.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self.inner.layer(x, constants=constants)

  def step(self, x, state, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self.inner.step(x, state, constants=constants)


class DeferredDepthwiseConv1D(types.SequenceLayer):
  """DepthwiseConv1D that defers weight creation until first input."""

  def __init__(self, config):
    super().__init__()
    self._config = config
    self.inner = None

  def _ensure_initialized(self, in_features):
    if self.inner is not None:
      return
    c = self._config
    compute_dtype = getattr(c, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    param_dtype = init_mapping._to_mx_dtype(c.param_dtype)
    activation = init_mapping.map_activation(getattr(c, 'activation', None))
    self.inner = DepthwiseConv1D(
        in_features=in_features,
        kernel_size=c.kernel_size,
        depth_multiplier=c.depth_multiplier,
        strides=c.strides,
        dilation_rate=c.dilation_rate,
        padding=c.padding,
        use_bias=c.use_bias,
        activation=activation,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )

  @property
  def supports_step(self):
    return _supports_step(self._config.padding)

  @property
  def block_size(self):
    return self._config.strides

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self._config.strides)

  @property
  def input_latency(self):
    ek = _effective_kernel_size(
        self._config.kernel_size, self._config.dilation_rate
    )
    if self._config.padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self._config.padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
        PaddingMode.SEMICAUSAL_FULL.value,
    ):
      return ek - 1
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    return (input_shape[0] * self._config.depth_multiplier,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    cd = getattr(self._config, 'compute_dtype', None)
    if cd is not None:
      return init_mapping._to_mx_dtype(cd)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    self._ensure_initialized(input_spec.shape[-1])
    return self.inner.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self.inner.layer(x, constants=constants)

  def step(self, x, state, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self.inner.step(x, state, constants=constants)


class DeferredConv1DTranspose(types.SequenceLayer):
  """Conv1DTranspose that defers weight creation until first input."""

  def __init__(self, config):
    super().__init__()
    self._config = config
    self.inner = None

  def _ensure_initialized(self, in_features):
    if self.inner is not None:
      return
    c = self._config
    compute_dtype = getattr(c, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    param_dtype = init_mapping._to_mx_dtype(c.param_dtype)
    activation = init_mapping.map_activation(getattr(c, 'activation', None))
    self.inner = Conv1DTranspose(
        in_features=in_features,
        filters=c.filters,
        kernel_size=c.kernel_size,
        strides=c.strides,
        dilation_rate=getattr(c, 'dilation_rate', 1),
        padding=c.padding,
        groups=getattr(c, 'groups', 1),
        use_bias=c.use_bias,
        activation=activation,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )

  @property
  def supports_step(self):
    return self._config.padding == PaddingMode.CAUSAL.value

  @property
  def block_size(self):
    return 1

  @property
  def output_ratio(self):
    return fractions.Fraction(self._config.strides)

  @property
  def input_latency(self):
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    return (self._config.filters,)

  def get_output_dtype(self, input_dtype, *, constants=None):
    cd = getattr(self._config, 'compute_dtype', None)
    if cd is not None:
      return init_mapping._to_mx_dtype(cd)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    self._ensure_initialized(input_spec.shape[-1])
    return self.inner.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def layer(self, x, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self.inner.layer(x, constants=constants)

  def step(self, x, state, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self.inner.step(x, state, constants=constants)
