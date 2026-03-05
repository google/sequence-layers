"""DSP layers for MLX."""

import dataclasses
import fractions
import math

import mlx.core as mx
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import convolution as conv_utils
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence
PaddingMode = bt.PaddingMode


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------


def hann_window(window_length, periodic=True, dtype=np.float32):
  """Compute a periodic Hann window."""
  if window_length == 1:
    return np.ones([1], dtype=dtype)
  even = 1 - window_length % 2
  n = np.asarray(window_length + int(periodic) * even - 1, dtype=dtype)
  count = np.arange(window_length, dtype=dtype)
  return np.asarray(0.5 - 0.5 * np.cos(2 * np.pi * count / n), dtype)


def frame(values, frame_length, frame_step, pad_mode='valid', axis=1):
  """Produce overlapping frames of a signal along `axis`.

  Args:
    values: [..., T, ...] array.
    frame_length: Length of each frame.
    frame_step: Stride between frames.
    pad_mode: Padding mode string or 'valid'.
    axis: Axis along which to frame.

  Returns:
    [..., num_frames, frame_length, ...] array.
  """
  # Normalize axis.
  if axis < 0:
    axis += values.ndim

  t = values.shape[axis]

  # Apply padding if needed.
  if isinstance(pad_mode, str) and pad_mode != PaddingMode.VALID.value:
    pad_left, pad_right = conv_utils._explicit_padding(
        pad_mode, frame_length, frame_step, 1
    )
    pad_widths = [(0, 0)] * values.ndim
    pad_widths[axis] = (pad_left, pad_right)
    values = mx.pad(values, pad_widths)
    t = values.shape[axis]

  # Compute number of frames.
  num_frames = max(0, (t - frame_length) // frame_step + 1)

  # Move target axis to position 1 for uniform handling.
  if axis != 1:
    perm = list(range(values.ndim))
    perm[1], perm[axis] = perm[axis], perm[1]
    values = mx.transpose(values, perm)

  # values shape: [batch, t, ...]
  batch = values.shape[0]
  rest_shape = values.shape[2:]

  # Fast path: zero-copy strided view for contiguous data.
  rest_size = 1
  for d in rest_shape:
    rest_size *= d

  batch_stride = t * rest_size
  frame_stride = frame_step * rest_size
  time_stride = rest_size

  # Compute rest strides from contiguous layout.
  rest_strides = []
  s = 1
  for d in reversed(rest_shape):
    rest_strides.append(s)
    s *= d
  rest_strides.reverse()

  result = mx.as_strided(
      values,
      shape=(batch, num_frames, frame_length) + rest_shape,
      strides=(batch_stride, frame_stride, time_stride) + tuple(rest_strides),
  )

  if axis != 1:
    # Move back.
    perm = list(range(result.ndim))
    perm[1], perm[axis] = perm[axis], perm[1]
    if axis > 1:
      perm.insert(axis + 1, perm.pop(2))
    result = mx.transpose(result, perm)

  return result


def overlap_and_add(signal_arr, frame_step):
  """Overlap-add framed signal.

  Args:
    signal_arr: [..., frames, frame_length] array.
    frame_step: Stride between frames.

  Returns:
    [..., output_length] array where
    output_length = (frames - 1) * frame_step + frame_length.
  """
  shape = signal_arr.shape
  outer_dims = shape[:-2]
  frames = shape[-2]
  frame_length = shape[-1]
  output_length = frame_length + frame_step * (frames - 1)

  if frame_length == frame_step:
    return signal_arr.reshape(outer_dims + (output_length,))

  # Vectorized overlap-add via scatter.
  outer_size = 1
  for d in outer_dims:
    outer_size *= d

  flat = signal_arr.reshape(outer_size, frames, frame_length)

  # Build output position indices: [frames, frame_length].
  offsets = mx.arange(frames)[:, None] * frame_step
  positions = offsets + mx.arange(frame_length)[None, :]
  flat_positions = positions.reshape(-1)  # [frames * frame_length]

  # Flatten signal and scatter-add all frame contributions at once.
  flat_signal = flat.reshape(outer_size, frames * frame_length)
  result = mx.zeros((outer_size, output_length), dtype=flat.dtype)
  result = result.at[:, flat_positions].add(flat_signal)

  return result.reshape(outer_dims + (output_length,))


def linear_to_mel_weight_matrix(
    num_mel_bins,
    num_spectrogram_bins,
    sample_rate,
    lower_edge_hertz,
    upper_edge_hertz,
    dtype=np.float64,
):
  """Create a weight matrix for converting linear spectrogram to mel."""

  # Mel scale conversion (HTK formula).
  def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

  def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

  nyquist = sample_rate / 2.0
  freq_bins = np.linspace(0, nyquist, num_spectrogram_bins)

  mel_low = hz_to_mel(lower_edge_hertz)
  mel_high = hz_to_mel(upper_edge_hertz)
  mel_points = np.linspace(mel_low, mel_high, num_mel_bins + 2)
  hz_points = mel_to_hz(mel_points)

  lower = hz_points[:-2][np.newaxis, :]   # [1, num_mel_bins]
  center = hz_points[1:-1][np.newaxis, :]  # [1, num_mel_bins]
  upper = hz_points[2:][np.newaxis, :]     # [1, num_mel_bins]
  freq = freq_bins[:, np.newaxis]          # [num_spectrogram_bins, 1]

  rising = np.where(
      (freq >= lower) & (freq <= center) & (center > lower),
      (freq - lower) / np.maximum(center - lower, 1e-10),
      0.0,
  )
  falling = np.where(
      (freq > center) & (freq <= upper) & (upper > center),
      (upper - freq) / np.maximum(upper - center, 1e-10),
      0.0,
  )
  return (rising + falling).astype(dtype)


# ---------------------------------------------------------------------------
# Delay
# ---------------------------------------------------------------------------


class Delay(types.PreservesShape, types.PreservesType, types.SequenceLayer):
  """Delays input by `length` timesteps."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    length: int = 0
    delay_layer_output: bool = True
    name: str | None = None

    def make(self) -> 'Delay':
      return Delay.from_config(self)

  def __init__(self, *, length, delay_layer_output=True):
    super().__init__()
    if length < 0:
      raise ValueError(f'length must be non-negative, got {length}.')
    self.length = length
    self.delay_layer_output = delay_layer_output

  @property
  def input_latency(self):
    return self.length

  @property
  def output_latency(self):
    return 0 if self.delay_layer_output else self.length

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if not self.length:
      return ()
    return Sequence(
        mx.zeros(
            (batch_size, self.length) + input_spec.shape,
            dtype=input_spec.dtype,
        ),
        mx.zeros(
            (batch_size, self.length),
            dtype=bt.MASK_DTYPE,
        ),
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    if not self.length:
      return x, state
    state = state.concatenate(x)
    t = x.shape[1]
    y = Sequence(state.values[:, :t], state.mask[:, :t])
    state = Sequence(state.values[:, t:], state.mask[:, t:])
    return y, state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self.delay_layer_output:
      return x.pad_time(self.length, 0, valid=False)
    return x

  @classmethod
  def from_config(cls, config):
    return cls(
        length=config.length,
        delay_layer_output=config.delay_layer_output,
    )


# ---------------------------------------------------------------------------
# Lookahead
# ---------------------------------------------------------------------------


class Lookahead(types.PreservesShape, types.PreservesType, types.SequenceLayer):
  """Drops the first `length` timesteps from the input."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    length: int = 0
    preserve_length_in_layer: bool = False
    name: str | None = None

    def make(self) -> 'Lookahead':
      return Lookahead.from_config(self)

  def __init__(self, *, length, preserve_length_in_layer=False):
    super().__init__()
    if length < 0:
      raise ValueError(f'length must be non-negative, got {length}.')
    self.length = length
    self.preserve_length_in_layer = preserve_length_in_layer

  @property
  def input_latency(self):
    return 0

  @property
  def output_latency(self):
    return self.length

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if not self.length:
      return ()
    return mx.full(
        (batch_size,),
        self.length + 1,
        dtype=mx.int32,
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    if not self.length:
      return x, state
    increments = mx.cumsum(x.mask.astype(mx.int32), axis=1)
    countdown = mx.maximum(0, state[:, None] - increments)
    mask = mx.logical_and(x.mask, countdown == 0)
    y = Sequence(x.values, mask)
    state = countdown[:, -1]
    return y, state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if not self.length:
      return x
    x = x[:, self.length :]
    if self.preserve_length_in_layer:
      return x.pad_time(0, self.length, valid=False)
    return x

  @classmethod
  def from_config(cls, config):
    return cls(
        length=config.length,
        preserve_length_in_layer=config.preserve_length_in_layer,
    )


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


class Window(types.PreservesShape, types.PreservesType, types.Stateless):
  """Applies a window function along a channel axis."""

  def __init__(self, *, axis, window_fn=None):
    super().__init__()
    self._axis = axis
    self._window_fn = window_fn or hann_window

  def _get_axis(self, x):
    axis = self._axis
    if axis < 0:
      axis += x.ndim
    if axis < 2:
      raise ValueError(
          f'Window axis must be a channel axis (>= 2), got {axis}.'
      )
    return axis

  @types.check_layer
  def layer(self, x, *, constants=None):
    axis = self._get_axis(x)
    window_length = x.shape[axis]
    window = self._window_fn(window_length)
    window = mx.array(window, dtype=x.dtype)
    shape = [1] * x.ndim
    shape[axis] = window_length
    window = window.reshape(shape)
    return x.apply_values_masked(lambda v: v * window)

  @classmethod
  def from_config(cls, config):
    return cls(
        axis=config.axis,
        window_fn=config.window_fn,
    )


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


class Frame(types.PreservesType, types.SequenceLayer):
  """Produces overlapping frames of the input sequence."""

  def __init__(self, *, frame_length, frame_step, padding='valid'):
    super().__init__()
    if frame_length <= 0:
      raise ValueError(f'frame_length must be positive: {frame_length}')
    if frame_step <= 0:
      raise ValueError(f'frame_step must be positive: {frame_step}')
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.padding = padding

  @property
  def supports_step(self):
    return conv_utils._supports_step(self.padding)

  @property
  def block_size(self):
    return self.frame_step

  @property
  def output_ratio(self):
    return fractions.Fraction(1, self.frame_step)

  @property
  def input_latency(self):
    if self.padding in (
        PaddingMode.CAUSAL_VALID.value,
        PaddingMode.CAUSAL.value,
        PaddingMode.SEMICAUSAL.value,
    ):
      return 0
    elif self.padding in (
        PaddingMode.REVERSE_CAUSAL_VALID.value,
        PaddingMode.REVERSE_CAUSAL.value,
    ):
      return self.frame_length - 1
    return 0

  @property
  def _buffer_width(self):
    if self.padding == PaddingMode.SEMICAUSAL.value:
      return max(self.frame_length - self.frame_step, 0)
    elif self.padding in (
        PaddingMode.REVERSE_CAUSAL.value,
        PaddingMode.REVERSE_CAUSAL_VALID.value,
    ):
      return (self.frame_length - 1) // self.frame_step * self.frame_step
    elif self.padding in (
        PaddingMode.CAUSAL.value,
        PaddingMode.CAUSAL_VALID.value,
    ):
      return self.frame_length - 1
    else:
      raise ValueError(f'Unsupported step padding: {self.padding}')

  def get_output_shape(self, input_shape, *, constants=None):
    return (self.frame_length,) + tuple(input_shape)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return input_dtype

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    bw = self._buffer_width
    if not bw:
      return ()
    return conv_utils._compute_initial_state(
        batch_size,
        input_spec,
        bw,
        self.padding,
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    if self.frame_length > 1:
      x = x.mask_invalid()

    bw = self._buffer_width
    if bw:
      state = state.concatenate(x)
    else:
      state = x

    values = frame(
        state.values,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        pad_mode=PaddingMode.VALID.value,
        axis=1,
    )
    mask = conv_utils._compute_conv_mask(
        state.mask,
        self.frame_length,
        self.frame_step,
        1,
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
    if self.frame_length > 1:
      x = x.mask_invalid()

    values = frame(
        x.values,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        pad_mode=self.padding,
        axis=1,
    )
    mask = conv_utils._compute_conv_mask(
        x.mask,
        self.frame_length,
        self.frame_step,
        1,
        self.padding,
        is_step=False,
    )
    return Sequence(values, mask)

  @classmethod
  def from_config(cls, config):
    return cls(
        frame_length=config.frame_length,
        frame_step=config.frame_step,
        padding=config.padding,
    )


# ---------------------------------------------------------------------------
# OverlapAdd
# ---------------------------------------------------------------------------


class OverlapAdd(types.PreservesType, types.SequenceLayer):
  """Overlap-adds windows of [b, t, frame_length, ...].

  Output shape: [b, to, ...] where
  to = (ti - 1) * frame_step + frame_length.
  """

  def __init__(self, *, frame_length, frame_step, padding='valid'):
    super().__init__()
    if frame_length <= 0:
      raise ValueError(f'frame_length must be positive: {frame_length}')
    if frame_step <= 0:
      raise ValueError(f'frame_step must be positive: {frame_step}')
    if frame_length < frame_step:
      raise ValueError('frame_length must be >= frame_step.')
    if padding not in (
        PaddingMode.CAUSAL.value,
        PaddingMode.VALID.value,
        PaddingMode.SEMICAUSAL_FULL.value,
    ):
      raise ValueError(f'Unsupported padding: {padding}')
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.padding = padding

  @property
  def supports_step(self):
    return self.padding == PaddingMode.CAUSAL.value

  @property
  def output_ratio(self):
    return fractions.Fraction(self.frame_step)

  @property
  def _buffer_width(self):
    return max(0, self.frame_length - self.frame_step)

  def get_output_shape(self, input_shape, *, constants=None):
    if not input_shape or input_shape[0] != self.frame_length:
      raise ValueError(
          f'OverlapAdd expects (frame_length, ...) input, got {input_shape}.'
      )
    return tuple(input_shape[1:])

  def get_output_dtype(self, input_dtype, *, constants=None):
    return input_dtype

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if not input_shape_valid(input_spec.shape, self.frame_length):
      raise ValueError(f'Invalid input_spec shape: {input_spec.shape}')
    bw = self._buffer_width
    if not bw:
      return ()
    out_shape = tuple(input_spec.shape[1:])
    return mx.zeros(
        (batch_size, bw) + out_shape,
        dtype=input_spec.dtype,
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    if self.frame_length > 1:
      x = x.mask_invalid()

    # Transpose [num_frames, frame_length] to end.
    if x.ndim > 3:
      # Move axes 1,2 to -2,-1.
      axes = list(range(x.ndim))
      axes.remove(1)
      axes.remove(2)
      axes.extend([1, 2])
      values = mx.transpose(x.values, axes)
    else:
      values = x.values

    values = overlap_and_add(values, self.frame_step)

    if x.ndim > 3:
      # Move back.
      values = mx.moveaxis(values, -1, 1)

    mask = conv_utils._compute_conv_transpose_mask(
        x.mask,
        self.frame_length,
        self.frame_step,
        1,
        self.padding,
    )

    bw = self._buffer_width
    if bw:
      time = x.shape[1]
      # Pad state to extend to values length.
      pad_right = max(0, values.shape[1] - bw)
      pad_widths = [(0, 0)] * state.ndim
      pad_widths[1] = (0, pad_right)
      padded_state = mx.pad(state, pad_widths)

      values = values + padded_state

      output_samples = self.frame_step * time
      output = values[:, :output_samples]
      state = values[:, output_samples : output_samples + bw]
      if state.shape[1] < bw:
        pad_widths = [(0, 0)] * state.ndim
        pad_widths[1] = (0, bw - state.shape[1])
        state = mx.pad(state, pad_widths)
      values = output

    return Sequence(values, mask), state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self.frame_length > 1:
      x = x.mask_invalid()

    if x.ndim > 3:
      axes = list(range(x.ndim))
      axes.remove(1)
      axes.remove(2)
      axes.extend([1, 2])
      values = mx.transpose(x.values, axes)
    else:
      values = x.values

    values = overlap_and_add(values, self.frame_step)

    if x.ndim > 3:
      values = mx.moveaxis(values, -1, 1)

    mask = conv_utils._compute_conv_transpose_mask(
        x.mask,
        self.frame_length,
        self.frame_step,
        1,
        self.padding,
    )

    trim = max(self.frame_length - self.frame_step, 0)
    if self.padding == PaddingMode.CAUSAL.value:
      if trim:
        values = values[:, :-trim]
    elif self.padding == PaddingMode.SEMICAUSAL_FULL.value:
      if trim:
        values = values[:, trim:]
        mask = mask[:, trim:]
      size = min(values.shape[1], mask.shape[1])
      return Sequence(values[:, :size], mask[:, :size])

    return Sequence(values, mask)

  @classmethod
  def from_config(cls, config):
    return cls(
        frame_length=config.frame_length,
        frame_step=config.frame_step,
        padding=config.padding,
    )


def input_shape_valid(shape, frame_length):
  return shape and shape[0] == frame_length


# ---------------------------------------------------------------------------
# FFT layers
# ---------------------------------------------------------------------------


def _validate_and_normalize_axis(axis, input_shape):
  """Normalize axis for FFT, ensuring it's a channel axis."""
  if axis < 0:
    axis += len(input_shape)
  if axis < 0 or axis >= len(input_shape):
    raise ValueError(f'Axis {axis} out of range for shape {input_shape}.')
  if axis in (0, 1):
    raise ValueError(f'FFT over batch/time not allowed. Got axis={axis}.')
  return axis


def _pad_or_truncate_for_fft(x, axis, required_length, padding):
  """Pad or truncate sequence for FFT."""
  input_dim = x.shape[axis]
  if input_dim == required_length:
    return x
  if input_dim < required_length:
    pad_amount = required_length - input_dim
    if padding == 'center':
      pad_left = pad_amount // 2
      pad_right = pad_amount - pad_left
    else:
      pad_left = 0
      pad_right = pad_amount
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (pad_left, pad_right)
    return x.apply_values_masked(mx.pad, pad_widths)
  else:
    # Truncate.
    if padding == 'center':
      start = (input_dim - required_length) // 2
    else:
      start = 0
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(start, start + required_length)
    return x.apply_values_masked(lambda v: v[tuple(slices)])


class FFT(types.PreservesType, types.Stateless):
  """Applies FFT to a channel dimension."""

  def __init__(self, *, fft_length=None, axis=-1, padding='right'):
    super().__init__()
    self.fft_length = fft_length
    self._axis = axis
    self._padding = padding

  def _get_output_length(self, input_size):
    return self.fft_length or input_size

  def get_output_shape(self, input_shape, *, constants=None):
    shape = list(input_shape)
    axis = (
        _validate_and_normalize_axis(
            self._axis, (None, None) + tuple(input_shape)
        )
        - 2
    )
    shape[axis] = self._get_output_length(shape[axis])
    return tuple(shape)

  @types.check_layer
  def layer(self, x, *, constants=None):
    if x.ndim <= 2:
      raise ValueError('FFT requires rank >= 3 input.')
    axis = _validate_and_normalize_axis(self._axis, x.shape)
    required = self._get_output_length(x.shape[axis])
    x = _pad_or_truncate_for_fft(x, axis, required, self._padding)
    return x.apply_values(mx.fft.fft, axis=axis)

  @classmethod
  def from_config(cls, config):
    return cls(
        fft_length=config.fft_length,
        axis=config.axis,
        padding=config.padding,
    )


class IFFT(types.PreservesType, types.Stateless):
  """Applies IFFT to a channel dimension."""

  def __init__(
      self,
      *,
      fft_length=None,
      frame_length=None,
      axis=-1,
      padding='right',
  ):
    super().__init__()
    self.fft_length = fft_length
    self.frame_length = frame_length
    self._axis = axis
    self._padding = padding

  def _get_output_length(self, input_size):
    return self.frame_length or input_size

  def get_output_shape(self, input_shape, *, constants=None):
    shape = list(input_shape)
    axis = (
        _validate_and_normalize_axis(
            self._axis, (None, None) + tuple(input_shape)
        )
        - 2
    )
    shape[axis] = self._get_output_length(shape[axis])
    return tuple(shape)

  @types.check_layer
  def layer(self, x, *, constants=None):
    if x.ndim <= 2:
      raise ValueError('IFFT requires rank >= 3 input.')
    axis = _validate_and_normalize_axis(self._axis, x.shape)
    x = x.apply_values(mx.fft.ifft, axis=axis)
    required = self._get_output_length(x.shape[axis])
    return _pad_or_truncate_for_fft(x, axis, required, self._padding)

  @classmethod
  def from_config(cls, config):
    return cls(
        fft_length=config.fft_length,
        frame_length=config.frame_length,
        axis=config.axis,
        padding=config.padding,
    )


class RFFT(types.Stateless):
  """Applies RFFT to a channel dimension."""

  def __init__(self, *, fft_length=None, axis=-1, padding='right'):
    super().__init__()
    self.fft_length = fft_length
    self._axis = axis
    self._padding = padding

  def _get_fft_length(self, input_size):
    return self.fft_length or input_size

  def _get_output_length(self, input_size):
    return self._get_fft_length(input_size) // 2 + 1

  def get_output_shape(self, input_shape, *, constants=None):
    shape = list(input_shape)
    axis = (
        _validate_and_normalize_axis(
            self._axis, (None, None) + tuple(input_shape)
        )
        - 2
    )
    shape[axis] = self._get_output_length(shape[axis])
    return tuple(shape)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return mx.complex64

  @types.check_layer
  def layer(self, x, *, constants=None):
    if x.ndim <= 2:
      raise ValueError('RFFT requires rank >= 3 input.')
    axis = _validate_and_normalize_axis(self._axis, x.shape)
    fft_len = self._get_fft_length(x.shape[axis])
    x = _pad_or_truncate_for_fft(x, axis, fft_len, self._padding)

    def rfft_fn(v):
      if v.dtype == mx.bfloat16:
        v = v.astype(mx.float32)
      return mx.fft.rfft(v, n=fft_len, axis=axis)

    return x.apply_values(rfft_fn)

  @classmethod
  def from_config(cls, config):
    return cls(
        fft_length=config.fft_length,
        axis=config.axis,
        padding=config.padding,
    )


class IRFFT(types.Stateless):
  """Applies IRFFT to a channel dimension."""

  def __init__(
      self,
      *,
      fft_length=None,
      frame_length=None,
      axis=-1,
      padding='right',
  ):
    super().__init__()
    self.fft_length = fft_length
    self.frame_length = frame_length
    self._axis = axis
    self._padding = padding

  def _get_fft_length(self, input_size):
    return self.fft_length or (input_size - 1) * 2

  def _get_output_length(self, input_size):
    return self.frame_length or self._get_fft_length(input_size)

  def get_output_shape(self, input_shape, *, constants=None):
    shape = list(input_shape)
    axis = (
        _validate_and_normalize_axis(
            self._axis, (None, None) + tuple(input_shape)
        )
        - 2
    )
    shape[axis] = self._get_output_length(shape[axis])
    return tuple(shape)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return mx.float32

  @types.check_layer
  def layer(self, x, *, constants=None):
    if x.ndim <= 2:
      raise ValueError('IRFFT requires rank >= 3 input.')
    axis = _validate_and_normalize_axis(self._axis, x.shape)
    fft_len = self._get_fft_length(x.shape[axis])
    x = x.apply_values(lambda v: mx.fft.irfft(v, n=fft_len, axis=axis))
    required = self.frame_length or x.shape[axis]
    return _pad_or_truncate_for_fft(x, axis, required, self._padding)

  @classmethod
  def from_config(cls, config):
    return cls(
        fft_length=config.fft_length,
        frame_length=config.frame_length,
        axis=config.axis,
        padding=config.padding,
    )


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------


class STFT(types.SequenceLayer):
  """Short-Time Fourier Transform."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    frame_length: int = 0
    frame_step: int = 0
    fft_length: int = 0
    window_fn: object = None
    time_padding: str = 'reverse_causal_valid'
    fft_padding: str = 'right'
    output_magnitude: bool = False
    name: str | None = None

    def make(self) -> 'STFT':
      return STFT.from_config(self)

  def __init__(
      self,
      *,
      frame_length,
      frame_step,
      fft_length,
      window_fn=None,
      time_padding='reverse_causal_valid',
      fft_padding='right',
      output_magnitude=False,
  ):
    super().__init__()
    self._frame_length = frame_length
    self._frame_step = frame_step
    self._fft_length = fft_length
    self._window_fn = window_fn or hann_window
    self._time_padding = time_padding
    self._fft_padding = fft_padding
    self._output_magnitude = output_magnitude

    self.framer = Frame(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=time_padding,
    )
    self.fft = RFFT(
        fft_length=fft_length,
        axis=2,
        padding=fft_padding,
    )

  @property
  def supports_step(self):
    return self.framer.supports_step

  @property
  def block_size(self):
    return self.framer.block_size

  @property
  def output_ratio(self):
    return self.framer.output_ratio

  @property
  def input_latency(self):
    return self.framer.input_latency

  def get_output_shape(self, input_shape, *, constants=None):
    frame_shape = self.framer.get_output_shape(input_shape, constants=constants)
    return self.fft.get_output_shape(frame_shape, constants=constants)

  def get_output_dtype(self, input_dtype, *, constants=None):
    fft_dtype = self.fft.get_output_dtype(input_dtype, constants=constants)
    if self._output_magnitude:
      return mx.float32
    return fft_dtype

  def _apply_window(self, x):
    if self._window_fn:
      window = self._window_fn(self._frame_length)
      window = mx.array(window, dtype=x.dtype)
      shape = [1] * x.ndim
      shape[2] = self._frame_length
      window = window.reshape(shape)
      return x.apply_values_masked(lambda v: v * window)
    return x

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    return self.framer.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    framed, state = self.framer.step(x, state, constants=constants)
    framed = self._apply_window(framed)
    dft = self.fft.layer(framed, constants=constants)
    if self._output_magnitude:
      dft = dft.apply_values_masked(lambda v: mx.abs(v))
    return dft, state

  @types.check_layer
  def layer(self, x, *, constants=None):
    framed = self.framer.layer(x, constants=constants)
    framed = self._apply_window(framed)
    dft = self.fft.layer(framed, constants=constants)
    if self._output_magnitude:
      dft = dft.apply_values_masked(lambda v: mx.abs(v))
    return dft

  @classmethod
  def from_config(cls, config):
    return cls(
        frame_length=config.frame_length,
        frame_step=config.frame_step,
        fft_length=config.fft_length,
        window_fn=config.window_fn,
        time_padding=config.time_padding,
        fft_padding=config.fft_padding,
        output_magnitude=config.output_magnitude,
    )


# ---------------------------------------------------------------------------
# InverseSTFT
# ---------------------------------------------------------------------------


class InverseSTFT(types.SequenceLayer):
  """Inverse Short-Time Fourier Transform."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    frame_length: int = 0
    frame_step: int = 0
    fft_length: int = 0
    window_fn: object = None
    time_padding: str = 'causal'
    fft_padding: str = 'right'
    name: str | None = None

    def make(self) -> 'InverseSTFT':
      return InverseSTFT.from_config(self)

  def __init__(
      self,
      *,
      frame_length,
      frame_step,
      fft_length,
      window_fn=None,
      time_padding='causal',
      fft_padding='right',
  ):
    super().__init__()
    self._frame_length = frame_length
    self._frame_step = frame_step
    self._fft_length = fft_length
    self._window_fn = window_fn or hann_window
    self._time_padding = time_padding
    self._fft_padding = fft_padding

    self.overlap_add = OverlapAdd(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=time_padding,
    )
    self.irfft = IRFFT(
        fft_length=fft_length,
        frame_length=frame_length,
        axis=2,
        padding=fft_padding,
    )

  @property
  def supports_step(self):
    return self.overlap_add.supports_step

  @property
  def block_size(self):
    return 1

  @property
  def output_ratio(self):
    return self.overlap_add.output_ratio

  @property
  def input_latency(self):
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    irfft_shape = list(
        self.irfft.get_output_shape(input_shape, constants=constants)
    )
    irfft_shape[0] = self._frame_length
    return self.overlap_add.get_output_shape(irfft_shape, constants=constants)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.irfft.get_output_dtype(input_dtype, constants=constants)

  def _apply_window(self, irfft):
    """Pad/truncate to frame_length and apply window."""
    fft_len = irfft.shape[2]
    if fft_len > self._frame_length:
      irfft = irfft.apply_values_masked(lambda v: v[:, :, : self._frame_length])
    elif fft_len < self._frame_length:
      pad_amount = self._frame_length - fft_len
      if self._fft_padding == 'center':
        pl = pad_amount // 2
        pr = pad_amount - pl
      else:
        pl, pr = 0, pad_amount
      pad_widths = [(0, 0)] * irfft.ndim
      pad_widths[2] = (pl, pr)
      irfft = irfft.apply_values_masked(mx.pad, pad_widths)

    if self._window_fn:
      window = self._window_fn(self._frame_length)
      window = mx.array(window, dtype=irfft.dtype)
      shape = [1] * irfft.ndim
      shape[2] = self._frame_length
      window = window.reshape(shape)
      irfft = irfft.apply_values_masked(lambda v: v * window)
    return irfft

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    irfft_spec = self.irfft.get_output_spec(input_spec, constants=constants)
    irfft_shape = list(irfft_spec.shape)
    irfft_shape[0] = self._frame_length
    irfft_spec = bt.ShapeDType(tuple(irfft_shape), irfft_spec.dtype)
    return self.overlap_add.get_initial_state(
        batch_size, irfft_spec, constants=constants
    )

  @types.check_step
  def step(self, x, state, *, constants=None):
    if x.ndim < 3:
      raise ValueError(f'Expected [b,t,fft_bins,...] input, got {x.shape}.')
    irfft = self.irfft.layer(x, constants=constants)
    irfft = self._apply_window(irfft)
    ola, state = self.overlap_add.step(irfft, state, constants=constants)
    return ola, state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if x.ndim < 3:
      raise ValueError(f'Expected [b,t,fft_bins,...] input, got {x.shape}.')
    irfft = self.irfft.layer(x, constants=constants)
    irfft = self._apply_window(irfft)
    ola = self.overlap_add.layer(irfft, constants=constants)
    return ola

  @classmethod
  def from_config(cls, config):
    return cls(
        frame_length=config.frame_length,
        frame_step=config.frame_step,
        fft_length=config.fft_length,
        window_fn=config.window_fn,
        time_padding=config.time_padding,
        fft_padding=config.fft_padding,
    )


# ---------------------------------------------------------------------------
# LinearToMelSpectrogram
# ---------------------------------------------------------------------------


class LinearToMelSpectrogram(types.PreservesType, types.Stateless):
  """Converts linear spectrogram to mel spectrogram."""

  def __init__(
      self,
      *,
      num_mel_bins,
      sample_rate,
      lower_edge_hertz,
      upper_edge_hertz,
  ):
    super().__init__()
    self.num_mel_bins = num_mel_bins
    self.sample_rate = sample_rate
    self.lower_edge_hertz = lower_edge_hertz
    self.upper_edge_hertz = upper_edge_hertz
    self._cached_weights = None
    self._cached_num_bins = None
    self._cached_dtype = None

  def get_output_shape(self, input_shape, *, constants=None):
    if not input_shape:
      raise ValueError('LinearToMelSpectrogram requires rank >= 1 input.')
    return tuple(input_shape[:-1]) + (self.num_mel_bins,)

  def _get_weights(self, num_bins, dtype):
    if (
        self._cached_weights is None
        or self._cached_num_bins != num_bins
        or self._cached_dtype != dtype
    ):
      weights = linear_to_mel_weight_matrix(
          num_mel_bins=self.num_mel_bins,
          num_spectrogram_bins=num_bins,
          sample_rate=self.sample_rate,
          lower_edge_hertz=self.lower_edge_hertz,
          upper_edge_hertz=self.upper_edge_hertz,
      )
      self._cached_weights = mx.array(weights, dtype=dtype)
      self._cached_num_bins = num_bins
      self._cached_dtype = dtype
    return self._cached_weights

  @types.check_layer
  def layer(self, x, *, constants=None):
    weights = self._get_weights(x.shape[-1], x.dtype)
    return x.apply_values_masked(lambda v: v @ weights)

  @classmethod
  def from_config(cls, config):
    return cls(
        num_mel_bins=config.num_mel_bins,
        sample_rate=config.sample_rate,
        lower_edge_hertz=config.lower_edge_hertz,
        upper_edge_hertz=config.upper_edge_hertz,
    )
