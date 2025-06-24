# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Signal utilities, generally ported from tensorflow.signal."""

from collections.abc import Sequence
import functools
import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal
from sequence_layers.jax import types
from sequence_layers.jax import utils

# Type for the name argument of the get_window function of scipy
# The functions covered by the alternative types are listed
# in the in-line comments.
_WindowNameT = (
    str  # Window function without parameters.
    | float  # Kaiser window's beta parameter.
    | tuple[
        str, float
    ]  # kaiser, kaiser_bessel_derived, gaussian, general_hamming, dpss, chebwin
    | tuple[str, float, float]  # general_gaussian
    | tuple[str | Sequence[float]]  # general_cosine
)


def _raised_cosine_window(
    window_length: int,
    periodic: bool,
    dtype: npt.DTypeLike,
    a: float,
    b: float,
) -> np.ndarray:
  """Computes a raised cosine window. Ported from tf.signal."""
  if window_length == 1:
    return np.ones([1], dtype=dtype)

  even = 1 - window_length % 2
  n = np.asarray(window_length + int(periodic) * even - 1, dtype=dtype)
  count = np.arange(window_length, dtype=dtype)
  cos_arg = 2 * np.pi * count / n
  return a - b * np.cos(cos_arg)


def hann_window(
    window_length: int,
    periodic: bool = True,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
  """Computes a hann window. Ported from tf.signal."""
  # Note: numpy.hanning does not have the `periodic` and `dtype` parameters.
  return _raised_cosine_window(window_length, periodic, dtype, 0.5, 0.5)


def hamming_window(
    window_length: int,
    periodic: bool = True,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
  """Computes a Hamming window."""
  # The coefficient is chosen identical to scipy.signal.windows.
  # Note: numpy.hamming does not have the `periodic` and `dtype` parameters.
  a0 = 0.54
  return _raised_cosine_window(window_length, periodic, dtype, a0, 1.0 - a0)


def get_window(
    name: _WindowNameT,
    window_length: int,
    periodic: bool = True,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
  """A simple wrapper around scipy.signal.get_windwow.

  This function wraps the scipy.signal.get_window function.
  It can provide all the windows listed at the following link.
  https://docs.scipy.org/doc/scipy/reference/signal.windows.html

  If the window requires no parameters, then `name` can be a string.

  If the window requires parameters, then `name` must be a tuple
  with the first argument the string name of the window, and the next
  arguments the needed parameters.

  If `name` is a floating point number, it is interpreted as the beta
  parameter of the `~scipy.signal.windows.kaiser` window.

  The `periodic` argument is helpful to get windows that satisfy
  the COLA (constant overlap-add) condition necessary to get perfect
  reconstruction from overlap-add procedure with certain frame_step sizes.
  https://ccrma.stanford.edu/~jos/sasp/Overlap_Add_Decomposition.html#22206
  Note that COLA, is different from NOLA (non-zero overlap-add), which
  is the required condition for perfect reconstruction from STFT/iSTFT.

  Args:
    name: The name of the window.
    window_length: The length of the window.
    periodic: Set to True if the window is to be used with an STFT.
    dtype: Desired data type of the output array.

  Returns:
    An array containing the window of desired type and length.
  """
  even = window_length % 2 == 0
  return np.asarray(
      sp_signal.get_window(name, Nx=window_length, fftbins=even and periodic),
      dtype=dtype,
  )


def get_window_fn(
    name: _WindowNameT,
) -> Callable[..., np.ndarray]:
  """Return an API compatible window function of the requested type.

  This function wraps the scipy.signal.get_window function.
  It can provide all the windows listed at the following link.
  https://docs.scipy.org/doc/scipy/reference/signal.windows.html

  Args:
    name: The name of the window.

  Returns:
    A function taking as argument a length, a periodic flag, and a dtype.
  """

  return functools.partial(get_window, name)


def inverse_stft_window_fn(frame_step: int, forward_window_fn=hann_window):
  """Generates a window function that can be used in `inverse_stft`.

  Constructs a window that is equal to the forward window with a further
  pointwise amplitude correction.  `inverse_stft_window_fn` is equivalent to
  `forward_window_fn` in the case where it would produce an exact inverse.

  Args:
    frame_step: The number of samples to step.
    forward_window_fn: window_fn used in the forward STFT transform.

  Returns:
    A callable that takes a window length and a `dtype` keyword argument and
      returns a `[window_length]` `Tensor` of samples in the provided datatype.
      The returned window is suitable for reconstructing original waveform in
      inverse_stft.
  """

  def inverse_stft_window_fn_inner(frame_length, dtype):
    """Computes a window that can be used in `inverse_stft`.

    Args:
      frame_length: The window length in samples.
      dtype: Data type of waveform passed to `stft`.

    Returns:
      A window suitable for reconstructing the original waveform with an inverse
      STFT.
    """
    # Use equation 7 from Griffin + Lim.
    forward_window = forward_window_fn(frame_length, dtype=dtype)
    denom = jnp.square(forward_window)
    overlaps = -(-frame_length // frame_step)  # Ceiling division.
    denom = jnp.pad(denom, [(0, overlaps * frame_step - frame_length)])
    denom = jnp.reshape(denom, [overlaps, frame_step])
    denom = jnp.sum(denom, 0, keepdims=True)
    denom = jnp.tile(denom, [overlaps, 1])
    denom = jnp.reshape(denom, [overlaps * frame_step])
    denom = denom[:frame_length]
    return jnp.where(denom == 0.0, 0, forward_window / denom)

  return inverse_stft_window_fn_inner


def overlap_and_add(signal: jax.Array, frame_step: int) -> jax.Array:
  """Reconstructs a signal from a framed representation.

  Adds potentially overlapping frames of a signal with shape
  `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
  The resulting tensor has shape `[..., output_size]` where

      output_size = (frames - 1) * frame_step + frame_length

  Args:
    signal: A [..., frames, frame_length] array.
    frame_step: An integer denoting overlap offsets. Must be less than or equal
      to `frame_length`.

  Returns:
    An array with shape `[..., output_size]` containing the overlap-added
    frames of `signal`'s inner-most two dimensions.

  Raises:
    ValueError: If `signal`'s rank is less than 2, or `frame_step` is not a
      scalar integer.
  """
  signal_shape = signal.shape

  # All dimensions that are not part of the overlap-and-add. Can be empty for
  # rank 2 inputs.
  outer_dimensions = signal_shape[:-2]
  outer_rank = len(outer_dimensions)

  def full_shape(inner_shape):
    return outer_dimensions + inner_shape

  frame_length = signal_shape[-1]
  frames = signal_shape[-2]

  # Compute output length.
  output_length = frame_length + frame_step * (frames - 1)

  # If frame_length is equal to frame_step, there's no overlap so just
  # reshape the tensor.
  if frame_length == frame_step:
    return jnp.reshape(signal, outer_dimensions + (output_length,))

  # The following code is documented using this example:
  #
  # frame_step = 2
  # signal.shape = (3, 5)
  # a b c d e
  # f g h i j
  # k l m n o

  # Compute the number of segments, per frame.
  segments = -(-frame_length // frame_step)  # Divide and round up.

  # Pad the frame_length dimension to a multiple of the frame step.
  # Pad the frames dimension by `segments` so that signal.shape = (6, 6)
  # a b c d e 0
  # f g h i j 0
  # k l m n o 0
  # 0 0 0 0 0 0
  # 0 0 0 0 0 0
  # 0 0 0 0 0 0
  paddings = [[0, segments], [0, segments * frame_step - frame_length]]
  outer_paddings = [(0, 0)] * outer_rank
  signal = jnp.pad(signal, outer_paddings + paddings)

  # Reshape so that signal.shape = (6, 3, 2)
  # ab cd e0
  # fg hi j0
  # kl mn o0
  # 00 00 00
  # 00 00 00
  # 00 00 00
  shape = full_shape((frames + segments, segments, frame_step))
  signal = jnp.reshape(signal, shape)

  # Transpose dimensions so that signal.shape = (3, 6, 2)
  # ab fg kl 00 00 00
  # cd hi mn 00 00 00
  # e0 j0 o0 00 00 00

  perm = list(range(outer_rank)) + [
      outer_rank + 1,
      outer_rank + 0,
      outer_rank + 2,
  ]
  signal = jnp.transpose(signal, perm)

  # Reshape so that signal.shape = (18, 2)
  # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0 00 00 00
  shape = full_shape(((frames + segments) * segments, frame_step))
  signal = jnp.reshape(signal, shape)

  # Truncate so that signal.shape = (15, 2)
  # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0
  signal = signal[..., : (frames + segments - 1) * segments, :]

  # Reshape so that signal.shape = (3, 5, 2)
  # ab fg kl 00 00
  # 00 cd hi mn 00
  # 00 00 e0 j0 o0
  shape = full_shape((segments, (frames + segments - 1), frame_step))
  signal = jnp.reshape(signal, shape)

  # Now, reduce over the columns, to achieve the desired sum.
  signal = jnp.sum(signal, -3)

  # Flatten the array.
  shape = full_shape(((frames + segments - 1) * frame_step,))
  signal = jnp.reshape(signal, shape)

  # Truncate to final length.
  signal = signal[..., :output_length]

  return signal


def frame(
    x: jax.Array,
    frame_length: int,
    frame_step: int,
    axis: int = -1,
    pad_mode: tuple[int, int] | str = 'reverse_causal_valid',
    pad_value: complex = 0,
) -> jax.Array:
  """Expands `x`'s `axis` dimension into frames of `frame_length`.

  This is a JAX version of `tf.signal.frame` with convolution-like padding
  support. The output shape is exactly equivalent to convolution of kernel size
  `frame_length` and stride `frame_step`, with padding mode `pad_mode`.

  Args:
    x: A `(..., samples, ...)` array. Rank must be at least 1.
    frame_length: The frame length in samples.
    frame_step: The frame hop size in samples.
    axis: Indicating the axis to frame. Defaults to the last axis. Supports
      negative values for indexing from the end.
    pad_mode: The padding mode to use.
    pad_value: The padding value to use.

  Returns:
    An array of frames, size `(..., num_frames, frame_length, ...)`.
  """
  if frame_length < 0:
    raise ValueError('frame_length must be non-negative.')
  if frame_step < 0:
    raise ValueError('frame_step must be non-negative.')
  if x.ndim < 1:
    raise ValueError('signal must have rank at least 1.')

  axis = axis % x.ndim
  outer_dimensions = x.shape[:axis]
  inner_dimensions = x.shape[axis + 1 :]

  if isinstance(pad_mode, str):
    pad_mode = types.validate_padding(pad_mode)
  num_frames = utils.convolution_padding_output_size(
      x.shape[axis], pad_mode, frame_length, frame_step, dilation_rate=1
  )

  if not num_frames:
    return jnp.zeros(
        outer_dimensions + (0, frame_length) + inner_dimensions,
        dtype=x.dtype,
    )

  # Performance optimization: If frame_length and frame_step have common
  # factors, we can reduce the number of gather indices by dividing frames into
  # "subframes". This can speed up framing by up by 10-20x if frame_length is
  # divisible by frame_step.
  subframe_factor = math.gcd(frame_length, frame_step)

  # If subframing is enabled, pad to a multiple of frame_length. This guarantees
  # it is divisible by subframe_factor.
  pad_left, pad_right = utils.convolution_explicit_padding(
      pad_mode, frame_length, frame_step, dilation_rate=1
  )
  if subframe_factor > 1:
    pad_right += (-x.shape[axis] - pad_left - pad_right) % frame_length

  paddings = [(0, 0)] * x.ndim
  paddings[axis] = (pad_left, pad_right)
  x = jnp.pad(x, paddings, constant_values=jnp.array(pad_value, dtype=x.dtype))

  if subframe_factor > 1:
    assert x.shape[axis] % subframe_factor == 0, (x.shape, subframe_factor)
    x = x.reshape(outer_dimensions + (-1, subframe_factor) + inner_dimensions)

    # After extracting subframe_factor from the frame axis, divide the length
    # and step appropriately.
    frame_length = frame_length // subframe_factor
    frame_step = frame_step // subframe_factor

  # Build a [num_frames, frame_length] array of indices to take from x along
  # the frame axis, where selector[i] corresponds to the range [frame_step *
  # i, frame_step * i + frame_length).
  start_indices = (np.arange(num_frames) * frame_step)[:, np.newaxis]
  window_indices = np.arange(frame_length)[np.newaxis, :]
  x = jnp.reshape(
      jnp.take(x, indices=start_indices + window_indices, axis=axis),
      outer_dimensions
      + (-1, frame_length * subframe_factor)
      + inner_dimensions,
  )
  return x


# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _hertz_to_mel(frequencies_hertz: float | int | np.ndarray) -> np.ndarray:
  """Converts hertz to mel."""
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
  )


def linear_to_mel_weight_matrix(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    sample_rate: int | float = 8000,
    lower_edge_hertz: int | float = 125.0,
    upper_edge_hertz: int | float = 3800.0,
    dtype=np.float32,
) -> np.ndarray:
  r"""Numpy-port of `tf.signal.linear_to_mel_weight_matrix`.

  Copied from google3/third_party/py/praxis/layers/asr/frontend.py.

  Args:
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
      source spectrogram data, which is understood to be `fft_size // 2 + 1`,
      i.e., the spectrogram only contains the nonredundant FFT bins.
    sample_rate: An integer or float `Tensor`. Samples per second of the input
      signal used to create the spectrogram. Used to figure out the frequencies
      corresponding to each spectrogram bin, which dictates how they are mapped
      into the mel scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    dtype: The `DType` of the result matrix. Must be a floating point type.

  Returns:
    An array of shape `[num_spectrogram_bins, num_mel_bins]`.
  Raises:
    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
      ordered, `upper_edge_hertz` is larger than the Nyquist frequency.
  [mel]: https://en.wikipedia.org/wiki/Mel_scale
  """

  # Input validator from tensorflow/python/ops/signal/mel_ops.py#L71
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if lower_edge_hertz < 0.0:
    raise ValueError(
        'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz
    )
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError(
        'lower_edge_hertz %.1f >= upper_edge_hertz %.1f'
        % (lower_edge_hertz, upper_edge_hertz)
    )
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError(
        'upper_edge_hertz must not be larger than the Nyquist '
        'frequency (sample_rate / 2). Got %s for sample_rate: %s'
        % (upper_edge_hertz, sample_rate)
    )

  # For better precision, we internally use float64.  It will not slow down
  # feature extraction because this function is called only once for obtaining
  # a constant matrix.
  internal_dtype = np.float64

  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = np.linspace(
      0.0, nyquist_hertz, num_spectrogram_bins, dtype=internal_dtype
  )[bands_to_zero:]
  spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, np.newaxis]

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  edges = np.linspace(
      _hertz_to_mel(lower_edge_hertz),
      _hertz_to_mel(upper_edge_hertz),
      num_mel_bins + 2,
      dtype=internal_dtype,
  )

  # Split the triples up and reshape them into [1, num_mel_bins] tensors.
  lower_edge_mel, center_mel, upper_edge_mel = (
      edges[:-2][np.newaxis, :],
      edges[1:-1][np.newaxis, :],
      edges[2:][np.newaxis, :],
  )

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel
  )
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel
  )

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

  # Re-add the zeroed lower bins we sliced out above.
  return np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]]).astype(dtype)
