# Copyright 2023 Google LLC
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
"""DSP layers."""

import fractions
import functools
from typing import Callable, Optional, Tuple

import numpy as np
from sequence_layers.tensorflow import convolution
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class Delay(types.SequenceLayer):
  """A layer that delays its input by length timesteps."""

  def __init__(self, length: int, name: Optional[str] = None):
    super().__init__(name=name)
    if length < 0:
      raise ValueError(
          f'Negative delay ({length}) is not supported by sl.Delay layer.'
      )
    self._length = length

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    if self._length == 0:
      return ()

    channels_dims = x.values.shape.dims[2:]
    batch_size = utils.smart_dimension_size(x.values, 0)
    return types.Sequence(
        tf.zeros(
            [batch_size, self._length] + channels_dims, dtype=x.values.dtype
        ),
        tf.ones((batch_size, self._length), dtype=x.mask.dtype),
    )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if not input_shape.is_fully_defined():
      raise ValueError(
          '%s depends on input shape, but input has unknown inner shape: %s'
          % (self, input_shape)
      )
    return input_shape

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    if self._length == 0:
      return x, state
    block_size = utils.smart_dimension_size(x.values, 1)
    x = state.concatenate(x)
    values, state_values = tf.split(
        x.values, [block_size, self._length], axis=1
    )
    mask, state_mask = tf.split(x.mask, [block_size, self._length], axis=1)
    state = types.Sequence(state_values, state_mask)
    # No masking because timesteps are unchanged, just delayed.
    return types.Sequence(values, mask), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    if self._length == 0:
      return x
    time = utils.smart_dimension_size(x.values, 1)
    x = x.pad_time(self._length, 0, valid=True)
    # No masking because timesteps are unchanged, just delayed.
    return x[:, :time]


def _pad_or_truncate_for_fft(
    x: types.Sequence, required_input_length: int, padding: str
) -> types.Sequence:
  """Pads or truncates the provided sequence for an FFT/IFFT/RFFT/IRFFT."""
  input_dim = x.values.shape.dims[-1].value
  if input_dim == required_input_length:
    return x
  if input_dim < required_input_length:
    pad_amount = required_input_length - input_dim
    if padding == 'center':  # Center padding.
      pad_left = pad_amount // 2
      pad_right = pad_amount - pad_left
    else:
      assert padding == 'right'
      pad_left = 0
      pad_right = pad_amount
    rank = x.values.shape.rank
    return x.apply_values(
        lambda v: tf.pad(v, [[0, 0]] * (rank - 1) + [[pad_left, pad_right]])
    )
  else:
    assert input_dim > required_input_length

    def slice_last_dim(v, start, length):
      # Avoid a strided slice with ellipsis for tf.lite compatibility.
      return tf.slice(
          v,
          [0] * (v.shape.rank - 1) + [start],
          [-1] * (v.shape.rank - 1) + [length],
      )

    if padding == 'center':
      trim_left = (input_dim - required_input_length) // 2
      return x.apply_values(slice_last_dim, trim_left, required_input_length)
    else:
      assert padding == 'right'
      return x.apply_values(slice_last_dim, 0, required_input_length)


def _validate(fft_length: int, padding: str):
  if fft_length is not None and fft_length <= 0:
    raise ValueError('fft_length must be positive, got: %d' % fft_length)
  if padding not in ('center', 'right'):
    raise ValueError('padding must be "center" or "right", got: %s' % padding)


class FFTBase(types.Stateless):
  """A base class for shared FFT logic."""

  def __init__(
      self, fft_length: Optional[int] = None, padding: str = 'right', name=None
  ):
    """Creates an FFT layer.

    Args:
      fft_length: The length of the FFT to perform.
      padding: One of 'right' or 'center'. Whether to pad inputs to `fft_length`
        by padding with zeros from the right or equally on both sides.
      name: An optional name for this layer.
    """
    super().__init__(name=name)
    _validate(fft_length, padding)
    self._fft_length = fft_length
    self._padding = padding

  def _get_fft_fn(self, fft_length: int) -> Callable[..., tf.Tensor]:
    raise NotImplementedError

  def _get_fft_length(self, input_shape: tf.TensorShape) -> int:
    return self._fft_length or input_shape.dims[-1].value

  def _get_required_input_length(self, input_shape: tf.TensorShape) -> int:
    return self._get_fft_length(input_shape)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    # FFT operates over the inner-most dimension.
    return input_shape[:-1].concatenate(self._get_fft_length(input_shape))

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    required_input_length = self._get_required_input_length(x.channel_shape)
    fft_length = self._get_fft_length(x.channel_shape)
    fft_fn = self._get_fft_fn(fft_length)
    x = _pad_or_truncate_for_fft(x, required_input_length, self._padding)
    return x.apply_values(fft_fn).mask_invalid()


class FFT(FFTBase):
  """A layer that applies an FFT to the channels dimension."""

  def _get_fft_fn(self, fft_length: int) -> Callable[..., tf.Tensor]:
    del fft_length
    return tf.signal.fft


class IFFT(FFTBase):
  """A layer that applies an IFFT to the channels dimension."""

  def _get_fft_fn(self, fft_length: int) -> Callable[..., tf.Tensor]:
    del fft_length
    return tf.signal.ifft


class RFFT(FFTBase):
  """A layer that applies an RFFT to the channels dimension."""

  def _get_fft_fn(self, fft_length: int) -> Callable[..., tf.Tensor]:
    return functools.partial(tf.signal.rfft, fft_length=[fft_length])

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return input_shape[:-1].concatenate(
        self._get_fft_length(input_shape) // 2 + 1
    )

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    return tf.complex64


class IRFFT(FFTBase):
  """A layer that applies an IRFFT to the channels dimension."""

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    return tf.float32

  def _get_fft_fn(self, fft_length: int):
    return functools.partial(tf.signal.irfft, fft_length=[fft_length])

  def _get_fft_length(self, input_shape: tf.TensorShape) -> int:
    return self._fft_length or (input_shape.dims[-1].value - 1) * 2

  def _get_required_input_length(self, input_shape: tf.TensorShape) -> int:
    return self._get_fft_length(input_shape) // 2 + 1


class RDFT(types.Stateless):
  """Computes the Real Discrete Fourier Transform using a matrix multiply.

  This matches tf.signal.rfft's conventions in terms of normalization and
  conjugation.
  """

  def __init__(
      self,
      dft_length: int,
      padding: str = 'right',
      output_magnitude: bool = False,
      name=None,
  ):
    """Creates an RDFT layer.

    Args:
      dft_length: The length of the DFT to perform.
      padding: One of 'right' or 'center'. Whether to pad inputs to `dft_length`
        by padding with zeros from the right or equally on both sides.
      output_magnitude: If true, output magnitude of the RDFT, otherwise the
        complex spectrum.
      name: An optional name for this layer.
    """
    super().__init__(name=name)
    self._dft_length = dft_length
    self._padding = padding
    self._output_magnitude = output_magnitude
    self._real_dft_tensor = None
    self._imag_dft_tensor = None

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return input_shape[:-1].concatenate(self._dft_length // 2 + 1)

  def get_output_dtype(self, input_dtype):
    return tf.float32 if self._output_magnitude else tf.complex64

  def build(self, input_shape):
    if self._real_dft_tensor is not None:
      return

    # See https://en.wikipedia.org/wiki/DFT_matrix
    omega = (0 + 1j) * 2.0 * np.pi / float(self._dft_length)
    # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
    dft_matrix = np.exp(
        omega
        * np.outer(np.arange(self._dft_length), np.arange(self._dft_length))
    )

    # We are right-multiplying by the DFT matrix, and we are keeping
    # only the first half ("positive frequencies").
    # So discard the second half of rows, but transpose the array for
    # right-multiplication.
    # The DFT matrix is symmetric, so we could have done it more
    # directly, but this reflects our intention better.
    complex_dft_matrix_kept_values = dft_matrix[
        : self._dft_length // 2 + 1, :
    ].transpose()
    self._real_dft_tensor = tf.constant(
        np.real(complex_dft_matrix_kept_values).astype(np.float32),
        name='real_dft_matrix',
    )
    self._imag_dft_tensor = tf.constant(
        np.imag(complex_dft_matrix_kept_values).astype(np.float32),
        name='imaginary_dft_matrix',
    )

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self.build(x.channel_shape)
    padded_signal = _pad_or_truncate_for_fft(
        x, self._dft_length, padding=self._padding
    )
    result_real_part = tf.matmul(padded_signal.values, self._real_dft_tensor)
    result_imag_part = tf.matmul(padded_signal.values, self._imag_dft_tensor)
    if self._output_magnitude:
      values = tf.sqrt(
          tf.square(result_real_part) + tf.square(result_imag_part)
      )
    else:
      # For compatibility with tf.signal.rfft, use the conjugate.
      values = tf.complex(result_real_part, -result_imag_part)
    return types.Sequence(values, padded_signal.mask).mask_invalid()


class Frame(types.SequenceLayer):
  """Produce a sequence of overlapping frames of the input sequence."""

  def __init__(self, frame_length: int, frame_step: int, name=None):
    """Creates a framing layer which produces overlapping frames of the input.

    Args:
      frame_length: The length of frames to generate.
      frame_step: The step or stride of the frames.
      name: An optional name for the layer.
    """
    super().__init__(name=name)
    self._frame_length = frame_length
    self._frame_step = frame_step
    if frame_length <= 0:
      raise ValueError('frame_length must be positive, got: %d' % frame_length)
    if frame_step <= 0:
      raise ValueError('frame_step must be positive, got: %d' % frame_step)
    self._buffer_width = frame_length - 1

  @property
  def block_size(self) -> int:
    return self._frame_step

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._frame_step)

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    if self._buffer_width == 0:
      return ()
    inner_shape = x.channel_shape
    batch_size = utils.smart_dimension_size(x.values, 0)
    return types.Sequence(
        tf.zeros(
            [batch_size, self._buffer_width] + inner_shape.as_list(),
            dtype=x.values.dtype,
        ),
        tf.ones((batch_size, self._buffer_width), dtype=x.mask.dtype),
    )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if not input_shape.is_fully_defined():
      raise ValueError(
          '%s depends on input shape, but input has unknown inner shape: %s'
          % (self, input_shape)
      )
    return tf.TensorShape([self._frame_length]).concatenate(input_shape)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    if self._buffer_width > 0:
      state = state.concatenate(x)
    else:
      state = x
    values = tf.signal.frame(
        state.values,
        frame_length=self._frame_length,
        frame_step=self._frame_step,
        pad_end=False,
        axis=1,
    )
    mask = convolution.compute_conv_mask(
        state.mask,
        kernel_size=self._frame_length,
        stride=self._frame_step,
        dilation_rate=1,
        padding='causal',
    )

    # Keep the last buffer_width samples as state.
    if self._buffer_width > 0:
      state = state[:, -self._buffer_width :]
    else:
      state = ()
    return types.Sequence(values, mask).mask_invalid(), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    causal_padding = self._frame_length - 1
    x = x.pad_time(causal_padding, 0, valid=True)
    values = tf.signal.frame(
        x.values,
        frame_length=self._frame_length,
        frame_step=self._frame_step,
        pad_end=False,
        axis=1,
    )
    mask = convolution.compute_conv_mask(
        x.mask,
        kernel_size=self._frame_length,
        stride=self._frame_step,
        dilation_rate=1,
        padding='causal',
    )
    return types.Sequence(values, mask).mask_invalid()


class BaseSTFT(types.SequenceLayer):
  """Base class for STFT-like logic."""

  def __init__(
      self,
      frame_length: int,
      frame_step: int,
      fft_length: int,
      window_fn,
      padding: str,
      use_dft_matrix: bool,
      output_magnitude: bool,
      name=None,
  ):
    """Creates a BaseSTFT layer."""
    super().__init__(name=name)
    self._frame_length = frame_length
    self._framer = Frame(frame_length, frame_step)
    self._window = None
    self._window_fn = window_fn
    self._output_magnitude = output_magnitude
    if use_dft_matrix:
      self._dft = RDFT(
          fft_length, padding=padding, output_magnitude=output_magnitude
      )
    else:
      self._dft = RFFT(fft_length, padding=padding)

  @property
  def block_size(self) -> int:
    return self._framer.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._framer.output_ratio

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    return self._framer.get_initial_state(x, constants)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank != 0:
      raise ValueError(
          'Expected [b, t] input, but got %s:'
          % tf.TensorShape([None, None]).concatenate(input_shape)
      )
    frame_shape = self._framer.get_output_shape(input_shape, constants)
    dft_shape = self._dft.get_output_shape(frame_shape, constants)
    return dft_shape

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    if self._output_magnitude:
      return tf.float32
    else:
      return self._dft.get_output_dtype(input_dtype)

  def build(self, input_shape):
    if self._window is None and self._window_fn is not None:
      self._window = self._window_fn(self._frame_length)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    self.build(x.channel_shape)
    framed, state = self._framer.step(x, state, training, constants)
    if self._window is not None:
      framed = framed.apply_values(lambda v: v * self._window)
    dft, _ = self._dft.step(framed, (), training, constants)
    if self._output_magnitude and dft.values.dtype.is_complex:
      dft = dft.apply_values(tf.abs)
    return dft, state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self.build(x.channel_shape)
    framed = self._framer.layer(
        x, training, initial_state=initial_state, constants=constants
    )
    if self._window is not None:
      framed = framed.apply_values(lambda v: v * self._window)
    dft = self._dft.layer(framed, training, constants=constants)
    if self._output_magnitude and dft.values.dtype.is_complex:
      dft = dft.apply_values(tf.abs)
    return dft


class STFT(BaseSTFT):
  """Computes the Short-time Fourier Transform of input signals.

  When used with 'right' padding, equivalent to tf.signal.stft.
  """

  def __init__(
      self,
      frame_length: int,
      frame_step: int,
      fft_length: int,
      window_fn=tf.signal.hann_window,
      padding: str = 'right',
      use_dft_matrix: bool = False,
      name=None,
  ):
    """Creates an STFT layer.

    Args:
      frame_length: The frame length of the STFT.
      frame_step: The frame step of the STFT.
      fft_length: The FFT length of the STFT.
      window_fn: The window to use in the STFT. A callable that takes a window
        length and a dtype keyword argument and returns a [window_length] Tensor
        of samples. If None, no windowing is used.
      padding: One of 'right' or 'center'. Whether to pad inputs to `fft_length`
        by padding with zeros from the right or equally on both sides.
      use_dft_matrix: If true, uses an RDFT matrix multiply to compute the
        spectrum instead of an RFFT. Useful for tf.lite support.
      name: An optional name for the layer.
    """
    super().__init__(
        frame_length,
        frame_step,
        fft_length,
        window_fn,
        padding,
        use_dft_matrix,
        output_magnitude=False,
        name=name,
    )


class Spectrogram(BaseSTFT):
  """Computes a magnitude spectrogram of input signals.

  When used with 'right' padding, equivalent to tf.abs(tf.signal.stft(.)).
  """

  def __init__(
      self,
      frame_length: int,
      frame_step: int,
      fft_length: int,
      window_fn=tf.signal.hann_window,
      padding: str = 'right',
      use_dft_matrix: bool = False,
      name=None,
  ):
    """Creates a Spectrogram layer.

    Args:
      frame_length: The frame length of the STFT.
      frame_step: The frame step of the STFT.
      fft_length: The FFT length of the STFT.
      window_fn: The window to use in the STFT. A callable that takes a window
        length and a dtype keyword argument and returns a [window_length] Tensor
        of samples. If None, no windowing is used.
      padding: One of 'right' or 'center'. Whether to pad inputs to `fft_length`
        by padding with zeros from the right or equally on both sides.
      use_dft_matrix: If true, uses an RDFT matrix multiply to compute the
        spectrum instead of an RFFT. Useful for tf.lite support.
      name: An optional name for the layer.
    """
    super().__init__(
        frame_length,
        frame_step,
        fft_length,
        window_fn,
        padding,
        use_dft_matrix,
        output_magnitude=True,
        name=name,
    )


class LinearToMelSpectrogram(types.Stateless):
  """Converts linear-scale spectrogram to a mel-scale spectrogram.

  The spectrogram magnitudes should be uncompressed, *not* log compressed.
  Summation is performed across bands, which in log magnitdue would be
  multiplication.
  """

  def __init__(
      self,
      num_mel_bins: int,
      sample_rate: float,
      lower_edge_hertz: float,
      upper_edge_hertz: float,
      name: Optional[str] = None,
  ):
    """Creates a LinearToMelSpectrogram layer.

    Args:
      num_mel_bins: The number of mel bins to compute.
      sample_rate: The sample rate of the spectrum.
      lower_edge_hertz: Lower bound on the frequencies to be included in the mel
        spectrum. This corresponds to the lower edge of the lowest triangular
        band.
      upper_edge_hertz: The desired top edge of the highest frequency band.
      name: An optional name for the layer.
    """
    super().__init__(name=name)
    self._num_mel_bins = num_mel_bins
    self._sample_rate = sample_rate
    self._lower_edge_hertz = lower_edge_hertz
    self._upper_edge_hertz = upper_edge_hertz
    self._mel_kernel = None

  def _get_mel_kernel(self, num_spectrogram_bins, dtype):
    # TODO(rryan): Extract this into a general helper for caching tensors.
    # We recompute the mel kernel if:
    should_recompute = (
        # We have not cached the mel kernel yet.
        self._mel_kernel is None
        or
        # The dtype does not match the requested dtype.
        self._mel_kernel.dtype != dtype
        or
        # We are in Eager mode but the cached tensor is not an EagerTensor.
        (tf.executing_eagerly() and not hasattr(self._mel_kernel, 'numpy'))
        or
        # We are in graph mode, the cached tensor is not an Eager tensor and its
        # graph is not this graph.
        (
            not tf.executing_eagerly()
            and not hasattr(self._mel_kernel, 'numpy')
            and self._mel_kernel.graph != tf1.get_default_graph()
        )
    )
    if should_recompute:
      self._mel_kernel = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=self._num_mel_bins,
          num_spectrogram_bins=num_spectrogram_bins,
          sample_rate=self._sample_rate,
          lower_edge_hertz=self._lower_edge_hertz,
          upper_edge_hertz=self._upper_edge_hertz,
          dtype=dtype,
      )
    return self._mel_kernel

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if input_shape.rank == 0:
      raise ValueError(
          f'{self} requires input with at least rank 1, got: {input_shape}'
      )
    return input_shape[:-1].concatenate(self._num_mel_bins)

  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    num_spectrogram_bins = x.values.shape.dims[-1].value
    # Do not cache the linear_to_mel matrix since we may be inside a FuncGraph.
    linear_to_mel = self._get_mel_kernel(num_spectrogram_bins, x.values.dtype)
    # No masking is required because a weight matrix applied to a masked
    # timestep is also masked.
    return x.apply_values(tf.tensordot, linear_to_mel, 1)
