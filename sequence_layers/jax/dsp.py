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
"""DSP layers."""

import abc
import dataclasses
import fractions
import math
from typing import Callable, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution
from sequence_layers.jax import signal
from sequence_layers.jax import types

__all__ = (
    # go/keep-sorted start
    'Delay',
    'FFT',
    'Frame',
    'IFFT',
    'IRFFT',
    'InverseSTFT',
    'LinearToMelSpectrogram',
    'Lookahead',
    'OverlapAdd',
    'RFFT',
    'STFT',
    'Window',
    # go/keep-sorted end
)

_DEFAULT_FFT_PADDING = 'right'

FFTPaddingString = Literal['center', 'right']


class Frame(types.PreservesType, types.SequenceLayer):
  """Produce a sequence of overlapping frames of the input sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Frame layer."""

    # The length of frames to generate.
    frame_length: int
    # The step or stride of the frames.
    frame_step: int
    # Padding to use for the framing. If explicit padding is provided it must
    # sum to frame_length - 1. If 'causal_valid', 'reverse_causal_valid',
    # 'causal', 'reverse_causal', 'semicausal' or if the explicit padding sums
    # to `frame_length - 1`, then the `Frame` is streamable.
    # TODO(b/407645797): Support explicit padding in general.
    padding: tuple[int, int] | types.PaddingModeString = (
        types.PaddingMode.REVERSE_CAUSAL_VALID.value
    )
    # TODO(b/407645797): Remove this and migrate all users to SAME-like padding
    # in explicit mode.
    explicit_padding_is_same_like: bool = False
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      if isinstance(self.padding, str):
        object.__setattr__(
            self, 'padding', types.validate_padding(self.padding)
        )
      elif sum(self.padding) != self.frame_length - 1:
        raise NotImplementedError(
            f'{self.padding=} must sum to {self.frame_length - 1=}'
        )

    def make(self) -> 'Frame':
      return Frame(self, name=self.name)

  config: Config

  def setup(self) -> None:
    if self.config.frame_length <= 0:
      raise ValueError(
          'frame_length must be positive, got: %d' % self.config.frame_length
      )
    if self.config.frame_step <= 0:
      raise ValueError(
          'frame_step must be positive, got: %d' % self.config.frame_step
      )

  @property
  def supports_step(self) -> bool:
    if isinstance(self.config.padding, str):
      return self.config.padding in (
          types.PaddingMode.CAUSAL_VALID.value,
          types.PaddingMode.REVERSE_CAUSAL_VALID.value,
          types.PaddingMode.CAUSAL.value,
          types.PaddingMode.REVERSE_CAUSAL.value,
          types.PaddingMode.SEMICAUSAL.value,
      )
    else:
      past_pad, future_pad = self.config.padding
      # Allow stepping as long as total padding is frame_length - 1.
      return past_pad + future_pad == self.config.frame_length - 1

  @property
  def block_size(self) -> int:
    return self.config.frame_step

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self.config.frame_step)

  @property
  def input_latency(self) -> int:
    if not isinstance(self.config.padding, str):
      past_pad, _ = self.config.padding
      return max(0, (self.config.frame_length - 1) - past_pad)
    match self.config.padding:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        # Causal padding eliminates latency.
        return 0
      case (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL.value
      ):
        # Reverse causal introduces no past padding in layer-wise mode, so we
        # need a full effective_kernel_size kernel to compute the first output
        # layer-wise processing would produce. Since we do not count the current
        # input as part of the latency, the input latency is one smaller than
        # the effective kernel size.
        return self.config.frame_length - 1
      case types.PaddingMode.SEMICAUSAL_FULL.value:
        # SEMICAUSAL_FULL padding only needs to wait for frame_step samples.
        return self.config.frame_step - 1
      case _:
        # Unsupported.
        return 0

  @property
  def output_latency(self) -> int:
    """Returns the output latency of this layer.

    Output latency is defined as the number of output timesteps before the
    step-wise output of the layer matches its layer-wise output.
    """
    match self.config.padding:
      case types.PaddingMode.SEMICAUSAL_FULL.value:
        # The semicausal-full padding has both left and right padding, which
        # requires a different output_latency for correct operations.
        # It only requires to wait for strides[0] - 1 samples, and the
        # output_ratio is also strides[0], which means the latency is 0.
        return 0
      case _:
        # Other cases are handled in the parent class.
        return super().output_latency

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return convolution.conv_receptive_field_per_step(
        self.config.frame_length, self.config.frame_step, 1, self.config.padding
    )

  @property
  def _buffer_width(self) -> int:
    if not isinstance(self.config.padding, str):
      past_pad, future_pad = self.config.padding
      assert past_pad + future_pad == self.config.frame_length - 1
      return (
          self.config.frame_length - 1 - (future_pad % self.config.frame_step)
      )

    match self.config.padding:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(self.config.frame_length - self.config.frame_step, 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (self.config.frame_length - 1)
            // self.config.frame_step
            * self.config.frame_step
        )
      case (
          types.PaddingMode.CAUSAL.value | types.PaddingMode.CAUSAL_VALID.value
      ):
        return self.config.frame_length - 1
      case _:
        raise NotImplementedError(
            f'Unsupported padding mode: {self.config.padding}'
        )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Special case kernel_size 1 since it is stateless.
    if not (buffer_width := self._buffer_width):
      return ()

    match self.config.padding:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.SEMICAUSAL_FULL.value
      ):
        # For CAUSAL_VALID and REVERSE_CAUSAL_VALID, since the mask calculation
        # is a windowed logical-AND, we have to prepend the mask with valid
        # samples, otherwise the first output timesteps that touch this padding
        # become invalid.
        mask = jnp.ones((batch_size, buffer_width), dtype=types.MASK_DTYPE)
      case (
          types.PaddingMode.CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        mask = jnp.zeros((batch_size, buffer_width), dtype=types.MASK_DTYPE)
      case (unused_pad_left, unused_pad_right):
        if self.config.explicit_padding_is_same_like:
          mask = jnp.zeros((batch_size, buffer_width), dtype=types.MASK_DTYPE)
        else:
          mask = jnp.ones((batch_size, buffer_width), dtype=types.MASK_DTYPE)
      case _:
        raise ValueError(
            'Stepwise processing is not supported with padding:'
            f' {self._paddings[0]}'
        )

    return types.MaskedSequence(
        jnp.zeros(
            (batch_size, buffer_width) + input_spec.shape,
            dtype=input_spec.dtype,
        ),
        mask,
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (self.config.frame_length,) + tuple(input_shape)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if self.config.frame_length > 1:
      x = x.mask_invalid()

    if buffer_width := self._buffer_width:
      # Concatenate the new frames with the previous buffer_width frames.
      state = state.concatenate(x)
    else:
      state = x

    values = signal.frame(
        state.values,
        frame_length=self.config.frame_length,
        frame_step=self.config.frame_step,
        # Padding is already applied via the state.
        pad_mode=types.PaddingMode.VALID.value,
        axis=1,
    )
    mask = convolution.compute_conv_mask(
        state.mask,
        kernel_size=self.config.frame_length,
        stride=self.config.frame_step,
        dilation_rate=1,
        padding=self.config.padding,
        is_step=True,
    )

    # Keep the trailing buffer_width samples for the next step.
    if buffer_width:
      state = state[:, -buffer_width:]
    else:
      state = ()

    # Framing can leave unmasked values with non-zero values.
    return types.Sequence(values, mask), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Mask inputs if receptive field is greater than 1.
    if self.config.frame_length > 1:
      x = x.mask_invalid()

    values = signal.frame(
        x.values,
        frame_length=self.config.frame_length,
        frame_step=self.config.frame_step,
        pad_mode=self.config.padding,
        axis=1,
    )
    mask = convolution.compute_conv_mask(
        x.mask,
        kernel_size=self.config.frame_length,
        stride=self.config.frame_step,
        dilation_rate=1,
        padding=self.config.padding,
        is_step=False,
    )

    # If the frame receptive field is 1 then preserve the input mask state.
    result_type = types.Sequence if self.config.frame_length > 1 else type(x)
    return result_type(values, mask)


class OverlapAdd(types.PreservesType, types.SequenceLayer):
  """Overlap adds windows of [b, t, frame_length, ...].

  For a [b, ti, frame_length, ...] input signal, the resulting sequence has
  shape [b, to, ...], where:

  to = (ti - 1) * frame_step + frame_length

  Since OverlapAdd.step can only produce frame_step samples at a time, we trim
  the output of layer() to drop the final `frame_length - frame_step` timesteps.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for OverlapAdd layer."""

    # The length of frames to overlap-add.
    frame_length: int
    # The step or stride of the overlap-add operation.
    frame_step: int
    # Padding to use for the overlap-add. Only 'causal', 'valid', and
    # 'semicausal_full' are supported.
    # The 'semicausal_full' padding will trim (frame_length - frame_step)
    # samples from the right side of the signal. This behavior is intended
    # to remove the part that was padded by an hypothetical Frame layer also
    # using the same padding type. This allows sample aligned reconstruction in
    # layer mode.
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'padding', types.validate_padding(self.padding))

      if self.padding not in (
          types.PaddingMode.CAUSAL.value,
          types.PaddingMode.VALID.value,
          types.PaddingMode.SEMICAUSAL_FULL.value,
      ):
        raise ValueError(f'Unsupported padding mode: {self.padding}')

    def make(self) -> 'OverlapAdd':
      return OverlapAdd(self, name=self.name)

  config: Config

  def setup(self) -> None:
    if self.config.frame_length <= 0:
      raise ValueError(
          'frame_length must be positive, got: %d' % self.config.frame_length
      )
    if self.config.frame_step <= 0:
      raise ValueError(
          'frame_step must be positive, got: %d' % self.config.frame_step
      )
    # TODO(rryan): Fix issues with frame_length < frame_step.
    if self.config.frame_length < self.config.frame_step:
      raise ValueError('frame_length must be at least frame_step.')

  @property
  def supports_step(self) -> bool:
    return self.config.padding == types.PaddingMode.CAUSAL.value

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self.config.frame_step)

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    if self.config.padding == types.PaddingMode.SEMICAUSAL_FULL.value:
      past = 0
      future = (
          math.ceil(float(self.config.frame_length) / self.config.frame_step)
          - 1
      )
      return {0: (past, future)}
    return convolution.transpose_conv_receptive_field_per_step(
        self.config.frame_length,
        self.config.frame_step,
        1,
        self.config.padding,
    )

  @nn.nowrap
  def _validate_input_shape(self, input_shape: types.ShapeLike) -> None:
    if not input_shape or input_shape[0] != self.config.frame_length:
      raise ValueError(
          'OverlapAdd expects an input of shape (frame_length, ...), got:'
          f' {input_shape=}'
      )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    self._validate_input_shape(input_shape)
    return tuple(input_shape[1:])

  @property
  def _buffer_width(self) -> int:
    return max(0, self.config.frame_length - self.config.frame_step)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._validate_input_shape(input_spec.shape)
    if buffer_width := self._buffer_width:
      output_spec = self.get_output_spec(input_spec, constants=constants)
      return jnp.zeros(
          (batch_size, buffer_width) + output_spec.shape,
          dtype=output_spec.dtype,
      )
    else:
      return ()

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._validate_input_shape(x.channel_spec.shape)

    if self.config.frame_length > 1:
      x = x.mask_invalid()

    # Transpose [num_frames, frame_length] to end.
    if x.ndim > 3:
      values = jnp.moveaxis(x.values, (1, 2), (-2, -1))
    else:
      values = x.values
    values = signal.overlap_and_add(values, self.config.frame_step)
    # Move overlapped axis back if we moved it.
    if x.ndim > 3:
      values = jnp.moveaxis(values, -1, 1)

    mask = convolution.compute_conv_transpose_mask(
        x.mask,
        self.config.frame_length,
        self.config.frame_step,
        dilation_rate=1,
        padding=self.config.padding,
    )

    trim = max(self.config.frame_length - self.config.frame_step, 0)
    match self.config.padding:
      case types.PaddingMode.CAUSAL.value:
        # Trim the last frame_length - frame_step samples since we don't produce
        # these in step mode (we can only produce frame_step samples at a time
        # in step mode, so we can't produce the final frame_length - frame_step
        # samples).
        if trim:
          values = values[:, :-trim]

      case types.PaddingMode.SEMICAUSAL_FULL.value:
        # Remove the front padding that should be zero and invalid.
        if trim:
          values = values[:, trim:]
          mask = mask[:, trim:]

        size = min(values.shape[1], mask.shape[1])
        return types.Sequence(values[:, :size], mask[:, :size])

      case _:
        pass

    # Overlap add leaves padding with nonzero values.
    y = types.Sequence(values, mask)

    return y

  @types.check_step
  @nn.nowrap
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:

    if self.config.frame_length > 1:
      x = x.mask_invalid()

    # Transpose [num_frames, frame_length] to end.
    if x.ndim > 3:
      values = jnp.moveaxis(x.values, (1, 2), (-2, -1))
    else:
      values = x.values
    values = signal.overlap_and_add(values, self.config.frame_step)
    # Move overlapped axis back if we moved it.
    if x.ndim > 3:
      values = jnp.moveaxis(values, -1, 1)

    mask = convolution.compute_conv_transpose_mask(
        x.mask,
        self.config.frame_length,
        self.config.frame_step,
        dilation_rate=1,
        padding=self.config.padding,
    )

    if self._buffer_width:
      time = x.shape[1]

      # Pad the state to extend it to the length of the layer output.
      # output_time is at least frame_length and buffer_width is at most
      # frame_length - 1, so output_time - buffer_width is positive.
      paddings = [(0, 0)] * state.ndim
      paddings[1] = (0, values.shape[1] - self._buffer_width)
      state = jnp.pad(state, paddings)

      # Overlap-add outputs from previous timesteps into values.
      values = values + state

      # Stride samples are "ready" for output after one timestep, so the number
      # of output samples for the block is stride * time.
      output_samples = self.config.frame_step * time

      # We need to store (effective_kernel_size - stride) samples for future
      # steps, since their value depends on future inputs.
      values, state = jnp.split(values, [output_samples], axis=1)

    # Overlap add leaves padding with nonzero values.
    return types.Sequence(values, mask), state


def _validate_and_normalize_axis(
    axis: int,
    input_shape: types.Shape,
) -> int:
  """Normalizes user-provided axis and checks batch/time are not specified."""
  if axis < 0:
    axis += len(input_shape)
  if axis < 0 or axis > len(input_shape) - 1:
    raise ValueError(
        f'Axis out of range {axis=} for {axis=} with {input_shape=}.'
    )
  if axis in (0, 1):
    raise ValueError(
        'Computing FFTs over the batch or time dimension is '
        f'not allowed. Got: {axis}'
    )
  return axis


def _pad_or_truncate_for_fft(
    x: types.Sequence,
    normalized_axis: int,
    required_input_length: int,
    padding: str,
) -> types.Sequence:
  """Pads or truncates the provided sequence for an FFT/IFFT/RFFT/IRFFT."""
  assert normalized_axis > 1 and normalized_axis < x.ndim
  input_dim = x.shape[normalized_axis]
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
    paddings = [(0, 0)] * x.ndim
    paddings[normalized_axis] = (pad_left, pad_right)
    return x.apply_values_masked(jnp.pad, paddings)
  else:
    assert input_dim > required_input_length

    def slice_in_dim(v, start, length):
      return jax.lax.slice_in_dim(
          v, start_index=start, limit_index=start + length, axis=normalized_axis
      )

    if padding == 'center':
      trim_left = (input_dim - required_input_length) // 2
      return x.apply_values(slice_in_dim, trim_left, required_input_length)
    else:
      assert padding == 'right'
      return x.apply_values(slice_in_dim, 0, required_input_length)


def _validate(fft_length: int, padding: str):
  if fft_length is not None and fft_length <= 0:
    raise ValueError('fft_length must be positive, got: %d' % fft_length)
  if padding not in ('center', 'right'):
    raise ValueError('padding must be "center" or "right", got: %s' % padding)


class FFTBase(types.Stateless, metaclass=abc.ABCMeta):
  """A base class for shared FFT logic."""

  @property
  @abc.abstractmethod
  def _axis(self) -> int:
    pass

  @property
  @abc.abstractmethod
  def _fft_length(self) -> int | None:
    pass

  @property
  @abc.abstractmethod
  def _padding(self) -> str:
    pass

  @abc.abstractmethod
  def _get_fft_fn(self) -> Callable[..., types.Sequence]:
    pass

  @abc.abstractmethod
  def _get_output_length(self, input_size: int) -> int:
    pass

  @nn.nowrap
  def _get_fft_length(self, input_size: int) -> int:
    return self._fft_length or input_size

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[int, ...]:
    input_shape = list(input_shape)
    axis = _validate_and_normalize_axis(
        self._axis, (None, None) + tuple(input_shape)
    )
    axis -= 2
    input_shape[axis] = self._get_output_length(input_shape[axis])
    return tuple(input_shape)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    if x.ndim <= 2:
      raise ValueError('FFT requires an input of rank at least 3.')

    axis = _validate_and_normalize_axis(self._axis, x.shape)
    fft_fn = self._get_fft_fn()
    return fft_fn(x, axis=axis)


class FFT(types.PreservesType, FFTBase):
  """A layer that applies an FFT to the channels dimension."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    fft_length: int | None = None
    axis: int = -1
    padding: FFTPaddingString = _DEFAULT_FFT_PADDING
    name: str | None = None

    def make(self) -> 'FFT':
      return FFT(self, name=self.name)

  config: Config

  @property
  def _axis(self) -> int:
    return self.config.axis

  @property
  def _fft_length(self) -> int | None:
    return self.config.fft_length

  @property
  def _padding(self) -> str:
    return self.config.padding

  def _get_output_length(self, input_size: int) -> int:
    return self.config.fft_length or input_size

  def _get_fft_fn(self) -> Callable[..., types.Sequence]:
    def fft_fn(x, axis):
      required_length = self._get_output_length(x.shape[axis])
      x = _pad_or_truncate_for_fft(
          x, axis, required_length, self.config.padding
      )
      return x.apply_values(jnp.fft.fft, axis=axis)

    return fft_fn


class IFFT(types.PreservesType, FFTBase):
  """A layer that applies an IFFT to the channels dimension."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    fft_length: int | None = None
    frame_length: int | None = None
    axis: int = -1
    padding: FFTPaddingString = _DEFAULT_FFT_PADDING
    name: str | None = None

    def make(self) -> 'IFFT':
      return IFFT(self, name=self.name)

  config: Config

  @property
  def _axis(self) -> int:
    return self.config.axis

  @property
  def _fft_length(self) -> int | None:
    return self.config.fft_length

  @property
  def _padding(self) -> str:
    return self.config.padding

  def _get_output_length(self, input_size: int) -> int:
    return self.config.frame_length or input_size

  def _get_fft_fn(self) -> Callable[..., types.Sequence]:
    def ifft_fn(a, axis):
      a = a.apply_values(jnp.fft.ifft, axis=axis)
      required_length = self._get_output_length(a.shape[axis])
      return _pad_or_truncate_for_fft(
          a, axis, required_length, self.config.padding
      )

    return ifft_fn


class RFFT(FFTBase):
  """A layer that applies an RFFT to the channels dimension."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    fft_length: int | None = None
    axis: int = -1
    padding: FFTPaddingString = _DEFAULT_FFT_PADDING
    name: str | None = None

    def make(self) -> 'RFFT':
      return RFFT(self, name=self.name)

  config: Config

  @property
  def _axis(self) -> int:
    return self.config.axis

  @property
  def _fft_length(self) -> int | None:
    return self.config.fft_length

  @property
  def _padding(self) -> str:
    return self.config.padding

  def _get_fft_length(self, input_size: int) -> int:
    return self.config.fft_length or input_size

  def _get_output_length(self, input_size: int) -> int:
    return self._get_fft_length(input_size) // 2 + 1

  def _get_fft_fn(self) -> Callable[..., types.Sequence]:

    def rfft(a, axis=-1):
      # Cast input if the dtype is not supported by rfft.
      fft_length = self._get_fft_length(a.shape[axis])
      a = _pad_or_truncate_for_fft(a, axis, fft_length, self.config.padding)
      rfft_inner = lambda v: jnp.fft.rfft(v, n=fft_length, axis=axis)
      if a.dtype == jnp.bfloat16:
        return a.apply_values(lambda v: rfft_inner(v.astype(jnp.float32)))
      return a.apply_values(rfft_inner)

    return rfft

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    match input_dtype:
      case jnp.bfloat16:
        return jnp.complex64
      case jnp.float16:
        return jnp.complex64
      case jnp.float32:
        return jnp.complex64
      case jnp.float64:
        return jnp.complex128
      case _:
        raise ValueError(f'Unsupported input dtype: {input_dtype}')


class IRFFT(FFTBase):
  """A layer that applies an IRFFT to the channels dimension."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    fft_length: int | None = None
    frame_length: int | None = None
    axis: int = -1
    padding: FFTPaddingString = _DEFAULT_FFT_PADDING
    name: str | None = None

    def make(self) -> 'IRFFT':
      return IRFFT(self, name=self.name)

  config: Config

  @property
  def _axis(self) -> int:
    return self.config.axis

  @property
  def _fft_length(self) -> int | None:
    return self.config.fft_length

  @property
  def _padding(self) -> str:
    return self.config.padding

  def setup(self) -> None:
    _validate(self.config.fft_length, self.config.padding)

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    match input_dtype:
      case jnp.complex64:
        return jnp.float32
      case jnp.complex128:
        return jnp.float64
      case _:
        raise ValueError(f'Unsupported input dtype: {input_dtype}')

  @nn.nowrap
  def _get_fft_length(self, input_size: int) -> int:
    return self.config.fft_length or (input_size - 1) * 2

  @nn.nowrap
  def _get_output_length(self, input_size: int) -> int:
    return self.config.frame_length or self._get_fft_length(input_size)

  @nn.nowrap
  def _get_fft_fn(self) -> Callable[..., types.Sequence]:
    def irfft_fn(a, axis=-1):
      fft_length = self._get_fft_length(a.shape[axis])
      a = a.apply_values(jnp.fft.irfft, axis=axis, n=fft_length)
      required_length = self.config.frame_length or a.shape[axis]
      return _pad_or_truncate_for_fft(
          a, axis, required_length, self.config.padding
      )

    return irfft_fn


class STFT(types.SequenceLayer):
  """Computes the Short-time Fourier Transform of input signals.

  When used with 'right' FFT padding, equivalent to tf.signal.stft.

  For an input batch of multi-channel signals shaped [b, t, ...], this layer
  produces a [b, t // frame_step, fft_length // 2 + 1, ...] batch of
  multi-channel output STFTs, where the STFT for each channel in ... is computed
  independently.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for STFT layer."""

    # The frame length of the STFT.
    frame_length: int
    # The frame step of the STFT.
    frame_step: int
    # The FFT length of the STFT.
    fft_length: int
    # The window to use in the STFT. A callable that takes a window length and
    # returns a [window_length] array of samples. If None, no windowing is used.
    # The default is a periodic hann window.
    window_fn: Callable[..., jax.Array | np.ndarray] | None = signal.hann_window
    # Time padding to apply to the sequence. If 'causal_valid',
    # 'reverse_causal_valid', 'causal', 'reverse_causal', or 'semicausal' then
    # the STFT is streamable.
    time_padding: types.PaddingModeString = (
        types.PaddingMode.REVERSE_CAUSAL_VALID.value
    )
    # One of 'right' or 'center'. Whether to pad inputs to `fft_length` by
    # padding with zeros from the right or equally on both sides.
    fft_padding: FFTPaddingString = _DEFAULT_FFT_PADDING
    # If true, outputs a magnitude STFT (i.e. a spectrogram).
    output_magnitude: bool = False
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )

    def make(self) -> 'STFT':
      return STFT(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.framer = Frame.Config(
        frame_length=self.config.frame_length,
        frame_step=self.config.frame_step,
        padding=self.config.time_padding,
    ).make()
    self.fft = RFFT.Config(
        self.config.fft_length, axis=2, padding=self.config.fft_padding
    ).make()

  @property
  def supports_step(self) -> bool:
    return self.framer.supports_step and self.fft.supports_step

  @property
  def block_size(self) -> int:
    return self.framer.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self.framer.output_ratio

  @property
  def input_latency(self) -> int:
    return self.framer.input_latency

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    if self.config.window_fn:
      window = self.config.window_fn(self.config.frame_length)
    else:
      window = np.ones(self.config.frame_length)
    receptive_field = self.framer.receptive_field
    assert receptive_field is not None, 'Layer has no receptive field.'
    past, future = receptive_field
    receptive_field_list = np.array(range(past, future + 1)).astype(np.int32)

    if self.config.fft_length < self.config.frame_length:
      if self.config.fft_padding == 'center':
        trim_left = (self.config.frame_length - self.config.fft_length) // 2
        trim_right = (
            self.config.frame_length - self.config.fft_length - trim_left
        )
      elif self.config.fft_padding == 'right':
        trim_left = 0
        trim_right = (
            self.config.frame_length - self.config.fft_length - trim_left
        )
      else:
        raise ValueError(f'Unsupported FFT padding: {self.config.fft_padding}')
      trim_mask = np.array(
          [0] * trim_left
          + [1] * (self.config.frame_length - trim_left - trim_right)
          + [0] * trim_right
      )
      window = window * trim_mask

    receptive_field_list = receptive_field_list[window > 0]
    if receptive_field_list.size > 0:
      past = np.nanmin(receptive_field_list).astype(np.int32).item()
      future = np.nanmax(receptive_field_list).astype(np.int32).item()
    else:
      # Indicating no receptive field.
      past = np.inf
      future = -np.inf
    return {0: (past, future)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    return self.framer.get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    frame_shape = self.framer.get_output_shape(input_shape, constants=constants)
    return self.fft.get_output_shape(frame_shape, constants=constants)

  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    fft_output_type = self.fft.get_output_dtype(input_dtype)

    if self.config.output_magnitude:
      match fft_output_type:
        case jnp.complex64:
          return jnp.float32
        case jnp.complex128:
          return jnp.float64
        case _:
          raise ValueError(
              f'Unsupported FFT output dtype: {input_dtype=} {fft_output_type=}'
          )
    else:
      output_type = fft_output_type
    return output_type

  @nn.nowrap
  def _apply_window(self, x: types.Sequence) -> types.Sequence:
    if self.config.window_fn:
      window = self.config.window_fn(self.config.frame_length).reshape(
          (1, 1, self.config.frame_length) + (1,) * (x.ndim - 3)
      )
      window = jnp.asarray(window, x.dtype)
      return x.apply_values_masked(lambda v: v * window)
    return x

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    framed, state = self.framer.step(
        x, state, training=training, constants=constants
    )
    framed = self._apply_window(framed)
    dft, _ = self.fft.step(framed, (), training=training, constants=constants)
    if self.config.output_magnitude:
      dft = dft.apply_values_masked(jnp.abs)
    return dft, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    framed = self.framer.layer(x, training=training, constants=constants)
    framed = self._apply_window(framed)
    dft = self.fft.layer(framed, training=training, constants=constants)
    if self.config.output_magnitude:
      dft = dft.apply_values_masked(jnp.abs)
    return dft


class InverseSTFT(types.SequenceLayer):
  """Computes the inverse Short-time Fourier Transform of input signals.

  When used with 'right' FFT padding, equivalent to tf.signal.inverse_stft.

  For an input batch of multi-channel STFTs shaped [b, t, fft_length // 2 + 1,
  ...], this layer produces a [b, t * frame_step, ...] batch of multi-channel
  output signals, where the inverse STFT for each channel in ... is computed
  independently.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for the InverseSTFT layer."""

    # The frame length of the inverse STFT.
    frame_length: int
    # The frame step of the inverse STFT.
    frame_step: int
    # The FFT length of the inverse STFT.
    fft_length: int
    # The window to use for the inverse STFT. A callable that takes a window
    # length and returns a [window_length] array of samples. If None, no
    # windowing is used. The default is a periodic hann window, but it is best
    # to adjust the desired window function for an inverse STFT with
    # signal.inverse_stft_window_fn to achieve invertibility.
    window_fn: Callable[..., jax.Array | np.ndarray] | None = signal.hann_window
    # Time padding to apply to the sequence. If 'causal' then the inverse STFT
    # is streamable. Only 'causal' and 'valid' are supported.
    time_padding: types.PaddingModeString = types.PaddingMode.CAUSAL.value
    # One of 'right' or 'center'. Whether to pad inputs to `fft_length` by
    # padding with zeros from the right or equally on both sides.
    fft_padding: FFTPaddingString = _DEFAULT_FFT_PADDING
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )

    def make(self) -> 'InverseSTFT':
      return InverseSTFT(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.overlap_add = OverlapAdd.Config(
        frame_length=self.config.frame_length,
        frame_step=self.config.frame_step,
        padding=self.config.time_padding,
    ).make()
    self.irfft = IRFFT.Config(
        self.config.fft_length,
        self.config.frame_length,
        axis=2,
        padding=self.config.fft_padding,
    ).make()

  @property
  def supports_step(self) -> bool:
    return self.overlap_add.supports_step and self.irfft.supports_step

  @property
  def block_size(self) -> int:
    assert self.irfft.block_size == 1
    return self.overlap_add.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    assert self.irfft.output_ratio == 1
    return self.overlap_add.output_ratio

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    fft_length = self.config.fft_length
    frame_length = self.config.frame_length
    frame_step = self.config.frame_step
    time_padding = self.config.time_padding
    fft_padding = self.config.fft_padding

    explicit_padding = convolution.transpose_conv_explicit_padding(
        frame_length,
        frame_step,
        1,
        time_padding,
    )
    # Shift in the transpoed frame in OverlapAdd in the first step, i.e., the
    # number of frame items that are that are unused in the first step.
    shift_left = frame_length - explicit_padding[0] - 1
    if (pad_amount := frame_length - fft_length) > 0:
      if fft_padding == 'right':
        pad_left = 0
        pad_right = pad_amount
      elif fft_padding == 'center':
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
      else:
        raise ValueError(f'Unsupported FFT padding: {fft_padding}')
    else:
      pad_left = 0
      pad_right = 0

    if self.config.window_fn:
      window = self.config.window_fn(frame_length, dtype=jnp.int32)
      if pad_amount > 0:
        window = np.array([
            v if pad_left <= i < pad_left + fft_length else 0
            for i, v in enumerate(window)
        ])
      window_mask = window != 0
    else:
      window_mask = jnp.ones(frame_length, dtype=jnp.bool_)

    p = -math.ceil(float(explicit_padding[0] + 1) / frame_step) + 1
    f = math.ceil(float(shift_left) / frame_step) - 1
    rf_per_step = {}
    for i in range(frame_step):
      if (explicit_padding[0] + 1) % frame_step == 0 or (
          i < ((explicit_padding[0] + 1)) % frame_step
      ):
        p_i = p
      else:
        p_i = p + 1
      f_i = f if i < -shift_left % frame_step else f + 1

      if pad_amount > 0:
        padding_mask = np.array(
            [0] * pad_left + [1] * fft_length + [0] * pad_right, dtype=jnp.bool_
        )
      else:
        padding_mask = jnp.ones(frame_length, dtype=jnp.bool_)
      rf_frame_index = (i + shift_left) - np.arange(p_i, f_i + 1) * frame_step
      rf_frame_index_mask = (window_mask & padding_mask)[rf_frame_index]
      receptive_field_list = np.array(range(p_i, f_i + 1)).astype(np.int32)
      receptive_field_list = receptive_field_list[rf_frame_index_mask]
      if receptive_field_list.size > 0:
        p_i = np.nanmin(receptive_field_list).astype(np.int32).item()
        f_i = np.nanmax(receptive_field_list).astype(np.int32).item()
        rf_per_step[i] = (p_i, f_i) if p_i <= f_i else None
      else:
        rf_per_step[i] = None
    return rf_per_step

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    irfft_spec = self.irfft.get_output_spec(input_spec, constants=constants)
    # IRFFT output is padded or truncated to frame_length.
    irfft_shape = list(irfft_spec.shape)
    irfft_shape[0] = self.config.frame_length
    irfft_spec = types.ShapeDType(
        irfft_shape,
        irfft_spec.dtype,
    )
    return self.overlap_add.get_initial_state(
        batch_size, irfft_spec, training=training, constants=constants
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    irfft_shape = list(
        self.irfft.get_output_shape(input_shape, constants=constants)
    )
    # IRFFT output is padded or truncated to frame_length.
    irfft_shape[0] = self.config.frame_length
    return self.overlap_add.get_output_shape(irfft_shape, constants=constants)

  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return self.irfft.get_output_dtype(input_dtype)

  @nn.nowrap
  def _apply_window(self, irfft: types.Sequence) -> types.Sequence:
    # Pad or truncate the resulting fft_length vectors to frame_length.
    fft_length = irfft.shape[2]
    if fft_length > self.config.frame_length:
      # Trim to frame_length.
      irfft = irfft.apply_values_masked(
          lambda v: v[:, :, : self.config.frame_length]
      )
    elif fft_length < self.config.frame_length:
      pad_amount = self.config.frame_length - fft_length
      match self.config.fft_padding:
        case 'right':
          pad_left, pad_right = 0, pad_amount
        case 'center':
          pad_left, pad_right = pad_amount // 2, pad_amount - pad_amount // 2
        case _:
          raise ValueError(
              f'Unsupported FFT padding: {self.config.fft_padding=}'
          )

      paddings = [(0, 0)] * irfft.ndim
      paddings[2] = (pad_left, pad_right)
      irfft = irfft.apply_values_masked(jnp.pad, paddings)

    if self.config.window_fn:
      window = self.config.window_fn(
          self.config.frame_length, dtype=irfft.dtype
      ).reshape((1, 1, self.config.frame_length) + (1,) * (irfft.ndim - 3))
      window = jnp.asarray(window, irfft.dtype)
      irfft = irfft.apply_values_masked(lambda v: v * window)
    return irfft

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if x.ndim < 3:
      raise ValueError(
          f'Expected [b, t, num_frequency_bins, ...] input, but got {x.shape=}'
      )

    irfft, _ = self.irfft.step(x, (), training=training, constants=constants)
    irfft = self._apply_window(irfft)
    ola, state = self.overlap_add.step(
        irfft, state, training=training, constants=constants
    )
    return ola, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if x.ndim < 3:
      raise ValueError(
          f'Expected [b, t, num_frequency_bins, ...] input, but got {x.shape=}'
      )

    irfft = self.irfft.layer(x, training=training, constants=constants)
    irfft = self._apply_window(irfft)
    ola = self.overlap_add.layer(irfft, training=training, constants=constants)
    return ola


class LinearToMelSpectrogram(types.PreservesType, types.Stateless):
  """Converts linear-scale spectrogram to a mel-scale spectrogram.

  The spectrogram magnitudes should be uncompressed, *not* log compressed.
  Summation is performed across bands, which in log magnitdue would be
  multiplication.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for LinearToMelSpectrogram layer."""

    # The number of mel bins to compute.
    num_mel_bins: int
    # The sample rate of the spectrum.
    sample_rate: float
    # Lower bound on the frequencies to be included in the mel spectrum. This
    # corresponds to the lower edge of the lowest triangular band.
    lower_edge_hertz: float
    # The desired top edge of the highest frequency band.
    upper_edge_hertz: float
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'LinearToMelSpectrogram':
      return LinearToMelSpectrogram(self, name=self.name)

  config: Config

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.ShapeLike:
    if not input_shape:
      raise ValueError(
          f'{self} requires input with at least rank 1, got: {input_shape}'
      )
    return tuple(input_shape[:-1]) + (self.config.num_mel_bins,)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    num_spectrogram_bins = x.shape[-1]

    linear_to_mel = jnp.asarray(
        signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.config.num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.config.sample_rate,
            lower_edge_hertz=self.config.lower_edge_hertz,
            upper_edge_hertz=self.config.upper_edge_hertz,
            dtype=np.float64,
        ),
        x.dtype,
    )

    # No masking is required because a weight matrix applied to a masked
    # timestep is also masked.
    return x.apply_values_masked(
        lambda v: jnp.einsum('...a,ab->...b', v, linear_to_mel)
    )


class Delay(types.PreservesShape, types.PreservesType, types.SequenceLayer):
  """A layer that delays its input by `length` timesteps.

  In contrast to sl.Lookahead, which drops `length` timesteps from the start of
  the sequence, sl.Delay inserts `length` invalid timesteps at the start of the
  sequence (delaying the input sequence by `length`).
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Delay layer."""

    # The non-negative length of the delay to apply. A length of zero is a
    # no-op.
    length: int
    # If true, delays layer-wise outputs by the specified amount. If false, the
    # delay is only inserted in the step-wise processing. This is useful if the
    # Delay is intended to support matching latency of a layer that introduces
    # step-wise latency but no layer-wise latency (such as
    # LocalDotProductSelfAttention with max_future_horizon > 0, or a convolution
    # with reverse_causal padding).
    delay_layer_output: bool = True
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'Delay':
      return Delay(self, name=self.name)

  config: Config

  @property
  def input_latency(self) -> int:
    # The delay length is the amount of input latency we introduce.
    return self.config.length

  @property
  def output_latency(self) -> int:
    # The current definitions of input and output latency are very confusing.
    # Output latency is defined as the number of output timesteps before the
    # step-wise output of the layer matches its layer-wise output. Since delay
    # is added in both layer() and step(), the layer-wise and step-wise outputs
    # match immediately and therefore have 0 output latency.
    if self.config.delay_layer_output:
      return 0
    else:
      return self.config.length

  @property
  def receptive_field_per_step(self):
    if self.config.delay_layer_output:
      return {0: (-self.config.length, -self.config.length)}
    else:
      return {0: (0, 0)}

  def setup(self) -> None:
    if self.config.length < 0:
      raise ValueError(f'Expected nonnegative delay length. Got: {self.config}')

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Special case no delay.
    if not self.config.length:
      return ()
    return types.Sequence(
        jnp.zeros(
            (batch_size, self.config.length) + input_spec.shape,
            dtype=input_spec.dtype,
        ),
        jnp.zeros((batch_size, self.config.length), types.MASK_DTYPE),
    )

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if not self.config.length:
      return x, state

    state = state.concatenate(x)
    y_values, state_values = jnp.split(state.values, [x.shape[1]], axis=1)
    y_mask, state_mask = jnp.split(state.mask, [x.shape[1]], axis=1)
    y = types.Sequence(y_values, y_mask)
    state = types.Sequence(state_values, state_mask)
    return y, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if self.config.delay_layer_output:
      return x.pad_time(self.config.length, 0, valid=False)
    else:
      return x


class Lookahead(types.PreservesShape, types.PreservesType, types.SequenceLayer):
  """A layer that drops the first `length` timesteps from its input.

  In contrast to sl.Delay, which inserts `length` invalid timesteps at the start
  of the sequence (delaying the input sequence by `length`), sl.Lookahead drops
  `length` timesteps from the start of the sequence.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Lookahead layer."""

    # The non-negative length of the lookahead to apply. A length of zero is a
    # no-op.
    length: int
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'Lookahead':
      return Lookahead(self, name=self.name)

  config: Config

  @property
  def input_latency(self) -> int:
    return 0

  @property
  def output_latency(self) -> int:
    return self.config.length

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return {0: (self.config.length, self.config.length)}

  def setup(self) -> None:
    if self.config.length < 0:
      raise ValueError(
          f'Expected nonnegative lookahead length. Got: {self.config}'
      )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Special case no lookahead.
    if not self.config.length:
      return ()
    return jnp.full((batch_size,), jnp.array(self.config.length + 1, jnp.int32))

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if not self.config.length:
      return x, state
    assert x.shape[1] > 0

    increments = jnp.cumsum(x.mask, axis=1)
    countdown = jnp.maximum(0, state[:, jnp.newaxis] - increments)
    mask = jnp.logical_and(x.mask, countdown == 0)
    y = types.Sequence(x.values, mask)
    state = countdown[:, -1]
    return y, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    if not self.config.length:
      return x
    return x[:, self.config.length :]


class Window(types.PreservesShape, types.PreservesType, types.Stateless):
  """Applies a window function as in the STFT/InverseSTFT."""

  @dataclasses.dataclass(frozen=True, slots=True)
  class Config(types.SequenceLayerConfig):
    """Config of this layer."""

    # The axis onto which the window is applied.
    axis: int
    # The window function to use.
    window_fn: Callable[..., jax.Array | np.ndarray] | None = signal.hann_window
    # Optional name for this layer.
    name: str | None = None

    def make(self) -> 'Window':
      return Window(self, name=self.name)

  config: Config

  def _get_axis(self, x: types.Sequence):
    axis = self.config.axis
    if axis < 0:
      axis = x.ndim + axis

    if axis in (0, 1):
      raise ValueError(
          f'The axis cannot be 0 (batch) or 1 (time) (got {self.config.axis}).'
      )

    if axis >= x.ndim:
      raise ValueError(
          f'Axis {self.config.axis} larger than inputs ndim ({{x.ndim}})'
      )

    return axis

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:

    axis = self._get_axis(x)

    window = self.config.window_fn(x.shape[axis], dtype=x.dtype)
    shape_broadcast = [jnp.newaxis] * x.ndim
    shape_broadcast[axis] = slice(None, None)

    return x.apply_values(lambda v: v * window[tuple(shape_broadcast)])
