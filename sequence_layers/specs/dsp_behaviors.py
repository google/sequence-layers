# Copyright 2026 Google LLC
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
"""Behavior tests for digital signal processing (DSP) layers.

Backend-specific test files should inherit from these tests.
"""

# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation

from absl.testing import parameterized
import numpy as np

from sequence_layers.specs import test_utils


def _pad_or_truncate_for_fft(
    values: np.ndarray, padding: str, axis: int, required_input_length: int
) -> np.ndarray:
  """Pads or truncates values to required_input_length along axis."""
  axis_size = values.shape[axis]
  pad_amount = max(0, required_input_length - axis_size)
  if padding == 'center':
    left = pad_amount // 2
    right = pad_amount - left
  else:
    assert padding == 'right'
    left, right = 0, pad_amount

  paddings = [(0, 0)] * values.ndim
  paddings[axis] = (left, right)
  values = np.pad(values, paddings)
  axis_size = values.shape[axis]

  trim_amount = max(0, axis_size - required_input_length)
  if padding == 'center':
    left = trim_amount // 2
  else:
    left = 0

  slices = [slice(None)] * values.ndim
  slices[axis] = slice(left, left + required_input_length)
  return values[tuple(slices)]


class FFTTest(test_utils.SequenceLayerTest):
  """Test behavior of FFT layer."""

  @parameterized.product(
      shape_axis=[((2, 3, 32), -1), ((2, 3, 5, 32), -1), ((2, 3, 5, 32), -2)],
      fft_length=[31, 32, 33],
      padding=['center', 'right'],
  )
  def test_fft(self, shape_axis, fft_length, padding):
    shape, axis = shape_axis
    x = self.random_sequence(*shape, low_length=1)
    config = self.sl.FFT.Config(
        fft_length=fft_length,
        axis=axis,
        padding=padding,
        name='fft',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'fft')

    channel_shape = list(shape[2:])
    channel_shape[axis] = fft_length
    self.assertEqual(l.get_output_shape(shape[2:]), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)

    # Check that the result is the same as manually padding/truncating followed by FFT.
    def apply_fft(values):
      values = _pad_or_truncate_for_fft(values, padding, axis, fft_length)
      return np.fft.fft(values, n=fft_length, axis=axis)

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-4, rtol=1e-4)
    self.assertEqual(y.shape[axis], fft_length)


class IFFTTest(test_utils.SequenceLayerTest):
  """Test behavior of IFFT layer."""

  @parameterized.product(
      shape_axis=[((2, 3, 32), -1), ((2, 3, 5, 32), -1), ((2, 3, 5, 32), -2)],
      frame_length=[31, 32, 33, None],
      padding=['center', 'right'],
  )
  def test_ifft(self, shape_axis, frame_length, padding):
    shape, axis = shape_axis
    fft_length = shape[axis]
    x = self.random_sequence(*shape)
    config = self.sl.IFFT.Config(
        fft_length=fft_length,
        frame_length=frame_length,
        axis=axis,
        padding=padding,
        name='ifft',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)

    if frame_length is None:
      frame_length = fft_length

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'ifft')

    channel_shape = list(shape[2:])
    channel_shape[axis] = frame_length
    self.assertEqual(l.get_output_shape(shape[2:]), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)

    def apply_ifft(values):
      values = np.fft.ifft(values, n=fft_length, axis=axis)
      return _pad_or_truncate_for_fft(values, padding, axis, frame_length)

    y_expected = x.apply_values(apply_ifft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-4, rtol=1e-4)
    self.assertEqual(y.shape[axis], frame_length)


class RFFTTest(test_utils.SequenceLayerTest):
  """Test behavior of RFFT layer."""

  @parameterized.product(
      shape_axis=[((2, 3, 32), -1), ((2, 3, 5, 32), -1), ((2, 3, 5, 32), -2)],
      fft_length=[31, 32, 33],
      padding=['center', 'right'],
  )
  def test_rfft(self, shape_axis, fft_length, padding):
    shape, axis = shape_axis
    x = self.random_sequence(*shape)
    config = self.sl.RFFT.Config(
        fft_length=fft_length,
        axis=axis,
        padding=padding,
        name='rfft',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'rfft')

    channel_shape = list(shape[2:])
    channel_shape[axis] = fft_length // 2 + 1
    self.assertEqual(l.get_output_shape(shape[2:]), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)

    def apply_rfft(values):
      values = _pad_or_truncate_for_fft(values, padding, axis, fft_length)
      return np.fft.rfft(values, n=fft_length, axis=axis)

    y_expected = x.apply_values(apply_rfft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-4, rtol=1e-4)
    self.assertEqual(y.shape[axis], fft_length // 2 + 1)


class IRFFTTest(test_utils.SequenceLayerTest):
  """Test behavior of IRFFT layer."""

  @parameterized.product(
      shape_axis=[((2, 3, 17), -1), ((2, 3, 5, 17), -1)],
      frame_length=[31, 32, 33, None],
      padding=['center', 'right'],
  )
  def test_irfft(self, shape_axis, frame_length, padding):
    shape, axis = shape_axis
    fft_length = (shape[axis] - 1) * 2
    x = self.random_sequence(*shape)
    config = self.sl.IRFFT.Config(
        fft_length=fft_length,
        frame_length=frame_length,
        axis=axis,
        padding=padding,
        name='irfft',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)

    if frame_length is None:
      frame_length = fft_length

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'irfft')

    channel_shape = list(shape[2:])
    channel_shape[axis] = frame_length
    self.assertEqual(l.get_output_shape(shape[2:]), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)

    def apply_irfft(values):
      values = np.fft.irfft(values, n=fft_length, axis=axis)
      return _pad_or_truncate_for_fft(values, padding, axis, frame_length)

    y_expected = x.apply_values(apply_irfft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-4, rtol=1e-4)
    self.assertEqual(y.shape[axis], frame_length)


class FrameTest(test_utils.SequenceLayerTest):
  """Test behavior of Frame layer."""

  @parameterized.product(
      frame_length=[1, 2, 3, 4],
      frame_step=[1, 2, 3, 4],
      padding=[
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
  )
  def test_frame(self, frame_length, frame_step, padding):
    batch_size, time, channels = 2, 20 * frame_step, 3
    x = self.random_sequence(batch_size, time, channels)
    config = self.sl.Frame.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=padding,
        name='frame',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.block_size, frame_step)
    self.assertEqual(1 / l.output_ratio, frame_step)
    self.assertEqual(l.name, 'frame')
    self.assertTrue(l.supports_step)
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)


class OverlapAddTest(test_utils.SequenceLayerTest):
  """Test behavior of OverlapAdd layer."""

  @parameterized.product(
      frame_length=[1, 2, 3, 4],
      frame_step=[1, 2, 3, 4],
      padding=['causal', 'valid', 'semicausal_full'],
  )
  def test_overlap_add(self, frame_length, frame_step, padding):
    if frame_length < frame_step:
      return  # Pre-condition requirement
    batch_size, time, channels = 2, 20, 3
    x = self.random_sequence(batch_size, time, frame_length, channels)
    config = self.sl.OverlapAdd.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=padding,
        name='overlap_add',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.output_ratio, frame_step)
    self.assertEqual(l.name, 'overlap_add')
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)


class STFTTest(test_utils.SequenceLayerTest):
  """Test behavior of STFT layer."""

  @parameterized.product(
      frame_length=[4, 8],
      frame_step=[2, 4],
      fft_length=[8, 16],
      time_padding=[
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
      fft_padding=['center', 'right'],
  )
  def test_stft(
      self, frame_length, frame_step, fft_length, time_padding, fft_padding
  ):
    if fft_length < frame_length:
      return
    batch_size, time, channels = 2, 20 * frame_step, 3
    x = self.random_sequence(batch_size, time, channels)
    config = self.sl.STFT.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding=time_padding,
        fft_padding=fft_padding,
        name='stft',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.block_size, frame_step)
    self.assertEqual(1 / l.output_ratio, frame_step)
    self.assertEqual(l.name, 'stft')
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)


class InverseSTFTTest(test_utils.SequenceLayerTest):
  """Test behavior of InverseSTFT layer."""

  @parameterized.product(
      frame_length=[4, 8],
      frame_step=[2, 4],
      fft_length=[8, 16],
      time_padding=['causal', 'valid'],
      fft_padding=['center', 'right'],
  )
  def test_inverse_stft(
      self, frame_length, frame_step, fft_length, time_padding, fft_padding
  ):
    if fft_length < frame_length:
      return
    if frame_length < frame_step:
      return
    batch_size, time, channels = 2, 20, 3
    # Input to InverseSTFT must be complex spectrogram bins: fft_length // 2 + 1
    x = self.random_sequence(batch_size, time, fft_length // 2 + 1, channels)
    config = self.sl.InverseSTFT.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding=time_padding,
        fft_padding=fft_padding,
        name='istft',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.output_ratio, frame_step)
    self.assertEqual(l.name, 'istft')
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)


class LinearToMelSpectrogramTest(test_utils.SequenceLayerTest):
  """Test behavior of LinearToMelSpectrogram layer."""

  def test_mel_spectrogram(self):
    batch_size, time, channels = 2, 8, 257  # 257 linear spectrogram bins
    x = self.random_sequence(batch_size, time, channels)
    config = self.sl.LinearToMelSpectrogram.Config(
        num_mel_bins=40,
        sample_rate=16000.0,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
        name='mel',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.name, 'mel')
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)


class DelayTest(test_utils.SequenceLayerTest):
  """Test behavior of Delay layer."""

  @parameterized.product(
      length=[0, 1, 3],
      delay_layer_output=[True, False],
  )
  def test_delay(self, length, delay_layer_output):
    batch_size, time, channels = 2, 15, 3
    x = self.random_sequence(batch_size, time, channels)
    config = self.sl.Delay.Config(
        length=length,
        delay_layer_output=delay_layer_output,
        name='delay',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.name, 'delay')
    self.assertEqual(l.input_latency, length)
    self.assertEqual(l.output_latency, 0 if delay_layer_output else length)
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)


class LookaheadTest(test_utils.SequenceLayerTest):
  """Test behavior of Lookahead layer."""

  @parameterized.product(
      length=[0, 1, 3],
      preserve_length_in_layer=[True, False],
  )
  def test_lookahead(self, length, preserve_length_in_layer):
    batch_size, time, channels = 2, 15, 3
    x = self.random_sequence(batch_size, time, channels)
    config = self.sl.Lookahead.Config(
        length=length,
        preserve_length_in_layer=preserve_length_in_layer,
        name='lookahead',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.name, 'lookahead')
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.output_latency, length)
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)


class WindowTest(test_utils.SequenceLayerTest):
  """Test behavior of Window layer."""

  def test_window(self):
    batch_size, time, channels = 2, 8, 16
    x = self.random_sequence(batch_size, time, channels)
    config = self.sl.Window.Config(
        axis=-1,
        name='window',
    )
    l = self.make_layer(config)
    l = self.init_layer(l, x)
    self.assertEqual(l.name, 'window')
    self.verify_contract(l, x, training=False, atol=1e-4, rtol=1e-4)
