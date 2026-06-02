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
"""DSP tests."""

import itertools
import math

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from sequence_layers.jax import combinators
from sequence_layers.jax import dsp
from sequence_layers.jax import signal
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.specs import dsp_behaviors as spec


class FFTTest(test_utils.SequenceLayerTest, spec.FFTTest):
  """Verify FFT contract."""


class IFFTTest(test_utils.SequenceLayerTest, spec.IFFTTest):
  """Verify IFFT contract."""


class RFFTTest(test_utils.SequenceLayerTest, spec.RFFTTest):
  """Verify RFFT contract."""


class IRFFTTest(test_utils.SequenceLayerTest, spec.IRFFTTest):
  """Verify IRFFT contract."""


class FFTInverseTTest(test_utils.SequenceLayerTest, parameterized.TestCase):
  """Tests that the FFT/IFFT and RFFT/IRFFT are inverses of each other."""

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 31), -1), ((2, 3, 5, 32), -1), ((2, 3, 5, 33), -2)),
          (31, 32, 33),
          (
              (dsp.RFFT.Config, dsp.IRFFT.Config, jnp.float32),
              (dsp.FFT.Config, dsp.IFFT.Config, jnp.complex64),
          ),
          ('center', 'right'),
      )
  )
  def test_fft_inverse(self, shape_axis, fft_length, fft_config_dtype, padding):
    shape, axis = shape_axis

    forward_config_fn, backward_config_fn, dtype = fft_config_dtype

    x = test_utils.random_sequence(*shape, dtype=dtype)

    # The input length is necessary for the backward transfrom.
    frame_length = x.shape[axis]

    forward = (
        forward_config_fn(
            fft_length,
            axis=axis,
            padding=padding,
            name='forward',
        )
        .make()
        .bind({})
    )
    backward = (
        backward_config_fn(
            fft_length,
            frame_length=frame_length,
            axis=axis,
            padding=padding,
            name='backward',
        )
        .make()
        .bind({})
    )

    # Shortcuts.
    def forward_fn(val):
      return forward(val, training=False)

    def backward_fn(val):
      return backward(val, training=False)

    y_A = forward_fn(x)  # pylint: disable=invalid-name
    y_BA = backward_fn(y_A)  # pylint: disable=invalid-name
    y_ABA = forward_fn(y_BA)  # pylint: disable=invalid-name
    y_BABA = backward_fn(y_ABA)  # pylint: disable=invalid-name

    # 1) A B A = A applied to x.
    self.assertSequencesClose(y_A, y_ABA, atol=1e-5, rtol=1e-3)
    # 2) B A B = B applied to A x.
    self.assertSequencesClose(y_BA, y_BABA, atol=1e-5, rtol=1e-3)


class FrameTest(test_utils.SequenceLayerTest, spec.FrameTest):
  """Verify Frame contract and JAX specific behaviors."""

  @parameterized.product(
      frame_length_frame_step=((1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)),
      channel_shape=((), (4,), (5, 9)),
      padding=(
          'causal_valid',
          'semicausal',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'same',
          'valid',
          'semicausal_full',
          'explicit_semicausal',
      ),
  )
  def test_frame_exhaustive(
      self, frame_length_frame_step, channel_shape, padding
  ):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    frame_length, frame_step = frame_length_frame_step
    if padding == 'explicit_semicausal':
      total_pad = frame_length - 1
      overlap = max(0, frame_length - frame_step)
      explicit_padding = (overlap, total_pad - overlap)
    else:
      explicit_padding = padding
    x = test_utils.random_sequence(batch_size, 1, *channel_shape)
    l = dsp.Frame.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=explicit_padding,
        name='frame',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(
        l.supports_step,
        padding
        in (
            'causal_valid',
            'semicausal',
            'reverse_causal_valid',
            'causal',
            'reverse_causal',
            'explicit_semicausal',
        ),
    )
    self.assertEqual(l.block_size, frame_step)
    self.assertEqual(1 / l.output_ratio, frame_step)
    match padding:
      case 'causal_valid' | 'causal' | 'semicausal':
        expected_input_latency = 0
      case 'reverse_causal_valid' | 'reverse_causal':
        expected_input_latency = frame_length - 1
      case 'semicausal_full':
        expected_input_latency = frame_step - 1
      case 'explicit_semicausal':
        # If frame_length >= frame_step, the below expression simplifies to
        # frame_step - 1. If frame_length < frame_step, the expression
        # simplifies to frame_length - 1. In both cases, the output latency will
        # be zero both expressions are less than frame_step.
        expected_input_latency = (frame_length - 1) - max(
            0, frame_length - frame_step
        )
      case _:
        # Unsupported defaults to zero.
        expected_input_latency = 0
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // frame_step)
    self.assertEqual(l.name, 'frame')
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        (frame_length,) + channel_shape,
    )
    self.assertEmpty(l.variables)

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = test_utils.random_sequence(
          batch_size, time, *channel_shape, low_length=time // 2
      )
      self.verify_contract(l, x, training=False)


class STFTTest(test_utils.SequenceLayerTest, spec.STFTTest):
  """Verify STFT contract."""


class InverseSTFTTest(test_utils.SequenceLayerTest, spec.InverseSTFTTest):
  """Verify InverseSTFT contract."""


class STFTPerfectReconstructionTest(
    test_utils.SequenceLayerTest, parameterized.TestCase
):
  """With padding SEMICAUSAL_FULL, the STFT/InverseSTFT should give perfect reconstruction."""

  @parameterized.parameters(
      itertools.product(
          (
              (32, 16, 32),
              (32, 8, 32),
              (32, 8, 64),
              (50, 15, 64),
          ),
          (signal.hann_window, signal.hamming_window),
          ('center', 'right'),
      )
  )
  def test_stft_perfect_reconstruction_padding_semicausal_full(
      self,
      length_frame_step_fft,
      window_fn,
      fft_padding,
  ):
    frame_length, frame_step, fft_length = length_frame_step_fft
    batch_size = 2
    overlap = math.ceil(frame_length / frame_step)
    time = 2 * overlap * frame_length + 3

    # Perfect reconstruction is possible with SEMICAUSAL_FULL at the cost of
    # steppability.
    time_padding = types.PaddingMode.SEMICAUSAL_FULL.value

    x = test_utils.random_sequence(batch_size, time, dtype=jnp.float32)
    forward = (
        dsp.STFT.Config(
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=window_fn,
            time_padding=time_padding,
            fft_padding=fft_padding,
            name='stft',
        )
        .make()
        .bind({})
    )
    backward = (
        dsp.InverseSTFT.Config(
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=signal.inverse_stft_window_fn(frame_step, window_fn),
            time_padding=time_padding,
            fft_padding=fft_padding,
            name='inverse_stft',
        )
        .make()
        .bind({})
    )

    y = forward(x, training=False)
    x_hat = backward(y, training=False)

    size = x.shape[1]
    self.assertLess(size, x_hat.shape[1])

    # Intersection should be the same.
    mask_and = jnp.logical_and(x.mask, x_hat.mask[:, :size])
    np.testing.assert_allclose(
        x.values * mask_and,
        x_hat.values[:, :size] * mask_and,
        atol=1e-5,
        rtol=1e-5,
    )

    # Difference should be zero in the output.
    mask_xor = jnp.logical_xor(
        jnp.pad(x.mask, ((0, 0), (0, x_hat.shape[1] - size))), x_hat.mask
    )

    x_hat = x_hat.mask_invalid()
    self.assertTrue(jnp.all(abs(x_hat.values[mask_xor]) < 1e-6))


class LinearToMelSpectrogramTest(
    test_utils.SequenceLayerTest, spec.LinearToMelSpectrogramTest
):
  """Verify LinearToMelSpectrogram contract."""


class OverlapAddTest(test_utils.SequenceLayerTest, spec.OverlapAddTest):
  """Verify OverlapAdd contract and perfect reconstruction."""

  @parameterized.parameters((1, 1), (2, 1), (2, 2), (3, 2))
  def test_frame_overlap_add_perfect(self, frame_length, frame_step):
    b, t = 2, 35
    x = test_utils.random_sequence(b, t)
    forward = (
        dsp.Frame.Config(
            frame_length=frame_length,
            frame_step=frame_step,
            padding='semicausal_full',
            name='forward',
        )
        .make()
        .bind({})
    )
    backward = (
        dsp.OverlapAdd.Config(
            frame_length=frame_length,
            frame_step=frame_step,
            padding='semicausal_full',
            name='backward',
        )
        .make()
        .bind({})
    )

    y = forward.layer(x, training=False)
    z = backward.layer(y, training=False)

    self.assertLessEqual(x.shape[1], z.shape[1])
    self.assertTrue(np.all(np.array(z.lengths()) >= np.array(x.lengths())))
    np.testing.assert_array_equal(
        z.mask[:, x.shape[1] :],
        jnp.zeros((z.shape[0], z.shape[1] - x.shape[1]), dtype=jnp.bool_),
    )

    z_values = z.values[:, : x.shape[1]]
    z_mask = z.mask[:, : x.shape[1]]
    difference_mask = jnp.logical_xor(x.mask, z_mask)
    self.assertTrue(jnp.all(z_values[difference_mask] == 0))


class DelayTest(test_utils.SequenceLayerTest, spec.DelayTest):
  """Verify Delay contract and nonnegative checks."""

  def test_delay_nonnegative(self):
    x = test_utils.random_sequence(2, 11, 3, 5)
    l = dsp.Delay.Config(length=-1, name='delay').make().bind({})
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class LookaheadTest(test_utils.SequenceLayerTest, spec.LookaheadTest):
  """Verify Lookahead contract and nonnegative checks."""

  def test_lookahead_nonnegative(self):
    x = test_utils.random_sequence(2, 11, 3, 5)
    l = dsp.Lookahead.Config(length=-1, name='lookahead').make().bind({})
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class WindowTest(test_utils.SequenceLayerTest, spec.WindowTest):
  """Verify Window contract and invalid axis handling."""

  @parameterized.parameters(
      (20, 2),
      (15, 3),
      (10, 4),
  )
  def test_window_perfect_reconstruction(
      self, frame_step, frame_length_multiplier
  ):
    frame_length = frame_step * frame_length_multiplier
    batch = 2
    time = 11 * frame_step + frame_length
    seq_in = test_utils.random_sequence(
        batch, time, low_length=2 * frame_length, dtype=jnp.float32
    )

    module = (
        combinators.Serial.Config(
            [
                dsp.Frame.Config(
                    frame_length=frame_length,
                    frame_step=frame_step,
                    padding='semicausal',
                ),
                dsp.Window.Config(
                    axis=2,
                    window_fn=signal.hamming_window,
                ),
                dsp.Window.Config(
                    axis=2,
                    window_fn=signal.inverse_stft_window_fn(
                        frame_step, signal.hamming_window
                    ),
                ),
                dsp.OverlapAdd.Config(
                    frame_length=frame_length,
                    frame_step=frame_step,
                    padding='causal',
                ),
                dsp.Lookahead.Config(frame_length - frame_step),
            ],
            name='test',
        )
        .make()
        .bind({})
    )

    seq_out = module.layer(seq_in, training=False)

    expected = types.Sequence.from_lengths(
        seq_in.values[:, : -frame_length + frame_step], seq_out.lengths()
    )

    self.assertSequencesClose(expected, seq_out)

  @parameterized.parameters((0,), (1,), (-2,), (-3,), (3,))
  def test_window_invalid_axis(self, axis):
    seq_in = test_utils.random_sequence(2, 5, 1)
    module = (
        dsp.Window.Config(
            axis=axis, window_fn=signal.hamming_window, name='test'
        )
        .make()
        .bind({})
    )

    with self.assertRaises(ValueError):
      module.layer(seq_in, training=False, constants=None)


if __name__ == '__main__':
  test_utils.main()
