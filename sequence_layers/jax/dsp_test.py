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
from sequence_layers.jax import utils
import tensorflow as tf


def _pad_or_truncate_for_fft(values, padding, axis, required_input_length):
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
  return jax.lax.slice_in_dim(values, left, required_input_length, axis=axis)


class FFTTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 32), -1), ((2, 3, 5, 32), -1), ((2, 3, 5, 32), -2)),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_fft(self, shape_axis, fft_length, padding):
    shape, axis = shape_axis
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape, dtype=jnp.complex64)
    l = dsp.FFT.Config(
        fft_length, axis=axis, padding=padding, name='fft'
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'fft')
    channel_shape = list(shape[2:])
    channel_shape[axis] = fft_length
    self.assertEqual(l.get_output_shape_for_sequence(x), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    # Check that the result is the same as manually padding/truncating followed
    # by the FFT.
    def apply_fft(values):
      values = _pad_or_truncate_for_fft(values, padding, axis, fft_length)
      return np.fft.fft(values, n=fft_length, axis=axis)

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-5, rtol=1e-5)
    self.assertEqual(y.shape[axis], fft_length)


class IFFTTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 32), -1), ((2, 3, 5, 32), -1), ((2, 3, 5, 32), -2)),
          (31, 32, 33, None),
          ('center', 'right'),
      )
  )
  def test_ifft(self, shape_axis, frame_length, padding):
    shape, axis = shape_axis
    key = jax.random.PRNGKey(1234)

    # The length of the input sequence.
    fft_length = shape[axis]

    x = test_utils.random_sequence(*shape, dtype=jnp.complex64)
    l = dsp.IFFT.Config(
        fft_length,
        frame_length=frame_length,
        axis=axis,
        padding=padding,
        name='ifft',
    ).make()

    # If frame_length is not provided, it should be infered from the input
    # length by IFFT to be the same as fft_length.
    if frame_length is None:
      frame_length = fft_length

    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'ifft')
    channel_shape = list(shape[2:])
    channel_shape[axis] = frame_length
    self.assertEqual(l.get_output_shape_for_sequence(x), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    # Check that the result is the same as manually padding/truncating followed
    # by the IFFT.
    def apply_fft(values):
      values = np.fft.ifft(values, n=fft_length, axis=axis)
      return _pad_or_truncate_for_fft(values, padding, axis, frame_length)

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-5, rtol=1e-5)
    self.assertEqual(y.shape[axis], frame_length)


class RFFTTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  def run_rfft_test(self, shape_axis, fft_length, padding, dtype):
    shape, axis = shape_axis
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape, dtype=dtype)
    l = dsp.RFFT.Config(
        fft_length, axis=axis, padding=padding, name='rfft'
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'rfft')

    channel_shape = list(shape[2:])
    channel_shape[axis] = fft_length // 2 + 1
    self.assertEqual(l.get_output_shape_for_sequence(x), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    # Check that the result is the same as manually padding/truncating followed
    # by the RFFT.
    def apply_fft(values):
      values = _pad_or_truncate_for_fft(values, padding, axis, fft_length)
      return np.fft.rfft(values, n=fft_length, axis=axis)

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-5, rtol=1e-5)
    self.assertEqual(y.shape[axis], fft_length // 2 + 1)

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 32), -1), ((2, 3, 5, 32), -1), ((2, 3, 5, 32), -2)),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_rfft(self, shape_axis, fft_length, padding):
    self.run_rfft_test(
        shape_axis=shape_axis,
        fft_length=fft_length,
        padding=padding,
        dtype=jnp.float32,
    )

  def test_rfft_bfloat16(self):
    self.run_rfft_test(
        shape_axis=((2, 3, 32), -1),
        fft_length=31,
        padding='center',
        dtype=jnp.bfloat16,
    )


class IRFFTTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (((2, 3, 17), -1), ((2, 3, 5, 17), -1), ((2, 3, 5, 17), -2)),
          (31, 32, 33, None),
          (32, None),
          ('center', 'right'),
      )
  )
  def test_irfft(self, shape_axis, frame_length, fft_length, padding):
    shape, axis = shape_axis
    key = jax.random.PRNGKey(1234)

    x = test_utils.random_sequence(*shape, dtype=jnp.complex64)
    l = dsp.IRFFT.Config(
        fft_length,
        frame_length=frame_length,
        axis=axis,
        padding=padding,
        name='irfft',
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    # If frame_length or fft_length are not provided, they are infered from the
    # input shape.
    if fft_length is None:
      fft_length = 2 * (shape[axis] - 1)

    if frame_length is None:
      frame_length = fft_length

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'irfft')
    channel_shape = list(shape[2:])
    channel_shape[axis] = frame_length
    self.assertEqual(l.get_output_shape_for_sequence(x), tuple(channel_shape))
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    # Check that the result is the same as manually padding/truncating followed
    # by the IRFFT.
    def apply_fft(values):
      values = np.fft.irfft(values, n=fft_length, axis=axis)
      return _pad_or_truncate_for_fft(values, padding, axis, frame_length)

    y_expected = x.apply_values(apply_fft).mask_invalid()
    self.assertSequencesClose(y, y_expected, atol=1e-5, rtol=1e-5)
    self.assertEqual(y.shape[axis], frame_length)


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
    forward_fn = lambda x: forward(x, training=False)
    backward_fn = lambda x: backward(x, training=False)

    # Depending on the padding parameters, there may be no inverse.
    # In that case the backward transform should be the pseudo-inverse.
    # For a general test, we test the pseudo-inverse properties. Let A and B
    # be the forward and backward transforms. They should satisfy:
    # 1) A B A = A
    # 2) B A B = B

    y_A = forward_fn(x)  # pylint: disable=invalid-name
    y_BA = backward_fn(y_A)  # pylint: disable=invalid-name
    y_ABA = forward_fn(y_BA)  # pylint: disable=invalid-name
    y_BABA = backward_fn(y_ABA)  # pylint: disable=invalid-name

    # 1) A B A = A applied to x.
    self.assertSequencesClose(y_A, y_ABA, atol=1e-5, rtol=1e-3)
    # 2) B A B = B applied to A x.
    self.assertSequencesClose(y_BA, y_BABA, atol=1e-5, rtol=1e-3)


class FrameTest(test_utils.SequenceLayerTest, parameterized.TestCase):

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
          'explicit_semicausal',
          'semicausal_full',
      ),
  )
  def test_frame(self, frame_length_frame_step, channel_shape, padding):
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
      case 'explicit_semicausal':
        # If frame_length >= frame_step, the below expression simplifies to
        # frame_step - 1. If frame_length < frame_step, the expression
        # simplifies to frame_length - 1. In both cases, the output latency will
        # be zero both expressions are less than frame_step.
        expected_input_latency = (frame_length - 1) - max(
            0, frame_length - frame_step
        )
      case 'semicausal_full':
        expected_input_latency = frame_step - 1
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


class STFTTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (True, False),
          (1, 2, 3, 4),
          (1, 2),
          (2, 3),
          (
              'causal_valid',
              'valid',
              'same',
              'reverse_causal_valid',
              'causal',
              'reverse_causal',
          ),
          ('center', 'right'),
      )
  )
  def test_stft(
      self,
      output_magnitude,
      frame_length,
      frame_step,
      fft_length,
      time_padding,
      fft_padding,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 20
    x = test_utils.random_sequence(batch_size, time)
    l = dsp.STFT.Config(
        output_magnitude=output_magnitude,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding=time_padding,
        fft_padding=fft_padding,
        name='stft',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, frame_step)
    self.assertEqual(
        l.supports_step,
        time_padding
        in ('causal_valid', 'reverse_causal_valid', 'causal', 'reverse_causal'),
    )
    self.assertEqual(1 / l.output_ratio, frame_step)
    match time_padding:
      case 'causal_valid' | 'causal':
        expected_input_latency = 0
      case 'reverse_causal_valid' | 'reverse_causal':
        expected_input_latency = frame_length - 1
      case 'semicausal':
        # If frame_length > frame_step, input_latency is frame_step - 1 so the
        # output latency is always zero.
        expected_input_latency = (frame_length - 1) - max(
            0, frame_length - frame_step
        )
      case _:
        # Unsupported defaults to zero.
        expected_input_latency = 0
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // frame_step)
    self.assertEqual(l.name, 'stft')
    self.assertEqual(l.get_output_shape_for_sequence(x), (fft_length // 2 + 1,))
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    # Check compatibility with tf.signal.stft (which only supports right
    # padding).
    if fft_padding == 'right':
      left, right = utils.convolution_explicit_padding(
          time_padding, frame_length, frame_step, dilation_rate=1
      )
      # Mask is unused so valid does not matter.
      x = x.pad_time(left, right, valid=False)
      y_tfs = tf.signal.stft(
          x.values,
          frame_length=frame_length,
          frame_step=frame_step,
          fft_length=fft_length,
          pad_end=False,
      )
      if output_magnitude:
        y_tfs = tf.abs(y_tfs)
      y_tfs = types.Sequence(y_tfs.numpy(), y.mask).mask_invalid().values
      self.assertAllClose(y.values, y_tfs)

  @parameterized.product(channel_shape=((1,), (2,), (2, 3)))
  def test_multichannel(self, channel_shape):
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 20
    x = test_utils.random_sequence(
        batch_size, time, low_length=time // 2, *channel_shape
    )
    l = dsp.STFT.Config(
        output_magnitude=True,
        frame_length=8,
        frame_step=3,
        fft_length=8,
        time_padding='causal',
        fft_padding='right',
        name='stft',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 3)
    self.assertTrue(l.supports_step)
    self.assertEqual(1 / l.output_ratio, 3)
    self.assertEqual(l.name, 'stft')
    y = self.verify_contract(l, x, training=False)

    x_flat = x.apply_values(lambda v: v.reshape(v.shape[:2] + (-1,)))
    ys = []
    for x_i in utils.sequence_unstack(x_flat, axis=2):
      ys.append(l.layer(x_i, training=False))

    y_expected = (
        utils.sequence_stack(ys, axis=3)
        .apply_values(lambda v: v.reshape(v.shape[:3] + channel_shape))
        .mask_invalid()
    )
    self.assertSequencesClose(y, y_expected)


class InverseSTFTTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (1, 2, 3, 4),
          (1, 2),
          (2, 3),
          (
              'causal',
              # 'same',  # TODO(rryan): Fix SAME tests.
              'valid',
          ),
          ('center', 'right'),
      )
  )
  def test_inverse_stft(
      self,
      frame_length,
      frame_step,
      fft_length,
      time_padding,
      fft_padding,
  ):
    if frame_length < frame_step:
      self.skipTest('TODO(rryan): Enable length < step tests.')
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 20
    x = test_utils.random_sequence(
        batch_size, time, fft_length // 2 + 1, dtype=jnp.complex64
    )
    l = dsp.InverseSTFT.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=signal.inverse_stft_window_fn(frame_step, signal.hann_window),
        time_padding=time_padding,
        fft_padding=fft_padding,
        name='inverse_stft',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, frame_step)
    self.assertEqual(l.name, 'inverse_stft')
    # Only streamable in causal mode.
    self.assertEqual(l.supports_step, time_padding == 'causal')
    self.assertEqual(l.get_output_shape_for_sequence(x), ())
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

    # Check compatibility with tf.signal.inverse_stft (which only supports right
    # FFT padding).
    if fft_padding == 'right':
      y_tfs = tf.signal.inverse_stft(
          x.values,
          frame_length=frame_length,
          frame_step=frame_step,
          fft_length=fft_length,
          window_fn=tf.signal.inverse_stft_window_fn(
              frame_step, tf.signal.hann_window
          ),
      )
      if time_padding == 'causal':
        if trim := max(frame_length - frame_step, 0):
          y_tfs = y_tfs[:, :-trim]
      y_tfs = types.Sequence(y_tfs.numpy(), y.mask).mask_invalid().values
      self.assertAllClose(y.values, y_tfs)

  @parameterized.product(channel_shape=((1,), (2,), (2, 3)))
  def test_multichannel(self, channel_shape):
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 20
    frame_length, frame_step, fft_length = 8, 3, 8
    x = test_utils.random_sequence(
        batch_size,
        time,
        fft_length // 2 + 1,
        *channel_shape,
        low_length=time // 2,
        dtype=jnp.complex64,
    )
    l = dsp.InverseSTFT.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=signal.inverse_stft_window_fn(frame_step, signal.hann_window),
        time_padding='causal',
        fft_padding='right',
        name='inverse_stft',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.output_ratio, frame_step)
    self.assertEqual(l.name, 'inverse_stft')
    y = self.verify_contract(l, x, training=False)

    x_flat = x.apply_values(lambda v: v.reshape(v.shape[:3] + (-1,)))
    ys = []
    for x_i in utils.sequence_unstack(x_flat, axis=3):
      ys.append(l.layer(x_i, training=False))

    y_expected = (
        utils.sequence_stack(ys, axis=2)
        .apply_values(lambda v: v.reshape(v.shape[:2] + channel_shape))
        .mask_invalid()
    )
    self.assertSequencesClose(y, y_expected)


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
            window_fn=signal.hann_window,
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
            window_fn=signal.inverse_stft_window_fn(
                frame_step, signal.hann_window
            ),
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
    test_utils.SequenceLayerTest, parameterized.TestCase
):

  def test_linear_to_mel_spectrogram(self):
    key = jax.random.PRNGKey(1234)
    batch_size, time, num_spectrogram_bins = 2, 3, 5
    x = test_utils.random_sequence(batch_size, time, num_spectrogram_bins)
    l = dsp.LinearToMelSpectrogram.Config(
        num_mel_bins=8,
        sample_rate=400,
        lower_edge_hertz=1.0,
        upper_edge_hertz=200.0,
        name='linear_to_mel_spectrogram',
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'linear_to_mel_spectrogram')
    self.assertEqual(l.get_output_shape_for_sequence(x), (8,))
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)


class OverlapAddTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (
              (1, 1),
              (2, 1),
              (2, 2),
              (3, 2),
              # TODO(rryan): Fix frame_length < frame_step tests.
              # (1, 2),
              # (2, 3),
          ),
          ((), (5, 9)),
          (
              'causal',
              # 'same',  # TODO(rryan): Fix SAME tests.
              'valid',
              'semicausal_full',
          ),
      )
  )
  def test_overlap_add(self, frame_length_frame_step, inner_shape, padding):
    key = jax.random.PRNGKey(1234)
    frame_length, frame_step = frame_length_frame_step

    # TODO(rryan): Check why test fails with t = 35.
    b, t = 2, 34
    x = test_utils.random_sequence(b, t, frame_length, *inner_shape)
    l = dsp.OverlapAdd.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        padding=padding,
        name='overlap_add',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.supports_step, padding == 'causal')
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, frame_step)
    self.assertEqual(l.name, 'overlap_add')
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        inner_shape,
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)

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

    # z should not be shorter than x and the extra part should only contain
    # zeros.
    self.assertLessEqual(x.shape[1], z.shape[1])
    self.assertTrue(jnp.all(z.lengths() >= x.lengths()))
    np.testing.assert_array_equal(
        z.mask[:, x.shape[1] :],
        jnp.zeros((z.shape[0], z.shape[1] - x.shape[1]), dtype=jnp.bool_),
    )

    # The extra valid entries of z should only contain zeros.
    z_values = z.values[:, : x.shape[1]]
    z_mask = z.mask[:, : x.shape[1]]
    difference_mask = jnp.logical_xor(x.mask, z_mask)
    self.assertTrue(jnp.all(z_values[difference_mask] == 0))


class DelayTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  def test_delay_nonnegative(self):
    x = test_utils.random_sequence(2, 11, 3, 5)
    l = dsp.Delay.Config(length=-1, name='delay').make().bind({})
    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  @parameterized.product(length=(0, 1, 4), delay_layer_output=(True, False))
  def test_delay(self, length, delay_layer_output):
    x = test_utils.random_sequence(2, 11, 3, 5)
    l = (
        dsp.Delay.Config(
            length=length, delay_layer_output=delay_layer_output, name='delay'
        )
        .make()
        .bind({})
    )
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, length)
    self.assertEqual(l.output_latency, 0 if delay_layer_output else length)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'delay')
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        (3, 5),
    )
    y = self.verify_contract(l, x, training=False)
    self.assertEqual(y.shape[1], 11 + length if delay_layer_output else 11)
    self.assertEmpty(l.variables)


class LookaheadTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  def test_lookahead_nonnegative(self):
    x = test_utils.random_sequence(2, 11, 3, 5)
    l = dsp.Lookahead.Config(length=-1, name='lookahead').make().bind({})
    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  @parameterized.product(length=(0, 1, 4))
  def test_lookahead(self, length):
    x = test_utils.random_sequence(2, 11, 3, 5)
    l = dsp.Lookahead.Config(length=length, name='lookahead').make().bind({})
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, 0)
    self.assertEqual(l.output_latency, length)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'lookahead')
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        (3, 5),
    )
    y = self.verify_contract(l, x, training=False)
    self.assertEqual(y.shape[1], 11 - length)
    self.assertEmpty(l.variables)


class WindowTest(test_utils.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (20, 2),
      (15, 3),
      (10, 4),
  )
  def test_window(self, frame_step, frame_length_multiplier):
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
                ),  # (B T1 T2 S+K O)
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
                ),  # (B T S O)
                dsp.Lookahead.Config(frame_length - frame_step),
            ],
            name='test',
        )
        .make()
        .bind({})
    )

    seq_out = module.layer(seq_in, training=False)

    # Trim the lengths.
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
