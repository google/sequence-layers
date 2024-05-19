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
"""Tests for sequence_layers.tensorflow.dsp."""

import itertools
import math

from absl.testing import parameterized
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v2 as tf


class DspTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (0, 1, 2), (tf.TensorShape((2, 10, 4)), tf.TensorShape((2, 3, 5, 9)))
      )
  )
  def test_delay(self, steps, shape):
    # Pad one invalid frame to "flush" the longest sequence.
    # Without the padding, delay fails the "step 2x" test in verify_contract,
    # because the 2x block size "flushes" a stored sample that layer trims off.
    #
    # input:  A B C D E
    # layer:  0 A B C D
    # step1x: 0 A B C D
    # step2x: 0 A B C D E
    #
    # If we add one timestep of padding, it "flushes" the full sequence:
    #
    # input:  A B C D E x
    # layer:  0 A B C D E
    # step1x: 0 A B C D E
    # step2x: 0 A B C D E
    x = self.random_sequence(*shape).pad_time(0, 1, valid=False)
    l = sl.Delay(steps)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

    # Verify behavior.
    y_layer = l.layer(x, training=False)
    y = x.pad_time(steps, 0, valid=True)
    y = y[:, : x.values.shape.dims[1].value]
    y, y_layer = self.evaluate([y, y_layer])
    self.assertAllEqual(y.values, y_layer.values)
    self.assertAllEqual(y.mask, y_layer.mask)

  def test_delay_invalid_length(self):
    with self.assertRaises(ValueError):
      sl.Delay(-3)

  @parameterized.parameters(
      itertools.product(
          (tf.TensorShape((2, 3, 32)), tf.TensorShape((2, 3, 5, 32))),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_fft(self, shape, fft_length, padding):
    x = self.random_sequence(*shape, dtype=tf.complex64)
    l = sl.FFT(fft_length, padding)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + [fft_length]
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    # FFT is not supported on tf.lite.

  @parameterized.parameters(
      itertools.product(
          (tf.TensorShape((2, 3, 32)), tf.TensorShape((2, 3, 5, 32))),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_ifft(self, shape, fft_length, padding):
    x = self.random_sequence(*shape, dtype=tf.complex64)
    l = sl.IFFT(fft_length, padding)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + [fft_length]
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    # IFFT is not supported on tf.lite.

  @parameterized.parameters(
      itertools.product(
          (tf.TensorShape((2, 3, 32)), tf.TensorShape((2, 3, 5, 32))),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_rfft(self, shape, fft_length, padding):
    x = self.random_sequence(*shape)
    l = sl.RFFT(fft_length, padding)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + [fft_length // 2 + 1]
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    # TFLite supports the RFFT op for powers of 2.
    if math.log2(fft_length).is_integer():
      self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(
      itertools.product(
          (tf.TensorShape((2, 3, 17)), tf.TensorShape((2, 3, 5, 17))),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_irfft(self, shape, fft_length, padding):
    x = self.random_sequence(*shape, dtype=tf.complex64)
    l = sl.IRFFT(fft_length, padding)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + [fft_length]
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    # A standalone IRFFT layer requires complex64 *inputs*, which aren't
    # supported by tf.lite.

  @parameterized.parameters(
      itertools.product(
          (tf.TensorShape((2, 3, 32)), tf.TensorShape((2, 3, 5, 32))),
          (31, 32, 33),
          ('center', 'right'),
      )
  )
  def test_rfft_irfft(self, shape, fft_length, padding):
    x = self.random_sequence(*shape)
    l = sl.Serial([sl.RFFT(fft_length, padding), sl.IRFFT(fft_length, padding)])
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:-1] + [fft_length]
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    # TFLite supports the RFFT/IRFFT ops for powers of 2.
    if math.log2(fft_length).is_integer():
      self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(
      itertools.product(
          ((1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)),
          (tf.TensorShape((2, 10, 4)), tf.TensorShape((2, 3, 5, 9))),
      )
  )
  def test_frame(self, frame_length_frame_step, shape):
    frame_length, frame_step = frame_length_frame_step
    x = self.random_sequence(*shape)
    l = sl.Frame(frame_length=frame_length, frame_step=frame_step)
    self.assertEqual(l.block_size, frame_step)
    self.assertEqual(1 / l.output_ratio, frame_step)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), [frame_length] + shape[2:]
    )
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  @parameterized.parameters(
      itertools.product(
          (sl.Spectrogram, sl.STFT),
          (1, 2, 3, 4),
          (1, 2),
          (2, 3),
          ('center', 'right'),
      )
  )
  def test_stft(
      self, layer_type, frame_length, frame_step, fft_length, padding
  ):
    """Tests both sl.STFT and sl.Spectrogram."""
    batch_size, time = 2, 20
    x = self.random_sequence(batch_size, time)
    l_fft = layer_type(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        padding=padding,
        use_dft_matrix=False,
    )
    self.assertEqual(l_fft.block_size, frame_step)
    self.assertEqual(1 / l_fft.output_ratio, frame_step)
    self.assertEqual(
        l_fft.get_output_shape_for_sequence(x), [fft_length // 2 + 1]
    )
    _, y_fft = self.verify_contract(l_fft, x, training=False)
    self.assertEmpty(l_fft.variables)
    self.assertEmpty(l_fft.trainable_variables)
    # TFLite supports the RFFT op for powers of 2.
    if layer_type is sl.Spectrogram and math.log2(fft_length).is_integer():
      self.verify_tflite_step(l_fft, x, use_flex=True)

    l_dft = layer_type(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        padding=padding,
        use_dft_matrix=True,
    )
    self.assertEqual(l_dft.block_size, frame_step)
    self.assertEqual(1 / l_dft.output_ratio, frame_step)
    self.assertEqual(
        l_dft.get_output_shape_for_sequence(x), [fft_length // 2 + 1]
    )
    _, y_dft = self.verify_contract(l_dft, x, training=False)
    self.assertEmpty(l_dft.variables)
    self.assertEmpty(l_dft.trainable_variables)
    is_spectrogram = layer_type is sl.Spectrogram
    if is_spectrogram:
      self.verify_tflite_step(l_dft, x)

    # Check that the FFT and DFT based approaches produce identical results.
    self.assertAllClose(y_fft.values, y_dft.values)
    self.assertAllEqual(y_fft.mask, y_dft.mask)

    # Check compatibility with tf.signal.stft.
    if padding == 'right':
      # Causal padding.
      x = x.pad_time(frame_length - 1, 0, valid=True)
      y_tfs = tf.signal.stft(
          x.values,
          frame_length=frame_length,
          frame_step=frame_step,
          fft_length=fft_length,
          pad_end=False,
      )
      if is_spectrogram:
        y_tfs = tf.abs(y_tfs)
      y_tfs = sl.Sequence(y_tfs, y_fft.mask).mask_invalid().values
      y_tfs = self.evaluate(y_tfs)
      self.assertAllClose(y_fft.values, y_tfs)

  def test_linear_to_mel_spectrogram(self):
    batch_size, time, num_spectrogram_bins = 2, 3, 5
    x = self.random_sequence(batch_size, time, num_spectrogram_bins)
    l = sl.LinearToMelSpectrogram(
        num_mel_bins=8,
        sample_rate=400,
        lower_edge_hertz=1.0,
        upper_edge_hertz=200.0,
    )

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), [8])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)

  def test_linear_to_mel_spectrogram_caching(self):
    batch_size, time, num_spectrogram_bins = 2, 3, 5
    x = self.random_sequence(batch_size, time, num_spectrogram_bins)
    l = sl.LinearToMelSpectrogram(
        num_mel_bins=8,
        sample_rate=400,
        lower_edge_hertz=1.0,
        upper_edge_hertz=200.0,
    )

    # Build in Eager (TF2) or graph (TF1) mode.
    is_tf2 = tf.executing_eagerly()
    l.layer(x, training=False)
    mel_kernel = l._mel_kernel

    @tf.function
    def fn():
      l.layer(x, training=False)

      # mel_kernel is not recomputed in TF2 mode, but is in TF1
      if is_tf2:
        self.assertIs(l._mel_kernel, mel_kernel)
      else:
        self.assertIsNot(l._mel_kernel, mel_kernel)

    fn()

    l = sl.LinearToMelSpectrogram(
        num_mel_bins=8,
        sample_rate=400,
        lower_edge_hertz=1.0,
        upper_edge_hertz=200.0,
    )

    @tf.function
    def fn2():
      self.assertIsNone(l._mel_kernel)
      return l.layer(x, training=False)

    fn2()
    mel_kernel = l._mel_kernel
    l.layer(x, training=False)
    # In both Eager and graph mode, the cached FuncGraph mel kernel is not used.
    self.assertIsNot(l._mel_kernel, mel_kernel)


if __name__ == '__main__':
  tf.test.main()
