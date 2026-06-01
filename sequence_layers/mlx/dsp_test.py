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
"""Tests for DSP MLX sequence layers."""

from absl.testing import absltest
from absl.testing import parameterized
import mlx.core as mx
import numpy as np

from sequence_layers.jax import dsp as jax_dsp
from sequence_layers.mlx import dsp
from sequence_layers.mlx import test_utils
from sequence_layers.specs import dsp_behaviors as spec


class DelayTest(test_utils.SequenceLayerTest, spec.DelayTest):

  def test_from_config(self):
    config = jax_dsp.Delay.Config(length=3)
    mlx_layer = dsp.Delay.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.Delay)


class LookaheadTest(test_utils.SequenceLayerTest, spec.LookaheadTest):

  def test_from_config(self):
    config = jax_dsp.Lookahead.Config(length=2)
    mlx_layer = dsp.Lookahead.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.Lookahead)


class WindowTest(test_utils.SequenceLayerTest, spec.WindowTest):

  def test_from_config(self):
    config = jax_dsp.Window.Config(axis=-1)
    mlx_layer = dsp.Window.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.Window)


class FrameTest(test_utils.SequenceLayerTest, spec.FrameTest):

  def test_from_config(self):
    config = jax_dsp.Frame.Config(
        frame_length=4,
        frame_step=2,
        padding='causal',
    )
    mlx_layer = dsp.Frame.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.Frame)


class OverlapAddTest(test_utils.SequenceLayerTest, spec.OverlapAddTest):

  def test_from_config(self):
    config = jax_dsp.OverlapAdd.Config(
        frame_length=4,
        frame_step=2,
        padding='causal',
    )
    mlx_layer = dsp.OverlapAdd.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.OverlapAdd)


class FFTTest(test_utils.SequenceLayerTest, spec.FFTTest):

  def test_from_config(self):
    config = jax_dsp.FFT.Config()
    mlx_layer = dsp.FFT.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.FFT)


class IFFTTest(test_utils.SequenceLayerTest, spec.IFFTTest):

  def test_from_config(self):
    config = jax_dsp.IFFT.Config()
    mlx_layer = dsp.IFFT.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.IFFT)


class RFFTTest(test_utils.SequenceLayerTest, spec.RFFTTest):

  def test_from_config(self):
    config = jax_dsp.RFFT.Config()
    mlx_layer = dsp.RFFT.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.RFFT)


class IRFFTTest(test_utils.SequenceLayerTest, spec.IRFFTTest):

  def test_from_config(self):
    config = jax_dsp.IRFFT.Config()
    mlx_layer = dsp.IRFFT.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.IRFFT)


class STFTTest(test_utils.SequenceLayerTest, spec.STFTTest):

  def test_from_config(self):
    config = jax_dsp.STFT.Config(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
    )
    mlx_layer = dsp.STFT.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.STFT)


class InverseSTFTTest(test_utils.SequenceLayerTest, spec.InverseSTFTTest):

  def test_from_config(self):
    config = jax_dsp.InverseSTFT.Config(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
    )
    mlx_layer = dsp.InverseSTFT.from_config(config)
    self.assertIsInstance(mlx_layer, dsp.InverseSTFT)


class LinearToMelSpectrogramTest(
    test_utils.SequenceLayerTest, spec.LinearToMelSpectrogramTest
):

  def test_from_config(self):
    config = jax_dsp.LinearToMelSpectrogram.Config(
        num_mel_bins=40,
        sample_rate=16000.0,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    mlx_layer = dsp.LinearToMelSpectrogram.from_config(config)
    self.assertIsInstance(
        mlx_layer,
        dsp.LinearToMelSpectrogram,
    )


class SignalUtilitiesTest(parameterized.TestCase):

  def test_hann_window(self):
    w = dsp.hann_window(4)
    self.assertEqual(len(w), 4)
    # Periodic Hann: endpoints should not both be zero.
    self.assertGreater(w[-1], 0.0)

  def test_frame(self):
    values = mx.arange(10).reshape(1, 10, 1).astype(mx.float32)
    framed = dsp.frame(values, 4, 2)
    self.assertEqual(framed.shape, (1, 4, 4, 1))

  def test_overlap_and_add_identity(self):
    signal_arr = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    result = dsp.overlap_and_add(signal_arr, 2)
    np.testing.assert_allclose(np.array(result), [[1.0, 2.0, 3.0, 4.0]])

  def test_mel_weight_matrix(self):
    w = dsp.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=129,
        sample_rate=16000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    self.assertEqual(w.shape, (129, 40))
    self.assertTrue(np.all(w >= 0))


if __name__ == '__main__':
  absltest.main()
