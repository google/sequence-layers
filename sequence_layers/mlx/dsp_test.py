"""Tests for DSP MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import dsp
from sequence_layers.mlx import test_utils


class DelayTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.Delay(length=3)
    # Delay with delay_layer_output=True pads output time,
    # so step/layer shapes differ. Test separately.
    x = test_utils.random_sequence(2, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (4,))
    # Layer pads: output time = 8 + 3 = 11.
    self.assertEqual(y.shape[1], 11)

  def test_step(self):
    layer = dsp.Delay(length=3)
    x = test_utils.random_sequence(1, 8, 4)
    y_step, _ = test_utils.step_by_step(layer, x)
    # Step output: same time as input.
    self.assertEqual(y_step.shape[1], 8)
    # First 3 outputs should be masked (invalid).
    np.testing.assert_array_equal(
        np.array(y_step.mask[0, :3]), [False, False, False]
    )
    np.testing.assert_array_equal(np.array(y_step.mask[0, 3:]), [True] * 5)

  def test_zero_delay(self):
    layer = dsp.Delay(length=0)
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_delays_output(self):
    layer = dsp.Delay(length=2)
    values = mx.array([[[1.0], [2.0], [3.0], [4.0]]])
    mask = mx.ones((1, 4), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    # First 2 timesteps should be invalid (padded).
    self.assertEqual(y.shape, (1, 6, 1))
    # Mask should have first 2 False.
    np.testing.assert_array_equal(
        np.array(y.mask[0]),
        [False, False, True, True, True, True],
    )

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.Delay.Config(length=3)
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.Delay)


class LookaheadTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.Lookahead(length=2)
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    # Drops first 2 timesteps.
    self.assertEqual(y.shape[1], 6)

  def test_zero_lookahead(self):
    layer = dsp.Lookahead(length=0)
    test_utils.verify_contract(
        self,
        layer,
        (4,),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_preserve_length(self):
    layer = dsp.Lookahead(length=2, preserve_length_in_layer=True)
    x = test_utils.random_sequence(1, 8, 4)
    y = layer.layer(x)
    self.assertEqual(y.shape[1], 8)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.Lookahead.Config(length=2)
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.Lookahead)


class WindowTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.Window(axis=-1)
    # Input: [batch, time, frame_length]
    x = test_utils.random_sequence(1, 4, 8)
    y = layer.layer(x)
    self.assertEqual(y.shape, x.shape)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.Window.Config(axis=-1)
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.Window)


class FrameTest(parameterized.TestCase):

  def test_valid_padding(self):
    layer = dsp.Frame(
        frame_length=4,
        frame_step=2,
        padding='valid',
    )
    x = test_utils.random_sequence(1, 8, 1)
    y = layer.layer(x)
    # (8 - 4) // 2 + 1 = 3 frames
    self.assertEqual(y.shape[1], 3)
    self.assertEqual(y.channel_shape, (4, 1))

  @parameterized.parameters(
      ('causal',),
      ('causal_valid',),
  )
  def test_causal_paddings(self, padding):
    layer = dsp.Frame(
        frame_length=4,
        frame_step=2,
        padding=padding,
    )
    test_utils.verify_contract(
        self,
        layer,
        (1,),
        time=8,
        atol=1e-5,
        rtol=1e-5,
    )

  def test_output_shape(self):
    layer = dsp.Frame(frame_length=4, frame_step=2)
    self.assertEqual(layer.get_output_shape((3,)), (4, 3))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.Frame.Config(
        frame_length=4,
        frame_step=2,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.Frame)


class OverlapAddTest(parameterized.TestCase):

  def test_valid(self):
    layer = dsp.OverlapAdd(
        frame_length=4,
        frame_step=2,
        padding='valid',
    )
    # Input: [batch, frames, frame_length]
    x = bt.MaskedSequence(
        mx.ones((1, 3, 4)),
        mx.ones((1, 3), dtype=mx.bool_),
    )
    y = layer.layer(x)
    # (3-1)*2 + 4 = 8
    self.assertEqual(y.shape, (1, 8))

  def test_causal(self):
    layer = dsp.OverlapAdd(
        frame_length=4,
        frame_step=2,
        padding='causal',
    )
    # Build input: [batch, frames, frame_length]
    x = bt.MaskedSequence(
        mx.ones((1, 3, 4)),
        mx.ones((1, 3), dtype=mx.bool_),
    )
    y = layer.layer(x)
    # Causal trims overlap: output = frames * frame_step = 6
    self.assertEqual(y.shape[1], 6)

  def test_output_shape(self):
    layer = dsp.OverlapAdd(
        frame_length=4,
        frame_step=2,
        padding='valid',
    )
    self.assertEqual(layer.get_output_shape((4, 3)), (3,))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.OverlapAdd.Config(
        frame_length=4,
        frame_step=2,
        padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.OverlapAdd)


class FFTTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.FFT()
    x = test_utils.random_sequence(1, 4, 8)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (8,))

  def test_fft_length(self):
    layer = dsp.FFT(fft_length=16)
    x = test_utils.random_sequence(1, 4, 8)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (16,))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.FFT.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.FFT)


class IFFTTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.IFFT()
    values = mx.random.normal(shape=(1, 4, 8)) + 0j
    values = values.astype(mx.complex64)
    mask = mx.ones((1, 4), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (8,))


class RFFTTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.RFFT()
    x = test_utils.random_sequence(1, 4, 8)
    y = layer.layer(x)
    # RFFT output size: 8 // 2 + 1 = 5
    self.assertEqual(y.channel_shape, (5,))
    self.assertEqual(y.dtype, mx.complex64)

  def test_fft_length(self):
    layer = dsp.RFFT(fft_length=16)
    x = test_utils.random_sequence(1, 4, 8)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (9,))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.RFFT.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.RFFT)


class IRFFTTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.IRFFT()
    # 5 complex bins -> irfft -> 8 real samples
    values = mx.random.normal(shape=(1, 4, 5)) + 0j
    values = values.astype(mx.complex64)
    mask = mx.ones((1, 4), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (8,))
    self.assertEqual(y.dtype, mx.float32)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.IRFFT.Config()
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.IRFFT)


class STFTTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.STFT(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
    )
    x = test_utils.random_sequence(1, 32, 1)
    y = layer.layer(x)
    # 32 / 8 = 4 frames, each with fft_length/2+1 = 9 bins.
    self.assertEqual(y.shape[1], 4)
    self.assertEqual(y.channel_shape, (9, 1))

  def test_layer_magnitude(self):
    layer = dsp.STFT(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
        output_magnitude=True,
    )
    x = test_utils.random_sequence(1, 32, 1)
    y = layer.layer(x)
    self.assertEqual(y.dtype, mx.float32)
    # All magnitudes should be >= 0.
    self.assertTrue(bool(mx.all(y.values >= 0)))

  def test_step(self):
    layer = dsp.STFT(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
    )
    test_utils.verify_contract(
        self,
        layer,
        (1,),
        time=32,
        atol=1e-4,
        rtol=1e-4,
    )

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.STFT.Config(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.STFT)


class InverseSTFTTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.InverseSTFT(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
    )
    # Input: [batch, frames, fft_bins]
    num_bins = 16 // 2 + 1
    values = mx.random.normal(shape=(1, 4, num_bins)) + 0j
    values = values.astype(mx.complex64)
    mask = mx.ones((1, 4), dtype=mx.bool_)
    x = bt.MaskedSequence(values, mask)
    y = layer.layer(x)
    # Causal: output = frames * frame_step = 32
    self.assertEqual(y.shape[1], 32)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.InverseSTFT.Config(
        frame_length=16,
        frame_step=8,
        fft_length=16,
        time_padding='causal',
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, dsp.InverseSTFT)


class LinearToMelSpectrogramTest(parameterized.TestCase):

  def test_layer(self):
    layer = dsp.LinearToMelSpectrogram(
        num_mel_bins=40,
        sample_rate=16000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    # Input: [batch, time, fft_bins]
    x = test_utils.random_sequence(1, 4, 129)
    y = layer.layer(x)
    self.assertEqual(y.channel_shape, (40,))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import dsp as jax_dsp

    config = jax_dsp.LinearToMelSpectrogram.Config(
        num_mel_bins=40,
        sample_rate=16000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    mlx_layer = config.make(backend='mlx')
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
    # (10 - 4) // 2 + 1 = 4 frames
    self.assertEqual(framed.shape, (1, 4, 4, 1))

  def test_overlap_and_add_identity(self):
    # No overlap: frame_step == frame_length.
    signal = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    result = dsp.overlap_and_add(signal, 2)
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
    # Weights should be non-negative.
    self.assertTrue(np.all(w >= 0))


if __name__ == '__main__':
  absltest.main()
