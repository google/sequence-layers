import unittest
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import signal as sp_signal

from sequence_layers.mlx import signal as mlx_signal
from sequence_layers.jax import signal as jax_signal


class WindowTest(parameterized.TestCase):

  @parameterized.product(
      length=(16, 32, 64, 65),
      periodic=(True, False),
      dtype=(np.float32, np.float64),
  )
  def test_hann_window(self, length, periodic, dtype):
    mlx_win = mlx_signal.hann_window(length, periodic=periodic, dtype=dtype)
    jax_win = jax_signal.hann_window(length, periodic=periodic, dtype=dtype)

    self.assertEqual(mlx_win.dtype, dtype)
    np.testing.assert_allclose(mlx_win, np.array(jax_win), atol=1e-6)

    # Also compare with scipy
    sym = not (periodic and (length % 2 == 0))
    scipy_win = sp_signal.windows.hann(length, sym=sym).astype(dtype)
    np.testing.assert_allclose(mlx_win, scipy_win, atol=1e-6)

  @parameterized.product(
      length=(16, 32, 64, 65),
      periodic=(True, False),
      dtype=(np.float32, np.float64),
  )
  def test_hamming_window(self, length, periodic, dtype):
    mlx_win = mlx_signal.hamming_window(length, periodic=periodic, dtype=dtype)
    jax_win = jax_signal.hamming_window(length, periodic=periodic, dtype=dtype)

    self.assertEqual(mlx_win.dtype, dtype)
    np.testing.assert_allclose(mlx_win, np.array(jax_win), atol=1e-6)

    # Also compare with scipy
    sym = not (periodic and (length % 2 == 0))
    scipy_win = sp_signal.windows.hamming(length, sym=sym).astype(dtype)
    np.testing.assert_allclose(mlx_win, scipy_win, atol=1e-6)


class InverseStftWindowFnTest(parameterized.TestCase):

  @parameterized.product(
      frame_length=(64, 128, 256),
      frame_step=(16, 32, 64),
      dtype=(np.float32, np.float64),
  )
  def test_inverse_stft_window_fn(self, frame_length, frame_step, dtype):
    if frame_step > frame_length:
      self.skipTest("frame_step must be <= frame_length")

    mlx_inv_fn = mlx_signal.inverse_stft_window_fn(frame_step)
    mlx_inv_win = mlx_inv_fn(frame_length, dtype=dtype)

    jax_inv_fn = jax_signal.inverse_stft_window_fn(frame_step)
    jax_inv_win = jax_inv_fn(frame_length, dtype=dtype)

    self.assertEqual(mlx_inv_win.dtype, dtype)
    np.testing.assert_allclose(mlx_inv_win, np.array(jax_inv_win), atol=1e-6)


if __name__ == '__main__':
  absltest.main()
