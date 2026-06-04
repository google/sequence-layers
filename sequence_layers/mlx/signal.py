"""Signal utilities for MLX, ported from sequence_layers.jax.signal."""

import numpy as np
import mlx.core as mx


def _raised_cosine_window(window_length, periodic, dtype, a, b):
  """Computes a raised cosine window."""
  if window_length == 1:
    return np.ones([1], dtype=dtype)
  even = 1 - window_length % 2
  n = np.asarray(window_length + int(periodic) * even - 1, dtype=dtype)
  count = np.arange(window_length, dtype=dtype)
  cos_arg = 2 * np.pi * count / n
  return a - b * np.cos(cos_arg)


def hann_window(window_length, periodic=True, dtype=np.float32):
  """Computes a hann window. Ported from tf.signal."""
  return _raised_cosine_window(window_length, periodic, dtype, 0.5, 0.5)


def hamming_window(window_length, periodic=True, dtype=np.float32):
  """Computes a Hamming window."""
  a0 = 0.54
  return _raised_cosine_window(window_length, periodic, dtype, a0, 1.0 - a0)


def inverse_stft_window_fn(frame_step, forward_window_fn=hann_window):
  """Generates a window function that can be used in inverse STFT.

  Constructs a window that is equal to the forward window with a further
  pointwise amplitude correction.

  Args:
    frame_step: The number of samples to step.
    forward_window_fn: Window function used in the forward STFT transform.

  Returns:
    A callable that takes a window length and a dtype keyword argument and
    returns a [window_length] array of window samples.
  """

  def inverse_stft_window_fn_inner(frame_length, dtype=np.float32):
    """Computes a window suitable for inverse STFT reconstruction."""
    # Use equation 7 from Griffin + Lim.
    forward_window = forward_window_fn(frame_length, dtype=dtype)
    # Convert to mx array for computation.
    fw = mx.array(forward_window, dtype=mx.float32)
    denom = mx.square(fw)
    overlaps = -(-frame_length // frame_step)  # Ceiling division.
    denom = mx.pad(denom, [(0, overlaps * frame_step - frame_length)])
    denom = mx.reshape(denom, [overlaps, frame_step])
    denom = mx.sum(denom, axis=0, keepdims=True)
    denom = mx.tile(denom, [overlaps, 1])
    denom = mx.reshape(denom, [overlaps * frame_step])
    denom = denom[:frame_length]
    result = mx.where(denom == 0.0, 0, fw / denom)
    # Convert back to numpy for consistency with the forward window.
    return np.array(result, dtype=dtype)

  return inverse_stft_window_fn_inner
