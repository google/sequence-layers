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
"""Tests for signal utilities."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import signal as sp_signal
from sequence_layers.jax import signal
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils
import tensorflow as tf

from google3.testing.pybase import parameterized


def _tf_signal_frame(
    x: jax.Array,
    frame_length: int,
    frame_step: int,
    axis: int,
    pad_mode: types.PaddingModeString | tuple[int, int],
):
  explicit_padding = utils.convolution_explicit_padding(
      pad_mode, kernel_size=frame_length, stride=frame_step, dilation_rate=1
  )

  paddings = [(0, 0)] * x.ndim
  paddings[axis] = explicit_padding
  x = jnp.pad(x, paddings)

  return tf.signal.frame(
      tf.convert_to_tensor(x),
      frame_length,
      frame_step,
      axis=axis,
      pad_end=False,
  ).numpy()


class FrameTest(test_utils.SequenceLayerTest):

  def test_frame(self):

    x = jnp.arange(1, 6, dtype=jnp.int32)

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=1, pad_mode='causal_valid'),
        jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=2, pad_mode='causal_valid'),
        jnp.asarray([[0, 1], [2, 3], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=3, pad_mode='causal_valid'),
        jnp.asarray([[0, 1], [3, 4]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=1, pad_mode='causal'),
        jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=2, pad_mode='causal'),
        jnp.asarray([[0, 1], [2, 3], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=3, pad_mode='causal'),
        jnp.asarray([[0, 1], [3, 4]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=1, pad_mode='semicausal'),
        jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=2, pad_mode='semicausal'),
        jnp.asarray([[1, 2], [3, 4], [5, 0]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=3, pad_mode='semicausal'),
        jnp.asarray([[1, 2], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(
            x, frame_length=2, frame_step=1, pad_mode='reverse_causal_valid'
        ),
        jnp.asarray([[1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]),
    )

    self.assertAllEqual(
        signal.frame(
            x, frame_length=2, frame_step=2, pad_mode='reverse_causal_valid'
        ),
        jnp.asarray([[1, 2], [3, 4], [5, 0]]),
    )

    self.assertAllEqual(
        signal.frame(
            x, frame_length=2, frame_step=3, pad_mode='reverse_causal_valid'
        ),
        jnp.asarray([[1, 2], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(
            x, frame_length=2, frame_step=1, pad_mode='reverse_causal'
        ),
        jnp.asarray([[1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]),
    )

    self.assertAllEqual(
        signal.frame(
            x, frame_length=2, frame_step=2, pad_mode='reverse_causal'
        ),
        jnp.asarray([[1, 2], [3, 4], [5, 0]]),
    )

    self.assertAllEqual(
        signal.frame(
            x, frame_length=2, frame_step=3, pad_mode='reverse_causal'
        ),
        jnp.asarray([[1, 2], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=1, pad_mode='valid'),
        jnp.asarray([[1, 2], [2, 3], [3, 4], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=2, pad_mode='valid'),
        jnp.asarray([[1, 2], [3, 4]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=3, pad_mode='valid'),
        jnp.asarray([[1, 2], [4, 5]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=1, pad_mode='same'),
        jnp.asarray([[1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=2, pad_mode='same'),
        jnp.asarray([[1, 2], [3, 4], [5, 0]]),
    )

    self.assertAllEqual(
        signal.frame(x, frame_length=2, frame_step=3, pad_mode='same'),
        jnp.asarray([[1, 2], [4, 5]]),
    )

  @parameterized.product(
      length_step=(
          # No common factors:
          (7, 11),
          (11, 7),
          (7, 7),
          # Common factors:
          (7 * 3, 11 * 3),
          (11 * 3, 7 * 3),
          (7 * 3, 7 * 3),
      ),
      padding=(
          'valid',
          'same',
          'reverse_causal_valid',
          'causal_valid',
          'reverse_causal',
          'causal',
          'semicausal',
      ),
      outer_dimensions=((), (1,), (2,), (3, 5)),
      inner_dimensions=((), (1,), (2,), (3, 5)),
      dtype=(jnp.int32, jnp.float32),
  )
  def test_tf_signal_equivalence(
      self,
      length_step,
      padding,
      inner_dimensions,
      outer_dimensions,
      dtype,
  ):
    frame_length, frame_step = length_step
    axis = len(outer_dimensions)

    for time in range(2 * frame_length + frame_step):
      with self.subTest(f'time{time}'):
        shape = (*outer_dimensions, time, *inner_dimensions)
        if dtype == jnp.int32:
          x = jnp.array(np.random.randint(0, 2048, size=shape), dtype=dtype)
        else:
          x = jnp.array(np.random.normal(size=shape), dtype=dtype)

        y = signal.frame(
            x, frame_length, frame_step, axis=axis, pad_mode=padding
        )

        y_expected = _tf_signal_frame(
            x, frame_length, frame_step, axis=axis, pad_mode=padding
        )
        self.assertEqual(y.dtype, dtype)
        self.assertAllEqual(y, y_expected)


class OverlapAndAddTest(test_utils.SequenceLayerTest):

  def test_overlap_and_add(self):
    x = jnp.arange(1, 6, dtype=jnp.int32)

    y = signal.frame(
        x, frame_length=2, frame_step=3, pad_mode='reverse_causal_valid'
    )
    z = signal.overlap_and_add(y, frame_step=3)
    self.assertAllEqual(z, jnp.asarray([1, 2, 0, 4, 5]))

    y = signal.frame(
        x, frame_length=3, frame_step=3, pad_mode='reverse_causal_valid'
    )
    z = signal.overlap_and_add(y, frame_step=3)
    self.assertAllEqual(z, jnp.asarray([1, 2, 3, 4, 5, 0]))

    y = signal.frame(
        x, frame_length=4, frame_step=3, pad_mode='reverse_causal_valid'
    )
    z = signal.overlap_and_add(y, frame_step=3)
    self.assertAllEqual(z, jnp.asarray([1, 2, 3, 8, 5, 0, 0]))


class WindowTest(parameterized.TestCase):

  @parameterized.parameters(
      (64, True, jnp.float32),
      (64, False, jnp.float32),
      (65, True, jnp.float32),
      (65, False, jnp.float32),
      (64, True, jnp.float64),
      (64, False, jnp.float64),
      (65, True, jnp.float64),
      (65, False, jnp.float64),
  )
  def test_hann_window(self, length, periodic, dtype):

    win_dsp = signal.hann_window(length, periodic=periodic, dtype=dtype)

    # This should call the scipy function.
    win_hann_fn = signal.get_window_fn('hann')
    win_npy = win_hann_fn(length, periodic=periodic, dtype=dtype)

    err = abs(win_dsp - win_npy).max()
    self.assertLess(err, 1e-5)

  @parameterized.parameters(
      (64, True, jnp.float32),
      (64, False, jnp.float32),
      (65, True, jnp.float32),
      (65, False, jnp.float32),
      (64, True, jnp.float64),
      (64, False, jnp.float64),
      (65, True, jnp.float64),
      (65, False, jnp.float64),
  )
  def test_hamming_window(self, length, periodic, dtype):

    # This should call the gemax implementation.
    win_dsp = signal.hamming_window(length, periodic=periodic, dtype=dtype)

    # This should call the scipy function.
    win_hamming_fn = signal.get_window_fn('hamming')
    win_npy = win_hamming_fn(length, periodic=periodic, dtype=dtype)

    err = abs(win_dsp - win_npy).max()
    self.assertLess(err, 1e-5)

  @parameterized.parameters(
      (
          14,  # kaiser with beta=14
          64,
          True,
          jnp.float32,
          True,
      ),
      (
          ('general_cosine', [0.54, 1.0 - 0.54]),  # i.e., Hamming window.
          50,
          True,
          jnp.float32,
          True,
      ),
      (
          ('gaussian', 7),  # gaussian with sigma == 7
          51,
          False,
          jnp.float32,
          False,
      ),
      ('blackman', 65, True, jnp.float32, False),
  )
  def test_get_window(self, name, length, periodic, dtype, fftbins):
    """Tests possible types for the name argument of get_window."""

    # Call scipy directly.
    win_sig = sp_signal.get_window(name, length, fftbins=fftbins)

    # This should call the scipy function.
    win_fn = signal.get_window_fn(name)
    win_npy = win_fn(length, periodic=periodic, dtype=dtype)

    err = abs(win_sig - win_npy).max()
    self.assertLess(err, 1e-5)


if __name__ == '__main__':
  test_utils.main()
