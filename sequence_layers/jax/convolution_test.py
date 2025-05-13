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
"""Tests for convolution helpers."""

from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
from sequence_layers.jax import convolution
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils


def _expected_conv_mask(
    mask: jax.Array,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: types.PaddingModeString,
) -> jax.Array:
  """Output timesteps are valid iff they only touch valid input timesteps."""
  match padding:
    case (
        types.PaddingMode.SAME.value
        | types.PaddingMode.CAUSAL.value
        | types.PaddingMode.REVERSE_CAUSAL.value
        | types.PaddingMode.SEMICAUSAL.value
    ):
      return jax.lax.conv_general_dilated(
          mask[:, :, jnp.newaxis].astype(jnp.float32),
          jnp.ones([1, 1, 1]),
          window_strides=[stride],
          padding='VALID',
          rhs_dilation=[dilation_rate],
          dimension_numbers=('NHC', 'HIO', 'NHC'),
      )[:, :, 0].astype(jnp.bool_)
    case (
        types.PaddingMode.VALID.value
        | types.PaddingMode.CAUSAL_VALID.value
        | types.PaddingMode.REVERSE_CAUSAL_VALID.value
    ):
      explicit_padding = utils.convolution_explicit_padding(
          # TODO(rryan): Why do we need to validate here?
          types.validate_padding(padding),
          kernel_size,
          stride,
          dilation_rate,
      )
      # Causal padding pads the mask with valid samples.
      mask = jnp.pad(
          mask,
          [(0, 0), explicit_padding],
          constant_values=padding == types.PaddingMode.CAUSAL_VALID.value,
      )
      mask_golden = jax.lax.conv_general_dilated(
          mask[:, :, jnp.newaxis].astype(jnp.float32),
          jnp.ones([kernel_size, 1, 1]),
          window_strides=[stride],
          padding='VALID',
          rhs_dilation=[dilation_rate],
          dimension_numbers=('NHC', 'HIO', 'NHC'),
      )[:, :, 0]
      # Only timesteps that add up to kernel_size are valid.
      return mask_golden > kernel_size - 1e-3
    case types.PaddingMode.SEMICAUSAL_FULL.value:
      explicit_padding = utils.convolution_explicit_padding(
          types.validate_padding(padding),
          kernel_size,
          stride,
          dilation_rate,
      )
      # Causal padding pads the mask with valid samples.
      mask = jnp.pad(
          mask,
          [(0, 0), explicit_padding],
          constant_values=False,
      )
      mask_golden = jax.lax.conv_general_dilated(
          mask[:, :, jnp.newaxis].astype(jnp.float32),
          jnp.ones([kernel_size, 1, 1]),
          window_strides=[stride],
          padding='VALID',
          rhs_dilation=[dilation_rate],
          dimension_numbers=('NHC', 'HIO', 'NHC'),
      )[:, :, 0]
      # All timesteps where the kernel overlaps with the mask.
      return mask_golden > 0


class ComputeConvMaskTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      params=[
          # kernel_size 1
          (1, 1, 1),
          (1, 2, 1),
          (1, 1, 2),
          (1, 3, 1),
          (1, 1, 3),
          # kernel_size > stride or dilation:
          (5, 1, 1),
          (5, 2, 1),
          (5, 1, 2),
          (5, 3, 1),
          (5, 1, 3),
          # kernel_size = stride or dilation:
          (2, 2, 1),
          (2, 1, 2),
          (3, 3, 1),
          (3, 1, 3),
          # kernel_size < stride or dilation:
          (3, 4, 1),
          (3, 1, 4),
          (3, 5, 1),
          (3, 1, 5),
      ],
      padding=[
          'valid',
          'same',
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
          'semicausal_full',
      ],
  )
  def test_dense_left_aligned_mask(self, params, padding):
    kernel_size, stride, dilation_rate = params
    # Try even and odd tensor lengths.
    for maxlen in [16, 17]:
      # Sweep all possible lengths.
      lengths = jnp.arange(maxlen + 1, dtype=jnp.int32)[:, jnp.newaxis]
      mask = jnp.arange(maxlen, dtype=jnp.int32)[jnp.newaxis, :] < lengths

      expected = _expected_conv_mask(
          mask, kernel_size, stride, dilation_rate, padding
      )
      actual = convolution.compute_conv_mask(
          mask,
          kernel_size,
          stride,
          dilation_rate,
          padding,
          is_step=False,
      )

      self.assertAllEqual(actual, expected)

  @parameterized.product(
      params=[
          # kernel_size 1
          (1, 1, 1),
          (1, 2, 1),
          (1, 1, 2),
          (1, 3, 1),
          (1, 1, 3),
          # kernel_size > stride or dilation:
          (5, 1, 1),
          (5, 2, 1),
          (5, 1, 2),
          (5, 3, 1),
          (5, 1, 3),
          # kernel_size = stride or dilation:
          (2, 2, 1),
          (2, 1, 2),
          (3, 3, 1),
          (3, 1, 3),
          # kernel_size < stride or dilation:
          (3, 4, 1),
          (3, 1, 4),
          (3, 5, 1),
          (3, 1, 5),
      ],
      padding=[
          'valid',
          'same',
          'causal_valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
  )
  def test_sparse_mask(self, params, padding):
    key = jax.random.PRNGKey(1234)
    kernel_size, stride, dilation_rate = params
    # Try even and odd tensor lengths.
    for key, maxlen in zip(jax.random.split(key, 2), [16, 17]):
      # Sweep all possible lengths.
      mask = jax.random.uniform(key, shape=[maxlen, maxlen]) > 0.5
      expected = _expected_conv_mask(
          mask, kernel_size, stride, dilation_rate, padding
      )
      actual = convolution.compute_conv_mask(
          mask,
          kernel_size,
          stride,
          dilation_rate,
          padding,
          is_step=False,
      )

      self.assertAllEqual(actual, expected)

  def test_valid_padding(self):
    def check(mask, kernel_size, stride, dilation_rate, padding, expected):
      actual = convolution.compute_conv_mask(
          jnp.asarray(mask),
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
          is_step=False,
      )
      self.assertAllEqual(actual, jnp.asarray(expected))

    # kernel_size > stride
    check(
        mask=[[True, True, True, False]],
        kernel_size=3,
        stride=2,
        dilation_rate=1,
        padding='valid',
        expected=[[
            True,
        ]],
    )

    # kernel_size == stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=3,
        stride=3,
        dilation_rate=1,
        padding='valid',
        expected=[[True, True]],
    )

    # kernel_size < stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=2,
        stride=3,
        dilation_rate=1,
        padding='valid',
        expected=[[True, True]],
    )

    # kernel dilation
    check(
        mask=[[True, False, True, False, True, False, True, False]],
        kernel_size=3,
        stride=1,
        dilation_rate=2,
        padding='valid',
        expected=[[True, False, True, False]],
    )

  def test_causal_valid_padding(self):
    # TODO(rryan): Consider merging with `test_valid_padding` since they are
    # the same except for whether the output length equals the input length.
    def check(mask, kernel_size, stride, dilation_rate, padding, expected):
      actual = convolution.compute_conv_mask(
          jnp.asarray(mask),
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
          is_step=False,
      )
      self.assertAllEqual(actual, jnp.asarray(expected))

    # kernel_size > stride
    check(
        mask=[[True, True, True, False]],
        kernel_size=3,
        stride=2,
        dilation_rate=1,
        padding='causal_valid',
        expected=[[
            True,
            True,
        ]],
    )

    # kernel_size == stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=3,
        stride=3,
        dilation_rate=1,
        padding='causal_valid',
        expected=[[True, True, False]],
    )

    # kernel_size < stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=2,
        stride=3,
        dilation_rate=1,
        padding='causal_valid',
        expected=[[True, True, False]],
    )

    # kernel dilation
    check(
        mask=[[True, False, True, False, True, False, True, False]],
        kernel_size=3,
        stride=1,
        dilation_rate=2,
        padding='causal_valid',
        expected=[[True, False, True, False, True, False, True, False]],
    )

  def test_reverse_causal_valid_padding(self):
    # TODO(rryan): Consider merging with `test_valid_padding` since they are
    # the same except for whether the output length equals the input length.
    def check(mask, kernel_size, stride, dilation_rate, padding, expected):
      actual = convolution.compute_conv_mask(
          jnp.asarray(mask),
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
          is_step=False,
      )
      self.assertAllEqual(actual, jnp.asarray(expected))

    # kernel_size > stride
    check(
        mask=[[True, True, True, False]],
        kernel_size=3,
        stride=2,
        dilation_rate=1,
        padding='reverse_causal_valid',
        expected=[[
            True,
            False,
        ]],
    )

    # kernel_size == stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=3,
        stride=3,
        dilation_rate=1,
        padding='reverse_causal_valid',
        expected=[[True, True, False]],
    )

    # kernel_size < stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=2,
        stride=3,
        dilation_rate=1,
        padding='reverse_causal_valid',
        expected=[[True, True, False]],
    )

    # kernel dilation
    check(
        mask=[[True, False, True, False, True, False, True, False]],
        kernel_size=3,
        stride=1,
        dilation_rate=2,
        padding='reverse_causal_valid',
        expected=[[True, False, True, False, False, False, False, False]],
    )

  @parameterized.parameters('same', 'causal', 'reverse_causal')
  def test_same_like_padding(self, padding):
    def check(mask, kernel_size, stride, dilation_rate, padding, expected):
      actual = convolution.compute_conv_mask(
          jnp.asarray(mask),
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
          is_step=False,
      )
      self.assertAllEqual(actual, jnp.asarray(expected))

    # kernel_size > stride
    check(
        mask=[[True, True, True, False]],
        kernel_size=3,
        stride=2,
        dilation_rate=1,
        padding=padding,
        expected=[[
            True,
            True,
        ]],
    )

    # kernel_size == stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=3,
        stride=3,
        dilation_rate=1,
        padding=padding,
        expected=[[True, True, False]],
    )

    # kernel_size < stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=2,
        stride=3,
        dilation_rate=1,
        padding=padding,
        expected=[[True, True, False]],
    )

    # kernel dilation
    check(
        mask=[[True, False, True, False, True, False, True, False]],
        kernel_size=3,
        stride=1,
        dilation_rate=2,
        padding=padding,
        expected=[[True, False, True, False, True, False, True, False]],
    )

  def test_semicausal(self):
    def check(mask, kernel_size, stride, dilation_rate, padding, expected):
      actual = convolution.compute_conv_mask(
          jnp.asarray(mask),
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
          is_step=False,
      )
      self.assertAllEqual(actual, jnp.asarray(expected))

    # kernel_size > stride
    check(
        mask=[[True, True, True, False]],
        kernel_size=3,
        stride=2,
        dilation_rate=1,
        padding='semicausal',
        expected=[[
            True,
            True,
        ]],
    )

    # kernel_size == stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=3,
        stride=3,
        dilation_rate=1,
        padding='semicausal',
        expected=[[True, True, False]],
    )

    # kernel_size < stride
    check(
        mask=[[True, True, True, True, True, True, False]],
        kernel_size=2,
        stride=3,
        dilation_rate=1,
        padding='semicausal',
        expected=[[True, True, False]],
    )

    # kernel dilation
    check(
        mask=[[True, False, True, False, True, False, True, False]],
        kernel_size=3,
        stride=1,
        dilation_rate=2,
        padding='semicausal',
        expected=[[True, False, True, False, True, False, True, False]],
    )

  def test_semicausal_full_padding(self):
    def check(mask, kernel_size, stride, dilation_rate, padding, expected):
      actual = convolution.compute_conv_mask(
          jnp.asarray(mask),
          kernel_size=kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
          padding=padding,
          is_step=False,
      )
      self.assertAllEqual(actual, jnp.asarray(expected))

    # kernel_size > stride
    check(
        mask=[[True, True, True, False]],
        kernel_size=3,
        stride=2,
        dilation_rate=1,
        padding='semicausal_full',
        expected=[[
            True,
            True,
            False,
        ]],
    )

    # kernel_size == stride
    check(
        mask=[[True, True, True, True, True, True, True, False, False]],
        kernel_size=3,
        stride=3,
        dilation_rate=1,
        padding='semicausal_full',
        expected=[[True, True, True]],
    )

    # kernel_size < stride
    check(
        mask=[[True, True, True, True, True, True, True, False, False]],
        kernel_size=2,
        stride=3,
        dilation_rate=1,
        padding='semicausal_full',
        expected=[[True, True, True]],
    )

    # kernel dilation
    check(
        mask=[[True, False, True, False, True, False, True, False]],
        kernel_size=3,
        stride=1,
        dilation_rate=2,
        padding='semicausal_full',
        expected=[[
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]],
    )

if __name__ == '__main__':
  test_utils.main()
