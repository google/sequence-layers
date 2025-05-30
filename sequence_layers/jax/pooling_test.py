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
"""Tests for pooling layers."""

from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import convolution
from sequence_layers.jax import pooling
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils


class Pooling1DTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      params=[
          # 1x1 conv.
          (1, 1, 1),
          # even pool_size with smaller, equal and larger strides.
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          # odd pool_size with smaller, equal and larger strides.
          (3, 2, 1),
          (3, 3, 1),
          (3, 4, 1),
          # pool_size smaller, equal and larger than even dilation_rate.
          (1, 1, 2),
          (2, 1, 2),
          (3, 1, 2),
          # pool_size smaller, equal and larger than odd dilation_rate.
          (1, 1, 3),
          (2, 1, 3),
          (3, 1, 3),
      ],
      padding=[
          'same',
          'valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
  )
  def test_pooling1d(self, pool_type_kwargs, params, padding):
    pool_type, kwargs = pool_type_kwargs
    return self._test_pooling1d(
        pool_type,
        params,
        (3,),
        padding,
        jnp.float32,
        **kwargs,
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      dtype=(jnp.float32, jnp.int32),
  )
  def test_dtypes(self, pool_type_kwargs, dtype):
    pool_type, kwargs = pool_type_kwargs
    return self._test_pooling1d(
        pool_type, (3, 2, 1), (3,), 'reverse_causal', dtype, **kwargs
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      channel_shape=(
          (),
          (3,),
          (3, 5),
      ),
  )
  def test_channel_shapes(self, pool_type_kwargs, channel_shape):
    pool_type, kwargs = pool_type_kwargs
    return self._test_pooling1d(
        pool_type,
        (3, 2, 1),
        channel_shape,
        'reverse_causal',
        jnp.float32,
        **kwargs,
    )

  @parameterized.product(
      masked_average=[True, False],
  )
  def test_masked_average(self, masked_average):
    key = jax.random.PRNGKey(1234)
    pool_size, stride, dilation_rate = 3, 3, 1
    padding = 'reverse_causal'
    l = pooling.AveragePooling1D.Config(
        pool_size=pool_size,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        name='pool_1d',
        masked_average=masked_average,
    ).make()

    x = types.Sequence(
        jnp.array([
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9, 0],
            [2, 3, 0, 6, 2, 1],
            [0, 6, 2, 1, 7, 8],
        ]).astype(jnp.float32),
        jnp.array([
            [False, False, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]),
    )
    l = self.init_and_bind_layer(key, l, x)
    y = l(x, training=False)
    if masked_average:
      expected_y_values = jnp.array([
          [0.0, 0.0],
          [(3 + 4 + 5) / 3.0, 0],
          [(5 + 6 + 7) / 3.0, 8],
          [(2 + 3 + 0) / 3.0, (6 + 2) / 2.0],
          [(0 + 6 + 2) / 3.0, (1 + 7 + 8) / 3.0],
      ])
    else:
      expected_y_values = jnp.array([
          [0.0, 0.0],
          [(3 + 4 + 5) / 3.0, 0],
          [(5 + 6 + 7) / 3.0, 8 / 3.0],
          [(2 + 3 + 0) / 3.0, (6 + 2) / 3.0],
          [(0 + 6 + 2) / 3.0, (1 + 7 + 8) / 3.0],
      ])
    expected_y_mask = jnp.array([
        [False, False],
        [True, False],
        [True, True],
        [True, True],
        [True, True],
    ])
    expected_y = types.Sequence(expected_y_values, expected_y_mask)
    self.assertSequencesEqual(y, expected_y)

  def _test_pooling1d(
      self, pool_type, params, channel_shape, padding, dtype, **kwargs
  ):
    key = jax.random.PRNGKey(1234)
    pool_size, stride, dilation_rate = params
    explicit_padding = utils.convolution_explicit_padding(
        padding, pool_size, stride, dilation_rate
    )
    match pool_type:
      case 'min':
        l = pooling.MinPooling1D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            padding=padding,
            name='pool_1d',
            **kwargs,
        ).make()
        pad_value = np.inf
        golden_fn = lambda x: nn.pooling.min_pool(
            x.values,
            window_shape=(pool_size,),
            strides=(stride,),
            padding=(explicit_padding,),
        )
      case 'max':
        l = pooling.MaxPooling1D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            padding=padding,
            name='pool_1d',
            **kwargs,
        ).make()
        pad_value = -np.inf
        golden_fn = lambda x: nn.pooling.max_pool(
            x.values,
            window_shape=(pool_size,),
            strides=(stride,),
            padding=(explicit_padding,),
        )
      case 'average':
        l = pooling.AveragePooling1D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            padding=padding,
            name='pool_1d',
            **kwargs,
        ).make()
        pad_value = 0
        if kwargs.get('masked_average', False):
          golden_fn = lambda x: (
              nn.pooling.avg_pool(
                  x.values,
                  window_shape=(pool_size,),
                  strides=(stride,),
                  padding=(explicit_padding,),
              )
              / nn.pooling.avg_pool(
                  x.mask.astype(x.values.dtype)[..., jnp.newaxis],
                  window_shape=(pool_size,),
                  strides=(stride,),
                  padding=(explicit_padding,),
              )
          )
        else:
          golden_fn = lambda x: nn.pooling.avg_pool(
              x.values,
              window_shape=(pool_size,),
              strides=(stride,),
              padding=(explicit_padding,),
          )
      case _:
        raise NotImplementedError()
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)
    self.assertEqual(l.name, 'pool_1d')
    self.assertEqual(
        l.supports_step,
        padding
        in (
            'reverse_causal_valid',
            'causal',
            'reverse_causal',
            'semicausal',
        ),
    )

    effective_pool_size = utils.convolution_effective_kernel_size(
        pool_size, dilation_rate
    )
    expected_input_latency = (
        effective_pool_size - 1
        if padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size = 2

    x = test_utils.random_sequence(batch_size, 1, *channel_shape, dtype=dtype)
    l = self.init_and_bind_layer(key, l, x)
    self.assertEmpty(l.variables)

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.dtype, dtype)
    self.assertEqual(output_spec.shape, channel_shape)

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = test_utils.random_sequence(
          batch_size, time, *channel_shape, dtype=dtype
      )
      y = self.verify_contract(
          l,
          x,
          training=False,
          # JAX does not support reduce_window gradients with dilation_rate > 1.
          # Don't compute gradients for integer types.
          test_gradients=dilation_rate == 1 and dtype == jnp.float32,
      )

      # Only test for flax compatibility on inputs that flax supports.
      if (
          len(channel_shape) == 1
          and dtype == jnp.float32
          and dilation_rate == 1
      ):

        # Flax does not have a concept of padding, so replace padded regions
        # with the pad value manually.
        x = x.mask_invalid(pad_value)

        values_golden = golden_fn(x)
        mask_golden = convolution.compute_conv_mask(
            x.mask,
            pool_size,
            stride,
            dilation_rate,
            padding,
            is_step=False,
        )

        # Apply masking.
        values_golden = (
            types.Sequence(values_golden, mask_golden).mask_invalid().values
        )

        chex.assert_trees_all_close(y.values, values_golden, atol=1e-5)
        chex.assert_trees_all_equal(y.mask, mask_golden)


class Pooling2DTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      params=[
          # 1x1 conv.
          (1, 1, 1),
          # even pool_size with smaller, equal and larger strides.
          (2, 1, 1),
          (2, 2, 1),
          (2, 3, 1),
          # odd pool_size with smaller, equal and larger strides.
          (3, 2, 1),
          (3, 3, 1),
          (3, 4, 1),
          # pool_size smaller, equal and larger than even dilation_rate.
          (1, 1, 2),
          (2, 1, 2),
          (3, 1, 2),
          # pool_size smaller, equal and larger than odd dilation_rate.
          (1, 1, 3),
          (2, 1, 3),
          (3, 1, 3),
      ],
      time_padding=[
          'same',
          'valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
  )
  def test_pooling2d(
      self,
      pool_type_kwargs,
      params,
      time_padding,
  ):
    pool_type, kwargs = pool_type_kwargs
    self._test_pooling2d(
        pool_type,
        params,
        (9,),
        time_padding,
        'same',
        jnp.float32,
        **kwargs,
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      spatial_padding=[
          'same',
          'valid',
          'reverse_causal_valid',
          'causal',
          'reverse_causal',
          'semicausal',
      ],
  )
  def test_spatial_padding(self, pool_type_kwargs, spatial_padding):
    pool_type, kwargs = pool_type_kwargs
    return self._test_pooling2d(
        pool_type,
        (3, 2, 1),
        (9,),
        'reverse_causal',
        spatial_padding,
        jnp.float32,
        **kwargs,
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      dtype=(jnp.float32, jnp.int32),
  )
  def test_dtypes(self, pool_type_kwargs, dtype):
    jax.config.update('jax_traceback_filtering', 'off')
    pool_type, kwargs = pool_type_kwargs
    return self._test_pooling2d(
        pool_type,
        (3, 2, 1),
        (9,),
        'reverse_causal',
        'reverse_causal',
        dtype,
        **kwargs,
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      channel_shape=(
          (9,),
          (9, 5),
          (9, 5, 7),
      ),
  )
  def test_channel_shapes(self, pool_type_kwargs, channel_shape):
    pool_type, kwargs = pool_type_kwargs
    return self._test_pooling2d(
        pool_type,
        (3, 2, 1),
        channel_shape,
        'reverse_causal',
        'reverse_causal',
        jnp.float32,
        **kwargs,
    )

  @parameterized.product(
      masked_average=[True, False],
  )
  def test_masked_average(self, masked_average):
    key = jax.random.PRNGKey(1234)
    pool_size, stride, dilation_rate = (3, 2), (3, 2), (1, 1)
    time_padding = 'reverse_causal'
    spatial_padding = 'reverse_causal'
    l = pooling.AveragePooling2D.Config(
        pool_size=pool_size,
        strides=stride,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        name='pool_2d',
        masked_average=masked_average,
    ).make()

    x = types.Sequence(
        jnp.array([
            [[1, 2], [2, 3], [5, 6], [7, 8], [9, 3], [4, 2]],
            [[2, 3], [5, 6], [7, 8], [9, 3], [3, 1], [2, 7]],
            [[5, 2], [7, 3], [0, 3], [3, 1], [2, 6], [1, 2]],
            [[7, 3], [0, 3], [3, 1], [2, 6], [1, 2], [3, 4]],
            [[0, 3], [3, 1], [2, 6], [1, 2], [3, 4], [5, 7]],
        ]).astype(jnp.float32),
        jnp.array([
            [False, False, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]),
    )
    l = self.init_and_bind_layer(key, l, x)
    y = l(x, training=False)
    if masked_average:
      expected_y_values = jnp.array([
          [[0.0], [0.0]],
          [[(2 + 5 + 7 + 3 + 6 + 8) / 6.0], [0]],
          [[(5 + 7 + 0 + 2 + 3 + 3) / 6.0], [(3 + 1) / 2.0]],
          [[(7 + 0 + 3 + 3 + 3 + 1) / 6.0], [(2 + 1 + 6 + 2) / 4.0]],
          [[(0 + 3 + 2 + 3 + 1 + 6) / 6.0], [(1 + 3 + 5 + 2 + 4 + 7) / 6.0]],
      ])
    else:
      expected_y_values = jnp.array([
          [[0.0], [0.0]],
          [[(2 + 5 + 7 + 3 + 6 + 8) / 6.0], [0]],
          [[(5 + 7 + 0 + 2 + 3 + 3) / 6.0], [(3 + 1) / 6.0]],
          [[(7 + 0 + 3 + 3 + 3 + 1) / 6.0], [(2 + 1 + 6 + 2) / 6.0]],
          [[(0 + 3 + 2 + 3 + 1 + 6) / 6.0], [(1 + 3 + 5 + 2 + 4 + 7) / 6.0]],
      ])
    expected_y_mask = jnp.array([
        [False, False],
        [True, False],
        [True, True],
        [True, True],
        [True, True],
    ])
    expected_y = types.Sequence(expected_y_values, expected_y_mask)
    self.assertSequencesEqual(y, expected_y)

  def _test_pooling2d(
      self,
      pool_type,
      params,
      channel_shape,
      time_padding,
      spatial_padding,
      dtype,
      **kwargs,
  ):
    key = jax.random.PRNGKey(1234)
    pool_size, stride, dilation_rate = params
    explicit_time_padding = utils.convolution_explicit_padding(
        time_padding, pool_size, stride, dilation_rate
    )
    explicit_spatial_padding = utils.convolution_explicit_padding(
        spatial_padding, pool_size, stride, dilation_rate
    )
    match pool_type:
      case 'min':
        l = pooling.MinPooling2D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=spatial_padding,
            name='pool_2d',
            **kwargs,
        ).make()
        pad_value = np.inf
        golden_fn = lambda x: nn.pooling.min_pool(
            x.values,
            window_shape=(pool_size, pool_size),
            strides=(stride, stride),
            padding=(explicit_time_padding, explicit_spatial_padding),
        )
      case 'max':
        l = pooling.MaxPooling2D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=spatial_padding,
            name='pool_2d',
            **kwargs,
        ).make()
        pad_value = -np.inf
        golden_fn = lambda x: nn.pooling.max_pool(
            x.values,
            window_shape=(pool_size, pool_size),
            strides=(stride, stride),
            padding=(explicit_time_padding, explicit_spatial_padding),
        )
      case 'average':
        l = pooling.AveragePooling2D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=spatial_padding,
            name='pool_2d',
            **kwargs,
        ).make()
        pad_value = 0
        if kwargs.get('masked_average', False):
          golden_fn = lambda x: (
              nn.pooling.avg_pool(
                  x.values,
                  window_shape=(pool_size, pool_size),
                  strides=(stride, stride),
                  padding=(explicit_time_padding, explicit_spatial_padding),
              )
              / jnp.expand_dims(
                  nn.pooling.avg_pool(
                      x.mask.astype(x.values.dtype)[..., jnp.newaxis],
                      window_shape=(pool_size,),
                      strides=(stride,),
                      padding=(explicit_time_padding,),
                  ),
                  range(3, x.values.ndim),
              )
          )
        else:
          golden_fn = lambda x: nn.pooling.avg_pool(
              x.values,
              window_shape=(pool_size, pool_size),
              strides=(stride, stride),
              padding=(explicit_time_padding, explicit_spatial_padding),
          )
      case _:
        raise NotImplementedError()
    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)
    self.assertEqual(l.name, 'pool_2d')
    self.assertEqual(
        l.supports_step,
        time_padding
        in (
            'reverse_causal_valid',
            'causal',
            'reverse_causal',
            'semicausal',
        ),
    )

    effective_pool_size = utils.convolution_effective_kernel_size(
        pool_size, dilation_rate
    )
    expected_input_latency = (
        effective_pool_size - 1
        if time_padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size = 2

    x = test_utils.random_sequence(batch_size, 1, *channel_shape, dtype=dtype)
    l = self.init_and_bind_layer(key, l, x)
    self.assertEmpty(l.variables)

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.dtype, dtype)

    spatial_output = utils.convolution_padding_output_size(
        x.channel_shape[0], spatial_padding, pool_size, stride, dilation_rate
    )
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        (spatial_output,) + x.channel_shape[1:],
    )

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = test_utils.random_sequence(
          batch_size, time, *channel_shape, dtype=dtype
      )
      y = self.verify_contract(
          l,
          x,
          training=False,
          # JAX does not support reduce_window gradients with dilation_rate > 1.
          # Don't compute gradients for integer types.
          test_gradients=dilation_rate == 1 and dtype == jnp.float32,
      )

      # Only test for flax compatibility on inputs that flax supports.
      if (
          len(channel_shape) == 2
          and dtype == jnp.float32
          and dilation_rate == 1
      ):

        # Flax does not have a concept of padding, so replace padded regions
        # with the pad value manually.
        x = x.mask_invalid(pad_value)

        values_golden = golden_fn(x)

        mask_golden = convolution.compute_conv_mask(
            x.mask,
            pool_size,
            stride,
            dilation_rate,
            time_padding,
            is_step=False,
        )

        # Apply masking.
        values_golden = (
            types.Sequence(values_golden, mask_golden).mask_invalid().values
        )

        chex.assert_trees_all_close(y.values, values_golden, atol=1e-5)
        chex.assert_trees_all_equal(y.mask, mask_golden)


if __name__ == '__main__':
  test_utils.main()
