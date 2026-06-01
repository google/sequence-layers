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
"""Behavior tests for pooling layers.

Backend-specific test files should inherit from these tests.
"""

# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation
from absl.testing import parameterized
import numpy as np

from sequence_layers.specs import test_utils


class Pooling1DTest(test_utils.SequenceLayerTest):
  """Test behavior of 1D pooling layers."""

  def test_defaults(self):
    self.assertConfigDefaults(
        self.sl.MaxPooling1D.Config,
        {
            'strides': 1,
            'dilation_rate': 1,
            'padding': 'valid',
            'name': None,
        },
        pool_size=3,
    )
    self.assertConfigDefaults(
        self.sl.MinPooling1D.Config,
        {
            'strides': 1,
            'dilation_rate': 1,
            'padding': 'valid',
            'name': None,
        },
        pool_size=3,
    )
    self.assertConfigDefaults(
        self.sl.AveragePooling1D.Config,
        {
            'strides': 1,
            'dilation_rate': 1,
            'padding': 'valid',
            'masked_average': False,
            'name': None,
        },
        pool_size=3,
    )

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
        self.xp.float32,
        **kwargs,
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      dtype_name=[
          'FLOAT32',
          'INT32',
      ],
  )
  def test_dtypes(self, pool_type_kwargs, dtype_name):
    pool_type, kwargs = pool_type_kwargs
    dtype = getattr(self.xp, dtype_name.lower())
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
        self.xp.float32,
        **kwargs,
    )

  @parameterized.product(
      masked_average=[True, False],
  )
  def test_masked_average(self, masked_average):
    pool_size, stride, dilation_rate = 3, 3, 1
    padding = 'reverse_causal'
    config = self.sl.AveragePooling1D.Config(
        pool_size=pool_size,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        name='pool_1d',
        masked_average=masked_average,
    )
    l = self.make_layer(config)

    x_values = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9, 0],
            [2, 3, 0, 6, 2, 1],
            [0, 6, 2, 1, 7, 8],
        ],
        dtype=np.float32,
    )

    x_mask = np.array(
        [
            [False, False, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ],
        dtype=bool,
    )

    x = self.sl.types.Sequence(
        self.xp.array(x_values),
        self.xp.array(x_mask),
    )
    l = self.init_layer(l, x)
    y = l.layer(x, training=False)

    if masked_average:
      expected_y_values = np.array(
          [
              [0.0, 0.0],
              [(3 + 4 + 5) / 3.0, 0],
              [(5 + 6 + 7) / 3.0, 8],
              [(2 + 3 + 0) / 3.0, (6 + 2) / 2.0],
              [(0 + 6 + 2) / 3.0, (1 + 7 + 8) / 3.0],
          ],
          dtype=np.float32,
      )
    else:
      expected_y_values = np.array(
          [
              [0.0, 0.0],
              [(3 + 4 + 5) / 3.0, 0],
              [(5 + 6 + 7) / 3.0, 8 / 3.0],
              [(2 + 3 + 0) / 3.0, (6 + 2) / 3.0],
              [(0 + 6 + 2) / 3.0, (1 + 7 + 8) / 3.0],
          ],
          dtype=np.float32,
      )

    expected_y_mask = np.array(
        [
            [False, False],
            [True, False],
            [True, True],
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )

    expected_y = self.sl.types.Sequence(
        self.xp.array(expected_y_values),
        self.xp.array(expected_y_mask),
    )
    self.assertSequencesClose(y, expected_y)

  def _test_pooling1d(
      self, pool_type, params, channel_shape, padding, dtype, **kwargs
  ):
    pool_size, stride, dilation_rate = params
    effective_pool_size = (pool_size - 1) * dilation_rate + 1

    match pool_type:
      case 'min':
        config = self.sl.MinPooling1D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            padding=padding,
            name='pool_1d',
            **kwargs,
        )
      case 'max':
        config = self.sl.MaxPooling1D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            padding=padding,
            name='pool_1d',
            **kwargs,
        )
      case 'average':
        config = self.sl.AveragePooling1D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            padding=padding,
            name='pool_1d',
            **kwargs,
        )
      case _:
        raise NotImplementedError()

    l = self.make_layer(config)

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

    expected_input_latency = (
        effective_pool_size - 1
        if padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size = 2
    x = self.random_sequence(batch_size, 1, *channel_shape, dtype=dtype)
    l = self.init_layer(l, x)
    self.assertEmpty(self.get_variables(l))

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.dtype, dtype)
    self.assertEqual(output_spec.shape, channel_shape)

    # Check contract compatibility on various sequence lengths.
    # JAX does not support reduce_window gradients with dilation_rate > 1.
    test_gradients = dilation_rate == 1 and self.xp.float32 == dtype
    test_receptive_field = dilation_rate == 1 and self.xp.float32 == dtype

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, *channel_shape, dtype=dtype)
      self.verify_contract(
          l,
          x,
          training=False,
          test_gradients=test_gradients,
          test_receptive_field=test_receptive_field,
      )


class Pooling2DTest(test_utils.SequenceLayerTest):
  """Test behavior of 2D pooling layers."""

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
  def test_pooling2d(self, pool_type_kwargs, params, time_padding):
    pool_type, kwargs = pool_type_kwargs
    self._test_pooling2d(
        pool_type,
        params,
        (9,),
        time_padding,
        'same',
        self.xp.float32,
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
        self.xp.float32,
        **kwargs,
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      dtype_name=[
          'FLOAT32',
          'INT32',
      ],
  )
  def test_dtypes(self, pool_type_kwargs, dtype_name):
    pool_type, kwargs = pool_type_kwargs
    dtype = getattr(self.xp, dtype_name.lower())
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
          (9, 5, 3),
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
        self.xp.float32,
        **kwargs,
    )

  @parameterized.product(
      masked_average=[True, False],
  )
  def test_masked_average(self, masked_average):
    pool_size, stride, dilation_rate = (3, 2), (3, 2), (1, 1)
    time_padding = 'reverse_causal'
    spatial_padding = 'reverse_causal'
    config = self.sl.AveragePooling2D.Config(
        pool_size=pool_size,
        strides=stride,
        dilation_rate=dilation_rate,
        time_padding=time_padding,
        spatial_padding=spatial_padding,
        name='pool_2d',
        masked_average=masked_average,
    )
    l = self.make_layer(config)

    x_values = np.array(
        [
            [[1, 2], [2, 3], [5, 6], [7, 8], [9, 3], [4, 2]],
            [[2, 3], [5, 6], [7, 8], [9, 3], [3, 1], [2, 7]],
            [[5, 2], [7, 3], [0, 3], [3, 1], [2, 6], [1, 2]],
            [[7, 3], [0, 3], [3, 1], [2, 6], [1, 2], [3, 4]],
            [[0, 3], [3, 1], [2, 6], [1, 2], [3, 4], [5, 7]],
        ],
        dtype=np.float32,
    )

    x_mask = np.array(
        [
            [False, False, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ],
        dtype=bool,
    )

    x = self.sl.types.Sequence(
        self.xp.array(x_values),
        self.xp.array(x_mask),
    )
    l = self.init_layer(l, x)
    y = l.layer(x, training=False)

    if masked_average:
      expected_y_values = np.array(
          [
              [[0.0], [0.0]],
              [[(2 + 5 + 7 + 3 + 6 + 8) / 6.0], [0]],
              [[(5 + 7 + 0 + 2 + 3 + 3) / 6.0], [(3 + 1) / 2.0]],
              [[(7 + 0 + 3 + 3 + 3 + 1) / 6.0], [(2 + 1 + 6 + 2) / 4.0]],
              [
                  [(0 + 3 + 2 + 3 + 1 + 6) / 6.0],
                  [(1 + 3 + 5 + 2 + 4 + 7) / 6.0],
              ],
          ],
          dtype=np.float32,
      )
    else:
      expected_y_values = np.array(
          [
              [[0.0], [0.0]],
              [[(2 + 5 + 7 + 3 + 6 + 8) / 6.0], [0]],
              [[(5 + 7 + 0 + 2 + 3 + 3) / 6.0], [(3 + 1) / 6.0]],
              [[(7 + 0 + 3 + 3 + 3 + 1) / 6.0], [(2 + 1 + 6 + 2) / 6.0]],
              [
                  [(0 + 3 + 2 + 3 + 1 + 6) / 6.0],
                  [(1 + 3 + 5 + 2 + 4 + 7) / 6.0],
              ],
          ],
          dtype=np.float32,
      )

    expected_y_mask = np.array(
        [
            [False, False],
            [True, False],
            [True, True],
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )

    expected_y = self.sl.types.Sequence(
        self.xp.array(expected_y_values),
        self.xp.array(expected_y_mask),
    )
    self.assertSequencesClose(y, expected_y)

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
    pool_size, stride, dilation_rate = params
    effective_pool_size = (pool_size - 1) * dilation_rate + 1

    match pool_type:
      case 'min':
        config = self.sl.MinPooling2D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=spatial_padding,
            name='pool_2d',
            **kwargs,
        )
      case 'max':
        config = self.sl.MaxPooling2D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=spatial_padding,
            name='pool_2d',
            **kwargs,
        )
      case 'average':
        config = self.sl.AveragePooling2D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=spatial_padding,
            name='pool_2d',
            **kwargs,
        )
      case _:
        raise NotImplementedError()

    l = self.make_layer(config)

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

    expected_input_latency = (
        effective_pool_size - 1
        if time_padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size = 2
    x = self.random_sequence(batch_size, 1, *channel_shape, dtype=dtype)
    l = self.init_layer(l, x)
    self.assertEmpty(self.get_variables(l))

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.dtype, dtype)

    # Verify verification contract
    test_gradients = dilation_rate == 1 and self.xp.float32 == dtype
    test_receptive_field = dilation_rate == 1 and self.xp.float32 == dtype

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, *channel_shape, dtype=dtype)
      self.verify_contract(
          l,
          x,
          training=False,
          test_gradients=test_gradients,
          test_receptive_field=test_receptive_field,
      )


class Pooling3DTest(test_utils.SequenceLayerTest):
  """Test behavior of 3D pooling layers."""

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
  def test_pooling3d(self, pool_type_kwargs, params, time_padding):
    pool_type, kwargs = pool_type_kwargs
    self._test_pooling3d(
        pool_type,
        params,
        (9, 9),
        time_padding,
        'same',
        self.xp.float32,
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
    return self._test_pooling3d(
        pool_type,
        (3, 2, 1),
        (9, 9),
        'reverse_causal',
        spatial_padding,
        self.xp.float32,
        **kwargs,
    )

  @parameterized.product(
      pool_type_kwargs=(
          ('min', {}),
          ('max', {}),
          ('average', {'masked_average': False}),
          ('average', {'masked_average': True}),
      ),
      dtype_name=[
          'FLOAT32',
          'INT32',
      ],
  )
  def test_dtypes(self, pool_type_kwargs, dtype_name):
    pool_type, kwargs = pool_type_kwargs
    dtype = getattr(self.xp, dtype_name.lower())
    return self._test_pooling3d(
        pool_type,
        (3, 2, 1),
        (9, 9),
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
          (9, 9),
          (9, 9, 5),
      ),
  )
  def test_channel_shapes(self, pool_type_kwargs, channel_shape):
    pool_type, kwargs = pool_type_kwargs
    return self._test_pooling3d(
        pool_type,
        (3, 2, 1),
        channel_shape,
        'reverse_causal',
        'reverse_causal',
        self.xp.float32,
        **kwargs,
    )

  def _test_pooling3d(
      self,
      pool_type,
      params,
      channel_shape,
      time_padding,
      spatial_padding,
      dtype,
      **kwargs,
  ):
    pool_size, stride, dilation_rate = params
    effective_pool_size = (pool_size - 1) * dilation_rate + 1

    match pool_type:
      case 'min':
        config = self.sl.MinPooling3D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=(spatial_padding, spatial_padding),
            name='pool_3d',
            **kwargs,
        )
      case 'max':
        config = self.sl.MaxPooling3D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=(spatial_padding, spatial_padding),
            name='pool_3d',
            **kwargs,
        )
      case 'average':
        config = self.sl.AveragePooling3D.Config(
            pool_size=pool_size,
            strides=stride,
            dilation_rate=dilation_rate,
            time_padding=time_padding,
            spatial_padding=(spatial_padding, spatial_padding),
            name='pool_3d',
            **kwargs,
        )
      case _:
        raise NotImplementedError()

    l = self.make_layer(config)

    self.assertEqual(l.block_size, stride)
    self.assertEqual(1 / l.output_ratio, stride)
    self.assertEqual(l.name, 'pool_3d')
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

    expected_input_latency = (
        effective_pool_size - 1
        if time_padding in ('reverse_causal_valid', 'reverse_causal')
        else 0
    )
    self.assertEqual(l.input_latency, expected_input_latency)
    self.assertEqual(l.output_latency, expected_input_latency // stride)

    batch_size = 2
    x = self.random_sequence(batch_size, 1, *channel_shape, dtype=dtype)
    l = self.init_layer(l, x)
    self.assertEmpty(self.get_variables(l))

    output_spec = l.get_output_spec(x.channel_spec)
    self.assertEqual(output_spec.dtype, dtype)

    # Verify verification contract
    test_gradients = dilation_rate == 1 and self.xp.float32 == dtype
    test_receptive_field = dilation_rate == 1 and self.xp.float32 == dtype

    for time in range(20 * l.block_size - 1, 20 * l.block_size + 2):
      x = self.random_sequence(batch_size, time, *channel_shape, dtype=dtype)
      self.verify_contract(
          l,
          x,
          training=False,
          test_gradients=test_gradients,
          test_receptive_field=test_receptive_field,
      )
