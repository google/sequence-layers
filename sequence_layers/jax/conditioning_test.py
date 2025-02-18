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
"""Conditioning tests."""

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import conditioning
from sequence_layers.jax import test_utils
from sequence_layers.jax import types

from google3.testing.pybase import parameterized


IDENTITY = conditioning.Conditioning.Projection.IDENTITY
LINEAR = conditioning.Conditioning.Projection.LINEAR
LINEAR_AFFINE = conditioning.Conditioning.Projection.LINEAR_AFFINE
ADD = conditioning.Conditioning.Combination.ADD
CONCAT = conditioning.Conditioning.Combination.CONCAT
AFFINE = conditioning.Conditioning.Combination.AFFINE


def _float_tensor(values):
  return jnp.asarray(values, dtype=jnp.float32)


class ConditioningTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (IDENTITY, ADD, tuple(), tuple(), tuple()),
      (IDENTITY, ADD, tuple(), (5,), (5,)),
      (IDENTITY, ADD, tuple(), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2,), tuple(), (2,)),
      (IDENTITY, ADD, (2, 5), tuple(), (2, 5)),
      (IDENTITY, ADD, (5,), (5,), (5,)),
      (IDENTITY, ADD, (5,), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2, 5), (5,), (2, 5)),
      (IDENTITY, ADD, (3, 1, 5), (2, 5), (3, 2, 5)),
      (IDENTITY, ADD, (2, 5), (3, 1, 5), (3, 2, 5)),
      (IDENTITY, CONCAT, tuple(), tuple(), (2,)),
      (IDENTITY, CONCAT, tuple(), (5,), (6,)),
      (IDENTITY, CONCAT, tuple(), (2, 5), (2, 6)),
      (IDENTITY, CONCAT, (2,), tuple(), (3,)),
      (IDENTITY, CONCAT, (2, 5), tuple(), (2, 6)),
      (IDENTITY, CONCAT, (5,), (7,), (12,)),
      (IDENTITY, CONCAT, (5,), (2, 7), (2, 12)),
      (IDENTITY, CONCAT, (2, 5), (7,), (2, 12)),
      (IDENTITY, CONCAT, (3, 1, 5), (2, 7), (3, 2, 12)),
      (IDENTITY, CONCAT, (2, 5), (3, 1, 7), (3, 2, 12)),
      (LINEAR, ADD, tuple(), tuple(), tuple()),
      (LINEAR, ADD, tuple(), (5,), tuple()),
      (LINEAR, ADD, tuple(), (2, 5), tuple()),
      (LINEAR, ADD, (2,), tuple(), (2,)),
      (LINEAR, ADD, (2, 5), tuple(), (2, 5)),
      (LINEAR, ADD, (5,), (7,), (5,)),
      (LINEAR, ADD, (7,), (2, 5), (7,)),
      (LINEAR, ADD, (2, 5), (7,), (2, 5)),
      (LINEAR, ADD, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR, ADD, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR_AFFINE, AFFINE, tuple(), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (5,), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (2, 5), tuple()),
      (LINEAR_AFFINE, AFFINE, (2,), tuple(), (2,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), tuple(), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (5,)),
      (LINEAR_AFFINE, AFFINE, (7,), (2, 5), (7,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), (7,), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR_AFFINE, AFFINE, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR, CONCAT, tuple(), tuple(), (2,)),
      (LINEAR, CONCAT, tuple(), (5,), (2,)),
      (LINEAR, CONCAT, tuple(), (2, 5), (2,)),
      (LINEAR, CONCAT, (2,), tuple(), (4,)),
      (LINEAR, CONCAT, (2, 5), tuple(), (2, 10)),
      (LINEAR, CONCAT, (5,), (7,), (10,)),
      (LINEAR, CONCAT, (7,), (2, 5), (14,)),
      (LINEAR, CONCAT, (2, 5), (7,), (2, 10)),
      (LINEAR, CONCAT, (3, 1, 5), (2, 7), (3, 1, 10)),
      (LINEAR, CONCAT, (2, 7), (3, 1, 5), (2, 14)),
  )
  def test_conditioning_sequence(
      self,
      projection,
      combination,
      x_channel_shape,
      c_channel_shape,
      expected_channel_shape,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 16
    x = test_utils.random_sequence(
        batch_size, time, *x_channel_shape, low_length=8
    )
    c = test_utils.random_sequence(
        batch_size, time, *c_channel_shape, low_length=8
    )
    l = conditioning.Conditioning.Config(
        'test', projection, combination, name='conditioning'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'conditioning')
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        expected_channel_shape,
    )
    self.verify_contract(
        l, x, training=False, pad_constants=True, constants=constants
    )

    # If either channel shape is scalar, use (1,).
    param_c_shape = c_channel_shape if c_channel_shape else (1,)
    if projection == conditioning.Conditioning.Projection.LINEAR_AFFINE:
      param_x_shape = (2,) + x_channel_shape
    else:
      param_x_shape = x_channel_shape if x_channel_shape else (1,)

    match projection:
      case conditioning.Conditioning.Projection.IDENTITY:
        expected_params = {}
      case (
          conditioning.Conditioning.Projection.LINEAR
          | conditioning.Conditioning.Projection.LINEAR_AFFINE
      ):
        expected_params = {
            'params': {
                'dense': {
                    'kernel': jnp.zeros(param_c_shape + param_x_shape),
                    'bias': jnp.zeros(param_x_shape),
                }
            }
        }
      case _:
        raise NotImplementedError(f'Unsupported projection: {projection}')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables), expected_params
    )

  @parameterized.parameters(
      (IDENTITY, CONCAT, (2, 5), tuple(), (2, 6)),
      (LINEAR, ADD, (5,), (7,), (5,)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (5,)),
  )
  def test_conditioning_sequence_streaming(
      self,
      projection,
      combination,
      x_channel_shape,
      c_channel_shape,
      expected_channel_shape,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 16
    x = test_utils.random_sequence(
        batch_size, time, *x_channel_shape, low_length=8
    )
    c = test_utils.random_sequence(
        batch_size, time, *c_channel_shape, low_length=8
    )
    l = conditioning.Conditioning.Config(
        'test', projection, combination, streaming=True, name='conditioning'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'conditioning')
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        expected_channel_shape,
    )
    self.verify_contract(
        l,
        x,
        training=False,
        stream_constants=True,
        pad_constants=True,
        constants=constants,
    )

    # If either channel shape is scalar, use (1,).
    param_c_shape = c_channel_shape if c_channel_shape else (1,)
    if projection == conditioning.Conditioning.Projection.LINEAR_AFFINE:
      param_x_shape = (2,) + x_channel_shape
    else:
      param_x_shape = x_channel_shape if x_channel_shape else (1,)

    match projection:
      case conditioning.Conditioning.Projection.IDENTITY:
        expected_params = {}
      case (
          conditioning.Conditioning.Projection.LINEAR
          | conditioning.Conditioning.Projection.LINEAR_AFFINE
      ):
        expected_params = {
            'params': {
                'dense': {
                    'kernel': jnp.zeros(param_c_shape + param_x_shape),
                    'bias': jnp.zeros(param_x_shape),
                }
            }
        }
      case _:
        raise NotImplementedError(f'Unsupported projection: {projection}')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables), expected_params
    )

  @parameterized.parameters(
      (LINEAR, ADD, tuple(), tuple(), tuple(), tuple()),
      (LINEAR, ADD, tuple(), (5,), tuple(), tuple()),
      (LINEAR, ADD, tuple(), (2, 5), tuple(), tuple()),
      (LINEAR, ADD, (2,), tuple(), (1,), (2,)),
      (LINEAR, ADD, (2, 5), tuple(), (5,), (2, 5)),
      (LINEAR, ADD, (5,), (7,), (1,), (5,)),
      (LINEAR, ADD, (7,), (2, 5), (1,), (7,)),
      (LINEAR, ADD, (2, 5), (7,), (2, 1), (2, 5)),
      (LINEAR, ADD, (3, 1, 5), (2, 7), (1, 1, 1), (3, 1, 5)),
      (LINEAR, ADD, (2, 7), (3, 1, 5), (1, 1), (2, 7)),
      (LINEAR_AFFINE, AFFINE, tuple(), tuple(), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (5,), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (2, 5), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, (2,), tuple(), (1,), (2,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), tuple(), (1, 1), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (1,), (5,)),
      (LINEAR_AFFINE, AFFINE, (7,), (2, 5), (1,), (7,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), (7,), (1, 1), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (3, 1, 5), (2, 7), (1, 1, 1), (3, 1, 5)),
      (LINEAR_AFFINE, AFFINE, (2, 7), (3, 1, 5), (1, 1), (2, 7)),
      (LINEAR, CONCAT, tuple(), tuple(), (2,), (3,)),
      (LINEAR, CONCAT, tuple(), (5,), (2,), (3,)),
      (LINEAR, CONCAT, tuple(), (2, 5), (2,), (3,)),
      (LINEAR, CONCAT, (2,), tuple(), (1,), (3,)),
      (LINEAR, CONCAT, (2, 5), tuple(), (2, 1), (2, 6)),
      (LINEAR, CONCAT, (5,), (7,), (1,), (6,)),
      (LINEAR, CONCAT, (7,), (2, 5), (1,), (8,)),
      (LINEAR, CONCAT, (2, 5), (7,), (2, 1), (2, 6)),
      (LINEAR, CONCAT, (3, 1, 5), (2, 7), (3, 1, 1), (3, 1, 6)),
      (LINEAR, CONCAT, (2, 7), (3, 1, 5), (2, 1), (2, 8)),
  )
  def test_projection_channel_shape(
      self,
      projection,
      combination,
      x_channel_shape,
      c_channel_shape,
      projection_channel_shape,
      expected_channel_shape,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 16
    x = test_utils.random_sequence(
        batch_size, time, *x_channel_shape, low_length=8
    )
    c = test_utils.random_sequence(
        batch_size, time, *c_channel_shape, low_length=8
    )
    l = conditioning.Conditioning.Config(
        'test',
        projection,
        combination,
        name='conditioning',
        projection_channel_shape=projection_channel_shape,
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'conditioning')
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        expected_channel_shape,
    )
    self.verify_contract(
        l,
        x,
        training=False,
        pad_constants=True,
        constants=constants,
        atol=1e-5,
    )

    # If either channel shape is scalar, use (1,).
    param_c_shape = c_channel_shape if c_channel_shape else (1,)
    if projection == conditioning.Conditioning.Projection.LINEAR_AFFINE:
      param_x_shape = (2,) + projection_channel_shape
    else:
      param_x_shape = (
          projection_channel_shape if projection_channel_shape else (1,)
      )

    match projection:
      case conditioning.Conditioning.Projection.IDENTITY:
        expected_params = {}
      case (
          conditioning.Conditioning.Projection.LINEAR
          | conditioning.Conditioning.Projection.LINEAR_AFFINE
      ):
        expected_params = {
            'params': {
                'dense': {
                    'kernel': jnp.zeros(param_c_shape + param_x_shape),
                    'bias': jnp.zeros(param_x_shape),
                }
            }
        }
      case _:
        raise NotImplementedError(f'Unsupported projection: {projection}')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables), expected_params
    )

  @parameterized.parameters(
      (IDENTITY, ADD, tuple(), tuple(), tuple()),
      (IDENTITY, ADD, tuple(), (5,), (5,)),
      (IDENTITY, ADD, tuple(), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2,), tuple(), (2,)),
      (IDENTITY, ADD, (2, 5), tuple(), (2, 5)),
      (IDENTITY, ADD, (5,), (5,), (5,)),
      (IDENTITY, ADD, (5,), (2, 5), (2, 5)),
      (IDENTITY, ADD, (2, 5), (5,), (2, 5)),
      (IDENTITY, ADD, (3, 1, 5), (2, 5), (3, 2, 5)),
      (IDENTITY, ADD, (2, 5), (3, 1, 5), (3, 2, 5)),
      (IDENTITY, CONCAT, tuple(), tuple(), (2,)),
      (IDENTITY, CONCAT, tuple(), (5,), (6,)),
      (IDENTITY, CONCAT, tuple(), (2, 5), (2, 6)),
      (IDENTITY, CONCAT, (2,), tuple(), (3,)),
      (IDENTITY, CONCAT, (2, 5), tuple(), (2, 6)),
      (IDENTITY, CONCAT, (5,), (7,), (12,)),
      (IDENTITY, CONCAT, (5,), (2, 7), (2, 12)),
      (IDENTITY, CONCAT, (2, 5), (7,), (2, 12)),
      (IDENTITY, CONCAT, (3, 1, 5), (2, 7), (3, 2, 12)),
      (IDENTITY, CONCAT, (2, 5), (3, 1, 7), (3, 2, 12)),
      (LINEAR, ADD, tuple(), tuple(), tuple()),
      (LINEAR, ADD, tuple(), (5,), tuple()),
      (LINEAR, ADD, tuple(), (2, 5), tuple()),
      (LINEAR, ADD, (2,), tuple(), (2,)),
      (LINEAR, ADD, (2, 5), tuple(), (2, 5)),
      (LINEAR, ADD, (5,), (7,), (5,)),
      (LINEAR, ADD, (7,), (2, 5), (7,)),
      (LINEAR, ADD, (2, 5), (7,), (2, 5)),
      (LINEAR, ADD, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR, ADD, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR_AFFINE, AFFINE, tuple(), tuple(), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (5,), tuple()),
      (LINEAR_AFFINE, AFFINE, tuple(), (2, 5), tuple()),
      (LINEAR_AFFINE, AFFINE, (2,), tuple(), (2,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), tuple(), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (5,), (7,), (5,)),
      (LINEAR_AFFINE, AFFINE, (7,), (2, 5), (7,)),
      (LINEAR_AFFINE, AFFINE, (2, 5), (7,), (2, 5)),
      (LINEAR_AFFINE, AFFINE, (3, 1, 5), (2, 7), (3, 1, 5)),
      (LINEAR_AFFINE, AFFINE, (2, 7), (3, 1, 5), (2, 7)),
      (LINEAR, CONCAT, tuple(), tuple(), (2,)),
      (LINEAR, CONCAT, tuple(), (5,), (2,)),
      (LINEAR, CONCAT, tuple(), (2, 5), (2,)),
      (LINEAR, CONCAT, (2,), tuple(), (4,)),
      (LINEAR, CONCAT, (2, 5), tuple(), (2, 10)),
      (LINEAR, CONCAT, (5,), (7,), (10,)),
      (LINEAR, CONCAT, (7,), (2, 5), (14,)),
      (LINEAR, CONCAT, (2, 5), (7,), (2, 10)),
      (LINEAR, CONCAT, (3, 1, 5), (2, 7), (3, 1, 10)),
      (LINEAR, CONCAT, (2, 7), (3, 1, 5), (2, 14)),
  )
  def test_conditioning_tensor(
      self,
      projection,
      combination,
      x_channel_shape,
      c_channel_shape,
      expected_channel_shape,
  ):
    key = jax.random.PRNGKey(1234)
    batch_size, time = 2, 16
    x = test_utils.random_sequence(
        batch_size, time, *x_channel_shape, low_length=8
    )
    c = jnp.asarray(
        np.random.normal(size=(batch_size,) + c_channel_shape).astype(
            np.float32
        )
    )
    l = conditioning.Conditioning.Config(
        'test', projection, combination, name='conditioning'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'conditioning')
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants=constants),
        expected_channel_shape,
    )
    self.verify_contract(
        l, x, training=False, pad_constants=True, constants=constants
    )

    # If either channel shape is scalar, use (1,).
    param_c_shape = c_channel_shape if c_channel_shape else (1,)
    if projection == conditioning.Conditioning.Projection.LINEAR_AFFINE:
      param_x_shape = (2,) + x_channel_shape
    else:
      param_x_shape = x_channel_shape if x_channel_shape else (1,)

    match projection:
      case conditioning.Conditioning.Projection.IDENTITY:
        expected_params = {}
      case (
          conditioning.Conditioning.Projection.LINEAR
          | conditioning.Conditioning.Projection.LINEAR_AFFINE
      ):
        expected_params = {
            'params': {
                'dense': {
                    'kernel': jnp.zeros(param_c_shape + param_x_shape),
                    'bias': jnp.zeros(param_x_shape),
                }
            }
        }
      case _:
        raise NotImplementedError(f'Unsupported projection: {projection}')

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables), expected_params
    )

  @parameterized.parameters(
      (IDENTITY, ADD, (5,), (6,)),
      (IDENTITY, ADD, (3, 4, 5), (2, 5)),
      (IDENTITY, CONCAT, (2, 5), (3, 7)),
  )
  def test_conditioning_invalid_shapes(
      self, projection, combination, x_channel_shape, c_channel_shape
  ):
    batch_size, time = 2, 16
    x = test_utils.random_sequence(
        batch_size, time, *x_channel_shape, low_length=8
    )
    c = test_utils.random_sequence(
        batch_size, time, *c_channel_shape, low_length=8
    )
    l = conditioning.Conditioning.Config(
        'test', projection, combination, name='conditioning_layer'
    ).make()
    constants = {'test': c}

    l = l.bind({})
    s0 = l.get_initial_state(
        x.shape[0], x.channel_spec, training=False, constants=constants
    )
    with self.assertRaises(ValueError):
      l.get_output_shape_for_sequence(x, constants=constants)
    with self.assertRaises((ValueError, TypeError)):
      l.layer(x, training=False, constants=constants)
    with self.assertRaises((ValueError, TypeError)):
      l.step(x, s0, training=False, constants=constants)

  def test_condition_sequence_add_combination(self):
    # [2, 3, 3]
    x = types.Sequence(
        _float_tensor([
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
            [[0, 0, 0], [2, 4, 6], [3, 5, 7]],
        ]),
        jnp.asarray([[True, True, False], [False, True, True]]),
    )

    # [2, 3, 2, 3]
    c = types.Sequence(
        _float_tensor([
            [
                [[0, 0, 0], [0, 0, 0]],
                [[-1, -2, -3], [-5, -6, -7]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            [
                [[-4, -3, -2], [-8, -7, -6]],
                [[0, 0, 0], [0, 0, 0]],
                [[-2, -4, -6], [-1, -3, -5]],
            ],
        ]),
        jnp.asarray([[False, True, False], [True, False, True]]),
    )

    # [2, 3, 2, 3]
    expected_conditioned_x = types.Sequence(
        _float_tensor([
            [
                [[0, 0, 0], [0, 0, 0]],
                [[4 - 1, 5 - 2, 6 - 3], [4 - 5, 5 - 6, 6 - 7]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
                [[3 - 2, 5 - 4, 7 - 6], [3 - 1, 5 - 3, 7 - 5]],
            ],
        ]),
        jnp.asarray([[False, True, False], [False, False, True]]),
    )

    key = jax.random.PRNGKey(1234)
    l = conditioning.Conditioning.Config(
        'test', IDENTITY, ADD, name='conditioning_layer'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    conditioned_x = l.layer(
        x, training=False, constants=constants
    ).mask_invalid()

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_condition_tensor_add_combination(self):
    # [2, 3, 3]
    x = types.Sequence(
        _float_tensor([
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
            [[0, 0, 0], [2, 4, 6], [3, 5, 7]],
        ]),
        jnp.asarray([[True, True, False], [False, True, True]]),
    )

    # [2, 2, 3]
    c = _float_tensor(
        [[[-1, -2, -3], [-5, -6, -7]], [[-4, -3, -2], [-8, -7, -6]]]
    )

    # [2, 3, 2, 3]
    expected_conditioned_x = types.Sequence(
        _float_tensor([
            [
                [[1 - 1, 2 - 2, 3 - 3], [1 - 5, 2 - 6, 3 - 7]],
                [[4 - 1, 5 - 2, 6 - 3], [4 - 5, 5 - 6, 6 - 7]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, 0]],
                [[2 - 4, 4 - 3, 6 - 2], [2 - 8, 4 - 7, 6 - 6]],
                [[3 - 4, 5 - 3, 7 - 2], [3 - 8, 5 - 7, 7 - 6]],
            ],
        ]),
        jnp.asarray([[True, True, False], [False, True, True]]),
    )

    key = jax.random.PRNGKey(1234)
    l = conditioning.Conditioning.Config(
        'test', IDENTITY, ADD, name='conditioning_layer'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    conditioned_x = l.layer(
        x, training=False, constants=constants
    ).mask_invalid()

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_condition_sequence_concat_combination(self):
    # [2, 3, 2, 4]
    x = types.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[4, 3, 2, 1], [8, 7, 6, 5]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[2, 4, 6, 8], [1, 3, 5, 7]],
            ],
        ]),
        jnp.asarray([[False, True, False], [True, False, True]]),
    )

    # [2, 3, 2]
    c = types.Sequence(
        _float_tensor([
            [[-1, -2], [-3, -4], [-5, -6]],
            [[-1, -3], [-3, -5], [-7, -9]],
        ]),
        jnp.asarray([[True, True, False], [False, True, True]]),
    )

    # [2, 3, 2, 6]
    expected_conditioned_x = types.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[1, 2, 3, 4, -3, -4], [5, 6, 7, 8, -3, -4]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[2, 4, 6, 8, -7, -9], [1, 3, 5, 7, -7, -9]],
            ],
        ]),
        jnp.asarray([[False, True, False], [False, False, True]]),
    )

    key = jax.random.PRNGKey(1234)
    l = conditioning.Conditioning.Config(
        'test', IDENTITY, CONCAT, name='conditioning_layer'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    conditioned_x = l.layer(
        x, training=False, constants=constants
    ).mask_invalid()

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_condition_tensor_concat_combination(self):
    # [2, 3, 2, 4]
    x = types.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[4, 3, 2, 1], [8, 7, 6, 5]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[2, 4, 6, 8], [1, 3, 5, 7]],
            ],
        ]),
        jnp.asarray([[False, True, False], [True, False, True]]),
    )

    # [2, 2]
    c = _float_tensor([[-1, -2], [-3, -4]])

    # [2, 3, 2, 6]
    expected_conditioned_x = types.Sequence(
        _float_tensor([
            [
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[1, 2, 3, 4, -1, -2], [5, 6, 7, 8, -1, -2]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            ],
            [
                [[4, 3, 2, 1, -3, -4], [8, 7, 6, 5, -3, -4]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[2, 4, 6, 8, -3, -4], [1, 3, 5, 7, -3, -4]],
            ],
        ]),
        jnp.asarray([[False, True, False], [True, False, True]]),
    )

    key = jax.random.PRNGKey(1234)
    l = conditioning.Conditioning.Config(
        'test', IDENTITY, CONCAT, name='conditioning_layer'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    conditioned_x = l.layer(
        x, training=False, constants=constants
    ).mask_invalid()

    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_conditioned_values_add_combination_2d_sequences(self):
    # [2, 3]
    x = types.Sequence(
        _float_tensor([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        jnp.asarray([[False, True, False], [True, False, True]]),
    )

    # [2, 3]
    c = types.Sequence(
        _float_tensor([
            [2, 4, 6],
            [3, 5, 7],
        ]),
        jnp.asarray([[True, True, False], [False, True, True]]),
    )

    # [2, 3]
    expected_conditioned_x = types.Sequence(
        _float_tensor([
            [0, 2 + 4, 0],
            [0, 0, 6 + 7],
        ]),
        jnp.asarray([[False, True, False], [False, False, True]]),
    )

    key = jax.random.PRNGKey(1234)
    l = conditioning.Conditioning.Config(
        'test', IDENTITY, ADD, name='conditioning_layer'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    conditioned_x = l.layer(
        x, training=False, constants=constants
    ).mask_invalid()
    self.assertSequencesClose(conditioned_x, expected_conditioned_x)

  def test_conditioned_values_concat_combination_2d_sequences(self):
    # [2, 3]
    x = types.Sequence(
        _float_tensor([
            [1, 2, 3],
            [4, 5, 6],
        ]),
        jnp.asarray([[False, True, False], [True, False, True]]),
    )

    # [2, 3]
    c = types.Sequence(
        _float_tensor([
            [2, 4, 6],
            [3, 5, 7],
        ]),
        jnp.asarray([[True, True, False], [False, True, True]]),
    )

    # [2, 3, 2]
    expected_conditioned_x = types.Sequence(
        _float_tensor([
            [[0, 0], [2, 4], [0, 0]],
            [[0, 0], [0, 0], [6, 7]],
        ]),
        jnp.asarray([[False, True, False], [False, False, True]]),
    )

    key = jax.random.PRNGKey(1234)
    l = conditioning.Conditioning.Config(
        'test', IDENTITY, CONCAT, name='conditioning_layer'
    ).make()
    constants = {'test': c}
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    conditioned_x = l.layer(
        x, training=False, constants=constants
    ).mask_invalid()
    self.assertSequencesClose(conditioned_x, expected_conditioned_x)


if __name__ == '__main__':
  test_utils.main()
