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
"""Normalization tests."""

import itertools

from absl.testing import parameterized
import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import combinators
from sequence_layers.jax import dense
from sequence_layers.jax import normalization
from sequence_layers.jax import test_utils
from sequence_layers.jax import types


class LayerNormalizationTest(test_utils.SequenceLayerTest):

  def test_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    key = jax.random.PRNGKey(1234)
    l = normalization.LayerNormalization.Config(axis=[-1, -2]).make()
    x = test_utils.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  @parameterized.parameters(
      itertools.product(
          (False, True),
          [
              ((2, 10, 32), [-1], [32]),
              ((2, 3, 5, 32), [-1], [32]),
              ((2, 3, 32, 9), [-2], [32]),
              ((2, 3, 32, 8), [-1, -2], [32, 8]),
          ],
      )
  )
  def test_layer_normalization(self, training, shape_axes):
    key = jax.random.PRNGKey(1234)
    shape, axes, expected_param_shape = shape_axes
    l = normalization.LayerNormalization.Config(
        axis=axes, name='layer_normalization'
    ).make()
    x = test_utils.random_sequence(*shape)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'layer_normalization')
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    y = self.verify_contract(l, x, training=training)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'scale': jnp.zeros(expected_param_shape),
                'bias': jnp.zeros(expected_param_shape),
            }
        },
    )

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a in axes or a - len(shape) in axes
    )
    mean = np.mean(y.values, axis=reduce_axes)
    var = np.var(y.values, axis=reduce_axes)

    # Invalid timesteps will have a mean and variance of zero.
    chex.assert_trees_all_close(mean, np.zeros_like(mean), rtol=1e-6, atol=1e-6)
    mask = y.mask.astype(jnp.float32)
    mask = np.reshape(
        mask, mask.shape + (1,) * (len(mean.shape) - len(mask.shape))
    )
    chex.assert_trees_all_close(
        var, np.broadcast_to(mask, mean.shape), rtol=1e-4, atol=1e-4
    )

  @parameterized.product(
      test_utils.standard_dtype_configs(param=True, input=True),
      config=(
          dict(training=True),
          dict(epsilon=1.0),
          dict(use_bias=False),
          dict(use_bias=False, use_scale=False),
          dict(reductions_in_at_least_fp32=False),
      ),
  )
  def test_layer_normalization_dtypes(self, param_dtype, input_dtype, config):
    key = jax.random.PRNGKey(1234)
    shape, axes, expected_param_shape = (2, 3, 32, 8), [-1, -2], [32, 8]
    training = config.pop('training', False)
    defaults = dict(
        axis=axes,
        epsilon=1e-6,
        use_bias=True,
        use_scale=True,
        reductions_in_at_least_fp32=True,
        param_dtype=param_dtype,
    )
    layer = normalization.LayerNormalization.Config(
        **(defaults | config)
    ).make()
    inputs = test_utils.random_sequence(*shape, dtype=input_dtype)
    layer = self.init_and_bind_layer(key, layer, inputs)
    unboxed_variables = flax.core.meta.unbox(layer.variables)

    params = {}
    if layer.config.use_scale:
      params['scale'] = jnp.ones(expected_param_shape, dtype=param_dtype)
    if layer.config.use_bias:
      params['bias'] = jnp.zeros(expected_param_shape, dtype=param_dtype)
    if params:
      chex.assert_trees_all_equal(unboxed_variables['params'], params)
      chex.assert_trees_all_equal_dtypes(unboxed_variables['params'], params)
    else:
      self.assertNotIn('params', unboxed_variables)

    self.verify_contract(
        layer,
        inputs,
        training=training,
        **test_utils.get_grad_tols(layer, inputs, param_dtype, input_dtype),
    )


class RMSNormalizationTest(test_utils.SequenceLayerTest):

  def test_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    key = jax.random.PRNGKey(1234)
    l = normalization.RMSNormalization.Config(
        axis=[-1, -2],
    ).make()
    x = test_utils.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  @parameterized.parameters(
      itertools.product(
          (False, True),
          [
              ((2, 10, 3), [-1], [3]),
              ((2, 3, 5, 9), [-1], [9]),
              ((2, 3, 5, 9), [-2], [5]),
              ((2, 3, 5, 9), [-1, -2], [5, 9]),
          ],
      )
  )
  def test_rms_normalization(self, training, shape_axes):
    key = jax.random.PRNGKey(1234)
    shape, axes, expected_param_shape = shape_axes
    epsilon = 1e-1
    l = normalization.RMSNormalization.Config(
        axes, epsilon=epsilon, name='rms_normalization'
    ).make()
    x = test_utils.random_sequence(*shape)
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'rms_normalization')
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    y = self.verify_contract(l, x, training=training)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'scale': jnp.zeros(expected_param_shape),
            }
        },
    )

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a in axes or a - len(shape) in axes
    )
    x_ss = np.mean(np.square(x.values), axis=reduce_axes, keepdims=True)

    y_expected = types.Sequence(
        x.values / np.sqrt(x_ss + epsilon), x.mask
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)

  @parameterized.product(
      test_utils.standard_dtype_configs(param=True, input=True),
      config=(
          dict(training=True),
          dict(epsilon=1.0),
          dict(use_scale=False),
          dict(reductions_in_at_least_fp32=False),
      ),
  )
  def test_rms_normalization_dtypes(self, param_dtype, input_dtype, config):
    key = jax.random.PRNGKey(1234)
    shape, axes, expected_param_shape = (2, 3, 32, 8), [-1, -2], [32, 8]
    training = config.pop('training', False)
    defaults = dict(
        axis=axes,
        epsilon=1e-6,
        use_scale=True,
        reductions_in_at_least_fp32=True,
        param_dtype=param_dtype,
    )
    layer = normalization.RMSNormalization.Config(**(defaults | config)).make()
    inputs = test_utils.random_sequence(*shape, dtype=input_dtype)
    layer = self.init_and_bind_layer(key, layer, inputs)
    unboxed_variables = flax.core.meta.unbox(layer.variables)

    expected_variables = {}
    if layer.config.use_scale:
      expected_variables['params'] = {
          'scale': jnp.ones(expected_param_shape, dtype=param_dtype)
      }
    chex.assert_trees_all_equal(unboxed_variables, expected_variables)
    chex.assert_trees_all_equal_dtypes(unboxed_variables, expected_variables)

    self.verify_contract(
        layer,
        inputs,
        training=training,
        **test_utils.get_grad_tols(layer, inputs, param_dtype, input_dtype),
    )


class BatchNormalizationTest(test_utils.SequenceLayerTest):

  def test_batch_normalization_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = normalization.BatchNormalization.Config(axis=0).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

    l = normalization.BatchNormalization.Config(axis=1).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

    l = normalization.BatchNormalization.Config(axis=2).make()
    self.init_and_bind_layer(key, l, x)

  @parameterized.parameters(
      ((32, 10, 3), -1, [3]),
      ((32, 3, 5, 9), -2, [5]),
      # TODO(rryan): Support multiple axes.
      # ((2, 3, 5, 9), [-1, -2], [5, 9]),
  )
  def test_batch_normalization(self, shape, axis, expected_param_shape):
    key = jax.random.PRNGKey(1234)
    epsilon = 1e-3
    l = normalization.BatchNormalization.Config(
        axis=axis, epsilon=epsilon, name='batch_normalization'
    ).make()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'batch_normalization')

    x = test_utils.random_sequence(*shape)
    l = self.init_and_bind_layer(key, l, x, randomize_weights=True)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    unboxed_variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        unboxed_variables,
        {
            'batch_stats': {
                'batch_normalization': {
                    'mean': jnp.zeros(expected_param_shape),
                    'var': jnp.zeros(expected_param_shape),
                },
            },
            'params': {
                'batch_normalization': {
                    'scale': jnp.zeros(expected_param_shape),
                    'bias': jnp.zeros(expected_param_shape),
                },
            },
        },
    )

    expanded_param_shape = [1] * x.values.ndim
    expanded_param_shape[axis] = x.values.shape[axis]

    scale = unboxed_variables['params']['batch_normalization']['scale'].reshape(
        expanded_param_shape
    )
    bias = unboxed_variables['params']['batch_normalization']['bias'].reshape(
        expanded_param_shape
    )

    y_train, _ = l.apply(
        l.variables, x, training=True, method=l.layer, mutable=['batch_stats']
    )

    # Step-wise training is not supported.
    state = l.get_initial_state(
        batch_size=1, input_spec=x.channel_spec, training=True
    )
    with self.assertRaises(ValueError):
      l.step(x, state, training=True)

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a != axis and a - len(shape) != axis
    )

    expanded_mask = y_train.expanded_mask()
    y_unscaled = (y_train.values - bias) / scale
    mean = np.mean(
        y_unscaled, axis=reduce_axes, keepdims=True, where=expanded_mask
    )
    var = np.var(
        y_unscaled, axis=reduce_axes, keepdims=True, where=expanded_mask
    )
    chex.assert_trees_all_close(mean, jnp.zeros_like(mean), atol=1e-6)
    chex.assert_trees_all_close(var, jnp.ones_like(var), atol=1e-2)

    # Verify that layer-wise and step-wise processing are identical in
    # non-training mode.
    y_test = self.verify_contract(l, x, training=False)

    # Check that y2_np is correct given x, scale, bias and moving mean/var.
    moving_mean = unboxed_variables['batch_stats']['batch_normalization'][
        'mean'
    ].reshape(expanded_param_shape)
    moving_variance = unboxed_variables['batch_stats']['batch_normalization'][
        'var'
    ].reshape(expanded_param_shape)

    y_test_expected = types.Sequence(
        scale * (x.values - moving_mean) / np.sqrt(epsilon + moving_variance)
        + bias,
        x.mask,
    ).mask_invalid()
    self.assertSequencesClose(y_test, y_test_expected)

  @parameterized.product(
      test_utils.standard_dtype_configs(param=True, input=True),
      config=(
          dict(epsilon=1.0),
          dict(use_bias=False),
          dict(use_bias=False, use_scale=False),
          dict(use_fast_variance=False),
      ),
  )
  def test_batch_normalization_dtypes(self, param_dtype, input_dtype, config):
    key = jax.random.PRNGKey(1234)
    name = 'batch_normalization'
    shape, axis, expected_param_shape = (32, 3, 5, 9), -2, [5]
    defaults = dict(
        axis=axis,
        epsilon=1e-3,
        use_bias=True,
        use_scale=True,
        use_fast_variance=True,
        param_dtype=param_dtype,
        name=name,
    )
    layer = normalization.BatchNormalization.Config(
        **(defaults | config)
    ).make()
    inputs = test_utils.random_sequence(*shape, dtype=input_dtype)
    layer = self.init_and_bind_layer(key, layer, inputs)
    unboxed_variables = flax.core.meta.unbox(layer.variables)

    params = {}
    if layer.config.use_scale:
      params['scale'] = jnp.ones(expected_param_shape, dtype=param_dtype)
    if layer.config.use_bias:
      params['bias'] = jnp.zeros(expected_param_shape, dtype=param_dtype)
    if params:
      chex.assert_trees_all_equal(unboxed_variables['params'], {name: params})
      chex.assert_trees_all_equal_dtypes(
          unboxed_variables['params'], {name: params}
      )
    else:
      self.assertNotIn('params', unboxed_variables)

    # BN computes statistics in at least float32:
    stats_dtype = jnp.promote_types(input_dtype, jnp.float32)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        unboxed_variables['batch_stats'],
        {
            name: {
                'mean': jnp.zeros(expected_param_shape, dtype=stats_dtype),
                'var': jnp.zeros(expected_param_shape, dtype=stats_dtype),
            }
        },
    )

    self.verify_contract(
        layer,
        inputs,
        training=False,
        **test_utils.get_grad_tols(layer, inputs, param_dtype, input_dtype),
    )


class GroupNormalizationTest(test_utils.SequenceLayerTest):

  def test_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = normalization.GroupNormalization.Config(num_groups=1, axis=0).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

    l = normalization.GroupNormalization.Config(num_groups=1, axis=1).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

    l = normalization.GroupNormalization.Config(num_groups=1, axis=2).make()
    self.init_and_bind_layer(key, l, x)

  def test_invalid_groups(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = normalization.GroupNormalization.Config(num_groups=2).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  @parameterized.parameters(
      itertools.product(
          [
              ((8, 32, 32), -1, 8, [32]),
              ((8, 32, 5, 32), -2, 5, [5]),
              ((8, 32, 5, 32), -2, 1, [5]),
          ],
          (False, True),
      )
  )
  def test_group_normalization(self, shape_axes, cumulative):
    key = jax.random.PRNGKey(1234)
    shape, axis, num_groups, expected_param_shape = shape_axes
    l = normalization.GroupNormalization.Config(
        num_groups=num_groups,
        cumulative=cumulative,
        axis=axis,
        name='group_normalization',
    ).make()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'group_normalization')

    x = test_utils.random_sequence(*shape)
    l = self.init_and_bind_layer(key, l, x, randomize_weights=True)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    y = self.verify_contract(
        l,
        x,
        training=True,
        grad_rtol=1e-5,
        grad_atol=1e-4,
    )
    y_test = self.verify_contract(
        l,
        x,
        training=False,
        grad_rtol=1e-5,
        grad_atol=1e-4,
    )

    # Training mode doesn't affect behavior.
    self.assertSequencesEqual(y, y_test)

    unboxed_variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        unboxed_variables,
        {
            'params': {
                'scale': jnp.zeros(expected_param_shape),
                'bias': jnp.zeros(expected_param_shape),
            }
        },
    )

    axis = axis + x.ndim if axis < 0 else axis
    axis_dim = y.values.shape[axis]
    group_size = axis_dim // num_groups
    outer_dims, _, inner_dims = np.split(y.values.shape, [axis, axis + 1])

    expanded_param_shape = [1] * y.values.ndim
    expanded_param_shape[axis] = axis_dim
    scale = unboxed_variables['params']['scale'].reshape(expanded_param_shape)
    bias = unboxed_variables['params']['bias'].reshape(expanded_param_shape)

    y_grouped = np.reshape(
        (y.values - bias) / scale,
        outer_dims.tolist() + [num_groups, group_size] + inner_dims.tolist(),
    )

    if cumulative:
      # TODO(rryan): Test cumulative mode numerically.
      return

    reduction_dims = [a for a in range(y_grouped.ndim) if a not in (0, axis)]

    expanded_mask = types.Sequence(y_grouped, x.mask).expanded_mask()

    # Check each group is mean zero and unit variance.
    mean = np.mean(
        y_grouped, axis=reduction_dims, keepdims=True, where=expanded_mask
    )
    var = np.var(
        y_grouped, axis=reduction_dims, keepdims=True, where=expanded_mask
    )

    # Handle zero length sequences. The moment calculation avoids NaNs by
    # capping divisors at 1.
    mean = np.where(np.isnan(mean), np.zeros_like(mean), mean)
    var = np.where(np.isnan(var), np.ones_like(var), var)

    chex.assert_trees_all_close(mean, jnp.zeros_like(mean), atol=1e-6)
    chex.assert_trees_all_close(var, jnp.ones_like(var), atol=1e-4)

  @parameterized.product(
      test_utils.standard_dtype_configs(param=True, input=True),
      config=(
          dict(epsilon=1.0),
          dict(cumulative=True),
          dict(use_bias=False),
          dict(use_bias=False, use_scale=False),
      ),
      training=(False, True),
  )
  def test_group_normalization_dtypes(
      self, param_dtype, input_dtype, config, training
  ):
    key = jax.random.PRNGKey(1234)
    shape, axis, num_groups, expected_param_shape = (8, 6, 5, 32), -2, 5, [5]
    defaults = dict(
        num_groups=num_groups,
        axis=axis,
        epsilon=1e-6,
        cumulative=False,
        use_scale=True,
        use_bias=True,
        param_dtype=param_dtype,
    )
    layer = normalization.GroupNormalization.Config(
        **(defaults | config)
    ).make()
    inputs = test_utils.random_sequence(*shape, dtype=input_dtype)
    layer = self.init_and_bind_layer(key, layer, inputs)
    unboxed_variables = flax.core.meta.unbox(layer.variables)

    params = {}
    if layer.config.use_scale:
      params['scale'] = jnp.ones(expected_param_shape, dtype=param_dtype)
    if layer.config.use_bias:
      params['bias'] = jnp.zeros(expected_param_shape, dtype=param_dtype)
    if params:
      chex.assert_trees_all_equal(unboxed_variables['params'], params)
      chex.assert_trees_all_equal_dtypes(unboxed_variables['params'], params)
    else:
      self.assertNotIn('params', unboxed_variables)

    self.verify_contract(
        layer,
        inputs,
        training=training,
        **test_utils.get_grad_tols(layer, inputs, param_dtype, input_dtype),
    )

  def test_group_normalization_zero_length_sequence(self):
    key = jax.random.PRNGKey(1234)
    l = normalization.GroupNormalization.Config(
        num_groups=4,
        cumulative=False,
        axis=-1,
        name='group_normalization',
    ).make()

    x = types.MaskedSequence(jnp.zeros((1, 3, 8)), jnp.zeros((1, 3), jnp.bool_))
    l = self.init_and_bind_layer(key, l, x, randomize_weights=True)

    y = l.layer(x, training=False).mask_invalid()
    # GroupNormalization leaves the length zero sequences unchanged.
    self.assertSequencesEqual(x, y)


class ZeroInputStabilityTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      normalization.LayerNormalization.Config(epsilon=1e-32, use_bias=False),
      normalization.RMSNormalization.Config(epsilon=1e-32),
      # TODO(rryan): Fix BatchNormalization stability with all zero input.
      # normalization.BatchNormalization.Config(epsilon=1e-32, use_bias=False),
      normalization.GroupNormalization.Config(
          num_groups=2, epsilon=1e-32, use_bias=False
      ),
      normalization.GroupNormalization.Config(
          num_groups=2,
          epsilon=1e-32,
          cumulative=True,
          use_bias=False,
      ),
  )
  def test_zero_input_gradient(self, norm_config):
    config = combinators.Repeat.Config(
        combinators.Serial.Config([
            dense.Dense.Config(8, use_bias=False),
            norm_config,
        ]),
        num_repeats=10,
    )
    x = types.Sequence.from_values(jnp.zeros((1, 1, 8)))
    l = self.init_and_bind_layer(jax.random.PRNGKey(42), config.make(), x)

    def f(params, x):
      y = (
          config.make()
          .bind(params, mutable='batch_stats')
          .layer(x, training=True)
          .mask_invalid()
      )
      return jnp.sum(y.values)

    grad_fn = jax.value_and_grad(f)
    outputs, grads = grad_fn(l.variables, x)

    self.assertAllEqual(jnp.isfinite(outputs), True)
    for grad in jax.tree.leaves(grads):
      self.assertAllEqual(jnp.isfinite(grad), True)


if __name__ == '__main__':
  test_utils.main()
