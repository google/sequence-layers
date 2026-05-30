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
from sequence_layers.specs import normalization_behaviors as spec


class L2NormalizeTest(spec.L2NormalizeTest, test_utils.SequenceLayerTest):

  @parameterized.product(
      test_utils.standard_dtype_configs(input=True),
      config=(
          dict(training=True),
          dict(epsilon=1.0),
      ),
  )
  def test_l2_normalization_dtypes(self, input_dtype, config):
    key = jax.random.PRNGKey(1234)
    shape, axes = (2, 3, 4, 8), [-1, -2]
    training = config.pop('training', False)
    layer = normalization.L2Normalize.Config(axis=axes, **config).make()
    inputs = test_utils.random_sequence(*shape, dtype=input_dtype)
    layer = self.init_and_bind_layer(key, layer, inputs)
    unboxed_variables = flax.core.meta.unbox(layer.variables)

    self.assertEmpty(unboxed_variables)

    self.verify_contract(
        layer,
        inputs,
        training=training,
        **test_utils.get_grad_tols(layer, inputs, jnp.float32, input_dtype),
    )


class LayerNormalizationTest(
    spec.LayerNormalizationTest, test_utils.SequenceLayerTest
):

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
    shape, axes, expected_param_shape = (2, 3, 4, 8), [-1, -2], [4, 8]
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


class RMSNormalizationTest(
    spec.RMSNormalizationTest, test_utils.SequenceLayerTest
):

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
    shape, axes, expected_param_shape = (2, 3, 4, 8), [-1, -2], [4, 8]
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


class BatchNormalizationTest(
    spec.BatchNormalizationTest, test_utils.SequenceLayerTest
):

  @parameterized.parameters(
      ((4, 10, 3), -1, [3]),
      ((4, 3, 5, 9), -2, [5]),
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
    l = self.init_and_bind_layer(key, l, x, randomize_weights=False)
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
    shape, axis, expected_param_shape = (4, 3, 5, 9), -2, [5]
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


class GroupNormalizationTest(
    spec.GroupNormalizationTest, test_utils.SequenceLayerTest
):

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
    shape, axis, num_groups, expected_param_shape = (8, 6, 5, 4), -2, 5, [5]
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


class WeightNormalizationTest(test_utils.SequenceLayerTest):

  def test_l2_normalization(self):
    epsilon = 1e-12
    normalizer = normalization.L2WeightNormalization.Config(
        name='l2_normalization',
        epsilon=epsilon,
    ).make()
    w = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    key = jax.random.PRNGKey(0)
    params = normalizer.init(key, w=w, training=False)
    normalized = normalizer.apply(params, w=w, training=False)
    self.assertAllClose(
        normalized,
        w / (jnp.sqrt((w**2).sum(axis=[0], keepdims=True)) + epsilon),
    )

  @parameterized.product(kernel_size=[2, 3, 5], training=[True, False])
  def test_spectral_norm(self, kernel_size: int, training: bool):
    epsilon = 1e-12
    normalizer = normalization.SpectralWeightNormalization.Config(
        name='spectral_normalization',
        epsilon=epsilon,
        n_power_iteration=10,
    ).make()
    # This test is based on keras SpectralNormalizationTest.test_normalization,
    # which checks that SN normalizes weights by the maximum eigen value.
    w = np.random.rand(kernel_size, kernel_size).astype(np.float32)
    w = w @ w.T
    eigen_val, _ = jnp.linalg.eig(w)
    expected = w / jnp.max(jnp.abs(eigen_val))

    key = jax.random.PRNGKey(0)
    params = normalizer.init(key, w=w, training=training)
    normalized, _ = normalizer.apply(
        params,
        w=w,
        training=training,
        mutable=['batch_stats'],
    )
    self.assertAllClose(normalized, expected)


if __name__ == '__main__':
  test_utils.main()
