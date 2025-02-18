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
"""Dense tests."""

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from sequence_layers.jax import dense
from sequence_layers.jax import test_utils

from google3.testing.pybase import parameterized


class DenseTest(test_utils.SequenceLayerTest):

  def test_rank2_unsupported(self):
    key = jax.random.PRNGKey(1234)
    l = dense.Dense.Config(
        3, bias_init=nn.initializers.normal(), name='dense'
    ).make()
    x = test_utils.random_sequence(2, 13)
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)

  @parameterized.parameters(((5,),), ((5, 7),))
  def test_dense(self, channels_shape):
    key = jax.random.PRNGKey(1234)
    l = dense.Dense.Config(
        3, bias_init=nn.initializers.normal(), name='dense'
    ).make()
    x = test_utils.random_sequence(2, 13, *channels_shape, random_mask=True)
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dense')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), channels_shape[:-1] + (3,)
    )
    self.verify_contract(l, x, training=False, grad_rtol=1e-5, grad_atol=1e-5)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'kernel': jnp.zeros((channels_shape[-1], 3)),
                'bias': jnp.zeros((3,)),
            }
        },
    )

  @parameterized.parameters(True, False)
  def test_use_bias(self, use_bias):
    """Check that use_bias controls whether a bias is created."""
    key = jax.random.PRNGKey(1234)
    l = dense.Dense.Config(3, use_bias=use_bias).make()
    x = test_utils.random_sequence(2, 3, 5)
    l = self.init_and_bind_layer(key, l, x)
    self.assertCountEqual(
        l.variables['params'], ['kernel', 'bias'] if use_bias else ['kernel']
    )

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      use_bias=(True, False),
  )
  def test_dtypes(self, param_dtype, input_dtype, compute_dtype, use_bias):
    channels_shape = (3, 5)
    key = jax.random.PRNGKey(1234)
    l = dense.Dense.Config(
        2,
        use_bias=use_bias,
        bias_init=nn.initializers.normal(),
        dtype=compute_dtype,
        param_dtype=param_dtype,
        name='dense',
    ).make()
    x = test_utils.random_sequence(
        7, 9, *channels_shape, random_mask=True, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)
    y = self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )
    if compute_dtype:
      self.assertEqual(y.dtype, compute_dtype)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'kernel': jnp.zeros((channels_shape[-1], 2), dtype=param_dtype),
            } | (
                {'bias': jnp.zeros((2,), dtype=param_dtype)} if use_bias else {}
            )
        },
    )


class DenseShapedTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      shape=((2, 13), (2, 13, 1), (2, 13, 5), (2, 13, 5, 7)),
      output_shape=((), (1,), (6,), (6, 8)),
  )
  def test_dense_shaped(self, shape, output_shape):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)
    l = dense.DenseShaped.Config(
        output_shape, bias_init=nn.initializers.normal(), name='dense_shaped'
    ).make()

    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dense_shaped')
    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)

    self.verify_contract(l, x, training=False)

    input_kernel_shape = shape[2:] if shape[2:] else (1,)
    output_kernel_shape = output_shape if output_shape else (1,)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'kernel': jnp.zeros(input_kernel_shape + output_kernel_shape),
                'bias': jnp.zeros(output_kernel_shape),
            }
        },
    )

  @parameterized.parameters(True, False)
  def test_use_bias(self, use_bias):
    """Check that use_bias controls whether a bias is created."""
    key = jax.random.PRNGKey(1234)
    l = dense.DenseShaped.Config([3], use_bias=use_bias).make()
    x = test_utils.random_sequence(2, 13)
    l = self.init_and_bind_layer(key, l, x)
    self.assertCountEqual(
        l.variables['params'], ['kernel', 'bias'] if use_bias else ['kernel']
    )

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      use_bias=(True, False),
  )
  def test_dtypes(self, param_dtype, input_dtype, compute_dtype, use_bias):
    shape = (5, 11)
    output_shape = (2, 7, 13)
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape, random_mask=True, dtype=input_dtype)
    l = dense.DenseShaped.Config(
        output_shape,
        use_bias=use_bias,
        bias_init=nn.initializers.normal(),
        dtype=compute_dtype,
        param_dtype=param_dtype,
        name='dense_shaped',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    y = self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )
    if compute_dtype:
      self.assertEqual(y.dtype, compute_dtype)

    input_kernel_shape = shape[2:] if shape[2:] else (1,)
    output_kernel_shape = output_shape if output_shape else (1,)
    params = {
        'kernel': jnp.zeros(
            input_kernel_shape + output_kernel_shape, dtype=param_dtype
        )
    }
    if use_bias:
      params['bias'] = jnp.zeros(output_kernel_shape, dtype=param_dtype)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables), {'params': params}
    )


class EinsumDenseTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (
          (2, 3, 5),
          '...a,ab->...b',
          (7,),
          (5, 7),
          (7,),
      ),
      (
          (2, 3, 5, 7),
          '...ab,ac->...cb',
          (11, 7),
          (5, 11),
          (11, 7),
      ),
      (
          (2, 3, 5, 7),
          '...ab,b->...a',
          (None,),
          (7,),
          (5,),
      ),
      (
          (2, 3, 5, 7),
          '...ab,ab->...ba',
          (None, None),
          (5, 7),
          (7, 5),
      ),
      (
          (2, 3, 5, 7),
          '...ab,abc->...bac',
          (None, None, 2),
          (5, 7, 2),
          (7, 5, 2),
      ),
  )
  def test_einsum_dense(
      self,
      shape,
      equation,
      output_shape,
      expected_kernel_shape,
      expected_output_shape,
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)

    # default_kernel_init doesn't work for 1D kernels.
    if len(expected_kernel_shape) == 1:
      kernel_init = nn.initializers.normal()
    else:
      kernel_init = nn.linear.default_kernel_init

    l = dense.EinsumDense.Config(
        equation, output_shape, kernel_init=kernel_init, name='einsum_dense'
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'einsum_dense')
    self.assertEqual(l.get_output_shape_for_sequence(x), expected_output_shape)
    y = self.verify_contract(l, x, training=False)

    # A kernel.
    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros(expected_kernel_shape),
            }
        },
    )

    y_expected = x.apply_values(
        lambda v: jnp.einsum(equation, v, variables['params']['kernel'])
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)

  @parameterized.parameters(
      (
          (2, 3, 5),
          '...a,ab->...b',
          'b',
          (7,),
          (5, 7),
          (7,),
          (7,),
      ),
      (
          (2, 3, 5),
          '...a,abc->...bc',
          'b',
          (7, 11),
          (5, 7, 11),
          # Bias includes a broadcast dimension for c.
          (7, 1),
          (7, 11),
      ),
      (
          (2, 3, 5),
          '...a,abcd->...bcd',
          'c',
          (7, 11, 13),
          (5, 7, 11, 13),
          # Bias includes a broadcast dimension for d but nothing for b.
          (11, 1),
          (7, 11, 13),
      ),
      (
          (2, 3, 5),
          '...a,abcd->...bcd',
          'bc',
          (7, 11, 13),
          (5, 7, 11, 13),
          (7, 11, 1),
          (7, 11, 13),
      ),
      (
          (2, 3, 5),
          '...a,abcd->...bcd',
          'cd',
          (7, 11, 13),
          (5, 7, 11, 13),
          (11, 13),
          (7, 11, 13),
      ),
  )
  def test_bias(
      self,
      shape,
      equation,
      bias_axes,
      output_shape,
      expected_kernel_shape,
      expected_bias_shape,
      expected_output_shape,
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape)

    l = dense.EinsumDense.Config(
        equation, output_shape, bias_axes, name='einsum_dense'
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'einsum_dense')
    self.assertEqual(l.get_output_shape_for_sequence(x), expected_output_shape)
    y = self.verify_contract(l, x, training=False)

    # A kernel.
    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros(expected_kernel_shape),
                'bias': jnp.zeros(expected_bias_shape),
            }
        },
    )

    kernel = variables['params']['kernel']
    bias = variables['params']['bias']

    y_expected = x.apply_values(
        lambda v: jnp.einsum(equation, v, kernel) + bias
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      (
          dict(
              shape=(2, 3, 5, 7, 11),
              equation='...abc,bd->...bd',
              output_shape=(None, 13),
              expected_kernel_shape=(7, 13),
              bias_axes='',
              expected_bias_shape=None,
          ),
          dict(
              shape=(2, 3, 5),
              equation='...a,abcd->...bcd',
              output_shape=(7, 11, 13),
              expected_kernel_shape=(5, 7, 11, 13),
              bias_axes='cd',
              expected_bias_shape=(11, 13),
          ),
      ),
  )
  def test_dtypes(
      self,
      param_dtype,
      input_dtype,
      compute_dtype,
      shape,
      equation,
      output_shape,
      expected_kernel_shape,
      bias_axes,
      expected_bias_shape,
  ):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(*shape, random_mask=True, dtype=input_dtype)

    l = dense.EinsumDense.Config(
        equation,
        output_shape,
        bias_axes,
        dtype=compute_dtype,
        param_dtype=param_dtype,
        name='einsum_dense',
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    y = self.verify_contract(
        l,
        x,
        training=False,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )
    if compute_dtype:
      self.assertEqual(y.dtype, compute_dtype)

    params = {'kernel': jnp.zeros(expected_kernel_shape, dtype=param_dtype)}
    if bias_axes:
      params['bias'] = jnp.zeros(expected_bias_shape, dtype=param_dtype)

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(variables, {'params': params})

    kernel = variables['params']['kernel']
    y_expected = x.apply_values(
        lambda v: jnp.einsum(
            equation,
            v.astype(compute_dtype if compute_dtype else v.dtype),
            kernel.astype(compute_dtype if compute_dtype else param_dtype),
        )
    ).mask_invalid()
    if bias_axes:
      bias = variables['params']['bias']
      y_expected = y_expected.apply_values(
          lambda v: v
          + bias.astype(compute_dtype if compute_dtype else param_dtype)
      ).mask_invalid()
    self.assertSequencesClose(y, y_expected)

  def test_einsum_dense_nonbroadcasting_equation(self):
    with self.assertRaises(ValueError):
      key = jax.random.PRNGKey(1234)
      x = test_utils.random_sequence(2, 3, 4, 5, 6)
      l = dense.EinsumDense.Config(
          'btabc,bc->btad', output_shape=[None, 2]
      ).make()
      self.init_and_bind_layer(key, l, x)

  def test_einsum_dense_inconsistent_input_shape(self):
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 3, 5)
    l = dense.EinsumDense.Config(
        '...abc,bc->...ad', output_shape=[None, 2]
    ).make()
    with self.assertRaises(ValueError):
      self.init_and_bind_layer(key, l, x)
    # Show it works with the right input shape.
    x = test_utils.random_sequence(2, 3, 5, 7, 11)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 2))


if __name__ == '__main__':
  test_utils.main()
