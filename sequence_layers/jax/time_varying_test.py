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
from absl.testing import parameterized
import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import test_utils
from sequence_layers.jax import time_varying
from sequence_layers.jax import types
from sequence_layers.jax import utils


def _round_up_to_multiple_of(x: int, multiple_of: int) -> int:
  return (x + multiple_of - 1) // multiple_of * multiple_of


class SequenceEmbeddingTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      shape=((3, 4), (1, 2, 3), (2, 3, 5, 9)),
      round_multiple=(None, 128),
  )
  def test_constant_num_embeddings_per_step(
      self,
      shape: tuple[int, ...],
      round_multiple: int | None,
  ):
    key = jax.random.PRNGKey(1234)
    dimension, num_embeddings, num_steps = 8, 5, 7
    x = test_utils.random_sequence(
        *shape, dtype=jnp.int32, low=0, high=num_embeddings - 1
    )
    l = time_varying.SequenceEmbedding.Config(
        dimension=dimension,
        num_embeddings_per_step=num_embeddings,
        num_steps=num_steps,
        round_num_embeddings_to_multiple_of=round_multiple,
        name='sequence_embedding',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'sequence_embedding')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), shape[2:] + (dimension,)
    )
    y = self.verify_contract(
        l,
        x,
        training=False,
        # Integer tensors have no gradient to test.
        test_gradients=False,
    ).mask_invalid()

    variables = flax.core.meta.unbox(l.variables)
    expected_num_embs = num_embeddings * num_steps
    if round_multiple:
      expected_num_embs = _round_up_to_multiple_of(
          expected_num_embs, round_multiple
      )
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'embedding': jnp.zeros((expected_num_embs, dimension)),
            }
        },
    )

    embedding = variables['params']['embedding']
    seqlen = shape[1]
    offsets = jnp.reshape(
        jnp.arange(seqlen) % num_steps, [seqlen] + [1] * (len(shape) - 2)
    )
    expected = types.Sequence(
        jnp.take(
            embedding,
            x.values + offsets * num_embeddings,
            axis=0,
        ),
        x.mask,
    ).mask_invalid()
    self.assertSequencesClose(y, expected)

  @parameterized.parameters(None, 20)
  def test_num_embeddings_per_step(self, round_multiple: int | None):
    key = jax.random.PRNGKey(1234)
    dimension, num_steps = 8, 8
    num_embeddings_per_step = (3, 5, 7, 11, 13, 17, 19, 23)

    x = types.Sequence.concatenate_sequences(
        test_utils.random_sequence(
            2,
            1,
            3,
            4,
            dtype=jnp.int32,
            low=0,
            high=n,
            random_lengths=False,
        )
        for n in num_embeddings_per_step
    )

    l = time_varying.SequenceEmbedding.Config(
        dimension=dimension,
        num_embeddings_per_step=num_embeddings_per_step,
        num_steps=num_steps,
        round_num_embeddings_to_multiple_of=round_multiple,
        name='sequence_embedding',
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'sequence_embedding')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), x.channel_shape + (dimension,)
    )
    self.verify_contract(
        l,
        # Truncate for contract verification because of padding within
        # verify_contract.
        x[:4],
        training=False,
        # Integer tensors have no gradient to test.
        test_gradients=False,
    )

    expected_num_embs = sum(num_embeddings_per_step)
    if round_multiple:
      expected_num_embs = _round_up_to_multiple_of(
          expected_num_embs, round_multiple
      )

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'embedding': jnp.zeros((expected_num_embs, dimension)),
            }
        },
    )

    y = l.layer(x, training=False)
    embedding = variables['params']['embedding']
    seqlen = x.shape[1]
    steps = jnp.arange(seqlen)
    offsets = np.cumsum((0,) + num_embeddings_per_step[:-1])
    offsets = jnp.take(offsets, steps)
    offsets = jnp.reshape(offsets, [1, seqlen] + [1] * (x.ndim - 2))
    expected = types.Sequence(
        jnp.take(
            embedding,
            x.values + offsets,
            axis=0,
        ),
        x.mask,
    ).mask_invalid()

    self.assertSequencesClose(y, expected)

  def test_out_of_range(self):
    key = jax.random.PRNGKey(1234)
    dimension, num_steps = 8, 8
    num_embeddings_per_step = (3, 5, 7, 11, 13, 17, 19, 23)

    x = types.Sequence.from_values(
        jnp.stack([
            jnp.array([-1, 0, 2, 3]),
            jnp.array([-1, 0, 4, 5]),
            jnp.array([-1, 0, 6, 7]),
            jnp.array([-1, 0, 10, 11]),
            jnp.array([-1, 0, 12, 13]),
            jnp.array([-1, 0, 16, 17]),
            jnp.array([-1, 0, 18, 19]),
            jnp.array([-1, 0, 22, 23]),
            # Step out of range.
            jnp.array([0, 0, 0, 0]),
        ])[jnp.newaxis, ...]
    )

    l = time_varying.SequenceEmbedding.Config(
        dimension=dimension,
        num_embeddings_per_step=num_embeddings_per_step,
        num_steps=num_steps,
        name='sequence_embedding',
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    embedding = flax.core.meta.unbox(l.variables)['params']['embedding']
    y = l.layer(x, training=False)

    # The -1 and num_embeddings_per_step values get replaced with NaN.
    self.assertAllEqual(jnp.isnan(y.values[0, :-1, 0]), True)
    self.assertAllEqual(jnp.isnan(y.values[0, :-1, 3]), True)

    # The final step is all NaN since it's out of range.
    self.assertAllEqual(jnp.isnan(y.values[0, -1, :]), True)

    def get_embeddings(*indices):
      return jnp.take(embedding, jnp.array(indices), axis=0)

    expected_embeddings = jnp.stack([
        get_embeddings(0 + 0, 0 + 2),
        get_embeddings(3 + 0, 3 + 4),
        get_embeddings(8 + 0, 8 + 6),
        get_embeddings(15 + 0, 15 + 10),
        get_embeddings(26 + 0, 26 + 12),
        get_embeddings(39 + 0, 39 + 16),
        get_embeddings(56 + 0, 56 + 18),
        get_embeddings(75 + 0, 75 + 22),
    ])
    self.assertAllEqual(y.values[0, :-1, 1:3], expected_embeddings)

  @parameterized.product(
      test_utils.standard_dtype_configs(param=True, input=False, compute=True)
  )
  def test_dtypes(self, param_dtype, compute_dtype):
    channels_shape = (3, 5)
    key = jax.random.PRNGKey(1234)
    dimension, num_embeddings, num_groups = 8, 5, 7
    l = time_varying.SequenceEmbedding.Config(
        dimension=dimension,
        num_embeddings_per_step=num_embeddings,
        num_steps=num_groups,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        name='sequence_embedding',
    ).make()
    x = test_utils.random_sequence(
        7, 9, *channels_shape, dtype=jnp.int32, low=0, high=num_embeddings - 1
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
                'embedding': jnp.zeros(
                    (num_embeddings * num_groups, dimension), dtype=param_dtype
                ),
            }
        },
    )


class SequenceDenseTest(test_utils.SequenceLayerTest):

  def test_sequence_dense(self):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    # We need to support more steps than the input time dim because
    # verify_contract() pads the input in various ways.
    input_time, num_steps = 4, 8
    input_dim, output_dim = 5, 6
    l = time_varying.SequenceDense.Config(
        features=output_dim,
        num_steps=num_steps,
        bias_init=nn.initializers.normal(),
        name='sequence_dense',
    ).make()
    x = test_utils.random_sequence(batch_size, input_time, input_dim)
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'sequence_dense')
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        (output_dim,),
    )

    y = self.verify_contract(l, x, training=False)

    params = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        params,
        {
            'params': {
                'kernel': jnp.zeros((num_steps, input_dim, output_dim)),
                'bias': jnp.zeros((
                    num_steps,
                    output_dim,
                )),
            }
        },
    )

    kernel = params['params']['kernel']
    bias = params['params']['bias']

    ys = []
    for i in range(x.shape[1]):
      kernel_i = kernel[i]
      bias_i = bias[i]
      x_i = x[:, i : i + 1]

      ys.append(
          x_i.apply_values(
              lambda v: jnp.einsum('...a,ab->...b', v, kernel_i) + bias_i  # pylint: disable=cell-var-from-loop
          )
      )

    expected = types.Sequence.concatenate_sequences(ys).mask_invalid()
    self.assertSequencesClose(y, expected)

  def test_out_of_range(self):
    key = jax.random.PRNGKey(1234)
    dimension, num_steps = 8, 8
    # Include one out of range step.
    x = types.Sequence.from_values(
        jnp.arange(1, num_steps + 2)[jnp.newaxis, :, jnp.newaxis]
    )

    l = time_varying.SequenceDense.Config(
        features=dimension,
        num_steps=num_steps,
        name='sequence_dense',
    ).make()
    l = self.init_and_bind_layer(key, l, x)

    variables = flax.core.meta.unbox(l.variables)
    kernel = variables['params']['kernel']
    bias = variables['params']['bias']
    y = l.layer(x, training=False)
    y_step, _, _ = utils.step_by_step_dynamic(l, x, training=False)

    # The final step is all NaN since it's out of range.
    self.assertAllEqual(jnp.isnan(y.values[0, -1, :]), True)
    self.assertAllEqual(jnp.isnan(y_step.values[0, -1, :]), True)

    def get_projection_and_bias(i):
      return kernel[i, 0] + bias[i]

    expected_embeddings = jnp.stack([
        1 * get_projection_and_bias(0),
        2 * get_projection_and_bias(1),
        3 * get_projection_and_bias(2),
        4 * get_projection_and_bias(3),
        5 * get_projection_and_bias(4),
        6 * get_projection_and_bias(5),
        7 * get_projection_and_bias(6),
        8 * get_projection_and_bias(7),
    ])
    self.assertAllEqual(y.values[0, :-1], expected_embeddings)
    self.assertAllEqual(y_step.values[0, :-1], expected_embeddings)

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      use_bias=(True, False),
  )
  def test_dtypes(self, param_dtype, input_dtype, compute_dtype, use_bias):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    # We need to support more steps than the input time dim because
    # verify_contract() pads the input in various ways.
    input_time, num_steps = 4, 8
    input_dim, output_dim = 5, 6
    l = time_varying.SequenceDense.Config(
        output_dim,
        num_steps=num_steps,
        use_bias=use_bias,
        bias_init=nn.initializers.normal(),
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        name='sequence_dense',
    ).make()
    x = test_utils.random_sequence(
        batch_size, input_time, input_dim, random_mask=True, dtype=input_dtype
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

    bias_parameters = {
        'bias': jnp.zeros(
            (
                num_steps,
                output_dim,
            ),
            dtype=param_dtype,
        )
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'kernel': jnp.zeros(
                    (num_steps, input_dim, output_dim), dtype=param_dtype
                ),
            } | (bias_parameters if use_bias else {})
        },
    )


class MaskedDenseTest(test_utils.SequenceLayerTest):

  def test_causality(self):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    num_steps = 3
    input_dim, output_dim = 4, 5

    input_shape = (batch_size, num_steps, input_dim)
    output_shape = (batch_size, num_steps, output_dim)

    l = time_varying.MaskedDense.Config(
        features=output_dim,
        num_steps=num_steps,
        bias_init=nn.initializers.normal(),
    ).make()
    x = test_utils.random_sequence(*input_shape, random_lengths=False)
    l = self.init_and_bind_layer(key, l, x)

    @jax.jit
    def forward_fn(x):
      # We can't pass mask through jacfwd.
      x = types.Sequence.from_values(x)
      return l.layer(x, training=False).values

    y = forward_fn(x.values)
    self.assertEqual(y.shape, output_shape)

    # Confirm that autoregressive causality is obeyed.
    # Take Jacobian of y=[By, Ty, O] wrt x=[Bx, Tx, I].
    jac = jax.jacfwd(forward_fn)(x.values)  # [By, Ty, O, Bx, Tx, I]

    self.assertEqual(jac.shape, output_shape + input_shape)

    # Reduce Jacobian across channel dim to get boolean connectedness.
    # [By, Ty, O, Bx, Tx, I] -> [By, Ty, Bx, Tx]
    connected = jnp.any(jnp.not_equal(jac, 0.0), axis=[2, 5])

    for by in range(batch_size):
      for bx in range(batch_size):
        for ty in range(num_steps):  # y[by, ty]
          if bx == by:
            # For matching batch index, y[n] is only connected to x[<=n].
            for tx in range(ty + 1):  # x[bx=by, tx<=ty]
              self.assertTrue(connected[by, ty, bx, tx])
            for tx in range(ty + 1, num_steps):  # x[bx=by, tx>ty]
              self.assertFalse(connected[by, ty, bx, tx])
          else:
            # For non-matching batch index, nothing is connected.
            for tx in range(num_steps):  # x[bx!=by, tx]
              self.assertFalse(connected[by, ty, bx, tx])

  def test_masked_dense(self):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    # We need to support more steps than the input time dim because
    # verify_contract() pads the input in various ways.
    input_time, num_steps = 4, 8
    input_dim, output_dim = 5, 6

    l = time_varying.MaskedDense.Config(
        features=output_dim,
        num_steps=num_steps,
        bias_init=nn.initializers.normal(),
    ).make()
    x = test_utils.random_sequence(batch_size, input_time, input_dim)
    l = self.init_and_bind_layer(key, l, x)
    self.assertTrue(l.supports_step)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), (output_dim,))

    # Don't use NaN for padding invariance checks since they can pass through
    # the causality mask.
    self.verify_contract(
        l,
        x,
        training=False,
        padding_invariance_pad_value=1e9,
    )

    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'kernel': jnp.zeros(
                    (num_steps, num_steps, input_dim, output_dim)
                ),
                'bias': jnp.zeros((
                    num_steps,
                    output_dim,
                )),
            }
        },
    )

  @parameterized.product(
      test_utils.standard_dtype_configs(),
      use_bias=(True, False),
  )
  def test_dtypes(self, param_dtype, input_dtype, compute_dtype, use_bias):
    key = jax.random.PRNGKey(1234)
    batch_size = 2
    # We need to support more steps than the input time dim because
    # verify_contract() pads the input in various ways.
    input_time, num_steps = 4, 8
    input_dim, output_dim = 5, 6
    l = time_varying.MaskedDense.Config(
        output_dim,
        num_steps=num_steps,
        use_bias=use_bias,
        bias_init=nn.initializers.normal(),
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        name='masked_dense',
    ).make()
    x = test_utils.random_sequence(
        batch_size, input_time, input_dim, random_mask=True, dtype=input_dtype
    )
    l = self.init_and_bind_layer(key, l, x)
    # Don't use NaN for padding invariance checks since they can pass through
    # the causality mask.
    y = self.verify_contract(
        l,
        x,
        training=False,
        padding_invariance_pad_value=1e9,
        **test_utils.get_grad_tols(l, x, param_dtype, compute_dtype),
    )
    if compute_dtype:
      self.assertEqual(y.dtype, compute_dtype)

    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': (
                {
                    'kernel': jnp.zeros(
                        (num_steps, num_steps, input_dim, output_dim),
                        dtype=param_dtype,
                    ),
                }
                | (
                    {
                        'bias': jnp.zeros(
                            (
                                num_steps,
                                output_dim,
                            ),
                            dtype=param_dtype,
                        )
                    }
                    if use_bias
                    else {}
                )
            )
        },
    )


if __name__ == '__main__':
  test_utils.main()
