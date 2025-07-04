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
"""Utilities test."""

import fractions
import typing
from typing import Any

from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import meta
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils


class FlaxEinsumDenseTest(test_utils.SequenceLayerTest):

  def test_basic_dense(self):
    key = jax.random.PRNGKey(1234)
    l = utils.FlaxEinsumDense(
        equation='abc,cd->abd',
        output_shape=(3, 7),
        bias_axes='d',
        bias_init=nn.initializers.normal(),
    )

    x = jnp.zeros((2, 3, 5))
    l = l.bind(l.init(key, x))

    chex.assert_trees_all_equal_shapes_and_dtypes(
        l.variables,
        {
            'params': {
                'kernel': jnp.zeros((5, 7)),
                'bias': jnp.zeros((7,)),
            }
        },
    )

    y = l(x)
    self.assertEqual(y.shape, (2, 3, 7))
    y_expected = jnp.einsum('abc,cd->abd', x, l.variables['params']['kernel'])
    y_expected += l.variables['params']['bias']
    chex.assert_trees_all_close(y, y_expected)

  def test_basic_dense_with_einsum_factory(self):
    def einsum_factory() -> types.JnpEinsumT:
      def custom_einsum(equation, *args, **kwargs):
        return jnp.multiply(jnp.einsum(equation, *args, **kwargs), 3)

      return custom_einsum

    key = jax.random.PRNGKey(1234)
    l = utils.FlaxEinsumDense(
        equation='abc,cd->abd',
        output_shape=(3, 7),
        bias_axes='d',
        bias_init=nn.initializers.zeros_init(),
        einsum_factory=einsum_factory,
    )

    x = jax.random.normal(key, (2, 3, 5))
    l = l.bind(l.init(key, x))

    chex.assert_trees_all_equal_shapes_and_dtypes(
        l.variables,
        {
            'params': {
                'kernel': jnp.zeros((5, 7)),
                'bias': jnp.zeros((7,)),
            }
        },
    )

    y = l(x)
    self.assertEqual(y.shape, (2, 3, 7))
    y_expected = jnp.einsum('abc,cd->abd', x, l.variables['params']['kernel'])
    y_expected *= 3
    chex.assert_trees_all_close(y, y_expected)


class AddCountLayer(types.PreservesShape, types.PreservesType, types.Emitting):
  """A test layer that just adds a counter to the input."""

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return {0: (0, 0)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: Any | None = None,
  ) -> types.State:
    return jnp.zeros((batch_size,), dtype=jnp.int32)

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: Any | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    count = jnp.tile(jnp.arange(x.shape[1])[jnp.newaxis, :], (x.shape[0], 1))
    count_reshaped = count.reshape(count.shape + (1,) * (x.ndim - 2)).astype(
        x.dtype
    )
    y = x.apply_values(lambda v: v + count_reshaped)
    emits = {'count': count, 'x': x}
    return y, emits

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: Any | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    count = (
        state[:, jnp.newaxis]
        + jnp.arange(x.shape[1], dtype=state.dtype)[jnp.newaxis, :]
    )
    count_reshaped = count.reshape(count.shape + (1,) * (x.ndim - 2)).astype(
        x.dtype
    )

    y = x.apply_values(lambda v: v + count_reshaped)
    state += x.shape[1]
    emits = {'count': count, 'x': x}
    return y, state, emits


class StepByStepTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((), False),
      ((5,), False),
      ((5, 7), False),
      ((5, 7, 11), False),
      ((5,), True),
  )
  def test_step_by_step_static(
      self,
      channel_shape,
      stream_constants,
  ):
    key = jax.random.PRNGKey(1234)
    l = AddCountLayer()

    if stream_constants:
      constants = {'test': test_utils.random_sequence(2, 15)}
    else:
      constants = {'test': jnp.array([42, 42])}

    x = test_utils.random_sequence(2, 15, *channel_shape)
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        stream_constants=stream_constants,
    )

    for t in range(15, 20):
      for blocks_per_step in range(1, 4):
        if stream_constants:
          constants['test'] = test_utils.random_sequence(2, t)
        x = test_utils.random_sequence(2, t, *channel_shape)
        y_layer, y_layer_emits = l.layer_with_emits(
            x, training=False, constants=constants
        )
        y_step, _, y_step_emits = utils.step_by_step_static(
            l,
            x,
            training=False,
            blocks_per_step=blocks_per_step,
            constants=constants,
            stream_constants=stream_constants,
        )
        self.assertSequencesClose(y_step_emits['x'], y_layer_emits['x'])
        trim = y_layer_emits['count'].shape[1]
        chex.assert_trees_all_equal(
            y_step_emits['count'][:, :trim], y_layer_emits['count']
        )
        self.assertSequencesClose(y_step.mask_invalid(), y_layer.mask_invalid())

  @parameterized.parameters(
      ((), False),
      ((5,), False),
      ((5, 7), False),
      ((5, 7, 11), False),
      ((5,), True),
  )
  def test_step_by_step_dynamic(
      self,
      channel_shape,
      stream_constants,
  ):
    key = jax.random.PRNGKey(1234)
    l = AddCountLayer()

    if stream_constants:
      constants = {'test': test_utils.random_sequence(2, 15)}
    else:
      constants = {'test': jnp.array([42, 42])}

    x = test_utils.random_sequence(2, 15, *channel_shape)
    l = self.init_and_bind_layer(key, l, x, constants=constants)
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        stream_constants=stream_constants,
    )

    for t in range(15, 20):
      for blocks_per_step in range(1, 4):
        if stream_constants:
          constants['test'] = test_utils.random_sequence(2, t)
        x = test_utils.random_sequence(2, t, *channel_shape)
        y_layer, y_layer_emits = l.layer_with_emits(x, training=False)
        y_step, _, y_step_emits = utils.step_by_step_dynamic(
            l, x, training=False, blocks_per_step=blocks_per_step
        )
        self.assertSequencesClose(y_step_emits['x'], y_layer_emits['x'])
        trim = y_layer_emits['count'].shape[1]
        chex.assert_trees_all_equal(
            y_step_emits['count'][:, :trim], y_layer_emits['count']
        )
        self.assertSequencesClose(y_step.mask_invalid(), y_layer.mask_invalid())


class ShardInitializerTest(test_utils.SequenceLayerTest):

  def test_no_sharding_or_kwargs(self):
    key = jax.random.PRNGKey(1234)
    initializer = utils.shard_initializer(nn.initializers.normal(), None)
    init = initializer(key, (), jnp.float32)
    self.assertIsInstance(init, jax.Array)

  def test_sharding(self):
    key = jax.random.PRNGKey(1234)
    initializer = utils.shard_initializer(
        nn.initializers.normal(), ('data', None, 'model')
    )
    init = initializer(key, (2, 3, 5), jnp.float32)
    self.assertIsInstance(init, meta.Partitioned)
    self.assertEqual(
        init.get_partition_spec(),
        jax.sharding.PartitionSpec('data', None, 'model'),
    )

  def test_axes_types_no_sharding(self):
    key = jax.random.PRNGKey(1234)
    initializer = utils.shard_initializer(
        nn.initializers.normal(),
        None,
        projectable=True,
        axes_types=(meta.AxisType.FANIN, None, None),
    )
    init = initializer(key, (2, 3, 5), jnp.float32)
    self.assertIsInstance(init, meta.Partitioned)


class SequenceBroadcastTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      [
          dict(bt_shapes=((2, 4), (2, 4), (1, 1)), expected_bt_shape=(2, 4)),
          dict(bt_shapes=((1, 2), (3, 1), (1, 2)), expected_bt_shape=(3, 2)),
      ],
      [
          dict(channels=(tuple(), tuple()), expected_channels=tuple()),
          dict(channels=(tuple(), (5,)), expected_channels=(5,)),
          dict(channels=(tuple(), (2, 5)), expected_channels=(2, 5)),
          dict(channels=((2,), tuple()), expected_channels=(2,)),
          dict(channels=((2, 5), tuple()), expected_channels=(2, 5)),
          dict(channels=((5,), (5,)), expected_channels=(5,)),
          dict(channels=((5,), (2, 5)), expected_channels=(2, 5)),
          dict(channels=((2, 5), (5,)), expected_channels=(2, 5)),
          dict(channels=((3, 1, 5), (2, 5)), expected_channels=(3, 2, 5)),
          dict(channels=((2, 5), (3, 1, 5)), expected_channels=(3, 2, 5)),
          dict(channels=(tuple(), tuple(), tuple()), expected_channels=tuple()),
          dict(channels=((5,), (1, 5), (2, 1, 1)), expected_channels=(2, 1, 5)),
          dict(
              channels=((3, 1, 2), (1, 4, 2), (3, 1, 1)),
              expected_channels=(3, 4, 2),
          ),
      ],
  )
  def test_sequence_broadcast_add(
      self,
      bt_shapes,
      expected_bt_shape,
      channels,
      expected_channels,
  ):
    x_bt, y_bt, z_bt = bt_shapes
    x_ch = channels[0]
    y_ch = channels[1]
    z_ch = channels[2] if len(channels) == 3 else None

    seqs = []
    seqs.append(test_utils.random_sequence(*x_bt, *x_ch))
    seqs.append(test_utils.random_sequence(*y_bt, *y_ch))
    if z_ch is not None:
      seqs.append(test_utils.random_sequence(*z_bt, *z_ch))

    output = utils.sequence_broadcast_add(*seqs)
    self.assertEqual(output.shape[:2], expected_bt_shape)
    self.assertEqual(output.channel_shape, expected_channels)

    self.assertEqual(
        utils.sequence_broadcast_combine_output_channel_shape(
            utils.CombinationMode.ADD, *(seq.channel_shape for seq in seqs)
        ),
        expected_channels,
    )

  @parameterized.product(
      [
          dict(bt_shapes=((2, 4), (2, 4), (1, 1)), expected_bt_shape=(2, 4)),
          dict(bt_shapes=((1, 2), (3, 1), (1, 2)), expected_bt_shape=(3, 2)),
      ],
      [
          dict(channels=(tuple(), tuple()), expected_channels=tuple()),
          dict(channels=(tuple(), (5,)), expected_channels=(5,)),
          dict(channels=(tuple(), (2, 5)), expected_channels=(2, 5)),
          dict(channels=((2,), tuple()), expected_channels=(2,)),
          dict(channels=((2, 5), tuple()), expected_channels=(2, 5)),
          dict(channels=((5,), (5,)), expected_channels=(5,)),
          dict(channels=((5,), (2, 5)), expected_channels=(2, 5)),
          dict(channels=((2, 5), (5,)), expected_channels=(2, 5)),
          dict(channels=((3, 1, 5), (2, 5)), expected_channels=(3, 2, 5)),
          dict(channels=((2, 5), (3, 1, 5)), expected_channels=(3, 2, 5)),
          # 3-input cases
          dict(channels=(tuple(), tuple(), tuple()), expected_channels=tuple()),
          dict(channels=((5,), (1, 5), (2, 1, 1)), expected_channels=(2, 1, 5)),
          dict(
              channels=((3, 1, 2), (1, 4, 2), (3, 1, 1)),
              expected_channels=(3, 4, 2),
          ),
      ],
  )
  def test_sequence_broadcast_mean(
      self,
      bt_shapes,
      expected_bt_shape,
      channels,
      expected_channels,
  ):
    x_bt, y_bt, z_bt = bt_shapes
    x_ch = channels[0]
    y_ch = channels[1]
    z_ch = channels[2] if len(channels) == 3 else None

    seqs = []
    seqs.append(test_utils.random_sequence(*x_bt, *x_ch))
    seqs.append(test_utils.random_sequence(*y_bt, *y_ch))
    if z_ch is not None:
      seqs.append(test_utils.random_sequence(*z_bt, *z_ch))

    output = utils.sequence_broadcast_mean(*seqs)
    self.assertEqual(output.shape[:2], expected_bt_shape)
    self.assertEqual(output.channel_shape, expected_channels)

    self.assertEqual(
        utils.sequence_broadcast_combine_output_channel_shape(
            utils.CombinationMode.MEAN, *(seq.channel_shape for seq in seqs)
        ),
        expected_channels,
    )

  @parameterized.product(
      [
          dict(bt_shapes=((2, 4), (2, 4), (1, 1)), expected_bt_shape=(2, 4)),
          dict(bt_shapes=((1, 2), (3, 1), (1, 2)), expected_bt_shape=(3, 2)),
      ],
      [
          dict(channels=(tuple(), tuple()), expected_channels=(2,)),
          dict(channels=(tuple(), (5,)), expected_channels=(6,)),
          dict(channels=(tuple(), (2, 5)), expected_channels=(2, 6)),
          dict(channels=((2,), tuple()), expected_channels=(3,)),
          dict(channels=((2, 5), tuple()), expected_channels=(2, 6)),
          dict(channels=((5,), (7,)), expected_channels=(12,)),
          dict(channels=((5,), (2, 7)), expected_channels=(2, 12)),
          dict(channels=((2, 5), (7,)), expected_channels=(2, 12)),
          dict(channels=((3, 1, 5), (2, 7)), expected_channels=(3, 2, 12)),
          dict(channels=((2, 5), (3, 1, 7)), expected_channels=(3, 2, 12)),
          dict(channels=(tuple(), tuple(), tuple()), expected_channels=(3,)),
          dict(channels=((5,), (1, 1), (2, 1, 2)), expected_channels=(2, 1, 8)),
          dict(
              channels=((3, 1, 2), (1, 4, 1), (3, 1, 3)),
              expected_channels=(3, 4, 6),
          ),
      ],
  )
  def test_sequence_broadcast_concat(
      self,
      bt_shapes,
      expected_bt_shape,
      channels,
      expected_channels,
  ):
    x_bt, y_bt, z_bt = bt_shapes
    x_ch = channels[0]
    y_ch = channels[1]
    z_ch = channels[2] if len(channels) == 3 else None

    seqs = []
    seqs.append(test_utils.random_sequence(*x_bt, *x_ch))
    seqs.append(test_utils.random_sequence(*y_bt, *y_ch))
    if z_ch is not None:
      seqs.append(test_utils.random_sequence(*z_bt, *z_ch))

    output = utils.sequence_broadcast_concat(*seqs)
    self.assertEqual(output.shape[:2], expected_bt_shape)
    self.assertEqual(output.channel_shape, expected_channels)

    self.assertEqual(
        utils.sequence_broadcast_combine_output_channel_shape(
            utils.CombinationMode.CONCAT, *(seq.channel_shape for seq in seqs)
        ),
        expected_channels,
    )

  @parameterized.product(
      [
          dict(bt_shapes=((2, 4), (2, 4), (1, 1)), expected_bt_shape=(2, 4)),
          dict(bt_shapes=((1, 2), (3, 1), (1, 2)), expected_bt_shape=(3, 2)),
      ],
      [
          dict(channels=(tuple(), tuple()), expected_channels=(2,)),
          dict(channels=(tuple(), (5,)), expected_channels=(2, 5)),
          dict(channels=(tuple(), (2, 5)), expected_channels=(2, 2, 5)),
          dict(channels=((2,), tuple()), expected_channels=(2, 2)),
          dict(channels=((2, 5), tuple()), expected_channels=(2, 2, 5)),
          dict(channels=((3, 5), (3, 5)), expected_channels=(2, 3, 5)),
          dict(channels=((5, 1), (1, 7)), expected_channels=(2, 5, 7)),
          dict(channels=((5,), (2, 5)), expected_channels=(2, 2, 5)),
          dict(channels=((2, 5), (5,)), expected_channels=(2, 2, 5)),
          dict(channels=((3, 1, 5), (2, 5)), expected_channels=(2, 3, 2, 5)),
          dict(channels=((2, 5), (3, 1, 5)), expected_channels=(2, 3, 2, 5)),
          dict(channels=(tuple(), tuple(), tuple()), expected_channels=(3,)),
          dict(
              channels=((5,), (1, 5), (2, 1, 1)), expected_channels=(3, 2, 1, 5)
          ),
          dict(
              channels=((3, 1, 2), (1, 4, 2), (3, 1, 1)),
              expected_channels=(3, 3, 4, 2),
          ),
      ],
  )
  def test_sequence_broadcast_stack(
      self,
      bt_shapes,
      expected_bt_shape,
      channels,
      expected_channels,
  ):
    x_bt, y_bt, z_bt = bt_shapes
    x_ch = channels[0]
    y_ch = channels[1]
    z_ch = channels[2] if len(channels) == 3 else None

    seqs = []
    seqs.append(test_utils.random_sequence(*x_bt, *x_ch))
    seqs.append(test_utils.random_sequence(*y_bt, *y_ch))
    if z_ch is not None:
      seqs.append(test_utils.random_sequence(*z_bt, *z_ch))

    output = utils.sequence_broadcast_stack(*seqs).mask_invalid()
    self.assertEqual(output.shape[:2], expected_bt_shape)
    self.assertEqual(output.channel_shape, expected_channels)

    self.assertEqual(
        utils.sequence_broadcast_combine_output_channel_shape(
            utils.CombinationMode.STACK, *(seq.channel_shape for seq in seqs)
        ),
        expected_channels,
    )

  @parameterized.product(
      [
          dict(bt_shapes=((2, 4), (2, 4), (1, 1)), expected_bt_shape=(2, 4)),
          dict(bt_shapes=((1, 2), (3, 1), (1, 2)), expected_bt_shape=(3, 2)),
      ],
      [
          dict(channels=(tuple(), tuple()), expected_channels=tuple()),
          dict(channels=(tuple(), (5,)), expected_channels=(5,)),
          dict(channels=(tuple(), (2, 5)), expected_channels=(2, 5)),
          dict(channels=((2,), tuple()), expected_channels=(2,)),
          dict(channels=((2, 5), tuple()), expected_channels=(2, 5)),
          dict(channels=((5,), (5,)), expected_channels=(5,)),
          dict(channels=((5,), (2, 5)), expected_channels=(2, 5)),
          dict(channels=((2, 5), (5,)), expected_channels=(2, 5)),
          dict(channels=((3, 1, 5), (2, 5)), expected_channels=(3, 2, 5)),
          dict(channels=((2, 5), (3, 1, 5)), expected_channels=(3, 2, 5)),
          dict(channels=(tuple(), tuple(), tuple()), expected_channels=tuple()),
          dict(channels=((5,), (1, 5), (2, 1, 1)), expected_channels=(2, 1, 5)),
          dict(
              channels=((3, 1, 2), (1, 4, 2), (3, 1, 1)),
              expected_channels=(3, 4, 2),
          ),
      ],
  )
  def test_sequence_broadcast_product(
      self,
      bt_shapes,
      expected_bt_shape,
      channels,
      expected_channels,
  ):
    x_bt, y_bt, z_bt = bt_shapes
    x_ch = channels[0]
    y_ch = channels[1]
    z_ch = channels[2] if len(channels) == 3 else None

    seqs = []
    seqs.append(test_utils.random_sequence(*x_bt, *x_ch))
    seqs.append(test_utils.random_sequence(*y_bt, *y_ch))
    if z_ch is not None:
      seqs.append(test_utils.random_sequence(*z_bt, *z_ch))

    output = utils.sequence_broadcast_product(*seqs)
    self.assertEqual(output.shape[:2], expected_bt_shape)
    self.assertEqual(output.channel_shape, expected_channels)

    self.assertEqual(
        utils.sequence_broadcast_combine_output_channel_shape(
            utils.CombinationMode.PRODUCT, *(seq.channel_shape for seq in seqs)
        ),
        expected_channels,
    )

  @parameterized.parameters(
      (utils.CombinationMode.STACK, (2, 3, 5)),
      (utils.CombinationMode.CONCAT, (3, 10)),
      (utils.CombinationMode.ADD, (3, 5)),
      (utils.CombinationMode.MEAN, (3, 5)),
      (utils.CombinationMode.PRODUCT, (3, 5)),
  )
  def test_sequence_broadcast_combine(self, mode, expected_channel_shape):
    batch_size, time = 2, 4
    x = test_utils.random_sequence(batch_size, time, 3, 5)
    y = test_utils.random_sequence(batch_size, time, 5)
    output = utils.sequence_broadcast_combine(mode, x, y)
    self.assertEqual(output.channel_shape, expected_channel_shape)
    self.assertEqual(
        utils.sequence_broadcast_combine_output_channel_shape(
            mode, x.channel_shape, y.channel_shape
        ),
        expected_channel_shape,
    )


class SequenceSplitTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      (
          dict(
              shape=(1, 1, 1, 3),
              indices_or_sections=3,
              axis=3,
              expected_shapes=[(1, 1, 1, 1)] * 3,
          ),
          dict(
              shape=(1, 1, 1, 1),
              indices_or_sections=1,
              axis=-1,
              expected_shapes=[(1, 1, 1, 1)],
          ),
          dict(
              shape=(2, 3, 4, 7),
              indices_or_sections=2,
              axis=2,
              expected_shapes=[(2, 3, 2, 7)] * 2,
          ),
          dict(
              shape=(2, 3, 4, 7),
              indices_or_sections=3,
              axis=1,
              expected_shapes=[(2, 1, 4, 7)] * 3,
          ),
          dict(
              shape=(9, 1),
              indices_or_sections=[2, 5, 9],
              axis=0,
              expected_shapes=[(2, 1), (3, 1), (4, 1), (0, 1)],
          ),
          dict(
              shape=(2, 3, 4, 7),
              indices_or_sections=[1, 3, 6],
              axis=-1,
              expected_shapes=[
                  (2, 3, 4, 1),
                  (2, 3, 4, 2),
                  (2, 3, 4, 3),
                  (2, 3, 4, 1),
              ],
          ),
      ),
      expected_seq_type=(types.Sequence, types.MaskedSequence),
  )
  def test_sequence_split_result(
      self, shape, indices_or_sections, axis, expected_shapes, expected_seq_type
  ):
    seq = expected_seq_type.from_values(jnp.ones(shape))
    expected = [
        expected_seq_type.from_values(jnp.ones(shape))
        for shape in expected_shapes
    ]
    result = utils.sequence_split(seq, indices_or_sections, axis)
    with self.subTest('shapes'):
      self.assertEqual(len(result), len(expected))
      for expected_seq, result_seq in zip(expected, result):
        self.assertSequencesEqual(result_seq, expected_seq)
    with self.subTest('sequence_type'):
      for expected_seq, result_seq in zip(expected, result):
        self.assertEqual(type(expected_seq), type(result_seq))

  @parameterized.parameters(
      dict(shape=(1, 1, 1), axis=3),
      dict(shape=(1, 1, 1), axis=-4),
  )
  def test_sequence_split_raises_on_invalid_axis(self, shape, axis):
    seq = types.Sequence.from_values(jnp.zeros(shape))
    with self.assertRaisesRegex(ValueError, 'split'):
      utils.sequence_split(seq, 1, axis)

  @parameterized.parameters(
      dict(shape=(3, 1, 1), indices_or_sections=2, axis=0),
      dict(shape=(5, 1, 1), indices_or_sections=5, axis=-1),
  )
  def test_sequence_stack_raises_on_uneven_split(
      self, shape, indices_or_sections, axis
  ):
    seq = types.Sequence.from_values(jnp.zeros(shape))
    with self.assertRaisesRegex(ValueError, 'equal'):
      utils.sequence_split(seq, indices_or_sections, axis)


class SequenceStackTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      (
          dict(shapes=[(1, 1, 1)] * 3, axis=3, expected_shape=(1, 1, 1, 3)),
          dict(shapes=[(1, 1, 1)], axis=-1, expected_shape=(1, 1, 1, 1)),
          dict(
              shapes=tuple([(2, 3, 4)] * 7), axis=2, expected_shape=(2, 3, 7, 4)
          ),
      ),
      expected_seq_type=(types.Sequence, types.MaskedSequence),
  )
  def test_sequence_stack_result(
      self, shapes, axis, expected_shape, expected_seq_type
  ):
    seqs = [expected_seq_type.from_values(jnp.ones(shape)) for shape in shapes]
    expected = expected_seq_type.from_values(jnp.ones(expected_shape))
    result = utils.sequence_stack(seqs, axis)
    with self.subTest('shape'):
      self.assertSequencesEqual(result, expected)
    with self.subTest('sequence_type'):
      self.assertEqual(type(result), type(expected))

  @parameterized.parameters(
      dict(shapes=[(1, 3, 4), (1, 5, 4)], axis=2),
      dict(shapes=((1, 1), (2, 1)), axis=-1),
  )
  def test_sequence_stack_raises_on_mismatched_seqs(self, shapes, axis):
    seqs = [types.Sequence.from_values(jnp.zeros(shape)) for shape in shapes]
    with self.assertRaisesRegex(ValueError, 'dimensions'):
      utils.sequence_stack(seqs, axis)

  @parameterized.parameters(
      dict(shapes=[(1, 3, 5), (1, 3, 5)], axis=1),
      dict(shapes=[(1, 1, 1), (1, 1, 1), (1, 1, 1)], axis=0),
      dict(shapes=((1, 1, 1), (1, 1, 1), (1, 1, 1)), axis=-3),
  )
  def test_sequence_stack_raises_on_invalid_axis(self, shapes, axis):
    seqs = [types.Sequence.from_values(jnp.zeros(shape)) for shape in shapes]
    with self.assertRaisesRegex(ValueError, 'axis'):
      utils.sequence_stack(seqs, axis)

  def test_sequence_stack_raises_on_empty(self):
    with self.assertRaisesRegex(ValueError, '(?i)no'):
      utils.sequence_stack([], axis=-1)


class SequenceUnstackTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      (
          dict(shape=(1, 1, 1, 3), axis=3, expected_shapes=[(1, 1, 1)] * 3),
          dict(shape=(1, 1, 1, 1), axis=-1, expected_shapes=[(1, 1, 1)]),
          dict(shape=(2, 3, 7, 4), axis=2, expected_shapes=[(2, 3, 4)] * 7),
      ),
      expected_seq_type=(types.Sequence, types.MaskedSequence),
  )
  def test_sequence_unstack_result(
      self, shape, axis, expected_shapes, expected_seq_type
  ):
    seq = expected_seq_type.from_values(jnp.ones(shape))
    expected = [
        expected_seq_type.from_values(jnp.ones(shape))
        for shape in expected_shapes
    ]
    result = utils.sequence_unstack(seq, axis)
    with self.subTest('shapes'):
      self.assertEqual(len(result), len(expected))
      for expected_seq, result_seq in zip(expected, result):
        self.assertSequencesEqual(result_seq, expected_seq)
    with self.subTest('sequence_type'):
      for expected_seq, result_seq in zip(expected, result):
        self.assertEqual(type(expected_seq), type(result_seq))

  @parameterized.parameters(
      dict(shape=(1, 3, 5), axis=1),
      dict(shape=(1, 1, 1), axis=0),
      dict(shape=(1, 1, 1), axis=-2),
  )
  def test_sequence_stack_raises_on_invalid_axis(self, shape, axis):
    seq = types.Sequence.from_values(jnp.zeros(shape))
    with self.assertRaisesRegex(ValueError, 'axis'):
      utils.sequence_unstack(seq, axis)


class CastableTest(test_utils.SequenceLayerTest):

  def test_checkable(self):
    self.assertIsInstance(jnp.asarray([[3.14]]), utils.Castable)
    self.assertIsInstance(np.asarray([[3.14]]), utils.Castable)
    self.assertIsInstance(
        types.Sequence.from_values(jnp.asarray([[3.14]])), utils.Castable
    )
    self.assertNotIsInstance(3.14, utils.Castable)


class InAtLeastFp32Tests(test_utils.SequenceLayerTest):

  @parameterized.parameters(True, False)
  def test_basic(self, restore_dtypes):
    @utils.run_in_at_least_fp32(restore_dtypes=restore_dtypes)
    def fn(x: types.Sequence, y: jax.Array):
      with self.subTest('casts_values'):
        self.assertEqual(x.dtype, jnp.float32)
        self.assertEqual(y.dtype, jnp.float32)
      return x, y + 1.0

    in_seq = types.Sequence.from_values(jnp.array([[3.14]], dtype=jnp.bfloat16))
    in_arr = jnp.array([1.0], dtype=jnp.float16)

    out_seq, out_arr = fn(in_seq, in_arr)

    with self.subTest('return_value_dtypes'):
      self.assertEqual(
          out_seq.dtype, in_seq.dtype if restore_dtypes else jnp.float32
      )
      self.assertEqual(
          out_arr.dtype, in_arr.dtype if restore_dtypes else jnp.float32
      )
    with self.subTest('performs_fn'):
      self.assertEqual(out_arr.item(), 2.0)

  @parameterized.parameters(True, False)
  def test_supports_none_args(self, restore_dtypes):
    @utils.run_in_at_least_fp32(restore_dtypes=restore_dtypes)
    def fn(x: jax.Array, _: None):
      del _
      return x, 'this is a test'

    in_arr = jnp.array([1.0], dtype=jnp.float16)
    out_arr, aux_val = fn(in_arr, None)
    self.assertEqual(
        out_arr.dtype, in_arr.dtype if restore_dtypes else jnp.float32
    )
    self.assertIsInstance(aux_val, str)

  def test_raises_on_mismatched_args(self):
    @utils.run_in_at_least_fp32
    def fn(x: jax.Array):
      return x, x + 1.0

    in_arr = jnp.array([1.0], dtype=jnp.float16)
    with self.assertRaisesRegex(TypeError, 'number of outputs'):
      _, _ = fn(in_arr)

  def test_raises_on_unpromotable_args(self):
    @utils.run_in_at_least_fp32
    def fn(x: jax.Array, y: str):
      del y
      return x, 'test'

    in_arr = jnp.array([1.0], dtype=jnp.float16)
    with self.assertRaisesRegex(TypeError, 'Cannot promote'):
      _, _ = fn(in_arr, 'test')

  def test_raises_on_keyword_args(self):
    @utils.run_in_at_least_fp32
    def fn(x: jax.Array, y: jax.Array):
      return x, y + 1.0

    in_arr = jnp.array([1.0], dtype=jnp.float16)
    with self.assertRaisesRegex(TypeError, 'unexpected keyword argument'):
      _, _ = fn(in_arr, y=in_arr)

  def test_raises_on_uncastable_results_when_restoring(self):
    @utils.run_in_at_least_fp32(restore_dtypes=True)
    def fn(x: jax.Array, y: jax.Array):
      del y
      return x, 'test'

    in_arr = jnp.array([1.0], dtype=jnp.float16)
    with self.assertRaisesRegex(TypeError, 'Cannot cast'):
      _, _ = fn(in_arr, in_arr)

  def test_allow_uncastable_results_when_not_restoring(self):
    @utils.run_in_at_least_fp32(restore_dtypes=False)
    def fn(x: jax.Array, y: jax.Array):
      del y
      return x, 'test'

    in_arr = jnp.array([1.0], dtype=jnp.float16)
    _, y = fn(in_arr, in_arr)
    self.assertEqual(y, 'test')

  @parameterized.parameters(True, False)
  def test_conditional_casting(self, promote: bool):
    @utils.maybe_in_at_least_fp32(promote)
    def fn(x: types.Sequence, y: jax.Array):
      out_type = jnp.float32 if promote else jnp.float16
      self.assertEqual(x.dtype, out_type)
      self.assertEqual(y.dtype, out_type)
      return x, y

    in_seq = types.Sequence.from_values(jnp.array([[3.14]], dtype=jnp.float16))
    in_arr = jnp.array([1.0], dtype=jnp.float16)
    _, _ = fn(in_seq, in_arr)


class ShiftTest(test_utils.SequenceLayerTest):

  @parameterized.product(channel_shape=((), (2,), (2, 3)))
  def test_left_shift(self, channel_shape):
    batch_size = 3
    x = test_utils.random_sequence(
        batch_size, 16, *channel_shape, low_length=5, high_length=10
    )

    y = utils.left_shift(x, 0)
    self.assertSequencesEqual(y, x)
    self.assertEqual(y.shape[1], x.shape[1])

    y = utils.left_shift(x, 5)
    self.assertSequencesEqual(y, x[:, 5:])
    self.assertEqual(y.shape[1], x.shape[1])

    y = utils.left_shift(x, jnp.full([batch_size], 0))
    self.assertSequencesEqual(y, x)
    self.assertEqual(y.shape[1], x.shape[1])

    shifts = jnp.array([3, 10, 12])
    y = utils.left_shift(x, shifts)
    self.assertEqual(y.shape[1], x.shape[1])

    for i, shift in enumerate(shifts):
      y_i_expected = x[i : i + 1, shift:]
      self.assertSequencesEqual(y[i : i + 1, :], y_i_expected)

  @parameterized.product(channel_shape=((), (2,), (2, 3)))
  def test_right_shift(self, channel_shape):
    batch_size = 3
    x = test_utils.random_sequence(
        batch_size, 16, *channel_shape, low_length=5, high_length=10
    )

    y = utils.right_shift(x, 0)
    self.assertSequencesEqual(y, x)
    self.assertEqual(y.shape[1], x.shape[1])

    y = utils.right_shift(x, 5)
    self.assertSequencesEqual(y, x[:, :-5].pad_time(5, 0, valid=False))
    self.assertEqual(y.shape[1], x.shape[1])

    y = utils.right_shift(x, jnp.full([batch_size], 0))
    self.assertSequencesEqual(y, x)
    self.assertEqual(y.shape[1], x.shape[1])

    shifts = jnp.array([3, 10, 12])
    y = utils.right_shift(x, shifts)
    self.assertEqual(y.shape[1], x.shape[1])

    for i, shift in enumerate(shifts):
      y_i_expected = x[i : i + 1, :-shift].pad_time(shift, 0, valid=False)
      self.assertSequencesEqual(y[i : i + 1, :], y_i_expected)


class RaggedSequenceConcatTest(test_utils.SequenceLayerTest):

  def test_wrong_shape_dtype(self):
    x = test_utils.random_sequence(3, 5, 2)
    y = test_utils.random_sequence(2, 9, 2)

    with self.assertRaises(ValueError):
      utils.ragged_sequence_concat(x, y)

    x = test_utils.random_sequence(2, 5, 2)
    y = test_utils.random_sequence(2, 9, 3)

    with self.assertRaises(ValueError):
      utils.ragged_sequence_concat(x, y)

    x = test_utils.random_sequence(2, 5, 3)
    y = test_utils.random_sequence(2, 9, 3)
    utils.ragged_sequence_concat(x, y)

  @parameterized.product(channel_shape=((), (2,), (2, 3)))
  def test_ragged_sequence_concat(self, channel_shape):
    x = test_utils.random_sequence(3, 5, *channel_shape)
    y = test_utils.random_sequence(3, 9, *channel_shape)

    x_lengths = x.lengths()
    y_lengths = y.lengths()

    concat = utils.ragged_sequence_concat(x, y)

    for i in range(3):
      expected_i = types.Sequence.concatenate_sequences(
          [x[i : i + 1, : x_lengths[i]], y[i : i + 1, : y_lengths[i]]]
      )
      self.assertSequencesEqual(concat[i : i + 1, :], expected_i)


class MatchShapeAlongAxesTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      dict(
          channel_shape=(),
          axes=None,
          expected_shape=(),
      ),
      dict(
          channel_shape=(2,),
          axes=None,
          expected_shape=(2,),
      ),
      dict(
          channel_shape=(2,),
          axes=-1,
          expected_shape=(2,),
      ),
      dict(
          channel_shape=(2, 3),
          axes=0,
          expected_shape=(2, 1),
      ),
      dict(
          channel_shape=(2, 3),
          axes=(1,),
          expected_shape=(1, 3),
      ),
      dict(
          channel_shape=(2, 3, 7),
          axes=[2, 0],
          expected_shape=(2, 1, 7),
      ),
      dict(
          channel_shape=(2, 3, 7),
          axes=None,
          expected_shape=(2, 3, 7),
      ),
  )
  def test_returns_matched_shape(self, channel_shape, axes, expected_shape):
    result = utils.match_shape_along_axes(channel_shape, axes)
    self.assertEqual(result, expected_shape)

  @parameterized.parameters(
      ((), -1),
      ((2,), 1),
      ((2,), -2),
      ((2, 3), 2),
      ((2, 3), (-3,)),
      ((2, 3, 7), [0, 3]),
  )
  def test_raises_error_on_invalid_axes(self, channel_shape, axes):
    with self.assertRaises(ValueError):
      utils.match_shape_along_axes(channel_shape, axes)


class ConvolutionPaddingOutputSizeTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      padding=('valid', 'same', 'causal_valid', 'reverse_causal_valid'),
      kernel_size=(1, 2, 3),
      stride=(1, 2, 3),
      dilation_rate=(1, 2, 3),
  )
  def test_convolution_padding_output_size(
      self,
      padding: types.PaddingModeString,
      kernel_size: int,
      stride: int,
      dilation_rate: int,
  ):
    kernel = jnp.zeros((kernel_size, 1, 1))
    for i in range(10):
      with self.subTest(f'{i}'):
        x = jnp.zeros((1, i, 1))
        padding = utils.convolution_explicit_padding(
            padding, kernel_size, stride, dilation_rate
        )
        y = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=[stride],
            padding=(padding,),
            lhs_dilation=[1],
            rhs_dilation=[dilation_rate],
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        )
        self.assertEqual(
            utils.convolution_padding_output_size(
                i, padding, kernel_size, stride, dilation_rate
            ),
            y.shape[1],
        )


class ConvolutionExplicitPaddingTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      dict(
          padding='valid',
          kernel_size=3,
          dilation_rate=1,
          expected_padding=(0, 0),
      ),
      dict(
          padding='valid',
          kernel_size=3,
          dilation_rate=2,
          expected_padding=(0, 0),
      ),
      dict(
          padding='valid',
          kernel_size=3,
          dilation_rate=3,
          expected_padding=(0, 0),
      ),
      dict(
          padding='same',
          kernel_size=3,
          dilation_rate=1,
          expected_padding=(1, 1),
      ),
      dict(
          padding='same',
          kernel_size=3,
          dilation_rate=2,
          expected_padding=(2, 2),
      ),
      dict(
          padding='same',
          kernel_size=3,
          dilation_rate=3,
          expected_padding=(3, 3),
      ),
      dict(
          padding='causal_valid',
          kernel_size=3,
          dilation_rate=1,
          expected_padding=(2, 0),
      ),
      dict(
          padding='causal_valid',
          kernel_size=3,
          dilation_rate=2,
          expected_padding=(4, 0),
      ),
      dict(
          padding='causal_valid',
          kernel_size=3,
          dilation_rate=3,
          expected_padding=(6, 0),
      ),
      dict(
          padding='reverse_causal_valid',
          kernel_size=3,
          dilation_rate=1,
          expected_padding=(0, 2),
      ),
      dict(
          padding='reverse_causal_valid',
          kernel_size=3,
          dilation_rate=2,
          expected_padding=(0, 4),
      ),
      dict(
          padding='reverse_causal_valid',
          kernel_size=3,
          dilation_rate=3,
          expected_padding=(0, 6),
      ),
      dict(
          padding='causal',
          kernel_size=3,
          dilation_rate=1,
          expected_padding=(2, 0),
      ),
      dict(
          padding='causal',
          kernel_size=3,
          dilation_rate=2,
          expected_padding=(4, 0),
      ),
      dict(
          padding='causal',
          kernel_size=3,
          dilation_rate=3,
          expected_padding=(6, 0),
      ),
      dict(
          padding='reverse_causal',
          kernel_size=3,
          dilation_rate=1,
          expected_padding=(0, 2),
      ),
      dict(
          padding='reverse_causal',
          kernel_size=3,
          dilation_rate=2,
          expected_padding=(0, 4),
      ),
      dict(
          padding='reverse_causal',
          kernel_size=3,
          dilation_rate=3,
          expected_padding=(0, 6),
      ),
      dict(
          padding='semicausal',
          kernel_size=3,
          dilation_rate=1,
          stride=1,
          expected_padding=(2, 0),
      ),
      dict(
          padding='semicausal',
          kernel_size=3,
          dilation_rate=1,
          stride=2,
          expected_padding=(1, 1),
      ),
      dict(
          padding='semicausal',
          kernel_size=3,
          dilation_rate=1,
          stride=3,
          expected_padding=(0, 2),
      ),
      dict(
          padding='semicausal',
          kernel_size=3,
          dilation_rate=1,
          stride=4,
          expected_padding=(0, 2),
      ),
      dict(
          padding='semicausal',
          kernel_size=3,
          dilation_rate=2,
          stride=1,
          expected_padding=(4, 0),
      ),
      dict(
          padding='semicausal',
          kernel_size=3,
          dilation_rate=2,
          stride=3,
          expected_padding=(2, 2),
      ),
  )
  def test_convolution_explicit_padding(
      self,
      padding: types.PaddingModeString,
      kernel_size: int,
      dilation_rate: int,
      expected_padding: tuple[int, int],
      stride: int = 1,
  ):
    self.assertEqual(
        utils.convolution_explicit_padding(
            padding, kernel_size, stride, dilation_rate
        ),
        expected_padding,
    )


class BatchWhereTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (tuple(),),
      ((1,),),
      ((2, 3),),
  )
  def test_basic(self, channel_shape):
    cond = jnp.array([True, False, True])
    a = test_utils.random_sequence(3, 5, *channel_shape)
    b = test_utils.random_sequence(3, 5, *channel_shape)

    c = utils.batch_where(cond, a, b)

    self.assertSequencesEqual(c[0:1], a[0:1])
    self.assertSequencesEqual(c[1:2], b[1:2])
    self.assertSequencesEqual(c[2:3], a[2:3])


class SplitDimensionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 5), 0, (2, 1), (2, 1, 3, 5)),
      ((2, 3, 5), 1, (1, 3, 1), (2, 1, 3, 1, 5)),
      ((7, 11, 30), 2, (3, 5, 2), (7, 11, 3, 5, 2)),
  )
  def test_basic(self, shape, axis, split_shape, expected_shape):
    x = jax.random.normal(jax.random.PRNGKey(0), shape)
    y = utils.split_dimension(x, axis=axis, shape=split_shape)
    y_expected = jnp.reshape(x, expected_shape)
    self.assertAllEqual(y, y_expected)

  def test_error(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 5))
    with self.assertRaisesRegex(ValueError, 'incorrect number of elements'):
      utils.split_dimension(x, axis=1, shape=(2, 2))


class DTypeTest(test_utils.SequenceLayerTest):

  def test_is_floating(self):
    self.assertTrue(utils.is_floating(jnp.bfloat16))
    self.assertTrue(utils.is_floating(jnp.float16))
    self.assertTrue(utils.is_floating(jnp.float32))
    self.assertTrue(utils.is_floating(jnp.float64))
    self.assertFalse(utils.is_floating(jnp.int32))
    self.assertFalse(utils.is_floating(jnp.bool_))

    self.assertTrue(utils.is_floating(jnp.zeros((), jnp.bfloat16).dtype))
    self.assertTrue(utils.is_floating(jnp.zeros((), jnp.float16).dtype))
    self.assertTrue(utils.is_floating(jnp.zeros((), jnp.float32).dtype))
    self.assertTrue(utils.is_floating(jnp.zeros((), jnp.float64).dtype))
    self.assertFalse(utils.is_floating(jnp.zeros((), jnp.int32).dtype))
    self.assertFalse(utils.is_floating(jnp.zeros((), jnp.bool_).dtype))

  def test_is_integral(self):
    self.assertFalse(utils.is_integral(jnp.bfloat16))
    self.assertFalse(utils.is_integral(jnp.float16))
    self.assertFalse(utils.is_integral(jnp.float32))
    self.assertFalse(utils.is_integral(jnp.float64))
    self.assertTrue(utils.is_integral(jnp.int8))
    self.assertTrue(utils.is_integral(jnp.int16))
    self.assertTrue(utils.is_integral(jnp.int32))
    self.assertTrue(utils.is_integral(jnp.int64))
    self.assertFalse(utils.is_integral(jnp.bool_))

    self.assertFalse(utils.is_integral(jnp.zeros((), jnp.bfloat16).dtype))
    self.assertFalse(utils.is_integral(jnp.zeros((), jnp.float16).dtype))
    self.assertFalse(utils.is_integral(jnp.zeros((), jnp.float32).dtype))
    self.assertFalse(utils.is_integral(jnp.zeros((), jnp.float64).dtype))
    self.assertTrue(utils.is_integral(jnp.zeros((), jnp.int8).dtype))
    self.assertTrue(utils.is_integral(jnp.zeros((), jnp.int16).dtype))
    self.assertTrue(utils.is_integral(jnp.zeros((), jnp.int32).dtype))
    self.assertTrue(utils.is_integral(jnp.zeros((), jnp.int64).dtype))
    self.assertFalse(utils.is_integral(jnp.zeros((), jnp.bool_).dtype))


class BatchedTimeSliceTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      channel_shape=[
          (),
          (3,),
          (3, 5),
      ],
      dtype=[jnp.float32, jnp.int32],
  )
  def test_batched_time_slice(self, channel_shape, dtype):
    x = test_utils.random_sequence(
        3, 16, *channel_shape, dtype=dtype, random_lengths=False
    )
    x = types.Sequence.from_lengths(
        x.values, jnp.array([3, 6, 9])
    ).mask_invalid()
    y = utils.batched_time_slice(x, jnp.array([0, 4, 9]), 3)
    self.assertAllEqual(y.lengths(), jnp.array([3, 2, 0]))
    self.assertSequencesEqual(y[0:1], x[0:1, 0:])
    self.assertSequencesEqual(y[1:2], x[1:2, 4:])
    self.assertSequencesEqual(y[2:3], x[2:3, 9:])


class GetConstantTest(test_utils.SequenceLayerTest):

  def test_get_constant_array(self):
    layer = test_utils.AssertConstantsLayer.Config().make()
    foo = jnp.zeros((2, 3, 5))
    constants = {'foo': foo}
    self.assertAllEqual(
        utils.get_constant_array(
            layer, constants, 'foo', (2, 3, 5), jnp.float32
        ),
        foo,
    )

  def test_get_constant_array_unpack(self):
    layer = test_utils.AssertConstantsLayer.Config().make()
    foo_seq = types.Sequence.from_values(jnp.zeros((2, 3, 5)))
    constants = {'foo_seq': foo_seq}
    self.assertAllEqual(
        utils.get_constant_array(
            layer,
            constants,
            'foo_seq',
            (2, 3, 5),
            jnp.float32,
            unpack_sequence=True,
        ),
        foo_seq.values,
    )

  def test_get_constant_sequence(self):
    layer = test_utils.AssertConstantsLayer.Config().make()
    foo_seq = types.Sequence.from_values(jnp.zeros((2, 3, 5)))
    constants = {'foo_seq': foo_seq}
    self.assertSequencesEqual(
        utils.get_constant_sequence(
            layer, constants, 'foo_seq', (2, 3, 5), jnp.float32
        ),
        foo_seq,
    )

  def test_errors(self):
    layer = test_utils.AssertConstantsLayer.Config().make()
    foo = jnp.zeros((2, 3, 5))
    foo_seq = types.Sequence.from_values(foo)

    constants = {'foo': foo, 'foo_seq': foo_seq}
    self.assertIsInstance(
        utils.get_constant_array(
            layer, constants, 'foo', (2, 3, 5), jnp.float32
        ),
        jax.Array,
    )
    self.assertIsInstance(
        utils.get_constant_sequence(
            layer, constants, 'foo_seq', (2, 3, 5), jnp.float32
        ),
        types.Sequence,
    )

    with self.assertRaises(ValueError):
      utils.get_constant_sequence(layer, None, 'foo', None, None)
    with self.assertRaises(ValueError):
      utils.get_constant_array(layer, None, 'foo', None, None)

    with self.assertRaises(ValueError):
      utils.get_constant_array(layer, {}, 'foo', None, None)
    with self.assertRaises(ValueError):
      utils.get_constant_sequence(layer, {}, 'foo_seq', None, None)

    with self.assertRaises(ValueError):
      utils.get_constant_array(layer, constants, 'foo', (2, 3, 6), None)
    with self.assertRaises(ValueError):
      utils.get_constant_sequence(layer, constants, 'foo_seq', (2, 3, 6), None)

    with self.assertRaises(ValueError):
      utils.get_constant_array(layer, constants, 'foo', (2, 3, 5), jnp.int32)
    with self.assertRaises(ValueError):
      utils.get_constant_sequence(
          layer, constants, 'foo_seq', (2, 3, 5), jnp.int32
      )


class ReceptiveFieldUtilsTest(test_utils.SequenceLayerTest):

  class _MockSequenceLayer:

    def __init__(self, receptive_field_per_step=None, output_ratio=None):
      self.receptive_field_per_step = receptive_field_per_step
      self.output_ratio = output_ratio

  @parameterized.named_parameters(
      (
          'both_none',
          None,
          None,
          None,
      ),
      (
          'first_none',
          None,
          (0, 1),
          (0, 1),
      ),
      (
          'second_none',
          (0, 1),
          None,
          (0, 1),
      ),
      (
          'overlap',
          (0, 2),
          (1, 3),
          (0, 3),
      ),
      (
          'disjoint',
          (0, 1),
          (1, 3),
          (0, 3),
      ),
      (
          'contained',
          (0, 3),
          (1, 2),
          (0, 3),
      ),
      (
          'negative_values',
          (-2, 0),
          (-1, 1),
          (-2, 1),
      ),
  )
  def test_receptive_field_union(self, rf_a, rf_b, expected_rf):
    self.assertEqual(utils.receptive_field_union(rf_a, rf_b), expected_rf)

  @parameterized.named_parameters(
      (
          'ratio_1_step_0',
          {0: (0, 0), 1: (-1, 1)},
          1,
          0,
          (0, 0),
      ),
      (
          'ratio_1_step_1',
          {0: (0, 0), 1: (-1, 1)},
          1,
          1,
          (-1, 1),
      ),
      (
          'ratio_2_upsample_step_0',
          {0: (0, 1), 1: (0, 1)},
          2,
          0,
          (0, 1),
      ),
      (
          'ratio_2_upsample_step_3',
          {0: (0, 1), 1: (0, 1)},
          2,
          3,
          (1, 2),  # Shifted (0,1) by (3-1)//2 = 1
      ),
      (
          'ratio_0.5_downsample_step_0',
          {0: (0, 2)},
          0.5,
          0,
          (0, 2),
      ),
      (
          'ratio_0.5_downsample_step_1',
          {0: (0, 2)},
          0.5,
          1,
          (2, 4),  # Shifted (0,2) by (1-0)//0.5 = 2
      ),
      (
          'rf_is_none',
          {0: None, 1: (0, 0)},
          1,
          0,
          None,
      ),
      (
          'rf_is_none_wrap_around',
          {0: (0, 0), 1: None},
          1,
          1,
          None,
      ),
      (
          'single_rf_entry_step_0',
          {0: (-2, 2)},
          1,
          0,
          (-2, 2),
      ),
      (
          'single_rf_entry_step_5',
          {0: (-2, 2)},
          1,
          5,
          (3, 7),  # Shifted (-2,2) by 5//1 = 5
      ),
  )
  def test_receptive_field_at(
      self, receptive_field_per_step, output_ratio, step, expected_rf
  ):

    mock_layer = typing.cast(
        types.SequenceLayer,
        self._MockSequenceLayer(receptive_field_per_step, output_ratio),
    )
    self.assertEqual(
        utils.layer_receptive_field_at(mock_layer, step), expected_rf
    )

  @parameterized.named_parameters(
      (
          'empty_rf_per_step',
          {},
          fractions.Fraction(1, 1),
          None,
      ),
      (
          'all_none_rf',
          {0: None, 1: None},
          fractions.Fraction(1, 1),
          None,
      ),
      (
          'single_rf',
          {0: (0, 1)},
          fractions.Fraction(1, 1),
          (0, 1),
      ),
      (
          'multiple_rf_overlap',
          {0: (0, 2), 1: (-1, 1)},
          fractions.Fraction(1, 1),
          (-2, 2),  # (0,2) -> (0,2), (-1,1) -> (-2,0). Union is (-2,2)
      ),
      (
          'multiple_rf_disjoint',
          {0: (0, 1), 1: (3, 4)},
          fractions.Fraction(1, 1),
          (
              0,
              3,
          ),  # (0,1) -> (0,1), (3,4) -> (2,3).
      ),
      (
          'multiple_rf_with_none',
          {0: (0, 1), 1: None, 2: (-2, -1)},
          fractions.Fraction(1, 1),
          (-4, 1),  # (0,1)->(0,1), (-2,-1)->(-4,-3). Union is (-4,1)
      ),
      (
          'output_ratio_2',
          {0: (0, 1), 1: (-1, 0)},
          fractions.Fraction(
              2, 1
          ),  # (0,1)->(0,1), (-1,0)->(-1,0). Union is (-1,1)
          (-1, 1),
      ),
      (
          'output_ratio_half',
          {0: (0, 1), 1: (-1, 0)},
          fractions.Fraction(1, 2),
          (
              -3,
              1,
          ),  # (0,1)->(0,1), (-1,0)->(-3,-2). Union is (-3,1).
      ),
  )
  def test_reduce_receptive_field_per_step(
      self, rf_per_step, output_ratio, expected_rf
  ):
    self.assertEqual(
        utils.reduce_receptive_field_per_step(rf_per_step, output_ratio),
        expected_rf,
    )

  @parameterized.named_parameters(
      (
          'single_layer',
          [{0: (0, 0), 1: (-1, 1)}],
          {0: (0, 0), 1: (-1, 1)},
      ),
      (
          'multiple_layers_same_steps',
          [
              {0: (0, 0), 1: (-2, 0)},
              {0: (-1, 1), 1: (0, 1)},
          ],
          {0: (-1, 1), 1: (-2, 1)},
      ),
      (
          'multiple_layers_different_steps',
          [
              {0: (0, 0)},
              {0: (-1, 1), 1: (0, 1)},
          ],
          {0: (-1, 1), 1: (0, 1)},
      ),
      (
          'layers_with_none',
          [
              {0: (0, 0), 1: None},
              {0: None, 1: (0, 1)},
          ],
          {0: (0, 0), 1: (0, 1)},
      ),
      ('empty_list', [], {}),
  )
  def test_aggregate_layers_receptive_field_per_steps(
      self,
      rf_per_step_list: list[dict[int, types.ReceptiveField]],
      expected_rf_dict: dict[int, types.ReceptiveField | None],
  ):
    layers = [
        typing.cast(
            types.SequenceLayer,
            self._MockSequenceLayer(receptive_field_per_step=rf_per_step),
        )
        for rf_per_step in rf_per_step_list
    ]
    self.assertEqual(
        utils.aggregate_layers_receptive_field_per_steps(layers),
        expected_rf_dict,
    )

  @parameterized.named_parameters(
      (
          'identity_propagation',
          {0: (0, 0)},
          _MockSequenceLayer({0: (-1, 0)}, fractions.Fraction(1, 1)),
          {0: (-1, 0)},
      ),
      (
          'wider_rf_next',
          {0: (-1, 1)},
          _MockSequenceLayer({0: (0, 0)}, fractions.Fraction(1, 1)),
          {0: (-1, 1)},
      ),
      (
          'prev_is_downsample',
          {0: (-1, 0)},
          _MockSequenceLayer({0: (0, 1)}, fractions.Fraction(1, 2)),
          {0: (-2, 1)},
      ),
      (
          'cnn_then_downsample',
          {0: (-2, 0)},
          _MockSequenceLayer(
              receptive_field_per_step={0: (0, 1)},
              output_ratio=fractions.Fraction(1, 2),
          ),
          {0: (-4, 1)},
      ),
      (
          'rf_next_is_none',
          {0: None},
          _MockSequenceLayer({0: (0, 1)}, fractions.Fraction(1, 2)),
          {0: None},
      ),
      (
          'all_prev_rf_at_are_none',
          {0: (0, 0)},
          _MockSequenceLayer({0: None}, fractions.Fraction(1, 1)),
          {0: None},
      ),
  )
  def test_propagate_receptive_field_to_prev_layer(
      self,
      layer_rf_per_step_next: dict[int, types.ReceptiveField | None],
      layer_prev: _MockSequenceLayer,
      expected_rf_dict: dict[int, types.ReceptiveField | None],
  ):
    layer_prev = typing.cast(types.SequenceLayer, layer_prev)
    self.assertEqual(
        utils.propagate_receptive_field_to_prev_layer(
            layer_rf_per_step_next,
            layer_prev.receptive_field_per_step,
            layer_prev.output_ratio,
        ),
        expected_rf_dict,
    )

  @parameterized.named_parameters(
      ('empty_layers', [], {0: (0, 0)}),
      (
          'single_layer',
          [_MockSequenceLayer({0: (-1, 0)}, fractions.Fraction(1, 1))],
          {0: (-1, 0)},
      ),
      (
          'identity_plus_cnn',
          [
              _MockSequenceLayer({0: (0, 0)}, fractions.Fraction(1, 1)),
              _MockSequenceLayer({0: (-1, 0)}, fractions.Fraction(1, 1)),
          ],
          {0: (-1, 0)},
      ),
      (
          'downsample_plus_cnn',
          [
              _MockSequenceLayer(
                  {0: (0, 1)}, fractions.Fraction(1, 2)
              ),  # DS(2)
              _MockSequenceLayer({0: (-2, 0)}, fractions.Fraction(1, 1)),
          ],
          {0: (-4, 1)},
      ),
      (
          'layers_with_different_rf_len',
          [
              _MockSequenceLayer(
                  {0: (0, 0), 1: (0, 0)}, fractions.Fraction(1, 1)
              ),
              _MockSequenceLayer({0: (-1, 0)}, fractions.Fraction(1, 1)),
          ],
          {0: (-2, 0), 1: (0, 0)},
      ),
  )
  def test_receptive_field_per_step_of_serial_layers(
      self,
      layers: list[_MockSequenceLayer],
      expected_rf_dict: dict[int, types.ReceptiveField | None],
  ):
    layers = [typing.cast(types.SequenceLayer, layer) for layer in layers]
    self.assertEqual(
        utils.receptive_field_per_step_of_serial_layers(layers),
        expected_rf_dict,
    )


if __name__ == '__main__':
  test_utils.main()
