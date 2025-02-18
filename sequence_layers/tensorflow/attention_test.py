# Copyright 2023 Google LLC
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
"""Tests for sequence_layers.tensorflow.attention."""

import itertools

from absl.testing import parameterized
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class AdditiveAttentionTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (1, 2),
      (3, 5),
  )
  def test_additive_attention(self, num_heads, units_per_head):
    x = None
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.AdditiveAttention(
          source_name, num_heads=num_heads, units_per_head=units_per_head
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}

    channels = 3
    for time in range(5 * l.block_size, 7 * l.block_size):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants),
          tf.TensorShape([num_heads, source_channels]),
      )
      self.verify_contract(l, x, training=False, constants=constants)
      # Projection for queries / keys, plus a combination matrix.
      self.assertLen(l.variables, 4)
      self.assertLen(l.trainable_variables, 4)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/additive_attention/v:0',
              'test/additive_attention/query_projection/kernel:0',
              'test/additive_attention/source_projection/kernel:0',
              'test/additive_attention/source_projection/bias:0',
          ],
      )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  def test_additive_attention_emit_outputs(self):
    num_heads, units_per_head = 3, 5
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = sl.AdditiveAttention(
        source_name, num_heads=num_heads, units_per_head=units_per_head
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}
    time, channels = 7, 3
    x = self.random_sequence(batch_size, time, channels)
    _, emits = l.layer_with_emits(x, training=False, constants=constants)
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, time, num_heads, source_time],
    )

    # Only run three timesteps of the sequence.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(
        x,
        l.get_initial_state(x, constants=constants),
        training=False,
        constants=constants,
    )
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, 3, num_heads, source_time],
    )

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_additive_attention_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.AdditiveAttention(source_name, num_heads=3, units_per_head=5)

    source = self.random_sequence(
        batch_size, source_time, source_channels, dtype=utils.compute_dtype()
    )
    constants = {source_name: source}

    x = self.random_sequence(batch_size, 5, 3, dtype=utils.compute_dtype())
    _, y_np = self.verify_contract(l, x, training=True, constants=constants)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


class GmmAttentionTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(
      (1, 2, True),
      (1, 2, False),
      (3, 5, True),
      (3, 5, False),
  )
  def test_gmm_attention(self, num_heads, units_per_head, monotonic):
    x = None
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.GmmAttention(
          source_name,
          num_heads=num_heads,
          units_per_head=units_per_head,
          num_components=5,
          monotonic=monotonic,
          init_offset_bias=1.0,
          init_scale_bias=1.0,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}

    channels = 3
    for time in range(5 * l.block_size, 7 * l.block_size):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants),
          tf.TensorShape([num_heads, source_channels]),
      )
      self.verify_contract(l, x, training=False, constants=constants)
      # Two dense layers for computing GMM parameters.
      self.assertLen(l.variables, 4)
      self.assertLen(l.trainable_variables, 4)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/gmm_attention/gmm_mlp_hidden/kernel:0',
              'test/gmm_attention/gmm_mlp_hidden/bias:0',
              'test/gmm_attention/gmm_mlp_output/kernel:0',
              'test/gmm_attention/gmm_mlp_output/bias:0',
          ],
      )

    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  def test_gmm_attention_emit_outputs(self):
    num_heads, units_per_head, monotonic = 3, 5, True
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = sl.GmmAttention(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        num_components=5,
        monotonic=monotonic,
        init_offset_bias=1.0,
        init_scale_bias=1.0,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}
    time, channels = 7, 3
    x = self.random_sequence(batch_size, time, channels)
    _, emits = l.layer_with_emits(x, training=False, constants=constants)
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, time, num_heads, source_time],
    )

    # Only run three timesteps of the sequence.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(
        x,
        l.get_initial_state(x, constants=constants),
        training=False,
        constants=constants,
    )
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, 3, num_heads, source_time],
    )

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_gmm_attention_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.GmmAttention(
          source_name,
          num_heads=3,
          units_per_head=5,
          num_components=5,
          monotonic=True,
      )

    source = self.random_sequence(
        batch_size, source_time, source_channels, dtype=utils.compute_dtype()
    )
    constants = {source_name: source}

    x = self.random_sequence(batch_size, 5, 3, dtype=utils.compute_dtype())
    _, y_np = self.verify_contract(l, x, training=True, constants=constants)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


class DotProductAttentionTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (1, 2),
      (3, 5),
  )
  def test_dot_product_attention(self, num_heads, units_per_head):
    x = None
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.DotProductAttention(
          source_name, num_heads=num_heads, units_per_head=units_per_head
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}

    channels = 3
    for time in range(5 * l.block_size, 7 * l.block_size):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants),
          tf.TensorShape([num_heads, units_per_head]),
      )
      self.verify_contract(l, x, training=False, constants=constants)
      # Two bias-less dense layers for q and kv projections.
      self.assertLen(l.variables, 2)
      self.assertLen(l.trainable_variables, 2)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/dot_product_attention/key_value_projection/kernel:0',
              'test/dot_product_attention/query_projection/kernel:0',
          ],
      )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  def test_dot_product_attention_emit_outputs(self):
    num_heads, units_per_head = 3, 5
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = sl.DotProductAttention(
        source_name, num_heads=num_heads, units_per_head=units_per_head
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}
    time, channels = 7, 3
    x = self.random_sequence(batch_size, time, channels)
    _, emits = l.layer_with_emits(x, training=False, constants=constants)
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, time, num_heads, source_time],
    )

    # Only run three timesteps of the sequence.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(
        x,
        l.get_initial_state(x, constants=constants),
        training=False,
        constants=constants,
    )
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, 3, num_heads, source_time],
    )

  @parameterized.parameters(True, False)
  def test_dot_product_attention_dropout(
      self, broadcast_dropout_across_queries
  ):
    source_name = 'source'
    l = sl.DotProductAttention(
        source_name,
        num_heads=3,
        units_per_head=5,
        attention_probabilities_dropout_rate=0.99999,
        broadcast_dropout_across_queries=broadcast_dropout_across_queries,
    )
    source = self.random_sequence(2, 4, 5)
    constants = {source_name: source}
    x = self.random_sequence(2, 12, 3, random_mask=True)
    y_dropout = l.layer(x, training=True, constants=constants)
    y_no_dropout = l.layer(x, training=False, constants=constants)
    y_no_dropout2 = l.layer(x, training=False, constants=constants)

    y_dropout, y_no_dropout, y_no_dropout2 = self.evaluate(
        [y_dropout, y_no_dropout, y_no_dropout2]
    )

    self.assertAllEqual(y_dropout.values, tf.zeros_like(y_dropout.values))
    self.assertSequencesClose(y_no_dropout, y_no_dropout2)

  def test_dot_product_attention_logits_soft_cap(self):
    num_heads, units_per_head = 3, 5
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.DotProductAttention(
          source_name,
          num_heads=num_heads,
          units_per_head=units_per_head,
          attention_logits_soft_cap=50.0,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}

    time, channels = 6 * l.block_size, 3
    x = self.random_sequence(batch_size, time, channels)
    self.assertEqual(
        l.get_output_shape_for_sequence(x, constants),
        tf.TensorShape([num_heads, units_per_head]),
    )
    self.verify_contract(l, x, training=False, constants=constants)
    # Two bias-less dense layers for q and kv projections.
    self.assertLen(l.variables, 2)
    self.assertLen(l.trainable_variables, 2)
    self.assertCountEqual(
        [v.name for v in l.variables],
        [
            'test/dot_product_attention/key_value_projection/kernel:0',
            'test/dot_product_attention/query_projection/kernel:0',
        ],
    )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_dot_product_attention_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.DotProductAttention(source_name, num_heads=3, units_per_head=5)

    source = self.random_sequence(
        batch_size, source_time, source_channels, dtype=utils.compute_dtype()
    )
    constants = {source_name: source}

    x = self.random_sequence(batch_size, 5, 3, dtype=utils.compute_dtype())
    rtol, atol = test_util.rtol_atol_for_dtype(x.values.dtype)
    _, y_np = self.verify_contract(
        l, x, training=True, constants=constants, rtol=rtol, atol=atol
    )
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)

  @parameterized.parameters(
      itertools.product(
          ('mixed_bfloat16', 'float32'),
          (None, 2),
      )
  )
  def test_dot_product_attention_chunked(
      self,
      policy: str,
      key_chunk_size: int | None,
  ):
    x = None
    constants = {}
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    with test_util.keras_precision_policy_scope(policy):
      compute_dtype = utils.compute_dtype()
      batch_size, num_heads, units_per_head, source_channels = 2, 3, 5, 2
      query_chunk_size = 3
      source_name = 'source'

      with tf.name_scope('test'):
        l = sl.DotProductAttention(
            source_name,
            num_heads=num_heads,
            units_per_head=units_per_head,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
        )
      self.assertEqual(l.block_size, 1)
      self.assertEqual(l.output_ratio, 1)

      channels = 12
      # Sweep from one less than a multiple of key_chunk_size to one more
      # than a multiple of key_chunk_size.
      for source_time in range(10, 13):
        source = self.random_sequence(
            batch_size, source_time, source_channels, dtype=compute_dtype
        )
        constants = {source_name: source}

        # Sweep from one less than a multiple of query_chunk_size to one more
        # than a multiple of query_chunk_size.
        for time in range(5 * l.block_size, 8 * l.block_size):
          x = self.random_sequence(
              batch_size, time, channels, random_mask=False, dtype=compute_dtype
          )
          self.assertEqual(
              l.get_output_shape_for_sequence(x, constants),
              tf.TensorShape([num_heads, units_per_head]),
          )

          y_layer_chunked = l.layer(x, training=False, constants=constants)
          l._query_chunk_size = None
          l._key_chunk_size = None
          y_layer_no_chunk = l.layer(x, training=False, constants=constants)
          l._query_chunk_size = query_chunk_size
          l._key_chunk_size = key_chunk_size

          self.assertEqual(
              y_layer_chunked.values.shape,
              [batch_size, time, num_heads, units_per_head],
          )

          self.assertEqual(
              y_layer_no_chunk.values.shape,
              [batch_size, time, num_heads, units_per_head],
          )

          rtol, atol = test_util.rtol_atol_for_dtype(compute_dtype)
          self.assertSequencesClose(
              y_layer_chunked, y_layer_no_chunk, rtol=rtol, atol=atol
          )
          self.verify_contract(
              l, x, training=False, constants=constants, rtol=rtol, atol=atol
          )
          # Two bias-less dense layers for q and kv projections.
          self.assertLen(l.variables, 2)
          self.assertLen(l.trainable_variables, 2)
          self.assertCountEqual(
              [v.name for v in l.variables],
              [
                  'test/dot_product_attention/key_value_projection/kernel:0',
                  'test/dot_product_attention/query_projection/kernel:0',
              ],
          )
      # Use flex for Einsum.
      if policy == 'float32':
        self.verify_tflite_step(l, x, constants=constants, use_flex=True)


class DotProductSelfAttentionTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (1, 2),
      (3, 5),
  )
  def test_dot_product_attention_query_key_value_network(
      self, num_heads, units_per_head
  ):
    x = None
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.DotProductAttention(
          source_name,
          num_heads=num_heads,
          units_per_head=units_per_head,
          query_network=sl.AddTimingSignal(),
          key_network=sl.AddTimingSignal(),
          # Doesn't really make sense, but just testing something stateful.
          value_network=sl.AddTimingSignal(),
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}

    channels = 3
    for time in range(5 * l.block_size, 7 * l.block_size):
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants),
          tf.TensorShape([num_heads, units_per_head]),
      )
      self.verify_contract(l, x, training=False, constants=constants)
      # Two bias-less dense layers for q and kv projections.
      self.assertLen(l.variables, 2)
      self.assertLen(l.trainable_variables, 2)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/dot_product_attention/key_value_projection/kernel:0',
              'test/dot_product_attention/query_projection/kernel:0',
          ],
      )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  def test_dot_product_attention_query_key_value_network_supports_step(self):
    l = sl.DotProductAttention(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=sl.AddTimingSignal(),
        key_network=sl.AddTimingSignal(),
        value_network=sl.AddTimingSignal(),
    )
    self.assertTrue(l.supports_step)

    l = sl.DotProductAttention(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=test_util.NonSteppableLayer(),
        key_network=sl.AddTimingSignal(),
        value_network=sl.AddTimingSignal(),
    )
    self.assertFalse(l.supports_step)

    l = sl.DotProductAttention(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=sl.AddTimingSignal(),
        key_network=test_util.NonSteppableLayer(),
        value_network=sl.AddTimingSignal(),
    )
    # Even if key / value network not steppable, we can still step.
    self.assertTrue(l.supports_step)

    l = sl.DotProductAttention(
        'source',
        num_heads=3,
        units_per_head=5,
        query_network=sl.AddTimingSignal(),
        key_network=sl.AddTimingSignal(),
        value_network=test_util.NonSteppableLayer(),
    )
    # Even if key / value network not steppable, we can still step.
    self.assertTrue(l.supports_step)

  @parameterized.parameters(
      # max_past_horizon > 0, max_future_horizon == 0
      (1, 2, 3, 0, False, False),
      (3, 5, 3, 0, False, False),
      (1, 2, 3, 0, True, False),
      (3, 5, 3, 0, True, False),
      (1, 2, 3, 0, False, True),
      (3, 5, 3, 0, False, True),
      (1, 2, 3, 0, True, True),
      (3, 5, 3, 0, True, True),
      # max_past_horizon > 0, max_future_horizon > 0
      (3, 5, 3, 2, False, False),
      (3, 5, 3, 2, False, True),
      # max_past_horizon == -1, max_future_horizon > 0
      (3, 5, -1, 2, False, False),
      (3, 5, -1, 2, False, True),
      # max_past_horizon > 0, max_future_horizon == -1
      (3, 5, 3, -1, False, False),
      (3, 5, 3, -1, False, True),
      # max_past_horizon == -1, max_future_horizon == -1 (unmasked)
      (3, 5, -1, -1, False, False),
      (3, 5, -1, -1, False, True),
  )
  def test_dot_product_self_attention(
      self,
      num_heads,
      units_per_head,
      max_past_horizon,
      max_future_horizon,
      relative,
      random_mask,
  ):
    x = None
    batch_size = 2
    with tf.name_scope('test'):
      l = sl.DotProductSelfAttention(
          num_heads=num_heads,
          units_per_head=units_per_head,
          max_horizon=max_past_horizon,
          max_future_horizon=max_future_horizon,
          use_relative_position_embedding=relative,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    channels = 1
    # Sweep time dimension shorter and longer than max_horizon.
    for time in range(1, 6):
      with self.subTest(f'time{time}'):
        x = self.random_sequence(
            batch_size, time, channels, random_mask=random_mask
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x),
            tf.TensorShape([num_heads, units_per_head]),
        )
        self.verify_contract(l, x, training=False)
        # A bias-less dense layer for qkv projection.
        self.assertLen(l.variables, 2 if relative else 1)
        self.assertLen(l.trainable_variables, 2 if relative else 1)
        self.assertCountEqual(
            [v.name for v in l.variables],
            [
                'test/dot_product_self_attention/query_key_value_projection/kernel:0'
            ]
            + (
                [
                    'test/dot_product_self_attention/shaw_relative_position_embedding/embedding:0'
                ]
                if relative
                else []
            ),
        )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, use_flex=True)

  def test_dot_product_self_attention_logits_soft_cap(self):
    batch_size, num_heads, units_per_head = 2, 3, 5

    with tf.name_scope('test'):
      l = sl.DotProductSelfAttention(
          num_heads=num_heads,
          units_per_head=units_per_head,
          max_horizon=3,
          max_future_horizon=10,
          attention_logits_soft_cap=50.0,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    time, channels = 5, 1

    x = self.random_sequence(batch_size, time, channels, random_mask=False)
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        tf.TensorShape([num_heads, units_per_head]),
    )
    self.verify_contract(l, x, training=False)
    # A bias-less dense layer for qkv projection.
    self.assertLen(l.variables, 1)
    self.assertLen(l.trainable_variables, 1)
    self.assertCountEqual(
        [v.name for v in l.variables],
        ['test/dot_product_self_attention/query_key_value_projection/kernel:0'],
    )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(
      # Fully unmasked.
      (-1, -1, 8, True, 16),
      (-1, -1, 8, False, 16),
      # Causal, fully unmasked past.
      (-1, 0, 8, True, 16),
      (-1, 0, 8, False, 16),
      # No past, fully unmasked future.
      (0, -1, 8, True, 16),
      # Limited past and future.
      (4, 4, 8, True, 16),
      (4, 4, 8, False, 16),
      # Causal, limited past context (streaming capable).
      (8, 0, 8, False, 16),  # Causal / streaming.
  )
  def test_t5_relative_position_bias(
      self,
      max_past_horizon,
      max_future_horizon,
      num_buckets,
      bidirectional,
      max_distance,
  ):
    x = None
    batch_size, num_heads, units_per_head = 2, 3, 5
    with tf.name_scope('test'):
      relative_embedding = sl.T5RelativePositionEmbedding(
          num_buckets=num_buckets,
          bidirectional=bidirectional,
          max_distance=max_distance,
          num_heads=num_heads,
      )
      l = sl.DotProductSelfAttention(
          num_heads=num_heads,
          units_per_head=units_per_head,
          max_horizon=max_past_horizon,
          max_future_horizon=max_future_horizon,
          relative_position_embedding=relative_embedding,
          use_relative_position_embedding=True,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    channels = 1
    for time in range(15, 20):
      with self.subTest(f'time{time}'):
        x = self.random_sequence(batch_size, time, channels, random_mask=False)
        self.assertEqual(
            l.get_output_shape_for_sequence(x),
            tf.TensorShape([num_heads, units_per_head]),
        )
        self.verify_contract(l, x, training=False)
        # A bias-less dense layer for qkv projection.
        self.assertLen(l.variables, 2)
        self.assertLen(l.trainable_variables, 2)
        self.assertCountEqual(
            [v.name for v in l.variables],
            [
                'test/dot_product_self_attention/query_key_value_projection/kernel:0',
                'test/t5_relative_position_embedding/rel_embedding/embeddings:0',
            ],
        )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(True, False)
  def test_dot_product_self_attention_dropout(
      self, broadcast_dropout_across_queries
  ):
    with tf.name_scope('test'):
      l = sl.DotProductSelfAttention(
          num_heads=8,
          units_per_head=3,
          max_horizon=-1,
          max_future_horizon=-1,
          use_relative_position_embedding=False,
          attention_probabilities_dropout_rate=0.99999,
          broadcast_dropout_across_queries=broadcast_dropout_across_queries,
      )
    x = self.random_sequence(2, 12, 3, random_mask=True)
    y_dropout = l.layer(x, training=True)
    y_no_dropout = l.layer(x, training=False)
    y_no_dropout2 = l.layer(x, training=False)

    y_dropout, y_no_dropout, y_no_dropout2 = self.evaluate(
        [y_dropout, y_no_dropout, y_no_dropout2]
    )

    self.assertAllEqual(y_dropout.values, tf.zeros_like(y_dropout.values))
    self.assertSequencesClose(y_no_dropout, y_no_dropout2)

  def test_dot_product_self_attention_emit_outputs(self):
    num_heads, units_per_head, max_horizon = 3, 5, 10
    l = sl.DotProductSelfAttention(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_horizon=max_horizon,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    batch_size, time, channels = 2, 7, 11
    x = self.random_sequence(batch_size, time, channels)
    _, emits = l.layer_with_emits(x, training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, time, num_heads, time],
    )

    # Only run three timesteps of the sequence. The combination weights
    # are 3 + max_horizon.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(x, l.get_initial_state(x), training=False)
    emit_specs = l.get_emit_specs_for_sequence(x)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, 3, num_heads, 3 + max_horizon],
    )

  @parameterized.parameters(
      itertools.product(
          (True, False),
          (-1, 3),
          (-1, 0, 5),
          ('mixed_bfloat16', 'float32'),
          (None, 2),
      )
  )
  def test_dot_product_self_attention_chunked(
      self,
      relative_embedding: bool,
      past_horizon: int,
      future_horizon: int,
      policy: str,
      key_chunk_size: int | None,
  ):
    x = None
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    with test_util.keras_precision_policy_scope(policy):
      compute_dtype = utils.compute_dtype()
      batch_size, num_heads, units_per_head = 2, 3, 5
      query_chunk_size = 3
      with tf.name_scope('test'):
        relative_position_embedding = None
        if relative_embedding:
          relative_position_embedding = sl.T5RelativePositionEmbedding(
              num_buckets=32,
              num_heads=num_heads,
              max_distance=128,
              bidirectional=False,
          )
        l = sl.DotProductSelfAttention(
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_horizon=past_horizon,
            max_future_horizon=future_horizon,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            use_relative_position_embedding=relative_embedding,
            relative_position_embedding=relative_position_embedding,
        )
      self.assertEqual(l.block_size, 1)
      self.assertEqual(l.output_ratio, 1)

      channels = 1
      for time in range(5 * l.block_size, 8 * l.block_size):
        x = self.random_sequence(
            batch_size, time, channels, random_mask=False, dtype=compute_dtype
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x),
            tf.TensorShape([num_heads, units_per_head]),
        )

        y_layer_chunked = l.layer(x, training=False)
        l._query_chunk_size = None
        l._key_chunk_size = None
        y_layer_no_chunk = l.layer(x, training=False)
        l._query_chunk_size = query_chunk_size
        l._key_chunk_size = key_chunk_size

        self.assertEqual(
            y_layer_chunked.values.shape,
            [batch_size, time, num_heads, units_per_head],
        )

        self.assertEqual(
            y_layer_no_chunk.values.shape,
            [batch_size, time, num_heads, units_per_head],
        )

        rtol, atol = test_util.rtol_atol_for_dtype(compute_dtype)
        self.assertSequencesClose(
            y_layer_chunked, y_layer_no_chunk, rtol=rtol, atol=atol
        )
        self.verify_contract(
            l,
            x,
            training=False,
            rtol=rtol,
            atol=atol,
        )
        # A bias-less dense layer for qkv projection and optional relative
        # position embeddings.
        self.assertLen(l.variables, 2 if relative_embedding else 1)
        self.assertLen(l.trainable_variables, 2 if relative_embedding else 1)
        expected_variables = [
            'test/dot_product_self_attention/query_key_value_projection/kernel:0'
        ]
        if relative_embedding:
          expected_variables.append(
              'test/t5_relative_position_embedding/rel_embedding/embeddings:0'
          )
        self.assertCountEqual([v.name for v in l.variables], expected_variables)
      # Use flex for Einsum.
      if policy == 'float32':
        self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(
      (1, 2, False),
      (3, 5, False),
      (1, 2, True),
      (3, 5, True),
  )
  def test_dot_product_self_attention_query_key_value_network(
      self,
      num_heads,
      units_per_head,
      random_mask,
  ):
    x = None
    batch_size = 2
    with tf.name_scope('test'):
      l = sl.DotProductSelfAttention(
          num_heads=num_heads,
          units_per_head=units_per_head,
          max_horizon=3,
          max_future_horizon=0,
          use_relative_position_embedding=False,
          key_network=sl.AddTimingSignal(),
          query_network=sl.AddTimingSignal(),
          # Doesn't really make sense, but just testing something stateful.
          value_network=sl.AddTimingSignal(),
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertTrue(l.supports_step)

    channels = 1
    # Sweep time dimension shorter and longer than max_horizon.
    for time in range(1, 6):
      with self.subTest(f'time{time}'):
        x = self.random_sequence(
            batch_size, time, channels, random_mask=random_mask
        )
        self.assertEqual(
            l.get_output_shape_for_sequence(x),
            tf.TensorShape([num_heads, units_per_head]),
        )
        self.verify_contract(l, x, training=False)
        # A bias-less dense layer for qkv projection.
        self.assertCountEqual(
            [v.name for v in l.variables],
            [
                'test/dot_product_self_attention/query_key_value_projection/kernel:0'
            ],
        )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, use_flex=True)

  def test_dot_product_self_attention_query_key_value_network_supports_step(
      self,
  ):
    l = sl.DotProductSelfAttention(
        num_heads=3,
        units_per_head=5,
        max_horizon=3,
        query_network=sl.AddTimingSignal(),
        key_network=sl.AddTimingSignal(),
        value_network=sl.AddTimingSignal(),
    )
    self.assertTrue(l.supports_step)

    l = sl.DotProductSelfAttention(
        num_heads=3,
        units_per_head=5,
        max_horizon=3,
        query_network=test_util.NonSteppableLayer(),
        key_network=sl.AddTimingSignal(),
        value_network=sl.AddTimingSignal(),
    )
    self.assertFalse(l.supports_step)

    l = sl.DotProductSelfAttention(
        num_heads=3,
        units_per_head=5,
        max_horizon=3,
        query_network=sl.AddTimingSignal(),
        key_network=test_util.NonSteppableLayer(),
        value_network=sl.AddTimingSignal(),
    )
    self.assertFalse(l.supports_step)

    l = sl.DotProductSelfAttention(
        num_heads=3,
        units_per_head=5,
        max_horizon=3,
        query_network=sl.AddTimingSignal(),
        key_network=sl.AddTimingSignal(),
        value_network=test_util.NonSteppableLayer(),
    )
    self.assertFalse(l.supports_step)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_dot_product_self_attention_precision_policy(self, precision_policy):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    with tf.name_scope('test'):
      l = sl.DotProductSelfAttention(
          num_heads=3, units_per_head=5, max_horizon=12
      )

    x = self.random_sequence(2, 5, 3, dtype=utils.compute_dtype())
    rtol, atol = test_util.rtol_atol_for_dtype(x.values.dtype)
    _, y_np = self.verify_contract(l, x, training=True, rtol=rtol, atol=atol)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


class LocationSensitiveAttentionTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(
      (1, 2, 'same'),
      (1, 2, 'valid'),
      (3, 5, 'same'),
  )
  def test_location_sensitive_attention(
      self, num_heads, units_per_head, padding
  ):
    x = None
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.LocationSensitiveAttention(
          source_name,
          num_heads=num_heads,
          units_per_head=units_per_head,
          location_num_filters=3,
          location_filter_size=7,
          location_filter_padding=padding,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}

    channels = 3
    for i in range(5 * l.block_size, 7 * l.block_size):
      time = i + 1
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants),
          tf.TensorShape([num_heads, source_channels]),
      )
      self.verify_contract(l, x, training=False, constants=constants)
      # Projection for queries / keys, plus a combination matrix.
      self.assertLen(l.variables, 6)
      self.assertLen(l.trainable_variables, 6)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/location_sensitive_attention/v:0',
              'test/location_sensitive_attention/location_projection:0',
              'test/location_sensitive_attention/location_filters/depthwise_kernel:0',
              'test/location_sensitive_attention/query_projection/kernel:0',
              'test/location_sensitive_attention/source_projection/kernel:0',
              'test/location_sensitive_attention/source_projection/bias:0',
          ],
      )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  def test_location_sensitive_attention_emit_outputs(self):
    num_heads, units_per_head = 1, 5
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = sl.LocationSensitiveAttention(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        location_num_filters=3,
        location_filter_size=7,
    )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}
    time, channels = 7, 3
    x = self.random_sequence(batch_size, time, channels)
    _, emits = l.layer_with_emits(
        x,
        initial_state=l.get_initial_state(x, constants=constants),
        training=False,
        constants=constants,
    )
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, time, num_heads, source_time],
    )

    # Only run three timesteps of the sequence.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(
        x,
        l.get_initial_state(x, constants=constants),
        training=False,
        constants=constants,
    )
    emit_specs = l.get_emit_specs_for_sequence(x, constants=constants)
    self.assertEmitsCompatible(emit_specs, emits)
    self.assertEqual(
        emits.probabilities.values.shape.as_list(),
        [batch_size, 3, num_heads, source_time],
    )

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_location_sensitive_attention_precision_policy(
      self, precision_policy
  ):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.LocationSensitiveAttention(
          source_name,
          num_heads=3,
          units_per_head=5,
          location_num_filters=3,
          location_filter_size=7,
      )

    source = self.random_sequence(
        batch_size, source_time, source_channels, dtype=utils.compute_dtype()
    )
    constants = {source_name: source}

    x = self.random_sequence(batch_size, 5, 3, dtype=utils.compute_dtype())
    _, y_np = self.verify_contract(l, x, training=True, constants=constants)
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


class DynamicConvolutionAttentionTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(10, 11)
  def test_dynamic_convolution_attention(self, source_time):
    x = None
    num_heads = 1
    batch_size, source_channels = 2, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.DynamicConvolutionAttention(
          source_name,
          max_forward_step=3,
          prior_alpha=1.0,
          prior_beta=1.0,
          num_static_filters=4,
          num_dynamic_filters=8,
          dynamic_filter_hidden_dim=2,
      )
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}

    channels = 3
    for i in range(5 * l.block_size, 7 * l.block_size):
      time = i + 1
      x = self.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants),
          tf.TensorShape([num_heads, source_channels]),
      )
      self.verify_contract(l, x, training=False, constants=constants)
      self.assertLen(l.variables, 8)
      self.assertLen(l.trainable_variables, 8)
      self.assertCountEqual(
          [v.name for v in l.variables],
          [
              'test/dynamic_convolution_attention/v/kernel:0',
              'test/dynamic_convolution_attention/v/bias:0',
              'test/dynamic_convolution_attention/dynamic_location_filters_einsum_dense0/kernel:0',
              'test/dynamic_convolution_attention/dynamic_location_filters_einsum_dense0/bias:0',
              'test/dynamic_convolution_attention/dynamic_location_filters_einsum_dense1/kernel:0',
              'test/dynamic_convolution_attention/dynamic_location_filters_einsum_dense1/bias:0',
              'test/dynamic_convolution_attention/static_location_filters/kernel:0',
              'test/dynamic_convolution_attention/static_location_filters/bias:0',
          ],
      )

    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  @parameterized.parameters(*test_util.SUPPORTED_PRECISION_POLICIES)
  def test_dynamic_convolution_attention_precision_policy(
      self, precision_policy
  ):
    if not tf.executing_eagerly():
      self.skipTest('Mixed precision is TF2 only.')
    default_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    with tf.name_scope('test'):
      l = sl.DynamicConvolutionAttention(
          source_name,
          max_forward_step=3,
          prior_alpha=1.0,
          prior_beta=1.0,
          num_static_filters=4,
          num_dynamic_filters=8,
          dynamic_filter_hidden_dim=2,
      )

    source = self.random_sequence(
        batch_size, source_time, source_channels, dtype=utils.compute_dtype()
    )
    constants = {source_name: source}

    x = self.random_sequence(batch_size, 5, 3, dtype=utils.compute_dtype())
    rtol, atol = test_util.rtol_atol_for_dtype(x.values.dtype)
    _, y_np = self.verify_contract(
        l, x, training=True, constants=constants, rtol=rtol, atol=atol
    )
    self.assertEqual(y_np.dtype, utils.compute_dtype())
    for variable in l.variables:
      self.assertEqual(variable.dtype, utils.variable_dtype())
    tf.keras.mixed_precision.set_global_policy(default_policy)


if __name__ == '__main__':
  tf.test.main()
