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
"""Tests for sequence_layers.tensorflow.normalization."""

import contextlib
import functools
import itertools

from absl import flags
from absl.testing import parameterized
import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import normalization
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow import utils
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

_TEST_ON_TPU = flags.DEFINE_boolean(
    'test_on_tpu', False, 'Whether to run on a TPU.'
)


def _maybe_tpu_rewrite(f):
  def wrapper():
    if _TEST_ON_TPU.value:
      return test_util.structured_tpu_rewrite(f)
    else:
      return f()

  return wrapper


@contextlib.contextmanager
def _maybe_tpu_initialize():
  if _TEST_ON_TPU.value:
    tf.tpu.experimental.initialize_tpu_system()
  yield
  if _TEST_ON_TPU.value:
    tf.tpu.experimental.shutdown_tpu_system()


class NormalizationTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_layer_normalization_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    l = sl.LayerNormalization(axis=[-1, -2])
    x = self.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l.layer(x, training=True)

  @parameterized.parameters(
      itertools.product(
          (False, True),
          [((2, 10, 3), [-1]), ((2, 3, 5, 9), [-1]), ((2, 3, 5, 9), [-1, -2])],
      )
  )
  def test_layer_normalization(self, training, shape_axes):
    shape, axes = shape_axes
    with tf.name_scope('test'):
      l = sl.LayerNormalization(axes)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    x = self.random_sequence(*shape)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape(shape[2:])
    )
    _, y_layer_np = self.verify_contract(l, x, training=training)
    self.assertLen(l.variables, 2)
    self.assertLen(l.trainable_variables, 2)
    self.assertCountEqual(
        [v.name for v in l.variables],
        ['test/layer_normalization/gamma:0', 'test/layer_normalization/beta:0'],
    )
    # Requires flex ops due to fused batchnorm usage.
    self.verify_tflite_step(l, x, use_flex=True)

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a in axes or a - len(shape) in axes
    )
    mean = np.mean(y_layer_np.values, axis=reduce_axes)
    var = np.var(y_layer_np.values, axis=reduce_axes)

    # Invalid timesteps will have a mean and variance of zero.
    self.assertAllClose(mean, np.zeros_like(mean))
    mask = y_layer_np.mask
    mask = np.reshape(
        mask, mask.shape + (1,) * (len(mean.shape) - len(mask.shape))
    )
    self.assertAllClose(
        var, np.broadcast_to(mask, mean.shape), rtol=1e-2, atol=1e-2
    )

  def test_rms_normalization_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    l = sl.RMSNormalization(axis=[-1, -2])
    x = self.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l.layer(x, training=True)

  @parameterized.parameters(
      itertools.product(
          (False, True),
          [((2, 10, 3), [-1]), ((2, 3, 5, 9), [-1]), ((2, 3, 5, 9), [-1, -2])],
      )
  )
  def test_rms_normalization(self, training, shape_axes):
    shape, axes = shape_axes
    epsilon = 1e-1
    with tf.name_scope('test'):
      l = sl.RMSNormalization(axes, epsilon=epsilon)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    x = self.random_sequence(*shape)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape(shape[2:])
    )
    _, y_layer_np = self.verify_contract(l, x, training=training)
    self.assertLen(l.variables, 1)
    self.assertLen(l.trainable_variables, 1)
    self.assertCountEqual(
        [v.name for v in l.variables], ['test/rms_normalization/gamma:0']
    )
    self.verify_tflite_step(l, x)

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a in axes or a - len(shape) in axes
    )
    x_np = self.evaluate(x.values)
    x_ss = np.mean(np.square(x_np), axis=reduce_axes, keepdims=True)

    y_expected = sl.Sequence(
        x_np / np.sqrt(x_ss + epsilon), x.mask
    ).mask_invalid()
    self.assertSequencesClose(y_layer_np, y_expected)

  def test_batch_normalization_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    l = sl.BatchNormalization(axis=[-1, -2])
    x = self.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l.layer(x, training=True)

  @parameterized.parameters(
      ((2, 10, 3), [-1]),
      ((2, 3, 5, 9), [-1]),
      ((2, 3, 5, 9), [-1, -2]),
      ((2, 3, 5, 9), [-2]),
  )
  def test_batch_normalization(self, shape, axes):
    epsilon = 1e-3
    with tf.name_scope('test'):
      l = sl.BatchNormalization(axis=axes, epsilon=epsilon)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    # Use full-length sequences so we can verify statistics.
    x = self.random_sequence(*shape, random_lengths=False)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape(shape[2:])
    )
    y_layer_train = l.layer(x, training=True)
    self.assertLen(l.variables, 4)
    self.assertLen(l.trainable_variables, 2)
    self.assertCountEqual(
        [v.name for v in l.variables],
        [
            'test/batch_normalization/moving_mean:0',
            'test/batch_normalization/moving_variance:0',
            'test/batch_normalization/gamma:0',
            'test/batch_normalization/beta:0',
        ],
    )

    # Step-wise training is not supported.
    state = l.get_initial_state(x)
    with self.assertRaises(ValueError):
      utils.step_by_step_dynamic(l, x, training=True, initial_state=state)

    # Run the update ops in graph mode.
    y_layer_train_np = self.evaluate([y_layer_train])[0]

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a
        for a in range(len(shape))
        if a not in axes and a - len(shape) not in axes
    )
    mean = np.mean(y_layer_train_np.values, axis=reduce_axes)
    var = np.var(y_layer_train_np.values, axis=reduce_axes)
    self.assertAllClose(mean, np.zeros_like(mean))
    self.assertAllClose(var, np.ones_like(mean), rtol=1e-2, atol=1e-2)

    moving_mean, moving_variance, gamma, beta = l.variables
    assert 'gamma' in gamma.name
    assert 'beta' in beta.name
    assert 'moving_mean' in moving_mean.name
    assert 'moving_variance' in moving_variance.name

    # Set gamma and beta to non-default values.
    self.evaluate([
        gamma.assign(tf.ones_like(gamma) * 5.0),
        beta.assign(tf.ones_like(beta) * -3.0),
    ])

    # Verify that layer-wise and step-wise processing are identical in
    # non-training mode.
    x2 = self.random_sequence(*shape, random_lengths=False)
    x2_np, y2_np = self.verify_contract(l, x2, training=False)
    self.verify_tflite_step(l, x2, use_flex=False)

    # Check that y2_np is correct given x2, gamma, beta and moving mean/var.
    gamma, beta, moving_mean, moving_variance = self.evaluate(
        [gamma, beta, moving_mean, moving_variance]
    )

    # Need to expand_dims ("broadcast") for single-axis where axis is not the
    # last dimension.
    if axes == [-2]:
      gamma = gamma[..., tf.newaxis]
      beta = beta[..., tf.newaxis]
      moving_mean = moving_mean[..., tf.newaxis]
      moving_variance = moving_variance[..., tf.newaxis]

    y2_expected = (
        gamma
        * (x2_np.values - moving_mean)
        / np.sqrt(epsilon + moving_variance)
        + beta
    )
    self.assertAllClose(y2_np.values, y2_expected)

  def test_instance_normalization_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    l = sl.InstanceNormalization(axis=[-1, -2])
    x = self.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l.layer(x, training=True)

  @parameterized.parameters(
      ((2, 10, 3), [-1]),
      ((2, 13, 5, 9), [-1]),
      ((2, 13, 5, 9), [-2]),
      ((2, 13, 5, 9), [-1, -2]),
  )
  def test_instance_normalization(self, shape, axes):
    epsilon = 1e-3
    with tf.name_scope('test'):
      l = sl.InstanceNormalization(axis=axes, epsilon=epsilon)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    # Use full-length sequences so we can verify statistics.
    x = self.random_sequence(*shape, random_lengths=False)
    self.assertEqual(
        l.get_output_shape_for_sequence(x), tf.TensorShape(shape[2:])
    )
    y_layer_train = l.layer(x, training=True)
    self.assertLen(l.variables, 2)
    self.assertLen(l.trainable_variables, 2)
    self.assertCountEqual(
        [v.name for v in l.variables],
        [
            'test/instance_normalization/gamma:0',
            'test/instance_normalization/beta:0',
        ],
    )

    # Step-wise processing is not supported.
    self.assertFalse(l.supports_step)

    y_layer_train_np = self.evaluate(y_layer_train)

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a
        for a in range(1, len(shape))
        if a not in axes and a - len(shape) not in axes
    )
    mean = np.mean(y_layer_train_np.values, axis=reduce_axes)
    var = np.var(y_layer_train_np.values, axis=reduce_axes)
    self.assertAllClose(mean, np.zeros_like(mean))
    self.assertAllClose(var, np.ones_like(mean), rtol=1e-2, atol=1e-2)

    gamma, beta = l.variables
    assert 'gamma' in gamma.name
    assert 'beta' in beta.name

    # Set gamma and beta to non-default values.
    self.evaluate([
        gamma.assign(tf.ones_like(gamma) * 5.0),
        beta.assign(tf.ones_like(beta) * -3.0),
    ])

    # Verify contract.
    x2 = self.random_sequence(*shape, random_lengths=False)
    x2_np, y2_np = self.verify_contract(l, x2, training=False)
    self.verify_tflite_step(l, x2, use_flex=False)

    # Check that y2_np is correct given x2, gamma, and beta.
    gamma, beta = self.evaluate([gamma, beta])

    broadcast_axes = [1] * len(shape)
    for axe in axes:
      if axe < 0:
        axe += len(shape)
      broadcast_axes[axe] = shape[axe]
    gamma = np.reshape(gamma, broadcast_axes)
    beta = np.reshape(beta, broadcast_axes)

    mean = np.mean(x2_np.values, axis=reduce_axes, keepdims=True)
    var = np.var(x2_np.values, axis=reduce_axes, keepdims=True)

    y2_expected = gamma * (x2_np.values - mean) / np.sqrt(epsilon + var) + beta
    self.assertAllClose(y2_np.values, y2_expected)


class SequenceBatchNormalizationTest(
    test_util.SequenceLayerTest, parameterized.TestCase
):

  @parameterized.parameters(([-1],), ([-1, -2],))
  def test_tf1_sequence_lengths(self, axis):
    """Moments are computed across valid timesteps and outputs are masked."""
    if tf.executing_eagerly():
      return

    epsilon = 1e-9
    # Set momentum to zero so that we know the expected value for moving
    # means/variances after one update.
    with tf.name_scope('test'):
      layer = normalization._SequenceBatchNormalization(
          axis=axis,
          epsilon=epsilon,
          momentum=0.0,
          gamma_initializer=tf.initializers.constant(3.0),
          beta_initializer=tf.initializers.constant(-1.0),
      )

    @_maybe_tpu_rewrite
    def train_fn():
      x = self.random_sequence(2, 20, 3, 5)
      y = layer((x.values, x.mask), training=True)
      self.assertLen(layer.trainable_variables, 2)
      self.assertLen(layer.variables, 4)

      return tf.nest.map_structure(
          tf.identity, (x.values, x.expanded_mask(), y)
      )

    x, x_mask, y = train_fn()

    # Compute moments from valid timesteps over axes not in `axis`.
    reduce_axes = [a for a in range(4) if a not in axis and a - 4 not in axis]
    x_mean, x_var = tf.nn.weighted_moments(
        x, axes=reduce_axes, frequency_weights=x_mask, keepdims=len(axis) > 1
    )
    # Undo beta/gamma scaling.
    y_mean, y_var = tf.nn.weighted_moments(
        (y + 1.0) / 3.0,
        axes=reduce_axes,
        frequency_weights=x_mask,
        keepdims=len(axis) > 1,
    )

    @_maybe_tpu_rewrite
    def test_fn():
      x2_values = tf.random.uniform((2, 15, 3, 5))
      x2_lengths = tf.convert_to_tensor([0, 9])
      x2_mask = tf.sequence_mask(x2_lengths, 15, dtype=tf.float32)
      x2_mask_expanded = x2_mask[:, :, tf.newaxis, tf.newaxis]
      y2 = layer((x2_values, x2_mask), training=False)
      self.assertLen(layer.trainable_variables, 2)
      self.assertLen(layer.variables, 4)
      self.assertCountEqual(
          [v.name for v in layer.variables],
          [
              'private__sequence_batch_normalization/gamma:0',
              'private__sequence_batch_normalization/beta:0',
              'private__sequence_batch_normalization/moving_mean:0',
              'private__sequence_batch_normalization/moving_variance:0',
          ],
      )
      return x2_values, x2_mask_expanded, y2

    x2, x2_mask, y2 = test_fn()

    with _maybe_tpu_initialize():
      x_mean, x_var, y_mean, y_var = self.evaluate(
          [x_mean, x_var, y_mean, y_var]
      )

      self.assertAllClose(y_mean, np.zeros_like(y_mean))
      self.assertAllClose(y_var, np.ones_like(y_var))

      x2_np, x2_mask_np, y2_np = self.evaluate([x2, x2_mask, y2])
      moving_mean, moving_variance = self.evaluate(
          [layer.moving_mean, layer.moving_variance]
      )
      # The moving mean and variance are exactly equal to x's moments because
      # momentum is zero.
      self.assertAllClose(moving_mean, x_mean)
      self.assertAllClose(moving_variance, x_var)
      # We initialized gamma and beta to 3 and -1 above.
      y2_expected = (
          3.0 * (x2_np - moving_mean) / np.sqrt(epsilon + moving_variance) - 1.0
      )
      self.assertAllClose(y2_np, y2_expected * x2_mask_np)

  @parameterized.parameters(([-1],), ([-1, -2],))
  def test_tf2_sequence_lengths(self, axis):
    """Moments are computed across valid timesteps and outputs are masked."""
    if not tf.executing_eagerly():
      return

    if _TEST_ON_TPU.value:
      # Instantiate the TPUStrategy object, with a tpu cluster resolver.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
      strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    with strategy.scope():
      epsilon = 1e-9
      # Set momentum to zero so that we know the expected value for moving
      # means/variances after one update.
      layer = normalization._SequenceBatchNormalization(
          axis=axis,
          epsilon=epsilon,
          momentum=0.0,
          gamma_initializer=tf.initializers.constant(3.0),
          beta_initializer=tf.initializers.constant(-1.0),
      )

      x = self.random_sequence(2, 20, 3, 5)
      x_mask = x.expanded_mask()
      y = layer((x.values, x.mask), training=True)
      self.assertLen(layer.trainable_variables, 2)
      self.assertLen(layer.variables, 4)
      self.assertCountEqual(
          [v.name for v in layer.variables],
          [
              'private__sequence_batch_normalization/gamma:0',
              'private__sequence_batch_normalization/beta:0',
              'private__sequence_batch_normalization/moving_mean:0',
              'private__sequence_batch_normalization/moving_variance:0',
          ],
      )

      # Compute moments from valid timesteps over axes not in `axis`.
      reduce_axes = [a for a in range(4) if a not in axis and a - 4 not in axis]
      x_mean, x_var = tf.nn.weighted_moments(
          x.values,
          axes=reduce_axes,
          frequency_weights=x_mask,
          keepdims=len(axis) > 1,
      )
      # Undo beta/gamma scaling.
      y_mean, y_var = tf.nn.weighted_moments(
          (y + 1.0) / 3.0,
          axes=reduce_axes,
          frequency_weights=x_mask,
          keepdims=len(axis) > 1,
      )

      x2 = tf.random.uniform((2, 15, 3, 5))
      x2_lengths = tf.convert_to_tensor([0, 9])
      x2_mask = tf.sequence_mask(x2_lengths, 15, dtype=tf.float32)
      x2_mask_expanded = x2_mask[:, :, tf.newaxis, tf.newaxis]
      y2 = layer((x2, x2_mask), training=False)
      self.assertLen(layer.trainable_variables, 2)
      self.assertLen(layer.variables, 4)
      self.assertAllClose(y_mean, np.zeros_like(y_mean))
      self.assertAllClose(y_var, np.ones_like(y_var))

      # The moving mean and variance are exactly equal to x's moments because
      # momentum is zero.
      self.assertAllClose(layer.moving_mean, x_mean)
      self.assertAllClose(layer.moving_variance, x_var)
      # We initialized gamma and beta to 3 and -1 above.
      y2_expected = (
          3.0
          * (x2 - layer.moving_mean)
          / np.sqrt(epsilon + layer.moving_variance)
          - 1.0
      )
      self.assertAllClose(y2, y2_expected * x2_mask_expanded)

  def test_tf1_tpu_cross_replica_sum(self):
    """TPU cross replica sum aggregates moments across replicas."""
    if not _TEST_ON_TPU.value or tf.executing_eagerly():
      return

    # Use separate layer instances so that we don't double-execute update ops
    # for the layer on the second call.
    layer = normalization._SequenceBatchNormalization(
        axis=-1, use_cross_replica_sum=True
    )
    layer_sharded = normalization._SequenceBatchNormalization(
        axis=-1, use_cross_replica_sum=True
    )

    def train_fn(values, lengths, layer):
      mask = tf.sequence_mask(
          lengths, values.shape.dims[1].value, dtype=tf.float32
      )
      return layer((values, mask), training=True)

    values = tf.random.uniform((8, 20, 5))
    lengths = tf.convert_to_tensor([5, 15, 0, 3, 4, 10, 20, 1])
    # Forge TPUs are 1x1, so 2 cores. Compare sharding across the 2 cores to
    # running everything on one core.
    y_values = tf1.tpu.rewrite(
        functools.partial(train_fn, layer=layer), [values, lengths]
    )
    y_values_sharded = tf1.tpu.batch_parallel(
        functools.partial(train_fn, layer=layer_sharded),
        [values, lengths],
        num_shards=2,
    )

    with _maybe_tpu_initialize():
      y_values_np, y_values_sharded_np = self.evaluate(
          [y_values, y_values_sharded]
      )
      mean, mean_sharded, variance, variance_sharded = self.evaluate([
          layer.moving_mean,
          layer_sharded.moving_mean,
          layer.moving_variance,
          layer_sharded.moving_variance,
      ])

      # No difference between computing as one versus sharding 2 ways.
      # If moments were not computed cross-replica then this will fail.
      self.assertAllClose(y_values_np, y_values_sharded_np)
      self.assertAllClose(mean, mean_sharded)
      self.assertAllClose(variance, variance_sharded)

  def test_tf2_tpu_strategy_cross_replica_sum(self):
    """TPU cross replica sum aggregates moments across replicas."""
    if not _TEST_ON_TPU.value or not tf.executing_eagerly():
      return

    devices = [
        d for d in tf.config.list_logical_devices() if d.device_type == 'TPU'
    ]
    self.assertNotEmpty(devices)
    # Pick one core to use.
    device = devices[0]

    # Instantiate the TPUStrategy object, with a tpu cluster resolver.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    def train_fn(values, lengths, layer):
      mask = tf.sequence_mask(
          lengths, values.shape.dims[1].value, dtype=tf.float32
      )
      y = layer((values, mask), training=True)
      return y

    # Use separate layer instances so that we don't double-execute update ops
    # for the layer on the second call.
    layer = normalization._SequenceBatchNormalization(
        axis=-1, use_cross_replica_sum=True
    )

    values = tf.random.uniform((8, 20, 5))
    lengths = tf.convert_to_tensor([5, 15, 0, 3, 4, 10, 20, 1])

    # Forge TPUs are 1x1, so 2 cores. Compare sharding across the 2 cores to
    # running everything on one core.

    with tf.device(device):
      y_values = train_fn(values, lengths, layer)

    with strategy.scope():

      def split_fn(ctx, x):
        """Distribute x across replicas by splitting it across batch."""
        xs = tf.split(x, ctx.num_replicas_in_sync, axis=0)
        return xs[ctx.replica_id_in_sync_group]

      values_dist = strategy.experimental_distribute_values_from_function(
          functools.partial(split_fn, x=values)
      )
      lengths_dist = strategy.experimental_distribute_values_from_function(
          functools.partial(split_fn, x=lengths)
      )

      layer_sharded = normalization._SequenceBatchNormalization(
          axis=-1, use_cross_replica_sum=True
      )
      y_values_sharded = strategy.run(
          tf.function(functools.partial(train_fn, layer=layer_sharded)),
          [values_dist, lengths_dist],
      )
      y_values_sharded = tf.concat(
          strategy.experimental_local_results(y_values_sharded), 0
      )

    # No difference between computing as one versus sharding 2 ways.
    # If moments were not computed cross-replica then this will fail.
    self.assertAllClose(y_values, y_values_sharded)
    self.assertAllClose(layer.moving_mean, layer_sharded.moving_mean)
    self.assertAllClose(layer.moving_variance, layer_sharded.moving_variance)

  @parameterized.parameters(
      (tf.TensorShape((2, 3, 5)), 2),
      (tf.TensorShape((2, 3, 5)), [-1]),
      (tf.TensorShape((2, 3, 5, 6)), -1),
      (tf.TensorShape((2, 3, 5, 6)), [2, 3]))  # pyformat: disable
  def test_l2_normalize(self, shape, axis):
    x = self.random_sequence(*shape)
    l = sl.L2Normalize(axis)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    self.verify_contract(l, x, training=False)
    self.assertEmpty(l.variables)
    self.assertEmpty(l.trainable_variables)
    self.verify_tflite_step(l, x)


if __name__ == '__main__':
  tf.test.main()
