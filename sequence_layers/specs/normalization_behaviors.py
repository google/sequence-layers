"""Behavior tests for normalization layers.

Backend-specific test files should inherit from these tests.
"""

# pylint: disable=abstract-method

import itertools

from absl.testing import parameterized
import numpy as np

from sequence_layers.specs import test_utils


class L2NormalizeTest(test_utils.SequenceLayerTest):
  """Test behavior of L2Normalize layer."""

  def test_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    l = self.sl.L2Normalize.Config(axis=[-1, -2]).make()
    x = self.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

  @parameterized.parameters(
      itertools.product(
          (False, True),
          [
              ((2, 10, 3), [-1]),
              ((2, 3, 5, 9), [-1]),
              ((2, 3, 5, 9), [-2]),
              ((2, 3, 5, 9), [-1, -2]),
          ],
      )
  )
  def test_l2_normalization(self, training, shape_axes):
    shape, axes = shape_axes
    epsilon = 1e-12
    l = self.sl.L2Normalize.Config(
        axis=axes, epsilon=epsilon, name='l2_normalization'
    ).make()
    x = self.random_sequence(*shape)
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'l2_normalization')
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    y = self.verify_contract(l, x, training=training)

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a in axes or a - len(shape) in axes
    )
    x_ss = np.sum(np.square(x.values), axis=reduce_axes, keepdims=True)

    y_expected = self.sl.types.Sequence(
        x.values / np.sqrt(x_ss + epsilon), x.mask
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)


class RMSNormalizationTest(test_utils.SequenceLayerTest):
  """Test behavior of RMSNormalization layer."""

  def test_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    l = self.sl.RMSNormalization.Config(axis=[-1, -2]).make()
    x = self.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

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
    shape, axes, expected_param_shape = shape_axes
    epsilon = 1e-1
    l = self.sl.RMSNormalization.Config(
        axis=axes, epsilon=epsilon, name='rms_normalization'
    ).make()
    x = self.random_sequence(*shape)
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'rms_normalization')
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    y = self.verify_contract(l, x, training=training)

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a in axes or a - len(shape) in axes
    )
    x_ss = np.mean(np.square(x.values), axis=reduce_axes, keepdims=True)

    y_expected = self.sl.types.Sequence(
        x.values / np.sqrt(x_ss + epsilon), x.mask
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)


class LayerNormalizationTest(test_utils.SequenceLayerTest):
  """Test behavior of LayerNormalization layer."""

  def test_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    l = self.sl.LayerNormalization.Config(axis=[-1, -2]).make()
    x = self.random_sequence(2, 3, 5)
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

  @parameterized.parameters(
      itertools.product(
          (False, True),
          [
              ((2, 10, 4), [-1], [4]),
              ((2, 3, 5, 4), [-1], [4]),
              ((2, 3, 4, 9), [-2], [4]),
              ((2, 3, 4, 8), [-1, -2], [4, 8]),
          ],
      )
  )
  def test_layer_normalization(self, training, shape_axes):
    shape, axes, expected_param_shape = shape_axes
    l = self.sl.LayerNormalization.Config(
        axis=axes, name='layer_normalization'
    ).make()
    x = self.random_sequence(*shape)
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'layer_normalization')
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    y = self.verify_contract(l, x, training=training)

    # Verify the train batch is normalized correctly.
    reduce_axes = tuple(
        a for a in range(len(shape)) if a in axes or a - len(shape) in axes
    )
    mean = np.mean(y.values, axis=reduce_axes)
    var = np.var(y.values, axis=reduce_axes)

    # Invalid timesteps will have a mean and variance of zero.
    np.testing.assert_allclose(mean, np.zeros_like(mean), rtol=1e-5, atol=1e-5)
    mask = y.mask.astype(np.float32)
    mask = np.reshape(
        mask, mask.shape + (1,) * (len(mean.shape) - len(mask.shape))
    )
    np.testing.assert_allclose(
        var, np.broadcast_to(mask, mean.shape), rtol=1e-4, atol=1e-4
    )


class BatchNormalizationTest(test_utils.SequenceLayerTest):
  """Test behavior of BatchNormalization layer."""

  def test_batch_normalization_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    x = self.random_sequence(2, 3, 5)
    l = self.sl.BatchNormalization.Config(axis=0).make()
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

    l = self.sl.BatchNormalization.Config(axis=1).make()
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

    l = self.sl.BatchNormalization.Config(axis=2).make()
    l = self.init_layer(l, x)


class GroupNormalizationTest(test_utils.SequenceLayerTest):
  """Test behavior of GroupNormalization layer."""

  def test_invalid_axis(self):
    """Normalizing over the batch or time dimension is not allowed."""
    x = self.random_sequence(2, 3, 5)
    l = self.sl.GroupNormalization.Config(num_groups=1, axis=0).make()
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

    l = self.sl.GroupNormalization.Config(num_groups=1, axis=1).make()
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

    l = self.sl.GroupNormalization.Config(num_groups=1, axis=2).make()
    l = self.init_layer(l, x)

  def test_invalid_groups(self):
    x = self.random_sequence(2, 3, 5)
    l = self.sl.GroupNormalization.Config(num_groups=2).make()
    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)

  @parameterized.parameters(
      itertools.product(
          [
              ((8, 6, 6), -1, 3, [6]),
              ((8, 6, 5, 6), -2, 5, [5]),
              ((8, 6, 5, 6), -2, 1, [5]),
          ],
          (False, True),
      )
  )
  def test_group_normalization(self, shape_axes, cumulative):
    shape, axis, num_groups, expected_param_shape = shape_axes
    l = self.sl.GroupNormalization.Config(
        num_groups=num_groups,
        cumulative=cumulative,
        axis=axis,
        name='group_normalization',
    ).make()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'group_normalization')

    x = self.random_sequence(*shape)
    l = self.init_layer(l, x)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])

    # Test inference (training=False) - both backends match.
    y_test = self.verify_contract(l, x, training=False)

    axis = axis + x.ndim if axis < 0 else axis
    axis_dim = y_test.values.shape[axis]
    group_size = axis_dim // num_groups
    outer_dims, _, inner_dims = np.split(y_test.values.shape, [axis, axis + 1])

    # Unscale and verify group normalization per-timestep.
    if cumulative:
      # Skip testing cumulative mode numerically.
      return

    y_vals = y_test.values
    y_grouped = np.reshape(
        y_vals,
        outer_dims.tolist() + [num_groups, group_size] + inner_dims.tolist(),
    )

    reduction_dims = [a for a in range(y_grouped.ndim) if a not in (0, axis)]
    expanded_mask = self.sl.types.Sequence(y_grouped, x.mask).expanded_mask()

    mean = np.mean(
        y_grouped, axis=reduction_dims, keepdims=True, where=expanded_mask
    )
    var = np.var(
        y_grouped, axis=reduction_dims, keepdims=True, where=expanded_mask
    )

    # Avoid NaNs.
    mean = np.where(np.isnan(mean), np.zeros_like(mean), mean)
    var = np.where(np.isnan(var), np.ones_like(var), var)

    np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-5)
    np.testing.assert_allclose(var, np.ones_like(var), atol=1e-3)
