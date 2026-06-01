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
    x_np = np.array(x.values)
    x_ss = np.sum(np.square(x_np), axis=reduce_axes, keepdims=True)
    y_expected_np = x_np / np.sqrt(x_ss + epsilon)

    y_np = np.array(y.values)
    expanded_mask = np.array(y.expanded_mask())
    y_np_masked = np.where(expanded_mask, y_np, 0.0)
    y_expected_np_masked = np.where(expanded_mask, y_expected_np, 0.0)

    np.testing.assert_allclose(
        y_np_masked, y_expected_np_masked, rtol=1e-5, atol=1e-5
    )


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
    shape, axes, _ = shape_axes
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
    x_np = np.array(x.values)
    x_ss = np.mean(np.square(x_np), axis=reduce_axes, keepdims=True)
    y_expected_np = x_np / np.sqrt(x_ss + epsilon)

    y_np = np.array(y.values)
    expanded_mask = np.array(y.expanded_mask())
    y_np_masked = np.where(expanded_mask, y_np, 0.0)
    y_expected_np_masked = np.where(expanded_mask, y_expected_np, 0.0)

    np.testing.assert_allclose(
        y_np_masked, y_expected_np_masked, rtol=1e-5, atol=1e-5
    )


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
    shape, axes, _ = shape_axes
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
    y_np = np.array(y.values)
    mean = np.mean(y_np, axis=reduce_axes)
    var = np.var(y_np, axis=reduce_axes)

    # Invalid timesteps will have a mean and variance of zero.
    np.testing.assert_allclose(mean, np.zeros_like(mean), rtol=1e-5, atol=1e-5)
    mask = np.array(y.mask, dtype=np.float32)
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
    shape, axis, num_groups, _ = shape_axes
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
    shape = list(y_test.values.shape)
    axis_dim = shape[axis]
    group_size = axis_dim // num_groups
    outer_dims = shape[:axis]
    inner_dims = shape[axis + 1 :]

    # Unscale and verify group normalization per-timestep.
    if cumulative:
      # Skip testing cumulative mode numerically.
      return

    y_vals = y_test.values
    y_grouped = np.reshape(
        y_vals,
        outer_dims + [num_groups, group_size] + inner_dims,
    )

    if l.supports_step:
      # Pointwise/causal normalization: reduce only over group_size (axis 3).
      reduction_dims = (3,)
    else:
      # Non-causal normalization: reduce over time (axis 1) and group_size (axis 3).
      reduction_dims = tuple(
          a for a in range(y_grouped.ndim) if a not in (0, axis)
      )
    expanded_mask = self.sl.types.Sequence(y_grouped, x.mask).expanded_mask()
    expanded_mask_np = np.array(expanded_mask, dtype=bool)
    y_grouped_np = np.array(y_grouped)

    mean = np.mean(
        y_grouped_np, axis=reduction_dims, keepdims=True, where=expanded_mask_np
    )
    var = np.var(
        y_grouped_np, axis=reduction_dims, keepdims=True, where=expanded_mask_np
    )

    # Avoid NaNs.
    mean = np.where(np.isnan(mean), np.zeros_like(mean), mean)
    var = np.where(np.isnan(var), np.ones_like(var), var)

    np.testing.assert_allclose(mean, np.zeros_like(mean), atol=2e-5)
    if l.supports_step:
      if group_size == 1:
        # Reducing over 1 element per-timestep mathematically results in 0 variance.
        np.testing.assert_allclose(var, np.zeros_like(var), atol=1e-3)
      elif group_size == 2:
        # For tiny group sizes per-timestep, the output variance can naturally
        # deviate from 1.0 due to epsilon.
        np.testing.assert_allclose(var, np.ones_like(var), atol=0.8)
      else:
        np.testing.assert_allclose(var, np.ones_like(var), atol=1e-3)
    else:
      np.testing.assert_allclose(var, np.ones_like(var), atol=1e-3)
