# Copyright 2026 Google LLC
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
"""Behavior tests for conditioning layers."""

# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation

from absl.testing import parameterized
import numpy as np

from sequence_layers.specs import conditioning as conditioning_spec
from sequence_layers.specs import test_utils


class ConditioningTest(test_utils.SequenceLayerTest):
  """Test behavior of Conditioning layer."""

  def _make_constants(self, conditioning_seq, name='cond'):
    return {name: conditioning_seq}

  def test_identity_add(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    self.verify_contract(layer, x, pad_constants=True, constants=constants)

  def test_identity_add_output_shape(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))

  def test_identity_add_broadcast(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 1)
    constants = self._make_constants(cond_seq)
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))

  def test_tensor_conditioning(self):
    """Conditioning with a [B, dim] tensor (not a Sequence)."""
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    # Generate a random sequence and extract its values to get a raw tensor.
    cond_seq = self.random_sequence(2, 1, 4)
    # Squeeze the time dimension to get [B, C]
    cond_values = cond_seq.values
    if hasattr(cond_values, 'squeeze'):
      cond_tensor = cond_values.squeeze(axis=1)
    else:
      # Fallback if squeeze is not available (should be for both JAX and MLX)
      cond_tensor = cond_values[:, 0]

    constants = self._make_constants(cond_tensor)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    y = layer.layer(x, training=False, constants=constants)
    self.assertEqual(y.channel_shape, (4,))

  def test_step_non_streaming(self):
    """Non-streaming: full conditioning passed, layer slices per step."""
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
        streaming=False,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)

    layer = self.init_layer(layer, x, constants=constants)

    # Layer mode.
    y_layer = layer.layer(x, training=False, constants=constants)

    # Step mode (pass full conditioning; layer slices internally).
    y_step, _ = self._step_by_step(layer, x, block_size=1, constants=constants)
    self.assertSequencesClose(y_step, y_layer)

  def test_step_streaming(self):
    """Streaming: conditioning chunks arrive with input chunks."""
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
        streaming=True,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    x = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)

    layer = self.init_layer(layer, x, constants=constants)

    # Layer mode.
    y_layer = layer.layer(x, training=False, constants=constants)

    # Step mode with stream_constants.
    y_step, _ = self._step_by_step(
        layer,
        x,
        block_size=1,
        stream_constants=constants,
    )
    self.assertSequencesClose(y_step, y_layer)

  def test_identity_concat(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.CONCAT,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 3)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    y = layer.layer(x, training=False, constants=constants)
    self.assertEqual(y.channel_shape, (7,))

  def test_concat_before(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.CONCAT_BEFORE,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 3)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    y = layer.layer(x, training=False, constants=constants)
    self.assertEqual(y.channel_shape, (7,))
    # CONCAT_BEFORE should have conditioning first.
    y_cond = y[:, :, :3]
    # Align cond_seq mask with y mask for comparison (y.mask is c.mask & x.mask)
    cond_seq_aligned = self.sl.Sequence(cond_seq.values, y.mask).mask_invalid()
    self.assertSequencesClose(y_cond, cond_seq_aligned)

  def test_identity_mul(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.MUL,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    self.verify_contract(layer, x, pad_constants=True, constants=constants)

  def test_linear_add(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 6)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    self.verify_contract(layer, x, pad_constants=True, constants=constants)

  def test_linear_add_output_shape(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 6)
    constants = self._make_constants(cond_seq)
    # LINEAR projects conditioning to input channel shape.
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))

  def test_with_projection_channel_shape(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR,
        combination=conditioning_spec.Combination.ADD,
        projection_channel_shape=(8,),
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 6)
    constants = self._make_constants(cond_seq)
    # Projects to (8,), then broadcast-add with input (8,).
    self.assertEqual(layer.get_output_shape((8,), constants=constants), (8,))

  def test_linear_affine_shift(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR,
        combination=conditioning_spec.Combination.AFFINE_SHIFT,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 6)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    self.verify_contract(layer, x, pad_constants=True, constants=constants)

  def test_linear_affine_scale(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR,
        combination=conditioning_spec.Combination.AFFINE_SCALE,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 6)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    self.verify_contract(layer, x, pad_constants=True, constants=constants)

  def test_linear_affine(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR_AFFINE,
        combination=conditioning_spec.Combination.AFFINE,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 6)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, constants=constants)
    self.verify_contract(layer, x, pad_constants=True, constants=constants)

  def test_linear_affine_output_shape(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR_AFFINE,
        combination=conditioning_spec.Combination.AFFINE,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 6)
    constants = self._make_constants(cond_seq)
    # AFFINE combination strips the '2' dim from projected shape.
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))

  def test_affine_requires_linear_affine(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR,
        combination=conditioning_spec.Combination.AFFINE,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    with self.assertRaises(ValueError):
      layer.get_output_shape((4,), constants=constants)

  def test_affine_shift_requires_linear(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.AFFINE_SHIFT,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    with self.assertRaises(ValueError):
      layer.get_output_shape((4,), constants=constants)

  def test_affine_scale_requires_linear(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.AFFINE_SCALE,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    with self.assertRaises(ValueError):
      layer.get_output_shape((4,), constants=constants)

  def test_linear_affine_requires_affine(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.LINEAR_AFFINE,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    cond_seq = self.random_sequence(2, 8, 4)
    constants = self._make_constants(cond_seq)
    with self.assertRaises(ValueError):
      layer.get_output_shape((4,), constants=constants)

  def test_missing_constants(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, bind_only=True)
    with self.assertRaises(ValueError):
      layer.layer(x, training=False, constants=None)

  def test_missing_key(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x, bind_only=True)
    dummy_seq = self.random_sequence(2, 8, 4)
    with self.assertRaises(ValueError):
      layer.layer(x, training=False, constants={'other': dummy_seq})

  def test_from_config_identity_add(self):
    config = self.sl.Conditioning.Config(
        conditioning_name='cond',
        projection=conditioning_spec.Projection.IDENTITY,
        combination=conditioning_spec.Combination.ADD,
    )
    layer = self.make_layer(config)
    self.assertIsInstance(layer, self.sl.Conditioning)

    cond_seq = self.random_sequence(2, 5, 8)
    constants = self._make_constants(cond_seq)
    x = self.random_sequence(2, 5, 8)
    layer = self.init_layer(layer, x, constants=constants)
    y = layer.layer(x, training=False, constants=constants)
    self.assertEqual(y.channel_shape, (8,))
