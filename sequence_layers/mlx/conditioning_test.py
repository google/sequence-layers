"""Tests for Conditioning MLX sequence layer."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import conditioning
from sequence_layers.mlx import test_utils

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence


def _make_constants(conditioning_seq, name='cond'):
  return {name: conditioning_seq}


class ConditioningIdentityAddTest(parameterized.TestCase):
  """Tests for IDENTITY projection + ADD combination."""

  def test_layer(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    cond_seq = test_utils.random_sequence(2, 8, 4)
    constants = _make_constants(cond_seq)
    test_utils.verify_contract(self, layer, (4,), constants=constants)

  def test_output_shape(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    cond_seq = test_utils.random_sequence(2, 8, 4)
    constants = _make_constants(cond_seq)
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))

  def test_broadcast_add(self):
    """x: [B,T,4], c: [B,T,1] → [B,T,4] via broadcast."""
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    cond_seq = test_utils.random_sequence(2, 8, 1)
    constants = _make_constants(cond_seq)
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))

  def test_tensor_conditioning(self):
    """Conditioning with a [B, dim] tensor (not a Sequence)."""
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    cond_tensor = mx.random.normal(shape=(2, 4))
    constants = _make_constants(cond_tensor)
    x = test_utils.random_sequence(2, 8, 4)
    y = layer.layer(x, constants=constants)
    self.assertEqual(y.channel_shape, (4,))

  def test_step_non_streaming(self):
    """Non-streaming: full conditioning passed, layer slices per step."""
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
        streaming=False,
    )
    cond_seq = test_utils.random_sequence(2, 8, 4)
    constants = _make_constants(cond_seq)
    x = test_utils.random_sequence(2, 8, 4)

    # Layer mode.
    y_layer = layer.layer(x, constants=constants)

    # Step mode (pass full conditioning; layer slices internally).
    y_step, _ = test_utils.step_by_step(
        layer, x, block_size=1, constants=constants
    )
    np.testing.assert_allclose(
        np.array(y_step.values),
        np.array(y_layer.values),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_step_streaming(self):
    """Streaming: conditioning chunks arrive with input chunks."""
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
        streaming=True,
    )
    cond_seq = test_utils.random_sequence(2, 8, 4)
    x = test_utils.random_sequence(2, 8, 4)

    # Layer mode.
    y_layer = layer.layer(x, constants=_make_constants(cond_seq))

    # Step mode with stream_constants.
    y_step, _ = test_utils.step_by_step(
        layer,
        x,
        block_size=1,
        stream_constants=_make_constants(cond_seq),
    )
    np.testing.assert_allclose(
        np.array(y_step.values),
        np.array(y_layer.values),
        atol=1e-5,
        rtol=1e-5,
    )


class ConditioningIdentityConcatTest(parameterized.TestCase):

  def test_layer(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.CONCAT,
    )
    cond_seq = test_utils.random_sequence(2, 8, 3)
    constants = _make_constants(cond_seq)
    x = test_utils.random_sequence(2, 8, 4)
    y = layer.layer(x, constants=constants)
    self.assertEqual(y.channel_shape, (7,))

  def test_concat_before(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.CONCAT_BEFORE,
    )
    cond_seq = test_utils.random_sequence(2, 8, 3)
    constants = _make_constants(cond_seq)
    x = test_utils.random_sequence(2, 8, 4)
    y = layer.layer(x, constants=constants)
    self.assertEqual(y.channel_shape, (7,))
    # CONCAT_BEFORE should have conditioning first.
    np.testing.assert_allclose(
        np.array(y.values[:, :, :3]),
        np.array(cond_seq.values),
        atol=1e-5,
    )


class ConditioningIdentityMulTest(parameterized.TestCase):

  def test_layer(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.MUL,
    )
    cond_seq = test_utils.random_sequence(2, 8, 4)
    constants = _make_constants(cond_seq)
    test_utils.verify_contract(self, layer, (4,), constants=constants)


class ConditioningLinearAddTest(parameterized.TestCase):

  def test_layer(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.LINEAR,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    cond_seq = test_utils.random_sequence(2, 8, 6)
    constants = _make_constants(cond_seq)
    # input shape (4,), conditioning shape (6,), projected to (4,).
    test_utils.verify_contract(self, layer, (4,), constants=constants)

  def test_output_shape(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.LINEAR,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    cond_seq = test_utils.random_sequence(2, 8, 6)
    constants = _make_constants(cond_seq)
    # LINEAR projects conditioning to input channel shape.
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))

  def test_with_projection_channel_shape(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.LINEAR,
        combination=conditioning.Conditioning.Combination.ADD,
        projection_channel_shape=(8,),
    )
    cond_seq = test_utils.random_sequence(2, 8, 6)
    constants = _make_constants(cond_seq)
    # Projects to (8,), then broadcast-add with input (8,).
    self.assertEqual(layer.get_output_shape((8,), constants=constants), (8,))


class ConditioningLinearAffineShiftTest(parameterized.TestCase):

  def test_layer(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.LINEAR,
        combination=conditioning.Conditioning.Combination.AFFINE_SHIFT,
    )
    cond_seq = test_utils.random_sequence(2, 8, 6)
    constants = _make_constants(cond_seq)
    test_utils.verify_contract(self, layer, (4,), constants=constants)


class ConditioningLinearAffineScaleTest(parameterized.TestCase):

  def test_layer(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.LINEAR,
        combination=conditioning.Conditioning.Combination.AFFINE_SCALE,
    )
    cond_seq = test_utils.random_sequence(2, 8, 6)
    constants = _make_constants(cond_seq)
    test_utils.verify_contract(self, layer, (4,), constants=constants)


class ConditioningLinearAffineTest(parameterized.TestCase):

  def test_layer(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.LINEAR_AFFINE,
        combination=conditioning.Conditioning.Combination.AFFINE,
    )
    cond_seq = test_utils.random_sequence(2, 8, 6)
    constants = _make_constants(cond_seq)
    test_utils.verify_contract(self, layer, (4,), constants=constants)

  def test_output_shape(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.LINEAR_AFFINE,
        combination=conditioning.Conditioning.Combination.AFFINE,
    )
    cond_seq = test_utils.random_sequence(2, 8, 6)
    constants = _make_constants(cond_seq)
    # AFFINE combination strips the '2' dim from projected shape.
    self.assertEqual(layer.get_output_shape((4,), constants=constants), (4,))


class ConditioningValidationTest(parameterized.TestCase):

  def test_affine_requires_linear_affine(self):
    with self.assertRaises(ValueError):
      conditioning.Conditioning(
          conditioning_name='cond',
          projection=conditioning.Conditioning.Projection.LINEAR,
          combination=conditioning.Conditioning.Combination.AFFINE,
      )

  def test_affine_shift_requires_linear(self):
    with self.assertRaises(ValueError):
      conditioning.Conditioning(
          conditioning_name='cond',
          projection=conditioning.Conditioning.Projection.IDENTITY,
          combination=conditioning.Conditioning.Combination.AFFINE_SHIFT,
      )

  def test_affine_scale_requires_linear(self):
    with self.assertRaises(ValueError):
      conditioning.Conditioning(
          conditioning_name='cond',
          projection=conditioning.Conditioning.Projection.IDENTITY,
          combination=conditioning.Conditioning.Combination.AFFINE_SCALE,
      )

  def test_linear_affine_requires_affine(self):
    with self.assertRaises(ValueError):
      conditioning.Conditioning(
          conditioning_name='cond',
          projection=conditioning.Conditioning.Projection.LINEAR_AFFINE,
          combination=conditioning.Conditioning.Combination.ADD,
      )

  def test_missing_constants(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    x = test_utils.random_sequence(2, 8, 4)
    with self.assertRaises(ValueError):
      layer.layer(x, constants=None)

  def test_missing_key(self):
    layer = conditioning.Conditioning(
        conditioning_name='cond',
        projection=conditioning.Conditioning.Projection.IDENTITY,
        combination=conditioning.Conditioning.Combination.ADD,
    )
    x = test_utils.random_sequence(2, 8, 4)
    with self.assertRaises(ValueError):
      layer.layer(x, constants={'other': mx.zeros((2, 4))})


class ConditioningFromConfigTest(parameterized.TestCase):

  def test_from_config_identity_add(self):
    import sequence_layers.mlx
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.IDENTITY,
        combination=jax_cond.BaseConditioning.Combination.ADD,
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, conditioning.Conditioning)

    cond_seq = test_utils.random_sequence(2, 5, 8)
    constants = _make_constants(cond_seq)
    x = test_utils.random_sequence(2, 5, 8)
    y = mlx_layer.layer(x, constants=constants)
    self.assertEqual(y.channel_shape, (8,))

  def test_from_config_linear_add(self):
    import sequence_layers.mlx
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.LINEAR,
        combination=jax_cond.BaseConditioning.Combination.ADD,
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, conditioning.Conditioning)

  def test_from_config_linear_affine(self):
    import sequence_layers.mlx
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.LINEAR_AFFINE,
        combination=jax_cond.BaseConditioning.Combination.AFFINE,
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(mlx_layer, conditioning.Conditioning)


if __name__ == '__main__':
  absltest.main()
