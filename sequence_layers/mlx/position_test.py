"""Tests for position encoding MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import position
from sequence_layers.mlx import test_utils


class ApplyRotaryPositionalEncodingTest(parameterized.TestCase):

  def test_layer(self):
    layer = position.ApplyRotaryPositionalEncoding(max_wavelength=10000.0)
    test_utils.verify_contract(self, layer, (8,))

  def test_layer_multi_channel(self):
    layer = position.ApplyRotaryPositionalEncoding(
        max_wavelength=10000.0, axis=-1
    )
    test_utils.verify_contract(self, layer, (4, 8))

  def test_step_vs_layer(self):
    layer = position.ApplyRotaryPositionalEncoding(
        max_wavelength=10000.0,
        only_advance_position_for_valid_timesteps=False,
    )
    test_utils.verify_contract(self, layer, (8,), atol=1e-4, rtol=1e-4)

  def test_step_positions_advance(self):
    layer = position.ApplyRotaryPositionalEncoding(
        max_wavelength=10000.0,
        only_advance_position_for_valid_timesteps=True,
    )
    spec = bt.ShapeDType((8,), mx.float32)
    state = layer.get_initial_state(1, spec)

    # Step with valid mask.
    x1 = bt.MaskedSequence(
        mx.ones((1, 1, 8)),
        mx.ones((1, 1), dtype=mx.bool_),
    )
    _, state = layer.step(x1, state)
    # State starts at -1, cumsum(True)=1, position=-1+1=0.
    self.assertEqual(int(state[0, 0]), 0)

    # Step with invalid mask.
    x2 = bt.MaskedSequence(
        mx.ones((1, 1, 8)),
        mx.zeros((1, 1), dtype=mx.bool_),
    )
    _, state = layer.step(x2, state)
    # cumsum(False)=0, position=0+0=0. No advance.
    self.assertEqual(int(state[0, 0]), 0)

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax import position as jax_pos

    config = jax_pos.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=10000.0
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        position.ApplyRotaryPositionalEncoding,
    )
    test_utils.verify_contract(self, mlx_layer, (8,))


if __name__ == '__main__':
  absltest.main()
