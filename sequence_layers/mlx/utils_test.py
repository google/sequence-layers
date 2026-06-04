import unittest
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import sequence_layers.jax as jax_sl
import sequence_layers.mlx as mlx_sl
from sequence_layers.mlx import utils as mlx_utils
from sequence_layers.jax import utils as jax_utils


class UtilsTest(parameterized.TestCase):

  def test_make_layer_simple(self):
    jax_config = jax_sl.Scale.Config(scale=0.5)
    mlx_layer = mlx_utils.make_layer(jax_config)
    self.assertIsInstance(mlx_layer, mlx_sl.Scale)
    np.testing.assert_allclose(mlx_layer.config.scale.to_array(), 0.5)

  def test_make_layer_gated_unit(self):
    # Verifies our fix for gated units
    jax_config = jax_sl.GatedLinearUnit.Config()
    mlx_layer = mlx_utils.make_layer(jax_config)
    self.assertIsInstance(mlx_layer, mlx_sl.GatedLinearUnit)
    # Activations should be populated correctly
    self.assertIsNone(mlx_layer._feature_activation)
    self.assertIsNotNone(mlx_layer._gate_activation) # should be mx.sigmoid

  def test_get_required_stepwise_delay(self):
    from fractions import Fraction
    ratios = [Fraction(1, 2), Fraction(1, 4), Fraction(2, 1), Fraction(4, 1)]
    latencies = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for ratio in ratios:
      for latency in latencies:
        with self.subTest(ratio=str(ratio), latency=latency):
          try:
            mlx_delay = mlx_utils.get_required_stepwise_delay(ratio, latency)
            jax_delay = jax_utils.get_required_stepwise_delay(ratio, latency)
            self.assertEqual(mlx_delay, jax_delay)
          except NotImplementedError:
            with self.assertRaises(NotImplementedError):
              jax_utils.get_required_stepwise_delay(ratio, latency)

  @parameterized.product(
      accumulated_latency=(0, 1, 2, 3, 4),
  )
  def test_get_output_latency_simple(self, accumulated_latency):
    jax_config = jax_sl.Scale.Config(scale=0.5)
    mlx_config = mlx_sl.Scale.Config(scale=0.5)
    
    mlx_lat = mlx_utils.get_output_latency(mlx_config, accumulated_latency)
    jax_lat = jax_utils.get_output_latency(jax_config, accumulated_latency)
    
    self.assertEqual(mlx_lat, jax_lat)

  @parameterized.product(
      accumulated_latency=(0, 1, 2, 3, 4),
  )
  def test_get_output_latency_serial(self, accumulated_latency):
    jax_config = jax_sl.Serial.Config(
        layers=[
            jax_sl.Scale.Config(scale=0.5),
            jax_sl.Add.Config(shift=1.0),
        ]
    )
    mlx_config = mlx_sl.Serial.Config(
        layers=[
            mlx_sl.Scale.Config(scale=0.5),
            mlx_sl.Add.Config(shift=1.0),
        ]
    )
    
    mlx_lat = mlx_utils.get_output_latency(mlx_config, accumulated_latency)
    jax_lat = jax_utils.get_output_latency(jax_config, accumulated_latency)
    
    self.assertEqual(mlx_lat, jax_lat)

  def test_get_output_latency_validation(self):
    # Pooling with stride=2 has output_ratio = 1/2.
    # Divisor for latency is 1 / (1/2) = 2.
    # If accumulated_latency is odd (e.g. 1), it should raise ValueError in JAX.
    jax_config = jax_sl.MaxPooling1D.Config(pool_size=2, strides=2)
    mlx_config = mlx_sl.MaxPooling1D.Config(pool_size=2, strides=2)
    
    # For even latency, both should succeed and match
    mlx_lat_even = mlx_utils.get_output_latency(mlx_config, 2)
    jax_lat_even = jax_utils.get_output_latency(jax_config, 2)
    self.assertEqual(mlx_lat_even, jax_lat_even)
    
    # For odd latency, JAX should raise ValueError
    with self.assertRaises(ValueError):
      jax_utils.get_output_latency(jax_config, 1)
      
    # Currently, MLX might NOT raise ValueError because _get_accumulated_output_latency bypasses it.
    # We assert it raises ValueError to enforce parity.
    with self.assertRaises(ValueError):
      mlx_utils.get_output_latency(mlx_config, 1)


if __name__ == '__main__':
  absltest.main()
