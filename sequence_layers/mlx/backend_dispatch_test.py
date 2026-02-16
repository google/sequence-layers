"""Tests for the backend dispatch mechanism."""

from absl.testing import absltest
from absl.testing import parameterized


class BackendDispatchTest(parameterized.TestCase):

  def test_make_default_returns_linen(self):
    """config.make() (no args) still returns Linen module."""
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Identity.Config()
    linen_model = config.make()
    self.assertEqual(type(linen_model).__name__, 'Identity')
    # Check it's a Linen module, not MLX.
    from flax import linen as nn

    self.assertIsInstance(linen_model, nn.Module)

  def test_make_backend_mlx(self):
    """config.make(backend='mlx') returns MLX module."""
    import sequence_layers.mlx  # Register backends.
    from sequence_layers.jax import simple as jax_simple
    from sequence_layers.mlx import simple as mlx_simple

    config = jax_simple.Identity.Config()
    mlx_model = config.make(backend='mlx')
    self.assertIsInstance(mlx_model, mlx_simple.Identity)

  def test_unregistered_backend_raises(self):
    from sequence_layers.jax import simple as jax_simple

    config = jax_simple.Identity.Config()
    with self.assertRaises(ValueError):
      config.make(backend='nonexistent')

  def test_nested_configs_dispatch(self):
    """Nested configs (e.g. Serial) correctly dispatch children."""
    import sequence_layers.mlx
    import sequence_layers.jax as sl

    config = sl.Serial.Config([
        sl.Identity.Config(),
        sl.Dense.Config(features=8),
    ])
    mlx_model = config.make(backend='mlx')
    from sequence_layers.mlx import combinators

    self.assertIsInstance(mlx_model, combinators.Serial)
    self.assertEqual(len(mlx_model.layers), 2)

  def test_mro_lookup(self):
    """Config subclasses inherit backend factories via MRO."""
    import sequence_layers.mlx
    import dataclasses
    from sequence_layers.jax import simple as jax_simple

    # Create a subclass of Identity.Config.
    @dataclasses.dataclass(frozen=True)
    class MyIdentityConfig(jax_simple.Identity.Config):
      pass

    # Should still find the factory via MRO.
    mlx_model = MyIdentityConfig().make(backend='mlx')
    from sequence_layers.mlx import simple as mlx_simple

    self.assertIsInstance(mlx_model, mlx_simple.Identity)


if __name__ == '__main__':
  absltest.main()
