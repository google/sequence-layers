"""Tests for attention MLX sequence layers."""

from absl.testing import absltest
from absl.testing import parameterized
import mlx.core as mx
import numpy as np

from sequence_layers.mlx import attention
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import position
from sequence_layers.mlx import test_utils
from sequence_layers.specs import attention_behaviors as spec


class DotProductSelfAttentionTest(
    test_utils.SequenceLayerTest, spec.DotProductSelfAttentionTest
):

  def test_step_builds_kv_cache(self):
    layer = attention.DotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=10,
    )
    spec = bt.ShapeDType((8,), mx.float32)
    state = layer.get_initial_state(1, spec, training=False)

    for i in range(5):
      x = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 8)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      _, state = layer.step(x, state, training=False)

    # Check KV cache has been populated.
    kv_mask = state[2]
    self.assertEqual(mx.sum(kv_mask).item(), 5)

  def test_with_query_key_networks(self):
    """Test with RoPE on Q/K."""
    rope = position.ApplyRotaryPositionalEncoding(
        max_wavelength=10000.0, axis=-1
    )
    layer = attention.DotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=32,
        query_network=rope,
        key_network=position.ApplyRotaryPositionalEncoding(
            max_wavelength=10000.0, axis=-1
        ),
    )
    x = test_utils.random_sequence(1, 5, 8)
    y = layer.layer(x, training=False)
    self.assertEqual(y.shape, (1, 5, 2, 4))

  def test_per_dim_scale(self):
    """Test per_dim_scale creates parameter and affects output."""
    layer = attention.DotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=32,
        per_dim_scale=True,
    )
    self.assertIsNotNone(layer._per_dim_scale)
    self.assertEqual(layer._per_dim_scale.shape, (4,))
    np.testing.assert_array_equal(layer._per_dim_scale, np.zeros(4))

    # At initialization (zeros), output should match per_dim_scale=False.
    layer_no_pds = attention.DotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=32,
        per_dim_scale=False,
    )
    # Copy weights so projections match.
    layer_no_pds.q_proj = layer.q_proj
    layer_no_pds.kv_proj = layer.kv_proj

    x = test_utils.random_sequence(1, 5, 8)
    y_pds = layer.layer(x, training=False)
    y_no_pds = layer_no_pds.layer(x, training=False)
    np.testing.assert_allclose(
        np.array(y_pds.values), np.array(y_no_pds.values), atol=1e-5
    )

    # After modifying per_dim_scale, output should differ.
    layer._per_dim_scale = mx.ones((4,))
    y_modified = layer.layer(x, training=False)
    self.assertFalse(
        np.allclose(
            np.array(y_pds.values), np.array(y_modified.values), atol=1e-5
        )
    )


class DotProductSelfAttentionFromConfigTest(test_utils.SequenceLayerTest):

  def test_from_config(self):
    from sequence_layers.jax.attention import dot_product_self_attention as jax_attn
    import sequence_layers.mlx

    config = jax_attn.DotProductSelfAttention.Config(
        num_heads=4,
        units_per_head=8,
        max_past_horizon=32,
    )
    mlx_layer = attention.DotProductSelfAttention.from_config(config)
    self.assertIsInstance(
        mlx_layer,
        attention.DotProductSelfAttention,
    )

    x = test_utils.random_sequence(1, 5, 16)
    y = mlx_layer.layer(x, training=False)
    self.assertEqual(y.channel_shape, (4, 8))


class DotProductAttentionTest(
    test_utils.SequenceLayerTest,
    spec.DotProductAttentionTest,
):
  """Tests for cross-attention."""

  def test_from_config(self):
    from sequence_layers.jax.attention import dot_product_attention as jax_cross_attn
    import sequence_layers.mlx

    config = jax_cross_attn.DotProductAttention.Config(
        source_name='enc',
        num_heads=4,
        units_per_head=8,
    )
    mlx_layer = attention.DotProductAttention.from_config(config)
    self.assertIsInstance(
        mlx_layer,
        attention.DotProductAttention,
    )
    source = test_utils.random_sequence(1, 6, 16)
    constants = {'enc': source}
    x = test_utils.random_sequence(1, 4, 16)
    y = mlx_layer.layer(x, constants=constants, training=False)
    self.assertEqual(y.channel_shape, (4, 8))


class StreamingDotProductAttentionTest(
    test_utils.SequenceLayerTest, spec.StreamingDotProductAttentionTest
):
  """Tests for streaming cross-attention."""

  def _make_source(self, batch, time, features, name='source'):
    return test_utils.random_sequence(batch, time, features)

  def test_step_builds_kv_cache(self):
    """KV buffer grows correctly during step mode."""
    layer = attention.StreamingDotProductAttention(
        in_features=8,
        source_features=12,
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=10,
    )
    source = self._make_source(1, 1, 12)
    spec = bt.ShapeDType((8,), mx.float32)
    state = layer.get_initial_state(
        1, spec, training=False, constants={'source': source}
    )

    for _ in range(5):
      x = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 8)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      src = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 12)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      _, state, _ = layer.step_with_emits(
          x, state, training=False, constants={'source': src}
      )

    kv_keys = state[0]
    self.assertEqual(kv_keys.shape[1], 10)  # buffer size

  def test_no_query_delay_buffer(self):
    """use_query_delay_buffer=False has no delay."""
    layer = attention.StreamingDotProductAttention(
        in_features=8,
        source_features=8,
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=4,
        max_future_horizon=2,
        use_query_delay_buffer=False,
    )
    self.assertEqual(layer.input_latency, 0)
    source = self._make_source(1, 8, 8)
    spec = bt.ShapeDType((8,), mx.float32)
    state = layer.get_initial_state(
        1, spec, constants={'source': source}, training=False
    )
    # Delay buffer should be empty tuples.
    self.assertIsInstance(state[7], tuple)
    self.assertEqual(state[7], ())

  def test_from_config(self):
    """Both Streaming and StreamingLocal configs produce correct layer."""
    from sequence_layers.jax.attention import streaming_dot_product_attention as jax_streaming_attn
    from sequence_layers.jax.attention import streaming_local_dot_product_attention as jax_streaming_local_attn
    import sequence_layers.mlx

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
    )
    mlx_layer = attention.StreamingDotProductAttention.from_config(config)
    self.assertIsInstance(
        mlx_layer,
        attention.StreamingDotProductAttention,
    )

    source = test_utils.random_sequence(1, 6, 8)
    x = test_utils.random_sequence(1, 6, 8)
    y = mlx_layer.layer(x, constants={'source': source}, training=False)
    self.assertEqual(y.channel_shape, (2, 4))

    # StreamingLocal config should also work.
    local_config = (
        jax_streaming_local_attn.StreamingLocalDotProductAttention.Config(
            source_name='source',
            num_heads=2,
            units_per_head=4,
            block_size=2,
            max_past_horizon=8,
        )
    )
    mlx_local = attention.StreamingDotProductAttention.from_config(local_config)
    self.assertIsInstance(
        mlx_local,
        attention.StreamingDotProductAttention,
    )


class LocalDotProductSelfAttentionTest(
    test_utils.SequenceLayerTest, spec.LocalDotProductSelfAttentionTest
):

  test_step_in_future_horizon = False

  def test_from_config(self):
    from sequence_layers.jax.attention import local_dot_product_self_attention as jax_local_attn
    import sequence_layers.mlx

    config = jax_local_attn.LocalDotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        block_size=2,
        max_past_horizon=8,
    )
    mlx_layer = attention.LocalDotProductSelfAttention.from_config(config)
    self.assertIsInstance(
        mlx_layer,
        attention.LocalDotProductSelfAttention,
    )
    self.assertEqual(mlx_layer.block_size, 1)

    x = test_utils.random_sequence(1, 8, 8)
    y = mlx_layer.layer(x, training=False)
    self.assertEqual(y.channel_shape, (2, 4))


if __name__ == '__main__':
  absltest.main()
