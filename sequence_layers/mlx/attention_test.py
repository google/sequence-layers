"""Tests for attention MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import attention
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import position
from sequence_layers.mlx import test_utils


class DotProductSelfAttentionTest(parameterized.TestCase):

  def test_layer(self):
    layer = attention.DotProductSelfAttention(
        in_features=16,
        num_heads=4,
        units_per_head=8,
        max_past_horizon=32,
    )
    test_utils.verify_contract(self, layer, (16,), atol=1e-4, rtol=1e-4)

  def test_causal(self):
    layer = attention.DotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=64,
        max_future_horizon=0,
    )
    test_utils.verify_contract(self, layer, (8,), atol=1e-4, rtol=1e-4)

  def test_gqa(self):
    """Test Grouped Query Attention (fewer KV heads)."""
    layer = attention.DotProductSelfAttention(
        in_features=16,
        num_heads=8,
        units_per_head=4,
        max_past_horizon=32,
        num_kv_heads=2,
    )
    test_utils.verify_contract(self, layer, (16,), atol=1e-4, rtol=1e-4)

  def test_output_shape(self):
    layer = attention.DotProductSelfAttention(
        in_features=16,
        num_heads=4,
        units_per_head=8,
        max_past_horizon=32,
    )
    self.assertEqual(layer.get_output_shape((16,)), (4, 8))

  def test_step_builds_kv_cache(self):
    layer = attention.DotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=10,
    )
    spec = bt.ShapeDType((8,), mx.float32)
    state = layer.get_initial_state(1, spec)

    for i in range(5):
      x = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 8)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      _, state = layer.step(x, state)

    # Check KV cache has been populated.
    kv_keys = state[0]
    self.assertEqual(kv_keys.shape[1], 10)  # buffer size
    kv_mask = state[2]
    self.assertEqual(mx.sum(kv_mask).item(), 5)  # 5 of 10 slots filled

  @parameterized.parameters(
      (2, 4, 4, 0),
  )
  def test_use_kv_cache_ringbuffer(
      self, num_heads, units_per_head, max_past_horizon, max_future_horizon
  ):
    """Test ring buffer wrap-around: layer() vs step() parity.

    With block_size=1 (default), once the ring buffer wraps, the current
    write-before-read implementation overwrites the oldest key in the
    attention window before the query can attend to it. This causes
    step() to see max_past keys while layer() sees max_past + 1 keys
    for the same query position, breaking bitwise parity.

    Sweep time shorter, equal, and longer than max_past_horizon to
    exercise the wrap-around.
    """
    config = attention.DotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
    )
    layer = config.make(backend='mlx')

    for time in [1, max_past_horizon, max_past_horizon + 2]:
      with self.subTest(f'time_{time}'):
        test_utils.verify_contract(
            self, layer, (8,), time=time, atol=1e-4, rtol=1e-4
        )

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
    y = layer.layer(x)
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
    y_pds = layer.layer(x)
    y_no_pds = layer_no_pds.layer(x)
    np.testing.assert_allclose(
        np.array(y_pds.values), np.array(y_no_pds.values), atol=1e-5
    )

    # After modifying per_dim_scale, output should differ.
    layer._per_dim_scale = mx.ones((4,))
    y_modified = layer.layer(x)
    self.assertFalse(
        np.allclose(
            np.array(y_pds.values), np.array(y_modified.values), atol=1e-5
        )
    )

  def test_per_dim_scale_step(self):
    """Test per_dim_scale works in step mode."""
    layer = attention.DotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=10,
        per_dim_scale=True,
    )
    test_utils.verify_contract(self, layer, (8,), atol=1e-4, rtol=1e-4)


class DeferredDotProductSelfAttentionTest(parameterized.TestCase):

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax.attention import (
        dot_product_self_attention as jax_attn,
    )

    config = jax_attn.DotProductSelfAttention.Config(
        num_heads=4,
        units_per_head=8,
        max_past_horizon=32,
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        attention.DeferredDotProductSelfAttention,
    )

    x = test_utils.random_sequence(1, 5, 16)
    y = mlx_layer.layer(x)
    self.assertEqual(y.channel_shape, (4, 8))


class DotProductAttentionTest(parameterized.TestCase):
  """Tests for cross-attention."""

  def _make_constants(self, batch, time, features, name='source'):
    source = test_utils.random_sequence(batch, time, features)
    return {name: source}

  def test_layer(self):
    layer = attention.DotProductAttention(
        in_features=8,
        source_features=12,
        source_name='source',
        num_heads=2,
        units_per_head=4,
    )
    constants = self._make_constants(2, 6, 12)
    test_utils.verify_contract(
        self,
        layer,
        (8,),
        constants=constants,
        atol=1e-4,
        rtol=1e-4,
    )

  def test_output_shape(self):
    layer = attention.DotProductAttention(
        in_features=16,
        source_features=16,
        source_name='enc',
        num_heads=4,
        units_per_head=8,
    )
    self.assertEqual(layer.get_output_shape((16,)), (4, 8))

  def test_step_reuses_precomputed_kv(self):
    layer = attention.DotProductAttention(
        in_features=8,
        source_features=12,
        source_name='source',
        num_heads=2,
        units_per_head=4,
    )
    constants = self._make_constants(1, 6, 12)
    spec = bt.ShapeDType((8,), mx.float32)
    state = layer.get_initial_state(1, spec, constants=constants)
    # KV should be pre-computed.
    keys_v = state[0]
    self.assertEqual(keys_v.shape, (1, 6, 2, 4))

    for _ in range(3):
      x = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 8)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      y, state = layer.step(x, state, constants=constants)
      self.assertEqual(y.channel_shape, (2, 4))

  def test_missing_source_raises(self):
    layer = attention.DotProductAttention(
        in_features=8,
        source_features=8,
        source_name='missing',
        num_heads=2,
        units_per_head=4,
    )
    x = test_utils.random_sequence(1, 3, 8)
    with self.assertRaises(ValueError):
      layer.layer(x, constants={})

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    config = jax_cross_attn.DotProductAttention.Config(
        source_name='enc',
        num_heads=4,
        units_per_head=8,
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        attention.DeferredDotProductAttention,
    )
    source = test_utils.random_sequence(1, 6, 16)
    constants = {'enc': source}
    x = test_utils.random_sequence(1, 4, 16)
    y = mlx_layer.layer(x, constants=constants)
    self.assertEqual(y.channel_shape, (4, 8))


class StreamingDotProductAttentionTest(parameterized.TestCase):
  """Tests for streaming cross-attention."""

  def _make_source(self, batch, time, features, name='source'):
    return test_utils.random_sequence(batch, time, features)

  def test_layer_basic(self):
    """Basic layer mode with banded visibility mask."""
    layer = attention.StreamingDotProductAttention(
        in_features=8,
        source_features=12,
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=4,
    )
    source = self._make_source(2, 8, 12)
    x = test_utils.random_sequence(2, 8, 8)
    y = layer.layer(x, constants={'source': source})
    self.assertEqual(y.channel_shape, (2, 4))
    self.assertEqual(y.shape, (2, 8, 2, 4))

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
    state = layer.get_initial_state(1, spec, constants={'source': source})

    for _ in range(5):
      x = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 8)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      src = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 12)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      _, state, _ = layer.step_with_emits(x, state, constants={'source': src})

    kv_keys = state[0]
    self.assertEqual(kv_keys.shape[1], 10)  # buffer size

  def test_step_matches_layer(self):
    """Step-by-step with streaming constants matches layer()."""
    layer = attention.StreamingDotProductAttention(
        in_features=8,
        source_features=12,
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=16,
    )
    batch, time = 1, 8
    x = test_utils.random_sequence(batch, time, 8)
    source = self._make_source(batch, time, 12)
    constants = {'source': source}

    # Layer mode.
    y_layer = layer.layer(x, constants=constants)

    # Step-by-step mode.
    y_step, _ = test_utils.step_by_step(
        layer,
        x,
        block_size=1,
        stream_constants={'source': source},
    )

    np.testing.assert_allclose(
        np.array(y_step.values),
        np.array(y_layer.values),
        atol=1e-4,
        rtol=1e-4,
        err_msg='step vs layer mismatch',
    )

  def test_with_future_horizon(self):
    """Query delay buffer with max_future_horizon > 0."""
    layer = attention.StreamingDotProductAttention(
        in_features=8,
        source_features=8,
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=4,
        max_future_horizon=2,
        use_query_delay_buffer=True,
    )
    self.assertEqual(layer.input_latency, 2)
    source = self._make_source(1, 8, 8)
    spec = bt.ShapeDType((8,), mx.float32)
    state = layer.get_initial_state(1, spec, constants={'source': source})

    # Verify delay buffer is in state.
    q_delay_values = state[7]
    self.assertFalse(isinstance(q_delay_values, tuple))
    self.assertEqual(q_delay_values.shape[1], 2)

    # Run a few steps to make sure it doesn't crash.
    for _ in range(5):
      x = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 8)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      src = bt.MaskedSequence(
          mx.random.normal(shape=(1, 1, 8)),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      y, state, _ = layer.step_with_emits(x, state, constants={'source': src})
      self.assertEqual(y.channel_shape, (2, 4))

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
    state = layer.get_initial_state(1, spec, constants={'source': source})
    # Delay buffer should be empty tuples.
    self.assertIsInstance(state[7], tuple)
    self.assertEqual(state[7], ())

  def test_with_rope(self):
    """Q/K processing networks (RoPE)."""
    rope_q = position.ApplyRotaryPositionalEncoding(
        max_wavelength=10000.0, axis=-1
    )
    rope_k = position.ApplyRotaryPositionalEncoding(
        max_wavelength=10000.0, axis=-1
    )
    layer = attention.StreamingDotProductAttention(
        in_features=8,
        source_features=12,
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=16,
        query_network=rope_q,
        key_network=rope_k,
    )
    source = self._make_source(1, 5, 12)
    x = test_utils.random_sequence(1, 5, 8)
    y = layer.layer(x, constants={'source': source})
    self.assertEqual(y.shape, (1, 5, 2, 4))

  def test_output_shape(self):
    layer = attention.StreamingDotProductAttention(
        in_features=16,
        source_features=16,
        source_name='source',
        num_heads=4,
        units_per_head=8,
        max_past_horizon=8,
    )
    self.assertEqual(layer.get_output_shape((16,)), (4, 8))

  def test_from_config(self):
    """Both Streaming and StreamingLocal configs produce correct layer."""
    import sequence_layers.mlx
    from sequence_layers.jax.attention import (
        streaming_dot_product_attention as jax_streaming_attn,
    )
    from sequence_layers.jax.attention import (
        streaming_local_dot_product_attention as jax_streaming_local_attn,
    )

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='source',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        attention.DeferredStreamingDotProductAttention,
    )

    source = test_utils.random_sequence(1, 6, 8)
    x = test_utils.random_sequence(1, 6, 8)
    y = mlx_layer.layer(x, constants={'source': source})
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
    mlx_local = local_config.make(backend='mlx')
    self.assertIsInstance(
        mlx_local,
        attention.DeferredStreamingDotProductAttention,
    )


class LocalDotProductSelfAttentionTest(parameterized.TestCase):

  def test_layer(self):
    layer = attention.LocalDotProductSelfAttention(
        in_features=16,
        num_heads=4,
        units_per_head=4,
        max_past_horizon=8,
        block_size_config=2,
    )
    test_utils.verify_contract(self, layer, (16,), atol=1e-4, rtol=1e-4)
    # Also test with time > max_past_horizon to exercise ring buffer wrap.
    test_utils.verify_contract(self, layer, (16,), time=10, atol=1e-4, rtol=1e-4)

  def test_block_size(self):
    layer = attention.LocalDotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=4,
        block_size_config=4,
    )
    self.assertEqual(layer.block_size, 4)

  def test_with_future_horizon(self):
    layer = attention.LocalDotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=4,
        max_future_horizon=2,
        block_size_config=1,
    )
    self.assertEqual(layer.input_latency, 2)
    test_utils.verify_contract(
        self, layer, (8,), atol=1e-4, rtol=1e-4, test_step=False
    )

  def test_with_soft_cap(self):
    layer = attention.LocalDotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
        block_size_config=1,
        attention_logits_soft_cap=50.0,
    )
    test_utils.verify_contract(self, layer, (8,), atol=1e-4, rtol=1e-4)

  def test_with_rope(self):
    rope = position.ApplyRotaryPositionalEncoding(
        max_wavelength=10000.0, axis=-1
    )
    layer = attention.LocalDotProductSelfAttention(
        in_features=8,
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
        block_size_config=1,
        query_network=rope,
        key_network=position.ApplyRotaryPositionalEncoding(
            max_wavelength=10000.0, axis=-1
        ),
    )
    test_utils.verify_contract(self, layer, (8,), atol=1e-4, rtol=1e-4)

  def test_output_shape(self):
    layer = attention.LocalDotProductSelfAttention(
        in_features=16,
        num_heads=4,
        units_per_head=8,
        max_past_horizon=8,
        block_size_config=2,
    )
    self.assertEqual(layer.get_output_shape((16,)), (4, 8))

  def test_from_config(self):
    import sequence_layers.mlx
    from sequence_layers.jax.attention import (
        local_dot_product_self_attention as jax_local_attn,
    )

    config = jax_local_attn.LocalDotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        block_size=2,
        max_past_horizon=8,
    )
    mlx_layer = config.make(backend='mlx')
    self.assertIsInstance(
        mlx_layer,
        attention.DeferredLocalDotProductSelfAttention,
    )
    self.assertEqual(mlx_layer.block_size, 2)

    x = test_utils.random_sequence(1, 8, 8)
    y = mlx_layer.layer(x)
    self.assertEqual(y.channel_shape, (2, 4))


if __name__ == '__main__':
  absltest.main()
