"""Behavior tests for attention layers.

Backend-specific test files should inherit from these tests.
"""

# pylint: disable=abstract-method

from absl.testing import parameterized

from sequence_layers.specs import test_utils


class DotProductSelfAttentionTest(test_utils.SequenceLayerTest):
  """Test behavior of DotProductSelfAttention layer."""

  def test_layer(self):
    layer = self.sl.DotProductSelfAttention.Config(
        num_heads=4,
        units_per_head=8,
        max_past_horizon=32,
        name='dot_product_self_attention',
    ).make()
    x = self.random_sequence(2, 5, 16)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, atol=1e-4, rtol=1e-4)

  def test_causal(self):
    layer = self.sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=64,
        max_future_horizon=0,
    ).make()
    x = self.random_sequence(2, 5, 8)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, atol=1e-4, rtol=1e-4)

  def test_gqa(self):
    """Test Grouped Query Attention (fewer KV heads)."""
    layer = self.sl.DotProductSelfAttention.Config(
        num_heads=8,
        units_per_head=4,
        max_past_horizon=32,
        num_kv_heads=2,
        input_projection=self.sl.SeparateQueryKeyValueProjection(),
    ).make()
    x = self.random_sequence(2, 5, 16)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, atol=1e-4, rtol=1e-4)

  def test_output_shape(self):
    layer = self.sl.DotProductSelfAttention.Config(
        num_heads=4,
        units_per_head=8,
        max_past_horizon=32,
    ).make()
    x = self.random_sequence(2, 5, 16)
    layer = self.init_layer(layer, x)
    self.assertEqual(layer.get_output_shape((16,)), (4, 8))

  def test_step_builds_kv_cache(self):
    # We cannot easily test this generically if `state` structures differ,
    # but we can check that `state` is returned and updated.
    # Let's write a generic version or keep it backend-specific if it relies
    # too much on backend-specific types.
    # Actually, the state structure in JAX is:
    # (keys, values, state_index, ...) or similar.
    # In MLX it's:
    # (keys, values, step)
    # But wait, both have a step/index in state.
    # Let's make this backend-specific since the state layout is too divergent
    # (e.g. JAX uses flax.FrozenDict/tuple, MLX uses list/tuple).
    pass

  def test_per_dim_scale(self):
    # Also relies on checking internal attributes like `layer._per_dim_scale`
    # which are named differently or don't exist in the same way.
    # In JAX it is a parameter. In MLX it is a parameter.
    # Let's keep per_dim_scale testing backend-specific for now, or at least the
    # parameter checking part.
    pass

  def test_per_dim_scale_step(self):
    layer = self.sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=10,
        per_dim_scale=True,
    ).make()
    x = self.random_sequence(2, 5, 8)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, atol=1e-4, rtol=1e-4)

  @parameterized.product(
      (
          # CombinedQueryKeyValueProjection. GQA is not supported.
          {
              'input_projection_name': 'CombinedQueryKeyValueProjection',
              'share_kv_projection': False,
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {
              'input_projection_name': 'CombinedQueryKeyValueProjection',
              'share_kv_projection': True,
              'num_heads': 3,
              'num_kv_heads': None,
          },
          # SeparateQueryKeyValueProjection. MHA and GQA supported.
          {
              'input_projection_name': 'SeparateQueryKeyValueProjection',
              'share_kv_projection': False,
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {
              'input_projection_name': 'SeparateQueryKeyValueProjection',
              'share_kv_projection': False,
              'num_heads': 6,
              'num_kv_heads': 3,
          },
          # QueryAndKeyValueProjection. MHA and GQA supported.
          {
              'input_projection_name': 'QueryAndKeyValueProjection',
              'share_kv_projection': False,
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {
              'input_projection_name': 'QueryAndKeyValueProjection',
              'share_kv_projection': False,
              'num_heads': 6,
              'num_kv_heads': 3,
          },
          # QueryAndSharedKeyValueProjection. MHA and GQA supported.
          {
              'input_projection_name': 'QueryAndSharedKeyValueProjection',
              'share_kv_projection': False,
              'num_heads': 3,
              'num_kv_heads': None,
          },
          {
              'input_projection_name': 'QueryAndSharedKeyValueProjection',
              'share_kv_projection': False,
              'num_heads': 6,
              'num_kv_heads': 3,
          },
      ),
  )
  def test_projection_config_contract(
      self,
      input_projection_name: str,
      share_kv_projection: bool,
      num_heads: int,
      num_kv_heads: int | None,
  ):
    proj_cls = getattr(self.sl.attention, input_projection_name)
    if input_projection_name == 'CombinedQueryKeyValueProjection':
      input_projection = proj_cls(share_kv_projection=share_kv_projection)
    else:
      input_projection = proj_cls()

    batch_size, units_per_head = 2, 5
    max_past_horizon = 7
    max_future_horizon = 11

    l = self.sl.DotProductSelfAttention.Config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        units_per_head=units_per_head,
        input_projection=input_projection,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        name='dot_product_self_attention',
    ).make()

    x = self.random_sequence(batch_size, 16, 2)
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_self_attention')
    self.assertEqual(
        l.get_output_shape((2,)), (num_heads, units_per_head)
    )
    self.assertTrue(l.supports_step)
    self.assertEqual(l.input_latency, max(0, max_future_horizon))

    self.verify_contract(
        l,
        x,
        training=False,
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )




class DotProductAttentionTest(test_utils.SequenceLayerTest):
  """Test behavior of DotProductAttention layer."""

  def _make_constants(self, batch, time, features, name='source'):
    """Helper to create random sequence for cross-attention constants."""
    source = self.random_sequence(batch, time, features)
    return {name: source}

  def test_layer(self):
    layer = self.sl.DotProductAttention.Config(
        source_name='source',
        num_heads=2,
        units_per_head=4,
        name='dot_product_attention',
    ).make()
    constants = self._make_constants(2, 6, 12)
    x = self.random_sequence(2, 5, 8)
    layer = self.init_layer(layer, x, constants=constants)
    self.verify_contract(
        layer,
        x,
        constants=constants,
        atol=1e-4,
        rtol=1e-4,
    )

  def test_output_shape(self):
    layer = self.sl.DotProductAttention.Config(
        source_name='enc',
        num_heads=4,
        units_per_head=8,
    ).make()
    x = self.random_sequence(1, 5, 16)
    constants = self._make_constants(1, 5, 16, name='enc')
    layer = self.init_layer(layer, x, constants=constants)
    self.assertEqual(layer.get_output_shape((16,)), (4, 8))

  def test_step_reuses_precomputed_kv(self):
    layer = self.sl.DotProductAttention.Config(
        source_name='source',
        num_heads=2,
        units_per_head=4,
    ).make()
    constants = self._make_constants(1, 6, 12)
    x = self.random_sequence(1, 1, 8)
    layer = self.init_layer(layer, x, constants=constants)

    input_spec = self.sl.types.ShapeDType((8,), x.dtype)
    state = layer.get_initial_state(
        1, input_spec, training=False, constants=constants
    )
    # KV should be pre-computed.
    keys_v = state[0]
    self.assertEqual(keys_v.shape, (1, 6, 2, 4))

    for _ in range(3):
      x_step = self.random_sequence(1, 1, 8)
      y, state = layer.step(x_step, state, training=False, constants=constants)
      self.assertEqual(y.channel_shape, (2, 4))

  def test_missing_source_raises(self):
    layer = self.sl.DotProductAttention.Config(
        source_name='missing',
        num_heads=2,
        units_per_head=4,
    ).make()
    x = self.random_sequence(1, 3, 8)
    layer = self.init_layer(layer, x, bind_only=True)
    with self.assertRaises(ValueError):
      layer.layer(x, constants={}, training=False)

  def test_logits_soft_cap(self):
    num_heads, units_per_head = 3, 5
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = self.sl.DotProductAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        attention_logits_soft_cap=50.0,
        name='dot_product_attention',
    ).make()

    source = self.random_sequence(batch_size, source_time, source_channels)
    constants = {source_name: source}
    time, channels = 21, 3
    x = self.random_sequence(batch_size, time, channels)
    l = self.init_layer(l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dot_product_attention')

    self.assertEqual(
        l.get_output_shape((channels,)),
        (num_heads, units_per_head),
    )
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        grad_atol=1e-5,
        grad_rtol=1e-5,
    )
