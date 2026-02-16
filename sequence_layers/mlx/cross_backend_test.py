"""Cross-backend numerical tests: JAX (Linen) vs MLX.

Verifies that both backends produce numerically identical outputs for all
ported layer types when initialised from the same random Linen parameters.
"""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import sequence_layers.jax as sl
from sequence_layers.jax import types as jax_types
from sequence_layers.jax.attention import common as attn_common
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import export
from sequence_layers.mlx import weight_converter

Sequence = bt.Sequence
ShapeDType = bt.ShapeDType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_stateless_float(
    test_case,
    config,
    input_shape,
    *,
    batch_size=2,
    time=8,
    atol=1e-5,
    rtol=1e-5,
    seed=42,
):
  """Compare a stateless layer that requires no parameters (float inputs)."""
  rng = np.random.RandomState(seed)
  values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
  mask = np.ones((batch_size, time), dtype=bool)

  # JAX.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
  jax_out = np.array(jax_model.apply(variables, x_jax, training=False).values)

  # MLX.
  mlx_model = config.make(backend='mlx')
  x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
  mlx_out = np.array(mlx_model.layer(x_mx).values)

  np.testing.assert_allclose(
      mlx_out,
      jax_out,
      atol=atol,
      rtol=rtol,
      err_msg=f'{config.__class__.__qualname__}: outputs differ',
  )


def _compare_parametric_float(
    test_case,
    config,
    input_shape,
    *,
    batch_size=2,
    time=8,
    atol=1e-5,
    rtol=1e-5,
    seed=42,
):
  """Compare a parametric layer with float inputs (Conv, Dense, Norm, etc.)."""
  rng = np.random.RandomState(seed)
  values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
  mask = np.ones((batch_size, time), dtype=bool)

  # JAX: init + run.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
  jax_params = variables['params']
  jax_out = np.array(
      jax_model.apply({'params': jax_params}, x_jax, training=False).values
  )

  # MLX: create, load weights, run.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(
      mlx_model,
      jax_params,
      config,
      input_spec=ShapeDType(input_shape, mx.float32),
  )
  x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
  mlx_out = np.array(mlx_model.layer(x_mx).values)

  np.testing.assert_allclose(
      mlx_out,
      jax_out,
      atol=atol,
      rtol=rtol,
      err_msg=f'{config.__class__.__qualname__}: outputs differ',
  )


def _compare_parametric_int(
    test_case,
    config,
    *,
    batch_size=2,
    time=8,
    atol=1e-5,
    rtol=1e-5,
    seed=42,
):
  """Compare a parametric layer with integer token inputs (Embedding)."""
  rng = np.random.RandomState(seed)
  # Infer vocab size from config.
  vocab = getattr(config, 'num_embeddings', 32)
  tokens = rng.randint(0, vocab, size=(batch_size, time)).astype(np.int32)
  mask = np.ones((batch_size, time), dtype=bool)

  # JAX.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(tokens), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
  jax_params = variables['params']
  jax_out = np.array(
      jax_model.apply({'params': jax_params}, x_jax, training=False).values
  )

  # MLX.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(mlx_model, jax_params, config)
  x_mx = Sequence(
      mx.array(tokens, dtype=mx.int32),
      mx.array(mask, dtype=mx.bool_),
  )
  mlx_out = np.array(mlx_model.layer(x_mx).values)

  np.testing.assert_allclose(
      mlx_out,
      jax_out,
      atol=atol,
      rtol=rtol,
      err_msg=f'{config.__class__.__qualname__}: outputs differ',
  )


def _compare_with_constants(
    test_case,
    config,
    input_shape,
    constants_fn,
    *,
    batch_size=2,
    time=8,
    atol=1e-4,
    rtol=1e-4,
    seed=42,
):
  """Compare a parametric layer that needs constants (cross-attention)."""
  rng = np.random.RandomState(seed)
  values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
  mask = np.ones((batch_size, time), dtype=bool)

  jax_constants, mlx_constants = constants_fn(batch_size, time, rng)

  # JAX.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(
      jax.random.PRNGKey(0),
      x_jax,
      training=False,
      constants=jax_constants,
  )
  jax_params = variables['params']
  jax_out = np.array(
      jax_model.apply(
          {'params': jax_params},
          x_jax,
          training=False,
          constants=jax_constants,
      ).values
  )

  # MLX.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(
      mlx_model,
      jax_params,
      config,
      input_spec=ShapeDType(input_shape, mx.float32),
      constants=mlx_constants,
  )
  x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
  mlx_out = np.array(mlx_model.layer(x_mx, constants=mlx_constants).values)

  np.testing.assert_allclose(
      mlx_out,
      jax_out,
      atol=atol,
      rtol=rtol,
      err_msg=f'{config.__class__.__qualname__}: outputs differ',
  )


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class StatelessActivationsTest(parameterized.TestCase):
  """Stateless activations: JAX vs MLX."""

  @parameterized.named_parameters(
      ('relu', sl.Relu.Config()),
      ('gelu', sl.Gelu.Config(approximate=False)),
      ('swish', sl.Swish.Config()),
      ('tanh', sl.Tanh.Config()),
      ('sigmoid', sl.Sigmoid.Config()),
      ('leaky_relu', sl.LeakyRelu.Config()),
      ('elu', sl.Elu.Config()),
      ('softmax', sl.Softmax.Config()),
      ('softplus', sl.Softplus.Config()),
  )
  def test_activation(self, config):
    _compare_stateless_float(self, config, (16,))


class StatelessShapeOpsTest(parameterized.TestCase):
  """Stateless shape operations: JAX vs MLX."""

  @parameterized.named_parameters(
      ('flatten_2d', sl.Flatten.Config(), (4, 3)),
      ('reshape', sl.Reshape.Config(output_shape=(2, 4)), (8,)),
      ('expand_dims', sl.ExpandDims.Config(axis=-1), (8,)),
      ('squeeze', sl.Squeeze.Config(), (8, 1)),
      ('transpose', sl.Transpose.Config(), (4, 3)),
  )
  def test_shape_op(self, config, input_shape):
    _compare_stateless_float(self, config, input_shape)


class StatelessMiscTest(parameterized.TestCase):
  """Stateless misc layers: JAX vs MLX."""

  @parameterized.named_parameters(
      ('scale', sl.Scale.Config(scale=0.5), (8,)),
      ('add', sl.Add.Config(shift=1.0), (8,)),
      ('gated_linear_unit', sl.GatedLinearUnit.Config(), (16,)),
      ('gated_tanh_unit', sl.GatedTanhUnit.Config(), (16,)),
  )
  def test_misc(self, config, input_shape):
    _compare_stateless_float(self, config, input_shape)

  def test_one_hot(self):
    config = sl.OneHot.Config(depth=8)
    rng = np.random.RandomState(42)
    tokens = rng.randint(0, 8, size=(2, 8)).astype(np.int32)
    mask = np.ones((2, 8), dtype=bool)

    # JAX.
    jax_model = config.make()
    x_jax = jax_types.Sequence(
        jnp.array(tokens), jnp.array(mask, dtype=jnp.bool_)
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
    jax_out = np.array(jax_model.apply(variables, x_jax, training=False).values)

    # MLX.
    mlx_model = config.make(backend='mlx')
    x_mx = Sequence(
        mx.array(tokens, dtype=mx.int32),
        mx.array(mask, dtype=mx.bool_),
    )
    mlx_out = np.array(mlx_model.layer(x_mx).values)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-5,
        rtol=1e-5,
        err_msg='OneHot outputs differ',
    )


class SamplingTest(parameterized.TestCase):
  """Downsample1D / Upsample1D: JAX vs MLX."""

  @parameterized.named_parameters(
      ('downsample_2', sl.Downsample1D.Config(rate=2), (8,)),
      ('downsample_3', sl.Downsample1D.Config(rate=3), (8,)),
      ('upsample_2', sl.Upsample1D.Config(rate=2), (8,)),
      ('upsample_3', sl.Upsample1D.Config(rate=3), (8,)),
      ('downsample_4', sl.Downsample1D.Config(rate=4), (16,)),
  )
  def test_sampling(self, config, input_shape):
    _compare_stateless_float(self, config, input_shape, time=12)


class PoolingCrossBackendTest(parameterized.TestCase):
  """Pooling layers: JAX vs MLX."""

  @parameterized.named_parameters(
      (
          'max_pool_2_valid',
          sl.MaxPooling1D.Config(pool_size=2, padding='valid'),
          (8,),
      ),
      (
          'max_pool_3_causal',
          sl.MaxPooling1D.Config(pool_size=3, padding='causal'),
          (8,),
      ),
      (
          'min_pool_2_valid',
          sl.MinPooling1D.Config(pool_size=2, padding='valid'),
          (8,),
      ),
      (
          'min_pool_3_causal',
          sl.MinPooling1D.Config(pool_size=3, padding='causal'),
          (8,),
      ),
      (
          'avg_pool_2_valid',
          sl.AveragePooling1D.Config(pool_size=2, padding='valid'),
          (8,),
      ),
      (
          'avg_pool_3_causal',
          sl.AveragePooling1D.Config(pool_size=3, padding='causal'),
          (8,),
      ),
      (
          'max_pool_stride2',
          sl.MaxPooling1D.Config(pool_size=2, strides=2, padding='valid'),
          (8,),
      ),
      (
          'avg_pool_masked',
          sl.AveragePooling1D.Config(
              pool_size=2, padding='valid', masked_average=True
          ),
          (8,),
      ),
  )
  def test_pooling(self, config, input_shape):
    _compare_stateless_float(self, config, input_shape)


class EmbeddingCrossBackendTest(parameterized.TestCase):
  """Embedding: JAX vs MLX."""

  def test_embedding(self):
    config = sl.Embedding.Config(num_embeddings=32, dimension=16)
    _compare_parametric_int(self, config)


class DenseCrossBackendTest(parameterized.TestCase):
  """Dense: JAX vs MLX."""

  def test_dense_plain(self):
    config = sl.Dense.Config(features=16)
    _compare_parametric_float(self, config, (8,))

  def test_dense_with_bias(self):
    config = sl.Dense.Config(features=16, use_bias=True)
    _compare_parametric_float(self, config, (8,))

  def test_dense_with_activation(self):
    config = sl.Dense.Config(features=16, activation=jax.nn.relu)
    _compare_parametric_float(self, config, (8,))


class ConvolutionCrossBackendTest(parameterized.TestCase):
  """Convolution: JAX vs MLX."""

  def test_conv1d_causal(self):
    config = sl.Conv1D.Config(filters=8, kernel_size=3, padding='causal')
    _compare_parametric_float(self, config, (4,))

  def test_conv1d_causal_valid(self):
    config = sl.Conv1D.Config(filters=8, kernel_size=3, padding='causal_valid')
    _compare_parametric_float(self, config, (4,))

  def test_depthwise_conv1d(self):
    config = sl.DepthwiseConv1D.Config(kernel_size=3, padding='causal')
    _compare_parametric_float(self, config, (4,))

  def test_conv1d_transpose(self):
    config = sl.Conv1DTranspose.Config(
        filters=8, kernel_size=3, strides=2, padding='causal'
    )
    _compare_parametric_float(self, config, (4,))

  def test_conv1d_with_bias(self):
    config = sl.Conv1D.Config(
        filters=8, kernel_size=3, padding='causal', use_bias=True
    )
    _compare_parametric_float(self, config, (4,))


class NormalizationCrossBackendTest(parameterized.TestCase):
  """Normalization: JAX vs MLX."""

  def test_rms_norm(self):
    config = sl.RMSNormalization.Config()
    _compare_parametric_float(self, config, (16,))

  def test_layer_norm(self):
    config = sl.LayerNormalization.Config()
    _compare_parametric_float(self, config, (16,))

  def test_l2_normalize(self):
    config = sl.L2Normalize.Config()
    _compare_stateless_float(self, config, (16,))

  def test_l2_normalize_multi_axis(self):
    config = sl.L2Normalize.Config(axis=(-2, -1))
    _compare_stateless_float(self, config, (4, 3))

  def test_batch_norm(self):
    config = sl.BatchNormalization.Config()
    rng = np.random.RandomState(42)
    batch_size, time = 2, 8
    input_shape = (16,)
    values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
    mask = np.ones((batch_size, time), dtype=bool)

    # JAX: init returns both 'params' and 'batch_stats'.
    jax_model = config.make()
    x_jax = jax_types.Sequence(
        jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
    jax_params = variables['params']
    jax_batch_stats = variables['batch_stats']
    jax_out = np.array(jax_model.apply(variables, x_jax, training=False).values)

    # MLX: load params + batch_stats.
    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType(input_shape, mx.float32),
        batch_stats=jax_batch_stats,
    )
    x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
    mlx_out = np.array(mlx_model.layer(x_mx).values)

    np.testing.assert_allclose(mlx_out, jax_out, atol=1e-5, rtol=1e-5)

  def test_batch_norm_no_affine(self):
    config = sl.BatchNormalization.Config(use_scale=False, use_bias=False)
    rng = np.random.RandomState(42)
    batch_size, time = 2, 8
    input_shape = (16,)
    values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
    mask = np.ones((batch_size, time), dtype=bool)

    jax_model = config.make()
    x_jax = jax_types.Sequence(
        jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
    jax_batch_stats = variables.get('batch_stats', {})
    # No params when scale/bias disabled — only batch_stats.
    jax_params = variables.get('params', {})
    jax_out = np.array(jax_model.apply(variables, x_jax, training=False).values)

    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType(input_shape, mx.float32),
        batch_stats=jax_batch_stats,
    )
    x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
    mlx_out = np.array(mlx_model.layer(x_mx).values)

    np.testing.assert_allclose(mlx_out, jax_out, atol=1e-5, rtol=1e-5)

  # GroupNorm: JAX layer() reduces over time (non-cumulative), MLX normalizes
  # per-timestep by design. Cross-backend comparison requires cumulative mode
  # which differs semantically. Skipped.


class SelfAttentionCrossBackendTest(parameterized.TestCase):
  """Self-attention: JAX vs MLX."""

  def test_basic(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=8,
        max_past_horizon=16,
        max_future_horizon=0,
    )
    _compare_parametric_float(self, config, (16,), atol=1e-4, rtol=1e-4)

  def test_with_rope(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=8,
        max_past_horizon=16,
        max_future_horizon=0,
        query_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
        key_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
    )
    _compare_parametric_float(self, config, (16,), atol=1e-4, rtol=1e-4)


class LocalSelfAttentionCrossBackendTest(parameterized.TestCase):
  """Local self-attention: JAX vs MLX."""

  def test_basic(self):
    from sequence_layers.jax.attention import (
        local_dot_product_self_attention as jax_local_attn,
    )

    config = jax_local_attn.LocalDotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        block_size=1,
        max_past_horizon=8,
        max_future_horizon=0,
    )
    _compare_parametric_float(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_with_soft_cap(self):
    from sequence_layers.jax.attention import (
        local_dot_product_self_attention as jax_local_attn,
    )

    config = jax_local_attn.LocalDotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        block_size=1,
        max_past_horizon=8,
        max_future_horizon=0,
        attention_logits_soft_cap=50.0,
    )
    _compare_parametric_float(self, config, (8,), atol=1e-4, rtol=1e-4)


class StepModeLocalSelfAttentionTest(parameterized.TestCase):
  """Step-mode cross-backend: local self-attention."""

  def test_causal(self):
    from sequence_layers.jax.attention import (
        local_dot_product_self_attention as jax_local_attn,
    )

    config = jax_local_attn.LocalDotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        block_size=1,
        max_past_horizon=8,
        max_future_horizon=0,
    )
    _compare_step_mode(self, config, (8,), atol=1e-4, rtol=1e-4)


class DSPCrossBackendTest(parameterized.TestCase):
  """DSP layers: JAX vs MLX."""

  def test_delay(self):
    config = sl.Delay.Config(length=2)
    _compare_stateless_float(self, config, (8,))

  def test_lookahead(self):
    config = sl.Lookahead.Config(length=3)
    _compare_stateless_float(self, config, (8,))

  def test_window(self):
    config = sl.Window.Config(axis=-1)
    _compare_stateless_float(self, config, (8,))

  def test_frame(self):
    config = sl.Frame.Config(frame_length=4, frame_step=2)
    _compare_stateless_float(self, config, (1,), time=8)

  def test_frame_causal(self):
    config = sl.Frame.Config(frame_length=4, frame_step=2, padding='causal')
    _compare_stateless_float(self, config, (1,), time=8)

  def test_overlap_add_causal(self):
    config = sl.OverlapAdd.Config(
        frame_length=4, frame_step=2, padding='causal'
    )
    _compare_stateless_float(self, config, (4,), time=8)

  def test_fft(self):
    config = sl.FFT.Config()
    _compare_stateless_float(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_ifft(self):
    config = sl.IFFT.Config()
    _compare_stateless_float(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_rfft(self):
    config = sl.RFFT.Config()
    _compare_stateless_float(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_rfft_irfft_roundtrip(self):
    # IRFFT needs complex input; test via RFFT→IRFFT roundtrip.
    config = sl.Serial.Config([
        sl.RFFT.Config(),
        sl.IRFFT.Config(),
    ])
    _compare_stateless_float(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_stft(self):
    config = sl.STFT.Config(
        frame_length=8,
        frame_step=4,
        fft_length=8,
        output_magnitude=True,
    )
    _compare_stateless_float(self, config, (1,), time=16, atol=1e-4, rtol=1e-4)

  def test_stft_complex(self):
    config = sl.STFT.Config(
        frame_length=8,
        frame_step=4,
        fft_length=8,
        output_magnitude=False,
    )
    _compare_stateless_float(self, config, (1,), time=16, atol=1e-4, rtol=1e-4)

  def test_stft_inverse_stft_roundtrip(self):
    # InverseSTFT needs complex input; test via STFT→InverseSTFT roundtrip.
    config = sl.Serial.Config([
        sl.STFT.Config(
            frame_length=8,
            frame_step=4,
            fft_length=8,
            output_magnitude=False,
        ),
        sl.InverseSTFT.Config(
            frame_length=8,
            frame_step=4,
            fft_length=8,
            time_padding='causal',
        ),
    ])
    _compare_stateless_float(self, config, (1,), time=16, atol=1e-4, rtol=1e-4)

  def test_mel_spectrogram(self):
    config = sl.LinearToMelSpectrogram.Config(
        num_mel_bins=10,
        sample_rate=16000.0,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    # Mel filterbank computation may differ slightly between backends
    # due to different float64 vs float32 precision paths.
    _compare_stateless_float(self, config, (5,), atol=0.05, rtol=0.1)


class CombinatorsCrossBackendTest(parameterized.TestCase):
  """Combinators: JAX vs MLX."""

  def test_serial(self):
    config = sl.Serial.Config([
        sl.Dense.Config(features=16),
        sl.Relu.Config(),
        sl.Dense.Config(features=8),
    ])
    _compare_parametric_float(self, config, (8,))

  def test_residual(self):
    config = sl.Residual.Config([
        sl.Dense.Config(features=8),
        sl.Relu.Config(),
    ])
    _compare_parametric_float(self, config, (8,))

  def test_repeat(self):
    config = sl.Repeat.Config(
        num_repeats=2,
        layer=sl.Serial.Config([
            sl.Dense.Config(features=8),
            sl.Relu.Config(),
        ]),
    )
    _compare_parametric_float(self, config, (8,), atol=1e-4, rtol=1e-4)


class CrossAttentionCrossBackendTest(parameterized.TestCase):
  """Cross-attention (DotProductAttention): JAX vs MLX."""

  def _make_constants(self, batch_size, time, source_features, rng):
    source_values = rng.randn(batch_size, time, source_features).astype(
        np.float32
    )
    source_mask = np.ones((batch_size, time), dtype=bool)
    jax_source = jax_types.Sequence(
        jnp.array(source_values), jnp.array(source_mask, dtype=jnp.bool_)
    )
    mlx_source = Sequence(
        mx.array(source_values), mx.array(source_mask, dtype=mx.bool_)
    )
    return {'enc': jax_source}, {'enc': mlx_source}

  def test_basic(self):
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    config = jax_cross_attn.DotProductAttention.Config(
        source_name='enc',
        num_heads=2,
        units_per_head=4,
    )
    _compare_with_constants(
        self,
        config,
        (8,),
        lambda b, t, rng: self._make_constants(b, t, 12, rng),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_same_features(self):
    """Source and input have the same feature dimension."""
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    config = jax_cross_attn.DotProductAttention.Config(
        source_name='enc',
        num_heads=4,
        units_per_head=4,
    )
    _compare_with_constants(
        self,
        config,
        (16,),
        lambda b, t, rng: self._make_constants(b, t, 16, rng),
        atol=1e-4,
        rtol=1e-4,
    )


class StreamingAttentionCrossBackendTest(parameterized.TestCase):
  """Streaming cross-attention: JAX vs MLX weight conversion."""

  def _make_constants(self, batch_size, time, source_features, rng):
    source_values = rng.randn(batch_size, time, source_features).astype(
        np.float32
    )
    source_mask = np.ones((batch_size, time), dtype=bool)
    jax_source = jax_types.Sequence(
        jnp.array(source_values), jnp.array(source_mask, dtype=jnp.bool_)
    )
    mlx_source = Sequence(
        mx.array(source_values), mx.array(source_mask, dtype=mx.bool_)
    )
    return {'src': jax_source}, {'src': mlx_source}

  def test_basic(self):
    from sequence_layers.jax.attention import (
        streaming_dot_product_attention as jax_streaming_attn,
    )

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='src',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
    )
    _compare_with_constants(
        self,
        config,
        (8,),
        lambda b, t, rng: self._make_constants(b, t, 12, rng),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_with_future_horizon(self):
    from sequence_layers.jax.attention import (
        streaming_dot_product_attention as jax_streaming_attn,
    )

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='src',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=4,
        max_future_horizon=2,
    )
    _compare_with_constants(
        self,
        config,
        (8,),
        lambda b, t, rng: self._make_constants(b, t, 8, rng),
        atol=1e-4,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Step-mode cross-backend tests
# ---------------------------------------------------------------------------


def _compare_step_mode(
    test_case,
    config,
    input_shape,
    *,
    batch_size=1,
    num_steps=6,
    block_size=1,
    atol=1e-5,
    rtol=1e-5,
    seed=42,
    constants_fn=None,
    stream_constants_fn=None,
):
  """Compare JAX and MLX step-by-step outputs with shared weights.

  Args:
    test_case: A TestCase instance.
    config: A SequenceLayerConfig.
    input_shape: Channel shape, e.g. (8,).
    batch_size: Batch dimension.
    num_steps: Number of step invocations.
    block_size: Number of timesteps per step. Must match the layer's
        block_size for layers that require it (e.g. Frame, OverlapAdd).
    atol: Absolute tolerance.
    rtol: Relative tolerance.
    seed: Random seed.
    constants_fn: For static cross-attention. Returns (jax_constants,
        mlx_constants) given (batch_size, rng).
    stream_constants_fn: For streaming cross-attention. Returns
        (jax_constants, mlx_constants) given (batch_size, time, rng).
        Each has shape [batch, time, features]. Will be sliced per step.
  """
  rng = np.random.RandomState(seed)
  step_values = [
      rng.randn(batch_size, block_size, *input_shape).astype(np.float32)
      for _ in range(num_steps)
  ]
  step_masks = [
      np.ones((batch_size, block_size), dtype=bool) for _ in range(num_steps)
  ]
  total_time = num_steps * block_size

  jax_constants = None
  mlx_constants = None
  jax_stream_constants = None
  mlx_stream_constants = None

  if constants_fn is not None:
    jax_constants, mlx_constants = constants_fn(batch_size, rng)

  if stream_constants_fn is not None:
    jax_stream_constants, mlx_stream_constants = stream_constants_fn(
        batch_size, total_time, rng
    )

  # --- JAX init + step ---
  jax_model = config.make()
  # Init with a full sequence to get params.
  full_values = np.concatenate(step_values, axis=1)
  full_mask = np.ones((batch_size, total_time), dtype=bool)
  x_init = jax_types.Sequence(
      jnp.array(full_values), jnp.array(full_mask, dtype=jnp.bool_)
  )
  init_constants = jax_constants
  if init_constants is None and jax_stream_constants is not None:
    init_constants = jax_stream_constants
  variables = jax_model.init(
      jax.random.PRNGKey(0),
      x_init,
      training=False,
      constants=init_constants,
  )
  jax_params = variables.get('params', {})
  jax_variables = {'params': jax_params} if jax_params else variables

  jax_spec = jax.ShapeDtypeStruct(input_shape, jnp.float32)
  jax_state = jax_model.apply(
      jax_variables,
      batch_size,
      jax_spec,
      training=False,
      constants=init_constants,
      method=jax_model.get_initial_state,
  )

  jax_outputs = []
  for i in range(num_steps):
    x_jax = jax_types.Sequence(
        jnp.array(step_values[i]),
        jnp.array(step_masks[i], dtype=jnp.bool_),
    )
    step_c = jax_constants
    if jax_stream_constants is not None:
      s = i * block_size
      e = s + block_size
      step_c = {
          k: jax_types.Sequence(v.values[:, s:e], v.mask[:, s:e])
          for k, v in jax_stream_constants.items()
      }
    y_jax, jax_state = jax_model.apply(
        jax_variables,
        x_jax,
        jax_state,
        training=False,
        constants=step_c,
        method=jax_model.step,
    )
    jax_outputs.append(np.array(y_jax.values))

  # --- MLX init + step ---
  mlx_model = config.make(backend='mlx')
  mlx_init_constants = mlx_constants
  if mlx_init_constants is None and mlx_stream_constants is not None:
    mlx_init_constants = mlx_stream_constants
  if jax_params:
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType(input_shape, mx.float32),
        constants=mlx_init_constants,
    )
  # Skip _materialize_deferred for param-less layers — no deferred weights.

  mlx_spec = ShapeDType(input_shape, mx.float32)
  # Slice stream constants to time=1 for get_initial_state.
  state_constants = mlx_constants
  if mlx_stream_constants is not None:
    state_constants = {
        k: Sequence(v.values[:, :1], v.mask[:, :1])
        for k, v in mlx_stream_constants.items()
    }
  mlx_state = mlx_model.get_initial_state(
      batch_size, mlx_spec, constants=state_constants
  )

  mlx_outputs = []
  for i in range(num_steps):
    x_mx = Sequence(
        mx.array(step_values[i]),
        mx.array(step_masks[i], dtype=mx.bool_),
    )
    step_c = mlx_constants
    if mlx_stream_constants is not None:
      s = i * block_size
      e = s + block_size
      step_c = {
          k: Sequence(v.values[:, s:e], v.mask[:, s:e])
          for k, v in mlx_stream_constants.items()
      }
    y_mx, mlx_state = mlx_model.step(x_mx, mlx_state, constants=step_c)
    mx.eval(y_mx.values)
    mlx_outputs.append(np.array(y_mx.values))

  # --- Compare ---
  for i, (jax_out, mlx_out) in enumerate(zip(jax_outputs, mlx_outputs)):
    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=atol,
        rtol=rtol,
        err_msg=f'{config.__class__.__qualname__} step {i}: outputs differ',
    )


class StepModeConvolutionTest(parameterized.TestCase):
  """Step-mode cross-backend: convolution layers."""

  def test_conv1d_causal(self):
    config = sl.Conv1D.Config(filters=8, kernel_size=3, padding='causal')
    _compare_step_mode(self, config, (4,))

  def test_depthwise_conv1d_causal(self):
    config = sl.DepthwiseConv1D.Config(kernel_size=3, padding='causal')
    _compare_step_mode(self, config, (4,))

  def test_conv1d_transpose_causal(self):
    config = sl.Conv1DTranspose.Config(
        filters=8, kernel_size=3, strides=2, padding='causal'
    )
    _compare_step_mode(self, config, (4,))


class StepModeDenseNormTest(parameterized.TestCase):
  """Step-mode cross-backend: Dense and normalization."""

  def test_dense(self):
    config = sl.Dense.Config(features=16)
    _compare_step_mode(self, config, (8,))

  def test_rms_norm(self):
    config = sl.RMSNormalization.Config()
    _compare_step_mode(self, config, (16,))

  def test_layer_norm(self):
    config = sl.LayerNormalization.Config()
    _compare_step_mode(self, config, (16,))


class StepModeSelfAttentionTest(parameterized.TestCase):
  """Step-mode cross-backend: self-attention."""

  def test_causal(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=16,
        max_future_horizon=0,
    )
    _compare_step_mode(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_causal_with_bias(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=16,
        max_future_horizon=0,
        use_bias=True,
    )
    _compare_step_mode(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_causal_with_rope(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=16,
        max_future_horizon=0,
        query_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
        key_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
    )
    _compare_step_mode(self, config, (8,), atol=1e-4, rtol=1e-4)

  def test_gqa_with_rope(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=4,
        units_per_head=4,
        max_past_horizon=16,
        max_future_horizon=0,
        num_kv_heads=2,
        input_projection=attn_common.SeparateQueryKeyValueProjection(),
        query_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
        key_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
    )
    _compare_step_mode(self, config, (16,), atol=1e-4, rtol=1e-4)


class StepModeCrossAttentionTest(parameterized.TestCase):
  """Step-mode cross-backend: cross-attention."""

  def _make_constants(self, batch_size, rng, source_features=12, source_time=8):
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    source_values = rng.randn(batch_size, source_time, source_features).astype(
        np.float32
    )
    source_mask = np.ones((batch_size, source_time), dtype=bool)
    jax_source = jax_types.Sequence(
        jnp.array(source_values), jnp.array(source_mask, dtype=jnp.bool_)
    )
    mlx_source = Sequence(
        mx.array(source_values), mx.array(source_mask, dtype=mx.bool_)
    )
    return {'enc': jax_source}, {'enc': mlx_source}

  def test_cross_attention(self):
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    config = jax_cross_attn.DotProductAttention.Config(
        source_name='enc',
        num_heads=2,
        units_per_head=4,
    )
    _compare_step_mode(
        self,
        config,
        (8,),
        constants_fn=lambda b, rng: self._make_constants(b, rng),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_cross_attention_different_dims(self):
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    config = jax_cross_attn.DotProductAttention.Config(
        source_name='enc',
        num_heads=2,
        units_per_head=4,
    )
    _compare_step_mode(
        self,
        config,
        (16,),
        constants_fn=lambda b, rng: self._make_constants(
            b, rng, source_features=8, source_time=12
        ),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_cross_attention_with_bias(self):
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    config = jax_cross_attn.DotProductAttention.Config(
        source_name='enc',
        num_heads=2,
        units_per_head=4,
        use_bias=True,
    )
    _compare_step_mode(
        self,
        config,
        (8,),
        constants_fn=lambda b, rng: self._make_constants(b, rng),
        atol=1e-4,
        rtol=1e-4,
    )


class StepModeStreamingAttentionTest(parameterized.TestCase):
  """Step-mode cross-backend: streaming cross-attention."""

  def _make_stream_constants(self, batch_size, time, rng, source_features=12):
    source_values = rng.randn(batch_size, time, source_features).astype(
        np.float32
    )
    source_mask = np.ones((batch_size, time), dtype=bool)
    jax_source = jax_types.Sequence(
        jnp.array(source_values), jnp.array(source_mask, dtype=jnp.bool_)
    )
    mlx_source = Sequence(
        mx.array(source_values), mx.array(source_mask, dtype=mx.bool_)
    )
    return {'src': jax_source}, {'src': mlx_source}

  def test_streaming_attention(self):
    from sequence_layers.jax.attention import (
        streaming_dot_product_attention as jax_streaming_attn,
    )

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='src',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
    )
    _compare_step_mode(
        self,
        config,
        (8,),
        stream_constants_fn=lambda b, t, rng: self._make_stream_constants(
            b, t, rng, source_features=12
        ),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_streaming_with_future_horizon(self):
    from sequence_layers.jax.attention import (
        streaming_dot_product_attention as jax_streaming_attn,
    )

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='src',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=6,
        max_future_horizon=2,
    )
    _compare_step_mode(
        self,
        config,
        (8,),
        stream_constants_fn=lambda b, t, rng: self._make_stream_constants(
            b, t, rng, source_features=12
        ),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_streaming_with_rope(self):
    from sequence_layers.jax.attention import (
        streaming_dot_product_attention as jax_streaming_attn,
    )

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='src',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
        query_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
        key_network=sl.ApplyRotaryPositionalEncoding.Config(
            max_wavelength=10_000.0,
        ),
    )
    _compare_step_mode(
        self,
        config,
        (8,),
        stream_constants_fn=lambda b, t, rng: self._make_stream_constants(
            b, t, rng, source_features=12
        ),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_streaming_local(self):
    from sequence_layers.jax.attention import (
        streaming_local_dot_product_attention as jax_streaming_local_attn,
    )

    config = jax_streaming_local_attn.StreamingLocalDotProductAttention.Config(
        source_name='src',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
        block_size=1,
    )
    _compare_step_mode(
        self,
        config,
        (8,),
        stream_constants_fn=lambda b, t, rng: self._make_stream_constants(
            b, t, rng, source_features=12
        ),
        atol=1e-4,
        rtol=1e-4,
    )

  def test_streaming_with_bias(self):
    from sequence_layers.jax.attention import (
        streaming_dot_product_attention as jax_streaming_attn,
    )

    config = jax_streaming_attn.StreamingDotProductAttention.Config(
        source_name='src',
        num_heads=2,
        units_per_head=4,
        max_past_horizon=8,
        use_bias=True,
    )
    _compare_step_mode(
        self,
        config,
        (8,),
        stream_constants_fn=lambda b, t, rng: self._make_stream_constants(
            b, t, rng, source_features=12
        ),
        atol=1e-4,
        rtol=1e-4,
    )


class StepModeDSPTest(parameterized.TestCase):
  """Step-mode cross-backend: DSP layers."""

  def test_delay(self):
    config = sl.Delay.Config(length=3)
    _compare_step_mode(self, config, (8,))

  def test_lookahead(self):
    config = sl.Lookahead.Config(length=3)
    _compare_step_mode(self, config, (8,))

  def test_window(self):
    config = sl.Window.Config(axis=-1)
    _compare_step_mode(self, config, (8,))

  def test_frame_causal(self):
    config = sl.Frame.Config(frame_length=4, frame_step=2, padding='causal')
    _compare_step_mode(self, config, (1,), block_size=2, num_steps=6)

  def test_overlap_add_causal(self):
    config = sl.OverlapAdd.Config(
        frame_length=4, frame_step=2, padding='causal'
    )
    _compare_step_mode(self, config, (4,), num_steps=6)

  def test_overlap_add_causal_large(self):
    config = sl.OverlapAdd.Config(
        frame_length=8, frame_step=4, padding='causal'
    )
    _compare_step_mode(self, config, (8,), num_steps=6)


class StepModeCombinatorTest(parameterized.TestCase):
  """Step-mode cross-backend: combinators."""

  def test_serial(self):
    config = sl.Serial.Config([
        sl.Dense.Config(features=16),
        sl.Relu.Config(),
        sl.Dense.Config(features=8),
    ])
    _compare_step_mode(self, config, (8,))

  def test_residual(self):
    config = sl.Residual.Config([
        sl.Dense.Config(features=8),
        sl.Relu.Config(),
    ])
    _compare_step_mode(self, config, (8,))

  def test_repeat_with_attention(self):
    config = sl.Repeat.Config(
        num_repeats=2,
        layer=sl.Serial.Config([
            sl.Residual.Config([
                sl.RMSNormalization.Config(),
                sl.DotProductSelfAttention.Config(
                    num_heads=2,
                    units_per_head=4,
                    max_past_horizon=16,
                    max_future_horizon=0,
                ),
                sl.Flatten.Config(),
            ]),
            sl.Residual.Config([
                sl.RMSNormalization.Config(),
                sl.Dense.Config(features=8),
            ]),
        ]),
    )
    _compare_step_mode(self, config, (8,), atol=1e-3, rtol=1e-3)


class StepModePoolingTest(parameterized.TestCase):
  """Step-mode cross-backend: pooling layers."""

  def test_max_pool_causal(self):
    config = sl.MaxPooling1D.Config(pool_size=3, padding='causal')
    _compare_step_mode(self, config, (8,))

  def test_min_pool_causal(self):
    config = sl.MinPooling1D.Config(pool_size=3, padding='causal')
    _compare_step_mode(self, config, (8,))

  def test_avg_pool_causal(self):
    config = sl.AveragePooling1D.Config(pool_size=3, padding='causal')
    _compare_step_mode(self, config, (8,))


class StepModeGQATest(parameterized.TestCase):
  """Step-mode cross-backend: grouped query attention."""

  def test_gqa(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=4,
        units_per_head=4,
        max_past_horizon=16,
        max_future_horizon=0,
        num_kv_heads=2,
        input_projection=attn_common.SeparateQueryKeyValueProjection(),
    )
    _compare_step_mode(self, config, (16,), atol=1e-4, rtol=1e-4)


class GQACrossBackendTest(parameterized.TestCase):
  """Layer-mode cross-backend: grouped query attention."""

  def test_gqa(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=4,
        units_per_head=4,
        max_past_horizon=16,
        max_future_horizon=0,
        num_kv_heads=2,
        input_projection=attn_common.SeparateQueryKeyValueProjection(),
    )
    _compare_parametric_float(self, config, (16,), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Parallel combinator cross-backend tests
# ---------------------------------------------------------------------------


class ParallelCrossBackendTest(parameterized.TestCase):
  """Cross-backend: Parallel combinator (layer + step)."""

  def test_parallel_add_layer(self):
    config = sl.Parallel.Config(
        layers=[
            sl.Dense.Config(features=8),
            sl.Dense.Config(features=8),
        ],
        combination=sl.CombinationMode.ADD,
    )
    _compare_parametric_float(self, config, (8,))

  def test_parallel_concat_layer(self):
    config = sl.Parallel.Config(
        layers=[
            sl.Dense.Config(features=4),
            sl.Dense.Config(features=4),
        ],
        combination=sl.CombinationMode.CONCAT,
    )
    _compare_parametric_float(self, config, (8,))

  def test_parallel_stack_layer(self):
    config = sl.Parallel.Config(
        layers=[
            sl.Dense.Config(features=8),
            sl.Dense.Config(features=8),
        ],
        combination=sl.CombinationMode.STACK,
    )
    _compare_parametric_float(self, config, (8,))

  def test_parallel_add_step(self):
    config = sl.Parallel.Config(
        layers=[
            sl.Dense.Config(features=8),
            sl.Dense.Config(features=8),
        ],
        combination=sl.CombinationMode.ADD,
    )
    _compare_step_mode(self, config, (8,))

  def test_parallel_concat_step(self):
    config = sl.Parallel.Config(
        layers=[
            sl.Dense.Config(features=4),
            sl.Dense.Config(features=4),
        ],
        combination=sl.CombinationMode.CONCAT,
    )
    _compare_step_mode(self, config, (8,))


# ---------------------------------------------------------------------------
# Partially-masked input tests
# ---------------------------------------------------------------------------


def _compare_parametric_float_masked(
    test_case,
    config,
    input_shape,
    *,
    batch_size=2,
    time=8,
    atol=1e-5,
    rtol=1e-5,
    seed=42,
):
  """Like _compare_parametric_float but with partially-masked inputs."""
  rng = np.random.RandomState(seed)
  values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
  # Create a mask where ~25% of timesteps are invalid.
  mask = rng.rand(batch_size, time) > 0.25

  # JAX.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
  jax_params = variables.get('params', {})
  jax_variables = {'params': jax_params} if jax_params else variables
  jax_out = jax_model.apply(jax_variables, x_jax, training=False)
  jax_values = np.array(jax_out.values)
  jax_mask = np.array(jax_out.mask)

  # MLX.
  mlx_model = config.make(backend='mlx')
  if jax_params:
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType(input_shape, mx.float32),
    )
  else:
    export._materialize_deferred(
        mlx_model,
        batch_size=1,
        input_spec=ShapeDType(input_shape, mx.float32),
    )
  x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
  mlx_out = mlx_model.layer(x_mx)
  mlx_values = np.array(mlx_out.values)
  mlx_mask = np.array(mlx_out.mask)

  # Compare valid timesteps only.
  out_mask = jax_mask
  if jax_values.shape != mlx_values.shape:
    test_case.fail(
        f'{config.__class__.__qualname__}: shape mismatch'
        f' jax={jax_values.shape} vs mlx={mlx_values.shape}'
    )

  # Flatten and compare only valid positions.
  for b in range(batch_size):
    for t in range(out_mask.shape[1]):
      if out_mask[b, t]:
        np.testing.assert_allclose(
            mlx_values[b, t],
            jax_values[b, t],
            atol=atol,
            rtol=rtol,
            err_msg=(
                f'{config.__class__.__qualname__} batch={b} time={t}:'
                ' valid outputs differ'
            ),
        )

  # Masks should match.
  np.testing.assert_array_equal(
      mlx_mask,
      jax_mask,
      err_msg=f'{config.__class__.__qualname__}: masks differ',
  )


class MaskedInputDenseTest(parameterized.TestCase):
  """Cross-backend with partially-masked inputs: Dense."""

  def test_dense_masked(self):
    config = sl.Dense.Config(features=16)
    _compare_parametric_float_masked(self, config, (8,))


class MaskedInputConvTest(parameterized.TestCase):
  """Cross-backend with partially-masked inputs: Conv1D."""

  def test_conv1d_causal_masked(self):
    config = sl.Conv1D.Config(filters=8, kernel_size=3, padding='causal')
    _compare_parametric_float_masked(self, config, (4,))

  def test_depthwise_conv1d_masked(self):
    config = sl.DepthwiseConv1D.Config(kernel_size=3, padding='causal')
    _compare_parametric_float_masked(self, config, (4,))


class MaskedInputNormTest(parameterized.TestCase):
  """Cross-backend with partially-masked inputs: normalization."""

  def test_rms_norm_masked(self):
    config = sl.RMSNormalization.Config()
    _compare_parametric_float_masked(self, config, (16,))

  def test_layer_norm_masked(self):
    config = sl.LayerNormalization.Config()
    _compare_parametric_float_masked(self, config, (16,))


class MaskedInputSelfAttentionTest(parameterized.TestCase):
  """Cross-backend with partially-masked inputs: self-attention."""

  def test_causal_masked(self):
    config = sl.DotProductSelfAttention.Config(
        num_heads=2,
        units_per_head=4,
        max_past_horizon=16,
        max_future_horizon=0,
    )
    _compare_parametric_float_masked(self, config, (8,), atol=1e-4, rtol=1e-4)


class MaskedInputPoolingTest(parameterized.TestCase):
  """Cross-backend with partially-masked inputs: pooling."""

  def test_max_pool_masked(self):
    config = sl.MaxPooling1D.Config(pool_size=2, padding='causal')
    _compare_parametric_float_masked(self, config, (8,))

  def test_avg_pool_masked(self):
    config = sl.AveragePooling1D.Config(pool_size=2, padding='causal')
    _compare_parametric_float_masked(self, config, (8,))


# ---------------------------------------------------------------------------
# Integration tests: full model cross-backend comparison
# ---------------------------------------------------------------------------


def _compare_integration_float(
    test_case,
    config,
    input_shape,
    *,
    batch_size=2,
    time=8,
    atol=1e-3,
    rtol=1e-3,
    seed=42,
    constants_fn=None,
):
  """Compare a full model (layer mode) between JAX and MLX."""
  rng = np.random.RandomState(seed)
  values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
  mask = np.ones((batch_size, time), dtype=bool)

  jax_constants = None
  mlx_constants = None
  if constants_fn is not None:
    jax_constants, mlx_constants = constants_fn(batch_size, time, rng)

  # JAX.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(
      jax.random.PRNGKey(0), x_jax, training=False, constants=jax_constants
  )
  jax_params = variables['params']
  jax_out = jax_model.apply(
      {'params': jax_params},
      x_jax,
      training=False,
      constants=jax_constants,
  )
  jax_values = np.array(jax_out.values)

  # MLX.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(
      mlx_model,
      jax_params,
      config,
      input_spec=ShapeDType(input_shape, mx.float32),
      constants=mlx_constants,
  )
  x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
  mlx_out = mlx_model.layer(x_mx, constants=mlx_constants)
  mlx_values = np.array(mlx_out.values)

  test_case.assertEqual(
      jax_values.shape,
      mlx_values.shape,
      f'Shape mismatch: jax={jax_values.shape} vs mlx={mlx_values.shape}',
  )
  np.testing.assert_allclose(
      mlx_values,
      jax_values,
      atol=atol,
      rtol=rtol,
      err_msg='Integration test: JAX vs MLX outputs differ',
  )
  return jax_params, jax_constants, mlx_constants


def _compare_integration_int(
    test_case,
    config,
    *,
    vocab_size=256,
    batch_size=2,
    time=8,
    atol=1e-3,
    rtol=1e-3,
    seed=42,
):
  """Compare a full model with integer token inputs (layer mode)."""
  rng = np.random.RandomState(seed)
  tokens = rng.randint(0, vocab_size, size=(batch_size, time)).astype(np.int32)
  mask = np.ones((batch_size, time), dtype=bool)

  # JAX.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(tokens), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(jax.random.PRNGKey(0), x_jax, training=False)
  jax_params = variables['params']
  jax_out = np.array(
      jax_model.apply({'params': jax_params}, x_jax, training=False).values
  )

  # MLX.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(mlx_model, jax_params, config)
  x_mx = Sequence(
      mx.array(tokens, dtype=mx.int32), mx.array(mask, dtype=mx.bool_)
  )
  mlx_out = np.array(mlx_model.layer(x_mx).values)

  test_case.assertEqual(
      jax_out.shape,
      mlx_out.shape,
      f'Shape mismatch: jax={jax_out.shape} vs mlx={mlx_out.shape}',
  )
  np.testing.assert_allclose(
      mlx_out,
      jax_out,
      atol=atol,
      rtol=rtol,
      err_msg='Integration test: JAX vs MLX outputs differ',
  )
  return jax_params


def _compare_integration_step(
    test_case,
    config,
    input_shape,
    jax_params,
    *,
    batch_size=2,
    num_steps=8,
    atol=1e-3,
    rtol=1e-3,
    seed=42,
    jax_constants=None,
    mlx_constants=None,
):
  """Compare step-by-step output of a full model between JAX and MLX."""
  rng = np.random.RandomState(seed + 1)
  step_values = [
      rng.randn(batch_size, 1, *input_shape).astype(np.float32)
      for _ in range(num_steps)
  ]
  step_masks = [np.ones((batch_size, 1), dtype=bool) for _ in range(num_steps)]

  # JAX step.
  jax_model = config.make()
  jax_spec = jax.ShapeDtypeStruct(input_shape, jnp.float32)
  jax_state = jax_model.apply(
      {'params': jax_params},
      batch_size,
      jax_spec,
      training=False,
      constants=jax_constants,
      method=jax_model.get_initial_state,
  )
  jax_outputs = []
  for i in range(num_steps):
    x_jax = jax_types.Sequence(
        jnp.array(step_values[i]),
        jnp.array(step_masks[i], dtype=jnp.bool_),
    )
    y_jax, jax_state = jax_model.apply(
        {'params': jax_params},
        x_jax,
        jax_state,
        training=False,
        constants=jax_constants,
        method=jax_model.step,
    )
    jax_outputs.append(np.array(y_jax.values))

  # MLX step.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(
      mlx_model,
      jax_params,
      config,
      input_spec=ShapeDType(input_shape, mx.float32),
      constants=mlx_constants,
  )
  mlx_spec = ShapeDType(input_shape, mx.float32)
  mlx_state = mlx_model.get_initial_state(
      batch_size, mlx_spec, constants=mlx_constants
  )
  mlx_outputs = []
  for i in range(num_steps):
    x_mx = Sequence(
        mx.array(step_values[i]),
        mx.array(step_masks[i], dtype=mx.bool_),
    )
    y_mx, mlx_state = mlx_model.step(x_mx, mlx_state, constants=mlx_constants)
    mx.eval(y_mx.values)
    mlx_outputs.append(np.array(y_mx.values))

  for i, (jax_out, mlx_out) in enumerate(zip(jax_outputs, mlx_outputs)):
    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=atol,
        rtol=rtol,
        err_msg=f'Integration step {i}: JAX vs MLX outputs differ',
    )


def _compare_integration_int_step(
    test_case,
    config,
    jax_params,
    *,
    vocab_size=256,
    batch_size=1,
    num_steps=8,
    atol=1e-3,
    rtol=1e-3,
    seed=42,
):
  """Compare step-by-step output of a token model between JAX and MLX."""
  rng = np.random.RandomState(seed + 1)
  step_tokens = [
      rng.randint(0, vocab_size, size=(batch_size, 1)).astype(np.int32)
      for _ in range(num_steps)
  ]
  step_masks = [np.ones((batch_size, 1), dtype=bool) for _ in range(num_steps)]

  # JAX step.
  jax_model = config.make()
  jax_spec = jax.ShapeDtypeStruct((), jnp.int32)
  jax_state = jax_model.apply(
      {'params': jax_params},
      batch_size,
      jax_spec,
      training=False,
      method=jax_model.get_initial_state,
  )
  jax_outputs = []
  for i in range(num_steps):
    x_jax = jax_types.Sequence(
        jnp.array(step_tokens[i]),
        jnp.array(step_masks[i], dtype=jnp.bool_),
    )
    y_jax, jax_state = jax_model.apply(
        {'params': jax_params},
        x_jax,
        jax_state,
        training=False,
        method=jax_model.step,
    )
    jax_outputs.append(np.array(y_jax.values))

  # MLX step.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(mlx_model, jax_params, config)
  mlx_spec = ShapeDType((), mx.int32)
  mlx_state = mlx_model.get_initial_state(batch_size, mlx_spec)
  mlx_outputs = []
  for i in range(num_steps):
    x_mx = Sequence(
        mx.array(step_tokens[i], dtype=mx.int32),
        mx.array(step_masks[i], dtype=mx.bool_),
    )
    y_mx, mlx_state = mlx_model.step(x_mx, mlx_state)
    mx.eval(y_mx.values)
    mlx_outputs.append(np.array(y_mx.values))

  for i, (jax_out, mlx_out) in enumerate(zip(jax_outputs, mlx_outputs)):
    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=atol,
        rtol=rtol,
        err_msg=f'Integration step {i}: JAX vs MLX outputs differ',
    )


class DecoderTransformerIntegrationTest(parameterized.TestCase):
  """Cross-backend: decoder-only transformer (token input)."""

  def _config(self, dim=32, num_heads=4, num_layers=2, vocab_size=64):
    return sl.Serial.Config([
        sl.Embedding.Config(num_embeddings=vocab_size, dimension=dim),
        sl.Repeat.Config(
            num_repeats=num_layers,
            layer=sl.Serial.Config([
                sl.Residual.Config([
                    sl.RMSNormalization.Config(),
                    sl.DotProductSelfAttention.Config(
                        num_heads=num_heads,
                        units_per_head=dim // num_heads,
                        max_past_horizon=64,
                        max_future_horizon=0,
                        query_network=(
                            sl.ApplyRotaryPositionalEncoding.Config(
                                max_wavelength=10_000.0,
                            )
                        ),
                        key_network=(
                            sl.ApplyRotaryPositionalEncoding.Config(
                                max_wavelength=10_000.0,
                            )
                        ),
                    ),
                    sl.Flatten.Config(),
                ]),
                sl.Residual.Config([
                    sl.RMSNormalization.Config(),
                    sl.Dense.Config(features=dim * 4, activation=jax.nn.gelu),
                    sl.Dense.Config(features=dim),
                ]),
            ]),
        ),
        sl.RMSNormalization.Config(),
        sl.Dense.Config(features=vocab_size),
    ])

  def test_layer(self):
    config = self._config()
    _compare_integration_int(self, config, vocab_size=64)

  def test_step(self):
    config = self._config()
    jax_params = _compare_integration_int(self, config, vocab_size=64)
    _compare_integration_int_step(
        self, config, jax_params, vocab_size=64, num_steps=6
    )


class GQADecoderIntegrationTest(parameterized.TestCase):
  """Cross-backend: decoder transformer with GQA."""

  def _config(self, dim=32, num_heads=4, num_kv_heads=2, vocab_size=64):
    return sl.Serial.Config([
        sl.Embedding.Config(num_embeddings=vocab_size, dimension=dim),
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            sl.DotProductSelfAttention.Config(
                num_heads=num_heads,
                units_per_head=dim // num_heads,
                max_past_horizon=64,
                max_future_horizon=0,
                num_kv_heads=num_kv_heads,
                input_projection=(
                    attn_common.SeparateQueryKeyValueProjection()
                ),
                query_network=(
                    sl.ApplyRotaryPositionalEncoding.Config(
                        max_wavelength=10_000.0,
                    )
                ),
                key_network=(
                    sl.ApplyRotaryPositionalEncoding.Config(
                        max_wavelength=10_000.0,
                    )
                ),
            ),
            sl.Flatten.Config(),
        ]),
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            sl.Dense.Config(features=dim * 4, activation=jax.nn.gelu),
            sl.Dense.Config(features=dim),
        ]),
        sl.Dense.Config(features=vocab_size),
    ])

  def test_layer(self):
    config = self._config()
    _compare_integration_int(self, config, vocab_size=64)

  def test_step(self):
    config = self._config()
    jax_params = _compare_integration_int(self, config, vocab_size=64)
    _compare_integration_int_step(
        self, config, jax_params, vocab_size=64, num_steps=6
    )


class ConvEncoderIntegrationTest(parameterized.TestCase):
  """Cross-backend: conv + dense encoder (float input)."""

  def _config(self, dim=16):
    return sl.Serial.Config([
        sl.Conv1D.Config(filters=dim, kernel_size=3, padding='causal'),
        sl.Relu.Config(),
        sl.Conv1D.Config(filters=dim, kernel_size=3, padding='causal'),
        sl.Relu.Config(),
        sl.LayerNormalization.Config(),
        sl.Dense.Config(features=dim * 2, activation=jax.nn.gelu),
        sl.Dense.Config(features=dim),
    ])

  def test_layer(self):
    config = self._config()
    _compare_integration_float(self, config, (8,))

  def test_step(self):
    config = self._config()
    jax_params, _, _ = _compare_integration_float(self, config, (8,))
    _compare_integration_step(self, config, (8,), jax_params)


class ConvAttentionIntegrationTest(parameterized.TestCase):
  """Cross-backend: conv + self-attention + pooling model (float input)."""

  def _config(self, dim=16):
    return sl.Serial.Config([
        sl.Conv1D.Config(filters=dim, kernel_size=3, padding='causal'),
        sl.Swish.Config(),
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            sl.DotProductSelfAttention.Config(
                num_heads=2,
                units_per_head=dim // 2,
                max_past_horizon=32,
                max_future_horizon=0,
            ),
            sl.Flatten.Config(),
        ]),
        sl.MaxPooling1D.Config(pool_size=2, padding='causal'),
        sl.Dense.Config(features=dim),
    ])

  def test_layer(self):
    config = self._config()
    _compare_integration_float(self, config, (8,), time=8)

  def test_step(self):
    config = self._config()
    jax_params, _, _ = _compare_integration_float(self, config, (8,), time=8)
    _compare_integration_step(
        self, config, (8,), jax_params, num_steps=8, atol=5e-3, rtol=5e-3
    )


class EncoderDecoderIntegrationTest(parameterized.TestCase):
  """Cross-backend: encoder-decoder with cross-attention (float input)."""

  def _encoder_config(self, dim=16):
    return sl.Serial.Config([
        sl.Dense.Config(features=dim, activation=jax.nn.relu),
        sl.Dense.Config(features=dim),
    ])

  def _decoder_config(self, dim=16):
    from sequence_layers.jax.attention import (
        dot_product_attention as jax_cross_attn,
    )

    return sl.Serial.Config([
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            sl.DotProductSelfAttention.Config(
                num_heads=2,
                units_per_head=dim // 2,
                max_past_horizon=32,
                max_future_horizon=0,
            ),
            sl.Flatten.Config(),
        ]),
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            jax_cross_attn.DotProductAttention.Config(
                source_name='encoder',
                num_heads=2,
                units_per_head=dim // 2,
            ),
            sl.Flatten.Config(),
        ]),
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            sl.Dense.Config(features=dim * 2, activation=jax.nn.gelu),
            sl.Dense.Config(features=dim),
        ]),
    ])

  def _make_constants(self, batch_size, time, rng, dim=16):
    source_values = rng.randn(batch_size, time, dim).astype(np.float32)
    source_mask = np.ones((batch_size, time), dtype=bool)
    jax_source = jax_types.Sequence(
        jnp.array(source_values), jnp.array(source_mask, dtype=jnp.bool_)
    )
    mlx_source = Sequence(
        mx.array(source_values), mx.array(source_mask, dtype=mx.bool_)
    )
    return {'encoder': jax_source}, {'encoder': mlx_source}

  def test_layer(self):
    config = self._decoder_config()
    _compare_integration_float(
        self,
        config,
        (16,),
        constants_fn=lambda b, t, rng: self._make_constants(b, t, rng),
    )

  def test_step(self):
    config = self._decoder_config()
    jax_params, jax_constants, mlx_constants = _compare_integration_float(
        self,
        config,
        (16,),
        constants_fn=lambda b, t, rng: self._make_constants(b, t, rng),
    )
    _compare_integration_step(
        self,
        config,
        (16,),
        jax_params,
        jax_constants=jax_constants,
        mlx_constants=mlx_constants,
    )


class DepthwiseConvPipelineIntegrationTest(parameterized.TestCase):
  """Cross-backend: depthwise conv + dense + normalization pipeline."""

  def _config(self, dim=16):
    return sl.Serial.Config([
        sl.Dense.Config(features=dim),
        sl.DepthwiseConv1D.Config(kernel_size=3, padding='causal'),
        sl.Swish.Config(),
        sl.LayerNormalization.Config(),
        sl.Dense.Config(features=dim * 2, activation=jax.nn.gelu),
        sl.Dense.Config(features=dim),
        sl.DepthwiseConv1D.Config(kernel_size=5, padding='causal'),
        sl.RMSNormalization.Config(),
        sl.Dense.Config(features=dim),
    ])

  def test_layer(self):
    config = self._config()
    _compare_integration_float(self, config, (8,), atol=2e-3, rtol=2e-3)

  def test_step(self):
    config = self._config()
    jax_params, _, _ = _compare_integration_float(
        self, config, (8,), atol=2e-3, rtol=2e-3
    )
    _compare_integration_step(
        self, config, (8,), jax_params, atol=2e-3, rtol=2e-3
    )


class ParallelBranchIntegrationTest(parameterized.TestCase):
  """Cross-backend: parallel branches with different processing."""

  def _config(self, dim=8):
    return sl.Serial.Config([
        sl.Parallel.Config(
            layers=[
                sl.Serial.Config([
                    sl.Dense.Config(features=dim, activation=jax.nn.relu),
                    sl.Dense.Config(features=dim),
                ]),
                sl.Serial.Config([
                    sl.Dense.Config(features=dim, activation=jax.nn.gelu),
                    sl.Dense.Config(features=dim),
                ]),
            ],
            combination=sl.CombinationMode.ADD,
        ),
        sl.RMSNormalization.Config(),
        sl.Dense.Config(features=dim),
    ])

  def test_layer(self):
    config = self._config()
    _compare_integration_float(self, config, (8,))

  def test_step(self):
    config = self._config()
    jax_params, _, _ = _compare_integration_float(self, config, (8,))
    _compare_integration_step(self, config, (8,), jax_params)


def _compare_conditioning(
    test_case,
    config,
    input_shape,
    cond_shape,
    *,
    batch_size=2,
    time=8,
    atol=1e-5,
    rtol=1e-5,
    seed=42,
):
  """Compare Conditioning layer: JAX vs MLX."""
  rng = np.random.RandomState(seed)
  values = rng.randn(batch_size, time, *input_shape).astype(np.float32)
  mask = np.ones((batch_size, time), dtype=bool)
  cond_values = rng.randn(batch_size, time, *cond_shape).astype(np.float32)
  cond_mask = np.ones((batch_size, time), dtype=bool)

  jax_constants = {
      'cond': jax_types.Sequence(
          jnp.array(cond_values), jnp.array(cond_mask, dtype=jnp.bool_)
      )
  }
  mlx_constants = {
      'cond': Sequence(
          mx.array(cond_values), mx.array(cond_mask, dtype=mx.bool_)
      )
  }

  # JAX.
  jax_model = config.make()
  x_jax = jax_types.Sequence(
      jnp.array(values), jnp.array(mask, dtype=jnp.bool_)
  )
  variables = jax_model.init(
      jax.random.PRNGKey(0),
      x_jax,
      training=False,
      constants=jax_constants,
  )
  jax_params = variables.get('params', {})
  jax_out = np.array(
      jax_model.apply(
          variables,
          x_jax,
          training=False,
          constants=jax_constants,
      ).values
  )

  # MLX.
  mlx_model = config.make(backend='mlx')
  if jax_params:
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType(input_shape, mx.float32),
        constants=mlx_constants,
    )
  x_mx = Sequence(mx.array(values), mx.array(mask, dtype=mx.bool_))
  mlx_out = np.array(mlx_model.layer(x_mx, constants=mlx_constants).values)

  np.testing.assert_allclose(
      mlx_out,
      jax_out,
      atol=atol,
      rtol=rtol,
      err_msg=f'{config.__class__.__qualname__}: outputs differ',
  )


class ConditioningCrossBackendTest(parameterized.TestCase):
  """Conditioning: JAX vs MLX layer-mode."""

  def test_identity_add(self):
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.IDENTITY,
        combination=jax_cond.BaseConditioning.Combination.ADD,
    )
    _compare_conditioning(self, config, (8,), (8,))

  def test_identity_mul(self):
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.IDENTITY,
        combination=jax_cond.BaseConditioning.Combination.MUL,
    )
    _compare_conditioning(self, config, (8,), (8,))

  def test_identity_concat(self):
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.IDENTITY,
        combination=jax_cond.BaseConditioning.Combination.CONCAT,
    )
    _compare_conditioning(self, config, (4,), (6,))

  def test_linear_add(self):
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.LINEAR,
        combination=jax_cond.BaseConditioning.Combination.ADD,
    )
    _compare_conditioning(self, config, (4,), (6,))

  def test_linear_affine_shift(self):
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.LINEAR,
        combination=jax_cond.BaseConditioning.Combination.AFFINE_SHIFT,
    )
    _compare_conditioning(self, config, (4,), (6,))

  def test_linear_affine_scale(self):
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.LINEAR,
        combination=jax_cond.BaseConditioning.Combination.AFFINE_SCALE,
    )
    _compare_conditioning(self, config, (4,), (6,))

  def test_linear_affine(self):
    from sequence_layers.jax import conditioning as jax_cond

    config = jax_cond.Conditioning.Config(
        conditioning_name='cond',
        projection=jax_cond.BaseConditioning.Projection.LINEAR_AFFINE,
        combination=jax_cond.BaseConditioning.Combination.AFFINE,
    )
    _compare_conditioning(self, config, (4,), (6,))


if __name__ == '__main__':
  absltest.main()
