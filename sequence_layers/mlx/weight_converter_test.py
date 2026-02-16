"""Tests for weight_converter: Linen → MLX param conversion.

Requires both JAX and MLX to be importable.
"""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import sequence_layers.jax as sl
from sequence_layers.jax import types as jax_types
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import export
from sequence_layers.mlx import weight_converter

Sequence = bt.Sequence
ShapeDType = bt.ShapeDType


def _run_jax_layer(config, jax_params, tokens):
  """Run a Linen model on integer token inputs.

  Args:
    config: A SequenceLayerConfig.
    jax_params: Linen params dict.
    tokens: A numpy array of shape [batch, time].

  Returns:
    numpy array of output values.
  """
  model = config.make()
  x = jax_types.Sequence(
      jnp.array(tokens, dtype=jnp.int32),
      jnp.ones(tokens.shape, dtype=jnp.bool_),
  )
  y = model.apply({'params': jax_params}, x, training=False)
  return np.array(y.values)


def _run_mlx_layer(mlx_model, tokens):
  """Run an MLX model on integer token inputs.

  Args:
    mlx_model: An MLX SequenceLayer.
    tokens: A numpy array of shape [batch, time].

  Returns:
    numpy array of output values.
  """
  x = Sequence(
      mx.array(tokens, dtype=mx.int32),
      mx.ones(tokens.shape, dtype=mx.bool_),
  )
  y = mlx_model.layer(x)
  mx.eval(y.values)
  return np.array(y.values)


class EmbeddingConversionTest(parameterized.TestCase):
  """Test embedding weight conversion."""

  def test_embedding_round_trip(self):
    config = sl.Embedding.Config(
        num_embeddings=32,
        dimension=16,
    )
    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x, training=False)
    jax_params = variables['params']

    # Create MLX model and load weights.
    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(mlx_model, jax_params, config)

    tokens = np.array([[0, 5, 10, 31]])
    jax_out = _run_jax_layer(config, jax_params, tokens)
    mlx_out = _run_mlx_layer(mlx_model, tokens)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-5,
        rtol=1e-5,
        err_msg='Embedding outputs differ',
    )


class DenseConversionTest(parameterized.TestCase):
  """Test Dense weight conversion."""

  def test_dense_round_trip(self):
    config = sl.Serial.Config([
        sl.Embedding.Config(num_embeddings=32, dimension=8),
        sl.Dense.Config(features=16),
    ])
    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x, training=False)
    jax_params = variables['params']

    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(mlx_model, jax_params, config)

    tokens = np.array([[0, 3, 7, 15]])
    jax_out = _run_jax_layer(config, jax_params, tokens)
    mlx_out = _run_mlx_layer(mlx_model, tokens)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-5,
        rtol=1e-5,
        err_msg='Dense outputs differ',
    )


class RMSNormConversionTest(parameterized.TestCase):
  """Test RMSNorm weight conversion."""

  def test_rms_norm_round_trip(self):
    config = sl.Serial.Config([
        sl.Embedding.Config(num_embeddings=32, dimension=8),
        sl.RMSNormalization.Config(),
    ])
    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x, training=False)
    jax_params = variables['params']

    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(mlx_model, jax_params, config)

    tokens = np.array([[0, 3, 7, 15]])
    jax_out = _run_jax_layer(config, jax_params, tokens)
    mlx_out = _run_mlx_layer(mlx_model, tokens)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-5,
        rtol=1e-5,
        err_msg='RMSNorm outputs differ',
    )


def _run_jax_layer_float(config, jax_params, values):
  """Run a Linen model on float inputs.

  Args:
    config: A SequenceLayerConfig.
    jax_params: Linen params dict.
    values: A numpy array of shape [batch, time, channels].

  Returns:
    numpy array of output values.
  """
  model = config.make()
  x = jax_types.Sequence(
      jnp.array(values, dtype=jnp.float32),
      jnp.ones(values.shape[:2], dtype=jnp.bool_),
  )
  y = model.apply({'params': jax_params}, x, training=False)
  return np.array(y.values)


def _run_mlx_layer_float(mlx_model, values):
  """Run an MLX model on float inputs.

  Args:
    mlx_model: An MLX SequenceLayer.
    values: A numpy array of shape [batch, time, channels].

  Returns:
    numpy array of output values.
  """
  x = Sequence(
      mx.array(values, dtype=mx.float32),
      mx.ones(values.shape[:2], dtype=mx.bool_),
  )
  y = mlx_model.layer(x)
  mx.eval(y.values)
  return np.array(y.values)


class Conv1DConversionTest(parameterized.TestCase):
  """Test Conv1D weight conversion."""

  def test_conv1d_round_trip(self):
    config = sl.Conv1D.Config(
        filters=8,
        kernel_size=3,
        padding='causal',
    )
    in_channels = 4
    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 8, in_channels), dtype=jnp.float32),
        jnp.ones((1, 8), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x, training=False)
    jax_params = variables['params']

    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType((in_channels,), mx.float32),
    )

    values = (
        np.random.RandomState(42).randn(1, 8, in_channels).astype(np.float32)
    )
    jax_out = _run_jax_layer_float(config, jax_params, values)
    mlx_out = _run_mlx_layer_float(mlx_model, values)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-5,
        rtol=1e-5,
        err_msg='Conv1D outputs differ',
    )


class DepthwiseConv1DConversionTest(parameterized.TestCase):
  """Test DepthwiseConv1D weight conversion."""

  def test_depthwise_conv1d_round_trip(self):
    config = sl.DepthwiseConv1D.Config(
        kernel_size=3,
        padding='causal',
    )
    in_channels = 4
    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 8, in_channels), dtype=jnp.float32),
        jnp.ones((1, 8), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x, training=False)
    jax_params = variables['params']

    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType((in_channels,), mx.float32),
    )

    values = (
        np.random.RandomState(42).randn(1, 8, in_channels).astype(np.float32)
    )
    jax_out = _run_jax_layer_float(config, jax_params, values)
    mlx_out = _run_mlx_layer_float(mlx_model, values)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-5,
        rtol=1e-5,
        err_msg='DepthwiseConv1D outputs differ',
    )


class Conv1DTransposeConversionTest(parameterized.TestCase):
  """Test Conv1DTranspose weight conversion."""

  def test_conv1d_transpose_round_trip(self):
    config = sl.Conv1DTranspose.Config(
        filters=8,
        kernel_size=3,
        strides=2,
        padding='causal',
    )
    in_channels = 4
    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 8, in_channels), dtype=jnp.float32),
        jnp.ones((1, 8), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(0), x, training=False)
    jax_params = variables['params']

    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(
        mlx_model,
        jax_params,
        config,
        input_spec=ShapeDType((in_channels,), mx.float32),
    )

    values = (
        np.random.RandomState(42).randn(1, 8, in_channels).astype(np.float32)
    )
    jax_out = _run_jax_layer_float(config, jax_params, values)
    mlx_out = _run_mlx_layer_float(mlx_model, values)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-5,
        rtol=1e-5,
        err_msg='Conv1DTranspose outputs differ',
    )


class DecoderTransformerConversionTest(parameterized.TestCase):
  """Test full decoder transformer weight conversion."""

  def _decoder_config(self):
    return sl.Serial.Config([
        sl.Embedding.Config(
            num_embeddings=32,
            dimension=16,
        ),
        sl.Repeat.Config(
            num_repeats=2,
            layer=sl.Serial.Config([
                sl.Residual.Config([
                    sl.RMSNormalization.Config(),
                    sl.DotProductSelfAttention.Config(
                        num_heads=2,
                        units_per_head=8,
                        max_past_horizon=16,
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
                    sl.Dense.Config(
                        features=64,
                        activation=jax.nn.gelu,
                    ),
                    sl.Dense.Config(features=16),
                ]),
            ]),
        ),
        sl.RMSNormalization.Config(),
        sl.Dense.Config(features=32),
    ])

  def test_layer_output_match(self):
    """Full transformer: JAX and MLX produce same layer() output."""
    config = self._decoder_config()

    # Build and init JAX model.
    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(42), x, training=False)
    jax_params = variables['params']

    # Build MLX model and load Linen weights.
    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(mlx_model, jax_params, config)

    tokens = np.array([[0, 5, 10, 31]])
    jax_out = _run_jax_layer(config, jax_params, tokens)
    mlx_out = _run_mlx_layer(mlx_model, tokens)

    np.testing.assert_allclose(
        mlx_out,
        jax_out,
        atol=1e-3,
        rtol=1e-3,
        err_msg='Decoder transformer outputs differ',
    )

  def test_step_output_match(self):
    """Full transformer: JAX step and MLX step produce same output."""
    config = self._decoder_config()

    jax_model = config.make()
    x = jax_types.Sequence(
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.ones((1, 4), dtype=jnp.bool_),
    )
    variables = jax_model.init(jax.random.PRNGKey(42), x, training=False)
    jax_params = variables['params']

    mlx_model = config.make(backend='mlx')
    weight_converter.load_linen_params(mlx_model, jax_params, config)

    # Run step-by-step on both.
    tokens = [5, 10, 31]

    # JAX step.
    jax_spec = jax.ShapeDtypeStruct((), jnp.int32)
    jax_state = jax_model.apply(
        {'params': jax_params},
        1,
        jax_spec,
        training=False,
        method=jax_model.get_initial_state,
    )
    jax_outputs = []
    for t in tokens:
      x_jax = jax_types.Sequence(
          jnp.array([[t]], dtype=jnp.int32),
          jnp.ones((1, 1), dtype=jnp.bool_),
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
    input_spec = ShapeDType((), mx.int32)
    export._materialize_deferred(mlx_model, 1, input_spec)
    mlx_state = mlx_model.get_initial_state(1, input_spec)
    mlx_outputs = []
    for t in tokens:
      x_mx = Sequence(
          mx.array([[t]], dtype=mx.int32),
          mx.ones((1, 1), dtype=mx.bool_),
      )
      y_mx, mlx_state = mlx_model.step(x_mx, mlx_state)
      mx.eval(y_mx.values)
      mlx_outputs.append(np.array(y_mx.values))

    for i, (jax_out, mlx_out) in enumerate(zip(jax_outputs, mlx_outputs)):
      np.testing.assert_allclose(
          mlx_out,
          jax_out,
          atol=1e-3,
          rtol=1e-3,
          err_msg=f'Step {i} (token={tokens[i]}): outputs differ',
      )


if __name__ == '__main__':
  absltest.main()
