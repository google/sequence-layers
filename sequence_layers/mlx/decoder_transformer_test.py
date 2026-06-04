"""End-to-end test: decoder-only transformer on MLX.

Defines a small decoder transformer using MLX configs, builds an MLX
model via config.make(), and tests inference + export.
"""

import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from sequence_layers.mlx import attention as mlx_attn
from sequence_layers.mlx import combinators as mlx_comb
from sequence_layers.mlx import dense as mlx_dense
from sequence_layers.mlx import export
from sequence_layers.mlx import normalization as mlx_norm
from sequence_layers.mlx import position as mlx_pos
from sequence_layers.mlx import simple as mlx_simple
from sequence_layers.mlx import test_utils
from sequence_layers.mlx import types as bt

Sequence = bt.Sequence
ShapeDType = bt.ShapeDType


def _decoder_config(vocab_size=256, dim=64, num_heads=4, num_layers=2):
  """A small decoder-only transformer config."""
  return mlx_comb.Serial.Config([
      mlx_simple.Embedding.Config(
          num_embeddings=vocab_size,
          dimension=dim,
      ),
      mlx_comb.Repeat.Config(
          num_repeats=num_layers,
          layer=mlx_comb.Serial.Config([
              mlx_comb.Residual.Config([
                  mlx_norm.RMSNormalization.Config(),
                  mlx_attn.DotProductSelfAttention.Config(
                      num_heads=num_heads,
                      units_per_head=dim // num_heads,
                      max_past_horizon=128,
                      max_future_horizon=0,
                      query_network=(
                          mlx_pos.ApplyRotaryPositionalEncoding.Config(
                              max_wavelength=10000.0,
                          )
                      ),
                      key_network=(
                          mlx_pos.ApplyRotaryPositionalEncoding.Config(
                              max_wavelength=10000.0,
                          )
                      ),
                  ),
                  mlx_simple.Flatten.Config(),
              ]),
              mlx_comb.Residual.Config([
                  mlx_norm.RMSNormalization.Config(),
                  mlx_dense.Dense.Config(
                      features=dim * 4,
                      activation=nn.gelu,
                  ),
                  mlx_dense.Dense.Config(features=dim),
              ]),
          ]),
      ),
      mlx_norm.RMSNormalization.Config(),
      mlx_dense.Dense.Config(features=vocab_size),
  ])


def _make_token_sequence(tokens):
  """Create a Sequence from integer token ids.

  Args:
    tokens: A 2D list [[t1, t2, ...], ...] of shape [batch, time].

  Returns:
    Sequence with values shape [batch, time] and all-valid mask.
  """
  arr = mx.array(tokens, dtype=mx.int32)
  if arr.ndim != 2:
    raise ValueError(f'Expected 2D token array, got shape {arr.shape}')
  mask = mx.ones(arr.shape, dtype=mx.bool_)
  return Sequence(arr, mask)


@unittest.skip("Core dumps on Linux due to activations")
class DecoderTransformerTest(test_utils.SequenceLayerTest, parameterized.TestCase):
  """End-to-end tests for a decoder transformer on MLX."""

  def _make_model(self, config=None):
    if config is None:
      config = _decoder_config()
    model = config.make()
    return model

  def test_make_mlx(self):
    """config.make() produces an MLX SequenceLayer."""
    config = _decoder_config()
    model = config.make()
    from sequence_layers.mlx import types

    self.assertIsInstance(model, types.SequenceLayer)
    self.assertTrue(model.supports_step)

  def test_layer(self):
    """model.layer() produces correct output shape and dtype."""
    model = self._make_model()
    batch, time, vocab_size = 2, 8, 256
    # Input: integer token ids with scalar channel shape ().
    x = _make_token_sequence([[0] * time] * batch)
    y = model.layer(x, training=False)
    self.assertEqual(y.shape, (batch, time, vocab_size))

  def test_step(self):
    """model.step() runs and output shape is correct."""
    model = self._make_model()
    batch, vocab_size = 1, 256
    input_spec = ShapeDType((), mx.int32)

    export._materialize_deferred(model, batch, input_spec)
    state = model.get_initial_state(batch, input_spec, training=False)

    # Step with a single token.
    x = _make_token_sequence([[42]])
    y, new_state = model.step(x, state, training=False)
    self.assertEqual(y.shape, (batch, 1, vocab_size))

    # Second step.
    x2 = _make_token_sequence([[7]])
    y2, state2 = model.step(x2, new_state, training=False)
    self.assertEqual(y2.shape, (batch, 1, vocab_size))

  def test_step_layer_match(self):
    """step() and layer() produce matching outputs."""
    model = self._make_model()
    batch, time = 2, 8
    values = mx.random.randint(0, 256, shape=(batch, time)).astype(mx.int32)
    mask = mx.ones((batch, time), dtype=mx.bool_)
    x = Sequence(values, mask)

    y_layer = model.layer(x, training=False)
    y_step, _ = self._step_by_step(model, x)

    np.testing.assert_allclose(
        np.array(y_step.values),
        np.array(y_layer.values),
        atol=1e-4,
        rtol=1e-4,
        err_msg='step() and layer() outputs differ',
    )

  def test_autoregressive_generation(self):
    """Token-by-token generation loop with random weights."""
    model = self._make_model()
    batch, vocab_size, max_len = 1, 256, 16
    input_spec = ShapeDType((), mx.int32)

    export._materialize_deferred(model, batch, input_spec)
    state = model.get_initial_state(batch, input_spec, training=False)

    token = 0
    generated = [token]

    for _ in range(max_len - 1):
      x = _make_token_sequence([[token]])
      y, state = model.step(x, state, training=False)
      mx.eval(y.values)

      logits = y.values[0, 0]  # [vocab_size]
      token = int(mx.argmax(logits))
      generated.append(token)

    self.assertLen(generated, max_len)
    for t in generated:
      self.assertGreaterEqual(t, 0)
      self.assertLess(t, vocab_size)

  def test_export_import(self):
    """Export step to .mlxfn, import, verify same outputs."""
    model = self._make_model()
    batch, vocab_size = 1, 256
    input_spec = ShapeDType((), mx.int32)

    export._materialize_deferred(model, batch, input_spec)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'decoder.mlxfn')
      export.export_step(model, path, batch_size=batch, input_spec=input_spec)
      self.assertTrue(os.path.exists(path))

      imported = mx.import_function(path)

      tokens = [42, 7, 13]
      inputs = [_make_token_sequence([[t]]) for t in tokens]

      # Native inference.
      state = model.get_initial_state(batch, input_spec, training=False)
      native_outputs = []
      for x in inputs:
        y, state = model.step(x, state, training=False)
        mx.eval(y.values)
        native_outputs.append(np.array(y.values))

      # Exported inference.
      flat_state, structure = export.get_initial_state_flat(
          model, batch, input_spec
      )
      exported_outputs = []
      for x in inputs:
        y_vals, y_mask, flat_state = export.run_exported(
            imported, x.values, x.mask, flat_state
        )
        mx.eval(y_vals)
        exported_outputs.append(np.array(y_vals))

      for i, (native, exported) in enumerate(
          zip(native_outputs, exported_outputs)
      ):
        np.testing.assert_allclose(
            exported,
            native,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f'Token {tokens[i]}: exported != native',
        )

  def test_export_file_size(self):
    """Exported .mlxfn file has reasonable size."""
    model = self._make_model()
    batch = 1
    input_spec = ShapeDType((), mx.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'decoder.mlxfn')
      export.export_step(model, path, batch_size=batch, input_spec=input_spec)
      size_mb = os.path.getsize(path) / (1024 * 1024)
      # Small model should be < 10 MB.
      self.assertLess(size_mb, 10.0)


if __name__ == '__main__':
  absltest.main()
