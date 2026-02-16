"""Tests for MLX export utilities."""

import os
import tempfile

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import export
from sequence_layers.mlx import test_utils

Sequence = bt.Sequence
ShapeDType = bt.ShapeDType


class StateFlattenTest(parameterized.TestCase):
  """Tests for state flatten/unflatten."""

  def test_empty_tuple(self):
    state = ()
    flat, structure = export._flatten_state(state)
    self.assertEmpty(flat)
    rebuilt = export._unflatten_state(flat, structure)
    self.assertEqual(rebuilt, ())

  def test_single_array(self):
    arr = mx.zeros((2, 3))
    state = (arr,)
    flat, structure = export._flatten_state(state)
    self.assertLen(flat, 1)
    rebuilt = export._unflatten_state(flat, structure)
    self.assertIsInstance(rebuilt, tuple)
    np.testing.assert_array_equal(rebuilt[0], arr)

  def test_nested_tuples(self):
    a = mx.ones((2,))
    b = mx.zeros((3, 4))
    c = mx.full((1,), 5.0)
    state = ((a, b), (c, ()))
    flat, structure = export._flatten_state(state)
    self.assertLen(flat, 3)
    rebuilt = export._unflatten_state(flat, structure)
    np.testing.assert_array_equal(rebuilt[0][0], a)
    np.testing.assert_array_equal(rebuilt[0][1], b)
    np.testing.assert_array_equal(rebuilt[1][0], c)
    self.assertEqual(rebuilt[1][1], ())

  def test_attention_state_round_trip(self):
    """Simulate attention state: (keys, values, mask, time, (), (), ())."""
    keys = mx.zeros((2, 8, 4, 16))
    values = mx.zeros((2, 8, 4, 16))
    mask = mx.zeros((2, 8), dtype=mx.bool_)
    time = mx.zeros((2,), dtype=mx.int32)
    state = (keys, values, mask, time, (), (), ())
    flat, structure = export._flatten_state(state)
    self.assertLen(flat, 4)
    rebuilt = export._unflatten_state(flat, structure)
    np.testing.assert_array_equal(rebuilt[0], keys)
    np.testing.assert_array_equal(rebuilt[1], values)
    np.testing.assert_array_equal(rebuilt[2], mask)
    np.testing.assert_array_equal(rebuilt[3], time)
    self.assertEqual(rebuilt[4], ())
    self.assertEqual(rebuilt[5], ())
    self.assertEqual(rebuilt[6], ())

  def test_serial_state_round_trip(self):
    """Simulate Serial state: tuple of per-layer states."""
    state = (
        (),  # Identity (stateless)
        (
            mx.zeros((2, 4, 4, 8)),  # Attention keys
            mx.zeros((2, 4, 4, 8)),  # values
            mx.zeros((2, 4), dtype=mx.bool_),  # mask
            mx.zeros((2,), dtype=mx.int32),  # time
            mx.full((2, 1), -1, dtype=mx.int32),  # q_net_state
            mx.full((2, 1), -1, dtype=mx.int32),  # k_net_state
            (),
        ),  # v_net_state
        (),  # Dense (stateless)
    )
    flat, structure = export._flatten_state(state)
    self.assertLen(flat, 6)
    rebuilt = export._unflatten_state(flat, structure)
    self.assertEqual(rebuilt[0], ())
    self.assertLen(rebuilt[1], 7)
    self.assertEqual(rebuilt[2], ())


class ExportDenseTest(parameterized.TestCase):
  """Test exporting a simple Dense layer."""

  def test_export_dense_step(self):
    from sequence_layers.mlx import dense

    layer = dense.Dense(in_features=8, features=16, use_bias=True)
    input_spec = ShapeDType((8,), mx.float32)
    batch_size = 2

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'dense.mlxfn')
      export.export_step(
          layer,
          path,
          batch_size=batch_size,
          input_spec=input_spec,
      )
      self.assertTrue(os.path.exists(path))

      # Import and run.
      imported = mx.import_function(path)
      flat_state, structure = export.get_initial_state_flat(
          layer, batch_size, input_spec
      )

      x = test_utils.random_sequence(batch_size, 1, (8,))
      mx.eval(x.values, x.mask)

      # Run native.
      state = layer.get_initial_state(batch_size, input_spec)
      y_native, _ = layer.step(x, state)

      # Run exported.
      y_vals, y_mask, new_state = export.run_exported(
          imported, x.values, x.mask, flat_state
      )
      mx.eval(y_native.values, y_vals)

      np.testing.assert_allclose(
          np.array(y_vals),
          np.array(y_native.values),
          atol=1e-5,
          rtol=1e-5,
      )

  def test_export_dense_no_bias(self):
    from sequence_layers.mlx import dense

    layer = dense.Dense(in_features=8, features=16, use_bias=False)
    input_spec = ShapeDType((8,), mx.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'dense_nobias.mlxfn')
      export.export_step(layer, path, batch_size=1, input_spec=input_spec)

      imported = mx.import_function(path)
      flat_state, _ = export.get_initial_state_flat(layer, 1, input_spec)

      x = test_utils.random_sequence(1, 1, (8,))
      mx.eval(x.values, x.mask)

      state = layer.get_initial_state(1, input_spec)
      y_native, _ = layer.step(x, state)

      y_vals, _, _ = export.run_exported(imported, x.values, x.mask, flat_state)
      mx.eval(y_native.values, y_vals)

      np.testing.assert_allclose(
          np.array(y_vals),
          np.array(y_native.values),
          atol=1e-5,
          rtol=1e-5,
      )


class ExportAttentionTest(parameterized.TestCase):
  """Test exporting attention with KV cache."""

  def test_export_attention_multi_step(self):
    from sequence_layers.mlx import attention

    layer = attention.DotProductSelfAttention(
        in_features=16,
        num_heads=2,
        units_per_head=8,
        max_past_horizon=32,
        max_future_horizon=0,
    )
    input_spec = ShapeDType((16,), mx.float32)
    batch_size = 1

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'attn.mlxfn')
      export.export_step(
          layer,
          path,
          batch_size=batch_size,
          input_spec=input_spec,
      )

      imported = mx.import_function(path)

      # Run 3 steps natively.
      state = layer.get_initial_state(batch_size, input_spec)
      flat_state, structure = export._flatten_state(state)
      mx.eval(*flat_state)

      # Use same inputs for both native and exported.
      inputs = []
      for _ in range(3):
        x = test_utils.random_sequence(batch_size, 1, (16,))
        mx.eval(x.values, x.mask)
        inputs.append(x)

      # Native.
      native_state = state
      native_outputs = []
      for x in inputs:
        y, native_state = layer.step(x, native_state)
        mx.eval(y.values)
        native_outputs.append(np.array(y.values))

      # Exported.
      exported_state = list(flat_state)
      exported_outputs = []
      for x in inputs:
        y_vals, y_mask, exported_state = export.run_exported(
            imported, x.values, x.mask, exported_state
        )
        mx.eval(y_vals)
        exported_outputs.append(np.array(y_vals))

      for i, (native, exported) in enumerate(
          zip(native_outputs, exported_outputs)
      ):
        np.testing.assert_allclose(
            exported,
            native,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f'Step {i} mismatch',
        )


class ExportSerialTest(parameterized.TestCase):
  """Test exporting a Serial model."""

  def test_export_serial(self):
    from sequence_layers.mlx import combinators
    from sequence_layers.mlx import dense
    from sequence_layers.mlx import normalization

    model = combinators.Serial([
        normalization.RMSNormalization(epsilon=1e-6),
        dense.Dense(
            in_features=8,
            features=16,
            use_bias=True,
            activation=mx.sigmoid,
        ),
        dense.Dense(in_features=16, features=8, use_bias=True),
    ])
    input_spec = ShapeDType((8,), mx.float32)
    batch_size = 2

    # Materialize deferred layers.
    export._materialize_deferred(model, batch_size, input_spec)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'serial.mlxfn')
      export.export_step(
          model,
          path,
          batch_size=batch_size,
          input_spec=input_spec,
      )

      imported = mx.import_function(path)
      flat_state, structure = export.get_initial_state_flat(
          model, batch_size, input_spec
      )

      x = test_utils.random_sequence(batch_size, 1, (8,))
      mx.eval(x.values, x.mask)

      # Native.
      state = model.get_initial_state(batch_size, input_spec)
      y_native, _ = model.step(x, state)

      # Exported.
      y_vals, y_mask, _ = export.run_exported(
          imported, x.values, x.mask, flat_state
      )
      mx.eval(y_native.values, y_vals)

      np.testing.assert_allclose(
          np.array(y_vals),
          np.array(y_native.values),
          atol=1e-5,
          rtol=1e-5,
      )


if __name__ == '__main__':
  absltest.main()
