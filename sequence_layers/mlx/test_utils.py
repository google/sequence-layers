"""Test utilities for MLX sequence layers."""

import mlx.core as mx
import numpy as np
import typing

from sequence_layers.abstract import test_utils
from sequence_layers.mlx import types

Sequence = types.Sequence
MaskedSequence = types.MaskedSequence
ShapeDType = types.ShapeDType

class SequenceLayerTest(test_utils.SequenceLayerTest):
  @typing.override
  def get_backend(self):
    return mx

  @property
  @typing.override
  def Sequence(self):
    return types.Sequence

  @property
  @typing.override
  def MaskedSequence(self):
    return types.MaskedSequence

  @typing.override
  def random_sequence(
      self,
      *dims: int,
      dtype=mx.float32,
      random_mask: bool = False,
      random_lengths: bool | None = None,
      low: int | None = 0,
      high: int | None = 10,
      low_length: int = 0,
      high_length: int | None = None,
  ) -> types.Sequence:
    if len(dims) < 2:
      raise ValueError("dims must be at least (batch, time)")
    batch_size = dims[0]
    time = dims[1]
    shape = dims[2:]
    values = mx.random.normal((batch_size, time) + shape, dtype=dtype)
    mask = mx.ones((batch_size, time), dtype=mx.bool_)
    return self.Sequence(values, mask)

  @typing.override
  def verify_contract(
      self,
      layer,
      x: types.Sequence,
      *,
      training: bool = False,
      constants=None,
      atol: float = 1e-5,
      rtol: float = 1e-5,
      test_step: bool = True,
      **kwargs,
  ):
    """Verify that a layer's layer() and step() outputs are consistent."""
    y_layer = self.call_layer(layer, x, training=training, constants=constants)

    if not test_step:
        return y_layer

    # Check output shape.
    expected_shape = layer.get_output_shape(x.channel_shape, constants=constants)

    # Test step().
    block_size = layer.block_size
    state = self.call_get_initial_state(
        layer, x.shape[0], types.ShapeDType(expected_shape, x.dtype), constants=constants
    )
    outputs_values = []
    outputs_masks = []
    
    for i in range(0, x.shape[1], block_size):
        x_block_vals = x.values[:, i : i + block_size]
        x_block_mask = x.mask[:, i : i + block_size]
        x_block = self.Sequence(x_block_vals, x_block_mask)
        y_block, state = layer.step(
            x_block,
            state,
            training=training,
            constants=constants,
        )
        outputs_values.append(y_block.values)
        outputs_masks.append(y_block.mask)

    y_values = mx.concatenate(outputs_values, axis=1)
    y_mask = mx.concatenate(outputs_masks, axis=1)
    y_step = self.Sequence(y_values, y_mask)

    # Check shapes match.
    self.assertEqual(y_step.shape, y_layer.shape)

    # Check values match.
    y_layer_np = np.array(y_layer.values)
    y_step_np = np.array(y_step.values)
    np.testing.assert_allclose(
        y_step_np,
        y_layer_np,
        atol=atol,
        rtol=rtol,
        err_msg=f'{layer.__class__.__name__}: step() and layer() outputs differ',
    )
    return y_layer
