"""Test utilities for MLX sequence layers."""

import mlx.core as mx
import numpy as np

from sequence_layers.mlx import basic_types as bt

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence
ShapeDType = bt.ShapeDType


def random_sequence(
    batch: int,
    time: int,
    channels: int | tuple[int, ...],
    *,
    dtype=mx.float32,
    mask: mx.array | None = None,
    masked: bool = True,
) -> Sequence:
  """Create a random Sequence for testing.

  Args:
    batch: Batch size.
    time: Sequence length.
    channels: Channel size (int) or channel shape (tuple).
    dtype: Values dtype.
    mask: Optional explicit mask. If None, all-valid mask is used.
    masked: If True, returns a MaskedSequence. If False, a Sequence.

  Returns:
    A random Sequence or MaskedSequence.
  """
  if isinstance(channels, int):
    channels = (channels,)
  shape = (batch, time) + channels
  values = mx.random.normal(shape=shape).astype(dtype)
  if mask is None:
    mask = mx.ones((batch, time), dtype=mx.bool_)
  if masked:
    return MaskedSequence(values, mask)
  return Sequence(values, mask)


def step_by_step(
    layer,
    x: Sequence,
    *,
    block_size: int = 1,
    constants=None,
    stream_constants=None,
) -> tuple[Sequence, object]:
  """Run a layer step-by-step and concatenate outputs.

  Args:
    layer: A SequenceLayer with supports_step.
    x: Input sequence [batch, time, ...].
    block_size: Number of timesteps per step.
    constants: Optional constants dict (static, passed as-is each step).
    stream_constants: Optional dict of source_name -> Sequence. These are
        sliced at the same block_size as input for each step, merging into
        the constants dict. Use this for streaming cross-attention sources.

  Returns:
    (output_sequence, final_state)
  """
  batch = x.shape[0]
  time = x.shape[1]
  spec = x.channel_spec

  # Build initial constants with full stream sources for get_initial_state.
  init_constants = dict(constants) if constants else {}
  if stream_constants:
    init_constants.update(stream_constants)

  state = layer.get_initial_state(batch, spec, constants=init_constants or None)

  outputs_values = []
  outputs_masks = []

  for t in range(0, time, block_size):
    x_block = Sequence(
        x.values[:, t : t + block_size],
        x.mask[:, t : t + block_size],
    )

    # Build per-step constants with sliced stream sources.
    step_constants = dict(constants) if constants else {}
    if stream_constants:
      for name, seq in stream_constants.items():
        step_constants[name] = Sequence(
            seq.values[:, t : t + block_size],
            seq.mask[:, t : t + block_size],
        )

    y_block, state = layer.step(
        x_block,
        state,
        constants=step_constants or None,
    )
    outputs_values.append(y_block.values)
    outputs_masks.append(y_block.mask)

  y_values = mx.concatenate(outputs_values, axis=1)
  y_mask = mx.concatenate(outputs_masks, axis=1)
  return Sequence(y_values, y_mask), state


def verify_contract(
    test_case,
    layer,
    input_shape,
    *,
    batch_size: int = 2,
    time: int = 8,
    dtype=mx.float32,
    constants=None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    test_step: bool = True,
):
  """Verify that a layer's layer() and step() outputs are consistent.

  Checks:
    1. layer() runs without error and produces correct output shape.
    2. step() runs without error and produces correct output shape.
    3. layer() and step() produce approximately equal outputs.

  Args:
    test_case: An absltest.TestCase (or similar) with assertion methods.
    layer: The SequenceLayer to test.
    input_shape: Channel shape (tuple), e.g. (16,).
    batch_size: Batch size for test inputs.
    time: Sequence length for test inputs.
    dtype: Input dtype.
    constants: Optional constants dict.
    atol: Absolute tolerance for output comparison.
    rtol: Relative tolerance for output comparison.
    test_step: Whether to test step() and compare with layer().
  """
  x = random_sequence(batch_size, time, input_shape, dtype=dtype)

  # Test layer().
  y_layer = layer.layer(x, constants=constants)

  # Check output shape.
  expected_shape = layer.get_output_shape(input_shape, constants=constants)
  test_case.assertEqual(y_layer.channel_shape, expected_shape)

  # Check output dtype.
  expected_dtype = layer.get_output_dtype(dtype, constants=constants)
  test_case.assertEqual(y_layer.dtype, expected_dtype)

  if not test_step or not layer.supports_step:
    return

  # Test step().
  block_size = layer.block_size
  y_step, _ = step_by_step(layer, x, block_size=block_size, constants=constants)

  # Check shapes match.
  test_case.assertEqual(y_step.shape, y_layer.shape)

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
