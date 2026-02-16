"""Position embeddings for MLX."""

import mlx.core as mx
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import types

Sequence = bt.Sequence


class ApplyRotaryPositionalEncoding(
    types.PreservesType,
    types.PreservesShape,
    types.SequenceLayer,
):
  """Applies Rotary Positional Encodings (RoPE) to the sequence."""

  def __init__(
      self,
      *,
      max_wavelength: float,
      axis: int = -1,
      only_advance_position_for_valid_timesteps: bool = True,
  ):
    super().__init__()
    self.max_wavelength = max_wavelength
    self._axis = axis
    self.only_advance_position_for_valid_timesteps = (
        only_advance_position_for_valid_timesteps
    )

  def _apply_rope(self, x, positions):
    """Apply rotary position encoding to x at given positions."""
    axis = self._axis + x.ndim if self._axis < 0 else self._axis
    channel_ndim = x.ndim - 2
    axis_dim = x.shape[axis]

    freq_exponents = (
        2.0 * mx.arange(axis_dim // 2).astype(mx.float32) / axis_dim
    )
    timescale = self.max_wavelength**freq_exponents

    broadcast_shape = [1] * x.ndim
    broadcast_shape[axis] = axis_dim // 2

    # Compute position angles.
    positions_f = positions.astype(mx.float32)
    radians = positions_f.reshape(
        positions_f.shape + (1,) * channel_ndim
    ) / timescale.reshape(broadcast_shape)
    sin_r = mx.sin(radians)
    cos_r = mx.cos(radians)

    # Split input along rotation axis, apply rotation.
    splits = mx.split(x, 2, axis=axis)
    x1, x2 = splits[0], splits[1]
    result = mx.concatenate(
        [x1 * cos_r - x2 * sin_r, x2 * cos_r + x1 * sin_r],
        axis=axis,
    )
    return result.astype(x.dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if self.only_advance_position_for_valid_timesteps:
      return mx.full((batch_size, 1), -1, dtype=mx.int32)
    else:
      return mx.zeros((batch_size, 1), dtype=mx.int32)

  @types.check_step
  def step(self, x, state, *, constants=None):
    x_time = x.shape[1]
    if self.only_advance_position_for_valid_timesteps:
      positions = state + mx.cumsum(x.mask.astype(mx.int32), axis=1)
      state = positions[:, -1:]
    else:
      positions = state + mx.arange(x_time, dtype=mx.int32)
      state = state + x_time
    y = x.apply_values(self._apply_rope, positions)
    return y, state

  @types.check_layer
  def layer(self, x, *, constants=None):
    if self.only_advance_position_for_valid_timesteps:
      positions = mx.maximum(
          0,
          mx.cumsum(x.mask.astype(mx.int32), axis=1) - 1,
      )
    else:
      positions = mx.broadcast_to(
          mx.arange(x.shape[1], dtype=mx.int32)[None, :],
          (x.shape[0], x.shape[1]),
      )
    return x.apply_values(self._apply_rope, positions)

  @classmethod
  def from_config(cls, config):
    return cls(
        max_wavelength=config.max_wavelength,
        axis=config.axis,
        only_advance_position_for_valid_timesteps=(
            config.only_advance_position_for_valid_timesteps
        ),
    )
